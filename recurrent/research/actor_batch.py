"""Batch-size invariants shared by recurrent PPO actor updates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np


DIAG_PREFIX = "[GATE_A_BATCH_DIAG]"


def stable_identity_int64(value: object) -> int:
    """Reconstruct the signed-positive int64 stored in distributed ledgers."""
    raw = hashlib.sha256(str(value).encode()).digest()[:8]
    return int.from_bytes(raw, "big", signed=False) & ((1 << 63) - 1)


@dataclass(frozen=True)
class ActorBatchPlan:
    global_rollout_batch_size: int
    global_prompt_mini_batch_size: int
    global_rollout_mini_batch_size: int
    data_parallel_world_size: int
    update_steps_per_batch: int
    local_mini_batch_size: int
    local_train_batch_size: int


def build_actor_batch_plan(
    *, train_batch_size: int, rollout_n: int, ppo_mini_batch_size: int, data_parallel_world_size: int
) -> ActorBatchPlan:
    """Translate global PPO sizes into per-data-parallel-rank sizes.

    Both configured sizes count prompts. Workers expand each by ``rollout_n``
    before converting the expanded mini-batch to a per-rank trajectory count.
    """
    values = {
        "train_batch_size": train_batch_size,
        "rollout_n": rollout_n,
        "ppo_mini_batch_size": ppo_mini_batch_size,
        "data_parallel_world_size": data_parallel_world_size,
    }
    if any(int(value) <= 0 for value in values.values()):
        raise ValueError(f"{DIAG_PREFIX} actor batch sizes must be positive: {values}")

    global_rollout_batch_size = int(train_batch_size) * int(rollout_n)
    if int(train_batch_size) % int(ppo_mini_batch_size):
        raise ValueError(
            f"{DIAG_PREFIX} configured prompt batch is not divisible by the prompt PPO mini-batch: "
            f"TRAIN_BATCH_SIZE={train_batch_size}, ROLLOUT_N={rollout_n}, "
            f"global_rollout_batch_size={global_rollout_batch_size}, "
            f"PPO_MINI_BATCH_SIZE={ppo_mini_batch_size}. PPO_MINI_BATCH_SIZE counts prompts."
        )
    global_rollout_mini_batch_size = int(ppo_mini_batch_size) * int(rollout_n)
    if global_rollout_mini_batch_size % int(data_parallel_world_size):
        raise ValueError(
            f"{DIAG_PREFIX} rollout-expanded PPO mini-batch cannot be evenly dispatched: "
            f"PPO_MINI_BATCH_SIZE={ppo_mini_batch_size}, ROLLOUT_N={rollout_n}, "
            f"global_rollout_mini_batch_size={global_rollout_mini_batch_size}, "
            f"data_parallel_world_size={data_parallel_world_size}. Increase the mini-batch "
            "or reduce the data-parallel world size; samples will not be duplicated."
        )

    update_steps = int(train_batch_size) // int(ppo_mini_batch_size)
    local_mini = global_rollout_mini_batch_size // int(data_parallel_world_size)
    # The checks above make both values positive; retain an explicit guard so a
    # future normalization change can never leak sections=0 to torch.
    if update_steps < 1 or local_mini < 1:
        raise ValueError(
            f"{DIAG_PREFIX} invalid actor split: global_rollout_batch_size={global_rollout_batch_size}, "
            f"global_prompt_mini_batch_size={ppo_mini_batch_size}, "
            f"global_rollout_mini_batch_size={global_rollout_mini_batch_size}, "
            f"data_parallel_world_size={data_parallel_world_size}, "
            f"computed_num_mini_batches={update_steps}, local_mini_batch_size={local_mini}"
        )
    return ActorBatchPlan(
        global_rollout_batch_size=global_rollout_batch_size,
        global_prompt_mini_batch_size=int(ppo_mini_batch_size),
        global_rollout_mini_batch_size=global_rollout_mini_batch_size,
        data_parallel_world_size=int(data_parallel_world_size),
        update_steps_per_batch=update_steps,
        local_mini_batch_size=local_mini,
        local_train_batch_size=update_steps * local_mini,
    )


def validate_active_actor_batch(*, active_batch_size: int, world_size: int, response_token_count: int) -> None:
    """Reject batches that cannot produce a meaningful update before dispatch."""
    if active_batch_size < 1:
        raise ValueError(f"{DIAG_PREFIX} active actor batch is empty: active_batch_size=0")
    if active_batch_size < world_size:
        raise ValueError(
            f"{DIAG_PREFIX} active batch is smaller than the data-parallel world size: "
            f"active_batch_size={active_batch_size}, world_size={world_size}. "
            "At least one real sample per rank is required; masked padding is not used to create empty ranks."
        )
    if response_token_count < 1:
        raise ValueError(
            f"{DIAG_PREFIX} actor batch has no trainable response tokens: "
            f"active_batch_size={active_batch_size}, response_valid_tokens=0"
        )


def align_recurrent_turn_identities(
    *,
    source_rows: Sequence[int],
    source_uids: Sequence[object],
    dataset_indices: Sequence[int],
    trajectory_seeds: Sequence[int],
) -> dict[str, np.ndarray]:
    """Build row-aligned internal grouping and stable audit identities.

    ``uid`` is the trainer's ephemeral GRPO grouping label.  It must remain
    aligned with every recurrent turn, but it is deliberately not used as an
    experimental identity.  Stable RWWPO identities instead derive from the
    frozen dataset row and the independently derived trajectory seed.
    """
    uids = np.asarray(source_uids, dtype=object)
    indices = np.asarray(dataset_indices)
    seeds = np.asarray(trajectory_seeds)
    rows = np.asarray(source_rows)
    source_count = len(uids)
    if len(indices) != source_count or len(seeds) != source_count:
        raise ValueError(
            "RWWPO stable identity source columns are not row aligned: "
            f"uids={source_count}, dataset_indices={len(indices)}, "
            f"trajectory_seeds={len(seeds)}"
        )
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("RWWPO stable identity source rows must be a 1D integer vector")
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= source_count):
        raise ValueError("RWWPO stable identity source row is out of range")

    source_example_ids = np.asarray(
        [f"frozen_train_row:{int(index)}" for index in indices], dtype=object
    )
    source_trajectory_ids = np.asarray(
        [f"{example_id}:seed:{int(seed)}" for example_id, seed in zip(source_example_ids, seeds)],
        dtype=object,
    )
    return {
        "uid": uids[rows],
        "stable_example_id": source_example_ids[rows],
        "trajectory_id": source_trajectory_ids[rows],
        "trajectory_seed": seeds.astype(np.uint64, copy=False)[rows],
    }
