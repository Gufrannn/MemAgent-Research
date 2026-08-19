"""Deterministic, auditable seeds for recurrent rollout trajectories."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


MAX_TORCH_SEED = (1 << 63) - 1
SUPPORTED_TRAJECTORY_SEED_MODES = ("independent", "matched")


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value % MAX_TORCH_SEED


def build_trajectory_seed_records(
    *, base_seed: int, global_step: int, batch_size: int, rollout_n: int, mode: str
) -> list[dict[str, int | str]]:
    """Build one reproducible base seed per repeated rollout row."""
    if mode not in SUPPORTED_TRAJECTORY_SEED_MODES:
        raise ValueError(f"unsupported trajectory seed mode: {mode}")
    if rollout_n < 1 or batch_size < 1 or batch_size % rollout_n:
        raise ValueError("batch_size must be positive and divisible by rollout_n")

    records: list[dict[str, int | str]] = []
    for row in range(batch_size):
        group, replica = divmod(row, rollout_n)
        stream_replica = replica if mode == "independent" else 0
        trajectory_seed = _stable_seed(
            "memagent-trajectory", int(base_seed), int(global_step), group, stream_replica
        )
        records.append({
            "row": row,
            "group": group,
            "replica": replica,
            "mode": mode,
            "base_seed": int(base_seed),
            "global_step": int(global_step),
            "trajectory_seed": trajectory_seed,
        })
    return records


def derive_turn_request_seeds(
    trajectory_base_seeds: Sequence[int], sample_indices: Sequence[int], recurrent_turn: int
) -> list[int]:
    """Derive the request seed for each active trajectory in one recurrent turn."""
    if recurrent_turn < 0:
        raise ValueError("recurrent_turn must be non-negative")
    result: list[int] = []
    for sample_index in sample_indices:
        sample_index = int(sample_index)
        if sample_index < 0 or sample_index >= len(trajectory_base_seeds):
            raise IndexError(f"sample index out of range: {sample_index}")
        result.append(_stable_seed(
            "memagent-recurrent-turn",
            int(trajectory_base_seeds[sample_index]),
            int(recurrent_turn),
        ))
    return result
