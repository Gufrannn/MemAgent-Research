"""Cross-Occupancy Role-Alternating Learning (CORAL) contracts.

The primary method deliberately has two *optimization masks*, not two separate
parameter vectors.  Both roles are implemented by one shared language model:
``memory_writer`` contains every non-terminal recurrent response and
``terminal_answer`` contains the final response.  Alternating the masks makes
the terminal-answer gradient at step 2k consume text memories materialized by
the preceding writer-active checkpoint at step 2k-1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

PHASES = ("memory_writer", "terminal_answer")


def phase_for_step(global_step: int) -> str:
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 1:
        raise ValueError("CORAL_NO_GO: global_step must be a positive integer")
    return "memory_writer" if global_step % 2 else "terminal_answer"


def role_masks(response_mask: torch.Tensor, final_mask: torch.Tensor, phase: str):
    """Return disjoint active/inactive token masks for a recurrent turn batch."""
    import torch
    if phase not in PHASES:
        raise ValueError("CORAL_NO_GO: invalid phase")
    if response_mask.ndim != 2 or final_mask.ndim != 1 \
            or len(response_mask) != len(final_mask) or final_mask.dtype != torch.bool:
        raise ValueError("CORAL_NO_GO: role-mask tensor contract")
    if not bool(torch.all((response_mask == 0) | (response_mask == 1))):
        raise ValueError("CORAL_NO_GO: response_mask must be binary")
    selector = ~final_mask if phase == "memory_writer" else final_mask
    active = response_mask * selector.to(response_mask.dtype).unsqueeze(-1)
    inactive = response_mask - active
    if not torch.equal(active + inactive, response_mask) \
            or bool(torch.any((active != 0) & (inactive != 0))):
        raise ValueError("CORAL_NO_GO: role masks do not partition response tokens")
    if int(active.sum().item()) < 1 or int(inactive.sum().item()) < 1:
        raise ValueError("CORAL_NO_GO: both recurrent roles must have tokens")
    return active, inactive


def role_covered_order(final_mask: torch.Tensor, world_size: int,
                       valid_rows: torch.Tensor | None = None) -> torch.Tensor:
    """Return equal contiguous DP partitions, each containing both roles.

    Recurrent generation stores terminal rows late in the flattened batch. A
    naive contiguous data-parallel split can consequently give one rank only
    writer rows and another only terminal rows. This deterministic reorder is
    applied after advantages are computed and before actor dispatch.
    """
    import torch

    if final_mask.ndim != 1 or final_mask.dtype != torch.bool \
            or isinstance(world_size, bool) or not isinstance(world_size, int) \
            or world_size < 1 or len(final_mask) % world_size:
        raise ValueError("CORAL_NO_GO: invalid role-coverage partition contract")
    if valid_rows is None:
        valid_rows = torch.ones_like(final_mask, dtype=torch.bool)
    if valid_rows.shape != final_mask.shape or valid_rows.dtype != torch.bool:
        raise ValueError("CORAL_NO_GO: invalid valid-row mask")
    final = torch.nonzero(final_mask & valid_rows, as_tuple=False).flatten().tolist()
    writer = torch.nonzero((~final_mask) & valid_rows, as_tuple=False).flatten().tolist()
    padding = torch.nonzero(~valid_rows, as_tuple=False).flatten().tolist()
    partition_size = len(final_mask) // world_size
    if len(final) < world_size or len(writer) < world_size or partition_size < 2:
        raise ValueError("CORAL_NO_GO: every DP rank requires both recurrent roles")
    final_counts = [len(final) // world_size + int(rank < len(final) % world_size)
                    for rank in range(world_size)]
    if any(count < 1 or count >= partition_size for count in final_counts):
        raise ValueError("CORAL_NO_GO: terminal rows cannot cover DP partitions")
    partitions = [[] for _ in range(world_size)]
    cursor = 0
    for rank, count in enumerate(final_counts):
        partitions[rank].extend(final[cursor:cursor + count])
        cursor += count
    for rank in range(world_size):
        partitions[rank].append(writer[rank])
    remaining = writer[world_size:] + padding
    remaining_cursor = 0
    for rank in range(world_size):
        capacity = partition_size - len(partitions[rank])
        partitions[rank].extend(remaining[remaining_cursor:remaining_cursor + capacity])
        remaining_cursor += capacity
        real = [index for index in partitions[rank] if bool(valid_rows[index])]
        if not any(bool(final_mask[index]) for index in real) \
                or not any(not bool(final_mask[index]) for index in real):
            raise ValueError("CORAL_NO_GO: role coverage failed")
    order = [index for partition in partitions for index in partition]
    if len(order) != len(final_mask) or sorted(order) != list(range(len(final_mask))):
        raise ValueError("CORAL_NO_GO: role reorder is not a permutation")
    return torch.tensor(order, dtype=torch.long, device=final_mask.device)


def validate_config(config: Any) -> dict[str, Any]:
    required = {
        "enabled", "active_from_update", "schedule", "role_partition",
        "require_recurrent", "require_grpo", "require_gate_a_sync",
    }
    value = dict(config)
    if set(value) != required or value["enabled"] is not True \
            or value["active_from_update"] != 1 \
            or value["schedule"] != "odd_writer_even_terminal_answer_v2" \
            or value["role_partition"] != "nonfinal_memory_writer_vs_final_answer" \
            or value["require_recurrent"] is not True \
            or value["require_grpo"] is not True \
            or value["require_gate_a_sync"] is not True:
        raise ValueError("CORAL_NO_GO: frozen algorithm config drifted")
    return value


def exposure_counts(total_updates: int) -> dict[str, int]:
    """Return the exact role-update exposure for a run of ``total_updates``."""
    if isinstance(total_updates, bool) or not isinstance(total_updates, int) or total_updates < 1:
        raise ValueError("CORAL_NO_GO: total_updates must be a positive integer")
    return {
        phase: sum(phase_for_step(step) == phase for step in range(1, total_updates + 1))
        for phase in PHASES
    }
