"""Certified writer-only secondary routing for exact QA ties."""

from __future__ import annotations
import torch


FORBIDDEN_OUTCOME_FIELDS = {"bot", "noop", "BOT", "NOOP", "bot_label", "noop_label", "outcome_label"}


def reject_forbidden_fields(rows: list[dict]) -> None:
    found = sorted(set().union(*(set(row) for row in rows)) & FORBIDDEN_OUTCOME_FIELDS)
    if found:
        raise ValueError(f"generic baseline received forbidden outcome fields: {found}")


def route_ncr(*, qa_advantage: torch.Tensor, secondary_score: torch.Tensor, uid: list[str],
              qa_reward: torch.Tensor, writer_mask: torch.Tensor, final_mask: torch.Tensor,
              eligible: torch.Tensor, exact_correct: torch.Tensor, lambda_: float) -> tuple[torch.Tensor, dict]:
    if not (len(uid) == len(qa_reward) == len(secondary_score) == len(eligible) == len(exact_correct)):
        raise ValueError("NCR trajectory metadata length mismatch")
    if writer_mask.shape != qa_advantage.shape or final_mask.shape[0] != qa_advantage.shape[0]:
        raise ValueError("NCR writer/final mask shape mismatch")
    traj_bonus = torch.zeros_like(secondary_score)
    tie_groups = 0
    for group in dict.fromkeys(uid):
        idx = torch.tensor([u == group for u in uid], device=qa_reward.device)
        rewards = qa_reward[idx]
        is_tie = bool(torch.all(rewards == rewards[0]))
        if not is_tie:
            continue
        if bool(torch.all(exact_correct[idx])):
            raise ValueError("NO_METHOD: all-exact-correct tie group is forbidden")
        if not bool(torch.all(eligible[idx])):
            continue
        scores = secondary_score[idx]
        traj_bonus[idx] = lambda_ * (scores - scores.mean())
        tie_groups += 1
    row_bonus = traj_bonus.unsqueeze(-1).expand_as(qa_advantage) * writer_mask.to(qa_advantage.dtype)
    row_bonus[final_mask] = 0
    result = qa_advantage + row_bonus
    non_tie_rows = row_bonus.abs().sum(dim=-1) == 0
    if not torch.equal(result[non_tie_rows], qa_advantage[non_tie_rows]):
        raise AssertionError("non-tie advantage changed")
    if torch.any(row_bonus[final_mask] != 0):
        raise AssertionError("final-answer NCR bonus is nonzero")
    return result, {"tie_groups_routed": tie_groups, "writer_only": True}


def generic_auxiliary(rows: list[dict], scorer_field: str) -> list[float]:
    reject_forbidden_fields(rows)
    if scorer_field not in {"generic_qa_score", "frozen_judge_score", "raw_judge_score", "uniform_direction"}:
        raise ValueError("unregistered generic scorer")
    return [float(row[scorer_field]) for row in rows]
