"""Fail-closed primitives for predictive rate--distortion MemRL.

The coding prior is evaluated on a deliberately smaller sigma-field than the
actor: (previous memory, turn index).  This module never constructs that
context from an actor prompt; callers must provide a typed, audited context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


FORBIDDEN_PRIOR_FIELDS = frozenset(
    {
        "evidence",
        "new_evidence",
        "history",
        "history_chunk",
        "chunk",
        "gold",
        "gold_answer",
        "answer",
        "future",
        "future_chunk",
        "outcome",
        "reward",
    }
)
ALLOWED_PRIOR_FIELDS = frozenset({"previous_memory", "turn_index"})


class PriorTaintError(ValueError):
    """Raised when the coding prior could observe actor-only information."""


@dataclass(frozen=True)
class PriorContext:
    previous_memory: str
    turn_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.previous_memory, str):
            raise PriorTaintError("previous_memory must be text")
        if not isinstance(self.turn_index, int) or self.turn_index < 0:
            raise PriorTaintError("turn_index must be a non-negative integer")

    def canonical_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()

    @property
    def context_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def validate_prior_record(record: Mapping[str, object]) -> PriorContext:
    """Convert an exact two-field record into a prior context.

    Exact-key equality is intentional: aliases and unused extra fields are a
    common route for evidence leakage and therefore fail closed.
    """

    keys = frozenset(record)
    forbidden = sorted(keys & FORBIDDEN_PRIOR_FIELDS)
    if forbidden:
        raise PriorTaintError(f"coding prior record contains forbidden fields: {forbidden}")
    if keys != ALLOWED_PRIOR_FIELDS:
        raise PriorTaintError(
            f"coding prior record keys must be exactly {sorted(ALLOWED_PRIOR_FIELDS)}; got {sorted(keys)}"
        )
    return PriorContext(
        previous_memory=record["previous_memory"],  # type: ignore[arg-type]
        turn_index=record["turn_index"],  # type: ignore[arg-type]
    )


def conditional_rate_nats(
    actor_log_probs: torch.Tensor,
    prior_log_probs: torch.Tensor,
    writer_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-trajectory and mean sampled conditional KL/code length.

    The estimator is sum_t log pi(M'|H,M) - log q(M'|M,t), never token
    count. Negative finite-sample values are retained in the ledger rather
    than clipped, so diagnostics cannot silently turn this into length.
    """

    if actor_log_probs.shape != prior_log_probs.shape or actor_log_probs.shape != writer_mask.shape:
        raise ValueError("actor_log_probs, prior_log_probs, and writer_mask must have identical shapes")
    if actor_log_probs.ndim != 2:
        raise ValueError("conditional-rate tensors must have shape [batch, response_tokens]")
    if not torch.isfinite(actor_log_probs).all() or not torch.isfinite(prior_log_probs).all():
        raise ValueError("conditional-rate log probabilities must be finite")
    mask = writer_mask.to(dtype=actor_log_probs.dtype)
    if torch.any(mask < 0) or torch.any(mask > 1):
        raise ValueError("writer_mask must be binary")
    per_trajectory = ((actor_log_probs - prior_log_probs) * mask).sum(dim=-1)
    active = mask.sum(dim=-1) > 0
    if not active.any():
        raise ValueError("conditional-rate batch has no writer tokens")
    return per_trajectory, per_trajectory[active].mean()


@dataclass
class ProjectedDual:
    capacity_nats: float
    learning_rate: float
    value: float = 0.0
    updates: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.capacity_nats) or self.capacity_nats < 0:
            raise ValueError("capacity_nats must be finite and non-negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("dual value must be finite and non-negative")

    def penalty(self, estimated_rate: torch.Tensor) -> torch.Tensor:
        return estimated_rate * float(self.value)

    def step(self, estimated_rate: float) -> float:
        if not math.isfinite(estimated_rate):
            raise ValueError("estimated_rate must be finite")
        self.value = max(0.0, self.value + self.learning_rate * (estimated_rate - self.capacity_nats))
        self.updates += 1
        return self.value

    def state_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ProjectedDual":
        required = {"capacity_nats", "learning_rate", "value", "updates"}
        if set(state) != required:
            raise ValueError(f"dual checkpoint fields must be exactly {sorted(required)}")
        obj = cls(float(state["capacity_nats"]), float(state["learning_rate"]), float(state["value"]))
        obj.updates = int(state["updates"])
        if obj.updates < 0:
            raise ValueError("dual updates must be non-negative")
        return obj


def validate_capacity_frontier(capacities: Sequence[float], *, minimum_points: int = 3) -> tuple[float, ...]:
    values = tuple(float(value) for value in capacities)
    if len(values) < minimum_points or len(set(values)) != len(values):
        raise ValueError(f"frontier requires at least {minimum_points} distinct capacity points")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("frontier capacities must be finite and non-negative")
    if tuple(sorted(values)) != values:
        raise ValueError("frontier capacities must be canonical ascending")
    return values


def assert_rate_not_length(rates: Iterable[float], token_lengths: Iterable[int]) -> float:
    """Return Kendall discordance; reject a pure length ordering."""

    r, n = list(rates), list(token_lengths)
    if len(r) != len(n) or len(r) < 3:
        raise ValueError("rate-not-length audit requires aligned vectors with at least three rows")
    pairs = [(i, j) for i in range(len(r)) for j in range(i + 1, len(r)) if n[i] != n[j] and r[i] != r[j]]
    if not pairs:
        raise ValueError("rate-not-length audit has no comparable pairs")
    discordant = sum((n[i] - n[j]) * (r[i] - r[j]) < 0 for i, j in pairs)
    fraction = discordant / len(pairs)
    if fraction == 0:
        raise ValueError("PRD_E1_FAIL_RATE_IS_LENGTH_ORDER")
    return fraction


def save_prd_checkpoint(
    directory: str | Path,
    *,
    actor_state: Mapping[str, object],
    prior_state: Mapping[str, object],
    dual: ProjectedDual,
) -> Path:
    """Atomically save all three scientific states; none is optional."""

    if not actor_state or not prior_state:
        raise ValueError("actor and prior checkpoint states must both be non-empty")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / "prd_state.pt.tmp"
    final = target / "prd_state.pt"
    torch.save({"schema_version": 1, "actor": dict(actor_state), "prior": dict(prior_state), "dual": dual.state_dict()}, temporary)
    temporary.replace(final)
    return final


def load_prd_checkpoint(path: str | Path) -> tuple[dict, dict, ProjectedDual]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if set(payload) != {"schema_version", "actor", "prior", "dual"} or payload["schema_version"] != 1:
        raise ValueError("incomplete or unsupported PRD checkpoint")
    if not payload["actor"] or not payload["prior"]:
        raise ValueError("PRD checkpoint is missing actor or prior state")
    return payload["actor"], payload["prior"], ProjectedDual.from_state_dict(payload["dual"])
