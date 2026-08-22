"""Transactional helpers for prefix-certified actor updates.

The helpers are deliberately independent of FSDP orchestration: parameters are
the local shards owned by a rank, while feasibility is decided from globally
gathered prefix rows by the caller.
"""
from __future__ import annotations

import copy
import hashlib
import io
import random
from dataclasses import dataclass

import torch


ALPHA_GRID = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def _cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def digest(value) -> str:
    buffer = io.BytesIO()
    torch.save(_cpu_clone(value), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def parameter_snapshot(module):
    return [parameter.detach().cpu().clone() for parameter in module.parameters()]


def set_interpolated_parameters(module, old, full, alpha: float):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("trial alpha is outside [0,1]")
    with torch.no_grad():
        for target, before, proposed in zip(module.parameters(), old, full):
            trial = before + (proposed - before) * alpha
            target.copy_(trial.to(device=target.device, dtype=target.dtype))


def displacement_norm(old, new) -> float:
    squared = sum(float((after.double() - before.double()).square().sum())
                  for before, after in zip(old, new))
    return squared ** 0.5


def rng_snapshot():
    state = {
        "torch_cpu": torch.get_rng_state().cpu(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state):
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass(frozen=True)
class TrialDecision:
    alpha: float
    accepted_nonzero: bool
    proposal_zero: bool
    tested: tuple


def largest_tested_feasible(trials, proposal_zero=False, alpha_grid=ALPHA_GRID):
    """Return the first feasible grid point in declared descending test order.

    No monotonicity is assumed.  ``trials`` maps an actually evaluated alpha to
    its global feasibility decision; an untested alpha can never be committed.
    """
    order = tuple(float(value) for value in alpha_grid)
    if order != tuple(sorted(order, reverse=True)) or len(set(order)) != len(order):
        raise ValueError("alpha grid must be unique and descending")
    for alpha in order:
        if alpha not in trials:
            raise ValueError(f"alpha {alpha} was not actually tested")
        if bool(trials[alpha]):
            if proposal_zero:
                return TrialDecision(0.0, False, True, order)
            return TrialDecision(alpha, True, False, order)
    return TrialDecision(0.0, False, bool(proposal_zero), order)
