"""Transactional helpers for prefix-certified actor updates.

The helpers are deliberately independent of FSDP orchestration: parameters are
the local shards owned by a rank, while feasibility is decided from globally
gathered prefix rows by the caller.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import random
from dataclasses import dataclass

import numpy as np
import torch


ALPHA_GRID = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
RWWPO2_INNER_TRANSACTIONS = 2
RWWPO2_SCHEDULE_KIND = "constant_with_linear_warmup"
RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS = 8_388_608


def local_gradient_sketch_sufficient_statistics(
        parameters, *,
        chunk_elements: int = RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS):
    """Compute the registered local gradient projection with bounded memory.

    The numeric oracle and the live actor must call this exact implementation.
    Coordinates are local-FSDP flattened parameter coordinates; the caller is
    responsible for the rank all-reduce and the final square root of element 0.
    """
    chunk_elements = int(chunk_elements)
    if chunk_elements < 1:
        raise ValueError("gradient sketch chunk must be positive")
    values = None
    for parameter_index, parameter in enumerate(parameters):
        if parameter.grad is None:
            continue
        raw_gradient = parameter.grad.detach()
        if not raw_gradient.is_contiguous():
            raise RuntimeError(
                "RWWPO2_GRADIENT_SKETCH_NONCONTIGUOUS_GRADIENT_NO_GO")
        # The explicit contiguity gate makes view(-1) zero-copy.  A flatten or
        # reshape here could allocate an unbounded full FSDP-shard temporary.
        flattened = raw_gradient.view(-1)
        if values is None:
            values = torch.zeros(4, dtype=torch.float64, device=flattened.device)
        for chunk_start in range(0, flattened.numel(), chunk_elements):
            chunk_stop = min(flattened.numel(), chunk_start + chunk_elements)
            gradient = flattened[chunk_start:chunk_stop].to(dtype=torch.float64)
            coordinate = torch.arange(
                chunk_start, chunk_stop, device=gradient.device, dtype=torch.int64)
            alternating = torch.bitwise_and(
                coordinate + parameter_index, 1
            ).to(dtype=torch.float64).mul_(2).sub_(1)
            coordinate.add_(17 * parameter_index).remainder_(257)
            saw = coordinate.to(dtype=torch.float64).sub_(128.0).div_(128.0)
            values[0].add_(torch.dot(gradient, gradient))
            values[1].add_(gradient.sum())
            values[2].add_(torch.dot(gradient, alternating))
            values[3].add_(torch.dot(gradient, saw))
            del gradient, coordinate, alternating, saw
    if values is None:
        raise RuntimeError("RWWPO2_GRADIENT_SKETCH_MISSING_GRADIENT")
    return values


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


def named_buffer_snapshot(module):
    """Snapshot every persistent and non-persistent named model buffer.

    Transaction rollback must cover forward-mutated state as well as trainable
    parameters.  ``named_buffers`` includes non-persistent buffers such as
    rotary/cache state that are absent from a normal ``state_dict``.
    """
    return [
        (name, buffer.detach().cpu().clone())
        for name, buffer in module.named_buffers()
    ]


def restore_named_buffers(module, snapshot):
    """Restore an exact named-buffer snapshot, rejecting inventory drift."""
    current = list(module.named_buffers())
    expected_names = [name for name, _ in snapshot]
    current_names = [name for name, _ in current]
    if current_names != expected_names:
        raise RuntimeError("RWWPO2_MODEL_BUFFER_INVENTORY_DRIFT")
    with torch.no_grad():
        for (name, target), (_, source) in zip(current, snapshot):
            if target.shape != source.shape or target.dtype != source.dtype:
                raise RuntimeError(
                    f"RWWPO2_MODEL_BUFFER_METADATA_DRIFT:{name}")
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def module_state_digest(parameters, buffers):
    """Bind parameter shards and all named buffers into one model digest."""
    return digest({"parameters": parameters, "named_buffers": buffers})


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


def relative_displacement_norm(old, new, epsilon: float = 1e-30) -> float:
    """Global-shard-local relative displacement used by the exposure oracle.

    Distributed callers all-reduce the numerator and denominator squared sums
    before taking their ratio.  This helper is exact for an unsharded module and
    intentionally does not hide the distributed reduction requirement.
    """
    numerator = sum(float((after.double() - before.double()).square().sum())
                    for before, after in zip(old, new))
    denominator = sum(float(before.double().square().sum()) for before in old)
    return math.sqrt(numerator) / (math.sqrt(denominator) + float(epsilon))


def proposal_clock(round_id: int, inner_id: int, inner_transactions: int = RWWPO2_INNER_TRANSACTIONS) -> int:
    """One-based immutable logical proposal coordinate."""
    round_id = int(round_id)
    inner_id = int(inner_id)
    inner_transactions = int(inner_transactions)
    if round_id < 1 or inner_transactions < 1 or not 1 <= inner_id <= inner_transactions:
        raise ValueError("invalid RWWPO-2 round/inner proposal coordinate")
    return inner_transactions * (round_id - 1) + inner_id


def stateless_proposal_lr(*, base_lr: float, warmup_proposals: int,
                          total_proposals: int, proposal_id: int,
                          kind: str = RWWPO2_SCHEDULE_KIND) -> float:
    """Return S(p) without reading or mutating a scheduler object."""
    base_lr = float(base_lr)
    warmup_proposals = int(warmup_proposals)
    total_proposals = int(total_proposals)
    proposal_id = int(proposal_id)
    if kind != RWWPO2_SCHEDULE_KIND:
        raise ValueError(f"unsupported stateless proposal schedule: {kind}")
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("base LR must be finite and positive")
    if warmup_proposals < 0 or total_proposals < 1 or not 1 <= proposal_id <= total_proposals:
        raise ValueError("invalid proposal schedule coordinate")
    if warmup_proposals == 0:
        return base_lr
    return base_lr * min(1.0, proposal_id / warmup_proposals)


def set_stateless_proposal_lr(optimizer, **schedule) -> float:
    """Write S(p) to every param group and return the exact written value."""
    lr = stateless_proposal_lr(**schedule)
    if not optimizer.param_groups:
        raise ValueError("optimizer has no parameter groups")
    for group in optimizer.param_groups:
        group["lr"] = lr
    if any(float(group["lr"]) != lr for group in optimizer.param_groups):
        raise RuntimeError("RWWPO-2 param-group LR drift")
    return lr


def logical_transaction_seed(*, experiment_seed: int, round_id: int,
                             inner_id: int, rank: int, stream: str) -> int:
    """Attempt-independent seed H(experiment, round, inner, rank, stream)."""
    if not stream or any(int(value) < 0 for value in (experiment_seed, rank)):
        raise ValueError("invalid logical seed coordinate")
    # proposal_clock validates round/inner and prevents ambiguous coordinates.
    proposal_clock(round_id, inner_id)
    payload = "\x1f".join(("rwwpo2", str(int(experiment_seed)), str(int(round_id)),
                            str(int(inner_id)), str(int(rank)), str(stream))).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % ((1 << 63) - 1)


def seed_transaction_rng(seed: int):
    """Set Python, NumPy, Torch CPU, and every CUDA RNG from a logical seed."""
    seed = int(seed)
    if not 0 <= seed < (1 << 63):
        raise ValueError("logical transaction seed is outside torch range")
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def writer_logprob_rms(current_log_prob, behavior_log_prob, writer_mask,
                       sample_index) -> float:
    """Trajectory-balanced RMS writer-logprob displacement.

    Each trajectory contributes one within-trajectory token mean, irrespective of
    its number of recurrent writes or writer tokens.  Distributed callers must
    gather the returned per-trajectory sufficient statistics rather than average
    rank-local scalar RMS values.
    """
    active = writer_mask.bool()
    if sample_index.ndim != 1 or len(sample_index) != len(active):
        raise ValueError("writer exposure sample identity is not row aligned")
    trajectory_means = []
    squared = (current_log_prob - behavior_log_prob).detach().double().square()
    for sid in torch.unique(sample_index, sorted=True):
        rows = sample_index == sid
        trajectory_mask = active & rows.unsqueeze(-1)
        if not trajectory_mask.any():
            raise ValueError("writer exposure requires writer tokens for every trajectory")
        trajectory_means.append(squared[trajectory_mask].mean())
    if not trajectory_means:
        raise ValueError("writer exposure requires at least one trajectory")
    return float(torch.stack(trajectory_means).mean().sqrt().item())


def writer_logprob_rms_sufficient_statistics(current_log_prob, behavior_log_prob,
                                              writer_mask, sample_index):
    """Return `(sum trajectory MSE, trajectory count)` for exact all-reduce."""
    active = writer_mask.bool()
    if sample_index.ndim != 1 or len(sample_index) != len(active):
        raise ValueError("writer exposure sample identity is not row aligned")
    squared = (current_log_prob - behavior_log_prob).detach().double().square()
    values = []
    for sid in torch.unique(sample_index, sorted=True):
        trajectory_mask = active & (sample_index == sid).unsqueeze(-1)
        if not trajectory_mask.any():
            raise ValueError("writer exposure requires writer tokens for every trajectory")
        values.append(squared[trajectory_mask].mean())
    if not values:
        raise ValueError("writer exposure requires at least one trajectory")
    return float(torch.stack(values).sum().item()), len(values)


def off_behavior_exposed(*, relative_parameter_displacement: float,
                         writer_logprob_rms_value: float,
                         tau_theta: float, tau_logprob: float) -> bool:
    values = (relative_parameter_displacement, writer_logprob_rms_value,
              tau_theta, tau_logprob)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("invalid off-behavior exposure value")
    return (float(relative_parameter_displacement) > float(tau_theta) and
            float(writer_logprob_rms_value) > float(tau_logprob))


def _normalized_weight_stats(log_weights):
    values = [float(value) for value in log_weights]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("prefix weights must be nonempty and finite")
    peak = max(values)
    raw = [math.exp(value - peak) for value in values]
    total = sum(raw)
    weights = [value / total for value in raw]
    chi2 = len(weights) * sum(value * value for value in weights) - 1.0
    return {
        "count": len(values),
        "ess_fraction": 1.0 / (1.0 + chi2),
        "chi2": chi2,
        "max_abs_log_ratio": max(abs(value) for value in values),
        "mean_log_ratio": sum(values) / len(values),
    }


def prefix_distribution_stats(prefix_rows, *, q_min: float, root_q_min: float,
                              log_ratio_cap: float):
    """Compute trajectory/root ESS and leave-one-root-out stability by turn.

    `root_identity_hash` identifies the independent prompt root; multiple sampled
    trajectories under that root are averaged in probability space.  This keeps
    the trajectory ESS as a concentration diagnostic while preventing rollout
    replicas from being counted as independent roots.
    """
    if not 0 < float(q_min) <= 1 or not 0 < float(root_q_min) <= 1:
        raise ValueError("ESS thresholds must be in (0,1]")
    if not math.isfinite(float(log_ratio_cap)) or float(log_ratio_cap) <= 0:
        raise ValueError("log-ratio cap must be finite and positive")
    required = {"turn", "sample_index", "root_identity_hash", "log_ratio"}
    if not prefix_rows or any(set(row) < required for row in prefix_rows):
        raise ValueError("prefix rows lack stable root evidence")
    results = []
    for turn in sorted({int(row["turn"]) for row in prefix_rows}):
        turn_rows = [row for row in prefix_rows if int(row["turn"]) == turn]
        trajectory = _normalized_weight_stats([row["log_ratio"] for row in turn_rows])
        roots = {}
        for row in turn_rows:
            roots.setdefault(str(row["root_identity_hash"]), []).append(float(row["log_ratio"]))
        root_log_weights = {}
        for root, values in roots.items():
            peak = max(values)
            root_log_weights[root] = peak + math.log(
                sum(math.exp(value - peak) for value in values) / len(values))
        root = _normalized_weight_stats(root_log_weights.values())
        full_feasible = (
            trajectory["ess_fraction"] >= float(q_min)
            and root["ess_fraction"] >= float(root_q_min)
            and trajectory["max_abs_log_ratio"] <= float(log_ratio_cap)
        )
        loo = []
        for removed in sorted(roots):
            kept = [row for row in turn_rows if str(row["root_identity_hash"]) != removed]
            if not kept:
                loo.append({"removed_root": removed, "complete": False, "feasible": False})
                continue
            kept_trajectory = _normalized_weight_stats([row["log_ratio"] for row in kept])
            kept_roots = {}
            for row in kept:
                kept_roots.setdefault(str(row["root_identity_hash"]), []).append(
                    float(row["log_ratio"]))
            kept_root_log_weights = []
            for values in kept_roots.values():
                peak = max(values)
                kept_root_log_weights.append(
                    peak + math.log(sum(math.exp(value - peak) for value in values) / len(values)))
            kept_root = _normalized_weight_stats(kept_root_log_weights)
            feasible = (
                kept_trajectory["ess_fraction"] >= float(q_min)
                and kept_root["ess_fraction"] >= float(root_q_min)
                and kept_trajectory["max_abs_log_ratio"] <= float(log_ratio_cap)
            )
            loo.append({
                "removed_root": removed,
                "complete": True,
                "feasible": bool(feasible),
                "trajectory_ess_fraction": kept_trajectory["ess_fraction"],
                "root_ess_fraction": kept_root["ess_fraction"],
                "max_abs_log_ratio": kept_trajectory["max_abs_log_ratio"],
            })
        complete_loo = [row for row in loo if row["complete"]]
        # LOO stability is interpretable only when deletion leaves at least two
        # independent prompt roots. Sparse late turns still face the full-data
        # ESS/cap constraints; they are reported as unsupported, not failures.
        root_loo_supported = len(roots) >= 3
        results.append({
            "turn": turn,
            "batch_size": trajectory["count"],
            "root_count": root["count"],
            "ess_fraction": trajectory["ess_fraction"],
            "chi2": trajectory["chi2"],
            "root_ess_fraction": root["ess_fraction"],
            "root_chi2": root["chi2"],
            "max_abs_log_ratio": trajectory["max_abs_log_ratio"],
            "mean_log_ratio": trajectory["mean_log_ratio"],
            "feasible": bool(full_feasible),
            "root_loo": loo,
            "root_loo_supported": root_loo_supported,
            "root_loo_complete_fraction": len(complete_loo) / len(loo),
            "root_loo_feasibility_flip_fraction": (
                sum(bool(row["feasible"]) != bool(full_feasible) for row in complete_loo)
                / len(complete_loo) if root_loo_supported else 0.0
            ),
        })
    return results


def rng_snapshot():
    state = {
        "torch_cpu": torch.get_rng_state().cpu(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state):
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def ordered_rng_state_digests(states):
    """Return the ordered digest vector that identifies a replay RNG schedule."""
    states = list(states)
    if not states:
        raise ValueError("RWWPO2 replay RNG schedule is empty")
    return [digest(state) for state in states]


def replay_with_rng_snapshots(items, forward):
    """Replay ordered payloads under their behavior RNG without advancing RNG.

    Each item is ``(payload, rng_snapshot)``.  The caller owns model-state
    restoration; this helper closes the complete Python/NumPy/Torch RNG side of
    a diagnostic forward and restores the algorithmic terminal RNG even when a
    replay forward raises.
    """
    items = list(items)
    if not items:
        raise ValueError("RWWPO2 replay requires at least one microbatch")
    terminal = rng_snapshot()
    outputs = []
    try:
        for payload, state in items:
            restore_rng(state)
            outputs.append(forward(payload))
    finally:
        restore_rng(terminal)
    return outputs


@dataclass(frozen=True)
class TrialDecision:
    alpha: float
    accepted_nonzero: bool
    proposal_zero: bool
    tested: tuple


def largest_tested_feasible(trials, proposal_zero=False, alpha_grid=ALPHA_GRID):
    """Validate actual backtracking evidence and return its canonical decision.

    ``trials`` is either an insertion-ordered mapping or an ordered iterable of
    ``(alpha, feasible)`` pairs.  It must be the exact descending prefix that a
    fail-closed runtime evaluated: the first feasible point terminates the
    prefix, while an all-infeasible decision must exhaust the declared grid.
    """
    order = tuple(float(value) for value in alpha_grid)
    if order != tuple(sorted(order, reverse=True)) or len(set(order)) != len(order):
        raise ValueError("alpha grid must be unique and descending")
    raw_items = list(trials.items()) if hasattr(trials, "items") else list(trials)
    if not raw_items:
        raise ValueError("backtracking trial evidence is empty")
    if any(not isinstance(feasible, (bool, np.bool_))
           for _, feasible in raw_items):
        raise ValueError("trial feasibility evidence must be boolean")
    items = [(float(alpha), bool(feasible)) for alpha, feasible in raw_items]
    tested_order = tuple(alpha for alpha, _ in items)
    if tested_order != order[:len(tested_order)]:
        raise ValueError("trials are not an exact descending alpha-grid prefix")
    feasible_indices = [index for index, (_, feasible) in enumerate(items) if feasible]
    if feasible_indices:
        if feasible_indices != [len(items) - 1]:
            raise ValueError("first feasible trial did not terminate tested prefix")
        selected = tested_order[-1]
    else:
        if len(items) != len(order):
            raise ValueError("all-infeasible trials did not exhaust alpha grid")
        selected = 0.0
    alpha = 0.0 if proposal_zero else selected
    return TrialDecision(alpha, alpha > 0.0, bool(proposal_zero), tested_order)
