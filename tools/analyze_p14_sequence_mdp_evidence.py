#!/usr/bin/env python3
"""P14 MDP-evidence audit for sequence memory-computation probes.

This script refines the coarse P14 gates into paper-safer diagnostics:

1. ordering is decomposed into answer-quality change and cost/footprint change;
2. composition is reported with positive margins, not epsilon-only gates;
3. budget is measured by value gain, optimal step usage, and unused budget;
4. path-dependent marginal value is computed from existing sequence outcomes;
5. optional EXPAND-Q vs EXPAND-State operator competence control.

All numbers are exploratory unless the input matrix uses the official
LongMemEval judge.  The script consumes only frozen generation outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2]
DEFAULT_BOOTSTRAP_SEED = 20260831
DEFAULT_BOOTSTRAP_SAMPLES = 10000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except ValueError:
        return math.nan


def mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / max(1, len(values))


def quantiles(values: list[float]) -> dict[str, float]:
    values = sorted(value for value in values if not math.isnan(value))
    if not values:
        return {"mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    positives = [value for value in values if value > 0]
    return {
        "mean": mean(values),
        "median": median(values),
        "min": values[0],
        "max": values[-1],
        "positive_mean": mean(positives),
        "positive_median": median(positives) if positives else math.nan,
        "n_positive": len(positives),
    }


def threshold_counts(values: list[float], thresholds: list[float]) -> dict[str, int]:
    return {f">{threshold:g}": sum(1 for value in values if not math.isnan(value) and value > threshold) for threshold in thresholds}


def sign(value: float, eps: float) -> int:
    if math.isnan(value):
        return 0
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def sign_transition(before: float, after: float, eps: float) -> str:
    before_sign = sign(before, eps)
    after_sign = sign(after, eps)
    if before_sign == after_sign:
        return f"same_{before_sign}"
    if before_sign != 0 and after_sign != 0:
        return f"strict_reversal_{before_sign}_to_{after_sign}"
    return f"zero_boundary_{before_sign}_to_{after_sign}"


def bootstrap_ci_mean(values: list[float], *, seed: int, n_bootstrap: int, alpha: float = 0.05) -> dict[str, float]:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0, "n_bootstrap": n_bootstrap}
    rng = random.Random(seed)
    boot = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot.append(mean(sample))
    boot.sort()
    lo_idx = max(0, int((alpha / 2) * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))
    return {
        "mean": mean(values),
        "ci_low": boot[lo_idx],
        "ci_high": boot[hi_idx],
        "n": n,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_ci_proportion(values: list[float], threshold: float, *, seed: int, n_bootstrap: int, alpha: float = 0.05) -> dict[str, float]:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return {"threshold": threshold, "proportion": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0, "n_bootstrap": n_bootstrap}
    indicators = [1.0 if abs(value) > threshold else 0.0 for value in values]
    result = bootstrap_ci_mean(indicators, seed=seed, n_bootstrap=n_bootstrap, alpha=alpha)
    return {
        "threshold": threshold,
        "proportion": result["mean"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "n": result["n"],
        "n_bootstrap": n_bootstrap,
    }


def best_set(values: dict[str, float], eps: float) -> tuple[float, list[str]]:
    finite = {key: value for key, value in values.items() if not math.isnan(value)}
    if not finite:
        return math.nan, []
    best = max(finite.values())
    return best, [key for key, value in finite.items() if abs(value - best) <= eps]


def step_count(operation: str) -> int:
    return {
        "stop": 0,
        "answer_now": 0,
        "base": 0,
        "refine": 1,
        "expand": 1,
        "expand_q": 1,
        "refine_expand": 2,
        "expand_refine": 2,
    }.get(operation, operation.count("_") + 1)


def op_value(row: dict[str, str], op: str | None, suffix: str) -> float:
    if op is None:
        return math.nan
    return to_float(row.get(f"{op}_{suffix}"))


def analyze(
    rows: list[dict[str, str]],
    *,
    stop: str,
    refine: str,
    expand: str,
    refine_expand: str,
    expand_refine: str,
    expand_q: str | None,
    metric: str,
    eps: float,
    thresholds: list[float],
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    per_qid: list[dict[str, Any]] = []
    comp_gains: list[float] = []
    order_reward_diffs: list[float] = []
    order_cost_diffs: list[float] = []
    budget_gains_b2_b1: list[float] = []
    budget_gains_b1_b0: list[float] = []
    delta_expand_direct: list[float] = []
    delta_expand_after_refine: list[float] = []
    delta_expand_path_shift: list[float] = []
    delta_refine_direct: list[float] = []
    delta_refine_after_expand: list[float] = []
    delta_refine_path_shift: list[float] = []
    expand_state_minus_query: list[float] = []
    order_counts = Counter()
    order_quality_cost_joint = Counter()
    budget_step_counts_min = Counter()
    budget_step_counts_max = Counter()
    unused_budget_counts = Counter()
    sign_transition_counts = Counter()
    path_shift_by_type: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        qid = row["qid"]
        qtype = row.get("question_type", "unknown")
        values = {
            stop: op_value(row, stop, metric),
            refine: op_value(row, refine, metric),
            expand: op_value(row, expand, metric),
            refine_expand: op_value(row, refine_expand, metric),
            expand_refine: op_value(row, expand_refine, metric),
        }
        rewards = {
            stop: op_value(row, stop, "reward"),
            refine: op_value(row, refine, "reward"),
            expand: op_value(row, expand, "reward"),
            refine_expand: op_value(row, refine_expand, "reward"),
            expand_refine: op_value(row, expand_refine, "reward"),
        }
        costs = {
            stop: op_value(row, stop, "cost"),
            refine: op_value(row, refine, "cost"),
            expand: op_value(row, expand, "cost"),
            refine_expand: op_value(row, refine_expand, "cost"),
            expand_refine: op_value(row, expand_refine, "cost"),
        }
        if any(math.isnan(value) for mapping in [values, rewards, costs] for value in mapping.values()):
            continue

        one_step_best = max(values[refine], values[expand])
        two_step_best = max(values[refine_expand], values[expand_refine])
        comp_gain = two_step_best - one_step_best
        comp_gains.append(comp_gain)

        order_reward_diff = rewards[refine_expand] - rewards[expand_refine]
        order_cost_diff = costs[refine_expand] - costs[expand_refine]
        order_reward_diffs.append(order_reward_diff)
        order_cost_diffs.append(order_cost_diff)
        quality_sign = sign(order_reward_diff, eps)
        cost_sign = sign(order_cost_diff, eps)
        order_counts[f"reward_sign_{quality_sign}"] += 1
        order_counts[f"cost_sign_{cost_sign}"] += 1
        order_quality_cost_joint[f"reward_{quality_sign}_cost_{cost_sign}"] += 1

        b0_value, b0_winners = best_set({stop: values[stop]}, eps)
        b1_value, b1_winners = best_set({stop: values[stop], refine: values[refine], expand: values[expand]}, eps)
        b2_value, b2_winners = best_set(values, eps)
        budget_gain_b1 = b1_value - b0_value
        budget_gain_b2 = b2_value - b1_value
        budget_gains_b1_b0.append(budget_gain_b1)
        budget_gains_b2_b1.append(budget_gain_b2)
        min_steps = min(step_count(op) for op in b2_winners)
        max_steps = max(step_count(op) for op in b2_winners)
        budget_step_counts_min[str(min_steps)] += 1
        budget_step_counts_max[str(max_steps)] += 1
        if min_steps < 2:
            unused_budget_counts["B2_can_leave_budget_unused_under_min_step_tie_break"] += 1
        if max_steps < 2:
            unused_budget_counts["B2_all_best_leave_budget_unused"] += 1

        expand_direct = values[expand] - values[stop]
        expand_after_refine = values[refine_expand] - values[refine]
        expand_shift = expand_after_refine - expand_direct
        refine_direct = values[refine] - values[stop]
        refine_after_expand = values[expand_refine] - values[expand]
        refine_shift = refine_after_expand - refine_direct
        delta_expand_direct.append(expand_direct)
        delta_expand_after_refine.append(expand_after_refine)
        delta_expand_path_shift.append(expand_shift)
        delta_refine_direct.append(refine_direct)
        delta_refine_after_expand.append(refine_after_expand)
        delta_refine_path_shift.append(refine_shift)
        path_shift_by_type[qtype].append(max(abs(expand_shift), abs(refine_shift)))
        expand_transition = sign_transition(expand_direct, expand_after_refine, eps)
        refine_transition = sign_transition(refine_direct, refine_after_expand, eps)
        sign_transition_counts[f"expand_after_refine:{expand_transition}"] += 1
        sign_transition_counts[f"refine_after_expand:{refine_transition}"] += 1

        expand_q_value = op_value(row, expand_q, metric)
        expand_q_reward = op_value(row, expand_q, "reward")
        if expand_q and not math.isnan(expand_q_value):
            expand_state_minus_query.append(values[expand] - expand_q_value)

        per_qid.append(
            {
                "qid": qid,
                "question_type": qtype,
                "metric": metric,
                "composition_gain": comp_gain,
                "order_reward_diff_RE_minus_ER": order_reward_diff,
                "order_cost_diff_RE_minus_ER": order_cost_diff,
                "budget_gain_B1_minus_B0": budget_gain_b1,
                "budget_gain_B2_minus_B1": budget_gain_b2,
                "best_B0": "|".join(b0_winners),
                "best_B1": "|".join(b1_winners),
                "best_B2": "|".join(b2_winners),
                "best_B2_min_steps": min_steps,
                "best_B2_max_steps": max_steps,
                "expand_direct_gain": expand_direct,
                "expand_after_refine_gain": expand_after_refine,
                "expand_path_shift": expand_shift,
                "expand_sign_transition_after_refine": expand_transition,
                "refine_direct_gain": refine_direct,
                "refine_after_expand_gain": refine_after_expand,
                "refine_path_shift": refine_shift,
                "refine_sign_transition_after_expand": refine_transition,
                "expand_state_minus_expand_q_metric": values[expand] - expand_q_value if expand_q and not math.isnan(expand_q_value) else math.nan,
                "expand_state_minus_expand_q_reward": rewards[expand] - expand_q_reward if expand_q and not math.isnan(expand_q_reward) else math.nan,
            }
        )

    path_shift_abs = [max(abs(a), abs(b)) for a, b in zip(delta_expand_path_shift, delta_refine_path_shift)]
    summary = {
        "n_states": len(per_qid),
        "metric": metric,
        "operations": {
            "answer_now": stop,
            "shrink": refine,
            "grow": expand,
            "shrink_then_grow": refine_expand,
            "grow_then_shrink": expand_refine,
            "grow_query_only_control": expand_q,
        },
        "ordering_decomposition": {
            "reward_RE_minus_ER": {
                **quantiles(order_reward_diffs),
                "threshold_abs_counts": threshold_counts([abs(v) for v in order_reward_diffs], thresholds),
            },
            "cost_RE_minus_ER": {
                **quantiles(order_cost_diffs),
                "threshold_abs_counts": threshold_counts([abs(v) for v in order_cost_diffs], thresholds),
            },
            "sign_counts": dict(order_counts),
            "quality_cost_joint_sign_counts": dict(order_quality_cost_joint),
        },
        "composition_margin": {
            **quantiles(comp_gains),
            "threshold_counts": threshold_counts(comp_gains, thresholds),
        },
        "budget_redesign": {
            "gain_B1_minus_B0": {
                **quantiles(budget_gains_b1_b0),
                "threshold_counts": threshold_counts(budget_gains_b1_b0, thresholds),
            },
            "gain_B2_minus_B1": {
                **quantiles(budget_gains_b2_b1),
                "threshold_counts": threshold_counts(budget_gains_b2_b1, thresholds),
            },
            "B2_best_min_step_counts": dict(budget_step_counts_min),
            "B2_best_max_step_counts": dict(budget_step_counts_max),
            "unused_budget_counts": dict(unused_budget_counts),
        },
        "path_dependent_marginal_value": {
            "expand_direct_gain": quantiles(delta_expand_direct),
            "expand_after_refine_gain": quantiles(delta_expand_after_refine),
            "expand_path_shift_after_refine_minus_direct": {
                **quantiles(delta_expand_path_shift),
                "threshold_abs_counts": threshold_counts([abs(v) for v in delta_expand_path_shift], thresholds),
            },
            "refine_direct_gain": quantiles(delta_refine_direct),
            "refine_after_expand_gain": quantiles(delta_refine_after_expand),
            "refine_path_shift_after_expand_minus_direct": {
                **quantiles(delta_refine_path_shift),
                "threshold_abs_counts": threshold_counts([abs(v) for v in delta_refine_path_shift], thresholds),
            },
            "any_abs_path_shift": {
                **quantiles(path_shift_abs),
                "threshold_abs_counts": threshold_counts(path_shift_abs, thresholds),
            },
            "sign_transition_counts": dict(sign_transition_counts),
            "strict_polarity_reversal_counts": {
                "expand_value_reversal_after_refine": sum(
                    count
                    for key, count in sign_transition_counts.items()
                    if key.startswith("expand_after_refine:strict_reversal")
                ),
                "refine_value_reversal_after_expand": sum(
                    count
                    for key, count in sign_transition_counts.items()
                    if key.startswith("refine_after_expand:strict_reversal")
                ),
            },
            "zero_boundary_transition_counts": {
                "expand_value_zero_boundary_after_refine": sum(
                    count
                    for key, count in sign_transition_counts.items()
                    if key.startswith("expand_after_refine:zero_boundary")
                ),
                "refine_value_zero_boundary_after_expand": sum(
                    count
                    for key, count in sign_transition_counts.items()
                    if key.startswith("refine_after_expand:zero_boundary")
                ),
            },
            "mean_abs_path_shift_by_type": {qtype: mean(values) for qtype, values in sorted(path_shift_by_type.items())},
        },
        "bootstrap_uncertainty": {
            "seed": bootstrap_seed,
            "n_bootstrap": bootstrap_samples,
            "mean_abs_path_shift_ci": bootstrap_ci_mean(
                path_shift_abs,
                seed=bootstrap_seed,
                n_bootstrap=bootstrap_samples,
            ),
            "proportion_abs_path_shift_gt_0.05_ci": bootstrap_ci_proportion(
                path_shift_abs,
                0.05,
                seed=bootstrap_seed + 1,
                n_bootstrap=bootstrap_samples,
            ),
            "proportion_abs_path_shift_gt_0.10_ci": bootstrap_ci_proportion(
                path_shift_abs,
                0.10,
                seed=bootstrap_seed + 2,
                n_bootstrap=bootstrap_samples,
            ),
            "mean_composition_gain_ci": bootstrap_ci_mean(
                comp_gains,
                seed=bootstrap_seed + 3,
                n_bootstrap=bootstrap_samples,
            ),
            "proportion_composition_gain_gt_0.10_ci": bootstrap_ci_proportion(
                comp_gains,
                0.10,
                seed=bootstrap_seed + 4,
                n_bootstrap=bootstrap_samples,
            ),
        },
        "expand_operator_competence_control": {
            "available": expand_q is not None and len(expand_state_minus_query) > 0,
            "expand_state_minus_expand_q": {
                **quantiles(expand_state_minus_query),
                "threshold_counts": threshold_counts(expand_state_minus_query, thresholds),
            },
        },
        "interpretation_guardrails": [
            "Ordering in reward space supports state-transition semantics; ordering only in cost space supports budgeted scheduling but not semantic order.",
            "Budget gain uses value improvement rather than best-set changes, avoiding super-set tie artifacts.",
            "Unused-budget counts are reported with tie-aware min/max step usage.",
            "Path-dependent marginal value is the cleanest current MDP diagnostic, but remains exploratory under surrogate F1.",
            "Strict polarity reversals are separated from zero-boundary transitions; do not call zero-to-positive/negative movements harmful-beneficial reversals.",
        ],
    }
    return {"summary": summary, "per_qid": per_qid}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-operation", default="stop")
    parser.add_argument("--refine-operation", default="refine")
    parser.add_argument("--expand-operation", default="expand")
    parser.add_argument("--refine-expand-operation", default="refine_expand")
    parser.add_argument("--expand-refine-operation", default="expand_refine")
    parser.add_argument("--expand-q-operation")
    parser.add_argument("--metric", choices=["reward", "utility"], default="reward")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--threshold", type=float, action="append", default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    thresholds = args.threshold if args.threshold is not None else DEFAULT_THRESHOLDS
    result = analyze(
        read_csv(args.wide_matrix),
        stop=args.stop_operation,
        refine=args.refine_operation,
        expand=args.expand_operation,
        refine_expand=args.refine_expand_operation,
        expand_refine=args.expand_refine_operation,
        expand_q=args.expand_q_operation,
        metric=args.metric,
        eps=args.eps,
        thresholds=thresholds,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"p14_mdp_evidence_{args.metric}_summary.json").write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / f"p14_mdp_evidence_{args.metric}_per_qid.csv", result["per_qid"])
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
