#!/usr/bin/env python3
"""P28 static admission oracle-gap analysis.

This is a read-only analysis over an existing operation-value wide matrix.  It
does not run generation, tune prompts, train a controller, or use privileged
metadata as an online feature.  Gold-session labels are already present in the
matrix only for offline grouping.

Purpose:
    Compare original greedy admission against a fixed set of legal heuristic
    C0 -> W admission/repack policies, then quantify:

    - best fixed policy;
    - per-group best fixed policy;
    - per-example static oracle over supplied legal policies;
    - oracle gap after best fixed;
    - Group-B rescue and Group-C disruption tradeoff.

The oracle is an offline upper bound.  It is not a deployable online policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> float:
    if value in {None, "", "nan", "NaN"}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return sum(valid) / len(valid) if valid else math.nan


def bool01(value: Any) -> int | None:
    number = parse_float(value)
    if math.isnan(number):
        return None
    return int(number >= 1.0)


def wtl(delta: float, eps: float) -> str:
    if math.isnan(delta):
        return "missing"
    if delta > eps:
        return "win"
    if delta < -eps:
        return "loss"
    return "tie"


def group_label(row: dict[str, str], c0_col: str) -> str:
    c0_complete = bool01(row.get(c0_col))
    w0_complete = bool01(row.get("stop_all_evidence_present"))
    if c0_complete is None:
        return "UNKNOWN_C0_RETRIEVAL_STATUS"
    if c0_complete == 0:
        return "A_retrieval_incomplete_gold_not_subset_C0"
    if w0_complete == 0:
        return "B_admission_incomplete_gold_in_C0_not_W0"
    if w0_complete == 1:
        return "C_admitted_complete_gold_subset_W0"
    return "UNKNOWN_W0_ADMISSION_STATUS"


def value(row: dict[str, str], policy: str, metric: str) -> float:
    if metric == "reward":
        return parse_float(row.get(f"{policy}_reward"))
    if metric == "proxy_utility_context":
        key = f"{policy}_proxy_utility_context"
        if key not in row:
            key = f"{policy}_utility"
        return parse_float(row.get(key))
    raise ValueError(f"Unsupported metric: {metric}")


def choose_best_policy(row: dict[str, str], policies: list[str], metric: str) -> tuple[str, float]:
    candidates = [(policy, value(row, policy, metric), parse_float(row.get(f"{policy}_cost"))) for policy in policies]
    valid = [(policy, val, cost) for policy, val, cost in candidates if not math.isnan(val)]
    if not valid:
        return "", math.nan
    # Tie-break by higher value, then lower final context cost, then policy order.
    order = {policy: idx for idx, policy in enumerate(policies)}
    best = max(valid, key=lambda item: (item[1], -item[2] if not math.isnan(item[2]) else 0.0, -order[item[0]]))
    return best[0], best[1]


def summarize_policy(rows: list[dict[str, Any]], policy: str, metric: str, eps: float) -> dict[str, Any]:
    baseline_values = [value(row, "stop", metric) for row in rows]
    policy_values = [value(row, policy, metric) for row in rows]
    deltas = [p - b for p, b in zip(policy_values, baseline_values)]
    reward_deltas = [
        parse_float(row.get(f"{policy}_delta_reward_vs_stop"))
        if policy != "stop"
        else 0.0
        for row in rows
    ]
    stop_complete = [bool01(row.get("stop_all_evidence_present")) for row in rows]
    op_complete = [bool01(row.get(f"{policy}_all_evidence_present")) for row in rows]
    c0_complete = [bool01(row.get("_c0_all_evidence_present")) for row in rows]
    rescue_deltas: list[float] = []
    admitted_complete_reward_deltas: list[float] = []
    rescue_count = 0
    preservation_loss_count = 0
    for delta_r, s, o, c in zip(reward_deltas, stop_complete, op_complete, c0_complete):
        if s == 0 and o == 1 and c == 1:
            rescue_count += 1
            rescue_deltas.append(delta_r)
        if s == 1:
            admitted_complete_reward_deltas.append(delta_r)
            if o == 0:
                preservation_loss_count += 1
    return {
        "policy": policy,
        "metric": metric,
        "n": len(rows),
        "mean_value": mean(policy_values),
        "mean_delta_vs_stop": mean(deltas),
        "wins_eps": sum(1 for delta in deltas if wtl(delta, eps) == "win"),
        "ties_eps": sum(1 for delta in deltas if wtl(delta, eps) == "tie"),
        "losses_eps": sum(1 for delta in deltas if wtl(delta, eps) == "loss"),
        "mean_cost": mean([parse_float(row.get(f"{policy}_cost")) for row in rows]),
        "gold_session_rescue_count": rescue_count,
        "gold_session_rescue_precision_eps": mean([float(delta > eps) for delta in rescue_deltas]),
        "gold_session_preservation_loss_count": preservation_loss_count,
        "admitted_complete_disruption_risk_eps": mean(
            [float(delta < -eps) for delta in admitted_complete_reward_deltas]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-matrix", type=Path, required=True)
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--baseline-policy", default="stop")
    parser.add_argument("--metric", choices=["reward", "proxy_utility_context"], default="reward")
    parser.add_argument("--c0-complete-column", default="stop_retrieved_all_evidence_present")
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.wide_matrix)
    if not rows:
        raise ValueError(f"empty wide matrix: {args.wide_matrix}")
    if args.baseline_policy != "stop":
        raise ValueError("P28 currently expects stop as baseline")
    if args.c0_complete_column not in rows[0]:
        raise ValueError(f"Missing C0 completeness column: {args.c0_complete_column}")

    policies = [args.baseline_policy] + [policy for policy in args.policy if policy != args.baseline_policy]
    for row in rows:
        row["_c0_all_evidence_present"] = row.get(args.c0_complete_column, "")
        row["evidence_bottleneck_group"] = group_label(row, args.c0_complete_column)

    baseline_values = [value(row, args.baseline_policy, args.metric) for row in rows]
    fixed_summaries = [summarize_policy(rows, policy, args.metric, args.eps) for policy in policies]
    best_fixed = max(fixed_summaries, key=lambda item: (item["mean_value"], -item["mean_cost"]))

    per_qid: list[dict[str, Any]] = []
    oracle_values: list[float] = []
    best_fixed_values: list[float] = []
    group_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_rows[row["evidence_bottleneck_group"]].append(row)
        oracle_policy, oracle_value = choose_best_policy(row, policies, args.metric)
        best_fixed_value = value(row, best_fixed["policy"], args.metric)
        base_value = value(row, args.baseline_policy, args.metric)
        oracle_values.append(oracle_value)
        best_fixed_values.append(best_fixed_value)
        per_qid.append(
            {
                "qid": row.get("qid"),
                "question_type": row.get("question_type"),
                "evidence_bottleneck_group": row["evidence_bottleneck_group"],
                "metric": args.metric,
                "baseline_policy": args.baseline_policy,
                "best_fixed_policy": best_fixed["policy"],
                "oracle_static_policy": oracle_policy,
                "baseline_value": base_value,
                "best_fixed_value": best_fixed_value,
                "oracle_static_value": oracle_value,
                "oracle_minus_best_fixed": oracle_value - best_fixed_value,
                "oracle_minus_stop": oracle_value - base_value,
            }
        )

    group_summaries: list[dict[str, Any]] = []
    for group, items in sorted(group_rows.items()):
        group_fixed = [summarize_policy(items, policy, args.metric, args.eps) for policy in policies]
        group_best = max(group_fixed, key=lambda item: (item["mean_value"], -item["mean_cost"]))
        group_oracle = [choose_best_policy(row, policies, args.metric)[1] for row in items]
        group_best_values = [value(row, group_best["policy"], args.metric) for row in items]
        group_base = [value(row, args.baseline_policy, args.metric) for row in items]
        group_summaries.append(
            {
                "group": group,
                "metric": args.metric,
                "n": len(items),
                "best_fixed_policy_in_group": group_best["policy"],
                "mean_stop": mean(group_base),
                "mean_group_best_fixed": mean(group_best_values),
                "mean_static_oracle": mean(group_oracle),
                "oracle_minus_group_best_fixed": mean(
                    [o - b for o, b in zip(group_oracle, group_best_values)]
                ),
                "oracle_minus_stop": mean([o - b for o, b in zip(group_oracle, group_base)]),
            }
        )

    policy_choice_counts = Counter(row["oracle_static_policy"] for row in per_qid)
    group_counts = Counter(row["evidence_bottleneck_group"] for row in rows)
    report = {
        "status": "EXPLORATORY_SESSION_LEVEL_STATIC_ORACLE_UPPER_BOUND",
        "wide_matrix": str(args.wide_matrix),
        "metric": args.metric,
        "n": len(rows),
        "policies": policies,
        "best_fixed_policy": best_fixed["policy"],
        "mean_stop": mean(baseline_values),
        "mean_best_fixed": best_fixed["mean_value"],
        "mean_static_oracle": mean(oracle_values),
        "oracle_minus_best_fixed": mean([o - b for o, b in zip(oracle_values, best_fixed_values)]),
        "oracle_minus_stop": mean([o - b for o, b in zip(oracle_values, baseline_values)]),
        "oracle_policy_counts": dict(sorted(policy_choice_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "guardrails": [
            "Static oracle is an offline upper bound, not an online policy.",
            "Evidence groups are gold-session-level, not answer-bearing span-level.",
            "Surrogate F1 is exploratory and must not be reported as official LongMemEval performance.",
            "A nonzero oracle gap supports conditional admission headroom, not RL necessity.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "p28_static_admission_fixed_policy_summary.csv", fixed_summaries)
    write_csv(args.output_dir / "p28_static_admission_group_oracle_summary.csv", group_summaries)
    write_csv(args.output_dir / "p28_static_admission_oracle_per_qid.csv", per_qid)
    (args.output_dir / "p28_static_admission_oracle_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
