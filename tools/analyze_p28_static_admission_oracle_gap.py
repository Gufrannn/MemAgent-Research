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
import random
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


def policy_cost(row: dict[str, str], policy: str) -> float:
    return parse_float(row.get(f"{policy}_cost"))


def validate_policy_columns(rows: list[dict[str, str]], policies: list[str], metric: str) -> None:
    missing: list[str] = []
    for policy in policies:
        value_key = f"{policy}_reward" if metric == "reward" else f"{policy}_proxy_utility_context"
        if metric == "proxy_utility_context" and value_key not in rows[0]:
            value_key = f"{policy}_utility"
        required = [value_key, f"{policy}_cost", f"{policy}_all_evidence_present"]
        for key in required:
            if key not in rows[0]:
                missing.append(key)
    if missing:
        raise ValueError(f"Missing required P28 columns: {sorted(set(missing))}")

    for row in rows:
        qid = row.get("qid", "")
        for policy in policies:
            val = value(row, policy, metric)
            cost = policy_cost(row, policy)
            if math.isnan(val):
                raise ValueError(f"Missing value for policy={policy}, metric={metric}, qid={qid}")
            if math.isnan(cost):
                raise ValueError(
                    f"Missing cost for policy={policy}, qid={qid}. "
                    "P28 tie-break is fail-closed; missing cost cannot receive an oracle advantage."
                )


def choose_best_policy(
    row: dict[str, str],
    policies: list[str],
    metric: str,
    tie_eps: float,
) -> tuple[str, float, float]:
    candidates = [(policy, value(row, policy, metric), policy_cost(row, policy)) for policy in policies]
    if not candidates:
        return "", math.nan, math.nan
    max_value = max(val for _, val, _ in candidates)
    tied = [(policy, val, cost) for policy, val, cost in candidates if max_value - val <= tie_eps]
    # Tie-break by task-equivalent value, then lower final context cost, then policy order.
    order = {policy: idx for idx, policy in enumerate(policies)}
    best = min(tied, key=lambda item: (item[2], order[item[0]]))
    return best[0], best[1], best[2]


def choose_best_summary(summaries: list[dict[str, Any]], policies: list[str], tie_eps: float) -> dict[str, Any]:
    max_value = max(float(item["mean_value"]) for item in summaries)
    tied = [item for item in summaries if max_value - float(item["mean_value"]) <= tie_eps]
    order = {policy: idx for idx, policy in enumerate(policies)}
    return min(tied, key=lambda item: (float(item["mean_cost"]), order[str(item["policy"])]))


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return math.nan
    out = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        out -= p * math.log(p + 1e-12)
    return max(0.0, out)


def bootstrap_ci(values: list[float], samples: int, seed: int) -> dict[str, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    if samples <= 0:
        return {"mean": mean(clean), "ci_low": math.nan, "ci_high": math.nan, "n": len(clean)}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        draw = [clean[rng.randrange(len(clean))] for _ in clean]
        means.append(sum(draw) / len(draw))
    means.sort()
    lo = means[max(0, int(0.025 * samples) - 1)]
    hi = means[min(samples - 1, int(0.975 * samples))]
    return {"mean": mean(clean), "ci_low": lo, "ci_high": hi, "n": len(clean)}


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
    parser.add_argument(
        "--margin-eps",
        action="append",
        type=float,
        default=None,
        help="Margin-aware oracle thresholds against global best fixed. Defaults to 0.05 and 0.10.",
    )
    parser.add_argument(
        "--tie-eps",
        type=float,
        default=0.01,
        help="Task-equivalence tolerance for oracle/fixed tie-breaks before choosing lower cost.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
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
    margin_eps_values = args.margin_eps if args.margin_eps is not None else [0.05, 0.10]
    validate_policy_columns(rows, policies, args.metric)
    for row in rows:
        row["_c0_all_evidence_present"] = row.get(args.c0_complete_column, "")
        row["evidence_bottleneck_group"] = group_label(row, args.c0_complete_column)

    baseline_values = [value(row, args.baseline_policy, args.metric) for row in rows]
    fixed_summaries = [summarize_policy(rows, policy, args.metric, args.eps) for policy in policies]
    best_fixed = choose_best_summary(fixed_summaries, policies, args.tie_eps)

    per_qid: list[dict[str, Any]] = []
    raw_oracle_values: list[float] = []
    best_fixed_values: list[float] = []
    margin_oracle_values: dict[float, list[float]] = {eps: [] for eps in margin_eps_values}
    margin_oracle_policies: dict[float, Counter[str]] = {eps: Counter() for eps in margin_eps_values}
    group_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_rows[row["evidence_bottleneck_group"]].append(row)
        oracle_policy, oracle_value, oracle_cost = choose_best_policy(row, policies, args.metric, args.tie_eps)
        best_fixed_value = value(row, best_fixed["policy"], args.metric)
        base_value = value(row, args.baseline_policy, args.metric)
        raw_oracle_values.append(oracle_value)
        best_fixed_values.append(best_fixed_value)
        out_row = {
            "qid": row.get("qid"),
            "question_type": row.get("question_type"),
            "evidence_bottleneck_group": row["evidence_bottleneck_group"],
            "metric": args.metric,
            "baseline_policy": args.baseline_policy,
            "best_fixed_policy": best_fixed["policy"],
            "raw_oracle_static_policy": oracle_policy,
            "raw_oracle_static_cost": oracle_cost,
            "baseline_value": base_value,
            "best_fixed_value": best_fixed_value,
            "raw_oracle_static_value": oracle_value,
            "raw_oracle_minus_best_fixed": oracle_value - best_fixed_value,
            "raw_oracle_minus_stop": oracle_value - base_value,
        }
        for margin_eps in margin_eps_values:
            if oracle_value - best_fixed_value > margin_eps:
                margin_policy = oracle_policy
                margin_value = oracle_value
            else:
                margin_policy = str(best_fixed["policy"])
                margin_value = best_fixed_value
            margin_oracle_values[margin_eps].append(margin_value)
            margin_oracle_policies[margin_eps][margin_policy] += 1
            suffix = str(margin_eps).replace(".", "p")
            out_row[f"margin_{suffix}_oracle_policy"] = margin_policy
            out_row[f"margin_{suffix}_oracle_value"] = margin_value
            out_row[f"margin_{suffix}_positive_sample"] = int(margin_value - best_fixed_value > 0)
            out_row[f"margin_{suffix}_oracle_minus_best_fixed"] = margin_value - best_fixed_value
            out_row[f"margin_{suffix}_oracle_minus_stop"] = margin_value - base_value
        per_qid.append(out_row)

    group_summaries: list[dict[str, Any]] = []
    for group, items in sorted(group_rows.items()):
        group_fixed = [summarize_policy(items, policy, args.metric, args.eps) for policy in policies]
        group_best = choose_best_summary(group_fixed, policies, args.tie_eps)
        group_oracle_pairs = [choose_best_policy(row, policies, args.metric, args.tie_eps) for row in items]
        group_oracle = [item[1] for item in group_oracle_pairs]
        group_best_values = [value(row, group_best["policy"], args.metric) for row in items]
        global_best_fixed_values = [value(row, best_fixed["policy"], args.metric) for row in items]
        group_base = [value(row, args.baseline_policy, args.metric) for row in items]
        row_out: dict[str, Any] = {
            "group": group,
            "metric": args.metric,
            "n": len(items),
            "best_fixed_policy_in_group": group_best["policy"],
            "global_best_fixed_policy": best_fixed["policy"],
            "mean_stop": mean(group_base),
            "mean_group_best_fixed": mean(group_best_values),
            "mean_global_best_fixed": mean(global_best_fixed_values),
            "mean_raw_static_oracle": mean(group_oracle),
            "raw_oracle_minus_group_best_fixed": mean(
                [o - b for o, b in zip(group_oracle, group_best_values)]
            ),
            "raw_oracle_minus_global_best_fixed": mean(
                [o - b for o, b in zip(group_oracle, global_best_fixed_values)]
            ),
            "raw_oracle_minus_stop": mean([o - b for o, b in zip(group_oracle, group_base)]),
            "raw_positive_margin_sample_count_vs_global_best": sum(
                1 for o, b in zip(group_oracle, global_best_fixed_values) if o - b > 0
            ),
            "raw_oracle_policy_counts": json.dumps(
                dict(sorted(Counter(item[0] for item in group_oracle_pairs).items())),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        for margin_eps in margin_eps_values:
            suffix = str(margin_eps).replace(".", "p")
            margin_values: list[float] = []
            margin_policy_counts: Counter[str] = Counter()
            positive = 0
            for oracle_pair, row_item in zip(group_oracle_pairs, items):
                oracle_policy, oracle_value, _ = oracle_pair
                global_best = value(row_item, best_fixed["policy"], args.metric)
                if oracle_value - global_best > margin_eps:
                    margin_policy_counts[oracle_policy] += 1
                    margin_values.append(oracle_value)
                    positive += 1
                else:
                    margin_policy_counts[str(best_fixed["policy"])] += 1
                    margin_values.append(global_best)
            row_out[f"margin_{suffix}_mean_static_oracle"] = mean(margin_values)
            row_out[f"margin_{suffix}_oracle_minus_global_best_fixed"] = mean(
                [o - b for o, b in zip(margin_values, global_best_fixed_values)]
            )
            row_out[f"margin_{suffix}_oracle_minus_stop"] = mean(
                [o - b for o, b in zip(margin_values, group_base)]
            )
            row_out[f"margin_{suffix}_positive_sample_count"] = positive
            row_out[f"margin_{suffix}_oracle_policy_counts"] = json.dumps(
                dict(sorted(margin_policy_counts.items())), ensure_ascii=False, sort_keys=True
            )
        group_summaries.append(row_out)

    policy_choice_counts = Counter(row["raw_oracle_static_policy"] for row in per_qid)
    group_counts = Counter(row["evidence_bottleneck_group"] for row in rows)
    raw_gaps = [o - b for o, b in zip(raw_oracle_values, best_fixed_values)]
    margin_reports: list[dict[str, Any]] = []
    for margin_eps in margin_eps_values:
        margin_values = margin_oracle_values[margin_eps]
        margin_gaps = [o - b for o, b in zip(margin_values, best_fixed_values)]
        margin_stop_gaps = [o - b for o, b in zip(margin_values, baseline_values)]
        ci = bootstrap_ci(margin_gaps, args.bootstrap_samples, args.seed + int(margin_eps * 10000))
        margin_reports.append(
            {
                "margin_eps": margin_eps,
                "metric": args.metric,
                "mean_margin_static_oracle": mean(margin_values),
                "margin_oracle_minus_best_fixed": mean(margin_gaps),
                "margin_oracle_minus_best_fixed_ci_low": ci["ci_low"],
                "margin_oracle_minus_best_fixed_ci_high": ci["ci_high"],
                "margin_oracle_minus_stop": mean(margin_stop_gaps),
                "positive_margin_sample_count": sum(1 for gap in margin_gaps if gap > 0),
                "positive_margin_sample_rate": mean([float(gap > 0) for gap in margin_gaps]),
                "oracle_policy_entropy": entropy(margin_oracle_policies[margin_eps]),
                "oracle_policy_counts": json.dumps(
                    dict(sorted(margin_oracle_policies[margin_eps].items())),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    raw_ci = bootstrap_ci(raw_gaps, args.bootstrap_samples, args.seed)
    report = {
        "status": "EXPLORATORY_SESSION_LEVEL_MARGIN_AWARE_STATIC_ORACLE_UPPER_BOUND",
        "wide_matrix": str(args.wide_matrix),
        "metric": args.metric,
        "n": len(rows),
        "policies": policies,
        "tie_eps": args.tie_eps,
        "margin_eps_values": margin_eps_values,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "best_fixed_policy": best_fixed["policy"],
        "mean_stop": mean(baseline_values),
        "mean_best_fixed": best_fixed["mean_value"],
        "mean_raw_static_oracle": mean(raw_oracle_values),
        "raw_oracle_minus_best_fixed": mean(raw_gaps),
        "raw_oracle_minus_best_fixed_ci_low": raw_ci["ci_low"],
        "raw_oracle_minus_best_fixed_ci_high": raw_ci["ci_high"],
        "raw_oracle_minus_stop": mean([o - b for o, b in zip(raw_oracle_values, baseline_values)]),
        "raw_positive_margin_sample_count_vs_best_fixed": sum(1 for gap in raw_gaps if gap > 0),
        "raw_oracle_policy_entropy": entropy(policy_choice_counts),
        "raw_oracle_policy_counts": dict(sorted(policy_choice_counts.items())),
        "margin_oracle_reports": margin_reports,
        "group_counts": dict(sorted(group_counts.items())),
        "guardrails": [
            "Static oracle is an offline upper bound, not an online policy.",
            "Raw per-example max can overstate learnable headroom under noisy surrogate F1.",
            "Margin-aware oracle reports are the gate-relevant quantities.",
            "Tie-equivalent values are resolved by lower final context cost and then fixed policy order.",
            "Missing policy costs fail closed before oracle selection.",
            "Evidence groups are gold-session-level, not answer-bearing span-level.",
            "Surrogate F1 is exploratory and must not be reported as official LongMemEval performance.",
            "A nonzero oracle gap supports conditional admission headroom, not RL necessity.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "p28_static_admission_fixed_policy_summary.csv", fixed_summaries)
    write_csv(args.output_dir / "p28_static_admission_group_oracle_summary.csv", group_summaries)
    write_csv(args.output_dir / "p28_static_admission_margin_oracle_summary.csv", margin_reports)
    write_csv(args.output_dir / "p28_static_admission_oracle_per_qid.csv", per_qid)
    (args.output_dir / "p28_static_admission_oracle_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
