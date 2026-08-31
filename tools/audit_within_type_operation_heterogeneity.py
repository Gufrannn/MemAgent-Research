#!/usr/bin/env python3
"""Audit whether operation-value heterogeneity survives within question types.

This script separates two possibilities:

1. task-type prior: each benchmark question_type mostly prefers a fixed action;
2. state-dependent heterogeneity: positive and negative operation values coexist
   inside the same question_type.

The script uses question_type only for offline analysis. It must not be treated
as an online policy feature unless explicitly marked privileged elsewhere.

The ANOVA-style decomposition emitted here is descriptive. It reports sums of
squares and eta-squared, not a predictive R^2. The within-type share should be
read as "variation not accounted for by benchmark type labels", not as direct
evidence-state variance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / max(1, len(values))


def variance(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return sum((value - mu) ** 2 for value in values) / (len(values) - 1)


def sign(value: float, eps: float) -> str:
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "zero"


def column_value(row: dict[str, str], operation: str, metric: str) -> float:
    return to_float(row.get(f"{operation}_{metric}"))


def column_cost(row: dict[str, str], operation: str) -> float:
    return to_float(row.get(f"{operation}_cost"))


def threshold_stats(deltas: list[float], threshold: float) -> dict[str, Any]:
    n = len(deltas)
    pos = sum(1 for value in deltas if value > threshold)
    neg = sum(1 for value in deltas if value < -threshold)
    neutral = n - pos - neg
    return {
        f"positive_gt_{threshold:g}_count": pos,
        f"negative_lt_minus_{threshold:g}_count": neg,
        f"neutral_abs_le_{threshold:g}_count": neutral,
        f"p_positive_gt_{threshold:g}": pos / max(1, n),
        f"p_negative_lt_minus_{threshold:g}": neg / max(1, n),
        f"p_neutral_abs_le_{threshold:g}": neutral / max(1, n),
        f"nontrivial_mixed_pos_neg_{threshold:g}": pos > 0 and neg > 0,
    }


def ss_decomposition_for_groups(grouped_values: dict[str, list[float]]) -> dict[str, float]:
    all_values = [value for values in grouped_values.values() for value in values]
    if not all_values:
        return {
            "n": 0,
            "n_types": 0,
            "total_ss": math.nan,
            "between_type_ss": math.nan,
            "within_type_ss": math.nan,
            "type_eta_squared": math.nan,
            "within_share": math.nan,
            "within_mse": math.nan,
            "total_sample_variance": math.nan,
        }
    global_mean = mean(all_values)
    sst = sum((value - global_mean) ** 2 for value in all_values)
    ssb = 0.0
    ssw = 0.0
    for values in grouped_values.values():
        type_mean = mean(values)
        ssb += len(values) * (type_mean - global_mean) ** 2
        ssw += sum((value - type_mean) ** 2 for value in values)
    n = len(all_values)
    n_types = len(grouped_values)
    return {
        "n": n,
        "n_types": n_types,
        "total_ss": sst,
        "between_type_ss": ssb,
        "within_type_ss": ssw,
        "type_eta_squared": ssb / sst if sst > 0 else 0.0,
        "within_share": ssw / sst if sst > 0 else 0.0,
        "within_mse": ssw / max(1, n - n_types),
        "total_sample_variance": sst / max(1, n - 1),
    }


def audit(
    rows: list[dict[str, str]],
    *,
    baseline_operation: str,
    operations: list[str],
    metric: str,
    eps: float,
    thresholds: list[float],
    operation_access: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    long_rows: list[dict[str, Any]] = []
    for row in rows:
        qtype = row.get("question_type") or "unknown"
        qid = row.get("qid") or row.get("raw_qid") or ""
        base_value = column_value(row, baseline_operation, metric)
        if math.isnan(base_value):
            continue
        for operation in operations:
            op_value = column_value(row, operation, metric)
            if math.isnan(op_value):
                continue
            delta = op_value - base_value
            grouped[(qtype, operation)].append(delta)
            long_rows.append(
                {
                    "qid": qid,
                    "question_type": qtype,
                    "baseline_operation": baseline_operation,
                    "operation": operation,
                    "operation_access": operation_access,
                    "metric": metric,
                    "baseline_value": base_value,
                    "operation_value": op_value,
                    "delta": delta,
                    "delta_sign": sign(delta, eps),
                    "baseline_cost": column_cost(row, baseline_operation),
                    "operation_cost": column_cost(row, operation),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    mixed_type_operation = 0
    total_type_operation = 0
    for (qtype, operation), deltas in sorted(grouped.items()):
        n = len(deltas)
        pos = sum(1 for value in deltas if value > eps)
        neg = sum(1 for value in deltas if value < -eps)
        zero = n - pos - neg
        mixed = pos > 0 and neg > 0
        mixed_type_operation += int(mixed)
        total_type_operation += 1
        summary_rows.append(
            {
                "question_type": qtype,
                "operation": operation,
                "operation_access": operation_access,
                "metric": metric,
                "n": n,
                "positive_count": pos,
                "negative_count": neg,
                "zero_count": zero,
                "p_positive": pos / max(1, n),
                "p_negative": neg / max(1, n),
                "p_zero": zero / max(1, n),
                "mean_delta": mean(deltas),
                "var_delta": variance(deltas),
                "min_delta": min(deltas) if deltas else math.nan,
                "max_delta": max(deltas) if deltas else math.nan,
                "mixed_positive_negative": mixed,
                **{
                    key: value
                    for threshold in thresholds
                    for key, value in threshold_stats(deltas, threshold).items()
                },
            }
        )

    all_deltas_by_op: dict[str, list[float]] = defaultdict(list)
    values_by_op_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (qtype, operation), deltas in grouped.items():
        all_deltas_by_op[operation].extend(deltas)
        values_by_op_type[operation][qtype].extend(deltas)

    ss_decomposition = {}
    operation_summary_rows: list[dict[str, Any]] = []
    for operation, deltas in all_deltas_by_op.items():
        decomp = ss_decomposition_for_groups(dict(values_by_op_type[operation]))
        ss_decomposition[operation] = decomp
        operation_row: dict[str, Any] = {
            "operation": operation,
            "operation_access": operation_access,
            "metric": metric,
            "n": len(deltas),
            "n_types": decomp["n_types"],
            "mean_delta": mean(deltas),
            "total_ss": decomp["total_ss"],
            "between_type_ss": decomp["between_type_ss"],
            "within_type_ss": decomp["within_type_ss"],
            "type_eta_squared": decomp["type_eta_squared"],
            "within_share": decomp["within_share"],
            "within_mse": decomp["within_mse"],
            "total_sample_variance": decomp["total_sample_variance"],
        }
        for threshold in thresholds:
            cell_stats = [
                threshold_stats(values, threshold)
                for values in values_by_op_type[operation].values()
            ]
            pos_key = f"positive_gt_{threshold:g}_count"
            neg_key = f"negative_lt_minus_{threshold:g}_count"
            mixed_key = f"nontrivial_mixed_pos_neg_{threshold:g}"
            operation_row.update(
                {
                    f"within_type_positive_tail_cells_gt_{threshold:g}": sum(1 for stats in cell_stats if stats[pos_key] > 0),
                    f"within_type_negative_tail_cells_lt_minus_{threshold:g}": sum(1 for stats in cell_stats if stats[neg_key] > 0),
                    f"within_type_nontrivial_mixed_cells_{threshold:g}": sum(1 for stats in cell_stats if stats[mixed_key]),
                    f"within_type_nontrivial_mixed_cell_rate_{threshold:g}": (
                        sum(1 for stats in cell_stats if stats[mixed_key]) / max(1, len(cell_stats))
                    ),
                    f"global_positive_tail_rate_gt_{threshold:g}": sum(1 for value in deltas if value > threshold) / max(1, len(deltas)),
                    f"global_negative_tail_rate_lt_minus_{threshold:g}": sum(1 for value in deltas if value < -threshold) / max(1, len(deltas)),
                }
            )
        operation_summary_rows.append(operation_row)

    summary = {
        "metric": metric,
        "baseline_operation": baseline_operation,
        "operations": operations,
        "operation_access": operation_access,
        "n_long_rows": len(long_rows),
        "n_type_operation_cells": total_type_operation,
        "mixed_positive_negative_cells": mixed_type_operation,
        "mixed_positive_negative_cell_rate": mixed_type_operation / max(1, total_type_operation),
        "thresholds": thresholds,
        "ss_decomposition_by_operation": ss_decomposition,
        "interpretation_guardrails": [
            "question_type is used only for offline heterogeneity analysis.",
            "Tiny sign flips can be noise; prefer thresholded non-trivial tails for claims.",
            "Reward-space heterogeneity is the primary scientific phenomenon; utility-space heterogeneity additionally includes cost.",
            "Mixed non-trivial tails within a type support heterogeneity beyond a pure type prior.",
            "Uniform signs within a type suggest a strong task-type prior but do not by themselves prove leakage.",
            "type_eta_squared is a descriptive ANOVA effect size, not a predictive R^2.",
            "within_share is variation not accounted for by benchmark type labels; do not call it evidence-state variance without a state-prediction test.",
            "Rows marked privileged_upper_bound are diagnostic controls and must not be merged into the legal online-operation main table.",
        ],
    }
    return summary_rows, operation_summary_rows, {"summary": summary, "long_rows": long_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-operation", default="stop")
    parser.add_argument("--operations", nargs="+", required=True)
    parser.add_argument(
        "--operation-access",
        choices=["legal_online_operation", "privileged_upper_bound"],
        default="legal_online_operation",
        help="Protocol status of the audited operations; oracle/evidence-gold controls should be privileged_upper_bound.",
    )
    parser.add_argument("--metric", choices=["reward", "utility"], default="utility")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.1])
    args = parser.parse_args()

    summary_rows, operation_summary_rows, payload = audit(
        read_csv(args.wide_matrix),
        baseline_operation=args.baseline_operation,
        operations=args.operations,
        metric=args.metric,
        eps=args.eps,
        thresholds=args.thresholds,
        operation_access=args.operation_access,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"within_type_heterogeneity_{args.metric}_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / f"within_type_heterogeneity_{args.metric}_operation_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operation_summary_rows[0]))
        writer.writeheader()
        writer.writerows(operation_summary_rows)
    with (args.output_dir / f"within_type_heterogeneity_{args.metric}_long.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["long_rows"][0]))
        writer.writeheader()
        writer.writerows(payload["long_rows"])
    (args.output_dir / f"within_type_heterogeneity_{args.metric}_audit.json").write_text(
        json.dumps(payload["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
