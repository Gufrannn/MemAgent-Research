#!/usr/bin/env python3
"""P26 admission-vs-transformation factorization for memory operations.

This is an offline analysis over frozen generation traces.  It does not train
or tune a controller.  Gold evidence labels are used only after generation to
separate retrieval availability, context-budget admission, and transformation
effects:

    C0 = initial retrieved candidate pool
    W0 = initial admitted working memory visible to the reader

The core comparison is:

    stop              : answer from W0
    shrink_visible    : transform W0 only; no new sources allowed
    repack_candidates : reselect/repack from fixed C0 into W1

Any operation that can access C0 but not just W0 is treated as an admission
operation, not as pure visible-state compression.
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


def wtl(delta: float, eps: float) -> str:
    if math.isnan(delta):
        return "missing"
    if delta > eps:
        return "win"
    if delta < -eps:
        return "loss"
    return "tie"


def bool01(value: Any) -> int | None:
    number = parse_float(value)
    if math.isnan(number):
        return None
    return int(number >= 1.0)


def evidence_retrieval_group(row: dict[str, str]) -> str:
    """Classify each qid by initial retrieval and initial admitted evidence.

    The matrix builder reports final-operation admitted evidence.  For stop,
    this is W0.  ``stop_evidence_recall`` is therefore admitted recall.  When
    available, ``stop_retrieved_evidence_recall`` should represent C0 recall.
    Current builders may not yet expose C0 recall, so we fail closed unless the
    caller explicitly supplies a column name.
    """

    c0_complete = bool01(row.get("_c0_all_evidence_present"))
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


def summarize(rows: list[dict[str, Any]], group_key: str, operations: list[str], eps: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for op in operations:
            grouped[(str(row.get(group_key, "ALL")), op)].append(row)

    out: list[dict[str, Any]] = []
    for (group, op), items in sorted(grouped.items()):
        deltas = [parse_float(item.get(f"{op}_delta_reward_vs_stop")) for item in items]
        proxy_utility_key = f"{op}_delta_proxy_utility_context_vs_stop"
        if proxy_utility_key not in items[0]:
            proxy_utility_key = f"{op}_delta_utility_vs_stop"
        utility_deltas = [parse_float(item.get(proxy_utility_key)) for item in items]
        stop_costs = [parse_float(item.get("stop_cost")) for item in items]
        op_costs = [parse_float(item.get(f"{op}_cost")) for item in items]
        stop_complete = [bool01(item.get("stop_all_evidence_present")) for item in items]
        op_complete = [bool01(item.get(f"{op}_all_evidence_present")) for item in items]
        c0_complete = [bool01(item.get("_c0_all_evidence_present")) for item in items]

        admission_rescue: list[bool] = []
        admission_rescue_deltas: list[float] = []
        preservation_loss: list[bool] = []
        admitted_complete_deltas: list[float] = []
        for delta, s, o, c in zip(deltas, stop_complete, op_complete, c0_complete):
            if s is not None and o is not None and c is not None:
                rescued = s == 0 and o == 1 and c == 1
                admission_rescue.append(rescued)
                if rescued:
                    admission_rescue_deltas.append(delta)
            if s is not None and o is not None:
                lost = s == 1 and o == 0
                preservation_loss.append(lost)
                if s == 1:
                    admitted_complete_deltas.append(delta)
        wins = sum(1 for delta in deltas if wtl(delta, eps) == "win")
        ties = sum(1 for delta in deltas if wtl(delta, eps) == "tie")
        losses = sum(1 for delta in deltas if wtl(delta, eps) == "loss")
        out.append(
            {
                "group_key": group_key,
                "group": group,
                "operation": op,
                "n": len(items),
                "mean_delta_reward_vs_stop": mean(deltas),
                "mean_delta_proxy_utility_context_vs_stop": mean(utility_deltas),
                "mean_stop_context_cost": mean(stop_costs),
                "mean_op_context_cost": mean(op_costs),
                "mean_delta_context_cost": mean(
                    [op_cost - stop_cost for op_cost, stop_cost in zip(op_costs, stop_costs)]
                ),
                "wins_eps": wins,
                "ties_eps": ties,
                "losses_eps": losses,
                "admission_rescue_count": sum(admission_rescue),
                "admission_rescue_rate": mean([float(x) for x in admission_rescue]),
                "gold_session_rescue_precision_eps": mean(
                    [float(delta > eps) for delta in admission_rescue_deltas]
                ),
                "preservation_loss_count": sum(preservation_loss),
                "preservation_loss_rate": mean([float(x) for x in preservation_loss]),
                "admitted_complete_disruption_risk_eps": mean(
                    [float(delta < -eps) for delta in admitted_complete_deltas]
                ),
                "mean_stop_all_evidence_present": mean(
                    [float(x) for x in stop_complete if x is not None]
                ),
                "mean_op_all_evidence_present": mean(
                    [float(x) for x in op_complete if x is not None]
                ),
                "mean_c0_all_evidence_present": mean(
                    [float(x) for x in c0_complete if x is not None]
                ),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-matrix", type=Path, required=True)
    parser.add_argument(
        "--c0-complete-column",
        default="stop_retrieved_all_evidence_present",
        help=(
            "Column containing whether gold evidence is present in C0. "
            "If absent, P26 exits unless --allow-missing-c0 is set."
        ),
    )
    parser.add_argument("--baseline-operation", default="stop")
    parser.add_argument("--operation", action="append", required=True)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--allow-missing-c0", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.baseline_operation != "stop":
        raise ValueError("P26 factorization currently expects stop as baseline_operation")

    rows = read_csv(args.wide_matrix)
    if not rows:
        raise ValueError(f"empty wide matrix: {args.wide_matrix}")

    has_c0 = args.c0_complete_column in rows[0]
    if not has_c0 and not args.allow_missing_c0:
        raise ValueError(
            f"Missing C0 retrieval-completeness column '{args.c0_complete_column}'. "
            "Do not conflate C0 availability with W0 admission. Rebuild the matrix "
            "with retrieved-source evidence diagnostics, or rerun with "
            "--allow-missing-c0 only for legacy exploratory output."
        )

    enriched: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = dict(row)
        out["_c0_all_evidence_present"] = row.get(args.c0_complete_column, "")
        out["evidence_bottleneck_group"] = evidence_retrieval_group(out)
        for op in args.operation:
            delta = parse_float(row.get(f"{op}_delta_reward_vs_stop"))
            out[f"{op}_primary_wtl_eps_{args.eps}"] = wtl(delta, args.eps)
            stop_complete = bool01(row.get("stop_all_evidence_present"))
            op_complete = bool01(row.get(f"{op}_all_evidence_present"))
            c0_complete = bool01(out.get("_c0_all_evidence_present"))
            out[f"{op}_admission_rescue"] = int(stop_complete == 0 and op_complete == 1 and c0_complete == 1)
            out[f"{op}_preservation_loss"] = int(stop_complete == 1 and op_complete == 0)
            out[f"{op}_gold_session_rescue_win_eps_{args.eps}"] = int(
                out[f"{op}_admission_rescue"] == 1 and delta > args.eps
            )
            out[f"{op}_admitted_complete_disruption_eps_{args.eps}"] = int(
                stop_complete == 1 and delta < -args.eps
            )
        enriched.append(out)

    group_counts = Counter(row["evidence_bottleneck_group"] for row in enriched)
    summaries = []
    summaries.extend(summarize(enriched, "__all__", args.operation, args.eps))
    summaries.extend(summarize(enriched, "evidence_bottleneck_group", args.operation, args.eps))
    if "question_type" in enriched[0]:
        summaries.extend(summarize(enriched, "question_type", args.operation, args.eps))

    for row in enriched:
        row["__all__"] = "ALL"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "p26_admission_transformation_per_qid.csv", enriched)
    write_csv(args.output_dir / "p26_admission_transformation_summary.csv", summaries)
    report = {
        "status": "EXPLORATORY",
        "wide_matrix": str(args.wide_matrix),
        "n": len(enriched),
        "operations": args.operation,
        "eps": args.eps,
        "c0_complete_column": args.c0_complete_column if has_c0 else None,
        "has_c0_retrieval_completeness": has_c0,
        "evidence_bottleneck_group_counts": dict(sorted(group_counts.items())),
        "guardrails": [
            "Gold labels are used only after frozen generation for mechanism decomposition.",
            "SHRINK_VISIBLE should be interpreted as W_t -> W_{t+1} with no new sources.",
            "REPACK_CANDIDATES should be interpreted as fixed C0 -> new W_{t+1} admission.",
            "If C0 retrieval completeness is unavailable, retrieval vs admission cannot be separated.",
            "Surrogate F1 is exploratory and must not be written as official LongMemEval performance.",
        ],
    }
    (args.output_dir / "p26_admission_transformation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
