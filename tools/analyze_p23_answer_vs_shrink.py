#!/usr/bin/env python3
"""P23 ANSWER-vs-SHRINK audit for LongMemEval.

This is an offline analysis script.  Gold evidence labels and benchmark
``question_type`` are used only to audit mechanisms after frozen generations
exist.  They must not be treated as online controller features.

Main question:

    Is SHRINK/REFINE merely a universally good preprocessing step, or does it
    preserve non-trivial wins and losses relative to ANSWER/STOP?

The script reports W/T/L under multiple thresholds, by question type and by
gold-mechanism bins.  It can be run independently for each model scale and the
outputs can later be joined by qid for model-conditioning analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DATE_RE = re.compile(r"^(\d{4}/\d{2}/\d{2}) \([^)]+\) (\d{2}:\d{2})$")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path or not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_qid(qid: str) -> str:
    qid = str(qid)
    return qid if qid.startswith("longmemeval_") else f"longmemeval_{qid}"


def raw_qid(qid: str) -> str:
    qid = str(qid)
    return qid[len("longmemeval_") :] if qid.startswith("longmemeval_") else qid


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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


def pearson(xs: list[float], ys: list[float]) -> float:
    paired = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(paired) <= 1:
        return math.nan
    xvals = [x for x, _ in paired]
    yvals = [y for _, y in paired]
    xmu = mean(xvals)
    ymu = mean(yvals)
    num = sum((x - xmu) * (y - ymu) for x, y in paired)
    denx = math.sqrt(sum((x - xmu) ** 2 for x in xvals))
    deny = math.sqrt(sum((y - ymu) ** 2 for y in yvals))
    return num / (denx * deny) if denx > 0 and deny > 0 else math.nan


def sign_label(delta: float, eps: float) -> str:
    if delta > eps:
        return "win"
    if delta < -eps:
        return "loss"
    return "tie"


def wtl_stats(deltas: list[float], eps: float) -> dict[str, Any]:
    n = len([d for d in deltas if not math.isnan(d)])
    wins = sum(1 for value in deltas if value > eps)
    losses = sum(1 for value in deltas if value < -eps)
    ties = n - wins - losses
    return {
        f"wins_eps_{eps:g}": wins,
        f"ties_eps_{eps:g}": ties,
        f"losses_eps_{eps:g}": losses,
        f"win_rate_eps_{eps:g}": wins / max(1, n),
        f"tie_rate_eps_{eps:g}": ties / max(1, n),
        f"loss_rate_eps_{eps:g}": losses / max(1, n),
        f"mixed_win_loss_eps_{eps:g}": int(wins > 0 and losses > 0),
    }


def parse_date(value: Any) -> datetime | None:
    text = str(value or "")
    match = DATE_RE.match(text)
    if not match:
        return None
    return datetime.strptime(" ".join(match.groups()), "%Y/%m/%d %H:%M")


def load_manifest_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = read_json(path)
    if isinstance(payload, dict):
        ids = payload.get("question_ids") or [item.get("question_id") for item in payload.get("items", [])]
    elif isinstance(payload, list):
        ids = payload
    else:
        raise ValueError(f"Unsupported manifest format: {path}")
    return {normalize_qid(str(qid)) for qid in ids if qid}


def load_raw_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("LongMemEval raw file must be a JSON list")
    return {normalize_qid(str(row.get("question_id") or "")): row for row in payload}


def nonempty_session_ids(raw_row: dict[str, Any]) -> list[str]:
    session_ids = raw_row.get("haystack_session_ids") or []
    sessions = raw_row.get("haystack_sessions") or []
    out: list[str] = []
    for idx, session in enumerate(sessions):
        if not session:
            continue
        if idx < len(session_ids):
            out.append(str(session_ids[idx]))
        else:
            out.append(str(idx))
    return out


def load_response_query_hashes(path: Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if path is None or not path.exists():
        return out
    for row in iter_jsonl(path):
        qid = normalize_qid(str(row.get("qid") or row.get("question_id") or ""))
        query = str(row.get("query") or "")
        if qid and query:
            out[qid] = sha1_text(query)
    return out


def latest_op_record(trace_row: dict[str, Any], allowed_ops: set[str]) -> dict[str, Any] | None:
    for record in reversed(trace_row.get("op_records") or []):
        if record.get("operation") in allowed_ops:
            return record
    return None


def load_trace_by_hash(path: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if path is None or not path.exists():
        return out
    for row in iter_jsonl(path):
        if row.get("phase") == "qa" and row.get("query_sha1"):
            out[str(row["query_sha1"])] = row
    return out


def initial_retrieval_features(
    *,
    qid: str,
    raw_row: dict[str, Any],
    response_hash_by_qid: dict[str, str],
    trace_by_hash: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    query_hash = response_hash_by_qid.get(qid)
    trace_row = trace_by_hash.get(query_hash or "")
    session_ids = nonempty_session_ids(raw_row)
    gold = {str(item) for item in (raw_row.get("answer_session_ids") or [])}
    out: dict[str, Any] = {
        "stop_trace_found": int(trace_row is not None),
        "stop_trace_query_sha1": query_hash or "",
        "initial_selected_session_count": math.nan,
        "initial_selected_gold_count": math.nan,
        "initial_selected_distractor_count": math.nan,
        "initial_trace_evidence_recall": math.nan,
        "initial_trace_all_gold_present": math.nan,
    }
    if not trace_row:
        return out
    record = latest_op_record(trace_row, {"RETRIEVE", "RETRIEVE_RECENT", "FILTER", "REFINE", "EXPAND", "RETRIEVE_MORE"})
    if not record:
        return out
    raw_indices = record.get("selected_indices")
    if raw_indices is None:
        raw_indices = record.get("admitted_source_indices") or []
    selected_ids: set[str] = set()
    for idx in raw_indices:
        if isinstance(idx, str) and idx.isdigit():
            idx = int(idx)
        if isinstance(idx, int) and 0 <= idx < len(session_ids):
            selected_ids.add(session_ids[idx])
    gold_hits = selected_ids & gold
    out["initial_selected_session_count"] = len(selected_ids)
    out["initial_selected_gold_count"] = len(gold_hits)
    out["initial_selected_distractor_count"] = max(0, len(selected_ids) - len(gold_hits))
    if gold:
        out["initial_trace_evidence_recall"] = len(gold_hits) / len(gold)
        out["initial_trace_all_gold_present"] = 1.0 if gold <= selected_ids else 0.0
    return out


def gold_mechanism_features(raw_row: dict[str, Any]) -> dict[str, Any]:
    session_ids = [str(x) for x in raw_row.get("haystack_session_ids") or []]
    gold = [str(x) for x in raw_row.get("answer_session_ids") or []]
    total_sessions = len(session_ids)
    gold_positions = [session_ids.index(x) for x in gold if x in session_ids]
    qdate = parse_date(raw_row.get("question_date"))
    gold_dates = [
        parse_date(raw_row.get("haystack_dates", [])[pos])
        for pos in gold_positions
        if pos < len(raw_row.get("haystack_dates") or [])
    ]
    gold_dates = [dt for dt in gold_dates if dt is not None]
    if gold_positions:
        position_span = max(gold_positions) - min(gold_positions) + 1
        position_density = len(gold_positions) / max(1, position_span)
    else:
        position_span = math.nan
        position_density = math.nan
    if len(gold_dates) >= 2:
        temporal_spread_days = (max(gold_dates) - min(gold_dates)).total_seconds() / 86400.0
    else:
        temporal_spread_days = 0.0 if gold_dates else math.nan
    if qdate and gold_dates:
        latest_gold_age_days = (qdate - max(gold_dates)).total_seconds() / 86400.0
    else:
        latest_gold_age_days = math.nan
    relevant_count = len(gold)
    return {
        "gold_relevant_session_count": relevant_count,
        "gold_present_in_haystack_count": len(gold_positions),
        "haystack_session_count": total_sessions,
        "gold_concentration": relevant_count / total_sessions if total_sessions else math.nan,
        "gold_position_min": min(gold_positions) if gold_positions else math.nan,
        "gold_position_max": max(gold_positions) if gold_positions else math.nan,
        "gold_position_span": position_span,
        "gold_position_density": position_density,
        "gold_temporal_spread_days": temporal_spread_days,
        "latest_gold_age_days": latest_gold_age_days,
        "gold_relevant_count_bin": "0" if relevant_count == 0 else "1" if relevant_count == 1 else "2" if relevant_count == 2 else "3plus",
        "gold_position_span_bin": bin_numeric(position_span, [(1, "span1"), (5, "span2to5"), (20, "span6to20")], "span_gt20"),
        "gold_temporal_spread_bin": bin_numeric(
            temporal_spread_days,
            [(0.01, "same_time"), (1, "within_day"), (7, "within_week"), (30, "within_month")],
            "over_month",
        ),
        "gold_latest_age_bin": bin_numeric(
            latest_gold_age_days,
            [(1, "age_within_day"), (7, "age_within_week"), (30, "age_within_month")],
            "age_over_month",
        ),
    }


def bin_numeric(value: float, cutoffs: list[tuple[float, str]], overflow: str) -> str:
    if value is None or math.isnan(float(value)):
        return "missing"
    for cutoff, label in cutoffs:
        if float(value) <= cutoff:
            return label
    return overflow


def group_summary(rows: list[dict[str, Any]], group_field: str, eps_values: list[float]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field, "missing"))].append(row)
    out: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items()):
        deltas = [float(row["delta_reward"]) for row in items]
        utility_deltas = [float(row["delta_utility"]) for row in items]
        out_row: dict[str, Any] = {
            "group_field": group_field,
            "group_value": group,
            "n": len(items),
            "mean_stop_reward": mean([float(row["stop_reward"]) for row in items]),
            "mean_shrink_reward": mean([float(row["shrink_reward"]) for row in items]),
            "mean_delta_reward": mean(deltas),
            "var_delta_reward": variance(deltas),
            "mean_delta_utility": mean(utility_deltas),
            "mean_delta_cost": mean([float(row["delta_cost"]) for row in items]),
            "mean_stop_evidence_recall": mean([to_float(row["stop_evidence_recall"]) for row in items]),
            "mean_shrink_evidence_recall": mean([to_float(row["shrink_evidence_recall"]) for row in items]),
        }
        for eps in eps_values:
            out_row.update(wtl_stats(deltas, eps))
        out.append(out_row)
    return out


def attach_rows(
    *,
    wide_rows: list[dict[str, str]],
    raw_rows: dict[str, dict[str, Any]],
    manifest_ids: set[str] | None,
    model_label: str,
    baseline_operation: str,
    shrink_operation: str,
    stop_response: Path | None,
    stop_trace: Path | None,
    eps_primary: float,
) -> list[dict[str, Any]]:
    response_hash_by_qid = load_response_query_hashes(stop_response)
    trace_by_hash = load_trace_by_hash(stop_trace)
    out: list[dict[str, Any]] = []
    for row in wide_rows:
        qid = normalize_qid(row.get("qid") or row.get("raw_qid") or "")
        if manifest_ids is not None and qid not in manifest_ids:
            continue
        raw_row = raw_rows.get(qid)
        if not raw_row:
            raise ValueError(f"Missing raw LongMemEval row for {qid}")
        stop_reward = to_float(row.get(f"{baseline_operation}_reward"))
        shrink_reward = to_float(row.get(f"{shrink_operation}_reward"))
        stop_cost = to_float(row.get(f"{baseline_operation}_cost"))
        shrink_cost = to_float(row.get(f"{shrink_operation}_cost"))
        stop_utility = to_float(row.get(f"{baseline_operation}_utility"))
        shrink_utility = to_float(row.get(f"{shrink_operation}_utility"))
        delta_reward = shrink_reward - stop_reward
        delta_utility = shrink_utility - stop_utility
        gold_features = gold_mechanism_features(raw_row)
        retrieval_features = initial_retrieval_features(
            qid=qid,
            raw_row=raw_row,
            response_hash_by_qid=response_hash_by_qid,
            trace_by_hash=trace_by_hash,
        )
        initial_recall = to_float(row.get(f"{baseline_operation}_evidence_recall"))
        selected_distractors = to_float(retrieval_features.get("initial_selected_distractor_count"))
        item: dict[str, Any] = {
            "model_label": model_label,
            "qid": qid,
            "raw_qid": raw_qid(qid),
            "question_type": row.get("question_type") or raw_row.get("question_type") or "unknown",
            "is_abstention": int(qid.endswith("_abs") or len(raw_row.get("answer_session_ids") or []) == 0),
            "baseline_operation": baseline_operation,
            "shrink_operation": shrink_operation,
            "stop_reward": stop_reward,
            "shrink_reward": shrink_reward,
            "delta_reward": delta_reward,
            "stop_cost": stop_cost,
            "shrink_cost": shrink_cost,
            "delta_cost": shrink_cost - stop_cost,
            "stop_utility": stop_utility,
            "shrink_utility": shrink_utility,
            "delta_utility": delta_utility,
            "stop_evidence_recall": initial_recall,
            "shrink_evidence_recall": to_float(row.get(f"{shrink_operation}_evidence_recall")),
            "stop_all_evidence_present": to_float(row.get(f"{baseline_operation}_all_evidence_present")),
            "shrink_all_evidence_present": to_float(row.get(f"{shrink_operation}_all_evidence_present")),
            "primary_sign_eps": eps_primary,
            "primary_wtl": sign_label(delta_reward, eps_primary),
            "initial_evidence_recall_bin": bin_numeric(
                initial_recall,
                [(0, "recall_0"), (0.5, "recall_0_to_0p5"), (0.999999, "recall_partial")],
                "recall_complete",
            ),
            "initial_matrix_all_gold_bin": "complete"
            if to_float(row.get(f"{baseline_operation}_all_evidence_present")) == 1.0
            else "incomplete_or_missing",
            "initial_distractor_count_bin": bin_numeric(
                selected_distractors,
                [(5, "distractor_0to5"), (10, "distractor_6to10"), (20, "distractor_11to20")],
                "distractor_gt20",
            ),
            **gold_features,
            **retrieval_features,
        }
        out.append(item)
    return sorted(out, key=lambda x: x["qid"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-matrix", type=Path, required=True)
    parser.add_argument("--raw-longmemeval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-label", default="unknown_model")
    parser.add_argument("--baseline-operation", default="stop")
    parser.add_argument("--shrink-operation", default="refine")
    parser.add_argument("--stop-response", type=Path)
    parser.add_argument("--stop-trace", type=Path)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.05, 0.1])
    parser.add_argument("--primary-eps", type=float, default=0.1)
    parser.add_argument("--compare", action="append", default=[], help="Optional label=per_qid.csv from another model scale.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_ids = load_manifest_ids(args.manifest)
    wide_rows = read_csv(args.wide_matrix)
    raw_rows = load_raw_rows(args.raw_longmemeval)
    rows = attach_rows(
        wide_rows=wide_rows,
        raw_rows=raw_rows,
        manifest_ids=manifest_ids,
        model_label=args.model_label,
        baseline_operation=args.baseline_operation,
        shrink_operation=args.shrink_operation,
        stop_response=args.stop_response,
        stop_trace=args.stop_trace,
        eps_primary=args.primary_eps,
    )
    if not rows:
        raise ValueError("No rows available for P23 analysis")

    per_qid_path = args.output_dir / "p23_answer_vs_shrink_per_qid.csv"
    write_csv(per_qid_path, rows)
    by_type = group_summary(rows, "question_type", args.eps)
    write_csv(args.output_dir / "p23_answer_vs_shrink_by_type.csv", by_type)

    mechanism_fields = [
        "initial_matrix_all_gold_bin",
        "initial_evidence_recall_bin",
        "gold_relevant_count_bin",
        "gold_position_span_bin",
        "gold_temporal_spread_bin",
        "gold_latest_age_bin",
        "initial_distractor_count_bin",
    ]
    mechanism_rows: list[dict[str, Any]] = []
    for field in mechanism_fields:
        mechanism_rows.extend(group_summary(rows, field, args.eps))
    write_csv(args.output_dir / "p23_gold_mechanism_audit.csv", mechanism_rows)

    all_deltas = [float(row["delta_reward"]) for row in rows]
    summary: dict[str, Any] = {
        "experiment": "P23_answer_vs_shrink",
        "model_label": args.model_label,
        "n": len(rows),
        "wide_matrix": str(args.wide_matrix),
        "raw_longmemeval": str(args.raw_longmemeval),
        "manifest": str(args.manifest) if args.manifest else None,
        "baseline_operation": args.baseline_operation,
        "shrink_operation": args.shrink_operation,
        "metric": "surrogate_f1_reward_unless_matrix_was_built_with_judge",
        "mean_stop_reward": mean([float(row["stop_reward"]) for row in rows]),
        "mean_shrink_reward": mean([float(row["shrink_reward"]) for row in rows]),
        "mean_delta_reward": mean(all_deltas),
        "var_delta_reward": variance(all_deltas),
        "min_delta_reward": min(all_deltas),
        "max_delta_reward": max(all_deltas),
        "mean_delta_cost": mean([float(row["delta_cost"]) for row in rows]),
        "mean_delta_utility": mean([float(row["delta_utility"]) for row in rows]),
        "stop_all_evidence_present_rate": mean([to_float(row["stop_all_evidence_present"]) for row in rows]),
        "shrink_all_evidence_present_rate": mean([to_float(row["shrink_all_evidence_present"]) for row in rows]),
        "trace_alignment": {
            "stop_response": str(args.stop_response) if args.stop_response else None,
            "stop_trace": str(args.stop_trace) if args.stop_trace else None,
            "trace_found_rate": mean([float(row["stop_trace_found"]) for row in rows]),
            "mean_initial_trace_evidence_recall": mean([to_float(row["initial_trace_evidence_recall"]) for row in rows]),
        },
        "threshold_wtl": {str(eps): wtl_stats(all_deltas, eps) for eps in args.eps},
        "question_type_summary_csv": str(args.output_dir / "p23_answer_vs_shrink_by_type.csv"),
        "gold_mechanism_audit_csv": str(args.output_dir / "p23_gold_mechanism_audit.csv"),
        "per_qid_csv": str(per_qid_path),
        "interpretation_guardrails": [
            "This is exploratory/dev analysis unless the manifest is preregistered and untouched.",
            "question_type, answer_session_ids, and derived gold-mechanism fields are offline diagnostics only.",
            "A positive mean refine delta alone does not establish a control problem; inspect wins and losses.",
            "If SHRINK has wins and losses at epsilon=0.1, this supports non-trivial operation-value heterogeneity.",
            "Surrogate F1 is not the official LongMemEval judge; claims must be weakened until official judge is run.",
            "Do not compare 0.5B and 7B unless they use the same manifest, same operations, same K, and same prompts.",
        ],
    }

    comparisons: list[dict[str, Any]] = []
    this_by_qid = {row["qid"]: row for row in rows}
    for spec in args.compare:
        if "=" not in spec:
            raise ValueError("--compare expects label=per_qid.csv")
        label, path_text = spec.split("=", 1)
        other_rows = read_csv(Path(path_text))
        other_by_qid = {normalize_qid(row["qid"]): row for row in other_rows}
        common = sorted(set(this_by_qid) & set(other_by_qid))
        xs = [float(this_by_qid[qid]["delta_reward"]) for qid in common]
        ys = [to_float(other_by_qid[qid].get("delta_reward")) for qid in common]
        sign_overlap_eps01 = mean(
            [
                float(sign_label(x, 0.1) == sign_label(y, 0.1))
                for x, y in zip(xs, ys)
                if not math.isnan(x) and not math.isnan(y)
            ]
        )
        comparisons.append(
            {
                "reference_model_label": args.model_label,
                "comparison_model_label": label,
                "comparison_path": path_text,
                "n_common": len(common),
                "delta_reward_pearson": pearson(xs, ys),
                "sign_overlap_eps_0.1": sign_overlap_eps01,
            }
        )
    if comparisons:
        summary["model_comparisons"] = comparisons
        write_csv(args.output_dir / "p23_model_scale_comparison.csv", comparisons)

    (args.output_dir / "p23_answer_vs_shrink_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
