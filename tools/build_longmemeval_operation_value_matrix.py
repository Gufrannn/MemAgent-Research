#!/usr/bin/env python3
"""Build a LongMemEval operation-value matrix from frozen generations.

The matrix is an offline analysis artifact.  It joins:

- official LongMemEval raw JSON, including evidence labels;
- id-only manifest for the pilot/confirm boundary;
- one or more operation generation files;
- optional operation trace files;
- optional judge scores.

Important leakage rule: ``answer_session_ids`` and ``has_answer`` are consumed
only here, after generation, to compute diagnostics such as evidence recall.
They are never inference-time features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_qid(qid: str) -> str:
    qid = str(qid)
    return qid if qid.startswith("longmemeval_") else f"longmemeval_{qid}"


def raw_qid(qid: str) -> str:
    qid = str(qid)
    return qid[len("longmemeval_") :] if qid.startswith("longmemeval_") else qid


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def answer_f1(prediction: str, answer: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(answer)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / max(1e-12, precision + recall)


def is_abstention(row: dict[str, Any]) -> bool:
    qid = str(row.get("question_id") or "")
    answer_sessions = row.get("answer_session_ids") or []
    return qid.endswith("_abs") or len(answer_sessions) == 0


def load_manifest_ids(path: Path) -> set[str]:
    payload = read_json(path)
    if isinstance(payload, dict):
        ids = payload.get("question_ids") or [item.get("question_id") for item in payload.get("items", [])]
    elif isinstance(payload, list):
        ids = payload
    else:
        raise ValueError(f"Unsupported manifest format: {path}")
    return {normalize_qid(str(qid)) for qid in ids if qid}


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


def load_raw_rows(path: Path, manifest_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("LongMemEval raw input must be a list")
    by_qid: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = normalize_qid(str(row.get("question_id") or ""))
        if qid in manifest_ids:
            by_qid[qid] = row
    missing = sorted(manifest_ids - set(by_qid))
    if missing:
        raise ValueError(f"Manifest ids not found in raw data, first missing: {missing[:5]}")
    return by_qid


def load_responses(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        qid = normalize_qid(str(row.get("qid") or row.get("question_id") or ""))
        if qid:
            out[qid] = row
    return out


def load_trace(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("phase") != "qa":
            continue
        qhash = row.get("query_sha1")
        if qhash:
            out[str(qhash)] = row
    return out


def load_judge(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    rows = iter_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    if isinstance(rows, dict):
        rows = rows.get("items") or rows.get("scores") or []
    if not isinstance(rows, list):
        raise ValueError("Judge file must be a JSON list, JSONL rows, or dict with items/scores")
    out: dict[str, float] = {}
    for row in rows:
        qid = normalize_qid(str(row.get("qid") or row.get("question_id") or ""))
        raw_score = row.get("judge_correct", row.get("correct", row.get("score", row.get("judge_score"))))
        if raw_score is None and isinstance(row.get("autoeval_label"), dict):
            raw_score = row["autoeval_label"].get("label")
        if qid and raw_score is not None:
            out[qid] = float(raw_score)
    return out


def parse_operation_spec(spec: str) -> tuple[str, Path, Path | None, Path | None, Path | None]:
    parts = spec.split("=")
    if len(parts) != 2 or not parts[0].strip():
        raise ValueError("Operation spec must be name=responses.jsonl[:trace.jsonl[:raw_longmemeval.json[:judge_log]]]")
    name = parts[0].strip()
    paths = parts[1].split(":")
    if not paths[0]:
        raise ValueError(f"Missing response path in operation spec: {spec}")
    response_path = Path(paths[0])
    trace_path = Path(paths[1]) if len(paths) > 1 and paths[1] else None
    raw_override = Path(paths[2]) if len(paths) > 2 and paths[2] else None
    judge_path = Path(paths[3]) if len(paths) > 3 and paths[3] else None
    return name, response_path, trace_path, raw_override, judge_path


def latest_op_record(trace_row: dict[str, Any], allowed_ops: set[str]) -> dict[str, Any] | None:
    records = trace_row.get("op_records") or []
    for record in reversed(records):
        if record.get("operation") in allowed_ops:
            return record
    return None


def first_op_record(trace_row: dict[str, Any], allowed_ops: set[str]) -> dict[str, Any] | None:
    records = trace_row.get("op_records") or []
    for record in records:
        if record.get("operation") in allowed_ops:
            return record
    return None


def session_ids_from_indices(indices: Any, raw_row: dict[str, Any]) -> set[str]:
    session_ids = nonempty_session_ids(raw_row)
    return {
        session_ids[int(idx)]
        for idx in (indices or [])
        if isinstance(idx, int) or (isinstance(idx, str) and idx.isdigit())
        for _ in [None]
        if 0 <= int(idx) < len(session_ids)
    }


def admitted_session_ids(trace_row: dict[str, Any] | None, raw_row: dict[str, Any]) -> tuple[set[str], str]:
    if not trace_row:
        return set(), "missing_trace"
    record = latest_op_record(
        trace_row,
        {"FILTER", "REFINE", "SHRINK_VISIBLE", "REPACK_CANDIDATES", "RETRIEVE_MORE", "EXPAND"},
    ) or latest_op_record(trace_row, {"RETRIEVE", "RETRIEVE_RECENT"})
    if not record:
        return set(), "missing_record"
    indices = record.get("admitted_source_indices")
    source = "admitted_source_indices"
    if indices is None:
        indices = record.get("selected_indices") or []
        source = "selected_indices_legacy_fallback"
    return session_ids_from_indices(indices, raw_row), source


def retrieved_session_ids(trace_row: dict[str, Any] | None, raw_row: dict[str, Any]) -> tuple[set[str], str]:
    """Return C0 session ids from the first retrieval record.

    This intentionally uses ``retrieved_source_indices`` rather than final
    admitted indices.  It supports P26's separation of retrieval availability
    (gold in C0) from working-memory admission (gold in W0).
    """

    if not trace_row:
        return set(), "missing_trace"
    record = first_op_record(trace_row, {"RETRIEVE", "RETRIEVE_RECENT"})
    if not record:
        return set(), "missing_initial_retrieve_record"
    indices = record.get("retrieved_source_indices")
    source = "retrieved_source_indices"
    if indices is None:
        indices = record.get("selected_indices") or []
        source = "selected_indices_legacy_fallback"
    return session_ids_from_indices(indices, raw_row), source


def evidence_scores(admitted: set[str], raw_row: dict[str, Any]) -> dict[str, float | int | str]:
    gold = {str(item) for item in (raw_row.get("answer_session_ids") or [])}
    if not gold:
        return {
            "gold_evidence_sessions": 0,
            "admitted_evidence_sessions": 0,
            "evidence_session_recall": math.nan,
            "all_evidence_present": math.nan,
            "is_abstention": 1,
        }
    hit = len(admitted & gold)
    return {
        "gold_evidence_sessions": len(gold),
        "admitted_evidence_sessions": hit,
        "evidence_session_recall": hit / len(gold),
        "all_evidence_present": 1.0 if gold <= admitted else 0.0,
        "is_abstention": 0,
    }


def cost_value(response_row: dict[str, Any], trace_row: dict[str, Any] | None, field: str) -> float:
    if field == "generation_time":
        return float(response_row.get("generation_time") or 0.0)
    if not trace_row:
        return 0.0
    if field == "latency_s":
        return float(trace_row.get("latency_s") or 0.0)
    if field == "ops":
        return float(len(trace_row.get("operations") or []))
    if field == "context_kchars":
        return float(trace_row.get("final_context_chars") or 0.0) / 1000.0
    raise ValueError(f"Unsupported cost field: {field}")


def mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / max(1, len(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-longmemeval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--operation",
        action="append",
        required=True,
        help="Operation spec: name=responses.jsonl[:trace.jsonl[:raw_longmemeval.json[:judge_log]]]. Repeat for each operation.",
    )
    parser.add_argument("--judge-log", type=Path, help="Optional JSON/JSONL judge scores.")
    parser.add_argument("--baseline-operation", default="d1")
    parser.add_argument("--reward-source", choices=["judge", "surrogate_f1"], default="surrogate_f1")
    parser.add_argument("--cost-field", choices=["context_kchars", "ops", "latency_s", "generation_time"], default="context_kchars")
    parser.add_argument("--lambda-cost", type=float, default=0.02)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Use the intersection of completed response ids across operations. Intended for dev10 smoke only.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_ids = load_manifest_ids(args.manifest)
    raw_rows = load_raw_rows(args.raw_longmemeval, manifest_ids)
    judge_scores = load_judge(args.judge_log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    operation_data: dict[str, dict[str, Any]] = {}
    for spec in args.operation:
        name, response_path, trace_path, raw_override, judge_path = parse_operation_spec(spec)
        responses = load_responses(response_path)
        traces = load_trace(trace_path)
        operation_judge_scores = load_judge(judge_path)
        operation_raw_rows = load_raw_rows(raw_override, manifest_ids) if raw_override else raw_rows
        operation_data[name] = {
            "responses": responses,
            "traces": traces,
            "judge_scores": operation_judge_scores,
            "raw_rows": operation_raw_rows,
            "raw_override": str(raw_override) if raw_override else None,
            "judge_log": str(judge_path) if judge_path else None,
        }

    if args.baseline_operation not in operation_data:
        raise ValueError(f"Baseline operation not supplied: {args.baseline_operation}")

    requested_manifest_ids = set(manifest_ids)
    if args.allow_partial:
        completed_sets = [set(payload["responses"]) for payload in operation_data.values()]
        manifest_ids = set.intersection(requested_manifest_ids, *completed_sets) if completed_sets else set()
        if not manifest_ids:
            raise ValueError("No common completed response ids across operations")

    long_rows: list[dict[str, Any]] = []
    by_qid_op: dict[tuple[str, str], dict[str, Any]] = {}
    for op_name, payload in operation_data.items():
        responses = payload["responses"]
        traces = payload["traces"]
        op_raw_rows = payload["raw_rows"]
        missing = sorted(manifest_ids - set(responses))
        if missing and not args.allow_partial:
            raise ValueError(f"{op_name} responses missing manifest ids, first missing: {missing[:5]}")
        for qid in sorted(manifest_ids):
            raw_row = raw_rows[qid]
            evidence_raw_row = op_raw_rows[qid]
            response_row = responses[qid]
            query = str(response_row.get("query") or f"[{raw_row.get('question_date')}] {raw_row.get('question')}")
            trace_row = traces.get(sha1_text(query))
            admitted, evidence_source = admitted_session_ids(trace_row, evidence_raw_row)
            retrieved, retrieved_evidence_source = retrieved_session_ids(trace_row, evidence_raw_row)
            ev = evidence_scores(admitted, evidence_raw_row)
            retrieved_ev = evidence_scores(retrieved, evidence_raw_row)
            surrogate_f1 = answer_f1(str(response_row.get("response") or ""), str(raw_row.get("answer") or ""))
            operation_judge_scores = payload.get("judge_scores") or {}
            judge_correct = operation_judge_scores.get(qid, judge_scores.get(qid, math.nan))
            if args.reward_source == "judge":
                reward = judge_correct
            else:
                reward = surrogate_f1
            cost = cost_value(response_row, trace_row, args.cost_field)
            proxy_utility_context = reward - args.lambda_cost * cost if not math.isnan(reward) else math.nan
            row = {
                "qid": qid,
                "raw_qid": raw_qid(qid),
                "operation": op_name,
                "question_type": raw_row.get("question_type") or "unknown",
                "is_abstention": ev["is_abstention"],
                "reward": reward,
                "surrogate_answer_f1": surrogate_f1,
                "judge_correct": judge_correct,
                "cost": cost,
                "cost_field": args.cost_field,
                "proxy_utility_context": proxy_utility_context,
                "utility": proxy_utility_context,
                "utility_deprecated_alias": "proxy_utility_context",
                "evidence_session_recall": ev["evidence_session_recall"],
                "all_evidence_present": ev["all_evidence_present"],
                "retrieved_evidence_session_recall": retrieved_ev["evidence_session_recall"],
                "retrieved_all_evidence_present": retrieved_ev["all_evidence_present"],
                "retrieved_evidence_sessions": retrieved_ev["admitted_evidence_sessions"],
                "gold_evidence_sessions": ev["gold_evidence_sessions"],
                "admitted_evidence_sessions": ev["admitted_evidence_sessions"],
                "admitted_session_count": len(admitted),
                "evidence_source": evidence_source,
                "retrieved_evidence_source": retrieved_evidence_source,
                "trace_found": int(trace_row is not None),
                "final_context_chars": trace_row.get("final_context_chars") if trace_row else math.nan,
                "n_operations": len(trace_row.get("operations") or []) if trace_row else math.nan,
                "generation_time": float(response_row.get("generation_time") or 0.0),
            }
            long_rows.append(row)
            by_qid_op[(qid, op_name)] = row

    wide_rows: list[dict[str, Any]] = []
    for qid in sorted(manifest_ids):
        raw_row = raw_rows[qid]
        base = by_qid_op[(qid, args.baseline_operation)]
        wide = {
            "qid": qid,
            "raw_qid": raw_qid(qid),
            "question_type": raw_row.get("question_type") or "unknown",
            "is_abstention": int(is_abstention(raw_row)),
            "baseline_operation": args.baseline_operation,
        }
        for op_name in operation_data:
            row = by_qid_op[(qid, op_name)]
            prefix = op_name
            wide[f"{prefix}_reward"] = row["reward"]
            wide[f"{prefix}_cost"] = row["cost"]
            wide[f"{prefix}_proxy_utility_context"] = row["proxy_utility_context"]
            wide[f"{prefix}_utility"] = row["utility"]
            wide[f"{prefix}_evidence_recall"] = row["evidence_session_recall"]
            wide[f"{prefix}_all_evidence_present"] = row["all_evidence_present"]
            wide[f"{prefix}_retrieved_evidence_recall"] = row["retrieved_evidence_session_recall"]
            wide[f"{prefix}_retrieved_all_evidence_present"] = row["retrieved_all_evidence_present"]
            wide[f"{prefix}_delta_proxy_utility_context_vs_{args.baseline_operation}"] = (
                row["proxy_utility_context"] - base["proxy_utility_context"]
            )
            wide[f"{prefix}_delta_reward_vs_{args.baseline_operation}"] = row["reward"] - base["reward"]
            wide[f"{prefix}_delta_utility_vs_{args.baseline_operation}"] = row["utility"] - base["utility"]
        wide_rows.append(wide)

    summary_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        groups[("ALL", row["operation"])].append(row)
        groups[(str(row["question_type"]), row["operation"])].append(row)
    for (group, op_name), rows in sorted(groups.items()):
        base_rows = [by_qid_op[(row["qid"], args.baseline_operation)] for row in rows]
        summary_rows.append(
            {
                "group": group,
                "operation": op_name,
                "n": len(rows),
                "mean_reward": mean([float(row["reward"]) for row in rows]),
                "mean_cost": mean([float(row["cost"]) for row in rows]),
                "mean_proxy_utility_context": mean([float(row["proxy_utility_context"]) for row in rows]),
                "mean_utility": mean([float(row["utility"]) for row in rows]),
                "mean_delta_reward_vs_baseline": mean(
                    [float(row["reward"]) - float(base["reward"]) for row, base in zip(rows, base_rows)]
                ),
                "mean_delta_utility_vs_baseline": mean(
                    [float(row["utility"]) - float(base["utility"]) for row, base in zip(rows, base_rows)]
                ),
                "mean_delta_proxy_utility_context_vs_baseline": mean(
                    [
                        float(row["proxy_utility_context"]) - float(base["proxy_utility_context"])
                        for row, base in zip(rows, base_rows)
                    ]
                ),
                "mean_evidence_session_recall": mean([float(row["evidence_session_recall"]) for row in rows]),
                "mean_all_evidence_present": mean([float(row["all_evidence_present"]) for row in rows]),
                "trace_coverage": mean([float(row["trace_found"]) for row in rows]),
            }
        )

    write_csv(args.output_dir / "longmemeval_operation_value_long.csv", long_rows)
    write_csv(args.output_dir / "longmemeval_operation_value_wide.csv", wide_rows)
    write_csv(args.output_dir / "longmemeval_operation_value_summary.csv", summary_rows)

    audit = {
        "raw_longmemeval": str(args.raw_longmemeval),
        "manifest": str(args.manifest),
        "allow_partial": args.allow_partial,
        "n_requested_manifest_questions": len(requested_manifest_ids),
        "n_questions": len(manifest_ids),
        "operations": sorted(operation_data),
        "operation_raw_overrides": {
            name: payload["raw_override"]
            for name, payload in operation_data.items()
            if payload["raw_override"]
        },
        "operation_judge_logs": {
            name: payload["judge_log"]
            for name, payload in operation_data.items()
            if payload["judge_log"]
        },
        "baseline_operation": args.baseline_operation,
        "reward_source": args.reward_source,
        "cost_field": args.cost_field,
        "lambda_cost": args.lambda_cost,
        "outputs": [
            "longmemeval_operation_value_long.csv",
            "longmemeval_operation_value_wide.csv",
            "longmemeval_operation_value_summary.csv",
        ],
        "leakage_boundary": "Gold evidence labels are used only in this offline matrix builder.",
        "warning": "surrogate_f1 is for smoke/debug only; final claims require official LongMemEval judge or a calibrated judge surrogate.",
    }
    (args.output_dir / "longmemeval_operation_value_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
