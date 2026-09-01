#!/usr/bin/env python3
"""P31 Oracle Exception Audit.

Read-only diagnostic over frozen operation-value artifacts.  It asks why the
offline oracle beats a default policy on substantive eps-margin examples:

    1. answer-bearing admission deficit:
       answer-bearing session is in C0 but absent from W0/default and admitted
       by the winning policy.
    2. organization/interface:
       answer-bearing sessions are already admitted by the base policy, so the
       oracle win is not explained by simple session-level recall rescue.
    3. no candidate opportunity:
       answer-bearing sessions are absent from C0, so admission cannot rescue.

This script uses gold answer_session_ids only for offline audit labels.  It
does not train, generate, score, or alter any protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_p29_selective_admission_gate import (
    cost,
    first_op_record,
    group_label,
    iter_jsonl,
    mean,
    normalize_qid,
    read_csv,
    value,
    write_csv,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest_ids(path: Path) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, dict):
        ids = payload.get("question_ids")
        if ids is None:
            ids = [item.get("question_id") for item in payload.get("items", [])]
    elif isinstance(payload, list):
        ids = payload
    else:
        raise ValueError(f"Unsupported manifest format: {path}")
    out = [normalize_qid(str(qid)) for qid in ids if qid]
    if not out:
        raise ValueError(f"empty manifest: {path}")
    return out


def load_raw_rows(path: Path, manifest_qids: list[str]) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("raw LongMemEval file must be a JSON list")
    wanted = set(manifest_qids)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = normalize_qid(str(row.get("question_id") or ""))
        if qid in wanted:
            out[qid] = row
    missing = sorted(wanted - set(out))
    if missing:
        raise ValueError(f"manifest ids missing from raw LongMemEval, first={missing[:5]}")
    return out


def load_responses_qid_to_query_sha1(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in iter_jsonl(path):
        qid = normalize_qid(str(row.get("qid") or row.get("question_id") or ""))
        qhash = str(row.get("query_sha1") or "")
        if not qid or not qhash:
            continue
        if qid in out:
            raise ValueError(f"duplicate response qid in {path}: {qid}")
        out[qid] = qhash
    return out


def load_trace_query_sha1_to_first_retrieve(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("phase") != "qa":
            continue
        qhash = str(row.get("query_sha1") or "")
        if not qhash:
            continue
        record = first_op_record(row, {"RETRIEVE"})
        if not record:
            continue
        if qhash in out:
            raise ValueError(f"duplicate qa trace query_sha1 in {path}: {qhash}")
        out[qhash] = record
    return out


def answer_source_indices(raw_row: dict[str, Any]) -> list[int]:
    session_ids = [str(x) for x in raw_row.get("haystack_session_ids") or []]
    answer_ids = set(str(x) for x in raw_row.get("answer_session_ids") or [])
    return [idx for idx, sid in enumerate(session_ids) if sid in answer_ids]


def int_list(value_in: Any) -> list[int]:
    if value_in is None:
        return []
    if isinstance(value_in, list):
        return [int(x) for x in value_in]
    if isinstance(value_in, str):
        text = value_in.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [int(x) for x in payload]
        except json.JSONDecodeError:
            pass
        return [int(x) for x in text.split(",") if x.strip()]
    return []


def choose_tiecost_fixed(rows: list[dict[str, str]], indices: list[int], policies: list[str], metric: str, tie_eps: float) -> str:
    means = {p: mean([value(rows[i], p, metric) for i in indices]) for p in policies}
    costs = {p: mean([cost(rows[i], p) for i in indices]) for p in policies}
    best = max(means.values())
    tied = [p for p in policies if best - means[p] <= tie_eps]
    order = {p: pos for pos, p in enumerate(policies)}
    return min(tied, key=lambda p: (costs[p], order[p]))


def choose_raw_best(rows: list[dict[str, str]], indices: list[int], policies: list[str], metric: str) -> str:
    means = {p: mean([value(rows[i], p, metric) for i in indices]) for p in policies}
    best = max(means.values())
    order = {p: pos for pos, p in enumerate(policies)}
    tied = [p for p in policies if abs(means[p] - best) <= 1e-12]
    return min(tied, key=lambda p: order[p])


def winning_policy(row: dict[str, str], base_policy: str, policies: list[str], metric: str, eps: float, tie_eps: float) -> tuple[str, float, int]:
    base_value = value(row, base_policy, metric)
    best_policy = base_policy
    best_delta = 0.0
    for policy in policies:
        delta = value(row, policy, metric) - base_value
        if delta > best_delta + tie_eps or (
            abs(delta - best_delta) <= tie_eps and cost(row, policy) < cost(row, best_policy)
        ):
            best_policy = policy
            best_delta = delta
    return best_policy, best_delta, int(best_delta > eps)


def classify_failure(
    c0_has_answer: bool,
    base_has_answer: bool,
    winner_has_answer: bool,
    winner_delta: float,
    eps: float,
) -> str:
    if winner_delta <= eps:
        return "default_safe_no_substantive_exception"
    if not c0_has_answer:
        return "no_candidate_opportunity_answer_not_in_C0"
    if not base_has_answer and winner_has_answer:
        return "answer_bearing_admission_deficit_rescued"
    if not base_has_answer and not winner_has_answer:
        return "answer_bearing_admission_deficit_not_rescued"
    if base_has_answer:
        return "organization_or_reader_interface_or_surrogate"
    return "unclassified"


def bool_str(flag: bool) -> str:
    return "1" if flag else "0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wide-matrix", type=Path)
    parser.add_argument("--raw-longmemeval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metric", choices=["reward", "proxy_utility_context"], default="reward")
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--tie-eps", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    wide_matrix = args.wide_matrix or args.run_root / "matrix" / "longmemeval_operation_value_wide.csv"
    rows = read_csv(wide_matrix)
    if not rows:
        raise ValueError(f"empty wide matrix: {wide_matrix}")
    policies = ["stop"] + [policy for policy in args.policy if policy != "stop"]
    qids = [normalize_qid(str(row.get("qid") or "")) for row in rows]
    for row, qid in zip(rows, qids):
        row["qid"] = qid
        row["evidence_bottleneck_group"] = group_label(row, "stop_retrieved_all_evidence_present")
    qid_to_row = {str(row["qid"]): row for row in rows}

    manifest_qids = load_manifest_ids(args.manifest)
    if set(qids) - set(manifest_qids):
        raise ValueError("wide matrix contains qids outside manifest")
    raw_rows = load_raw_rows(args.raw_longmemeval, qids)

    policy_traces: dict[str, dict[str, dict[str, Any]]] = {}
    missing_trace_files: list[str] = []
    for policy in policies:
        trace_path = args.run_root / policy / "trace.jsonl"
        response_path = args.run_root / policy / f"responses_{policy}.jsonl"
        if not trace_path.exists():
            missing_trace_files.append(str(trace_path))
            continue
        if not response_path.exists():
            missing_trace_files.append(str(response_path))
            continue
        qid_to_hash = load_responses_qid_to_query_sha1(response_path)
        hash_to_retrieve = load_trace_query_sha1_to_first_retrieve(trace_path)
        mapped: dict[str, dict[str, Any]] = {}
        for qid in qids:
            qhash = qid_to_hash.get(qid, "")
            if not qhash:
                continue
            record = hash_to_retrieve.get(qhash)
            if record:
                mapped[qid] = record
        policy_traces[policy] = mapped
    if missing_trace_files:
        raise ValueError(f"missing trace files: {missing_trace_files}")
    for policy, mapped in policy_traces.items():
        missing_qids = sorted(set(qids) - set(mapped))
        if missing_qids:
            raise ValueError(f"{policy} missing mapped first RETRIEVE records, first={missing_qids[:5]}")

    raw_best_global = choose_raw_best(rows, list(range(len(rows))), policies, args.metric)
    tiecost_global = choose_tiecost_fixed(rows, list(range(len(rows))), policies, args.metric, args.tie_eps)

    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        qid = str(row["qid"])
        train_idx = [i for i in range(len(rows)) if i != idx]
        raw_best_fold = choose_raw_best(rows, train_idx, policies, args.metric)
        tiecost_fold = choose_tiecost_fixed(rows, train_idx, policies, args.metric, args.tie_eps)
        raw_winner, raw_delta, raw_exception = winning_policy(row, raw_best_fold, policies, args.metric, args.eps, args.tie_eps)
        tie_winner, tie_delta, tie_exception = winning_policy(row, tiecost_fold, policies, args.metric, args.eps, args.tie_eps)

        raw_row = raw_rows[qid]
        answer_indices = set(answer_source_indices(raw_row))
        stop_record = policy_traces["stop"][qid]
        c0_indices = set(int_list(stop_record.get("retrieved_source_indices")))

        def admitted(policy: str) -> set[int]:
            record = policy_traces[policy][qid]
            return set(int_list(record.get("admitted_source_indices")))

        w0_indices = admitted("stop")
        raw_base_indices = admitted(raw_best_fold)
        tie_base_indices = admitted(tiecost_fold)
        raw_winner_indices = admitted(raw_winner)
        tie_winner_indices = admitted(tie_winner)

        c0_has = bool(answer_indices & c0_indices)
        w0_has = bool(answer_indices & w0_indices)
        raw_base_has = bool(answer_indices & raw_base_indices)
        tie_base_has = bool(answer_indices & tie_base_indices)
        raw_winner_has = bool(answer_indices & raw_winner_indices)
        tie_winner_has = bool(answer_indices & tie_winner_indices)

        out_rows.append(
            {
                "qid": qid,
                "question_type": row.get("question_type", ""),
                "metric": args.metric,
                "answer_session_count": len(answer_indices),
                "answer_source_indices_json": json.dumps(sorted(answer_indices)),
                "C0_has_answer_session": bool_str(c0_has),
                "W0_stop_has_answer_session": bool_str(w0_has),
                "raw_best_global_policy": raw_best_global,
                "tiecost_global_policy": tiecost_global,
                "raw_best_fold_policy": raw_best_fold,
                "tiecost_fold_policy": tiecost_fold,
                "raw_exception_winner": raw_winner,
                "raw_exception_delta": raw_delta,
                "raw_exception_gt_eps": raw_exception,
                "raw_base_has_answer_session": bool_str(raw_base_has),
                "raw_winner_has_answer_session": bool_str(raw_winner_has),
                "raw_exception_class": classify_failure(c0_has, raw_base_has, raw_winner_has, raw_delta, args.eps),
                "tiecost_exception_winner": tie_winner,
                "tiecost_exception_delta": tie_delta,
                "tiecost_exception_gt_eps": tie_exception,
                "tiecost_base_has_answer_session": bool_str(tie_base_has),
                "tiecost_winner_has_answer_session": bool_str(tie_winner_has),
                "tiecost_exception_class": classify_failure(c0_has, tie_base_has, tie_winner_has, tie_delta, args.eps),
                "stop_reward": value(row, "stop", args.metric),
                "raw_fold_base_value": value(row, raw_best_fold, args.metric),
                "raw_winner_value": value(row, raw_winner, args.metric),
                "tiecost_fold_base_value": value(row, tiecost_fold, args.metric),
                "tiecost_winner_value": value(row, tie_winner, args.metric),
                "evidence_bottleneck_group": row.get("evidence_bottleneck_group", ""),
            }
        )

    summary: list[dict[str, Any]] = []
    for base_name, class_key, exception_key in [
        ("raw_fold_best", "raw_exception_class", "raw_exception_gt_eps"),
        ("tiecost_fold_default", "tiecost_exception_class", "tiecost_exception_gt_eps"),
    ]:
        total = len(out_rows)
        exceptions = [r for r in out_rows if int(r[exception_key]) == 1]
        class_counts = Counter(str(r[class_key]) for r in out_rows)
        exception_class_counts = Counter(str(r[class_key]) for r in exceptions)
        summary.append(
            {
                "base": base_name,
                "metric": args.metric,
                "n": total,
                "n_exceptions_gt_eps": len(exceptions),
                "exception_rate": len(exceptions) / max(1, total),
                "C0_answer_present_rate": mean([int(r["C0_has_answer_session"]) for r in out_rows]),
                "W0_answer_present_rate": mean([int(r["W0_stop_has_answer_session"]) for r in out_rows]),
                "class_counts_json": json.dumps(dict(sorted(class_counts.items())), sort_keys=True),
                "exception_class_counts_json": json.dumps(dict(sorted(exception_class_counts.items())), sort_keys=True),
            }
        )

    by_type: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in out_rows:
        buckets[("raw_fold_best", str(r.get("question_type", "")))].append(r)
        buckets[("tiecost_fold_default", str(r.get("question_type", "")))].append(r)
    for (base_name, qtype), items in sorted(buckets.items()):
        exception_key = "raw_exception_gt_eps" if base_name == "raw_fold_best" else "tiecost_exception_gt_eps"
        class_key = "raw_exception_class" if base_name == "raw_fold_best" else "tiecost_exception_class"
        by_type.append(
            {
                "base": base_name,
                "question_type": qtype,
                "n": len(items),
                "n_exceptions_gt_eps": sum(int(r[exception_key]) for r in items),
                "C0_answer_present_rate": mean([int(r["C0_has_answer_session"]) for r in items]),
                "W0_answer_present_rate": mean([int(r["W0_stop_has_answer_session"]) for r in items]),
                "exception_class_counts_json": json.dumps(
                    dict(sorted(Counter(str(r[class_key]) for r in items if int(r[exception_key]) == 1).items())),
                    sort_keys=True,
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"p31_oracle_exception_per_qid_{args.metric}.csv", out_rows)
    write_csv(args.output_dir / f"p31_oracle_exception_summary_{args.metric}.csv", summary)
    write_csv(args.output_dir / f"p31_oracle_exception_by_type_{args.metric}.csv", by_type)
    report = {
        "status": "P31_ORACLE_EXCEPTION_AUDIT_COMPLETE",
        "scope": "read-only offline audit using gold answer_session_ids; no training/generation/rescoring",
        "run_root": str(args.run_root),
        "wide_matrix": str(wide_matrix),
        "raw_longmemeval": str(args.raw_longmemeval),
        "manifest": str(args.manifest),
        "metric": args.metric,
        "eps": args.eps,
        "tie_eps": args.tie_eps,
        "policies": policies,
        "raw_best_global_policy": raw_best_global,
        "tiecost_global_policy": tiecost_global,
        "summary": summary,
        "guardrails": [
            "answer_session_ids are used only as offline_labels for mechanism audit.",
            "This audit identifies session-level answer-bearing admission opportunity; it does not prove span-level sufficiency.",
            "Surrogate F1 exception labels remain exploratory until official judge validation.",
            "raw-best, tie/cost-default, and fold-local default exceptions must not be merged without naming them.",
        ],
    }
    (args.output_dir / f"p31_oracle_exception_report_{args.metric}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
