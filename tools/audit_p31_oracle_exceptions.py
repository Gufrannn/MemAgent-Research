#!/usr/bin/env python3
"""P31 Oracle Exception Audit.

Read-only diagnostic over frozen operation-value artifacts.  It asks why the
offline oracle beats a default policy on substantive eps-margin examples:

    1. retrieval/candidate missing:
       not all gold answer sessions are present in C0.
    2. admission incomplete:
       C0 contains all gold sessions, but the base final visible working memory
       does not.
       This is further split into no rescue, partial rescue, and complete rescue.
    3. gold-session complete organization/interface candidate:
       the base final visible working memory already contains all gold sessions, so a
       substantive oracle win is not explained by session-level recall.

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
    sha1_text,
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
        if not qhash and row.get("query") is not None:
            qhash = sha1_text(str(row.get("query") or ""))
        if not qid or not qhash:
            continue
        if qid in out:
            raise ValueError(f"duplicate response qid in {path}: {qid}")
        out[qid] = qhash
    return out


def final_visible_record(trace_row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the final reader-visible working-memory record.

    The first RETRIEVE record defines C0 and initial W0.  Repack/admission
    policies may later replace the reader-visible working memory in
    REPACK_CANDIDATES.  For base/winner completeness, use the last op record
    that explicitly carries admitted_source_indices rather than the initial
    RETRIEVE record.
    """

    for record in reversed(trace_row.get("op_records") or []):
        if record.get("admitted_source_indices") is not None:
            return record
    return None


def load_trace_query_sha1_to_state_records(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("phase") != "qa":
            continue
        qhash = str(row.get("query_sha1") or "")
        if not qhash:
            continue
        retrieve = first_op_record(row, {"RETRIEVE"})
        final = final_visible_record(row)
        if not retrieve or not final:
            continue
        if qhash in out:
            raise ValueError(f"duplicate qa trace query_sha1 in {path}: {qhash}")
        out[qhash] = {
            "first_retrieve": retrieve,
            "final_visible": final,
            "final_visible_operation": str(final.get("operation") or ""),
        }
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


def gold_stats(gold: set[int], selected: set[int]) -> dict[str, Any]:
    overlap = gold & selected
    denom = max(1, len(gold))
    return {
        "any": bool(overlap),
        "all": bool(gold) and gold.issubset(selected),
        "recall": len(overlap) / denom,
        "n_overlap": len(overlap),
        "overlap_json": json.dumps(sorted(overlap)),
    }


def classify_failure(
    c0_recall: float,
    base_recall: float,
    winner_recall: float,
    winner_delta: float,
    eps: float,
) -> str:
    if winner_delta <= eps:
        return "default_safe_no_substantive_exception"
    if c0_recall < 1.0:
        return "retrieval_or_candidate_missing_gold_sessions"
    if base_recall < 1.0:
        if winner_recall <= base_recall:
            return "admission_incomplete_not_rescued"
        if winner_recall >= 1.0:
            return "admission_incomplete_complete_rescue"
        return "admission_incomplete_partial_rescue"
    if base_recall >= 1.0:
        return "gold_session_complete_organization_interface_or_surrogate_candidate"
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
        hash_to_state = load_trace_query_sha1_to_state_records(trace_path)
        mapped: dict[str, dict[str, Any]] = {}
        for qid in qids:
            qhash = qid_to_hash.get(qid, "")
            if not qhash:
                continue
            state_records = hash_to_state.get(qhash)
            if state_records:
                mapped[qid] = state_records
        policy_traces[policy] = mapped
    if missing_trace_files:
        raise ValueError(f"missing trace files: {missing_trace_files}")
    for policy, mapped in policy_traces.items():
        missing_qids = sorted(set(qids) - set(mapped))
        if missing_qids:
            raise ValueError(f"{policy} missing mapped trace state records, first={missing_qids[:5]}")

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
        stop_record = policy_traces["stop"][qid]["first_retrieve"]
        c0_indices = set(int_list(stop_record.get("retrieved_source_indices")))

        def admitted(policy: str) -> set[int]:
            record = policy_traces[policy][qid]["final_visible"]
            return set(int_list(record.get("admitted_source_indices")))

        def final_op(policy: str) -> str:
            return str(policy_traces[policy][qid].get("final_visible_operation") or "")

        w0_indices = admitted("stop")
        raw_base_indices = admitted(raw_best_fold)
        tie_base_indices = admitted(tiecost_fold)
        raw_winner_indices = admitted(raw_winner)
        tie_winner_indices = admitted(tie_winner)

        c0_gold = gold_stats(answer_indices, c0_indices)
        w0_gold = gold_stats(answer_indices, w0_indices)
        raw_base_gold = gold_stats(answer_indices, raw_base_indices)
        tie_base_gold = gold_stats(answer_indices, tie_base_indices)
        raw_winner_gold = gold_stats(answer_indices, raw_winner_indices)
        tie_winner_gold = gold_stats(answer_indices, tie_winner_indices)

        out_rows.append(
            {
                "qid": qid,
                "question_type": row.get("question_type", ""),
                "metric": args.metric,
                "answer_session_count": len(answer_indices),
                "answer_source_indices_json": json.dumps(sorted(answer_indices)),
                "C0_any_gold_session": bool_str(c0_gold["any"]),
                "C0_all_gold_sessions": bool_str(c0_gold["all"]),
                "C0_gold_session_recall": c0_gold["recall"],
                "C0_gold_session_overlap_n": c0_gold["n_overlap"],
                "C0_gold_session_overlap_json": c0_gold["overlap_json"],
                "W0_any_gold_session": bool_str(w0_gold["any"]),
                "W0_all_gold_sessions": bool_str(w0_gold["all"]),
                "W0_gold_session_recall": w0_gold["recall"],
                "W0_gold_session_overlap_n": w0_gold["n_overlap"],
                "W0_gold_session_overlap_json": w0_gold["overlap_json"],
                "legacy_C0_has_answer_session_any": bool_str(c0_gold["any"]),
                "legacy_W0_stop_has_answer_session_any": bool_str(w0_gold["any"]),
                "raw_best_global_policy": raw_best_global,
                "tiecost_global_policy": tiecost_global,
                "raw_best_fold_policy": raw_best_fold,
                "raw_best_fold_final_visible_operation": final_op(raw_best_fold),
                "tiecost_fold_policy": tiecost_fold,
                "tiecost_fold_final_visible_operation": final_op(tiecost_fold),
                "raw_exception_winner": raw_winner,
                "raw_exception_winner_final_visible_operation": final_op(raw_winner),
                "raw_exception_delta": raw_delta,
                "raw_exception_gt_eps": raw_exception,
                "raw_base_any_gold_session": bool_str(raw_base_gold["any"]),
                "raw_base_all_gold_sessions": bool_str(raw_base_gold["all"]),
                "raw_base_gold_session_recall": raw_base_gold["recall"],
                "raw_base_gold_session_overlap_n": raw_base_gold["n_overlap"],
                "raw_base_gold_session_overlap_json": raw_base_gold["overlap_json"],
                "raw_winner_any_gold_session": bool_str(raw_winner_gold["any"]),
                "raw_winner_all_gold_sessions": bool_str(raw_winner_gold["all"]),
                "raw_winner_gold_session_recall": raw_winner_gold["recall"],
                "raw_winner_gold_session_overlap_n": raw_winner_gold["n_overlap"],
                "raw_winner_gold_session_overlap_json": raw_winner_gold["overlap_json"],
                "raw_exception_class": classify_failure(c0_gold["recall"], raw_base_gold["recall"], raw_winner_gold["recall"], raw_delta, args.eps),
                "tiecost_exception_winner": tie_winner,
                "tiecost_exception_winner_final_visible_operation": final_op(tie_winner),
                "tiecost_exception_delta": tie_delta,
                "tiecost_exception_gt_eps": tie_exception,
                "tiecost_base_any_gold_session": bool_str(tie_base_gold["any"]),
                "tiecost_base_all_gold_sessions": bool_str(tie_base_gold["all"]),
                "tiecost_base_gold_session_recall": tie_base_gold["recall"],
                "tiecost_base_gold_session_overlap_n": tie_base_gold["n_overlap"],
                "tiecost_base_gold_session_overlap_json": tie_base_gold["overlap_json"],
                "tiecost_winner_any_gold_session": bool_str(tie_winner_gold["any"]),
                "tiecost_winner_all_gold_sessions": bool_str(tie_winner_gold["all"]),
                "tiecost_winner_gold_session_recall": tie_winner_gold["recall"],
                "tiecost_winner_gold_session_overlap_n": tie_winner_gold["n_overlap"],
                "tiecost_winner_gold_session_overlap_json": tie_winner_gold["overlap_json"],
                "tiecost_exception_class": classify_failure(c0_gold["recall"], tie_base_gold["recall"], tie_winner_gold["recall"], tie_delta, args.eps),
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
                "C0_any_gold_session_rate": mean([int(r["C0_any_gold_session"]) for r in out_rows]),
                "C0_all_gold_sessions_rate": mean([int(r["C0_all_gold_sessions"]) for r in out_rows]),
                "C0_mean_gold_session_recall": mean([float(r["C0_gold_session_recall"]) for r in out_rows]),
                "W0_any_gold_session_rate": mean([int(r["W0_any_gold_session"]) for r in out_rows]),
                "W0_all_gold_sessions_rate": mean([int(r["W0_all_gold_sessions"]) for r in out_rows]),
                "W0_mean_gold_session_recall": mean([float(r["W0_gold_session_recall"]) for r in out_rows]),
                "exception_C0_all_gold_sessions_rate": mean([int(r["C0_all_gold_sessions"]) for r in exceptions]),
                "exception_base_all_gold_sessions_rate": mean([
                    int(r["raw_base_all_gold_sessions"] if base_name == "raw_fold_best" else r["tiecost_base_all_gold_sessions"])
                    for r in exceptions
                ]),
                "exception_winner_all_gold_sessions_rate": mean([
                    int(r["raw_winner_all_gold_sessions"] if base_name == "raw_fold_best" else r["tiecost_winner_all_gold_sessions"])
                    for r in exceptions
                ]),
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
                "C0_any_gold_session_rate": mean([int(r["C0_any_gold_session"]) for r in items]),
                "C0_all_gold_sessions_rate": mean([int(r["C0_all_gold_sessions"]) for r in items]),
                "C0_mean_gold_session_recall": mean([float(r["C0_gold_session_recall"]) for r in items]),
                "W0_any_gold_session_rate": mean([int(r["W0_any_gold_session"]) for r in items]),
                "W0_all_gold_sessions_rate": mean([int(r["W0_all_gold_sessions"]) for r in items]),
                "W0_mean_gold_session_recall": mean([float(r["W0_gold_session_recall"]) for r in items]),
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
            "Canonical session-level audit reports any/all/recall; legacy has_answer_session fields are any-session only and must not be interpreted as complete evidence availability.",
            "C0 is taken from the first RETRIEVE retrieved_source_indices; policy working memory is taken from the final visible op record carrying admitted_source_indices.",
            "This audit identifies session-level answer-bearing admission opportunity; it does not prove turn/span-level sufficiency.",
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
