#!/usr/bin/env python3
"""P32 Exception Content/Path Audit.

Read-only diagnostic over frozen P27/P31 artifacts.  This script does not
generate, train, rescore, or modify prompts/operators/splits/metrics.  It only
examines substantive P31.6 exceptions and asks whether session-level complete
cases also look answer-content complete under a conservative string probe.

Important boundary:
    LongMemEval answer_session_ids are session-level labels.  This script does
    not have official answer-bearing turn/span annotations.  Exact answer-string
    and token-coverage probes are diagnostic hints, not proof of content
    sufficiency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from audit_p31_oracle_exceptions import (
    answer_source_indices,
    gold_stats,
    int_list,
    load_raw_rows,
    load_responses_qid_to_query_sha1,
    load_trace_query_sha1_to_state_records,
)
from train_p29_selective_admission_gate import (
    iter_jsonl,
    mean,
    normalize_qid,
    read_csv,
    sha1_text,
    tokens,
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


def norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def answer_exact_in_text(answer: str, text: str) -> bool:
    ans = norm_text(answer)
    if not ans:
        return False
    return ans in norm_text(text)


def answer_token_coverage(answer: str, text: str) -> float:
    ans_tokens = [tok for tok in tokens(answer) if tok]
    if not ans_tokens:
        return 0.0
    text_tokens = set(tokens(text))
    return sum(1 for tok in ans_tokens if tok in text_tokens) / len(ans_tokens)


def content_probe(answer: str, text: str, threshold: float) -> bool:
    return answer_exact_in_text(answer, text) or answer_token_coverage(answer, text) >= threshold


def stringify_session(session: Any) -> str:
    if isinstance(session, list):
        parts: list[str] = []
        for turn in session:
            if isinstance(turn, dict):
                role = str(turn.get("role") or "")
                content = str(turn.get("content") or "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return str(session or "")


def raw_gold_text(raw_row: dict[str, Any], gold_indices: list[int]) -> str:
    sessions = raw_row.get("haystack_sessions") or []
    chunks: list[str] = []
    for idx in gold_indices:
        if 0 <= idx < len(sessions):
            chunks.append(stringify_session(sessions[idx]))
    return "\n\n".join(chunks)


def load_policy_maps(run_root: Path, policies: list[str], qids: list[str]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    traces: dict[str, dict[str, dict[str, Any]]] = {}
    responses: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in policies:
        trace_path = run_root / policy / "trace.jsonl"
        response_path = run_root / policy / f"responses_{policy}.jsonl"
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
        if not response_path.exists():
            raise FileNotFoundError(response_path)
        qid_to_hash = load_responses_qid_to_query_sha1(response_path)
        hash_to_state = load_trace_query_sha1_to_state_records(trace_path)
        mapped: dict[str, dict[str, Any]] = {}
        for qid in qids:
            qhash = qid_to_hash.get(qid, "")
            if qhash and qhash in hash_to_state:
                mapped[qid] = hash_to_state[qhash]
        missing = sorted(set(qids) - set(mapped))
        if missing:
            raise ValueError(f"{policy} missing mapped trace state records, first={missing[:5]}")
        traces[policy] = mapped

        resp_map: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(response_path):
            qid = normalize_qid(str(row.get("qid") or row.get("question_id") or ""))
            if qid:
                resp_map[qid] = row
        missing_resp = sorted(set(qids) - set(resp_map))
        if missing_resp:
            raise ValueError(f"{policy} missing responses, first={missing_resp[:5]}")
        responses[policy] = resp_map
    return traces, responses


def indices_from_record(record: dict[str, Any], key: str = "admitted_source_indices") -> list[int]:
    return int_list(record.get(key))


def source_positions(selected: list[int], gold: set[int]) -> dict[str, Any]:
    positions = [i for i, idx in enumerate(selected) if idx in gold]
    if not selected:
        return {
            "gold_rank_positions_json": "[]",
            "gold_rank_min": math.nan,
            "gold_rank_max": math.nan,
            "gold_rank_span": math.nan,
            "gold_rank_mean": math.nan,
            "gold_rank_mean_relative": math.nan,
            "non_gold_count": 0,
            "gold_density": math.nan,
        }
    return {
        "gold_rank_positions_json": json.dumps(positions),
        "gold_rank_min": min(positions) if positions else math.nan,
        "gold_rank_max": max(positions) if positions else math.nan,
        "gold_rank_span": (max(positions) - min(positions)) if positions else math.nan,
        "gold_rank_mean": mean([float(x) for x in positions]) if positions else math.nan,
        "gold_rank_mean_relative": mean([float(x) / max(1, len(selected) - 1) for x in positions]) if positions else math.nan,
        "non_gold_count": len([idx for idx in selected if idx not in gold]),
        "gold_density": len([idx for idx in selected if idx in gold]) / max(1, len(selected)),
    }


def classify_p32(
    p31_class: str,
    base_all: bool,
    winner_all: bool,
    base_probe: bool,
    winner_probe: bool,
) -> str:
    if "retrieval_or_candidate_missing" in p31_class:
        return "route_A_candidate_retrieval_missing"
    if not base_all:
        if winner_all:
            return "route_A_session_admission_rescued"
        return "route_A_session_or_content_admission_incomplete"
    if not base_probe and winner_probe:
        return "route_A_content_surface_exposed_by_winner_probe"
    if not base_probe and not winner_probe:
        return "route_B_or_C_session_complete_answer_string_absent_both_needs_turn_span_or_judge"
    if base_probe and winner_probe:
        return "route_B_or_C_content_present_organization_interface_or_surrogate_candidate"
    if base_probe and not winner_probe:
        return "route_C_surrogate_or_response_format_candidate_winner_lacks_answer_string"
    return "needs_manual_review"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--p31-per-qid", type=Path, required=True)
    parser.add_argument("--raw-longmemeval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metric", choices=["reward", "proxy_utility_context"], default="reward")
    parser.add_argument("--base", choices=["raw_fold_best", "tiecost_fold_default", "both"], default="both")
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--answer-token-threshold", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    p31_rows_all = read_csv(args.p31_per_qid)
    if not p31_rows_all:
        raise ValueError(f"empty P31 per-qid: {args.p31_per_qid}")
    qids = [normalize_qid(str(row.get("qid") or "")) for row in p31_rows_all]
    manifest_qids = set(load_manifest_ids(args.manifest))
    if set(qids) - manifest_qids:
        raise ValueError("P31 rows contain qids outside manifest")
    raw_rows = load_raw_rows(args.raw_longmemeval, qids)
    policies = ["stop"] + [policy for policy in args.policy if policy != "stop"]
    traces, responses = load_policy_maps(args.run_root, policies, qids)

    base_specs = []
    if args.base in {"raw_fold_best", "both"}:
        base_specs.append(("raw_fold_best", "raw_best_fold_policy", "raw_exception_winner", "raw_exception_delta", "raw_exception_gt_eps", "raw_exception_class"))
    if args.base in {"tiecost_fold_default", "both"}:
        base_specs.append(("tiecost_fold_default", "tiecost_fold_policy", "tiecost_exception_winner", "tiecost_exception_delta", "tiecost_exception_gt_eps", "tiecost_exception_class"))

    out_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    for p31 in p31_rows_all:
        qid = normalize_qid(str(p31.get("qid") or ""))
        raw_row = raw_rows[qid]
        answer = str(raw_row.get("answer") or "")
        gold_indices = answer_source_indices(raw_row)
        gold_set = set(gold_indices)
        gold_text = raw_gold_text(raw_row, gold_indices)
        c0_indices = indices_from_record(traces["stop"][qid]["first_retrieve"], "retrieved_source_indices")
        w0_indices = indices_from_record(traces["stop"][qid]["final_visible"], "admitted_source_indices")
        c0_gold = gold_stats(gold_set, set(c0_indices))
        w0_gold = gold_stats(gold_set, set(w0_indices))

        for base_name, base_policy_key, winner_key, delta_key, exception_key, class_key in base_specs:
            if int(float(p31.get(exception_key) or 0)) != 1:
                continue
            base_policy = str(p31.get(base_policy_key) or "")
            winner_policy = str(p31.get(winner_key) or "")
            if not base_policy or not winner_policy:
                raise ValueError(f"missing base/winner policy for {qid}/{base_name}")
            base_record = traces[base_policy][qid]["final_visible"]
            winner_record = traces[winner_policy][qid]["final_visible"]
            base_indices = indices_from_record(base_record, "admitted_source_indices")
            winner_indices = indices_from_record(winner_record, "admitted_source_indices")
            base_state = str(base_record.get("state_text") or "")
            winner_state = str(winner_record.get("state_text") or "")
            base_resp = str(responses[base_policy][qid].get("response") or "")
            winner_resp = str(responses[winner_policy][qid].get("response") or "")
            query = str(responses[base_policy][qid].get("query") or raw_row.get("question") or "")

            base_gold = gold_stats(gold_set, set(base_indices))
            winner_gold = gold_stats(gold_set, set(winner_indices))
            base_pos = source_positions(base_indices, gold_set)
            winner_pos = source_positions(winner_indices, gold_set)

            exact_gold = answer_exact_in_text(answer, gold_text)
            exact_base = answer_exact_in_text(answer, base_state)
            exact_winner = answer_exact_in_text(answer, winner_state)
            cov_gold = answer_token_coverage(answer, gold_text)
            cov_base = answer_token_coverage(answer, base_state)
            cov_winner = answer_token_coverage(answer, winner_state)
            probe_base = content_probe(answer, base_state, args.answer_token_threshold)
            probe_winner = content_probe(answer, winner_state, args.answer_token_threshold)

            route = classify_p32(
                str(p31.get(class_key) or ""),
                bool(base_gold["all"]),
                bool(winner_gold["all"]),
                probe_base,
                probe_winner,
            )
            row = {
                "base": base_name,
                "qid": qid,
                "question_type": p31.get("question_type", ""),
                "p31_exception_class": p31.get(class_key, ""),
                "p32_route_probe": route,
                "metric": args.metric,
                "exception_delta": p31.get(delta_key, ""),
                "base_policy": base_policy,
                "winner_policy": winner_policy,
                "answer_sha1": sha1_text(answer),
                "answer_session_count": len(gold_indices),
                "answer_source_indices_json": json.dumps(gold_indices),
                "gold_answer_exact_in_gold_sessions": int(exact_gold),
                "gold_answer_token_coverage_in_gold_sessions": cov_gold,
                "C0_any_gold_session": int(c0_gold["any"]),
                "C0_all_gold_sessions": int(c0_gold["all"]),
                "C0_gold_session_recall": c0_gold["recall"],
                "W0_any_gold_session": int(w0_gold["any"]),
                "W0_all_gold_sessions": int(w0_gold["all"]),
                "W0_gold_session_recall": w0_gold["recall"],
                "base_any_gold_session": int(base_gold["any"]),
                "base_all_gold_sessions": int(base_gold["all"]),
                "base_gold_session_recall": base_gold["recall"],
                "winner_any_gold_session": int(winner_gold["any"]),
                "winner_all_gold_sessions": int(winner_gold["all"]),
                "winner_gold_session_recall": winner_gold["recall"],
                "base_answer_exact_in_visible_context": int(exact_base),
                "winner_answer_exact_in_visible_context": int(exact_winner),
                "base_answer_token_coverage": cov_base,
                "winner_answer_token_coverage": cov_winner,
                "base_answer_content_probe": int(probe_base),
                "winner_answer_content_probe": int(probe_winner),
                "base_context_chars": len(base_state),
                "winner_context_chars": len(winner_state),
                "context_char_delta_winner_minus_base": len(winner_state) - len(base_state),
                "base_n_admitted_sources": len(base_indices),
                "winner_n_admitted_sources": len(winner_indices),
                "n_shared_sources": len(set(base_indices) & set(winner_indices)),
                "n_base_only_sources": len(set(base_indices) - set(winner_indices)),
                "n_winner_only_sources": len(set(winner_indices) - set(base_indices)),
                "base_response_sha1": sha1_text(base_resp),
                "winner_response_sha1": sha1_text(winner_resp),
                "base_response_answer_exact": int(answer_exact_in_text(answer, base_resp)),
                "winner_response_answer_exact": int(answer_exact_in_text(answer, winner_resp)),
                "base_response_prefix": base_resp[:160],
                "winner_response_prefix": winner_resp[:160],
                "query_sha1": sha1_text(query),
            }
            for k, v in base_pos.items():
                row[f"base_{k}"] = v
            for k, v in winner_pos.items():
                row[f"winner_{k}"] = v
            out_rows.append(row)

            for role, policy, response in [("base", base_policy, base_resp), ("winner", winner_policy, winner_resp)]:
                judge_rows.append(
                    {
                        "base": base_name,
                        "qid": qid,
                        "role": role,
                        "policy": policy,
                        "query": query,
                        "expected_answer": answer,
                        "response": response,
                        "query_sha1": sha1_text(query),
                        "expected_answer_sha1": sha1_text(answer),
                        "response_sha1": sha1_text(response),
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for base_name in sorted(set(r["base"] for r in out_rows)):
        items = [r for r in out_rows if r["base"] == base_name]
        route_counts = Counter(str(r["p32_route_probe"]) for r in items)
        class_counts = Counter(str(r["p31_exception_class"]) for r in items)
        summary_rows.append(
            {
                "base": base_name,
                "n_exceptions": len(items),
                "route_counts_json": json.dumps(dict(sorted(route_counts.items())), sort_keys=True),
                "p31_class_counts_json": json.dumps(dict(sorted(class_counts.items())), sort_keys=True),
                "base_all_gold_session_rate": mean([float(r["base_all_gold_sessions"]) for r in items]),
                "winner_all_gold_session_rate": mean([float(r["winner_all_gold_sessions"]) for r in items]),
                "base_answer_content_probe_rate": mean([float(r["base_answer_content_probe"]) for r in items]),
                "winner_answer_content_probe_rate": mean([float(r["winner_answer_content_probe"]) for r in items]),
                "base_response_answer_exact_rate": mean([float(r["base_response_answer_exact"]) for r in items]),
                "winner_response_answer_exact_rate": mean([float(r["winner_response_answer_exact"]) for r in items]),
                "mean_context_char_delta_winner_minus_base": mean([float(r["context_char_delta_winner_minus_base"]) for r in items]),
                "mean_winner_only_sources": mean([float(r["n_winner_only_sources"]) for r in items]),
                "mean_base_only_sources": mean([float(r["n_base_only_sources"]) for r in items]),
            }
        )

    by_type_rows: list[dict[str, Any]] = []
    for key in sorted(set((r["base"], r["question_type"]) for r in out_rows)):
        base_name, qtype = key
        items = [r for r in out_rows if r["base"] == base_name and r["question_type"] == qtype]
        route_counts = Counter(str(r["p32_route_probe"]) for r in items)
        by_type_rows.append(
            {
                "base": base_name,
                "question_type": qtype,
                "n_exceptions": len(items),
                "route_counts_json": json.dumps(dict(sorted(route_counts.items())), sort_keys=True),
                "base_answer_content_probe_rate": mean([float(r["base_answer_content_probe"]) for r in items]),
                "winner_answer_content_probe_rate": mean([float(r["winner_answer_content_probe"]) for r in items]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"p32_exception_content_path_per_exception_{args.metric}.csv", out_rows)
    write_csv(args.output_dir / f"p32_exception_content_path_summary_{args.metric}.csv", summary_rows)
    write_csv(args.output_dir / f"p32_exception_content_path_by_type_{args.metric}.csv", by_type_rows)
    write_csv(args.output_dir / f"p32_official_judge_queue_{args.metric}.csv", judge_rows)
    report = {
        "status": "P32_EXCEPTION_CONTENT_PATH_AUDIT_COMPLETE",
        "scope": "read-only diagnostic over P31.6 substantive exceptions; no generation/training/rescoring",
        "run_root": str(args.run_root),
        "p31_per_qid": str(args.p31_per_qid),
        "raw_longmemeval": str(args.raw_longmemeval),
        "manifest": str(args.manifest),
        "metric": args.metric,
        "base": args.base,
        "answer_token_threshold": args.answer_token_threshold,
        "summary": summary_rows,
        "guardrails": [
            "answer_session_ids and answer strings are offline audit labels only.",
            "Exact answer-string and token-coverage probes are not official turn/span annotations.",
            "Do not infer official correctness from surrogate F1 or answer-string probes.",
            "This script creates an official-judge queue but does not call any judge.",
            "P32 is failure-mode identification, not a method or RL experiment.",
        ],
    }
    (args.output_dir / f"p32_exception_content_path_report_{args.metric}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
