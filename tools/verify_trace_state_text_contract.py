#!/usr/bin/env python3
"""Verify trace ``state_text`` contracts against recorded prompt/context hashes.

The P29 online selector may use the first RETRIEVE op record's ``state_text`` as
W0.  This verifier checks producer-level contracts instead of trusting field
names:

1. Each response query maps to one QA trace by query_sha1.
2. The first RETRIEVE/RETRIEVE_RECENT op record has state_text when required.
3. sha1(state_text) equals that op record's context_sha1.
4. If the trace is STOP-only, first RETRIEVE state_text is also the final reader
   context recorded by final_context_sha1 and prompt_sha1.
5. For any trace whose final op record has state_text, its hash must equal both
   the final op context_sha1 and the trace final_context_sha1; prompt_sha1 is
   recomputed from that final state_text.

This script is read-only.  It does not validate semantic answer-bearing spans;
it only validates that trace text fields correspond to the prompt-visible memory
context recorded by the producer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from agents.concat_agent import QA_PROMPT
except Exception:  # pragma: no cover - fail closed in main with explicit row status
    QA_PROMPT = None


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def normalize_qid(qid: str) -> str:
    qid = str(qid)
    return qid if qid.startswith("longmemeval_") else f"longmemeval_{qid}"


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_responses(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in iter_jsonl(path):
        qid = normalize_qid(str(row.get("qid") or row.get("question_id") or ""))
        if not qid:
            continue
        if qid in out:
            duplicates.append(qid)
        out[qid] = row
    if duplicates:
        raise ValueError(f"duplicate response qids: {duplicates[:5]}")
    return out


def load_traces(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in iter_jsonl(path):
        if row.get("phase") != "qa":
            continue
        qhash = str(row.get("query_sha1") or "")
        if not qhash:
            continue
        if qhash in out:
            duplicates.append(qhash)
        out[qhash] = row
    if duplicates:
        raise ValueError(f"duplicate trace query_sha1 rows: {duplicates[:5]}")
    return out


def first_op_record(trace_row: dict[str, Any], allowed_ops: set[str]) -> dict[str, Any] | None:
    for record in trace_row.get("op_records") or []:
        if record.get("operation") in allowed_ops:
            return record
    return None


def final_op_record(trace_row: dict[str, Any]) -> dict[str, Any] | None:
    records = trace_row.get("op_records") or []
    return records[-1] if records else None


def prompt_hash(context: str, query: str) -> str | None:
    if QA_PROMPT is None:
        return None
    return sha1_text(f"Your memory:\n{context}\n\n{QA_PROMPT.format(query)}")


def verify_row(qid: str, response: dict[str, Any], trace: dict[str, Any] | None) -> dict[str, Any]:
    query = str(response.get("query") or "")
    qhash = sha1_text(query)
    out: dict[str, Any] = {
        "qid": qid,
        "query_sha1": qhash,
        "trace_found": int(trace is not None),
        "ok": 0,
        "violations": "",
    }
    violations: list[str] = []
    if trace is None:
        violations.append("missing_trace")
        out["violations"] = ";".join(violations)
        return out

    operations = list(trace.get("operations") or [])
    first = first_op_record(trace, {"RETRIEVE", "RETRIEVE_RECENT"})
    final = final_op_record(trace)
    out["operations"] = ",".join(str(op) for op in operations)
    out["trace_schema_version"] = str((first or {}).get("trace_schema_version") or "")
    out["first_retrieve_found"] = int(first is not None)
    out["final_record_found"] = int(final is not None)

    if first is None:
        violations.append("missing_first_retrieve")
    else:
        first_state = str(first.get("state_text") or "")
        first_state_hash = sha1_text(first_state) if first_state else ""
        first_context_hash = str(first.get("context_sha1") or "")
        out["first_state_text_present"] = int(bool(first_state))
        out["first_state_text_sha1"] = first_state_hash
        out["first_context_sha1"] = first_context_hash
        out["first_state_text_matches_context_sha1"] = int(bool(first_state_hash) and first_state_hash == first_context_hash)
        out["first_admitted_source_indices"] = json.dumps(first.get("admitted_source_indices") or [], sort_keys=True)
        out["first_retrieved_source_indices"] = json.dumps(first.get("retrieved_source_indices") or [], sort_keys=True)
        if not first_state:
            violations.append("missing_first_state_text")
        elif first_state_hash != first_context_hash:
            violations.append("first_state_text_context_sha1_mismatch")

    if final is None:
        violations.append("missing_final_record")
    else:
        final_state = str(final.get("state_text") or "")
        final_state_hash = sha1_text(final_state) if final_state else ""
        final_context_hash = str(final.get("context_sha1") or "")
        trace_final_hash = str(trace.get("final_context_sha1") or "")
        out["final_state_text_present"] = int(bool(final_state))
        out["final_state_text_sha1"] = final_state_hash
        out["final_record_context_sha1"] = final_context_hash
        out["trace_final_context_sha1"] = trace_final_hash
        out["final_state_text_matches_record_context_sha1"] = int(
            bool(final_state_hash) and final_state_hash == final_context_hash
        )
        out["final_state_text_matches_trace_final_context_sha1"] = int(
            bool(final_state_hash) and final_state_hash == trace_final_hash
        )
        out["final_admitted_source_indices"] = json.dumps(final.get("admitted_source_indices") or [], sort_keys=True)
        if final_state:
            recomputed_prompt_hash = prompt_hash(final_state, query)
            out["recomputed_prompt_sha1"] = recomputed_prompt_hash or ""
            out["trace_prompt_sha1"] = str(trace.get("prompt_sha1") or "")
            out["prompt_sha1_matches_final_state_text"] = int(
                recomputed_prompt_hash is not None and recomputed_prompt_hash == str(trace.get("prompt_sha1") or "")
            )
            if final_state_hash != final_context_hash:
                violations.append("final_state_text_record_context_sha1_mismatch")
            if final_state_hash != trace_final_hash:
                violations.append("final_state_text_trace_final_context_sha1_mismatch")
            if recomputed_prompt_hash is None:
                violations.append("qa_prompt_import_failed")
            elif recomputed_prompt_hash != str(trace.get("prompt_sha1") or ""):
                violations.append("prompt_sha1_final_state_text_mismatch")
        else:
            violations.append("missing_final_state_text")

    if operations in [["RETRIEVE"], ["RETRIEVE_RECENT"]] and first is not None:
        first_state_hash = out.get("first_state_text_sha1", "")
        trace_final_hash = str(trace.get("final_context_sha1") or "")
        out["stop_first_state_text_matches_trace_final_context_sha1"] = int(
            bool(first_state_hash) and first_state_hash == trace_final_hash
        )
        if first_state_hash != trace_final_hash:
            violations.append("stop_first_state_not_final_context")

    out["violations"] = ";".join(violations)
    out["ok"] = int(not violations)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fail-on-violation", action="store_true")
    args = parser.parse_args()

    responses = load_responses(args.responses)
    traces = load_traces(args.trace)
    rows = [
        verify_row(qid, response, traces.get(sha1_text(str(response.get("query") or ""))))
        for qid, response in sorted(responses.items())
    ]
    violation_counter: Counter[str] = Counter()
    for row in rows:
        for item in str(row.get("violations") or "").split(";"):
            if item:
                violation_counter[item] += 1
    report = {
        "status": "VALID_TRACE_STATE_TEXT_CONTRACT" if not violation_counter else "TRACE_STATE_TEXT_CONTRACT_VIOLATION",
        "responses": str(args.responses),
        "trace": str(args.trace),
        "n_responses": len(responses),
        "n_trace_rows": len(traces),
        "n_checked": len(rows),
        "ok_rows": sum(int(row["ok"]) for row in rows),
        "violation_rows": sum(1 for row in rows if not int(row["ok"])),
        "violations": dict(sorted(violation_counter.items())),
        "guardrails": [
            "This verifies trace text/hash/prompt contracts only.",
            "It does not prove session-level evidence contains the answer-bearing span.",
            "It does not use gold evidence, answers, or judge labels.",
            "P29 may use first RETRIEVE state_text as W0 only if this verifier passes for the STOP trace.",
        ],
    }
    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_violation and violation_counter:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
