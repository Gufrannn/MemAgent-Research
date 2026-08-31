#!/usr/bin/env python3
"""Export compact protocol-parity hashes from memory-operation traces.

The output intentionally avoids full query, answer, prompt, trace state_text,
and evidence text.  It is designed for H20/JumServer workflows where compact
copy-paste is easier than full artifact transfer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha1_obj(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def normalize_qid(qid: str) -> str:
    qid = str(qid)
    return qid if qid.startswith("longmemeval_") else f"longmemeval_{qid}"


def load_response_qids(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in iter_jsonl(path):
        qid = normalize_qid(str(row.get("qid") or row.get("question_id") or ""))
        query = str(row.get("query") or "")
        if qid and query:
            out[hashlib.sha1(query.encode("utf-8")).hexdigest()] = qid
    return out


def first_retrieve(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        if record.get("operation") in {"RETRIEVE", "RETRIEVE_RECENT"}:
            return record
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", action="append", required=True, help="name=responses.jsonl:trace.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for spec in args.operation:
        if "=" not in spec or ":" not in spec:
            raise ValueError("--operation must be name=responses.jsonl:trace.jsonl")
        name, paths = spec.split("=", 1)
        response_path_text, trace_path_text = paths.split(":", 1)
        qid_by_query_sha1 = load_response_qids(Path(response_path_text))
        for trace in iter_jsonl(Path(trace_path_text)):
            if trace.get("phase") != "qa":
                continue
            query_sha1 = str(trace.get("query_sha1") or "")
            records = trace.get("op_records") or []
            retrieve = first_retrieve(records)
            final_record = records[-1] if records else {}
            retrieved = retrieve.get("retrieved_source_indices")
            admitted = retrieve.get("admitted_source_indices")
            final_admitted = final_record.get("admitted_source_indices")
            final_n_admitted = final_record.get("n_admitted_sources", "")
            rows.append(
                {
                    "operation": name,
                    "qid": qid_by_query_sha1.get(query_sha1, ""),
                    "query_sha1": query_sha1,
                    "initial_state_sha1": retrieve.get("context_sha1", ""),
                    "final_state_sha1": trace.get("final_context_sha1", ""),
                    "retrieved_source_indices_sha1": sha1_obj(retrieved if retrieved is not None else []),
                    "admitted_source_indices_sha1": sha1_obj(admitted if admitted is not None else []),
                    "final_admitted_source_indices_sha1": sha1_obj(
                        final_admitted if final_admitted is not None else []
                    ),
                    "n_retrieved_sources": retrieve.get("n_retrieved_sources", ""),
                    "n_admitted_sources": retrieve.get("n_admitted_sources", ""),
                    "n_final_admitted_sources": final_n_admitted,
                    "final_admitted_empty": int(final_n_admitted == 0),
                    "final_operator_contract": final_record.get("operator_contract", ""),
                    "final_input_state": final_record.get("input_state", ""),
                    "final_contract_violation": final_record.get("contract_violation", ""),
                    "final_admitted_content_sha1_by_source": sha1_obj(
                        final_record.get("admitted_content_sha1_by_source") or {}
                    ),
                    "final_unassigned_admitted_content_line_count": final_record.get(
                        "unassigned_admitted_content_line_count", ""
                    ),
                    "prompt_sha1": trace.get("prompt_sha1", ""),
                    "model": trace.get("model", ""),
                    "temperature": trace.get("temperature", ""),
                    "top_p": trace.get("top_p", ""),
                    "max_tokens": trace.get("max_tokens", ""),
                    "top_k": trace.get("top_k", ""),
                    "trace_schema_version": retrieve.get("trace_schema_version", ""),
                    "ok": trace.get("ok", ""),
                    "error": trace.get("error", ""),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "operation",
        "qid",
        "query_sha1",
        "initial_state_sha1",
        "final_state_sha1",
        "retrieved_source_indices_sha1",
        "admitted_source_indices_sha1",
        "final_admitted_source_indices_sha1",
        "n_retrieved_sources",
        "n_admitted_sources",
        "n_final_admitted_sources",
        "final_admitted_empty",
        "final_operator_contract",
        "final_input_state",
        "final_contract_violation",
        "final_admitted_content_sha1_by_source",
        "final_unassigned_admitted_content_line_count",
        "prompt_sha1",
        "model",
        "temperature",
        "top_p",
        "max_tokens",
        "top_k",
        "trace_schema_version",
        "ok",
        "error",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["operation"], row["qid"])))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(rows),
                "operations": sorted({row["operation"] for row in rows}),
                "missing_qid_rows": sum(1 for row in rows if not row["qid"]),
                "missing_prompt_sha1_rows": sum(1 for row in rows if not row["prompt_sha1"]),
                "missing_trace_schema_rows": sum(1 for row in rows if not row["trace_schema_version"]),
                "final_admitted_empty_rows_by_operation": {
                    operation: sum(
                        1
                        for row in rows
                        if row["operation"] == operation and row["final_admitted_empty"] == 1
                    )
                    for operation in sorted({row["operation"] for row in rows})
                },
                "final_contract_violation_rows": sum(
                    1 for row in rows if str(row["final_contract_violation"]).lower() == "true"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
