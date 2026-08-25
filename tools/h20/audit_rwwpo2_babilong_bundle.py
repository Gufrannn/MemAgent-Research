#!/usr/bin/env python3
"""Independent semantic audit of a materialized RWWPO-2 BABILong bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.rwwpo2_babilong import (
    CHUNK_SIZE, LENGTHS, MAX_CHUNKS, TASK_DEPTH, adapt_partition,
    partition_indices, validate_frozen_contract,
)
from recurrent.research.stable_eval_identity import (
    MANIFEST_ROW_FIELDS, canonical_sha256, validate_resolved_manifest,
)
from tools.h20.materialize_rwwpo2_babilong import read_source_bundle, sha256_file
from tools.h20.preflight_qwen25_7b_stable_i4x2 import _load_parquet_rows


def authenticated_report(path: Path, *, file_sha256: str, decision: str) -> dict:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != file_sha256:
        raise ValueError("materialization report file SHA")
    report = json.loads(path.read_text(encoding="utf-8"))
    declared = report.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared != actual or report.get("status") != "PASS" or report.get("decision") != decision:
        raise ValueError("materialization report authentication")
    return {**report, "report_sha256": declared}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "manifest", "manifest-sha256", "source-root", "source-manifest-sha256",
        "tokenizer-root", "bundle-root", "materialization-report-sha256",
        "expected-commit", "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_BABILONG_BUNDLE_AUDIT_NO_GO:checkout")
    manifest_path = Path(args.manifest).resolve()
    source_root = Path(args.source_root).resolve()
    tokenizer_root = Path(args.tokenizer_root).resolve()
    bundle_root = Path(args.bundle_root).resolve()
    output = Path(args.output)
    if any(path.is_symlink() for path in (manifest_path, source_root, tokenizer_root, bundle_root)) \
            or output.exists() or output.is_symlink():
        raise SystemExit("RWWPO2_BABILONG_BUNDLE_AUDIT_NO_GO:symlink/append-only")
    if sha256_file(manifest_path) != args.manifest_sha256:
        raise SystemExit("RWWPO2_BABILONG_BUNDLE_AUDIT_NO_GO:manifest SHA")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_frozen_contract(manifest)
        source_report, sources = read_source_bundle(
            source_root, expected_sha256=args.source_manifest_sha256,
            expected_commit=head, adapter_manifest_sha256=args.manifest_sha256,
        )
        materialized = authenticated_report(
            bundle_root / "materialization_report.json",
            file_sha256=args.materialization_report_sha256,
            decision="RWWPO2_BABILONG_MATERIALIZATION_PASS",
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_root, local_files_only=True, trust_remote_code=False
        )
        expected = adapt_partition(
            sources, partition=materialized["partition"],
            context_token_length=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        )
    except (ImportError, KeyError, OSError, ValueError) as error:
        raise SystemExit("RWWPO2_BABILONG_BUNDLE_AUDIT_NO_GO:" + str(error)) from error
    audited_outputs = []
    output_by_length = {row["length"]: row for row in materialized["outputs"]}
    for length in LENGTHS:
        item = output_by_length.get(length)
        if not isinstance(item, dict):
            raise SystemExit("RWWPO2_BABILONG_BUNDLE_AUDIT_NO_GO:output inventory")
        validation = Path(item["validation_path"])
        resolved_path = Path(item["resolved_path"])
        if validation.parent.resolve() != bundle_root or resolved_path.parent.resolve() != bundle_root \
                or validation.is_symlink() or resolved_path.is_symlink() \
                or sha256_file(validation) != item["validation_sha256"] \
                or sha256_file(resolved_path) != item["resolved_sha256"]:
            raise SystemExit("RWWPO2_BABILONG_BUNDLE_AUDIT_NO_GO:artifact path/SHA")
        actual_rows = _load_parquet_rows(validation)
        expected_rows, expected_identities = expected[length]
        if canonical_sha256(actual_rows) != canonical_sha256(expected_rows):
            raise SystemExit("RWWPO2_BABILONG_BUNDLE_AUDIT_NO_GO:parquet reconstruction")
        resolved = validate_resolved_manifest(json.loads(resolved_path.read_text(encoding="utf-8")))
        binding = resolved.get("babilong_binding", {})
        frozen = resolved["identity_payload"]["rows"]
        if resolved["eval_manifest_hash"] != item["eval_manifest_hash"] \
                or frozen != expected_identities \
                or binding.get("validation_sha256") != item["validation_sha256"] \
                or binding.get("source_bundle_report_sha256") != source_report["report_sha256"] \
                or binding.get("partition") != materialized["partition"] \
                or binding.get("length") != length \
                or int(binding.get("chunk_size", -1)) != CHUNK_SIZE \
                or int(binding.get("max_chunks", -1)) != MAX_CHUNKS[length] \
                or binding.get("outcomes_consumed_for_membership") is not False:
            raise SystemExit("RWWPO2_BABILONG_BUNDLE_AUDIT_NO_GO:resolved binding")
        expected_membership = {
            (task, source_index)
            for task in TASK_DEPTH
            for source_index in partition_indices(length, task, materialized["partition"])
        }
        actual_membership = {
            (str(row["babilong_task"]), int(row["babilong_source_index"]))
            for row in frozen
        }
        if actual_membership != expected_membership \
                or [int(row["source_order_index"]) for row in frozen] != list(range(len(frozen))) \
                or any(int(row["context_token_count"]) > CHUNK_SIZE * MAX_CHUNKS[length]
                       for row in frozen) \
                or any(any(field not in row for field in MANIFEST_ROW_FIELDS) for row in frozen):
            raise SystemExit("RWWPO2_BABILONG_BUNDLE_AUDIT_NO_GO:membership/identity")
        audited_outputs.append({
            "length": length, "rows": len(actual_rows),
            "validation_sha256": item["validation_sha256"],
            "resolved_sha256": item["resolved_sha256"],
            "eval_manifest_hash": resolved["eval_manifest_hash"],
            "membership_sha256": canonical_sha256(sorted(actual_membership)),
            "maximum_context_tokens": max(int(row["context_token_count"]) for row in frozen),
        })
    report = {
        "schema_version": "rwwpo2-babilong-bundle-audit-v1",
        "status": "PASS", "decision": "RWWPO2_BABILONG_BUNDLE_AUDIT_PASS",
        "git_commit": head, "adapter_manifest_sha256": args.manifest_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_report_sha256": source_report["report_sha256"],
        "materialization_report_file_sha256": args.materialization_report_sha256,
        "materialization_report_sha256": materialized["report_sha256"],
        "partition": materialized["partition"], "outputs": audited_outputs,
        "output_inventory_sha256": canonical_sha256(audited_outputs),
        "outcomes_consumed_for_membership": False,
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "partition": report["partition"], "outputs": audited_outputs,
        "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
