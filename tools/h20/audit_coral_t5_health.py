#!/usr/bin/env python3
"""Fail-closed CORAL T5 health and authenticated Original-T5 comparison."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.cosi import canonical_sha256, checkpoint_sha256, sha256_file, validate_ledger
from recurrent.research.gate_a_execution import validate_jsonl_chain
from tools.h20.audit_qwen25_7b_cosi import build_anchor_comparison


def auth(path, decision):
    value = json.loads(path.read_text())
    unsigned = {key: child for key, child in value.items() if key != "report_sha256"}
    if value.get("status") != "PASS" or value.get("decision") != decision \
            or value.get("report_sha256") != canonical_sha256(unsigned):
        raise ValueError(f"CORAL_T5_NO_GO:{path}")
    return value


def validate_role_mechanism(updates, gate_rows):
    """Validate that CORAL was active at every T5 update and was synchronized."""
    expected = [
        (1, "memory_writer"), (2, "terminal_answer"), (3, "memory_writer"),
        (4, "terminal_answer"), (5, "memory_writer"),
    ]
    if [(int(row["global_step"]), row["phase"]) for row in updates] != expected:
        raise ValueError("CORAL_T5_NO_GO: phase schedule/activity")
    for row in updates:
        digest = row.get("actor_vllm_sampled_tensor_digest")
        if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None \
                or int(row.get("active_tokens", 0)) < 1 \
                or int(row.get("inactive_tokens", 0)) < 1 \
                or not math.isfinite(float(row.get("active_grad_norm", float("nan")))) \
                or float(row["active_grad_norm"]) <= 0:
            raise ValueError("CORAL_T5_NO_GO: role mechanism ledger")
    if validate_jsonl_chain(gate_rows):
        raise ValueError("CORAL_T5_NO_GO: Gate A ledger chain")
    post_sync = [row for row in gate_rows if row.get("record_type") == "weight_sync_summary"
                 and row.get("sync_kind") == "post_actor_update"]
    if [(int(row["global_step"]), row["sampled_tensor_digest"]) for row in post_sync] != [
        (int(item["global_step"]), item["actor_vllm_sampled_tensor_digest"])
        for item in updates
    ]:
        raise ValueError("CORAL_T5_NO_GO: role ledger / weight-sync binding")
    role_norms = {
        role: min(float(row["active_grad_norm"]) for row in updates if row["phase"] == role)
        for role in ("memory_writer", "terminal_answer")
    }
    return role_norms, post_sync


def validate_t5_evaluation(evaluation, baseline, checkpoint_hash):
    if evaluation.get("step") != 5 \
            or evaluation.get("checkpoint_inventory_sha256") != checkpoint_hash:
        raise ValueError("CORAL_T5_NO_GO: method eval/checkpoint binding")
    return build_anchor_comparison(5, evaluation, baseline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-import", required=True)
    parser.add_argument("--method-eval", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    rows = validate_ledger(root / "coral_execution_ledger.jsonl")
    updates = [row["payload"] for row in rows if row["payload"].get("event") == "coral_role_update"]
    gate_rows = [json.loads(line) for line in (root / "gate_a_execution_ledger.jsonl").read_text().splitlines() if line.strip()]
    role_norms, post_sync = validate_role_mechanism(updates, gate_rows)
    baseline_path = Path(args.baseline_import)
    expected_baseline_sha = os.environ.get("MEMAGENT_COSI_BASELINE_REPORT_SHA256", "")
    if re.fullmatch(r"[0-9a-f]{64}", expected_baseline_sha) is None \
            or sha256_file(baseline_path) != expected_baseline_sha:
        raise ValueError("CORAL_T5_NO_GO: external baseline report binding")
    baseline = auth(baseline_path, "COSI_BASELINE_IMPORT_PASS")
    evaluation = auth(Path(args.method_eval), "CORAL_S128_EVAL_PASS")
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    comparison = validate_t5_evaluation(evaluation, baseline, checkpoint_hash)
    method_f1 = comparison["method"]["token_f1"]
    original_f1 = comparison["original"]["token_f1"]
    passed = method_f1 >= original_f1 - 0.02
    report = {
        "schema": "memagent.coral.t5-health.v2",
        "status": "PASS" if passed else "FAIL",
        "decision": "COSI_T5_HEALTH_PASS" if passed else "CORAL_T5_NO_GO",
        "checkpoint_sha256": checkpoint_hash,
        "memory_writer_updates": 3, "terminal_answer_updates": 2,
        "minimum_active_gradient_norm_by_role": role_norms,
        "method_token_f1": method_f1, "original_token_f1": original_f1,
        "token_f1_delta": method_f1 - original_f1,
        "anchor_comparison": comparison,
        "weight_sync_records": len(post_sync),
    }
    report["report_sha256"] = canonical_sha256(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
