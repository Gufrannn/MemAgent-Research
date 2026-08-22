#!/usr/bin/env python3
"""Cheap fail-closed T5 training-health audit; no evaluation or baseline access."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.cosi import canonical_sha256, checkpoint_sha256, validate_ledger
from recurrent.research.gate_a_execution import validate_jsonl_chain


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
                or float(row["active_grad_norm"]) <= 0 \
                or not math.isfinite(float(row.get("active_pg_loss", float("nan")))):
            raise ValueError("CORAL_T5_NO_GO: role mechanism ledger")
    if validate_jsonl_chain(gate_rows):
        raise ValueError("CORAL_T5_NO_GO: Gate A ledger chain")
    post_sync = [row for row in gate_rows if row.get("record_type") == "weight_sync_summary"
                 and row.get("sync_kind") == "post_actor_update"
                 and int(row.get("global_step", -1)) <= 5]
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


def select_t5_updates(all_updates, *, exact_boundary):
    if exact_boundary and [int(row.get("global_step", -1)) for row in all_updates] \
            != [1, 2, 3, 4, 5]:
        raise ValueError("CORAL_T5_NO_GO: recovery requires exact step5 ledger boundary")
    return [row for row in all_updates if int(row.get("global_step", -1)) <= 5]


def validate_exact_gate_boundary(gate_rows):
    advanced = [row for row in gate_rows
                if type(row.get("global_step")) is int and row["global_step"] > 5]
    if advanced:
        raise ValueError("CORAL_T5_NO_GO: Gate-A ledger advanced beyond exact step5 boundary")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exact-boundary", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root)
    rows = validate_ledger(root / "coral_execution_ledger.jsonl")
    all_updates = [row["payload"] for row in rows
                   if row["payload"].get("event") == "coral_role_update"]
    updates = select_t5_updates(all_updates, exact_boundary=args.exact_boundary)
    gate_rows = [json.loads(line) for line in (root / "gate_a_execution_ledger.jsonl").read_text().splitlines() if line.strip()]
    if args.exact_boundary:
        validate_exact_gate_boundary(gate_rows)
    role_norms, post_sync = validate_role_mechanism(updates, gate_rows)
    checkpoint = Path(args.checkpoint)
    if not (checkpoint / "actor").is_dir() or not (checkpoint / "data.pt").is_file():
        raise ValueError("CORAL_T5_NO_GO: incomplete T5 checkpoint")
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    report = {
        "schema": "memagent.coral.t5-health.v3",
        "status": "PASS", "decision": "COSI_T5_HEALTH_PASS",
        "checkpoint_sha256": checkpoint_hash,
        "memory_writer_updates": 3, "terminal_answer_updates": 2,
        "minimum_active_gradient_norm_by_role": role_norms,
        "finite_policy_loss_updates": 5,
        "weight_sync_records": len(post_sync),
        "evaluation_performed": False, "original_baseline_accessed": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != report:
            raise ValueError("CORAL_T5_NO_GO: existing health certificate differs")
    else:
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
