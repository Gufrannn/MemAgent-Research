#!/usr/bin/env python3
"""Read-only research, T5, and final CORAL closure audit."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.coral import phase_for_step
from recurrent.research.cosi import canonical_sha256, checkpoint_sha256, validate_ledger
from recurrent.research.gate_a_execution import validate_jsonl_chain


def auth(path, decision):
    value = json.loads(path.read_text())
    unsigned = {key: child for key, child in value.items() if key != "report_sha256"}
    if value.get("status") != "PASS" or value.get("decision") != decision \
            or value.get("report_sha256") != canonical_sha256(unsigned):
        raise ValueError(f"CORAL_AUDIT_NO_GO:{path}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--training-root")
    parser.add_argument("--stage", choices=("research", "t5", "final"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    work = Path(os.environ["MEMAGENT_COSI_WORK_ROOT"]).resolve()
    gate_root = work / "logs/cosi_preflight/certificates"
    gates = {}
    for name, decision in (
        ("paper_framing_review", "CORAL_PAPER_FRAMING_GO"),
        ("coral_e0", "CORAL_E0_PASS"),
        ("coral_e1_final_report", "CORAL_E1_PASS"),
        ("baseline_import", "COSI_BASELINE_IMPORT_PASS"),
    ):
        gates[name] = auth(gate_root / f"{name}.json", decision)["report_sha256"]
    if args.stage == "research":
        updates = []
        gate_tail = None
    else:
        training_root = Path(args.training_root).resolve()
        ledger = validate_ledger(root / "coral_execution_ledger.jsonl")
        updates = [row["payload"] for row in ledger if row["payload"].get("event") == "coral_role_update"]
        required_last = 5 if args.stage == "t5" else 25
        if [(int(row["global_step"]), row["phase"]) for row in updates] != [
            (step, phase_for_step(step)) for step in range(1, required_last + 1)
        ]:
            raise ValueError("CORAL_AUDIT_NO_GO: exact update schedule/continuation")
        if any(re.fullmatch(r"[0-9a-f]{64}", str(row.get("actor_vllm_sampled_tensor_digest"))) is None
               for row in updates):
            raise ValueError("CORAL_AUDIT_NO_GO: missing role weight digest")
        gate_rows = [json.loads(line) for line in (root / "gate_a_execution_ledger.jsonl").read_text().splitlines() if line.strip()]
        if validate_jsonl_chain(gate_rows):
            raise ValueError("CORAL_AUDIT_NO_GO: Gate A ledger")
        sync = [row for row in gate_rows if row.get("record_type") == "weight_sync_summary"
                and row.get("sync_kind") == "post_actor_update"]
        if [(int(row["global_step"]), row["sampled_tensor_digest"]) for row in sync] != [
            (int(row["global_step"]), row["actor_vllm_sampled_tensor_digest"]) for row in updates
        ]:
            raise ValueError("CORAL_AUDIT_NO_GO: update/sync mismatch")
        gate_tail = gate_rows[-1]["record_sha256"]
        health = auth(gate_root / "t5_health.json", "COSI_T5_HEALTH_PASS")
        gates["t5_health"] = health["report_sha256"]
        anchors = (5,) if args.stage == "t5" else (5, 10, 15, 20, 25)
        for step in anchors:
            evaluation = auth(root / f"fixed_s128/T{step}/certificates/final_report.json", "CORAL_S128_EVAL_PASS")
            checkpoint = training_root / f"global_step_{step}"
            if evaluation["checkpoint_inventory_sha256"] != checkpoint_sha256(checkpoint):
                raise ValueError("CORAL_AUDIT_NO_GO: checkpoint/evaluation tamper")
            gates[f"s128_t{step}"] = evaluation["report_sha256"]
        if args.stage == "final":
            resume = [row for row in gate_rows if row.get("record_type") == "resume_load"]
            if len(resume) != 1 or int(resume[0].get("global_step", -1)) != 5 \
                    or resume[0].get("actor_model_optimizer_extra_loaded") is not True \
                    or resume[0].get("data_loaded") is not True:
                raise ValueError("CORAL_AUDIT_NO_GO: exact resume closure")
    report = {
        "schema": "memagent.coral.audit.v2", "status": "PASS",
        "decision": f"CORAL_{args.stage.upper()}_AUDIT_PASS", "stage": args.stage,
        "gate_hashes": gates, "update_records": len(updates),
        "gate_a_ledger_tail_sha256": gate_tail,
    }
    report["report_sha256"] = canonical_sha256(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
