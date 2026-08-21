#!/usr/bin/env python3
"""Read-only research, T5, and final CORAL closure audit."""
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
from recurrent.research.coral import phase_for_step
from recurrent.research.cosi import canonical_sha256, checkpoint_sha256, sha256_file, validate_ledger
from recurrent.research.gate_a_execution import validate_jsonl_chain

ANCHOR_METRICS = ("normalized_exact_match", "token_f1", "format_success")


def auth(path, decision):
    value = json.loads(path.read_text())
    unsigned = {key: child for key, child in value.items() if key != "report_sha256"}
    if value.get("status") != "PASS" or value.get("decision") != decision \
            or value.get("report_sha256") != canonical_sha256(unsigned):
        raise ValueError(f"CORAL_AUDIT_NO_GO:{path}")
    return value


def _metric_summary(value, label):
    required = {
        "denominator", "normalized_exact_match", "token_f1", "format_success",
        "historical_sub_exact_match_diagnostic",
    }
    if not isinstance(value, dict) or set(value) != required \
            or type(value.get("denominator")) is not int \
            or value["denominator"] != 128:
        raise ValueError(f"CORAL_AUDIT_NO_GO: {label} metric schema")
    result = {"denominator": 128}
    for metric in required - {"denominator"}:
        raw = value[metric]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) \
                or not math.isfinite(float(raw)) or not 0 <= float(raw) <= 1:
            raise ValueError(f"CORAL_AUDIT_NO_GO: {label} metric range")
        result[metric] = float(raw)
    return result


def build_anchor_comparison(step, evaluation, baseline):
    """Authenticate one Method-vs-Original fixed-S128 anchor comparison."""
    if type(step) is not int or step not in (5, 10, 15, 20, 25) \
            or evaluation.get("step") != step:
        raise ValueError("CORAL_AUDIT_NO_GO: evaluation anchor mismatch")
    for field in ("eval_manifest_hash", "stable_inventory_sha256"):
        if not isinstance(evaluation.get(field), str) \
                or evaluation[field] != baseline.get(field):
            raise ValueError(f"CORAL_AUDIT_NO_GO: baseline/method {field} drift")
    original_key = f"Original{step}"
    aggregates = baseline.get("aggregates")
    if not isinstance(aggregates, dict) or original_key not in aggregates:
        raise ValueError("CORAL_AUDIT_NO_GO: Original anchor missing")
    method = _metric_summary(evaluation.get("metrics"), f"Method-T{step}")
    original = _metric_summary(aggregates[original_key], original_key)
    deltas = {metric: method[metric] - original[metric] for metric in ANCHOR_METRICS}
    return {
        "step": step,
        "method_interface": f"CORAL_T{step}",
        "original_interface": original_key,
        "denominator": 128,
        "method": {metric: method[metric] for metric in ANCHOR_METRICS},
        "original": {metric: original[metric] for metric in ANCHOR_METRICS},
        "method_minus_original": deltas,
        "population_inference": False,
    }


def summarize_anchor_curve(comparisons):
    if [row.get("step") for row in comparisons] not in (
        [5], [5, 10, 15, 20, 25],
    ):
        raise ValueError("CORAL_AUDIT_NO_GO: anchor curve inventory")
    summary = {}
    for metric in ANCHOR_METRICS:
        values = [row["method_minus_original"][metric] for row in comparisons]
        summary[metric] = {
            "mean_delta": math.fsum(values) / len(values),
            "worst_anchor_delta": min(values),
            "best_anchor_delta": max(values),
            "last_anchor_delta": values[-1],
        }
    return summary


def validate_resume_record(record, expected_source):
    """Require rank-complete model/optimizer/scheduler/RNG/data resume evidence."""
    expected_source = Path(expected_source).resolve()
    data_path = expected_source / "data.pt"
    if not isinstance(record, dict) \
            or int(record.get("global_step", -1)) != 5 \
            or record.get("resume_source") != str(expected_source) \
            or record.get("actor_model_optimizer_extra_loaded") is not True \
            or record.get("data_loaded") is not True \
            or not data_path.is_file() \
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("data_sha256"))) is None \
            or record["data_sha256"] != sha256_file(data_path):
        raise ValueError("CORAL_AUDIT_NO_GO: exact resume closure")
    acknowledgements = record.get("actor_load_worker_acks")
    if not isinstance(acknowledgements, list) or len(acknowledgements) != 2 \
            or any(not isinstance(ack, dict) or type(ack.get("rank")) is not int
                   for ack in acknowledgements) \
            or sorted(ack["rank"] for ack in acknowledgements) != [0, 1]:
        raise ValueError("CORAL_AUDIT_NO_GO: rank-complete resume acknowledgement")
    expected_rng = ["cpu", "cuda", "numpy", "random"]
    for ack in acknowledgements:
        if any(ack.get(field) is not True for field in (
            "model_loaded", "optimizer_loaded", "extra_loaded", "rng_restored",
            "lr_scheduler_loaded",
        )) or ack.get("rng_state_keys") != expected_rng:
            raise ValueError("CORAL_AUDIT_NO_GO: training state not fully restored")
        integer_fields = (
            "optimizer_state_entry_count", "optimizer_step_entry_count",
            "optimizer_step_min", "optimizer_step_max", "lr_scheduler_last_epoch",
        )
        if any(type(ack.get(field)) is not int or int(ack[field]) < 1
               for field in integer_fields) \
                or ack["optimizer_step_min"] != ack["optimizer_step_max"]:
            raise ValueError("CORAL_AUDIT_NO_GO: optimizer/scheduler resume state")


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
    gate_reports = {}
    for name, decision, expected_variable in (
        ("paper_framing_review", "CORAL_PAPER_FRAMING_GO", "MEMAGENT_COSI_PAPER_REVIEW_SHA256"),
        ("coral_e0", "CORAL_E0_PASS", "MEMAGENT_COSI_E0_REPORT_SHA256"),
        ("coral_e1_final_report", "CORAL_E1_PASS", "MEMAGENT_COSI_E1_REPORT_SHA256"),
        ("baseline_import", "COSI_BASELINE_IMPORT_PASS", "MEMAGENT_COSI_BASELINE_REPORT_SHA256"),
    ):
        gate_path = gate_root / f"{name}.json"
        expected_sha = os.environ.get(expected_variable, "")
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None \
                or sha256_file(gate_path) != expected_sha:
            raise ValueError(f"CORAL_AUDIT_NO_GO: external gate binding {name}")
        gate_reports[name] = auth(gate_path, decision)
        gates[name] = gate_reports[name]["report_sha256"]
    if args.stage == "research":
        updates = []
        gate_tail = None
        anchor_comparisons = []
        anchor_curve = None
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
        anchor_comparisons = []
        for step in anchors:
            evaluation = auth(root / f"fixed_s128/T{step}/certificates/final_report.json", "CORAL_S128_EVAL_PASS")
            checkpoint = training_root / f"global_step_{step}"
            if evaluation["checkpoint_inventory_sha256"] != checkpoint_sha256(checkpoint):
                raise ValueError("CORAL_AUDIT_NO_GO: checkpoint/evaluation tamper")
            anchor_comparisons.append(build_anchor_comparison(
                step, evaluation, gate_reports["baseline_import"],
            ))
            gates[f"s128_t{step}"] = evaluation["report_sha256"]
        anchor_curve = summarize_anchor_curve(anchor_comparisons)
        if args.stage == "final":
            resume = [row for row in gate_rows if row.get("record_type") == "resume_load"]
            if len(resume) != 1:
                raise ValueError("CORAL_AUDIT_NO_GO: exact resume closure")
            validate_resume_record(resume[0], training_root / "global_step_5")
    report = {
        "schema": "memagent.coral.audit.v3", "status": "PASS",
        "decision": f"CORAL_{args.stage.upper()}_AUDIT_PASS", "stage": args.stage,
        "gate_hashes": gates, "update_records": len(updates),
        "gate_a_ledger_tail_sha256": gate_tail,
        "anchor_comparisons": anchor_comparisons,
        "anchor_curve_summary": anchor_curve,
        "comparison_estimand": (
            "descriptive Method-minus-authenticated-Original fixed-S128 aggregates; "
            "no population inference"
        ),
        "performance_claim_authorized": False,
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
