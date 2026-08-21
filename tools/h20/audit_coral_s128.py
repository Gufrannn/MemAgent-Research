#!/usr/bin/env python3
"""Independently score and authenticate one fixed-S128 CORAL interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.cosi import canonical_sha256, checkpoint_inventory
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_fixed_s128
from recurrent.research.stable_eval_identity import validate_resolved_manifest
from tools.h20.preflight_qwen25_7b_stable_i4x2 import _load_parquet_rows


def mapping(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("CORAL_S128_NO_GO: parquet mapping field")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, checkpoint = Path(args.evaluation_root).resolve(), Path(args.checkpoint).resolve()
    p0 = json.loads((root / "certificates/p0.json").read_text())
    unsigned = {key: value for key, value in p0.items() if key != "report_sha256"}
    if p0.get("decision") != "CORAL_S128_P0_PASS" \
            or p0.get("report_sha256") != canonical_sha256(unsigned):
        raise ValueError("CORAL_S128_NO_GO: P0 authentication")
    if canonical_sha256(checkpoint_inventory(checkpoint)) != p0["checkpoint_inventory_sha256"]:
        raise ValueError("CORAL_S128_NO_GO: checkpoint changed after P0")
    resolved = json.loads((root / "certificates/resolved_eval_manifest.json").read_text())
    validate_resolved_manifest(resolved)
    if resolved["eval_manifest_hash"] != p0["eval_manifest_hash"]:
        raise ValueError("CORAL_S128_NO_GO: identity manifest drift")
    summary = json.loads((root / "execution_summary.json").read_text())
    required_summary = {
        "interface_id": f"CORAL_T{p0['step']}", "eval_manifest_hash": p0["eval_manifest_hash"],
        "resolved_runtime_config_sha256": p0["runtime_config_sha256"],
        "global_step": p0["step"], "actor_update_calls": 0,
        "optimizer_step_calls": 0, "checkpoint_save_calls": 0,
        "resume_mode": "actor_only_eval", "weight_source": "actor_checkpoint",
        "checkpoint_load_mode": "actor_only", "checkpoint_source": str(checkpoint),
    }
    for key, value in required_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"CORAL_S128_NO_GO: execution summary {key}")
    terminal_path = root / "terminal" / f"{p0['step']}.jsonl"
    terminal = [json.loads(line) for line in terminal_path.read_text().splitlines() if line.strip()]
    if len(terminal) != 128:
        raise ValueError("CORAL_S128_NO_GO: exact terminal denominator")
    rows = _load_parquet_rows(Path(args.validation))
    ground_truth = {}
    for source in rows:
        extra = mapping(source.get("extra_info"))
        reward = mapping(source.get("reward_model"))
        ground_truth[str(int(extra["index"]))] = reward["ground_truth"]
    identities = resolved["identity_payload"]["rows"]
    expected_ids = [str(row["example_id"]) for row in identities]
    actual_ids = [str(row.get("example_id")) for row in terminal]
    if actual_ids != expected_ids or len(set(actual_ids)) != 128:
        raise ValueError("CORAL_S128_NO_GO: stable identity/order drift")
    scored = []
    for row in terminal:
        example_id = str(row["example_id"])
        if example_id not in ground_truth:
            raise ValueError("CORAL_S128_NO_GO: ground-truth join")
        metric = score_terminal_output(row.get("output", ""), ground_truth[example_id])
        scored.append({"stable_id": example_id, **metric})
    aggregate = summarize_fixed_s128(scored)
    report = {
        "schema": "memagent.coral.s128-report.v1", "status": "PASS",
        "decision": "CORAL_S128_EVAL_PASS", "step": p0["step"],
        "checkpoint_inventory_sha256": p0["checkpoint_inventory_sha256"],
        "eval_manifest_hash": p0["eval_manifest_hash"], "metrics": aggregate,
        "metric_source": "terminal text independently joined to parquet ground truth",
        "dense_reward_used_as_performance": False,
        "stable_inventory_sha256": canonical_sha256(actual_ids),
        "terminal_jsonl_sha256": __import__("hashlib").sha256(terminal_path.read_bytes()).hexdigest(),
    }
    report["report_sha256"] = canonical_sha256(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
