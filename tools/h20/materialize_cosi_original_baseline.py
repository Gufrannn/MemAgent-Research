#!/usr/bin/env python3
"""Materialize CORAL's baseline report from authenticated Original curve artifacts."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recurrent.research.coral_evidence import (
    validate_curve_authority,
    validate_original_training_authority,
    validate_stable_s128_authority,
)
from recurrent.research.cosi import canonical_sha256, sha256_file
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_fixed_s128
from recurrent.research.stable_eval_identity import (
    MANIFEST_ROW_FIELDS,
    canonical_sha256 as stable_canonical_sha256,
    evaluation_trajectory_seed,
    stable_key,
    stable_trajectory_id,
)
from tools.h20.preflight_qwen25_7b_stable_i4x2 import _load_parquet_rows

INTERFACES = ("I", "Original5", "Original10", "Original15", "Original20", "Original25")
STEPS = {"I": 0, "Original5": 5, "Original10": 10, "Original15": 15,
         "Original20": 20, "Original25": 25}


def _mapping(value: object, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"COSI_BASELINE_NO_GO: {label} mapping")
    return dict(value)


def ground_truth_by_source_order(validation: Path, stable: Mapping[str, Any]):
    rows = _load_parquet_rows(validation)
    result = {}
    for frozen in stable["identity_payload"]["rows"]:
        order = int(frozen["source_order_index"])
        raw_position = int(frozen["raw_row_position"])
        if raw_position < 0 or raw_position >= len(rows):
            raise ValueError("COSI_BASELINE_NO_GO: S128 raw position outside validation")
        reward = _mapping(rows[raw_position].get("reward_model"), "reward_model")
        if "ground_truth" not in reward \
                or stable_canonical_sha256(reward["ground_truth"]) != frozen["ground_truth_hash"]:
            raise ValueError("COSI_BASELINE_NO_GO: parquet ground-truth identity/hash")
        result[order] = reward["ground_truth"]
    if set(result) != set(range(128)):
        raise ValueError("COSI_BASELINE_NO_GO: exact S128 ground-truth coverage")
    return result


def _artifact_paths(plan: Mapping[str, Any], details: Mapping[str, Any]):
    if details.get("root") != plan.get("root"):
        raise ValueError("COSI_BASELINE_NO_GO: final-inventory/root binding")
    root = Path(str(details.get("root", ""))).resolve()
    artifacts = details.get("artifacts")
    if not root.is_dir() or not isinstance(artifacts, dict) or set(artifacts) != {
        f"terminal/{int(plan['global_step'])}.jsonl", "trajectory_turns.jsonl",
        "execution_summary.json", "run.log",
    }:
        raise ValueError("COSI_BASELINE_NO_GO: authenticated artifact inventory schema")
    result = {}
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() \
                or ".." in Path(relative).parts or not isinstance(expected, dict) \
                or set(expected) != {"sha256", "size"}:
            raise ValueError("COSI_BASELINE_NO_GO: artifact inventory entry")
        candidate = root / relative
        path = candidate.resolve()
        if root not in path.parents or candidate.is_symlink() or not path.is_file() \
                or re.fullmatch(r"[0-9a-f]{64}", str(expected["sha256"])) is None \
                or type(expected["size"]) is not int or expected["size"] < 1 \
                or sha256_file(path) != expected["sha256"] \
                or path.stat().st_size != expected["size"]:
            raise ValueError("COSI_BASELINE_NO_GO: artifact differs from final inventory")
        result[relative] = {"absolute_path": str(path), **expected}
    return result


def score_interface(interface: str, *, plan: Mapping[str, Any], details: Mapping[str, Any],
                    stable: Mapping[str, Any], ground_truth: Mapping[int, object],
                    expected_digest: str):
    if int(plan.get("global_step", -1)) != STEPS[interface]:
        raise ValueError("COSI_BASELINE_NO_GO: interface/global-step plan")
    inventory = _artifact_paths(plan, details)
    terminal_relative = f"terminal/{STEPS[interface]}.jsonl"
    terminal_path = Path(inventory[terminal_relative]["absolute_path"])
    terminal = [json.loads(line) for line in terminal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if len(terminal) != 128 or [row.get("source_repeated_row") for row in terminal] != list(range(128)):
        raise ValueError("COSI_BASELINE_NO_GO: terminal denominator/order")
    frozen_rows = stable["identity_payload"]["rows"]
    metric_rows = []
    for index, (row, frozen) in enumerate(zip(terminal, frozen_rows)):
        if not isinstance(row, dict) or any(row.get(field) != frozen[field]
                                             for field in MANIFEST_ROW_FIELDS) \
                or row.get("eval_manifest_hash") != stable["eval_manifest_hash"] \
                or row.get("replica_id") != 0 \
                or row.get("source_order_index") != index:
            raise ValueError("COSI_BASELINE_NO_GO: terminal/stable identity drift")
        seed = evaluation_trajectory_seed(
            base_seed=2026, eval_manifest_hash=stable["eval_manifest_hash"],
            example_id=str(row["example_id"]), source_order_index=index, replica_id=0,
        )
        if row.get("trajectory_seed") != seed \
                or row.get("trajectory_id") != stable_trajectory_id(
                    eval_manifest_hash=stable["eval_manifest_hash"],
                    example_id=str(row["example_id"]), replica_id=0,
                    trajectory_seed=seed,
                ):
            raise ValueError("COSI_BASELINE_NO_GO: terminal seed/trajectory identity")
        output = row.get("output")
        if not isinstance(output, str):
            raise ValueError("COSI_BASELINE_NO_GO: terminal output is not text")
        scored = score_terminal_output(output, ground_truth[index])
        metric_rows.append({
            "stable_key": json.dumps(stable_key(row), separators=(",", ":")),
            "source_order_index": index,
            "eval_manifest_hash": row["eval_manifest_hash"],
            "example_id": row["example_id"], "replica_id": row["replica_id"],
            "trajectory_seed": row["trajectory_seed"],
            "trajectory_id": row["trajectory_id"], **scored,
        })
    digest = stable_canonical_sha256(metric_rows)
    metrics = summarize_fixed_s128(metric_rows)
    if digest != expected_digest \
            or details.get("independent_metric_rows_sha256") != digest \
            or details.get("metrics") != metrics:
        raise ValueError("COSI_BASELINE_NO_GO: independent metric rows/aggregate mismatch")
    # Detect mutation between inventory authentication and completed scoring.
    for relative, item in inventory.items():
        if sha256_file(item["absolute_path"]) != item["sha256"]:
            raise ValueError("COSI_BASELINE_NO_GO: source artifact changed during materialization")
    return metrics, digest, inventory


def materialize(manifest: Mapping[str, Any], *, original_resolved_path: Path,
                original_resolved_sha256: str, stable_resolved_path: Path,
                stable_resolved_sha256: str, validation: Path):
    authority = manifest.get("evidence_authority")
    if not isinstance(authority, dict) or set(authority) != {
        "original_s128_curve", "original_training", "stable_s128", "actual_loss",
    } or authority["actual_loss"] != {
        "status": "PENDING_ACTUAL_LOSS_LEDGER", "original_rank_ledgers_available": False,
        "forbid_metric_as_loss": True, "forbid_original_rerun": True,
    }:
        raise ValueError("COSI_BASELINE_NO_GO: evidence authority schema")
    training = validate_original_training_authority(
        authority["original_training"], resolved_path=original_resolved_path,
        expected_resolved_sha256=original_resolved_sha256,
    )
    stable_receipt = validate_stable_s128_authority(
        authority["stable_s128"], resolved_path=stable_resolved_path,
        expected_resolved_sha256=stable_resolved_sha256,
    )
    curve = validate_curve_authority(
        authority["original_s128_curve"], stable_resolved=stable_receipt["resolved"],
    )
    if sha256_file(validation) != manifest["data"]["validation_sha256"]:
        raise ValueError("COSI_BASELINE_NO_GO: validation parquet SHA")
    ground_truth = ground_truth_by_source_order(validation, stable_receipt["resolved"])
    plan = curve["resolved"].get("execution_binding", {}).get("interface_plan")
    interfaces = curve["final"].get("evidence", {}).get("interfaces")
    digests = authority["original_s128_curve"]["canonical_metric_row_digests"]
    if not isinstance(plan, dict) or set(plan) != set(INTERFACES) \
            or not isinstance(interfaces, dict) or set(interfaces) != set(INTERFACES):
        raise ValueError("COSI_BASELINE_NO_GO: curve interface inventory")
    aggregates, row_digests, inventories = {}, {}, {}
    for name in INTERFACES:
        aggregates[name], row_digests[name], inventories[name] = score_interface(
            name, plan=plan[name], details=interfaces[name],
            stable=stable_receipt["resolved"], ground_truth=ground_truth,
            expected_digest=digests[name],
        )
    report = {
        "schema": "memagent.coral.baseline-materialization.v2",
        "status": "PASS", "decision": "COSI_BASELINE_IMPORT_PASS",
        "source_mode": "authenticated_original_curve_artifacts_read_only",
        "source_commit": authority["original_training"]["git_commit"],
        "eval_manifest_hash": stable_receipt["resolved"]["eval_manifest_hash"],
        "stable_inventory_sha256": stable_receipt["stable_inventory_sha256"],
        "aggregates": aggregates, "metric_rows_sha256": row_digests,
        "artifact_inventories": inventories,
        "curve_authority": {key: curve[key] for key in (
            "final_sha256", "p0_sha256", "resolved_sha256", "ledger_sha256", "ledger_tail"
        )},
        "training_authority": {key: training[key] for key in (
            "resolved_sha256", "p0_sha256", "final_sha256", "ledger_sha256", "ledger_tail"
        )},
        "stable_authority": {key: stable_receipt[key] for key in (
            "resolved_sha256", "final_sha256", "ledger_sha256", "ledger_tail"
        )},
        "validation_sha256": sha256_file(validation),
        "ground_truth_source": "authenticated validation parquet by frozen raw_row_position",
        "dense_reward_used_as_performance": False,
        "original_rerun": False,
        "actual_loss_status": "PENDING_ACTUAL_LOSS_LEDGER",
        "materialized_at": datetime.now(timezone.utc).isoformat(),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    env = os.environ
    required = (
        "MEMAGENT_COSI_WORK_ROOT", "MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST",
        "MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST_SHA256",
        "MEMAGENT_COSI_S128_RESOLVED_MANIFEST", "MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256",
    )
    if any(not env.get(name) for name in required):
        raise ValueError("COSI_BASELINE_NO_GO: explicit authority environment missing")
    work = Path(env["MEMAGENT_COSI_WORK_ROOT"]).resolve()
    report = materialize(
        manifest,
        original_resolved_path=Path(env["MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST"]).resolve(),
        original_resolved_sha256=env["MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST_SHA256"],
        stable_resolved_path=Path(env["MEMAGENT_COSI_S128_RESOLVED_MANIFEST"]).resolve(),
        stable_resolved_sha256=env["MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256"],
        validation=work / "datasets/hotpotqa/hotpotqa_dev.parquet",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
