#!/usr/bin/env python3
"""Aggregate authenticated attempt DAG segments into the performance-free R50 gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CELLS = ("A", "B", "C", "D", "E")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list:
    """Descriptive 95% interval over independent rollout-round clusters."""
    if total <= 0:
        return [None, None]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, *, commit: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"attempt audit is missing or a symlink: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    expected_decision = f"RWWPO2_R{int(row.get('target_round', -1))}_ATTEMPT_AUDIT_PASS"
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != expected_decision \
            or row.get("git_commit") != commit \
            or row.get("program_version") != "rwwpo2-k2" \
            or row.get("s128_consumed") is not False \
            or row.get("performance_evaluated") is not False:
        raise ValueError(f"invalid attempt audit: {path}")
    return {**row, "report_sha256": declared, "file_sha256": sha256_file(path),
            "path": str(path.resolve())}


def verified_resolved(path: Path, *, expected_sha: str, commit: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("resolved contract is missing or a symlink")
    if sha256_file(path) != expected_sha:
        raise ValueError("resolved contract SHA mismatch")
    row = json.loads(path.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != "RWWPO2_RESOLVED_CONTRACT_PASS" \
            or row.get("git_commit") != commit:
        raise ValueError("resolved contract receipt invalid")
    return {**row, "report_sha256": declared}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-audit", action="append", required=True)
    parser.add_argument("--resolved-contract", required=True)
    parser.add_argument("--resolved-contract-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_R50_NO_GO:checkout")
    try:
        raw_resolved = Path(args.resolved_contract)
        if raw_resolved.is_symlink() \
                or any(Path(path).is_symlink() for path in args.attempt_audit):
            raise ValueError("R50 input symlink")
        resolved = verified_resolved(
            raw_resolved.resolve(),
            expected_sha=args.resolved_contract_sha256, commit=head,
        )
        reports = [verified(Path(path).resolve(), commit=head) for path in args.attempt_audit]
    except ValueError as error:
        raise SystemExit("RWWPO2_R50_NO_GO:" + str(error)) from error
    manifest = resolved["manifest"]
    seeds = tuple(int(value) for value in manifest["training"]["mechanism_seeds"])
    expected_assignments = {(cell, seed) for cell in CELLS for seed in seeds}
    grouped = defaultdict(list)
    for row in reports:
        if row.get("resolved_contract_file_sha256") != \
                args.resolved_contract_sha256 \
                or row.get("resolved_contract_report_sha256") != \
                resolved["report_sha256"] \
                or row.get("source_manifest_sha256") != \
                resolved["source_manifest_sha256"]:
            raise SystemExit("RWWPO2_R50_NO_GO:segment contract binding")
        key = (str(row["cell"]), int(row["experiment_seed"]))
        if key not in expected_assignments or int(row["target_round"]) > 50:
            raise SystemExit("RWWPO2_R50_NO_GO:unexpected cell/seed/round")
        grouped[key].append(row)
    if set(grouped) != expected_assignments:
        raise SystemExit("RWWPO2_R50_NO_GO:matrix coverage")

    gates = resolved["r50_mechanism_gate"]
    assignment_summaries = {}
    for key in sorted(grouped):
        segments = sorted(grouped[key], key=lambda row: int(row["start_round"]))
        if int(segments[0]["start_round"]) != 1 \
                or int(segments[-1]["target_round"]) != 50:
            raise SystemExit(f"RWWPO2_R50_NO_GO:lineage endpoints {key}")
        for previous, current in zip(segments, segments[1:]):
            if int(current["start_round"]) != int(previous["target_round"]) + 1:
                raise SystemExit(f"RWWPO2_R50_NO_GO:lineage gap/overlap {key}")
            lineage = current.get("lineage_parent") or {}
            if lineage.get("checkpoint_inventory_event_sha256") != \
                    previous.get("checkpoint_inventory_record_sha256") \
                    or lineage.get("failed_suffix_imported") is not False:
                raise SystemExit(f"RWWPO2_R50_NO_GO:lineage parent binding {key}")
        eligible = sum(int(row["mechanism"]["eligible_round_count"]) for row in segments)
        exposed = sum(int(row["mechanism"]["exposed_round_count"]) for row in segments)
        activated = sum(int(row["mechanism"][
            "geometry_activation_count_given_exposed"]) for row in segments)
        exposure_rate = exposed / eligible
        activation_rate = activated / exposed if exposed else 0.0
        round_diagnostics = [
            diagnostic
            for row in segments
            for diagnostic in row["mechanism"].get("round_diagnostics", [])
        ]
        round_ids = [int(row["round_id"]) for row in round_diagnostics]
        if round_ids != list(range(1, 51)) \
                or any(row.get("cluster_unit") != "rollout_round"
                       or row.get("eligible") is not True for row in round_diagnostics) \
                or sum(bool(row["exposed"]) for row in round_diagnostics) != exposed \
                or sum(bool(row["geometry_activated_given_exposed"])
                       for row in round_diagnostics) != activated:
            raise SystemExit(f"RWWPO2_R50_NO_GO:round-cluster evidence {key}")
        if eligible < int(gates["r50_minimum_eligible_rounds_per_host"]):
            raise SystemExit(f"RWWPO2_R50_NO_GO:eligible rounds {key}")
        if exposed < int(gates["r50_minimum_exposed_rounds_per_host"]) \
                or exposure_rate < float(gates["r50_minimum_exposure_rate_per_host"]):
            raise SystemExit(f"RWWPO2_R50_NO_GO:off-behavior exposure {key}")
        if activated < int(gates[
                "r50_minimum_geometry_activation_count_given_exposed"]) \
                or activation_rate < float(gates[
                    "r50_minimum_geometry_activation_rate_given_exposed"]):
            raise SystemExit(f"RWWPO2_R50_NO_GO:geometry activation {key}")
        assignment_summaries[f"{key[0]}:seed{key[1]}"] = {
            "segment_count": len(segments), "eligible_rounds": eligible,
            "exposed_rounds": exposed, "exposure_rate": exposure_rate,
            "activated_exposed_rounds": activated,
            "geometry_activation_rate_given_exposed": activation_rate,
            "exposure_rate_wilson95_rollout_round_cluster": wilson_interval(
                exposed, eligible),
            "geometry_activation_wilson95_exposed_round_cluster": wilson_interval(
                activated, exposed),
            "uncertainty_unit": "rollout_round; prompt-root LOO within round",
            "round_diagnostics": round_diagnostics,
            "inner2_salvaged_partial_commit_count": sum(int(row["mechanism"][
                "inner2_salvaged_partial_commit_count"]) for row in segments),
            "segment_receipts": [{
                "path": row["path"], "file_sha256": row["file_sha256"],
                "report_sha256": row["report_sha256"],
                "start_round": row["start_round"], "target_round": row["target_round"],
            } for row in segments],
        }

    b_salvage = sum(
        row["inner2_salvaged_partial_commit_count"] for name, row in
        assignment_summaries.items() if name.startswith("B:")
    )
    if b_salvage < 1:
        raise SystemExit("RWWPO2_R50_NO_GO:no B hard-rollback counterfactual aperture")
    report = {
        "schema_version": "rwwpo2-r50-program-gate-v1",
        "status": "PASS", "decision": "RWWPO2_R50_MECHANISM_GATE_PASS",
        "git_commit": head, "program_version": "rwwpo2-k2",
        "resolved_contract_file_sha256": args.resolved_contract_sha256,
        "resolved_contract_report_sha256": resolved["report_sha256"],
        "source_manifest_sha256": resolved["source_manifest_sha256"],
        "mechanism_seeds": list(seeds), "cells": list(CELLS),
        "assignment_summaries": assignment_summaries,
        "b_shared_proposal_hard_rollback_counterfactual_count": b_salvage,
        "s128_consumed": False, "performance_evaluated": False,
        "decision_scope": "performance-free mechanism and engineering gate only",
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "b_salvage": b_salvage, "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
