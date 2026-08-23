#!/usr/bin/env python3
"""Independent read-only audit of the two-rank RWWPO-2 numeric oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MULTIPLIER = 16.0
GRADIENT_SKETCH_CHUNK_ELEMENTS = 8_388_608
STREAMED_ORACLE_MICROBATCHES = 7
STREAMED_ORACLE_SEQUENCE_LENGTH = 8191
FLOORS = {
    "tau_theta": 1e-12,
    "tau_logprob": 1e-6,
    "tau_gradient": 1e-8,
    "tau_coefficient": 1e-10,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", required=True)
    parser.add_argument("--oracle-report-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:checkout")
    raw_root = Path(args.oracle_root)
    if raw_root.is_symlink():
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:root symlink")
    root = raw_root.resolve()
    report_path = root / "numeric_oracle.json"
    if report_path.is_symlink() or not report_path.is_file() \
            or sha256_file(report_path) != args.oracle_report_sha256:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:report identity")
    row = json.loads(report_path.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != "RWWPO2_NUMERIC_ORACLE_PASS" \
            or row.get("git_commit") != head or int(row.get("world_size", 0)) != 2:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:receipt")
    if float(row.get("threshold_multiplier", -1)) != MULTIPLIER \
            or int(row.get("gradient_sketch_chunk_elements", -1)) != \
                GRADIENT_SKETCH_CHUNK_ELEMENTS \
            or row.get("streamed_replay_calibration") != {
                "microbatches": STREAMED_ORACLE_MICROBATCHES,
                "sequence_length": STREAMED_ORACLE_SEQUENCE_LENGTH,
                "active_response_tokens": 1024,
                "synthetic_label_free": True,
            } \
            or row.get("threshold_floors") != FLOORS:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:threshold rule")
    observed = row.get("observed", {})
    if set(observed) != {
        "repeated_logprob_max_abs", "repeated_gradient_projection_relative_l2",
        "streamed_replay_gradient_projection_relative_l2",
        "save_load_parameter_relative_l2", "save_load_logprob_max_abs",
        "save_load_gradient_projection_relative_l2",
        "behavior_actual_loss_logprob_max_abs",
        "behavior_actual_loss_coefficient_max_abs",
        "behavior_actual_loss_gradient_projection_relative_l2",
        "allreduce_max_abs",
    } or any(not math.isfinite(float(value)) or float(value) < 0
             for value in observed.values()):
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:observed numerics")
    expected_thresholds = {
        "tau_theta": max(FLOORS["tau_theta"], MULTIPLIER * float(
            observed["save_load_parameter_relative_l2"])),
        "tau_logprob": max(FLOORS["tau_logprob"], MULTIPLIER * max(
            float(observed["repeated_logprob_max_abs"]),
            float(observed["save_load_logprob_max_abs"]),
            float(observed["behavior_actual_loss_logprob_max_abs"]))),
        "tau_gradient": max(FLOORS["tau_gradient"], MULTIPLIER * max(
            float(observed["repeated_gradient_projection_relative_l2"]),
            float(observed["streamed_replay_gradient_projection_relative_l2"]),
            float(observed["save_load_gradient_projection_relative_l2"]),
            float(observed["behavior_actual_loss_gradient_projection_relative_l2"]))),
        "tau_coefficient": max(FLOORS["tau_coefficient"], MULTIPLIER * float(
            observed["behavior_actual_loss_coefficient_max_abs"])),
    }
    if row.get("thresholds") != expected_thresholds \
            or float(observed["allreduce_max_abs"]) != 0.0:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:threshold reconstruction")
    gpu_pair = row.get("gpu_pair")
    binding = row.get("gpu_binding")
    if not isinstance(gpu_pair, list) or len(gpu_pair) != 2 \
            or gpu_pair != sorted(set(int(value) for value in gpu_pair)) \
            or not isinstance(binding, list) or len(binding) != 2 \
            or any("NVIDIA H20" not in value or "GPU-" not in value for value in binding):
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:GPU binding")
    state_inventory = []
    evidence = row.get("rank_state_evidence", [])
    if sorted(int(item.get("rank", -1)) for item in evidence) != [0, 1]:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:rank closure")
    for item in evidence:
        relative = item.get("relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute() \
                or re.fullmatch(r"state/rank_[01]\.pt", relative) is None:
            raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:state path")
        path = root / relative
        if path.is_symlink() or not path.is_file() or root not in path.resolve().parents \
                or path.stat().st_size != int(item.get("state_size", -1)) \
                or sha256_file(path) != item.get("state_sha256"):
            raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:state evidence")
        state_inventory.append({
            "rank": int(item["rank"]), "relative_path": relative,
            "size": path.stat().st_size, "sha256": sha256_file(path),
        })
    model_path = Path(row.get("model_path", ""))
    if not model_path.is_absolute() or not model_path.joinpath("config.json").is_file() \
            or sha256_file(model_path / "config.json") != row.get("model_config_sha256"):
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:model identity")
    audit = {
        "schema_version": "rwwpo2-numeric-oracle-audit-v1",
        "status": "PASS", "decision": "RWWPO2_NUMERIC_ORACLE_AUDIT_PASS",
        "git_commit": head, "oracle_root": str(root),
        "oracle_report_file_sha256": args.oracle_report_sha256,
        "oracle_report_sha256": declared, "thresholds": expected_thresholds,
        "gradient_sketch_chunk_elements": GRADIENT_SKETCH_CHUNK_ELEMENTS,
        "streamed_replay_calibration": row["streamed_replay_calibration"],
        "gpu_pair": gpu_pair, "gpu_binding": binding,
        "rank_state_inventory": state_inventory,
    }
    raw = json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False)
    audit["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": audit["decision"],
        "output": str(output.resolve()), "thresholds": expected_thresholds,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
