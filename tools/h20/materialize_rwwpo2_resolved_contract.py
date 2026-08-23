#!/usr/bin/env python3
"""Bind the pre-R50 numeric oracle into an immutable RWWPO-2 run contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"
NUMERIC_FIELDS = ("tau_theta", "tau_logprob", "tau_gradient", "tau_coefficient")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_receipt(path: Path, *, commit: str) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != "RWWPO2_NUMERIC_ORACLE_PASS" \
            or row.get("git_commit") != commit:
        raise ValueError("numeric oracle receipt is not authentic for this commit")
    return {**row, "report_sha256": declared}


def verified_oracle_audit(path: Path, *, commit: str, oracle: dict) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != "RWWPO2_NUMERIC_ORACLE_AUDIT_PASS" \
            or row.get("git_commit") != commit \
            or row.get("oracle_report_sha256") != oracle["report_sha256"] \
            or row.get("thresholds") != oracle.get("thresholds"):
        raise ValueError("numeric oracle audit is not authentic for this commit")
    return {**row, "report_sha256": declared}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--manifest-schema", required=True)
    parser.add_argument("--manifest-schema-sha256", required=True)
    parser.add_argument("--numeric-oracle", required=True)
    parser.add_argument("--numeric-oracle-sha256", required=True)
    parser.add_argument("--numeric-oracle-audit", required=True)
    parser.add_argument("--numeric-oracle-audit-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:checkout")

    raw_inputs = tuple(Path(value) for value in (
        args.manifest, args.manifest_schema,
        args.numeric_oracle, args.numeric_oracle_audit,
    ))
    if any(path.is_symlink() for path in raw_inputs):
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:source symlink")
    manifest_path, schema_path, oracle_path, oracle_audit_path = (
        path.resolve() for path in raw_inputs
    )
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:output already exists")
    if sha256_file(manifest_path) != args.manifest_sha256:
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:manifest SHA")
    if sha256_file(schema_path) != args.manifest_schema_sha256:
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:manifest schema SHA")
    if sha256_file(oracle_path) != args.numeric_oracle_sha256:
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:oracle file SHA")
    if sha256_file(oracle_audit_path) != args.numeric_oracle_audit_sha256:
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:oracle audit file SHA")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        import jsonschema
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(manifest)
    except Exception as error:
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:manifest schema validation") from error
    oracle = verified_receipt(oracle_path, commit=head)
    oracle_audit = verified_oracle_audit(oracle_audit_path, commit=head, oracle=oracle)
    thresholds = oracle.get("thresholds")
    if set(thresholds or {}) != set(NUMERIC_FIELDS):
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:threshold fields")
    if any(not math.isfinite(float(thresholds[name])) or float(thresholds[name]) <= 0
           for name in NUMERIC_FIELDS):
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:threshold values")
    if manifest.get("program") != "RWWPO-2" \
            or manifest.get("method", {}).get("numeric_thresholds") != \
            "PENDING_PRE_R50_BF16_FSDP_ORACLE_FAILS_GPU_PREFLIGHT":
        raise SystemExit("RWWPO2_RESOLVE_NO_GO:manifest numeric contract")

    resolved = {
        "schema_version": "rwwpo2-resolved-contract-v1",
        "status": "PASS",
        "decision": "RWWPO2_RESOLVED_CONTRACT_PASS",
        "git_commit": head,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": args.manifest_sha256,
        "source_manifest_schema_path": str(schema_path),
        "source_manifest_schema_sha256": args.manifest_schema_sha256,
        "numeric_oracle_path": str(oracle_path),
        "numeric_oracle_file_sha256": args.numeric_oracle_sha256,
        "numeric_oracle_report_sha256": oracle["report_sha256"],
        "numeric_oracle_audit_path": str(oracle_audit_path),
        "numeric_oracle_audit_file_sha256": args.numeric_oracle_audit_sha256,
        "numeric_oracle_audit_report_sha256": oracle_audit["report_sha256"],
        "numeric_thresholds": {name: float(thresholds[name]) for name in NUMERIC_FIELDS},
        "behavior_coefficient_tolerance": float(thresholds["tau_coefficient"]),
        "behavior_gradient_tolerance": float(thresholds["tau_gradient"]),
        "maximum_root_loo_feasibility_flip_fraction": float(
            manifest["method"]["maximum_root_loo_feasibility_flip_fraction"]
        ),
        "r50_mechanism_gate": {
            name: manifest["method"][name]
            for name in (
                "r50_minimum_eligible_rounds_per_host",
                "r50_minimum_exposed_rounds_per_host",
                "r50_minimum_exposure_rate_per_host",
                "r50_minimum_geometry_activation_count_given_exposed",
                "r50_minimum_geometry_activation_rate_given_exposed",
            )
        },
        "proposal_schedule": manifest["method"]["proposal_schedule"],
        "manifest": manifest,
    }
    raw = json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False)
    resolved["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(resolved, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": resolved["decision"],
        "output": str(output.resolve()), "sha256": sha256_file(output),
        "thresholds": resolved["numeric_thresholds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
