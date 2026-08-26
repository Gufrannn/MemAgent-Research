#!/usr/bin/env python3
"""Signed descriptive B/D/E comparison on adaptive fixed-S128 T20."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    "token_f1", "normalized_exact_match", "format_success",
    "historical_sub_exact_match_diagnostic",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(path: Path, *, commit: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("certificate path")
    row = json.loads(path.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared != actual or row.get("status") != "DIAGNOSTIC_ONLY" \
            or row.get("decision") != "RWWPO_T20_S128_DIAGNOSTIC_ONLY" \
            or row.get("git_commit") != commit or int(row.get("step", -1)) != 20 \
            or set(row.get("metrics", {})) != {
                "denominator", *METRICS
            } or int(row["metrics"]["denominator"]) != 128:
        raise ValueError("certificate identity")
    return {**row, "report_sha256": declared, "file_sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", required=True)
    parser.add_argument("--d", required=True)
    parser.add_argument("--e", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_T20_COMPARE_NO_GO:checkout")
    try:
        rows = {
            cell: verified(Path(path).resolve(), commit=head)
            for cell, path in (("B", args.b), ("D", args.d), ("E", args.e))
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit("RWWPO2_T20_COMPARE_NO_GO:" + str(error)) from error
    if len({row["resolved_manifest_sha256"] for row in rows.values()}) != 1:
        raise SystemExit("RWWPO2_T20_COMPARE_NO_GO:identity manifest drift")
    aggregates = {cell: row["metrics"] for cell, row in rows.items()}
    differences = {}
    for left, right in (("B", "D"), ("E", "D"), ("B", "E")):
        differences[f"{left}-{right}"] = {
            name: float(aggregates[left][name]) - float(aggregates[right][name])
            for name in METRICS
        }
    report = {
        "schema_version": "rwwpo2-hotpot-t20-bde-diagnostic-v1",
        "status": "DIAGNOSTIC_ONLY",
        "decision": "RWWPO2_HOTPOT_T20_BDE_DIAGNOSTIC_ONLY",
        "git_commit": head,
        "step": 20,
        "adaptive_dataset_role": "development_diagnostic_not_blind_final",
        "primary_descriptive_metric": "token_f1",
        "aggregates": aggregates,
        "paired_descriptive_differences": differences,
        "input_certificates": {
            cell: {
                "path": str(Path(path).resolve()),
                "file_sha256": rows[cell]["file_sha256"],
                "report_sha256": rows[cell]["report_sha256"],
            }
            for cell, path in (("B", args.b), ("D", args.d), ("E", args.e))
        },
        "claim_scope": "single_seed_fixed_S128_descriptive_only",
    }
    raw = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    report["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise SystemExit("RWWPO2_T20_COMPARE_NO_GO:output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "decision": report["decision"],
        "token_f1": {cell: aggregates[cell]["token_f1"] for cell in "BDE"},
        "token_f1_differences": {
            name: row["token_f1"] for name, row in differences.items()
        },
        "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
