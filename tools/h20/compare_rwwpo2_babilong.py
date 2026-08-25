#!/usr/bin/env python3
"""Pair BABILong metric rows from two RWWPO-2 interfaces."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.rwwpo2_babilong import paired_descriptive_difference
from recurrent.research.rwwpo2_confirmation import sha256_file
from recurrent.research.stable_eval_identity import canonical_sha256


def read_rows(paths: list[str]) -> list[dict]:
    rows = []
    for value in paths:
        path = Path(value)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing/symlink metric rows: {value}")
        rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def authenticate_reports(
    report_paths: list[str], row_paths: list[str], *, expected_commit: str,
) -> list[dict]:
    if len(report_paths) != 2 or len(row_paths) != 2:
        raise ValueError("development comparison requires exactly two lengths")
    reports = []
    for report_value, row_value in zip(report_paths, row_paths, strict=True):
        report_path = Path(report_value)
        row_path = Path(row_value)
        if report_path.is_symlink() or not report_path.is_file() \
                or row_path.is_symlink() or not row_path.is_file():
            raise ValueError("missing/symlink audit input")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        declared = report.pop("report_sha256", None)
        actual = hashlib.sha256(json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        if declared != actual or report.get("git_commit") != expected_commit \
                or report.get("status") != "DIAGNOSTIC_ONLY" \
                or report.get("decision") != "RWWPO2_BABILONG_DEVELOPMENT_DIAGNOSTIC_ONLY" \
                or report.get("partition") != "development" \
                or Path(str(report.get("metric_rows_path", ""))).resolve() != row_path.resolve() \
                or report.get("metric_rows_sha256") != sha256_file(row_path):
            raise ValueError("development audit report authentication")
        reports.append({**report, "report_sha256": declared})
    if {report.get("length") for report in reports} != {"32k", "128k"} \
            or len({report.get("cell") for report in reports}) != 1 \
            or len({int(report.get("evaluation_step", -1)) for report in reports}) != 1:
        raise ValueError("development audit report cell/step/length binding")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--left-rows", nargs="+", required=True)
    parser.add_argument("--right-rows", nargs="+", required=True)
    parser.add_argument("--left-reports", nargs="+", required=True)
    parser.add_argument("--right-reports", nargs="+", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_BABILONG_COMPARE_NO_GO:checkout")
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise SystemExit("RWWPO2_BABILONG_COMPARE_NO_GO:append-only output")
    try:
        left_reports = authenticate_reports(
            args.left_reports, args.left_rows, expected_commit=head
        )
        right_reports = authenticate_reports(
            args.right_reports, args.right_rows, expected_commit=head
        )
        left_cell = str(left_reports[0]["cell"])
        right_cell = str(right_reports[0]["cell"])
        if left_cell == right_cell \
                or not args.left_name.startswith(left_cell + "-") \
                or not args.right_name.startswith(right_cell + "-"):
            raise ValueError("paired contrast requires two distinct cells")
        left = read_rows(args.left_rows)
        right = read_rows(args.right_rows)
        paired = paired_descriptive_difference(
            left, right, left_name=args.left_name, right_name=args.right_name
        )
    except ValueError as error:
        raise SystemExit("RWWPO2_BABILONG_COMPARE_NO_GO:" + str(error)) from error
    if len(left) != 48 or len(right) != 48:
        raise SystemExit("RWWPO2_BABILONG_COMPARE_NO_GO:development denominator must be 48")
    report = {
        "schema_version": "rwwpo2-babilong-paired-development-v1",
        "status": "DIAGNOSTIC_ONLY",
        "decision": "RWWPO2_BABILONG_PAIRED_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "git_commit": head, "left": args.left_name, "right": args.right_name,
        "left_rows_sha256": canonical_sha256(left),
        "right_rows_sha256": canonical_sha256(right),
        "left_audit_report_sha256": [
            report["report_sha256"] for report in left_reports
        ],
        "right_audit_report_sha256": [
            report["report_sha256"] for report in right_reports
        ],
        "paired": paired, "adaptive_development": True,
        "population_inference": False,
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "decision": report["decision"],
        "left": args.left_name, "right": args.right_name,
        "paired": paired, "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
