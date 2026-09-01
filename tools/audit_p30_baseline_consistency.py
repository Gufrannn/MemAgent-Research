#!/usr/bin/env python3
"""P30.1 baseline consistency audit.

This is a read-only report repair/audit script.  It does not train, generate,
or re-score model outputs.  It checks whether a P30 method table's reported
"best fixed" identity is the raw mean-best fixed policy or a tie/cost-aware
fixed policy.  If a cheaper policy within tie_eps was labeled "best fixed",
the qualitative comparison may remain valid, but the reported identity/delta
must be renamed or recomputed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIXED_METHOD_PREFIXES = {
    "Greedy_STOP",
    "Lexical_BM25",
    "TFIDF_Jaccard",
    "Graph_Bridge",
    "Temporal_Session",
    "STOP",
    "TFIDF_full_best_fixed",
    "Graph",
    "Lexical",
    "Temporal",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def policy_from_choices(row: dict[str, str]) -> str:
    method = row.get("Method") or row.get("method") or ""
    method_policy = {
        "Greedy_STOP": "stop",
        "STOP": "stop",
        "Lexical_BM25": "repack_lexical_bm25",
        "Lexical": "repack_lexical_bm25",
        "TFIDF_Jaccard": "repack_tfidf_jaccard",
        "TFIDF_full_best_fixed": "repack_tfidf_jaccard",
        "Graph_Bridge": "repack_graph_bridge",
        "Graph": "repack_graph_bridge",
        "Temporal_Session": "repack_temporal_session",
        "Temporal": "repack_temporal_session",
    }
    if method in method_policy:
        return method_policy[method]
    choices = row.get("Choices") or row.get("choices") or ""
    if not choices:
        return method
    try:
        payload = json.loads(choices)
    except json.JSONDecodeError:
        only = choices.strip()
        if only == "stop80":
            return "stop"
        if only == "tfidf80":
            return "repack_tfidf_jaccard"
        if only == "graph80":
            return "repack_graph_bridge"
        if only == "lexical80":
            return "repack_lexical_bm25"
        if only == "temporal80":
            return "repack_temporal_session"
        return only
    if isinstance(payload, dict) and len(payload) == 1:
        return str(next(iter(payload)))
    return ""


def is_fixed_method(row: dict[str, str]) -> bool:
    method = row.get("Method") or row.get("method") or ""
    if method in FIXED_METHOD_PREFIXES:
        return True
    if method.startswith("Margin_Oracle") or method.startswith("SSA"):
        return False
    return False


def reported_fixed_policy(rows: list[dict[str, str]]) -> str:
    candidates: list[str] = []
    for row in rows:
        for key in (
            "Full_TieCost_Fixed_Policy",
            "Full_Best_Fixed_Policy",
            "full_best_fixed_policy",
        ):
            value = row.get(key, "")
            if value:
                candidates.append(value)
    if not candidates:
        for row in rows:
            method = row.get("Method") or row.get("method") or ""
            if "full_best_fixed" in method.lower():
                policy = policy_from_choices(row)
                if policy:
                    candidates.append(policy)
    if not candidates:
        return ""
    return Counter(candidates).most_common(1)[0][0]


def audit_one(path: Path, label: str, tie_eps: float) -> dict[str, Any]:
    rows = read_csv(path)
    fixed = [row for row in rows if is_fixed_method(row)]
    if not fixed:
        raise ValueError(f"no fixed-policy rows detected in {path}")
    fixed_rows: list[dict[str, Any]] = []
    for row in fixed:
        method = row.get("Method") or row.get("method") or ""
        policy = policy_from_choices(row)
        fixed_rows.append(
            {
                "method": method,
                "policy": policy,
                "mean": parse_float(row.get("Mean") or row.get("mean")),
            }
        )
    raw_best = max(fixed_rows, key=lambda row: row["mean"])
    reported = reported_fixed_policy(rows)
    reported_row = next((row for row in fixed_rows if row["policy"] == reported), None)
    reported_mean = reported_row["mean"] if reported_row else float("nan")
    gap_raw_minus_reported = raw_best["mean"] - reported_mean
    if not reported:
        status = "NO_REPORTED_FIXED_POLICY"
    elif raw_best["policy"] == reported:
        status = "CONSISTENT_RAW_BEST"
    elif gap_raw_minus_reported <= tie_eps + 1e-12:
        status = "NAMING_INCONSISTENCY_TIECOST_WITHIN_TIE_EPS"
    else:
        status = "INCONSISTENT_NEEDS_MATRIX_AUDIT"
    return {
        "label": label,
        "path": str(path),
        "tie_eps": tie_eps,
        "status": status,
        "raw_best_method": raw_best["method"],
        "raw_best_policy": raw_best["policy"],
        "raw_best_mean": raw_best["mean"],
        "reported_fixed_policy": reported,
        "reported_fixed_mean": reported_mean,
        "raw_minus_reported": gap_raw_minus_reported,
        "fixed_policy_means_json": json.dumps(fixed_rows, ensure_ascii=False, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--tie-eps", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reports = [audit_one(Path(path), label, args.tie_eps) for label, path in args.table]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "p30_1_baseline_consistency_audit.csv", reports)
    (args.output_dir / "p30_1_baseline_consistency_audit.json").write_text(
        json.dumps(
            {
                "status": "P30_1_BASELINE_CONSISTENCY_AUDIT_COMPLETE",
                "scope": "read-only method-table audit; no generation/training/rescoring",
                "reports": reports,
                "interpretation": [
                    "CONSISTENT_RAW_BEST means the reported fixed policy is the highest raw mean fixed policy.",
                    "NAMING_INCONSISTENCY_TIECOST_WITHIN_TIE_EPS means a lower-mean tie/cost-aware policy was reported as best fixed; rename/recompute deltas but this does not imply matrix mixing.",
                    "INCONSISTENT_NEEDS_MATRIX_AUDIT means raw minus reported exceeds tie_eps and may indicate table/matrix mismatch.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
