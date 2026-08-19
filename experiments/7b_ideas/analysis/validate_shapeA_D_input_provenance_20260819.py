#!/usr/bin/env python3
"""Outcome-blind D-input provenance preflight; never authorizes analysis/training."""

from __future__ import annotations
import argparse, json
from pathlib import Path

M = "M_RELATIONAL_COMPRESSION"
F = "F_DETERMINISTIC_INDUCTIVE_BIAS"
A = "A_ACCESS_MISMATCH_NO_GO"
U = "UNKNOWN_PROVENANCE_NO_GO"
FORBIDDEN = {"harm", "outcome", "tau", "y_factual", "y_noop", "reward"}
REQUIRED = ("d_response_cells", "d_role_metadata", "d_target_metadata", "d_structural_ops", "d_normalization",
            "d_human_oracle_labels", "baseline_visible_cells", "baseline_visible_metadata", "baseline_marginal_summaries",
            "baseline_discarded_relational_structure", "baseline_contains_full_transcript_metadata",
            "d_deterministically_reconstructable", "d_object", "baseline_object", "d_budget", "baseline_budget")


def classify(value: dict) -> str:
    if set(value) & FORBIDDEN: raise ValueError("outcome-blind preflight received forbidden harm/outcome field")
    if any(key not in value for key in REQUIRED): return U
    if value["d_object"] != value["baseline_object"] or value["d_budget"] != value["baseline_budget"]:
        return A
    if bool(value["baseline_contains_full_transcript_metadata"]) and bool(value["d_deterministically_reconstructable"]):
        return F
    if value["baseline_discarded_relational_structure"] and value["baseline_marginal_summaries"]:
        return M
    return U


def validate(value: dict) -> dict:
    actual = classify(value); expected = value.get("expected_classification")
    if expected is not None and expected != actual:
        raise ValueError(f"expected classification mismatch: expected={expected}, actual={actual}")
    return {"classification": actual, "preflight_only": True, "reads_outcomes": False,
            "outcome_analysis_authorized": False, "training_authorized": False}


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--input", type=Path, required=True); args = p.parse_args()
    print(json.dumps(validate(json.loads(args.input.read_text())), sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
