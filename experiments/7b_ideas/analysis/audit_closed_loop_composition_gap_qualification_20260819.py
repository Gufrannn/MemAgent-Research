#!/usr/bin/env python3
"""CLI and self-test for the closed-loop composition-gap qualification firewall."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.closed_loop_composition import audit_composition_gap  # noqa: E402


def _manifest(gaps=(1.0, -1.0)):
    rows = []
    for index, gap in enumerate(gaps):
        rows.append({"stable_example_id": f"e{index}", "direct_source_row_hash": f"{index + 10:064x}",
                     "splice_row_hash": f"{index + 20:064x}", "direct_gc_terminal": 1.0,
                     "splice_terminal": 1.0 + gap, "best_control_terminal": 0.0})
    contract = {key: f"{index + 100:064x}" for index, key in enumerate((
        "initial_manifest_hash", "checkpoint_hash", "writer_reader_contract_hash",
        "horizon_contract_hash", "endpoint_definition_hash", "missing_rule_hash"))}
    return {"schema_version": "closed-loop-composition-gap-v1",
            "gap_direction": "V_splice_minus_V_direct_GC", "complete_per_example_bijection": True,
            "splice_algorithm_frozen_before_outcome": True, "source_rows_frozen_before_outcome": True,
            "composition_gap_actionability_gate": False, "composition_gap_rescues_terminal_IUT": False,
            "feedback_claim_authorized": False, "optimizer_steps": 0, "new_rollouts": False,
            "direct_contract": dict(contract), "splice_contract": dict(contract),
            "splice_algorithm_hash": "a" * 64,
            "prefrozen_source_row_hashes": [row["direct_source_row_hash"] for row in rows],
            "composition_transport_SESOI": .5, "rows": rows}


def self_test():
    zero_mean = audit_composition_gap(_manifest())
    assert zero_mean["status"] == "COMPOSITION_GAP_QUALIFIED"
    assert zero_mean["signed_mean_composition_gap"] == 0.0
    assert zero_mean["composition_gap_MAE_error_mass"] == 1.0
    mismatch = _manifest((.6, .6))
    mismatch["splice_contract"]["endpoint_definition_hash"] = "f" * 64
    result = audit_composition_gap(mismatch)
    assert result["status"] == "COMPOSITION_GAP_NOT_IDENTIFIED" and not result["outcomes_read"]
    assert sum((.6, .6)) / 2 == .6  # pseudo-gap that the firewall refuses to interpret
    missing = _manifest(); missing["rows"].pop()
    assert audit_composition_gap(missing)["status"] == "COMPOSITION_GAP_NOT_IDENTIFIED"
    print("closed_loop_composition_gap_qualification_self_test=ok")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.manifest: parser.error("--manifest required")
    print(json.dumps(audit_composition_gap(json.loads(args.manifest.read_text())), indent=2, sort_keys=True))


if __name__ == "__main__": main()
