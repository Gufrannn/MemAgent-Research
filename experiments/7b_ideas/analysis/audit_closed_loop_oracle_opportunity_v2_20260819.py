#!/usr/bin/env python3
"""CPU-only closed-loop v7 oracle/opportunity semantics audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.closed_loop_oracle import audit_oracle_opportunity  # noqa: E402


def _base(mode: str) -> dict:
    return {"schema_version": "closed-loop-oracle-opportunity-v2", "primary_mode": mode,
            "role": "orthogonal_auxiliary_not_terminal_pairwise_IUT_gate",
            "failure_can_veto_terminal_pairwise_IUT": False,
            "seed_or_replicate_increases_scientific_n": False,
            "optimizer_steps": 0, "new_rollouts": False}


def self_test() -> None:
    assigned = ["e0", "e1"]; policies = ["A", "B"]
    terminal = {("e0", "A"): 1.0, ("e0", "B"): 0.0,
                ("e1", "A"): 0.0, ("e1", "B"): 1.0}
    d = {**_base("D"), "deterministic_pointwise_package_oracle_authorized": True}
    report = audit_oracle_opportunity(d, primary_mode="D", terminal_values=terminal,
                                      assigned=assigned, policies=policies)
    assert report["V_fixed_star"] == .5 and report["V_pointwise_package_oracle"] == 1
    f = {**_base("F"), "seed_manifest_hindsight_envelope_only": True,
         "example_heterogeneity_oracle_claim_authorized": False}
    assert audit_oracle_opportunity(f, primary_mode="F", terminal_values=terminal,
                                    assigned=assigned, policies=policies)["status"].endswith("ONLY")
    s = {**_base("S"), "raw_pointwise_max_is_example_heterogeneity_oracle": False,
         "raw_pointwise_max_role": "coupling_dependent_hindsight_luck_envelope",
         "assignment_evaluation_seed_folds_independent": True,
         "plugin_max_winners_curse_acknowledged": True, "within_example_MC_reported": True,
         "stable_per_example_oracle_labels_authorized": False,
         "audit16_K4_scope": "luck_envelope_coupling_MC_feasibility_only", "K_per_fold": 4}
    records = []
    for fold_index, fold in enumerate(("assignment", "evaluation")):
        for policy_index, policy in enumerate(policies):
            for example_index, example in enumerate(assigned):
                for replicate in range(4):
                    records.append({"stable_example_id": example, "policy": policy, "fold": fold,
                                    "replicate": replicate,
                                    "seed": 10000 * fold_index + 1000 * policy_index +
                                            10 * example_index + replicate,
                                    "official_endpoint_value": .5})
    s["fold_records"] = records
    report = audit_oracle_opportunity(s, primary_mode="S", terminal_values={key: .5 for key in terminal},
                                      assigned=assigned, policies=policies)
    assert report["status"] == "STOCHASTIC_CONDITIONAL_MEAN_ORACLE_MC_FEASIBILITY_ONLY"
    bad = json.loads(json.dumps(s)); bad["fold_records"][-1]["seed"] = bad["fold_records"][0]["seed"]
    assert audit_oracle_opportunity(bad, primary_mode="S", terminal_values={key: .5 for key in terminal},
                                    assigned=assigned, policies=policies)["status"] == "ORACLE_OPPORTUNITY_INVALID"
    print("closed_loop_oracle_opportunity_v2_self_test=ok")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path)
    parser.add_argument("--terminal-ledger", type=Path); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.manifest or not args.terminal_ledger: parser.error("--manifest and --terminal-ledger required")
    manifest = json.loads(args.manifest.read_text()); rows = json.loads(args.terminal_ledger.read_text())
    terminal = {(str(row["stable_example_id"]), str(row["policy"])): float(row["official_endpoint_value"])
                for row in rows}
    assigned = sorted({key[0] for key in terminal}); policies = sorted({key[1] for key in terminal})
    print(json.dumps(audit_oracle_opportunity(manifest, primary_mode=manifest.get("primary_mode"),
      terminal_values=terminal, assigned=assigned, policies=policies), indent=2, sort_keys=True))


if __name__ == "__main__": main()
