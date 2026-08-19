#!/usr/bin/env python3
"""Fail-closed OOTT O2 checkpoint/seed-coupling preflight; reads no outcomes."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _fail(reason):
    raise ValueError(f"O2_SEED_COUPLING_NO_GO: {reason}")


def validate(value):
    if value.get("schema_version") != "oott-o2-seed-coupling-v1":
        _fail("schema version mismatch")
    if value.get("o2_existing_gate_authorized") is not True:
        return {"status": "O2_NOT_AUTHORIZED", "outcomes_read": False, "training_authorized": False,
                "new_rollouts_authorized": False, "queue_changed": False}
    required = {
        "primary_checkpoint_seed_namespaces_disjoint": True,
        "checkpoint_specific_seed_namespaces": True,
        "seeds_per_example_per_checkpoint": 4,
        "within_checkpoint_seed_mean_before_example_contrast": True,
        "independent_unit": "stable_example_id",
        "seed_repeats_increase_independent_n": False,
        "per_example_sign_certification_authorized": False,
        "coupling_selected_by_pilot_variance": False,
        "coupling_selected_by_narrower_ci": False,
        "coupling_selected_by_direction": False,
        "policy_marginal_estimand_changes_with_coupling": False,
        "optimizer_steps": 0,
        "new_rollouts": False,
    }
    wrong = {key: (value.get(key), expected) for key, expected in required.items() if value.get(key) != expected}
    if wrong:
        _fail(f"primary contract failed {wrong}")
    namespaces = value.get("primary_seed_namespaces")
    if not isinstance(namespaces, dict) or set(namespaces) != {"T25", "T200"}:
        _fail("primary namespaces must be declared for T25 and T200")
    seed_sets = []
    for checkpoint in ("T25", "T200"):
        seeds = namespaces[checkpoint]
        if not isinstance(seeds, list) or len(seeds) != 4 or len(set(seeds)) != 4:
            _fail(f"{checkpoint} requires exactly four unique primary seeds")
        seed_sets.append(set(seeds))
    if seed_sets[0] & seed_sets[1]:
        _fail("T25 and T200 primary seed namespaces overlap")
    if not SHA256.fullmatch(str(value.get("primary_seed_manifest_hash", ""))):
        _fail("primary seed manifest hash missing")
    crn = value.get("crn_sensitivity", {})
    requested = crn.get("requested")
    if not isinstance(requested, bool):
        _fail("CRN sensitivity requested flag missing")
    classification = "PRIMARY_INDEPENDENT_NAMESPACE_ONLY"
    if requested:
        crn_required = {
            "corrected_per_trajectory_seeds": True,
            "bci_status": "PASS_COUPLED",
            "role": "implementation_coupling_sensitivity_only",
            "natural_cross_policy_trajectory_identity": False,
            "individual_or_causal_paired_effect_authorized": False,
            "namespace_prefrozen": True,
        }
        wrong = {key: (crn.get(key), expected) for key, expected in crn_required.items()
                 if crn.get(key) != expected}
        if wrong:
            _fail(f"CRN sensitivity contract failed {wrong}")
        if not SHA256.fullmatch(str(crn.get("coupling_manifest_hash", ""))):
            _fail("CRN coupling manifest hash missing")
        seeds = crn.get("seed_namespace")
        if not isinstance(seeds, list) or len(seeds) != 4 or len(set(seeds)) != 4:
            _fail("CRN sensitivity requires four unique prefrozen seeds")
        if set(seeds) & (seed_sets[0] | seed_sets[1]):
            _fail("CRN namespace overlaps primary namespace")
        if value.get("primary_crn_direction_conflict") is True:
            classification = "COUPLING_SENSITIVE_STOCHASTIC_TRANSPORT"
        elif value.get("primary_crn_direction_conflict") is not False:
            _fail("primary/CRN direction-conflict declaration missing")
    return {"status": "O2_SEED_COUPLING_PREFLIGHT_PASS", "primary_role": "policy_marginal_estimand",
            "crn_role": "implementation_coupling_sensitivity_only" if requested else "not_requested",
            "classification": classification, "checkpoint_seed_means_before_contrast": True,
            "independent_unit": "stable_example_id", "seed_repeats_increase_n": False,
            "coupling_changes_monte_carlo_variance_only": True, "outcomes_read": False,
            "training_authorized": False, "new_rollouts_authorized": False, "queue_changed": False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.manifest.read_text())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
