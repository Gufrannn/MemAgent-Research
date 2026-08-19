#!/usr/bin/env python3
"""Fail-closed, outcome-timing-aware OOTT stratum manifest validator."""
import argparse, json
from pathlib import Path

ALLOWED = {
    "P": {"effect_modifier_screening", "baseline_heterogeneity", "paired_transport_by_frozen_stratum"},
    "O": {"occupancy_distribution_shift", "common_support_description", "selected_post_policy_description"},
    "Y": {"scorer_calibration", "error_type_decomposition", "blinded_error_analysis"},
}
REQUIRED = {"name", "source_fields", "measured_at", "depends_on_policy_output",
            "depends_on_final_response", "depends_on_gold", "depends_on_reward",
            "expected_class", "requested_claim"}

def classify(item):
    missing = REQUIRED - set(item)
    if missing: raise ValueError("missing_fields:" + ",".join(sorted(missing)))
    if not isinstance(item["source_fields"], list) or not item["source_fields"]: raise ValueError("source_fields_empty")
    flags = [item[key] for key in ("depends_on_policy_output", "depends_on_final_response", "depends_on_gold", "depends_on_reward")]
    if not all(isinstance(value, bool) for value in flags): raise ValueError("dependency_flags_must_be_boolean")
    if item["depends_on_final_response"] or item["depends_on_gold"] or item["depends_on_reward"]: actual = "Y"
    elif item["depends_on_policy_output"]: actual = "O"
    elif item["measured_at"] in {"pre_policy", "pre_write", "initial_state"}: actual = "P"
    else: raise ValueError("unknown_provenance")
    if item["expected_class"] != actual: raise ValueError(f"class_mismatch:{item['expected_class']}!={actual}")
    if item["requested_claim"] not in ALLOWED[actual]: raise ValueError(f"claim_not_allowed_for_{actual}:{item['requested_claim']}")
    route = "O3a_pre_outcome_transport_heterogeneity" if actual == "P" else "O3b_induced_occupancy_support_map" if actual == "O" else "RED_calibration_error_analysis"
    return {"name": item["name"], "class": actual, "claim": item["requested_claim"], "route": route, "valid": True}

def validate_manifest(data):
    if not isinstance(data, dict): raise ValueError("manifest_must_be_object")
    rows = data.get("strata")
    if not isinstance(rows, list) or not rows: raise ValueError("manifest_strata_empty")
    classified = [classify(row) for row in rows]
    if any(row["class"] == "O" for row in classified) and data.get("unstratified_total_contrast_reported") is not True:
        raise ValueError("O_strata_require_unstratified_total_contrast")
    if data.get("manifest_frozen_before_stratum_analysis") is not True or not data.get("manifest_hash"):
        raise ValueError("manifest_not_frozen_or_unhashed")
    return {"valid": True, "strata": classified, "training_authorized": False,
            "adds_rollout": False, "adds_training": False, "causal_or_mediator_claim_authorized": False}

def self_test():
    base = {"name": "x", "source_fields": ["x"], "measured_at": "pre_policy", "depends_on_policy_output": False,
      "depends_on_final_response": False, "depends_on_gold": False, "depends_on_reward": False,
      "expected_class": "P", "requested_claim": "baseline_heterogeneity"}
    assert classify(base)["class"] == "P"
    occ = dict(base, name="generated_memory_length", measured_at="post_writer_pre_reader", depends_on_policy_output=True,
      expected_class="O", requested_claim="occupancy_distribution_shift")
    assert classify(occ)["class"] == "O"
    outcome = dict(base, name="exact_correct", measured_at="post_endpoint", depends_on_final_response=True,
      depends_on_gold=True, expected_class="Y", requested_claim="scorer_calibration")
    assert classify(outcome)["class"] == "Y"
    for bad in (dict(occ, requested_claim="effect_modifier_screening"), dict(outcome, requested_claim="occupancy_distribution_shift")):
        try: classify(bad)
        except ValueError as exc: assert "claim_not_allowed" in str(exc)
        else: raise AssertionError("post-policy/postoutcome claim was accepted")
    manifest = {"strata": [base, occ, outcome], "unstratified_total_contrast_reported": True,
      "manifest_frozen_before_stratum_analysis": True, "manifest_hash": "sha256:test"}
    assert validate_manifest(manifest)["training_authorized"] is False
    try: validate_manifest({**manifest, "unstratified_total_contrast_reported": False})
    except ValueError as exc: assert "unstratified" in str(exc)
    else: raise AssertionError("O stratum without total contrast accepted")
    print("oott_stratum_timing_self_test=ok")

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest"); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.manifest: parser.error("--manifest required")
    print(json.dumps(validate_manifest(json.loads(Path(args.manifest).read_text())), indent=2))
if __name__ == "__main__": main()
