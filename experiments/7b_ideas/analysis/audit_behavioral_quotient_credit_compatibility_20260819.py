#!/usr/bin/env python3
"""CPU-only BQCC controlled-pair audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.bqcc import audit_bqcc, canonical_manifest_sha256  # noqa: E402


def _manifest(invariant_delta: float, necessary_delta: float) -> dict:
    rows = []
    for example in ("e0", "e1"):
        rows.extend([
          {"stable_example_id": example, "pair_id": "inv", "path_id": "p0", "pair_type": "invariant",
           "reward_left": .5 + invariant_delta, "reward_right": .5, "group_baseline": 0., "group_scale": 1.,
           "actual_advantage_left": .5 + invariant_delta, "actual_advantage_right": .5,
           "discovery_frozen_context_id": "low"},
          {"stable_example_id": example, "pair_id": "need", "path_id": "p0", "pair_type": "necessary",
           "reward_left": .5 + necessary_delta, "reward_right": .5, "group_baseline": 0., "group_scale": 1.,
           "actual_advantage_left": .5 + necessary_delta, "actual_advantage_right": .5,
           "frozen_expected_direction": 1, "discovery_frozen_context_id": "low"}])
    manifest = {"schema_version": "bqcc-v1", "controlled_pair_ledger_generated": True,
      "shapeA_structural_gate_passed": True, "role": "unique_prioritized_RL_bridge_after_ShapeA",
      "credit_funnel_is_naturally_nested_quotient_chain": False,
      "pair_or_path_increases_independent_n": False,
      "blocked_pair_centering_authorized_for_invariant_nuisance": False,
      "second_contribution_authorized": False, "optimizer_steps": 0, "new_rollouts": False,
      "training_authorized": False, "tie_epsilon": .01, "split_mass_threshold": .1,
      "merge_mass_threshold": .1, "identity_tolerance": 1e-12, "min_examples_per_relation": 2,
      "bootstrap_iterations": 200, "bootstrap_seed": 20260819,
      "context_sign_flip_discovery_frozen": True,
      "controlled_pairs": rows}
    manifest["canonical_manifest_sha256"] = canonical_manifest_sha256(manifest)
    return manifest


def self_test() -> None:
    assert audit_bqcc(_manifest(0., .5))["status"] == "FINITE_QUOTIENT_COMPATIBLE"
    assert audit_bqcc(_manifest(.2, .5))["status"] == "NUISANCE_SPECIFICITY_FAILURE"
    assert audit_bqcc(_manifest(0., 0.))["status"] == "REWARD_ALIASING_REGROUPING_CANNOT_REPAIR"
    assert audit_bqcc(_manifest(.2, 0.))["status"] == "PARTITION_CROSSING_GROUPING_ONLY_NO_GO"
    sign = _manifest(0., .1)
    for row in list(sign["controlled_pairs"]):
        if row["pair_id"] == "need":
            clone = dict(row); clone["path_id"] = "p1"; clone["group_baseline"] = 1.
            clone["actual_advantage_left"] = clone["reward_left"] - 1.
            clone["actual_advantage_right"] = clone["reward_right"] - 1.
            clone["discovery_frozen_context_id"] = "high"; sign["controlled_pairs"].append(clone)
    sign["canonical_manifest_sha256"] = canonical_manifest_sha256(sign)
    report = audit_bqcc(sign)
    assert report["context_reference_set_sign_flips"]
    identity = _manifest(0., .5); identity["controlled_pairs"][0]["actual_advantage_left"] += .1
    identity["canonical_manifest_sha256"] = canonical_manifest_sha256(identity)
    assert audit_bqcc(identity)["status"] == "IMPLEMENTATION_IDENTITY_FAILURE"
    bad = _manifest(0., .5); bad["canonical_manifest_sha256"] = "0" * 64
    assert audit_bqcc(bad)["status"] == "BQCC_LEDGER_INVALID"
    duplicate = _manifest(0., .5); duplicate["controlled_pairs"].append(dict(duplicate["controlled_pairs"][0]))
    duplicate["canonical_manifest_sha256"] = canonical_manifest_sha256(duplicate)
    assert audit_bqcc(duplicate)["status"] == "BQCC_LEDGER_INVALID"
    missing_n = _manifest(0., .5)
    missing_n["controlled_pairs"] = [row for row in missing_n["controlled_pairs"] if row["pair_type"] == "invariant"]
    missing_n["canonical_manifest_sha256"] = canonical_manifest_sha256(missing_n)
    assert audit_bqcc(missing_n)["status"] == "BQCC_LEDGER_INVALID"
    too_few = _manifest(0., .5)
    too_few["controlled_pairs"] = [row for row in too_few["controlled_pairs"] if row["stable_example_id"] == "e0"]
    too_few["canonical_manifest_sha256"] = canonical_manifest_sha256(too_few)
    assert audit_bqcc(too_few)["status"] == "BQCC_LEDGER_INVALID"
    direction = _manifest(0., .5); direction["controlled_pairs"][1]["frozen_expected_direction"] = 0
    direction["canonical_manifest_sha256"] = canonical_manifest_sha256(direction)
    assert audit_bqcc(direction)["status"] == "BQCC_LEDGER_INVALID"
    scale = _manifest(0., .5); scale["controlled_pairs"][0]["group_scale"] = 0
    scale["canonical_manifest_sha256"] = canonical_manifest_sha256(scale)
    assert audit_bqcc(scale)["status"] == "BQCC_LEDGER_INVALID"
    print("behavioral_quotient_credit_compatibility_self_test=ok")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.manifest: parser.error("--manifest required")
    print(json.dumps(audit_bqcc(json.loads(args.manifest.read_text())), indent=2, sort_keys=True))


if __name__ == "__main__": main()
