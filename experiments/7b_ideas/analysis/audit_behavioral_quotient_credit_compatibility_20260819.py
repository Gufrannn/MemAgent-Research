#!/usr/bin/env python3
"""CPU-only BQCC three-input controlled-pair audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.bqcc import (  # noqa: E402
    audit_bqcc, canonical_ledger_sha256, canonical_manifest_sha256,
)


def _inputs(invariant_deltas=(0., 0.), necessary_deltas=(.5, .5),
            *, extra_unadmitted_examples: int = 0) -> tuple[dict, list[dict], list[dict]]:
    pairs = []; target = []
    for relation, deltas in (("invariant", invariant_deltas), ("necessary", necessary_deltas)):
        for index, delta in enumerate(deltas):
            example = f"e{index}"; pair_id = "inv" if relation == "invariant" else "need"
            row = {"stable_example_id": example, "pair_id": pair_id, "path_id": "p0",
                   "pair_type": relation, "analysis_status": "analyzed",
                   "reward_left": .5 + delta, "reward_right": .5,
                   "group_baseline": 0., "group_scale": 1.,
                   "actual_advantage_left": .5 + delta, "actual_advantage_right": .5,
                   "discovery_frozen_context_id": "low"}
            if relation == "necessary": row["frozen_expected_direction"] = 1
            pairs.append(row)
            target.append({"stable_example_id": example, "pair_id": pair_id, "path_id": "p0",
                           "pair_type": relation, "target_relation": True, "admitted": True,
                           "admission_reason": "prefrozen_relation_admitted", "infrastructure_failure": False})
        for extra in range(extra_unadmitted_examples):
            target.append({"stable_example_id": f"x{extra}", "pair_id": f"{relation}_extra",
                           "path_id": "p0", "pair_type": relation, "target_relation": True,
                           "admitted": False, "admission_reason": "prefrozen_relation_not_admitted",
                           "infrastructure_failure": False})
    manifest = {"schema_version": "bqcc-v2-coverage-interval",
      "controlled_pair_ledger_generated": True, "target_ledger_complete": True,
      "shapeA_structural_gate_passed": True, "role": "unique_prioritized_RL_bridge_after_ShapeA",
      "credit_funnel_is_naturally_nested_quotient_chain": False,
      "coverage_run_gate_equals_population_identification_gate": False,
      "pair_or_path_increases_independent_n": False,
      "blocked_pair_centering_authorized_for_invariant_nuisance": False,
      "second_contribution_authorized": False, "optimizer_steps": 0, "new_rollouts": False,
      "training_authorized": False, "tie_epsilon": .01,
      "invariant_max_defect": .1, "invariant_high_threshold": .1,
      "necessary_merge_max_defect": .1, "necessary_merge_high_threshold": .1,
      "identity_tolerance": 1e-12, "min_examples_per_relation": 2,
      "minimum_pair_coverage_to_run": .5, "minimum_example_coverage_to_run": .5,
      "bootstrap_iterations": 500, "bootstrap_seed": 20260819,
      "context_sign_flip_discovery_frozen": True,
      "missingness_partial_id_model": {"enabled": False},
      "pairs_sha256": canonical_ledger_sha256(pairs),
      "target_ledger_sha256": canonical_ledger_sha256(target)}
    manifest["canonical_manifest_sha256"] = canonical_manifest_sha256(manifest)
    return manifest, pairs, target


def _rehash(manifest: dict, pairs: list[dict], target: list[dict]) -> None:
    manifest["pairs_sha256"] = canonical_ledger_sha256(pairs)
    manifest["target_ledger_sha256"] = canonical_ledger_sha256(target)
    manifest["canonical_manifest_sha256"] = canonical_manifest_sha256(manifest)


def self_test() -> None:
    manifest, pairs, target = _inputs()
    result = audit_bqcc(manifest, pairs, target)
    assert result["status"] == "FINITE_QUOTIENT_COMPATIBLE"
    assert result["claim_scope"] == "full_target_population"
    assert result["InvariantSplit_axis_state"] == result["NecessaryMerge_axis_state"] == "LOW_CERTIFIED"
    for inv, need, expected in (( (.2, .2), (.5, .5), "NUISANCE_SPECIFICITY_FAILURE"),
                                ( (0., 0.), (0., 0.), "REWARD_ALIASING_REGROUPING_CANNOT_REPAIR"),
                                ( (.2, .2), (0., 0.), "PARTITION_CROSSING_GROUPING_ONLY_NO_GO")):
        manifest, pairs, target = _inputs(inv, need)
        assert audit_bqcc(manifest, pairs, target)["status"] == expected
    manifest, pairs, target = _inputs((0., .2), (.5, .5))
    result = audit_bqcc(manifest, pairs, target)
    assert result["status"] == "BQCC_INCONCLUSIVE_THRESHOLD_UNCERTAINTY"
    assert not result["specific_defect_label_authorized"] and not result["point_verdict_authorized"]
    manifest, pairs, target = _inputs((0., 0.), (.5, 0.))
    assert audit_bqcc(manifest, pairs, target)["status"] == "BQCC_INCONCLUSIVE_THRESHOLD_UNCERTAINTY"
    manifest, pairs, target = _inputs(extra_unadmitted_examples=2)
    result = audit_bqcc(manifest, pairs, target)
    assert result["status"] == "FINITE_QUOTIENT_COMPATIBLE"
    assert result["claim_scope"] == "admitted_relation_stratum_only"
    manifest["missingness_partial_id_model"] = {"enabled": True, "frozen_before_outcome": True,
      "validated": True, "model_hash": "a" * 64}
    manifest["canonical_manifest_sha256"] = canonical_manifest_sha256(manifest)
    assert audit_bqcc(manifest, pairs, target)["claim_scope"] == \
        "full_target_population_via_validated_missingness_partial_ID"
    manifest["minimum_pair_coverage_to_run"] = .75; manifest["minimum_example_coverage_to_run"] = .75
    manifest["canonical_manifest_sha256"] = canonical_manifest_sha256(manifest)
    assert audit_bqcc(manifest, pairs, target)["status"] == "BQCC_COVERAGE_RUN_GATE_FAIL"
    manifest, pairs, target = _inputs(); target[0]["infrastructure_failure"] = True; _rehash(manifest, pairs, target)
    assert audit_bqcc(manifest, pairs, target)["status"] == "BQCC_INFRASTRUCTURE_FAILURE_POINT_VERDICT_BLOCKED"
    manifest, pairs, target = _inputs(); pairs.pop(); _rehash(manifest, pairs, target)
    assert audit_bqcc(manifest, pairs, target)["status"] == "BQCC_LEDGER_INVALID"
    manifest, pairs, target = _inputs(); pairs[0]["actual_advantage_left"] += .1; _rehash(manifest, pairs, target)
    assert audit_bqcc(manifest, pairs, target)["status"] == "IMPLEMENTATION_IDENTITY_FAILURE"
    manifest, pairs, target = _inputs(); manifest["canonical_manifest_sha256"] = "0" * 64
    assert audit_bqcc(manifest, pairs, target)["status"] == "BQCC_LEDGER_INVALID"
    manifest, pairs, target = _inputs(); pairs.append(dict(pairs[0])); _rehash(manifest, pairs, target)
    assert audit_bqcc(manifest, pairs, target)["status"] == "BQCC_LEDGER_INVALID"
    manifest, pairs, target = _inputs(); target = [row for row in target if row["pair_type"] == "invariant"]
    _rehash(manifest, pairs, target)
    assert audit_bqcc(manifest, pairs, target)["status"] == "BQCC_LEDGER_INVALID"
    manifest, pairs, target = _inputs(); manifest["min_examples_per_relation"] = 3
    manifest["canonical_manifest_sha256"] = canonical_manifest_sha256(manifest)
    assert audit_bqcc(manifest, pairs, target)["status"] == "BQCC_LEDGER_INVALID"
    manifest, pairs, target = _inputs(); pairs[2]["frozen_expected_direction"] = 0; _rehash(manifest, pairs, target)
    assert audit_bqcc(manifest, pairs, target)["status"] == "BQCC_LEDGER_INVALID"
    manifest, pairs, target = _inputs(); pairs[0]["group_scale"] = 0; _rehash(manifest, pairs, target)
    assert audit_bqcc(manifest, pairs, target)["status"] == "BQCC_LEDGER_INVALID"
    print("behavioral_quotient_credit_compatibility_v2_self_test=ok")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path); parser.add_argument("--pairs", type=Path)
    parser.add_argument("--target-ledger", type=Path); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.manifest or not args.pairs or not args.target_ledger:
        parser.error("--manifest, --pairs, and --target-ledger are required")
    print(json.dumps(audit_bqcc(json.loads(args.manifest.read_text()), _read_jsonl(args.pairs),
                                _read_jsonl(args.target_ledger)), indent=2, sort_keys=True))


if __name__ == "__main__": main()
