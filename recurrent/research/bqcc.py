"""Behavioral-Quotient Credit Compatibility CPU audit."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")
RELATIONS = {"invariant", "necessary"}


def _invalid(reason: str, status: str = "BQCC_LEDGER_INVALID") -> dict[str, Any]:
    return {"status": status, "reason": reason, "point_verdict_authorized": False,
            "training_authorized": False, "second_contribution_authorized": False}


def canonical_manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "canonical_manifest_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_ledger_sha256(rows: list[dict[str, Any]]) -> str:
    lines = sorted(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                   for row in rows)
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def _cluster_interval(values: dict[str, float], *, iterations: int, seed: int) -> list[float]:
    examples = sorted(values); rng = random.Random(seed); estimates = []
    for _ in range(iterations):
        sampled = [rng.choice(examples) for _ in examples]
        estimates.append(sum(values[example] for example in sampled) / len(sampled))
    estimates.sort()
    return [estimates[int(.025 * (iterations - 1))], estimates[int(.975 * (iterations - 1))]]


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
    raw = (row.get("stable_example_id"), row.get("pair_id"), row.get("path_id"), row.get("pair_type"))
    if (any(item is None or str(item) == "" for item in raw) or
            not isinstance(raw[3], str) or raw[3] not in RELATIONS):
        return None
    return tuple(map(str, raw))  # type: ignore[return-value]


def _axis_state(interval: list[float], low_max_defect: float, high_threshold: float) -> str:
    if interval[1] <= low_max_defect:
        return "LOW_CERTIFIED"
    if interval[0] > high_threshold:
        return "HIGH_ESTABLISHED"
    return "THRESHOLD_UNCERTAIN"


def audit_bqcc(manifest: dict[str, Any], pairs: list[dict[str, Any]] | None = None,
               target_ledger: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    required = {"schema_version": "bqcc-v2-coverage-interval",
                "controlled_pair_ledger_generated": True, "target_ledger_complete": True,
                "shapeA_structural_gate_passed": True,
                "role": "unique_prioritized_RL_bridge_after_ShapeA",
                "credit_funnel_is_naturally_nested_quotient_chain": False,
                "coverage_run_gate_equals_population_identification_gate": False,
                "pair_or_path_increases_independent_n": False,
                "blocked_pair_centering_authorized_for_invariant_nuisance": False,
                "second_contribution_authorized": False, "optimizer_steps": 0,
                "new_rollouts": False, "training_authorized": False}
    wrong = {key: (manifest.get(key), expected) for key, expected in required.items()
             if manifest.get(key) != expected}
    if wrong:
        return _invalid(f"contract_failed:{wrong}")
    if not isinstance(pairs, list) or not isinstance(target_ledger, list):
        return _invalid("three_inputs_required:PAIRS_jsonl+TARGET_LEDGER_jsonl+MANIFEST_json")
    if manifest.get("canonical_manifest_sha256") != canonical_manifest_sha256(manifest):
        return _invalid("canonical_manifest_SHA256_mismatch")
    if manifest.get("pairs_sha256") != canonical_ledger_sha256(pairs):
        return _invalid("PAIRS_canonical_SHA256_mismatch")
    if manifest.get("target_ledger_sha256") != canonical_ledger_sha256(target_ledger):
        return _invalid("TARGET_LEDGER_canonical_SHA256_mismatch")
    epsilon = manifest.get("tie_epsilon")
    split_low = manifest.get("invariant_max_defect"); split_high = manifest.get("invariant_high_threshold")
    merge_low = manifest.get("necessary_merge_max_defect")
    merge_high = manifest.get("necessary_merge_high_threshold")
    identity_tolerance = manifest.get("identity_tolerance")
    pair_run_cut = manifest.get("minimum_pair_coverage_to_run")
    example_run_cut = manifest.get("minimum_example_coverage_to_run")
    min_examples = manifest.get("min_examples_per_relation")
    bootstrap_iterations = manifest.get("bootstrap_iterations"); bootstrap_seed = manifest.get("bootstrap_seed")
    thresholds_valid = all(
        isinstance(item, (int, float)) and math.isfinite(float(item)) and item >= 0
        for item in (epsilon, split_low, split_high, merge_low, merge_high, identity_tolerance)
    ) and all(isinstance(item, (int, float)) and 0 <= float(item) <= 1
              for item in (pair_run_cut, example_run_cut))
    if (not thresholds_valid or not isinstance(min_examples, int) or min_examples < 1 or
            not isinstance(bootstrap_iterations, int) or bootstrap_iterations < 200 or
            not isinstance(bootstrap_seed, int) or float(split_low) > float(split_high) or
            float(merge_low) > float(merge_high)):
        return _invalid("thresholds_invalid")

    target_keys: set[tuple[str, str, str, str]] = set(); admitted_keys = set()
    target_pairs = defaultdict(int); admitted_pairs = defaultdict(int)
    target_examples = defaultdict(set); admitted_examples = defaultdict(set)
    infrastructure = []
    for index, row in enumerate(target_ledger):
        key = _pair_key(row)
        if (key is None or key in target_keys or row.get("target_relation") is not True or
                not isinstance(row.get("admitted"), bool) or
                not isinstance(row.get("infrastructure_failure"), bool)):
            return _invalid(f"target_row_invalid_or_duplicate:{index}")
        if not isinstance(row.get("admission_reason"), str) or not row["admission_reason"]:
            return _invalid(f"target_admission_reason_missing:{index}")
        target_keys.add(key); relation = key[3]; target_pairs[relation] += 1; target_examples[relation].add(key[0])
        if row["infrastructure_failure"]:
            infrastructure.append(key)
        if row["admitted"]:
            admitted_keys.add(key); admitted_pairs[relation] += 1; admitted_examples[relation].add(key[0])
    if not target_keys or not target_examples["invariant"] or not target_examples["necessary"]:
        return _invalid("complete_target_ledger_requires_I_and_N_relations")
    if infrastructure:
        return _invalid("infrastructure_failure_blocks_point_verdict",
                        status="BQCC_INFRASTRUCTURE_FAILURE_POINT_VERDICT_BLOCKED")

    analyzed_keys = set(); invariant = defaultdict(list); necessary_tie = defaultdict(list)
    necessary_wrong = defaultdict(list); sign_contexts = defaultdict(list); identity_errors = []
    for index, row in enumerate(pairs):
        key = _pair_key(row)
        if key is None or key in analyzed_keys or row.get("analysis_status") != "analyzed":
            return _invalid(f"analyzed_pair_invalid_or_duplicate:{index}")
        analyzed_keys.add(key)
        pair_type = key[3]; left = row.get("reward_left"); right = row.get("reward_right")
        baseline = row.get("group_baseline"); scale = row.get("group_scale")
        actual_left = row.get("actual_advantage_left"); actual_right = row.get("actual_advantage_right")
        if (not all(isinstance(item, (int, float)) and math.isfinite(float(item))
                    for item in (left, right, baseline, scale, actual_left, actual_right)) or scale <= 0):
            return _invalid(f"pair_values_invalid:{index}")
        delta_r = float(left) - float(right)
        advantage_left = (float(left) - float(baseline)) / float(scale)
        advantage_right = (float(right) - float(baseline)) / float(scale)
        identity_errors.append(abs((float(actual_left) - float(actual_right)) - delta_r / float(scale)))
        example = key[0]
        if pair_type == "invariant":
            invariant[example].append(abs(delta_r))
        else:
            direction = row.get("frozen_expected_direction")
            if direction not in {-1, 1}:
                return _invalid(f"necessary_direction_missing:{index}")
            directed = float(direction) * delta_r
            necessary_tie[example].append(1.0 if abs(delta_r) <= float(epsilon) else 0.0)
            necessary_wrong[example].append(max(0.0, -directed))
        context = row.get("discovery_frozen_context_id")
        if context is not None:
            sign_contexts[(example, key[1])].append(
                (str(context), math.copysign(1, advantage_left) if advantage_left else 0,
                 math.copysign(1, advantage_right) if advantage_right else 0))
    if admitted_keys != analyzed_keys:
        return _invalid(f"target_to_admitted_to_analyzed_closure_failure:admitted={len(admitted_keys)}:analyzed={len(analyzed_keys)}")
    if not invariant or not necessary_tie:
        return _invalid("both_admitted_invariant_and_necessary_pairs_required")
    if len(invariant) < min_examples or len(necessary_tie) < min_examples:
        return _invalid(f"minimum_examples_per_relation_not_met:I={len(invariant)}:N={len(necessary_tie)}")

    coverage = {}
    for relation in sorted(RELATIONS):
        coverage[relation] = {
            "pair_coverage": admitted_pairs[relation] / target_pairs[relation],
            "example_coverage": len(admitted_examples[relation]) / len(target_examples[relation]),
            "target_pairs": target_pairs[relation], "admitted_analyzed_pairs": admitted_pairs[relation],
            "target_examples": len(target_examples[relation]),
            "admitted_analyzed_examples": len(admitted_examples[relation])}
    if any(row[metric] < float(cut) for row in coverage.values()
           for metric, cut in (("pair_coverage", pair_run_cut), ("example_coverage", example_run_cut))):
        return {**_invalid("coverage_below_prefrozen_run_threshold", status="BQCC_COVERAGE_RUN_GATE_FAIL"),
                "coverage": coverage, "claim_scope": "NO_POINT_VERDICT"}

    full_coverage = all(row[metric] == 1.0 for row in coverage.values()
                        for metric in ("pair_coverage", "example_coverage"))
    partial_id = manifest.get("missingness_partial_id_model")
    if not isinstance(partial_id, dict) or not isinstance(partial_id.get("enabled"), bool):
        return _invalid("missingness_partial_ID_declaration_missing")
    partial_id_valid = False
    if partial_id["enabled"]:
        partial_id_valid = (partial_id.get("frozen_before_outcome") is True and
                            partial_id.get("validated") is True and
                            SHA256.fullmatch(str(partial_id.get("model_hash", ""))) is not None)
        if not partial_id_valid:
            return _invalid("missingness_partial_ID_model_not_prefrozen_and_validated")
    claim_scope = ("full_target_population" if full_coverage else
                   "full_target_population_via_validated_missingness_partial_ID" if partial_id_valid else
                   "admitted_relation_stratum_only")

    inv_example = {example: sum(items) / len(items) for example, items in invariant.items()}
    tie_example = {example: sum(items) / len(items) for example, items in necessary_tie.items()}
    wrong_example = {example: sum(items) / len(items) for example, items in necessary_wrong.items()}
    merge_example = {example: tie_example[example] + wrong_example[example] for example in tie_example}
    split_mass = sum(inv_example.values()) / len(inv_example)
    tie_mass = sum(tie_example.values()) / len(tie_example)
    wrong_mass = sum(wrong_example.values()) / len(wrong_example)
    merge_mass = sum(merge_example.values()) / len(merge_example)
    split_interval = _cluster_interval(inv_example, iterations=bootstrap_iterations, seed=bootstrap_seed)
    merge_interval = _cluster_interval(merge_example, iterations=bootstrap_iterations, seed=bootstrap_seed + 1)
    split_state = _axis_state(split_interval, float(split_low), float(split_high))
    merge_state = _axis_state(merge_interval, float(merge_low), float(merge_high))
    routes = {("LOW_CERTIFIED", "LOW_CERTIFIED"): "FINITE_QUOTIENT_COMPATIBLE",
              ("HIGH_ESTABLISHED", "LOW_CERTIFIED"): "NUISANCE_SPECIFICITY_FAILURE",
              ("LOW_CERTIFIED", "HIGH_ESTABLISHED"): "REWARD_ALIASING_REGROUPING_CANNOT_REPAIR",
              ("HIGH_ESTABLISHED", "HIGH_ESTABLISHED"): "PARTITION_CROSSING_GROUPING_ONLY_NO_GO"}
    sign_flips = []
    if manifest.get("context_sign_flip_discovery_frozen") is not True:
        return _invalid("context_sign_flip_discovery_not_frozen")
    for key, items in sign_contexts.items():
        left_signs = {item[1] for item in items}; right_signs = {item[2] for item in items}
        if len(left_signs - {0}) > 1 or len(right_signs - {0}) > 1:
            sign_flips.append({"stable_example_id": key[0], "pair_id": key[1],
                               "contexts": sorted({item[0] for item in items})})
    max_identity_error = max(identity_errors)
    if max_identity_error > float(identity_tolerance):
        status = "IMPLEMENTATION_IDENTITY_FAILURE"
    elif "THRESHOLD_UNCERTAIN" in {split_state, merge_state}:
        status = "BQCC_INCONCLUSIVE_THRESHOLD_UNCERTAINTY"
    else:
        status = routes[(split_state, merge_state)]
    return {"status": status,
            "point_verdict_authorized": status != "BQCC_INCONCLUSIVE_THRESHOLD_UNCERTAINTY",
            "claim_scope": claim_scope, "coverage": coverage,
            "coverage_run_gate_equals_population_identification_gate": False,
            "InvariantSplitMass": split_mass, "InvariantSplitMass_cluster_interval": split_interval,
            "InvariantSplit_axis_state": split_state,
            "InvariantSplit_frozen_thresholds": {"max_defect": split_low, "high_threshold": split_high},
            "NecessaryTieMass": tie_mass, "NecessaryWrongMass": wrong_mass,
            "NecessaryTieWrongMass": merge_mass,
            "NecessaryTieWrongMass_cluster_interval": merge_interval,
            "NecessaryMerge_axis_state": merge_state,
            "NecessaryMerge_frozen_thresholds": {"max_defect": merge_low, "high_threshold": merge_high},
            "example_cluster_bootstrap_confidence": 0.95,
            "specific_defect_label_authorized": status in {
                "NUISANCE_SPECIFICITY_FAILURE", "REWARD_ALIASING_REGROUPING_CANNOT_REPAIR",
                "PARTITION_CROSSING_GROUPING_ONLY_NO_GO"},
            "context_reference_set_sign_flips": sign_flips,
            "absolute_advantage_sign_is_reference_set_dependent": True,
            "shared_baseline_or_std_repairs_reward_tie": False,
            "shared_baseline_or_std_removes_invariant_split": False,
            "max_pair_advantage_identity_error": max_identity_error,
            "identity_error_aggregated_within_example_before_max": False,
            "pair_to_example_then_example_cluster_aggregation": True,
            "gradient_delivery_inferred_from_BQCC": False,
            "irreducible_delivery_sequence": ["G_credit", "G_task", "G_total", "Delta_theta_actual"],
            "BQCC_pass_and_G_task_silent_interpretation": "trust_region_delivery_bottleneck_not_reward_failure",
            "large_G_total_or_gradient_norm_repairs_BQCC_failure": False,
            "CSFGW_W4_required_only_for_actual_update_credit_or_repair_unlock_claim": True,
            "cluster_unit": "stable_example_id", "pair_or_path_increases_independent_n": False,
            "training_authorized": False, "second_contribution_authorized": False}
