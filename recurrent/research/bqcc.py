"""Behavioral-Quotient Credit Compatibility CPU audit."""
from __future__ import annotations

import math
import hashlib
import json
import random
from collections import defaultdict
from typing import Any


def _invalid(reason: str) -> dict[str, Any]:
    return {"status": "BQCC_LEDGER_INVALID", "reason": reason,
            "training_authorized": False, "second_contribution_authorized": False}


def canonical_manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "canonical_manifest_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cluster_interval(values: dict[str, float], *, iterations: int, seed: int) -> list[float]:
    examples = sorted(values); rng = random.Random(seed); estimates = []
    for _ in range(iterations):
        sampled = [rng.choice(examples) for _ in examples]
        estimates.append(sum(values[example] for example in sampled) / len(sampled))
    estimates.sort()
    return [estimates[int(.025 * (iterations - 1))], estimates[int(.975 * (iterations - 1))]]


def audit_bqcc(manifest: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version": "bqcc-v1", "controlled_pair_ledger_generated": True,
                "shapeA_structural_gate_passed": True,
                "role": "unique_prioritized_RL_bridge_after_ShapeA",
                "credit_funnel_is_naturally_nested_quotient_chain": False,
                "pair_or_path_increases_independent_n": False,
                "blocked_pair_centering_authorized_for_invariant_nuisance": False,
                "second_contribution_authorized": False, "optimizer_steps": 0,
                "new_rollouts": False, "training_authorized": False}
    wrong = {key: (manifest.get(key), expected) for key, expected in required.items()
             if manifest.get(key) != expected}
    if wrong: return _invalid(f"contract_failed:{wrong}")
    if manifest.get("canonical_manifest_sha256") != canonical_manifest_sha256(manifest):
        return _invalid("canonical_manifest_SHA256_mismatch")
    epsilon = manifest.get("tie_epsilon"); split_cut = manifest.get("split_mass_threshold")
    merge_cut = manifest.get("merge_mass_threshold"); identity_tolerance = manifest.get("identity_tolerance")
    min_examples = manifest.get("min_examples_per_relation")
    bootstrap_iterations = manifest.get("bootstrap_iterations"); bootstrap_seed = manifest.get("bootstrap_seed")
    thresholds_valid = all(
        isinstance(item, (int, float)) and math.isfinite(float(item)) and item >= 0
        for item in (epsilon, split_cut, merge_cut, identity_tolerance)
    )
    if (not thresholds_valid or not isinstance(min_examples, int) or min_examples < 1 or
            not isinstance(bootstrap_iterations, int) or bootstrap_iterations < 200 or
            not isinstance(bootstrap_seed, int)):
        return _invalid("thresholds_invalid")
    rows = manifest.get("controlled_pairs")
    if not isinstance(rows, list) or not rows: return _invalid("controlled_pair_ledger_missing")
    seen = set(); invariant = defaultdict(list); necessary_tie = defaultdict(list)
    necessary_wrong = defaultdict(list); sign_contexts = defaultdict(list); identity_errors = []
    for index, row in enumerate(rows):
        key = (str(row.get("stable_example_id")), str(row.get("pair_id")), str(row.get("path_id")))
        if not all(key) or key in seen: return _invalid(f"pair_key_invalid_or_duplicate:{index}")
        seen.add(key)
        pair_type = row.get("pair_type"); left = row.get("reward_left"); right = row.get("reward_right")
        baseline = row.get("group_baseline"); scale = row.get("group_scale")
        actual_left = row.get("actual_advantage_left"); actual_right = row.get("actual_advantage_right")
        if (pair_type not in {"invariant", "necessary"} or not all(
                isinstance(item, (int, float)) and math.isfinite(float(item))
                for item in (left, right, baseline, scale, actual_left, actual_right)) or scale <= 0):
            return _invalid(f"pair_values_invalid:{index}")
        delta_r = float(left) - float(right)
        advantage_left = (float(left) - float(baseline)) / float(scale)
        advantage_right = (float(right) - float(baseline)) / float(scale)
        actual_delta_a = float(actual_left) - float(actual_right)
        identity_errors.append(abs(actual_delta_a - delta_r / float(scale)))
        example = key[0]
        if pair_type == "invariant":
            invariant[example].append(abs(delta_r))
        else:
            direction = row.get("frozen_expected_direction")
            if direction not in {-1, 1}: return _invalid(f"necessary_direction_missing:{index}")
            directed = float(direction) * delta_r
            necessary_tie[example].append(1.0 if abs(delta_r) <= float(epsilon) else 0.0)
            necessary_wrong[example].append(max(0.0, -directed))
        context = row.get("discovery_frozen_context_id")
        if context is not None:
            sign_contexts[(example, str(row["pair_id"]))].append(
                (str(context), math.copysign(1, advantage_left) if advantage_left else 0,
                 math.copysign(1, advantage_right) if advantage_right else 0))
    if not invariant or not necessary_tie:
        return _invalid("both_invariant_and_necessary_pairs_required")
    if len(invariant) < min_examples or len(necessary_tie) < min_examples:
        return _invalid(f"minimum_examples_per_relation_not_met:I={len(invariant)}:N={len(necessary_tie)}")
    inv_example = {example: sum(items) / len(items) for example, items in invariant.items()}
    tie_example = {example: sum(items) / len(items) for example, items in necessary_tie.items()}
    wrong_example = {example: sum(items) / len(items) for example, items in necessary_wrong.items()}
    split_mass = sum(inv_example.values()) / len(inv_example)
    tie_mass = sum(tie_example.values()) / len(tie_example)
    wrong_mass = sum(wrong_example.values()) / len(wrong_example)
    merge_mass = tie_mass + wrong_mass
    high_split = split_mass > float(split_cut); high_merge = merge_mass > float(merge_cut)
    routes = {(False, False): "FINITE_QUOTIENT_COMPATIBLE",
              (True, False): "NUISANCE_SPECIFICITY_FAILURE",
              (False, True): "REWARD_ALIASING_REGROUPING_CANNOT_REPAIR",
              (True, True): "PARTITION_CROSSING_GROUPING_ONLY_NO_GO"}
    sign_flips = []
    if manifest.get("context_sign_flip_discovery_frozen") is not True:
        return _invalid("context_sign_flip_discovery_not_frozen")
    for key, items in sign_contexts.items():
        left_signs = {item[1] for item in items}; right_signs = {item[2] for item in items}
        if len(left_signs - {0}) > 1 or len(right_signs - {0}) > 1:
            sign_flips.append({"stable_example_id": key[0], "pair_id": key[1],
                               "contexts": sorted({item[0] for item in items})})
    max_identity_error = max(identity_errors)
    status = ("IMPLEMENTATION_IDENTITY_FAILURE" if max_identity_error > float(identity_tolerance)
              else routes[(high_split, high_merge)])
    return {"status": status,
            "InvariantSplitMass": split_mass, "NecessaryTieMass": tie_mass,
            "NecessaryWrongMass": wrong_mass, "NecessaryTieWrongMass": merge_mass,
            "InvariantSplitMass_cluster_interval": _cluster_interval(
                inv_example, iterations=bootstrap_iterations, seed=bootstrap_seed),
            "NecessaryTieMass_cluster_interval": _cluster_interval(
                tie_example, iterations=bootstrap_iterations, seed=bootstrap_seed + 1),
            "NecessaryWrongMass_cluster_interval": _cluster_interval(
                wrong_example, iterations=bootstrap_iterations, seed=bootstrap_seed + 2),
            "high_invariant_split": high_split, "high_necessary_merge": high_merge,
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
