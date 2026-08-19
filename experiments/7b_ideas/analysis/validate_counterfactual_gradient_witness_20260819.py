#!/usr/bin/env python3
"""Fail-closed W4 v8 endpoint, policy-derivative, and geometry router."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.counterfactual_gradient_witness import (  # noqa: E402
    FORBIDDEN_EVIDENCE_BASES, group_estimators, objective_mismatch_estimators, vector_hash)

SHA256 = re.compile(r"^[0-9a-f]{64}$")
ATOL = 1e-10
PARITY_DIMENSIONS = {
    "horizon", "future_policy", "suffix_contract", "reader", "answer_cell",
    "answer_normalization", "reward_components", "reward_weights", "reward_scale",
    "invalid_rule", "truncation_rule", "missing_rule", "row_weights",
    "candidate_joinability", "score_mask_joinability",
}
COMMON_REQUIRED = {
    "schema_version": "counterfactual-gradient-witness-v8",
    "on_policy_same_checkpoint_candidates": True,
    "candidate_groups_prefrozen": True,
    "candidate_groups_independent": True,
    "iid_candidates_within_group": True,
    "reader_seed_derivation": "pre_candidate_state_coupling_manifest",
    "rng_advance_candidate_length_dependent": False,
    "writer_token_mask_exact": True,
    "writer_mask_includes_eos_or_stop": True,
    "validity_frozen_before_return": True,
    "truncation_frozen_before_return": True,
    "row_selection_frozen_before_return": True,
    "return_or_outcome_conditioned_selection": False,
    "actual_group_reconstructable": True,
    "actual_bonus_reconstructable": True,
    "actual_logprob_reconstructable": True,
    "loss_reconstruction_exact": True,
    "single_batch_directional_evidence_authorized": False,
    "algorithm_novelty_authorized": False,
    "raw_euclidean_cosine_role": "fixed_coordinate_secondary_diagnostic_only",
    "raw_euclidean_cosine_parameterization_invariant": False,
    "geometry_selected_after_endpoint": False,
    "optimizer_steps": 0,
    "new_rollouts": False,
}
CV_REQUIRED = {
    "exact_noop_v2_qualified": True,
    "exact_noop_role": "control_variate_not_new_action_value_target",
    "noop_baseline_candidate_independent": True,
    "noop_rng_independent": True,
    "noop_cache_independent": True,
    "noop_coupling_frozen_before_candidate": True,
    "noop_coupling_exogenous_given_state": True,
    "including_self_all_mean_estimator": True,
    "including_self_expected_scale_formula": "(n-1)/n",
    "including_self_debias_formula": "n/(n-1)",
    "scientific_null": "paired_group_mean_G_credit_debiased_minus_G_CF_equals_zero",
}
OM_REQUIRED = {
    "red_calibration_pass": True,
    "endpoint_label_shuffle_null_registered": True,
    "equal_scale_component_ablation_registered": True,
    "component_ablation_scales_match": True,
    "objective_mismatch_label": "surrogate_objective_gradient_mismatch",
    "same_reward_credit_loss_claim_authorized": False,
}
PROHIBITED_INFERENCES = sorted(FORBIDDEN_EVIDENCE_BASES | {
    "lost_credit_from_single_batch", "recovered_credit_from_single_batch",
    "same_reward_credit_estimator_loss_under_distinct_endpoints",
})


def _no_go(reason: str, status: str = "W4_NO_GO") -> None:
    raise ValueError(f"{status}: {reason}; highest_level=W3")


def _wrong(value, expected):
    return {key: (value.get(key), target) for key, target in expected.items() if value.get(key) != target}


def _close(left, right, atol=ATOL):
    return len(left) == len(right) and all(abs(float(a) - float(b)) <= atol for a, b in zip(left, right))


def _subtract(left, right):
    return [float(a) - float(b) for a, b in zip(left, right)]


def _mean(vectors):
    return [sum(float(row[j]) for row in vectors) / len(vectors) for j in range(len(vectors[0]))]


def _variance(vectors):
    means = _mean(vectors)
    return [sum((float(row[j]) - means[j]) ** 2 for row in vectors) / (len(vectors) - 1)
            for j in range(len(vectors[0]))]


def _cluster_interval(vectors):
    means = _mean(vectors)
    variances = _variance(vectors)
    half = [1.96 * math.sqrt(value / len(vectors)) for value in variances]
    return {"mean": means, "lower_95": [m - h for m, h in zip(means, half)],
            "upper_95": [m + h for m, h in zip(means, half)],
            "cluster_unit": "candidate_group"}


def _mse(vectors, references):
    return [sum((float(row[j]) - float(ref[j])) ** 2 for row, ref in zip(vectors, references)) / len(vectors)
            for j in range(len(vectors[0]))]


def _endpoint_mode(value):
    rows = value.get("endpoint_parity_ledger")
    if not isinstance(rows, list):
        _no_go("endpoint parity ledger missing", "ENDPOINT_TARGET_AMBIGUOUS")
    by_dimension = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("dimension"), str):
            _no_go("malformed endpoint parity row", "ENDPOINT_TARGET_AMBIGUOUS")
        name = row["dimension"]
        if name in by_dimension:
            _no_go(f"duplicate endpoint parity dimension={name}", "ENDPOINT_TARGET_AMBIGUOUS")
        train_hash = row.get("training_definition_hash")
        science_hash = row.get("scientific_definition_hash")
        same = row.get("same_definition")
        if not SHA256.fullmatch(str(train_hash or "")) or not SHA256.fullmatch(str(science_hash or "")):
            _no_go(f"unknown endpoint definition for {name}", "ENDPOINT_TARGET_AMBIGUOUS")
        if not isinstance(same, bool) or same != (train_hash == science_hash):
            _no_go(f"inconsistent endpoint parity declaration for {name}", "ENDPOINT_TARGET_AMBIGUOUS")
        by_dimension[name] = same
    if set(by_dimension) != PARITY_DIMENSIONS:
        _no_go(f"endpoint parity dimensions must equal {sorted(PARITY_DIMENSIONS)}", "ENDPOINT_TARGET_AMBIGUOUS")
    derived = "CV_same_endpoint" if all(by_dimension.values()) else "OM_distinct_endpoint"
    if value.get("endpoint_mode") != derived:
        _no_go(f"declared endpoint_mode does not match complete parity ledger; derived={derived}",
               "ENDPOINT_TARGET_AMBIGUOUS")
    return derived, sorted(name for name, same in by_dimension.items() if not same)


def _policy_derivative_scope(value):
    mode = value.get("policy_derivative_mode")
    if mode not in {"L_frozen_future_policy", "T_tied_recurrent_policy"}:
        _no_go("unknown policy derivative mode")
    rows = value.get("policy_node_ledger")
    if not isinstance(rows, list) or not rows:
        _no_go("policy-node ledger missing")
    required_fields = {"node_id", "role", "checkpoint_hash", "parameter_identity_hash", "shares_theta",
                       "semantics", "token_span", "includes_eos_or_stop", "mask_hash",
                       "actual_included", "reference_included", "arm_parity", "stopgrad"}
    required_roles = {"current_writer", "future_writer", "future_answer", "future_reader"}
    seen = set()
    roles = set()
    actual_nodes = set()
    reference_nodes = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required_fields.issubset(row):
            _no_go(f"policy node={index} ledger incomplete")
        node_id = row["node_id"]
        if not node_id or node_id in seen:
            _no_go(f"duplicate/missing policy node id={node_id}")
        seen.add(node_id)
        roles.add(row["role"])
        if row["semantics"] not in {"target_policy", "frozen_environment"}:
            _no_go(f"policy node={index} invalid semantics")
        if (not SHA256.fullmatch(str(row["checkpoint_hash"])) or
                not SHA256.fullmatch(str(row["parameter_identity_hash"])) or
                not SHA256.fullmatch(str(row["mask_hash"]))):
            _no_go(f"policy node={index} invalid immutable hash")
        if not isinstance(row["token_span"], list) or len(row["token_span"]) != 2:
            _no_go(f"policy node={index} token span missing")
        if any(not isinstance(row[key], bool) for key in
               ("shares_theta", "includes_eos_or_stop", "actual_included", "reference_included",
                "arm_parity", "stopgrad")):
            _no_go(f"policy node={index} boolean metadata malformed")
        if row["arm_parity"] is not True:
            _no_go(f"policy node={index} lacks arm parity")
        if row["actual_included"]:
            actual_nodes.add(node_id)
        if row["reference_included"]:
            reference_nodes.add(node_id)
    if roles != required_roles:
        _no_go(f"policy-node roles must equal {sorted(required_roles)}")
    if actual_nodes != reference_nodes:
        _no_go("actual/reference policy score-node sets differ")
    current = [row for row in rows if row["role"] == "current_writer"]
    if len(current) != 1:
        _no_go("exactly one current-writer policy node is required")
    current = current[0]
    if (current["shares_theta"] is not True or current["semantics"] != "target_policy" or
            current["stopgrad"] is not False or current["actual_included"] is not True or
            current["reference_included"] is not True or current["includes_eos_or_stop"] is not True):
        _no_go("current candidate must carry the complete target-policy score including EOS/stop")
    future = [row for row in rows if row["role"] != "current_writer"]
    if mode == "L_frozen_future_policy":
        if any(row["shares_theta"] or row["semantics"] != "frozen_environment" or not row["stopgrad"] or
               row["actual_included"] or row["reference_included"] for row in future):
            _no_go("L mode requires every future policy node to be frozen and stop-gradient")
        label = "frozen_future_local_writer_gradient"
        full_recurrent = False
    else:
        theta_nodes = {row["node_id"] for row in rows if row["shares_theta"] and row["semantics"] == "target_policy"}
        if any(row["shares_theta"] and row["stopgrad"] for row in rows):
            _no_go("theta-sharing target-policy node cannot be stop-gradient in T mode")
        full = theta_nodes == actual_nodes == reference_nodes
        declared = value.get("policy_gradient_scope_label")
        if full:
            label = "full_terminal_recurrent_policy_gradient"
            if declared != label:
                _no_go("complete T graph must use full terminal recurrent-policy label")
        else:
            label = "local_recurrent_semi_gradient"
            if declared != label or actual_nodes != {current["node_id"]}:
                _no_go("incomplete T graph is only valid as a current-writer local semi-gradient")
        full_recurrent = full
    if value.get("policy_gradient_scope_label") != label:
        _no_go(f"policy gradient scope label must be {label}")
    return {"policy_derivative_mode": mode, "policy_gradient_scope_label": label,
            "policy_node_ledger_complete": True,
            "actual_reference_policy_node_sets_match": True,
            "full_recurrent_policy_gradient_authorized": full_recurrent}


def _gradient_geometry(value):
    direction = value.get("direction_adjudication_requested")
    mode = value.get("gradient_geometry_mode")
    if not isinstance(direction, bool):
        _no_go("direction_adjudication_requested must be boolean")
    for key in ("actual_parameter_block_hash", "reference_parameter_block_hash"):
        if not SHA256.fullmatch(str(value.get(key, ""))):
            _no_go(f"missing/invalid {key}")
    if value["actual_parameter_block_hash"] != value["reference_parameter_block_hash"]:
        _no_go("actual/reference parameter blocks differ")
    if not direction:
        if mode != "none_no_direction_adjudication":
            _no_go("geometry mode must be none when direction adjudication is not requested")
        return {"gradient_geometry_mode": mode, "geometry_claim": "none",
                "raw_euclidean_cosine_role": "fixed_coordinate_secondary_diagnostic_only",
                "direction_adjudication_authorized": False}
    if mode not in {"Fisher_tested_subspace", "optimizer_delivery"}:
        _no_go("direction adjudication must preregister Fisher or optimizer-delivery geometry")
    euclidean = value.get("fixed_coordinate_euclidean_pairing")
    if not isinstance(euclidean, (int, float)) or not math.isfinite(float(euclidean)):
        _no_go("fixed-coordinate Euclidean pairing missing")
    if mode == "Fisher_tested_subspace":
        config = value.get("fisher_geometry", {})
        required_hashes = ("parameter_block_hash", "projection_hash", "sensitivity_manifest_hash")
        if any(not SHA256.fullmatch(str(config.get(key, ""))) for key in required_hashes):
            _no_go("Fisher geometry immutable hashes missing")
        if config["parameter_block_hash"] != value["actual_parameter_block_hash"]:
            _no_go("Fisher parameter block differs from actual/reference block")
        rank = config.get("effective_rank")
        condition = config.get("condition_number")
        damping = config.get("relative_damping")
        cutoff = config.get("eigen_cutoff")
        pairing = config.get("endpoint_reference_fisher_bilinear")
        if (not isinstance(rank, int) or rank < 1 or
                any(not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) <= 0
                    for x in (condition, damping, cutoff)) or
                not isinstance(pairing, (int, float)) or not math.isfinite(float(pairing))):
            _no_go("Fisher rank/conditioning/damping/cutoff/pairing metadata invalid")
        report = {"gradient_geometry_mode": mode,
                  "geometry_claim": "empirical_Fisher_tested_subspace_geometry",
                  "direction_adjudication_authorized": True,
                  "invariant_pairing": float(pairing),
                  "fisher_effective_rank": rank, "fisher_condition_number": float(condition)}
    else:
        config = value.get("optimizer_delivery", {})
        required_true = ("adam_moments_included", "learning_rate_included", "clip_included",
                         "weight_decay_included", "accumulation_included", "scaling_included",
                         "state_mutation_disabled")
        if any(config.get(key) is not True for key in required_true):
            _no_go("optimizer delivery reconstruction is incomplete or stateful")
        if (not SHA256.fullmatch(str(config.get("parameter_block_hash", ""))) or
                config["parameter_block_hash"] != value["actual_parameter_block_hash"] or
                not SHA256.fullmatch(str(config.get("optimizer_state_hash_before", ""))) or
                config.get("optimizer_state_hash_after") != config.get("optimizer_state_hash_before")):
            _no_go("optimizer delivery block/state hash mismatch")
        endpoint = config.get("endpoint_gradient")
        delta = config.get("delta_theta_actual")
        if (not isinstance(endpoint, list) or not isinstance(delta, list) or not endpoint or
                len(endpoint) != len(delta) or not all(math.isfinite(float(x)) for x in endpoint + delta)):
            _no_go("optimizer delivery endpoint/delta vectors invalid")
        pairing = sum(float(left) * float(right) for left, right in zip(endpoint, delta))
        report = {"gradient_geometry_mode": mode, "geometry_claim": "optimizer_delivery_evidence",
                  "direction_adjudication_authorized": True,
                  "g_endpoint_dot_delta_theta_actual": pairing,
                  "optimizer_state_unchanged": True,
                  "pure_credit_evidence": False}
    invariant = report.get("invariant_pairing", report.get("g_endpoint_dot_delta_theta_actual"))
    if float(euclidean) < 0 <= float(invariant):
        report["geometry_adjudication"] = "COORDINATE_SCALE_ARTIFACT"
    elif mode == "optimizer_delivery" and float(invariant) < 0:
        report["geometry_adjudication"] = "DELIVERY_CONFLICT_NOT_AUTOMATIC_CREDIT_FAILURE"
    else:
        report["geometry_adjudication"] = "NO_DIRECTIONAL_CONFLICT_CLASSIFIED"
    report["raw_euclidean_cosine_role"] = "fixed_coordinate_secondary_diagnostic_only"
    return report


def _validate_group_common(group, index, manifest_hash, mode):
    required = ("candidate_group_id", "candidate_group_manifest_hash", "checkpoint_hash", "state_hash",
                "subspace_hash", "loss_graph_hash", "candidate_hashes", "score_gradients",
                "policy_controlled_token_kinds", "writer_score_mask_complete",
                "writer_score_mask_includes_eos_or_stop", "candidate_group_prefrozen", "capture_hash")
    missing = [key for key in required if key not in group]
    if missing:
        _no_go(f"group={index} missing={missing}")
    for key in ("candidate_group_manifest_hash", "checkpoint_hash", "state_hash", "subspace_hash", "loss_graph_hash"):
        if not SHA256.fullmatch(str(group[key])):
            _no_go(f"group={index} invalid {key}")
    if group["candidate_group_manifest_hash"] != manifest_hash or group["candidate_group_prefrozen"] is not True:
        _no_go(f"group={index} not from prefrozen group manifest")
    if (group["writer_score_mask_complete"] is not True or
            group["writer_score_mask_includes_eos_or_stop"] is not True or
            "eos_or_stop" not in group["policy_controlled_token_kinds"]):
        _no_go(f"group={index} incomplete writer score mask/EOS-stop coverage")
    candidates = group["candidate_hashes"]
    if not isinstance(candidates, list) or len(candidates) < 2 or len(candidates) != len(set(candidates)):
        _no_go(f"group={index} requires unique candidates and n>=2")
    if any(not SHA256.fullmatch(str(item)) for item in candidates):
        _no_go(f"group={index} invalid candidate hash")
    if mode == "CV_same_endpoint":
        flattened = ([float(x) for x in group["commit_returns"]] +
                     [float(x) for x in group["noop_baseline_returns"]] +
                     [float(x) for row in group["score_gradients"] for x in row])
    else:
        flattened = ([float(x) for x in group["training_commit_returns"]] +
                     [float(x) for x in group["scientific_eval_returns"]] +
                     [float(x) for x in group["endpoint_label_shuffle_returns"]] +
                     [float(x) for rows in group["component_ablation_returns"].values() for x in rows] +
                     [float(x) for row in group["score_gradients"] for x in row])
    if group["capture_hash"] != vector_hash(flattened):
        _no_go(f"group={index} capture hash mismatch")


def _validate_groups(value, mode):
    groups = value.get("groups")
    if not isinstance(groups, list) or len(groups) < 4:
        _no_go("at least four prefrozen independent candidate groups are required for plumbing")
    if "events" in value:
        _no_go("legacy event-level W4 manifest is forbidden")
    seen = set()
    outputs = []
    component_names = None
    for index, group in enumerate(groups):
        identity = group.get("candidate_group_id")
        if not identity or identity in seen:
            _no_go(f"duplicate/missing candidate_group_id={identity}")
        seen.add(identity)
        try:
            _validate_group_common(group, index, value["candidate_group_manifest_hash"], mode)
            output = group_estimators(group) if mode == "CV_same_endpoint" else objective_mismatch_estimators(group)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            _no_go(f"group={index} malformed capture: {exc}")
        if mode == "CV_same_endpoint" and not _close(output["g_credit_debiased"], output["g_credit_loo"]):
            _no_go(f"group={index} debiased all-mean != LOO")
        if mode == "OM_distinct_endpoint":
            names = set(output["g_component_ablations"])
            if component_names is None:
                component_names = names
            elif names != component_names:
                _no_go("component ablation family differs across independent groups")
        outputs.append(output)
    if len({len((row["g_cf_external_noop"] if mode == "CV_same_endpoint" else row["g_train"]))
            for row in outputs}) != 1:
        _no_go("gradient subspace differs across independent groups")
    return groups, outputs


def _cv_report(value, groups, outputs, mismatches):
    if mismatches:
        _no_go("CV mode cannot contain endpoint mismatch", "ENDPOINT_TARGET_AMBIGUOUS")
    if _wrong(value, CV_REQUIRED):
        _no_go(f"CV control-variate contract failed {_wrong(value, CV_REQUIRED)}")
    if not SHA256.fullmatch(str(value.get("exact_noop_v2_manifest_hash", ""))):
        _no_go("missing/invalid exact_noop_v2_manifest_hash")
    reference_available = value.get("independent_many_action_reference_available")
    if not isinstance(reference_available, bool):
        _no_go("independent_many_action_reference_available must be boolean")
    credit = [row["g_credit_debiased"] for row in outputs]
    loo = [row["g_credit_loo"] for row in outputs]
    g_cf = [row["g_cf_external_noop"] for row in outputs]
    raw = [row["g_raw_commit"] for row in outputs]
    differences = [_subtract(left, right) for left, right in zip(credit, g_cf)]
    report = {
        "audit_label": "same_endpoint_control_variate_expected_equivalence",
        "theoretical_null": "E_group[G_credit_debiased-G_CF]=0",
        "paired_group_mean_g_credit_debiased_minus_g_cf": _mean(differences),
        "paired_group_variance_g_credit_debiased_minus_g_cf": _variance(differences),
        "estimator_cross_group_variance": {
            "raw_commit": _variance(raw), "debiased_all_mean_and_loo": _variance(credit),
            "external_exact_noop_control_variate": _variance(g_cf),
        },
        "including_self_all_mean": {
            "expected_scale_formula": "(n-1)/n", "debias_formula": "n/(n-1)",
            "n4_expected_scale": .75, "n4_debias_factor": 4 / 3,
            "debiased_is_batchwise_identical_to_loo": all(_close(a, b) for a, b in zip(credit, loo)),
        },
        "exact_noop_role": "control_variate_not_new_or_truer_action_value_target",
        "equal_reward_external_nonzero_interpretation": "zero_mean_finite_batch_score_noise_not_credit_evidence",
        "nonzero_paired_mean_interpretation": (
            "diagnose_endpoint_parity_IID_baseline_independence_selection_mask_or_loss_reconstruction_not_credit_discovery"),
    }
    if reference_available:
        references = []
        for index, group in enumerate(groups):
            reference = group.get("many_action_reference_gradient")
            if not isinstance(reference, list) or len(reference) != len(g_cf[index]):
                _no_go(f"group={index} missing independent many-action reference")
            references.append([float(x) for x in reference])
        report["mse_status"] = "reported_against_independent_many_action_or_replication_reference"
        report["estimator_mse"] = {
            "raw_commit": _mse(raw, references), "debiased_all_mean_and_loo": _mse(credit, references),
            "external_exact_noop_control_variate": _mse(g_cf, references),
        }
    else:
        if any("many_action_reference_gradient" in group for group in groups):
            _no_go("reference supplied while independent reference availability is false")
        report["mse_status"] = "not_reported_no_independent_many_action_or_replication_reference"
    review = value.get("engineering_application_review", {})
    review_keys = {"expected_equivalence_pass", "noop_variance_or_mse_reduction_pass",
                   "beats_equal_cost_loo", "beats_equal_cost_state_value", "fresh_endpoint_safety_pass"}
    if set(review) != review_keys or any(not isinstance(review[key], bool) for key in review_keys):
        _no_go("engineering_application_review is incomplete")
    report["engineering_application_eligible_to_request"] = all(review.values())
    return report


def _om_report(value, groups, outputs, mismatches):
    if not mismatches:
        _no_go("OM mode requires at least one endpoint mismatch", "ENDPOINT_TARGET_AMBIGUOUS")
    if _wrong(value, OM_REQUIRED):
        _no_go(f"OM surrogate-objective contract failed {_wrong(value, OM_REQUIRED)}")
    differences = [row["g_eval_minus_g_train"] for row in outputs]
    shuffles = [row["g_endpoint_label_shuffle"] for row in outputs]
    names = sorted(outputs[0]["g_component_ablations"])
    return {
        "audit_label": "surrogate_objective_gradient_mismatch",
        "same_reward_credit_estimator_loss_claim_authorized": False,
        "mismatched_endpoint_dimensions": mismatches,
        "g_eval_minus_g_train_cluster_interval": _cluster_interval(differences),
        "endpoint_label_shuffle_null_cluster_interval": _cluster_interval(shuffles),
        "equal_scale_component_ablation_cluster_intervals": {
            name: _cluster_interval([row["g_component_ablations"][name] for row in outputs])
            for name in names
        },
        "exact_noop_can_remedy_endpoint_mismatch": False,
        "interpretation": "objective_mismatch_not_lost_credit_for_same_reward",
        "engineering_application_eligible_to_request": False,
    }


def validate(value: dict) -> dict:
    wrong = _wrong(value, COMMON_REQUIRED)
    if wrong:
        _no_go(f"common v6 contract failed {wrong}")
    if not SHA256.fullmatch(str(value.get("candidate_group_manifest_hash", ""))):
        _no_go("missing/invalid candidate_group_manifest_hash")
    mode, mismatches = _endpoint_mode(value)
    policy_scope = _policy_derivative_scope(value)
    geometry = _gradient_geometry(value)
    evidence = set(value.get("evidence_basis", []))
    forbidden = sorted(evidence & FORBIDDEN_EVIDENCE_BASES)
    if forbidden:
        _no_go(f"forbidden single-batch evidence basis {forbidden}")
    expected_evidence = ({"paired_group_mean", "cross_group_variance"} if mode == "CV_same_endpoint" else
                         {"clustered_objective_difference", "endpoint_label_shuffle_null", "component_ablation"})
    if evidence != expected_evidence:
        _no_go(f"evidence_basis must equal {sorted(expected_evidence)} for mode={mode}")
    groups, outputs = _validate_groups(value, mode)
    mode_report = (_cv_report(value, groups, outputs, mismatches) if mode == "CV_same_endpoint" else
                   _om_report(value, groups, outputs, mismatches))
    return {
        "status": ("W4_V8_SCIENTIFIC_AUDIT_READY" if len(groups) >= 20 else "W4_V8_PLUMBING_ONLY"),
        "endpoint_mode": mode,
        "endpoint_parity_complete": True,
        "highest_claim_level": "W3",
        "w4_claim_authorized": False,
        "training_authorized": False,
        "engineering_two_step_authorized": False,
        "algorithm_novelty": False,
        "optimizer_steps": 0,
        "new_rollouts": False,
        "independent_candidate_groups": len(groups),
        "scientific_audit_minimum_groups": 20,
        "single_batch_directional_evidence_authorized": False,
        "prohibited_inferences": PROHIBITED_INFERENCES,
        "engineering_application_requires_new_frozen_authorization": True,
        **policy_scope,
        **geometry,
        **mode_report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.manifest.read_text())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
