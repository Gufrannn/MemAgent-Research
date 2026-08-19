#!/usr/bin/env python3
"""Closed-loop v8 resource-mode adjudicator with terminal pairwise IUT."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.closed_loop_composition import audit_composition_gap  # noqa: E402
from recurrent.research.closed_loop_randomness import audit_randomness_estimand  # noqa: E402
from recurrent.research.closed_loop_oracle import audit_oracle_opportunity  # noqa: E402
from recurrent.research.closed_loop_resources import audit_resource_mode  # noqa: E402

SHA256 = re.compile(r"^[0-9a-f]{64}$")
SCIENTIFIC_TERMINALS = {"complete", "scientific_invalid", "truncation", "parser_failure", "natural_stop"}


def _fail(reason):
    raise ValueError(f"CLOSED_LOOP_TOTALITY_NO_GO: {reason}")


def _horizon_contract(value):
    mode = value.get("horizon_mode")
    if mode == "H_fixed":
        required = {"horizon_frozen_before_confirm32_policy_outcomes": True,
                    "horizon_freeze_basis": "plumbing_resource_failure_outcome_blind_claim_need_only",
                    "complete_frozen_horizon_executed": True,
                    "audit16_horizon_selected_by_policy_value_direction": False,
                    "additional_resources_authorized": False}
        wrong = {key: (value.get(key), target) for key, target in required.items() if value.get(key) != target}
        if value.get("horizon") not in {2, 3} or wrong:
            _fail(f"H_fixed contract failed horizon={value.get('horizon')} wrong={wrong}")
        return {"horizon_mode": mode, "primary_horizon": value["horizon"],
                "third_turn_claim_role": "confirmatory_only_if_horizon_3_was_prefrozen",
                "maximum_claim": f"FINITE_FIXED_{value['horizon']}_TURN_ACTIONABILITY"}
    if mode == "H_selected":
        required = {"confirm32_two_turn_outcome_accessed_before_third_turn": True,
                    "two_turn_primary_adjudication_prefrozen": True,
                    "third_turn_role": "outcome_triggered_selected_horizon_stress_description",
                    "third_turn_ordinary_unselected_ci_or_pvalue": False,
                    "third_turn_confirmatory_upgrade": False,
                    "third_turn_positive_replaces_two_turn_negative": False,
                    "untouched_three_turn_confirmation_authorized": False,
                    "additional_resources_authorized": False}
        wrong = {key: (value.get(key), target) for key, target in required.items() if value.get(key) != target}
        if wrong: _fail(f"H_selected contract failed {wrong}")
        return {"horizon_mode": mode, "primary_horizon": 2,
                "third_turn_claim_role": "outcome_triggered_selected_horizon_stress_description",
                "maximum_claim": "FINITE_TWO_TURN_ACTIONABILITY_WITH_SELECTED_THREE_TURN_DESCRIPTION"}
    _fail("horizon_mode must be H_fixed or H_selected")


def adjudicate(value):
    if value.get("schema_version") != "closed-loop-commit-v8-resource-mode":
        _fail("schema version mismatch")
    if value.get("closed_loop_existing_gate_authorized") is not True:
        return {"status": "CLOSED_LOOP_NOT_AUTHORIZED", "point_value_authorized": False,
                "training_authorized": False, "new_rollouts_authorized": False, "queue_changed": False}
    horizon = _horizon_contract(value)
    required = {"intent_to_execute_primary": True, "common_valid_intersection_primary": False,
                "intersection_diagnostic_only": True, "retry_to_success": False,
                "infrastructure_failure_scientific_zero": False, "audit_size": 16,
                "policy_totality_and_attrition_handling": "adjudicator_v2_hard_gate",
                "terminal_attribution_gate": "terminal_pairwise_IUT",
                "oracle_auxiliary_gate": "oracle_semantics_and_opportunity",
                "oracle_failure_can_veto_terminal_IUT": False,
                "resource_axis_role": "prefrozen_resource_estimand_and_cost_qualification",
                "resource_mode_frozen_before_outcome": True,
                "raw_QA_IUT_equals_equal_budget_or_practical_advantage": False,
                "local_action_attribution_authorized": False,
                "new_local_interventions": False,
                "clairvoyant_assignments_feed_selector_or_gate": False,
                "package_selector_training_authorized": False,
                "new_independent_selector_confirmation_authorized": False,
                "composition_gap_role": "orthogonal_transport_diagnostic_not_actionability_gate",
                "composition_gap_can_veto_terminal_IUT": False,
                "composition_gap_can_rescue_terminal_IUT": False,
                "randomness_axis_role": "orthogonal_execution_estimand",
                "randomness_mode_frozen_before_outcome": True,
                "seed_or_replicate_increases_scientific_n": False,
                "optimizer_steps": 0, "new_rollouts": False}
    wrong = {key: (value.get(key), expected) for key, expected in required.items() if value.get(key) != expected}
    if wrong: _fail(f"intent-to-execute contract failed {wrong}")
    if not SHA256.fullmatch(str(value.get("assignment_manifest_hash", ""))):
        _fail("assignment manifest hash missing")
    assigned = value.get("assigned_stable_example_ids")
    policies = value.get("policies")
    if (not isinstance(assigned, list) or len(assigned) != 16 or len(set(assigned)) != 16 or
            not isinstance(policies, list) or set(policies) != {"GC", "GF", "GN", "GS"} or len(policies) != 4):
        _fail("Audit16 requires sixteen unique examples and exactly GC/GF/GN/GS")
    expected = {(str(stable_id), str(policy)) for stable_id in assigned for policy in policies}
    rows = value.get("executions")
    if not isinstance(rows, list): _fail("execution ledger missing")
    seen = set(); by_policy = defaultdict(list); terminal_values = {}; infrastructure = []; fallback_weight = defaultdict(float)
    certificate_not_total = []
    for index, row in enumerate(rows):
        key = (str(row.get("stable_example_id")), str(row.get("policy")))
        if key not in expected: _fail(f"row={index} not in prefrozen assignment/policy product")
        if key in seen: _fail(f"duplicate execution row={key}")
        seen.add(key)
        status = row.get("execution_status")
        if status == "scientific_terminal":
            terminal = row.get("terminal_class")
            endpoint = row.get("official_endpoint_value")
            if terminal not in SCIENTIFIC_TERMINALS or not isinstance(endpoint, (int, float)) or not math.isfinite(endpoint):
                _fail(f"row={index} scientific terminal/official endpoint invalid")
            if row.get("retry_count") != 0 or row.get("retry_to_success") is not False:
                _fail(f"row={index} retry-to-success forbidden")
            if row.get("certificate_defined") is False:
                fallback = row.get("certificate_fallback")
                if (not isinstance(fallback, dict) or fallback.get("frozen_before_outcome") is not True or
                        not fallback.get("action") or not fallback.get("rule_hash") or
                        fallback.get("applied") is not True):
                    certificate_not_total.append(key)
                else:
                    fallback_weight[str(row["policy"])] += 1.0 / len(assigned)
            elif row.get("certificate_defined") is not True:
                _fail(f"row={index} certificate-defined flag missing")
            by_policy[str(row["policy"])].append(float(endpoint))
            terminal_values[key] = float(endpoint)
        elif status == "infrastructure_failure":
            if row.get("official_endpoint_value") is not None:
                _fail(f"row={index} infrastructure failure assigned a scientific endpoint")
            if (row.get("failure_class") not in {"OOM", "server", "identity", "data"} or
                    not row.get("incident_ledger_hash") or row.get("state_preserved") is not True or
                    row.get("full_manifest_rerun_required") is not True or
                    row.get("old_ledger_retained") is not True or not row.get("new_unique_experiment_name")):
                _fail(f"row={index} infrastructure failure handling incomplete")
            infrastructure.append(key)
        else:
            _fail(f"row={index} must be scientific_terminal or infrastructure_failure")
    missing = sorted(expected - seen)
    if certificate_not_total:
        return {"status": "CERTIFICATE_POLICY_NOT_TOTAL", "non_total_rows": certificate_not_total,
                "point_value_authorized": False, "training_authorized": False,
                "policy_totality_and_attrition_handling_pass": False}
    if missing or len(seen) != 16 * len(policies):
        return {"status": "AUDIT16_CONSTRUCTION_DIAGNOSTIC_ONLY", "completed_cells": len(seen),
                "required_cells": 16 * len(policies), "missing_cells": missing,
                "point_value_authorized": False, "training_authorized": False,
                "policy_totality_and_attrition_handling_pass": False}
    if infrastructure:
        return {"status": "INFRASTRUCTURE_FAILURE_FULL_MANIFEST_RERUN_REQUIRED",
                "infrastructure_failure_cells": infrastructure, "point_value_authorized": False,
                "scientific_zero_imputed": False, "old_ledger_retained": True,
                "training_authorized": False, "policy_totality_and_attrition_handling_pass": False}
    values = {policy: sum(by_policy[policy]) / len(assigned) for policy in policies}
    randomness_manifest = value.get("randomness_manifest")
    primary_mode = value.get("randomness_primary_mode")
    if (primary_mode not in {"D", "S", "F"} or not isinstance(randomness_manifest, dict) or
            randomness_manifest.get("primary_mode") != primary_mode or
            randomness_manifest.get("mode_frozen_before_outcome") is not True):
        randomness_report = {"status": "RANDOMNESS_ESTIMAND_INVALID",
                             "reason": "frozen_primary_mode_or_manifest_missing_or_mismatched",
                             "training_authorized": False}
    else:
        randomness_report = audit_randomness_estimand(randomness_manifest)
    if primary_mode == "D" and randomness_report["status"] in {
            "DETERMINISTIC_PROTOCOL_PRIMARY", "STOCHASTIC_NONTRANSPORT"}:
        if randomness_manifest.get("stochastic_sensitivity_requested") is True:
            observed = values["GC"] - max(values[control] for control in ("GF", "GN", "GS"))
            declared = randomness_manifest.get("deterministic_gc_minus_best_control")
            if not isinstance(declared, (int, float)) or not math.isclose(
                    float(declared), observed, rel_tol=0.0, abs_tol=1e-12):
                randomness_report = {"status": "RANDOMNESS_ESTIMAND_INVALID",
                                     "reason": "deterministic_sensitivity_contrast_not_joined_to_terminal_ledger",
                                     "training_authorized": False}
    elif primary_mode == "S" and randomness_report["status"] == "SEED_MARGINAL_STOCHASTIC_POLICY_VALUE_QUALIFIED":
        means = {(row["stable_example_id"], row["policy"]): row["replicate_mean_endpoint"]
                 for row in randomness_report["policy_example_replicate_means"]}
        exact_join = set(means) == expected and all(math.isclose(
            means[key], terminal_values[key], rel_tol=0.0, abs_tol=1e-12) for key in expected)
        if not exact_join:
            randomness_report = {"status": "STOCHASTIC_POLICY_MEAN_INVALID",
                                 "reason": "replicate_means_do_not_biject_execution_ledger",
                                 "policy_mean_identified": False, "training_authorized": False}
    valid_randomness_statuses = {"DETERMINISTIC_PROTOCOL_PRIMARY", "STOCHASTIC_NONTRANSPORT",
                                 "SEED_MARGINAL_STOCHASTIC_POLICY_VALUE_QUALIFIED",
                                 "SINGLE_FROZEN_SEED_REALIZED_SCREENING_ONLY"}
    if randomness_report["status"] not in valid_randomness_statuses:
        status = ("STOCHASTIC_POLICY_MEAN_INVALID" if primary_mode == "S"
                  else "RANDOMNESS_ESTIMAND_INVALID")
        return {"status": status, "randomness_estimand_audit": randomness_report,
                "point_value_authorized": False, "control_dominance_claim_authorized": False,
                "training_authorized": False, "new_rollouts_authorized": False,
                "queue_changed": False}
    resource_manifest = value.get("resource_manifest")
    primary_resource_mode = value.get("primary_resource_mode")
    if (primary_resource_mode not in {"A", "B"} or not isinstance(resource_manifest, dict) or
            resource_manifest.get("primary_resource_mode") != primary_resource_mode or
            resource_manifest.get("mode_frozen_before_outcome") is not True):
        resource_report = {"status": ("FIXED_BUDGET_POLICY_VALUE_INVALID"
                                      if primary_resource_mode == "B" else "COST_UNQUALIFIED"),
                           "reason": "frozen_primary_resource_mode_or_manifest_missing_or_mismatched",
                           "fixed_budget_policy_value_authorized": False,
                           "training_authorized": False}
    else:
        resource_report = audit_resource_mode(resource_manifest, terminal_values=terminal_values,
                                              assigned=[str(item) for item in assigned],
                                              policies=[str(item) for item in policies])
    contrasts = value.get("prefrozen_terminal_contrasts")
    if contrasts != ["GC-GF", "GC-GN", "GC-GS"]:
        _fail("terminal contrasts must be prefrozen as GC-GF, GC-GN, GC-GS")
    sesoi = value.get("terminal_contrast_SESOI")
    lower = value.get("pairwise_interval_lower_bounds")
    if (not isinstance(sesoi, (int, float)) or not math.isfinite(sesoi) or sesoi < 0 or
            not isinstance(lower, dict) or set(lower) != set(contrasts) or
            not all(isinstance(bound, (int, float)) and math.isfinite(bound) for bound in lower.values()) or
            not SHA256.fullmatch(str(value.get("pairwise_interval_method_hash", "")))):
        _fail("shared SESOI/interval gate metadata incomplete")
    pairwise = {}
    for control in ("GF", "GN", "GS"):
        name = f"GC-{control}"
        effects = [terminal_values[(str(stable_id), "GC")] - terminal_values[(str(stable_id), control)]
                   for stable_id in assigned]
        wins = sum(effect > 0 for effect in effects); ties = sum(effect == 0 for effect in effects)
        pairwise[name] = {"mean_terminal_contrast": sum(effects) / len(effects),
                          "win_tie_loss": [wins, ties, len(effects) - wins - ties],
                          "positive_terminal_effect_mass": sum(max(effect, 0.0) for effect in effects) / len(effects),
                          "negative_terminal_effect_mass": sum(max(-effect, 0.0) for effect in effects) / len(effects),
                          "interval_lower_bound": float(lower[name]),
                          "passes_shared_SESOI_interval_gate": float(lower[name]) >= float(sesoi)}
    oracle_manifest = value.get("oracle_opportunity_manifest")
    if isinstance(oracle_manifest, dict):
        oracle_report = audit_oracle_opportunity(
            oracle_manifest, primary_mode=primary_mode, terminal_values=terminal_values,
            assigned=[str(item) for item in assigned], policies=[str(item) for item in policies])
    else:
        oracle_report = {"status": "ORACLE_OPPORTUNITY_INVALID",
                         "reason": "oracle_opportunity_manifest_missing",
                         "can_veto_terminal_pairwise_IUT": False, "training_authorized": False}
    terminal_report = {"terminal_pairwise_contrasts": pairwise,
                       "oracle_semantics_and_opportunity": oracle_report,
                       "oracle_auxiliary_can_veto_terminal_pairwise_IUT": False,
                       "clairvoyant_assignments_feed_selector_or_gate": False,
                       "package_selector_training_authorized": False,
                       "GC_minus_max_control_point_summary": min(pairwise[name]["mean_terminal_contrast"]
                                                                  for name in contrasts),
                       "control_dominance_is_intersection_union": True,
                       "posthoc_single_control_confirmation_authorized": False,
                       "ordinary_best_control_interval_authorized": False,
                       "terminal_attribution_scope": "complete_policy_package_total_difference_only",
                       "prohibited_local_attributions": ["harmful_commit", "beneficial_rejection", "turn_local_credit",
                         "rollback_caused_rescue", "certificate_decision_mediated_gain"]}
    composition_requested = value.get("composition_gap_requested")
    if not isinstance(composition_requested, bool):
        _fail("composition_gap_requested must be boolean")
    if composition_requested:
        composition_manifest = value.get("composition_gap_manifest")
        if not isinstance(composition_manifest, dict):
            composition_report = {"status": "COMPOSITION_GAP_NOT_IDENTIFIED",
                                  "reason": "composition_gap_manifest_missing", "outcomes_read": False,
                                  "feedback_claim_authorized": False, "actionability_gate": False,
                                  "training_authorized": False}
        else:
            composition_report = audit_composition_gap(composition_manifest)
            if composition_report["status"] == "COMPOSITION_GAP_QUALIFIED":
                composition_rows = {str(row["stable_example_id"]): row for row in composition_manifest["rows"]}
                exact_join = set(composition_rows) == {str(item) for item in assigned}
                if exact_join:
                    for stable_id in assigned:
                        row = composition_rows[str(stable_id)]
                        actual_gc = terminal_values[(str(stable_id), "GC")]
                        actual_control = max(terminal_values[(str(stable_id), control)]
                                             for control in ("GF", "GN", "GS"))
                        if (float(row["direct_gc_terminal"]) != actual_gc or
                                float(row["best_control_terminal"]) != actual_control):
                            exact_join = False; break
                if not exact_join:
                    composition_report = {"status": "COMPOSITION_GAP_NOT_IDENTIFIED",
                                          "reason": "direct_GC_or_control_rows_do_not_biject_terminal_IUT_ledger",
                                          "outcomes_read": False, "feedback_claim_authorized": False,
                                          "actionability_gate": False, "training_authorized": False}
    else:
        composition_report = {"status": "COMPOSITION_GAP_NOT_REQUESTED",
                              "feedback_claim_authorized": False, "actionability_gate": False,
                              "training_authorized": False}
    terminal_report.update({"composition_transport_audit": composition_report,
                            "composition_gap_orthogonal_to_actionability": True,
                            "composition_gap_can_veto_terminal_IUT": False,
                            "composition_gap_can_rescue_terminal_IUT": False,
                            "randomness_estimand_audit": randomness_report,
                            "randomness_status": randomness_report["status"],
                            "randomness_axis_role": "orthogonal_execution_estimand",
                            "seed_or_replicate_increases_scientific_n": False,
                            "resource_mode_audit": resource_report,
                            "terminal_IUT_scope": "current_protocol_raw_QA_not_equal_budget_utility_or_cost_advantage",
                            "raw_QA_IUT_equals_equal_budget_or_practical_advantage": False})
    if primary_resource_mode == "B" and resource_report["status"] != "FIXED_BUDGET_POLICY_VALUE_QUALIFIED":
        return {"status": "FIXED_BUDGET_POLICY_VALUE_INVALID",
                "unconstrained_outcome_description": {"policy_values": values,
                                                       "terminal_pairwise_contrasts": pairwise},
                "fixed_budget_policy_value_authorized": False,
                "control_dominance_claim_authorized": False,
                "training_authorized": False, **horizon, **terminal_report}
    cost_unqualified = primary_resource_mode == "A" and resource_report["status"] != "ACCURACY_FIRST_RESOURCE_LEDGER_QUALIFIED"
    if primary_mode == "F":
        status = "SINGLE_FROZEN_SEED_REALIZED_SCREENING_ONLY"
        if cost_unqualified: status += "_WITH_COST_UNQUALIFIED"
        return {"status": status, "policy_values": values, "point_value_authorized": True,
                "confirmatory_claim_authorized": False,
                "control_dominance_claim_authorized": False,
                "training_authorized": False, **horizon, **terminal_report}
    if not all(row["passes_shared_SESOI_interval_gate"] for row in pairwise.values()):
        status = "CLOSED_LOOP_CONTROL_DOMINANCE_NO_GO"
        if cost_unqualified: status += "_WITH_COST_UNQUALIFIED"
        return {"status": status, "policy_values": values,
                "point_value_authorized": True, "control_dominance_claim_authorized": False,
                "training_authorized": False, **horizon, **terminal_report}
    if (composition_report["status"] == "COMPOSITION_GAP_QUALIFIED" and
            composition_report.get("myopic_nontransport") is True):
        status = "CLOSED_LOOP_ACTIONABILITY_WITH_MYOPIC_NONTRANSPORT"
    else:
        status = ("CLOSED_LOOP_INTENT_TO_EXECUTE_POINT_VALUE_QUALIFIED" if horizon["horizon_mode"] == "H_fixed"
                  else "FINITE_TWO_TURN_ACTIONABILITY_WITH_SELECTED_THREE_TURN_DESCRIPTION")
    if oracle_report["status"] == "ORACLE_OPPORTUNITY_INVALID":
        status += "_WITH_ORACLE_OPPORTUNITY_INVALID"
    if cost_unqualified:
        status += "_WITH_COST_UNQUALIFIED"
    report = {"status": status,
            "policy_values": values, "assigned_examples": 16, "execution_cells": len(seen),
            "fallback_mass_by_policy": dict(fallback_weight),
            "common_valid_intersection_role": "diagnostic_appendix_only",
            "retry_to_success": False, "point_value_authorized": True,
            "policy_totality_and_attrition_handling_pass": True,
              "training_authorized": False, "new_rollouts_authorized": False, "queue_changed": False,
              "control_dominance_claim_authorized": True, **terminal_report}
    report.update(horizon)
    return report


def self_test():
    examples = [f"e{i}" for i in range(16)]; policies = ["GC", "GF", "GN", "GS"]
    base = {"schema_version": "closed-loop-commit-v8-resource-mode", "closed_loop_existing_gate_authorized": True,
            "intent_to_execute_primary": True, "common_valid_intersection_primary": False,
            "intersection_diagnostic_only": True, "retry_to_success": False,
            "infrastructure_failure_scientific_zero": False, "audit_size": 16,
            "policy_totality_and_attrition_handling": "adjudicator_v2_hard_gate",
            "terminal_attribution_gate": "terminal_pairwise_IUT",
            "oracle_auxiliary_gate": "oracle_semantics_and_opportunity",
            "oracle_failure_can_veto_terminal_IUT": False,
            "resource_axis_role": "prefrozen_resource_estimand_and_cost_qualification",
            "resource_mode_frozen_before_outcome": True,
            "primary_resource_mode": "A",
            "raw_QA_IUT_equals_equal_budget_or_practical_advantage": False,
            "local_action_attribution_authorized": False, "new_local_interventions": False,
            "clairvoyant_assignments_feed_selector_or_gate": False,
            "package_selector_training_authorized": False,
            "new_independent_selector_confirmation_authorized": False,
            "composition_gap_role": "orthogonal_transport_diagnostic_not_actionability_gate",
            "composition_gap_can_veto_terminal_IUT": False,
            "composition_gap_can_rescue_terminal_IUT": False,
            "composition_gap_requested": False,
            "randomness_axis_role": "orthogonal_execution_estimand",
            "randomness_mode_frozen_before_outcome": True,
            "randomness_primary_mode": "D",
            "seed_or_replicate_increases_scientific_n": False,
            "randomness_manifest": {"schema_version": "closed-loop-randomness-estimand-v1",
              "primary_mode": "D", "mode_frozen_before_outcome": True,
              "estimand": "temperature0_deterministic_protocol_value", "temperature": 0,
              "deterministic_protocol_frozen": True, "deterministic_protocol_hash": "d" * 64,
              "stochastic_sensitivity_requested": False, "optimizer_steps": 0, "new_rollouts": False},
            "oracle_opportunity_manifest": {"schema_version": "closed-loop-oracle-opportunity-v2",
              "role": "orthogonal_auxiliary_not_terminal_pairwise_IUT_gate",
              "failure_can_veto_terminal_pairwise_IUT": False,
              "seed_or_replicate_increases_scientific_n": False,
              "primary_mode": "D", "deterministic_pointwise_package_oracle_authorized": True,
              "optimizer_steps": 0, "new_rollouts": False},
            "optimizer_steps": 0, "new_rollouts": False, "assignment_manifest_hash": "a" * 64,
            "assigned_stable_example_ids": examples, "policies": policies,
            "horizon_mode": "H_fixed", "horizon": 2,
            "horizon_frozen_before_confirm32_policy_outcomes": True,
            "horizon_freeze_basis": "plumbing_resource_failure_outcome_blind_claim_need_only",
            "complete_frozen_horizon_executed": True,
            "audit16_horizon_selected_by_policy_value_direction": False,
            "additional_resources_authorized": False,
            "prefrozen_terminal_contrasts": ["GC-GF", "GC-GN", "GC-GS"],
            "terminal_contrast_SESOI": 0.0,
            "pairwise_interval_lower_bounds": {"GC-GF": 0.0, "GC-GN": 0.0, "GC-GS": 0.0},
            "pairwise_interval_method_hash": "c" * 64}
    rows = [{"stable_example_id": stable_id, "policy": policy, "execution_status": "scientific_terminal",
             "terminal_class": "complete", "official_endpoint_value": .5, "retry_count": 0,
             "retry_to_success": False, "certificate_defined": True}
            for stable_id in examples for policy in policies]
    resource_rows = [{"stable_example_id": stable_id, "policy": policy,
      "cumulative_calls": 1, "input_tokens": 10, "generated_tokens": 5, "context_tokens": 15,
      "walltime_seconds": .1, "fallback_used": False, "certificate_readout_calls": 0,
      "policy_induced_memory_tokens": 0} for stable_id in examples for policy in policies]
    base["resource_manifest"] = {"schema_version": "closed-loop-resource-mode-v1",
      "mode_frozen_before_outcome": True, "primary_resource_mode": "A",
      "estimand": "accuracy_first_current_protocol_raw_QA",
      "raw_QA_IUT_equals_equal_budget_or_practical_advantage": False,
      "writer_proposal_parity_solves_certificate_or_memory_cost_parity": False,
      "utility_reporting_requested": False, "posthoc_lambda_sweep": False,
      "gain_per_token_only_reporting": False, "certificate_practical_gain_claim_requested": False,
      "resource_ledger": resource_rows, "optimizer_steps": 0, "new_rollouts": False}
    assert adjudicate({**base, "executions": rows})["point_value_authorized"]
    assert adjudicate({**base, "executions": rows[:-1]})["status"] == "AUDIT16_CONSTRUCTION_DIAGNOSTIC_ONLY"
    undefined = [dict(row) for row in rows]; undefined[2]["certificate_defined"] = False
    assert adjudicate({**base, "executions": undefined})["status"] == "CERTIFICATE_POLICY_NOT_TOTAL"
    infra = [dict(row) for row in rows]; infra[0].update({"execution_status": "infrastructure_failure",
      "official_endpoint_value": None, "failure_class": "OOM", "incident_ledger_hash": "b" * 64,
      "state_preserved": True, "full_manifest_rerun_required": True, "old_ledger_retained": True,
      "new_unique_experiment_name": "audit16_rerun_2"})
    assert adjudicate({**base, "executions": infra})["status"].startswith("INFRASTRUCTURE_FAILURE")
    print("closed_loop_commit_v8_resource_mode_self_test=ok")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.manifest: parser.error("--manifest required")
    print(json.dumps(adjudicate(json.loads(args.manifest.read_text())), indent=2, sort_keys=True))


if __name__ == "__main__": main()
