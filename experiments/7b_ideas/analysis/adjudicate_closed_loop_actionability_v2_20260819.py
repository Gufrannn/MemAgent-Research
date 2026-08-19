#!/usr/bin/env python3
"""Closed-loop v4 terminal-IUT adjudicator with totality v2 and horizon v3."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

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
    if value.get("schema_version") != "closed-loop-commit-v4-terminal-IUT":
        _fail("schema version mismatch")
    if value.get("closed_loop_existing_gate_authorized") is not True:
        return {"status": "CLOSED_LOOP_NOT_AUTHORIZED", "point_value_authorized": False,
                "training_authorized": False, "new_rollouts_authorized": False, "queue_changed": False}
    horizon = _horizon_contract(value)
    required = {"intent_to_execute_primary": True, "common_valid_intersection_primary": False,
                "intersection_diagnostic_only": True, "retry_to_success": False,
                "infrastructure_failure_scientific_zero": False, "audit_size": 16,
                "policy_totality_and_attrition_handling": "adjudicator_v2_hard_gate",
                "terminal_attribution_gate": "terminal_pairwise_IUT_and_regret",
                "local_action_attribution_authorized": False,
                "new_local_interventions": False,
                "clairvoyant_assignments_feed_selector_or_gate": False,
                "package_selector_training_authorized": False,
                "new_independent_selector_confirmation_authorized": False,
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
    v_fixed_star = max(values.values())
    best_fixed_policy = max(sorted(values), key=lambda policy: values[policy])
    v_clair = sum(max(terminal_values[(str(stable_id), policy)] for policy in policies)
                  for stable_id in assigned) / len(assigned)
    terminal_report = {"terminal_pairwise_contrasts": pairwise,
                       "V_fixed_star": v_fixed_star, "best_fixed_policy": best_fixed_policy,
                       "V_clair": v_clair, "V_clair_role": "clairvoyant_sample_upper_bound_not_executable_policy",
                       "Opportunity_package": v_clair - v_fixed_star,
                       "Regret_GC_clair": v_clair - values["GC"],
                       "Regret_GC_clair_role": "descriptive_clairvoyant_gap_not_regret_to_best_fixed",
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
    if not all(row["passes_shared_SESOI_interval_gate"] for row in pairwise.values()):
        return {"status": "CLOSED_LOOP_CONTROL_DOMINANCE_NO_GO", "policy_values": values,
                "point_value_authorized": True, "control_dominance_claim_authorized": False,
                "training_authorized": False, **horizon, **terminal_report}
    status = ("CLOSED_LOOP_INTENT_TO_EXECUTE_POINT_VALUE_QUALIFIED" if horizon["horizon_mode"] == "H_fixed"
              else "FINITE_TWO_TURN_ACTIONABILITY_WITH_SELECTED_THREE_TURN_DESCRIPTION")
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
    base = {"schema_version": "closed-loop-commit-v4-terminal-IUT", "closed_loop_existing_gate_authorized": True,
            "intent_to_execute_primary": True, "common_valid_intersection_primary": False,
            "intersection_diagnostic_only": True, "retry_to_success": False,
            "infrastructure_failure_scientific_zero": False, "audit_size": 16,
            "policy_totality_and_attrition_handling": "adjudicator_v2_hard_gate",
            "terminal_attribution_gate": "terminal_pairwise_IUT_and_regret",
            "local_action_attribution_authorized": False, "new_local_interventions": False,
            "clairvoyant_assignments_feed_selector_or_gate": False,
            "package_selector_training_authorized": False,
            "new_independent_selector_confirmation_authorized": False,
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
    assert adjudicate({**base, "executions": rows})["point_value_authorized"]
    assert adjudicate({**base, "executions": rows[:-1]})["status"] == "AUDIT16_CONSTRUCTION_DIAGNOSTIC_ONLY"
    undefined = [dict(row) for row in rows]; undefined[2]["certificate_defined"] = False
    assert adjudicate({**base, "executions": undefined})["status"] == "CERTIFICATE_POLICY_NOT_TOTAL"
    infra = [dict(row) for row in rows]; infra[0].update({"execution_status": "infrastructure_failure",
      "official_endpoint_value": None, "failure_class": "OOM", "incident_ledger_hash": "b" * 64,
      "state_preserved": True, "full_manifest_rerun_required": True, "old_ledger_retained": True,
      "new_unique_experiment_name": "audit16_rerun_2"})
    assert adjudicate({**base, "executions": infra})["status"].startswith("INFRASTRUCTURE_FAILURE")
    print("closed_loop_commit_v4_terminal_IUT_self_test=ok")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.manifest: parser.error("--manifest required")
    print(json.dumps(adjudicate(json.loads(args.manifest.read_text())), indent=2, sort_keys=True))


if __name__ == "__main__": main()
