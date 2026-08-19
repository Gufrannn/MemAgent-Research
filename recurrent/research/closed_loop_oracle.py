"""Closed-loop v7 oracle/opportunity auxiliary audit.

The terminal pairwise IUT is deliberately outside this module.  Every result
here is an orthogonal reporting qualification and never authorizes execution.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _invalid(reason: str) -> dict[str, Any]:
    return {"status": "ORACLE_OPPORTUNITY_INVALID", "reason": reason,
            "can_veto_terminal_pairwise_IUT": False, "training_authorized": False}


def audit_oracle_opportunity(manifest: dict[str, Any], *, primary_mode: str,
                             terminal_values: dict[tuple[str, str], float],
                             assigned: list[str], policies: list[str]) -> dict[str, Any]:
    required = {"schema_version": "closed-loop-oracle-opportunity-v2",
                "role": "orthogonal_auxiliary_not_terminal_pairwise_IUT_gate",
                "failure_can_veto_terminal_pairwise_IUT": False,
                "seed_or_replicate_increases_scientific_n": False,
                "optimizer_steps": 0, "new_rollouts": False}
    wrong = {key: (manifest.get(key), expected) for key, expected in required.items()
             if manifest.get(key) != expected}
    if wrong or manifest.get("primary_mode") != primary_mode:
        return _invalid(f"oracle_contract_failed:{wrong}")
    expected = {(str(example), str(policy)) for example in assigned for policy in policies}
    if set(terminal_values) != expected:
        return _invalid("terminal_ledger_not_complete_example_x_policy")
    policy_values = {policy: sum(terminal_values[(str(example), policy)] for example in assigned) / len(assigned)
                     for policy in policies}
    v_fixed = max(policy_values.values())
    if primary_mode == "D":
        if manifest.get("deterministic_pointwise_package_oracle_authorized") is not True:
            return _invalid("D_pointwise_package_oracle_not_prefrozen")
        v_pointwise = sum(max(terminal_values[(str(example), policy)] for policy in policies)
                          for example in assigned) / len(assigned)
        return {"status": "DETERMINISTIC_POINTWISE_PACKAGE_ORACLE_QUALIFIED",
                "V_fixed_star": v_fixed, "V_pointwise_package_oracle": v_pointwise,
                "Opportunity_package": v_pointwise - v_fixed,
                "oracle_scope": "temperature0_deterministic_protocol_packages",
                "can_veto_terminal_pairwise_IUT": False, "training_authorized": False}
    raw_hindsight = sum(max(terminal_values[(str(example), policy)] for policy in policies)
                        for example in assigned) / len(assigned)
    if primary_mode == "F":
        if (manifest.get("seed_manifest_hindsight_envelope_only") is not True or
                manifest.get("example_heterogeneity_oracle_claim_authorized") is not False):
            return _invalid("F_hindsight_envelope_contract_failed")
        return {"status": "SEED_MANIFEST_HINDSIGHT_ENVELOPE_ONLY",
                "seed_manifest_hindsight_envelope": raw_hindsight,
                "V_fixed_star_single_seed": v_fixed,
                "selection_opportunity_claim_authorized": False,
                "example_heterogeneity_oracle_claim_authorized": False,
                "can_veto_terminal_pairwise_IUT": False, "training_authorized": False}
    if primary_mode != "S":
        return _invalid("primary_mode_must_be_D_S_or_F")
    s_required = {"raw_pointwise_max_is_example_heterogeneity_oracle": False,
                  "raw_pointwise_max_role": "coupling_dependent_hindsight_luck_envelope",
                  "assignment_evaluation_seed_folds_independent": True,
                  "plugin_max_winners_curse_acknowledged": True,
                  "within_example_MC_reported": True,
                  "stable_per_example_oracle_labels_authorized": False,
                  "audit16_K4_scope": "luck_envelope_coupling_MC_feasibility_only"}
    wrong = {key: (manifest.get(key), expected) for key, expected in s_required.items()
             if manifest.get(key) != expected}
    if wrong:
        return _invalid(f"S_oracle_contract_failed:{wrong}")
    k = manifest.get("K_per_fold")
    records = manifest.get("fold_records")
    if not isinstance(k, int) or k < 2 or not isinstance(records, list):
        return _invalid("K_per_fold_or_fold_records_invalid")
    seen = set(); seeds_by_fold = defaultdict(set); values = defaultdict(list)
    for index, row in enumerate(records):
        key = (str(row.get("stable_example_id")), str(row.get("policy")),
               str(row.get("fold")), row.get("replicate"))
        if (key[0] not in {str(item) for item in assigned} or key[1] not in policies or
                key[2] not in {"assignment", "evaluation"} or
                not isinstance(key[3], int) or not 0 <= key[3] < k or key in seen):
            return _invalid(f"fold_record_key_invalid_or_duplicate:{index}")
        seed = row.get("seed"); endpoint = row.get("official_endpoint_value")
        if not isinstance(seed, int) or not isinstance(endpoint, (int, float)) or not math.isfinite(float(endpoint)):
            return _invalid(f"fold_record_seed_or_endpoint_invalid:{index}")
        seen.add(key); seeds_by_fold[key[2]].add(seed); values[key[:3]].append(float(endpoint))
    required_keys = {(str(example), policy, fold, replicate) for example in assigned
                     for policy in policies for fold in ("assignment", "evaluation")
                     for replicate in range(k)}
    if seen != required_keys:
        return _invalid("fold_ledger_not_complete_example_x_policy_x_fold_x_K")
    if seeds_by_fold["assignment"] & seeds_by_fold["evaluation"]:
        return _invalid("assignment_evaluation_seed_folds_overlap")
    fold_means = {(example, policy, fold): sum(items) / k
                  for (example, policy, fold), items in values.items()}
    crossfit_values = []
    within_mc = []
    for example in map(str, assigned):
        assignment_winner = max(sorted(policies), key=lambda p: fold_means[(example, p, "assignment")])
        evaluation_winner = max(sorted(policies), key=lambda p: fold_means[(example, p, "evaluation")])
        crossfit_values.append((fold_means[(example, assignment_winner, "evaluation")] +
                                fold_means[(example, evaluation_winner, "assignment")]) / 2.0)
        for policy in policies:
            pooled = values[(example, policy, "assignment")] + values[(example, policy, "evaluation")]
            mean = sum(pooled) / len(pooled)
            variance = sum((item - mean) ** 2 for item in pooled) / max(1, len(pooled) - 1)
            within_mc.append({"stable_example_id": example, "policy": policy,
                              "replicates": len(pooled), "within_example_MC_variance": variance})
    return {"status": "STOCHASTIC_CONDITIONAL_MEAN_ORACLE_MC_FEASIBILITY_ONLY",
            "raw_hindsight_luck_envelope": raw_hindsight,
            "crossfit_conditional_mean_oracle_estimate": sum(crossfit_values) / len(crossfit_values),
            "best_fixed_policy_value_from_execution_ledger": v_fixed,
            "within_example_MC": within_mc,
            "stable_per_example_oracle_labels_authorized": False,
            "audit16_K4_scope": "luck_envelope_coupling_MC_feasibility_only" if k == 4
                                else "finite_replication_crossfit_MC_audit",
            "seed_or_replicate_increases_n": False,
            "can_veto_terminal_pairwise_IUT": False, "training_authorized": False}
