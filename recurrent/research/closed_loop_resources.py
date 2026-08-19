"""Closed-loop v8 resource-mode qualification audit."""
from __future__ import annotations

import math
import re
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")
COUNTS = ("cumulative_calls", "input_tokens", "generated_tokens", "context_tokens", "walltime_seconds")


def _result(status: str, reason: str | None = None) -> dict[str, Any]:
    result = {"status": status, "resource_claim_authorized": False,
              "raw_QA_terminal_IUT_preserved_as_unconstrained_description": True,
              "fixed_budget_policy_value_authorized": False,
              "training_authorized": False, "new_rollouts": False}
    if reason: result["reason"] = reason
    return result


def audit_resource_mode(manifest: dict[str, Any], *, terminal_values: dict[tuple[str, str], float],
                        assigned: list[str], policies: list[str]) -> dict[str, Any]:
    required = {"schema_version": "closed-loop-resource-mode-v1",
                "mode_frozen_before_outcome": True,
                "raw_QA_IUT_equals_equal_budget_or_practical_advantage": False,
                "writer_proposal_parity_solves_certificate_or_memory_cost_parity": False,
                "posthoc_lambda_sweep": False, "gain_per_token_only_reporting": False,
                "optimizer_steps": 0, "new_rollouts": False}
    wrong = {key: (manifest.get(key), expected) for key, expected in required.items()
             if manifest.get(key) != expected}
    if wrong:
        return _result("COST_UNQUALIFIED", f"resource_contract_failed:{wrong}")
    mode = manifest.get("primary_resource_mode")
    expected = {(str(example), str(policy)) for example in assigned for policy in policies}
    if set(terminal_values) != expected:
        status = "FIXED_BUDGET_POLICY_VALUE_INVALID" if mode == "B" else "COST_UNQUALIFIED"
        return _result(status, "terminal_ledger_not_complete_example_x_policy")
    rows = manifest.get("resource_ledger")
    if not isinstance(rows, list):
        status = "FIXED_BUDGET_POLICY_VALUE_INVALID" if mode == "B" else "COST_UNQUALIFIED"
        return _result(status, "resource_ledger_missing")
    seen = set(); normalized_costs: dict[tuple[str, str], float] = {}
    for index, row in enumerate(rows):
        key = (str(row.get("stable_example_id")), str(row.get("policy")))
        if key not in expected or key in seen:
            status = "FIXED_BUDGET_POLICY_VALUE_INVALID" if mode == "B" else "COST_UNQUALIFIED"
            return _result(status, f"resource_row_key_invalid_or_duplicate:{index}")
        if (any(not isinstance(row.get(field), (int, float)) or
                not math.isfinite(float(row[field])) or float(row[field]) < 0 for field in COUNTS) or
                not isinstance(row.get("fallback_used"), bool) or
                not isinstance(row.get("certificate_readout_calls"), int) or row["certificate_readout_calls"] < 0 or
                not isinstance(row.get("policy_induced_memory_tokens"), int) or row["policy_induced_memory_tokens"] < 0):
            status = "FIXED_BUDGET_POLICY_VALUE_INVALID" if mode == "B" else "COST_UNQUALIFIED"
            return _result(status, f"resource_accounting_incomplete:{index}")
        if "normalized_cost" in row:
            cost = row["normalized_cost"]
            if not isinstance(cost, (int, float)) or not math.isfinite(float(cost)) or cost < 0:
                return _result("COST_UNQUALIFIED", f"normalized_cost_invalid:{index}")
            normalized_costs[key] = float(cost)
        seen.add(key)
    if seen != expected:
        status = "FIXED_BUDGET_POLICY_VALUE_INVALID" if mode == "B" else "COST_UNQUALIFIED"
        return _result(status, "resource_ledger_not_complete_example_x_policy")
    utility_report = None
    if manifest.get("utility_reporting_requested") is True:
        lam = manifest.get("utility_lambda")
        if (not isinstance(lam, (int, float)) or not math.isfinite(float(lam)) or lam < 0 or
                manifest.get("utility_lambda_frozen_from_use_case_before_outcome") is not True or
                not SHA256.fullmatch(str(manifest.get("cost_normalization_hash", ""))) or
                set(normalized_costs) != expected):
            return _result("COST_UNQUALIFIED", "utility_lambda_normalization_or_cost_ledger_invalid")
        utilities = {key: terminal_values[key] - float(lam) * normalized_costs[key] for key in expected}
        utility_report = {policy: sum(utilities[(str(example), policy)] for example in assigned) / len(assigned)
                          for policy in policies}
    elif manifest.get("utility_reporting_requested") is not False:
        return _result("COST_UNQUALIFIED", "utility_reporting_requested_missing")
    if manifest.get("certificate_practical_gain_claim_requested") is True:
        matched = manifest.get("matched_certificate_control")
        required_control = {"same_calls": True, "same_token_envelope": True,
                            "control_type": "content_random_or_sham", "frozen_before_outcome": True}
        if not isinstance(matched, dict) or any(matched.get(k) != v for k, v in required_control.items()):
            return _result("COST_UNQUALIFIED", "matched_certificate_control_missing_or_invalid")
    elif manifest.get("certificate_practical_gain_claim_requested") is not False:
        return _result("COST_UNQUALIFIED", "certificate_practical_gain_claim_requested_missing")
    if mode == "A":
        if manifest.get("estimand") != "accuracy_first_current_protocol_raw_QA":
            return _result("COST_UNQUALIFIED", "accuracy_first_estimand_invalid")
        return {"status": "ACCURACY_FIRST_RESOURCE_LEDGER_QUALIFIED",
                "raw_QA_terminal_IUT_preserved": True,
                "equal_budget_or_practical_advantage_claim_authorized":
                    bool(manifest.get("certificate_practical_gain_claim_requested")),
                "utility_policy_values": utility_report,
                "resource_claim_authorized": True, "training_authorized": False, "new_rollouts": False}
    if mode != "B":
        return _result("COST_UNQUALIFIED", "primary_resource_mode_must_be_A_or_B")
    b_required = {"estimand": "fixed_budget_policy_value", "common_resource_vector_frozen": True,
                  "overbudget_examples_deleted": False, "overbudget_rule_frozen_before_outcome": True,
                  "unconstrained_outcome_role": "description_only_not_fixed_budget_value"}
    wrong = {key: (manifest.get(key), expected_value) for key, expected_value in b_required.items()
             if manifest.get(key) != expected_value}
    vector = manifest.get("common_resource_vector")
    if wrong or not isinstance(vector, dict) or any(
            not isinstance(vector.get(field), (int, float)) or vector[field] < 0 for field in COUNTS):
        return _result("FIXED_BUDGET_POLICY_VALUE_INVALID", f"fixed_budget_contract_failed:{wrong}")
    allowed_dispositions = {"within_budget", "prefrozen_truncation", "prefrozen_skip", "prefrozen_fallback"}
    if any(row.get("budget_disposition") not in allowed_dispositions or
           row.get("included_in_fixed_budget_value") is not True for row in rows):
        return _result("FIXED_BUDGET_POLICY_VALUE_INVALID", "overbudget_disposition_or_inclusion_invalid")
    for index, row in enumerate(rows):
        overbudget = any(float(row[field]) > float(vector[field]) for field in COUNTS)
        if overbudget == (row["budget_disposition"] == "within_budget"):
            return _result("FIXED_BUDGET_POLICY_VALUE_INVALID",
                           f"resource_vector_and_budget_disposition_inconsistent:{index}")
    return {"status": "FIXED_BUDGET_POLICY_VALUE_QUALIFIED", "resource_claim_authorized": True,
            "raw_QA_terminal_IUT_preserved": True, "utility_policy_values": utility_report,
            "training_authorized": False, "new_rollouts": False}
