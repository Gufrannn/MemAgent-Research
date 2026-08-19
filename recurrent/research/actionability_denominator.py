"""Shared fail-closed denominator semantics for multi-write actionability audits."""
from __future__ import annotations

from typing import Any

FORBIDDEN_RAW_POOL_KEYS = {
    "raw_pool_harmful_commit_probability", "raw_pool_beneficial_rejection_probability",
    "deployment_harm_probability", "deployment_risk", "raw_pool_policy_value",
}


def raw_pool_event_bound(ledger: dict[str, Any] | None) -> dict[str, Any]:
    if ledger is None:
        return {"status": "not_available_without_complete_target_to_R1_to_pair_ledger"}
    required_true = ("complete_target_to_r1_to_pair_ledger",
                     "raw_target_includes_r0",
                     "raw_target_includes_construct_failures",
                     "raw_target_includes_unpaired_rows")
    if any(ledger.get(key) is not True for key in required_true):
        raise ValueError("RAW_POOL_DENOMINATOR_FALSE_CLAIM: incomplete raw target attrition ledger")
    n_raw = ledger.get("N_raw")
    observed = ledger.get("M_obs")
    missing = ledger.get("M_miss")
    if (not all(isinstance(value, int) for value in (n_raw, observed, missing)) or n_raw <= 0 or
            observed < 0 or missing < 0 or observed + missing > n_raw):
        raise ValueError("RAW_POOL_DENOMINATOR_FALSE_CLAIM: invalid N_raw/M_obs/M_miss counts")
    return {"status": "worst_case_selection_bound_only", "event_range": [0, 1],
            "lower": observed / n_raw, "upper": (observed + missing) / n_raw,
            "formula": "[M_obs/N_raw,(M_obs+M_miss)/N_raw]",
            "raw_pool_policy_value_identified": False}


def validate_output_claims(report: dict[str, Any]) -> bool:
    forbidden = sorted(key for key in FORBIDDEN_RAW_POOL_KEYS if key in report and report[key] is not None)
    if forbidden:
        raise ValueError(f"RAW_POOL_DENOMINATOR_FALSE_CLAIM: forbidden fields={forbidden}")
    required = ("eligible_target_harmful_commit_probability",
                "eligible_target_beneficial_rejection_probability", "raw_pool_probability_identified")
    missing = [key for key in required if key not in report]
    if missing or report.get("raw_pool_probability_identified") is not False:
        raise ValueError(f"RAW_POOL_DENOMINATOR_FALSE_CLAIM: missing/invalid required labels={missing}")
    if report.get("raw_pool_policy_value_identified") is not False or report.get("raw_pool_policy_value") is not None:
        raise ValueError("RAW_POOL_DENOMINATOR_FALSE_CLAIM: raw policy value must remain undefined")
    return True
