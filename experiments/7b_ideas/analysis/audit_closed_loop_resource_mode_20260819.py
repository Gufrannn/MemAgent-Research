#!/usr/bin/env python3
"""CPU-only closed-loop v8 resource-mode audit."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.closed_loop_resources import COUNTS, audit_resource_mode  # noqa: E402


def _rows(examples: list[str], policies: list[str]) -> list[dict]:
    return [{"stable_example_id": example, "policy": policy, "cumulative_calls": 2,
             "input_tokens": 10, "generated_tokens": 5, "context_tokens": 15,
             "walltime_seconds": .2, "fallback_used": False,
             "certificate_readout_calls": 1 if policy == "GC" else 0,
             "policy_induced_memory_tokens": 3 if policy == "GC" else 0,
             "normalized_cost": 2.0 if policy == "GC" else 0.0}
            for example in examples for policy in policies]


def _base(mode: str, rows: list[dict]) -> dict:
    return {"schema_version": "closed-loop-resource-mode-v1", "primary_resource_mode": mode,
            "mode_frozen_before_outcome": True,
            "raw_QA_IUT_equals_equal_budget_or_practical_advantage": False,
            "writer_proposal_parity_solves_certificate_or_memory_cost_parity": False,
            "utility_reporting_requested": False, "posthoc_lambda_sweep": False,
            "gain_per_token_only_reporting": False,
            "certificate_practical_gain_claim_requested": False,
            "resource_ledger": rows, "optimizer_steps": 0, "new_rollouts": False}


def self_test() -> None:
    examples = ["e0"]; policies = ["GC", "GF"]
    terminal = {("e0", "GC"): .70, ("e0", "GF"): .68}; rows = _rows(examples, policies)
    a = {**_base("A", rows), "estimand": "accuracy_first_current_protocol_raw_QA"}
    assert audit_resource_mode(a, terminal_values=terminal, assigned=examples,
                               policies=policies)["status"] == "ACCURACY_FIRST_RESOURCE_LEDGER_QUALIFIED"
    utility = {**a, "utility_reporting_requested": True, "utility_lambda": .02,
               "utility_lambda_frozen_from_use_case_before_outcome": True,
               "cost_normalization_hash": "a" * 64}
    report = audit_resource_mode(utility, terminal_values=terminal, assigned=examples, policies=policies)
    assert math.isclose(report["utility_policy_values"]["GC"], .66)
    assert math.isclose(report["utility_policy_values"]["GF"], .68)
    bad = json.loads(json.dumps(a)); del bad["resource_ledger"][0]["context_tokens"]
    assert audit_resource_mode(bad, terminal_values=terminal, assigned=examples,
                               policies=policies)["status"] == "COST_UNQUALIFIED"
    b_rows = _rows(examples, policies)
    for row in b_rows: row.update({"budget_disposition": "within_budget",
                                   "included_in_fixed_budget_value": True})
    b = {**_base("B", b_rows), "estimand": "fixed_budget_policy_value",
         "common_resource_vector_frozen": True, "overbudget_examples_deleted": False,
         "overbudget_rule_frozen_before_outcome": True,
         "unconstrained_outcome_role": "description_only_not_fixed_budget_value",
         "common_resource_vector": {field: 100 for field in COUNTS}}
    assert audit_resource_mode(b, terminal_values=terminal, assigned=examples,
                               policies=policies)["status"] == "FIXED_BUDGET_POLICY_VALUE_QUALIFIED"
    b["resource_ledger"][0]["included_in_fixed_budget_value"] = False
    assert audit_resource_mode(b, terminal_values=terminal, assigned=examples,
                               policies=policies)["status"] == "FIXED_BUDGET_POLICY_VALUE_INVALID"
    print("closed_loop_resource_mode_self_test=ok")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path)
    parser.add_argument("--terminal-ledger", type=Path); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.manifest or not args.terminal_ledger: parser.error("--manifest and --terminal-ledger required")
    manifest = json.loads(args.manifest.read_text()); rows = json.loads(args.terminal_ledger.read_text())
    terminal = {(str(row["stable_example_id"]), str(row["policy"])): float(row["official_endpoint_value"])
                for row in rows}
    assigned = sorted({key[0] for key in terminal}); policies = sorted({key[1] for key in terminal})
    print(json.dumps(audit_resource_mode(manifest, terminal_values=terminal,
      assigned=assigned, policies=policies), indent=2, sort_keys=True))


if __name__ == "__main__": main()
