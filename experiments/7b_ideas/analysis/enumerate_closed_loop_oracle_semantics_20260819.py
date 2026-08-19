#!/usr/bin/env python3
"""Exact complementary-outcome distinction between fixed and clairvoyant value."""
from __future__ import annotations

import argparse
import json


def exact_enumeration():
    outcomes = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
    fixed = {policy: sum(values) / len(values) for policy, values in outcomes.items()}
    v_fixed_star = max(fixed.values())
    v_clair = sum(max(outcomes[policy][index] for policy in outcomes) for index in range(2)) / 2
    assert v_fixed_star == .5 and v_clair == 1.0 and v_clair - v_fixed_star == .5
    return {"status": "PASS_EXACT_ENUMERATION", "V_fixed_star": v_fixed_star,
            "V_clair": v_clair, "Opportunity_package": v_clair - v_fixed_star,
            "fixed_policy_with_value_one_exists": False,
            "clairvoyant_selector_is_executable_fixed_policy": False,
            "selector_training_authorized": False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if not args.self_test: parser.error("--self-test required")
    print(json.dumps(exact_enumeration(), indent=2, sort_keys=True))


if __name__ == "__main__": main()
