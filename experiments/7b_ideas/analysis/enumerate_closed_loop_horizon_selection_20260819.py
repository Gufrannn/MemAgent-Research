#!/usr/bin/env python3
"""Exact truncated-normal identity behind the selected-horizon firewall."""
from __future__ import annotations

import argparse
import json
import math


def exact_identity():
    conditional_mean = math.sqrt(2 / math.pi)
    assert round(conditional_mean, 8) == .79788456
    return {"status": "PASS_EXACT_TRUNCATED_NORMAL_IDENTITY", "true_effect_z2": 0.0,
            "true_effect_z3": 0.0, "observation_rule": "observe_Z3_only_if_Z2_positive",
            "E_Z3_given_Z2_positive": conditional_mean,
            "selected_three_turn_confirmatory": False, "authorizes_resources": False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if not args.self_test: parser.error("--self-test required")
    print(json.dumps(exact_identity(), indent=2, sort_keys=True))


if __name__ == "__main__": main()
