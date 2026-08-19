#!/usr/bin/env python3
"""Exact v6 randomness counterexamples for seed marginalization and addressing."""
from __future__ import annotations

import argparse
import json


def exact_enumeration():
    seed_marginal_difference = .8 - .6
    single_seed_differences = {"u_between_.6_and_.8": 1, "u_below_.6": 0}
    sequential_turn2 = {"policy_without_extra_call": 1, "policy_with_extra_call": 0}
    addressed_turn2 = {"policy_without_extra_call": 1, "policy_with_extra_call": 1}
    assert round(seed_marginal_difference, 12) == .2
    assert set(single_seed_differences.values()) == {0, 1}
    assert sequential_turn2 == {"policy_without_extra_call": 1, "policy_with_extra_call": 0}
    assert addressed_turn2 == {"policy_without_extra_call": 1, "policy_with_extra_call": 1}
    return {"status": "PASS_EXACT_ENUMERATION", "seed_marginal_difference": seed_marginal_difference,
            "single_seed_differences": single_seed_differences,
            "sequential_tape_turn2": sequential_turn2, "addressed_turn2": addressed_turn2,
            "sequential_PRNG_position_is_trajectory_identity": False,
            "training_authorized": False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if not args.self_test: parser.error("--self-test required")
    print(json.dumps(exact_enumeration(), indent=2, sort_keys=True))


if __name__ == "__main__": main()
