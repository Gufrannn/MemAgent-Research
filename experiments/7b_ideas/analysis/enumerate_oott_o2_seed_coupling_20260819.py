#!/usr/bin/env python3
"""Exact Bernoulli coupling enumeration for the OOTT O2 firewall."""
from __future__ import annotations

import argparse
import json


def _variance(joint_11):
    p_a, p_b = .8, .6
    p_10 = p_a - joint_11
    p_01 = p_b - joint_11
    mean = p_10 - p_01
    second = p_10 + p_01
    return second - mean * mean


def enumerate_variances():
    values = {"comonotone": round(_variance(.6), 12), "independent": round(_variance(.8 * .6), 12),
              "countermonotone": round(_variance(.4), 12)}
    assert values == {"comonotone": .16, "independent": .4, "countermonotone": .56}
    return {"status": "PASS_EXACT_ENUMERATION", "marginal_A": .8, "marginal_B": .6,
            "mean_difference": .2, "difference_variance": values,
            "policy_marginal_estimand_unchanged": True, "authorizes_training": False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if not args.self_test: parser.error("--self-test required")
    print(json.dumps(enumerate_variances(), indent=2, sort_keys=True))


if __name__ == "__main__": main()
