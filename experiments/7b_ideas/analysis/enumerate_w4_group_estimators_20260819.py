#!/usr/bin/env python3
"""Exact CPU checks for W4 all-mean/LOO/control-variate identities."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.counterfactual_gradient_witness import group_estimators  # noqa: E402


def _close(left, right, atol=1e-12):
    return all(abs(a - b) <= atol for a, b in zip(left, right))


def enumerate_identities():
    checked = 0
    for n in range(2, 9):
        scores = [[((i + 1) * (j + 2) % 7 - 3) / 5 for j in range(3)] for i in range(n)]
        for rewards in itertools.product((0.0, 1.0), repeat=n):
            row = {"commit_returns": list(rewards), "noop_baseline_returns": [.25] * n,
                   "score_gradients": scores}
            estimates = group_estimators(row)
            assert _close(estimates["g_credit_debiased"], estimates["g_credit_loo"])
            assert estimates["including_self_expected_scale"] == (n - 1) / n
            assert estimates["including_self_debias_factor"] == n / (n - 1)
            checked += 1
    equal = group_estimators({"commit_returns": [2.0] * 4, "noop_baseline_returns": [0.0] * 4,
                              "score_gradients": [[1.0], [0.0], [0.0], [0.0]]})
    assert equal["g_credit_including_self_all_mean"] == [0.0]
    assert equal["g_cf_external_noop"] == [.5]
    return {"status": "W4_GROUP_ESTIMATOR_ENUMERATION_PASS", "enumerated_groups": checked,
            "n4_expected_scale": .75, "n4_debias_factor": 4 / 3,
            "debiased_all_mean_equals_loo_batchwise": True,
            "equal_reward_external_nonzero_is_finite_batch_score_noise": True,
            "authorizes_training": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("--self-test required")
    print(json.dumps(enumerate_identities(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
