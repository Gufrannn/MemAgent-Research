#!/usr/bin/env python3
"""Exact two-action counterexample for W4 endpoint-target mismatch."""
from __future__ import annotations

import argparse
import json


def enumerate_counterexample():
    candidates = (0.0, 1.0)
    probabilities = (.5, .5)
    scores = (-.5, .5)
    train = (2.0, 2.0)
    evaluation = tuple(2.0 + candidate for candidate in candidates)
    g_train = sum(p * reward * score for p, reward, score in zip(probabilities, train, scores))
    g_eval = sum(p * reward * score for p, reward, score in zip(probabilities, evaluation, scores))
    assert g_train == 0.0
    assert g_eval == .25
    return {"status": "W4_ENDPOINT_TARGET_ENUMERATION_PASS", "g_train": g_train, "g_eval": g_eval,
            "classification": "surrogate_objective_gradient_mismatch",
            "same_reward_credit_loss_claim_authorized": False, "exact_noop_remedy": False,
            "authorizes_training": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("--self-test required")
    print(json.dumps(enumerate_counterexample(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
