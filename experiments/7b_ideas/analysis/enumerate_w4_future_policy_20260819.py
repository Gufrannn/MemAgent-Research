#!/usr/bin/env python3
"""Exact W4 v8 counterexample for omitted tied future-policy score nodes."""
from __future__ import annotations

import argparse
import itertools
import json


def enumerate_counterexample():
    current = 0.0
    future = 0.0
    for a0, a1 in itertools.product((0.0, 1.0), repeat=2):
        probability = .25
        reward = a0 * a1
        current += probability * reward * (a0 - .5)
        future += probability * reward * (a1 - .5)
    assert current == .125
    assert future == .125
    assert current + future == .25
    return {"status": "W4_FUTURE_POLICY_ENUMERATION_PASS", "current_score_term": current,
            "future_score_term": future, "full_tied_recurrent_gradient": current + future,
            "current_only_fraction": .5,
            "current_only_label": "local_recurrent_semi_gradient",
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
