#!/usr/bin/env python3
"""Exact Bernoulli counterexamples for closed-loop v7 oracle semantics."""
from __future__ import annotations

import itertools
import json


def exact_enumeration() -> dict:
    two = list(itertools.product((0, 1), repeat=2))
    four = list(itertools.product((0, 1), repeat=4))
    return {"two_identical_Bernoulli_half_best_fixed": .5,
            "two_identical_Bernoulli_half_conditional_mean_oracle": .5,
            "two_identical_Bernoulli_half_true_selection_opportunity": 0.0,
            "two_policy_independent_draw_pointwise_max": sum(max(row) for row in two) / len(two),
            "two_policy_common_random_draw_pointwise_max": .5,
            "four_policy_independent_draw_pointwise_max": sum(max(row) for row in four) / len(four),
            "raw_random_pointwise_max_is_coupling_dependent": True}


if __name__ == "__main__": print(json.dumps(exact_enumeration(), indent=2, sort_keys=True))
