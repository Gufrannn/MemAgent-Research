#!/usr/bin/env python3
"""Exact v8 resource-mode sign-reversal counterexample."""
from __future__ import annotations

import json


def exact_example() -> dict:
    gc_y, control_y, gc_cost, control_cost, lam, budget = .70, .68, 2.0, 0.0, .02, 1.0
    return {"raw_GC_minus_control": gc_y - control_y,
            "GC_utility": gc_y - lam * gc_cost,
            "control_utility": control_y - lam * control_cost,
            "utility_prefers_control": gc_y - lam * gc_cost < control_y - lam * control_cost,
            "fixed_budget": budget, "GC_executable_under_fixed_budget": gc_cost <= budget}


if __name__ == "__main__": print(json.dumps(exact_example(), indent=2, sort_keys=True))
