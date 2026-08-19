#!/usr/bin/env python3
"""Exact scalar PPO-clip examples separating BQCC from gradient delivery."""
from __future__ import annotations

import json


def local_task_delivery(advantage: float, ratio: float, epsilon: float = .2) -> float:
    if advantage > 0 and ratio > 1 + epsilon:
        return 0.0
    if advantage < 0 and ratio < 1 - epsilon:
        return 0.0
    return ratio


def exact_examples() -> dict:
    return {"positive_A_ratio_1_3_delivery": local_task_delivery(1., 1.3),
            "negative_A_ratio_0_7_delivery": local_task_delivery(-1., .7),
            "unclipped_ratio_1_1_delivery": local_task_delivery(1., 1.1),
            "BQCC_pass_implies_gradient_delivery": False,
            "BQCC_failure_repaired_by_large_total_gradient": False}


if __name__ == "__main__": print(json.dumps(exact_examples(), indent=2, sort_keys=True))
