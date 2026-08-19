#!/usr/bin/env python3
"""Exact 2D reparameterization counterexample for W4 v8 geometry."""
from __future__ import annotations

import argparse
import json
import math


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _cos(left, right):
    return _dot(left, right) / math.sqrt(_dot(left, left) * _dot(right, right))


def enumerate_counterexample():
    g = [1.0, 1.0]
    h = [2.0, -1.0]
    # theta'=A theta, A=diag(10,1); covectors transform by A^-T.
    g_prime = [.1, 1.0]
    h_prime = [.2, -1.0]
    euclidean_before = _cos(g, h)
    euclidean_after = _cos(g_prime, h_prime)
    # Empirical-Fisher inverse bilinear: M=.5I, M'=A M A^T=diag(50,.5).
    fisher_before = g[0] * .5 * h[0] + g[1] * .5 * h[1]
    fisher_after = g_prime[0] * 50 * h_prime[0] + g_prime[1] * .5 * h_prime[1]
    # Covector-tangent delivery pairing, delta' = A delta.
    delivery_before = _dot(g, [.1, 0.0])
    delivery_after = _dot(g_prime, [1.0, 0.0])
    assert round(euclidean_before, 3) == .316
    assert round(euclidean_after, 3) == -.956
    assert fisher_before == fisher_after == .5
    assert delivery_before == delivery_after == .1
    return {"status": "PASS_EXACT_LINEAR_ALGEBRA", "euclidean_cosine_before": euclidean_before,
            "euclidean_cosine_after": euclidean_after, "fisher_bilinear_before": fisher_before,
            "fisher_bilinear_after": fisher_after, "covector_tangent_before": delivery_before,
            "covector_tangent_after": delivery_after, "euclidean_role": "fixed_coordinate_secondary_only",
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
