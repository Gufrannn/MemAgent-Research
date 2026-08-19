#!/usr/bin/env python3
"""Validate that actionability reports do not overclaim raw-pool risk."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.actionability_denominator import validate_output_claims  # noqa: E402


def validate(value):
    validate_output_claims(value)
    return {"status": "ACTIONABILITY_DENOMINATOR_LABELS_PASS",
            "identified_population": "uniform_example_then_uniform_R1_eligible_write",
            "raw_pool_probability_identified": False, "raw_pool_policy_value_identified": False,
            "training_authorized": False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--report", type=Path, required=True); args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.report.read_text())), indent=2, sort_keys=True))


if __name__ == "__main__": main()
