#!/usr/bin/env python3
"""Certificate-gated commit/rollback v3 multi-write audit; no training authority."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.actionability_denominator import raw_pool_event_bound, validate_output_claims  # noqa: E402

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def quantile(values, probability):
    ordered = sorted(values); position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else (ordered[lower] * (upper - position) +
                                                   ordered[upper] * (position - lower))


def validate(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty_certificate_rows")
    checkpoints = {row.get("checkpoint_hash") for row in rows}
    if len(checkpoints) != 1 or not SHA256.fullmatch(str(next(iter(checkpoints), ""))):
        raise ValueError("analysis_requires_exactly_one_valid_checkpoint")
    seen = set(); by_example = defaultdict(list)
    for source in rows:
        row = dict(source)
        key = (str(row.get("stable_example_id")), str(row.get("write_id")), str(row.get("checkpoint_hash")))
        if None in (source.get("stable_example_id"), source.get("write_id")):
            raise ValueError("missing_decision_key")
        if key in seen:
            raise ValueError("duplicate_example_write_checkpoint_key")
        seen.add(key)
        count = row.get("eligible_write_count")
        if not isinstance(count, int) or count < 1:
            raise ValueError("invalid_eligible_write_count")
        if (row.get("pair_complete") is not True or row.get("pair_qualified") is not True or
                row.get("postbranch_missing") is not False):
            raise ValueError("certificate_primary_requires_100_percent_branch_closure")
        if not isinstance(row.get("certificate_commit"), bool):
            raise ValueError("certificate_decision_must_be_boolean")
        values = (row.get("y_commit"), row.get("y_rollback"))
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise ValueError("missing_or_nonfinite_potential_outcome")
        row["stable_example_id"] = str(row["stable_example_id"]); row["write_id"] = str(row["write_id"])
        row["row_weight"] = 1.0 / count
        row["commit_effect"] = float(row["y_commit"]) - float(row["y_rollback"])
        by_example[row["stable_example_id"]].append(row)
    for stable_id, example_rows in by_example.items():
        counts = {row["eligible_write_count"] for row in example_rows}
        if len(counts) != 1 or len(example_rows) != next(iter(counts)):
            raise ValueError(f"eligible_write_count_mismatch:{stable_id}")
        if abs(sum(row["row_weight"] for row in example_rows) - 1.0) > 1e-12:
            raise ValueError(f"eligible_weight_not_one:{stable_id}")
    return [row for stable_id in sorted(by_example) for row in by_example[stable_id]], len(by_example), checkpoints.pop()


def metrics(rows, raw_pool_ledger=None):
    rows, n_examples, checkpoint = validate(rows)
    denominator = float(n_examples)
    weighted = lambda fn: sum(row["row_weight"] * fn(row) for row in rows) / denominator
    commit = weighted(lambda row: float(row["y_commit"]))
    rollback = weighted(lambda row: float(row["y_rollback"]))
    best_constant = max(commit, rollback)
    oracle = weighted(lambda row: max(float(row["y_commit"]), float(row["y_rollback"])))
    value = weighted(lambda row: float(row["y_commit"] if row["certificate_commit"] else row["y_rollback"]))
    opportunity = oracle - best_constant; gain = value - best_constant
    if gain > opportunity + 1e-10:
        raise AssertionError("gain_exceeds_opportunity")
    regret = weighted(lambda row: max(float(row["y_commit"]), float(row["y_rollback"])) -
                      float(row["y_commit"] if row["certificate_commit"] else row["y_rollback"]))
    harm_probability = weighted(lambda row: float(row["certificate_commit"] and row["commit_effect"] < 0))
    benefit_probability = weighted(lambda row: float(not row["certificate_commit"] and row["commit_effect"] > 0))
    harm_mass = weighted(lambda row: max(-row["commit_effect"], 0.0) if row["certificate_commit"] else 0.0)
    benefit_mass = weighted(lambda row: max(row["commit_effect"], 0.0) if not row["certificate_commit"] else 0.0)
    commit_coverage = weighted(lambda row: float(row["certificate_commit"]))
    result = {"schema_version": "certificate-gated-commit-rollback-v3-multiwrite",
            "checkpoint_hash": checkpoint,
            "decision_key": ["stable_example_id", "write_id", "checkpoint_hash"],
            "target_population": "uniform_example_then_uniform_prefrozen_eligible_write",
            "independent_examples": n_examples, "eligible_writes": len(rows),
            "eligible_weight_coverage": 1.0, "postbranch_missing_weight": 0.0,
            "always_commit_value": commit, "always_rollback_value": rollback,
            "best_constant_value": best_constant, "rowwise_oracle_value": oracle,
            "certificate_value": value, "selection_opportunity": opportunity,
            "certificate_gain": gain, "effect_weighted_regret": regret,
            "certificate_commit_coverage": commit_coverage,
            "eligible_target_harmful_commit_probability": harm_probability,
            "eligible_target_beneficial_rejection_probability": benefit_probability,
            "eligible_target_harmful_commit_effect_mass": harm_mass,
            "eligible_target_beneficial_rejection_effect_mass": benefit_mass,
            "identified_population": "uniform_example_then_uniform_R1_eligible_write",
            "raw_pool_probability_identified": False,
            "raw_pool_event_selection_bound": raw_pool_event_bound(raw_pool_ledger),
            "raw_pool_policy_value_identified": False, "raw_pool_policy_value": None,
            "bootstrap_unit": "stable_example_id_full_write_cluster",
            "writes_increase_independent_n": False, "row_bootstrap_authorized": False,
            "within_example_potential_outcome_average_before_oracle": False,
            "training_authorized": False, "sequential_or_closed_loop_value_authorized": False}
    validate_output_claims(result)
    return result


def bootstrap(rows, reps, seed):
    valid, _, _ = validate(rows); grouped = defaultdict(list)
    for row in valid: grouped[row["stable_example_id"]].append(row)
    ids = sorted(grouped); rng = random.Random(seed); gains = []
    for _ in range(reps):
        sample = []
        for cluster_index in range(len(ids)):
            selected = rng.choice(ids)
            for row in grouped[selected]:
                copied = dict(row); copied["stable_example_id"] = f"{selected}__cluster{cluster_index}"
                sample.append(copied)
        gains.append(metrics(sample)["certificate_gain"])
    return {"reps": reps, "cluster_unit": "stable_example_id",
            "certificate_gain_ci95": [quantile(gains, .025), quantile(gains, .975)]}


def self_test():
    rows = []
    for example in ("e0", "e1"):
        rows.extend([
            {"stable_example_id": example, "write_id": "good", "checkpoint_hash": "a" * 64,
             "eligible_write_count": 2, "certificate_commit": True, "y_commit": 1., "y_rollback": 0.,
             "pair_complete": True, "pair_qualified": True, "postbranch_missing": False},
            {"stable_example_id": example, "write_id": "bad", "checkpoint_hash": "a" * 64,
             "eligible_write_count": 2, "certificate_commit": False, "y_commit": -1., "y_rollback": 0.,
             "pair_complete": True, "pair_qualified": True, "postbranch_missing": False},
        ])
    result = metrics(rows)
    assert result["selection_opportunity"] == .5 and result["certificate_gain"] == .5
    assert result["independent_examples"] == 2 and result["eligible_writes"] == 4
    assert bootstrap(rows, 8, 3)["cluster_unit"] == "stable_example_id"
    bad = [dict(row) for row in rows]; bad[0]["checkpoint_hash"] = "b" * 64
    try: metrics(bad)
    except ValueError as exc: assert "one_valid_checkpoint" in str(exc)
    else: raise AssertionError("multiple checkpoints accepted")
    print("certificate_gated_commit_rollback_v3_multiwrite_self_test=ok")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input"); parser.add_argument("--output")
    parser.add_argument("--raw-pool-ledger")
    parser.add_argument("--bootstrap-reps", type=int, default=2000); parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.input: parser.error("--input required")
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    raw_ledger = json.loads(Path(args.raw_pool_ledger).read_text()) if args.raw_pool_ledger else None
    result = metrics(rows, raw_pool_ledger=raw_ledger); result["bootstrap"] = bootstrap(rows, args.bootstrap_reps, args.seed)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output: Path(args.output).write_text(text)
    else: print(text, end="")


if __name__ == "__main__": main()
