#!/usr/bin/env python3
"""Shape A actionability v2: weighted multi-write, myopic, and analysis-only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.actionability_denominator import raw_pool_event_bound, validate_output_claims  # noqa: E402


CLAIM_NAME = "myopic eligible-write actionability under a fixed shared-suffix protocol"


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else (ordered[lower] * (upper - position) +
                                                   ordered[upper] * (position - lower))


def tie_key(row):
    payload = f"{row['stable_example_id']}\x1f{row['write_id']}".encode()
    return hashlib.sha256(payload).hexdigest()


def validate(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty_actionability_rows")
    seen = set()
    by_example = defaultdict(list)
    for source in rows:
        row = dict(source)
        stable_id = row.get("stable_example_id")
        write_id = row.get("write_id")
        if stable_id is None or write_id is None:
            raise ValueError("missing_stable_example_id_or_write_id")
        key = (str(stable_id), str(write_id))
        if key in seen:
            raise ValueError("duplicate_stable_example_id_write_id")
        seen.add(key)
        count = row.get("eligible_write_count")
        if not isinstance(count, int) or count < 1:
            raise ValueError("invalid_prefrozen_eligible_write_count")
        if (row.get("pair_complete") is not True or row.get("pair_qualified") is not True or
                row.get("postbranch_missing") is not False):
            raise ValueError("primary_actionability_requires_100_percent_branch_closure")
        values = (row.get("score"), row.get("y_factual"), row.get("y_noop"))
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise ValueError("missing_or_nonfinite_required_value")
        row["stable_example_id"] = str(stable_id)
        row["write_id"] = str(write_id)
        row["harm"] = float(row["y_noop"]) - float(row["y_factual"])
        row["row_weight"] = 1.0 / count
        by_example[row["stable_example_id"]].append(row)
    for stable_id, example_rows in by_example.items():
        counts = {row["eligible_write_count"] for row in example_rows}
        if len(counts) != 1 or len(example_rows) != next(iter(counts)):
            raise ValueError(f"eligible_write_denominator_or_closure_mismatch:{stable_id}")
        if abs(sum(row["row_weight"] for row in example_rows) - 1.0) > 1e-12:
            raise ValueError(f"example_weight_not_one:{stable_id}")
    return [row for stable_id in sorted(by_example) for row in by_example[stable_id]], len(by_example)


def validate_gate_ledger(value):
    required = {"schema_version": "paired-write-harm-gate-v2", "shapeA_v8_primary": "pass",
                "exact_paired_replay": "exact_noop_v2_qualified",
                "multi_write_estimand": "uniform_example_then_uniform_prefrozen_eligible_write",
                "branch_closure": 1.0, "threshold_frozen_outside_confirmation": True,
                "adds_rollout": False, "adds_training": False}
    wrong = {key: (value.get(key), expected) for key, expected in required.items()
             if value.get(key) != expected}
    hashes = ("shapeA_evidence_hash", "paired_replay_hash", "eligibility_manifest_hash",
              "threshold_manifest_hash")
    missing = [key for key in hashes if not value.get(key)]
    if wrong or missing:
        raise ValueError(f"secondary_gate_ledger_fail_closed: wrong={wrong}, missing_hashes={missing}")
    return True


def _weighted_mean(rows, field, n_examples):
    return sum(row["row_weight"] * float(row[field]) for row in rows) / n_examples


def metrics(rows, q_values=(.10, .25, .50), raw_pool_ledger=None):
    valid, n_examples = validate(rows)
    ranked = sorted(valid, key=lambda row: (-float(row["score"]), tie_key(row)))
    total_weight = float(n_examples)
    mean_harm = _weighted_mean(ranked, "harm", n_examples)
    factual = _weighted_mean(ranked, "y_factual", n_examples)
    noop = _weighted_mean(ranked, "y_noop", n_examples)
    best_constant = max(factual, noop)
    oracle = sum(row["row_weight"] * max(float(row["y_factual"]), float(row["y_noop"]))
                 for row in ranked) / n_examples
    opportunity = oracle - best_constant

    def at_k(k):
        selected = {(row["stable_example_id"], row["write_id"]) for row in ranked[:k]}
        selected_weight = sum(row["row_weight"] for row in ranked[:k])
        selected_harm = sum(row["row_weight"] * row["harm"] for row in ranked[:k])
        value = regret = captured_harm = rejected_benefit = 0.0
        harmful_commit_probability = beneficial_rejection_probability = 0.0
        harmful_commit_effect_mass = beneficial_rejection_effect_mass = 0.0
        for row in ranked:
            use_noop = (row["stable_example_id"], row["write_id"]) in selected
            chosen = float(row["y_noop"] if use_noop else row["y_factual"])
            weight = row["row_weight"]
            value += weight * chosen
            regret += weight * (max(float(row["y_factual"]), float(row["y_noop"])) - chosen)
            if use_noop:
                captured_harm += weight * max(row["harm"], 0.0)
                rejected_benefit += weight * max(-row["harm"], 0.0)
                beneficial_rejection_probability += weight * float(row["harm"] < 0)
                beneficial_rejection_effect_mass += weight * max(-row["harm"], 0.0)
            else:
                harmful_commit_probability += weight * float(row["harm"] > 0)
                harmful_commit_effect_mass += weight * max(row["harm"], 0.0)
        value /= n_examples
        gain = value - best_constant
        if gain > opportunity + 1e-10:
            raise AssertionError("gain_exceeds_opportunity")
        return {"selected_write_count": k, "selected_eligible_weight": selected_weight,
                "q_realized": selected_weight / total_weight,
                "prioritized_harm": selected_harm / selected_weight - mean_harm,
                "myopic_eligible_write_value": value, "gain_vs_best_constant": gain,
                "capture_fraction": None if opportunity <= 1e-12 else gain / opportunity,
                "effect_weighted_regret": regret / n_examples,
                "captured_harm_mass": captured_harm / n_examples,
                "wrongly_rejected_benefit_mass": rejected_benefit / n_examples,
                "eligible_target_harmful_commit_probability": harmful_commit_probability / n_examples,
                "eligible_target_beneficial_rejection_probability": beneficial_rejection_probability / n_examples,
                "eligible_target_harmful_commit_effect_mass": harmful_commit_effect_mass / n_examples,
                "eligible_target_beneficial_rejection_effect_mass": beneficial_rejection_effect_mass / n_examples}

    points = [at_k(k) for k in range(1, len(ranked) + 1)]
    previous_q = 0.0
    auphc = 0.0
    for point in points:
        auphc += (point["q_realized"] - previous_q) * point["prioritized_harm"]
        previous_q = point["q_realized"]

    def at_q(q):
        target = q * total_weight
        cumulative = 0.0
        for index, row in enumerate(ranked, 1):
            cumulative += row["row_weight"]
            if cumulative + 1e-12 >= target:
                return at_k(index)
        return at_k(len(ranked))

    curve = {str(q): at_q(q) for q in q_values}
    result = {"schema_version": "paired-write-harm-prioritization-v2", "claim_name": CLAIM_NAME,
            "decision_key": ["stable_example_id", "write_id"],
            "target_population": "uniform_example_then_uniform_prefrozen_eligible_write",
            "independent_examples": n_examples, "eligible_writes": len(ranked),
            "eligible_weight_coverage": 1.0, "postbranch_missing_weight": 0.0,
            "mean_commit_harm": mean_harm, "always_factual_value": factual,
            "always_noop_value": noop, "best_constant_value": best_constant,
            "rowwise_oracle_value": oracle, "selection_opportunity": opportunity,
            "AUPHC_weight_integral": auphc,
            "curve": curve,
            "eligible_target_harmful_commit_probability": {
                q: point["eligible_target_harmful_commit_probability"] for q, point in curve.items()},
            "eligible_target_beneficial_rejection_probability": {
                q: point["eligible_target_beneficial_rejection_probability"] for q, point in curve.items()},
            "identified_population": "uniform_example_then_uniform_R1_eligible_write",
            "raw_pool_probability_identified": False,
            "raw_pool_event_selection_bound": raw_pool_event_bound(raw_pool_ledger),
            "raw_pool_policy_value_identified": False, "raw_pool_policy_value": None,
            "tie_break": "outcome_blind_sha256_composite_example_write_key",
            "bootstrap_unit": "stable_example_id_full_write_cluster",
            "per_write_iid_authorized": False, "sequential_or_closed_loop_value_authorized": False,
            "secondary_only": True, "adds_rollout": False, "adds_training": False,
            "training_authorized": False,
            "privileged_score_claim_ceiling": "offline_myopic_actionability_ceiling"}
    validate_output_claims(result)
    return result


def bootstrap(rows, reps, seed):
    valid, _ = validate(rows)
    by_example = defaultdict(list)
    for row in valid:
        by_example[row["stable_example_id"]].append(row)
    ids = sorted(by_example)
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        sample = []
        for cluster_index in range(len(ids)):
            selected = rng.choice(ids)
            for row in by_example[selected]:
                copied = dict(row)
                copied["stable_example_id"] = f"{selected}__cluster{cluster_index}"
                sample.append(copied)
        values.append(metrics(sample)["AUPHC_weight_integral"])
    return {"reps": reps, "cluster_unit": "stable_example_id",
            "AUPHC_ci95": [quantile(values, .025), quantile(values, .975)]}


def self_test():
    counterexample = [
        {"stable_example_id": "e0", "write_id": "w0", "eligible_write_count": 2, "score": 1.,
         "y_factual": 0., "y_noop": 1., "pair_complete": True, "pair_qualified": True,
         "postbranch_missing": False},
        {"stable_example_id": "e0", "write_id": "w1", "eligible_write_count": 2, "score": -1.,
         "y_factual": 0., "y_noop": -1., "pair_complete": True, "pair_qualified": True,
         "postbranch_missing": False},
    ]
    result = metrics(counterexample)
    assert result["mean_commit_harm"] == 0.0
    assert result["selection_opportunity"] == .5
    assert result["curve"]["0.5"]["gain_vs_best_constant"] == .5
    assert bootstrap(counterexample, 8, 7)["cluster_unit"] == "stable_example_id"
    bad = [dict(row) for row in counterexample]
    bad[0]["postbranch_missing"] = True
    try:
        metrics(bad)
    except ValueError as exc:
        assert "100_percent_branch_closure" in str(exc)
    else:
        raise AssertionError("postbranch missing row was accepted")
    ledger = {"schema_version": "paired-write-harm-gate-v2", "shapeA_v8_primary": "pass",
              "shapeA_evidence_hash": "a", "exact_paired_replay": "exact_noop_v2_qualified",
              "paired_replay_hash": "b", "eligibility_manifest_hash": "c",
              "multi_write_estimand": "uniform_example_then_uniform_prefrozen_eligible_write",
              "branch_closure": 1.0, "threshold_frozen_outside_confirmation": True,
              "threshold_manifest_hash": "d", "adds_rollout": False, "adds_training": False}
    assert validate_gate_ledger(ledger)
    print("paired_write_harm_prioritization_v2_self_test=ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input"); parser.add_argument("--output"); parser.add_argument("--gate-ledger")
    parser.add_argument("--raw-pool-ledger")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260819); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not args.input or not args.gate_ledger:
        parser.error("--input and --gate-ledger required")
    validate_gate_ledger(json.loads(Path(args.gate_ledger).read_text()))
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    raw_ledger = json.loads(Path(args.raw_pool_ledger).read_text()) if args.raw_pool_ledger else None
    result = metrics(rows, raw_pool_ledger=raw_ledger); result["bootstrap"] = bootstrap(rows, args.bootstrap_reps, args.seed)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
