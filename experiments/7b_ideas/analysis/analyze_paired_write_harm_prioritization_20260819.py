#!/usr/bin/env python3
"""Secondary paired write-harm ranking/actionability audit; no training authority."""
import argparse, hashlib, json, math, random
from pathlib import Path

def mean(xs): return sum(xs) / len(xs)
def quantile(xs, p):
    ys = sorted(xs)
    if not ys: return float("nan")
    x = (len(ys) - 1) * p; lo, hi = math.floor(x), math.ceil(x)
    return ys[lo] if lo == hi else ys[lo] * (hi - x) + ys[hi] * (x - lo)
def tie_key(row): return hashlib.sha256(str(row["stable_example_id"]).encode()).hexdigest()

def validate(rows):
    if len(rows) < 4: raise ValueError("need_at_least_four_independent_examples")
    ids = [str(r.get("stable_example_id")) for r in rows]
    if len(set(ids)) != len(ids): raise ValueError("duplicate_stable_example_id")
    out = []
    for row in rows:
        if row.get("valid") is not True: continue
        values = [row.get("score"), row.get("y_factual"), row.get("y_noop")]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("missing_or_nonfinite_required_value")
        item = dict(row); item["harm"] = float(item["y_noop"]) - float(item["y_factual"]); out.append(item)
    if len(out) < 4: raise ValueError("fewer_than_four_valid_examples")
    return out

def validate_gate_ledger(value):
    required = {"schema_version": "paired-write-harm-gate-v1", "shapeA_v8_primary": "pass",
      "exact_paired_replay": "exact_noop_v2_qualified", "threshold_frozen_outside_confirmation": True,
      "adds_rollout": False, "adds_training": False}
    wrong = {key: (value.get(key), expected) for key, expected in required.items() if value.get(key) != expected}
    hashes = ("shapeA_evidence_hash", "paired_replay_hash", "threshold_manifest_hash")
    missing_hashes = [key for key in hashes if not value.get(key)]
    if wrong or missing_hashes: raise ValueError(f"secondary_gate_ledger_fail_closed: wrong={wrong}, missing_hashes={missing_hashes}")
    return True

def metrics(rows, q_values=(0.10, 0.25, 0.50)):
    ranked = sorted(validate(rows), key=lambda r: (-float(r["score"]), tie_key(r)))
    n = len(ranked); harms = [r["harm"] for r in ranked]
    yf = [float(r["y_factual"]) for r in ranked]; yn = [float(r["y_noop"]) for r in ranked]
    base_h = mean(harms); vf, vn = mean(yf), mean(yn); best_const = max(vf, vn)
    oracle = mean([max(a, b) for a, b in zip(yf, yn)]); opportunity = oracle - best_const
    def at_k(k):
        selected = {str(r["stable_example_id"]) for r in ranked[:k]}
        values, regrets = [], []; captured_harm = rejected_benefit = 0.0
        for row in ranked:
            use_noop = str(row["stable_example_id"]) in selected
            value = float(row["y_noop"] if use_noop else row["y_factual"]); values.append(value)
            regrets.append(max(float(row["y_factual"]), float(row["y_noop"])) - value)
            if use_noop:
                captured_harm += max(row["harm"], 0.0); rejected_benefit += max(-row["harm"], 0.0)
        value = mean(values); gain = value - best_const
        if gain > opportunity + 1e-10: raise AssertionError("gain_exceeds_opportunity")
        return {"k": k, "q_realized": k / n, "prioritized_harm": mean(harms[:k]) - base_h,
          "policy_value": value, "gain_vs_best_constant": gain,
          "capture_fraction": None if opportunity <= 1e-12 else gain / opportunity,
          "effect_weighted_regret": mean(regrets), "captured_harm_mass": captured_harm / n,
          "wrongly_rejected_benefit_mass": rejected_benefit / n}
    full_curve = [at_k(k) for k in range(1, n)]
    return {"n_valid": n, "mean_commit_harm": base_h, "always_factual_value": vf,
      "always_noop_value": vn, "best_constant_value": best_const, "oracle_value": oracle,
      "selection_opportunity": opportunity, "AUPHC": mean([x["prioritized_harm"] for x in full_curve]),
      "curve": {str(q): at_k(min(n - 1, max(1, math.ceil(q * n)))) for q in q_values},
      "secondary_only": True, "adds_rollout": False, "adds_training": False,
      "training_authorized": False, "online_or_sequential_safety_authorized": False,
      "privileged_score_claim_ceiling": "offline_actionability_ceiling"}

def bootstrap(rows, reps, seed):
    valid = validate(rows); rng = random.Random(seed); values = []
    for _ in range(reps):
        sample = []
        for j in range(len(valid)):
            row = dict(rng.choice(valid)); row["stable_example_id"] = f'{row["stable_example_id"]}__boot{j}'; sample.append(row)
        values.append(metrics(sample)["AUPHC"])
    return {"reps": reps, "AUPHC_ci95": [quantile(values, .025), quantile(values, .975)]}

def self_test():
    rows = [{"stable_example_id": f"e{i}", "score": h, "y_factual": 0.0, "y_noop": h, "valid": True}
            for i, h in enumerate([1.0, .8, .4, .1, -.1, -.4, -.8, -1.0])]
    result = metrics(rows); assert result["AUPHC"] > 0 and result["selection_opportunity"] > 0
    assert metrics([dict(r, score=-r["score"]) for r in rows])["AUPHC"] < 0
    try: metrics(rows + [dict(rows[0])])
    except ValueError as exc: assert "duplicate" in str(exc)
    else: raise AssertionError("duplicate_id_not_rejected")
    ledger = {"schema_version": "paired-write-harm-gate-v1", "shapeA_v8_primary": "pass",
      "shapeA_evidence_hash": "a", "exact_paired_replay": "exact_noop_v2_qualified", "paired_replay_hash": "b",
      "threshold_frozen_outside_confirmation": True, "threshold_manifest_hash": "c",
      "adds_rollout": False, "adds_training": False}
    assert validate_gate_ledger(ledger)
    try: validate_gate_ledger({**ledger, "shapeA_v8_primary": "not_run"})
    except ValueError as exc: assert "fail_closed" in str(exc)
    else: raise AssertionError("invalid_secondary_gate_ledger_accepted")
    print("paired_write_harm_prioritization_self_test=ok")

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input"); parser.add_argument("--output")
    parser.add_argument("--gate-ledger"); parser.add_argument("--bootstrap-reps", type=int, default=2000); parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.input: parser.error("--input required")
    if not args.gate_ledger: parser.error("--gate-ledger required")
    validate_gate_ledger(json.loads(Path(args.gate_ledger).read_text()))
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    result = metrics(rows); result["bootstrap"] = bootstrap(rows, args.bootstrap_reps, args.seed)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output: Path(args.output).write_text(text)
    else: print(text, end="")
if __name__ == "__main__": main()
