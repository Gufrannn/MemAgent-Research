#!/usr/bin/env python3
"""P29 learned static admission gate over frozen P27 generations.

This script is deliberately read-only with respect to generation artifacts.  It
does not call a model, change prompts, or recompute operation outputs.  It asks
whether legal online state features available before an admission/repack action
can predict when to keep the greedy admitted working memory (STOP) versus when
to use a fixed READMIT policy over the already retrieved candidate pool C0.

Leakage boundary:
    Default features are query text, first-retrieval/admitted trace statistics,
    and optionally the initial visible working-memory text W0.  Gold evidence,
    answers, judge labels, question_type, raw source-index statistics, and
    operation outcomes are never used as formal online features.
    ``--include-question-type`` and ``--include-source-index-features`` are
    diagnostic controls and are marked as such in the report.

Evaluation boundary:
    All models are evaluated by outer LOOCV.  Hyperparameters and thresholds
    are selected only inside the outer train fold.  Results are exploratory
    because the input matrix is dev80 + surrogate F1 unless an official judge
    matrix is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_qid(qid: str) -> str:
    qid = str(qid)
    return qid if qid.startswith("longmemeval_") else f"longmemeval_{qid}"


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def parse_float(value: Any) -> float:
    if value in {None, "", "nan", "NaN"}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else math.nan


def wtl(delta: float, eps: float) -> str:
    if math.isnan(delta):
        return "missing"
    if delta > eps:
        return "win"
    if delta < -eps:
        return "loss"
    return "tie"


def bool01(value: Any) -> int | None:
    number = parse_float(value)
    if math.isnan(number):
        return None
    return int(number >= 1.0)


def group_label(row: dict[str, str], c0_col: str) -> str:
    c0_complete = bool01(row.get(c0_col))
    w0_complete = bool01(row.get("stop_all_evidence_present"))
    if c0_complete is None:
        return "UNKNOWN_C0_RETRIEVAL_STATUS"
    if c0_complete == 0:
        return "A_retrieval_incomplete_gold_not_subset_C0"
    if w0_complete == 0:
        return "B_admission_incomplete_gold_in_C0_not_W0"
    if w0_complete == 1:
        return "C_admitted_complete_gold_subset_W0"
    return "UNKNOWN_W0_ADMISSION_STATUS"


def value(row: dict[str, str], policy: str, metric: str) -> float:
    if metric == "reward":
        return parse_float(row.get(f"{policy}_reward"))
    if metric == "proxy_utility_context":
        key = f"{policy}_proxy_utility_context"
        if key not in row:
            key = f"{policy}_utility"
        return parse_float(row.get(key))
    raise ValueError(f"unsupported metric: {metric}")


def cost(row: dict[str, str], policy: str) -> float:
    return parse_float(row.get(f"{policy}_cost"))


def load_responses(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in iter_jsonl(path):
        qid = normalize_qid(str(row.get("qid") or row.get("question_id") or ""))
        if not qid:
            continue
        if qid in out:
            duplicates.append(qid)
        out[qid] = row
    if duplicates:
        raise ValueError(f"duplicate qids in responses: {duplicates[:5]}")
    return out


def load_traces_by_query_sha1(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in iter_jsonl(path):
        if row.get("phase") != "qa":
            continue
        qhash = str(row.get("query_sha1") or "")
        if not qhash:
            continue
        if qhash in out:
            duplicates.append(qhash)
        out[qhash] = row
    if duplicates:
        raise ValueError(f"duplicate query_sha1 rows in trace: {duplicates[:5]}")
    return out


def first_op_record(trace_row: dict[str, Any], allowed_ops: set[str]) -> dict[str, Any] | None:
    for record in trace_row.get("op_records") or []:
        if record.get("operation") in allowed_ops:
            return record
    return None


def numeric_summary(values: list[float], prefix: str) -> dict[str, float]:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_span": 0.0,
        }
    return {
        f"{prefix}_count": float(len(clean)),
        f"{prefix}_min": min(clean),
        f"{prefix}_max": max(clean),
        f"{prefix}_mean": sum(clean) / len(clean),
        f"{prefix}_span": max(clean) - min(clean),
    }


def add_hashed_tokens(vec: np.ndarray, text: str, prefix: str, weight: float) -> None:
    counts = Counter(tokens(text))
    for tok, count in counts.items():
        key = f"{prefix}:{tok}"
        h = stable_hash(key)
        idx = h % len(vec)
        sign = 1.0 if ((h >> 63) & 1) else -1.0
        vec[idx] += sign * weight * math.log1p(count)


def build_online_features(
    qids: list[str],
    wide_rows: dict[str, dict[str, str]],
    responses: dict[str, dict[str, Any]],
    traces_by_query: dict[str, dict[str, Any]],
    *,
    feature_set: str,
    hash_dim: int,
    include_question_type: bool,
    include_source_index_features: bool,
    allow_missing_trace: bool,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    numeric_dicts: list[dict[str, float]] = []
    hashed_rows: list[np.ndarray] = []
    feature_audit: list[dict[str, Any]] = []
    text_enabled = feature_set in {"text", "stats_text"}
    stats_enabled = feature_set in {"stats", "stats_text"}

    for qid in qids:
        response = responses.get(qid)
        if response is None:
            raise ValueError(f"missing stop response for qid={qid}")
        query = str(response.get("query") or "")
        qhash = sha1_text(query)
        trace_row = traces_by_query.get(qhash)
        if trace_row is None and not allow_missing_trace:
            raise ValueError(f"missing stop trace for qid={qid}, query_sha1={qhash}")

        retrieve = first_op_record(trace_row, {"RETRIEVE", "RETRIEVE_RECENT"}) if trace_row else None
        if retrieve is None and not allow_missing_trace:
            raise ValueError(f"missing first retrieve record for qid={qid}")

        retrieved_indices = list((retrieve or {}).get("retrieved_source_indices") or [])
        admitted_indices = list((retrieve or {}).get("admitted_source_indices") or [])
        context_chars = parse_float((retrieve or {}).get("context_chars"))
        budget_chars = parse_float((retrieve or {}).get("budget_chars"))
        if math.isnan(context_chars):
            context_chars = parse_float((trace_row or {}).get("final_context_chars"))
        if math.isnan(budget_chars):
            budget_chars = parse_float((trace_row or {}).get("budget_chars"))
        if math.isnan(context_chars):
            context_chars = 0.0
        if math.isnan(budget_chars) or budget_chars <= 0:
            budget_chars = 1.0

        q_toks = tokens(query)
        q_unique = len(set(q_toks))
        numeric: dict[str, float] = {
            "bias": 1.0,
            "query_chars": float(len(query)),
            "query_tokens": float(len(q_toks)),
            "query_unique_tokens": float(q_unique),
            "query_unique_ratio": float(q_unique / max(1, len(q_toks))),
            "query_digit_tokens": float(sum(1 for tok in q_toks if any(ch.isdigit() for ch in tok))),
            "query_question_marks": float(query.count("?")),
            "trace_found": float(trace_row is not None),
            "initial_context_kchars": context_chars / 1000.0,
            "budget_kchars": budget_chars / 1000.0,
            "context_budget_ratio": context_chars / max(1.0, budget_chars),
            "initial_n_retrieved_sources": float(len(retrieved_indices)),
            "initial_n_admitted_sources": float(len(admitted_indices)),
            "candidate_minus_admitted_sources": float(max(0, len(retrieved_indices) - len(admitted_indices))),
            "admitted_source_ratio": float(len(admitted_indices) / max(1, len(retrieved_indices))),
            "trace_top_k": parse_float((trace_row or {}).get("top_k")),
            "trace_total_memory_kchars": parse_float((trace_row or {}).get("total_memory_chars")) / 1000.0,
        }
        if math.isnan(numeric["trace_top_k"]):
            numeric["trace_top_k"] = 0.0
        if math.isnan(numeric["trace_total_memory_kchars"]):
            numeric["trace_total_memory_kchars"] = 0.0
        # Raw source indices are dataset-order artifacts, not semantic memory
        # state.  Exclude them from the formal online feature set by default.
        # If enabled, their names are prefixed as diagnostics so downstream
        # audits can separate them from deployable state features.
        if include_source_index_features:
            numeric.update(
                numeric_summary([float(x) for x in retrieved_indices], "diagnostic_source_index_retrieved")
            )
            numeric.update(
                numeric_summary([float(x) for x in admitted_indices], "diagnostic_source_index_admitted")
            )

        if include_question_type:
            # Privileged control only.  Wide matrix carries question_type from
            # benchmark metadata; default is false and reports mark true as
            # PRIVILEGED_CONTROL.
            qtype = str(wide_rows[qid].get("question_type") or "unknown")
            for idx in range(32):
                numeric[f"privileged_question_type_hash_{idx}"] = 0.0
            numeric[f"privileged_question_type_hash_{stable_hash(qtype) % 32}"] = 1.0

        hashed = np.zeros(hash_dim, dtype=float)
        state_text = str((retrieve or {}).get("state_text") or "")
        if text_enabled:
            add_hashed_tokens(hashed, query, "query", 1.0)
            if state_text:
                add_hashed_tokens(hashed, state_text, "w0", 0.5)
            if include_question_type:
                add_hashed_tokens(hashed, str(wide_rows[qid].get("question_type") or "unknown"), "privileged_qtype", 1.0)

        numeric_dicts.append(numeric)
        hashed_rows.append(hashed)
        feature_audit.append(
            {
                "qid": qid,
                "query_sha1": qhash,
                "trace_found": int(trace_row is not None),
                "first_retrieve_found": int(retrieve is not None),
                "feature_set": feature_set,
                "include_question_type": int(include_question_type),
                "include_source_index_features": int(include_source_index_features),
                "query_chars": len(query),
                "w0_state_text_available": int(bool(state_text)),
                "retrieved_source_count": len(retrieved_indices),
                "admitted_source_count": len(admitted_indices),
                "used_online_feature_families": (
                    "query_text;w0_state_text;trace_stats"
                    if feature_set == "stats_text"
                    else feature_set
                ),
                "forbidden_online_features": ";".join(
                    item
                    for item in [
                        "question_type_privileged_control" if include_question_type else "",
                        "raw_source_index_diagnostic" if include_source_index_features else "",
                    ]
                    if item
                )
                or "none",
            }
        )

    if stats_enabled:
        non_bias_names = sorted({name for row in numeric_dicts for name in row if name != "bias"})
        numeric_names = ["bias"] + non_bias_names
        numeric_matrix = [
            [row.get(name, 0.0) for name in numeric_names]
            for row in numeric_dicts
        ]
    else:
        numeric_names = ["bias"]
        numeric_matrix = [[1.0] for _ in numeric_dicts]
    rows = [numeric + hashed.tolist() for numeric, hashed in zip(numeric_matrix, hashed_rows)]
    feature_names = numeric_names + [f"hash_{i}" for i in range(hash_dim)]
    return np.asarray(rows, dtype=float), feature_names, feature_audit


def standardize_train_test(
    x: np.ndarray, train_idx: list[int], test_idx: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    train = x[train_idx].copy()
    test = x[test_idx].copy()
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    train = (train - mu) / sigma
    test = (test - mu) / sigma
    train[:, 0] = 1.0
    test[:, 0] = 1.0
    return train, test


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    lhs = x.T @ x + penalty
    rhs = x.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def kfold_indices(indices: list[int], folds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    folds = min(max(2, folds), len(shuffled))
    out = [[] for _ in range(folds)]
    for pos, idx in enumerate(shuffled):
        out[pos % folds].append(idx)
    return out


def choose_best_fixed(rows: list[dict[str, str]], policies: list[str], metric: str, tie_eps: float) -> str:
    means = {policy: mean([value(row, policy, metric) for row in rows]) for policy in policies}
    costs = {policy: mean([cost(row, policy) for row in rows]) for policy in policies}
    best_val = max(means.values())
    tied = [policy for policy in policies if best_val - means[policy] <= tie_eps]
    order = {policy: idx for idx, policy in enumerate(policies)}
    return min(tied, key=lambda policy: (costs[policy], order[policy]))


def choose_static_oracle(row: dict[str, str], policies: list[str], metric: str, tie_eps: float) -> str:
    vals = {policy: value(row, policy, metric) for policy in policies}
    best_val = max(vals.values())
    tied = [policy for policy in policies if best_val - vals[policy] <= tie_eps]
    order = {policy: idx for idx, policy in enumerate(policies)}
    return min(tied, key=lambda policy: (cost(row, policy), order[policy]))


def inner_binary_score(
    x: np.ndarray,
    y_delta: np.ndarray,
    rows: list[dict[str, str]],
    train_indices: list[int],
    *,
    readmit_policy: str,
    metric: str,
    alphas: list[float],
    thresholds: list[float],
    folds: int,
    seed: int,
) -> tuple[float, float]:
    best: tuple[float, float, float] | None = None
    split = kfold_indices(train_indices, folds, seed)
    for alpha in alphas:
        for threshold in thresholds:
            realized: list[float] = []
            for fold_id, val_idx in enumerate(split):
                fit_idx = [idx for idx in train_indices if idx not in set(val_idx)]
                x_fit, x_val = standardize_train_test(x, fit_idx, val_idx)
                beta = fit_ridge(x_fit, y_delta[fit_idx], alpha)
                pred = x_val @ beta
                for idx, pred_delta in zip(val_idx, pred):
                    chosen = readmit_policy if pred_delta > threshold else "stop"
                    realized.append(value(rows[idx], chosen, metric))
            score = mean(realized)
            candidate = (score, -abs(threshold), -alpha)
            if best is None or candidate > best:
                best = candidate
                best_alpha = alpha
                best_threshold = threshold
    return best_alpha, best_threshold


def run_binary_loocv(
    x: np.ndarray,
    qids: list[str],
    rows: list[dict[str, str]],
    *,
    readmit_policy: str,
    metric: str,
    alphas: list[float],
    thresholds: list[float],
    folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    y_delta = np.asarray([value(row, readmit_policy, metric) - value(row, "stop", metric) for row in rows])
    outputs: list[dict[str, Any]] = []
    all_indices = list(range(len(rows)))
    for test_idx in all_indices:
        train_idx = [idx for idx in all_indices if idx != test_idx]
        alpha, threshold = inner_binary_score(
            x,
            y_delta,
            rows,
            train_idx,
            readmit_policy=readmit_policy,
            metric=metric,
            alphas=alphas,
            thresholds=thresholds,
            folds=folds,
            seed=seed + test_idx,
        )
        x_train, x_test = standardize_train_test(x, train_idx, [test_idx])
        beta = fit_ridge(x_train, y_delta[train_idx], alpha)
        pred_delta = float((x_test @ beta)[0])
        chosen = readmit_policy if pred_delta > threshold else "stop"
        outputs.append(
            {
                "qid": qids[test_idx],
                "selector": f"binary_keep_vs_{readmit_policy}",
                "chosen_policy": chosen,
                "predicted_delta": pred_delta,
                "inner_alpha": alpha,
                "inner_threshold": threshold,
            }
        )
    return outputs


def choose_predicted_policy(
    predicted: dict[str, float],
    train_mean_costs: dict[str, float],
    policies: list[str],
    tie_eps: float,
) -> str:
    best = max(predicted.values())
    tied = [policy for policy in policies if best - predicted[policy] <= tie_eps]
    order = {policy: idx for idx, policy in enumerate(policies)}
    return min(tied, key=lambda policy: (train_mean_costs[policy], order[policy]))


def inner_multiclass_alpha(
    x: np.ndarray,
    rows: list[dict[str, str]],
    train_indices: list[int],
    *,
    policies: list[str],
    metric: str,
    alphas: list[float],
    folds: int,
    seed: int,
    tie_eps: float,
) -> float:
    best: tuple[float, float] | None = None
    split = kfold_indices(train_indices, folds, seed)
    for alpha in alphas:
        realized: list[float] = []
        for val_idx in split:
            fit_idx = [idx for idx in train_indices if idx not in set(val_idx)]
            x_fit, x_val = standardize_train_test(x, fit_idx, val_idx)
            betas: dict[str, np.ndarray] = {}
            for policy in policies:
                y = np.asarray([value(rows[idx], policy, metric) for idx in fit_idx])
                betas[policy] = fit_ridge(x_fit, y, alpha)
            train_mean_costs = {
                policy: mean([cost(rows[idx], policy) for idx in fit_idx])
                for policy in policies
            }
            for local_pos, idx in enumerate(val_idx):
                preds = {policy: float(x_val[local_pos] @ betas[policy]) for policy in policies}
                chosen = choose_predicted_policy(preds, train_mean_costs, policies, tie_eps)
                realized.append(value(rows[idx], chosen, metric))
        score = mean(realized)
        candidate = (score, -alpha)
        if best is None or candidate > best:
            best = candidate
            best_alpha = alpha
    return best_alpha


def run_multiclass_loocv(
    x: np.ndarray,
    qids: list[str],
    rows: list[dict[str, str]],
    *,
    policies: list[str],
    metric: str,
    alphas: list[float],
    folds: int,
    seed: int,
    tie_eps: float,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    all_indices = list(range(len(rows)))
    for test_idx in all_indices:
        train_idx = [idx for idx in all_indices if idx != test_idx]
        alpha = inner_multiclass_alpha(
            x,
            rows,
            train_idx,
            policies=policies,
            metric=metric,
            alphas=alphas,
            folds=folds,
            seed=seed + 1000 + test_idx,
            tie_eps=tie_eps,
        )
        x_train, x_test = standardize_train_test(x, train_idx, [test_idx])
        betas: dict[str, np.ndarray] = {}
        for policy in policies:
            y = np.asarray([value(rows[idx], policy, metric) for idx in train_idx])
            betas[policy] = fit_ridge(x_train, y, alpha)
        train_mean_costs = {policy: mean([cost(rows[idx], policy) for idx in train_idx]) for policy in policies}
        preds = {policy: float(x_test[0] @ betas[policy]) for policy in policies}
        chosen = choose_predicted_policy(preds, train_mean_costs, policies, tie_eps)
        outputs.append(
            {
                "qid": qids[test_idx],
                "selector": "multiclass_static_admission_selector",
                "chosen_policy": chosen,
                "predicted_delta": preds[chosen] - preds["stop"],
                "inner_alpha": alpha,
                "inner_threshold": "",
                "predicted_policy_values_json": json.dumps(preds, sort_keys=True),
            }
        )
    return outputs


def bootstrap_ci(values: list[float], samples: int, seed: int) -> dict[str, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    lo = draws[max(0, int(0.025 * samples) - 1)]
    hi = draws[min(samples - 1, int(0.975 * samples))]
    return {"mean": mean(clean), "ci_low": lo, "ci_high": hi, "n": len(clean)}


def summarize_selector(
    selector_rows: list[dict[str, Any]],
    qid_to_row: dict[str, dict[str, str]],
    *,
    selector: str,
    metric: str,
    best_fixed_policy: str,
    policies: list[str],
    eps: float,
    tie_eps: float,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    learned_values: list[float] = []
    stop_values: list[float] = []
    fixed_values: list[float] = []
    oracle_values: list[float] = []
    per_row_gaps: list[float] = []
    choice_counts: Counter[str] = Counter()
    for item in selector_rows:
        row = qid_to_row[str(item["qid"])]
        chosen = str(item["chosen_policy"])
        choice_counts[chosen] += 1
        learned = value(row, chosen, metric)
        fixed = value(row, best_fixed_policy, metric)
        oracle_policy = choose_static_oracle(row, policies, metric, tie_eps)
        learned_values.append(learned)
        stop_values.append(value(row, "stop", metric))
        fixed_values.append(fixed)
        oracle_values.append(value(row, oracle_policy, metric))
        per_row_gaps.append(learned - fixed)
    ci = bootstrap_ci(per_row_gaps, bootstrap_samples, seed)
    deltas_stop = [l - s for l, s in zip(learned_values, stop_values)]
    deltas_fixed = [l - f for l, f in zip(learned_values, fixed_values)]
    return {
        "selector": selector,
        "metric": metric,
        "n": len(selector_rows),
        "mean_learned_value": mean(learned_values),
        "mean_stop": mean(stop_values),
        "best_fixed_policy": best_fixed_policy,
        "mean_best_fixed": mean(fixed_values),
        "mean_static_oracle": mean(oracle_values),
        "learned_minus_stop": mean(deltas_stop),
        "learned_minus_best_fixed": mean(deltas_fixed),
        "learned_minus_best_fixed_ci_low": ci["ci_low"],
        "learned_minus_best_fixed_ci_high": ci["ci_high"],
        "static_oracle_minus_learned": mean([o - l for o, l in zip(oracle_values, learned_values)]),
        "wins_vs_stop_eps": sum(1 for delta in deltas_stop if wtl(delta, eps) == "win"),
        "ties_vs_stop_eps": sum(1 for delta in deltas_stop if wtl(delta, eps) == "tie"),
        "losses_vs_stop_eps": sum(1 for delta in deltas_stop if wtl(delta, eps) == "loss"),
        "wins_vs_best_fixed_eps": sum(1 for delta in deltas_fixed if wtl(delta, eps) == "win"),
        "ties_vs_best_fixed_eps": sum(1 for delta in deltas_fixed if wtl(delta, eps) == "tie"),
        "losses_vs_best_fixed_eps": sum(1 for delta in deltas_fixed if wtl(delta, eps) == "loss"),
        "chosen_policy_counts": json.dumps(dict(sorted(choice_counts.items())), sort_keys=True),
    }


def parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-matrix", type=Path, required=True)
    parser.add_argument("--stop-responses", type=Path, required=True)
    parser.add_argument("--stop-trace", type=Path, required=True)
    parser.add_argument("--metric", choices=["reward", "proxy_utility_context"], default="reward")
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--readmit-policy", action="append", default=None)
    parser.add_argument("--feature-set", choices=["stats", "text", "stats_text"], default="stats_text")
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--include-question-type", action="store_true")
    parser.add_argument(
        "--include-source-index-features",
        action="store_true",
        help="Diagnostic only. Raw source-index statistics are excluded from formal online features by default.",
    )
    parser.add_argument("--allow-missing-trace", action="store_true")
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--thresholds", default="-0.025,0,0.025,0.05,0.1")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--tie-eps", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.wide_matrix)
    if not rows:
        raise ValueError(f"empty wide matrix: {args.wide_matrix}")
    policies = ["stop"] + [policy for policy in args.policy if policy != "stop"]
    if "stop" not in policies:
        raise ValueError("stop must be present")
    for row in rows:
        qid = normalize_qid(str(row.get("qid") or ""))
        row["qid"] = qid
        row["evidence_bottleneck_group"] = group_label(row, "stop_retrieved_all_evidence_present")

    missing_columns: list[str] = []
    for policy in policies:
        metric_col = f"{policy}_reward" if args.metric == "reward" else f"{policy}_proxy_utility_context"
        if args.metric == "proxy_utility_context" and metric_col not in rows[0]:
            metric_col = f"{policy}_utility"
        for col in [metric_col, f"{policy}_cost"]:
            if col not in rows[0]:
                missing_columns.append(col)
    if missing_columns:
        raise ValueError(f"missing required matrix columns: {sorted(set(missing_columns))}")

    qids = [str(row["qid"]) for row in rows]
    if len(qids) != len(set(qids)):
        dupes = [qid for qid, n in Counter(qids).items() if n > 1]
        raise ValueError(f"duplicate qids in wide matrix: {dupes[:5]}")
    qid_to_row = {str(row["qid"]): row for row in rows}

    responses = load_responses(args.stop_responses)
    traces = load_traces_by_query_sha1(args.stop_trace)
    x, feature_names, feature_audit = build_online_features(
        qids,
        qid_to_row,
        responses,
        traces,
        feature_set=args.feature_set,
        hash_dim=args.hash_dim,
        include_question_type=args.include_question_type,
        include_source_index_features=args.include_source_index_features,
        allow_missing_trace=args.allow_missing_trace,
    )

    alphas = parse_float_list(args.alphas)
    thresholds = parse_float_list(args.thresholds)
    readmit_policies = args.readmit_policy or [policy for policy in policies if policy != "stop"]
    for policy in readmit_policies:
        if policy not in policies or policy == "stop":
            raise ValueError(f"invalid readmit policy: {policy}")

    best_fixed_policy = choose_best_fixed(rows, policies, args.metric, args.tie_eps)
    selector_predictions: list[dict[str, Any]] = []
    for policy in readmit_policies:
        selector_predictions.extend(
            run_binary_loocv(
                x,
                qids,
                rows,
                readmit_policy=policy,
                metric=args.metric,
                alphas=alphas,
                thresholds=thresholds,
                folds=args.inner_folds,
                seed=args.seed,
            )
        )
    selector_predictions.extend(
        run_multiclass_loocv(
            x,
            qids,
            rows,
            policies=policies,
            metric=args.metric,
            alphas=alphas,
            folds=args.inner_folds,
            seed=args.seed,
            tie_eps=args.tie_eps,
        )
    )

    per_qid: list[dict[str, Any]] = []
    for item in selector_predictions:
        row = qid_to_row[str(item["qid"])]
        chosen = str(item["chosen_policy"])
        oracle_policy = choose_static_oracle(row, policies, args.metric, args.tie_eps)
        out = dict(item)
        out.update(
            {
                "metric": args.metric,
                "feature_set": args.feature_set,
                "include_question_type": int(args.include_question_type),
                "include_source_index_features": int(args.include_source_index_features),
                "question_type_offline": row.get("question_type", ""),
                "evidence_bottleneck_group_offline": row.get("evidence_bottleneck_group", ""),
                "stop_value": value(row, "stop", args.metric),
                "chosen_value": value(row, chosen, args.metric),
                "best_fixed_policy": best_fixed_policy,
                "best_fixed_value": value(row, best_fixed_policy, args.metric),
                "static_oracle_policy": oracle_policy,
                "static_oracle_value": value(row, oracle_policy, args.metric),
                "chosen_minus_stop": value(row, chosen, args.metric) - value(row, "stop", args.metric),
                "chosen_minus_best_fixed": value(row, chosen, args.metric) - value(row, best_fixed_policy, args.metric),
            }
        )
        per_qid.append(out)

    selectors = sorted(set(row["selector"] for row in per_qid))
    summary_rows = [
        summarize_selector(
            [row for row in per_qid if row["selector"] == selector],
            qid_to_row,
            selector=selector,
            metric=args.metric,
            best_fixed_policy=best_fixed_policy,
            policies=policies,
            eps=args.eps,
            tie_eps=args.tie_eps,
            seed=args.seed + idx,
            bootstrap_samples=args.bootstrap_samples,
        )
        for idx, selector in enumerate(selectors)
    ]

    group_rows: list[dict[str, Any]] = []
    for selector in selectors:
        selected = [row for row in per_qid if row["selector"] == selector]
        for group_key, items in sorted(
            defaultdict(list, {
                key: [row for row in selected if row["evidence_bottleneck_group_offline"] == key]
                for key in sorted(set(row["evidence_bottleneck_group_offline"] for row in selected))
            }).items()
        ):
            if not items:
                continue
            group_rows.append(
                {
                    "selector": selector,
                    "metric": args.metric,
                    "evidence_bottleneck_group_offline": group_key,
                    "n": len(items),
                    "mean_chosen_value": mean([float(row["chosen_value"]) for row in items]),
                    "mean_stop_value": mean([float(row["stop_value"]) for row in items]),
                    "mean_best_fixed_value": mean([float(row["best_fixed_value"]) for row in items]),
                    "chosen_minus_stop": mean([float(row["chosen_minus_stop"]) for row in items]),
                    "chosen_minus_best_fixed": mean([float(row["chosen_minus_best_fixed"]) for row in items]),
                    "chosen_policy_counts": json.dumps(
                        dict(sorted(Counter(str(row["chosen_policy"]) for row in items).items())),
                        sort_keys=True,
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"p29_selector_summary_{args.metric}.csv", summary_rows)
    write_csv(args.output_dir / f"p29_selector_per_qid_{args.metric}.csv", per_qid)
    write_csv(args.output_dir / f"p29_selector_group_summary_{args.metric}.csv", group_rows)
    write_csv(args.output_dir / "p29_online_feature_audit.csv", feature_audit)

    report = {
        "status": (
            "PRIVILEGED_CONTROL_EXPLORATORY_NESTED_LOOCV"
            if args.include_question_type
            else "EXPLORATORY_ONLINE_ONLY_NESTED_LOOCV"
        ),
        "wide_matrix": str(args.wide_matrix),
        "stop_responses": str(args.stop_responses),
        "stop_trace": str(args.stop_trace),
        "metric": args.metric,
        "n": len(rows),
        "policies": policies,
        "readmit_policies": readmit_policies,
        "best_fixed_policy": best_fixed_policy,
        "feature_set": args.feature_set,
        "hash_dim": args.hash_dim,
        "include_question_type": args.include_question_type,
        "include_source_index_features": args.include_source_index_features,
        "alphas": alphas,
        "thresholds": thresholds,
        "inner_folds": args.inner_folds,
        "eps": args.eps,
        "tie_eps": args.tie_eps,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "feature_count": len(feature_names),
        "feature_audit_rows": len(feature_audit),
        "missing_trace_rows": sum(1 for row in feature_audit if int(row["trace_found"]) == 0),
        "missing_first_retrieve_rows": sum(1 for row in feature_audit if int(row["first_retrieve_found"]) == 0),
        "w0_state_text_available_rows": sum(1 for row in feature_audit if int(row["w0_state_text_available"]) == 1),
        "selector_summary": summary_rows,
        "guardrails": [
            "Read-only over frozen generation matrix and stop trace; no prompt/operator/protocol changes.",
            "Default online features exclude question_type, gold evidence, answer text, judge labels, response text, and raw source-index statistics.",
            "--include-question-type is a privileged control, not a legal online policy.",
            "--include-source-index-features is a dataset-order diagnostic control, not a formal online policy feature.",
            "Outer LOOCV evaluates each qid once; alpha and binary threshold are chosen only inside the train fold.",
            "Static oracle is an offline upper bound and not a deployable policy.",
            "Evidence bottleneck groups use gold-session labels only for post-hoc audit.",
            "Surrogate F1 remains exploratory unless the input matrix was built from an official judge.",
            "A positive learned selector result supports static admission learnability, not RL necessity.",
        ],
    }
    (args.output_dir / f"p29_selector_report_{args.metric}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
