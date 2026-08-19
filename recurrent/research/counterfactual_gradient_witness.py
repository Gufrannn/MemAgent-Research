"""CPU-only W4 v8 group capture and control-variate algebra.

This module consumes gradients captured elsewhere. It never calls backward,
``optimizer.step``, a model, or a rollout engine.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence


FORBIDDEN_EVIDENCE_BASES = {
    "single_batch_alignment",
    "alignment",
    "captured_signed_ratio",
    "effect_weighted_mass",
    "silent_mass",
    "opposed_mass",
    "gradient_difference_norm",
    "gradient_norm_only",
    "nonzero_gradient_norm",
    "nonzero_gradient_rate",
    "scalar_advantage_sign",
    "single_parameter_delta",
}


def vector_hash(values: Sequence[float]) -> str:
    payload = json.dumps([float(x) for x in values], separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _mean_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("W4_NO_GO: empty gradient collection")
    dimensions = {len(row) for row in vectors}
    if len(dimensions) != 1 or next(iter(dimensions)) == 0:
        raise ValueError("W4_NO_GO: gradient subspace dimensions mismatch")
    return [sum(float(row[j]) for row in vectors) / len(vectors) for j in range(len(vectors[0]))]


def _weighted_score_mean(weights: Sequence[float], scores: Sequence[Sequence[float]]) -> list[float]:
    if len(weights) != len(scores) or not weights:
        raise ValueError("W4_NO_GO: reward/score group length mismatch")
    dimensions = {len(row) for row in scores}
    if len(dimensions) != 1 or next(iter(dimensions)) == 0:
        raise ValueError("W4_NO_GO: gradient subspace dimensions mismatch")
    n = len(weights)
    return [sum(float(weight) * float(score[j]) for weight, score in zip(weights, scores)) / n
            for j in range(len(scores[0]))]


def group_estimators(group: dict[str, Any]) -> dict[str, Any]:
    """Compute W4 v8 same-endpoint estimators for one fixed IID candidate group."""
    rewards = [float(x) for x in group["commit_returns"]]
    baselines = [float(x) for x in group["noop_baseline_returns"]]
    scores = [[float(x) for x in row] for row in group["score_gradients"]]
    n = len(rewards)
    if n < 2 or len(baselines) != n or len(scores) != n:
        raise ValueError("W4_NO_GO: candidate group requires n>=2 aligned rewards/baselines/scores")
    values = rewards + baselines + [x for row in scores for x in row]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("W4_NO_GO: non-finite group capture")
    if max(baselines) - min(baselines) > 1e-12:
        raise ValueError("W4_NO_GO: exact-NOOP baseline is candidate-dependent within group")

    reward_mean = sum(rewards) / n
    centered = [reward - reward_mean for reward in rewards]
    all_mean = _weighted_score_mean(centered, scores)
    debias_factor = n / (n - 1)
    all_mean_debiased = [debias_factor * value for value in all_mean]
    loo_weights = [reward - (sum(rewards) - reward) / (n - 1) for reward in rewards]
    loo = _weighted_score_mean(loo_weights, scores)
    g_cf = _weighted_score_mean([reward - baseline for reward, baseline in zip(rewards, baselines)], scores)
    raw = _weighted_score_mean(rewards, scores)
    return {
        "group_size": n,
        "including_self_expected_scale": (n - 1) / n,
        "including_self_debias_factor": debias_factor,
        "g_raw_commit": raw,
        "g_credit_including_self_all_mean": all_mean,
        "g_credit_debiased": all_mean_debiased,
        "g_credit_loo": loo,
        "g_cf_external_noop": g_cf,
        "finite_batch_mean_score": _mean_vectors(scores),
        "all_equal_commit_returns": max(rewards) - min(rewards) <= 1e-12,
    }


def capture_w4_group(*, candidate_group_id: str, candidate_group_manifest_hash: str,
                     checkpoint_hash: str, state_hash: str, subspace_hash: str,
                     loss_graph_hash: str, candidate_hashes: list[str],
                     commit_returns: list[float], noop_baseline_returns: list[float],
                     score_gradients: list[list[float]],
                     policy_controlled_token_kinds: list[str],
                     many_action_reference_gradient: list[float] | None = None) -> dict[str, Any]:
    """Create an immutable group row from prefrozen, already-captured arrays."""
    n = len(candidate_hashes)
    if len(set(candidate_hashes)) != n:
        raise ValueError("W4_NO_GO: duplicate candidate in fixed group")
    if "eos_or_stop" not in policy_controlled_token_kinds:
        raise ValueError("W4_NO_GO: writer score mask must include EOS/stop decision")
    group = {
        "candidate_group_id": str(candidate_group_id),
        "candidate_group_manifest_hash": str(candidate_group_manifest_hash),
        "checkpoint_hash": str(checkpoint_hash),
        "state_hash": str(state_hash),
        "subspace_hash": str(subspace_hash),
        "loss_graph_hash": str(loss_graph_hash),
        "candidate_hashes": [str(x) for x in candidate_hashes],
        "commit_returns": [float(x) for x in commit_returns],
        "noop_baseline_returns": [float(x) for x in noop_baseline_returns],
        "score_gradients": [[float(x) for x in row] for row in score_gradients],
        "policy_controlled_token_kinds": [str(x) for x in policy_controlled_token_kinds],
        "writer_score_mask_complete": True,
        "writer_score_mask_includes_eos_or_stop": True,
        "candidate_group_prefrozen": True,
    }
    estimates = group_estimators(group)
    if many_action_reference_gradient is not None:
        reference = [float(x) for x in many_action_reference_gradient]
        if len(reference) != len(estimates["g_cf_external_noop"]) or not all(math.isfinite(x) for x in reference):
            raise ValueError("W4_NO_GO: invalid independent many-action reference")
        group["many_action_reference_gradient"] = reference
    flattened = (group["commit_returns"] + group["noop_baseline_returns"] +
                 [x for row in group["score_gradients"] for x in row])
    group["capture_hash"] = vector_hash(flattened)
    return group


def objective_mismatch_estimators(group: dict[str, Any]) -> dict[str, Any]:
    """Compute train/eval objective gradients and registered OM negative controls."""
    scores = [[float(x) for x in row] for row in group["score_gradients"]]
    train = [float(x) for x in group["training_commit_returns"]]
    evaluation = [float(x) for x in group["scientific_eval_returns"]]
    shuffled = [float(x) for x in group["endpoint_label_shuffle_returns"]]
    if len(train) < 2 or not (len(train) == len(evaluation) == len(shuffled) == len(scores)):
        raise ValueError("W4_NO_GO: OM endpoint rows must align within an n>=2 candidate group")
    ablations = group.get("component_ablation_returns")
    if not isinstance(ablations, dict) or not ablations:
        raise ValueError("W4_NO_GO: OM requires registered equal-scale component ablations")
    if any(not isinstance(values, list) or len(values) != len(train) for values in ablations.values()):
        raise ValueError("W4_NO_GO: OM component ablation rows do not align")
    values = train + evaluation + shuffled + [x for row in scores for x in row]
    values += [float(x) for rows in ablations.values() for x in rows]
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("W4_NO_GO: non-finite OM capture")
    g_train = _weighted_score_mean(train, scores)
    g_eval = _weighted_score_mean(evaluation, scores)
    return {
        "group_size": len(train),
        "g_train": g_train,
        "g_eval": g_eval,
        "g_eval_minus_g_train": [left - right for left, right in zip(g_eval, g_train)],
        "g_endpoint_label_shuffle": _weighted_score_mean(shuffled, scores),
        "g_component_ablations": {
            name: _weighted_score_mean([float(x) for x in rows], scores)
            for name, rows in sorted(ablations.items())
        },
    }


def capture_w4_objective_mismatch_group(*, candidate_group_id: str,
                                        candidate_group_manifest_hash: str,
                                        checkpoint_hash: str, state_hash: str,
                                        subspace_hash: str, loss_graph_hash: str,
                                        candidate_hashes: list[str],
                                        training_commit_returns: list[float],
                                        scientific_eval_returns: list[float],
                                        endpoint_label_shuffle_returns: list[float],
                                        component_ablation_returns: dict[str, list[float]],
                                        score_gradients: list[list[float]],
                                        policy_controlled_token_kinds: list[str]) -> dict[str, Any]:
    """Create an immutable W4 v8 distinct-endpoint group capture."""
    if len(set(candidate_hashes)) != len(candidate_hashes):
        raise ValueError("W4_NO_GO: duplicate candidate in fixed OM group")
    if "eos_or_stop" not in policy_controlled_token_kinds:
        raise ValueError("W4_NO_GO: writer score mask must include EOS/stop decision")
    group = {
        "candidate_group_id": str(candidate_group_id),
        "candidate_group_manifest_hash": str(candidate_group_manifest_hash),
        "checkpoint_hash": str(checkpoint_hash),
        "state_hash": str(state_hash),
        "subspace_hash": str(subspace_hash),
        "loss_graph_hash": str(loss_graph_hash),
        "candidate_hashes": [str(x) for x in candidate_hashes],
        "training_commit_returns": [float(x) for x in training_commit_returns],
        "scientific_eval_returns": [float(x) for x in scientific_eval_returns],
        "endpoint_label_shuffle_returns": [float(x) for x in endpoint_label_shuffle_returns],
        "component_ablation_returns": {
            str(name): [float(x) for x in values]
            for name, values in sorted(component_ablation_returns.items())
        },
        "score_gradients": [[float(x) for x in row] for row in score_gradients],
        "policy_controlled_token_kinds": [str(x) for x in policy_controlled_token_kinds],
        "writer_score_mask_complete": True,
        "writer_score_mask_includes_eos_or_stop": True,
        "candidate_group_prefrozen": True,
    }
    objective_mismatch_estimators(group)
    flattened = (group["training_commit_returns"] + group["scientific_eval_returns"] +
                 group["endpoint_label_shuffle_returns"] +
                 [x for values in group["component_ablation_returns"].values() for x in values] +
                 [x for row in group["score_gradients"] for x in row])
    group["capture_hash"] = vector_hash(flattened)
    return group
