"""Frozen numeric and feature primitives for MIC-v2 E1.

This module contains no rollout or holdout-opening entry.  It operates only on
sealed, time-safe feature rows and makes the finite E1-dev selection rule
auditable.  Stable identities are used for joins, weights, and folds only; they
are never feature inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import optimize
from scipy.special import expit


E1_SELECTION_NAMESPACE = "e1-selection"
HASH_SEED = 20260825
HASH_DIMENSION = 4096
ACTOR_COMPONENT_DIMENSION = 64
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
HEADS = ("fractional_logistic", "bounded_ridge")
REPRESENTATIONS = (
    ("turn_length", 5, 0),
    ("signed_text_hash", 4096, 1),
    ("actor_hidden_rademacher_128", 128, 2),
    ("actor_hidden_rademacher_256", 256, 3),
)
WORD_OR_PUNCT = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CHUNK_BOUNDARY = "\n\n<CHUNK_BOUNDARY>\n\n"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"MIC_V2_E1_NO_GO: {message}")


@dataclass(frozen=True)
class CandidateSpec:
    representation: str
    dimension: int
    representation_order: int
    head: str
    regularization: float

    def receipt(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "dimension": self.dimension,
            "head": self.head,
            "regularization": self.regularization,
        }


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec(name, dimension, order, head, regularization)
        for name, dimension, order in REPRESENTATIONS
        for head in HEADS
        for regularization in LAMBDAS
    )


def stable_selection_fold(content_root_id: str) -> int:
    _require(HEX64.fullmatch(content_root_id) is not None,
             "content root must be lowercase hex64")
    digest = hashlib.sha256(canonical_json(
        [content_root_id, E1_SELECTION_NAMESPACE]
    ).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 4


def signed_text_hash(
    components: Mapping[str, str], *, turn: int, dimension: int = HASH_DIMENSION,
) -> np.ndarray:
    """Fixed signed hashing of declared text components; identities are forbidden."""
    _require(dimension == HASH_DIMENSION, "signed-text dimension differs")
    allowed = ("question", "arrived_history", "current_memory")
    _require(tuple(components) == allowed
             and all(isinstance(components[key], str) for key in allowed),
             "signed-text component schema/order differs")
    _require(type(turn) is int and 0 <= turn <= 8, "signed-text turn differs")
    result = np.zeros(dimension, dtype=np.float64)
    count = 0
    for component in allowed:
        tokens = WORD_OR_PUNCT.findall(components[component])
        for position, token in enumerate(tokens):
            raw = hashlib.sha256(canonical_json(
                [HASH_SEED, component, position, token]
            ).encode("utf-8")).digest()
            index = int.from_bytes(raw[:8], "big") % dimension
            sign = 1.0 if raw[8] & 1 else -1.0
            result[index] += sign
            count += 1
    turn_raw = hashlib.sha256(canonical_json(
        [HASH_SEED, "turn-covariate", turn]
    ).encode("utf-8")).digest()
    result[int.from_bytes(turn_raw[:8], "big") % dimension] += (
        turn / 8.0 if turn_raw[8] & 1 else -turn / 8.0
    )
    count += 1
    if count:
        result /= math.sqrt(count)
    return result


def text_components_from_state(
    state: Mapping[str, Any], *, no_memory_text: str,
) -> dict[str, str]:
    _require(isinstance(no_memory_text, str) and no_memory_text,
             "no-memory text differs")
    question = state.get("question")
    chunks = state.get("arrived_chunks")
    memory = state.get("current_memory")
    _require(isinstance(question, str) and isinstance(chunks, list)
             and all(isinstance(chunk, str) for chunk in chunks)
             and isinstance(memory, str), "state text components differ")
    return {
        "question": question,
        "arrived_history": CHUNK_BOUNDARY.join(chunks),
        "current_memory": memory if memory else no_memory_text,
    }


def turn_length_features(
    *, turn: int, arrived_chunk_count: int, prior_active_turn_count: int,
    arrived_context_token_count: int, current_memory_token_count: int,
) -> np.ndarray:
    values = (
        turn, arrived_chunk_count, prior_active_turn_count,
        arrived_context_token_count, current_memory_token_count,
    )
    _require(all(type(value) is int and value >= 0 for value in values)
             and 0 <= turn <= 8 and arrived_chunk_count == turn,
             "turn/length feature receipt differs")
    return np.asarray(values, dtype=np.float64)


@lru_cache(maxsize=16)
def rademacher_matrix(
    input_dimension: int, output_dimension: int,
    namespace: str = "actor-hidden-rademacher",
) -> np.ndarray:
    _require(type(input_dimension) is int and input_dimension > 0
             and output_dimension in (64, 128, 256)
             and namespace in ("actor-hidden-component-rademacher",
                               "actor-hidden-rademacher"),
             "projection dimensions/namespace differ")
    matrix = np.empty((input_dimension, output_dimension), dtype=np.float64)
    scale = 1.0 / math.sqrt(output_dimension)
    for source in range(input_dimension):
        for target in range(output_dimension):
            digest = hashlib.sha256(canonical_json(
                [HASH_SEED, namespace, source, target]
            ).encode("utf-8")).digest()
            matrix[source, target] = scale if digest[0] & 1 else -scale
    matrix.setflags(write=False)
    return matrix


def rademacher_projection(
    hidden_components: Sequence[Sequence[float]], output_dimension: int, *, turn: int,
) -> np.ndarray:
    hidden = np.asarray(hidden_components, dtype=np.float64)
    _require(hidden.ndim == 1 and hidden.size > 0 and np.isfinite(hidden).all()
             and type(turn) is int and 0 <= turn <= 8,
             "actor-hidden component vector differs")
    augmented = np.concatenate((hidden, np.asarray([turn / 8.0], dtype=np.float64)))
    return augmented @ rademacher_matrix(augmented.size, output_dimension)


def actor_hidden_interactions(
    question_hidden: Sequence[float], arrived_chunk_hidden: Sequence[Sequence[float]],
    memory_hidden: Sequence[float],
) -> np.ndarray:
    """Freeze question-conditioned history aggregation and interactions.

    The behavior-policy hidden states are first compressed by one common fixed
    64-dimensional component projection.  The final candidate projection is
    applied only after the symmetric interaction vector has been constructed.
    """
    question = np.asarray(question_hidden, dtype=np.float64)
    chunks = np.asarray(arrived_chunk_hidden, dtype=np.float64)
    memory = np.asarray(memory_hidden, dtype=np.float64)
    if chunks.size == 0 and question.ndim == 1:
        chunks = np.empty((0, question.size), dtype=np.float64)
    _require(question.ndim == memory.ndim == 1 and question.size == memory.size > 0
             and chunks.ndim == 2
             and chunks.shape[1] == question.size
             and np.isfinite(question).all() and np.isfinite(chunks).all()
             and np.isfinite(memory).all(), "actor-hidden state components differ")

    def unit(value: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        _require(norm > 0 and math.isfinite(norm), "actor-hidden component norm is zero")
        return value / norm

    if chunks.shape[0] == 0:
        history = np.zeros_like(question)
    else:
        question_unit = unit(question)
        chunk_units = np.stack([unit(row) for row in chunks], axis=0)
        scores = chunk_units @ question_unit
        scores -= scores.max()
        attention = np.exp(scores)
        attention /= attention.sum()
        history = attention @ chunks
    component_matrix = rademacher_matrix(
        question.size, ACTOR_COMPONENT_DIMENSION,
        "actor-hidden-component-rademacher",
    )
    q = question @ component_matrix
    h = history @ component_matrix
    m = memory @ component_matrix
    result = np.concatenate((q, h, m, q * h, h * m, q * m, np.abs(h - m)))
    _require(result.shape == (7 * ACTOR_COMPONENT_DIMENSION,)
             and np.isfinite(result).all(), "actor-hidden interaction vector differs")
    return result


def root_trajectory_turn_weights(
    root_ids: Sequence[str], trajectory_ids: Sequence[str], turns: Sequence[int],
) -> np.ndarray:
    """Equal root -> trajectory -> available stage-state weights, summing to one."""
    _require(len(root_ids) == len(trajectory_ids) == len(turns) > 0,
             "weight row lengths differ")
    grouped: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str, int]] = set()
    for root, trajectory, turn in zip(root_ids, trajectory_ids, turns):
        _require(HEX64.fullmatch(root) is not None and isinstance(trajectory, str)
                 and trajectory and type(turn) is int and 0 <= turn <= 8,
                 "weight identity differs")
        key = (root, trajectory, turn)
        _require(key not in seen, "duplicate root/trajectory/turn pair")
        seen.add(key)
        grouped.setdefault(root, {})[trajectory] = grouped.setdefault(root, {}).get(trajectory, 0) + 1
    root_count = len(grouped)
    weights = []
    for root, trajectory in zip(root_ids, trajectory_ids):
        trajectory_count = len(grouped[root])
        turn_count = grouped[root][trajectory]
        weights.append(1.0 / (root_count * trajectory_count * turn_count))
    result = np.asarray(weights, dtype=np.float64)
    _require(abs(float(result.sum()) - 1.0) <= 1e-12, "root weights do not sum to one")
    return result


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def apply(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        _require(values.ndim == 2 and values.shape[1] == self.mean.size
                 and np.isfinite(values).all(), "feature matrix differs")
        return (values - self.mean) / self.scale


def fit_standardizer(features: np.ndarray, weights: np.ndarray) -> Standardizer:
    values = np.asarray(features, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] == weight.size
             and values.shape[1] > 0 and np.isfinite(values).all()
             and np.isfinite(weight).all() and (weight >= 0).all()
             and float(weight.sum()) > 0, "fit-fold feature/weight matrix differs")
    normalized = weight / weight.sum()
    mean = normalized @ values
    variance = normalized @ np.square(values - mean)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale == 0.0] = 1.0
    return Standardizer(mean=mean, scale=scale)


@dataclass(frozen=True)
class FittedHead:
    spec: CandidateSpec
    standardizer: Standardizer
    intercept: float
    coefficients: np.ndarray
    iterations: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        linear = self.intercept + self.standardizer.apply(features) @ self.coefficients
        if self.spec.head == "fractional_logistic":
            prediction = expit(linear)
        else:
            prediction = np.clip(linear, 0.0, 1.0)
        _require(np.isfinite(prediction).all(), "head prediction is non-finite")
        return prediction

    def receipt(self) -> dict[str, Any]:
        return {
            "specification": self.spec.receipt(),
            "standardizer_mean": self.standardizer.mean.tolist(),
            "standardizer_scale": self.standardizer.scale.tolist(),
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
            "iterations": self.iterations,
        }

    @classmethod
    def from_receipt(cls, value: Mapping[str, Any]) -> "FittedHead":
        _require(set(value) == {
            "specification", "standardizer_mean", "standardizer_scale",
            "intercept", "coefficients", "iterations",
        }, "fitted-head receipt schema differs")
        spec_value = value["specification"]
        _require(set(spec_value) == {"representation", "dimension", "head", "regularization"},
                 "fitted-head specification differs")
        order = {name: item_order for name, _dimension, item_order in REPRESENTATIONS}
        dimensions = {name: dimension for name, dimension, _item_order in REPRESENTATIONS}
        _require(spec_value["representation"] in order
                 and spec_value["dimension"] == dimensions[spec_value["representation"]]
                 and spec_value["head"] in HEADS
                 and spec_value["regularization"] in LAMBDAS,
                 "fitted-head specification is outside preregistration")
        spec = CandidateSpec(
            spec_value["representation"], spec_value["dimension"],
            order[spec_value["representation"]], spec_value["head"],
            spec_value["regularization"],
        )
        mean = np.asarray(value["standardizer_mean"], dtype=np.float64)
        scale = np.asarray(value["standardizer_scale"], dtype=np.float64)
        coefficients = np.asarray(value["coefficients"], dtype=np.float64)
        _require(mean.shape == scale.shape == coefficients.shape == (spec.dimension,)
                 and np.isfinite(mean).all() and np.isfinite(scale).all()
                 and np.isfinite(coefficients).all() and (scale > 0).all()
                 and math.isfinite(float(value["intercept"]))
                 and type(value["iterations"]) is int and value["iterations"] >= 0,
                 "fitted-head numeric receipt differs")
        return cls(
            spec, Standardizer(mean=mean, scale=scale), float(value["intercept"]),
            coefficients, value["iterations"],
        )


def fit_head(
    spec: CandidateSpec, features: np.ndarray, target: np.ndarray, weights: np.ndarray,
    *, tolerance: float = 1e-9, maximum_iterations: int = 500,
) -> FittedHead:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    _require(x.ndim == 2 and x.shape == (y.size, spec.dimension)
             and weight.shape == y.shape and np.isfinite(x).all()
             and np.isfinite(y).all() and np.isfinite(weight).all()
             and (0 <= y).all() and (y <= 1).all() and (weight >= 0).all()
             and float(weight.sum()) > 0, "head fit inputs differ")
    weight = weight / weight.sum()
    standardizer = fit_standardizer(x, weight)
    z = standardizer.apply(x)
    if spec.head == "bounded_ridge":
        y_mean = float(weight @ y)
        square_root_weight = np.sqrt(weight)
        design = square_root_weight[:, None] * z
        centered_target = square_root_weight * (y - y_mean)
        if spec.dimension <= y.size:
            gram = design.T @ design
            right = design.T @ centered_target
            coefficients = np.linalg.solve(
                gram + spec.regularization * np.eye(spec.dimension, dtype=np.float64), right,
            )
        else:
            # Algebraically identical dual ridge solve.  The 4096-dimensional
            # signed-hash candidate has fewer E1 fit rows than features, so
            # this avoids an unnecessary 4096 x 4096 factorization without
            # changing the preregistered closed-form estimator.
            dual = np.linalg.solve(
                design @ design.T
                + spec.regularization * np.eye(y.size, dtype=np.float64),
                centered_target,
            )
            coefficients = design.T @ dual
        result = FittedHead(spec, standardizer, y_mean, coefficients, 1)
    elif spec.head == "fractional_logistic":
        clipped_mean = min(max(float(weight @ y), 1e-8), 1.0 - 1e-8)
        initial = np.zeros(spec.dimension + 1, dtype=np.float64)
        initial[0] = math.log(clipped_mean / (1.0 - clipped_mean))

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            return _fractional_logistic_objective(
                parameters, z, y, weight, spec.regularization,
            )

        optimized = optimize.minimize(
            objective, initial, method="L-BFGS-B", jac=True,
            options={"ftol": tolerance, "gtol": tolerance, "maxiter": maximum_iterations,
                     "maxls": 50},
        )
        _require(bool(optimized.success) and optimized.nit <= maximum_iterations
                 and np.isfinite(optimized.x).all(),
                 f"fractional-logistic fit did not converge: {optimized.message}")
        result = FittedHead(
            spec, standardizer, float(optimized.x[0]), optimized.x[1:].copy(), int(optimized.nit),
        )
    else:
        raise ValueError("MIC_V2_E1_NO_GO: unknown head family")
    _require(np.isfinite(result.coefficients).all() and math.isfinite(result.intercept),
             "fitted head is non-finite")
    return result


def _fractional_logistic_objective(
    parameters: np.ndarray, features: np.ndarray, target: np.ndarray,
    weights: np.ndarray, regularization: float,
) -> tuple[float, np.ndarray]:
    """Stable fractional-logistic loss and its exact analytic gradient.

    ``logaddexp`` evaluates the unmodified logit objective without clipping, so
    the returned Jacobian remains the derivative of the returned scalar even
    for logits far outside the usual floating-point sigmoid transition region.
    """
    vector = np.asarray(parameters, dtype=np.float64)
    z = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    _require(vector.ndim == 1 and z.ndim == 2
             and vector.size == z.shape[1] + 1
             and y.shape == weight.shape == (z.shape[0],)
             and np.isfinite(vector).all() and np.isfinite(z).all()
             and np.isfinite(y).all() and np.isfinite(weight).all()
             and math.isfinite(regularization) and regularization >= 0,
             "fractional-logistic objective inputs differ")
    intercept, coefficients = float(vector[0]), vector[1:]
    linear = intercept + z @ coefficients
    loss = float(np.sum(weight * (np.logaddexp(0.0, linear) - y * linear)))
    loss += regularization * float(coefficients @ coefficients)
    residual = weight * (expit(linear) - y)
    gradient = np.concatenate((
        np.asarray([residual.sum()]),
        z.T @ residual + 2.0 * regularization * coefficients,
    ))
    _require(math.isfinite(loss) and np.isfinite(gradient).all(),
             "fractional-logistic objective is non-finite")
    return loss, gradient


def weighted_mse(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    y, p, w = map(lambda value: np.asarray(value, dtype=np.float64),
                  (target, prediction, weights))
    _require(y.shape == p.shape == w.shape and y.ndim == 1 and y.size > 0
             and np.isfinite(y).all() and np.isfinite(p).all() and np.isfinite(w).all()
             and (w >= 0).all() and float(w.sum()) > 0, "weighted MSE inputs differ")
    return float((w / w.sum()) @ np.square(y - p))


def calibration(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    y, p, w = map(lambda value: np.asarray(value, dtype=np.float64),
                  (target, prediction, weights))
    w = w / w.sum()
    p_mean, y_mean = float(w @ p), float(w @ y)
    variance = float(w @ np.square(p - p_mean))
    _require(variance > 0 and math.isfinite(variance), "calibration prediction variance is zero")
    slope = float(w @ ((p - p_mean) * (y - y_mean))) / variance
    intercept = y_mean - slope * p_mean
    _require(math.isfinite(slope) and math.isfinite(intercept), "calibration is non-finite")
    return {"slope": slope, "intercept": intercept}


def select_specification(
    *, root_ids: Sequence[str], trajectory_ids: Sequence[str], turns: Sequence[int],
    target: Sequence[float], features: Mapping[str, Mapping[str, np.ndarray]],
    stage_masks: Mapping[str, Sequence[bool]] | None = None,
    tolerance: float = 1e-9, maximum_iterations: int = 500,
) -> dict[str, Any]:
    """Run the frozen four-fold E1-dev selection over all 48 specifications."""
    y = np.asarray(target, dtype=np.float64)
    _require(y.ndim == 1 and y.size == len(root_ids), "selection target rows differ")
    masks = {
        stage: (np.ones(y.size, dtype=bool) if stage_masks is None
                else np.asarray(stage_masks[stage], dtype=bool))
        for stage in ("pre", "post")
    }
    _require(all(mask.shape == (y.size,) and mask.any() for mask in masks.values())
             and not np.any(masks["pre"] & ~masks["post"]),
             "selection stage availability differs")
    reports = []
    for spec in candidate_specs():
        stage_predictions: dict[str, np.ndarray] = {}
        failed = None
        try:
            for stage in ("pre", "post"):
                full_matrix = np.asarray(features[spec.representation][stage], dtype=np.float64)
                _require(full_matrix.shape == (y.size, spec.dimension),
                         f"{spec.representation}/{stage} feature matrix differs")
                mask = masks[stage]
                matrix, stage_y = full_matrix[mask], y[mask]
                stage_roots = np.asarray(root_ids, dtype=object)[mask].tolist()
                stage_trajectories = np.asarray(trajectory_ids, dtype=object)[mask].tolist()
                stage_turns = np.asarray(turns, dtype=np.int64)[mask].tolist()
                weights = root_trajectory_turn_weights(
                    stage_roots, stage_trajectories, stage_turns,
                )
                folds = np.asarray(
                    [stable_selection_fold(root) for root in stage_roots], dtype=np.int64,
                )
                _require(set(folds.tolist()) == {0, 1, 2, 3},
                         "all four E1 selection folds are required")
                oof = np.full(stage_y.size, np.nan, dtype=np.float64)
                for fold in range(4):
                    fit = folds != fold
                    heldout = folds == fold
                    _require(fit.any() and heldout.any(), "empty selection fold")
                    head = fit_head(
                        spec, matrix[fit], stage_y[fit], weights[fit],
                        tolerance=tolerance, maximum_iterations=maximum_iterations,
                    )
                    oof[heldout] = head.predict(matrix[heldout])
                _require(np.isfinite(oof).all(), "OOF prediction is incomplete")
                stage_predictions[stage] = oof
        except (ValueError, np.linalg.LinAlgError) as exc:
            failed = str(exc)
        if failed is None:
            stage_mse = {}
            for stage in ("pre", "post"):
                mask = masks[stage]
                stage_weights = root_trajectory_turn_weights(
                    np.asarray(root_ids, dtype=object)[mask].tolist(),
                    np.asarray(trajectory_ids, dtype=object)[mask].tolist(),
                    np.asarray(turns, dtype=np.int64)[mask].tolist(),
                )
                stage_mse[stage] = weighted_mse(
                    y[mask], stage_predictions[stage], stage_weights,
                )
            pre_mse, post_mse = stage_mse["pre"], stage_mse["post"]
            score = 0.5 * (pre_mse + post_mse)
            prediction_sha = sha256_json({
                "pre": stage_predictions["pre"].tolist(),
                "post": stage_predictions["post"].tolist(),
            })
            reports.append({
                "status": "PASS", "specification": spec.receipt(), "score": score,
                "pre_oof_mse": pre_mse, "post_oof_mse": post_mse,
                "prediction_sha256": prediction_sha,
            })
        else:
            reports.append({"status": "FAIL", "specification": spec.receipt(), "failure": failed})
    passing = [report for report in reports if report["status"] == "PASS"]
    _require(passing, "every E1-dev candidate failed")
    minimum = min(report["score"] for report in passing)
    tied = [report for report in passing if report["score"] <= minimum + 1e-6]
    order_lookup = {name: order for name, _dimension, order in REPRESENTATIONS}
    tied.sort(key=lambda report: (
        report["specification"]["dimension"],
        -report["specification"]["regularization"],
        0 if report["specification"]["head"] == "fractional_logistic" else 1,
        order_lookup[report["specification"]["representation"]],
    ))
    selected = tied[0]
    return {
        "schema": "memagent.mic.v2.e1-selection",
        "status": "PASS", "decision": "MIC_V2_E1_SPECIFICATION_SELECTED",
        "fold_namespace": E1_SELECTION_NAMESPACE, "fold_count": 4,
        "row_count": y.size,
        "stage_row_counts": {stage: int(mask.sum()) for stage, mask in masks.items()},
        "root_count": len(set(root_ids)),
        "fold_assignments_sha256": sha256_json({
            stage: [
                [root_ids[index], stable_selection_fold(root_ids[index])]
                for index in range(y.size) if mask[index]
            ] for stage, mask in masks.items()
        }),
        "candidate_count": len(reports), "selected": selected,
        "candidates_sha256": sha256_json(reports), "candidates": reports,
    }


def cross_fitted_predictions(
    spec: CandidateSpec, *, root_ids: Sequence[str], trajectory_ids: Sequence[str],
    turns: Sequence[int], target: Sequence[float],
    stage_features: Mapping[str, np.ndarray],
    stage_masks: Mapping[str, Sequence[bool]] | None = None,
    tolerance: float = 1e-9,
    maximum_iterations: int = 500,
) -> dict[str, np.ndarray]:
    y = np.asarray(target, dtype=np.float64)
    result = {}
    for stage in ("pre", "post"):
        matrix = np.asarray(stage_features[stage], dtype=np.float64)
        _require(matrix.shape == (y.size, spec.dimension),
                 f"selected {stage} feature matrix differs")
        mask = (np.ones(y.size, dtype=bool) if stage_masks is None
                else np.asarray(stage_masks[stage], dtype=bool))
        _require(mask.shape == (y.size,) and mask.any(),
                 f"selected {stage} availability differs")
        stage_roots = np.asarray(root_ids, dtype=object)[mask].tolist()
        stage_trajectories = np.asarray(trajectory_ids, dtype=object)[mask].tolist()
        stage_turns = np.asarray(turns, dtype=np.int64)[mask].tolist()
        stage_y, stage_matrix = y[mask], matrix[mask]
        weights = root_trajectory_turn_weights(stage_roots, stage_trajectories, stage_turns)
        folds = np.asarray([stable_selection_fold(root) for root in stage_roots], dtype=np.int64)
        oof = np.full(stage_y.size, np.nan, dtype=np.float64)
        for fold in range(4):
            fit, heldout = folds != fold, folds == fold
            head = fit_head(
                spec, stage_matrix[fit], stage_y[fit], weights[fit],
                tolerance=tolerance, maximum_iterations=maximum_iterations,
            )
            oof[heldout] = head.predict(stage_matrix[heldout])
        _require(np.isfinite(oof).all(), "selected OOF predictions are incomplete")
        full = np.full(y.size, np.nan, dtype=np.float64)
        full[mask] = oof
        result[stage] = full
    return result
