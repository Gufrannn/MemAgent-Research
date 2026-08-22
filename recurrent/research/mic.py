"""Materialized Memory Innovation Credit (MIC), fail-closed core.

This module intentionally separates admissible critic features from outcomes.
Feature records are constructed first; outcomes are supplied only to fitting and
evaluation functions.  Cross-fitted predictions never use an example/root in
the corresponding fold's training set.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "memagent.mic.v1"
FOLD_RULE = "sha256_stable_root_id_modulo_v1"
FORBIDDEN_KEYS = frozenset({
    "gold", "gold_answer", "ground_truth", "reference_answer", "label",
    "reward", "outcome", "score", "token_f1", "exact_match", "em",
    "generated_answer", "final_answer", "answer_text", "future_chunk",
    "future_chunks", "next_chunk", "all_chunks",
})
ALLOWED_STATE_KEYS = frozenset({
    "stable_example_id", "stable_root_id", "trajectory_id", "turn_index",
    "question", "visible_chunks", "materialized_memory", "memory_token_count",
    "materialized_memory_history", "visible_chunk_token_count", "is_prewrite", "state_sha256",
})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"MIC_NO_GO: {field} must be finite")
    return float(value)


def validate_admissible_state(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a critic row without ever accepting an outcome-like field."""
    if not isinstance(record, Mapping):
        raise ValueError("MIC_NO_GO: critic state must be a mapping")
    keys = set(record)
    forbidden = keys & FORBIDDEN_KEYS
    unknown = keys - ALLOWED_STATE_KEYS
    if forbidden:
        raise ValueError(f"MIC_NO_GO: forbidden critic fields: {sorted(forbidden)}")
    if unknown:
        raise ValueError(f"MIC_NO_GO: unknown critic fields: {sorted(unknown)}")
    required = {
        "stable_example_id", "stable_root_id", "trajectory_id", "turn_index",
        "question", "visible_chunks", "materialized_memory", "materialized_memory_history",
        "is_prewrite",
    }
    if not required <= keys:
        raise ValueError(f"MIC_NO_GO: missing critic fields: {sorted(required - keys)}")
    checked = dict(record)
    for key in ("stable_example_id", "stable_root_id", "trajectory_id"):
        if not isinstance(checked[key], str) or not checked[key]:
            raise ValueError(f"MIC_NO_GO: {key} must be a non-empty stable string")
    if isinstance(checked["turn_index"], bool) or not isinstance(checked["turn_index"], int) \
            or checked["turn_index"] < 0:
        raise ValueError("MIC_NO_GO: turn_index must be an integer >= 0")
    if not isinstance(checked["is_prewrite"], bool):
        raise ValueError("MIC_NO_GO: is_prewrite must be boolean")
    for key in ("question", "materialized_memory"):
        if not isinstance(checked[key], str):
            raise ValueError(f"MIC_NO_GO: {key} must be text")
    chunks = checked["visible_chunks"]
    if not isinstance(chunks, list) or any(not isinstance(chunk, str) for chunk in chunks):
        raise ValueError("MIC_NO_GO: visible_chunks must be a list of text")
    if len(chunks) != checked["turn_index"] and not checked["is_prewrite"]:
        raise ValueError("MIC_NO_GO: post-write visible chunk count must equal turn_index")
    memories = checked["materialized_memory_history"]
    if not isinstance(memories, list) or any(not isinstance(memory, str) for memory in memories):
        raise ValueError("MIC_NO_GO: materialized_memory_history must be a list of text")
    expected_history = 0 if checked["is_prewrite"] else checked["turn_index"]
    if len(memories) != expected_history:
        raise ValueError("MIC_NO_GO: memory history must contain every materialized write through turn")
    if memories and memories[-1] != checked["materialized_memory"]:
        raise ValueError("MIC_NO_GO: current memory differs from materialized history tail")
    state_payload = {
        key: checked[key] for key in (
            "stable_example_id", "stable_root_id", "trajectory_id", "turn_index",
            "question", "visible_chunks", "materialized_memory", "materialized_memory_history",
            "is_prewrite",
        )
    }
    expected_hash = sha256_json(state_payload)
    supplied = checked.get("state_sha256")
    if supplied is not None and supplied != expected_hash:
        raise ValueError("MIC_NO_GO: materialized state hash mismatch")
    checked["state_sha256"] = expected_hash
    checked["memory_token_count"] = int(checked.get(
        "memory_token_count", len(checked["materialized_memory"].split())
    ))
    checked["visible_chunk_token_count"] = int(checked.get(
        "visible_chunk_token_count", sum(len(x.split()) for x in chunks)
    ))
    if checked["memory_token_count"] < 0 or checked["visible_chunk_token_count"] < 0:
        raise ValueError("MIC_NO_GO: token counts must be non-negative")
    return checked


def stable_fold_assignments(root_ids: Sequence[str], fold_count: int) -> dict[str, int]:
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError("MIC_NO_GO: fold_count must be >= 2")
    unique = sorted(set(root_ids))
    if len(unique) < fold_count:
        raise ValueError("MIC_NO_GO: fewer stable roots than folds")
    return {root: int.from_bytes(hashlib.sha256(root.encode("utf-8")).digest()[:8], "big") % fold_count
            for root in unique}


def stable_source_identities(dataset_indices: Sequence[int],
                             prompt_token_ids: Sequence[Sequence[int]],
                             rollout_n: int) -> tuple[list[str], list[str]]:
    """Bind recurrent source rows to deterministic dataset/prompt coordinates."""
    if isinstance(rollout_n, bool) or not isinstance(rollout_n, int) or rollout_n < 1:
        raise ValueError("MIC_NO_GO: rollout_n must be a positive integer")
    if len(dataset_indices) != len(prompt_token_ids) or not dataset_indices:
        raise ValueError("MIC_NO_GO: stable source identity coverage mismatch")
    roots, examples = [], []
    for row, (dataset_index, token_ids) in enumerate(zip(dataset_indices, prompt_token_ids)):
        if isinstance(dataset_index, bool) or not isinstance(dataset_index, int) \
                or not isinstance(token_ids, Sequence) \
                or any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids):
            raise ValueError("MIC_NO_GO: invalid stable source coordinates")
        root = sha256_json({"dataset_index": dataset_index,
                            "prompt_ids": [int(token) for token in token_ids]})
        roots.append(root)
        examples.append(sha256_json({"stable_root_id": root, "replica": row % rollout_n}))
    return roots, examples


def _hashed_text_features(text: str, dimension: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float64)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[bucket] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def state_features(record: Mapping[str, Any], *, dimension: int = 64,
                   restricted: bool = False) -> np.ndarray:
    row = validate_admissible_state(record)
    numeric = np.asarray([
        1.0,
        math.log1p(row["turn_index"]),
        math.log1p(row["memory_token_count"]),
        math.log1p(row["visible_chunk_token_count"]),
        float(row["is_prewrite"]),
    ], dtype=np.float64)
    if restricted:
        return numeric
    text = "\n".join([row["question"], *row["visible_chunks"],
                       *row["materialized_memory_history"]])
    return np.concatenate([numeric, _hashed_text_features(text, dimension)])


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError("MIC_NO_GO: invalid critic fit arrays")
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def cross_fitted_values(states: Sequence[Mapping[str, Any]], outcomes: Mapping[str, float],
                        *, fold_count: int = 4, alpha: float = 1.0,
                        dimension: int = 64, restricted: bool = False) -> dict[str, Any]:
    """Fit fold-exclusive ridge critics and return auditable OOF predictions."""
    checked = [validate_admissible_state(row) for row in states]
    if not checked:
        raise ValueError("MIC_NO_GO: empty state set")
    trajectory_roots: dict[str, str] = {}
    for row in checked:
        prior = trajectory_roots.setdefault(row["trajectory_id"], row["stable_root_id"])
        if prior != row["stable_root_id"]:
            raise ValueError("MIC_NO_GO: trajectory spans stable roots")
    if set(outcomes) != set(trajectory_roots):
        raise ValueError("MIC_NO_GO: outcome trajectory coverage mismatch")
    y_by_trajectory = {key: _require_finite(value, f"outcome[{key}]")
                       for key, value in outcomes.items()}
    folds = stable_fold_assignments(list(trajectory_roots.values()), fold_count)
    predictions: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    fold_models: list[dict[str, Any]] = []
    for fold in range(fold_count):
        train = [row for row in checked if folds[row["stable_root_id"]] != fold]
        held = [row for row in checked if folds[row["stable_root_id"]] == fold]
        if not held:
            continue
        if not train:
            raise ValueError("MIC_NO_GO: all stable roots collapsed into one fold")
        train_roots = {row["stable_root_id"] for row in train}
        held_roots = {row["stable_root_id"] for row in held}
        if train_roots & held_roots:
            raise ValueError("MIC_NO_GO: stable-root fold leakage")
        x_train = np.stack([state_features(row, dimension=dimension, restricted=restricted)
                            for row in train])
        y_train = np.asarray([y_by_trajectory[row["trajectory_id"]] for row in train])
        weights = _ridge_fit(x_train, y_train, alpha)
        train_ids = sorted({row["stable_example_id"] for row in train})
        held_ids = sorted({row["stable_example_id"] for row in held})
        receipt = {
            "fold": fold,
            "train_root_sha256": sha256_json(sorted(train_roots)),
            "held_root_sha256": sha256_json(sorted(held_roots)),
            "train_example_sha256": sha256_json(train_ids),
            "held_example_sha256": sha256_json(held_ids),
            "train_count": len(train), "held_count": len(held),
            "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        }
        receipts.append(receipt)
        fold_models.append({
            "fold": fold, "weights": [float(value) for value in weights],
            "weight_sha256": receipt["weight_sha256"],
            "train_root_sha256": receipt["train_root_sha256"],
        })
        for row in held:
            value = float(state_features(row, dimension=dimension, restricted=restricted) @ weights)
            predictions.append({
                "trajectory_id": row["trajectory_id"],
                "stable_example_id": row["stable_example_id"],
                "stable_root_id": row["stable_root_id"],
                "turn_index": row["turn_index"],
                "is_prewrite": row["is_prewrite"],
                "state_sha256": row["state_sha256"],
                "fold": fold, "value": value,
            })
    predictions.sort(key=lambda row: (
        row["trajectory_id"], row["turn_index"], not row["is_prewrite"]
    ))
    return {
        "schema": SCHEMA, "kind": "oof_values", "fold_rule": FOLD_RULE,
        "fold_count": fold_count, "alpha": alpha, "dimension": dimension,
        "restricted": restricted, "predictions": predictions, "receipts": receipts,
        "fold_models": fold_models,
        "bundle_sha256": sha256_json({"predictions": predictions, "receipts": receipts,
                                       "fold_models": fold_models}),
    }


def innovation_ledger(oof: Mapping[str, Any], outcomes: Mapping[str, float],
                      *, tolerance: float = 1e-12) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oof.get("predictions", []):
        grouped[str(row["trajectory_id"])].append(dict(row))
    if set(grouped) != set(outcomes):
        raise ValueError("MIC_NO_GO: prediction/outcome trajectory mismatch")
    trajectories = []
    maximum = 0.0
    for trajectory_id in sorted(grouped):
        rows = grouped[trajectory_id]
        pre = [row for row in rows if row["is_prewrite"]]
        post = sorted((row for row in rows if not row["is_prewrite"]),
                      key=lambda row: row["turn_index"])
        if len(pre) != 1 or pre[0]["turn_index"] != 0 or not post:
            raise ValueError("MIC_NO_GO: each trajectory needs one prewrite and postwrites")
        expected_turns = list(range(1, len(post) + 1))
        if [row["turn_index"] for row in post] != expected_turns:
            raise ValueError("MIC_NO_GO: post-write turns are not contiguous")
        values = [float(pre[0]["value"]), *[float(row["value"]) for row in post]]
        increments = [values[index] - values[index - 1] for index in range(1, len(values))]
        outcome = _require_finite(outcomes[trajectory_id], f"outcome[{trajectory_id}]")
        residual = outcome - values[-1]
        closure = values[0] + math.fsum(increments) + residual - outcome
        maximum = max(maximum, abs(closure))
        trajectories.append({
            "trajectory_id": trajectory_id,
            "stable_root_id": pre[0]["stable_root_id"],
            "fold": pre[0]["fold"], "v0": values[0], "values": values[1:],
            "writer_innovations": increments, "answer_residual": residual,
            "outcome": outcome, "closure_error": closure,
        })
    if maximum > tolerance:
        raise ValueError(f"MIC_NO_GO: telescoping error {maximum} > {tolerance}")
    return {
        "schema": SCHEMA, "kind": "innovation_ledger", "tolerance": tolerance,
        "maximum_closure_error": maximum, "trajectories": trajectories,
        "ledger_sha256": sha256_json(trajectories),
    }


def calibration_report(ledger: Mapping[str, Any]) -> dict[str, float]:
    trajectories = ledger.get("trajectories", [])
    if len(trajectories) < 2:
        raise ValueError("MIC_NO_GO: calibration requires at least two trajectories")
    prediction = np.asarray([row["values"][-1] for row in trajectories], dtype=np.float64)
    outcome = np.asarray([row["outcome"] for row in trajectories], dtype=np.float64)
    mse = float(np.mean((prediction - outcome) ** 2))
    mae = float(np.mean(np.abs(prediction - outcome)))
    centered = prediction - prediction.mean()
    denominator = float(centered @ centered)
    slope = float(centered @ (outcome - outcome.mean()) / denominator) if denominator else 0.0
    intercept = float(outcome.mean() - slope * prediction.mean())
    innovations = np.asarray([value for row in trajectories
                              for value in row["writer_innovations"]], dtype=np.float64)
    return {
        "mse": mse, "mae": mae, "calibration_slope": slope,
        "calibration_intercept": intercept,
        "writer_innovation_mean": float(innovations.mean()),
        "writer_innovation_variance": float(innovations.var()),
        "answer_residual_variance": float(np.asarray(
            [row["answer_residual"] for row in trajectories]
        ).var()),
    }


def route_role_advantages(*, sample_index: Any, final_mask: Any, turn_index: Any,
                          response_mask: Any, ledger_rows: Sequence[Mapping[str, Any]],
                          trajectory_ids: Sequence[str]):
    """Route writer innovations and answer residuals to disjoint token masks."""
    import torch

    if sample_index.ndim != 1 or final_mask.ndim != 1 or turn_index.ndim != 1 \
            or response_mask.ndim != 2:
        raise ValueError("MIC_NO_GO: role-routing tensor ranks invalid")
    if not (len(sample_index) == len(final_mask) == len(turn_index) == len(response_mask)):
        raise ValueError("MIC_NO_GO: role-routing row alignment invalid")
    if final_mask.dtype != torch.bool:
        raise ValueError("MIC_NO_GO: final_mask must be boolean")
    if len(trajectory_ids) != len(ledger_rows):
        raise ValueError("MIC_NO_GO: trajectory ledger alignment invalid")
    by_id = {str(row["trajectory_id"]): row for row in ledger_rows}
    if set(by_id) != set(trajectory_ids) or len(by_id) != len(ledger_rows):
        raise ValueError("MIC_NO_GO: trajectory IDs are missing or duplicated")
    output = torch.zeros_like(response_mask, dtype=torch.float32)
    writer_rows = torch.zeros_like(response_mask, dtype=torch.bool)
    answer_rows = torch.zeros_like(response_mask, dtype=torch.bool)
    for row_index in range(len(sample_index)):
        source = int(sample_index[row_index])
        if source < 0 or source >= len(trajectory_ids):
            raise ValueError("MIC_NO_GO: sample_index out of range")
        ledger = by_id[str(trajectory_ids[source])]
        if bool(final_mask[row_index]):
            scalar = float(ledger["answer_residual"])
            answer_rows[row_index] = response_mask[row_index].bool()
        else:
            turn = int(turn_index[row_index])
            if turn < 1 or turn > len(ledger["writer_innovations"]):
                raise ValueError("MIC_NO_GO: writer turn has no innovation")
            scalar = float(ledger["writer_innovations"][turn - 1])
            writer_rows[row_index] = response_mask[row_index].bool()
        output[row_index] = scalar * response_mask[row_index]
    if torch.any(writer_rows & answer_rows):
        raise ValueError("MIC_NO_GO: writer/answer delivery overlap")
    return output, {
        "writer_active_tokens": int(writer_rows.sum().item()),
        "answer_active_tokens": int(answer_rows.sum().item()),
        "writer_advantage_sha256": hashlib.sha256(
            output.masked_select(writer_rows).detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "answer_advantage_sha256": hashlib.sha256(
            output.masked_select(answer_rows).detach().cpu().numpy().tobytes()
        ).hexdigest(),
    }


@dataclass(frozen=True)
class CriticCheckpoint:
    actor_commit: str
    fold_bundle_sha256: str
    state_feature_schema_sha256: str
    critic_payload: Mapping[str, Any]

    def write_new(self, path: str | Path) -> str:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"MIC_NO_GO: critic checkpoint exists: {target}")
        payload = {
            "schema": SCHEMA, "kind": "critic_checkpoint",
            "actor_commit": self.actor_commit,
            "fold_bundle_sha256": self.fold_bundle_sha256,
            "state_feature_schema_sha256": self.state_feature_schema_sha256,
            "critic_payload": self.critic_payload,
        }
        payload["checkpoint_sha256"] = sha256_json(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload["checkpoint_sha256"]

    @staticmethod
    def read(path: str | Path, *, expected_actor_commit: str) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        digest = payload.pop("checkpoint_sha256", None)
        if digest != sha256_json(payload):
            raise ValueError("MIC_NO_GO: critic checkpoint digest mismatch")
        if payload.get("actor_commit") != expected_actor_commit:
            raise ValueError("MIC_NO_GO: critic checkpoint actor commit mismatch")
        payload["checkpoint_sha256"] = digest
        return payload


def append_jsonl_new(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = "0" * 64
    if target.exists():
        lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            last = json.loads(lines[-1])
            previous = str(last.get("entry_sha256", ""))
            unsigned = {key: value for key, value in last.items() if key != "entry_sha256"}
            if previous != sha256_json(unsigned):
                raise ValueError("MIC_NO_GO: append-only ledger tail is corrupted")
    unsigned = dict(row)
    unsigned["previous_entry_sha256"] = previous
    unsigned["sequence"] = 0 if previous == "0" * 64 else len(lines)
    entry = dict(unsigned)
    entry["entry_sha256"] = sha256_json(unsigned)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(entry) + "\n")
