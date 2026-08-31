#!/usr/bin/env python3
"""P30 Semantic Selective Admission over frozen P27/P29.5 artifacts.

This script is read-only with respect to generation artifacts.  It does not
call a reader, change prompts, change operators, or open confirm data.

Method contract:
    SSA is a Default + Override policy.  For every outer held-out qid, the
    default fixed policy is selected only from the outer training fold:

        a0^(-i) = argmax_a mean_{j != i} R_j(a)

    The model learns predicted margins Delta_a(s) = R(s, a) - R(s, a0^(-i)).
    It overrides the fold-local default only if max_a Delta_hat_a(s) > tau.

Feature contract:
    Formal SSA features are online semantic summaries of the retrieved
    candidate pool C0 and admitted working memory W0.  Gold evidence, answers,
    judge labels, operation outcomes, question_type, response text, and raw
    source-index statistics are not used as features.  Source indices are used
    only as pointers to reconstruct candidate/admitted texts from raw
    LongMemEval sessions.

Evaluation boundary:
    Outer LOOCV is used for realized policy evaluation.  Alpha and threshold
    are selected inside the outer train fold.  Inner folds are stratified by
    rare override opportunity labels computed only from the inner training
    data's operation-value matrix; final realized policy is evaluated under
    the original sample prevalence, with no oversampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from train_p29_selective_admission_gate import (
    bootstrap_ci,
    choose_static_oracle,
    cost,
    first_op_record,
    group_label,
    iter_jsonl,
    mean,
    normalize_qid,
    parse_float,
    parse_float_list,
    read_csv,
    sha1_text,
    value,
    write_csv,
    wtl,
)


CONV_START_PROMPT = "Below is a conversation between {} and {}.\n\n"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest_ids(path: Path) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, dict):
        ids = payload.get("question_ids")
        if ids is None:
            ids = [item.get("question_id") for item in payload.get("items", [])]
    elif isinstance(payload, list):
        ids = payload
    else:
        raise ValueError(f"Unsupported manifest format: {path}")
    out = [normalize_qid(str(qid)) for qid in ids if qid]
    if not out:
        raise ValueError(f"empty manifest: {path}")
    return out


def raw_question_id(qid: str) -> str:
    qid = normalize_qid(qid)
    return qid[len("longmemeval_") :]


def load_raw_rows(path: Path, manifest_qids: list[str]) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("raw LongMemEval file must be a JSON list")
    wanted = set(manifest_qids)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = normalize_qid(str(row.get("question_id") or ""))
        if qid in wanted:
            out[qid] = row
    missing = sorted(wanted - set(out))
    if missing:
        raise ValueError(f"manifest ids missing from raw LongMemEval, first={missing[:5]}")
    return out


def longmemeval_chunks(raw_row: dict[str, Any]) -> list[str]:
    sessions = raw_row.get("haystack_sessions") or []
    dates = raw_row.get("haystack_dates") or []
    chunks: list[str] = []
    start_prompt = CONV_START_PROMPT.format("user", "assistant")
    for idx, session in enumerate(sessions):
        if not session:
            continue
        session_date = dates[idx] if idx < len(dates) else "Unknown"
        session_conv = ""
        for turn in session:
            role = turn.get("role", "")
            content = turn.get("content", "")
            turn_role = "User" if role == "user" else "Assistant"
            session_conv += f'{turn_role} said, "{content}"\n'
        if not session_conv:
            session_conv = "NO CONVERSATION"
        query_conv = f"DATE: {session_date}\nCONVERSATION:\n{session_conv}\n\n"
        chunks.append(start_prompt + query_conv)
    return chunks


def load_responses(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        qid = normalize_qid(str(row.get("qid") or row.get("question_id") or ""))
        if qid in out:
            raise ValueError(f"duplicate qid in responses: {qid}")
        out[qid] = row
    return out


def load_traces_by_query_sha1(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("phase") != "qa":
            continue
        qhash = str(row.get("query_sha1") or "")
        if not qhash:
            continue
        if qhash in out:
            raise ValueError(f"duplicate query_sha1 in trace: {qhash}")
        out[qhash] = row
    return out


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a_norm @ b_norm.T


class SentenceTransformerEncoder:
    backend = "sentence_transformers"

    def __init__(self, model_name_or_path: str, batch_size: int):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence_transformers is required for formal SSA. "
                "Do not fall back to hashed/TF-IDF features for SSA v1."
            ) from exc
        self.model = SentenceTransformer(model_name_or_path)
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)
        arr = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=float)


class TransformersMeanEncoder:
    backend = "transformers_mean"

    def __init__(self, model_name_or_path: str, batch_size: int, max_length: int):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "transformers_mean backend requires torch and transformers. "
                "Do not fall back to hashed/TF-IDF features for SSA v1."
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.batch_size = batch_size
        self.max_length = max_length

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)
        outputs: list[np.ndarray] = []
        torch = self.torch
        device = next(self.model.parameters()).device
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(float)


def text_fingerprint(texts: list[str]) -> str:
    payload = "\n".join(hashlib.sha1(text.encode("utf-8")).hexdigest() for text in texts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_or_build_embeddings(
    texts: list[str],
    *,
    encoder_name: str,
    encoder_backend: str,
    cache_path: Path | None,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    fingerprint = text_fingerprint(texts)
    if cache_path and cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        if (
            str(payload["fingerprint"]) == fingerprint
            and str(payload["encoder_name"]) == encoder_name
            and str(payload["encoder_backend"]) == encoder_backend
        ):
            return np.asarray(payload["embeddings"], dtype=float)
    if encoder_backend == "sentence_transformers":
        encoder = SentenceTransformerEncoder(encoder_name, batch_size)
    elif encoder_backend == "transformers_mean":
        encoder = TransformersMeanEncoder(encoder_name, batch_size, max_length)
    else:
        raise ValueError(f"unsupported encoder backend: {encoder_backend}")
    embeddings = encoder.encode(texts)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            fingerprint=fingerprint,
            encoder_name=encoder_name,
            encoder_backend=encoder_backend,
            embeddings=embeddings,
        )
    return embeddings


def safe_float(value_: float) -> float:
    if math.isnan(value_) or math.isinf(value_):
        return 0.0
    return float(value_)


def describe(values: list[float], prefix: str) -> dict[str, float]:
    clean = [safe_float(v) for v in values if not math.isnan(v)]
    if not clean:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_mass_pos": 0.0,
        }
    arr = np.asarray(clean, dtype=float)
    return {
        f"{prefix}_count": float(len(clean)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_mass_pos": float(np.sum(np.maximum(arr, 0.0))),
    }


def entropy_from_scores(scores: list[float]) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=float)
    arr = arr - float(np.max(arr))
    weights = np.exp(arr)
    probs = weights / max(float(np.sum(weights)), 1e-12)
    return float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))


def mean_pairwise_dissimilarity(embeddings: np.ndarray) -> float:
    if embeddings.shape[0] < 2:
        return 0.0
    sims = cosine_matrix(embeddings, embeddings)
    tri = sims[np.triu_indices(sims.shape[0], k=1)]
    return float(1.0 - np.mean(tri)) if tri.size else 0.0


def build_semantic_feature_table(
    qids: list[str],
    wide_rows: dict[str, dict[str, str]],
    raw_rows: dict[str, dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    traces_by_query: dict[str, dict[str, Any]],
    *,
    encoder_name: str,
    encoder_backend: str,
    embedding_cache: Path | None,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    text_keys: list[str] = []
    texts: list[str] = []
    per_qid_items: dict[str, dict[str, Any]] = {}

    def add_text(key: str, text: str) -> None:
        if key not in seen:
            seen.add(key)
            text_keys.append(key)
            texts.append(text)

    seen: set[str] = set()
    for qid in qids:
        response = responses.get(qid)
        if response is None:
            raise ValueError(f"missing stop response for {qid}")
        query = str(response.get("query") or "")
        qhash = sha1_text(query)
        trace_row = traces_by_query.get(qhash)
        if trace_row is None:
            raise ValueError(f"missing stop trace for {qid}, query_sha1={qhash}")
        retrieve = first_op_record(trace_row, {"RETRIEVE", "RETRIEVE_RECENT"})
        if retrieve is None:
            raise ValueError(f"missing first retrieve record for {qid}")
        retrieved = [int(x) for x in retrieve.get("retrieved_source_indices") or []]
        admitted = [int(x) for x in retrieve.get("admitted_source_indices") or []]
        chunk_map = {idx: text for idx, text in enumerate(longmemeval_chunks(raw_rows[qid]))}
        missing_indices = sorted(set(retrieved + admitted) - set(chunk_map))
        if missing_indices:
            raise ValueError(f"{qid} has trace indices absent from raw chunks: {missing_indices[:5]}")

        add_text(f"query::{qid}", query)
        for idx in retrieved:
            add_text(f"chunk::{qid}::{idx}", chunk_map[idx])
        per_qid_items[qid] = {
            "query": query,
            "retrieve": retrieve,
            "retrieved": retrieved,
            "admitted": admitted,
            "chunk_map": chunk_map,
        }

    embeddings = load_or_build_embeddings(
        texts,
        encoder_name=encoder_name,
        encoder_backend=encoder_backend,
        cache_path=embedding_cache,
        batch_size=batch_size,
        max_length=max_length,
    )
    emb_by_key = {key: embeddings[pos] for pos, key in enumerate(text_keys)}

    feature_rows: list[dict[str, float]] = []
    audit_rows: list[dict[str, Any]] = []
    for qid in qids:
        item = per_qid_items[qid]
        retrieved = item["retrieved"]
        admitted = item["admitted"]
        admitted_set = set(admitted)
        omitted = [idx for idx in retrieved if idx not in admitted_set]
        q_emb = emb_by_key[f"query::{qid}"].reshape(1, -1)
        cand_emb = np.asarray([emb_by_key[f"chunk::{qid}::{idx}"] for idx in retrieved], dtype=float)
        scores = cosine_matrix(q_emb, cand_emb)[0].tolist() if retrieved else []
        score_by_idx = {idx: float(score) for idx, score in zip(retrieved, scores)}
        admitted_scores = [score_by_idx[idx] for idx in admitted if idx in score_by_idx]
        omitted_scores = [score_by_idx[idx] for idx in omitted if idx in score_by_idx]
        all_scores = [score_by_idx[idx] for idx in retrieved if idx in score_by_idx]
        admitted_emb = np.asarray([emb_by_key[f"chunk::{qid}::{idx}"] for idx in admitted if idx in score_by_idx], dtype=float)
        omitted_emb = np.asarray([emb_by_key[f"chunk::{qid}::{idx}"] for idx in omitted if idx in score_by_idx], dtype=float)
        cand_emb_valid = np.asarray([emb_by_key[f"chunk::{qid}::{idx}"] for idx in retrieved if idx in score_by_idx], dtype=float)

        max_omitted = max(omitted_scores) if omitted_scores else -1.0
        min_admitted = min(admitted_scores) if admitted_scores else 1.0
        mean_omitted = mean(omitted_scores)
        mean_admitted = mean(admitted_scores)
        if math.isnan(mean_omitted):
            mean_omitted = -1.0
        if math.isnan(mean_admitted):
            mean_admitted = 0.0
        top_all = max(all_scores) if all_scores else 0.0
        second_all = sorted(all_scores, reverse=True)[1] if len(all_scores) > 1 else top_all
        context_chars = parse_float(item["retrieve"].get("context_chars"))
        budget_chars = parse_float(item["retrieve"].get("budget_chars"))
        if math.isnan(context_chars):
            context_chars = 0.0
        if math.isnan(budget_chars) or budget_chars <= 0:
            budget_chars = 1.0

        row: dict[str, float] = {
            "bias": 1.0,
            "semantic_gap_max_omitted_minus_min_admitted": max_omitted - min_admitted,
            "semantic_gap_max_omitted_minus_mean_admitted": max_omitted - mean_admitted,
            "semantic_gap_mean_omitted_minus_mean_admitted": mean_omitted - mean_admitted,
            "semantic_best_omitted_score": max_omitted,
            "semantic_worst_admitted_score": min_admitted,
            "semantic_best_candidate_score": top_all,
            "semantic_top_margin_all": top_all - second_all,
            "semantic_mass_outside_w0": sum(max(score, 0.0) for score in omitted_scores),
            "semantic_mass_inside_w0": sum(max(score, 0.0) for score in admitted_scores),
            "semantic_mass_outside_minus_inside": sum(max(score, 0.0) for score in omitted_scores)
            - sum(max(score, 0.0) for score in admitted_scores),
            "coverage_candidate_count": float(len(retrieved)),
            "coverage_admitted_count": float(len(admitted)),
            "coverage_omitted_count": float(len(omitted)),
            "coverage_omitted_ratio": float(len(omitted) / max(1, len(retrieved))),
            "budget_context_kchars": context_chars / 1000.0,
            "budget_kchars": budget_chars / 1000.0,
            "budget_context_ratio": context_chars / max(1.0, budget_chars),
            "concentration_all_entropy": entropy_from_scores(all_scores),
            "concentration_admitted_entropy": entropy_from_scores(admitted_scores),
            "concentration_omitted_entropy": entropy_from_scores(omitted_scores),
            "diversity_c0_pairwise_dissim": mean_pairwise_dissimilarity(cand_emb_valid),
            "diversity_w0_pairwise_dissim": mean_pairwise_dissimilarity(admitted_emb),
            "diversity_omitted_pairwise_dissim": mean_pairwise_dissimilarity(omitted_emb),
        }
        row.update(describe(all_scores, "semantic_c0_score"))
        row.update(describe(admitted_scores, "semantic_w0_score"))
        row.update(describe(omitted_scores, "semantic_omitted_score"))
        feature_rows.append(row)
        audit_rows.append(
            {
                "qid": qid,
                "query_sha1": sha1_text(item["query"]),
                "encoder": encoder_name,
                "encoder_backend": encoder_backend,
                "retrieved_count": len(retrieved),
                "admitted_count": len(admitted),
                "omitted_count": len(omitted),
                "state_text_sha1": sha1_text(str(item["retrieve"].get("state_text") or "")),
                "feature_contract": "semantic_C0_W0_gap_no_gold_no_question_type_no_source_index",
                "source_indices_used_as": "pointers_only_not_features",
                "question_type_used": 0,
                "gold_used": 0,
                "answer_used": 0,
                "judge_used": 0,
                "operation_outcome_used": 0,
                "raw_source_index_feature_used": 0,
                "question_type_offline": wide_rows[qid].get("question_type", ""),
            }
        )

    names = ["bias"] + sorted({key for row in feature_rows for key in row if key != "bias"})
    matrix = np.asarray([[row.get(name, 0.0) for name in names] for row in feature_rows], dtype=float)
    return matrix, names, audit_rows


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    train_out = (train - mu) / sigma
    test_out = (test - mu) / sigma
    train_out[:, 0] = 1.0
    test_out[:, 0] = 1.0
    return train_out, test_out


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    lhs = x.T @ x + penalty
    rhs = x.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def choose_best_fixed_fold(rows: list[dict[str, str]], indices: list[int], policies: list[str], metric: str, tie_eps: float) -> str:
    means = {
        policy: mean([value(rows[idx], policy, metric) for idx in indices])
        for policy in policies
    }
    costs = {
        policy: mean([cost(rows[idx], policy) for idx in indices])
        for policy in policies
    }
    best_val = max(means.values())
    tied = [policy for policy in policies if best_val - means[policy] <= tie_eps]
    order = {policy: pos for pos, policy in enumerate(policies)}
    return min(tied, key=lambda policy: (costs[policy], order[policy]))


def stratified_folds(indices: list[int], labels: dict[int, int], folds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    buckets: dict[int, list[int]] = {0: [], 1: []}
    for idx in indices:
        buckets[int(labels.get(idx, 0))].append(idx)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    folds = min(max(2, folds), len(indices))
    out = [[] for _ in range(folds)]
    for bucket in buckets.values():
        for pos, idx in enumerate(bucket):
            out[pos % folds].append(idx)
    return [fold for fold in out if fold]


def opportunity_labels(
    rows: list[dict[str, str]],
    indices: list[int],
    policies: list[str],
    metric: str,
    default_policy: str,
    margin_eps: float,
) -> dict[int, int]:
    labels: dict[int, int] = {}
    for idx in indices:
        base = value(rows[idx], default_policy, metric)
        best_alt = max(
            value(rows[idx], policy, metric) - base
            for policy in policies
            if policy != default_policy
        )
        labels[idx] = int(best_alt > margin_eps)
    return labels


def fit_margin_models(
    x_train: np.ndarray,
    rows: list[dict[str, str]],
    train_row_idx: list[int],
    policies: list[str],
    metric: str,
    default_policy: str,
    alpha: float,
) -> dict[str, np.ndarray]:
    if x_train.shape[0] != len(train_row_idx):
        raise ValueError("x_train rows must align with train_row_idx")
    betas: dict[str, np.ndarray] = {}
    for policy in policies:
        if policy == default_policy:
            continue
        y = np.asarray(
            [
                value(rows[idx], policy, metric) - value(rows[idx], default_policy, metric)
                for idx in train_row_idx
            ],
            dtype=float,
        )
        betas[policy] = fit_ridge(x_train, y, alpha)
    return betas


def choose_override(
    preds: dict[str, float],
    default_policy: str,
    threshold: float,
    train_mean_costs: dict[str, float],
    policies: list[str],
    tie_eps: float,
) -> tuple[str, float]:
    best_delta = max(preds.values()) if preds else -math.inf
    if best_delta <= threshold:
        return default_policy, best_delta
    tied = [policy for policy, pred in preds.items() if best_delta - pred <= tie_eps]
    order = {policy: pos for pos, policy in enumerate(policies)}
    return min(tied, key=lambda policy: (train_mean_costs.get(policy, 0.0), order[policy])), best_delta


def inner_select_alpha_threshold(
    x: np.ndarray,
    rows: list[dict[str, str]],
    outer_train_idx: list[int],
    policies: list[str],
    metric: str,
    alphas: list[float],
    thresholds: list[float],
    folds: int,
    seed: int,
    tie_eps: float,
    margin_eps: float,
) -> tuple[float, float]:
    outer_default = choose_best_fixed_fold(rows, outer_train_idx, policies, metric, tie_eps)
    labels = opportunity_labels(rows, outer_train_idx, policies, metric, outer_default, margin_eps)
    split = stratified_folds(outer_train_idx, labels, folds, seed)
    best_key: tuple[float, float, float] | None = None
    best_alpha = alphas[0]
    best_threshold = thresholds[0]
    for alpha in alphas:
        for threshold in thresholds:
            realized: list[float] = []
            for val_idx in split:
                val_set = set(val_idx)
                fit_idx = [idx for idx in outer_train_idx if idx not in val_set]
                default_policy = choose_best_fixed_fold(rows, fit_idx, policies, metric, tie_eps)
                x_fit_raw = x[fit_idx]
                x_val_raw = x[val_idx]
                x_fit, x_val = standardize(x_fit_raw.copy(), x_val_raw.copy())
                betas = fit_margin_models(x_fit, rows, fit_idx, policies, metric, default_policy, alpha)
                train_mean_costs = {policy: mean([cost(rows[idx], policy) for idx in fit_idx]) for policy in policies}
                for local_pos, idx in enumerate(val_idx):
                    preds = {policy: float(x_val[local_pos] @ beta) for policy, beta in betas.items()}
                    chosen, _ = choose_override(preds, default_policy, threshold, train_mean_costs, policies, tie_eps)
                    realized.append(value(rows[idx], chosen, metric))
            score = mean(realized)
            key = (score, -abs(threshold), -alpha)
            if best_key is None or key > best_key:
                best_key = key
                best_alpha = alpha
                best_threshold = threshold
    return best_alpha, best_threshold


def run_ssa_loocv(
    x: np.ndarray,
    qids: list[str],
    rows: list[dict[str, str]],
    policies: list[str],
    metric: str,
    alphas: list[float],
    thresholds: list[float],
    folds: int,
    seed: int,
    tie_eps: float,
    margin_eps: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    all_idx = list(range(len(rows)))
    for test_idx in all_idx:
        train_idx = [idx for idx in all_idx if idx != test_idx]
        alpha, threshold = inner_select_alpha_threshold(
            x,
            rows,
            train_idx,
            policies,
            metric,
            alphas,
            thresholds,
            folds,
            seed + test_idx,
            tie_eps,
            margin_eps,
        )
        default_policy = choose_best_fixed_fold(rows, train_idx, policies, metric, tie_eps)
        x_train, x_test = standardize(x[train_idx].copy(), x[[test_idx]].copy())
        betas = fit_margin_models(x_train, rows, train_idx, policies, metric, default_policy, alpha)
        train_mean_costs = {policy: mean([cost(rows[idx], policy) for idx in train_idx]) for policy in policies}
        preds = {policy: float(x_test[0] @ beta) for policy, beta in betas.items()}
        chosen, best_pred_delta = choose_override(preds, default_policy, threshold, train_mean_costs, policies, tie_eps)
        out.append(
            {
                "qid": qids[test_idx],
                "method": "SSA_v1_default_override",
                "chosen_policy": chosen,
                "fold_default_policy": default_policy,
                "overrode_default": int(chosen != default_policy),
                "best_predicted_delta": best_pred_delta,
                "inner_alpha": alpha,
                "inner_threshold": threshold,
                "predicted_deltas_json": json.dumps(preds, sort_keys=True),
            }
        )
    return out


def fixed_policy_rows(qids: list[str], method: str, policy: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "qid": qid,
            "method": method,
            "chosen_policy": policy,
            "fold_default_policy": "",
            "overrode_default": "",
            "best_predicted_delta": "",
            "inner_alpha": "",
            "inner_threshold": "",
            "predicted_deltas_json": "",
        }
        for qid in qids
    ]


def summarize_method(
    method_rows: list[dict[str, Any]],
    qid_to_row: dict[str, dict[str, str]],
    rows: list[dict[str, str]],
    policies: list[str],
    metric: str,
    method: str,
    eps: float,
    tie_eps: float,
    seed: int,
    bootstrap_samples: int,
    context: str,
) -> dict[str, Any]:
    chosen_values: list[float] = []
    stop_values: list[float] = []
    best_fixed_values: list[float] = []
    group_b_gaps: list[float] = []
    group_c_gaps: list[float] = []
    chosen_minus_default_for_overrides: list[float] = []
    choice_counts: Counter[str] = Counter()
    default_counts: Counter[str] = Counter()
    for item in method_rows:
        qid = str(item["qid"])
        row = qid_to_row[qid]
        train_idx = [idx for idx, candidate in enumerate(rows) if str(candidate["qid"]) != qid]
        fold_best = choose_best_fixed_fold(rows, train_idx, policies, metric, tie_eps)
        chosen = str(item["chosen_policy"])
        choice_counts[chosen] += 1
        if item.get("fold_default_policy"):
            default_counts[str(item["fold_default_policy"])] += 1
        chosen_val = value(row, chosen, metric)
        best_val = value(row, fold_best, metric)
        chosen_values.append(chosen_val)
        stop_values.append(value(row, "stop", metric))
        best_fixed_values.append(best_val)
        group = row.get("evidence_bottleneck_group", "")
        gap = chosen_val - best_val
        if group.startswith("B_"):
            group_b_gaps.append(gap)
        if group.startswith("C_"):
            group_c_gaps.append(gap)
        if str(item.get("overrode_default")) == "1":
            default_policy = str(item["fold_default_policy"])
            chosen_minus_default_for_overrides.append(chosen_val - value(row, default_policy, metric))

    deltas_best = [a - b for a, b in zip(chosen_values, best_fixed_values)]
    ci = bootstrap_ci(deltas_best, bootstrap_samples, seed)
    override_values = [
        delta for delta in chosen_minus_default_for_overrides if not math.isnan(delta)
    ]
    override_rate = len(override_values) / max(1, len(method_rows))
    exception_precision = (
        sum(1 for delta in override_values if delta > eps) / len(override_values)
        if override_values
        else 0.0
    )
    return {
        "Method": method,
        "Metric": metric,
        "Mean": mean(chosen_values),
        "Delta_Stop": mean([a - b for a, b in zip(chosen_values, stop_values)]),
        "Delta_Best_Fixed": mean(deltas_best),
        "CI_vs_Best_Fixed": f"[{ci['ci_low']:.6f}, {ci['ci_high']:.6f}]",
        "Override_Rate": override_rate,
        "Exception_Precision": exception_precision,
        "B_Delta_Best_Fixed": mean(group_b_gaps),
        "C_Delta_Best_Fixed": mean(group_c_gaps),
        "Choices": json.dumps(dict(sorted(choice_counts.items())), sort_keys=True),
        "Fold_Defaults": json.dumps(dict(sorted(default_counts.items())), sort_keys=True),
        "Context": context,
    }


def margin_oracle_rows(
    qids: list[str],
    qid_to_row: dict[str, dict[str, str]],
    rows: list[dict[str, str]],
    policies: list[str],
    metric: str,
    margin_eps: float,
    tie_eps: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for qid in qids:
        row = qid_to_row[qid]
        train_idx = [idx for idx, candidate in enumerate(rows) if str(candidate["qid"]) != qid]
        default_policy = choose_best_fixed_fold(rows, train_idx, policies, metric, tie_eps)
        base = value(row, default_policy, metric)
        best_policy = default_policy
        best_delta = 0.0
        for policy in policies:
            if policy == default_policy:
                continue
            delta = value(row, policy, metric) - base
            if delta > best_delta + tie_eps or (abs(delta - best_delta) <= tie_eps and cost(row, policy) < cost(row, best_policy)):
                best_delta = delta
                best_policy = policy
        if best_delta <= margin_eps:
            best_policy = default_policy
        out.append(
            {
                "qid": qid,
                "method": f"Margin_Oracle_eps_{margin_eps:g}",
                "chosen_policy": best_policy,
                "fold_default_policy": default_policy,
                "overrode_default": int(best_policy != default_policy),
                "best_predicted_delta": best_delta,
                "inner_alpha": "",
                "inner_threshold": margin_eps,
                "predicted_deltas_json": "",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-matrix", type=Path, required=True)
    parser.add_argument("--stop-responses", type=Path, required=True)
    parser.add_argument("--stop-trace", type=Path, required=True)
    parser.add_argument("--raw-longmemeval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metric", choices=["reward", "proxy_utility_context"], default="reward")
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--encoder-model", required=True)
    parser.add_argument(
        "--encoder-backend",
        choices=["sentence_transformers", "transformers_mean"],
        default="sentence_transformers",
    )
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--encoder-max-length", type=int, default=512)
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--thresholds", default="0,0.025,0.05,0.075,0.1,0.15")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--margin-eps", type=float, default=0.1)
    parser.add_argument("--tie-eps", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--context-label", default="dev80_k20")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.wide_matrix)
    if not rows:
        raise ValueError(f"empty matrix: {args.wide_matrix}")
    policies = ["stop"] + [policy for policy in args.policy if policy != "stop"]
    qids = [normalize_qid(str(row.get("qid") or "")) for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate qids in wide matrix")
    manifest_qids = load_manifest_ids(args.manifest)
    if set(qids) - set(manifest_qids):
        raise ValueError("wide matrix contains qids outside manifest")
    raw_rows = load_raw_rows(args.raw_longmemeval, qids)
    responses = load_responses(args.stop_responses)
    traces = load_traces_by_query_sha1(args.stop_trace)

    for row, qid in zip(rows, qids):
        row["qid"] = qid
        row["evidence_bottleneck_group"] = group_label(row, "stop_retrieved_all_evidence_present")
    qid_to_row = {str(row["qid"]): row for row in rows}

    missing_cols: list[str] = []
    for policy in policies:
        metric_col = f"{policy}_reward" if args.metric == "reward" else f"{policy}_proxy_utility_context"
        if args.metric == "proxy_utility_context" and metric_col not in rows[0]:
            metric_col = f"{policy}_utility"
        for col in [metric_col, f"{policy}_cost"]:
            if col not in rows[0]:
                missing_cols.append(col)
    if missing_cols:
        raise ValueError(f"missing required columns: {sorted(set(missing_cols))}")

    x, feature_names, feature_audit = build_semantic_feature_table(
        qids,
        qid_to_row,
        raw_rows,
        responses,
        traces,
        encoder_name=args.encoder_model,
        encoder_backend=args.encoder_backend,
        embedding_cache=args.embedding_cache,
        batch_size=args.encoder_batch_size,
        max_length=args.encoder_max_length,
    )

    alphas = parse_float_list(args.alphas)
    thresholds = parse_float_list(args.thresholds)
    ssa_rows = run_ssa_loocv(
        x,
        qids,
        rows,
        policies,
        args.metric,
        alphas,
        thresholds,
        args.inner_folds,
        args.seed,
        args.tie_eps,
        args.margin_eps,
    )

    method_rows: list[dict[str, Any]] = []
    name_map = {
        "stop": "Greedy_STOP",
        "repack_lexical_bm25": "Lexical_BM25",
        "repack_tfidf_jaccard": "TFIDF_Jaccard",
        "repack_graph_bridge": "Graph_Bridge",
        "repack_temporal_session": "Temporal_Session",
    }
    for policy in policies:
        method_rows.extend(fixed_policy_rows(qids, name_map.get(policy, policy), policy, rows))
    method_rows.extend(ssa_rows)
    method_rows.extend(margin_oracle_rows(qids, qid_to_row, rows, policies, args.metric, args.margin_eps, args.tie_eps))

    summary_rows = []
    for method in sorted(set(str(row["method"]) for row in method_rows)):
        selected = [row for row in method_rows if str(row["method"]) == method]
        summary_rows.append(
            summarize_method(
                selected,
                qid_to_row,
                rows,
                policies,
                args.metric,
                method,
                args.eps,
                args.tie_eps,
                args.seed + len(summary_rows),
                args.bootstrap_samples,
                args.context_label,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"p30_ssa_method_table_{args.metric}.csv", summary_rows)
    write_csv(args.output_dir / f"p30_ssa_per_qid_{args.metric}.csv", method_rows)
    write_csv(args.output_dir / "p30_ssa_semantic_feature_audit.csv", feature_audit)
    (args.output_dir / "p30_ssa_feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "status": "EXPLORATORY_SSA_V1_DEFAULT_OVERRIDE_SEMANTIC_ONLINE_STATE",
        "wide_matrix": str(args.wide_matrix),
        "stop_responses": str(args.stop_responses),
        "stop_trace": str(args.stop_trace),
        "raw_longmemeval": str(args.raw_longmemeval),
        "manifest": str(args.manifest),
        "metric": args.metric,
        "n": len(rows),
        "policies": policies,
        "encoder_model": args.encoder_model,
        "encoder_backend": args.encoder_backend,
        "encoder_max_length": args.encoder_max_length,
        "feature_count": len(feature_names),
        "alphas": alphas,
        "thresholds": thresholds,
        "inner_folds": args.inner_folds,
        "inner_fold_design": "stratified_by_train_fold_margin_exception_no_resampling",
        "default_policy_design": "selected_inside_each_outer_train_fold",
        "eps": args.eps,
        "margin_eps": args.margin_eps,
        "tie_eps": args.tie_eps,
        "seed": args.seed,
        "method_table": summary_rows,
        "guardrails": [
            "Read-only over frozen generation artifacts; no prompt/operator/split/metric/generation changes.",
            "SSA chooses a fold-local default inside each outer train fold; no full-dev best-fixed leakage.",
            "SSA overrides the default only when predicted semantic margin exceeds threshold.",
            "Source indices are pointers to reconstruct C0/W0 texts, not raw numeric features.",
            "Semantic features use query/C0/W0 text available at inference time; no question_type/gold/answer/judge/outcome features.",
            "Inner folds are stratified for rare exception stability but final realized policy keeps original prevalence.",
            "This is Method 1 development evidence, not official judge or confirm evidence.",
            "A positive SSA result supports static semantic admission learnability, not RL necessity.",
        ],
    }
    (args.output_dir / f"p30_ssa_report_{args.metric}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
