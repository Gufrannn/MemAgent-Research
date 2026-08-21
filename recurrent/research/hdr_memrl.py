"""Fail-closed primitives for horizon-distributionally robust MemRL.

This module deliberately keeps scheduling, DRO state, and evaluation as pure,
serialisable operations.  GPU launch code consumes their receipts; it does not
infer horizon identity from sequence length or mutable row order.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class HDRContractError(ValueError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def stable_root_id(*, dataset_sha256: str, source_index: int, query: str) -> str:
    if len(dataset_sha256) != 64 or source_index < 0 or not query:
        raise HDRContractError("stable root requires dataset SHA, nonnegative source index, and query")
    return hashlib.sha256(f"{dataset_sha256}\0{source_index}\0{query}".encode()).hexdigest()


def _partition(n: int, horizon: int) -> tuple[tuple[int, int], ...]:
    if n <= 0 or horizon <= 0 or horizon > n:
        raise HDRContractError(f"illegal evidence-token/horizon pair: n={n}, horizon={horizon}")
    q, r = divmod(n, horizon)
    out, start = [], 0
    for i in range(horizon):
        width = q + (1 if i < r else 0)
        out.append((start, start + width))
        start += width
    return tuple(out)


@dataclass(frozen=True)
class HorizonReceipt:
    root_id: str
    horizon: int
    terminal_query_sha256: str
    evidence_sha256: str
    evidence_token_count: int
    chunk_bounds: tuple[tuple[int, int], ...]
    chunk_sha256: tuple[str, ...]
    chunks: tuple[tuple[int, ...], ...]

    def as_dict(self) -> dict:
        return {
            "root_id": self.root_id, "horizon": self.horizon,
            "terminal_query_sha256": self.terminal_query_sha256,
            "evidence_sha256": self.evidence_sha256,
            "evidence_token_count": self.evidence_token_count,
            "chunk_bounds": [list(x) for x in self.chunk_bounds],
            "chunk_sha256": list(self.chunk_sha256),
            "chunks": [list(x) for x in self.chunks],
        }


def build_horizon_receipt(root_id: str, query: str, evidence_tokens: Sequence[int], horizon: int) -> HorizonReceipt:
    if len(root_id) != 64 or not query:
        raise HDRContractError("invalid root/query")
    tokens = tuple(int(x) for x in evidence_tokens)
    bounds = _partition(len(tokens), int(horizon))
    chunks = [tokens[a:b] for a, b in bounds]
    if tuple(x for chunk in chunks for x in chunk) != tokens:
        raise HDRContractError("partition changed evidence order or coverage")
    return HorizonReceipt(
        root_id=root_id, horizon=int(horizon),
        terminal_query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        evidence_sha256=sha256_json(tokens), evidence_token_count=len(tokens),
        chunk_bounds=bounds, chunk_sha256=tuple(sha256_json(c) for c in chunks),
        chunks=tuple(chunks),
    )


def validate_evidence_equated(receipts: Sequence[HorizonReceipt], expected_horizons: Sequence[int]) -> dict:
    if not receipts:
        raise HDRContractError("empty horizon suite")
    roots: dict[str, list[HorizonReceipt]] = {}
    for r in receipts:
        roots.setdefault(r.root_id, []).append(r)
    expected = tuple(sorted(set(map(int, expected_horizons))))
    if not expected or expected[0] <= 0:
        raise HDRContractError("invalid frozen horizon set")
    pair_ids: set[tuple[str, int]] = set()
    for root, rows in roots.items():
        got = tuple(sorted(r.horizon for r in rows))
        if got != expected:
            raise HDRContractError(f"root {root} horizon closure mismatch: {got} != {expected}")
        if len(set((r.terminal_query_sha256, r.evidence_sha256, r.evidence_token_count) for r in rows)) != 1:
            raise HDRContractError(f"root {root} changed terminal query/evidence/token coverage")
        for r in rows:
            key = (root, r.horizon)
            if key in pair_ids:
                raise HDRContractError(f"duplicate root×horizon: {key}")
            pair_ids.add(key)
            if len(r.chunk_bounds) != r.horizon or r.chunk_bounds[0][0] != 0:
                raise HDRContractError(f"invalid chunk closure for {key}")
            if any(a >= b for a, b in r.chunk_bounds) or any(
                r.chunk_bounds[i][1] != r.chunk_bounds[i + 1][0] for i in range(len(r.chunk_bounds) - 1)
            ) or r.chunk_bounds[-1][1] != r.evidence_token_count:
                raise HDRContractError(f"accidental truncation/gap/overlap for {key}")
            if len(r.chunks) != r.horizon:
                raise HDRContractError(f"missing chunk payloads for {key}")
            flattened = tuple(token for chunk in r.chunks for token in chunk)
            if len(flattened) != r.evidence_token_count or sha256_json(flattened) != r.evidence_sha256:
                raise HDRContractError(f"chunk payload/evidence digest mismatch for {key}")
            if tuple(sha256_json(chunk) for chunk in r.chunks) != r.chunk_sha256:
                raise HDRContractError(f"chunk payload digest mismatch for {key}")
            if tuple(len(chunk) for chunk in r.chunks) != tuple(b-a for a,b in r.chunk_bounds):
                raise HDRContractError(f"chunk payload/boundary mismatch for {key}")
    return {"status": "PASS", "root_count": len(roots), "horizons": list(expected),
            "pair_count": len(pair_ids), "suite_sha256": sha256_json([r.as_dict() for r in receipts])}


class BalancedHorizonScheduler:
    """Deterministic Latin rotation: one horizon per root, fixed trajectories/update."""
    def __init__(self, horizons: Sequence[int], trajectories_per_update: int, seed: int):
        self.horizons = tuple(sorted(set(map(int, horizons))))
        if not self.horizons or self.horizons[0] <= 0 or trajectories_per_update <= 0:
            raise HDRContractError("invalid scheduler configuration")
        self.trajectories_per_update = int(trajectories_per_update)
        self.seed = int(seed)

    def assign(self, root_ids: Sequence[str], update: int) -> list[dict]:
        if len(root_ids) != self.trajectories_per_update or len(set(root_ids)) != len(root_ids):
            raise HDRContractError("update must contain exactly the budgeted number of unique stable roots")
        offset = (self.seed + int(update)) % len(self.horizons)
        ordered = sorted(root_ids, key=lambda r: hashlib.sha256(f"{self.seed}:{update}:{r}".encode()).digest())
        assigned = {root: self.horizons[(i + offset) % len(self.horizons)] for i, root in enumerate(ordered)}
        return [{"root_id": root, "horizon": assigned[root], "update": int(update)} for root in root_ids]


@dataclass
class OnlineGroupDRO:
    horizons: tuple[int, ...]
    eta: float
    rho: float
    weights: list[float]

    @classmethod
    def create(cls, horizons: Sequence[int], eta: float, rho: float) -> "OnlineGroupDRO":
        hs = tuple(sorted(set(map(int, horizons))))
        if not hs or eta <= 0 or rho < 0:
            raise HDRContractError("invalid DRO configuration")
        return cls(hs, float(eta), float(rho), [1 / len(hs)] * len(hs))

    def update(self, losses: Mapping[int, float], counts: Mapping[int, int]) -> dict:
        if set(losses) != set(self.horizons) or any(counts.get(h, 0) <= 0 for h in self.horizons):
            raise HDRContractError("every frozen horizon needs an observed finite group loss")
        vals = [float(losses[h]) for h in self.horizons]
        if not all(math.isfinite(x) for x in vals):
            raise HDRContractError("non-finite group loss")
        logits = [math.log(max(w, 1e-300)) + self.eta * loss for w, loss in zip(self.weights, vals)]
        m = max(logits); raw = [math.exp(x - m) for x in logits]; z = sum(raw)
        candidate = [x / z for x in raw]
        # KL-ball projection along the segment to nominal; deterministic bisection.
        nominal = [1 / len(candidate)] * len(candidate)
        def kl(ws): return sum(w * math.log(w / q) for w, q in zip(ws, nominal) if w > 0)
        if kl(candidate) > self.rho:
            lo, hi = 0.0, 1.0
            for _ in range(80):
                mid = (lo + hi) / 2
                ws = [(1-mid)*q + mid*w for q, w in zip(nominal, candidate)]
                if kl(ws) <= self.rho: lo = mid
                else: hi = mid
            candidate = [(1-lo)*q + lo*w for q, w in zip(nominal, candidate)]
        self.weights = candidate
        return self.state_dict()

    def sample_multipliers(self, horizon_ids: Sequence[int]) -> list[float]:
        counts = {h: horizon_ids.count(h) for h in self.horizons}
        if any(counts[h] == 0 for h in self.horizons):
            raise HDRContractError("actor batch missing a horizon")
        table = {h: self.weights[i] * len(horizon_ids) / counts[h] for i, h in enumerate(self.horizons)}
        return [table[int(h)] for h in horizon_ids]

    def state_dict(self) -> dict:
        return {"horizons": list(self.horizons), "eta": self.eta, "rho": self.rho,
                "weights": self.weights, "sha256": sha256_json(self.weights)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "OnlineGroupDRO":
        obj = cls(tuple(map(int, state["horizons"])), float(state["eta"]), float(state["rho"]),
                  list(map(float, state["weights"])))
        if abs(sum(obj.weights) - 1) > 1e-10 or any(w < 0 for w in obj.weights):
            raise HDRContractError("invalid checkpointed DRO simplex")
        return obj


def evaluate_horizons(rows: Iterable[Mapping[str, object]], nominal: int, unseen: Sequence[int]) -> dict:
    grouped: dict[int, list[dict]] = {}
    seen_pairs = set()
    root_horizons: dict[str, set[int]] = {}
    for row in rows:
        root, h = str(row["root_id"]), int(row["horizon"])
        if (root, h) in seen_pairs:
            raise HDRContractError("duplicate evaluator root×horizon")
        seen_pairs.add((root, h))
        root_horizons.setdefault(root, set()).add(h)
        if not bool(row.get("evidence_equated", False)) or bool(row.get("truncated", True)):
            raise HDRContractError("evaluator row lacks evidence-equated/no-truncation receipt")
        grouped.setdefault(h, []).append({k: float(row[k]) for k in ("em", "token_f1", "format")})
    if nominal not in grouped:
        raise HDRContractError("nominal horizon absent")
    expected=set(grouped)
    if any(hs != expected for hs in root_horizons.values()):
        raise HDRContractError("evaluator roots do not share complete horizon closure")
    metrics = {h: {k: sum(r[k] for r in rs)/len(rs) for k in ("em", "token_f1", "format")}
               for h, rs in grouped.items()}
    worst_h = min(metrics, key=lambda h: metrics[h]["token_f1"])
    missing_unseen = sorted(set(map(int, unseen)) - set(grouped))
    if missing_unseen:
        raise HDRContractError(f"unseen horizons absent: {missing_unseen}")
    return {"status": "PASS", "nominal_horizon": nominal, "nominal": metrics[nominal],
            "worst_horizon": worst_h, "worst": metrics[worst_h],
            "unseen": {h: metrics[h] for h in map(int, unseen)}, "by_horizon": metrics}


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def prediction_metrics(prediction: str, gold: str) -> dict[str, float]:
    pred, target = normalize_answer(prediction), normalize_answer(gold)
    em = float(pred == target)
    pt, gt = pred.split(), target.split()
    overlap = sum((Counter(pt) & Counter(gt)).values())
    f1 = 0.0 if overlap == 0 else 2 * overlap / (len(pt) + len(gt))
    valid = float(bool(re.search(r"\\boxed\{[^{}]+\}", str(prediction))))
    return {"em": em, "token_f1": f1, "format": valid}


def aggregate_predictions(rows: Sequence[Mapping[str, object]]) -> dict:
    if not rows:
        raise HDRContractError("empty predictions")
    ids = [str(r["stable_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise HDRContractError("duplicate stable prediction ID")
    vals = [prediction_metrics(str(r["prediction"]), str(r["gold"])) for r in rows]
    return {k: sum(v[k] for v in vals) / len(vals) for k in ("em", "token_f1", "format")} | {"count": len(vals)}


def write_json(path: str | Path, value: object) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
