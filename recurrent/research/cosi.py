"""Fail-closed primitives for Cross-Occupancy Update Certificates (COSI).

This module deliberately contains no reward model and no candidate selector.  It
validates an externally produced, content-addressed four-cell replay, computes
root-clustered contrasts, and decides whether an immutable provisional
checkpoint may become the next accepted checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "memagent.cosi.four-cell.v1"
DECISION_SCHEMA = "memagent.cosi.decision.v1"
LEDGER_SCHEMA = "memagent.cosi.ledger.v1"
CELLS = ("OO", "NO", "ON", "NN")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"COSI_NO_GO: {field} must be lowercase SHA-256")
    return value


def derive_seed(*, base_seed: int, root_id: str, writer: str, replica: int,
                phase: str, turn: int = 0) -> int:
    if writer not in ("O", "N") or phase not in ("writer", "future"):
        raise ValueError("COSI_NO_GO: invalid seed namespace")
    payload = ["cosi-seed-v1", int(base_seed), root_id, writer, int(replica), phase, int(turn)]
    return int.from_bytes(hashlib.sha256(canonical_json(payload).encode()).digest()[:8], "big")


def checkpoint_inventory(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"COSI_NO_GO: checkpoint is not a directory: {root}")
    result = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        result.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not result:
        raise ValueError("COSI_NO_GO: checkpoint inventory is empty")
    return result


def checkpoint_sha256(root: str | Path) -> str:
    return canonical_sha256(checkpoint_inventory(root))


def _finite_score(value: Any, field: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"COSI_NO_GO: {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < low or number > high:
        raise ValueError(f"COSI_NO_GO: {field} outside [{low},{high}]")
    return number


def validate_four_cell_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "contract", "records", "bundle_sha256"}
    if not isinstance(bundle, Mapping) or set(bundle) != required:
        raise ValueError("COSI_NO_GO: four-cell bundle fields drifted")
    unsigned = {k: v for k, v in bundle.items() if k != "bundle_sha256"}
    require_sha256(bundle["bundle_sha256"], "bundle_sha256")
    if bundle["schema"] != SCHEMA or bundle["bundle_sha256"] != canonical_sha256(unsigned):
        raise ValueError("COSI_NO_GO: four-cell bundle authentication failed")
    contract = bundle["contract"]
    contract_fields = {
        "old_weight_sha256", "new_weight_sha256", "base_seed", "reward_low",
        "reward_high", "candidate_sampling", "future_seed_coupling",
        "root_inventory_sha256", "git_commit", "transport_manifest_sha256",
    }
    if not isinstance(contract, Mapping) or set(contract) != contract_fields:
        raise ValueError("COSI_NO_GO: transport contract fields drifted")
    old_weight = require_sha256(contract["old_weight_sha256"], "old_weight_sha256")
    new_weight = require_sha256(contract["new_weight_sha256"], "new_weight_sha256")
    require_sha256(contract["root_inventory_sha256"], "root_inventory_sha256")
    require_sha256(contract["transport_manifest_sha256"], "transport_manifest_sha256")
    if old_weight == new_weight:
        raise ValueError("COSI_NO_GO: old/new weights are identical; method inactive")
    if contract["candidate_sampling"] != "cache_once_no_resample" \
            or contract["future_seed_coupling"] != "common_within_root_writer_replica":
        raise ValueError("COSI_NO_GO: replay coupling contract drifted")
    if not isinstance(contract["base_seed"], int) or isinstance(contract["base_seed"], bool):
        raise ValueError("COSI_NO_GO: base_seed must be an integer")
    if not isinstance(contract["git_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", contract["git_commit"]) is None:
        raise ValueError("COSI_NO_GO: git_commit must be exact SHA")
    low = _finite_score(contract["reward_low"], "reward_low", -1e100, 1e100)
    high = _finite_score(contract["reward_high"], "reward_high", -1e100, 1e100)
    if not low < high:
        raise ValueError("COSI_NO_GO: reward bounds invalid")

    rows = bundle["records"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("COSI_NO_GO: records must be non-empty")
    exact_fields = {
        "root_id", "replica", "cell", "writer_checkpoint", "continuation_checkpoint",
        "candidate_sha256", "candidate_token_ids_sha256", "writer_seed", "future_seeds",
        "writer_weight_sha256", "continuation_weight_sha256", "score", "score_evidence_sha256",
    }
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    roots = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != exact_fields:
            raise ValueError(f"COSI_NO_GO: record {index} fields drifted")
        root = row["root_id"]
        replica = row["replica"]
        cell = row["cell"]
        if not isinstance(root, str) or not root or not isinstance(replica, int) or isinstance(replica, bool) or replica < 0:
            raise ValueError("COSI_NO_GO: invalid root/replica identity")
        if cell not in CELLS or cell in grouped[(root, replica)]:
            raise ValueError("COSI_NO_GO: invalid or duplicate cell")
        writer, continuation = cell
        if row["writer_checkpoint"] != writer or row["continuation_checkpoint"] != continuation:
            raise ValueError("COSI_NO_GO: cell/checkpoint role mismatch")
        require_sha256(row["candidate_sha256"], "candidate_sha256")
        require_sha256(row["candidate_token_ids_sha256"], "candidate_token_ids_sha256")
        require_sha256(row["score_evidence_sha256"], "score_evidence_sha256")
        expected_writer_weight = old_weight if writer == "O" else new_weight
        expected_cont_weight = old_weight if continuation == "O" else new_weight
        if row["writer_weight_sha256"] != expected_writer_weight \
                or row["continuation_weight_sha256"] != expected_cont_weight:
            raise ValueError("COSI_NO_GO: cell weight authentication failed")
        expected_writer_seed = derive_seed(base_seed=contract["base_seed"], root_id=root,
                                           writer=writer, replica=replica, phase="writer")
        if row["writer_seed"] != expected_writer_seed:
            raise ValueError("COSI_NO_GO: writer seed drifted")
        if not isinstance(row["future_seeds"], list) or not row["future_seeds"] \
                or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in row["future_seeds"]):
            raise ValueError("COSI_NO_GO: future seed schedule invalid")
        _finite_score(row["score"], "score", low, high)
        grouped[(root, replica)][cell] = row
        roots.add(root)
    if contract["root_inventory_sha256"] != canonical_sha256(sorted(roots)):
        raise ValueError("COSI_NO_GO: root inventory authentication failed")
    for identity, cells in grouped.items():
        if set(cells) != set(CELLS):
            raise ValueError(f"COSI_NO_GO: incomplete four-cell support for {identity}")
        for writer, pair in (("O", ("OO", "ON")), ("N", ("NO", "NN"))):
            left, right = (cells[name] for name in pair)
            for field in ("candidate_sha256", "candidate_token_ids_sha256", "writer_seed"):
                if left[field] != right[field]:
                    raise ValueError(f"COSI_NO_GO: cached candidate mismatch for {identity} writer={writer}")
            if left["future_seeds"] != right["future_seeds"]:
                raise ValueError(f"COSI_NO_GO: common future seeds mismatch for {identity} writer={writer}")
    return json.loads(canonical_json(bundle))


def root_contrasts(bundle: Mapping[str, Any], *, closure_tol: float = 1e-12) -> list[dict[str, float | str]]:
    checked = validate_four_cell_bundle(bundle)
    replicas: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in checked["records"]:
        replicas[(row["root_id"], row["replica"])][row["cell"]] = float(row["score"])
    roots: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (root, _), y in replicas.items():
        values = {
            "writer_old": y["NO"] - y["OO"],
            "continuation_old": y["ON"] - y["OO"],
            "interaction": y["NN"] - y["NO"] - y["ON"] + y["OO"],
            "closed": y["NN"] - y["OO"],
        }
        if abs(values["closed"] - values["writer_old"] - values["continuation_old"] - values["interaction"]) > closure_tol:
            raise ValueError("COSI_NO_GO: decomposition did not close")
        roots[root].append(values)
    result = []
    for root, values in sorted(roots.items()):
        result.append({"root_id": root, **{
            key: math.fsum(item[key] for item in values) / len(values)
            for key in ("writer_old", "continuation_old", "interaction", "closed")
        }})
    return result


def hoeffding_lcb(values: Sequence[float], *, alpha: float, low: float, high: float) -> float:
    if not values or not 0 < alpha < 1 or not low < high:
        raise ValueError("COSI_NO_GO: invalid LCB inputs")
    if any(not math.isfinite(v) or v < low or v > high for v in values):
        raise ValueError("COSI_NO_GO: LCB value outside registered bounds")
    mean = math.fsum(values) / len(values)
    radius = (high - low) * math.sqrt(math.log(1.0 / alpha) / (2.0 * len(values)))
    return mean - radius


def decide(bundle: Mapping[str, Any], *, alpha: float, delta: float,
           attempt: int, alpha_schedule: Sequence[float]) -> dict[str, Any]:
    checked = validate_four_cell_bundle(bundle)
    if not isinstance(attempt, int) or attempt < 0 or attempt >= len(alpha_schedule):
        raise ValueError("COSI_NO_GO: invalid preregistered attempt")
    if abs(float(alpha_schedule[attempt]) - float(alpha)) > 1e-15:
        raise ValueError("COSI_NO_GO: alpha differs from preregistered spending schedule")
    contrasts = root_contrasts(checked)
    reward_range = float(checked["contract"]["reward_high"]) - float(checked["contract"]["reward_low"])
    closed = [float(row["closed"]) for row in contrasts]
    lcb = hoeffding_lcb(closed, alpha=alpha, low=-reward_range, high=reward_range)
    decision = "ACCEPT" if lcb >= -float(delta) else ("BACKTRACK" if attempt + 1 < len(alpha_schedule) else "ROLLBACK")
    unsigned = {
        "schema": DECISION_SCHEMA, "bundle_sha256": checked["bundle_sha256"],
        "root_count": len(contrasts), "alpha": float(alpha), "delta": float(delta),
        "attempt": attempt, "alpha_schedule": list(alpha_schedule), "lcb_method": "root_hoeffding_v1",
        "closed_mean": math.fsum(closed) / len(closed), "closed_lcb": lcb,
        "reversal_root_count": sum(float(row["writer_old"]) > 0 and float(row["closed"]) < 0 for row in contrasts),
        "decision": decision,
    }
    return {**unsigned, "decision_sha256": canonical_sha256(unsigned)}


def append_ledger(path: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = "0" * 64
    sequence = 0
    if path.exists():
        rows = validate_ledger(path)
        previous = rows[-1]["record_sha256"]
        sequence = len(rows)
    unsigned = {"schema": LEDGER_SCHEMA, "sequence": sequence,
                "previous_sha256": previous, "payload": dict(record)}
    signed = {**unsigned, "record_sha256": canonical_sha256(unsigned)}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (canonical_json(signed) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return signed


def validate_ledger(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    previous = "0" * 64
    with Path(path).open(encoding="utf-8") as stream:
        for sequence, line in enumerate(stream):
            row = json.loads(line)
            if set(row) != {"schema", "sequence", "previous_sha256", "payload", "record_sha256"}:
                raise ValueError("COSI_NO_GO: ledger fields drifted")
            unsigned = {k: v for k, v in row.items() if k != "record_sha256"}
            if row["schema"] != LEDGER_SCHEMA or row["sequence"] != sequence \
                    or row["previous_sha256"] != previous \
                    or row["record_sha256"] != canonical_sha256(unsigned):
                raise ValueError("COSI_NO_GO: ledger chain invalid")
            previous = row["record_sha256"]
            rows.append(row)
    if not rows:
        raise ValueError("COSI_NO_GO: ledger is empty")
    return rows
