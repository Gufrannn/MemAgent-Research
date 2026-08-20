"""Append-only, process-safe evidence helpers for the frozen H20 Gate A."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from numbers import Real
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REQUIRED_RUNTIME_BINDINGS = ("MEMAGENT_GATEA_WORK_ROOT", "MEMAGENT_GATEA_REPO_DIR")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_environment(value: Any, environment: Mapping[str, str] | None = None) -> Any:
    """Resolve only the task-scoped H20 path bindings; reject implicit defaults."""
    source = os.environ if environment is None else environment
    missing = [name for name in REQUIRED_RUNTIME_BINDINGS if not source.get(name)]
    if missing:
        raise ValueError(f"missing explicit runtime path bindings: {missing}")
    bindings = {name: str(source[name]) for name in REQUIRED_RUNTIME_BINDINGS}
    non_absolute = [name for name, path in bindings.items() if not Path(path).is_absolute()]
    if non_absolute:
        raise ValueError(f"runtime path bindings must be absolute: {non_absolute}")

    def resolve(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: resolve(child) for key, child in item.items()}
        if isinstance(item, list):
            return [resolve(child) for child in item]
        if isinstance(item, str):
            result = item
            for name, path in bindings.items():
                result = result.replace(f"${{{name}}}", path)
            if "${" in result:
                raise ValueError(f"unresolved manifest placeholder: {result}")
            return result
        return item

    return resolve(value)


def load_frozen_manifest(path: str | Path, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    source = os.environ if environment is None else environment
    expected_commit = str(source.get("MEMAGENT_GATEA_EXPECTED_COMMIT", ""))
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError(
            "missing or invalid explicit runtime binding: MEMAGENT_GATEA_EXPECTED_COMMIT"
        )
    return resolve_manifest_environment(raw, environment)


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one canonical, hash-chained JSON record under a process lock."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    supplied = dict(record)
    reserved = {"record_index", "previous_record_sha256", "record_sha256"} & supplied.keys()
    if reserved:
        raise ValueError(f"hash-chain fields are writer-owned: {sorted(reserved)}")
    with target.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        lines = [line for line in stream.read().splitlines() if line.strip()]
        if lines:
            previous = json.loads(lines[-1])
            previous_digest = previous.get("record_sha256")
            previous_index = previous.get("record_index")
            if not isinstance(previous_index, int) or previous_index != len(lines) - 1:
                raise ValueError("existing Gate A ledger has a non-contiguous record index")
            if previous_digest != record_sha256(previous):
                raise ValueError("existing Gate A ledger tail hash is invalid")
        else:
            previous_digest = "0" * 64
        chained = {
            **supplied,
            "record_index": len(lines),
            "previous_record_sha256": previous_digest,
        }
        chained["record_sha256"] = record_sha256(chained)
        payload = json.dumps(chained, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.seek(0, os.SEEK_END)
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def record_sha256(record: Mapping[str, Any]) -> str:
    """Digest a ledger record while excluding its self-authenticating digest field."""
    payload = dict(record)
    payload.pop("record_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_jsonl_chain(records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    previous_digest = "0" * 64
    for index, record in enumerate(records):
        if record.get("record_index") != index:
            failures.append(
                f"ledger record index mismatch at position {index}: {record.get('record_index')}"
            )
        if record.get("previous_record_sha256") != previous_digest:
            failures.append(f"ledger previous hash mismatch at record {index}")
        actual_digest = record_sha256(record)
        if record.get("record_sha256") != actual_digest:
            failures.append(f"ledger record hash mismatch at record {index}")
        previous_digest = actual_digest
    return failures


def gate_a_enabled() -> bool:
    return os.getenv("GATE_A_FROZEN_AUDIT", "0") == "1"


def append_gate_a_record(record_type: str, **fields: Any) -> None:
    if not gate_a_enabled():
        return
    ledger = os.environ["GATE_A_EXECUTION_LEDGER"]
    base = {
        "record_type": record_type,
        "experiment_name": os.environ["GATE_A_EXPERIMENT_NAME"],
        "git_commit": os.environ["GATE_A_GIT_COMMIT"],
        "run_id": os.environ["GATE_A_RUN_ID"],
        "recorded_at": utc_now(),
    }
    base.update(fields)
    append_jsonl(ledger, base)


def partition_numeric_metrics(metrics: Mapping[str, Any]) -> tuple[dict[str, float], list[str]]:
    finite: dict[str, float] = {}
    nonfinite: list[str] = []
    for key, value in metrics.items():
        if not any(token in key.lower() for token in ("reward", "adv", "grad", "loss", "kl")):
            continue
        if not isinstance(value, Real) and hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                continue
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        if math.isfinite(float(value)):
            finite[str(key)] = float(value)
        else:
            nonfinite.append(str(key))
    return finite, sorted(nonfinite)


def finite_numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Compatibility helper; formal evidence must also record non-finite names."""
    return partition_numeric_metrics(metrics)[0]


def checkpoint_inventory(step_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(step_dir)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
