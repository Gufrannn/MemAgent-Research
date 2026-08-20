"""Append-only, process-safe evidence helpers for the frozen H20 Gate A."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append exactly one JSON record under an advisory inter-process lock."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
    with target.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


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
        "recorded_at": utc_now(),
    }
    base.update(fields)
    append_jsonl(ledger, base)


def finite_numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if not any(token in key.lower() for token in ("reward", "adv", "grad", "loss", "kl")):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[str(key)] = float(value)
    return result


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
