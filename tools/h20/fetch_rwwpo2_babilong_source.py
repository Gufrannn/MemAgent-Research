#!/usr/bin/env python3
"""Fetch the pinned official 100-row BABILong source into an append-only bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.rwwpo2_babilong import (
    LENGTHS, SOURCE_DATASET_ID, SOURCE_REVISION, SOURCE_ROWS_PER_CELL,
    TASK_DEPTH, validate_frozen_contract,
)
from recurrent.research.stable_eval_identity import canonical_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_BABILONG_SOURCE_NO_GO:checkout")
    manifest_path = Path(args.manifest).resolve()
    root = Path(args.output_root)
    if manifest_path.is_symlink() or root.exists() or root.is_symlink():
        raise SystemExit("RWWPO2_BABILONG_SOURCE_NO_GO:symlink/one-use root")
    if sha256_file(manifest_path) != args.manifest_sha256:
        raise SystemExit("RWWPO2_BABILONG_SOURCE_NO_GO:manifest SHA")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frozen_contract(manifest)
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("RWWPO2_BABILONG_SOURCE_NO_GO:datasets package missing") from error
    root.mkdir(parents=True)
    inventory = []
    for length in LENGTHS:
        for task in TASK_DEPTH:
            dataset = load_dataset(
                SOURCE_DATASET_ID, name=length, split=task,
                revision=SOURCE_REVISION, trust_remote_code=False,
            )
            if len(dataset) != SOURCE_ROWS_PER_CELL:
                raise SystemExit(
                    f"RWWPO2_BABILONG_SOURCE_NO_GO:{length}/{task} row count {len(dataset)}"
                )
            rows = []
            for index, source in enumerate(dataset):
                if any(not isinstance(source.get(field), str) or not source[field].strip()
                       for field in ("input", "question", "target")):
                    raise SystemExit(
                        f"RWWPO2_BABILONG_SOURCE_NO_GO:{length}/{task}/{index} schema"
                    )
                rows.append({field: source[field] for field in ("input", "question", "target")})
            path = root / "source" / length / f"{task}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            inventory.append({
                "length": length, "task": task, "rows": len(rows),
                "relative_path": str(path.relative_to(root)),
                "file_sha256": sha256_file(path),
                "canonical_rows_sha256": canonical_sha256(rows),
            })
    bundle = {
        "schema_version": "rwwpo2-babilong-source-bundle-v1",
        "status": "PASS", "decision": "RWWPO2_BABILONG_SOURCE_BUNDLE_PASS",
        "git_commit": head, "adapter_manifest_sha256": args.manifest_sha256,
        "dataset_id": SOURCE_DATASET_ID, "dataset_revision": SOURCE_REVISION,
        "cells": inventory, "cell_inventory_sha256": canonical_sha256(inventory),
    }
    bundle["report_sha256"] = hashlib.sha256(json.dumps(
        bundle, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    path = root / "bundle_manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(bundle, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": bundle["decision"],
        "output": str(root.resolve()), "bundle_manifest_sha256": sha256_file(path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
