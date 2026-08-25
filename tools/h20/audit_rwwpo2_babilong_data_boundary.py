#!/usr/bin/env python3
"""Audit exact canonical overlap between actor training roots and BABILong."""
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

from recurrent.research.stable_eval_identity import canonical_sha256, sha256_text
from tools.h20.audit_tf_rwwpo_budget_leakage import (
    EXPECTED_TRAIN_SHA256, _actor_consumed_rows, _read_parquet, sha256_file,
)
from tools.h20.materialize_rwwpo2_babilong import read_source_bundle


EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "manifest", "manifest-sha256", "adapter-manifest-sha256",
        "train", "tokenizer-root", "source-root", "source-manifest-sha256",
        "expected-commit", "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_BABILONG_BOUNDARY_NO_GO:checkout")
    raw = tuple(Path(value) for value in (
        args.manifest, args.train, args.tokenizer_root, args.source_root,
    ))
    output = Path(args.output)
    if any(path.is_symlink() for path in raw) or output.exists() or output.is_symlink():
        raise SystemExit("RWWPO2_BABILONG_BOUNDARY_NO_GO:symlink/append-only")
    manifest_path, train, tokenizer_root, source_root = (
        path.resolve() for path in raw
    )
    if sha256_file(manifest_path) != args.manifest_sha256 \
            or sha256_file(train) != EXPECTED_TRAIN_SHA256:
        raise SystemExit("RWWPO2_BABILONG_BOUNDARY_NO_GO:manifest/train SHA")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training = manifest.get("training", {})
    if manifest.get("program") != "RWWPO-2" \
            or manifest.get("data", {}).get("train_sha256") != EXPECTED_TRAIN_SHA256:
        raise SystemExit("RWWPO2_BABILONG_BOUNDARY_NO_GO:manifest identity")
    target_groups = int(training["target_rounds"]) * int(training["train_batch_size"])
    effective_filter = int(training["chunk_size"]) * int(training["max_chunks"])
    try:
        actor_rows, actor_scan = _actor_consumed_rows(
            _read_parquet(train), tokenizer_root=tokenizer_root,
            target_prompt_groups=target_groups,
            effective_max_prompt_length=effective_filter,
        )
        source_report, sources = read_source_bundle(
            source_root, expected_sha256=args.source_manifest_sha256,
            expected_commit=head,
            adapter_manifest_sha256=args.adapter_manifest_sha256,
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit("RWWPO2_BABILONG_BOUNDARY_NO_GO:" + str(error)) from error
    benchmark_rows = []
    for (length, task), rows in sorted(sources.items()):
        for source_index, row in enumerate(rows):
            question_hash = sha256_text(row["question"])
            context_hash = sha256_text(row["input"])
            target_hash = canonical_sha256([row["target"]])
            benchmark_rows.append({
                "cell_key": canonical_sha256([length, task, source_index]),
                "question_hash": question_hash, "context_hash": context_hash,
                "ground_truth_hash": target_hash,
                "root_key": canonical_sha256([question_hash, context_hash]),
                "content_key": canonical_sha256(
                    [question_hash, context_hash, target_hash]
                ),
            })
    actor_question = {row["question_hash"] for row in actor_rows}
    actor_context = {row["context_hash"] for row in actor_rows}
    actor_target = {row["ground_truth_hash"] for row in actor_rows}
    actor_root = {row["root_key"] for row in actor_rows}
    actor_content = {row["content_key"] for row in actor_rows}
    benchmark_question = {row["question_hash"] for row in benchmark_rows}
    benchmark_context = {row["context_hash"] for row in benchmark_rows}
    benchmark_target = {row["ground_truth_hash"] for row in benchmark_rows}
    benchmark_root = {row["root_key"] for row in benchmark_rows}
    benchmark_content = {row["content_key"] for row in benchmark_rows}
    intersections = {
        "train_intersect_babilong_question_hash": len(actor_question & benchmark_question),
        "train_intersect_babilong_context_hash": len(actor_context & benchmark_context),
        "train_intersect_babilong_target_hash": len(actor_target & benchmark_target),
        "train_intersect_babilong_root": len(actor_root & benchmark_root),
        "train_intersect_babilong_content": len(actor_content & benchmark_content),
    }
    direct = bool(
        intersections["train_intersect_babilong_root"]
        or intersections["train_intersect_babilong_content"]
    )
    report = {
        "schema_version": "rwwpo2-babilong-data-boundary-v1",
        "status": "NO_GO" if direct else "PASS",
        "decision": (
            "RWWPO2_BABILONG_DIRECT_DATA_LEAKAGE_DETECTED"
            if direct else "RWWPO2_BABILONG_DATA_BOUNDARY_AUDIT_PASS"
        ),
        "git_commit": head, "manifest_sha256": args.manifest_sha256,
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_report_sha256": source_report["report_sha256"],
        "actor_training": {
            "planned_rounds": int(training["target_rounds"]),
            "prompt_groups_per_round": int(training["train_batch_size"]),
            "maximum_consumed_prompt_roots": target_groups,
            **actor_scan,
        },
        "babilong": {
            "cells": 6, "rows": len(benchmark_rows),
            "unique_cell_keys": len({row["cell_key"] for row in benchmark_rows}),
            "development_is_adaptive": True,
            "confirmation_model_evaluation_forbidden_until_r400_complete": True,
        },
        "intersections": intersections,
        "direct_leakage": direct,
        "canonical_identity": {
            "root_key": "sha256(canonical(question_hash,context_hash))",
            "content_key": (
                "sha256(canonical(question_hash,context_hash,ground_truth_hash))"
            ),
        },
        "raw_examples_emitted": False,
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "decision": report["decision"],
        "intersections": intersections, "output": str(output.resolve()),
    }, sort_keys=True))
    raise SystemExit(1 if direct else 0)


if __name__ == "__main__":
    main()
