#!/usr/bin/env python3
"""Read-only content/root overlap and adaptive-use audit for RWWPO-2.

The report contains only paths, hashes, and counts.  It never serializes a
question, context, answer, or reward target.
"""
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

from tools.h20.audit_tf_rwwpo_budget_leakage import (
    EXPECTED_S128_DATA_SHA256,
    EXPECTED_S128_MANIFEST_HASH,
    EXPECTED_S128_RESOLVED_SHA256,
    EXPECTED_S128_SIZE,
    EXPECTED_TRAIN_SHA256,
    _actor_consumed_rows,
    _load_s128,
    _read_parquet,
    overlap_counts,
    sha256_file,
)


EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"


def sign(row: dict) -> dict:
    raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**row, "report_sha256": hashlib.sha256(raw.encode()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--tokenizer-root", required=True)
    parser.add_argument("--s128-data", required=True)
    parser.add_argument("--s128-resolved", required=True)
    parser.add_argument("--s128-resolved-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_DATA_BOUNDARY_NO_GO:checkout")
    raw_inputs = tuple(Path(value) for value in (
        args.manifest, args.train, args.tokenizer_root,
        args.s128_data, args.s128_resolved,
    ))
    if any(path.is_symlink() for path in raw_inputs):
        raise SystemExit("RWWPO2_DATA_BOUNDARY_NO_GO:source symlink")
    manifest_path, train_path, tokenizer_root, s128_data, s128_resolved = (
        path.resolve() for path in raw_inputs
    )
    if sha256_file(manifest_path) != args.manifest_sha256:
        raise SystemExit("RWWPO2_DATA_BOUNDARY_NO_GO:manifest SHA")
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA256:
        raise SystemExit("RWWPO2_DATA_BOUNDARY_NO_GO:train SHA")
    if sha256_file(s128_data) != EXPECTED_S128_DATA_SHA256:
        raise SystemExit("RWWPO2_DATA_BOUNDARY_NO_GO:S128 data SHA")
    if args.s128_resolved_sha256 != EXPECTED_S128_RESOLVED_SHA256 \
            or sha256_file(s128_resolved) != EXPECTED_S128_RESOLVED_SHA256:
        raise SystemExit("RWWPO2_DATA_BOUNDARY_NO_GO:S128 resolved SHA")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("program") != "RWWPO-2" \
            or manifest.get("data", {}).get("train_sha256") != EXPECTED_TRAIN_SHA256 \
            or manifest.get("data", {}).get("s128_sha256") != EXPECTED_S128_DATA_SHA256:
        raise SystemExit("RWWPO2_DATA_BOUNDARY_NO_GO:manifest data identity")

    training = manifest["training"]
    target_groups = int(training["target_rounds"]) * int(training["train_batch_size"])
    effective_filter = int(training["chunk_size"]) * int(training["max_chunks"])
    if effective_filter != int(training["runtime_effective_prompt_filter_length"]):
        raise SystemExit("RWWPO2_DATA_BOUNDARY_NO_GO:effective prompt filter")
    actor_rows, scan = _actor_consumed_rows(
        _read_parquet(train_path), tokenizer_root=tokenizer_root,
        target_prompt_groups=target_groups,
        effective_max_prompt_length=effective_filter,
    )
    resolved, s128_rows = _load_s128(s128_resolved)
    overlaps = overlap_counts(actor_rows, s128_rows)
    direct = bool(
        overlaps["train_intersect_s128_content"]
        or overlaps["train_intersect_s128_root"]
    )
    report = sign({
        "schema_version": "rwwpo2-data-boundary-v1",
        "status": "NO_GO" if direct else "PASS",
        "decision": (
            "RWWPO2_DIRECT_DATA_LEAKAGE_DETECTED"
            if direct else "RWWPO2_DATA_BOUNDARY_AUDIT_PASS"
        ),
        "git_commit": head,
        "manifest_path": str(manifest_path),
        "manifest_sha256": args.manifest_sha256,
        "actor_training": {
            "path": str(train_path), "sha256": sha256_file(train_path),
            "sampler": "SequentialSampler", "shuffle": False,
            "planned_rounds": int(training["target_rounds"]),
            "prompt_groups_per_round": int(training["train_batch_size"]),
            "rollout_trajectories_per_prompt": int(training["rollout_n"]),
            "maximum_consumed_prompt_roots": target_groups,
            "maximum_rollout_trajectories": target_groups * int(training["rollout_n"]),
            "memory_dataset_prompt_filter_length_after_hydra_mutation": effective_filter,
            "manifest_prompt_length_before_mutation": int(training["max_prompt_length"]),
            **scan,
        },
        "critic_fit": {"path": None, "sha256": None, "optimizer_updates": 0},
        "prior_fit": {"path": None, "sha256": None, "optimizer_updates": 0},
        "auxiliary_fit": {"path": None, "sha256": None, "optimizer_updates": 0},
        "hyperparameter_selection": {
            "uses_s128": True,
            "basis": "K1 T5/T10/T15/T20/T25 results were viewed before RWWPO-2",
            "performance_use_during_r50_or_training": False,
        },
        "early_stopping": {
            "performance_based": False, "validation_dataset": None,
            "test_freq": -1, "val_before_train": False,
        },
        "fixed_s128": {
            "data_path": str(s128_data),
            "data_sha256": sha256_file(s128_data),
            "resolved_path": str(s128_resolved),
            "resolved_sha256": sha256_file(s128_resolved),
            "eval_manifest_hash": resolved["eval_manifest_hash"],
            "rows": EXPECTED_S128_SIZE,
        },
        "canonical_identity": {
            "root_key": "sha256(canonical(question_hash,context_hash))",
            "content_key": "sha256(canonical(question_hash,context_hash,ground_truth_hash))",
            "dataset_local_index_is_diagnostic_only": True,
        },
        "intersections": {
            **overlaps,
            "critic_fit_intersect_s128_content": 0,
            "critic_fit_intersect_s128_root": 0,
            "prior_fit_intersect_s128_content": 0,
            "prior_fit_intersect_s128_root": 0,
            "auxiliary_fit_intersect_s128_content": 0,
            "auxiliary_fit_intersect_s128_root": 0,
            "selection_intersect_s128_content": EXPECTED_S128_SIZE,
            "selection_intersect_s128_root": EXPECTED_S128_SIZE,
        },
        "direct_leakage": direct,
        "s128_role": "ADAPTIVE_DEVELOPMENT_NOT_BLIND_FINAL",
        "blind_final_claim_authorized": False,
        "confirmatory_set": "PENDING_SEALED_AT_LEAST_512_DISJOINT_ROWS",
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "decision": report["decision"],
        "direct_leakage": direct, "intersections": report["intersections"],
        "output": str(output.resolve()),
    }, sort_keys=True))
    raise SystemExit(1 if direct else 0)


if __name__ == "__main__":
    main()
