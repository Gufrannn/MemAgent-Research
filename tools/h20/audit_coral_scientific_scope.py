#!/usr/bin/env python3
"""Read-only CORAL budget, leakage, and adaptive-benchmark audit.

This command never trains, evaluates, loads model weights, or modifies source
datasets.  It authenticates the exact parquet and frozen S128 manifest before
computing content-addressed set intersections.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recurrent.research.coral_scope_audit import (  # noqa: E402
    actual_budget, identity_inventory, overlap, parquet_row_identity,
    stable_row_identity, static_budget,
)
from recurrent.research.cosi import (  # noqa: E402
    canonical_sha256, sha256_file,
)
from recurrent.research.stable_eval_identity import validate_resolved_manifest  # noqa: E402


def _load_identity_rows(path: Path) -> list[dict[str, Any]]:
    """Read only outcome-free columns needed for content identity."""
    import pyarrow.parquet as parquet
    return parquet.read_table(path, columns=["prompt", "context", "extra_info"]).to_pylist()


def _exact_sha(path: Path, expected: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None:
        raise ValueError(f"CORAL_SCOPE_NO_GO: invalid expected SHA for {label}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"CORAL_SCOPE_NO_GO: {label} SHA mismatch")
    return actual


def _public_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inventory.items() if not isinstance(value, set)}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"CORAL_SCOPE_NO_GO: JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stable-resolved", required=True)
    parser.add_argument("--stable-resolved-sha256", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) is None:
        raise ValueError("CORAL_SCOPE_NO_GO: exact Git commit required")
    repo = Path(args.repo_dir).resolve()
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout
    if actual_commit != args.expected_commit or dirty:
        raise ValueError("CORAL_SCOPE_NO_GO: checkout commit/cleanliness binding")
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("branch") != "h20/qwen25-7b-cosi-t25-frozen-20260822":
        raise ValueError("CORAL_SCOPE_NO_GO: wrong method manifest")
    work = Path(args.work_root).resolve()
    train_path = work / "datasets/hotpotqa/hotpotqa_train_32k.parquet"
    s128_path = work / "datasets/hotpotqa/hotpotqa_dev.parquet"
    train_sha = _exact_sha(train_path, manifest["data"]["train_sha256"], "actor train")
    s128_sha = _exact_sha(s128_path, manifest["data"]["validation_sha256"], "S128 parquet")

    stable_path = Path(args.stable_resolved).resolve()
    stable_file_sha = _exact_sha(
        stable_path, args.stable_resolved_sha256, "stable resolved manifest"
    )
    stable = validate_resolved_manifest(_load_json(stable_path))
    if stable["eval_manifest_hash"] != manifest["evaluation"]["eval_manifest_hash"]:
        raise ValueError("CORAL_SCOPE_NO_GO: stable evaluation identity drift")
    stable_rows = stable["identity_payload"]["rows"]
    if len(stable_rows) != 128:
        raise ValueError("CORAL_SCOPE_NO_GO: stable evaluation is not S128")

    train_rows = _load_identity_rows(train_path)
    s128_rows = _load_identity_rows(s128_path)
    train_inventory = identity_inventory(parquet_row_identity(row) for row in train_rows)
    stable_inventory = identity_inventory(
        (stable_row_identity(row) for row in stable_rows), require_unique_content=True
    )
    s128_parquet_inventory = identity_inventory(
        (parquet_row_identity(row) for row in s128_rows), require_unique_content=True
    )
    if stable_inventory["content_identities"] != s128_parquet_inventory["content_identities"]:
        raise ValueError("CORAL_SCOPE_NO_GO: stable rows do not equal authenticated S128 parquet")

    empty = identity_inventory([])
    actor_overlap = overlap(train_inventory, stable_inventory)
    critic_overlap = overlap(empty, stable_inventory)
    # The benchmark's Original curve and Capture32 facts were already observed
    # while CORAL was framed.  It is therefore explicitly part of selection,
    # irrespective of whether the Method checkpoints themselves have run.
    selection_inventory = dict(train_inventory)
    selection_inventory["content_identities"] = (
        train_inventory["content_identities"] | stable_inventory["content_identities"]
    )
    selection_inventory["question_identities"] = (
        train_inventory["question_identities"] | stable_inventory["question_identities"]
    )
    selection_inventory["context_identities"] = (
        train_inventory["context_identities"] | stable_inventory["context_identities"]
    )
    selection_overlap = overlap(selection_inventory, stable_inventory)
    direct_clear = all(
        actor_overlap[field] == 0
        for field in ("canonical_content_pair_count", "question_hash_count", "context_hash_count")
    )

    report = {
        "schema": "memagent.coral.scientific-scope-audit.v1",
        "status": "PASS" if direct_clear else "FAIL",
        "decision": (
            "CORAL_SCOPE_DIRECT_LEAKAGE_CLEAR_S128_ADAPTIVE_DEV_ONLY"
            if direct_clear else "CORAL_SCOPE_NO_GO:DIRECT_OR_PARTIAL_ACTOR_TRAIN_S128_OVERLAP"
        ),
        "git_commit": args.expected_commit,
        "repository": str(repo),
        "method_manifest": {
            "path": str(manifest_path), "sha256": sha256_file(manifest_path),
        },
        "static_training_budget": static_budget(manifest),
        "actual_training_budget": actual_budget(
            None, None,
            expected_commit=args.expected_commit,
        ),
        "data_roles": {
            "actor_training": {
                "path": str(train_path), "parquet_sha256": train_sha,
                "resolved_row_manifest": "derived_read_only_content_inventory_in_this_report",
                **_public_inventory(train_inventory),
            },
            "critic_fit": {"status": "EMPTY_GRPO_HAS_NO_CRITIC", "row_count": 0},
            "prior_or_reference_fit": {
                "status": "EMPTY_REFERENCE_POLICY_IS_FROZEN_NOT_FIT", "row_count": 0,
            },
            "auxiliary_fit": {"status": "EMPTY_NO_LEARNED_AUXILIARY", "row_count": 0},
            "hyperparameter_and_method_selection": {
                "components": [
                    "E0 synthetic recurrent MDP (no HotpotQA rows)",
                    "E1 mechanism roots from authenticated actor-training parquet",
                    "previously observed Original/Capture32 fixed-S128 facts",
                ],
                "s128_is_in_selection_domain": True,
            },
            "early_stopping": {
                "status": "EMPTY_TEST_FREQ_MINUS_ONE_T5_HEALTH_HAS_NO_BENCHMARK",
                "row_count": 0,
            },
            "fixed_s128": {
                "parquet_path": str(s128_path), "parquet_sha256": s128_sha,
                "resolved_manifest_path": str(stable_path),
                "resolved_manifest_file_sha256": stable_file_sha,
                "eval_manifest_hash": stable["eval_manifest_hash"],
                **_public_inventory(stable_inventory),
            },
        },
        "set_intersections": {
            "actor_train_intersection_s128": actor_overlap,
            "critic_fit_intersection_s128": critic_overlap,
            "selection_intersection_s128": selection_overlap,
        },
        "adaptive_benchmark_classification": {
            "fixed_s128": "DEVELOPMENT_SCREEN_NOT_BLIND_FINAL_TEST",
            "reason": (
                "Authenticated Original and Capture32 S128 results were available during "
                "CORAL framing and gate design; later Method use is therefore adaptive reuse."
            ),
            "untouched_confirmation_required": True,
        },
    }
    unsigned = dict(report)
    report["report_sha256"] = canonical_sha256(unsigned)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if _load_json(output) != report:
            raise ValueError("CORAL_SCOPE_NO_GO: append-only report already differs")
    else:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
