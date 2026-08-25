#!/usr/bin/env python3
"""Materialize the label-blind MIC-v2 data split and overlap certificate.

This entry is CPU/read-only with respect to source parquet files.  It emits
only hashes and source positions, never question, context, or answer text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from recurrent.research.mic_v2 import (
    CONTRACT_SHA256,
    SCHEMA,
    canonical_json,
    sha256_file,
    sha256_json,
    write_json_new,
)


MANIFEST_REL = Path("manifests/h20/qwen25_7b_mic_v2_data_freeze.json")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SPLITS = (
    ("e1_dev", 128),
    ("e1_holdout", 128),
    ("formal_branching_oracle", 64),
    ("reference_length_calibration", 64),
    ("actor_training_b1_b8", 512),
    ("sealed_confirm", 512),
)
E0_IDS = tuple(f"E0-{index}" for index in range(1, 14))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MIC_V2_NO_GO: {message}")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _sha256_text(text: str) -> str:
    import hashlib

    _require(isinstance(text, str), "dataset question/context is not text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON authority is not an object: {path}")
    return value


def _require_json_finite(value: Any, path: str = "payload") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"non-finite value at {path}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_json_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_finite(item, f"{path}[{index}]")


def _verify_self_digest(payload: Mapping[str, Any], field: str, message: str) -> str:
    value = payload.get(field)
    _require(isinstance(value, str) and HEX64_PATTERN.fullmatch(value) is not None, message)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    _require(value == sha256_json(unsigned), message)
    return value


def _verify_e0(repo: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    authority = manifest["e0_authority"]
    path = Path(authority["certificate_path"])
    _require(path.is_absolute() and path.is_file(), "E0 certificate is unavailable")
    _require(sha256_file(path) == authority["certificate_file_sha256"],
             "E0 certificate file SHA differs")
    certificate = _load_json(path)
    _require_json_finite(certificate)
    digest = _verify_self_digest(
        certificate, "certificate_sha256", "E0 canonical certificate SHA differs",
    )
    _require(digest == authority["certificate_canonical_sha256"],
             "E0 canonical authority differs")
    preregistration = repo / authority["preregistration_manifest_path"]
    _require(sha256_file(preregistration) == authority["preregistration_manifest_sha256"],
             "E0 preregistration manifest SHA differs")
    _require(certificate.get("schema") == SCHEMA
             and certificate.get("kind") == "e0_certificate"
             and certificate.get("status") == "PASS"
             and certificate.get("decision") == "MIC_V2_E0_PASS",
             "E0 certificate is not PASS")
    _require(certificate.get("git_commit") == authority["e0_git_commit"]
             and certificate.get("run_id") == authority["run_id"]
             and certificate.get("output_path") == str(path)
             and certificate.get("contract_sha256") == CONTRACT_SHA256
             and certificate.get("preregistration_manifest_sha256")
                 == authority["preregistration_manifest_sha256"],
             "E0 certificate identity differs")
    tests = certificate.get("tests", [])
    _require([row.get("id") for row in tests] == list(E0_IDS)
             and all(row.get("status") == "PASS" for row in tests),
             "E0 certificate coverage differs")
    for row in tests:
        _require(row.get("evidence_sha256") == sha256_json(row.get("evidence")),
                 f"E0 evidence digest differs for {row.get('id')}")
    return {
        "path": str(path),
        "file_sha256": authority["certificate_file_sha256"],
        "canonical_sha256": digest,
        "git_commit": certificate["git_commit"],
        "run_id": certificate["run_id"],
    }


def _iter_parquet_rows(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    import pyarrow.parquet as pq

    position = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=64, columns=["prompt", "context", "reward_model", "extra_info"],
    ):
        for row in batch.to_pylist():
            yield position, row
            position += 1


def _identity(position: int, row: Mapping[str, Any], namespace: str) -> dict[str, Any]:
    context = row.get("context")
    prompt = row.get("prompt")
    extra = row.get("extra_info")
    reward = row.get("reward_model")
    _require(isinstance(extra, Mapping) and isinstance(reward, Mapping),
             f"dataset nested schema differs at row {position}")
    _require(isinstance(prompt, list) and len(prompt) == 1
             and isinstance(prompt[0], Mapping)
             and prompt[0].get("role") == "user"
             and isinstance(prompt[0].get("content"), str),
             f"dataset prompt schema differs at row {position}")
    question = prompt[0]["content"]
    auxiliary_question = extra.get("question")
    _require(auxiliary_question is None or auxiliary_question == question,
             f"extra_info.question differs from policy prompt at row {position}")
    ground_truth = reward.get("ground_truth")
    _require(isinstance(question, str) and isinstance(context, str),
             f"dataset text schema differs at row {position}")
    _require(isinstance(ground_truth, list) and ground_truth
             and all(isinstance(item, str) for item in ground_truth),
             f"dataset ground-truth schema differs at row {position}")
    question_sha = _sha256_text(question)
    context_sha = _sha256_text(context)
    ground_truth_sha = sha256_json(ground_truth)
    content_root = sha256_json({
        "namespace": namespace,
        "question_sha256": question_sha,
        "context_sha256": context_sha,
    })
    full_example = sha256_json({
        "namespace": namespace,
        "question_sha256": question_sha,
        "context_sha256": context_sha,
        "ground_truth_sha256": ground_truth_sha,
    })
    semantic_index = extra.get("index")
    _require(isinstance(semantic_index, int) and not isinstance(semantic_index, bool),
             f"dataset semantic index differs at row {position}")
    return {
        "source_position": position,
        "semantic_dataset_index": semantic_index,
        "question_sha256": question_sha,
        "context_sha256": context_sha,
        "ground_truth_sha256": ground_truth_sha,
        "content_root_id": content_root,
        "full_example_id": full_example,
    }


def _read_identities(path: Path, expected_rows: int, namespace: str) -> list[dict[str, Any]]:
    rows = [_identity(position, row, namespace) for position, row in _iter_parquet_rows(path)]
    _require(len(rows) == expected_rows, f"dataset row count differs for {path}")
    return rows


def _validate_s128_authority(
    repo: Path, manifest: Mapping[str, Any], identities: list[dict[str, Any]],
) -> dict[str, Any]:
    spec = manifest["sources"]["exposed_s128"]
    authority_path = repo / spec["authority_path"]
    _require(sha256_file(authority_path) == spec["authority_file_sha256"],
             "S128 authority file SHA differs")
    authority = _load_json(authority_path)
    _verify_self_digest(authority, "authority_sha256", "S128 authority digest differs")
    _require(authority.get("eval_manifest_hash") == spec["eval_manifest_hash"],
             "S128 eval manifest hash differs")
    source = authority.get("identity_payload", {}).get("source_dataset", {})
    _require(source.get("parquet_sha256") == spec["sha256"],
             "S128 parquet authority differs")
    frozen = authority.get("identity_payload", {}).get("rows", [])
    _require(len(frozen) == len(identities) == spec["rows"], "S128 row coverage differs")
    for position, (actual, expected) in enumerate(zip(identities, frozen)):
        _require(expected.get("raw_row_position") == position
                 and expected.get("source_order_index") == position
                 and expected.get("production_effective_position") == position
                 and expected.get("example_id") == str(actual["semantic_dataset_index"])
                 and expected.get("semantic_dataset_index") == actual["semantic_dataset_index"]
                 and expected.get("source_question_hash") == actual["question_sha256"]
                 and expected.get("source_context_hash") == actual["context_sha256"]
                 and expected.get("ground_truth_hash") == actual["ground_truth_sha256"],
                 f"S128 identity differs at row {position}")
    return {
        "authority_path": str(authority_path),
        "authority_file_sha256": spec["authority_file_sha256"],
        "authority_sha256": authority["authority_sha256"],
        "eval_manifest_hash": authority["eval_manifest_hash"],
        "content_root_count": len({row["content_root_id"] for row in identities}),
        "full_example_count": len({row["full_example_id"] for row in identities}),
    }


def _deduplicate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["content_root_id"]].append(row)
    result: dict[str, dict[str, Any]] = {}
    for root, aliases in grouped.items():
        ground_truths = {row["ground_truth_sha256"] for row in aliases}
        _require(len(ground_truths) == 1,
                 f"one content root has conflicting ground truths: {root}")
        aliases = sorted(aliases, key=lambda row: (row["source_position"], row["full_example_id"]))
        canonical = dict(aliases[0])
        canonical["alias_source_positions"] = [row["source_position"] for row in aliases]
        canonical["alias_count"] = len(aliases)
        result[root] = canonical
    return result


def _partition(
    manifest: Mapping[str, Any], train_rows: list[dict[str, Any]],
    s128_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped = _deduplicate(train_rows)
    s128_grouped = _deduplicate(s128_rows)
    for root in set(grouped) & set(s128_grouped):
        _require(grouped[root]["ground_truth_sha256"]
                 == s128_grouped[root]["ground_truth_sha256"],
                 f"one cross-source content root has conflicting ground truths: {root}")
    s128_content = {row["content_root_id"] for row in s128_rows}
    s128_full = {row["full_example_id"] for row in s128_rows}
    train_content = set(grouped)
    train_full = {row["full_example_id"] for row in train_rows}
    eligible = [row for root, row in grouped.items() if root not in s128_content]
    partition = manifest["partition"]
    order_namespace = partition["order_namespace"]
    seed = partition["seed"]
    for row in eligible:
        row["selection_order_sha256"] = sha256_json([
            order_namespace, seed, row["content_root_id"],
        ])
    eligible.sort(key=lambda row: (row["selection_order_sha256"], row["content_root_id"]))
    declared = tuple((item["name"], item["content_roots"]) for item in partition["splits"])
    _require(declared == EXPECTED_SPLITS, "data split sizes or order drifted")
    required = sum(size for _, size in EXPECTED_SPLITS)
    _require(len(eligible) >= required, "insufficient S128-disjoint unique training roots")
    splits: dict[str, Any] = {}
    start = 0
    for name, size in EXPECTED_SPLITS:
        selected = eligible[start:start + size]
        splits[name] = {
            "range_start": start,
            "range_stop_exclusive": start + size,
            "content_root_count": size,
            "rows": selected,
            "content_root_ids_sha256": sha256_json([row["content_root_id"] for row in selected]),
            "full_example_ids_sha256": sha256_json([row["full_example_id"] for row in selected]),
        }
        if name == "actor_training_b1_b8":
            splits[name]["blocks"] = [
                {
                    "block_id": f"B{block + 1}",
                    "range_start_within_split": block * 64,
                    "range_stop_exclusive_within_split": (block + 1) * 64,
                    "content_root_ids_sha256": sha256_json([
                        row["content_root_id"]
                        for row in selected[block * 64:(block + 1) * 64]
                    ]),
                }
                for block in range(8)
            ]
        start += size
    selected_sets = {
        name: {row["content_root_id"] for row in split["rows"]}
        for name, split in splits.items()
    }
    selected_full_sets = {
        name: {row["full_example_id"] for row in split["rows"]}
        for name, split in splits.items()
    }
    cross_content: dict[str, int] = {}
    cross_full: dict[str, int] = {}
    names = list(selected_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            key = f"{left}__{right}"
            cross_content[key] = len(selected_sets[left] & selected_sets[right])
            cross_full[key] = len(selected_full_sets[left] & selected_full_sets[right])
    _require(not any(cross_content.values()) and not any(cross_full.values()),
             "materialized data splits overlap")
    audit = {
        "training_rows": len(train_rows),
        "training_unique_content_roots": len(train_content),
        "training_unique_full_examples": len(train_full),
        "training_duplicate_rows": len(train_rows) - len(train_content),
        "s128_rows": len(s128_rows),
        "s128_unique_content_roots": len(s128_content),
        "s128_unique_full_examples": len(s128_full),
        "train_intersection_s128_content_roots": len(train_content & s128_content),
        "train_intersection_s128_full_examples": len(train_full & s128_full),
        "eligible_content_roots_after_s128_exclusion": len(eligible),
        "selected_content_roots": required,
        "split_pair_content_intersections": cross_content,
        "split_pair_full_example_intersections": cross_full,
        "all_selected_intersection_s128_content_roots": len(
            set().union(*selected_sets.values()) & s128_content
        ),
        "all_selected_intersection_s128_full_examples": len(
            set().union(*selected_full_sets.values()) & s128_full
        ),
    }
    _require(audit["all_selected_intersection_s128_content_roots"] == 0
             and audit["all_selected_intersection_s128_full_examples"] == 0,
             "selected data intersects exposed S128")
    return splits, audit


def _load_and_validate_manifest(repo: Path) -> tuple[dict[str, Any], Path, str]:
    path = repo / MANIFEST_REL
    manifest = _load_json(path)
    _require(manifest.get("schema") == "memagent.mic.v2.data-freeze-preregistration"
             and manifest.get("status") == "FROZEN_BEFORE_E1_OUTCOMES"
             and manifest.get("scientific_contract_sha256") == CONTRACT_SHA256,
             "data-freeze preregistration drifted")
    _require(manifest.get("identity", {}).get("selection_uses_ground_truth") is False,
             "data selection is not label-blind")
    _require(manifest.get("runtime", {}).get("gpu_required") is False,
             "data freeze unexpectedly requests a GPU")
    return manifest, path, sha256_file(path)


def _construct_resolved(
    repo: Path,
    actual_commit: str,
    output_root: Path,
    run_id: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha: str,
) -> dict[str, Any]:
    e0 = _verify_e0(repo, manifest)
    sources = manifest["sources"]
    train_spec = sources["training_pool"]
    s128_spec = sources["exposed_s128"]
    train_path = Path(train_spec["path"])
    s128_path = Path(s128_spec["path"])
    _require(train_path.is_file() and s128_path.is_file(), "source parquet is missing")
    _require(sha256_file(train_path) == train_spec["sha256"],
             "training parquet SHA differs")
    _require(sha256_file(s128_path) == s128_spec["sha256"], "S128 parquet SHA differs")
    namespace = manifest["identity"]["namespace"]
    s128_rows = _read_identities(s128_path, s128_spec["rows"], namespace)
    s128_authority = _validate_s128_authority(repo, manifest, s128_rows)
    train_rows = _read_identities(train_path, train_spec["rows"], namespace)
    splits, overlap = _partition(manifest, train_rows, s128_rows)
    resolved = {
        "schema": "memagent.mic.v2.resolved-data-split",
        "status": "PASS",
        "decision": "MIC_V2_DATA_SPLIT_FROZEN",
        "git_commit": actual_commit,
        "run_id": run_id,
        "output_root": str(output_root),
        "scientific_contract_sha256": CONTRACT_SHA256,
        "preregistration_manifest": str(manifest_path),
        "preregistration_manifest_sha256": manifest_sha,
        "e0_authority": e0,
        "source_files": {
            "training_pool": {"path": str(train_path), "sha256": train_spec["sha256"]},
            "exposed_s128": {"path": str(s128_path), "sha256": s128_spec["sha256"]},
        },
        "s128_authority": s128_authority,
        "identity_contract": manifest["identity"],
        "partition_contract": manifest["partition"],
        "overlap_audit": overlap,
        "splits": splits,
    }
    resolved["resolved_manifest_sha256"] = sha256_json(resolved)
    return resolved


def _build_certificate(
    *, actual_commit: str, run_id: str, output_root: Path,
    manifest: Mapping[str, Any], manifest_sha: str,
    resolved_path: Path, resolved: Mapping[str, Any],
) -> dict[str, Any]:
    e0 = resolved["e0_authority"]
    splits = resolved["splits"]
    report = {
        "schema": "memagent.mic.v2.data-freeze-certificate",
        "status": "PASS",
        "decision": "MIC_V2_DATA_FREEZE_PASS",
        "git_commit": actual_commit,
        "run_id": run_id,
        "output_root": str(output_root),
        "scientific_contract_sha256": CONTRACT_SHA256,
        "preregistration_manifest_sha256": manifest_sha,
        "e0_certificate_file_sha256": e0["file_sha256"],
        "e0_certificate_canonical_sha256": e0["canonical_sha256"],
        "resolved_manifest": str(resolved_path),
        "resolved_manifest_file_sha256": sha256_file(resolved_path),
        "resolved_manifest_canonical_sha256": resolved["resolved_manifest_sha256"],
        "overlap_audit": resolved["overlap_audit"],
        "split_receipts": {
            name: {
                "content_root_count": split["content_root_count"],
                "content_root_ids_sha256": split["content_root_ids_sha256"],
                "full_example_ids_sha256": split["full_example_ids_sha256"],
                **({"blocks": split["blocks"]} if "blocks" in split else {}),
            }
            for name, split in splits.items()
        },
        "blocked_commands": manifest["blocked_after_success"],
    }
    report["certificate_sha256"] = sha256_json(report)
    return report


def materialize(
    repo: Path, expected_commit: str, output_root: Path, run_id: str,
) -> dict[str, Any]:
    _require(sys.flags.optimize == 0, "optimized Python is forbidden")
    _require(RUN_ID_PATTERN.fullmatch(run_id) is not None, "unsafe data run ID")
    _require(repo.is_absolute() and output_root.is_absolute(), "paths must be absolute")
    actual_commit = _git(repo, "rev-parse", "HEAD")
    _require(actual_commit == expected_commit, "exact Git commit mismatch")
    _require(not _git(repo, "status", "--porcelain"), "worktree is dirty")
    manifest, manifest_path, manifest_sha = _load_and_validate_manifest(repo)
    _require(sha256_file(repo / "docs/papers/mic_v2_scientific_contract_20260825.md")
             == CONTRACT_SHA256, "scientific contract differs")
    resolved_path = output_root / "resolved_split_manifest.json"
    certificate_path = output_root / "certificates/data_freeze.json"
    try:
        resolved = _construct_resolved(
            repo, actual_commit, output_root, run_id,
            manifest, manifest_path, manifest_sha,
        )
        write_json_new(resolved_path, resolved)
        report = _build_certificate(
            actual_commit=actual_commit, run_id=run_id, output_root=output_root,
            manifest=manifest, manifest_sha=manifest_sha,
            resolved_path=resolved_path, resolved=resolved,
        )
        write_json_new(certificate_path, report)
        return report
    except Exception:
        failure = {
            "schema": "memagent.mic.v2.data-freeze-certificate",
            "status": "FAIL",
            "decision": "MIC_V2_DATA_FREEZE_NO_GO",
            "git_commit": actual_commit,
            "run_id": run_id,
            "output_root": str(output_root),
            "scientific_contract_sha256": CONTRACT_SHA256,
            "preregistration_manifest_sha256": manifest_sha,
            "traceback": traceback.format_exc(),
        }
        failure["certificate_sha256"] = sha256_json(failure)
        write_json_new(certificate_path, failure)
        return failure


def verify(repo: Path, expected_commit: str, output_root: Path, run_id: str) -> dict[str, Any]:
    _require(RUN_ID_PATTERN.fullmatch(run_id) is not None, "unsafe data run ID")
    _require(_git(repo, "rev-parse", "HEAD") == expected_commit, "verify commit differs")
    _require(not _git(repo, "status", "--porcelain"), "verify worktree is dirty")
    manifest, manifest_path, manifest_sha = _load_and_validate_manifest(repo)
    certificate_path = output_root / "certificates/data_freeze.json"
    resolved_path = output_root / "resolved_split_manifest.json"
    certificate = _load_json(certificate_path)
    canonical = _verify_self_digest(
        certificate, "certificate_sha256", "data-freeze certificate digest differs",
    )
    _require(certificate.get("status") == "PASS"
             and certificate.get("decision") == "MIC_V2_DATA_FREEZE_PASS",
             "data-freeze certificate is not PASS")
    _require(certificate.get("git_commit") == expected_commit
             and certificate.get("run_id") == run_id
             and certificate.get("output_root") == str(output_root),
             "data-freeze certificate identity differs")
    _require(certificate.get("preregistration_manifest_sha256") == manifest_sha,
             "data-freeze certificate manifest authority differs")
    _require(certificate.get("resolved_manifest") == str(resolved_path)
             and certificate.get("resolved_manifest_file_sha256") == sha256_file(resolved_path),
             "resolved data manifest file differs")
    resolved = _load_json(resolved_path)
    resolved_canonical = _verify_self_digest(
        resolved, "resolved_manifest_sha256", "resolved data manifest digest differs",
    )
    _require(resolved_canonical == certificate["resolved_manifest_canonical_sha256"],
             "resolved canonical authority differs")
    _require(resolved.get("git_commit") == expected_commit
             and resolved.get("run_id") == run_id
             and resolved.get("output_root") == str(output_root),
             "resolved data identity differs")
    overlap = resolved.get("overlap_audit", {})
    _require(overlap.get("all_selected_intersection_s128_content_roots") == 0
             and overlap.get("all_selected_intersection_s128_full_examples") == 0
             and not any(overlap.get("split_pair_content_intersections", {"missing": 1}).values())
             and not any(overlap.get(
                 "split_pair_full_example_intersections", {"missing": 1}).values()),
        "resolved data overlaps S128")
    _require({
        name: split.get("content_root_count")
        for name, split in resolved.get("splits", {}).items()
    } == dict(EXPECTED_SPLITS), "resolved split coverage differs")
    actor_blocks = resolved["splits"]["actor_training_b1_b8"].get("blocks", [])
    _require([block.get("block_id") for block in actor_blocks]
             == [f"B{index}" for index in range(1, 9)],
             "actor block partition differs")
    replay = _construct_resolved(
        repo, expected_commit, output_root, run_id,
        manifest, manifest_path, manifest_sha,
    )
    _require(replay == resolved, "resolved data manifest does not replay from source authority")
    expected_certificate = _build_certificate(
        actual_commit=expected_commit, run_id=run_id, output_root=output_root,
        manifest=manifest, manifest_sha=manifest_sha,
        resolved_path=resolved_path, resolved=resolved,
    )
    _require(expected_certificate == certificate,
             "data-freeze certificate does not replay from resolved authority")
    return {
        "status": "PASS",
        "decision": "MIC_V2_DATA_FREEZE_PASS",
        "git_commit": expected_commit,
        "run_id": run_id,
        "certificate": str(certificate_path),
        "certificate_sha256": canonical,
        "resolved_manifest": str(resolved_path),
        "resolved_manifest_sha256": resolved_canonical,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("materialize", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--repo", required=True, type=Path)
        sub.add_argument("--expected-commit", required=True)
        sub.add_argument("--output-root", required=True, type=Path)
        sub.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output_root = args.output_root.resolve()
    if args.command == "materialize":
        result = materialize(repo, args.expected_commit, output_root, args.run_id)
        print(canonical_json(result))
        return 0 if result["status"] == "PASS" else 2
    result = verify(repo, args.expected_commit, output_root, args.run_id)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
