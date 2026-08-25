#!/usr/bin/env python3
"""Fail-closed CPU authority and analysis entry for MIC-v2 E1."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from recurrent.research.mic_v2 import (
    CONTRACT_SHA256,
    MATERIALIZATION_PARSER_VERSION,
    canonical_json,
    sha256_file,
    sha256_json,
    write_json_new,
)
from tools.h20.mic_v2_data_freeze import _verify_self_digest
from tools.h20.mic_v2_reference_length_authority import (
    AUTHORITY_COMMIT as REFERENCE_AUTHORITY_COMMIT,
    AUTHORITY_FILE_SHA256 as REFERENCE_AUTHORITY_FILE_SHA256,
    verify_reference_length_authority,
)
from tools.h20.mic_v2_reference_length_calibration import (
    _tokenization_authority,
    _verify_model,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds


MANIFEST_REL = Path("manifests/h20/qwen25_7b_mic_v2_e1.json")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SPLIT_COUNTS = {"e1_dev": 128, "e1_holdout": 128}
GPU_FIELDS = (
    "source_position", "semantic_dataset_index", "question_sha256",
    "context_sha256", "content_root_id", "question", "context",
)
OUTCOME_FIELDS = (
    "source_position", "semantic_dataset_index", "content_root_id",
    "ground_truth_sha256", "ground_truth",
)
COLLECTION_FIELDS = {
    "execution": {
        "schema", "status", "decision", "git_commit", "run_id", "p0_sha256",
        "split", "gpu_pair", "physical_gpu_identity", "vllm_version",
        "config_loader_environment", "trajectory_count", "represented_generate_calls",
        "generate_calls_this_process", "ledger_file_sha256",
        "code_authority_sha256", "model_authority_sha256", "execution_sha256",
    },
    "replay": {
        "schema", "status", "decision", "git_commit", "run_id", "p0_sha256",
        "split", "gpu_pair", "physical_gpu_identity", "vllm_version",
        "config_loader_environment", "trajectory_count", "represented_generate_calls",
        "generate_calls_this_process", "ledger_file_sha256",
        "code_authority_sha256", "model_authority_sha256", "replay_sha256",
    },
}
MAX_SEED = (1 << 63) - 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MIC_V2_E1_NO_GO: {message}")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"JSONL is missing: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True,
    ).strip()


def _runtime(repo: Path, expected_commit: str, output_root: Path, run_id: str) -> None:
    _verify_inherited_lock_authority(run_id)
    _require(sys.flags.optimize == 0, "optimized Python is forbidden")
    _require(repo.is_absolute() and output_root.is_absolute(), "paths must be absolute")
    _require(RUN_ID.fullmatch(run_id) is not None, "unsafe E1 run ID")
    _require(_git(repo, "rev-parse", "HEAD") == expected_commit, "exact Git commit mismatch")
    _require(not _git(repo, "status", "--porcelain"), "worktree is dirty")
    _require(sha256_file(repo / "docs/papers/mic_v2_scientific_contract_20260825.md")
             == CONTRACT_SHA256, "scientific contract differs")


def _verify_inherited_lock_authority(run_id: str) -> list[int]:
    """Prove this process inherited the official run and two GPU lock descriptions."""
    lock_run_id = os.environ.get("MEMAGENT_MIC_V2_E1_LOCK_RUN_ID", "")
    replay_target = os.environ.get("MEMAGENT_MIC_V2_E1_REPLAY_TARGET_RUN_ID")
    _require(os.environ.get("MEMAGENT_MIC_V2_E1_OFFICIAL_ENTRY") == "locked-shell-v1"
             and RUN_ID.fullmatch(lock_run_id) is not None
             and (run_id == lock_run_id or replay_target == run_id),
             "E1 public CLI requires the locked official shell entry")
    work_root = os.environ.get("MEMAGENT_MIC_V2_E1_LOCK_WORK_ROOT", "")
    pair_text = os.environ.get("MEMAGENT_MIC_V2_E1_LOCK_GPU_PAIR", "")
    match = re.fullmatch(r"([0-9]+),([0-9]+)", pair_text)
    _require(work_root.startswith("/") and match is not None,
             "E1 inherited lock routing differs")
    pair = [int(match.group(1)), int(match.group(2))]
    expected_paths = [
        Path(work_root) / "locks" / f"memagent_mic_v2_e1_{lock_run_id}.lock",
        Path(work_root) / "locks" / f"memagent_h20_gpu_{pair[0]}.lock",
        Path(work_root) / "locks" / f"memagent_h20_gpu_{pair[1]}.lock",
    ]
    _require(os.environ.get("MEMAGENT_MIC_V2_E1_LOCK_FDS") == "7,8,9"
             and [os.environ.get(f"MEMAGENT_MIC_V2_E1_LOCK_PATH_{fd}")
                  for fd in (7, 8, 9)] == [str(path) for path in expected_paths],
             "E1 inherited lock path receipt differs")
    for descriptor, path in zip((7, 8, 9), expected_paths):
        try:
            opened, stated = os.fstat(descriptor), os.stat(path)
            _require((opened.st_dev, opened.st_ino) == (stated.st_dev, stated.st_ino),
                     "E1 inherited lock descriptor targets the wrong file")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "MIC_V2_E1_NO_GO: inherited run/GPU lock proof failed"
            ) from exc
    return pair


@contextmanager
def _scoped_replay_target(run_id: str):
    """Temporarily authorize a dev-evidence replay under holdout-owned locks."""
    _require(RUN_ID.fullmatch(run_id) is not None, "unsafe replay target run ID")
    name = "MEMAGENT_MIC_V2_E1_REPLAY_TARGET_RUN_ID"
    previous = os.environ.get(name)
    os.environ[name] = run_id
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _manifest(repo: Path) -> tuple[dict[str, Any], str]:
    path = repo / MANIFEST_REL
    value = _load(path)
    _require(value.get("schema") == "memagent.mic.v2.e1-preregistration"
             and value.get("status") == "FROZEN_BEFORE_E1_DEV_OUTCOMES"
             and value.get("scientific_contract_sha256") == CONTRACT_SHA256,
             "E1 preregistration differs")
    return value, sha256_file(path)


def _verify_e0(manifest: Mapping[str, Any]) -> dict[str, Any]:
    authority = manifest["e0_authority"]
    path = Path(authority["certificate_path"])
    _require(path.is_file() and sha256_file(path) == authority["certificate_file_sha256"],
             "E0 raw authority differs")
    value = _load(path)
    canonical = _verify_self_digest(value, "certificate_sha256", "E0 digest differs")
    _require(canonical == authority["certificate_canonical_sha256"]
             and value.get("status") == "PASS"
             and value.get("decision") == "MIC_V2_E0_PASS"
             and value.get("git_commit") == authority["git_commit"]
             and value.get("run_id") == authority["run_id"],
             "E0 authority differs")
    return {"file_sha256": authority["certificate_file_sha256"],
            "canonical_sha256": canonical}


def _verify_data_freeze(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = manifest["data_freeze_authority"]
    root = Path(authority["root"])
    certificate_path = root / "certificates/data_freeze.json"
    resolved_path = root / "resolved_split_manifest.json"
    _require(certificate_path.is_file() and resolved_path.is_file()
             and sha256_file(certificate_path) == authority["certificate_file_sha256"]
             and sha256_file(resolved_path) == authority["resolved_file_sha256"],
             "data-freeze raw authority differs")
    certificate, resolved = _load(certificate_path), _load(resolved_path)
    _require(_verify_self_digest(certificate, "certificate_sha256", "data certificate differs")
             == authority["certificate_canonical_sha256"]
             and _verify_self_digest(resolved, "resolved_manifest_sha256", "resolved differs")
                 == authority["resolved_canonical_sha256"]
             and certificate.get("status") == "PASS"
             and resolved.get("status") == "PASS",
             "data-freeze canonical authority differs")
    return certificate, resolved


def _project_split(resolved: Mapping[str, Any], split: str) -> dict[str, Any]:
    _require(split in SPLIT_COUNTS, "unknown E1 split")
    source = resolved.get("splits", {}).get(split, {})
    rows = source.get("rows", [])
    _require(isinstance(rows, list) and len(rows) == SPLIT_COUNTS[split]
             and source.get("content_root_count") == SPLIT_COUNTS[split],
             f"{split} coverage differs")
    projected = []
    for row in rows:
        item = {
            "source_position": row.get("source_position"),
            "semantic_dataset_index": row.get("semantic_dataset_index"),
            "question_sha256": row.get("question_sha256"),
            "context_sha256": row.get("context_sha256"),
            "content_root_id": row.get("content_root_id"),
            "ground_truth_sha256": row.get("ground_truth_sha256"),
        }
        _require(type(item["source_position"]) is int
                 and type(item["semantic_dataset_index"]) is int
                 and all(isinstance(item[key], str) and HEX64.fullmatch(item[key])
                         for key in ("question_sha256", "context_sha256",
                                     "content_root_id", "ground_truth_sha256")),
                 f"{split} identity row differs")
        projected.append(item)
    _require(len({row["content_root_id"] for row in projected}) == SPLIT_COUNTS[split],
             f"{split} content roots are not unique")
    return {
        "schema": "memagent.mic.v2.e1-split-projection", "status": "FROZEN",
        "split": split, "rows": projected,
        "content_root_ids_sha256": sha256_json([row["content_root_id"] for row in projected]),
    }


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def _materialize_gpu_source(
    manifest: Mapping[str, Any], split: Mapping[str, Any], output_root: Path,
    *, verify_existing: bool = False,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    source_spec = manifest["source"]
    source_path = Path(source_spec["path"])
    _require(source_path.is_file() and sha256_file(source_path) == source_spec["sha256"],
             "training source parquet differs")
    columns = ["prompt", "context", "extra_info"]
    table = pq.read_table(source_path, columns=columns).to_pylist()
    gpu_rows = []
    for frozen in split["rows"]:
        source = table[frozen["source_position"]]
        prompt, extra = source["prompt"], source["extra_info"]
        _require(isinstance(prompt, list) and len(prompt) == 1
                 and set(prompt[0]) == {"role", "content"} and prompt[0]["role"] == "user"
                 and isinstance(prompt[0]["content"], str)
                 and isinstance(source["context"], str)
                 and isinstance(extra, Mapping)
                 and set(extra) == {"index", "num_docs", "question"}
                 and extra.get("index") == frozen["semantic_dataset_index"]
                 and extra.get("question") == prompt[0]["content"]
                 and type(extra.get("num_docs")) is int,
                 "E1 source row schema differs")
        question, context = prompt[0]["content"], source["context"]
        _require(hashlib.sha256(question.encode()).hexdigest() == frozen["question_sha256"]
                 and hashlib.sha256(context.encode()).hexdigest() == frozen["context_sha256"],
                 "E1 source identity hash differs")
        gpu_rows.append({
            "source_position": frozen["source_position"],
            "semantic_dataset_index": frozen["semantic_dataset_index"],
            "question_sha256": frozen["question_sha256"],
            "context_sha256": frozen["context_sha256"],
            "content_root_id": frozen["content_root_id"],
            "question": question, "context": context,
        })
    _require(all(tuple(row) == GPU_FIELDS for row in gpu_rows),
             "E1 GPU artifact field order differs")
    gpu_path = output_root / "authorities/gpu_source.jsonl"
    if verify_existing:
        _require(gpu_path.is_file() and _jsonl(gpu_path) == gpu_rows,
                 "E1 GPU source replay differs")
    else:
        _write_jsonl_new(gpu_path, gpu_rows)
    return {"path": str(gpu_path), "rows": len(gpu_rows),
            "file_sha256": sha256_file(gpu_path), "schema_sha256": sha256_json(GPU_FIELDS)}


def _materialize_outcomes(
    manifest: Mapping[str, Any], split: Mapping[str, Any], output_root: Path,
    *, verify_existing: bool = False,
) -> dict[str, Any]:
    import pyarrow.parquet as pq
    source_path = Path(manifest["source"]["path"])
    _require(source_path.is_file() and sha256_file(source_path) == manifest["source"]["sha256"],
             "training source parquet differs before outcome opening")
    table = pq.read_table(source_path, columns=["reward_model"]).to_pylist()
    rows = []
    for frozen in split["rows"]:
        reward = table[frozen["source_position"]]["reward_model"]
        _require(isinstance(reward, Mapping) and set(reward) == {"ground_truth", "style"},
                 "E1 outcome row schema differs")
        ground_truth = reward["ground_truth"]
        _require(isinstance(ground_truth, list) and ground_truth
                 and all(isinstance(value, str) for value in ground_truth)
                 and sha256_json(ground_truth) == frozen["ground_truth_sha256"],
                 "E1 outcome identity differs")
        rows.append({
            "source_position": frozen["source_position"],
            "semantic_dataset_index": frozen["semantic_dataset_index"],
            "content_root_id": frozen["content_root_id"],
            "ground_truth_sha256": frozen["ground_truth_sha256"],
            "ground_truth": ground_truth,
        })
    _require(all(tuple(row) == OUTCOME_FIELDS for row in rows),
             "E1 outcome artifact field order differs")
    path = output_root / "authorities/cpu_outcomes.jsonl"
    if verify_existing:
        _require(path.is_file() and _jsonl(path) == rows,
                 "E1 CPU outcome replay differs")
    else:
        _write_jsonl_new(path, rows)
    return {"path": str(path), "rows": len(rows), "file_sha256": sha256_file(path),
            "schema_sha256": sha256_json(OUTCOME_FIELDS)}


def _materialize_sources(
    manifest: Mapping[str, Any], split: Mapping[str, Any], output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (_materialize_gpu_source(manifest, split, output_root),
            _materialize_outcomes(manifest, split, output_root))


def stable_e1_trajectory_seed(base_seed: int, content_root_id: str, replica: int) -> int:
    _require(type(base_seed) is int and HEX64.fullmatch(content_root_id) is not None
             and type(replica) is int and 0 <= replica < 4, "E1 seed input differs")
    raw = canonical_json([
        "memagent-mic-v2-e1-trajectory-v1", base_seed, content_root_id, replica,
    ]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % MAX_SEED


def _seed_authority(
    manifest: Mapping[str, Any], split: Mapping[str, Any],
    tokenization: Mapping[str, Any],
) -> dict[str, Any]:
    horizons = {row["content_root_id"]: row["active_writer_slots"]
                for row in tokenization["receipts"]}
    trajectories, requests = [], []
    for frozen in split["rows"]:
        root = frozen["content_root_id"]
        for replica in range(manifest["sampling"]["replicas"]):
            trajectory_seed = stable_e1_trajectory_seed(
                manifest["sampling"]["base_seed"], root, replica,
            )
            trajectories.append([root, replica, trajectory_seed])
            for slot in [*range(horizons[root]), manifest["recurrent"]["max_writer_slots"]]:
                requests.append([
                    root, replica, slot,
                    derive_turn_request_seeds([trajectory_seed], [0], slot)[0],
                ])
    trajectory_seeds = [row[-1] for row in trajectories]
    request_seeds = [row[-1] for row in requests]
    _require(len(set(trajectory_seeds)) == len(trajectory_seeds)
             and len(set(request_seeds)) == len(request_seeds)
             and not set(trajectory_seeds).intersection(request_seeds),
             "E1 seed schedule collides")
    return {
        "trajectory_count": len(trajectories),
        "trajectory_schedule_sha256": sha256_json(trajectories),
        "active_request_count": len(requests),
        "active_request_schedule_sha256": sha256_json(requests),
        "all_trajectory_seeds_unique": True,
        "all_active_request_seeds_unique": True,
        "trajectory_request_namespaces_disjoint": True,
    }


def _source_firewall(repo: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    forbidden = tuple(manifest["firewall"]["forbidden_gpu_source_terms"])
    for name in manifest["firewall"]["gpu_code_authorities"]:
        runner = repo / manifest["code_authority"][name]["path"]
        source = runner.read_text(encoding="utf-8").lower()
        _require(not any(term.lower() in source for term in forbidden),
                 f"E1 GPU source firewall failed: {name}")
    receipts = {}
    for name, receipt in manifest["code_authority"].items():
        path = repo / receipt["path"]
        _require(path.is_file() and sha256_file(path) == receipt["sha256"],
                 f"E1 code authority differs: {name}")
        receipts[receipt["path"]] = receipt["sha256"]
    return receipts


def _dev_gpu_reverification(
    repo: Path, expected_commit: str, dev_root: Path, dev_run_id: str,
    output_root: Path, holdout_run_id: str, gpu_pair: str, *, execute: bool,
) -> dict[str, Any]:
    """Authenticate a fresh full dev generation+feature replay before opening."""
    dev_p0 = _load(dev_root / "certificates/p0.json")
    _require(dev_p0.get("gpu_pair") == [int(value) for value in gpu_pair.split(",")],
             "holdout must lock the same canonical H20 pair used by E1-dev replay")
    if execute:
        environment = dict(__import__("os").environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu_pair
        environment["MEMAGENT_MIC_V2_E1_REPLAY_TARGET_RUN_ID"] = dev_run_id
        commands = (
            [
                sys.executable,
                str(repo / "tools/h20/run_qwen25_7b_mic_v2_e1_collect.py"),
                "--repo", str(repo), "--expected-commit", expected_commit,
                "--output-root", str(dev_root), "--run-id", dev_run_id,
                "--mode", "replay",
            ],
            [
                sys.executable,
                str(repo / "tools/h20/run_qwen25_7b_mic_v2_e1_features.py"),
                "--repo", str(repo), "--expected-commit", expected_commit,
                "--output-root", str(dev_root), "--run-id", dev_run_id,
                "--split", "e1_dev", "--mode", "replay",
            ],
        )
        for command in commands:
            completed = subprocess.run(
                command, env=environment, check=False, pass_fds=(7, 8, 9),
            )
            _require(completed.returncode == 0,
                     "fresh dev GPU reverification failed before holdout opening")
    collection_path = dev_root / "certificates/e1_dev_replay.json"
    feature_path = dev_root / "certificates/e1_dev_actor_hidden_features_replay.json"
    collection = _verified_self_receipt(
        collection_path, "replay_sha256", "memagent.mic.v2.e1-replay",
    )
    feature = _verified_self_receipt(
        feature_path, "features_sha256",
        "memagent.mic.v2.e1-actor-hidden-features-replay",
    )
    _require(collection.get("git_commit") == expected_commit
             and feature.get("git_commit") == expected_commit
             and collection.get("run_id") == feature.get("run_id") == dev_run_id
             and collection.get("physical_gpu_identity")
                 == feature.get("physical_gpu_identity")
             and feature.get("independent_exact_replay") is True,
             "dev GPU reverification identity differs")
    receipt = {
        "schema": "memagent.mic.v2.e1-dev-opening-gpu-reverification",
        "status": "PASS", "decision": "MIC_V2_E1_DEV_GPU_REVERIFIED_BEFORE_OPENING",
        "git_commit": expected_commit, "dev_run_id": dev_run_id,
        "dev_output_root": str(dev_root), "holdout_run_id": holdout_run_id,
        "holdout_output_root": str(output_root),
        "fresh_gpu_replay_required_before_holdout_opening": True,
        "collection_replay_file_sha256": sha256_file(collection_path),
        "collection_replay_canonical_sha256": collection["replay_sha256"],
        "feature_replay_file_sha256": sha256_file(feature_path),
        "feature_replay_canonical_sha256": feature["features_sha256"],
        "physical_gpu_identity": collection["physical_gpu_identity"],
    }
    receipt["reverification_sha256"] = sha256_json(receipt)
    path = output_root / "certificates/e1_dev_opening_gpu_reverification.json"
    if execute:
        write_json_new(path, receipt)
    else:
        _require(path.is_file() and _load(path) == receipt,
                 "dev opening GPU reverification receipt differs")
    return receipt


def _execute_split_gpu_replay(
    repo: Path, expected_commit: str, output_root: Path, run_id: str,
    split: str, gpu_pair: str,
) -> dict[str, Any]:
    """Actually regenerate every trajectory and hidden feature, then exact-compare."""
    environment = dict(__import__("os").environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu_pair
    commands = (
        [
            sys.executable, str(repo / "tools/h20/run_qwen25_7b_mic_v2_e1_collect.py"),
            "--repo", str(repo), "--expected-commit", expected_commit,
            "--output-root", str(output_root), "--run-id", run_id, "--mode", "replay",
        ],
        [
            sys.executable, str(repo / "tools/h20/run_qwen25_7b_mic_v2_e1_features.py"),
            "--repo", str(repo), "--expected-commit", expected_commit,
            "--output-root", str(output_root), "--run-id", run_id,
            "--split", split, "--mode", "replay",
        ],
    )
    for command in commands:
        completed = subprocess.run(
            command, env=environment, check=False, pass_fds=(7, 8, 9),
        )
        _require(completed.returncode == 0, f"fresh {split} GPU replay failed")
    collection_path = output_root / f"certificates/{split}_replay.json"
    feature_path = output_root / f"certificates/{split}_actor_hidden_features_replay.json"
    collection = _verified_self_receipt(
        collection_path, "replay_sha256", "memagent.mic.v2.e1-replay",
    )
    feature = _verified_self_receipt(
        feature_path, "features_sha256",
        "memagent.mic.v2.e1-actor-hidden-features-replay",
    )
    _require(collection.get("git_commit") == feature.get("git_commit") == expected_commit
             and collection.get("run_id") == feature.get("run_id") == run_id
             and collection.get("split") == feature.get("split") == split
             and collection.get("physical_gpu_identity")
                 == feature.get("physical_gpu_identity")
             and feature.get("independent_exact_replay") is True,
             f"fresh {split} GPU replay identity differs")
    return {
        "collection_replay_file_sha256": sha256_file(collection_path),
        "collection_replay_canonical_sha256": collection["replay_sha256"],
        "feature_replay_file_sha256": sha256_file(feature_path),
        "feature_replay_canonical_sha256": feature["features_sha256"],
        "physical_gpu_identity": collection["physical_gpu_identity"],
    }


def _replayed_dev_selection(
    repo: Path, dev_root: Path, expected_commit: str,
) -> tuple[dict[str, Any], str]:
    dev_p0 = _load(dev_root / "certificates/p0.json")
    unsigned_p0 = dict(dev_p0)
    dev_p0_sha = unsigned_p0.pop("p0_sha256", None)
    _require(dev_p0_sha == sha256_json(unsigned_p0)
             and dev_p0.get("git_commit") == expected_commit
             and dev_p0.get("output_root") == str(dev_root)
             and dev_p0.get("split") == "e1_dev",
             "E1-dev P0 cross-chain differs")
    # This is a full reconstruction from raw dev evidence, not validation of a
    # self-signed selection summary.  It must finish before the marker is made.
    dev_run_id = str(dev_p0.get("run_id", ""))
    with _scoped_replay_target(dev_run_id):
        selection = select_dev(
            repo, expected_commit, dev_root, dev_run_id,
            verify_existing=True,
        )
    _require(selection.get("p0_sha256") == dev_p0_sha,
             "E1-dev replay/P0 cross-chain differs")
    return selection, dev_p0_sha


def _holdout_opening_payload(
    repo: Path, dev_root: Path, output_root: Path, expected_commit: str, run_id: str,
) -> dict[str, Any]:
    selection, _dev_p0_sha = _replayed_dev_selection(repo, dev_root, expected_commit)
    selection_path = dev_root / "certificates/e1_dev_selection.json"
    selection_sha = selection["selection_sha256"]
    gpu_path = output_root / "certificates/e1_dev_opening_gpu_reverification.json"
    gpu_receipt = _load(gpu_path)
    unsigned_gpu = dict(gpu_receipt)
    gpu_sha = unsigned_gpu.pop("reverification_sha256", None)
    _require(gpu_sha == sha256_json(unsigned_gpu)
             and gpu_receipt.get("schema")
                 == "memagent.mic.v2.e1-dev-opening-gpu-reverification"
             and gpu_receipt.get("status") == "PASS"
             and gpu_receipt.get("git_commit") == expected_commit
             and gpu_receipt.get("dev_output_root") == str(dev_root)
             and gpu_receipt.get("holdout_output_root") == str(output_root)
             and gpu_receipt.get("holdout_run_id") == run_id,
             "dev GPU reverification/opening cross-chain differs")
    opening = {
        "schema": "memagent.mic.v2.e1-holdout-opening", "status": "OPENED_ONCE",
        "git_commit": expected_commit, "dev_run_id": selection["run_id"],
        "dev_selection_file_sha256": sha256_file(selection_path),
        "dev_selection_canonical_sha256": selection_sha,
        "dev_gpu_reverification_file_sha256": sha256_file(gpu_path),
        "dev_gpu_reverification_canonical_sha256": gpu_sha,
        "holdout_run_id": run_id, "holdout_output_root": str(output_root),
    }
    opening["opening_sha256"] = sha256_json(opening)
    return opening


def _open_holdout_once(
    repo: Path, dev_root: Path, output_root: Path, expected_commit: str, run_id: str,
) -> dict[str, Any]:
    opening = _holdout_opening_payload(
        repo, dev_root, output_root, expected_commit, run_id,
    )
    write_json_new(dev_root / "certificates/e1_holdout_opening.json", opening)
    return opening


def _verify_holdout_opening(
    repo: Path, dev_root: Path, output_root: Path, expected_commit: str, run_id: str,
) -> dict[str, Any]:
    expected = _holdout_opening_payload(
        repo, dev_root, output_root, expected_commit, run_id,
    )
    path = dev_root / "certificates/e1_holdout_opening.json"
    _require(path.is_file() and _load(path) == expected,
             "E1 holdout opening replay differs")
    return expected


def preflight(
    repo: Path, expected_commit: str, work_root: Path, output_root: Path,
    run_id: str, split_name: str, gpu_pair: str, dev_root: Path | None = None,
    *, verify_existing: bool = False,
) -> dict[str, Any]:
    _runtime(repo, expected_commit, output_root, run_id)
    manifest, manifest_sha = _manifest(repo)
    import numpy
    import scipy
    _require(numpy.__version__ == manifest["numeric"]["numpy_version"]
             and scipy.__version__ == manifest["numeric"]["scipy_version"],
             "E1 numeric runtime versions differ")
    _require(split_name in SPLIT_COUNTS, "unknown E1 preflight split")
    pair = [int(value) for value in gpu_pair.split(",")]
    _require(len(pair) == 2 and pair == sorted(set(pair)) and all(value >= 0 for value in pair),
             "GPU pair must be two unique ascending indices")
    e0 = _verify_e0(manifest)
    reference = verify_reference_length_authority(repo, work_root)
    verifier = manifest["reference_length_authority"]
    verifier_path = repo / "tools/h20/mic_v2_reference_length_authority.py"
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor",
         verifier["verifier_commit"], expected_commit], check=False,
    )
    _require(ancestor.returncode == 0
             and sha256_file(verifier_path) == verifier["verifier_file_sha256"]
             and reference.get("authority_commit") == REFERENCE_AUTHORITY_COMMIT
             and reference.get("authority_file_sha256") == REFERENCE_AUTHORITY_FILE_SHA256
             and reference.get("lbar_ref") == manifest["reference_length_authority"]["lbar_ref"],
             "reference-length authority differs")
    _certificate, resolved = _verify_data_freeze(manifest)
    dev_selection_authority = None
    dev_run_id = None
    if split_name == "e1_dev":
        _require(dev_root is None, "E1-dev preflight cannot accept a dev authority root")
    else:
        _require(dev_root is not None and dev_root.is_absolute()
                 and dev_root != output_root, "E1 holdout dev authority root differs")
        dev_p0 = _load(dev_root / "certificates/p0.json")
        unsigned_dev_p0 = dict(dev_p0)
        dev_p0_sha = unsigned_dev_p0.pop("p0_sha256", None)
        _require(dev_p0_sha == sha256_json(unsigned_dev_p0)
                 and dev_p0.get("git_commit") == expected_commit
                 and dev_p0.get("output_root") == str(dev_root)
                 and dev_p0.get("split") == "e1_dev",
                 "E1-dev P0 authority differs before holdout opening")
        dev_run_id = str(dev_p0.get("run_id", ""))
        _require(bool(dev_run_id), "E1-dev run identity is missing")
    split = _project_split(resolved, split_name)
    split["projection_sha256"] = sha256_json(split)
    # All label-blind/environment authorities are checked before the one-shot
    # holdout marker.  Ground truth is materialized only after that marker exists.
    gpu_source = _materialize_gpu_source(
        manifest, split, output_root, verify_existing=verify_existing,
    )
    model_files = _verify_model(manifest)
    tokenization = _tokenization_authority(manifest, Path(gpu_source["path"]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["path"], trust_remote_code=True, local_files_only=True,
    )
    tokenization["no_memory_token_ids_sha256"] = sha256_json(list(tokenizer.encode(
        manifest["recurrent"]["no_memory_text"], add_special_tokens=False,
    )))
    tokenization["parser_version"] = MATERIALIZATION_PARSER_VERSION
    seeds = _seed_authority(manifest, split, tokenization)
    code = _source_firewall(repo, manifest)
    dev_gpu_reverification = None
    if split_name == "e1_holdout":
        _require(dev_root is not None and dev_run_id is not None,
                 "E1-dev authority disappeared before holdout opening")
        dev_gpu_reverification = _dev_gpu_reverification(
            repo, expected_commit, dev_root, dev_run_id,
            output_root, run_id, gpu_pair,
            execute=not verify_existing,
        )
        if verify_existing:
            dev_selection_authority = _verify_holdout_opening(
                repo, dev_root, output_root, expected_commit, run_id,
            )
        else:
            dev_selection_authority = _open_holdout_once(
                repo, dev_root, output_root, expected_commit, run_id,
            )
    outcomes = _materialize_outcomes(
        manifest, split, output_root, verify_existing=verify_existing,
    )
    report = {
        "schema": "memagent.mic.v2.e1-p0", "status": "PASS",
        "decision": ("MIC_V2_E1_DEV_P0_PASS" if split_name == "e1_dev"
                     else "MIC_V2_E1_HOLDOUT_P0_PASS"),
        "git_commit": expected_commit,
        "run_id": run_id, "work_root": str(work_root),
        "output_root": str(output_root), "split": split_name,
        "gpu_pair": pair, "scientific_contract_sha256": CONTRACT_SHA256,
        "manifest_sha256": manifest_sha, "e0_authority": e0,
        "reference_length_authority": reference,
        "split_projection_sha256": split["projection_sha256"],
        "gpu_source": gpu_source, "cpu_outcomes": outcomes,
        "model_files": model_files, "tokenization_authority": tokenization,
        "seed_authority": seeds, "code_sha256": code,
        "lbar_ref": reference["lbar_ref"],
    }
    if dev_selection_authority is not None:
        report["dev_selection_authority"] = dev_selection_authority
        report["dev_selection_root"] = str(dev_root)
        report["dev_gpu_reverification"] = dev_gpu_reverification
    report["p0_sha256"] = sha256_json(report)
    p0_path = output_root / "certificates/p0.json"
    split_path = output_root / "authorities/split_projection.json"
    if verify_existing:
        _require(p0_path.is_file() and _load(p0_path) == report,
                 "E1 P0 full replay differs")
        _require(split_path.is_file() and _load(split_path) == split,
                 "E1 split projection replay differs")
    else:
        write_json_new(p0_path, report)
        write_json_new(split_path, split)
    return report


def _verified_collection(
    output_root: Path, expected_commit: str, run_id: str, split: str = "e1_dev",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p0 = _load(output_root / "certificates/p0.json")
    unsigned_p0 = dict(p0)
    p0_sha = unsigned_p0.pop("p0_sha256", None)
    _require(p0_sha == sha256_json(unsigned_p0)
             and p0.get("git_commit") == expected_commit
             and p0.get("run_id") == run_id
             and p0.get("output_root") == str(output_root)
             and p0.get("split") == split, "E1 P0 differs")
    receipts = {}
    for name in ("execution", "replay"):
        receipt = _load(output_root / f"certificates/{split}_{name}.json")
        digest = f"{name}_sha256"
        unsigned = dict(receipt)
        claimed = unsigned.pop(digest, None)
        _require(set(receipt) == COLLECTION_FIELDS[name]
                 and claimed == sha256_json(unsigned)
                 and receipt.get("schema") == f"memagent.mic.v2.e1-{name}"
                 and receipt.get("status") == "PASS"
                 and receipt.get("decision") == f"MIC_V2_E1_{name.upper()}_PASS"
                 and receipt.get("git_commit") == expected_commit
                 and receipt.get("run_id") == run_id
                 and receipt.get("split") == split
                 and receipt.get("p0_sha256") == p0_sha
                 and receipt.get("gpu_pair") == p0["gpu_pair"]
                 and receipt.get("trajectory_count") == 512
                 and receipt.get("represented_generate_calls")
                     == p0["seed_authority"]["active_request_count"]
                 and receipt.get("generate_calls_this_process")
                     == p0["seed_authority"]["active_request_count"],
                 f"E1 {name} receipt differs")
        _require(receipt.get("code_authority_sha256") == sha256_json(p0["code_sha256"])
                 and receipt.get("model_authority_sha256") == sha256_json(p0["model_files"]),
                 f"E1 {name} code/model authority differs")
        receipts[name] = receipt
    _require(receipts["execution"]["physical_gpu_identity"]
             == receipts["replay"]["physical_gpu_identity"]
             and receipts["execution"]["vllm_version"]
                 == receipts["replay"]["vllm_version"]
             and receipts["execution"]["config_loader_environment"]
                 == receipts["replay"]["config_loader_environment"]
             and receipts["execution"]["ledger_file_sha256"]
                 == receipts["replay"]["ledger_file_sha256"],
             "E1 producer/replay authority differs")
    ledger_path = output_root / f"trajectories/{split}.jsonl"
    _require(sha256_file(ledger_path) == receipts["execution"]["ledger_file_sha256"],
             "E1 trajectory ledger raw SHA differs")
    ledger = _jsonl(ledger_path)
    _require(len(ledger) == 512, "E1 trajectory coverage differs")
    expected_fields = {
        "schema", "split", "git_commit", "run_id", "p0_sha256",
        "content_root_id", "replica", "trajectory_id", "trajectory_seed",
        "active_writer_slots", "writer_slots", "answer_request_seed",
        "answer_completion", "terminal_text", "record_index",
        "previous_record_sha256", "record_sha256",
    }
    previous = "0" * 64
    pairs = set()
    active_requests = 0
    for index, record in enumerate(ledger):
        unsigned_record = dict(record)
        digest = unsigned_record.pop("record_sha256", None)
        pair = (record.get("content_root_id"), record.get("replica"))
        slots = record.get("writer_slots")
        _require(set(record) == expected_fields
                 and record.get("schema") == "memagent.mic.v2.e1-trajectory"
                 and record.get("split") == split
                 and record.get("git_commit") == expected_commit
                 and record.get("run_id") == run_id
                 and record.get("p0_sha256") == p0_sha
                 and record.get("record_index") == index
                 and record.get("previous_record_sha256") == previous
                 and digest == sha256_json(unsigned_record)
                 and pair not in pairs
                 and type(pair[1]) is int and 0 <= pair[1] < 4
                 and isinstance(slots, list)
                 and len(slots) == record.get("active_writer_slots")
                 and all(slot.get("turn") == turn
                         for turn, slot in enumerate(slots, start=1)),
                 "E1 trajectory ledger authority differs")
        previous = digest
        pairs.add(pair)
        active_requests += len(slots) + 1
    _require(len({root for root, _replica in pairs}) == 128
             and all(sum(candidate_root == root for candidate_root, _replica in pairs) == 4
                     for root, _replica in pairs)
             and active_requests == p0["seed_authority"]["active_request_count"],
             "E1 trajectory ledger root/request coverage differs")
    return p0, ledger


def seal_states(
    repo: Path, expected_commit: str, output_root: Path, run_id: str, split: str,
    *, verify_existing: bool = False,
) -> dict[str, Any]:
    _runtime(repo, expected_commit, output_root, run_id)
    _require(split in SPLIT_COUNTS, "unknown E1 state split")
    p0, ledger = _verified_collection(output_root, expected_commit, run_id, split)
    manifest, manifest_sha = _manifest(repo)
    _require(manifest_sha == p0["manifest_sha256"], "E1 state manifest differs")
    from transformers import AutoTokenizer
    from recurrent.research.mic_v2 import materialized_memory_receipt, validate_boundary_pair

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["path"], trust_remote_code=True, local_files_only=True,
    )
    from recurrent.research.serialization_credit_pilots import center_truncate_token_ids
    from recurrent.research.trajectory_seeding import derive_turn_request_seeds
    terminators = manifest["backend"]["termination_token_ids"]
    token_receipts = {
        row["content_root_id"]: row
        for row in p0["tokenization_authority"]["receipts"]
    }
    source_path = Path(p0["gpu_source"]["path"])
    _require(source_path.is_file()
             and sha256_file(source_path) == p0["gpu_source"]["file_sha256"],
             "E1 state GPU-source authority differs")
    sources = {row["content_root_id"]: row for row in _jsonl(source_path)}
    _require(len(sources) == 128, "E1 state GPU-source root coverage differs")

    state_rows = []
    seen = set()
    for record in ledger:
        root, replica = record.get("content_root_id"), record.get("replica")
        trajectory = record.get("trajectory_id")
        _require(isinstance(root, str) and HEX64.fullmatch(root) is not None
                 and type(replica) is int and 0 <= replica < 4
                 and isinstance(trajectory, str) and HEX64.fullmatch(trajectory) is not None,
                 "E1 state identity differs")
        slots = record.get("writer_slots")
        _require(isinstance(slots, list) and len(slots) == record.get("active_writer_slots"),
                 "E1 state slot coverage differs")
        _require(slots and root in token_receipts and root in sources,
                 "E1 state token/source authority is missing")
        source, token_receipt = sources[root], token_receipts[root]
        question_ids = list(tokenizer.encode(source["question"], add_special_tokens=False))
        context_ids = center_truncate_token_ids(
            list(tokenizer.encode(source["context"], add_special_tokens=False)),
            manifest["recurrent"]["max_context_tokens"],
        )
        chunk_size = manifest["recurrent"]["chunk_size"]
        chunk_ids = [
            context_ids[offset:offset + chunk_size]
            for offset in range(0, len(context_ids), chunk_size)
        ]
        decoded_chunks = [
            tokenizer.decode(chunk, skip_special_tokens=False) for chunk in chunk_ids
        ]
        expected_trajectory_seed = stable_e1_trajectory_seed(
            manifest["sampling"]["base_seed"], root, replica,
        )
        expected_trajectory_id = sha256_json([
            "mic-v2-e1-trajectory", split, root, replica,
        ])
        _require(record.get("trajectory_seed") == expected_trajectory_seed
                 and trajectory == expected_trajectory_id
                 and record.get("active_writer_slots") == len(chunk_ids) == len(slots)
                 and token_receipt["question_token_ids_sha256"] == sha256_json(question_ids)
                 and token_receipt["context_token_ids_sha256"] == sha256_json(context_ids)
                 and token_receipt["chunk_token_ids_sha256"]
                     == [sha256_json(chunk) for chunk in chunk_ids]
                 and token_receipt["active_writer_slots"] == len(chunk_ids),
                 "E1 independent source/seed/horizon reconstruction differs")
        expected_writer_seeds = [
            derive_turn_request_seeds([expected_trajectory_seed], [0], turn)[0]
            for turn in range(len(slots))
        ]
        expected_answer_seed = derive_turn_request_seeds(
            [expected_trajectory_seed], [0], manifest["recurrent"]["max_writer_slots"],
        )[0]
        _require([slot.get("request_seed") for slot in slots] == expected_writer_seeds
                 and record.get("answer_request_seed") == expected_answer_seed,
                 "E1 independent request-seed schedule differs")
        first_pre = slots[0].get("pre_state")
        _require(isinstance(first_pre, dict), "E1 first pre-write state is missing")
        chunk_schedule_id = first_pre.get("public_metadata", {}).get("chunk_schedule_id")
        expected_chunk_schedule_id = sha256_json(token_receipt["chunk_token_ids_sha256"])
        _require(isinstance(chunk_schedule_id, str)
                 and chunk_schedule_id == expected_chunk_schedule_id,
                 "E1 chunk schedule identity differs")
        initial_post = {
            "schema": "memagent.mic.v2.e1-initial-post-state",
            "phase": "initial_post", "content_root_id": root,
            "stable_example_id": root, "trajectory_id": trajectory,
            "turn_index": 0, "question": first_pre.get("question"),
            "arrived_chunks": [], "materialized_memory_history": [],
            "current_memory": "",
            "public_metadata": {
                "arrived_context_token_count": 0,
                "chunk_schedule_id": chunk_schedule_id,
                "exogenous_termination": False, "forced_truncation": False,
                "policy_termination": False, "prior_active_turn_count": 0,
            },
        }
        initial_post["state_sha256"] = sha256_json(initial_post)
        state_rows.append({
            "content_root_id": root, "replica": replica, "split": split,
            "trajectory_id": trajectory, "turn": 0,
            "available_stages": ["post"], "post_state": initial_post,
            "post_materialized_memory_token_count": 0,
            "post_materialized_memory_token_sha256": sha256_json([]),
        })
        prior_memory_token_ids: list[int] = []
        prior_memory_token_sha256: list[str] = []
        previous_post = initial_post
        for expected_turn, slot in enumerate(slots, start=1):
            before, after = validate_boundary_pair(slot.get("pre_state"), slot.get("post_state"))
            materialization = slot.get("materialization")
            _require(isinstance(materialization, dict),
                     "E1 materialized-memory receipt is missing")
            post_memory_token_ids = materialization.get("parsed_memory_token_ids")
            _require(isinstance(post_memory_token_ids, list)
                     and all(type(token) is int and token >= 0
                             for token in post_memory_token_ids)
                     and materialization.get("parsed_memory_sha256")
                         == sha256_json(post_memory_token_ids),
                     "E1 materialized-memory token authority differs")
            completion = slot.get("completion")
            _require(isinstance(completion, dict)
                     and isinstance(completion.get("sampled_token_ids"), list),
                     "E1 writer completion token authority is missing")
            parsed, recomputed_materialization = materialized_memory_receipt(
                token_ids=completion["sampled_token_ids"],
                termination_token_ids=terminators,
                content_root_id=root, trajectory_seed=record["trajectory_seed"],
                turn_index=expected_turn - 1,
                arrived_chunk_token_sha256=token_receipts[root][
                    "chunk_token_ids_sha256"
                ][:expected_turn],
                prior_memory_token_sha256=prior_memory_token_sha256,
            )
            decoded_post_memory = tokenizer.decode(parsed, skip_special_tokens=False)
            _require(recomputed_materialization == materialization
                     and after["current_memory"] == decoded_post_memory
                     and before["question"] == after["question"] == source["question"]
                     and before["arrived_chunks"] == after["arrived_chunks"]
                         == decoded_chunks[:expected_turn]
                     and before["public_metadata"]["arrived_context_token_count"]
                         == after["public_metadata"]["arrived_context_token_count"]
                         == sum(len(chunk) for chunk in chunk_ids[:expected_turn])
                     and after["materialized_memory_history"]
                         == [*previous_post["materialized_memory_history"], decoded_post_memory],
                     "E1 independent materialization/afterstate reconstruction differs")
            _validate_cross_turn_filtration(
                before, previous_post, expected_turn, chunk_schedule_id,
            )
            key = (root, replica, expected_turn)
            _require(key not in seen and slot.get("turn") == expected_turn
                     and before["content_root_id"] == root
                     and before["trajectory_id"] == trajectory
                     and before["turn_index"] == expected_turn,
                     "E1 time-safe state alignment differs")
            seen.add(key)
            state_rows.append({
                "content_root_id": root, "replica": replica, "split": split,
                "trajectory_id": trajectory, "turn": expected_turn,
                "available_stages": ["pre", "post"],
                "pre_state": before, "post_state": after,
                "pre_materialized_memory_token_count": len(prior_memory_token_ids),
                "pre_materialized_memory_token_sha256": sha256_json(prior_memory_token_ids),
                "post_materialized_memory_token_count": len(post_memory_token_ids),
                "post_materialized_memory_token_sha256": sha256_json(post_memory_token_ids),
            })
            prior_memory_token_ids = list(post_memory_token_ids)
            prior_memory_token_sha256.append(materialization["parsed_memory_sha256"])
            previous_post = after
    initial_count = sum(row["turn"] == 0 for row in state_rows)
    _require(len(state_rows) > 0 and len({row["content_root_id"] for row in state_rows}) == 128
             and initial_count == 512
             and all((row["available_stages"] == ["post"]) == (row["turn"] == 0)
                     for row in state_rows),
             "E1 time-safe state coverage differs")
    state_path = output_root / f"states/{split}_time_safe_states.jsonl"
    if verify_existing:
        _require(state_path.is_file() and _jsonl(state_path) == state_rows,
                 "E1 time-safe state replay differs")
    else:
        _write_jsonl_new(state_path, state_rows)
    report = {
        "schema": "memagent.mic.v2.e1-time-safe-states", "status": "PASS",
        "decision": "MIC_V2_E1_TIME_SAFE_STATES_SEALED", "split": split,
        "git_commit": expected_commit, "run_id": run_id,
        "p0_sha256": p0["p0_sha256"], "state_row_count": len(state_rows),
        "initial_post_row_count": initial_count,
        "writer_pair_row_count": len(state_rows) - initial_count,
        "root_count": 128, "state_path": str(state_path),
        "state_file_sha256": sha256_file(state_path),
        "state_canonical_sha256": sha256_json(state_rows),
        "time_safe_field_firewall_pass": True,
    }
    report["states_sha256"] = sha256_json(report)
    certificate_path = output_root / f"certificates/{split}_states.json"
    if verify_existing:
        _require(certificate_path.is_file() and _load(certificate_path) == report,
                 "E1 time-safe state certificate replay differs")
    else:
        write_json_new(certificate_path, report)
    return report


def _validate_cross_turn_filtration(
    before: Mapping[str, Any], previous_post: Mapping[str, Any],
    expected_turn: int, chunk_schedule_id: str,
) -> None:
    """Independently bind post(t-1) to pre(t), beyond within-slot validation."""
    _require(before["question"] == previous_post["question"]
             and before["content_root_id"] == previous_post["content_root_id"]
             and before["trajectory_id"] == previous_post["trajectory_id"]
             and before["arrived_chunks"][:-1] == previous_post["arrived_chunks"]
             and before["materialized_memory_history"]
                 == previous_post["materialized_memory_history"]
             and before["current_memory"] == previous_post["current_memory"]
             and before["public_metadata"]["chunk_schedule_id"] == chunk_schedule_id
             and previous_post["public_metadata"]["chunk_schedule_id"] == chunk_schedule_id
             and before["public_metadata"]["prior_active_turn_count"] == expected_turn - 1,
             "E1 cross-turn filtration continuity differs")


def score_split(
    repo: Path, expected_commit: str, output_root: Path, run_id: str, split: str,
    *, verify_existing: bool = False,
) -> dict[str, Any]:
    _runtime(repo, expected_commit, output_root, run_id)
    _require(split in SPLIT_COUNTS, "unknown E1 scoring split")
    manifest, manifest_sha = _manifest(repo)
    p0, ledger = _verified_collection(output_root, expected_commit, run_id, split)
    _require(manifest_sha == p0["manifest_sha256"], "E1 scoring manifest differs")
    scorer_path = repo / manifest["metric"]["code_path"]
    _require(scorer_path.is_file()
             and sha256_file(scorer_path) == manifest["metric"]["code_sha256"],
             "E1 independent metric code differs")
    from recurrent.research.s128_hotpot_metrics import score_terminal_output

    outcomes_path = Path(p0["cpu_outcomes"]["path"])
    _require(outcomes_path.is_file()
             and sha256_file(outcomes_path) == p0["cpu_outcomes"]["file_sha256"],
             "E1 CPU outcome authority differs")
    outcome_rows = _jsonl(outcomes_path)
    _require(len(outcome_rows) == 128
             and all(tuple(row) == OUTCOME_FIELDS for row in outcome_rows),
             "E1 CPU outcome coverage/schema differs")
    outcomes = {row["content_root_id"]: row for row in outcome_rows}
    _require(len(outcomes) == 128, "E1 outcome roots are not unique")
    metric_rows = []
    seen = set()
    for record in ledger:
        key = (record.get("content_root_id"), record.get("replica"))
        _require(key not in seen and key[0] in outcomes
                 and type(key[1]) is int and 0 <= key[1] < 4,
                 "E1 metric join identity differs")
        seen.add(key)
        metric = score_terminal_output(record.get("terminal_text"), outcomes[key[0]]["ground_truth"])
        _require(all(math.isfinite(float(metric[name]))
                     for name in ("token_f1", "exact_match", "format_success", "sub_exact_match")),
                 "E1 metric is non-finite")
        metric_rows.append({
            "content_root_id": key[0], "replica": key[1],
            "trajectory_id": record.get("trajectory_id"),
            "ground_truth_sha256": outcomes[key[0]]["ground_truth_sha256"],
            "terminal_text_sha256": hashlib.sha256(
                str(record.get("terminal_text", "")).encode("utf-8")
            ).hexdigest(),
            "target_raw_terminal_return": float(metric["token_f1"]),
            "exact_match": float(metric["exact_match"]),
            "format_success": float(metric["format_success"]),
            "sub_exact_match_diagnostic": float(metric["sub_exact_match"]),
            "prediction_sha256": hashlib.sha256(metric["prediction"].encode("utf-8")).hexdigest(),
            "extraction_route": metric["extraction_route"],
        })
    _require(len(seen) == 512 and all(
        sum(1 for candidate in seen if candidate[0] == root) == 4 for root in outcomes
    ), "E1 metric replica coverage differs")
    metrics_path = output_root / f"outcomes/{split}_terminal_metrics.jsonl"
    if verify_existing:
        _require(metrics_path.is_file() and _jsonl(metrics_path) == metric_rows,
                 "E1 terminal metric replay differs")
    else:
        _write_jsonl_new(metrics_path, metric_rows)
    report = {
        "schema": "memagent.mic.v2.e1-outcomes", "status": "PASS",
        "decision": "MIC_V2_E1_RAW_RETURN_SEALED", "split": split,
        "git_commit": expected_commit, "run_id": run_id,
        "p0_sha256": p0["p0_sha256"],
        "target_definition": "independently_recomputed_terminal_hotpot_token_f1",
        "target_dense_training_reward": False, "row_count": 512, "root_count": 128,
        "metric_rows_path": str(metrics_path),
        "metric_rows_file_sha256": sha256_file(metrics_path),
        "metric_rows_canonical_sha256": sha256_json(metric_rows),
        "metric_code_sha256": manifest["metric"]["code_sha256"],
        "mean_raw_terminal_return": sum(
            row["target_raw_terminal_return"] for row in metric_rows
        ) / len(metric_rows),
    }
    report["outcomes_sha256"] = sha256_json(report)
    certificate_path = output_root / f"certificates/{split}_outcomes.json"
    if verify_existing:
        _require(certificate_path.is_file() and _load(certificate_path) == report,
                 "E1 outcome certificate replay differs")
    else:
        write_json_new(certificate_path, report)
    return report


def _verified_self_receipt(path: Path, digest_field: str, schema: str) -> dict[str, Any]:
    value = _load(path)
    unsigned = dict(value)
    digest = unsigned.pop(digest_field, None)
    _require(digest == sha256_json(unsigned) and value.get("schema") == schema
             and value.get("status") == "PASS", f"receipt differs: {path.name}")
    return value


def _assemble_feature_rows(
    manifest: Mapping[str, Any], states: list[dict[str, Any]],
    metrics: list[dict[str, Any]], hidden_rows: list[dict[str, Any]],
) -> tuple[
    list[str], list[str], list[int], list[float],
    dict[str, dict[str, Any]], dict[str, Any],
]:
    import numpy as np
    from recurrent.research.mic_v2_e1 import (
        signed_text_hash, text_components_from_state, turn_length_features,
    )

    metric_map = {(row["content_root_id"], row["replica"]): row for row in metrics}
    hidden_map = {
        (row["content_root_id"], row["replica"], row["turn"]): row
        for row in hidden_rows
    }
    _require(len(metric_map) == 512 and len(hidden_map) == len(states),
             "E1 feature join coverage differs")
    root_ids, trajectory_ids, turns, target = [], [], [], []
    stage_masks: dict[str, list[bool]] = {"pre": [], "post": []}
    matrices: dict[str, dict[str, list[Any]]] = {
        name: {"pre": [], "post": []}
        for name in ("turn_length", "signed_text_hash",
                     "actor_hidden_rademacher_128", "actor_hidden_rademacher_256")
    }
    for row in states:
        key = (row["content_root_id"], row["replica"], row["turn"])
        metric, actor = metric_map.get(key[:2]), hidden_map.get(key)
        _require(metric is not None and actor is not None
                 and row["trajectory_id"] == metric["trajectory_id"] == actor["trajectory_id"]
                 and actor.get("available_stages") == row.get("available_stages")
                 and row.get("available_stages") in (["post"], ["pre", "post"]),
                 "E1 feature row join differs")
        root_ids.append(key[0])
        trajectory_ids.append(row["trajectory_id"])
        turns.append(row["turn"])
        target.append(float(metric["target_raw_terminal_return"]))
        for stage in ("pre", "post"):
            available = stage in row["available_stages"]
            stage_masks[stage].append(available)
            if not available:
                matrices["turn_length"][stage].append(np.zeros(5, dtype=np.float64))
                matrices["signed_text_hash"][stage].append(
                    np.zeros(4096, dtype=np.float64)
                )
                for dimension in (128, 256):
                    matrices[f"actor_hidden_rademacher_{dimension}"][stage].append(
                        np.zeros(dimension, dtype=np.float64)
                    )
                continue
            state = row[f"{stage}_state"]
            texts = text_components_from_state(
                state, no_memory_text=manifest["recurrent"]["no_memory_text"],
            )
            memory_count = _exact_materialized_memory_token_count(row, stage)
            matrices["turn_length"][stage].append(turn_length_features(
                turn=row["turn"], arrived_chunk_count=len(state["arrived_chunks"]),
                prior_active_turn_count=state["public_metadata"]["prior_active_turn_count"],
                arrived_context_token_count=state["public_metadata"]["arrived_context_token_count"],
                current_memory_token_count=memory_count,
            ))
            matrices["signed_text_hash"][stage].append(
                signed_text_hash(texts, turn=row["turn"])
            )
            for dimension in (128, 256):
                name = f"actor_hidden_rademacher_{dimension}"
                values = np.asarray(actor[f"{stage}_{name}"], dtype=np.float64)
                _require(values.shape == (dimension,) and np.isfinite(values).all(),
                         "E1 actor-hidden feature row differs")
                matrices[name][stage].append(values)
    features = {
        name: {stage: np.stack(rows, axis=0) for stage, rows in stages.items()}
        for name, stages in matrices.items()
    }
    masks = {stage: np.asarray(values, dtype=bool) for stage, values in stage_masks.items()}
    _require(masks["pre"].sum() + 512 == masks["post"].sum()
             and np.all(masks["pre"] <= masks["post"]),
             "E1 initial-post stage coverage differs")
    return root_ids, trajectory_ids, turns, target, features, masks


def _exact_materialized_memory_token_count(row: Mapping[str, Any], stage: str) -> int:
    """Read the behavior-ledger count; decoded text re-tokenization is not authoritative."""
    _require(stage in ("pre", "post"), "E1 materialized-memory stage differs")
    count = row.get(f"{stage}_materialized_memory_token_count")
    digest = row.get(f"{stage}_materialized_memory_token_sha256")
    _require(type(count) is int and count >= 0
             and isinstance(digest, str) and HEX64.fullmatch(digest) is not None,
             "E1 exact materialized-memory token receipt differs")
    return count


def select_dev(
    repo: Path, expected_commit: str, output_root: Path, run_id: str,
    *, verify_existing: bool = False,
) -> dict[str, Any]:
    _runtime(repo, expected_commit, output_root, run_id)
    stored_p0 = _load(output_root / "certificates/p0.json")
    _require(isinstance(stored_p0.get("work_root"), str)
             and Path(stored_p0["work_root"]).is_absolute()
             and isinstance(stored_p0.get("gpu_pair"), list),
             "E1-dev stored P0 routing differs")
    preflight(
        repo, expected_commit, Path(stored_p0["work_root"]), output_root, run_id,
        "e1_dev", ",".join(str(value) for value in stored_p0["gpu_pair"]),
        verify_existing=True,
    )
    seal_states(
        repo, expected_commit, output_root, run_id, "e1_dev", verify_existing=True,
    )
    score_split(
        repo, expected_commit, output_root, run_id, "e1_dev", verify_existing=True,
    )
    manifest, manifest_sha = _manifest(repo)
    p0, _ledger = _verified_collection(output_root, expected_commit, run_id)
    _require(manifest_sha == p0["manifest_sha256"], "E1 selection manifest differs")
    states_receipt = _verified_self_receipt(
        output_root / "certificates/e1_dev_states.json", "states_sha256",
        "memagent.mic.v2.e1-time-safe-states",
    )
    outcomes_receipt = _verified_self_receipt(
        output_root / "certificates/e1_dev_outcomes.json", "outcomes_sha256",
        "memagent.mic.v2.e1-outcomes",
    )
    hidden_receipt = _verified_self_receipt(
        output_root / "certificates/e1_dev_actor_hidden_features.json", "features_sha256",
        "memagent.mic.v2.e1-actor-hidden-features",
    )
    hidden_replay = _verified_self_receipt(
        output_root / "certificates/e1_dev_actor_hidden_features_replay.json", "features_sha256",
        "memagent.mic.v2.e1-actor-hidden-features-replay",
    )
    for receipt in (states_receipt, outcomes_receipt, hidden_receipt, hidden_replay):
        _require(receipt.get("git_commit") == expected_commit
                 and receipt.get("run_id") == run_id
                 and receipt.get("p0_sha256") == p0["p0_sha256"],
                 "E1 selection input identity differs")
    _require(states_receipt.get("split") == outcomes_receipt.get("split")
             == hidden_receipt.get("split") == hidden_replay.get("split") == "e1_dev",
             "E1 selection split differs")
    execution = _load(output_root / "certificates/e1_dev_execution.json")
    _require(hidden_receipt.get("physical_gpu_identity")
             == hidden_replay.get("physical_gpu_identity")
             == execution.get("physical_gpu_identity")
             and hidden_replay.get("independent_exact_replay") is True
             and hidden_receipt.get("feature_file_sha256")
                 == hidden_replay.get("feature_file_sha256")
             and hidden_receipt.get("feature_canonical_sha256")
                 == hidden_replay.get("feature_canonical_sha256"),
             "E1 collector/feature physical GPU identity differs")
    state_path = Path(states_receipt["state_path"])
    metric_path = Path(outcomes_receipt["metric_rows_path"])
    hidden_path = Path(hidden_receipt["feature_path"])
    _require(sha256_file(state_path) == states_receipt["state_file_sha256"]
             and sha256_file(metric_path) == outcomes_receipt["metric_rows_file_sha256"]
             and sha256_file(hidden_path) == hidden_receipt["feature_file_sha256"],
             "E1 selection input raw SHA differs")
    states, metrics, hidden_rows = map(_jsonl, (state_path, metric_path, hidden_path))
    import numpy as np
    from recurrent.research.mic_v2_e1 import (
        CandidateSpec,
        cross_fitted_predictions,
        fit_head,
        root_trajectory_turn_weights,
        select_specification,
    )
    root_ids, trajectory_ids, turns, target, features, stage_masks = _assemble_feature_rows(
        manifest, states, metrics, hidden_rows,
    )
    numeric = manifest["numeric"]
    selection = select_specification(
        root_ids=root_ids, trajectory_ids=trajectory_ids, turns=turns,
        target=target, features=features, stage_masks=stage_masks,
        tolerance=numeric["optimizer_tolerance"],
        maximum_iterations=numeric["optimizer_maximum_iterations"],
    )
    selected_row = selection["selected"]["specification"]
    representation_order = {
        "turn_length": 0, "signed_text_hash": 1,
        "actor_hidden_rademacher_128": 2, "actor_hidden_rademacher_256": 3,
    }
    selected_spec = CandidateSpec(
        selected_row["representation"], selected_row["dimension"],
        representation_order[selected_row["representation"]], selected_row["head"],
        selected_row["regularization"],
    )
    comparator_candidates = [
        row for row in selection["candidates"]
        if row["status"] == "PASS"
        and row["specification"]["representation"] == "turn_length"
    ]
    _require(comparator_candidates, "every turn/length comparator candidate failed")
    comparator_minimum = min(row["score"] for row in comparator_candidates)
    comparator_tied = [
        row for row in comparator_candidates if row["score"] <= comparator_minimum + 1e-6
    ]
    comparator_tied.sort(key=lambda row: (
        -row["specification"]["regularization"],
        0 if row["specification"]["head"] == "fractional_logistic" else 1,
    ))
    comparator_row = comparator_tied[0]["specification"]
    comparator_spec = CandidateSpec(
        "turn_length", 5, 0, comparator_row["head"], comparator_row["regularization"],
    )
    y = np.asarray(target, dtype=np.float64)
    weights = root_trajectory_turn_weights(root_ids, trajectory_ids, turns)
    selected_oof = cross_fitted_predictions(
        selected_spec, root_ids=root_ids, trajectory_ids=trajectory_ids, turns=turns,
        target=y, stage_features=features[selected_spec.representation],
        stage_masks=stage_masks,
        tolerance=numeric["optimizer_tolerance"],
        maximum_iterations=numeric["optimizer_maximum_iterations"],
    )
    comparator_oof = cross_fitted_predictions(
        comparator_spec, root_ids=root_ids, trajectory_ids=trajectory_ids, turns=turns,
        target=y, stage_features=features["turn_length"],
        stage_masks=stage_masks,
        tolerance=numeric["optimizer_tolerance"],
        maximum_iterations=numeric["optimizer_maximum_iterations"],
    )
    fitted = {}
    for label, spec, feature_name in (
        ("selected", selected_spec, selected_spec.representation),
        ("turn_length_comparator", comparator_spec, "turn_length"),
    ):
        fitted[label] = {
            stage: fit_head(
                spec, features[feature_name][stage][stage_masks[stage]],
                y[stage_masks[stage]],
                root_trajectory_turn_weights(
                    np.asarray(root_ids, dtype=object)[stage_masks[stage]].tolist(),
                    np.asarray(trajectory_ids, dtype=object)[stage_masks[stage]].tolist(),
                    np.asarray(turns, dtype=np.int64)[stage_masks[stage]].tolist(),
                ),
                tolerance=numeric["optimizer_tolerance"],
                maximum_iterations=numeric["optimizer_maximum_iterations"],
            ).receipt()
            for stage in ("pre", "post")
        }
    prediction_rows = [{
        "content_root_id": root_ids[index], "trajectory_id": trajectory_ids[index],
        "turn": turns[index], "target": target[index],
        "available_stages": [
            stage for stage in ("pre", "post") if bool(stage_masks[stage][index])
        ],
        "selected_pre_oof": (
            float(selected_oof["pre"][index]) if stage_masks["pre"][index] else None
        ),
        "selected_post_oof": float(selected_oof["post"][index]),
        "comparator_pre_oof": (
            float(comparator_oof["pre"][index]) if stage_masks["pre"][index] else None
        ),
        "comparator_post_oof": float(comparator_oof["post"][index]),
    } for index in range(len(states))]
    prediction_path = output_root / "selection/e1_dev_selected_oof.jsonl"
    model_path = output_root / "selection/e1_dev_refit_heads.json"
    model_bundle = {
        "schema": "memagent.mic.v2.e1-dev-refit-heads", "status": "PASS",
        "selected_specification": selected_spec.receipt(),
        "turn_length_comparator_specification": comparator_spec.receipt(),
        "heads": fitted,
    }
    model_bundle["heads_sha256"] = sha256_json(model_bundle)
    if verify_existing:
        _require(prediction_path.is_file() and _jsonl(prediction_path) == prediction_rows,
                 "E1-dev selected OOF replay differs")
        _require(model_path.is_file() and _load(model_path) == model_bundle,
                 "E1-dev refit-head replay differs")
    else:
        _write_jsonl_new(prediction_path, prediction_rows)
        write_json_new(model_path, model_bundle)
    report = {
        **selection, "git_commit": expected_commit, "run_id": run_id,
        "p0_sha256": p0["p0_sha256"],
        "selection_inputs_sha256": sha256_json({
            "states": states_receipt["states_sha256"],
            "outcomes": outcomes_receipt["outcomes_sha256"],
            "actor_hidden_features": hidden_receipt["features_sha256"],
            "actor_hidden_features_replay": hidden_replay["features_sha256"],
        }),
        "selected_oof_path": str(prediction_path),
        "selected_oof_file_sha256": sha256_file(prediction_path),
        "selected_oof_canonical_sha256": sha256_json(prediction_rows),
        "refit_heads_path": str(model_path), "refit_heads_file_sha256": sha256_file(model_path),
        "holdout_opened": False,
    }
    report["selection_sha256"] = sha256_json(report)
    selection_path = output_root / "certificates/e1_dev_selection.json"
    if verify_existing:
        _require(selection_path.is_file() and _load(selection_path) == report,
                 "E1-dev full selection replay differs")
    else:
        write_json_new(selection_path, report)
    return report


def _root_averages(
    values: Any, root_ids: list[str], trajectory_ids: list[str], turns: list[int],
) -> tuple[list[str], Any]:
    import numpy as np
    from recurrent.research.mic_v2_e1 import root_trajectory_turn_weights, stable_selection_fold
    array = np.asarray(values, dtype=np.float64)
    _require(array.shape == (len(root_ids),) and np.isfinite(array).all(),
             "root-average values differ")
    weights = root_trajectory_turn_weights(root_ids, trajectory_ids, turns)
    roots = sorted(set(root_ids))
    result = []
    for root in roots:
        selected = np.asarray([candidate == root for candidate in root_ids])
        conditional = weights[selected] / weights[selected].sum()
        result.append(float(conditional @ array[selected]))
    return roots, np.asarray(result, dtype=np.float64)


def _paired_root_bootstrap_lower(
    root_arrays: list[Any], contrast: Any, *, replicates: int, seed: int,
) -> tuple[float, float, float]:
    import numpy as np
    arrays = [np.asarray(value, dtype=np.float64) for value in root_arrays]
    _require(arrays and all(value.shape == arrays[0].shape for value in arrays)
             and arrays[0].ndim == 1 and arrays[0].size > 1,
             "root-bootstrap arrays differ")
    point = float(contrast(*[value.mean() for value in arrays]))
    generator = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = generator.integers(0, arrays[0].size, size=arrays[0].size)
        samples[index] = contrast(*[value[draw].mean() for value in arrays])
    _require(np.isfinite(samples).all(), "root bootstrap is non-finite")
    return point, float(np.percentile(samples, 5.0)), float(np.percentile(samples, 95.0))


def _root_cluster_diagnostic_intervals(
    *, target: Any, pre: Any, post: Any, credit: Any,
    root_ids: list[str], trajectory_ids: list[str], turns: list[int],
    occupied_cells: list[dict[str, Any]], minimum_roots_per_cell: int,
    replicates: int, seed: int,
) -> dict[str, Any]:
    """Paired content-root intervals for all non-MSE E1 diagnostics."""
    import numpy as np
    from recurrent.research.mic_v2_e1 import root_trajectory_turn_weights, stable_selection_fold

    roots = sorted(set(root_ids))
    _require(len(roots) > 1 and replicates > 0, "diagnostic bootstrap authority differs")
    weights = root_trajectory_turn_weights(root_ids, trajectory_ids, turns)
    vectors = {
        "y": np.asarray(target, dtype=np.float64),
        "pre": np.asarray(pre, dtype=np.float64),
        "post": np.asarray(post, dtype=np.float64),
        "pre2": np.square(np.asarray(pre, dtype=np.float64)),
        "post2": np.square(np.asarray(post, dtype=np.float64)),
        "ypre": np.asarray(target, dtype=np.float64) * np.asarray(pre, dtype=np.float64),
        "ypost": np.asarray(target, dtype=np.float64) * np.asarray(post, dtype=np.float64),
        "credit": np.asarray(credit, dtype=np.float64),
        "abs_credit": np.abs(np.asarray(credit, dtype=np.float64)),
        "credit2": np.square(np.asarray(credit, dtype=np.float64)),
    }
    _require(all(value.shape == (len(root_ids),) and np.isfinite(value).all()
                 for value in vectors.values()), "diagnostic bootstrap rows differ")
    root_moments = {name: [] for name in vectors}
    for root in roots:
        selected = np.asarray([candidate == root for candidate in root_ids])
        conditional = weights[selected] / weights[selected].sum()
        for name, values in vectors.items():
            root_moments[name].append(float(conditional @ values[selected]))
    root_moments = {
        name: np.asarray(values, dtype=np.float64) for name, values in root_moments.items()
    }
    membership = np.zeros((len(roots), len(occupied_cells)), dtype=np.float64)
    for root_index, root in enumerate(roots):
        fold = stable_selection_fold(root)
        active_turns = {
            turns[index] for index, candidate in enumerate(root_ids) if candidate == root
        }
        for cell_index, cell in enumerate(occupied_cells):
            membership[root_index, cell_index] = float(
                fold == cell["fold"] and cell["turn"] in active_turns
            )

    def statistics(indices: Any) -> dict[str, Any]:
        moment = {name: float(values[indices].mean()) for name, values in root_moments.items()}
        calibration_rows = {}
        for stage in ("pre", "post"):
            variance = moment[f"{stage}2"] - moment[stage] ** 2
            _require(variance > 0 and math.isfinite(variance),
                     "bootstrap calibration variance is zero")
            covariance = moment[f"y{stage}"] - moment["y"] * moment[stage]
            slope = covariance / variance
            calibration_rows[stage] = {
                "slope": slope, "intercept": moment["y"] - slope * moment[stage],
            }
        credit_variance = moment["credit2"] - moment["credit"] ** 2
        counts = membership[indices].sum(axis=0)
        eligible = counts >= minimum_roots_per_cell
        return {
            "calibration": calibration_rows,
            "credit_mean": moment["credit"],
            "credit_mean_absolute": moment["abs_credit"],
            "credit_variance": max(0.0, credit_variance),
            "eligible_cell_fraction": float(eligible.mean()),
            "cell_root_counts": counts,
            "cell_eligible": eligible,
        }

    point = statistics(np.arange(len(roots)))
    generator = np.random.Generator(np.random.PCG64(seed))
    samples = {
        "pre_slope": [], "pre_intercept": [], "post_slope": [], "post_intercept": [],
        "credit_mean": [], "credit_mean_absolute": [], "credit_variance": [],
        "eligible_cell_fraction": [],
    }
    cell_counts = np.empty((replicates, len(occupied_cells)), dtype=np.float64)
    cell_eligible = np.empty((replicates, len(occupied_cells)), dtype=np.float64)
    for bootstrap_index in range(replicates):
        draw = generator.integers(0, len(roots), size=len(roots))
        row = statistics(draw)
        for stage in ("pre", "post"):
            samples[f"{stage}_slope"].append(row["calibration"][stage]["slope"])
            samples[f"{stage}_intercept"].append(row["calibration"][stage]["intercept"])
        for name in ("credit_mean", "credit_mean_absolute", "credit_variance",
                     "eligible_cell_fraction"):
            samples[name].append(row[name])
        cell_counts[bootstrap_index] = row["cell_root_counts"]
        cell_eligible[bootstrap_index] = row["cell_eligible"]

    def interval(values: Any) -> list[float]:
        array = np.asarray(values, dtype=np.float64)
        _require(np.isfinite(array).all(), "diagnostic bootstrap is non-finite")
        return [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))]

    failure_cells = []
    for index, cell in enumerate(occupied_cells):
        failure_cells.append({
            **cell,
            "root_count_two_sided_95": interval(cell_counts[:, index]),
            "bootstrap_eligibility_probability": float(cell_eligible[:, index].mean()),
        })
    return {
        "calibration": {
            stage: {
                "slope": point["calibration"][stage]["slope"],
                "slope_two_sided_95": interval(samples[f"{stage}_slope"]),
                "intercept": point["calibration"][stage]["intercept"],
                "intercept_two_sided_95": interval(samples[f"{stage}_intercept"]),
            }
            for stage in ("pre", "post")
        },
        "writer_credit": {
            "weighted_mean": point["credit_mean"],
            "weighted_mean_two_sided_95": interval(samples["credit_mean"]),
            "weighted_mean_absolute": point["credit_mean_absolute"],
            "weighted_mean_absolute_two_sided_95": interval(samples["credit_mean_absolute"]),
            "weighted_variance": point["credit_variance"],
            "weighted_variance_two_sided_95": interval(samples["credit_variance"]),
        },
        "coverage": {
            "eligible_cell_fraction": point["eligible_cell_fraction"],
            "eligible_cell_fraction_two_sided_95": interval(
                samples["eligible_cell_fraction"]
            ),
            "cells": failure_cells,
        },
        "replicates": replicates, "seed": seed,
    }


def _root_cluster_calibration_interval(
    *, target: Any, prediction: Any, root_ids: list[str],
    trajectory_ids: list[str], turns: list[int], replicates: int, seed: int,
) -> dict[str, Any]:
    """Stage-specific calibration interval, including X0+ for the post head."""
    import numpy as np
    y = np.asarray(target, dtype=np.float64)
    value = np.asarray(prediction, dtype=np.float64)
    _require(y.shape == value.shape == (len(root_ids),)
             and np.isfinite(y).all() and np.isfinite(value).all(),
             "calibration bootstrap rows differ")
    from recurrent.research.mic_v2_e1 import root_trajectory_turn_weights
    weights = root_trajectory_turn_weights(root_ids, trajectory_ids, turns)
    roots = sorted(set(root_ids))
    root_moments = {name: [] for name in ("y", "p", "p2", "yp")}
    vectors = {"y": y, "p": value, "p2": np.square(value), "yp": y * value}
    for root in roots:
        selected = np.asarray([candidate == root for candidate in root_ids])
        conditional = weights[selected] / weights[selected].sum()
        for name, vector in vectors.items():
            root_moments[name].append(float(conditional @ vector[selected]))
    root_moments = {
        name: np.asarray(rows, dtype=np.float64) for name, rows in root_moments.items()
    }

    def statistic(indices: Any) -> tuple[float, float]:
        moment = {name: float(rows[indices].mean()) for name, rows in root_moments.items()}
        variance = moment["p2"] - moment["p"] ** 2
        _require(variance > 0 and math.isfinite(variance),
                 "bootstrap calibration variance is zero")
        slope = (moment["yp"] - moment["y"] * moment["p"]) / variance
        return slope, moment["y"] - slope * moment["p"]

    point = statistic(np.arange(len(roots)))
    generator = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty((replicates, 2), dtype=np.float64)
    for index in range(replicates):
        samples[index] = statistic(
            generator.integers(0, len(roots), size=len(roots))
        )
    _require(np.isfinite(samples).all(), "calibration bootstrap is non-finite")
    return {
        "slope": point[0],
        "slope_two_sided_95": [
            float(np.percentile(samples[:, 0], 2.5)),
            float(np.percentile(samples[:, 0], 97.5)),
        ],
        "intercept": point[1],
        "intercept_two_sided_95": [
            float(np.percentile(samples[:, 1], 2.5)),
            float(np.percentile(samples[:, 1], 97.5)),
        ],
    }


def _coverage_cells(
    root_ids: list[str], turns: list[int], maximum_writer_slots: int,
    minimum_roots_per_cell: int,
) -> tuple[list[dict[str, Any]], float]:
    from recurrent.research.mic_v2_e1 import stable_selection_fold
    _require(len(root_ids) == len(turns) and bool(root_ids), "coverage rows differ")
    occupied = []
    for fold in range(4):
        for turn in range(1, maximum_writer_slots + 1):
            roots = {
                root_ids[index] for index in range(len(root_ids))
                if stable_selection_fold(root_ids[index]) == fold and turns[index] == turn
            }
            if roots:
                occupied.append({
                    "fold": fold, "turn": turn, "root_count": len(roots),
                    "eligible": len(roots) >= minimum_roots_per_cell,
                })
    _require(bool(occupied), "holdout has no occupied fold-turn cells")
    return occupied, sum(cell["eligible"] for cell in occupied) / len(occupied)


def evaluate_holdout(
    repo: Path, expected_commit: str, output_root: Path, run_id: str,
    *, verify_existing: bool = False,
) -> dict[str, Any]:
    _runtime(repo, expected_commit, output_root, run_id)
    stored_p0 = _load(output_root / "certificates/p0.json")
    _require(isinstance(stored_p0.get("work_root"), str)
             and Path(stored_p0["work_root"]).is_absolute()
             and isinstance(stored_p0.get("gpu_pair"), list)
             and isinstance(stored_p0.get("dev_selection_root"), str),
             "holdout stored P0 routing differs")
    preflight(
        repo, expected_commit, Path(stored_p0["work_root"]), output_root, run_id,
        "e1_holdout", ",".join(str(value) for value in stored_p0["gpu_pair"]),
        Path(stored_p0["dev_selection_root"]), verify_existing=True,
    )
    seal_states(
        repo, expected_commit, output_root, run_id, "e1_holdout", verify_existing=True,
    )
    score_split(
        repo, expected_commit, output_root, run_id, "e1_holdout", verify_existing=True,
    )
    manifest, manifest_sha = _manifest(repo)
    p0, _ledger = _verified_collection(
        output_root, expected_commit, run_id, "e1_holdout",
    )
    _require(manifest_sha == p0["manifest_sha256"], "holdout manifest differs")
    states_receipt = _verified_self_receipt(
        output_root / "certificates/e1_holdout_states.json", "states_sha256",
        "memagent.mic.v2.e1-time-safe-states",
    )
    outcomes_receipt = _verified_self_receipt(
        output_root / "certificates/e1_holdout_outcomes.json", "outcomes_sha256",
        "memagent.mic.v2.e1-outcomes",
    )
    hidden_receipt = _verified_self_receipt(
        output_root / "certificates/e1_holdout_actor_hidden_features.json", "features_sha256",
        "memagent.mic.v2.e1-actor-hidden-features",
    )
    hidden_replay = _verified_self_receipt(
        output_root / "certificates/e1_holdout_actor_hidden_features_replay.json",
        "features_sha256", "memagent.mic.v2.e1-actor-hidden-features-replay",
    )
    for receipt in (states_receipt, outcomes_receipt, hidden_receipt, hidden_replay):
        _require(receipt.get("split") == "e1_holdout"
                 and receipt.get("git_commit") == expected_commit
                 and receipt.get("run_id") == run_id
                 and receipt.get("p0_sha256") == p0["p0_sha256"],
                 "holdout input identity differs")
    execution = _load(output_root / "certificates/e1_holdout_execution.json")
    _require(hidden_receipt.get("physical_gpu_identity")
             == hidden_replay.get("physical_gpu_identity")
             == execution.get("physical_gpu_identity")
             and hidden_replay.get("independent_exact_replay") is True
             and hidden_receipt.get("feature_file_sha256")
                 == hidden_replay.get("feature_file_sha256")
             and hidden_receipt.get("feature_canonical_sha256")
                 == hidden_replay.get("feature_canonical_sha256"),
             "holdout collector/feature physical GPU identity differs")
    state_path = Path(states_receipt["state_path"])
    metric_path = Path(outcomes_receipt["metric_rows_path"])
    hidden_path = Path(hidden_receipt["feature_path"])
    _require(sha256_file(state_path) == states_receipt["state_file_sha256"]
             and sha256_file(metric_path) == outcomes_receipt["metric_rows_file_sha256"]
             and sha256_file(hidden_path) == hidden_receipt["feature_file_sha256"],
             "holdout input SHA differs")
    states, metrics, hidden_rows = map(_jsonl, (state_path, metric_path, hidden_path))
    root_ids, trajectory_ids, turns, target, features, stage_masks = _assemble_feature_rows(
        manifest, states, metrics, hidden_rows,
    )

    import numpy as np
    from recurrent.research.mic_v2_e1 import FittedHead, calibration
    dev_root = Path(p0["dev_selection_root"])
    selection_path = dev_root / "certificates/e1_dev_selection.json"
    selection = _load(selection_path)
    unsigned_selection = dict(selection)
    selection_sha = unsigned_selection.pop("selection_sha256", None)
    opening = p0["dev_selection_authority"]
    _require(selection_sha == sha256_json(unsigned_selection)
             and sha256_file(selection_path) == opening["dev_selection_file_sha256"]
             and selection_sha == opening["dev_selection_canonical_sha256"],
             "holdout dev-selection authority differs")
    head_path = Path(selection["refit_heads_path"])
    _require(head_path.is_file()
             and sha256_file(head_path) == selection["refit_heads_file_sha256"],
             "holdout refit-head authority differs")
    bundle = _load(head_path)
    unsigned_bundle = dict(bundle)
    heads_sha = unsigned_bundle.pop("heads_sha256", None)
    _require(heads_sha == sha256_json(unsigned_bundle)
             and bundle.get("schema") == "memagent.mic.v2.e1-dev-refit-heads"
             and bundle.get("status") == "PASS", "holdout head bundle differs")
    selected_name = bundle["selected_specification"]["representation"]
    y = np.asarray(target, dtype=np.float64)
    predictions = {}
    for label, feature_name in (
        ("selected", selected_name), ("turn_length_comparator", "turn_length"),
    ):
        predictions[label] = {
            stage: np.where(
                stage_masks[stage],
                FittedHead.from_receipt(bundle["heads"][label][stage]).predict(
                    features[feature_name][stage]
                ),
                np.nan,
            )
            for stage in ("pre", "post")
        }
    squared = {
        label: {stage: np.square(y - predictions[label][stage]) for stage in ("pre", "post")}
        for label in predictions
    }
    root_order = None
    root_values = {}
    for label in squared:
        root_values[label] = {}
        for stage in ("pre", "post"):
            roots, values = _root_averages(
                squared[label][stage][stage_masks[stage]],
                np.asarray(root_ids, dtype=object)[stage_masks[stage]].tolist(),
                np.asarray(trajectory_ids, dtype=object)[stage_masks[stage]].tolist(),
                np.asarray(turns, dtype=np.int64)[stage_masks[stage]].tolist(),
            )
            _require(root_order is None or root_order == roots, "holdout root order differs")
            root_order = roots
            root_values[label][stage] = values
    bootstrap = manifest["holdout_gates"]["bootstrap"]
    contrasts = {}
    for stage in ("pre", "post"):
        point, lower, upper = _paired_root_bootstrap_lower(
            [root_values["turn_length_comparator"][stage], root_values["selected"][stage]],
            lambda baseline, selected: (baseline - selected) / baseline,
            replicates=bootstrap["replicates"], seed=bootstrap["seed"],
        )
        contrasts[f"selected_vs_turn_length_{stage}"] = {
            "relative_mse_reduction": point, "one_sided_95_lower": lower,
            "central_90_upper": upper,
        }
    point, lower, upper = _paired_root_bootstrap_lower(
        [
            _root_averages(
                squared["selected"]["pre"][stage_masks["pre"]],
                np.asarray(root_ids, dtype=object)[stage_masks["pre"]].tolist(),
                np.asarray(trajectory_ids, dtype=object)[stage_masks["pre"]].tolist(),
                np.asarray(turns, dtype=np.int64)[stage_masks["pre"]].tolist(),
            )[1],
            _root_averages(
                squared["selected"]["post"][stage_masks["pre"]],
                np.asarray(root_ids, dtype=object)[stage_masks["pre"]].tolist(),
                np.asarray(trajectory_ids, dtype=object)[stage_masks["pre"]].tolist(),
                np.asarray(turns, dtype=np.int64)[stage_masks["pre"]].tolist(),
            )[1],
        ],
        lambda pre, post: (pre - post) / pre,
        replicates=bootstrap["replicates"], seed=bootstrap["seed"],
    )
    contrasts["post_vs_pre"] = {
        "relative_mse_reduction": point, "one_sided_95_lower": lower,
        "central_90_upper": upper,
    }
    from recurrent.research.mic_v2_e1 import root_trajectory_turn_weights
    paired = stage_masks["pre"]
    paired_roots = np.asarray(root_ids, dtype=object)[paired].tolist()
    paired_trajectories = np.asarray(trajectory_ids, dtype=object)[paired].tolist()
    paired_turns = np.asarray(turns, dtype=np.int64)[paired].tolist()
    weights = root_trajectory_turn_weights(paired_roots, paired_trajectories, paired_turns)
    write_credit = (
        predictions["selected"]["post"][paired]
        - predictions["selected"]["pre"][paired]
    )
    weighted_credit_mean = float(weights @ write_credit)
    credit_diagnostics = {
        "weighted_mean": weighted_credit_mean,
        "weighted_mean_absolute": float(weights @ np.abs(write_credit)),
        "weighted_variance": float(weights @ np.square(write_credit - weighted_credit_mean)),
        "minimum": float(write_credit.min()), "maximum": float(write_credit.max()),
    }
    calibrations = {
        stage: calibration(
            y[stage_masks[stage]], predictions["selected"][stage][stage_masks[stage]],
            root_trajectory_turn_weights(
                np.asarray(root_ids, dtype=object)[stage_masks[stage]].tolist(),
                np.asarray(trajectory_ids, dtype=object)[stage_masks[stage]].tolist(),
                np.asarray(turns, dtype=np.int64)[stage_masks[stage]].tolist(),
            ),
        )
        for stage in ("pre", "post")
    }
    trajectory_rows: dict[str, list[int]] = {}
    for index, trajectory in enumerate(trajectory_ids):
        trajectory_rows.setdefault(trajectory, []).append(index)
    closure_errors = []
    for indices in trajectory_rows.values():
        indices.sort(key=lambda index: turns[index])
        _require(turns[indices[0]] == 0 and not stage_masks["pre"][indices[0]]
                 and all(stage_masks["pre"][index] for index in indices[1:]),
                 "holdout initial-post trajectory structure differs")
        pre = predictions["selected"]["pre"][indices[1:]]
        post = predictions["selected"]["post"][indices]
        returns = y[indices]
        _require(np.all(returns == returns[0]), "holdout trajectory target drifted by turn")
        reconstructed = (
            post[0] + np.sum(pre - post[:-1])
            + np.sum(post[1:] - pre) + (returns[0] - post[-1])
        )
        closure_errors.append(abs(float(reconstructed - returns[0])))
    maximum_closure_error = max(closure_errors)
    occupied, eligible_fraction = _coverage_cells(
        paired_roots, paired_turns, manifest["recurrent"]["max_writer_slots"],
        manifest["holdout_gates"]["minimum_roots_per_occupied_fold_turn"],
    )
    diagnostic_intervals = _root_cluster_diagnostic_intervals(
        target=y[paired], pre=predictions["selected"]["pre"][paired],
        post=predictions["selected"]["post"][paired], credit=write_credit,
        root_ids=paired_roots, trajectory_ids=paired_trajectories, turns=paired_turns,
        occupied_cells=occupied,
        minimum_roots_per_cell=manifest["holdout_gates"][
            "minimum_roots_per_occupied_fold_turn"
        ],
        replicates=bootstrap["replicates"], seed=bootstrap["seed"],
    )
    diagnostic_intervals["calibration"] = {
        stage: _root_cluster_calibration_interval(
            target=y[stage_masks[stage]],
            prediction=predictions["selected"][stage][stage_masks[stage]],
            root_ids=np.asarray(root_ids, dtype=object)[stage_masks[stage]].tolist(),
            trajectory_ids=np.asarray(trajectory_ids, dtype=object)[
                stage_masks[stage]
            ].tolist(),
            turns=np.asarray(turns, dtype=np.int64)[stage_masks[stage]].tolist(),
            replicates=bootstrap["replicates"], seed=bootstrap["seed"],
        ) for stage in ("pre", "post")
    }
    _require(all(
        abs(diagnostic_intervals["calibration"][stage][name]
            - calibrations[stage][name]) <= 1e-12
        for stage in ("pre", "post") for name in ("slope", "intercept")
    ), "calibration interval point estimate differs")
    _require(abs(diagnostic_intervals["coverage"]["eligible_cell_fraction"]
                 - eligible_fraction) <= 1e-15,
             "coverage point estimate differs from root-cluster reconstruction")
    gates = {
        "selected_pre_vs_comparator": (
            contrasts["selected_vs_turn_length_pre"]["relative_mse_reduction"]
                >= manifest["holdout_gates"]["minimum_relative_mse_reduction"]
            and contrasts["selected_vs_turn_length_pre"]["one_sided_95_lower"] > 0
        ),
        "selected_post_vs_comparator": (
            contrasts["selected_vs_turn_length_post"]["relative_mse_reduction"]
                >= manifest["holdout_gates"]["minimum_relative_mse_reduction"]
            and contrasts["selected_vs_turn_length_post"]["one_sided_95_lower"] > 0
        ),
        "post_vs_pre": (
                        contrasts["post_vs_pre"]["relative_mse_reduction"]
                            >= manifest["holdout_gates"]["minimum_relative_mse_reduction"]
                        and contrasts["post_vs_pre"]["one_sided_95_lower"] > 0),
        "calibration": all(
            manifest["holdout_gates"]["calibration_slope"][0]
                <= row["slope"]
                <= manifest["holdout_gates"]["calibration_slope"][1]
            and abs(row["intercept"])
                <= manifest["holdout_gates"]["maximum_absolute_calibration_intercept"]
            for row in calibrations.values()
        ),
        "coverage": eligible_fraction
            >= manifest["holdout_gates"]["minimum_eligible_cell_fraction"],
        "finite": all(
            np.isfinite(label[stage][stage_masks[stage]]).all()
            for label in predictions.values() for stage in ("pre", "post")
        ),
        "closure": math.isfinite(maximum_closure_error) and maximum_closure_error <= 1e-10,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    prediction_rows = [{
        "content_root_id": root_ids[index], "trajectory_id": trajectory_ids[index],
        "turn": turns[index], "target": target[index],
        "available_stages": [
            stage for stage in ("pre", "post") if bool(stage_masks[stage][index])
        ],
        "selected_pre": (
            float(predictions["selected"]["pre"][index])
            if stage_masks["pre"][index] else None
        ),
        "selected_post": float(predictions["selected"]["post"][index]),
        "comparator_pre": (
            float(predictions["turn_length_comparator"]["pre"][index])
            if stage_masks["pre"][index] else None
        ),
        "comparator_post": float(predictions["turn_length_comparator"]["post"][index]),
    } for index in range(len(states))]
    prediction_path = output_root / "evaluation/e1_holdout_predictions.jsonl"
    if verify_existing:
        _require(prediction_path.is_file() and _jsonl(prediction_path) == prediction_rows,
                 "holdout prediction replay differs")
    else:
        _write_jsonl_new(prediction_path, prediction_rows)
    report = {
        "schema": "memagent.mic.v2.e1-holdout-evaluation", "status": status,
        "decision": "MIC_V2_E1_HOLDOUT_PASS" if status == "PASS" else "MIC_V2_E1_HOLDOUT_NO_GO",
        "git_commit": expected_commit, "run_id": run_id, "p0_sha256": p0["p0_sha256"],
        "dev_selection_canonical_sha256": selection_sha,
        "dev_heads_canonical_sha256": heads_sha,
        "holdout_refit_performed": False, "holdout_recalibration_performed": False,
        "row_count": len(states), "root_count": 128,
        "contrasts": contrasts, "calibration": calibrations,
        "writer_credit_diagnostics": credit_diagnostics,
        "root_clustered_diagnostic_intervals": diagnostic_intervals,
        "maximum_algebraic_closure_error": maximum_closure_error,
        "coverage_cells": occupied, "eligible_cell_fraction": eligible_fraction,
        "gates": gates, "predictions_path": str(prediction_path),
        "predictions_file_sha256": sha256_file(prediction_path),
        "predictions_canonical_sha256": sha256_json(prediction_rows),
        "actor_experiment_licensed_by_e1_only": False,
        "actor_experiment_requires": ["E0_PASS", "E1_HOLDOUT_PASS", "FORMAL_BRANCHING_ORACLE_PASS"],
    }
    report["holdout_sha256"] = sha256_json(report)
    certificate_path = output_root / "certificates/e1_holdout_evaluation.json"
    if verify_existing:
        _require(certificate_path.is_file() and _load(certificate_path) == report,
                 "holdout full evaluation replay differs")
    else:
        write_json_new(certificate_path, report)
    _require(status == "PASS",
             "E1 holdout scientific gates failed; sealed NO-GO evidence was preserved")
    return report


def verify_holdout_final(
    repo: Path, expected_commit: str, output_root: Path, run_id: str,
) -> dict[str, Any]:
    """Final verifier: fresh GPU replay first, then full CPU evidence reconstruction."""
    stored_p0 = _load(output_root / "certificates/p0.json")
    pair = stored_p0.get("gpu_pair")
    _require(isinstance(pair, list) and len(pair) == 2,
             "holdout final GPU-pair routing differs")
    gpu = _execute_split_gpu_replay(
        repo, expected_commit, output_root, run_id, "e1_holdout",
        ",".join(str(value) for value in pair),
    )
    evaluation = evaluate_holdout(
        repo, expected_commit, output_root, run_id, verify_existing=True,
    )
    evaluation_path = output_root / "certificates/e1_holdout_evaluation.json"
    final = {
        "schema": "memagent.mic.v2.e1-holdout-final-verification",
        "status": "PASS", "decision": "MIC_V2_E1_HOLDOUT_FINAL_VERIFIED_PASS",
        "git_commit": expected_commit, "run_id": run_id,
        "output_root": str(output_root), "p0_sha256": stored_p0.get("p0_sha256"),
        "fresh_holdout_gpu_replay": gpu,
        "evaluation_file_sha256": sha256_file(evaluation_path),
        "evaluation_canonical_sha256": evaluation["holdout_sha256"],
        "actor_experiment_licensed_by_e1_only": False,
    }
    final["final_verification_sha256"] = sha256_json(final)
    path = output_root / "certificates/e1_holdout_final_verification.json"
    if path.exists():
        _require(_load(path) == final, "holdout final verification replay differs")
    else:
        write_json_new(path, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "preflight-dev", "preflight-holdout", "seal-dev-states", "seal-holdout-states",
        "score-dev", "score-holdout", "select-dev", "verify-dev",
        "evaluate-holdout", "verify-holdout",
    ))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-pair", required=True)
    parser.add_argument("--dev-root", type=Path)
    args = parser.parse_args()
    if args.command in ("preflight-dev", "preflight-holdout"):
        result = preflight(
            args.repo.resolve(), args.expected_commit, args.work_root.resolve(),
            args.output_root.resolve(), args.run_id,
            "e1_dev" if args.command == "preflight-dev" else "e1_holdout",
            args.gpu_pair, args.dev_root.resolve() if args.dev_root else None,
        )
    elif args.command in ("seal-dev-states", "seal-holdout-states"):
        result = seal_states(
            args.repo.resolve(), args.expected_commit,
            args.output_root.resolve(), args.run_id,
            "e1_dev" if args.command == "seal-dev-states" else "e1_holdout",
        )
    elif args.command in ("score-dev", "score-holdout"):
        result = score_split(
            args.repo.resolve(), args.expected_commit,
            args.output_root.resolve(), args.run_id,
            "e1_dev" if args.command == "score-dev" else "e1_holdout",
        )
    elif args.command in ("select-dev", "verify-dev"):
        result = select_dev(
            args.repo.resolve(), args.expected_commit,
            args.output_root.resolve(), args.run_id,
            verify_existing=args.command == "verify-dev",
        )
    elif args.command == "evaluate-holdout":
        result = evaluate_holdout(
            args.repo.resolve(), args.expected_commit,
            args.output_root.resolve(), args.run_id,
        )
    else:
        result = verify_holdout_final(
            args.repo.resolve(), args.expected_commit,
            args.output_root.resolve(), args.run_id,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
