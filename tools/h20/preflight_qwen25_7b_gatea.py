#!/usr/bin/env python3
"""Fail-closed, read-only P0 preflight for the frozen H20 Qwen2.5-7B Gate A."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_GIT_OBJECTS = (
    "scripts/h20/run_qwen25_7b_gatea_fresh2.sh",
    "scripts/h20/resume_qwen25_7b_gatea_step2_to3.sh",
    "tools/h20/preflight_qwen25_7b_gatea.py",
    "tools/h20/audit_qwen25_7b_gatea.py",
    "manifests/h20/qwen25_7b_gatea_seed2026.yaml",
    "manifests/h20/qwen25_7b_gatea_commands.json",
    "schemas/h20/gate_a_execution_ledger.schema.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def load_manifest(path: Path) -> dict:
    # JSON is a strict YAML subset, avoiding an undeclared parser dependency in P0.
    return json.loads(path.read_text(encoding="utf-8"))


def run_preflight(manifest_path: Path, check_runtime: bool) -> dict:
    manifest_path = manifest_path.resolve()
    repo = manifest_path.parents[2]
    manifest = load_manifest(manifest_path)
    failures: list[str] = []
    evidence: dict[str, object] = {}

    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    status = git(repo, "status", "--porcelain")
    evidence.update(git_commit=head, branch=branch, worktree_clean=not status)
    if branch != manifest["branch"]:
        failures.append(f"branch mismatch: {branch} != {manifest['branch']}")
    if status:
        failures.append(f"Git worktree is not clean: {status.splitlines()}")
    if subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", manifest["base_commit"], head],
        check=False,
    ).returncode:
        failures.append(f"HEAD does not contain base commit {manifest['base_commit']}")

    missing_git = [path for path in REQUIRED_GIT_OBJECTS if not (repo / path).is_file()]
    untracked_git = [
        path for path in REQUIRED_GIT_OBJECTS if subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
    ]
    if missing_git:
        failures.append(f"missing required Git artifacts: {missing_git}")
    if untracked_git:
        failures.append(f"required artifacts are not committed: {untracked_git}")

    gpu = manifest["gpu"]
    if gpu != {
        "declared_whitelist": [4, 5, 6, 7],
        "visible_devices": "4,5,6,7",
        "world_size": 4,
        "fsdp_size": 4,
        "trainer_gpus": 4,
        "tensor_parallel_size": 1,
    }:
        failures.append(f"GPU/FSDP configuration is not the frozen four-rank shape: {gpu}")
    backend = manifest["backend"]
    if backend != {
        "rollout": "vllm",
        "evaluation": "vllm",
        "allow_hf_fallback": False,
        "reward_manager": "naive",
    }:
        failures.append(f"backend configuration drift: {backend}")

    data_spec = dict(manifest["data"])
    expected_data_manifest_sha = data_spec.pop("manifest_sha256")
    actual_data_manifest_sha = canonical_sha256(data_spec)
    evidence["data_manifest_sha256"] = actual_data_manifest_sha
    if actual_data_manifest_sha != expected_data_manifest_sha:
        failures.append(
            f"data manifest SHA mismatch: {actual_data_manifest_sha} != {expected_data_manifest_sha}"
        )

    paths = manifest["paths"]
    if Path(paths["resume_source"]) != Path(paths["fresh_output"]) / "global_step_2":
        failures.append("resume source is not the exact P1 global_step_2 directory")
    for key in ("fresh_output", "resume_output"):
        if Path(paths[key]).exists():
            failures.append(f"frozen output already exists: {paths[key]}")

    model = manifest["model"]
    evidence["model_id"] = model["id"]
    evidence["model_revision"] = model["revision"]
    model_config = Path(model["path"]) / "config.json"
    train = Path(manifest["data"]["train"])
    validation = Path(manifest["data"]["validation"])
    runtime_paths = {
        "python": Path(manifest["python"]),
        "repository": Path(manifest["repository"]),
        "model_config": model_config,
        "train": train,
        "validation": validation,
    }
    for name, path in runtime_paths.items():
        if not path.exists():
            failures.append(f"missing frozen runtime path {name}: {path}")
    if repo.resolve() != Path(manifest["repository"]).resolve():
        failures.append(
            f"checkout path mismatch: {repo.resolve()} != {Path(manifest['repository']).resolve()}"
        )
    if model_config.is_file():
        actual = sha256_file(model_config)
        evidence["model_config_sha256"] = actual
        if actual != model["config_sha256"]:
            failures.append(f"model config SHA mismatch: {actual} != {model['config_sha256']}")
    for name, path in (("train", train), ("validation", validation)):
        if path.is_file():
            evidence[f"{name}_data_sha256"] = sha256_file(path)

    commands = json.loads((repo / "manifests/h20/qwen25_7b_gatea_commands.json").read_text())
    evidence["command_manifest_sha256"] = canonical_sha256(commands)
    json.loads((repo / "schemas/h20/gate_a_execution_ledger.schema.json").read_text())

    if check_runtime and Path(manifest["python"]).is_file():
        completed = subprocess.run(
            [manifest["python"], "-c", "import torch, vllm, verl; print(torch.__version__)"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        evidence["runtime_import_stdout"] = completed.stdout.strip()
        if completed.returncode:
            failures.append(f"torch/vLLM/verl import failed: {completed.stderr.strip()}")

    return {
        "gate": "P0",
        "status": "PASS" if not failures else "FAIL",
        "decision": "P0_PASS" if not failures else "GATE_A_NO_GO:P0",
        "failures": failures,
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--write-certificate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_preflight(args.manifest, args.check_runtime)
    manifest = load_manifest(args.manifest)
    if args.write_certificate:
        target = args.output or Path(manifest["paths"]["certificate_root"]) / "p0_preflight.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise SystemExit(f"refusing to overwrite append-only certificate: {target}")
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        ledger = Path(manifest["paths"]["execution_ledger"])
        ledger.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "record_type": "p0_preflight",
            "experiment_name": manifest["experiments"]["p0"],
            "git_commit": result["evidence"]["git_commit"],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "status": result["status"],
            "certificate": str(target),
            "manifest_sha256": sha256_file(args.manifest),
            "evidence": result["evidence"],
            "failures": result["failures"],
        }
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
