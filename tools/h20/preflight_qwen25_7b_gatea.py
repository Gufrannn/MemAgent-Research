#!/usr/bin/env python3
"""Fail-closed, read-only P0 preflight for the frozen H20 Qwen2.5-7B Gate A."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import append_jsonl, load_frozen_manifest

REQUIRED_GIT_OBJECTS = (
    "scripts/h20/run_qwen25_7b_gatea_fresh2.sh",
    "scripts/h20/resume_qwen25_7b_gatea_step2_to3.sh",
    "tools/h20/preflight_qwen25_7b_gatea.py",
    "tools/h20/audit_qwen25_7b_gatea.py",
    "manifests/h20/qwen25_7b_gatea_seed2026.yaml",
    "manifests/h20/qwen25_7b_gatea_commands.json",
    "gate_a_execution_ledger.schema.json",
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
    return load_frozen_manifest(path)


def run_preflight(manifest_path: Path, check_runtime: bool, phase: str = "p0") -> dict:
    manifest_path = manifest_path.resolve()
    repo = manifest_path.parents[2]
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = load_manifest(manifest_path)
    failures: list[str] = []
    evidence: dict[str, object] = {}
    evidence["resolved_manifest"] = manifest
    evidence["resolved_manifest_sha256"] = canonical_sha256(manifest)

    expected_binding = {
        "required_environment": [
            "MEMAGENT_GATEA_WORK_ROOT",
            "MEMAGENT_GATEA_REPO_DIR",
            "MEMAGENT_GATEA_EXPECTED_COMMIT",
        ],
        "path_environment": ["MEMAGENT_GATEA_WORK_ROOT", "MEMAGENT_GATEA_REPO_DIR"],
        "automatic_repository_selection": False,
    }
    if raw_manifest.get("runtime_binding") != expected_binding:
        failures.append(f"runtime binding is not explicit/fail-closed: {raw_manifest.get('runtime_binding')}")

    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    status = git(repo, "status", "--porcelain")
    evidence.update(git_commit=head, branch=branch, worktree_clean=not status)
    expected_commit = os.environ.get("MEMAGENT_GATEA_EXPECTED_COMMIT", "")
    evidence["expected_git_commit"] = expected_commit or None
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        failures.append("MEMAGENT_GATEA_EXPECTED_COMMIT must be an explicit 40-character Git SHA")
    elif head != expected_commit:
        failures.append(f"exact Git commit mismatch: {head} != {expected_commit}")
    if branch != manifest["branch"]:
        failures.append(f"branch mismatch: {branch} != {manifest['branch']}")
    if status:
        failures.append(f"Git worktree is not clean: {status.splitlines()}")
    for commit_key in ("base_commit", "derived_from_commit"):
        expected_ancestor = manifest.get(commit_key)
        if not expected_ancestor or subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", expected_ancestor, head],
            check=False,
        ).returncode:
            failures.append(f"HEAD does not contain {commit_key} {expected_ancestor}")

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
        "declared_whitelist": [6, 7],
        "visible_devices": "6,7",
        "world_size": 2,
        "fsdp_size": 2,
        "trainer_gpus": 2,
        "tensor_parallel_size": 1,
    }:
        failures.append(f"GPU/FSDP configuration is not the frozen physical-6,7 two-rank shape: {gpu}")
    backend = manifest["backend"]
    if backend != {
        "rollout": "vllm",
        "evaluation": "vllm",
        "allow_hf_fallback": False,
        "reward_manager": "naive",
    }:
        failures.append(f"backend configuration drift: {backend}")
    weight_sync = manifest["weight_sync"]
    if weight_sync.get("comparison_semantics") != "actor_projected_to_vllm_parameter_dtype":
        failures.append(
            f"weight-sync comparison semantics drifted: {weight_sync.get('comparison_semantics')}"
        )
    if weight_sync.get("transfer_format") != "dtensor":
        failures.append(f"weight-sync transfer format drifted: {weight_sync.get('transfer_format')}")
    if not any(
        name.endswith((".self_attn.o_proj.weight", ".mlp.down_proj.weight"))
        for name in weight_sync.get("parameter_names", [])
    ):
        failures.append("weight-sync sample set does not include transformer matrix weights")
    if weight_sync.get("expected_loaded_parameter_count") != 199:
        failures.append("vLLM full loader coverage is not frozen to 199 parameters")

    expected_training = {
        "seed": 2026,
        "trajectory_seed_mode": "independent",
        "train_batch_size": 4,
        "rollout_n": 2,
        "ppo_mini_batch_size": 4,
        "chunk_size": 5000,
        "max_prompt_length": 8192,
        "max_response_length": 1024,
        "ppo_max_token_len_per_gpu": 16384,
        "log_prob_max_token_len_per_gpu": 32768,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.55,
        "actor_learning_rate": 0.000001,
        "kl_loss_coefficient": 0.001,
        "fresh_total_steps": 2,
        "resume_total_steps": 3,
    }
    if manifest.get("training") != expected_training:
        failures.append("formal Qwen2.5-7B Gate A training contract drifted")

    data_spec = dict(raw_manifest["data"])
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
    if phase in ("p0", "fresh"):
        for key in ("fresh_output", "resume_output"):
            if Path(paths[key]).exists():
                failures.append(f"frozen output already exists: {paths[key]}")
    elif phase == "resume":
        if not Path(paths["fresh_output"]).is_dir():
            failures.append(f"frozen fresh output is missing before resume: {paths['fresh_output']}")
        if Path(paths["resume_output"]).exists():
            failures.append(f"frozen resume output already exists: {paths['resume_output']}")

    model = manifest["model"]
    if model.get("id") != "Qwen/Qwen2.5-7B-Instruct" or model.get("revision") != (
        "a09a35458c702b33eeacc393d103063234e8bc28"
    ):
        failures.append("model ID/revision is not the frozen Qwen2.5-7B-Instruct object")
    evidence["model_id"] = model["id"]
    evidence["model_revision"] = model["revision"]
    model_root = Path(model["path"])
    model_config = model_root / "config.json"
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
    model_inventory = []
    config_entries = [item for item in model.get("files", []) if item.get("path") == "config.json"]
    if len(config_entries) != 1 or config_entries[0].get("sha256") != model.get("config_sha256"):
        failures.append("model config SHA is not internally consistent with the frozen file inventory")
    for item in model.get("files", []):
        path = model_root / item["path"]
        if not path.is_file():
            failures.append(f"missing frozen model file: {path}")
            continue
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        model_inventory.append({"path": item["path"], "size": actual_size, "sha256": actual_sha})
        if actual_size != int(item["size"]) or actual_sha != item["sha256"]:
            failures.append(
                f"model file content mismatch for {item['path']}: "
                f"size/sha=({actual_size}, {actual_sha}) != ({item['size']}, {item['sha256']})"
            )
    evidence["model_file_inventory"] = model_inventory
    expected_weight_files = sorted(
        item["path"] for item in model.get("files", []) if item["path"].endswith(".safetensors")
    )
    actual_weight_files = sorted(path.name for path in model_root.glob("*.safetensors"))
    if actual_weight_files != expected_weight_files:
        failures.append(
            f"model safetensors file set mismatch: {actual_weight_files} != {expected_weight_files}"
        )
    for name, path, expected_sha in (
        ("train", train, manifest["data"].get("train_sha256")),
        ("validation", validation, manifest["data"].get("validation_sha256")),
    ):
        if path.is_file():
            actual_sha = sha256_file(path)
            evidence[f"{name}_data_sha256"] = actual_sha
            if not expected_sha or actual_sha != expected_sha:
                failures.append(f"{name} data SHA mismatch: {actual_sha} != {expected_sha}")

    commands = json.loads((repo / "manifests/h20/qwen25_7b_gatea_commands.json").read_text())
    evidence["command_manifest_sha256"] = canonical_sha256(commands)
    if commands.get("required_environment") != [
        "MEMAGENT_GATEA_WORK_ROOT",
        "MEMAGENT_GATEA_REPO_DIR",
        "MEMAGENT_GATEA_EXPECTED_COMMIT",
    ]:
        failures.append(
            "command manifest does not require explicit task-scoped Gate A path bindings"
        )
    if manifest.get("ledger_schema") != "gate_a_execution_ledger.schema.json" or commands.get(
        "ledger_schema"
    ) != "gate_a_execution_ledger.schema.json":
        failures.append("ledger schema Git object name drifted")
    expected_contract = {
        "kind": "formal_gate_a",
        "physical_gpus": [6, 7],
        "world_size": 2,
        "execution_revision": "20260821r4",
    }
    if manifest.get("contract") != expected_contract or commands.get("contract") != expected_contract:
        failures.append("formal two-GPU command/manifest contract drifted")
    json.loads((repo / "gate_a_execution_ledger.schema.json").read_text())

    if check_runtime and Path(manifest["python"]).is_file():
        completed = subprocess.run(
            [
                manifest["python"],
                "-c",
                (
                    "import json,torch,transformers,verl,vllm; "
                    "print(json.dumps({'torch':torch.__version__,'vllm':vllm.__version__,"
                    "'transformers':transformers.__version__,'verl':getattr(verl,'__version__','source-tree')},"
                    "sort_keys=True))"
                ),
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        evidence["runtime_import_stdout"] = completed.stdout.strip()
        if completed.returncode:
            failures.append(f"torch/vLLM/verl import failed: {completed.stderr.strip()}")
        else:
            try:
                evidence["runtime_versions"] = json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as error:
                failures.append(f"cannot parse frozen runtime versions: {error}")
        try:
            import pyarrow.parquet as parquet

            table = parquet.read_table(train, columns=["extra_info"]).slice(0, 12)
            cursor_indices = []
            for value in table.column("extra_info").to_pylist():
                if isinstance(value, str):
                    value = json.loads(value)
                cursor_indices.append(int(value["index"]))
            evidence["train_cursor_prefix"] = cursor_indices
            if cursor_indices != list(range(12)):
                failures.append(
                    f"frozen HotpotQA cursor prefix is not semantic indices 0..11: {cursor_indices}"
                )
        except Exception as error:
            failures.append(f"cannot verify frozen HotpotQA dataset indices: {error}")
        gpu_identity = subprocess.run(
            [
                "nvidia-smi", "-i", manifest["gpu"]["visible_devices"],
                "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if gpu_identity.returncode:
            failures.append(f"cannot identify frozen physical GPUs: {gpu_identity.stderr.strip()}")
        else:
            gpu_rows = [line.strip() for line in gpu_identity.stdout.splitlines() if line.strip()]
            evidence["physical_gpu_identity"] = gpu_rows
            actual_indices = [int(line.split(",", 1)[0].strip()) for line in gpu_rows]
            if actual_indices != manifest["gpu"]["declared_whitelist"]:
                failures.append(
                    f"physical GPU identity mismatch: {actual_indices} "
                    f"!= {manifest['gpu']['declared_whitelist']}"
                )
        if phase != "p0":
            p0_path = Path(paths["certificate_root"]) / "p0_preflight.json"
            if not p0_path.is_file():
                failures.append("missing standalone P0 certificate during runtime recheck")
            else:
                frozen_evidence = json.loads(p0_path.read_text(encoding="utf-8")).get("evidence", {})
                for field in ("runtime_versions", "physical_gpu_identity"):
                    if evidence.get(field) != frozen_evidence.get(field):
                        failures.append(
                            f"runtime field {field} changed since P0: "
                            f"{evidence.get(field)} != {frozen_evidence.get(field)}"
                        )

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
    parser.add_argument("--phase", choices=("p0", "fresh", "resume"), default="p0")
    parser.add_argument("--write-certificate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_preflight(args.manifest, args.check_runtime, args.phase)
        manifest = load_manifest(args.manifest)
    except ValueError as error:
        result = {
            "gate": "P0",
            "status": "FAIL",
            "decision": "GATE_A_NO_GO:P0",
            "failures": [str(error)],
            "evidence": {},
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    if args.write_certificate:
        if result["status"] != "PASS":
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
        target = args.output or Path(manifest["paths"]["certificate_root"]) / "p0_preflight.json"
        resolved_target = target.with_name("p0_resolved_manifest.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = [str(path) for path in (target, resolved_target) if path.exists()]
        if existing:
            raise SystemExit(f"refusing to overwrite append-only P0 artifacts: {existing}")
        run_id = secrets.token_hex(16)
        result["evidence"]["run_id"] = run_id
        result["evidence"]["resolved_manifest_path"] = str(resolved_target)
        with resolved_target.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with target.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        ledger = Path(manifest["paths"]["execution_ledger"])
        ledger.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "record_type": "p0_preflight",
            "experiment_name": manifest["experiments"]["p0"],
            "git_commit": result["evidence"]["git_commit"],
            "run_id": run_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "status": result["status"],
            "certificate": str(target),
            "manifest_sha256": sha256_file(args.manifest),
            "evidence": result["evidence"],
            "failures": result["failures"],
        }
        append_jsonl(ledger, record)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
