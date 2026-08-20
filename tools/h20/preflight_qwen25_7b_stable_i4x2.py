#!/usr/bin/env python3
"""Fail-closed P0 for the frozen recurrent-I 4x2x2 identity canary.

P0 reads the complete, already-existing HotpotQA S128 parquet.  It replays the
same prompt-length predicate used by ``MemoryDataset`` and refuses the canary
unless all 128 rows survive.  The four canary rows are a frozen prefix of that
S128, never a newly sampled dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import append_jsonl, validate_jsonl_chain
from recurrent.research.stable_eval_identity import (
    canonical_sha256,
    sha256_text,
    stable_eval_runtime_config_sha256,
    validate_resolved_manifest,
)


ENVIRONMENT_NAMES = (
    "MEMAGENT_STABLE_I_WORK_ROOT",
    "MEMAGENT_STABLE_I_REPO_DIR",
    "MEMAGENT_STABLE_I_EXPECTED_COMMIT",
)
REQUIRED_GIT_OBJECTS = (
    "manifests/h20/qwen25_7b_stable_i4x2_seed2026.json",
    "manifests/h20/qwen25_7b_stable_i4x2_commands.json",
    "scripts/h20/stable_i4x2_common.sh",
    "scripts/h20/run_qwen25_7b_stable_i4x2.sh",
    "tools/h20/preflight_qwen25_7b_stable_i4x2.py",
    "tools/h20/audit_qwen25_7b_stable_i4x2.py",
    "tests/h20/test_stable_i4x2_frozen.py",
    "stable_identity_execution_ledger.schema.json",
)
EXECUTION_CODE_OBJECTS = (
    "recurrent/impls/memory.py",
    "recurrent/generation_manager.py",
    "recurrent/research/stable_eval_identity.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/utils/dataset/rl_dataset.py",
    "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
    "verl/workers/sharding_manager/fsdp_vllm.py",
)
EXPECTED_BRANCH = "h20/qwen25-7b-stable-eval-i4x2-frozen-20260821"
EXPECTED_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
EXPECTED_VALIDATION_SHA256 = "54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6"
COMMAND_MANIFEST_PATH = "manifests/h20/qwen25_7b_stable_i4x2_commands.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def resolve_manifest_environment(
    value: Any, environment: Mapping[str, str] | None = None
) -> Any:
    source = os.environ if environment is None else environment
    missing = [name for name in ENVIRONMENT_NAMES if not source.get(name)]
    if missing:
        raise ValueError(f"missing explicit stable-I runtime bindings: {missing}")
    if re.fullmatch(r"[0-9a-f]{40}", str(source[ENVIRONMENT_NAMES[2]])) is None:
        raise ValueError("MEMAGENT_STABLE_I_EXPECTED_COMMIT must be a full Git SHA")
    for name in ENVIRONMENT_NAMES[:2]:
        if not Path(str(source[name])).is_absolute():
            raise ValueError(f"{name} must be an absolute path")

    def resolve(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: resolve(child) for key, child in item.items()}
        if isinstance(item, list):
            return [resolve(child) for child in item]
        if isinstance(item, str):
            result = item
            for name in ENVIRONMENT_NAMES[:2]:
                result = result.replace(f"${{{name}}}", str(source[name]))
            if "${" in result:
                raise ValueError(f"unresolved manifest placeholder: {result}")
            return result
        return item

    return resolve(value)


def load_manifest(
    path: str | Path, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    resolved = resolve_manifest_environment(raw, environment)
    if not isinstance(resolved, dict):
        raise TypeError("stable-I manifest must be a JSON object")
    return resolved


def _mapping(value: object, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return dict(value)


def _question_text(prompt: object) -> str:
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty chat-message list")
    first = _mapping(prompt[0], "prompt[0]")
    if first.get("role") != "user" or not isinstance(first.get("content"), str):
        raise ValueError("prompt[0] must be a user message with string content")
    return str(first["content"])


def freeze_existing_s128_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    prompt_token_length: Callable[[object], int],
    context_token_length: Callable[[str], int],
    max_prompt_length: int,
    max_context_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the production filter and freeze every surviving S128 row."""
    raw_rows = list(rows)
    frozen: list[dict[str, Any]] = []
    rejected_raw_positions: list[int] = []
    for raw_position, source in enumerate(raw_rows):
        row = dict(source)
        prompt = row.get("prompt")
        token_length = int(prompt_token_length(prompt))
        if token_length > int(max_prompt_length):
            rejected_raw_positions.append(raw_position)
            continue
        extra_info = _mapping(row.get("extra_info"), "extra_info")
        reward_model = _mapping(row.get("reward_model"), "reward_model")
        semantic_index = int(extra_info["index"])
        question = _question_text(prompt)
        context = row.get("context")
        if not isinstance(context, str):
            raise TypeError(f"context at raw row {raw_position} must be text")
        effective_context_tokens = min(
            int(context_token_length(context)), int(max_context_length)
        )
        if effective_context_tokens < 1:
            raise ValueError(f"context at raw row {raw_position} tokenizes to zero tokens")
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is None:
            raise ValueError(f"ground_truth is missing at raw row {raw_position}")
        source_order = len(frozen)
        frozen.append(
            {
                "example_id": str(semantic_index),
                "semantic_dataset_index": semantic_index,
                "source_order_index": source_order,
                "raw_row_position": raw_position,
                "production_effective_position": source_order,
                "context_token_count": effective_context_tokens,
                "source_question_hash": sha256_text(question),
                "source_context_hash": sha256_text(context),
                "ground_truth_hash": canonical_sha256(ground_truth),
            }
        )
    evidence = {
        "raw_row_count": len(raw_rows),
        "production_effective_row_count": len(frozen),
        "rejected_raw_positions": rejected_raw_positions,
        "production_effective_prompt_limit": int(max_prompt_length),
        "production_effective_context_limit": int(max_context_length),
    }
    return frozen, evidence


def validate_s128_freeze(
    rows: list[Mapping[str, Any]], evidence: Mapping[str, Any], data_contract: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    expected_raw = int(data_contract["expected_raw_rows"])
    expected_effective = int(data_contract["expected_effective_rows"])
    if int(evidence["raw_row_count"]) != expected_raw:
        failures.append(
            f"existing HotpotQA S128 raw row count {evidence['raw_row_count']} != {expected_raw}"
        )
    if int(evidence["production_effective_row_count"]) != expected_effective:
        failures.append(
            "MemoryDataset prompt filter did not retain the complete fixed S128: "
            f"{evidence['production_effective_row_count']} != {expected_effective}; "
            f"rejected={evidence['rejected_raw_positions']}"
        )
    if expected_raw != 128 or expected_effective != 128:
        failures.append("frozen dataset contract itself is not exactly S128 128/128")
    example_ids = [str(row["example_id"]) for row in rows]
    if len(example_ids) != len(set(example_ids)):
        failures.append("fixed S128 contains duplicate semantic example IDs")
    expected_orders = list(range(len(rows)))
    if [int(row["source_order_index"]) for row in rows] != expected_orders:
        failures.append("production-effective source order is not contiguous")
    canary = list(data_contract.get("canary_source_order_indices", []))
    if canary != [0, 1, 2, 3]:
        failures.append(f"canary is not the preregistered first-four prefix: {canary}")
    if len(rows) >= 4 and [int(rows[index]["source_order_index"]) for index in canary] != canary:
        failures.append("canary positions do not resolve to the first four effective S128 rows")
    return failures


def build_identity_payload(
    manifest: Mapping[str, Any], *, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the interface-neutral stable join domain shared by Q/G/R/I/T.

    Interface, checkpoint/model-weight inventory, Git commit, and canary subset
    deliberately live outside this payload so an I row and its future T row
    have the same stable key.
    """
    model = manifest["model"]
    tokenizer_files = [
        item for item in model["files"]
        if item["path"] in {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
    ]
    payload = {
        "schema_version": 1,
        "namespace": "memagent-existing-s128-interface-neutral-stable-identity-v1",
        "source_dataset": {
            "role": "existing_project_fixed_s128",
            "parquet_sha256": manifest["data"]["validation_sha256"],
            "raw_rows": 128,
            "production_effective_rows": 128,
            "shuffle": False,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": manifest["data"][
                "filter_overlong_prompts_workers"
            ],
            "dataloader_num_workers": manifest["data"]["dataloader_num_workers"],
            "production_effective_prompt_limit": manifest["data"][
                "production_effective_prompt_limit"
            ],
        },
        "base_model_protocol": {
            "id": model["id"],
            "revision": model["revision"],
        },
        "tokenizer": {
            "files": tokenizer_files,
            "manifest_sha256": canonical_sha256(tokenizer_files),
        },
        "identity_construction": {
            "version": 1,
            "example_id": "string form of source extra_info.index",
            "source_order_index": "position after the frozen production prompt filter",
            "row_hashes": "UTF-8 SHA-256 for question/context and canonical-JSON SHA-256 for ground truth",
        },
        "decode": {
            key: manifest["evaluation"][key]
            for key in ("do_sample", "temperature", "top_p", "top_k")
        },
        "backend": dict(manifest["backend"]),
        "rows": rows,
    }
    return payload


def build_execution_binding(
    manifest: Mapping[str, Any], *, repo: Path, git_commit: str,
    rows: list[Mapping[str, Any]],
    trainer_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind this I canary's executable/model artifact outside the stable key."""
    model = manifest["model"]
    code_hashes = {path: sha256_file(repo / path) for path in EXECUTION_CODE_OBJECTS}
    canary_orders = [int(value) for value in manifest["data"]["canary_source_order_indices"]]
    rows_by_order = {int(row["source_order_index"]): row for row in rows}
    chunk_size = int(manifest["recurrent"]["chunk_size"])
    active_turns_by_order = {
        str(order): (
            int(rows_by_order[order]["context_token_count"]) + chunk_size - 1
        ) // chunk_size
        for order in canary_orders
    }
    return {
        "interface_id": str(manifest["evaluation"]["interface_id"]),
        "git_commit": git_commit,
        "base_seed": int(manifest["evaluation"]["base_seed"]),
        "replicas": int(manifest["evaluation"]["replicas"]),
        "model_artifact": {
            "id": model["id"],
            "revision": model["revision"],
            "file_manifest_sha256": canonical_sha256(model["files"]),
            "checkpoint_inventory_sha256": None,
            "kind": "frozen_huggingface_base_model",
        },
        "recurrent": dict(manifest["recurrent"]),
        "canary_turn_schedule": {
            "active_turn_count_by_source_order": active_turns_by_order,
            "shared_final_turn": max(active_turns_by_order.values()),
        },
        "execution_code_sha256": code_hashes,
        "execution_code_combined_sha256": canonical_sha256(code_hashes),
        "trainer_configuration": dict(trainer_configuration or {}),
    }


def render_trainer_overrides(
    manifest: Mapping[str, Any],
    *,
    repo: Path,
    attempt_id: str,
    eval_manifest_hash: str,
    expected_runtime_config_sha256: str,
) -> list[str]:
    """Render the single frozen argv source used by both P0 and the runner."""
    attempts = list(manifest["evaluation"]["attempts"])
    if attempt_id not in attempts:
        raise ValueError(f"attempt_id is not preregistered: {attempt_id}")
    for name, value in (
        ("eval_manifest_hash", eval_manifest_hash),
        ("expected_runtime_config_sha256", expected_runtime_config_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
    commands = json.loads((repo / COMMAND_MANIFEST_PATH).read_text(encoding="utf-8"))
    templates = commands.get("trainer_overrides")
    if not isinstance(templates, list) or not templates:
        raise ValueError("command manifest has no frozen trainer_overrides argv")
    attempt_root = Path(manifest["paths"][attempt_id])
    replacements = {
        "${VALIDATION_PATH}": str(manifest["data"]["validation"]),
        "${MODEL_PATH}": str(manifest["model"]["path"]),
        "${REPO_DIR}": str(repo),
        "${EXPERIMENT_NAME}": (
            f"qwen25_7b_h20_2gpu_stable_i4x2_{attempt_id}_seed2026_20260821"
        ),
        "${ATTEMPT_ID}": attempt_id,
        "${ATTEMPT_ROOT}": str(attempt_root),
        "${TERMINAL_DIR}": str(attempt_root / "terminal"),
        "${RESOLVED_MANIFEST_PATH}": str(manifest["paths"]["resolved_manifest"]),
        "${EVAL_MANIFEST_HASH}": str(eval_manifest_hash),
        "${TURN_LEDGER_PATH}": str(attempt_root / "trajectory_turns.jsonl"),
        "${EXECUTION_SUMMARY_PATH}": str(attempt_root / "execution_summary.json"),
        "${EXPECTED_RUNTIME_CONFIG_SHA256}": str(expected_runtime_config_sha256),
    }
    rendered: list[str] = []
    for item in templates:
        if not isinstance(item, str) or "\n" in item or "\r" in item:
            raise ValueError("trainer override argv must contain one plain string per item")
        value = item
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        if "${" in value:
            raise ValueError(f"unresolved trainer override placeholder: {value}")
        if "=" not in value:
            raise ValueError(f"trainer override is not a Hydra assignment: {value}")
        rendered.append(value)
    return rendered


def compose_resolved_trainer_config(repo: Path, overrides: list[str]) -> dict[str, Any]:
    """Perform the same no-GPU Hydra composition used by main_ppo."""
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except Exception as error:
        raise ValueError(f"cannot import Hydra/OmegaConf for trainer config preflight: {error}") from error
    config_dir = (repo / "verl/trainer/config").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise TypeError("Hydra ppo_trainer did not resolve to an object")
    return resolved


def freeze_trainer_configuration(
    manifest: Mapping[str, Any], *, repo: Path, eval_manifest_hash: str
) -> dict[str, Any]:
    """Compose both attempts and freeze their complete resolved job configs."""
    attempts: dict[str, Any] = {}
    for attempt_id in manifest["evaluation"]["attempts"]:
        placeholder_overrides = render_trainer_overrides(
            manifest,
            repo=repo,
            attempt_id=attempt_id,
            eval_manifest_hash=eval_manifest_hash,
            expected_runtime_config_sha256="0" * 64,
        )
        placeholder_config = compose_resolved_trainer_config(repo, placeholder_overrides)
        resolved_sha = stable_eval_runtime_config_sha256(placeholder_config)
        final_overrides = render_trainer_overrides(
            manifest,
            repo=repo,
            attempt_id=attempt_id,
            eval_manifest_hash=eval_manifest_hash,
            expected_runtime_config_sha256=resolved_sha,
        )
        final_config = compose_resolved_trainer_config(repo, final_overrides)
        final_sha = stable_eval_runtime_config_sha256(final_config)
        if final_sha != resolved_sha:
            raise ValueError(
                f"trainer config self-hash is not stable for {attempt_id}: "
                f"{final_sha} != {resolved_sha}"
            )
        attempts[str(attempt_id)] = {
            "resolved_config_sha256": final_sha,
            "override_argv_sha256": canonical_sha256(final_overrides),
            "override_count": len(final_overrides),
        }
    return {
        "hydra_config_name": "ppo_trainer",
        "hydra_config_dir": "verl/trainer/config",
        "attempts": attempts,
    }


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(
        path, columns=["prompt", "context", "reward_model", "extra_info"]
    )
    return table.to_pylist()


def _runtime_versions(python: str, repo: Path) -> tuple[dict[str, str] | None, str | None]:
    completed = subprocess.run(
        [
            python,
            "-c",
            (
                "import json,torch,transformers,verl,vllm;"
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
    if completed.returncode:
        return None, completed.stderr.strip()
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1]), None
    except (IndexError, json.JSONDecodeError) as error:
        return None, f"cannot parse runtime versions: {error}"


def model_loading_relevant_paths(model_root: Path) -> list[str]:
    """Inventory every file that Transformers/vLLM could select at load time."""
    exact_names = {
        "config.json",
        "generation_config.json",
        "adapter_config.json",
        "quantization_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    }
    weight_suffixes = (
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".gguf",
        ".onnx",
    )
    relevant: list[str] = []
    for path in model_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if (
            name in exact_names
            or name.endswith(weight_suffixes)
            or name.endswith(".index.json")
            or name.startswith(("adapter_", "tokenizer", "vocab", "merges", "chat_template"))
        ):
            relevant.append(path.relative_to(model_root).as_posix())
    return sorted(relevant)


def capture_runtime_binding(manifest: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Re-hash mutable external inputs immediately around each attempt."""
    model_root = Path(manifest["model"]["path"])
    model_inventory = []
    for expected in manifest["model"]["files"]:
        path = model_root / expected["path"]
        if not path.is_file():
            raise ValueError(f"runtime binding lost frozen model file: {path}")
        model_inventory.append(
            {
                "path": expected["path"],
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    validation_path = Path(manifest["data"]["validation"])
    if not validation_path.is_file():
        raise ValueError(f"runtime binding lost fixed S128 parquet: {validation_path}")
    versions, version_error = _runtime_versions(str(manifest["python"]), repo)
    if version_error or versions is None:
        raise ValueError(f"runtime binding import failed: {version_error}")
    gpu_identity = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            manifest["gpu"]["visible_devices"],
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if gpu_identity.returncode:
        raise ValueError(
            f"runtime binding cannot identify GPU6-7: {gpu_identity.stderr.strip()}"
        )
    return {
        "git_commit": git(repo, "rev-parse", "HEAD"),
        "branch": git(repo, "branch", "--show-current"),
        "worktree_clean": not bool(git(repo, "status", "--porcelain")),
        "validation_data_sha256": sha256_file(validation_path),
        "model_file_inventory": model_inventory,
        "model_loading_relevant_paths": model_loading_relevant_paths(model_root),
        "runtime_versions": versions,
        "physical_gpu_identity": [
            line.strip() for line in gpu_identity.stdout.splitlines() if line.strip()
        ],
    }


def run_preflight(manifest_path: Path, *, check_runtime: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    manifest_path = manifest_path.resolve()
    repo = manifest_path.parents[2]
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = load_manifest(manifest_path)
    failures: list[str] = []
    evidence: dict[str, Any] = {
        "frozen_manifest_sha256": sha256_file(manifest_path),
        "resolved_runtime_manifest_sha256": canonical_sha256(manifest),
    }

    expected_runtime_binding = {
        "required_environment": list(ENVIRONMENT_NAMES),
        "automatic_repository_selection": False,
    }
    if raw_manifest.get("runtime_binding") != expected_runtime_binding:
        failures.append("runtime bindings are not the exact task-scoped, fail-closed contract")

    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    status = git(repo, "status", "--porcelain")
    expected_commit = os.environ.get("MEMAGENT_STABLE_I_EXPECTED_COMMIT", "")
    evidence.update(
        git_commit=head,
        expected_git_commit=expected_commit or None,
        branch=branch,
        worktree_clean=not status,
    )
    if head != expected_commit:
        failures.append(f"exact Git commit mismatch: {head} != {expected_commit}")
    if branch != EXPECTED_BRANCH or branch != manifest.get("branch"):
        failures.append(f"branch mismatch: {branch} != {EXPECTED_BRANCH}")
    if status:
        failures.append(f"Git worktree is not clean: {status.splitlines()}")
    base_commit = str(manifest.get("base_commit", ""))
    if subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", base_commit, head],
        check=False,
    ).returncode:
        failures.append(f"HEAD does not contain frozen base commit {base_commit}")

    missing = [path for path in REQUIRED_GIT_OBJECTS if not (repo / path).is_file()]
    untracked = [
        path for path in REQUIRED_GIT_OBJECTS
        if subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
    ]
    if missing:
        failures.append(f"required Git objects are missing: {missing}")
    if untracked:
        failures.append(f"required Git objects are not committed: {untracked}")

    if manifest.get("scope") != raw_manifest.get("scope"):
        failures.append("scope contract changed during environment resolution")
    scope = manifest.get("scope", {})
    for field in (
        "not_a_paper_performance_evaluation",
        "not_a_five_interface_gate",
        "does_not_authorize_q_g_r_t_claims",
        "does_not_replace_existing_s128",
        "eval_manifest_hash_is_interface_neutral",
    ):
        if scope.get(field) is not True:
            failures.append(f"scope limitation is not frozen true: {field}")
    if scope.get("stable_join_target_interfaces") != ["Q", "G", "R", "I", "T"]:
        failures.append("interface-neutral stable join target set drifted")
    if manifest.get("contract") != {
        "kind": "recurrent_i_stable_identity_canary",
        "physical_gpus": [6, 7],
        "world_size": 2,
        "execution_revision": "20260821-i4x2x2",
    }:
        failures.append("stable-I contract shape drifted")
    if manifest.get("gpu") != {
        "declared_whitelist": [6, 7],
        "visible_devices": "6,7",
        "world_size": 2,
        "fsdp_size": 2,
        "trainer_gpus": 2,
        "tensor_parallel_size": 1,
    }:
        failures.append("GPU/FSDP contract is not physical GPU6-7 with two ranks")
    if manifest.get("backend") != {
        "rollout": "vllm",
        "evaluation": "vllm",
        "rollout_mode": "sync",
        "allow_hf_fallback": False,
        "reward_manager": "naive",
    }:
        failures.append("backend is not synchronous strict vLLM recurrent-I")
    if manifest.get("evaluation") != {
        "base_seed": 2026,
        "interface_id": "I",
        "examples": 4,
        "replicas": 2,
        "attempts": ["repeat_a", "repeat_b"],
        "attempt_count": 2,
        "validation_only": True,
        "actor_update_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_save_calls": 0,
        "resume_mode": "disable",
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    }:
        failures.append("4x2x2 validation-only execution contract drifted")
    data_contract = manifest["data"]
    if data_contract.get("dataset_role") != "existing_project_fixed_s128":
        failures.append("validation data is not declared as the existing fixed S128")
    if data_contract.get("include_source_order_index") is not True:
        failures.append("source_order_index transport is not explicitly enabled")
    if {
        key: data_contract.get(key)
        for key in (
            "validation_sha256",
            "dataset_role",
            "expected_raw_rows",
            "expected_effective_rows",
            "shuffle",
            "filter_overlong_prompts",
            "filter_overlong_prompts_workers",
            "dataloader_num_workers",
            "include_source_order_index",
            "production_effective_prompt_limit",
            "canary_source_order_indices",
        )
    } != {
        "validation_sha256": EXPECTED_VALIDATION_SHA256,
        "dataset_role": "existing_project_fixed_s128",
        "expected_raw_rows": 128,
        "expected_effective_rows": 128,
        "shuffle": False,
        "filter_overlong_prompts": True,
        "filter_overlong_prompts_workers": 1,
        "dataloader_num_workers": 0,
        "include_source_order_index": True,
        "production_effective_prompt_limit": 40000,
        "canary_source_order_indices": [0, 1, 2, 3],
    }:
        failures.append("fixed S128/filter/canary contract drifted")
    if manifest.get("recurrent") != {
        "enable": "memory",
        "context_key": "context",
        "chunk_size": 5000,
        "max_chunks": 8,
        "max_prompt_length": 1024,
        "max_memorization_length": 1024,
        "max_final_response_length": 1024,
        "data_truncation": "center",
    }:
        failures.append("recurrent-I memory/chunk/budget contract drifted")
    if int(manifest["recurrent"]["chunk_size"]) * int(
        manifest["recurrent"]["max_chunks"]
    ) != int(data_contract["production_effective_prompt_limit"]):
        failures.append("MemoryDataset effective prompt-filter limit is internally inconsistent")
    expected_digest_parameters = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.27.input_layernorm.weight",
        "model.layers.27.self_attn.o_proj.weight",
        "model.layers.27.mlp.down_proj.weight",
        "model.norm.weight",
    ]
    weight_snapshot = manifest.get("weight_snapshot", {})
    if (
        weight_snapshot.get("samples_per_tensor") != 256
        or weight_snapshot.get("expected_loaded_parameter_count") != 199
        or weight_snapshot.get("expected_sampled_parameter_dtype") != "torch.bfloat16"
        or weight_snapshot.get("transfer_format") != "dtensor"
        or weight_snapshot.get("parameter_names") != expected_digest_parameters
        or weight_snapshot.get("required_worker_ranks") != [0, 1]
    ):
        failures.append("read-only actor/vLLM sampled-weight snapshot contract drifted")

    commands = json.loads(
        (repo / "manifests/h20/qwen25_7b_stable_i4x2_commands.json").read_text(
            encoding="utf-8"
        )
    )
    evidence["command_manifest_sha256"] = canonical_sha256(commands)
    if commands.get("required_environment") != list(ENVIRONMENT_NAMES):
        failures.append("command manifest does not use task-scoped stable-I bindings")
    if commands.get("contract") != manifest.get("contract"):
        failures.append("command/experiment manifest contract mismatch")
    if commands.get("required_sequence") != ["p0", "repeat_a", "repeat_b", "audit"]:
        failures.append("command manifest preregistered sequence drifted")
    if commands.get("gpu_execution_authorized_by_this_manifest") is not False:
        failures.append("command manifest improperly self-authorizes GPU execution")
    if commands.get("scientific_claim_authorized_by_this_manifest") is not False:
        failures.append("command manifest improperly self-authorizes a scientific claim")
    if manifest.get("ledger_schema") != "stable_identity_execution_ledger.schema.json":
        failures.append("stable identity ledger schema name drifted")
    json.loads((repo / manifest["ledger_schema"]).read_text(encoding="utf-8"))

    runtime_paths = {
        "python": Path(manifest["python"]),
        "repository": Path(manifest["repository"]),
        "model": Path(manifest["model"]["path"]),
        "validation": Path(data_contract["validation"]),
    }
    for name, path in runtime_paths.items():
        if not path.exists():
            failures.append(f"missing frozen runtime path {name}: {path}")
    evidence["invoked_python"] = str(Path(sys.executable).resolve())
    if runtime_paths["python"].exists() and Path(sys.executable).resolve() != runtime_paths[
        "python"
    ].resolve():
        failures.append(
            f"P0 was invoked with the wrong Python: {Path(sys.executable).resolve()} "
            f"!= {runtime_paths['python'].resolve()}"
        )
    if repo.resolve() != runtime_paths["repository"].resolve():
        failures.append(
            f"checkout path mismatch: {repo.resolve()} != {runtime_paths['repository'].resolve()}"
        )
    evidence_paths = [
        Path(manifest["paths"][key])
        for key in (
            "p0_certificate",
            "resolved_manifest",
            "final_report",
            "execution_ledger",
            "repeat_a",
            "repeat_b",
        )
    ]
    preexisting_evidence = [str(path) for path in evidence_paths if path.exists()]
    if preexisting_evidence:
        failures.append(
            f"append-only stable-I evidence path already exists before P0: {preexisting_evidence}"
        )

    model = manifest["model"]
    if model.get("id") != EXPECTED_MODEL_ID or model.get("revision") != EXPECTED_MODEL_REVISION:
        failures.append("model is not the frozen Qwen2.5-7B-Instruct revision")
    if model.get("loading_relevant_inventory_policy") != (
        "exact recursive inventory; undeclared model/tokenizer/index/adapter loading files are forbidden"
    ):
        failures.append("model loading-relevant exact-inventory policy drifted")
    model_root = runtime_paths["model"]
    model_inventory: list[dict[str, Any]] = []
    for item in model.get("files", []):
        path = model_root / item["path"]
        if not path.is_file():
            failures.append(f"missing frozen model file: {path}")
            continue
        actual = {"path": item["path"], "size": path.stat().st_size, "sha256": sha256_file(path)}
        model_inventory.append(actual)
        if actual != item:
            failures.append(f"frozen model file mismatch: {actual} != {item}")
    evidence["model_file_inventory"] = model_inventory
    expected_model_paths = sorted(item["path"] for item in model.get("files", []))
    actual_loading_paths = model_loading_relevant_paths(model_root) if model_root.is_dir() else []
    evidence["model_loading_relevant_paths"] = actual_loading_paths
    if actual_loading_paths != expected_model_paths:
        failures.append(
            "model directory loading-relevant inventory is not exact: "
            f"actual={actual_loading_paths}, expected={expected_model_paths}"
        )
    if model.get("config_sha256") != next(
        (item["sha256"] for item in model.get("files", []) if item["path"] == "config.json"),
        None,
    ):
        failures.append("model config hash is inconsistent with file inventory")

    validation_path = runtime_paths["validation"]
    if validation_path.is_file():
        validation_sha = sha256_file(validation_path)
        evidence["validation_data_sha256"] = validation_sha
        if validation_sha != EXPECTED_VALIDATION_SHA256 or validation_sha != data_contract.get(
            "validation_sha256"
        ):
            failures.append("existing fixed S128 parquet SHA-256 changed")

    resolved_identity_manifest: dict[str, Any] | None = None
    if runtime_paths["python"].is_file() and model_root.is_dir() and validation_path.is_file():
        try:
            from verl.utils import hf_tokenizer

            tokenizer = hf_tokenizer(str(model_root), trust_remote_code=False)
            parquet_rows = _load_parquet_rows(validation_path)
            frozen_rows, filter_evidence = freeze_existing_s128_rows(
                parquet_rows,
                prompt_token_length=lambda prompt: len(
                    tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
                ),
                context_token_length=lambda context: len(
                    tokenizer.encode(context, add_special_tokens=False)
                ),
                max_prompt_length=int(data_contract["production_effective_prompt_limit"]),
                max_context_length=int(data_contract["production_effective_prompt_limit"]),
            )
            evidence["s128_filter_replay"] = filter_evidence
            evidence["s128_rows"] = frozen_rows
            failures.extend(validate_s128_freeze(frozen_rows, filter_evidence, data_contract))
            payload = build_identity_payload(
                manifest, rows=frozen_rows
            )
            eval_manifest_hash = canonical_sha256(payload)
            trainer_configuration = freeze_trainer_configuration(
                manifest, repo=repo, eval_manifest_hash=eval_manifest_hash
            )
            resolved_identity_manifest = {
                "schema_version": 1,
                "frozen_manifest_sha256": evidence["frozen_manifest_sha256"],
                "identity_payload": payload,
                "eval_manifest_hash": eval_manifest_hash,
                "canary": {
                    "source_order_indices": list(
                        data_contract["canary_source_order_indices"]
                    ),
                    "examples": int(manifest["evaluation"]["examples"]),
                    "replicas": int(manifest["evaluation"]["replicas"]),
                    "attempts": list(manifest["evaluation"]["attempts"]),
                },
                "execution_binding": build_execution_binding(
                    manifest,
                    repo=repo,
                    git_commit=head,
                    rows=frozen_rows,
                    trainer_configuration=trainer_configuration,
                ),
            }
            validate_resolved_manifest(resolved_identity_manifest)
            evidence["eval_manifest_hash"] = resolved_identity_manifest["eval_manifest_hash"]
            evidence["execution_binding_sha256"] = canonical_sha256(
                resolved_identity_manifest["execution_binding"]
            )
            evidence["canary_rows"] = [
                frozen_rows[index]
                for index in data_contract["canary_source_order_indices"]
            ]
        except Exception as error:
            failures.append(f"cannot replay and freeze the complete existing S128: {error}")

    if check_runtime and runtime_paths["python"].is_file():
        versions, version_error = _runtime_versions(manifest["python"], repo)
        if version_error:
            failures.append(f"torch/vLLM/verl runtime import failed: {version_error}")
        else:
            evidence["runtime_versions"] = versions
        gpu_identity = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                manifest["gpu"]["visible_devices"],
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if gpu_identity.returncode:
            failures.append(f"cannot identify physical GPU6-7: {gpu_identity.stderr.strip()}")
        else:
            rows = [line.strip() for line in gpu_identity.stdout.splitlines() if line.strip()]
            evidence["physical_gpu_identity"] = rows
            indices = [int(row.split(",", 1)[0].strip()) for row in rows]
            if indices != [6, 7]:
                failures.append(f"physical GPU indices {indices} != [6, 7]")
            if len(rows) != 2 or any("H20" not in row for row in rows):
                failures.append(f"physical GPU6-7 are not both NVIDIA H20 devices: {rows}")

    runtime_binding_fields = (
        "git_commit",
        "branch",
        "worktree_clean",
        "validation_data_sha256",
        "model_file_inventory",
        "model_loading_relevant_paths",
        "runtime_versions",
        "physical_gpu_identity",
    )
    if check_runtime and all(field in evidence for field in runtime_binding_fields):
        runtime_binding = {
            field: evidence[field] for field in runtime_binding_fields
        }
        evidence["runtime_binding"] = runtime_binding
        evidence["runtime_binding_sha256"] = canonical_sha256(runtime_binding)

    result = {
        "gate": "P0",
        "status": "PASS" if not failures else "FAIL",
        "decision": "STABLE_I_P0_PASS" if not failures else "STABLE_I_NO_GO:P0",
        "scope": manifest["scope"],
        "failures": failures,
        "evidence": evidence,
    }
    return result, resolved_identity_manifest


def _certificate_is_current(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    p0_path = Path(manifest["paths"]["p0_certificate"])
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    if not p0_path.is_file() or not resolved_path.is_file():
        raise ValueError("standalone P0 certificate and resolved manifest are required")
    certificate = json.loads(p0_path.read_text(encoding="utf-8"))
    resolved = validate_resolved_manifest(json.loads(resolved_path.read_text(encoding="utf-8")))
    evidence = certificate.get("evidence", {})
    expected_commit = os.environ["MEMAGENT_STABLE_I_EXPECTED_COMMIT"]
    if certificate.get("status") != "PASS":
        raise ValueError("P0 certificate is not PASS")
    if evidence.get("git_commit") != expected_commit or git(
        Path(manifest["repository"]), "rev-parse", "HEAD"
    ) != expected_commit:
        raise ValueError("P0/HEAD/expected commit mismatch")
    if evidence.get("resolved_manifest_sha256") != sha256_file(resolved_path):
        raise ValueError("resolved manifest file changed after P0")
    if evidence.get("eval_manifest_hash") != resolved["eval_manifest_hash"]:
        raise ValueError("resolved evaluation manifest hash changed after P0")
    return certificate, resolved


def record_attempt_event(manifest_path: Path, *, attempt_id: str, event: str) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    attempts = list(manifest["evaluation"]["attempts"])
    if attempt_id not in attempts:
        raise ValueError(f"attempt_id is not preregistered: {attempt_id}")
    certificate, resolved = _certificate_is_current(manifest)
    runtime_binding = capture_runtime_binding(
        manifest, Path(manifest["repository"])
    )
    expected_runtime_binding = certificate.get("evidence", {}).get("runtime_binding")
    if runtime_binding != expected_runtime_binding:
        raise ValueError(
            "mutable runtime binding changed after standalone P0: "
            f"expected={expected_runtime_binding}, actual={runtime_binding}"
        )
    runtime_binding_sha256 = canonical_sha256(runtime_binding)
    attempt_root = Path(manifest["paths"][attempt_id])
    terminal = attempt_root / "terminal/0.jsonl"
    turns = attempt_root / "trajectory_turns.jsonl"
    summary = attempt_root / "execution_summary.json"
    run_log = attempt_root / "run.log"
    if event == "start":
        existing = [str(path) for path in (terminal, turns, summary) if path.exists()]
        if existing:
            raise ValueError(f"attempt start found pre-existing append-only evidence: {existing}")
        artifacts: dict[str, Any] = {}
        status = "PASS"
    else:
        missing = [str(path) for path in (terminal, turns, summary, run_log) if not path.is_file()]
        if missing:
            raise ValueError(f"attempt finish is missing required evidence: {missing}")
        artifacts = {
            str(path.relative_to(attempt_root)): {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for path in (terminal, turns, summary, run_log)
        }
        status = "PASS"

    ledger = Path(manifest["paths"]["execution_ledger"])
    existing_records = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    chain_failures = validate_jsonl_chain(existing_records)
    if chain_failures:
        raise ValueError(f"execution ledger hash chain is invalid: {chain_failures}")
    expected_prefix = [("s0_preflight", None)]
    for registered in attempts:
        if registered == attempt_id:
            expected_prefix.append(("run_start", registered))
            if event == "finish":
                expected_prefix.append(("run_finish", registered))
            break
        expected_prefix.extend((("run_start", registered), ("run_finish", registered)))
    candidate = [
        *[(record.get("record_type"), record.get("attempt_id")) for record in existing_records],
        ("run_start" if event == "start" else "run_finish", attempt_id),
    ]
    if candidate != expected_prefix:
        raise ValueError(
            f"attempt event order is not the preregistered sequence: {candidate} != {expected_prefix}"
        )

    record = {
        "record_type": "run_start" if event == "start" else "run_finish",
        "experiment_name": f"qwen25_7b_h20_2gpu_stable_i4x2_{attempt_id}_seed2026_20260821",
        "git_commit": certificate["evidence"]["git_commit"],
        "run_id": certificate["evidence"]["run_id"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": canonical_sha256(resolved["execution_binding"]),
        "runtime_binding_sha256": runtime_binding_sha256,
        "attempt_id": attempt_id,
        "status": status,
        "artifacts": artifacts,
    }
    append_jsonl(ledger, record)
    return record


def frozen_trainer_overrides(manifest_path: Path, *, attempt_id: str) -> list[str]:
    """Read the P0-frozen attempt hash and render its exact launch argv."""
    manifest = load_manifest(manifest_path)
    _, resolved = _certificate_is_current(manifest)
    attempt_binding = (
        resolved.get("execution_binding", {})
        .get("trainer_configuration", {})
        .get("attempts", {})
        .get(attempt_id)
    )
    if not isinstance(attempt_binding, Mapping):
        raise ValueError(f"P0 lacks frozen trainer configuration for {attempt_id}")
    expected_sha = str(attempt_binding.get("resolved_config_sha256", ""))
    overrides = render_trainer_overrides(
        manifest,
        repo=Path(manifest["repository"]),
        attempt_id=attempt_id,
        eval_manifest_hash=resolved["eval_manifest_hash"],
        expected_runtime_config_sha256=expected_sha,
    )
    actual_argv_sha = canonical_sha256(overrides)
    if actual_argv_sha != attempt_binding.get("override_argv_sha256"):
        raise ValueError(
            f"trainer override argv changed after P0 for {attempt_id}: "
            f"{actual_argv_sha} != {attempt_binding.get('override_argv_sha256')}"
        )
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--write-certificate", action="store_true")
    parser.add_argument("--record-attempt-event", choices=("start", "finish"))
    parser.add_argument("--emit-trainer-overrides", action="store_true")
    parser.add_argument("--attempt-id", choices=("repeat_a", "repeat_b"))
    args = parser.parse_args()

    if args.emit_trainer_overrides:
        if not args.attempt_id:
            parser.error("--attempt-id is required with --emit-trainer-overrides")
        try:
            overrides = frozen_trainer_overrides(
                args.manifest, attempt_id=args.attempt_id
            )
        except Exception as error:
            print(f"STABLE_I_NO_GO:CONFIG {error}", file=sys.stderr)
            return 1
        print("\n".join(overrides))
        return 0

    if args.record_attempt_event:
        if not args.attempt_id:
            parser.error("--attempt-id is required with --record-attempt-event")
        try:
            record = record_attempt_event(
                args.manifest, attempt_id=args.attempt_id, event=args.record_attempt_event
            )
        except Exception as error:
            print(json.dumps({"status": "FAIL", "decision": "STABLE_I_NO_GO:EVENT", "failures": [str(error)]}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    if args.write_certificate and not args.check_runtime:
        parser.error("--write-certificate requires --check-runtime")

    try:
        result, resolved = run_preflight(args.manifest, check_runtime=args.check_runtime)
        manifest = load_manifest(args.manifest)
    except Exception as error:
        result = {
            "gate": "P0",
            "status": "FAIL",
            "decision": "STABLE_I_NO_GO:P0",
            "failures": [str(error)],
            "evidence": {},
        }
        resolved = None
        manifest = None

    if args.write_certificate:
        if result["status"] != "PASS" or resolved is None or manifest is None:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
        p0_path = Path(manifest["paths"]["p0_certificate"])
        resolved_path = Path(manifest["paths"]["resolved_manifest"])
        ledger = Path(manifest["paths"]["execution_ledger"])
        existing = [str(path) for path in (p0_path, resolved_path, ledger) if path.exists()]
        if existing:
            raise SystemExit(f"refusing to overwrite append-only P0 evidence: {existing}")
        p0_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_payload = json.dumps(resolved, indent=2, sort_keys=True) + "\n"
        with resolved_path.open("x", encoding="utf-8") as stream:
            stream.write(resolved_payload)
        result["evidence"]["run_id"] = secrets.token_hex(16)
        result["evidence"]["resolved_manifest_path"] = str(resolved_path)
        result["evidence"]["resolved_manifest_sha256"] = sha256_file(resolved_path)
        with p0_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        append_jsonl(
            ledger,
            {
                "record_type": "s0_preflight",
                "experiment_name": "qwen25_7b_h20_2gpu_stable_i4x2_p0_seed2026_20260821",
                "git_commit": result["evidence"]["git_commit"],
                "run_id": result["evidence"]["run_id"],
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "eval_manifest_hash": resolved["eval_manifest_hash"],
                "execution_binding_sha256": canonical_sha256(
                    resolved["execution_binding"]
                ),
                "runtime_binding_sha256": result["evidence"][
                    "runtime_binding_sha256"
                ],
                "attempt_id": None,
                "status": "PASS",
                "artifact": str(p0_path),
                "artifact_sha256": sha256_file(p0_path),
                "row_count": 128,
            },
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
