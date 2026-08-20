#!/usr/bin/env python3
"""Fail-closed P0 for the corrected Original-style 2-GPU step3-to-T25 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import (
    append_jsonl,
    checkpoint_inventory,
    runtime_config_sha256,
    validate_jsonl_chain,
)


EXPECTED_BRANCH = "h20/qwen25-7b-original-t25-s128-frozen-20260821"
EXPECTED_BASE_COMMIT = "bd8b804c2cbf333f0f0650b729fd03a143d445b2"
ENVIRONMENT_NAMES = (
    "MEMAGENT_T25_WORK_ROOT",
    "MEMAGENT_T25_REPO_DIR",
    "MEMAGENT_T25_EXPECTED_COMMIT",
)
COMMAND_MANIFEST = "manifests/h20/qwen25_7b_original_t25_commands.json"
REQUIRED_GIT_OBJECTS = (
    "experiments/7b_gate_a/run_gate_a.sh",
    "scripts/h20/original_t25_common.sh",
    "scripts/h20/resume_qwen25_7b_original_step3_to25.sh",
    "tools/h20/preflight_qwen25_7b_original_t25.py",
    "tools/h20/audit_qwen25_7b_original_t25.py",
    "manifests/h20/qwen25_7b_original_t25_seed2026.json",
    COMMAND_MANIFEST,
    "gate_a_execution_ledger.schema.json",
    "recurrent/generation_manager.py",
    "recurrent/research/gate_a_execution.py",
    "recurrent/research/hotpotqa_dense_reward.py",
    "recurrent/research/trajectory_seeding.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/workers/fsdp_workers.py",
    "verl/workers/sharding_manager/fsdp_vllm.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ledger_prefix_sha256(lines: list[str], record_count: int) -> str:
    if record_count < 1 or len(lines) < record_count:
        raise ValueError(
            f"ledger prefix requires {record_count} records, found {len(lines)}"
        )
    payload = ("\n".join(lines[:record_count]) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def resolve_manifest(
    raw: Any, environment: Mapping[str, str] | None = None
) -> Any:
    source = os.environ if environment is None else environment
    missing = [name for name in ENVIRONMENT_NAMES if not source.get(name)]
    if missing:
        raise ValueError(f"missing explicit T25 runtime bindings: {missing}")
    for name in ENVIRONMENT_NAMES[:2]:
        if not Path(str(source[name])).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    expected_commit = str(source[ENVIRONMENT_NAMES[2]])
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("MEMAGENT_T25_EXPECTED_COMMIT must be a full lowercase Git SHA")

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: walk(child) for key, child in value.items()}
        if isinstance(value, list):
            return [walk(child) for child in value]
        if isinstance(value, str):
            result = value
            for name in ENVIRONMENT_NAMES:
                result = result.replace(f"${{{name}}}", str(source[name]))
            if "${" in result:
                raise ValueError(f"unresolved manifest placeholder: {result}")
            return result
        return value

    return walk(raw)


def load_manifest(
    path: Path, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = resolve_manifest(raw, environment)
    if not isinstance(result, dict):
        raise ValueError("T25 manifest must resolve to a JSON object")
    return result


def component_inventory(step_dir: Path, world_size: int) -> tuple[list[dict], list[str]]:
    inventory = checkpoint_inventory(step_dir) if step_dir.is_dir() else []
    names = {item["path"] for item in inventory if int(item["size"]) > 0}
    required = {"data.pt"}
    for prefix in ("model", "optim", "extra_state"):
        required.update(
            f"actor/{prefix}_world_size_{world_size}_rank_{rank}.pt"
            for rank in range(world_size)
        )
    return inventory, sorted(required - names)


def model_loading_relevant_paths(model_root: Path) -> list[str]:
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
        ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx"
    )
    result: list[str] = []
    if not model_root.is_dir():
        return result
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
            result.append(path.relative_to(model_root).as_posix())
    return sorted(result)


def collect_effective_cursor_prefix(
    rows, *, prompt_is_valid, expected_length: int
) -> tuple[list[int], list[int]]:
    semantic_indices: list[int] = []
    raw_positions: list[int] = []
    for raw_position, row in enumerate(rows):
        if not prompt_is_valid(row["prompt"]):
            continue
        extra_info = row["extra_info"]
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        semantic_indices.append(int(extra_info["index"]))
        raw_positions.append(raw_position)
        if len(semantic_indices) == expected_length:
            break
    return semantic_indices, raw_positions


def _runner_environment(
    manifest: Mapping[str, Any], *, reference_gate_a: bool
) -> dict[str, str]:
    training = manifest["training"]
    environment = dict(os.environ)
    if reference_gate_a:
        gate_manifest = json.loads(
            (REPO_ROOT / "manifests/h20/qwen25_7b_gatea_seed2026.yaml").read_text(
                encoding="utf-8"
            )
        )

        def gate_resolve(value: str) -> str:
            return value.replace(
                "${MEMAGENT_GATEA_WORK_ROOT}", str(manifest["work_root"])
            ).replace("${MEMAGENT_GATEA_REPO_DIR}", str(manifest["repository"]))

        experiment = gate_manifest["experiments"]["resume"]
        resume_from = gate_resolve(gate_manifest["paths"]["resume_source"])
        total_steps = int(gate_manifest["training"]["resume_total_steps"])
        source_step = 2
        save_freq = 1
        max_keep = 3
    else:
        experiment = str(manifest["experiment_name"])
        resume_from = str(manifest["source_gate_a"]["checkpoint"])
        total_steps = int(training["target_step"])
        source_step = int(training["source_step"])
        save_freq = int(training["save_freq"])
        max_keep = int(training["max_actor_ckpt_to_keep"])
    environment.update(
        {
            "WORK_ROOT": str(manifest["work_root"]),
            "CODE": str(manifest["repository"]),
            "PYTHON": str(manifest["python"]),
            "MODEL": str(manifest["model"]["path"]),
            "TRAIN": str(manifest["data"]["train"]),
            "VAL": str(manifest["data"]["validation"]),
            "PHASE": "resume",
            "EXP": experiment,
            "RESUME_FROM": resume_from,
            "RUN_SEED": str(training["seed"]),
            "TRAIN_BATCH_SIZE": str(training["train_batch_size"]),
            "ROLLOUT_N": str(training["rollout_n"]),
            "PPO_MINI_BATCH_SIZE": str(training["ppo_mini_batch_size"]),
            "N_GPUS": str(manifest["gpu"]["trainer_gpus"]),
            "FSDP_SIZE": str(manifest["gpu"]["fsdp_size"]),
            "REWARD_MANAGER": str(manifest["backend"]["reward_manager"]),
            "GPU_MEMORY_UTILIZATION": str(training["gpu_memory_utilization"]),
            "RESUME_TOTAL_STEPS": str(total_steps),
            "RESUME_SOURCE_STEP": str(source_step),
            "SAVE_FREQ": str(save_freq),
            "MAX_ACTOR_CKPT_TO_KEEP": str(max_keep),
            "CUDA_VISIBLE_DEVICES": str(manifest["gpu"]["visible_devices"]),
            "EMIT_TRAINER_OVERRIDES": "1",
        }
    )
    return environment


def emit_trainer_overrides(
    manifest: Mapping[str, Any], *, reference_gate_a: bool = False
) -> list[str]:
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "experiments/7b_gate_a/run_gate_a.sh")],
        cwd=REPO_ROOT,
        env=_runner_environment(manifest, reference_gate_a=reference_gate_a),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            "cannot emit trainer overrides from the production runner: "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("production runner emitted no trainer override JSON")
    try:
        overrides = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise ValueError(f"cannot parse production trainer overrides: {error}") from error
    if not isinstance(overrides, list) or not overrides or any(
        not isinstance(value, str) or "=" not in value for value in overrides
    ):
        raise ValueError("production trainer override payload is not a non-empty argv list")
    return overrides


def compose_resolved_trainer_config(overrides: list[str]) -> dict[str, Any]:
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except Exception as error:
        raise ValueError(f"cannot import Hydra/OmegaConf: {error}") from error
    config_dir = (REPO_ROOT / "verl/trainer/config").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise ValueError("Hydra ppo_trainer config did not resolve to an object")
    return resolved


def _replace_nested(config: dict[str, Any], dotted: str, value: object) -> None:
    target: dict[str, Any] = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"resolved config lacks allowed-difference path {dotted}")
        target = child
    if parts[-1] not in target:
        raise ValueError(f"resolved config lacks allowed-difference path {dotted}")
    target[parts[-1]] = value


def assert_only_t25_config_differences(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    left = json.loads(json.dumps(reference))
    right = json.loads(json.dumps(candidate))
    allowed = (
        "trainer.experiment_name",
        "trainer.default_local_dir",
        "trainer.total_training_steps",
        "trainer.save_freq",
        "trainer.max_actor_ckpt_to_keep",
        "trainer.resume_from_path",
    )
    for path in allowed:
        _replace_nested(left, path, "__T25_ALLOWED_DIFFERENCE__")
        _replace_nested(right, path, "__T25_ALLOWED_DIFFERENCE__")
    if left != right:
        raise ValueError(
            "resolved T25 Hydra config differs from Gate A outside total steps, output, "
            "resume source, and checkpoint-retention controls"
        )


def validate_resolved_t25_config(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> list[str]:
    """Check the scientific/runtime fields, independent of the reference diff."""

    def get(dotted: str) -> Any:
        value: Any = config
        for part in dotted.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return "__MISSING__"
            value = value[part]
        return value

    expected = {
        "recurrent.enable": "memory",
        "recurrent.memory.config.context_key": "context",
        "recurrent.memory.config.chunk_size": 5000,
        "recurrent.memory.config.max_chunks": 8,
        "recurrent.memory.config.max_prompt_length": 1024,
        "recurrent.memory.config.max_memorization_length": 1024,
        "recurrent.memory.config.max_final_response_length": 1024,
        "algorithm.adv_estimator": "grpo",
        "algorithm.grpo_use_adv": False,
        "algorithm.kl_ctrl.kl_coef": 0.001,
        "data.train_files": manifest["data"]["train"],
        "data.val_files": manifest["data"]["validation"],
        "data.shuffle": False,
        "data.filter_overlong_prompts": True,
        "data.train_batch_size": 4,
        "data.dataloader_num_workers": 0,
        "data.truncation": "center",
        "data.context_key": "context",
        "data.max_prompt_length": 8192,
        "data.max_response_length": 1024,
        "reward_model.reward_manager": "naive",
        "reward_model.enable": False,
        "custom_reward_function.path": str(
            Path(manifest["repository"])
            / "recurrent/research/hotpotqa_dense_reward.py"
        ),
        "custom_reward_function.name": "compute_score",
        "custom_reward_function.reward_kwargs.f1_weight": 0.95,
        "custom_reward_function.reward_kwargs.grounded_box_bonus": 0.05,
        "actor_rollout_ref.model.path": manifest["model"]["path"],
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.train_batch_size": 4,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 4,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 16384,
        "actor_rollout_ref.actor.clip_ratio_high": 0.2,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": 0.001,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.optim.lr": 0.000001,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 2,
        "actor_rollout_ref.actor.optim.warmup_style": "constant",
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": 2,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": 32768,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.rollout.load_format": "dummy_dtensor",
        "actor_rollout_ref.rollout.n": 2,
        "actor_rollout_ref.rollout.seed": 2026,
        "actor_rollout_ref.rollout.trajectory_seed_mode": "independent",
        "actor_rollout_ref.rollout.temperature": 1,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.55,
        "actor_rollout_ref.rollout.enforce_eager": False,
        "actor_rollout_ref.rollout.free_cache_engine": False,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 16384,
        "actor_rollout_ref.rollout.max_num_seqs": 16,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 32768,
        "actor_rollout_ref.rollout.val_kwargs.n": 2,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 0.7,
        "trainer.project_name": "memagent_7b_serialization_credit",
        "trainer.experiment_name": manifest["experiment_name"],
        "trainer.logger": ["console"],
        "trainer.val_before_train": False,
        "trainer.n_gpus_per_node": 2,
        "trainer.nnodes": 1,
        "trainer.save_freq": 5,
        "trainer.test_freq": -1,
        "trainer.total_epochs": 30,
        "trainer.total_training_steps": 25,
        "trainer.resume_mode": "resume_path",
        "trainer.resume_from_path": manifest["source_gate_a"]["checkpoint"],
        "trainer.max_actor_ckpt_to_keep": 5,
        "trainer.default_hdfs_dir": None,
        "trainer.default_local_dir": manifest["paths"]["output"],
        "ray_init.num_cpus": 64,
    }
    return [
        f"resolved config {path}={get(path)!r}, expected {value!r}"
        for path, value in expected.items()
        if get(path) != value
    ]


def validate_contract(manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if manifest.get("branch") != EXPECTED_BRANCH:
        failures.append("manifest branch is not the frozen T25 branch")
    if manifest.get("base_commit") != EXPECTED_BASE_COMMIT:
        failures.append("manifest base commit is not the stable-I closure commit")
    if manifest.get("study") != {
        "label": "corrected Original-style 2-GPU pilot",
        "not_original_paper_7b_reproduction": True,
        "paper_scale_difference": (
            "This pilot retains the passed Gate A learning configuration; it does not claim "
            "the original paper's 32-GPU, batch-128, rollout-n-16, thread-reward, warmup-20 setup."
        ),
        "primary_scientific_endpoint": 25,
        "secondary_learning_curve_anchor_steps": [5, 10, 15, 20],
    }:
        failures.append("scientific scope label drifted or overclaims paper reproduction")
    if manifest.get("contract") != {
        "kind": "corrected_original_style_2gpu_pilot_step3_to_t25",
        "physical_gpus": [6, 7],
        "world_size": 2,
        "execution_revision": "20260821-t25-v1",
    }:
        failures.append("T25 execution contract drifted")
    if manifest.get("gpu") != {
        "declared_whitelist": [6, 7],
        "visible_devices": "6,7",
        "world_size": 2,
        "fsdp_size": 2,
        "trainer_gpus": 2,
        "tensor_parallel_size": 1,
    }:
        failures.append("GPU contract is not physical GPU6-7 with world/FSDP size 2")
    if manifest.get("backend") != {
        "rollout": "vllm",
        "evaluation": "vllm",
        "allow_hf_fallback": False,
        "reward_manager": "naive",
    }:
        failures.append("backend drifted from naive strict-vLLM Gate A semantics")
    if manifest.get("storage") != {
        "retained_checkpoint_multiplier": 5,
        "safety_margin_bytes": 21474836480,
        "formula": (
            "available_bytes >= retained_checkpoint_multiplier * "
            "source_step3_inventory_bytes + safety_margin_bytes"
        ),
    }:
        failures.append("storage contract does not reserve all five complete anchors plus margin")

    training = manifest.get("training", {})
    expected_steps = list(range(4, 26))
    immutable_gate_a_values = {
        "seed": 2026,
        "trajectory_seed_mode": "independent",
        "source_step": 3,
        "first_update_step": 4,
        "target_step": 25,
        "update_steps": expected_steps,
        "update_count": 22,
        "train_batch_size": 4,
        "rollout_n": 2,
        "ppo_mini_batch_size": 4,
        "chunk_size": 5000,
        "max_chunks": 8,
        "max_prompt_length": 8192,
        "max_response_length": 1024,
        "ppo_max_token_len_per_gpu": 16384,
        "log_prob_max_token_len_per_gpu": 32768,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.55,
        "actor_learning_rate": 0.000001,
        "actor_lr_warmup_steps": 2,
        "clip_ratio_high": 0.2,
        "entropy_coefficient": 0.0,
        "kl_loss_coefficient": 0.001,
        "save_freq": 5,
        "max_actor_ckpt_to_keep": 5,
        "technical_checkpoint_steps": [5, 10, 15, 20, 25],
        "expected_retained_complete_actor_checkpoints": [5, 10, 15, 20, 25],
        "primary_scientific_endpoint": 25,
        "secondary_learning_curve_anchor_steps": [5, 10, 15, 20],
        "resume_mode": "resume_path",
    }
    if training != immutable_gate_a_values:
        failures.append("training contract is not exact Gate A learning config plus T25/save controls")
    data = manifest.get("data", {})
    if {
        key: data.get(key)
        for key in (
            "shuffle", "dataloader_num_workers", "filter_overlong_prompts",
            "production_effective_prompt_limit", "source_consumed_prompt_count",
            "continuation_source_order_start", "continuation_source_order_stop_exclusive",
            "continuation_prompt_count", "gate_a_cursor_prefix",
        )
    } != {
        "shuffle": False,
        "dataloader_num_workers": 0,
        "filter_overlong_prompts": True,
        "production_effective_prompt_limit": 40000,
        "source_consumed_prompt_count": 12,
        "continuation_source_order_start": 12,
        "continuation_source_order_stop_exclusive": 100,
        "continuation_prompt_count": 88,
        "gate_a_cursor_prefix": [2, 6, 7, 9, 10, 11, 12, 14, 16, 20, 21, 23],
    }:
        failures.append("step3-to-step25 data cursor contract drifted")
    source = manifest.get("source_gate_a", {})
    if (
        source.get("commit") != "c3f987be5513cad2a9e95622dd6773726a7bf12e"
        or source.get("global_step") != 3
        or source.get("final_report_sha256")
        != "5f8b67b496bd672cb6e89c9ec481c1de97adbf0a73c3459edd02aef79830dca4"
        or source.get("required_version_3_digest")
        != "e72701a91a57ee36032fa6979a26f5bf86f746dd28e8f0e5478dd907884a237a"
    ):
        failures.append("Gate A r5 source binding drifted")
    stable = manifest.get("stable_identity_prerequisite", {})
    if (
        stable.get("commit") != EXPECTED_BASE_COMMIT
        or stable.get("required_status") != "PASS"
        or stable.get("required_decision") != "I_RECURRENT_IDENTITY_CANARY_PASS"
        or stable.get("eval_manifest_hash")
        != "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a"
        or stable.get("execution_ledger_prefix_sha256")
        != "cace28d198f0dfa040c9c00973885ee77b24cb18c079162e77b4b84e6330b136"
        or stable.get("execution_ledger_prefix_record_count") != 5
        or stable.get("execution_ledger_total_record_count") != 6
    ):
        failures.append("stable-I prerequisite contract drifted")
    return failures


def _runtime_versions(python: str, repo: Path) -> tuple[dict[str, str] | None, str | None]:
    result = subprocess.run(
        [
            python,
            "-c",
            (
                "import json,torch,transformers,verl,vllm;"
                "print(json.dumps({'torch':torch.__version__,'vllm':vllm.__version__,"
                "'transformers':transformers.__version__,'verl':getattr(verl,'__version__','source-tree')},sort_keys=True))"
            ),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return None, result.stderr.strip()
    try:
        return json.loads(result.stdout.strip().splitlines()[-1]), None
    except (IndexError, json.JSONDecodeError) as error:
        return None, str(error)


def run_preflight(
    manifest_path: Path, *, check_runtime: bool, phase: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    repo = manifest_path.parents[2]
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = load_manifest(manifest_path)
    failures = validate_contract(manifest)
    evidence: dict[str, Any] = {
        "resolved_manifest": manifest,
        "resolved_manifest_sha256": canonical_sha256(manifest),
    }

    if raw_manifest.get("runtime_binding") != {
        "required_environment": list(ENVIRONMENT_NAMES),
        "automatic_repository_selection": False,
    }:
        failures.append("runtime binding is not the exact task-scoped T25 contract")
    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    status = git(repo, "status", "--porcelain")
    expected_commit = os.environ.get("MEMAGENT_T25_EXPECTED_COMMIT", "")
    evidence.update(
        git_commit=head,
        expected_git_commit=expected_commit,
        branch=branch,
        worktree_clean=not bool(status),
    )
    if head != expected_commit:
        failures.append(f"exact Git commit mismatch: {head} != {expected_commit}")
    if branch != EXPECTED_BRANCH:
        failures.append(f"branch mismatch: {branch} != {EXPECTED_BRANCH}")
    if status:
        failures.append(f"Git worktree is dirty: {status.splitlines()}")
    if subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", EXPECTED_BASE_COMMIT, head],
        check=False,
    ).returncode:
        failures.append(f"HEAD does not contain base commit {EXPECTED_BASE_COMMIT}")
    if repo.resolve() != Path(manifest["repository"]).resolve():
        failures.append("invoked checkout differs from MEMAGENT_T25_REPO_DIR")

    missing_git = [name for name in REQUIRED_GIT_OBJECTS if not (repo / name).is_file()]
    untracked_git = [
        name
        for name in REQUIRED_GIT_OBJECTS
        if subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
    ]
    if missing_git:
        failures.append(f"missing required Git objects: {missing_git}")
    if untracked_git:
        failures.append(f"required Git objects are not committed: {untracked_git}")
    evidence["execution_code_sha256"] = {
        name: sha256_file(repo / name)
        for name in REQUIRED_GIT_OBJECTS
        if (repo / name).is_file()
    }

    commands = json.loads((repo / COMMAND_MANIFEST).read_text(encoding="utf-8"))
    evidence["command_manifest_sha256"] = canonical_sha256(commands)
    if commands.get("required_environment") != list(ENVIRONMENT_NAMES):
        failures.append("command manifest does not require the exact task-scoped environment")
    if commands.get("required_sequence") != ["p0", "train_step4_to25", "readonly_audit"]:
        failures.append("command sequence drifted")
    if commands.get("gpu_execution_authorized_by_this_manifest") is not False:
        failures.append("Git manifest must not self-authorize GPU execution")

    paths = manifest["paths"]
    output = Path(paths["output"])
    log_root = Path(paths["log_root"])
    p0_path = Path(paths["p0_certificate"])
    if phase == "p0":
        if output.exists():
            failures.append(f"T25 output already exists: {output}")
        if log_root.exists():
            failures.append(f"T25 evidence root already exists: {log_root}")
    else:
        if output.exists():
            failures.append(f"T25 output exists before training: {output}")
        if not p0_path.is_file():
            failures.append("standalone T25 P0 certificate is missing")

    source_report_path = Path(manifest["source_gate_a"]["final_report"])
    source_step = Path(manifest["source_gate_a"]["checkpoint"])
    source_report: dict[str, Any] = {}
    if not source_report_path.is_file():
        failures.append(f"Gate A final report is missing: {source_report_path}")
    else:
        report_sha = sha256_file(source_report_path)
        evidence["source_gate_a_report_sha256"] = report_sha
        if report_sha != manifest["source_gate_a"]["final_report_sha256"]:
            failures.append("Gate A final report SHA-256 differs from the accepted r5 artifact")
        try:
            source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"cannot parse Gate A final report: {error}")
        if (
            source_report.get("status") != "PASS"
            or source_report.get("decision") != "GATE_A_PASS"
            or source_report.get("gates") != {"P0": "PASS", "P1": "PASS", "P2": "PASS"}
            or source_report.get("ledger_failures") != []
        ):
            failures.append("Gate A source report is not the complete P0/P1/P2 PASS")
        audits = source_report.get("audits") or {}
        if any((audits.get(name) or {}).get("status") != "PASS" for name in ("A1", "A2", "A3", "A4", "A5")):
            failures.append("Gate A source report does not contain a clean A1-A5 PASS")
        version3 = (audits.get("A4") or {}).get("version_digests", {}).get("3")
        if version3 != manifest["source_gate_a"]["required_version_3_digest"]:
            failures.append("Gate A version-3 effective actor digest drifted")

    source_inventory, source_missing = component_inventory(
        source_step, int(manifest["gpu"]["world_size"])
    )
    evidence["source_step3_inventory"] = source_inventory
    evidence["source_step3_inventory_sha256"] = canonical_sha256(source_inventory)
    evidence["source_step3_inventory_bytes"] = sum(int(item["size"]) for item in source_inventory)
    if source_missing:
        failures.append(f"Gate A step3 checkpoint is incomplete: {source_missing}")
    if source_report.get("step3_inventory") != source_inventory:
        failures.append("current Gate A step3 inventory differs from the accepted final report")

    stable_spec = manifest["stable_identity_prerequisite"]
    stable_p0_path = Path(stable_spec["p0_certificate"])
    stable_report_path = Path(stable_spec["final_report"])
    stable_resolved_path = Path(stable_spec["resolved_manifest"])
    stable_ledger_path = Path(stable_spec["execution_ledger"])
    stable_p0: dict[str, Any] = {}
    stable_report: dict[str, Any] = {}
    stable_resolved: dict[str, Any] = {}
    stable_records: list[dict[str, Any]] = []
    for label, path in (
        ("stable-I P0 certificate", stable_p0_path),
        ("stable-I final report", stable_report_path),
        ("stable-I resolved manifest", stable_resolved_path),
        ("stable-I execution ledger", stable_ledger_path),
    ):
        if not path.is_file():
            failures.append(f"{label} is missing: {path}")
    try:
        if stable_p0_path.is_file():
            stable_p0 = json.loads(stable_p0_path.read_text(encoding="utf-8"))
            evidence["stable_i_p0_certificate_sha256"] = sha256_file(stable_p0_path)
        if stable_report_path.is_file():
            stable_report = json.loads(stable_report_path.read_text(encoding="utf-8"))
            evidence["stable_i_final_report_sha256"] = sha256_file(stable_report_path)
        if stable_resolved_path.is_file():
            stable_resolved = json.loads(stable_resolved_path.read_text(encoding="utf-8"))
            evidence["stable_i_resolved_manifest_sha256"] = sha256_file(stable_resolved_path)
        if stable_ledger_path.is_file():
            stable_ledger_lines = [
                line
                for line in stable_ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            stable_records = [json.loads(line) for line in stable_ledger_lines]
            evidence["stable_i_execution_ledger_sha256"] = sha256_file(stable_ledger_path)
            stable_prefix_count = int(stable_spec["execution_ledger_prefix_record_count"])
            evidence["stable_i_execution_ledger_prefix_sha256"] = (
                ledger_prefix_sha256(stable_ledger_lines, stable_prefix_count)
            )
            evidence["stable_i_execution_ledger_record_count"] = len(stable_records)
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"cannot parse stable-I prerequisite artifacts: {error}")
    if (
        stable_p0.get("status") != "PASS"
        or stable_p0.get("decision") != "STABLE_I_P0_PASS"
        or stable_p0.get("evidence", {}).get("git_commit") != stable_spec["commit"]
        or stable_p0.get("evidence", {}).get("eval_manifest_hash")
        != stable_spec["eval_manifest_hash"]
        or stable_p0.get("evidence", {}).get("resolved_manifest_sha256")
        != evidence.get("stable_i_resolved_manifest_sha256")
    ):
        failures.append("stable-I P0 does not authenticate the accepted r2 resolved manifest")
    if (
        stable_report.get("status") != stable_spec["required_status"]
        or stable_report.get("decision") != stable_spec["required_decision"]
        or stable_report.get("failures") != []
        or stable_report.get("evidence", {}).get("git_commit") != stable_spec["commit"]
    ):
        failures.append(
            "stable-I prerequisite is not the frozen clean PASS at commit "
            f"{stable_spec['commit']}"
        )
    stable_eval_hash = stable_report.get("evidence", {}).get("eval_manifest_hash")
    evidence["stable_i_eval_manifest_hash"] = stable_eval_hash
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(stable_eval_hash or "")) is None
        or stable_eval_hash != stable_spec["eval_manifest_hash"]
        or stable_resolved.get("eval_manifest_hash") != stable_eval_hash
    ):
        failures.append("stable-I report/resolved-manifest evaluation hash disagrees")
    if evidence.get("stable_i_execution_ledger_prefix_sha256") != stable_spec[
        "execution_ledger_prefix_sha256"
    ]:
        failures.append("stable-I five-record ledger prefix is not the accepted r2 artifact")
    if (
        stable_report.get("evidence", {}).get("execution_ledger_sha256")
        != stable_spec["execution_ledger_prefix_sha256"]
        or stable_report.get("evidence", {}).get("execution_ledger_records")
        != stable_spec["execution_ledger_prefix_record_count"]
    ):
        failures.append("stable-I final report does not freeze the accepted five-record prefix")
    if len(stable_records) != int(stable_spec["execution_ledger_total_record_count"]):
        failures.append(
            f"stable-I execution ledger record count {len(stable_records)} is not 6"
        )
    stable_chain_failures = validate_jsonl_chain(stable_records)
    if stable_chain_failures:
        failures.append(f"stable-I execution ledger hash chain failed: {stable_chain_failures}")
    if not stable_records:
        failures.append("stable-I execution ledger is empty")
    else:
        stable_head = stable_records[0]
        stable_tail = stable_records[-1]
        if (
            stable_head.get("record_type") != "s0_preflight"
            or Path(str(stable_head.get("artifact", ""))).resolve()
            != stable_p0_path.resolve()
            or stable_head.get("artifact_sha256")
            != evidence.get("stable_i_p0_certificate_sha256")
        ):
            failures.append("stable-I ledger head does not authenticate its P0 certificate")
        if (
            stable_tail.get("record_type") != "audit_result"
            or stable_tail.get("status") != "PASS"
            or stable_tail.get("decision") != "I_RECURRENT_IDENTITY_CANARY_PASS"
            or stable_tail.get("git_commit") != stable_spec["commit"]
            or Path(str(stable_tail.get("artifact", ""))).resolve()
            != stable_report_path.resolve()
            or stable_tail.get("artifact_sha256")
            != evidence.get("stable_i_final_report_sha256")
            or stable_tail.get("eval_manifest_hash") != stable_eval_hash
        ):
            failures.append("stable-I ledger tail does not authenticate the required final PASS")

    model_root = Path(manifest["model"]["path"])
    model_inventory: list[dict[str, Any]] = []
    for expected in manifest["model"]["files"]:
        path = model_root / expected["path"]
        if not path.is_file():
            failures.append(f"missing model file: {path}")
            continue
        item = {
            "path": expected["path"],
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        model_inventory.append(item)
        if item != expected:
            failures.append(f"model file drifted: {expected['path']}")
    evidence["model_file_inventory"] = model_inventory
    actual_loading_paths = model_loading_relevant_paths(model_root)
    expected_loading_paths = sorted(item["path"] for item in manifest["model"]["files"])
    evidence["model_loading_relevant_paths"] = actual_loading_paths
    if actual_loading_paths != expected_loading_paths:
        failures.append(
            f"model loading-relevant file set drifted: {actual_loading_paths} != {expected_loading_paths}"
        )

    for label in ("train", "validation"):
        path = Path(manifest["data"][label])
        if not path.is_file():
            failures.append(f"missing {label} parquet: {path}")
            continue
        actual = sha256_file(path)
        evidence[f"{label}_data_sha256"] = actual
        if actual != manifest["data"][f"{label}_sha256"]:
            failures.append(f"{label} parquet SHA-256 drifted")
    python = Path(manifest["python"])
    if not python.is_file():
        failures.append(f"frozen Python is missing: {python}")

    checkpoint_bytes = int(evidence["source_step3_inventory_bytes"])
    storage = manifest["storage"]
    required_free = (
        int(storage["retained_checkpoint_multiplier"]) * checkpoint_bytes
        + int(storage["safety_margin_bytes"])
    )
    disk_probe = output.parent if output.parent.exists() else Path(manifest["work_root"])
    if disk_probe.exists():
        free_bytes = shutil.disk_usage(disk_probe).free
        evidence["available_bytes"] = free_bytes
        evidence["required_free_bytes"] = required_free
        if free_bytes < required_free:
            failures.append(
                "insufficient disk for five retained checkpoints plus margin: "
                f"{free_bytes} < {required_free}"
            )
    else:
        failures.append(f"cannot inspect target filesystem: {disk_probe}")

    if check_runtime and python.is_file():
        try:
            import torch

            source_training_state: list[dict[str, Any]] = []
            required_rng_keys = {"cpu", "cuda", "numpy", "random"}
            for rank in range(int(manifest["gpu"]["world_size"])):
                relative = (
                    f"actor/extra_state_world_size_{manifest['gpu']['world_size']}_"
                    f"rank_{rank}.pt"
                )
                extra_path = source_step / relative
                extra_state = torch.load(
                    extra_path, map_location="cpu", weights_only=False
                )
                if not isinstance(extra_state, Mapping):
                    raise ValueError(f"{relative} is not a mapping")
                scheduler = extra_state.get("lr_scheduler")
                rng = extra_state.get("rng")
                rng_keys = sorted(rng) if isinstance(rng, Mapping) else []
                scheduler_epoch = (
                    int(scheduler.get("last_epoch"))
                    if isinstance(scheduler, Mapping)
                    and scheduler.get("last_epoch") is not None
                    else None
                )
                source_training_state.append(
                    {
                        "rank": rank,
                        "path": relative,
                        "lr_scheduler_last_epoch": scheduler_epoch,
                        "rng_state_keys": rng_keys,
                    }
                )
                if scheduler_epoch != int(manifest["training"]["source_step"]):
                    failures.append(
                        f"{relative} scheduler last_epoch {scheduler_epoch} is not step3"
                    )
                if set(rng_keys) != required_rng_keys or any(
                    rng.get(key) is None for key in required_rng_keys
                ):
                    failures.append(
                        f"{relative} lacks the complete CPU/CUDA/NumPy/Python RNG state"
                    )
            evidence["source_step3_extra_state"] = source_training_state
        except Exception as error:
            failures.append(f"cannot inspect source step3 scheduler/RNG state: {error}")

        versions, version_error = _runtime_versions(str(python), repo)
        evidence["runtime_versions"] = versions
        if version_error:
            failures.append(f"torch/vLLM/verl runtime import failed: {version_error}")
        gpu_result = subprocess.run(
            [
                "nvidia-smi", "-i", manifest["gpu"]["visible_devices"],
                "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if gpu_result.returncode:
            failures.append(f"cannot identify physical GPU6-7: {gpu_result.stderr.strip()}")
        else:
            gpu_rows = [line.strip() for line in gpu_result.stdout.splitlines() if line.strip()]
            evidence["physical_gpu_identity"] = gpu_rows
            indices = [int(line.split(",", 1)[0].strip()) for line in gpu_rows]
            if indices != [6, 7]:
                failures.append(f"physical GPU indices drifted: {indices}")
            if len(gpu_rows) != 2 or any("H20" not in line.upper() for line in gpu_rows):
                failures.append(f"physical GPU6-7 are not both NVIDIA H20 devices: {gpu_rows}")
        app_result = subprocess.run(
            [
                "nvidia-smi", "-i", manifest["gpu"]["visible_devices"],
                "--query-compute-apps=pid", "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if app_result.returncode:
            failures.append(f"cannot inspect GPU processes: {app_result.stderr.strip()}")
        elif app_result.stdout.strip():
            failures.append(f"GPU6-7 are not idle: {app_result.stdout.splitlines()}")

        try:
            import pyarrow.parquet as parquet
            from verl.utils import hf_tokenizer

            tokenizer = hf_tokenizer(str(model_root), trust_remote_code=False)
            parquet_file = parquet.ParquetFile(manifest["data"]["train"])

            cursor: list[int] = []
            raw_positions: list[int] = []
            context_token_counts: list[int] = []
            raw_position = 0
            prompt_limit = int(manifest["data"]["production_effective_prompt_limit"])
            for batch in parquet_file.iter_batches(
                columns=["prompt", "extra_info", "context"]
            ):
                for row in batch.to_pylist():
                    prompt_valid = len(
                        tokenizer.apply_chat_template(
                            row["prompt"], add_generation_prompt=True
                        )
                    ) <= prompt_limit
                    if prompt_valid:
                        extra_info = row["extra_info"]
                        if isinstance(extra_info, str):
                            extra_info = json.loads(extra_info)
                        cursor.append(int(extra_info["index"]))
                        raw_positions.append(raw_position)
                        context_ids = tokenizer(
                            row["context"], add_special_tokens=False
                        )["input_ids"]
                        context_token_counts.append(min(len(context_ids), prompt_limit))
                        if len(cursor) == int(
                            manifest["data"]["continuation_source_order_stop_exclusive"]
                        ):
                            break
                    raw_position += 1
                if len(cursor) == int(
                    manifest["data"]["continuation_source_order_stop_exclusive"]
                ):
                    break
            evidence["train_cursor_semantic_indices_0_to_99"] = cursor
            evidence["train_cursor_raw_positions_0_to_99"] = raw_positions
            evidence["train_cursor_context_token_counts_0_to_99"] = context_token_counts
            chunk_size = int(manifest["training"]["chunk_size"])
            evidence["train_cursor_active_turn_counts_0_to_99"] = [
                (count + chunk_size - 1) // chunk_size
                for count in context_token_counts
            ]
            if len(cursor) != 100 or len(raw_positions) != 100:
                failures.append("production-effective train cursor did not yield 100 rows")
            if cursor[:12] != manifest["data"]["gate_a_cursor_prefix"]:
                failures.append("recomputed positions 0..11 differ from the Gate A r5 prefix")
            if raw_positions[:12] != list(range(12)):
                failures.append("Gate A r5 raw cursor positions 0..11 drifted")
        except Exception as error:
            failures.append(f"cannot reconstruct the production-effective train cursor: {error}")

        try:
            trainer_overrides = emit_trainer_overrides(manifest)
            reference_overrides = emit_trainer_overrides(
                manifest, reference_gate_a=True
            )
            resolved_trainer = compose_resolved_trainer_config(trainer_overrides)
            resolved_reference = compose_resolved_trainer_config(reference_overrides)
            assert_only_t25_config_differences(resolved_reference, resolved_trainer)
            resolved_contract_failures = validate_resolved_t25_config(
                manifest, resolved_trainer
            )
            if resolved_contract_failures:
                raise ValueError(
                    "resolved production trainer config violates the frozen T25 contract: "
                    + "; ".join(resolved_contract_failures)
                )
            evidence["trainer_override_argv"] = trainer_overrides
            evidence["trainer_override_argv_sha256"] = canonical_sha256(
                trainer_overrides
            )
            evidence["resolved_trainer_config_sha256"] = runtime_config_sha256(
                resolved_trainer
            )
            evidence["gate_a_reference_resolved_config_sha256"] = (
                runtime_config_sha256(resolved_reference)
            )
            evidence["hydra_config_name"] = "ppo_trainer"
            evidence["allowed_gate_a_to_t25_config_differences"] = [
                "trainer.experiment_name",
                "trainer.default_local_dir",
                "trainer.total_training_steps",
                "trainer.save_freq",
                "trainer.max_actor_ckpt_to_keep",
                "trainer.resume_from_path",
            ]
        except Exception as error:
            failures.append(f"cannot freeze/compare the production Hydra config: {error}")

    if phase == "run" and p0_path.is_file():
        try:
            p0 = json.loads(p0_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"cannot parse P0 certificate: {error}")
            p0 = {}
        p0_evidence = p0.get("evidence", {})
        if p0.get("status") != "PASS" or p0.get("decision") != "T25_P0_PASS":
            failures.append("standalone P0 is not T25_P0_PASS")
        for field in (
            "git_commit", "expected_git_commit", "resolved_manifest_sha256",
            "source_gate_a_report_sha256", "source_step3_inventory",
            "source_step3_inventory_sha256", "train_data_sha256",
            "validation_data_sha256", "model_file_inventory",
            "model_loading_relevant_paths", "runtime_versions", "physical_gpu_identity",
            "train_cursor_semantic_indices_0_to_99", "train_cursor_raw_positions_0_to_99",
            "train_cursor_context_token_counts_0_to_99",
            "train_cursor_active_turn_counts_0_to_99",
            "source_step3_extra_state",
            "stable_i_final_report_sha256", "stable_i_resolved_manifest_sha256",
            "stable_i_p0_certificate_sha256", "stable_i_execution_ledger_sha256",
            "stable_i_execution_ledger_prefix_sha256",
            "stable_i_execution_ledger_record_count", "stable_i_eval_manifest_hash",
            "trainer_override_argv", "trainer_override_argv_sha256",
            "resolved_trainer_config_sha256", "gate_a_reference_resolved_config_sha256",
            "allowed_gate_a_to_t25_config_differences",
        ):
            if evidence.get(field) != p0_evidence.get(field):
                failures.append(f"runtime field changed since P0: {field}")
        ledger = Path(paths["execution_ledger"])
        records = [
            json.loads(line)
            for line in ledger.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if ledger.is_file() else []
        run_id = p0_evidence.get("run_id")
        if (
            len(records) != 1
            or records[0].get("record_type") != "p0_preflight"
            or records[0].get("run_id") != run_id
            or records[0].get("git_commit") != head
            or Path(str(records[0].get("artifact", ""))).resolve() != p0_path.resolve()
            or records[0].get("artifact_sha256") != sha256_file(p0_path)
        ):
            failures.append("execution ledger is not the untouched standalone P0 prefix")

    return {
        "gate": "T25_P0",
        "status": "PASS" if not failures else "FAIL",
        "decision": "T25_P0_PASS" if not failures else "ORIGINAL_T25_NO_GO:P0",
        "failures": failures,
        "evidence": evidence,
    }, manifest


def write_p0(result: dict[str, Any], manifest: Mapping[str, Any]) -> None:
    if result["status"] != "PASS":
        raise ValueError("refusing to write a failed P0 certificate")
    certificate = Path(manifest["paths"]["p0_certificate"])
    certificate.parent.mkdir(parents=True, exist_ok=False)
    run_id = secrets.token_hex(16)
    result["evidence"]["run_id"] = run_id
    resolved_path = certificate.parent / "p0_resolved_manifest.json"
    with resolved_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result["evidence"]["resolved_manifest_path"] = str(resolved_path)
    with certificate.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    append_jsonl(
        manifest["paths"]["execution_ledger"],
        {
            "record_type": "p0_preflight",
            "experiment_name": manifest["experiment_name"],
            "git_commit": result["evidence"]["git_commit"],
            "run_id": run_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "decision": "T25_P0_PASS",
            "resolved_manifest_sha256": result["evidence"]["resolved_manifest_sha256"],
            "source_step3_inventory_sha256": result["evidence"]["source_step3_inventory_sha256"],
            "artifact": str(certificate),
            "artifact_sha256": sha256_file(certificate),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("p0", "run"), default="p0")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--write-certificate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.write_certificate and not args.check_runtime:
            raise ValueError("writing P0 requires --check-runtime")
        result, manifest = run_preflight(
            args.manifest, check_runtime=args.check_runtime, phase=args.phase
        )
        if args.write_certificate:
            if args.phase != "p0":
                raise ValueError("only standalone P0 may write a certificate")
            write_p0(result, manifest)
    except Exception as error:
        result = {
            "gate": "T25_P0",
            "status": "FAIL",
            "decision": "ORIGINAL_T25_NO_GO:P0",
            "failures": [str(error)],
            "evidence": {},
        }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
