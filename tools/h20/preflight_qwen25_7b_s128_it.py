#!/usr/bin/env python3
"""Fail-closed P0 and launch binding for the fixed-S128 I/T25 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import (
    append_jsonl,
    checkpoint_inventory,
    validate_jsonl_chain,
)
from recurrent.research.stable_eval_identity import (
    canonical_sha256,
    stable_eval_runtime_config_sha256,
    validate_resolved_manifest,
)
from tools.h20.preflight_qwen25_7b_stable_i4x2 import (
    _load_parquet_rows,
    _runtime_versions,
    build_identity_payload,
    compose_resolved_trainer_config,
    freeze_existing_s128_rows,
    model_loading_relevant_paths,
    sha256_file,
    validate_s128_freeze,
)


ENVIRONMENT_NAMES = (
    "MEMAGENT_S128_IT_WORK_ROOT",
    "MEMAGENT_S128_IT_REPO_DIR",
    "MEMAGENT_S128_IT_EXPECTED_COMMIT",
)
EXPECTED_BRANCH = "h20/qwen25-7b-original-t25-s128-frozen-20260821"
EXPECTED_DATA_SHA256 = "54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6"
COMMAND_MANIFEST = "manifests/h20/qwen25_7b_s128_it_commands.json"
EXPECTED_INTERFACES = ("I", "T25")
REQUIRED_GIT_OBJECTS = (
    "manifests/h20/qwen25_7b_s128_it_seed2026.json",
    COMMAND_MANIFEST,
    "scripts/h20/s128_it_common.sh",
    "scripts/h20/run_qwen25_7b_s128_it.sh",
    "tools/h20/preflight_qwen25_7b_s128_it.py",
    "tools/h20/audit_qwen25_7b_s128_it.py",
    "tests/h20/test_s128_it_frozen.py",
    "s128_it_execution_ledger.schema.json",
)
EXECUTION_CODE_OBJECTS = (
    "recurrent/impls/memory.py",
    "recurrent/generation_manager.py",
    "recurrent/research/stable_eval_identity.py",
    "recurrent/research/hotpotqa_dense_reward.py",
    "recurrent/research/s128_hotpot_metrics.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
    "verl/utils/dataset/rl_dataset.py",
    "verl/workers/fsdp_workers.py",
    "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
    "verl/workers/sharding_manager/fsdp_vllm.py",
)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def resolve_manifest_environment(
    value: Any, environment: Mapping[str, str] | None = None
) -> Any:
    source = os.environ if environment is None else environment
    missing = [name for name in ENVIRONMENT_NAMES if not source.get(name)]
    if missing:
        raise ValueError(f"missing explicit S128 I/T runtime bindings: {missing}")
    if re.fullmatch(r"[0-9a-f]{40}", str(source[ENVIRONMENT_NAMES[2]])) is None:
        raise ValueError("MEMAGENT_S128_IT_EXPECTED_COMMIT must be a full Git SHA")
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
    resolved = resolve_manifest_environment(
        json.loads(Path(path).read_text(encoding="utf-8")), environment
    )
    if not isinstance(resolved, dict):
        raise TypeError("S128 I/T manifest must be a JSON object")
    return resolved


def _expected_step(interface_id: str) -> int:
    if interface_id == "I":
        return 0
    if interface_id == "T25":
        return 25
    raise ValueError(f"unknown interface: {interface_id}")


def _attempt_id(manifest: Mapping[str, Any], interface_id: str) -> str:
    value = manifest["evaluation"]["attempt_ids"].get(interface_id)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing attempt ID for interface {interface_id}")
    return value


def render_trainer_overrides(
    manifest: Mapping[str, Any],
    *,
    repo: Path,
    interface_id: str,
    eval_manifest_hash: str,
    expected_runtime_config_sha256: str,
) -> list[str]:
    if interface_id not in EXPECTED_INTERFACES:
        raise ValueError(f"interface is not preregistered: {interface_id}")
    commands = json.loads((repo / COMMAND_MANIFEST).read_text(encoding="utf-8"))
    common = commands.get("common_trainer_overrides")
    specific = commands.get("interface_overrides", {}).get(interface_id)
    if not isinstance(common, list) or not common or not isinstance(specific, list):
        raise ValueError("command manifest lacks common/interface trainer overrides")
    interface_root = Path(manifest["paths"][interface_id])
    replacements = {
        "${VALIDATION_PATH}": str(manifest["data"]["validation"]),
        "${MODEL_PATH}": str(manifest["model"]["path"]),
        "${REPO_DIR}": str(repo),
        "${EXPERIMENT_NAME}": (
            "qwen25_7b_s128_i_base_seed2026_20260821"
            if interface_id == "I"
            else "qwen25_7b_s128_t25_corrected_original_style_seed2026_20260821"
        ),
        "${INTERFACE_ID}": interface_id,
        "${ATTEMPT_ID}": _attempt_id(manifest, interface_id),
        "${INTERFACE_ROOT}": str(interface_root),
        "${TERMINAL_DIR}": str(interface_root / "terminal"),
        "${RESOLVED_MANIFEST_PATH}": str(manifest["paths"]["resolved_manifest"]),
        "${EVAL_MANIFEST_HASH}": str(eval_manifest_hash),
        "${TURN_LEDGER_PATH}": str(interface_root / "trajectory_turns.jsonl"),
        "${EXECUTION_SUMMARY_PATH}": str(interface_root / "execution_summary.json"),
        "${EXPECTED_RUNTIME_CONFIG_SHA256}": str(expected_runtime_config_sha256),
        "${T25_CHECKPOINT}": str(manifest["training_anchor"]["checkpoint"]),
    }
    rendered: list[str] = []
    for item in [*common, *specific]:
        if not isinstance(item, str) or "\n" in item or "\r" in item:
            raise ValueError("trainer override argv must contain plain strings")
        value = item
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        if "${" in value or "=" not in value:
            raise ValueError(f"invalid or unresolved trainer override: {value}")
        rendered.append(value)
    return rendered


def generation_protocol_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that can affect recurrent generation/scoring."""
    return {
        "recurrent": config["recurrent"],
        "data": {
            key: config["data"][key]
            for key in (
                "val_files", "shuffle", "filter_overlong_prompts",
                "filter_overlong_prompts_workers", "dataloader_num_workers",
                "include_source_order_index", "truncation", "context_key",
                "val_max_samples", "max_prompt_length", "max_response_length",
            )
        },
        "model": {
            "path": config["actor_rollout_ref"]["model"]["path"],
            "use_remove_padding": config["actor_rollout_ref"]["model"]["use_remove_padding"],
        },
        "rollout": {
            key: config["actor_rollout_ref"]["rollout"][key]
            for key in (
                "name", "mode", "n", "tensor_model_parallel_size",
                "max_num_batched_tokens", "max_num_seqs", "val_kwargs",
            )
        },
        "reward_manager": config["reward_model"]["reward_manager"],
        "custom_reward_function": config["custom_reward_function"],
    }


def freeze_trainer_configuration(
    manifest: Mapping[str, Any], *, repo: Path, eval_manifest_hash: str
) -> dict[str, Any]:
    interfaces: dict[str, Any] = {}
    protocol_hashes: dict[str, str] = {}
    for interface_id in EXPECTED_INTERFACES:
        placeholder = render_trainer_overrides(
            manifest,
            repo=repo,
            interface_id=interface_id,
            eval_manifest_hash=eval_manifest_hash,
            expected_runtime_config_sha256="0" * 64,
        )
        config = compose_resolved_trainer_config(repo, placeholder)
        resolved_sha = stable_eval_runtime_config_sha256(config)
        final = render_trainer_overrides(
            manifest,
            repo=repo,
            interface_id=interface_id,
            eval_manifest_hash=eval_manifest_hash,
            expected_runtime_config_sha256=resolved_sha,
        )
        final_config = compose_resolved_trainer_config(repo, final)
        if stable_eval_runtime_config_sha256(final_config) != resolved_sha:
            raise ValueError(f"self-hashed Hydra config is unstable for {interface_id}")
        protocol_hash = canonical_sha256(generation_protocol_projection(final_config))
        protocol_hashes[interface_id] = protocol_hash
        interfaces[interface_id] = {
            "resolved_config_sha256": resolved_sha,
            "override_argv_sha256": canonical_sha256(final),
            "override_count": len(final),
            "generation_protocol_sha256": protocol_hash,
        }
    if len(set(protocol_hashes.values())) != 1:
        raise ValueError(
            "I/T generation-affecting Hydra configuration differs: "
            f"{protocol_hashes}"
        )
    return {
        "hydra_config_name": "ppo_trainer",
        "hydra_config_dir": "verl/trainer/config",
        "interfaces": interfaces,
        "shared_generation_protocol_sha256": next(iter(protocol_hashes.values())),
    }


def _checkpoint_contract(
    manifest: Mapping[str, Any], *, expected_git_commit: str | None = None
) -> dict[str, Any]:
    anchor = manifest["training_anchor"]
    checkpoint = Path(anchor["checkpoint"])
    report_path = Path(anchor["final_report"])
    if checkpoint.name != "global_step_25" or checkpoint.parent.name != anchor["experiment_name"]:
        raise ValueError("T25 checkpoint must be the exact frozen experiment/global_step_25 path")
    if not checkpoint.is_dir() or not report_path.is_file():
        raise ValueError("T25 checkpoint or corrected Original-style training final report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("corrected Original-style T25 training report is not PASS")
    if report.get("decision") != "ORIGINAL_T25_PASS":
        raise ValueError("corrected Original-style T25 training decision is not ORIGINAL_T25_PASS")
    if report.get("experiment_name") != anchor["experiment_name"]:
        raise ValueError("training report experiment name differs from the frozen anchor")
    if report.get("not_original_paper_7b_reproduction") is not True:
        raise ValueError("training report must explicitly disclaim original-paper reproduction")
    report_commit = report.get("git_commit")
    if re.fullmatch(r"[0-9a-f]{40}", str(report_commit or "")) is None:
        raise ValueError("training report lacks a full Git commit")
    if expected_git_commit is not None and report_commit != expected_git_commit:
        raise ValueError(
            "training report and S128 evaluation are not from the same frozen commit: "
            f"{report_commit} != {expected_git_commit}"
        )
    stable_canary = _stable_canary_contract(manifest)
    training_stable = report.get("stable_identity_prerequisite")
    expected_training_stable = {
        "status": "PASS",
        "decision": "I_RECURRENT_IDENTITY_CANARY_PASS",
        "commit": stable_canary["git_commit"],
        "report_sha256": stable_canary["sha256"],
        "eval_manifest_hash": stable_canary["eval_manifest_hash"],
        "execution_ledger_sha256": stable_canary["execution_ledger_sha256"],
    }
    if training_stable != expected_training_stable:
        raise ValueError(
            "T25 training report does not authenticate the current stable-I prerequisite: "
            f"actual={training_stable}, expected={expected_training_stable}"
        )
    version25_digest = (
        report.get("weight_sync", {}).get("version_digests", {}).get("25")
    )
    if re.fullmatch(r"[0-9a-f]{64}", str(version25_digest or "")) is None:
        raise ValueError("training report lacks a valid T25 effective actor/vLLM digest")
    frozen = report.get("step25_checkpoint")
    if not isinstance(frozen, Mapping):
        raise ValueError("training report lacks step25_checkpoint evidence")
    if Path(str(frozen.get("path", ""))).resolve() != checkpoint.resolve():
        raise ValueError("training report step25 checkpoint path differs from evaluation anchor")
    if frozen.get("global_step") != 25:
        raise ValueError("training report step25 checkpoint has wrong global step")
    declared_inventory = frozen.get("inventory")
    if not isinstance(declared_inventory, list):
        raise ValueError("training report step25 inventory is missing")
    current_inventory = checkpoint_inventory(checkpoint)
    if declared_inventory != current_inventory:
        raise ValueError("current T25 checkpoint inventory differs from training final report")
    inventory_sha = canonical_sha256(current_inventory)
    if frozen.get("inventory_sha256") != inventory_sha:
        raise ValueError("training report step25 inventory SHA is not canonical/current")

    ledger_path = Path(anchor["execution_ledger"])
    if not ledger_path.is_file():
        raise ValueError(f"T25 training execution ledger is missing: {ledger_path}")
    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chain_failures = validate_jsonl_chain(records)
    if chain_failures:
        raise ValueError(f"T25 training ledger hash chain failed: {chain_failures}")
    expected_type_counts = Counter(
        {
            "p0_preflight": 1,
            "runtime_config": 1,
            "resume_load": 1,
            "rollout_start": 22,
            "execution_signal": 22,
            "weight_sync_ack": 46,
            "weight_sync_summary": 23,
            "checkpoint_inventory": 1,
            "audit_result": 1,
        }
    )
    actual_type_counts = Counter(record.get("record_type") for record in records)
    if actual_type_counts != expected_type_counts or len(records) != 118:
        raise ValueError(
            "T25 training ledger is not the exact certified 116-record training "
            f"prefix plus two-record audit suffix: {dict(actual_type_counts)}"
        )
    if [record.get("record_type") for record in records[:3]] != [
        "p0_preflight",
        "runtime_config",
        "resume_load",
    ] or [record.get("record_type") for record in records[-2:]] != [
        "checkpoint_inventory",
        "audit_result",
    ]:
        raise ValueError("T25 training ledger prefix/suffix order differs")
    run_ids = {record.get("run_id") for record in records}
    if (
        len(run_ids) != 1
        or re.fullmatch(r"[0-9a-f]{32}", str(next(iter(run_ids), ""))) is None
        or any(record.get("git_commit") != report_commit for record in records)
        or any(
            record.get("experiment_name") != anchor["experiment_name"]
            for record in records
        )
    ):
        raise ValueError("T25 training ledger commit/run/experiment identity differs")
    training_records = records[:-2]
    checkpoint_record, audit_record = records[-2:]
    training_report_sha = sha256_file(report_path)
    if (
        audit_record.get("status") != "PASS"
        or audit_record.get("decision") != "ORIGINAL_T25_PASS"
        or Path(str(audit_record.get("report", ""))).resolve()
        != report_path.resolve()
        or audit_record.get("report_sha256") != training_report_sha
    ):
        raise ValueError("T25 audit-result ledger tail does not authenticate the final report")
    report_ledger = report.get("execution_ledger")
    if not isinstance(report_ledger, Mapping) or (
        report_ledger.get("training_prefix_record_count") != len(training_records)
        or report_ledger.get("training_prefix_sha256")
        != canonical_sha256(training_records)
        or report_ledger.get("training_prefix_tail_sha256")
        != training_records[-1].get("record_sha256")
    ):
        raise ValueError("T25 final report does not authenticate its training-ledger prefix")
    checkpoint_anchors = report.get("checkpoint_anchors")
    if (
        not isinstance(checkpoint_anchors, list)
        or [item.get("global_step") for item in checkpoint_anchors]
        != [5, 10, 15, 20, 25]
        or checkpoint_record.get("global_step") != 25
        or checkpoint_record.get("inventory") != declared_inventory
        or checkpoint_record.get("inventory_sha256") != inventory_sha
        or checkpoint_record.get("checkpoint_anchors") != checkpoint_anchors
        or checkpoint_record.get("checkpoint_anchors_sha256")
        != canonical_sha256(checkpoint_anchors)
    ):
        raise ValueError("T25 checkpoint-inventory ledger record does not bind all five anchors")
    required = set(anchor["required_actor_shards"])
    model_shards = {
        item["path"] for item in current_inventory
        if re.fullmatch(r"actor/model_world_size_\d+_rank_\d+\.pt", item["path"])
    }
    if model_shards != required:
        raise ValueError(
            f"T25 actor model shard inventory is not exact world-size 2: {sorted(model_shards)}"
        )
    actor_shards = [item for item in current_inventory if item["path"] in required]
    actor_shards.sort(key=lambda item: item["path"])
    return {
        "path": str(checkpoint.resolve()),
        "global_step": 25,
        "training_report": str(report_path.resolve()),
        "training_report_sha256": training_report_sha,
        "training_execution_ledger": str(ledger_path.resolve()),
        "training_execution_ledger_sha256": sha256_file(ledger_path),
        "training_execution_ledger_tail_sha256": audit_record["record_sha256"],
        "training_git_commit": report_commit,
        "training_decision": report["decision"],
        "training_stable_i_eval_manifest_hash": (
            report.get("stable_identity_prerequisite", {}).get("eval_manifest_hash")
        ),
        "training_stable_i_report_sha256": stable_canary["sha256"],
        "training_effective_actor_vllm_digest": version25_digest,
        "inventory": current_inventory,
        "inventory_sha256": inventory_sha,
        "actor_model_shards": actor_shards,
        "actor_model_shards_sha256": canonical_sha256(actor_shards),
        "fsdp_world_size": 2,
        "load_mode": "actor_only",
    }


def _stable_canary_contract(
    manifest: Mapping[str, Any], *, expected_eval_manifest_hash: str | None = None
) -> dict[str, Any]:
    contract = manifest["stable_identity_canary"]
    report_path = Path(contract["final_report"])
    resolved_path = Path(contract["resolved_manifest"])
    ledger_path = Path(contract["execution_ledger"])
    for label, path in (
        ("final report", report_path),
        ("resolved manifest", resolved_path),
        ("execution ledger", ledger_path),
    ):
        if not path.is_file():
            raise ValueError(f"stable-I canary {label} is missing: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != contract["required_status"]:
        raise ValueError("stable-I canary report is not PASS")
    if report.get("decision") != contract["required_decision"]:
        raise ValueError("stable-I canary decision is not I_RECURRENT_IDENTITY_CANARY_PASS")
    if report.get("failures") != []:
        raise ValueError("stable-I canary report is not a clean failure-free PASS")
    evidence = report.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("stable-I canary report lacks evidence")
    if evidence.get("git_commit") != contract["required_git_commit"]:
        raise ValueError("stable-I canary Git commit differs from the frozen prerequisite")
    canary_eval_hash = evidence.get("eval_manifest_hash")
    if re.fullmatch(r"[0-9a-f]{64}", str(canary_eval_hash or "")) is None:
        raise ValueError("stable-I canary report lacks a valid eval manifest hash")
    required_eval_hash = contract.get("required_eval_manifest_hash")
    if required_eval_hash is not None and canary_eval_hash != required_eval_hash:
        raise ValueError(
            "stable-I canary evaluation hash differs from the frozen r2 PASS: "
            f"{canary_eval_hash} != {required_eval_hash}"
        )
    resolved = validate_resolved_manifest(
        json.loads(resolved_path.read_text(encoding="utf-8"))
    )
    if resolved["eval_manifest_hash"] != canary_eval_hash:
        raise ValueError("stable-I report/resolved-manifest evaluation hash disagrees")
    ledger_lines = [
        line
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [json.loads(line) for line in ledger_lines]
    chain_failures = validate_jsonl_chain(records)
    if chain_failures:
        raise ValueError(f"stable-I execution ledger hash chain failed: {chain_failures}")
    if not records:
        raise ValueError("stable-I execution ledger is empty")
    expected_sequence = [
        ("s0_preflight", None),
        ("run_start", "repeat_a"),
        ("run_finish", "repeat_a"),
        ("run_start", "repeat_b"),
        ("run_finish", "repeat_b"),
        ("audit_result", None),
    ]
    actual_sequence = [
        (record.get("record_type"), record.get("attempt_id"))
        for record in records
    ]
    if actual_sequence != expected_sequence:
        raise ValueError(
            "stable-I execution ledger sequence differs from the completed r2 canary: "
            f"{actual_sequence}"
        )
    execution_binding_sha256 = canonical_sha256(
        resolved.get("execution_binding")
    )
    for index, record in enumerate(records):
        if (
            record.get("git_commit") != contract["required_git_commit"]
            or record.get("eval_manifest_hash") != canary_eval_hash
            or record.get("execution_binding_sha256") != execution_binding_sha256
        ):
            raise ValueError(
                f"stable-I execution ledger record {index} changed commit/eval/execution binding"
            )
    prefix_sha256 = hashlib.sha256(
        "".join(f"{line}\n" for line in ledger_lines[:-1]).encode("utf-8")
    ).hexdigest()
    if (
        evidence.get("execution_ledger_records") != len(records) - 1
        or evidence.get("execution_ledger_sha256") != prefix_sha256
    ):
        raise ValueError(
            "stable-I final report does not authenticate the pre-audit ledger prefix"
        )
    tail = records[-1]
    report_sha256 = sha256_file(report_path)
    if (
        tail.get("record_type") != "audit_result"
        or tail.get("status") != "PASS"
        or tail.get("decision") != "I_RECURRENT_IDENTITY_CANARY_PASS"
        or tail.get("git_commit") != contract["required_git_commit"]
        or tail.get("eval_manifest_hash") != canary_eval_hash
        or Path(str(tail.get("artifact", ""))).resolve() != report_path.resolve()
        or tail.get("artifact_sha256") != report_sha256
    ):
        raise ValueError("stable-I ledger tail does not authenticate the required final PASS")
    if (
        expected_eval_manifest_hash is not None
        and canary_eval_hash != expected_eval_manifest_hash
    ):
        raise ValueError(
            "stable-I canary and S128 evaluation do not share the same manifest hash: "
            f"{canary_eval_hash} != {expected_eval_manifest_hash}"
        )
    return {
        "path": str(report_path.resolve()),
        "sha256": report_sha256,
        "resolved_manifest": str(resolved_path.resolve()),
        "resolved_manifest_sha256": sha256_file(resolved_path),
        "execution_ledger": str(ledger_path.resolve()),
        "execution_ledger_sha256": sha256_file(ledger_path),
        "execution_ledger_tail_sha256": tail["record_sha256"],
        "status": report["status"],
        "decision": report["decision"],
        "git_commit": evidence["git_commit"],
        "eval_manifest_hash": canary_eval_hash,
    }


def capture_runtime_binding(
    manifest: Mapping[str, Any], repo: Path, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    model_root = Path(manifest["model"]["path"])
    model_files = [
        {
            "path": item["path"],
            "size": (model_root / item["path"]).stat().st_size,
            "sha256": sha256_file(model_root / item["path"]),
        }
        for item in manifest["model"]["files"]
    ]
    validation = Path(manifest["data"]["validation"])
    versions, error = _runtime_versions(str(manifest["python"]), repo)
    if error or versions is None:
        raise ValueError(f"runtime import failed: {error}")
    gpu = subprocess.run(
        [
            "nvidia-smi", "-i", manifest["gpu"]["visible_devices"],
            "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=False,
    )
    if gpu.returncode:
        raise ValueError(f"cannot identify GPU6-7: {gpu.stderr.strip()}")
    # P0 hashes the complete checkpoint once.  At each TOCTOU boundary, stat
    # the complete inventory and rehash only the two actor shards actually
    # consumed by actor-only evaluation plus the training report.
    if checkpoint.get("training_git_commit") != os.environ.get(ENVIRONMENT_NAMES[2]):
        raise ValueError("P0-frozen T25 training commit differs from current expected commit")
    checkpoint_root = Path(str(checkpoint["path"]))
    current_stat_inventory = [
        {"path": str(path.relative_to(checkpoint_root)), "size": path.stat().st_size}
        for path in sorted(item for item in checkpoint_root.rglob("*") if item.is_file())
    ]
    frozen_stat_inventory = [
        {"path": item["path"], "size": item["size"]}
        for item in checkpoint["inventory"]
    ]
    if current_stat_inventory != frozen_stat_inventory:
        raise ValueError("T25 checkpoint path/size inventory changed after P0")
    current_actor_shards = []
    for frozen in checkpoint["actor_model_shards"]:
        target = checkpoint_root / frozen["path"]
        current = {
            "path": frozen["path"], "size": target.stat().st_size,
            "sha256": sha256_file(target),
        }
        if current != frozen:
            raise ValueError(f"T25 actor shard changed after P0: {current} != {frozen}")
        current_actor_shards.append(current)
    report_path = Path(str(checkpoint["training_report"]))
    if sha256_file(report_path) != checkpoint["training_report_sha256"]:
        raise ValueError("T25 training final report changed after P0")
    stable_canary = _stable_canary_contract(manifest)
    return {
        "git_commit": git(repo, "rev-parse", "HEAD"),
        "branch": git(repo, "branch", "--show-current"),
        "worktree_clean": not bool(git(repo, "status", "--porcelain")),
        "validation_data_sha256": sha256_file(validation),
        "model_file_inventory": model_files,
        "model_loading_relevant_paths": model_loading_relevant_paths(model_root),
        "runtime_versions": versions,
        "physical_gpu_identity": [
            line.strip() for line in gpu.stdout.splitlines() if line.strip()
        ],
        "t25_checkpoint_inventory_sha256": checkpoint["inventory_sha256"],
        "t25_actor_model_shards_sha256": canonical_sha256(current_actor_shards),
        "t25_training_report_sha256": checkpoint["training_report_sha256"],
        "stable_i_canary_report_sha256": stable_canary["sha256"],
        "stable_i_canary_resolved_manifest_sha256": stable_canary[
            "resolved_manifest_sha256"
        ],
        "stable_i_canary_execution_ledger_sha256": stable_canary[
            "execution_ledger_sha256"
        ],
        "stable_i_canary_execution_ledger_tail_sha256": stable_canary[
            "execution_ledger_tail_sha256"
        ],
        "stable_i_canary_eval_manifest_hash": stable_canary["eval_manifest_hash"],
    }


def build_execution_binding(
    manifest: Mapping[str, Any], *, repo: Path, rows: list[Mapping[str, Any]],
    checkpoint: Mapping[str, Any], stable_canary: Mapping[str, Any],
    trainer_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    chunk_size = int(manifest["recurrent"]["chunk_size"])
    active = {
        str(row["source_order_index"]):
            (int(row["context_token_count"]) + chunk_size - 1) // chunk_size
        for row in rows
    }
    code = {path: sha256_file(repo / path) for path in EXECUTION_CODE_OBJECTS}
    return {
        "git_commit": git(repo, "rev-parse", "HEAD"),
        "interfaces": ["I", "T25"],
        "base_seed": int(manifest["evaluation"]["base_seed"]),
        "replicas": 1,
        "model_artifacts": {
            "I": {
                "kind": "frozen_huggingface_base_model",
                "file_manifest_sha256": canonical_sha256(manifest["model"]["files"]),
            },
            "T25": dict(checkpoint),
        },
        "stable_identity_canary_prerequisite": dict(stable_canary),
        "recurrent": dict(manifest["recurrent"]),
        "all_s128_turn_schedule": {
            "active_turn_count_by_source_order": active,
            "shared_final_turn": max(active.values()),
        },
        "execution_code_sha256": code,
        "execution_code_combined_sha256": canonical_sha256(code),
        "trainer_configuration": dict(trainer_configuration),
    }


def run_preflight(
    manifest_path: Path, *, check_runtime: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    manifest_path = manifest_path.resolve()
    repo = manifest_path.parents[2]
    manifest = load_manifest(manifest_path)
    failures: list[str] = []
    evidence: dict[str, Any] = {
        "frozen_manifest_sha256": sha256_file(manifest_path),
        "resolved_runtime_manifest_sha256": canonical_sha256(manifest),
    }
    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    status = git(repo, "status", "--porcelain")
    expected_commit = os.environ.get(ENVIRONMENT_NAMES[2], "")
    evidence.update(
        git_commit=head, expected_git_commit=expected_commit,
        branch=branch, worktree_clean=not bool(status),
    )
    if head != expected_commit:
        failures.append(f"exact Git commit mismatch: {head} != {expected_commit}")
    if subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor",
         "bd8b804c2cbf333f0f0650b729fd03a143d445b2", head],
        check=False,
    ).returncode:
        failures.append("HEAD does not contain the passed stable-I r2 commit bd8b804")
    if manifest.get("base_commit") != "bd8b804c2cbf333f0f0650b729fd03a143d445b2":
        failures.append("manifest base commit is not the passed stable-I canary commit")
    if branch != EXPECTED_BRANCH or manifest.get("branch") != EXPECTED_BRANCH:
        failures.append(f"branch mismatch: {branch} != {EXPECTED_BRANCH}")
    if status:
        failures.append(f"worktree is dirty: {status.splitlines()}")
    if Path(manifest["repository"]).resolve() != repo.resolve():
        failures.append("manifest repository binding differs from invoked checkout")
    missing = [name for name in REQUIRED_GIT_OBJECTS if not (repo / name).is_file()]
    if missing:
        failures.append(f"required Git objects are missing: {missing}")
    untracked = [
        name for name in REQUIRED_GIT_OBJECTS
        if subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode
    ]
    if untracked:
        failures.append(f"required Git objects are not committed: {untracked}")

    scope = manifest.get("scope", {})
    for name in (
        "existing_s128_only", "no_resampling", "raw_context_r_not_rerun",
        "paired_difference_is_descriptive_not_causal",
        "same_prompt_and_recurrent_protocol", "not_original_paper_7b_reproduction",
        "published_results_are_historical_reference_only",
    ):
        if scope.get(name) is not True:
            failures.append(f"scientific scope limitation is not frozen true: {name}")
    if scope.get("interfaces_run") != ["I", "T25"]:
        failures.append("only I and T25 may run in this evaluation")
    evaluation = manifest.get("evaluation", {})
    expected_eval = {
        "examples": 128, "replicas": 1, "interfaces": ["I", "T25"],
        "validation_only": True, "actor_update_calls": 0,
        "optimizer_step_calls": 0, "checkpoint_save_calls": 0,
        "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
    }
    for key, value in expected_eval.items():
        if evaluation.get(key) != value:
            failures.append(f"evaluation contract {key} differs: {evaluation.get(key)!r}")
    if evaluation.get("primary_metrics") != ["normalized_exact_match", "token_f1"]:
        failures.append("primary metric contract is not normalized EM/F1")
    if evaluation.get("diagnostic_metrics") != [
        "format_success", "historical_sub_exact_match"
    ]:
        failures.append("diagnostic metric contract drifted")
    if evaluation.get("training_dense_reward_excluded_from_evaluation_claims") is not True:
        failures.append("training dense reward is not explicitly excluded from evaluation claims")
    if evaluation.get("metrics_recomputed_from_terminal_output_and_parquet_ground_truth") is not True:
        failures.append("evaluation metrics are not required to be independently recomputed")
    data = manifest["data"]
    if any((
        data.get("validation_sha256") != EXPECTED_DATA_SHA256,
        data.get("dataset_role") != "existing_project_fixed_s128",
        data.get("expected_raw_rows") != 128,
        data.get("expected_effective_rows") != 128,
        data.get("shuffle") is not False,
        data.get("include_source_order_index") is not True,
    )):
        failures.append("fixed existing S128 data contract drifted")
    if manifest.get("backend") != {
        "rollout": "vllm", "evaluation": "vllm", "rollout_mode": "sync",
        "allow_hf_fallback": False, "reward_manager": "naive",
    }:
        failures.append("backend is not strict synchronous vLLM recurrent evaluation")
    commands = json.loads((repo / COMMAND_MANIFEST).read_text(encoding="utf-8"))
    if commands.get("required_sequence") != ["p0", "I", "T25", "audit"]:
        failures.append("command sequence is not P0 -> I -> T25 -> audit")
    if commands.get("gpu_execution_authorized_by_this_manifest") is not False:
        failures.append("command manifest improperly self-authorizes GPU execution")
    if commands.get("causal_claim_authorized_by_this_manifest") is not False:
        failures.append("command manifest improperly authorizes a causal claim")
    if manifest.get("ledger_schema") != "s128_it_execution_ledger.schema.json":
        failures.append("S128 I/T ledger schema binding drifted")
    try:
        json.loads((repo / str(manifest["ledger_schema"])).read_text(encoding="utf-8"))
    except Exception as error:
        failures.append(f"S128 I/T ledger schema cannot be loaded: {error}")

    paths = [
        Path(manifest["paths"][name])
        for name in ("p0_certificate", "resolved_manifest", "final_report", "execution_ledger", "I", "T25")
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        failures.append(f"append-only S128 I/T evidence already exists before P0: {existing}")

    model_root = Path(manifest["model"]["path"])
    validation = Path(data["validation"])
    for path in (Path(manifest["python"]), model_root, validation):
        if not path.exists():
            failures.append(f"required runtime path is missing: {path}")
    if validation.is_file() and sha256_file(validation) != EXPECTED_DATA_SHA256:
        failures.append("existing HotpotQA S128 parquet SHA changed")
    actual_model_inventory: list[dict[str, Any]] = []
    if model_root.is_dir():
        for item in manifest["model"]["files"]:
            target = model_root / item["path"]
            if not target.is_file():
                failures.append(f"frozen base-model file is missing: {target}")
                continue
            actual = {"path": item["path"], "size": target.stat().st_size, "sha256": sha256_file(target)}
            actual_model_inventory.append(actual)
            if actual != item:
                failures.append(f"base-model file differs: {actual} != {item}")
        if model_loading_relevant_paths(model_root) != sorted(
            item["path"] for item in manifest["model"]["files"]
        ):
            failures.append("base-model loading-relevant inventory is not exact")
    evidence["model_file_inventory"] = actual_model_inventory

    checkpoint: dict[str, Any] | None = None
    stable_canary: dict[str, Any] | None = None
    try:
        checkpoint = _checkpoint_contract(manifest, expected_git_commit=expected_commit)
        evidence["t25_checkpoint"] = checkpoint
    except Exception as error:
        failures.append(f"T25 checkpoint provenance failed: {error}")
    try:
        stable_canary = _stable_canary_contract(manifest)
        evidence["stable_identity_canary"] = stable_canary
    except Exception as error:
        failures.append(f"stable-I canary prerequisite failed: {error}")

    resolved: dict[str, Any] | None = None
    if (
        model_root.is_dir() and validation.is_file()
        and checkpoint is not None and stable_canary is not None
    ):
        try:
            from verl.utils import hf_tokenizer

            tokenizer = hf_tokenizer(str(model_root), trust_remote_code=False)
            frozen_rows, filter_evidence = freeze_existing_s128_rows(
                _load_parquet_rows(validation),
                prompt_token_length=lambda prompt: len(
                    tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
                ),
                context_token_length=lambda context: len(
                    tokenizer.encode(context, add_special_tokens=False)
                ),
                max_prompt_length=40000,
                max_context_length=40000,
            )
            failures.extend(validate_s128_freeze(frozen_rows, filter_evidence, data))
            if [row["source_order_index"] for row in frozen_rows] != list(range(128)):
                failures.append("S128 source order is not exactly 0..127")
            payload = build_identity_payload(manifest, rows=frozen_rows)
            eval_hash = canonical_sha256(payload)
            stable_canary = _stable_canary_contract(
                manifest, expected_eval_manifest_hash=eval_hash
            )
            if checkpoint.get("training_stable_i_eval_manifest_hash") != eval_hash:
                raise ValueError(
                    "T25 training report, stable-I canary, and S128 evaluation do not "
                    "share one eval manifest hash"
                )
            trainer = freeze_trainer_configuration(
                manifest, repo=repo, eval_manifest_hash=eval_hash
            )
            execution = build_execution_binding(
                manifest, repo=repo, rows=frozen_rows,
                checkpoint=checkpoint, stable_canary=stable_canary,
                trainer_configuration=trainer,
            )
            resolved = {
                "schema_version": 1,
                "frozen_manifest_sha256": evidence["frozen_manifest_sha256"],
                "identity_payload": payload,
                "eval_manifest_hash": eval_hash,
                "cohort": {
                    "source_order_indices": list(range(128)),
                    "examples": 128,
                    "replicas": 1,
                    "interfaces": ["I", "T25"],
                },
                "execution_binding": execution,
            }
            validate_resolved_manifest(resolved)
            evidence.update(
                eval_manifest_hash=eval_hash,
                execution_binding_sha256=canonical_sha256(execution),
                s128_filter_replay=filter_evidence,
            )
        except Exception as error:
            failures.append(f"cannot freeze complete S128/shared protocol: {error}")

    if check_runtime and checkpoint is not None:
        try:
            runtime = capture_runtime_binding(manifest, repo, checkpoint)
            evidence["runtime_binding"] = runtime
            evidence["runtime_binding_sha256"] = canonical_sha256(runtime)
            gpu_rows = runtime["physical_gpu_identity"]
            if len(gpu_rows) != 2 or [int(row.split(",", 1)[0]) for row in gpu_rows] != [6, 7]:
                failures.append(f"physical GPUs are not exact 6,7: {gpu_rows}")
            if any("H20" not in row for row in gpu_rows):
                failures.append(f"physical GPUs 6,7 are not both H20: {gpu_rows}")
        except Exception as error:
            failures.append(f"runtime binding failed: {error}")

    return ({
        "gate": "P0",
        "status": "PASS" if not failures else "FAIL",
        "decision": "S128_IT_P0_PASS" if not failures else "S128_IT_NO_GO:P0",
        "scope": manifest["scope"],
        "failures": failures,
        "evidence": evidence,
    }, resolved)


def _current_certificate(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    p0_path = Path(manifest["paths"]["p0_certificate"])
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    ledger_path = Path(manifest["paths"]["execution_ledger"])
    if not p0_path.is_file() or not resolved_path.is_file() or not ledger_path.is_file():
        raise ValueError("standalone P0 certificate/resolved manifest/ledger are required")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    resolved = validate_resolved_manifest(json.loads(resolved_path.read_text(encoding="utf-8")))
    expected = os.environ[ENVIRONMENT_NAMES[2]]
    evidence = p0.get("evidence")
    if (
        p0.get("status") != "PASS"
        or p0.get("decision") != "S128_IT_P0_PASS"
        or not isinstance(evidence, Mapping)
        or evidence.get("git_commit") != expected
        or evidence.get("expected_git_commit") != expected
        or re.fullmatch(r"[0-9a-f]{32}", str(evidence.get("run_id", ""))) is None
    ):
        raise ValueError("P0 is not the exact S128_IT_P0_PASS for this commit/run")
    if evidence.get("resolved_manifest_sha256") != sha256_file(resolved_path):
        raise ValueError("resolved manifest changed after P0")
    execution_sha = canonical_sha256(resolved["execution_binding"])
    runtime_sha = canonical_sha256(evidence.get("runtime_binding"))
    if (
        evidence.get("eval_manifest_hash") != resolved["eval_manifest_hash"]
        or evidence.get("execution_binding_sha256") != execution_sha
        or evidence.get("runtime_binding_sha256") != runtime_sha
    ):
        raise ValueError("P0 eval/execution/runtime binding is inconsistent")
    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chain_failures = validate_jsonl_chain(records)
    if chain_failures:
        raise ValueError(f"P0 ledger hash chain is invalid: {chain_failures}")
    if len(records) != 1:
        raise ValueError(f"P0 ledger must contain exactly one prefix record, got {len(records)}")
    record = records[0]
    expected_record = {
        "record_type": "s0_preflight",
        "experiment_name": "qwen25_7b_s128_it_p0_seed2026_20260821",
        "git_commit": expected,
        "run_id": evidence["run_id"],
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": execution_sha,
        "runtime_binding_sha256": runtime_sha,
        "interface_id": None,
        "status": "PASS",
        "artifact": str(p0_path),
        "artifact_sha256": sha256_file(p0_path),
        "row_count": 128,
    }
    for field, value in expected_record.items():
        if record.get(field) != value:
            raise ValueError(
                f"P0 ledger field {field}={record.get(field)!r}, expected {value!r}"
            )
    return p0, resolved


def frozen_trainer_overrides(manifest_path: Path, *, interface_id: str) -> list[str]:
    manifest = load_manifest(manifest_path)
    _, resolved = _current_certificate(manifest)
    binding = resolved["execution_binding"]["trainer_configuration"]["interfaces"].get(interface_id)
    if not isinstance(binding, Mapping):
        raise ValueError(f"P0 lacks trainer binding for {interface_id}")
    overrides = render_trainer_overrides(
        manifest,
        repo=Path(manifest["repository"]),
        interface_id=interface_id,
        eval_manifest_hash=resolved["eval_manifest_hash"],
        expected_runtime_config_sha256=str(binding["resolved_config_sha256"]),
    )
    if canonical_sha256(overrides) != binding.get("override_argv_sha256"):
        raise ValueError(f"trainer argv changed after P0 for {interface_id}")
    return overrides


def record_interface_event(
    manifest_path: Path, *, interface_id: str, event: str
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    p0, resolved = _current_certificate(manifest)
    checkpoint = resolved["execution_binding"]["model_artifacts"]["T25"]
    runtime = capture_runtime_binding(manifest, Path(manifest["repository"]), checkpoint)
    if runtime != p0["evidence"].get("runtime_binding"):
        raise ValueError("Git/data/model/checkpoint/runtime binding changed after P0")
    root = Path(manifest["paths"][interface_id])
    step = _expected_step(interface_id)
    artifacts_paths = (
        root / f"terminal/{step}.jsonl",
        root / "trajectory_turns.jsonl",
        root / "execution_summary.json",
        root / "run.log",
    )
    if event == "start":
        preexisting = [str(path) for path in artifacts_paths if path.exists()]
        if preexisting:
            raise ValueError(f"interface start found pre-existing evidence: {preexisting}")
        artifacts: dict[str, Any] = {}
    else:
        missing = [str(path) for path in artifacts_paths if not path.is_file()]
        if missing:
            raise ValueError(f"interface finish is missing evidence: {missing}")
        artifacts = {
            str(path.relative_to(root)): {
                "sha256": sha256_file(path), "size": path.stat().st_size,
            }
            for path in artifacts_paths
        }
    ledger = Path(manifest["paths"]["execution_ledger"])
    records = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chain_failures = validate_jsonl_chain(records)
    if chain_failures:
        raise ValueError(f"ledger hash chain is invalid: {chain_failures}")
    desired = [
        ("s0_preflight", None),
        ("interface_start", "I"), ("interface_finish", "I"),
        ("interface_start", "T25"), ("interface_finish", "T25"),
    ]
    next_pair = ("interface_start" if event == "start" else "interface_finish", interface_id)
    candidate = [(row.get("record_type"), row.get("interface_id")) for row in records] + [next_pair]
    if candidate != desired[: len(candidate)]:
        raise ValueError(f"interface event order differs from preregistration: {candidate}")
    record = {
        "record_type": next_pair[0],
        "experiment_name": (
            "qwen25_7b_s128_i_base_seed2026_20260821"
            if interface_id == "I"
            else "qwen25_7b_s128_t25_corrected_original_style_seed2026_20260821"
        ),
        "git_commit": p0["evidence"]["git_commit"],
        "run_id": p0["evidence"]["run_id"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": canonical_sha256(resolved["execution_binding"]),
        "runtime_binding_sha256": canonical_sha256(runtime),
        "interface_id": interface_id,
        "status": "PASS",
        "artifacts": artifacts,
    }
    append_jsonl(ledger, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--write-certificate", action="store_true")
    parser.add_argument("--emit-trainer-overrides", action="store_true")
    parser.add_argument("--validate-p0-prefix", action="store_true")
    parser.add_argument("--record-interface-event", choices=("start", "finish"))
    parser.add_argument("--interface", choices=EXPECTED_INTERFACES)
    args = parser.parse_args()
    if args.validate_p0_prefix:
        try:
            manifest = load_manifest(args.manifest)
            _current_certificate(manifest)
            print("S128_IT_P0_PREFIX_VALID=PASS")
            return 0
        except Exception as error:
            print(f"S128_IT_NO_GO:P0 {error}", file=sys.stderr)
            return 1
    if args.emit_trainer_overrides:
        if not args.interface:
            parser.error("--interface is required")
        try:
            print("\n".join(frozen_trainer_overrides(args.manifest, interface_id=args.interface)))
            return 0
        except Exception as error:
            print(f"S128_IT_NO_GO:CONFIG {error}", file=sys.stderr)
            return 1
    if args.record_interface_event:
        if not args.interface:
            parser.error("--interface is required")
        try:
            record = record_interface_event(
                args.manifest, interface_id=args.interface, event=args.record_interface_event
            )
            print(json.dumps(record, indent=2, sort_keys=True))
            return 0
        except Exception as error:
            print(json.dumps({"status": "FAIL", "decision": "S128_IT_NO_GO:EVENT", "failures": [str(error)]}, indent=2))
            return 1
    if args.write_certificate and not args.check_runtime:
        parser.error("--write-certificate requires --check-runtime")
    try:
        result, resolved = run_preflight(args.manifest, check_runtime=args.check_runtime)
        manifest = load_manifest(args.manifest)
    except Exception as error:
        result, resolved, manifest = ({
            "gate": "P0", "status": "FAIL", "decision": "S128_IT_NO_GO:P0",
            "failures": [str(error)], "evidence": {},
        }, None, None)
    if args.write_certificate:
        if result["status"] != "PASS" or resolved is None or manifest is None:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
        p0 = Path(manifest["paths"]["p0_certificate"])
        resolved_path = Path(manifest["paths"]["resolved_manifest"])
        ledger = Path(manifest["paths"]["execution_ledger"])
        existing = [str(path) for path in (p0, resolved_path, ledger) if path.exists()]
        if existing:
            raise SystemExit(f"refusing to overwrite append-only P0 evidence: {existing}")
        p0.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
        result["evidence"].update(
            run_id=secrets.token_hex(16),
            resolved_manifest_path=str(resolved_path),
            resolved_manifest_sha256=sha256_file(resolved_path),
        )
        with p0.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        append_jsonl(ledger, {
            "record_type": "s0_preflight",
            "experiment_name": "qwen25_7b_s128_it_p0_seed2026_20260821",
            "git_commit": result["evidence"]["git_commit"],
            "run_id": result["evidence"]["run_id"],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "eval_manifest_hash": resolved["eval_manifest_hash"],
            "execution_binding_sha256": canonical_sha256(resolved["execution_binding"]),
            "runtime_binding_sha256": result["evidence"]["runtime_binding_sha256"],
            "interface_id": None,
            "status": "PASS",
            "artifact": str(p0),
            "artifact_sha256": sha256_file(p0),
            "row_count": 128,
        })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
