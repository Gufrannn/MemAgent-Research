#!/usr/bin/env python3
"""Fail-closed P0 and launch binding for the fixed-S128 Original curve."""

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
from tools.h20.preflight_qwen25_7b_s128_it import (
    _stable_canary_contract,
    generation_protocol_projection,
    render_trainer_overrides as render_prior_s128_trainer_overrides,
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
    "MEMAGENT_ORIGINAL_CURVE_WORK_ROOT",
    "MEMAGENT_ORIGINAL_CURVE_REPO_DIR",
    "MEMAGENT_ORIGINAL_CURVE_EXPECTED_COMMIT",
)
EXPECTED_BRANCH = "h20/qwen25-7b-original-all-anchor-s128-frozen-20260821"
SOURCE_COMMIT = "b7bf64937b5825513df86ab963816b73604f102c"
EXPECTED_EVAL_HASH = "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a"
EXPECTED_METRIC_CODE_SHA = "addc282e3d48dc5e7b6ccf30205fde58a0c0515cb6a2341fd293a5b5b83da286"
INTERFACES = ("I", "Original5", "Original10", "Original15", "Original20", "Original25")
ANCHOR_STEPS = (5, 10, 15, 20, 25)
COMMAND_MANIFEST = "manifests/h20/qwen25_7b_original_s128_curve_commands.json"
REQUIRED_GIT_OBJECTS = (
    "manifests/h20/qwen25_7b_original_s128_curve_seed2026.json",
    COMMAND_MANIFEST,
    "original_s128_curve_execution_ledger.schema.json",
    "scripts/h20/original_s128_curve_common.sh",
    "scripts/h20/run_qwen25_7b_original_s128_curve.sh",
    "tools/h20/preflight_qwen25_7b_original_s128_curve.py",
    "tools/h20/audit_qwen25_7b_original_s128_curve.py",
    "tests/h20/test_original_s128_curve_frozen.py",
    "docs/h20/original_s128_all_anchor_curve_20260821.md",
)
EXECUTION_CODE_OBJECTS = (
    *REQUIRED_GIT_OBJECTS,
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
        raise ValueError(f"missing explicit Original curve runtime bindings: {missing}")
    if re.fullmatch(r"[0-9a-f]{40}", str(source[ENVIRONMENT_NAMES[2]])) is None:
        raise ValueError("MEMAGENT_ORIGINAL_CURVE_EXPECTED_COMMIT must be a full Git SHA")
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
    value = resolve_manifest_environment(
        json.loads(Path(path).read_text(encoding="utf-8")), environment
    )
    if not isinstance(value, dict):
        raise TypeError("Original curve manifest must be a JSON object")
    return value


def _load_inherited_contract(
    manifest: Mapping[str, Any], repo: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    inherited = manifest["inherited_s128_contract"]
    manifest_path = repo / inherited["manifest"]
    commands_path = repo / inherited["commands"]
    metric_path = repo / inherited["metric_code"]
    for path, expected in (
        (manifest_path, inherited["manifest_sha256"]),
        (commands_path, inherited["commands_sha256"]),
        (metric_path, inherited["metric_code_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"inherited b7bf S128 contract changed: {path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    work_root = str(manifest["work_root"])

    def resolve(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: resolve(value) for key, value in item.items()}
        if isinstance(item, list):
            return [resolve(value) for value in item]
        if isinstance(item, str):
            return item.replace("${MEMAGENT_S128_IT_WORK_ROOT}", work_root)
        return item

    inherited_manifest = resolve(raw)
    if inherited_manifest["stable_identity_canary"]["required_eval_manifest_hash"] != (
        EXPECTED_EVAL_HASH
    ):
        raise ValueError("inherited S128 evaluation hash is not the passed stable-I hash")
    return inherited_manifest, json.loads(commands_path.read_text(encoding="utf-8"))


def _step(interface_id: str) -> int:
    if interface_id == "I":
        return 0
    match = re.fullmatch(r"Original(5|10|15|20|25)", interface_id)
    if match is None:
        raise ValueError(f"unknown Original curve interface: {interface_id}")
    return int(match.group(1))


def _attempt_id(manifest: Mapping[str, Any], interface_id: str) -> str:
    value = manifest["evaluation"]["attempt_ids"].get(interface_id)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing attempt ID for {interface_id}")
    return value


def _experiment_name(interface_id: str) -> str:
    if interface_id == "I":
        return "qwen25_7b_s128_curve_i_base_seed2026_20260821"
    return f"qwen25_7b_s128_curve_original_step{_step(interface_id)}_seed2026_20260821"


def render_trainer_overrides(
    manifest: Mapping[str, Any], inherited: Mapping[str, Any], *, repo: Path,
    interface_id: str, eval_manifest_hash: str,
    expected_runtime_config_sha256: str,
) -> list[str]:
    if interface_id not in INTERFACES:
        raise ValueError(f"interface is not preregistered: {interface_id}")
    commands = json.loads((repo / COMMAND_MANIFEST).read_text(encoding="utf-8"))
    inherited_commands = json.loads(
        (repo / commands["inherited_common_trainer_overrides"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    common = inherited_commands[commands["inherited_common_trainer_overrides"]["key"]]
    specific_key = "I" if interface_id == "I" else "actor_checkpoint"
    specific = commands["interface_overrides"][specific_key]
    step = _step(interface_id)
    interface_root = Path(manifest["paths"][interface_id])
    replacements = {
        "${VALIDATION_PATH}": str(inherited["data"]["validation"]),
        "${MODEL_PATH}": str(inherited["model"]["path"]),
        "${REPO_DIR}": str(repo),
        "${EXPERIMENT_NAME}": _experiment_name(interface_id),
        "${INTERFACE_ID}": interface_id,
        "${ATTEMPT_ID}": _attempt_id(manifest, interface_id),
        "${INTERFACE_ROOT}": str(interface_root),
        "${TERMINAL_DIR}": str(interface_root / "terminal"),
        "${RESOLVED_MANIFEST_PATH}": str(manifest["paths"]["resolved_manifest"]),
        "${EVAL_MANIFEST_HASH}": str(eval_manifest_hash),
        "${TURN_LEDGER_PATH}": str(interface_root / "trajectory_turns.jsonl"),
        "${EXECUTION_SUMMARY_PATH}": str(interface_root / "execution_summary.json"),
        "${EXPECTED_RUNTIME_CONFIG_SHA256}": str(expected_runtime_config_sha256),
        "${CHECKPOINT}": str(
            Path(manifest["training_source"]["checkpoint_root"])
            / f"global_step_{step}"
        ),
        "${GLOBAL_STEP}": str(step),
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


def freeze_trainer_configuration(
    manifest: Mapping[str, Any], inherited: Mapping[str, Any], *,
    repo: Path, eval_manifest_hash: str,
) -> dict[str, Any]:
    interfaces: dict[str, Any] = {}
    protocol_hashes: dict[str, str] = {}
    for interface_id in INTERFACES:
        placeholder = render_trainer_overrides(
            manifest, inherited, repo=repo, interface_id=interface_id,
            eval_manifest_hash=eval_manifest_hash,
            expected_runtime_config_sha256="0" * 64,
        )
        config = compose_resolved_trainer_config(repo, placeholder)
        resolved_sha = stable_eval_runtime_config_sha256(config)
        final = render_trainer_overrides(
            manifest, inherited, repo=repo, interface_id=interface_id,
            eval_manifest_hash=eval_manifest_hash,
            expected_runtime_config_sha256=resolved_sha,
        )
        final_config = compose_resolved_trainer_config(repo, final)
        if stable_eval_runtime_config_sha256(final_config) != resolved_sha:
            raise ValueError(f"self-hashed Hydra config is unstable for {interface_id}")
        protocol_sha = canonical_sha256(
            repository_neutral_generation_protocol_projection(final_config, repo=repo)
        )
        protocol_hashes[interface_id] = protocol_sha
        interfaces[interface_id] = {
            "resolved_config_sha256": resolved_sha,
            "override_argv_sha256": canonical_sha256(final),
            "override_count": len(final),
            "generation_protocol_sha256": protocol_sha,
        }
    if len(set(protocol_hashes.values())) != 1:
        raise ValueError(f"Original curve generation protocols differ: {protocol_hashes}")
    return {
        "hydra_config_name": "ppo_trainer",
        "hydra_config_dir": "verl/trainer/config",
        "generation_protocol_projection": (
            "repository-relative-reward-code-sha256-v1"
        ),
        "interfaces": interfaces,
        "shared_generation_protocol_sha256": next(iter(protocol_hashes.values())),
    }


def repository_neutral_generation_protocol_projection(
    config: Mapping[str, Any], *, repo: Path
) -> dict[str, Any]:
    """Bind generation semantics without binding an absolute checkout location.

    The inherited S128 I/T25 run and this curve may execute the same committed
    reward implementation from different Git worktree paths.  An absolute
    checkout prefix is provenance, not a generation/scoring choice.  We remove
    only that prefix, while retaining both the repository-relative module path
    and its content digest.  A reward module outside ``repo`` fails closed.
    """
    projection = generation_protocol_projection(config)
    reward = dict(projection["custom_reward_function"])
    reward_path = Path(str(reward.get("path", ""))).resolve()
    repo_root = repo.resolve()
    try:
        relative = reward_path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(
            f"custom reward path is outside the bound repository: {reward_path}"
        ) from error
    if not reward_path.is_file():
        raise ValueError(f"custom reward implementation is missing: {reward_path}")
    reward["path"] = relative.as_posix()
    reward["path_sha256"] = sha256_file(reward_path)
    projection["custom_reward_function"] = reward
    return projection


def _validate_prior_protocol_attestation(
    trainer_configuration: Mapping[str, Any], *, expected_interfaces: tuple[str, ...]
) -> str:
    """Validate the legacy path-bound protocol hashes as an internal attestation."""
    shared = str(trainer_configuration.get("shared_generation_protocol_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", shared) is None:
        raise ValueError("prior shared generation protocol digest is malformed")
    interfaces = trainer_configuration.get("interfaces")
    if not isinstance(interfaces, Mapping) or set(interfaces) != set(expected_interfaces):
        raise ValueError("prior generation protocol interface set changed")
    for interface_id in expected_interfaces:
        item = interfaces[interface_id]
        if (
            not isinstance(item, Mapping)
            or item.get("generation_protocol_sha256") != shared
        ):
            raise ValueError(
                f"prior {interface_id} generation protocol does not match shared digest"
            )
    return shared


def _frozen_git_blob_sha256(repo: Path, *, commit: str, relative_path: str) -> str:
    """Hash one exact committed blob without consulting worktree bytes."""
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative_path:
        raise ValueError(f"frozen Git blob path is not canonical relative POSIX: {relative_path}")
    try:
        blob = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:{relative_path}"]
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"cannot read frozen Git blob {commit}:{relative_path}"
        ) from error
    return hashlib.sha256(blob).hexdigest()


def freeze_expected_prior_generation_protocol(
    inherited: Mapping[str, Any], *, repo: Path, eval_manifest_hash: str
) -> dict[str, Any]:
    """Recompose the immutable b7bf I/T25 protocol in this checkout.

    The inherited manifest and command manifest are content-hash checked before
    this function is called.  Recomposing them in the curve checkout makes the
    semantic comparison independent of the old checkout's absolute path.
    """
    hashes: dict[str, str] = {}
    for interface_id in ("I", "T25"):
        overrides = render_prior_s128_trainer_overrides(
            inherited,
            repo=repo,
            interface_id=interface_id,
            eval_manifest_hash=eval_manifest_hash,
            expected_runtime_config_sha256="0" * 64,
        )
        config = compose_resolved_trainer_config(repo, overrides)
        projection = repository_neutral_generation_protocol_projection(
            config, repo=repo
        )
        reward = projection["custom_reward_function"]
        reward["path_sha256"] = _frozen_git_blob_sha256(
            repo, commit=SOURCE_COMMIT, relative_path=reward["path"]
        )
        hashes[interface_id] = canonical_sha256(projection)
    if len(set(hashes.values())) != 1:
        raise ValueError(f"reconstructed prior I/T25 protocols differ: {hashes}")
    return {
        "projection": "repository-relative-reward-code-sha256-v1",
        "interfaces": hashes,
        "shared_generation_protocol_sha256": next(iter(hashes.values())),
    }


def _model_shards(inventory: list[dict[str, Any]], *, step: int) -> list[dict[str, Any]]:
    expected = {
        f"actor/model_world_size_2_rank_{rank}.pt" for rank in (0, 1)
    }
    actual = {
        item["path"] for item in inventory
        if re.fullmatch(r"actor/model_world_size_\d+_rank_\d+\.pt", item["path"])
    }
    if actual != expected:
        raise ValueError(
            f"Original step {step} actor shard inventory is not exact world-size 2: "
            f"{sorted(actual)}"
        )
    return sorted(
        (dict(item) for item in inventory if item["path"] in expected),
        key=lambda item: item["path"],
    )


def _json_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_symlinks(path: Path, *, label: str) -> None:
    """Refuse mutable indirection in every artifact frozen by P0."""
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if path.is_dir():
        linked = [str(item) for item in path.rglob("*") if item.is_symlink()]
        if linked:
            raise ValueError(f"{label} contains symlinks: {linked}")


def _complete_checkpoint_inventory(
    inventory: list[dict[str, Any]], *, step: int
) -> None:
    expected = {"data.pt"}
    for component in ("model", "optim", "extra_state"):
        expected.update(
            f"actor/{component}_world_size_2_rank_{rank}.pt" for rank in (0, 1)
        )
    actual = {str(item.get("path")) for item in inventory}
    malformed = any(
        not isinstance(item, Mapping)
        or not _json_integer(item.get("size"))
        or item.get("size", 0) <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None
        for item in inventory
    )
    if len(inventory) != 7 or actual != expected or malformed:
        raise ValueError(
            f"Original step {step} is not an exact complete 7-file world-size-2 "
            f"checkpoint: actual={sorted(actual)}, expected={sorted(expected)}"
        )


def _training_sync_contract(
    records: list[dict[str, Any]], *, step: int, report_digest: str,
    sync_kind: str = "post_actor_update",
) -> dict[str, str]:
    if not _json_integer(step):
        raise ValueError(f"Original training sync step is not a JSON integer: {step!r}")
    acks = [
        row for row in records
        if row.get("record_type") == "weight_sync_ack"
        and row.get("sync_kind") == sync_kind
        and row.get("actor_version") == step
        and row.get("global_step") == step
    ]
    if any(not _json_integer(row.get("vllm_worker_rank")) for row in acks):
        raise ValueError(f"Original step {step} training sync rank is not a JSON integer")
    ranks = sorted(row["vllm_worker_rank"] for row in acks)
    if len(acks) != 2 or ranks != [0, 1]:
        raise ValueError(
            f"Original step {step} training sync lacks exact worker-rank acks 0,1"
        )
    effective: set[str] = set()
    master: set[str] = set()
    loaded_names: set[str] = set()
    for ack in acks:
        effective_digest = ack.get("actor_rollout_sampled_tensor_digest")
        master_digest = ack.get("actor_master_sampled_tensor_digest")
        names_digest = ack.get("loaded_parameter_names_sha256")
        optimizer_state_count = ack.get("optimizer_state_entry_count")
        optimizer_step_count = ack.get("optimizer_step_entry_count")
        optimizer_histogram = ack.get("optimizer_step_histogram")
        audited_parameters = ack.get("audited_loaded_parameters")
        sampled_dtypes = ack.get("sampled_parameter_dtypes")
        if (
            not all(
                _json_integer(ack.get(field))
                for field in (
                    "global_step", "actor_version", "vllm_worker_rank",
                    "vllm_ack_version", "loaded_parameter_count",
                    "model_parameter_count", "optimizer_step_max",
                    "lr_scheduler_last_epoch", "optimizer_state_entry_count",
                    "optimizer_step_entry_count",
                )
            )
            or not isinstance(effective_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", effective_digest) is None
            or ack.get("actor_sampled_tensor_digest") != effective_digest
            or ack.get("vllm_sampled_tensor_digest") != effective_digest
            or ack.get("vllm_ack_version") != step
            or ack.get("weight_transfer_format") != "dtensor"
            or ack.get("loaded_parameter_count") != 199
            or ack.get("model_parameter_count") != 199
            or names_digest != ack.get("model_parameter_names_sha256")
            or re.fullmatch(r"[0-9a-f]{64}", str(names_digest or "")) is None
            or ack.get("optimizer_step_max") != step
            or ack.get("lr_scheduler_last_epoch") != step
            or not isinstance(optimizer_state_count, int)
            or isinstance(optimizer_state_count, bool)
            or optimizer_state_count < 1
            or optimizer_step_count != optimizer_state_count
            or not isinstance(optimizer_histogram, Mapping)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in optimizer_histogram.values()
            )
            or sum(
                value for value in optimizer_histogram.values()
                if isinstance(value, int) and not isinstance(value, bool)
            ) != optimizer_step_count
            or not isinstance(audited_parameters, list)
            or len(audited_parameters) != 8
            or not isinstance(sampled_dtypes, Mapping)
            or set(sampled_dtypes) != set(audited_parameters)
            or set(sampled_dtypes.values()) != {"torch.bfloat16"}
        ):
            raise ValueError(f"Original step {step} training sync ack is incomplete: {ack}")
        if re.fullmatch(r"[0-9a-f]{64}", str(master_digest or "")) is None:
            raise ValueError(f"Original step {step} training master digest is invalid")
        effective.add(effective_digest)
        master.add(str(master_digest))
        loaded_names.add(str(names_digest))
    summaries = [
        row for row in records
        if row.get("record_type") == "weight_sync_summary"
        and row.get("sync_kind") == sync_kind
        and row.get("actor_version") == step
        and row.get("global_step") == step
    ]
    if summaries and (
        not _json_integer(summaries[0].get("global_step"))
        or not _json_integer(summaries[0].get("actor_version"))
        or not isinstance(summaries[0].get("worker_ranks"), list)
        or any(
            not _json_integer(rank) for rank in summaries[0].get("worker_ranks", [])
        )
    ):
        raise ValueError(
            f"Original step {step} training sync summary has non-integer protocol fields"
        )
    if (
        len(effective) != 1
        or len(master) != 1
        or len(loaded_names) != 1
        or len(summaries) != 1
        or summaries[0].get("worker_ranks") != [0, 1]
        or summaries[0].get("sampled_tensor_digest") not in effective
        or summaries[0].get("actor_master_sampled_tensor_digest") not in master
        or report_digest not in effective
    ):
        raise ValueError(
            f"Original step {step} training acks/summary/report digests do not close"
        )
    return {
        "effective_actor_vllm_digest": next(iter(effective)),
        "actor_master_digest": next(iter(master)),
        "loaded_parameter_names_sha256": next(iter(loaded_names)),
    }


def _training_prefix_order_contract(
    records: list[dict[str, Any]], *, version_digests: Mapping[str, Any]
) -> None:
    expected_records = 116
    if len(records) != expected_records:
        raise ValueError(
            f"Original training ledger semantic prefix has {len(records)} records, "
            f"expected {expected_records}"
        )
    if [row.get("record_type") for row in records[:3]] != [
        "p0_preflight", "runtime_config", "resume_load"
    ]:
        raise ValueError("Original training ledger does not start P0/runtime/resume")
    cursor = 3
    for version in range(3, 26):
        if version > 3:
            if cursor >= len(records):
                raise ValueError(
                    f"Original training ledger ends before rollout at step {version}"
                )
            rollout = records[cursor]
            if (
                rollout.get("record_type") != "rollout_start"
                or not _json_integer(rollout.get("global_step"))
                or not _json_integer(rollout.get("actor_version"))
                or rollout.get("global_step") != version
                or rollout.get("actor_version") != version - 1
                or rollout.get("sampled_tensor_digest")
                != version_digests.get(str(version - 1))
            ):
                raise ValueError(
                    f"Original training ledger rollout order/binding failed at step {version}"
                )
            cursor += 1
        if cursor + 2 >= len(records):
            raise ValueError(
                f"Original training ledger ends inside sync group at version {version}"
            )
        acknowledgements = records[cursor : cursor + 2]
        summary = records[cursor + 2] if cursor + 2 < len(records) else {}
        kind = "resume_loaded" if version == 3 else "post_actor_update"
        if (
            len(acknowledgements) != 2
            or any(row.get("record_type") != "weight_sync_ack" for row in acknowledgements)
            or summary.get("record_type") != "weight_sync_summary"
            or any(row.get("sync_kind") != kind for row in acknowledgements)
            or summary.get("sync_kind") != kind
            or any(row.get("actor_version") != version for row in acknowledgements)
            or summary.get("actor_version") != version
        ):
            raise ValueError(
                f"Original training ledger sync group order failed at version {version}"
            )
        cursor += 3
        if version > 3:
            if cursor >= len(records):
                raise ValueError(
                    f"Original training ledger ends before execution signal at step {version}"
                )
            signal = records[cursor]
            if (
                signal.get("record_type") != "execution_signal"
                or not _json_integer(signal.get("global_step"))
                or not _json_integer(signal.get("actor_version"))
                or signal.get("global_step") != version
                or signal.get("actor_version") != version
                or not isinstance(signal.get("metrics"), Mapping)
                or signal.get("nonfinite_metric_names") != []
            ):
                raise ValueError(
                    f"Original training ledger execution signal failed at step {version}"
                )
            cursor += 1
    if cursor != len(records):
        raise ValueError(
            f"Original training ledger semantic prefix has trailing records: "
            f"cursor={cursor}, records={len(records)}"
        )


def validate_training_source(
    manifest: Mapping[str, Any], stable_canary: Mapping[str, Any]
) -> dict[str, Any]:
    source = manifest["training_source"]
    report_path = Path(source["final_report"])
    ledger_path = Path(source["execution_ledger"])
    if not report_path.is_file() or not ledger_path.is_file():
        raise ValueError("completed corrected Original training report/ledger is missing")
    _reject_symlinks(report_path, label="Original training final report")
    _reject_symlinks(ledger_path, label="Original training execution ledger")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != source["required_status"]
        or report.get("decision") != source["required_decision"]
        or report.get("failures") != []
        or report.get("git_commit") != source["required_git_commit"]
        or report.get("experiment_name") != source["experiment_name"]
        or report.get("not_original_paper_7b_reproduction") is not True
    ):
        raise ValueError("corrected Original training final report is not the frozen b7bf PASS")
    stable = report.get("stable_identity_prerequisite")
    if not isinstance(stable, Mapping) or (
        stable.get("report_sha256") != stable_canary["sha256"]
        or stable.get("eval_manifest_hash") != EXPECTED_EVAL_HASH
        or stable.get("commit") != stable_canary["git_commit"]
    ):
        raise ValueError("Original training report does not authenticate stable-I r2")
    report_sha = sha256_file(report_path)
    records = [
        json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chain_failures = validate_jsonl_chain(records)
    if chain_failures:
        raise ValueError(f"Original training ledger hash chain failed: {chain_failures}")
    expected_counts = Counter({
        "p0_preflight": 1,
        "runtime_config": 1,
        "resume_load": 1,
        "rollout_start": 22,
        "execution_signal": 22,
        "weight_sync_ack": 46,
        "weight_sync_summary": 23,
        "checkpoint_inventory": 1,
        "audit_result": 1,
    })
    actual_counts = Counter(row.get("record_type") for row in records)
    if len(records) != 118 or actual_counts != expected_counts:
        raise ValueError(
            "Original training ledger is not the exact 116-record training prefix "
            f"plus audit suffix: {dict(actual_counts)}"
        )
    if len(records) < 2 or [row.get("record_type") for row in records[-2:]] != [
        "checkpoint_inventory", "audit_result",
    ]:
        raise ValueError("Original training ledger lacks final checkpoint/audit suffix")
    if any(
        row.get("git_commit") != SOURCE_COMMIT
        or row.get("experiment_name") != source["experiment_name"]
        for row in records
    ):
        raise ValueError("Original training ledger commit/experiment identity changed")
    run_ids = {row.get("run_id") for row in records}
    if len(run_ids) != 1 or re.fullmatch(r"[0-9a-f]{32}", str(next(iter(run_ids), ""))) is None:
        raise ValueError("Original training ledger run_id is not one exact 32-hex identity")
    tail = records[-1]
    if (
        tail.get("status") != "PASS"
        or tail.get("decision") != "ORIGINAL_T25_PASS"
        or Path(str(tail.get("report", ""))).resolve() != report_path.resolve()
        or tail.get("report_sha256") != report_sha
    ):
        raise ValueError("Original training ledger tail does not authenticate final report")
    training_records = records[:-2]
    report_ledger = report.get("execution_ledger")
    if not isinstance(report_ledger, Mapping) or (
        report_ledger.get("training_prefix_record_count") != len(training_records)
        or report_ledger.get("training_prefix_sha256")
        != canonical_sha256(training_records)
        or report_ledger.get("training_prefix_tail_sha256")
        != training_records[-1].get("record_sha256")
    ):
        raise ValueError("Original final report does not authenticate its ledger prefix")

    declared = report.get("checkpoint_anchors")
    if (
        not isinstance(declared, list)
        or any(
            not isinstance(item, Mapping)
            or not _json_integer(item.get("global_step"))
            for item in declared
        )
        or [item.get("global_step") for item in declared] != list(ANCHOR_STEPS)
    ):
        raise ValueError("Original report does not freeze checkpoints 5/10/15/20/25")
    if report.get("checkpoint_anchors_sha256") != canonical_sha256(declared):
        raise ValueError("Original checkpoint-anchor map SHA is invalid")
    step25 = report.get("step25_checkpoint")
    if not isinstance(step25, Mapping) or any(
        step25.get(key) != declared[-1].get(key)
        for key in ("path", "global_step", "inventory", "inventory_sha256")
    ):
        raise ValueError("Original step25 checkpoint summary differs from anchor map")
    if (
        records[-2].get("checkpoint_anchors") != declared
        or records[-2].get("checkpoint_anchors_sha256") != canonical_sha256(declared)
        or records[-2].get("global_step") != 25
        or records[-2].get("inventory") != declared[-1].get("inventory")
        or records[-2].get("inventory_sha256")
        != declared[-1].get("inventory_sha256")
    ):
        raise ValueError("Original ledger checkpoint-anchor map differs from report")
    version_digests = report.get("weight_sync", {}).get("version_digests", {})
    if not isinstance(version_digests, Mapping):
        raise ValueError("Original training report version digests are not a mapping")
    version_syncs: dict[int, dict[str, str]] = {}
    for version in range(3, 26):
        digest = version_digests.get(str(version))
        if re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None:
            raise ValueError(
                f"Original training report lacks version-{version} weight digest"
            )
        version_syncs[version] = _training_sync_contract(
            training_records,
            step=version,
            report_digest=str(digest),
            sync_kind="resume_loaded" if version == 3 else "post_actor_update",
        )
    _training_prefix_order_contract(
        training_records, version_digests=version_digests
    )
    root = Path(source["checkpoint_root"])
    _reject_symlinks(root, label="Original checkpoint root")
    anchors: dict[str, Any] = {}
    for item, step in zip(declared, ANCHOR_STEPS):
        path = root / f"global_step_{step}"
        if Path(str(item.get("path", ""))).resolve() != path.resolve():
            raise ValueError(f"Original step {step} checkpoint path changed")
        current = checkpoint_inventory(path) if path.is_dir() else []
        if current != item.get("inventory"):
            raise ValueError(f"Original step {step} checkpoint inventory changed")
        if item.get("inventory_sha256") != canonical_sha256(current):
            raise ValueError(f"Original step {step} checkpoint inventory SHA changed")
        _reject_symlinks(path, label=f"Original step {step} checkpoint")
        _complete_checkpoint_inventory(current, step=step)
        digest = version_digests.get(str(step))
        if re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None:
            raise ValueError(f"Original training report lacks version-{step} weight digest")
        actor_shards = _model_shards(current, step=step)
        sync = version_syncs[step]
        anchors[f"Original{step}"] = {
            "path": str(path.resolve()),
            "global_step": step,
            "inventory": current,
            "inventory_sha256": canonical_sha256(current),
            "actor_model_shards": actor_shards,
            "actor_model_shards_sha256": canonical_sha256(actor_shards),
            "training_effective_actor_vllm_digest": sync[
                "effective_actor_vllm_digest"
            ],
            "training_actor_master_sampled_tensor_digest": sync[
                "actor_master_digest"
            ],
            "training_loaded_parameter_names_sha256": sync[
                "loaded_parameter_names_sha256"
            ],
            "fsdp_world_size": 2,
            "load_mode": "actor_only",
        }
    return {
        "report": str(report_path.resolve()),
        "report_sha256": report_sha,
        "ledger": str(ledger_path.resolve()),
        "ledger_sha256": sha256_file(ledger_path),
        "ledger_tail_sha256": tail["record_sha256"],
        "git_commit": SOURCE_COMMIT,
        "experiment_name": source["experiment_name"],
        "checkpoint_anchors_sha256": canonical_sha256(declared),
        "anchors": anchors,
    }


def _artifact_inventory(root: Path, *, step: int) -> dict[str, dict[str, Any]]:
    _reject_symlinks(root, label="interface evidence root")
    paths = (
        root / f"terminal/{step}.jsonl",
        root / "trajectory_turns.jsonl",
        root / "execution_summary.json",
        root / "run.log",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"interface evidence is incomplete: {missing}")
    return {
        str(path.relative_to(root)): {
            "sha256": sha256_file(path), "size": path.stat().st_size,
        }
        for path in paths
    }


def validate_prior_import(
    manifest: Mapping[str, Any], *, expected_eval_manifest_hash: str
) -> dict[str, Any]:
    spec = manifest["prior_s128_it_import"]
    paths = [
        Path(spec[name])
        for name in ("p0_certificate", "resolved_manifest", "final_report", "execution_ledger")
    ]
    paths.extend(Path(item["root"]) for item in spec["interfaces"].values())
    existing = [path.exists() for path in paths]
    if not any(existing):
        raise ValueError(
            "completed prior I/T25 evidence is required for read-only import; "
            "refusing to expand this run to I/Original25"
        )
    if not all(existing):
        missing = [str(path) for path, present in zip(paths, existing) if not present]
        raise ValueError(f"prior I/T25 evidence is partially present; refusing fallback run: {missing}")
    p0_path, resolved_path, report_path, ledger_path = paths[:4]
    for label, path in (
        ("prior P0 certificate", p0_path),
        ("prior resolved manifest", resolved_path),
        ("prior final report", report_path),
        ("prior execution ledger", ledger_path),
    ):
        _reject_symlinks(path, label=label)
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    resolved = validate_resolved_manifest(
        json.loads(resolved_path.read_text(encoding="utf-8"))
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_execution = resolved.get("execution_binding", {})
    if (
        p0.get("status") != "PASS"
        or p0.get("decision") != "S128_IT_P0_PASS"
        or p0.get("evidence", {}).get("git_commit") != spec["source_commit"]
        or p0.get("evidence", {}).get("branch")
        != "h20/qwen25-7b-original-t25-s128-frozen-20260821"
        or p0.get("evidence", {}).get("resolved_manifest_sha256") != sha256_file(resolved_path)
        or resolved.get("eval_manifest_hash") != expected_eval_manifest_hash
        or source_execution.get("git_commit") != spec["source_commit"]
        or len(resolved.get("identity_payload", {}).get("rows", [])) != 128
        or [
            row.get("source_order_index")
            for row in resolved.get("identity_payload", {}).get("rows", [])
        ] != list(range(128))
        or resolved.get("cohort") != {
            "source_order_indices": list(range(128)),
            "examples": 128,
            "replicas": 1,
            "interfaces": ["I", "T25"],
        }
        or source_execution.get("interfaces") != ["I", "T25"]
        or source_execution.get("base_seed") != 2026
        or source_execution.get("replicas") != 1
        or source_execution.get("execution_code_sha256", {}).get(
            "recurrent/research/s128_hotpot_metrics.py"
        ) != EXPECTED_METRIC_CODE_SHA
    ):
        raise ValueError("prior I/T25 P0/resolved manifest is not the frozen b7bf contract")
    if (
        report.get("status") != "PASS"
        or report.get("decision") != "S128_IT_PERFORMANCE_PASS"
        or report.get("failures") != []
        or report.get("evidence", {}).get("eval_manifest_hash") != expected_eval_manifest_hash
    ):
        raise ValueError("prior I/T25 final report is not a clean compatible PASS")
    records = [
        json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = validate_jsonl_chain(records)
    sequence = [
        ("s0_preflight", None),
        ("interface_start", "I"), ("interface_finish", "I"),
        ("interface_start", "T25"), ("interface_finish", "T25"),
        ("audit_result", None),
    ]
    if failures or [
        (row.get("record_type"), row.get("interface_id")) for row in records
    ] != sequence:
        raise ValueError(f"prior I/T25 ledger failed chain/sequence: {failures}")
    execution_sha = canonical_sha256(resolved["execution_binding"])
    runtime_sha = p0.get("evidence", {}).get("runtime_binding_sha256")
    run_id = p0.get("evidence", {}).get("run_id")
    expected_experiments = (
        "qwen25_7b_s128_it_p0_seed2026_20260821",
        "qwen25_7b_s128_i_base_seed2026_20260821",
        "qwen25_7b_s128_i_base_seed2026_20260821",
        "qwen25_7b_s128_t25_corrected_original_style_seed2026_20260821",
        "qwen25_7b_s128_t25_corrected_original_style_seed2026_20260821",
        "qwen25_7b_s128_it_audit_seed2026_20260821",
    )
    for index, row in enumerate(records):
        if (
            row.get("git_commit") != spec["source_commit"]
            or row.get("run_id") != run_id
            or row.get("eval_manifest_hash") != expected_eval_manifest_hash
            or row.get("execution_binding_sha256") != execution_sha
            or row.get("runtime_binding_sha256") != runtime_sha
            or row.get("experiment_name") != expected_experiments[index]
            or row.get("status") != "PASS"
        ):
            raise ValueError(f"prior I/T25 ledger record {index} identity changed")
    first = records[0]
    if (
        first.get("row_count") != 128
        or Path(str(first.get("artifact", ""))).resolve() != p0_path.resolve()
        or first.get("artifact_sha256") != sha256_file(p0_path)
    ):
        raise ValueError("prior I/T25 ledger does not authenticate its P0")
    raw_lines = [
        line for line in ledger_path.read_bytes().splitlines(keepends=True) if line.strip()
    ]
    prefix_sha = hashlib.sha256(b"".join(raw_lines[:5])).hexdigest()
    if (
        report.get("evidence", {}).get("execution_ledger_records") != 5
        or report.get("evidence", {}).get("execution_ledger_sha256") != prefix_sha
    ):
        raise ValueError("prior I/T25 report does not authenticate its five-record ledger prefix")
    report_sha = sha256_file(report_path)
    if (
        Path(str(records[-1].get("artifact", ""))).resolve() != report_path.resolve()
        or records[-1].get("artifact_sha256") != report_sha
        or records[-1].get("status") != "PASS"
        or records[-1].get("decision") != "S128_IT_PERFORMANCE_PASS"
    ):
        raise ValueError("prior I/T25 ledger tail does not authenticate final report")
    interface_evidence: dict[str, Any] = {}
    for target, finish_index in (("I", 2), ("Original25", 4)):
        source_interface = spec["interfaces"][target]["source_interface"]
        source_root = Path(spec["interfaces"][target]["root"])
        step = int(spec["interfaces"][target]["global_step"])
        artifacts = _artifact_inventory(source_root, step=step)
        start = records[finish_index - 1]
        finish = records[finish_index]
        if (
            start.get("interface_id") != source_interface
            or start.get("artifacts") != {}
            or finish.get("interface_id") != source_interface
            or finish.get("artifacts") != artifacts
        ):
            raise ValueError(f"prior {source_interface} artifact hashes differ from ledger")
        interface_evidence[target] = {
            "mode": "import",
            "source_interface": source_interface,
            "source_attempt": spec["interfaces"][target]["source_attempt"],
            "global_step": step,
            "root": str(source_root.resolve()),
            "artifacts": artifacts,
        }
    return {
        "available": True,
        "source_commit": spec["source_commit"],
        "p0_certificate": str(p0_path.resolve()),
        "p0_certificate_sha256": sha256_file(p0_path),
        "resolved_manifest": str(resolved_path.resolve()),
        "resolved_manifest_sha256": sha256_file(resolved_path),
        "resolved": resolved,
        "final_report": str(report_path.resolve()),
        "final_report_sha256": report_sha,
        "execution_ledger": str(ledger_path.resolve()),
        "execution_ledger_sha256": sha256_file(ledger_path),
        "execution_ledger_tail_sha256": records[-1]["record_sha256"],
        "interfaces": interface_evidence,
    }


def build_interface_plan(
    manifest: Mapping[str, Any], prior: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if not bool(prior.get("available")):
        raise ValueError("complete authenticated prior I/T25 import is mandatory")
    plan: dict[str, dict[str, Any]] = {}
    for interface_id in INTERFACES:
        if interface_id in prior["interfaces"]:
            plan[interface_id] = dict(prior["interfaces"][interface_id])
        else:
            plan[interface_id] = {
                "mode": "run",
                "source_interface": interface_id,
                "source_attempt": _attempt_id(manifest, interface_id),
                "global_step": _step(interface_id),
                "root": str(Path(manifest["paths"][interface_id]).resolve()),
            }
    return plan


def build_execution_binding(
    manifest: Mapping[str, Any], inherited: Mapping[str, Any], *, repo: Path,
    rows: list[Mapping[str, Any]], stable_canary: Mapping[str, Any],
    training: Mapping[str, Any], prior: Mapping[str, Any],
    trainer_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    chunk_size = int(inherited["recurrent"]["chunk_size"])
    active = {
        str(row["source_order_index"]):
            (int(row["context_token_count"]) + chunk_size - 1) // chunk_size
        for row in rows
    }
    code = {path: sha256_file(repo / path) for path in EXECUTION_CODE_OBJECTS}
    model_artifacts: dict[str, Any] = {
        "I": {
            "kind": "frozen_huggingface_base_model",
            "file_manifest_sha256": canonical_sha256(inherited["model"]["files"]),
        }
    }
    model_artifacts.update(training["anchors"])
    return {
        "git_commit": git(repo, "rev-parse", "HEAD"),
        "base_commit": SOURCE_COMMIT,
        "interfaces": list(INTERFACES),
        "base_seed": int(manifest["evaluation"]["base_seed"]),
        "replicas": 1,
        "model_artifacts": model_artifacts,
        "stable_identity_canary_prerequisite": dict(stable_canary),
        "training_source": {key: value for key, value in training.items() if key != "anchors"},
        "prior_s128_it_import": {
            key: value for key, value in prior.items() if key != "resolved"
        },
        "interface_plan": build_interface_plan(manifest, prior),
        "recurrent": dict(inherited["recurrent"]),
        "all_s128_turn_schedule": {
            "active_turn_count_by_source_order": active,
            "shared_final_turn": max(active.values()),
        },
        "trainer_configuration": dict(trainer_configuration),
        "execution_code_sha256": code,
        "execution_code_combined_sha256": canonical_sha256(code),
    }


def capture_runtime_binding(
    manifest: Mapping[str, Any], inherited: Mapping[str, Any], *, repo: Path,
    stable_canary: Mapping[str, Any], training: Mapping[str, Any],
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    model_root = Path(inherited["model"]["path"])
    model_files = [
        {
            "path": item["path"], "size": (model_root / item["path"]).stat().st_size,
            "sha256": sha256_file(model_root / item["path"]),
        }
        for item in inherited["model"]["files"]
    ]
    validation = Path(inherited["data"]["validation"])
    versions, error = _runtime_versions(str(manifest["python"]), repo)
    if error or versions is None:
        raise ValueError(f"runtime import failed: {error}")
    gpu = subprocess.run(
        ["nvidia-smi", "-i", "6,7", "--query-gpu=index,uuid,name",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if gpu.returncode:
        raise ValueError(f"cannot identify GPU6-7: {gpu.stderr.strip()}")
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
        "stable_i_canary_report_sha256": stable_canary["sha256"],
        "stable_i_canary_ledger_sha256": stable_canary["execution_ledger_sha256"],
        "training_report_sha256": training["report_sha256"],
        "training_ledger_sha256": training["ledger_sha256"],
        "checkpoint_anchor_inventory_sha256": {
            name: item["inventory_sha256"] for name, item in training["anchors"].items()
        },
        "prior_import_evidence_sha256": canonical_sha256(
            {key: value for key, value in prior.items() if key != "resolved"}
        ),
    }


def _validate_science_contract(manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if manifest.get("base_commit") != SOURCE_COMMIT:
        failures.append("manifest base commit is not b7bf649")
    if manifest.get("branch") != EXPECTED_BRANCH:
        failures.append("manifest branch changed")
    if manifest.get("evaluation", {}).get("interfaces") != list(INTERFACES):
        failures.append("evaluation interfaces are not the complete preregistered curve")
    if manifest.get("training_source", {}).get("anchor_steps") != list(ANCHOR_STEPS):
        failures.append("checkpoint anchors are not exact 5/10/15/20/25")
    scope = manifest.get("scope", {})
    for field in (
        "existing_s128_only", "no_resampling", "no_training", "raw_context_r_not_rerun",
        "same_prompt_recurrent_and_decode_protocol",
        "paired_differences_are_descriptive_not_causal",
        "not_original_paper_7b_reproduction",
        "published_results_are_historical_reference_only",
    ):
        if scope.get(field) is not True:
            failures.append(f"scientific scope field is not true: {field}")
    evaluation = manifest.get("evaluation", {})
    expected = {
        "examples": 128, "replicas": 1, "validation_only": True,
        "actor_update_calls": 0, "optimizer_step_calls": 0,
        "checkpoint_save_calls": 0, "do_sample": False,
        "temperature": 0.0, "top_p": 1.0, "top_k": -1,
    }
    for field, value in expected.items():
        if evaluation.get(field) != value:
            failures.append(f"evaluation field {field} changed")
    if evaluation.get("primary_metrics") != ["normalized_exact_match", "token_f1"]:
        failures.append("primary metrics are not normalized EM/F1")
    if evaluation.get("training_dense_reward_excluded_from_evaluation_claims") is not True:
        failures.append("dense training reward is not excluded from performance")
    return failures


def run_preflight(
    manifest_path: Path, *, check_runtime: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    manifest_path = manifest_path.resolve()
    repo = manifest_path.parents[2]
    manifest = load_manifest(manifest_path)
    failures = _validate_science_contract(manifest)
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
    if branch != EXPECTED_BRANCH:
        failures.append(f"branch mismatch: {branch} != {EXPECTED_BRANCH}")
    if status:
        failures.append(f"worktree is dirty: {status.splitlines()}")
    if Path(manifest["repository"]).resolve() != repo.resolve():
        failures.append("manifest repository differs from invoked checkout")
    if subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
        check=False,
    ).returncode:
        failures.append("HEAD does not contain the frozen b7bf Original/S128 closure")
    missing = [path for path in REQUIRED_GIT_OBJECTS if not (repo / path).is_file()]
    if missing:
        failures.append(f"required Git objects are missing: {missing}")
    untracked = [
        path for path in REQUIRED_GIT_OBJECTS
        if subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode
    ]
    if untracked:
        failures.append(f"required Git objects are not committed: {untracked}")
    append_only_paths = [
        Path(manifest["paths"][name])
        for name in ("p0_certificate", "resolved_manifest", "final_report", "execution_ledger", *INTERFACES)
    ]
    existing_new = [str(path) for path in append_only_paths if path.exists()]
    if existing_new:
        failures.append(f"new append-only curve evidence already exists: {existing_new}")

    inherited = None
    stable_canary = None
    training = None
    prior = None
    resolved = None
    try:
        inherited, _ = _load_inherited_contract(manifest, repo)
    except Exception as error:
        failures.append(f"cannot load inherited b7bf S128 contract: {error}")
    try:
        stable_canary = _stable_canary_contract(manifest)
        if stable_canary["eval_manifest_hash"] != EXPECTED_EVAL_HASH:
            raise ValueError("stable-I r2 evaluation hash changed")
        evidence["stable_identity_canary"] = stable_canary
    except Exception as error:
        failures.append(f"stable-I prerequisite failed: {error}")
    if stable_canary is not None:
        try:
            training = validate_training_source(manifest, stable_canary)
            evidence["training_source"] = training
        except Exception as error:
            failures.append(f"Original checkpoint provenance failed: {error}")

    if inherited is not None:
        data = inherited["data"]
        model = inherited["model"]
        validation = Path(data["validation"])
        model_root = Path(model["path"])
        for path in (Path(manifest["python"]), validation, model_root):
            if not path.exists():
                failures.append(f"required runtime path is missing: {path}")
        if validation.is_file() and sha256_file(validation) != data["validation_sha256"]:
            failures.append("fixed HotpotQA S128 parquet changed")
        if model_root.is_dir():
            for item in model["files"]:
                path = model_root / item["path"]
                if not path.is_file() or {
                    "path": item["path"], "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                } != item:
                    failures.append(f"base-model file differs: {path}")
            if model_loading_relevant_paths(model_root) != sorted(
                item["path"] for item in model["files"]
            ):
                failures.append("base-model loading-relevant inventory is not exact")

        if validation.is_file() and model_root.is_dir() and stable_canary is not None and training is not None:
            try:
                from verl.utils import hf_tokenizer

                tokenizer = hf_tokenizer(str(model_root), trust_remote_code=False)
                rows, filter_evidence = freeze_existing_s128_rows(
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
                failures.extend(validate_s128_freeze(rows, filter_evidence, data))
                if [row["source_order_index"] for row in rows] != list(range(128)):
                    failures.append("S128 source order is not exact 0..127")
                payload = build_identity_payload(inherited, rows=rows)
                eval_hash = canonical_sha256(payload)
                if eval_hash != EXPECTED_EVAL_HASH:
                    raise ValueError(f"S128 identity hash changed: {eval_hash}")
                prior = validate_prior_import(
                    manifest, expected_eval_manifest_hash=eval_hash
                )
                trainer = freeze_trainer_configuration(
                    manifest, inherited, repo=repo, eval_manifest_hash=eval_hash
                )
                if prior.get("available"):
                    prior_trainer = prior["resolved"]["execution_binding"][
                        "trainer_configuration"
                    ]
                    _validate_prior_protocol_attestation(
                        prior_trainer, expected_interfaces=("I", "T25")
                    )
                    expected_prior_protocol = freeze_expected_prior_generation_protocol(
                        inherited, repo=repo, eval_manifest_hash=eval_hash
                    )
                    if (
                        expected_prior_protocol["shared_generation_protocol_sha256"]
                        != trainer["shared_generation_protocol_sha256"]
                    ):
                        raise ValueError("imported I/T25 generation protocol differs from curve")
                execution = build_execution_binding(
                    manifest, inherited, repo=repo, rows=rows,
                    stable_canary=stable_canary, training=training, prior=prior,
                    trainer_configuration=trainer,
                )
                resolved = {
                    "schema_version": 1,
                    "frozen_manifest_sha256": evidence["frozen_manifest_sha256"],
                    "identity_payload": payload,
                    "eval_manifest_hash": eval_hash,
                    "cohort": {
                        "source_order_indices": list(range(128)),
                        "examples": 128, "replicas": 1,
                        "interfaces": list(INTERFACES),
                    },
                    "execution_binding": execution,
                }
                validate_resolved_manifest(resolved)
                evidence.update(
                    eval_manifest_hash=eval_hash,
                    execution_binding_sha256=canonical_sha256(execution),
                    interface_plan=execution["interface_plan"],
                    s128_filter_replay=filter_evidence,
                )
            except Exception as error:
                failures.append(f"cannot freeze complete S128 curve: {error}")

    if check_runtime and all(
        item is not None for item in (inherited, stable_canary, training, prior)
    ):
        try:
            runtime = capture_runtime_binding(
                manifest, inherited, repo=repo, stable_canary=stable_canary,
                training=training, prior=prior,
            )
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
        "decision": "ORIGINAL_S128_CURVE_P0_PASS" if not failures else "ORIGINAL_S128_CURVE_NO_GO:P0",
        "failures": failures,
        "evidence": evidence,
    }, resolved)


def _current_certificate(
    manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    p0_path = Path(manifest["paths"]["p0_certificate"])
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    if not p0_path.is_file() or not resolved_path.is_file():
        raise ValueError("standalone curve P0 certificate/resolved manifest are required")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    resolved = validate_resolved_manifest(
        json.loads(resolved_path.read_text(encoding="utf-8"))
    )
    expected = os.environ[ENVIRONMENT_NAMES[2]]
    evidence = p0.get("evidence", {})
    if (
        p0.get("status") != "PASS"
        or p0.get("decision") != "ORIGINAL_S128_CURVE_P0_PASS"
        or p0.get("failures") != []
        or evidence.get("git_commit") != expected
        or evidence.get("branch") != EXPECTED_BRANCH
        or re.fullmatch(r"[0-9a-f]{32}", str(evidence.get("run_id", ""))) is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(evidence.get("runtime_binding_sha256", ""))
        ) is None
        or resolved.get("eval_manifest_hash") != EXPECTED_EVAL_HASH
        or resolved.get("cohort", {}).get("interfaces") != list(INTERFACES)
    ):
        raise ValueError("curve P0 is not PASS for the expected commit")
    if evidence.get("resolved_manifest_sha256") != sha256_file(resolved_path):
        raise ValueError("curve resolved manifest changed after P0")
    return p0, resolved


def frozen_trainer_overrides(
    manifest_path: Path, *, interface_id: str
) -> list[str]:
    manifest = load_manifest(manifest_path)
    p0, resolved = _current_certificate(manifest)
    del p0
    inherited, _ = _load_inherited_contract(manifest, Path(manifest["repository"]))
    binding = resolved["execution_binding"]["trainer_configuration"]["interfaces"].get(
        interface_id
    )
    if not isinstance(binding, Mapping):
        raise ValueError(f"P0 lacks trainer binding for {interface_id}")
    if resolved["execution_binding"]["interface_plan"][interface_id]["mode"] != "run":
        raise ValueError(f"interface {interface_id} is a verified import and must not run")
    overrides = render_trainer_overrides(
        manifest, inherited, repo=Path(manifest["repository"]),
        interface_id=interface_id, eval_manifest_hash=resolved["eval_manifest_hash"],
        expected_runtime_config_sha256=str(binding["resolved_config_sha256"]),
    )
    if canonical_sha256(overrides) != binding["override_argv_sha256"]:
        raise ValueError(f"trainer argv changed after P0 for {interface_id}")
    return overrides


def interface_mode(manifest_path: Path, *, interface_id: str) -> str:
    manifest = load_manifest(manifest_path)
    _, resolved = _current_certificate(manifest)
    return str(resolved["execution_binding"]["interface_plan"][interface_id]["mode"])


def _stat_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(path.relative_to(root)), "size": path.stat().st_size}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def verify_toctou(
    manifest: Mapping[str, Any], p0: Mapping[str, Any],
    resolved: Mapping[str, Any], *, interface_id: str,
) -> None:
    repo = Path(manifest["repository"])
    if (
        git(repo, "rev-parse", "HEAD") != p0["evidence"]["git_commit"]
        or git(repo, "branch", "--show-current") != EXPECTED_BRANCH
        or git(repo, "status", "--porcelain")
    ):
        raise ValueError("Git binding changed after curve P0")
    inherited, _ = _load_inherited_contract(manifest, repo)
    validation = Path(inherited["data"]["validation"])
    if sha256_file(validation) != p0["evidence"]["runtime_binding"]["validation_data_sha256"]:
        raise ValueError("S128 parquet changed after curve P0")
    model_root = Path(inherited["model"]["path"])
    frozen_model = p0["evidence"]["runtime_binding"]["model_file_inventory"]
    current_model_stat = [
        {"path": item["path"], "size": (model_root / item["path"]).stat().st_size}
        for item in inherited["model"]["files"]
    ]
    if current_model_stat != [
        {"path": item["path"], "size": item["size"]} for item in frozen_model
    ]:
        raise ValueError("base-model path/size inventory changed after curve P0")
    if interface_id == "I":
        for frozen in frozen_model:
            path = model_root / frozen["path"]
            if sha256_file(path) != frozen["sha256"]:
                raise ValueError(f"base-model file changed after curve P0: {path}")
    training = resolved["execution_binding"]["model_artifacts"]
    for anchor in (f"Original{step}" for step in ANCHOR_STEPS):
        frozen = training[anchor]
        root = Path(frozen["path"])
        if _stat_inventory(root) != [
            {"path": item["path"], "size": item["size"]}
            for item in frozen["inventory"]
        ]:
            raise ValueError(f"{anchor} checkpoint path/size inventory changed after P0")
    if interface_id != "I":
        frozen = training[interface_id]
        root = Path(frozen["path"])
        for shard in frozen["actor_model_shards"]:
            path = root / shard["path"]
            current = {
                "path": shard["path"], "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if current != shard:
                raise ValueError(f"{interface_id} actor shard changed after P0")
    plan = resolved["execution_binding"]["interface_plan"][interface_id]
    if plan["mode"] == "import":
        current = _artifact_inventory(Path(plan["root"]), step=int(plan["global_step"]))
        if current != plan["artifacts"]:
            raise ValueError(f"imported {interface_id} artifacts changed after P0")
    small_files = (
        (manifest["stable_identity_canary"]["final_report"],
         p0["evidence"]["runtime_binding"]["stable_i_canary_report_sha256"]),
        (manifest["stable_identity_canary"]["execution_ledger"],
         p0["evidence"]["runtime_binding"]["stable_i_canary_ledger_sha256"]),
        (manifest["training_source"]["final_report"],
         p0["evidence"]["runtime_binding"]["training_report_sha256"]),
        (manifest["training_source"]["execution_ledger"],
         p0["evidence"]["runtime_binding"]["training_ledger_sha256"]),
    )
    for path, expected in small_files:
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"prerequisite evidence changed after curve P0: {path}")
    prior = resolved["execution_binding"]["prior_s128_it_import"]
    if prior.get("available"):
        for path_key, digest_key in (
            ("p0_certificate", "p0_certificate_sha256"),
            ("resolved_manifest", "resolved_manifest_sha256"),
            ("final_report", "final_report_sha256"),
            ("execution_ledger", "execution_ledger_sha256"),
        ):
            if sha256_file(Path(prior[path_key])) != prior[digest_key]:
                raise ValueError(
                    f"prior I/T25 prerequisite changed after curve P0: {prior[path_key]}"
                )


def record_interface_event(
    manifest_path: Path, *, interface_id: str, event: str
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    p0, resolved = _current_certificate(manifest)
    plan = resolved["execution_binding"]["interface_plan"][interface_id]
    verify_toctou(manifest, p0, resolved, interface_id=interface_id)
    root = Path(plan["root"])
    step = int(plan["global_step"])
    if event == "start":
        if plan["mode"] != "run":
            raise ValueError(f"imported interface {interface_id} cannot record a run start")
        if root.exists():
            raise ValueError(f"append-only run path already exists: {root}")
        artifacts: dict[str, Any] = {}
    elif event == "import":
        if plan["mode"] != "import":
            raise ValueError(f"run interface {interface_id} cannot record a source import")
        artifacts = _artifact_inventory(root, step=step)
        if artifacts != plan["artifacts"]:
            raise ValueError(f"imported {interface_id} artifact evidence changed")
    else:
        if plan["mode"] != "run":
            raise ValueError(f"imported interface {interface_id} cannot record a run finish")
        artifacts = _artifact_inventory(root, step=step)
    ledger = Path(manifest["paths"]["execution_ledger"])
    records = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chain_failures = validate_jsonl_chain(records)
    if chain_failures:
        raise ValueError(f"curve ledger hash chain is invalid: {chain_failures}")
    desired = [("p0_preflight", None)]
    for name in INTERFACES:
        interface_plan = resolved["execution_binding"]["interface_plan"][name]
        if interface_plan["mode"] == "import":
            desired.append(("source_import", name))
        else:
            desired.extend((("interface_start", name), ("interface_finish", name)))
    record_type = {
        "start": "interface_start",
        "finish": "interface_finish",
        "import": "source_import",
    }[event]
    pair = (record_type, interface_id)
    actual = [(row.get("record_type"), row.get("interface_id")) for row in records] + [pair]
    if actual != desired[: len(actual)]:
        raise ValueError(f"curve event order differs from preregistration: {actual}")
    record = {
        "record_type": pair[0],
        "experiment_name": _experiment_name(interface_id),
        "git_commit": p0["evidence"]["git_commit"],
        "run_id": p0["evidence"]["run_id"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": canonical_sha256(resolved["execution_binding"]),
        "runtime_binding_sha256": p0["evidence"]["runtime_binding_sha256"],
        "interface_id": interface_id,
        "mode": plan["mode"],
        "status": "PASS",
        "artifacts": artifacts,
    }
    if event == "import":
        prior = resolved["execution_binding"]["prior_s128_it_import"]
        record.update(
            source_interface=plan["source_interface"],
            source_attempt=plan["source_attempt"],
            source_root=plan["root"],
            source_p0_certificate_sha256=prior["p0_certificate_sha256"],
            source_resolved_manifest_sha256=prior["resolved_manifest_sha256"],
            source_final_report_sha256=prior["final_report_sha256"],
            source_execution_ledger_sha256=prior["execution_ledger_sha256"],
            source_execution_ledger_tail_sha256=prior[
                "execution_ledger_tail_sha256"
            ],
        )
    append_jsonl(ledger, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--write-certificate", action="store_true")
    parser.add_argument("--emit-trainer-overrides", action="store_true")
    parser.add_argument("--interface-mode", action="store_true")
    parser.add_argument(
        "--record-interface-event", choices=("start", "finish", "import")
    )
    parser.add_argument("--interface", choices=INTERFACES)
    args = parser.parse_args()
    if args.emit_trainer_overrides:
        if not args.interface:
            parser.error("--interface is required")
        try:
            print("\n".join(frozen_trainer_overrides(args.manifest, interface_id=args.interface)))
            return 0
        except Exception as error:
            print(f"ORIGINAL_S128_CURVE_NO_GO:CONFIG {error}", file=sys.stderr)
            return 1
    if args.interface_mode:
        if not args.interface:
            parser.error("--interface is required")
        try:
            print(interface_mode(args.manifest, interface_id=args.interface))
            return 0
        except Exception as error:
            print(f"ORIGINAL_S128_CURVE_NO_GO:PLAN {error}", file=sys.stderr)
            return 1
    if args.record_interface_event:
        if not args.interface:
            parser.error("--interface is required")
        try:
            record = record_interface_event(
                args.manifest, interface_id=args.interface,
                event=args.record_interface_event,
            )
            print(json.dumps(record, indent=2, sort_keys=True))
            return 0
        except Exception as error:
            print(json.dumps({
                "status": "FAIL", "decision": "ORIGINAL_S128_CURVE_NO_GO:EVENT",
                "failures": [str(error)],
            }, indent=2))
            return 1
    if args.write_certificate and not args.check_runtime:
        parser.error("--write-certificate requires --check-runtime")
    try:
        result, resolved = run_preflight(args.manifest, check_runtime=args.check_runtime)
        manifest = load_manifest(args.manifest)
    except Exception as error:
        result, resolved, manifest = ({
            "gate": "P0", "status": "FAIL",
            "decision": "ORIGINAL_S128_CURVE_NO_GO:P0",
            "failures": [str(error)], "evidence": {},
        }, None, None)
    if args.write_certificate:
        if result["status"] != "PASS" or resolved is None or manifest is None:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
        p0_path = Path(manifest["paths"]["p0_certificate"])
        resolved_path = Path(manifest["paths"]["resolved_manifest"])
        ledger = Path(manifest["paths"]["execution_ledger"])
        existing = [str(path) for path in (p0_path, resolved_path, ledger) if path.exists()]
        if existing:
            raise SystemExit(f"refusing to overwrite append-only curve P0: {existing}")
        p0_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
        result["evidence"].update(
            run_id=secrets.token_hex(16),
            resolved_manifest_path=str(resolved_path),
            resolved_manifest_sha256=sha256_file(resolved_path),
        )
        with p0_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        append_jsonl(ledger, {
            "record_type": "p0_preflight",
            "experiment_name": "qwen25_7b_original_s128_curve_p0_seed2026_20260821",
            "git_commit": result["evidence"]["git_commit"],
            "run_id": result["evidence"]["run_id"],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "eval_manifest_hash": resolved["eval_manifest_hash"],
            "execution_binding_sha256": canonical_sha256(resolved["execution_binding"]),
            "runtime_binding_sha256": result["evidence"]["runtime_binding_sha256"],
            "interface_id": None, "mode": None, "status": "PASS",
            "artifact": str(p0_path), "artifact_sha256": sha256_file(p0_path),
            "row_count": 128,
        })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
