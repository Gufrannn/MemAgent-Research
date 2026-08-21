#!/usr/bin/env python3
"""P0 freeze and append-only supervisor for the COMMIT/RETAIN capture."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.commit_retain_capture import (  # noqa: E402
    STATE_ENCODING,
    build_capture_envelope,
    canonical_json,
    canonical_sha256,
    stable_capture_ids,
    validate_capture_ledger,
)
from recurrent.research.gate_a_execution import (  # noqa: E402
    append_jsonl,
    sha256_file,
    validate_jsonl_chain,
)
from recurrent.research.serialization_credit_pilots import (  # noqa: E402
    center_truncate_token_ids,
    read_jsonl,
    write_json_exclusive,
)
from tools.h20.preflight_qwen25_7b_serialization_credit import (  # noqa: E402
    _file_stat,
    _load_parquet_rows,
    _model_loading_paths,
    _question,
    _runtime_versions,
    _stable_prerequisite,
    capture_lightweight_current_binding,
    select_pilot_rows,
)


MANIFEST_REL = "manifests/h20/qwen25_7b_commit_retain_capture_seed2026.json"
EXPERIMENT_NAME = "qwen25_7b_commit_retain_capture_seed2026"
REQUIRED_ENV = (
    "MEMAGENT_COMMIT_RETAIN_WORK_ROOT",
    "MEMAGENT_COMMIT_RETAIN_REPO_DIR",
    "MEMAGENT_COMMIT_RETAIN_EXPECTED_COMMIT",
    "MEMAGENT_COMMIT_RETAIN_RUN_ID",
    "MEMAGENT_COMMIT_RETAIN_GPU_PAIR",
)
CODE_OBJECTS = (
    "recurrent/impls/memory.py",
    "recurrent/utils.py",
    "recurrent/research/commit_retain_capture.py",
    "recurrent/research/gate_a_execution.py",
    "recurrent/research/s128_hotpot_metrics.py",
    "recurrent/research/serialization_credit_pilots.py",
    "recurrent/research/stable_eval_identity.py",
    "recurrent/research/trajectory_seeding.py",
    "tools/h20/preflight_qwen25_7b_s128_it.py",
    "tools/h20/preflight_qwen25_7b_serialization_credit.py",
    "tools/h20/preflight_qwen25_7b_commit_retain.py",
    "tools/h20/run_qwen25_7b_commit_retain.py",
    "tools/h20/audit_qwen25_7b_commit_retain.py",
    "scripts/h20/commit_retain_capture_common.sh",
    "scripts/h20/preflight_qwen25_7b_commit_retain.sh",
    "scripts/h20/run_qwen25_7b_commit_retain.sh",
    MANIFEST_REL,
    "manifests/h20/qwen25_7b_commit_retain_capture_commands.json",
    "commit_retain_capture_execution_ledger.schema.json",
    "docs/h20/commit_retain_capture_freeze_20260821.md",
)

GPU_PROFILES = {
    "gpu67": {
        "branch": "h20/qwen25-7b-commit-retain-capture-20260821",
        "base_commit": "ded5e1c0c98267d8ea4a29e658685b9b832b9622",
        "experiment_name": EXPERIMENT_NAME,
        "command_manifest": "manifests/h20/qwen25_7b_commit_retain_capture_commands.json",
        "entrypoints": {
            "p0": "scripts/h20/preflight_qwen25_7b_commit_retain.sh",
            "capture": "scripts/h20/run_qwen25_7b_commit_retain.sh",
            "gpu_runner": "tools/h20/run_qwen25_7b_commit_retain.py",
            "audit": "tools/h20/audit_qwen25_7b_commit_retain.py",
        },
        "profile_objects": (),
    },
    "gpu45": {
        "branch": "h20/qwen25-7b-commit-retain-capture-gpu45-20260821",
        "base_commit": "e019e7655046f34d368a82e7d5ea6d72c464ffc7",
        "experiment_name": "qwen25_7b_commit_retain_capture_gpu45_seed2026",
        "command_manifest": "manifests/h20/qwen25_7b_commit_retain_capture_gpu45_commands.json",
        "entrypoints": {
            "p0": "scripts/h20/preflight_qwen25_7b_commit_retain_gpu45.sh",
            "capture": "scripts/h20/run_qwen25_7b_commit_retain_gpu45.sh",
            "gpu_runner": "tools/h20/run_qwen25_7b_commit_retain.py",
            "audit": "tools/h20/audit_qwen25_7b_commit_retain.py",
        },
        "profile_objects": (
            "manifests/h20/qwen25_7b_commit_retain_capture_gpu45_seed2026.json",
            "manifests/h20/qwen25_7b_commit_retain_capture_gpu45_commands.json",
            "scripts/h20/preflight_qwen25_7b_commit_retain_gpu45.sh",
            "scripts/h20/run_qwen25_7b_commit_retain_gpu45.sh",
            "docs/h20/commit_retain_capture_gpu45_freeze_20260821.md",
        ),
    },
}


def _gpu_profile(manifest: Mapping[str, Any]) -> dict[str, Any]:
    name = manifest.get("execution_profile", "gpu67")
    if name not in GPU_PROFILES:
        raise ValueError(f"execution_profile is not preregistered: {name}")
    gpu = manifest.get("gpu", {})
    return {
        "name": name,
        **GPU_PROFILES[name],
        "physical_whitelist": gpu.get("physical_whitelist"),
        "visible_devices": gpu.get("visible_devices"),
        "pair_slug": gpu.get("pair_slug"),
    }


def parse_gpu_pair(value: str) -> tuple[list[int], str]:
    match = re.fullmatch(r"([0-9]+),([0-9]+)", value)
    if match is None:
        raise ValueError("MEMAGENT_COMMIT_RETAIN_GPU_PAIR must be A,B")
    first, second = (int(match.group(1)), int(match.group(2)))
    if value != f"{first},{second}" or first >= second:
        raise ValueError("GPU pair must be canonical ascending distinct A,B")
    return [first, second], f"gpu{first}_{second}"


def experiment_name(manifest: Mapping[str, Any]) -> str:
    return str(_gpu_profile(manifest)["experiment_name"])


def _pair_evidence(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any]
) -> dict[str, Any]:
    profile = _gpu_profile(manifest)
    whitelist = profile["physical_whitelist"]
    visible = profile["visible_devices"]
    pair_slug = profile["pair_slug"]
    identities = resolved["runtime_binding"]["physical_gpu_identity"]
    if (
        not isinstance(whitelist, list)
        or len(whitelist) != 2
        or any(type(index) is not int or index < 0 for index in whitelist)
        or whitelist[0] >= whitelist[1]
        or visible != ",".join(str(index) for index in whitelist)
        or pair_slug != f"gpu{whitelist[0]}_{whitelist[1]}"
    ):
        raise ValueError("resolved physical GPU pair is not canonical")
    if not isinstance(identities, list) or len(identities) != 2:
        raise ValueError("resolved physical GPU identity must contain two devices")
    parsed_indices: list[int] = []
    for identity in identities:
        fields = [field.strip() for field in str(identity).split(",", 2)]
        if (
            len(fields) != 3
            or re.fullmatch(r"GPU-[0-9A-Fa-f-]+", fields[1]) is None
            or fields[2] != "NVIDIA H20"
        ):
            raise ValueError("resolved physical GPU identity lacks index/UUID/H20 binding")
        try:
            parsed_indices.append(int(fields[0]))
        except ValueError as error:
            raise ValueError("resolved physical GPU identity index is invalid") from error
    if parsed_indices != whitelist:
        raise ValueError("resolved physical GPU UUID identities differ from GPU pair")
    return {
        "gpu_pair_slug": pair_slug,
        "physical_gpu_whitelist": whitelist,
        "visible_devices": visible,
        "physical_gpu_identity": identities,
    }


def expected_git_commit() -> str:
    return os.environ[REQUIRED_ENV[2]]


def _code_objects(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    profile = _gpu_profile(manifest)
    if profile["name"] == "gpu67":
        return CODE_OBJECTS
    # The GPU45 overlay inherits the immutable scientific contract from the
    # original manifest, so both the base and overlay are included in P0.
    return CODE_OBJECTS + tuple(profile["profile_objects"])


def _merge_manifest(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = _merge_manifest(merged[key], value) if key in merged else value
        return merged
    return overlay


def _load_raw_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_rel = raw.pop("extends_manifest", None)
    if base_rel is None:
        return raw
    if base_rel != MANIFEST_REL:
        raise ValueError("only the frozen COMMIT/RETAIN base manifest may be extended")
    allowed_overlay_fields = {
        "schema_version", "frozen_at", "execution_profile", "branch", "base_commit",
        "command_manifest",
    }
    if set(raw) != allowed_overlay_fields:
        raise ValueError(
            "GPU execution overlay may change only branch/profile identity"
        )
    base_path = REPO_ROOT / base_rel
    if manifest_path.resolve() == base_path.resolve():
        raise ValueError("manifest cannot extend itself")
    return _merge_manifest(_load_raw_manifest(base_path), raw)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def resolve_manifest_environment(
    value: Any, environment: Mapping[str, str] | None = None
) -> Any:
    source = os.environ if environment is None else environment
    missing = [name for name in REQUIRED_ENV if not source.get(name)]
    if missing:
        raise ValueError(f"missing task-scoped runtime bindings: {missing}")
    work_root = Path(str(source[REQUIRED_ENV[0]]))
    repo = Path(str(source[REQUIRED_ENV[1]]))
    expected_commit = str(source[REQUIRED_ENV[2]])
    run_id = str(source[REQUIRED_ENV[3]])
    gpu_pair, gpu_pair_slug = parse_gpu_pair(str(source[REQUIRED_ENV[4]]))
    if not work_root.is_absolute() or not repo.is_absolute():
        raise ValueError("task-scoped runtime paths must be absolute")
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("MEMAGENT_COMMIT_RETAIN_EXPECTED_COMMIT must be a full Git SHA")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,31}", run_id) is None:
        raise ValueError("MEMAGENT_COMMIT_RETAIN_RUN_ID has an invalid format")
    replacements = {
        "${MEMAGENT_COMMIT_RETAIN_WORK_ROOT}": str(work_root),
        "${MEMAGENT_COMMIT_RETAIN_REPO_DIR}": str(repo),
        "${MEMAGENT_COMMIT_RETAIN_RUN_ID}": run_id,
        "${MEMAGENT_COMMIT_RETAIN_GPU_PAIR}": f"{gpu_pair[0]},{gpu_pair[1]}",
        "${MEMAGENT_COMMIT_RETAIN_GPU_PAIR_SLUG}": gpu_pair_slug,
        "${MEMAGENT_COMMIT_RETAIN_GPU_FIRST}": str(gpu_pair[0]),
        "${MEMAGENT_COMMIT_RETAIN_GPU_SECOND}": str(gpu_pair[1]),
    }

    def resolve(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: resolve(child) for key, child in item.items()}
        if isinstance(item, list):
            return [resolve(child) for child in item]
        if isinstance(item, str):
            if item == "${MEMAGENT_COMMIT_RETAIN_GPU_PAIR_AS_LIST}":
                return list(gpu_pair)
            result = item
            for placeholder, replacement in replacements.items():
                result = result.replace(placeholder, replacement)
            if "${" in result:
                raise ValueError(f"unresolved manifest placeholder: {result}")
            return result
        return item

    return resolve(value)


def load_manifest(
    path: str | Path, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    return resolve_manifest_environment(
        _load_raw_manifest(path), environment
    )


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    profile = _gpu_profile(manifest)
    if manifest.get("branch") != profile["branch"]:
        raise ValueError("branch differs from the frozen GPU profile")
    if manifest.get("base_commit") != profile["base_commit"]:
        raise ValueError("base_commit differs from the frozen GPU profile")
    scope = manifest.get("scope", {})
    expected_scope = {
        "fixed_existing_s128_only": True,
        "examples": 4,
        "training": False,
        "trainer_attached": False,
        "actor_updates": 0,
        "optimizer_steps": 0,
        "paper_performance_result": False,
        "causal_effect_claim": False,
        "method_selection_status": "CAPTURE_ONLY_NO_METHOD_SELECTION",
        "training_authorized": False,
    }
    for field, expected in expected_scope.items():
        if scope.get(field) != expected or type(scope.get(field)) is not type(expected):
            raise ValueError(f"scope.{field} differs from the capture-only contract")
    if manifest.get("model", {}).get("id") != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("model must be Qwen2.5-7B-Instruct")
    if manifest.get("data", {}).get("expected_rows") != 128:
        raise ValueError("data must be the fixed 128 rows")
    if manifest["data"]["pilot_selection"].get("sorted_positions") != [15, 47, 79, 111]:
        raise ValueError("four outcome-blind pilot strata drifted")
    gpu = manifest.get("gpu", {})
    parsed_pair, parsed_slug = parse_gpu_pair(str(gpu.get("visible_devices", "")))
    if gpu.get("pair_environment") != "MEMAGENT_COMMIT_RETAIN_GPU_PAIR":
        raise ValueError("gpu.pair_environment drifted")
    physical_whitelist = gpu.get("physical_whitelist")
    if (
        not isinstance(physical_whitelist, list)
        or any(type(index) is not int for index in physical_whitelist)
        or physical_whitelist != parsed_pair
    ):
        raise ValueError("gpu.physical_whitelist differs from explicit GPU pair")
    if gpu.get("pair_slug") != parsed_slug:
        raise ValueError("gpu.pair_slug differs from explicit GPU pair")
    for field, expected in {
        "cuda_device_order": "PCI_BUS_ID",
        "tensor_parallel_size": 2,
        "max_num_seqs": 1,
        "one_prompt_per_generate_call": True,
    }.items():
        if gpu.get(field) != expected or type(gpu.get(field)) is not type(expected):
            raise ValueError(f"gpu.{field} drifted")
    if manifest.get("command_manifest") != profile["command_manifest"]:
        raise ValueError("command_manifest differs from the frozen GPU profile")
    work_root = str(manifest.get("work_root"))
    run_id = str(manifest.get("run_id"))
    log_root = (
        f"{work_root}/logs/commit_retain_capture_frozen_20260821/"
        f"{run_id}_{parsed_slug}"
    )
    expected_paths = {
        "log_root": log_root,
        "certificate_root": f"{log_root}/certificates",
        "p0_certificate": f"{log_root}/certificates/p0_preflight.json",
        "resolved_manifest": f"{log_root}/certificates/p0_resolved_manifest.json",
        "execution_ledger": f"{log_root}/commit_retain_capture_execution_ledger.jsonl",
        "capture_credential": f"{log_root}/credentials/capture_child.json",
        "capture_ledger": f"{log_root}/captures/commit_retain_pairs.jsonl",
        "capture_run_receipt": f"{log_root}/captures/run_receipt.json",
        "final_report": f"{log_root}/certificates/commit_retain_capture_final_report.json",
    }
    if manifest.get("paths") != expected_paths:
        raise ValueError("GPU-pair output/certificate paths drifted")
    expected_locks = [
        f"{work_root}/locks/memagent_gate_a_gpu_{index}.lock" for index in parsed_pair
    ]
    if manifest.get("execution_resources") != {
        "project_locks": expected_locks,
        "output_root": log_root,
    }:
        raise ValueError("per-GPU lock/output resource binding drifted")
    backend = manifest.get("backend", {})
    for field, expected in {
        "name": "vllm",
        "required_version": "0.8.2",
        "strict_vllm": True,
        "VLLM_USE_V1": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "parent_cuda_initialization_policy": "record_observed_spawn_required",
        "allow_huggingface_generation_fallback": False,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "max_num_seqs": 1,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.45,
        "swap_space_gib": 1,
        "max_model_len": 9216,
        "max_num_batched_tokens": 9216,
        "engine_seed": 2026,
    }.items():
        if backend.get(field) != expected or type(backend.get(field)) is not type(expected):
            raise ValueError(f"backend.{field} drifted")
    intervention = manifest.get("intervention", {})
    for field, expected in {
        "examples": 4,
        "timepoints_per_example": 1,
        "candidate_generation_count_per_pair": 1,
        "candidate_materialized_before_branching": True,
        "arm_execution_order": ["COMMIT", "RETAIN"],
        "state_encoding": STATE_ENCODING,
        "prefix_cache_enabled": False,
        "same_engine_for_both_arms": True,
        "trainer_attached": False,
        "actor_updates": 0,
    }.items():
        if intervention.get(field) != expected or type(intervention.get(field)) is not type(expected):
            raise ValueError(f"intervention.{field} drifted")
    if intervention.get("shared_contracts") != [
        "future_chunks", "horizon", "reader_checkpoint", "decode", "rng", "cache", "cost"
    ]:
        raise ValueError("intervention shared-contract inventory drifted")
    rule = intervention.get("timepoint_rule", {})
    if rule != {
        "kind": "outcome_blind_midpoint_with_prefix_and_future",
        "formula": "max(1,(total_writer_turns-1)//2)",
        "minimum_total_writer_turns": 3,
        "requires_prefix_writer_turn": True,
        "requires_future_writer_turn": True,
    }:
        raise ValueError("intervention timepoint rule drifted")
    decode_base = {
        "top_p": 1.0, "top_k": -1, "min_p": 0.0, "n": 1, "best_of": 1,
        "max_tokens": 1024,
    }
    if intervention.get("writer_decode") != {"temperature": 1.0, **decode_base}:
        raise ValueError("intervention.writer_decode drifted")
    if intervention.get("reader_decode") != {"temperature": 0.0, **decode_base}:
        raise ValueError("intervention.reader_decode drifted")
    if manifest.get("recurrent") != {
        "implementation": "project_native_recurrent.impls.memory.TokenTemplate",
        "context_key": "context",
        "context_truncation": "center",
        "max_context_tokens": 40000,
        "chunk_size": 5000,
        "max_chunks": 8,
        "max_memory_tokens": 1024,
        "max_final_tokens": 1024,
        "no_memory_text": "No previous memory",
    }:
        raise ValueError("recurrent protocol drifted")


def _current_binding(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any], *, full_model_sha: bool
) -> str:
    current = capture_lightweight_current_binding(manifest)
    current_sha = canonical_sha256(current)
    if current_sha != resolved.get("lightweight_current_binding_sha256"):
        raise ValueError("current Git/model-stat/data/runtime binding differs from P0")
    if full_model_sha:
        model_root = Path(manifest["model"]["path"])
        actual = [
            {
                "path": item["path"],
                "size": (model_root / item["path"]).stat().st_size,
                "sha256": sha256_file(model_root / item["path"]),
            }
            for item in manifest["model"]["files"]
        ]
        if actual != manifest["model"]["files"]:
            raise ValueError("full model SHA inventory differs from P0")
        if canonical_sha256(actual) != resolved["execution_binding"]["model_manifest_sha256"]:
            raise ValueError("full model manifest digest differs from P0")
    return current_sha


def _native_interface_evidence() -> dict[str, Any]:
    from recurrent.impls.memory import MemoryAgent

    source = inspect.getsource(MemoryAgent.update)
    overwrite = "self.memory[self.active_mask] = unpad(" in source
    if not overwrite:
        raise ValueError("native MemoryAgent.update overwrite statement drifted")
    lowered = source.lower()
    if "retain" in lowered or "commit" in lowered:
        raise ValueError("native interface unexpectedly gained an unreviewed intervention branch")
    return {
        "method": "recurrent.impls.memory.MemoryAgent.update",
        "method_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "unconditional_nonfinal_state_overwrite_observed": True,
        "native_exact_retain_branch_observed": False,
        "minimal_capture_adapter": "recurrent.research.commit_retain_capture",
        "adapter_state_encoding": STATE_ENCODING,
        "training_path_modified": False,
    }


def _worker_multiprocessing_runtime_binding(
    python: Path, repo: Path
) -> dict[str, Any]:
    """Ask the frozen vLLM runtime which worker start method it will use."""
    code = """
import json
import os
from vllm import envs
from vllm.utils import get_mp_context

configured = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
observed = envs.VLLM_WORKER_MULTIPROC_METHOD
context = get_mp_context().get_start_method()
print(json.dumps({
    "configured_environment": configured,
    "vllm_observed_method": observed,
    "multiprocessing_context_method": context,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(python), "-c", code],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"cannot inspect vLLM worker multiprocessing runtime: {detail}")
    try:
        binding = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("vLLM worker multiprocessing runtime emitted no binding") from error
    expected = {
        "configured_environment": "spawn",
        "vllm_observed_method": "spawn",
        "multiprocessing_context_method": "spawn",
    }
    if binding != expected:
        raise ValueError(f"vLLM worker multiprocessing binding drifted: {binding}")
    return binding


def _freeze_pairs(
    *,
    manifest: Mapping[str, Any],
    stable_resolved: Mapping[str, Any],
    tokenizer: Any,
    writer_template: Any,
) -> list[dict[str, Any]]:
    rows = _load_parquet_rows(Path(manifest["data"]["validation"]))
    no_memory = tokenizer.encode(
        manifest["recurrent"]["no_memory_text"], add_special_tokens=False
    )

    def writer_builder(
        question_ids: list[int], memory_ids: list[int], chunk_ids: list[int]
    ) -> list[int]:
        return writer_template.format(
            prompt=question_ids, memory=memory_ids, chunk=chunk_ids
        ).tolist()

    selected = select_pilot_rows(
        parquet_rows=rows,
        stable_rows=stable_resolved["identity_payload"]["rows"],
        tokenizer=tokenizer,
        writer_prompt_builder=writer_builder,
        sorted_positions=list(manifest["data"]["pilot_selection"]["sorted_positions"]),
        eval_manifest_hash=stable_resolved["eval_manifest_hash"],
        base_seed=int(manifest["backend"]["engine_seed"]),
        chunk_size=int(manifest["recurrent"]["chunk_size"]),
    )
    frozen: list[dict[str, Any]] = []
    chunk_size = int(manifest["recurrent"]["chunk_size"])
    for pilot in selected:
        source = rows[int(pilot["raw_row_position"])]
        question_ids = list(tokenizer.encode(_question(source), add_special_tokens=False))
        context_ids = center_truncate_token_ids(
            list(tokenizer.encode(str(source["context"]), add_special_tokens=False)),
            int(manifest["recurrent"]["max_context_tokens"]),
        )
        chunks = [context_ids[offset : offset + chunk_size] for offset in range(0, len(context_ids), chunk_size)]
        total = len(chunks)
        if total < 3 or total > int(manifest["recurrent"]["max_chunks"]):
            raise ValueError(
                f"pilot {pilot['example_id']} has {total} writer turns; capture requires 3..max_chunks"
            )
        writer_turn = max(1, (total - 1) // 2)
        if writer_turn >= total - 1:
            raise ValueError("frozen timepoint lacks a future writer chunk")
        ids = stable_capture_ids(
            pilot,
            trajectory_seed=int(pilot["trajectory_seed"]),
            writer_turn=writer_turn,
        )
        frozen.append(
            {
                **dict(pilot),
                **ids,
                "intervention_writer_turn": writer_turn,
                "total_writer_turns": total,
                "question_token_ids_sha256": canonical_sha256(question_ids),
                "context_token_ids_sha256": canonical_sha256(context_ids),
                "chunk_token_ids_sha256": [canonical_sha256(chunk) for chunk in chunks],
                "candidate_chunk_token_ids_sha256": canonical_sha256(chunks[writer_turn]),
                "future_chunk_token_ids_sha256": [
                    canonical_sha256(chunk) for chunk in chunks[writer_turn + 1 :]
                ],
                "no_memory_token_ids_sha256": canonical_sha256(list(no_memory)),
                "expected_pair_generate_calls": writer_turn + 1 + 2 * (total - writer_turn),
            }
        )
    if len(frozen) != 4 or len({item["stable_write_id"] for item in frozen}) != 4:
        raise ValueError("P0 did not freeze exactly four unique stable writes")
    return frozen


def run_preflight(
    manifest_path: Path, *, check_runtime: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    failures: list[str] = []
    if not check_runtime:
        failures.append("formal P0 requires --check-runtime")
    raw_manifest = _load_raw_manifest(manifest_path)
    try:
        manifest = load_manifest(manifest_path)
        _validate_manifest(manifest)
    except Exception as error:
        return {
            "gate": "P0",
            "status": "FAIL",
            "decision": "COMMIT_RETAIN_NO_GO:P0",
            "failures": [str(error)],
            "evidence": {},
        }, None
    repo = Path(manifest["repository"]).resolve()
    expected_commit = expected_git_commit()
    profile = _gpu_profile(manifest)
    gpu_indices = profile["physical_whitelist"]
    visible_devices = profile["visible_devices"]
    evidence: dict[str, Any] = {
        "frozen_manifest_sha256": sha256_file(manifest_path),
        "expected_git_commit": expected_commit,
        "run_id": manifest["run_id"],
        "gpu_pair_slug": profile["pair_slug"],
        "physical_gpu_whitelist": gpu_indices,
        "visible_devices": visible_devices,
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible_devices:
        failures.append(
            f"P0 CUDA_VISIBLE_DEVICES must be exactly {visible_devices}"
        )
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        failures.append("P0 CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if os.environ.get("VLLM_USE_V1") != "0":
        failures.append("P0 VLLM_USE_V1 must be 0")
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        failures.append("P0 VLLM_WORKER_MULTIPROC_METHOD must be spawn")
    try:
        if repo != REPO_ROOT.resolve():
            failures.append("invoked checkout differs from explicit repository")
        branch = git(repo, "branch", "--show-current")
        commit = git(repo, "rev-parse", "HEAD")
        dirty = git(repo, "status", "--porcelain")
        evidence.update(git_branch=branch, git_commit=commit, worktree_clean=not bool(dirty))
        if branch != raw_manifest["branch"]:
            failures.append(f"wrong branch: {branch}")
        if commit != expected_commit:
            failures.append(f"HEAD differs from expected commit: {commit} != {expected_commit}")
        if dirty:
            failures.append("worktree is dirty")
        if subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", raw_manifest["base_commit"], commit],
            check=False,
        ).returncode:
            failures.append("required base commit is not an ancestor of HEAD")
        if git(repo, "diff", "--name-only", raw_manifest["base_commit"], commit, "--", "sources"):
            failures.append("sources/ changed on the capture branch")
    except Exception as error:
        failures.append(f"Git closure failed: {error}")

    code_objects = _code_objects(manifest)
    missing_code = [name for name in code_objects if not (repo / name).is_file()]
    failures.extend(f"required code object is missing: {name}" for name in missing_code)
    code_hashes = {
        name: sha256_file(repo / name) for name in code_objects if (repo / name).is_file()
    }
    evidence["execution_code_sha256"] = code_hashes
    try:
        commands = json.loads((repo / manifest["command_manifest"]).read_text(encoding="utf-8"))
        json.loads((repo / manifest["ledger_schema"]).read_text(encoding="utf-8"))
        if commands.get("required_sequence") != [
            "p0", "capture_authorization", "single_engine_four_pair_capture",
            "readonly_audit",
        ]:
            raise ValueError("command sequence drifted")
        if commands.get("gpu_execution_authorized_by_this_manifest") is not False:
            raise ValueError("command manifest improperly self-authorizes GPU work")
        if commands.get("branch") != raw_manifest["branch"]:
            raise ValueError("command manifest branch drifted")
        if commands.get("entrypoints") != profile["entrypoints"]:
            raise ValueError("command entrypoints drifted")
        if profile["name"] == "gpu45" and commands.get("execution_profile") != "gpu45":
            raise ValueError("command execution profile drifted")
        command_execution = commands.get("execution", {})
        for field, expected in {
            "gpu_pair_environment": "MEMAGENT_COMMIT_RETAIN_GPU_PAIR",
            "canonical_ascending_distinct_pair_required": True,
            "per_gpu_locking": True,
            "cuda_device_order": "PCI_BUS_ID",
            "tensor_parallel_size": 2,
            "backend": "strict_vllm_0.8.2",
            "worker_multiproc_method": "spawn",
            "parent_cuda_initialization_policy": "record_observed_spawn_required",
            "huggingface_generation_fallback": False,
            "single_coordinator_invocation_single_engine": True,
            "one_prompt_per_generate_call": True,
            "max_num_seqs": 1,
            "prefix_cache_enabled": False,
            "candidate_generated_once_before_arms": True,
            "arm_order": ["COMMIT", "RETAIN"],
            "training_updates": 0,
        }.items():
            if command_execution.get(field) != expected:
                raise ValueError(f"command execution.{field} drifted")
        if commands.get("supervision", {}).get(
            "parent_issued_single_use_capture_credential"
        ) is not True:
            raise ValueError("command manifest lost parent capture authorization")
        if commands.get("claim_firewall") != {
            "capture_and_audit_only": True,
            "method_selected": False,
            "training_authorized": False,
            "paper_performance_result": False,
            "causal_effect_claim": False,
        }:
            raise ValueError("command claim firewall drifted")
    except Exception as error:
        failures.append(f"command/schema closure failed: {error}")

    model_root = Path(manifest["model"]["path"])
    actual_model_files: list[dict[str, Any]] = []
    for expected in manifest["model"]["files"]:
        path = model_root / expected["path"]
        if not path.is_file():
            failures.append(f"frozen model file is missing: {path}")
            continue
        actual = {
            "path": expected["path"],
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        actual_model_files.append(actual)
        if actual != expected:
            failures.append(f"frozen model file drifted: {expected['path']}")
    if model_root.is_dir() and _model_loading_paths(model_root) != sorted(
        item["path"] for item in manifest["model"]["files"]
    ):
        failures.append("model loading-relevant inventory drifted")
    model_manifest_sha = canonical_sha256(actual_model_files)
    tokenizer_manifest_sha = canonical_sha256(
        [
            item for item in actual_model_files
            if item["path"] in {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
        ]
    )
    data_path = Path(manifest["data"]["validation"])
    if not data_path.is_file() or sha256_file(data_path) != manifest["data"]["validation_sha256"]:
        failures.append("fixed S128 parquet is missing or drifted")

    stable_contract: dict[str, Any] | None = None
    stable_resolved: dict[str, Any] | None = None
    try:
        stable_contract, stable_resolved = _stable_prerequisite(manifest)
        evidence["stable_identity_prerequisite"] = stable_contract
    except Exception as error:
        failures.append(f"stable-I prerequisite failed: {error}")

    python = Path(manifest["python"])
    runtime_versions: dict[str, str] = {}
    worker_multiprocessing: dict[str, Any] = {}
    gpu_identity: list[str] = []
    if not python.is_file():
        failures.append(f"frozen Python is missing: {python}")
    if check_runtime and python.is_file():
        try:
            runtime_versions = _runtime_versions(python, repo)
            if runtime_versions.get("vllm") != "0.8.2":
                failures.append(f"vLLM version drifted: {runtime_versions.get('vllm')}")
            worker_multiprocessing = _worker_multiprocessing_runtime_binding(
                python, repo
            )
        except Exception as error:
            failures.append(str(error))
        completed = subprocess.run(
            [
                "nvidia-smi", "-i", visible_devices, "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            failures.append(
                f"cannot identify physical GPU{visible_devices}: {completed.stderr.strip()}"
            )
        else:
            gpu_identity = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            try:
                observed_indices = [int(line.split(",", 1)[0].strip()) for line in gpu_identity]
            except (ValueError, IndexError):
                observed_indices = []
            if observed_indices != gpu_indices:
                failures.append(
                    f"physical GPU indices {observed_indices} != {gpu_indices}"
                )
            if len(gpu_identity) != 2 or any("NVIDIA H20" not in line for line in gpu_identity):
                failures.append(
                    f"physical GPU{visible_devices} are not both NVIDIA H20: {gpu_identity}"
                )
    evidence["runtime_versions"] = runtime_versions
    evidence["worker_multiprocessing"] = worker_multiprocessing
    evidence["physical_gpu_identity"] = gpu_identity

    resolved: dict[str, Any] | None = None
    if not failures and stable_resolved is not None:
        try:
            from transformers import AutoTokenizer
            from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
            from recurrent.utils import TokenTemplate, chat_template

            tokenizer = AutoTokenizer.from_pretrained(
                model_root, trust_remote_code=True, local_files_only=True
            )
            writer_template_text = chat_template(tokenizer).format(message=TEMPLATE)
            reader_template_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
            writer_template = TokenTemplate(writer_template_text, tokenizer)
            frozen_pairs = _freeze_pairs(
                manifest=manifest,
                stable_resolved=stable_resolved,
                tokenizer=tokenizer,
                writer_template=writer_template,
            )
            native_evidence = _native_interface_evidence()
            engine_config = {
                **dict(manifest["backend"]),
                "physical_gpu_whitelist": gpu_indices,
                "visible_devices": visible_devices,
                "tensor_parallel_size": 2,
                "one_prompt_per_generate_call": True,
            }
            runtime_binding = {
                "git_commit": evidence["git_commit"],
                "branch": evidence["git_branch"],
                "worktree_clean": evidence["worktree_clean"],
                "model_manifest_sha256": model_manifest_sha,
                "tokenizer_manifest_sha256": tokenizer_manifest_sha,
                "validation_data_sha256": manifest["data"]["validation_sha256"],
                "runtime_versions": runtime_versions,
                "worker_multiprocessing": worker_multiprocessing,
                "physical_gpu_identity": gpu_identity,
                "gpu_pair_slug": profile["pair_slug"],
                "physical_gpu_whitelist": gpu_indices,
                "visible_devices": visible_devices,
            }
            execution_binding = {
                "git_commit": evidence["git_commit"],
                "eval_manifest_hash": stable_resolved["eval_manifest_hash"],
                "pilot_selection": manifest["data"]["pilot_selection"],
                "frozen_pairs": frozen_pairs,
                "timepoint_rule": manifest["intervention"]["timepoint_rule"],
                "model_manifest_sha256": model_manifest_sha,
                "tokenizer_manifest_sha256": tokenizer_manifest_sha,
                "writer_prompt_template_sha256": hashlib.sha256(
                    writer_template_text.encode("utf-8")
                ).hexdigest(),
                "reader_prompt_template_sha256": hashlib.sha256(
                    reader_template_text.encode("utf-8")
                ).hexdigest(),
                "engine_config": engine_config,
                "engine_config_sha256": canonical_sha256(engine_config),
                "recurrent": manifest["recurrent"],
                "intervention": manifest["intervention"],
                "native_memory_interface_evidence": native_evidence,
                "execution_code_sha256": code_hashes,
                "execution_code_combined_sha256": canonical_sha256(code_hashes),
                "expected_global_generate_call_count": sum(
                    int(item["expected_pair_generate_calls"]) for item in frozen_pairs
                ),
            }
            lightweight = capture_lightweight_current_binding(manifest)
            if lightweight["git_commit"] != expected_commit or lightweight["git_branch"] != raw_manifest["branch"]:
                raise ValueError("lightweight binding Git identity drifted")
            if lightweight["worktree_clean"] is not True:
                raise ValueError("lightweight binding worktree is dirty")
            resolved = {
                "schema_version": 1,
                "frozen_manifest_sha256": evidence["frozen_manifest_sha256"],
                "run_id": manifest["run_id"],
                "gpu_pair_slug": profile["pair_slug"],
                "physical_gpu_whitelist": gpu_indices,
                "visible_devices": visible_devices,
                "eval_manifest_hash": stable_resolved["eval_manifest_hash"],
                "stable_identity_resolved_manifest_sha256": sha256_file(
                    manifest["stable_identity_prerequisite"]["resolved_manifest"]
                ),
                "frozen_pairs": frozen_pairs,
                "runtime_binding": runtime_binding,
                "runtime_binding_sha256": canonical_sha256(runtime_binding),
                "lightweight_current_binding": lightweight,
                "lightweight_current_binding_sha256": canonical_sha256(lightweight),
                "execution_binding": execution_binding,
                "execution_binding_sha256": canonical_sha256(execution_binding),
            }
            evidence.update(
                eval_manifest_hash=resolved["eval_manifest_hash"],
                stable_write_ids=[item["stable_write_id"] for item in frozen_pairs],
                intervention_writer_turns=[item["intervention_writer_turn"] for item in frozen_pairs],
                expected_global_generate_call_count=execution_binding[
                    "expected_global_generate_call_count"
                ],
                native_memory_interface_evidence=native_evidence,
                runtime_binding_sha256=resolved["runtime_binding_sha256"],
                execution_binding_sha256=resolved["execution_binding_sha256"],
                lightweight_current_binding_sha256=resolved[
                    "lightweight_current_binding_sha256"
                ],
            )
        except Exception as error:
            failures.append(f"cannot freeze capture execution binding: {error}")
            resolved = None
    status = "PASS" if not failures and resolved is not None else "FAIL"
    return {
        "gate": "P0",
        "status": status,
        "decision": "COMMIT_RETAIN_P0_PASS" if status == "PASS" else "COMMIT_RETAIN_NO_GO:P0",
        "failures": failures,
        "evidence": evidence,
        "scope": manifest["scope"],
    }, resolved


def write_preflight(manifest_path: Path, *, check_runtime: bool) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    p0_path = Path(manifest["paths"]["p0_certificate"])
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    ledger_path = Path(manifest["paths"]["execution_ledger"])
    if any(path.exists() for path in (p0_path, resolved_path, ledger_path)):
        raise FileExistsError("refuse to overwrite append-only P0 evidence")
    report, resolved = run_preflight(manifest_path, check_runtime=check_runtime)
    if resolved is not None:
        write_json_exclusive(resolved_path, resolved)
        report["evidence"]["resolved_manifest_path"] = str(resolved_path.resolve())
        report["evidence"]["resolved_manifest_sha256"] = sha256_file(resolved_path)
    write_json_exclusive(p0_path, report)
    if report["status"] == "PASS" and resolved is not None:
        append_jsonl(
            ledger_path,
            {
                "record_type": "s0_preflight",
                "experiment_name": experiment_name(manifest),
                "git_commit": report["evidence"]["git_commit"],
                "run_id": manifest["run_id"],
                **_pair_evidence(manifest, resolved),
                "recorded_at": utc_now(),
                "eval_manifest_hash": resolved["eval_manifest_hash"],
                "execution_binding_sha256": resolved["execution_binding_sha256"],
                "runtime_binding_sha256": resolved["runtime_binding_sha256"],
                "current_binding_sha256": resolved["lightweight_current_binding_sha256"],
                "artifact": str(p0_path.resolve()),
                "artifact_sha256": sha256_file(p0_path),
                "resolved_manifest": str(resolved_path.resolve()),
                "resolved_manifest_sha256": sha256_file(resolved_path),
                "status": "PASS",
                "decision": "COMMIT_RETAIN_P0_PASS",
                "training_authorized": False,
                "method_selected": False,
            },
        )
    return report


def validate_p0(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    p0_path = Path(manifest["paths"]["p0_certificate"])
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    ledger_path = Path(manifest["paths"]["execution_ledger"])
    if not p0_path.is_file() or not resolved_path.is_file() or not ledger_path.is_file():
        raise ValueError("P0 certificate/resolved manifest/supervisor ledger is missing")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    records = read_jsonl(ledger_path)
    failures = validate_jsonl_chain(records)
    if failures or not records:
        raise ValueError(f"supervisor ledger P0 chain failed: {failures}")
    head = records[0]
    head_allowed = {
        "record_type", "experiment_name", "git_commit", "run_id", "recorded_at",
        "gpu_pair_slug", "physical_gpu_whitelist", "visible_devices",
        "physical_gpu_identity",
        "eval_manifest_hash", "execution_binding_sha256", "runtime_binding_sha256",
        "current_binding_sha256", "artifact", "artifact_sha256",
        "resolved_manifest", "resolved_manifest_sha256", "status", "decision",
        "training_authorized", "method_selected", "record_index",
        "previous_record_sha256", "record_sha256",
    }
    if set(head) != head_allowed:
        raise ValueError("P0 supervisor receipt has handcrafted fields")
    expected_commit = expected_git_commit()
    pair = _pair_evidence(manifest, resolved)
    valid = all(
        (
            p0.get("status") == "PASS",
            p0.get("decision") == "COMMIT_RETAIN_P0_PASS",
            p0.get("evidence", {}).get("git_commit") == expected_commit,
            p0.get("evidence", {}).get("expected_git_commit") == expected_commit,
            p0.get("evidence", {}).get("frozen_manifest_sha256") == sha256_file(manifest_path),
            resolved.get("frozen_manifest_sha256") == sha256_file(manifest_path),
            p0.get("evidence", {}).get("gpu_pair_slug") == pair["gpu_pair_slug"],
            p0.get("evidence", {}).get("physical_gpu_whitelist")
            == pair["physical_gpu_whitelist"],
            p0.get("evidence", {}).get("visible_devices") == pair["visible_devices"],
            p0.get("evidence", {}).get("physical_gpu_identity")
            == pair["physical_gpu_identity"],
            resolved.get("gpu_pair_slug") == pair["gpu_pair_slug"],
            resolved.get("physical_gpu_whitelist") == pair["physical_gpu_whitelist"],
            resolved.get("visible_devices") == pair["visible_devices"],
            resolved.get("runtime_binding", {}).get("gpu_pair_slug")
            == pair["gpu_pair_slug"],
            resolved.get("runtime_binding", {}).get("physical_gpu_whitelist")
            == pair["physical_gpu_whitelist"],
            resolved.get("runtime_binding", {}).get("visible_devices")
            == pair["visible_devices"],
            p0.get("evidence", {}).get("resolved_manifest_sha256") == sha256_file(resolved_path),
            canonical_sha256(resolved.get("runtime_binding")) == resolved.get("runtime_binding_sha256"),
            canonical_sha256(resolved.get("execution_binding")) == resolved.get("execution_binding_sha256"),
            canonical_sha256(resolved.get("lightweight_current_binding"))
            == resolved.get("lightweight_current_binding_sha256"),
            head.get("record_type") == "s0_preflight",
            head.get("experiment_name") == experiment_name(manifest),
            head.get("git_commit") == expected_commit,
            head.get("run_id") == manifest["run_id"],
            all(head.get(field) == value for field, value in pair.items()),
            head.get("eval_manifest_hash") == resolved.get("eval_manifest_hash"),
            head.get("artifact") == str(p0_path.resolve()),
            head.get("artifact_sha256") == sha256_file(p0_path),
            head.get("resolved_manifest") == str(resolved_path.resolve()),
            head.get("resolved_manifest_sha256") == sha256_file(resolved_path),
            head.get("execution_binding_sha256") == resolved.get("execution_binding_sha256"),
            head.get("runtime_binding_sha256") == resolved.get("runtime_binding_sha256"),
            head.get("current_binding_sha256") == resolved.get("lightweight_current_binding_sha256"),
            head.get("status") == "PASS",
            head.get("decision") == "COMMIT_RETAIN_P0_PASS",
            head.get("training_authorized") is False,
            head.get("method_selected") is False,
        )
    )
    if not valid:
        raise ValueError("P0 certificate prefix authentication failed")
    return p0, resolved


def issue_capture_credential(
    manifest_path: Path, *, output: Path, issuer_shell_pid: int
) -> dict[str, Any]:
    """Pre-authorize the one strict-vLLM capture child in the supervisor chain."""
    if type(issuer_shell_pid) is not int or issuer_shell_pid < 1:
        raise ValueError("issuer shell PID must be a positive integer")
    if os.getppid() != issuer_shell_pid:
        raise ValueError(
            f"credential issuer is not a direct shell child: {os.getppid()} != {issuer_shell_pid}"
        )
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    expected_output = Path(manifest["paths"]["capture_credential"]).resolve()
    if output.resolve() != expected_output:
        raise ValueError("capture credential path differs from the frozen path")
    ledger_path = Path(manifest["paths"]["execution_ledger"])
    records = read_jsonl(ledger_path)
    if validate_jsonl_chain(records) or [item.get("record_type") for item in records] != [
        "s0_preflight"
    ]:
        raise ValueError("capture credential must immediately follow P0")
    current_sha = _current_binding(manifest, resolved, full_model_sha=False)
    credential = {
        "schema": "memagent.commit-retain.parent-capture-credential.v1",
        "run_id": manifest["run_id"],
        **_pair_evidence(manifest, resolved),
        "git_commit": expected_git_commit(),
        "child_kind": "single_engine_four_pair_capture",
        "child_identity": f"{manifest['run_id']}:four-frozen-stable-writes",
        "parent_issuer_pid": issuer_shell_pid,
        "issued_at": utc_now(),
        "nonce": secrets.token_hex(32),
        "current_binding_sha256": current_sha,
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "child_full_model_sha_required": True,
        "single_use": True,
        "training_authorized": False,
        "method_selected": False,
    }
    credential["parent_credential_id"] = canonical_sha256(credential)
    write_json_exclusive(output, credential)
    append_jsonl(
        ledger_path,
        {
            "record_type": "capture_authorization",
            "experiment_name": experiment_name(manifest),
            "git_commit": expected_git_commit(),
            "run_id": manifest["run_id"],
            **_pair_evidence(manifest, resolved),
            "recorded_at": utc_now(),
            "eval_manifest_hash": resolved["eval_manifest_hash"],
            "execution_binding_sha256": resolved["execution_binding_sha256"],
            "runtime_binding_sha256": resolved["runtime_binding_sha256"],
            "current_binding_sha256": current_sha,
            "artifact": str(output.resolve()),
            "artifact_sha256": sha256_file(output),
            "parent_credential_id": credential["parent_credential_id"],
            "parent_issuer_pid": issuer_shell_pid,
            "status": "PASS",
            "decision": "COMMIT_RETAIN_CAPTURE_CHILD_AUTHORIZED",
            "training_authorized": False,
            "method_selected": False,
        },
    )
    authorization = read_jsonl(ledger_path)[-1]
    return {
        "parent_credential_id": credential["parent_credential_id"],
        "parent_credential_sha256": sha256_file(output),
        "parent_credential_path": str(output.resolve()),
        "parent_issuer_pid": issuer_shell_pid,
        "parent_authorization_record_sha256": authorization["record_sha256"],
    }


def validate_capture_credential(
    credential_path: Path,
    *,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    current_binding_sha256: str,
    require_live_parent: bool,
) -> dict[str, Any]:
    expected_path = Path(manifest["paths"]["capture_credential"]).resolve()
    if credential_path.resolve() != expected_path or not credential_path.is_file():
        raise ValueError("capture credential is missing or outside the frozen path")
    credential = json.loads(credential_path.read_text(encoding="utf-8"))
    credential_allowed = {
        "schema", "run_id", "git_commit", "child_kind", "child_identity",
        "gpu_pair_slug", "physical_gpu_whitelist", "visible_devices",
        "physical_gpu_identity",
        "parent_issuer_pid", "issued_at", "nonce", "current_binding_sha256",
        "runtime_binding_sha256", "execution_binding_sha256",
        "child_full_model_sha_required", "single_use", "training_authorized",
        "method_selected", "parent_credential_id",
    }
    if set(credential) != credential_allowed:
        raise ValueError("capture credential has handcrafted fields")
    credential_id = credential.get("parent_credential_id")
    unsigned = dict(credential)
    unsigned.pop("parent_credential_id", None)
    if credential_id != canonical_sha256(unsigned):
        raise ValueError("capture credential self-digest failed")
    parent_pid = credential.get("parent_issuer_pid")
    pair = _pair_evidence(manifest, resolved)
    if type(parent_pid) is not int or parent_pid < 1:
        raise ValueError("capture credential parent PID is invalid")
    try:
        issued_at = datetime.fromisoformat(credential.get("issued_at"))
    except (TypeError, ValueError) as error:
        raise ValueError("capture credential issued_at is invalid") from error
    if issued_at.tzinfo is None or issued_at.utcoffset() != timezone.utc.utcoffset(issued_at):
        raise ValueError("capture credential issued_at is not UTC")
    if require_live_parent and os.getppid() != parent_pid:
        raise ValueError(
            f"capture runner is not the authorized direct child: {os.getppid()} != {parent_pid}"
        )
    valid = all(
        (
            credential.get("schema") == "memagent.commit-retain.parent-capture-credential.v1",
            credential.get("run_id") == manifest["run_id"],
            credential.get("git_commit") == expected_git_commit(),
            credential.get("child_kind") == "single_engine_four_pair_capture",
            credential.get("child_identity")
            == f"{manifest['run_id']}:four-frozen-stable-writes",
            all(credential.get(field) == value for field, value in pair.items()),
            credential.get("current_binding_sha256") == current_binding_sha256,
            credential.get("runtime_binding_sha256") == resolved["runtime_binding_sha256"],
            credential.get("execution_binding_sha256") == resolved["execution_binding_sha256"],
            credential.get("child_full_model_sha_required") is True,
            credential.get("single_use") is True,
            credential.get("training_authorized") is False,
            credential.get("method_selected") is False,
            isinstance(credential.get("nonce"), str),
            re.fullmatch(r"[0-9a-f]{64}", credential.get("nonce", "")) is not None,
        )
    )
    if not valid:
        raise ValueError("capture credential binding differs from P0")
    ledger = read_jsonl(manifest["paths"]["execution_ledger"])
    if validate_jsonl_chain(ledger) or len(ledger) < 2:
        raise ValueError("capture authorization supervisor chain failed")
    record_types = [item.get("record_type") for item in ledger]
    if require_live_parent and record_types != ["s0_preflight", "capture_authorization"]:
        raise ValueError("live capture child must start immediately after its authorization")
    if not require_live_parent and record_types[:2] != [
        "s0_preflight", "capture_authorization"
    ]:
        raise ValueError("capture authorization is not the supervisor prefix")
    authorization = ledger[1]
    authorization_allowed = {
        "record_type", "experiment_name", "git_commit", "run_id", "recorded_at",
        "gpu_pair_slug", "physical_gpu_whitelist", "visible_devices",
        "physical_gpu_identity",
        "eval_manifest_hash", "execution_binding_sha256", "runtime_binding_sha256",
        "current_binding_sha256", "artifact", "artifact_sha256", "parent_credential_id",
        "parent_issuer_pid", "status", "decision", "training_authorized",
        "method_selected", "record_index", "previous_record_sha256", "record_sha256",
    }
    if set(authorization) != authorization_allowed:
        raise ValueError("capture authorization receipt has handcrafted fields")
    if not all(
        (
            authorization.get("record_type") == "capture_authorization",
            authorization.get("experiment_name") == experiment_name(manifest),
            authorization.get("artifact") == str(credential_path.resolve()),
            authorization.get("artifact_sha256") == sha256_file(credential_path),
            authorization.get("parent_credential_id") == credential_id,
            authorization.get("parent_issuer_pid") == parent_pid,
            authorization.get("current_binding_sha256") == current_binding_sha256,
            authorization.get("runtime_binding_sha256")
            == resolved["runtime_binding_sha256"],
            authorization.get("execution_binding_sha256")
            == resolved["execution_binding_sha256"],
            authorization.get("git_commit") == expected_git_commit(),
            authorization.get("run_id") == manifest["run_id"],
            all(authorization.get(field) == value for field, value in pair.items()),
            authorization.get("eval_manifest_hash") == resolved["eval_manifest_hash"],
            authorization.get("status") == "PASS",
            authorization.get("decision") == "COMMIT_RETAIN_CAPTURE_CHILD_AUTHORIZED",
            authorization.get("training_authorized") is False,
            authorization.get("method_selected") is False,
        )
    ):
        raise ValueError("capture authorization receipt differs from credential")
    return {
        **pair,
        "parent_credential_id": credential_id,
        "parent_credential_sha256": sha256_file(credential_path),
        "parent_credential_path": str(credential_path.resolve()),
        "parent_issuer_pid": parent_pid,
        "observed_parent_pid": parent_pid,
        "parent_authorization_record_sha256": authorization["record_sha256"],
    }


def _tokenizer(manifest: Mapping[str, Any]):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        manifest["model"]["path"], trust_remote_code=True, local_files_only=True
    )


def expected_pair_binding(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any], tokenizer: Any
) -> dict[str, Any]:
    execution = resolved["execution_binding"]
    checkpoint_sha = execution["model_manifest_sha256"]
    return {
        "writer_checkpoint_sha256": checkpoint_sha,
        "reader_checkpoint_sha256": checkpoint_sha,
        "writer_prompt_template_sha256": execution["writer_prompt_template_sha256"],
        "reader_prompt_template_sha256": execution["reader_prompt_template_sha256"],
        "writer_decode": manifest["intervention"]["writer_decode"],
        "reader_decode": manifest["intervention"]["reader_decode"],
        "gpu_pair_slug": _gpu_profile(manifest)["pair_slug"],
        "physical_gpu_whitelist": _gpu_profile(manifest)["physical_whitelist"],
        "visible_devices": _gpu_profile(manifest)["visible_devices"],
        "physical_gpu_identity": resolved["runtime_binding"]["physical_gpu_identity"],
        "engine_config_sha256": execution["engine_config_sha256"],
        "worker_multiproc_method": "spawn",
        "vllm_observed_worker_multiproc_method": "spawn",
        "multiprocessing_context_method": "spawn",
        "parent_cuda_initialization_policy": "record_observed_spawn_required",
        "global_generate_call_count": execution["expected_global_generate_call_count"],
        "eos_token_id": int(tokenizer.eos_token_id),
    }


def _expected_run_receipt(
    *,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    current_binding_sha256: str,
    capture_report: Mapping[str, Any],
    capture_path: Path,
) -> dict[str, Any]:
    execution = read_jsonl(capture_path)[0]["pair"]["execution"]
    receipt = {
        "schema": "memagent.commit-retain.capture-run-receipt.v1",
        "run_id": manifest["run_id"],
        "git_commit": expected_git_commit(),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": current_binding_sha256,
        **_pair_evidence(manifest, resolved),
        "capture_ledger": str(capture_path.resolve()),
        "capture_ledger_sha256": sha256_file(capture_path),
        "pair_count": capture_report["pair_count"],
        "pair_ids": capture_report["pair_ids"],
        "stable_write_ids": capture_report["stable_write_ids"],
        "generate_call_count": capture_report["generate_call_count"],
        "execution": execution,
        "training": capture_report["training"],
        "claim_boundary": capture_report["claim_boundary"],
    }
    receipt["run_receipt_id"] = canonical_sha256(receipt)
    return receipt


def validate_capture_artifacts(
    manifest_path: Path, *, require_supervisor_capture_receipt: bool
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    current_sha = _current_binding(manifest, resolved, full_model_sha=False)
    capture_path = Path(manifest["paths"]["capture_ledger"])
    run_receipt_path = Path(manifest["paths"]["capture_run_receipt"])
    if not capture_path.is_file() or not run_receipt_path.is_file():
        raise ValueError("capture ledger or run receipt is missing")
    tokenizer = _tokenizer(manifest)
    from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template

    chat = chat_template(tokenizer)
    writer_template = TokenTemplate(chat.format(message=TEMPLATE), tokenizer)
    reader_template = TokenTemplate(chat.format(message=TEMPLATE_FINAL_BOXED), tokenizer)
    report = validate_capture_ledger(
        read_jsonl(capture_path),
        frozen_pairs=resolved["frozen_pairs"],
        experiment_name=experiment_name(manifest),
        git_commit=expected_git_commit(),
        run_id=manifest["run_id"],
        execution_binding_sha256=resolved["execution_binding_sha256"],
        runtime_binding_sha256=resolved["runtime_binding_sha256"],
        current_binding_sha256=current_sha,
        decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        writer_prompt_builder=lambda question, memory, chunk: writer_template.format(
            prompt=question, memory=memory, chunk=chunk
        ).tolist(),
        reader_prompt_builder=lambda question, memory: reader_template.format(
            prompt=question, memory=memory
        ).tolist(),
        expected_pair_binding=expected_pair_binding(manifest, resolved, tokenizer),
    )
    credential_evidence = validate_capture_credential(
        Path(manifest["paths"]["capture_credential"]),
        manifest=manifest,
        resolved=resolved,
        current_binding_sha256=current_sha,
        require_live_parent=False,
    )
    executions = [item["pair"]["execution"] for item in read_jsonl(capture_path)]
    if any(
        execution.get(field) != value
        for execution in executions
        for field, value in credential_evidence.items()
    ):
        raise ValueError("capture execution differs from the parent authorization credential")
    expected_receipt = _expected_run_receipt(
        manifest=manifest,
        resolved=resolved,
        current_binding_sha256=current_sha,
        capture_report=report,
        capture_path=capture_path,
    )
    actual_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    if canonical_json(actual_receipt) != canonical_json(expected_receipt):
        raise ValueError("capture run receipt is handcrafted, stale, or incomplete")
    if require_supervisor_capture_receipt:
        records = read_jsonl(manifest["paths"]["execution_ledger"])
        failures = validate_jsonl_chain(records)
        if failures or len(records) not in (3, 4):
            raise ValueError(f"supervisor capture prefix failed: {failures}")
        expected_types = [
            "s0_preflight", "capture_authorization", "capture_complete"
        ] + (["audit_result"] if len(records) == 4 else [])
        if [record.get("record_type") for record in records] != expected_types:
            raise ValueError("supervisor stage order differs from the frozen sequence")
        capture_receipt = records[2]
        capture_allowed = {
            "record_type", "experiment_name", "git_commit", "run_id", "recorded_at",
            "gpu_pair_slug", "physical_gpu_whitelist", "visible_devices",
            "physical_gpu_identity",
            "eval_manifest_hash", "execution_binding_sha256", "runtime_binding_sha256",
            "current_binding_sha256", "artifact", "artifact_sha256", "training_authorized",
            "method_selected", "status", "decision", "pair_count", "pair_ids",
            "stable_write_ids", "generate_call_count", "run_receipt",
            "run_receipt_sha256", "record_index", "previous_record_sha256", "record_sha256",
        }
        if set(capture_receipt) != capture_allowed:
            raise ValueError("capture supervisor receipt has handcrafted fields")
        if not all(
            (
                capture_receipt.get("record_type") == "capture_complete",
                capture_receipt.get("artifact") == str(capture_path.resolve()),
                capture_receipt.get("artifact_sha256") == sha256_file(capture_path),
                capture_receipt.get("run_receipt") == str(run_receipt_path.resolve()),
                capture_receipt.get("run_receipt_sha256") == sha256_file(run_receipt_path),
                capture_receipt.get("pair_ids") == report["pair_ids"],
                capture_receipt.get("stable_write_ids") == report["stable_write_ids"],
                capture_receipt.get("pair_count") == 4,
                capture_receipt.get("generate_call_count") == report["generate_call_count"],
                capture_receipt.get("git_commit") == expected_git_commit(),
                capture_receipt.get("run_id") == manifest["run_id"],
                all(
                    capture_receipt.get(field) == value
                    for field, value in _pair_evidence(manifest, resolved).items()
                ),
                capture_receipt.get("eval_manifest_hash") == resolved["eval_manifest_hash"],
                capture_receipt.get("execution_binding_sha256")
                == resolved["execution_binding_sha256"],
                capture_receipt.get("runtime_binding_sha256")
                == resolved["runtime_binding_sha256"],
                capture_receipt.get("current_binding_sha256") == current_sha,
                capture_receipt.get("status") == "PASS",
                capture_receipt.get("decision") == "COMMIT_RETAIN_CAPTURE_COMPLETE",
                capture_receipt.get("training_authorized") is False,
                capture_receipt.get("method_selected") is False,
            )
        ):
            raise ValueError("append-only supervisor capture receipt differs from artifacts")
    return {
        **report,
        "git_commit": expected_git_commit(),
        "run_id": manifest["run_id"],
        **_pair_evidence(manifest, resolved),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": current_sha,
        "capture_ledger": str(capture_path.resolve()),
        "capture_ledger_sha256": sha256_file(capture_path),
        "capture_run_receipt": str(run_receipt_path.resolve()),
        "capture_run_receipt_sha256": sha256_file(run_receipt_path),
    }


def build_final_audit_report(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    capture = validate_capture_artifacts(
        manifest_path, require_supervisor_capture_receipt=True
    )
    supervisor = read_jsonl(manifest["paths"]["execution_ledger"])
    prefix = supervisor[:3]
    report = {
        "schema": "memagent.commit-retain.capture-final-audit.v1",
        "status": "PASS",
        "decision": "COMMIT_RETAIN_CAPTURE_AUDIT_COMPLETE",
        **{key: capture[key] for key in (
            "git_commit", "run_id", "eval_manifest_hash", "execution_binding_sha256",
            "gpu_pair_slug", "physical_gpu_whitelist", "visible_devices",
            "physical_gpu_identity",
            "runtime_binding_sha256", "current_binding_sha256", "capture_ledger",
            "capture_ledger_sha256", "capture_run_receipt", "capture_run_receipt_sha256",
            "pair_count", "stable_write_ids", "pair_ids", "generate_call_count", "outcomes",
            "training", "claim_boundary",
        )},
        "supervisor_prefix_record_count": 3,
        "supervisor_prefix_sha256": canonical_sha256(prefix),
        "native_memory_interface_evidence": validate_p0(manifest_path)[1][
            "execution_binding"
        ]["native_memory_interface_evidence"],
    }
    record_types = [item.get("record_type") for item in supervisor]
    if record_types not in (
        ["s0_preflight", "capture_authorization", "capture_complete"],
        ["s0_preflight", "capture_authorization", "capture_complete", "audit_result"],
    ):
        raise ValueError("supervisor ledger has an inadmissible terminal sequence")
    if len(supervisor) == 4:
        final_path = Path(manifest["paths"]["final_report"])
        audit = supervisor[3]
        allowed = {
            "record_type", "experiment_name", "git_commit", "run_id", "recorded_at",
            "gpu_pair_slug", "physical_gpu_whitelist", "visible_devices",
            "physical_gpu_identity",
            "eval_manifest_hash", "execution_binding_sha256", "runtime_binding_sha256",
            "current_binding_sha256", "artifact", "artifact_sha256",
            "training_authorized", "method_selected", "status", "decision", "pair_count",
            "pair_ids", "stable_write_ids", "generate_call_count", "record_index",
            "previous_record_sha256", "record_sha256",
        }
        if set(audit) != allowed:
            raise ValueError("terminal audit supervisor receipt has handcrafted fields")
        if not final_path.is_file() or not all(
            (
                audit.get("record_type") == "audit_result",
                audit.get("experiment_name") == experiment_name(manifest),
                audit.get("git_commit") == report["git_commit"],
                audit.get("run_id") == report["run_id"],
                all(audit.get(field) == report[field] for field in (
                    "gpu_pair_slug", "physical_gpu_whitelist", "visible_devices",
                    "physical_gpu_identity",
                )),
                audit.get("eval_manifest_hash") == report["eval_manifest_hash"],
                audit.get("execution_binding_sha256") == report["execution_binding_sha256"],
                audit.get("runtime_binding_sha256") == report["runtime_binding_sha256"],
                audit.get("current_binding_sha256") == report["current_binding_sha256"],
                audit.get("artifact") == str(final_path.resolve()),
                audit.get("artifact_sha256") == sha256_file(final_path),
                audit.get("status") == report["status"],
                audit.get("decision") == report["decision"],
                audit.get("pair_count") == report["pair_count"],
                audit.get("pair_ids") == report["pair_ids"],
                audit.get("stable_write_ids") == report["stable_write_ids"],
                audit.get("generate_call_count") == report["generate_call_count"],
                audit.get("training_authorized") is False,
                audit.get("method_selected") is False,
            )
        ):
            raise ValueError("terminal audit supervisor receipt differs from final report")
    return report


def record_stage(
    manifest_path: Path, *, record_type: str, artifact: Path
) -> dict[str, Any]:
    if record_type not in {"capture_complete", "audit_result"}:
        raise ValueError("unsupported supervisor stage")
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    current_sha = _current_binding(manifest, resolved, full_model_sha=False)
    ledger_path = Path(manifest["paths"]["execution_ledger"])
    existing = read_jsonl(ledger_path)
    if validate_jsonl_chain(existing):
        raise ValueError("existing supervisor ledger chain failed")
    if record_type == "capture_complete":
        expected_artifact = Path(manifest["paths"]["capture_ledger"])
        if artifact.resolve() != expected_artifact.resolve():
            raise ValueError("capture stage artifact path drifted")
        if [row.get("record_type") for row in existing] != [
            "s0_preflight", "capture_authorization"
        ]:
            raise ValueError("capture supervisor stage order drifted")
        report = validate_capture_artifacts(
            manifest_path, require_supervisor_capture_receipt=False
        )
        run_receipt = Path(manifest["paths"]["capture_run_receipt"])
        stage_fields = {
            "status": "PASS",
            "decision": "COMMIT_RETAIN_CAPTURE_COMPLETE",
            "pair_count": report["pair_count"],
            "pair_ids": report["pair_ids"],
            "stable_write_ids": report["stable_write_ids"],
            "generate_call_count": report["generate_call_count"],
            "run_receipt": str(run_receipt.resolve()),
            "run_receipt_sha256": sha256_file(run_receipt),
        }
    else:
        expected_artifact = Path(manifest["paths"]["final_report"])
        if artifact.resolve() != expected_artifact.resolve():
            raise ValueError("audit stage artifact path drifted")
        if [row.get("record_type") for row in existing] != [
            "s0_preflight", "capture_authorization", "capture_complete"
        ]:
            raise ValueError("audit supervisor stage order drifted")
        expected = build_final_audit_report(manifest_path)
        actual = json.loads(artifact.read_text(encoding="utf-8"))
        if canonical_json(actual) != canonical_json(expected):
            raise ValueError("final audit JSON was not produced by the read-only recomputation")
        stage_fields = {
            "status": expected["status"],
            "decision": expected["decision"],
            "pair_count": expected["pair_count"],
            "pair_ids": expected["pair_ids"],
            "stable_write_ids": expected["stable_write_ids"],
            "generate_call_count": expected["generate_call_count"],
        }
    if not artifact.is_file():
        raise ValueError("supervisor artifact is missing")
    record = {
        "record_type": record_type,
        "experiment_name": experiment_name(manifest),
        "git_commit": expected_git_commit(),
        "run_id": manifest["run_id"],
        **_pair_evidence(manifest, resolved),
        "recorded_at": utc_now(),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": current_sha,
        "artifact": str(artifact.resolve()),
        "artifact_sha256": sha256_file(artifact),
        "training_authorized": False,
        "method_selected": False,
        **stage_fields,
    }
    append_jsonl(ledger_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_REL)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--write-certificate", action="store_true")
    parser.add_argument("--validate-p0-prefix", action="store_true")
    parser.add_argument("--record-type", choices=("capture_complete", "audit_result"))
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--issue-capture-credential", type=Path)
    parser.add_argument("--issuer-shell-pid", type=int)
    args = parser.parse_args()
    if args.issue_capture_credential is not None:
        if args.issuer_shell_pid is None:
            parser.error("--issue-capture-credential requires --issuer-shell-pid")
        evidence = issue_capture_credential(
            args.manifest,
            output=args.issue_capture_credential,
            issuer_shell_pid=args.issuer_shell_pid,
        )
        print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
        return 0
    if args.validate_p0_prefix:
        _, resolved = validate_p0(args.manifest)
        print(json.dumps({
            "status": "PASS",
            "decision": "COMMIT_RETAIN_P0_PREFIX_VALID",
            "run_id": resolved["run_id"],
            "eval_manifest_hash": resolved["eval_manifest_hash"],
        }, sort_keys=True))
        return 0
    if args.record_type:
        if args.artifact is None:
            parser.error("--record-type requires --artifact")
        record = record_stage(args.manifest, record_type=args.record_type, artifact=args.artifact)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        return 0
    report = (
        write_preflight(args.manifest, check_runtime=args.check_runtime)
        if args.write_certificate
        else run_preflight(args.manifest, check_runtime=args.check_runtime)[0]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
