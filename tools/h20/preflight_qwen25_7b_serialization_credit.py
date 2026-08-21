#!/usr/bin/env python3
"""Fail-closed P0 and append-only event writer for 7B SMSB4/Tetrad4."""

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
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import (  # noqa: E402
    append_jsonl,
    sha256_file,
    validate_jsonl_chain,
)
from recurrent.research.serialization_credit_pilots import (  # noqa: E402
    canonical_sha256,
    center_truncate_token_ids,
    parent_authority_mac,
    validate_parent_launch_receipt,
    read_jsonl,
    require_finite_number,
    require_int,
    require_sha256,
    validate_single_request_token_budget,
    validate_sampling_params,
    write_json_exclusive,
)
from recurrent.research.stable_eval_identity import (  # noqa: E402
    evaluation_trajectory_seed,
    validate_resolved_manifest,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds  # noqa: E402


MANIFEST_REL = "manifests/h20/qwen25_7b_serialization_credit_pilots_seed2026.json"
EXPERIMENT_NAME = "qwen25_7b_serialization_credit_pilots_seed2026"
REQUIRED_ENV = (
    "MEMAGENT_SERIAL_CREDIT_WORK_ROOT",
    "MEMAGENT_SERIAL_CREDIT_REPO_DIR",
    "MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT",
    "MEMAGENT_SERIAL_CREDIT_RUN_ID",
)
CODE_OBJECTS = (
    "recurrent/impls/memory.py",
    "recurrent/utils.py",
    "recurrent/research/serialization_credit_pilots.py",
    "recurrent/research/gate_a_execution.py",
    "recurrent/research/s128_hotpot_metrics.py",
    "recurrent/research/stable_eval_identity.py",
    "recurrent/research/trajectory_seeding.py",
    "verl/utils/torch_functional.py",
    "tools/h20/preflight_qwen25_7b_s128_it.py",
    "tools/h20/preflight_qwen25_7b_serialization_credit.py",
    "tools/h20/launch_qwen25_7b_serialization_credit_child.py",
    "tools/h20/run_qwen25_7b_serialization_credit.py",
    "tools/h20/audit_qwen25_7b_serialization_credit.py",
    "scripts/h20/serialization_credit_pilots_common.sh",
    "scripts/h20/preflight_qwen25_7b_serialization_credit.sh",
    "scripts/h20/run_qwen25_7b_smsb4.sh",
    "scripts/h20/run_qwen25_7b_tetrad4.sh",
    MANIFEST_REL,
    "manifests/h20/qwen25_7b_serialization_credit_pilots_commands.json",
    "serialization_credit_pilot_execution_ledger.schema.json",
    "docs/h20/serialization_credit_pilots_freeze_20260821.md",
)


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
    if not work_root.is_absolute() or not repo.is_absolute():
        raise ValueError("task-scoped runtime paths must be absolute")
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT must be a full Git SHA")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,31}", run_id) is None:
        raise ValueError("MEMAGENT_SERIAL_CREDIT_RUN_ID has an invalid format")
    replacements = {
        "${MEMAGENT_SERIAL_CREDIT_WORK_ROOT}": str(work_root),
        "${MEMAGENT_SERIAL_CREDIT_REPO_DIR}": str(repo),
        "${MEMAGENT_SERIAL_CREDIT_RUN_ID}": run_id,
    }

    def resolve(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: resolve(child) for key, child in item.items()}
        if isinstance(item, list):
            return [resolve(child) for child in item]
        if isinstance(item, str):
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
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return resolve_manifest_environment(raw, environment)


def parent_authority_secret_path(manifest: Mapping[str, Any]) -> Path:
    return Path(manifest["paths"]["parent_authority_secret"]).resolve()


def _write_parent_authority_secret(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, secret)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return secret


def load_parent_authority_secret(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any]
) -> bytes:
    path = parent_authority_secret_path(manifest)
    authority = resolved.get("parent_receipt_authority", {})
    if path != Path(str(authority.get("secret_path", ""))).resolve():
        raise ValueError("parent authority secret path differs from P0")
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("parent authority secret is missing or not mode 0600")
    secret = path.read_bytes()
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != authority.get(
        "secret_sha256"
    ):
        raise ValueError("parent authority secret differs from P0")
    if authority.get("scheme") != "hmac-sha256-parent-receipt-v2":
        raise ValueError("parent authority scheme differs from P0")
    return secret


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return parquet.read_table(
        path, columns=["prompt", "context", "reward_model", "extra_info"]
    ).to_pylist()


def _question(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt")
    if (
        not isinstance(prompt, list)
        or len(prompt) != 1
        or prompt[0].get("role") != "user"
        or not isinstance(prompt[0].get("content"), str)
    ):
        raise ValueError("fixed S128 row has an unexpected prompt")
    return str(prompt[0]["content"])


def select_pilot_rows(
    *,
    parquet_rows: list[Mapping[str, Any]],
    stable_rows: list[Mapping[str, Any]],
    tokenizer: Any,
    writer_prompt_builder: Any,
    sorted_positions: list[int],
    eval_manifest_hash: str,
    base_seed: int = 2026,
    chunk_size: int = 5000,
) -> list[dict[str, Any]]:
    """Outcome-blind BCI-derived prompt-length-strata selection inside S128."""
    if len(parquet_rows) != 128 or len(stable_rows) != 128:
        raise ValueError("pilot selection requires the complete existing S128")
    base_seed = require_int(base_seed, "base_seed", minimum=0)
    chunk_size = require_int(chunk_size, "chunk_size", minimum=1)
    eval_manifest_hash = require_sha256(eval_manifest_hash, "eval_manifest_hash")
    if not isinstance(sorted_positions, list):
        raise ValueError("pilot sorted positions must be an integer array")
    sorted_positions = [
        require_int(value, f"sorted_positions[{index}]", minimum=0)
        for index, value in enumerate(sorted_positions)
    ]
    stable_by_raw = {
        require_int(row["raw_row_position"], "raw_row_position", minimum=0): row
        for row in stable_rows
    }
    candidates: list[dict[str, Any]] = []
    no_memory = tokenizer.encode("No previous memory", add_special_tokens=False)
    for raw_position, source in enumerate(parquet_rows):
        stable = stable_by_raw.get(raw_position)
        if stable is None:
            raise ValueError(f"stable identity has no row for raw position {raw_position}")
        row_eval_manifest_hash = stable.get("eval_manifest_hash")
        if (
            row_eval_manifest_hash is not None
            and row_eval_manifest_hash != eval_manifest_hash
        ):
            raise ValueError(
                "stable identity row evaluation hash disagrees with the resolved "
                f"manifest at raw row {raw_position}"
            )
        question = _question(source)
        context = str(source.get("context"))
        ground_truth = source.get("reward_model", {}).get("ground_truth")
        if canonical_sha256(ground_truth) != stable["ground_truth_hash"]:
            raise ValueError(f"ground-truth hash drift at raw row {raw_position}")
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != stable["source_question_hash"]:
            raise ValueError(f"question hash drift at raw row {raw_position}")
        if hashlib.sha256(context.encode("utf-8")).hexdigest() != stable["source_context_hash"]:
            raise ValueError(f"context hash drift at raw row {raw_position}")
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        context_ids = center_truncate_token_ids(
            list(tokenizer.encode(context, add_special_tokens=False)), 40000
        )
        first_chunk = context_ids[:chunk_size]
        writer_prompt_ids = list(writer_prompt_builder(question_ids, no_memory, first_chunk))
        trajectory_seed = evaluation_trajectory_seed(
            base_seed=base_seed,
            eval_manifest_hash=eval_manifest_hash,
            example_id=str(stable["example_id"]),
            source_order_index=require_int(
                stable["source_order_index"], "source_order_index", minimum=0
            ),
            replica_id=0,
        )
        candidates.append(
            {
                **dict(stable),
                # Stable-I deliberately stores this binding once at the resolved
                # manifest top level; identity_payload.rows contain only the
                # row-local fields committed by that hash.  Downstream pilot
                # records are executable identities and therefore must carry the
                # top-level binding explicitly.
                "eval_manifest_hash": eval_manifest_hash,
                "writer_turn0_prompt_token_length": len(writer_prompt_ids),
                "writer_turn0_prompt_token_sha256": canonical_sha256(writer_prompt_ids),
                "trajectory_seed": trajectory_seed,
                "writer_turn0_request_seed": derive_turn_request_seeds(
                    [trajectory_seed], [0], 0
                )[0],
            }
        )
    ordered = sorted(
        candidates,
        key=lambda row: (
            int(row["writer_turn0_prompt_token_length"]),
            int(row["source_order_index"]),
        ),
    )
    if sorted_positions != [15, 47, 79, 111]:
        raise ValueError("pilot selection strata drifted")
    selected = [ordered[position] for position in sorted_positions]
    if len({row["example_id"] for row in selected}) != 4:
        raise ValueError("pilot selection did not produce four unique examples")
    return selected


def build_generation_capacity_audit(
    *,
    selected: list[dict[str, Any]],
    parquet_rows: list[dict[str, Any]],
    tokenizer: Any,
    writer_template: Any,
    final_template: Any,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Prove at P0 that every possible frozen writer/final call fits vLLM."""
    recurrent = manifest["recurrent"]
    backend = manifest["backend"]
    no_memory = list(
        tokenizer.encode(
            recurrent["no_memory_text"], add_special_tokens=False
        )
    )
    maximum_memory = [0] * require_int(
        recurrent["max_memory_tokens"],
        "recurrent.max_memory_tokens",
        minimum=1,
    )
    chunk_size = require_int(
        recurrent["chunk_size"], "recurrent.chunk_size", minimum=1
    )
    audit_rows: list[dict[str, Any]] = []
    for pilot in selected:
        raw_position = require_int(
            pilot["raw_row_position"], "pilot.raw_row_position", minimum=0
        )
        source = parquet_rows[raw_position]
        question_ids = list(
            tokenizer.encode(_question(source), add_special_tokens=False)
        )
        context_ids = center_truncate_token_ids(
            list(
                tokenizer.encode(
                    str(source["context"]), add_special_tokens=False
                )
            ),
            recurrent["max_context_tokens"],
        )
        writer_turns: list[dict[str, Any]] = []
        for turn, offset in enumerate(range(0, len(context_ids), chunk_size)):
            chunk_ids = context_ids[offset : offset + chunk_size]
            memory_upper_bound = no_memory if turn == 0 else maximum_memory
            prompt_ids = writer_template.format(
                prompt=question_ids,
                memory=memory_upper_bound,
                chunk=chunk_ids,
            ).tolist()
            budget = validate_single_request_token_budget(
                prompt_ids,
                recurrent["max_memory_tokens"],
                max_model_len=backend["max_model_len"],
                max_num_batched_tokens=backend["max_num_batched_tokens"],
            )
            if (
                turn == 0
                and canonical_sha256(prompt_ids)
                != pilot["writer_turn0_prompt_token_sha256"]
            ):
                raise ValueError("P0 writer turn0 capacity prompt differs from selection")
            writer_turns.append(
                {
                    "turn": turn,
                    "chunk_start": offset,
                    "chunk_end": offset + len(chunk_ids),
                    "input_memory_token_upper_bound": len(memory_upper_bound),
                    **budget,
                }
            )
        if not writer_turns or len(writer_turns) > recurrent["max_chunks"]:
            raise ValueError("P0 capacity audit writer turn count is invalid")
        final_prompt_ids = final_template.format(
            prompt=question_ids, memory=maximum_memory
        ).tolist()
        final_budget = validate_single_request_token_budget(
            final_prompt_ids,
            recurrent["max_final_tokens"],
            max_model_len=backend["max_model_len"],
            max_num_batched_tokens=backend["max_num_batched_tokens"],
        )
        audit_rows.append(
            {
                "example_id": str(pilot["example_id"]),
                "writer_turns": writer_turns,
                "final_reader_upper_bound": final_budget,
            }
        )
    if len(audit_rows) != 4:
        raise ValueError("P0 capacity audit requires exactly four pilot examples")
    return audit_rows


def _model_loading_paths(model_root: Path) -> list[str]:
    exact = {
        "config.json", "generation_config.json", "adapter_config.json",
        "quantization_config.json", "preprocessor_config.json", "processor_config.json",
        "special_tokens_map.json", "added_tokens.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
    }
    suffixes = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx")
    return sorted(
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file()
        and (
            path.name.lower() in exact
            or path.name.lower().endswith(suffixes)
            or path.name.lower().endswith(".index.json")
        )
    )


def _stable_prerequisite(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.h20.preflight_qwen25_7b_s128_it import _stable_canary_contract

    spec = dict(manifest["stable_identity_prerequisite"])
    contract = _stable_canary_contract({"stable_identity_canary": spec})
    resolved = validate_resolved_manifest(
        json.loads(Path(spec["resolved_manifest"]).read_text(encoding="utf-8"))
    )
    if resolved["eval_manifest_hash"] != spec["required_eval_manifest_hash"]:
        raise ValueError("stable-I resolved manifest hash differs from the frozen r2 PASS")
    if len(resolved["identity_payload"]["rows"]) != 128:
        raise ValueError("stable-I resolved manifest does not bind all 128 rows")
    return contract, resolved


def _runtime_versions(python: Path, repo: Path) -> dict[str, str]:
    completed = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as md,json,pyarrow,sys,torch,transformers,vllm;"
                "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,"
                "'torch_cuda':torch.version.cuda,'transformers':transformers.__version__,"
                "'vllm':vllm.__version__,'pyarrow':pyarrow.__version__,"
                "'jsonschema':md.version('jsonschema')},sort_keys=True))"
            ),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise ValueError(f"runtime imports failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _validate_numeric_contract(manifest: Mapping[str, Any]) -> None:
    gpu = manifest["gpu"]
    whitelist = gpu.get("physical_whitelist")
    if not isinstance(whitelist, list) or [
        require_int(value, f"gpu.physical_whitelist[{index}]", minimum=0)
        for index, value in enumerate(whitelist)
    ] != [2, 3]:
        raise ValueError("GPU physical whitelist must be the strict integer array [2, 3]")
    if gpu.get("visible_devices") != "2,3":
        raise ValueError("GPU visible_devices must be exactly 2,3")
    if require_int(gpu.get("tensor_parallel_size"), "gpu.tensor_parallel_size", minimum=1) != 2:
        raise ValueError("tensor parallel size must be 2")
    if require_int(gpu.get("max_num_seqs"), "gpu.max_num_seqs", minimum=1) != 1:
        raise ValueError("GPU max_num_seqs must be 1")
    if gpu.get("one_prompt_per_generate_call") is not True:
        raise ValueError("one_prompt_per_generate_call must be true")
    if gpu.get("cuda_device_order") != "PCI_BUS_ID":
        raise ValueError("gpu.cuda_device_order must be PCI_BUS_ID")
    backend = manifest["backend"]
    if (
        backend.get("name") != "vllm"
        or backend.get("required_version") != "0.8.2"
        or backend.get("VLLM_USE_V1") != "0"
    ):
        raise ValueError("backend must be strict vLLM 0.8.2 with VLLM_USE_V1=0")
    if manifest.get("model", {}).get("id") != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("mechanism pilot model must be Qwen2.5-7B-Instruct")
    for field, minimum in (
        ("swap_space_gib", 0), ("max_model_len", 1),
        ("max_num_batched_tokens", 1), ("max_num_seqs", 1), ("engine_seed", 0),
    ):
        require_int(backend.get(field), f"backend.{field}", minimum=minimum)
    require_finite_number(
        backend.get("gpu_memory_utilization"),
        "backend.gpu_memory_utilization",
        minimum=0.0,
        maximum=1.0,
    )
    for field, expected in (
        ("strict_vllm", True), ("allow_huggingface_generation_fallback", False),
        ("enable_prefix_caching", False), ("enforce_eager", True),
        ("disable_custom_all_reduce", True),
    ):
        if backend.get(field) is not expected:
            raise ValueError(f"backend.{field} differs from the frozen boolean contract")
    recurrent = manifest["recurrent"]
    for field in (
        "max_context_tokens", "chunk_size", "max_chunks", "max_memory_tokens",
        "max_final_tokens",
    ):
        require_int(recurrent.get(field), f"recurrent.{field}", minimum=1)
    if recurrent.get("context_truncation") != "center":
        raise ValueError("recurrent context truncation must be center")
    if require_int(manifest["data"].get("expected_rows"), "data.expected_rows", minimum=1) != 128:
        raise ValueError("fixed S128 must contain 128 expected rows")
    for name in (
        "capture_writer_decode", "capture_final_stochastic_decode",
        "capture_final_deterministic_control",
    ):
        validate_sampling_params(manifest["smsb"].get(name), f"smsb.{name}")
    if require_int(manifest["smsb"].get("examples"), "smsb.examples", minimum=1) != 4:
        raise ValueError("SMSB pilot must freeze exactly four examples")
    if require_int(
        manifest["smsb"].get("fresh_replay_processes"),
        "smsb.fresh_replay_processes",
        minimum=1,
    ) != 12:
        raise ValueError("SMSB pilot must freeze exactly twelve fresh replay processes")
    if manifest["smsb"].get("replay_regimes") != [
        "temperature_zero", "matched_seed", "independent_seed"
    ]:
        raise ValueError("SMSB replay regimes/order drifted")
    if manifest["smsb"].get("L2_is_report_only") is not True:
        raise ValueError("SMSB L2 must remain report-only")
    tetrad = manifest["tetrad"]
    if require_int(tetrad.get("examples"), "tetrad.examples", minimum=0) != 4:
        raise ValueError("Tetrad pilot must freeze exactly four examples")
    if require_int(tetrad.get("requests"), "tetrad.requests", minimum=0) != 20:
        raise ValueError("Tetrad pilot must freeze exactly twenty requests")
    require_int(
        tetrad.get("maximum_shuffle_memory_token_caliper"),
        "tetrad.maximum_shuffle_memory_token_caliper",
        minimum=0,
    )
    if tetrad.get("states") != ["generated", "empty", "irrelevant", "shuffle", "gold"]:
        raise ValueError("Tetrad five-state contract/order drifted")
    if tetrad.get("requires_smsb_decision") != "PASS_E_DET_SINGLE_REQUEST":
        raise ValueError("Tetrad must remain gated by the exact SMSB E_det PASS")
    for field, expected in (
        ("fresh_process_per_request", True),
        ("deterministic_reader", True),
        ("effects_reportable", False),
        ("audit32_not_started", True),
    ):
        if tetrad.get(field) is not expected:
            raise ValueError(f"tetrad.{field} differs from the frozen boolean contract")
    if require_finite_number(tetrad.get("temperature"), "tetrad.temperature") != 0.0:
        raise ValueError("Tetrad reader must be deterministic temperature zero")
    scope = manifest.get("scope", {})
    if (
        scope.get("training") is not False
        or require_int(scope.get("actor_updates"), "scope.actor_updates", minimum=0) != 0
        or scope.get("paper_performance_result") is not False
        or scope.get("training_authorized") is not False
        or scope.get("method_selection_status") != "PENDING_EVIDENCE_NO_SELECTION"
    ):
        raise ValueError("pilot scope must forbid training, effects, and method selection")
    require_finite_number(
        manifest["tetrad"].get("canonical_competence_score_threshold_f1"),
        "tetrad.canonical_competence_score_threshold_f1",
        minimum=0.0,
        maximum=1.0,
    )
    require_finite_number(
        manifest["tetrad"].get("canonical_competence_rate_floor"),
        "tetrad.canonical_competence_rate_floor",
        minimum=0.0,
        maximum=1.0,
    )


def _file_stat(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to is not None else str(path.resolve()),
        "size": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def capture_lightweight_current_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Cheap per-process TOCTOU binding; model bytes are fully re-hashed at stage boundaries."""
    repo = Path(manifest["repository"]).resolve()
    model_root = Path(manifest["model"]["path"])
    data_path = Path(manifest["data"]["validation"])
    loading_paths = _model_loading_paths(model_root)
    expected_paths = sorted(item["path"] for item in manifest["model"]["files"])
    if loading_paths != expected_paths:
        raise ValueError("current model loading-relevant path inventory differs from P0")
    try:
        import importlib.metadata
        import pyarrow
        import sys
        import torch
        import transformers
        import vllm
    except Exception as error:
        raise ValueError(f"current runtime import failed: {error}") from error
    return {
        "git_commit": git(repo, "rev-parse", "HEAD"),
        "git_branch": git(repo, "branch", "--show-current"),
        "worktree_clean": not bool(git(repo, "status", "--porcelain")),
        "model_loading_relevant_paths": loading_paths,
        "model_file_stats": [
            _file_stat(model_root / relative, relative_to=model_root)
            for relative in loading_paths
        ],
        "validation_data_stat": _file_stat(data_path),
        "validation_data_sha256": sha256_file(data_path),
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "pyarrow": pyarrow.__version__,
            "jsonschema": importlib.metadata.version("jsonschema"),
        },
    }


def verify_current_binding(
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    *,
    full_model_sha: bool,
) -> str:
    current = capture_lightweight_current_binding(manifest)
    current_sha = canonical_sha256(current)
    if current_sha != resolved.get("lightweight_current_binding_sha256"):
        raise ValueError(
            "current repository/data/model-stat/runtime binding differs from P0: "
            f"{current_sha} != {resolved.get('lightweight_current_binding_sha256')}"
        )
    if full_model_sha:
        model_root = Path(manifest["model"]["path"])
        actual = [
            {
                "path": expected["path"],
                "size": (model_root / expected["path"]).stat().st_size,
                "sha256": sha256_file(model_root / expected["path"]),
            }
            for expected in manifest["model"]["files"]
        ]
        if actual != manifest["model"]["files"]:
            raise ValueError("full current model SHA inventory differs from P0")
        if canonical_sha256(actual) != resolved["execution_binding"]["model_manifest_sha256"]:
            raise ValueError("full current model manifest digest differs from P0")
    return current_sha


def run_preflight(
    manifest_path: Path, *, check_runtime: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    failures: list[str] = []
    if not check_runtime:
        failures.append("formal P0 requires --check-runtime")
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        manifest = load_manifest(manifest_path)
    except Exception as error:
        return {
            "gate": "P0", "status": "FAIL", "decision": "SERIAL_CREDIT_NO_GO:P0",
            "failures": [str(error)], "evidence": {},
        }, None
    try:
        _validate_numeric_contract(manifest)
    except Exception as error:
        failures.append(f"strict numeric/runtime contract failed: {error}")
    repo = Path(manifest["repository"]).resolve()
    expected_commit = os.environ["MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT"]
    evidence: dict[str, Any] = {
        "frozen_manifest_sha256": sha256_file(manifest_path),
        "expected_git_commit": expected_commit,
        "run_id": manifest["run_id"],
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES") != manifest["gpu"]["visible_devices"]:
        failures.append("P0 CUDA_VISIBLE_DEVICES differs from frozen physical whitelist")
    if os.environ.get("CUDA_DEVICE_ORDER") != manifest["gpu"]["cuda_device_order"]:
        failures.append("P0 CUDA_DEVICE_ORDER differs from PCI_BUS_ID")
    try:
        if repo != REPO_ROOT.resolve():
            failures.append(f"invoked checkout differs from explicit repository: {REPO_ROOT} != {repo}")
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
            failures.append("sources/ changed on the mechanism pilot branch")
    except Exception as error:
        failures.append(f"Git closure failed: {error}")

    for path_name in CODE_OBJECTS:
        path = repo / path_name
        if not path.is_file():
            failures.append(f"required code object is missing: {path_name}")
    code_hashes = {
        path_name: sha256_file(repo / path_name)
        for path_name in CODE_OBJECTS
        if (repo / path_name).is_file()
    }
    evidence["execution_code_sha256"] = code_hashes
    commands_path = repo / manifest["command_manifest"]
    schema_path = repo / manifest["ledger_schema"]
    try:
        commands = json.loads(commands_path.read_text(encoding="utf-8"))
        json.loads(schema_path.read_text(encoding="utf-8"))
        if commands["required_sequence"] != [
            "p0", "smsb_capture", "smsb_12_fresh_replays", "smsb_adjudication",
            "tetrad_construct_if_and_only_if_smsb_pass", "tetrad_20_fresh_requests",
            "tetrad_adjudication", "final_audit",
        ]:
            failures.append("command sequence drifted")
        if commands.get("branch") != raw_manifest["branch"]:
            failures.append("command manifest branch differs from the frozen branch")
        if commands.get("gpu_execution_authorized_by_this_manifest") is not False:
            failures.append("command manifest improperly self-authorizes GPU execution")
        command_execution = commands.get("execution", {})
        if (
            command_execution.get("physical_gpus") != manifest["gpu"]["physical_whitelist"]
            or command_execution.get("visible_devices") != manifest["gpu"]["visible_devices"]
        ):
            failures.append("command manifest GPU binding differs from the frozen manifest")
        for field in (
            "full_model_sha_per_fresh_child",
            "actual_gpu_uuid_name_bound_per_child",
            "parent_issued_single_use_credential_per_child",
            "parent_hmac_authenticated_receipt_per_child",
            "parent_observed_child_pid_ppid_exit_and_stdout_sha",
            "post_child_full_model_sha_per_child",
            "post_child_gpu_identity_query_per_child",
            "p0_and_per_call_token_capacity_gate",
            "writer_turn_token_chain_independently_rebuilt",
            "automatic_post_write_readonly_reaudit",
            "unique_parent_supervisor_pid_required",
            "unique_child_pid_required",
        ):
            if command_execution.get(field) is not True:
                failures.append(f"command manifest execution.{field} is not frozen true")
        if commands.get("claim_firewall", {}).get("tetrad4_effects_reportable") is not False:
            failures.append("command manifest improperly permits Tetrad4 effects")
    except Exception as error:
        failures.append(f"command/schema contract cannot be loaded: {error}")

    model_root = Path(manifest["model"]["path"])
    actual_model_files: list[dict[str, Any]] = []
    for expected in manifest["model"]["files"]:
        path = model_root / expected["path"]
        if not path.is_file():
            failures.append(f"frozen model file is missing: {path}")
            continue
        actual = {"path": expected["path"], "size": path.stat().st_size, "sha256": sha256_file(path)}
        actual_model_files.append(actual)
        if actual != expected:
            failures.append(f"frozen model file drifted: {expected['path']}")
    if model_root.is_dir() and _model_loading_paths(model_root) != sorted(
        item["path"] for item in manifest["model"]["files"]
    ):
        failures.append("model loading-relevant file inventory drifted")
    evidence["model_file_inventory"] = actual_model_files
    model_manifest_sha = canonical_sha256(actual_model_files)
    tokenizer_manifest_sha = canonical_sha256(
        [
            item
            for item in actual_model_files
            if item["path"] in {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
        ]
    )
    evidence["model_manifest_sha256"] = model_manifest_sha
    evidence["tokenizer_manifest_sha256"] = tokenizer_manifest_sha
    data_path = Path(manifest["data"]["validation"])
    if not data_path.is_file():
        failures.append(f"fixed S128 parquet is missing: {data_path}")
    elif sha256_file(data_path) != manifest["data"]["validation_sha256"]:
        failures.append("fixed S128 parquet SHA-256 drifted")
    else:
        evidence["validation_data_sha256"] = sha256_file(data_path)

    stable_contract: dict[str, Any] | None = None
    stable_resolved: dict[str, Any] | None = None
    try:
        stable_contract, stable_resolved = _stable_prerequisite(manifest)
        evidence["stable_identity_prerequisite"] = stable_contract
    except Exception as error:
        failures.append(f"stable-I prerequisite failed: {error}")

    python = Path(manifest["python"])
    if not python.is_file():
        failures.append(f"frozen Python is missing: {python}")
    runtime_versions: dict[str, str] = {}
    gpu_identity: list[str] = []
    if check_runtime and python.is_file():
        try:
            runtime_versions = _runtime_versions(python, repo)
            if runtime_versions.get("vllm") != manifest["backend"]["required_version"]:
                failures.append(f"vLLM version drifted: {runtime_versions.get('vllm')}")
        except Exception as error:
            failures.append(str(error))
        completed = subprocess.run(
            [
                "nvidia-smi", "-i", manifest["gpu"]["visible_devices"],
                "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            failures.append(f"cannot identify GPU2-3: {completed.stderr.strip()}")
        else:
            gpu_identity = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if len(gpu_identity) != 2 or any("NVIDIA H20" not in line for line in gpu_identity):
                failures.append(f"GPU2-3 are not both NVIDIA H20: {gpu_identity}")
    evidence["runtime_versions"] = runtime_versions
    evidence["physical_gpu_identity"] = gpu_identity

    lightweight_current_binding: dict[str, Any] | None = None
    if check_runtime and not failures:
        try:
            lightweight_current_binding = capture_lightweight_current_binding(manifest)
            if lightweight_current_binding["git_commit"] != expected_commit:
                raise ValueError("current binding Git commit differs from expected commit")
            if lightweight_current_binding["git_branch"] != raw_manifest["branch"]:
                raise ValueError("current binding branch differs from frozen branch")
            if lightweight_current_binding["worktree_clean"] is not True:
                raise ValueError("current binding worktree is dirty")
            if lightweight_current_binding["runtime_versions"].get("vllm") != manifest[
                "backend"
            ]["required_version"]:
                raise ValueError("current binding vLLM version differs")
            evidence["lightweight_current_binding_sha256"] = canonical_sha256(
                lightweight_current_binding
            )
        except Exception as error:
            failures.append(f"cannot freeze current TOCTOU binding: {error}")

    resolved: dict[str, Any] | None = None
    if not failures and stable_resolved is not None and lightweight_current_binding is not None:
        try:
            from transformers import AutoTokenizer
            from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
            from recurrent.utils import TokenTemplate, chat_template

            tokenizer = AutoTokenizer.from_pretrained(
                str(model_root), trust_remote_code=True, local_files_only=True
            )
            writer_template_text = chat_template(tokenizer).format(message=TEMPLATE)
            writer_template = TokenTemplate(writer_template_text, tokenizer)
            final_template_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
            final_template = TokenTemplate(final_template_text, tokenizer)

            def writer_builder(question_ids: list[int], memory_ids: list[int], chunk_ids: list[int]) -> list[int]:
                return writer_template.format(
                    prompt=question_ids, memory=memory_ids, chunk=chunk_ids
                ).tolist()

            parquet_rows = _load_parquet_rows(data_path)
            stable_rows = stable_resolved["identity_payload"]["rows"]
            selected = select_pilot_rows(
                parquet_rows=parquet_rows,
                stable_rows=stable_rows,
                tokenizer=tokenizer,
                writer_prompt_builder=writer_builder,
                sorted_positions=list(manifest["data"]["pilot_selection"]["sorted_positions"]),
                eval_manifest_hash=stable_resolved["eval_manifest_hash"],
                base_seed=int(manifest["backend"]["engine_seed"]),
                chunk_size=int(manifest["recurrent"]["chunk_size"]),
            )
            generation_capacity_audit = build_generation_capacity_audit(
                selected=selected,
                parquet_rows=parquet_rows,
                tokenizer=tokenizer,
                writer_template=writer_template,
                final_template=final_template,
                manifest=manifest,
            )
            engine_config = {
                **dict(manifest["backend"]),
                "physical_gpu_whitelist": manifest["gpu"]["physical_whitelist"],
                "visible_devices": manifest["gpu"]["visible_devices"],
                "tensor_parallel_size": manifest["gpu"]["tensor_parallel_size"],
                "one_prompt_per_generate_call": manifest["gpu"]["one_prompt_per_generate_call"],
            }
            runtime_binding = {
                "git_commit": evidence["git_commit"],
                "branch": evidence["git_branch"],
                "worktree_clean": evidence["worktree_clean"],
                "model_manifest_sha256": model_manifest_sha,
                "tokenizer_manifest_sha256": tokenizer_manifest_sha,
                "validation_data_sha256": evidence["validation_data_sha256"],
                "runtime_versions": runtime_versions,
                "physical_gpu_identity": gpu_identity,
            }
            execution_binding = {
                "git_commit": evidence["git_commit"],
                "eval_manifest_hash": stable_resolved["eval_manifest_hash"],
                "pilot_selection": manifest["data"]["pilot_selection"],
                "pilot_rows": selected,
                "model_manifest_sha256": model_manifest_sha,
                "tokenizer_manifest_sha256": tokenizer_manifest_sha,
                "writer_prompt_template_sha256": hashlib.sha256(
                    writer_template_text.encode("utf-8")
                ).hexdigest(),
                "final_prompt_template_sha256": hashlib.sha256(
                    final_template_text.encode("utf-8")
                ).hexdigest(),
                "engine_config": engine_config,
                "engine_config_sha256": canonical_sha256(engine_config),
                "generation_capacity_audit": generation_capacity_audit,
                "generation_capacity_audit_sha256": canonical_sha256(
                    generation_capacity_audit
                ),
                "recurrent": manifest["recurrent"],
                "smsb": manifest["smsb"],
                "tetrad": manifest["tetrad"],
                "execution_code_sha256": code_hashes,
                "execution_code_combined_sha256": canonical_sha256(code_hashes),
            }
            resolved = {
                "schema_version": 1,
                "frozen_manifest_sha256": evidence["frozen_manifest_sha256"],
                "run_id": manifest["run_id"],
                "eval_manifest_hash": stable_resolved["eval_manifest_hash"],
                "stable_identity_resolved_manifest_sha256": sha256_file(
                    manifest["stable_identity_prerequisite"]["resolved_manifest"]
                ),
                "pilot_rows": selected,
                "runtime_binding": runtime_binding,
                "runtime_binding_sha256": canonical_sha256(runtime_binding),
                "lightweight_current_binding": lightweight_current_binding,
                "lightweight_current_binding_sha256": canonical_sha256(
                    lightweight_current_binding
                ),
                "execution_binding": execution_binding,
                "execution_binding_sha256": canonical_sha256(execution_binding),
            }
            evidence.update(
                eval_manifest_hash=resolved["eval_manifest_hash"],
                pilot_example_ids=[row["example_id"] for row in selected],
                pilot_source_order_indices=[row["source_order_index"] for row in selected],
                runtime_binding_sha256=resolved["runtime_binding_sha256"],
                execution_binding_sha256=resolved["execution_binding_sha256"],
                generation_capacity_audit_sha256=execution_binding[
                    "generation_capacity_audit_sha256"
                ],
            )
        except Exception as error:
            failures.append(f"cannot freeze pilot execution binding: {error}")
            resolved = None

    status = "PASS" if not failures and resolved is not None else "FAIL"
    return {
        "gate": "P0",
        "status": status,
        "decision": "SERIAL_CREDIT_P0_PASS" if status == "PASS" else "SERIAL_CREDIT_NO_GO:P0",
        "failures": failures,
        "evidence": evidence,
        "scope": manifest["scope"],
    }, resolved


def write_preflight(
    manifest_path: Path, *, check_runtime: bool
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    p0_path = Path(manifest["paths"]["p0_certificate"])
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    ledger_path = Path(manifest["paths"]["execution_ledger"])
    authority_path = parent_authority_secret_path(manifest)
    if any(
        path.exists() for path in (p0_path, resolved_path, ledger_path, authority_path)
    ):
        raise FileExistsError("refuse to overwrite append-only P0 evidence")
    report, resolved = run_preflight(manifest_path, check_runtime=check_runtime)
    if resolved is not None:
        authority_secret = _write_parent_authority_secret(authority_path)
        authority_binding = {
            "scheme": "hmac-sha256-parent-receipt-v2",
            "secret_path": str(authority_path),
            "secret_sha256": hashlib.sha256(authority_secret).hexdigest(),
        }
        resolved["parent_receipt_authority"] = authority_binding
        report["evidence"]["parent_receipt_authority"] = authority_binding
        write_json_exclusive(resolved_path, resolved)
        report["evidence"]["resolved_manifest_path"] = str(resolved_path.resolve())
        report["evidence"]["resolved_manifest_sha256"] = sha256_file(resolved_path)
    write_json_exclusive(p0_path, report)
    if report["status"] == "PASS" and resolved is not None:
        append_jsonl(
            ledger_path,
            {
                "record_type": "s0_preflight",
                "experiment_name": EXPERIMENT_NAME,
                "git_commit": report["evidence"]["git_commit"],
                "run_id": manifest["run_id"],
                "recorded_at": utc_now(),
                "eval_manifest_hash": resolved["eval_manifest_hash"],
                "execution_binding_sha256": resolved["execution_binding_sha256"],
                "runtime_binding_sha256": resolved["runtime_binding_sha256"],
                "current_binding_sha256": resolved["lightweight_current_binding_sha256"],
                "artifact": str(p0_path.resolve()),
                "artifact_sha256": sha256_file(p0_path),
                "resolved_manifest": str(resolved_path.resolve()),
                "resolved_manifest_sha256": sha256_file(resolved_path),
                "parent_authority_secret_path": str(authority_path),
                "parent_authority_secret_sha256": resolved[
                    "parent_receipt_authority"
                ]["secret_sha256"],
                "status": "PASS",
                "decision": "SERIAL_CREDIT_P0_PASS",
                "training_authorized": False,
                "method_selection_status": "PENDING_EVIDENCE_NO_SELECTION",
            },
        )
    return report


def validate_p0(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    p0_path = Path(manifest["paths"]["p0_certificate"])
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    ledger_path = Path(manifest["paths"]["execution_ledger"])
    if not p0_path.is_file() or not resolved_path.is_file() or not ledger_path.is_file():
        raise ValueError("P0 certificate, resolved manifest, or execution ledger is missing")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    records = read_jsonl(ledger_path)
    failures = validate_jsonl_chain(records)
    if failures:
        raise ValueError(f"execution ledger hash chain failed: {failures}")
    if not records:
        raise ValueError("execution ledger is empty")
    head = records[0]
    authority_path = parent_authority_secret_path(manifest)
    authority = resolved.get("parent_receipt_authority", {})
    expected_commit = os.environ["MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT"]
    valid = all(
        (
            p0.get("status") == "PASS",
            p0.get("decision") == "SERIAL_CREDIT_P0_PASS",
            p0.get("evidence", {}).get("git_commit") == expected_commit,
            p0.get("evidence", {}).get("expected_git_commit") == expected_commit,
            p0.get("evidence", {}).get("frozen_manifest_sha256")
            == sha256_file(manifest_path),
            resolved.get("frozen_manifest_sha256") == sha256_file(manifest_path),
            p0.get("evidence", {}).get("resolved_manifest_sha256") == sha256_file(resolved_path),
            p0.get("evidence", {}).get("eval_manifest_hash") == resolved.get("eval_manifest_hash"),
            p0.get("evidence", {}).get("runtime_binding_sha256") == resolved.get("runtime_binding_sha256"),
            p0.get("evidence", {}).get("execution_binding_sha256") == resolved.get("execution_binding_sha256"),
            p0.get("evidence", {}).get("lightweight_current_binding_sha256")
            == resolved.get("lightweight_current_binding_sha256"),
            canonical_sha256(resolved.get("lightweight_current_binding"))
            == resolved.get("lightweight_current_binding_sha256"),
            canonical_sha256(resolved.get("runtime_binding"))
            == resolved.get("runtime_binding_sha256"),
            canonical_sha256(resolved.get("execution_binding"))
            == resolved.get("execution_binding_sha256"),
            head.get("record_type") == "s0_preflight",
            head.get("git_commit") == expected_commit,
            head.get("run_id") == manifest["run_id"],
            head.get("artifact_sha256") == sha256_file(p0_path),
            head.get("resolved_manifest_sha256") == sha256_file(resolved_path),
            head.get("eval_manifest_hash") == resolved.get("eval_manifest_hash"),
            head.get("runtime_binding_sha256") == resolved.get("runtime_binding_sha256"),
            head.get("execution_binding_sha256") == resolved.get("execution_binding_sha256"),
            head.get("current_binding_sha256") == resolved.get("lightweight_current_binding_sha256"),
            p0.get("evidence", {}).get("parent_receipt_authority") == authority,
            authority.get("scheme") == "hmac-sha256-parent-receipt-v2",
            authority.get("secret_path") == str(authority_path),
            isinstance(authority.get("secret_sha256"), str),
            head.get("parent_authority_secret_path") == str(authority_path),
            head.get("parent_authority_secret_sha256")
            == authority.get("secret_sha256"),
            authority_path.is_file(),
            not bool(authority_path.stat().st_mode & 0o077)
            if authority_path.is_file()
            else False,
        )
    )
    if not valid:
        raise ValueError("P0 certificate prefix authentication failed")
    return p0, resolved


def issue_child_credential(
    manifest_path: Path,
    *,
    output: Path,
    child_kind: str,
    child_identity: str,
    issuer_pid: int,
) -> dict[str, Any]:
    """Issue one HMAC-authenticated credential from the actual parent launcher."""
    if child_kind not in {"smsb_capture", "smsb_replay", "tetrad_replay"}:
        raise ValueError("child credential kind is not a supervised GPU child kind")
    if (
        not isinstance(child_identity, str)
        or not child_identity
        or any(character in child_identity for character in ("\t", "\r", "\n"))
    ):
        raise ValueError("child credential identity is unsafe or empty")
    issuer_pid = require_int(issuer_pid, "issuer_pid", minimum=1)
    if os.getpid() != issuer_pid:
        raise ValueError("credential issuer must be the current parent launcher process")
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    current_binding_sha = verify_current_binding(
        manifest, resolved, full_model_sha=False
    )
    authority_secret = load_parent_authority_secret(manifest, resolved)
    log_root = Path(manifest["paths"]["log_root"]).resolve()
    if not output.resolve().is_relative_to(log_root):
        raise ValueError("child credential path is outside the frozen run root")
    credential = {
        "schema": "memagent.serialization-credit.parent-child-credential.v2",
        "run_id": manifest["run_id"],
        "git_commit": os.environ["MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT"],
        "child_kind": child_kind,
        "child_identity": child_identity,
        "parent_issuer_pid": issuer_pid,
        "issued_at": utc_now(),
        "nonce": secrets.token_hex(32),
        "current_binding_sha256": current_binding_sha,
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "child_full_model_sha_required": True,
        "parent_authority_secret_sha256": hashlib.sha256(
            authority_secret
        ).hexdigest(),
    }
    credential["parent_credential_id"] = canonical_sha256(credential)
    credential["parent_credential_mac"] = parent_authority_mac(
        authority_secret, "child-credential-v2", credential
    )
    write_json_exclusive(output, credential)
    output.chmod(0o600)
    return credential


def _load_child_credential(
    credential_path: Path,
    *,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    current_binding_sha: str,
    child_kind: str,
    child_identity: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not credential_path.is_file():
        raise ValueError("parent-issued child credential is missing")
    log_root = Path(manifest["paths"]["log_root"]).resolve()
    if not credential_path.resolve().is_relative_to(log_root):
        raise ValueError("parent-issued child credential is outside the frozen run root")
    credential = json.loads(credential_path.read_text(encoding="utf-8"))
    credential_id = require_sha256(
        credential.get("parent_credential_id"), "parent_credential_id"
    )
    credential_mac = require_sha256(
        credential.get("parent_credential_mac"), "parent_credential_mac"
    )
    unsigned = dict(credential)
    unsigned.pop("parent_credential_id")
    unsigned.pop("parent_credential_mac")
    if canonical_sha256(unsigned) != credential_id:
        raise ValueError("parent-issued child credential canonical digest differs")
    parent_pid = require_int(
        credential.get("parent_issuer_pid"), "parent_issuer_pid", minimum=1
    )
    valid = all(
        (
            credential.get("schema")
            == "memagent.serialization-credit.parent-child-credential.v2",
            credential.get("run_id") == manifest["run_id"],
            credential.get("git_commit")
            == os.environ["MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT"],
            credential.get("child_kind") == child_kind,
            credential.get("child_identity") == child_identity,
            credential.get("current_binding_sha256") == current_binding_sha,
            credential.get("runtime_binding_sha256")
            == resolved["runtime_binding_sha256"],
            credential.get("execution_binding_sha256")
            == resolved["execution_binding_sha256"],
            credential.get("child_full_model_sha_required") is True,
            credential.get("parent_authority_secret_sha256")
            == resolved["parent_receipt_authority"]["secret_sha256"],
            isinstance(credential.get("nonce"), str),
            re.fullmatch(r"[0-9a-f]{64}", credential.get("nonce", ""))
            is not None,
        )
    )
    if not valid:
        raise ValueError("parent-issued child credential binding differs")
    evidence = {
        "parent_credential_id": credential_id,
        "parent_credential_mac": credential_mac,
        "parent_credential_sha256": sha256_file(credential_path),
        "parent_credential_path": str(credential_path.resolve()),
        "parent_issuer_pid": parent_pid,
    }
    return credential, evidence


def load_child_credential_claim(
    credential_path: Path,
    *,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    current_binding_sha: str,
    child_kind: str,
    child_identity: str,
) -> dict[str, Any]:
    """Load a credential in the child; only the parent receipt later authenticates it."""
    _, evidence = _load_child_credential(
        credential_path,
        manifest=manifest,
        resolved=resolved,
        current_binding_sha=current_binding_sha,
        child_kind=child_kind,
        child_identity=child_identity,
    )
    if evidence["parent_issuer_pid"] != os.getppid():
        raise ValueError("GPU child is not directly parented by the credential issuer")
    return {**evidence, "observed_parent_pid": os.getppid()}


def validate_child_credential(
    credential_path: Path,
    *,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    current_binding_sha: str,
    child_kind: str,
    child_identity: str,
    authority_secret: bytes,
    expected_issuer_pid: int,
) -> dict[str, Any]:
    """Authenticate an issuer credential in the parent supervisor/auditor."""
    credential, evidence = _load_child_credential(
        credential_path,
        manifest=manifest,
        resolved=resolved,
        current_binding_sha=current_binding_sha,
        child_kind=child_kind,
        child_identity=child_identity,
    )
    if evidence["parent_issuer_pid"] != require_int(
        expected_issuer_pid, "expected_issuer_pid", minimum=1
    ):
        raise ValueError("credential issuer PID differs from parent observation")
    signed = dict(credential)
    claimed_mac = signed.pop("parent_credential_mac")
    expected_mac = parent_authority_mac(
        authority_secret, "child-credential-v2", signed
    )
    if not secrets.compare_digest(claimed_mac, expected_mac):
        raise ValueError("parent-issued child credential MAC differs")
    return evidence


def record_stage(
    manifest_path: Path,
    *,
    record_type: str,
    artifact: Path,
    example_id: str | None,
    regime: str | None,
    request_id: str | None,
    state_role: str | None,
    parent_credential: Path | None = None,
    parent_receipt: Path | None = None,
) -> dict[str, Any]:
    allowed = {
        "smsb_capture", "smsb_replay", "smsb_adjudication",
        "tetrad_construct", "tetrad_replay", "tetrad_adjudication", "audit_result",
    }
    if record_type not in allowed:
        raise ValueError(f"unsupported stage record type: {record_type}")
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    actual_current_binding_sha = verify_current_binding(
        manifest, resolved, full_model_sha=False
    )
    if not artifact.is_file():
        raise ValueError(f"stage artifact is missing: {artifact}")
    ledger_path = Path(manifest["paths"]["execution_ledger"])
    existing = read_jsonl(ledger_path)
    artifact_resolved = str(artifact.resolve())
    if any(
        row.get("record_type") == record_type
        and row.get("artifact") == artifact_resolved
        for row in existing
    ):
        raise ValueError("stage artifact is already recorded")

    def artifact_current_binding() -> str:
        if record_type == "smsb_capture":
            payloads = read_jsonl(artifact)
            values = {row.get("current_binding_sha256") for row in payloads}
        elif record_type == "smsb_replay":
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            values = {payload.get("result", {}).get("current_binding_sha256")}
        elif record_type == "tetrad_construct":
            payloads = read_jsonl(artifact)
            values = {row.get("current_binding_sha256") for row in payloads}
        else:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            values = {payload.get("current_binding_sha256")}
        if values != {actual_current_binding_sha}:
            raise ValueError(
                "stage artifact current binding differs from P0: "
                f"{values} != {actual_current_binding_sha}"
            )
        return actual_current_binding_sha

    current_binding_sha = artifact_current_binding()
    record: dict[str, Any] = {
        "record_type": record_type,
        "experiment_name": EXPERIMENT_NAME,
        "git_commit": os.environ["MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT"],
        "run_id": manifest["run_id"],
        "recorded_at": utc_now(),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": current_binding_sha,
        "artifact": artifact_resolved,
        "artifact_sha256": sha256_file(artifact),
        "training_authorized": False,
        "method_selection_status": "PENDING_EVIDENCE_NO_SELECTION",
    }
    for key, value in (
        ("example_id", example_id), ("regime", regime),
        ("request_id", request_id), ("state_role", state_role),
    ):
        if value is not None:
            record[key] = value
    if record_type in {"smsb_adjudication", "tetrad_adjudication", "audit_result"}:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        record["status"] = payload.get("status")
        record["decision"] = payload.get("decision")
    if record_type == "tetrad_construct":
        authoring = Path(manifest["paths"]["tetrad_authoring"])
        if not authoring.is_file():
            raise ValueError("Tetrad construct authoring companion artifact is missing")
        record["authoring_artifact"] = str(authoring.resolve())
        record["authoring_artifact_sha256"] = sha256_file(authoring)
    if record_type in {"smsb_capture", "smsb_replay", "tetrad_replay"}:
        if parent_credential is None or parent_receipt is None:
            raise ValueError(
                "GPU child stage requires parent credential and launch receipt"
            )
        payload: Any = (
            read_jsonl(artifact)
            if record_type == "smsb_capture"
            else json.loads(artifact.read_text(encoding="utf-8"))
        )
        if record_type == "smsb_capture":
            if len(payload) != 4:
                raise ValueError("SMSB capture artifact must contain four rows")
            result = payload[0]["execution"]
            if any(
                canonical_sha256(row["execution"])
                != canonical_sha256(result)
                for row in payload[1:]
            ):
                raise ValueError("SMSB capture process evidence differs by row")
        else:
            result = payload["result"] if record_type == "smsb_replay" else payload
        child_identity = (
            "capture4"
            if record_type == "smsb_capture"
            else (
                f"{example_id}::{regime}"
                if record_type == "smsb_replay"
                else str(request_id)
            )
        )
        child_kind = record_type
        receipt_payload = json.loads(parent_receipt.read_text(encoding="utf-8"))
        authority_secret = load_parent_authority_secret(manifest, resolved)
        credential_evidence = validate_child_credential(
            parent_credential,
            manifest=manifest,
            resolved=resolved,
            current_binding_sha=current_binding_sha,
            child_kind=child_kind,
            child_identity=child_identity,
            authority_secret=authority_secret,
            expected_issuer_pid=require_int(
                receipt_payload.get("parent_launcher_pid"),
                "receipt.parent_launcher_pid",
                minimum=1,
            ),
        )
        if any(result.get(key) != value for key, value in credential_evidence.items()):
            raise ValueError("fresh replay result differs from parent-issued credential")
        validated_receipt = validate_parent_launch_receipt(
            receipt_payload,
            authority_secret=authority_secret,
            artifact_payload=payload,
            child_evidence=result,
            child_kind=child_kind,
            child_identity=child_identity,
        )
        if (
            Path(str(validated_receipt["artifact"])).resolve() != artifact.resolve()
            or validated_receipt["artifact_sha256"] != sha256_file(artifact)
            or not Path(str(validated_receipt["stdout_artifact"])).is_file()
            or validated_receipt["stdout_artifact_sha256"]
            != sha256_file(validated_receipt["stdout_artifact"])
        ):
            raise ValueError("parent launch receipt artifact/stdout binding differs")
        record.update(credential_evidence)
        record["process_pid"] = require_int(
            result.get("process_pid"), "result.process_pid", minimum=1
        )
        record.update(
            parent_receipt_path=str(parent_receipt.resolve()),
            parent_receipt_sha256=sha256_file(parent_receipt),
            parent_receipt_id=validated_receipt["receipt_id"],
            parent_receipt_mac=validated_receipt["receipt_mac"],
            parent_launcher_pid=validated_receipt["parent_launcher_pid"],
            observed_child_ppid=validated_receipt["observed_child_ppid"],
            child_exit_code=validated_receipt["child_exit_code"],
            child_stdout_artifact=validated_receipt["stdout_artifact"],
            child_stdout_artifact_sha256=validated_receipt[
                "stdout_artifact_sha256"
            ],
            pre_child_model_manifest_sha256=validated_receipt[
                "pre_child_model_manifest_sha256"
            ],
            post_child_model_manifest_sha256=validated_receipt[
                "post_child_model_manifest_sha256"
            ],
            post_child_current_binding_sha256=validated_receipt[
                "post_child_current_binding_sha256"
            ],
            pre_child_physical_gpu_identity_sha256=validated_receipt[
                "pre_child_physical_gpu_identity_sha256"
            ],
            post_child_physical_gpu_identity_sha256=validated_receipt[
                "post_child_physical_gpu_identity_sha256"
            ],
            post_child_full_model_sha_verified=validated_receipt[
                "post_child_full_model_sha_verified"
            ],
        )
    append_jsonl(ledger_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_REL)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--write-certificate", action="store_true")
    parser.add_argument("--validate-p0-prefix", action="store_true")
    parser.add_argument("--record-type")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--example-id")
    parser.add_argument("--regime")
    parser.add_argument("--request-id")
    parser.add_argument("--state-role")
    parser.add_argument("--parent-credential", type=Path)
    parser.add_argument("--parent-receipt", type=Path)
    args = parser.parse_args()
    if args.record_type:
        if args.artifact is None:
            parser.error("--record-type requires --artifact")
        record = record_stage(
            args.manifest,
            record_type=args.record_type,
            artifact=args.artifact,
            example_id=args.example_id,
            regime=args.regime,
            request_id=args.request_id,
            state_role=args.state_role,
            parent_credential=args.parent_credential,
            parent_receipt=args.parent_receipt,
        )
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        return 0
    if args.validate_p0_prefix:
        p0, resolved = validate_p0(args.manifest)
        print(json.dumps({
            "status": "PASS", "decision": "SERIAL_CREDIT_P0_PREFIX_VALID",
            "run_id": resolved["run_id"], "eval_manifest_hash": resolved["eval_manifest_hash"],
        }, sort_keys=True))
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
