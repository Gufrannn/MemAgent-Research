#!/usr/bin/env python3
"""Fail-closed P0, supervision, and audit for the preregistered capture32.

This is deliberately capture-only.  A locally produced provenance anchor is
an export candidate for external review; it is never treated as an external
signature and cannot authorize training.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.commit_retain_capture import (  # noqa: E402
    canonical_json,
    canonical_sha256,
    validate_capture_ledger,
)
from recurrent.research.gate_a_execution import (  # noqa: E402
    append_jsonl,
    sha256_file,
    validate_jsonl_chain,
)
from recurrent.research.paired_effect_credit import (  # noqa: E402
    CAPTURE32_COUNT,
    recompute_capture32_source_evidence,
    validate_capture32_authority_binding,
    validate_capture32_preregistration,
)
from recurrent.research.serialization_credit_pilots import (  # noqa: E402
    center_truncate_token_ids,
    read_jsonl,
    write_json_exclusive,
)
from tools.h20.preflight_qwen25_7b_commit_retain import (  # noqa: E402
    _model_loading_paths,
    _native_interface_evidence,
    _runtime_versions,
    _worker_multiprocessing_runtime_binding,
)
from tools.h20.preflight_qwen25_7b_serialization_credit import (  # noqa: E402
    capture_lightweight_current_binding,
)


MANIFEST_REL = "manifests/h20/qwen25_7b_commit_retain_capture32_seed2026.json"
BASE_MANIFEST_REL = "manifests/h20/qwen25_7b_commit_retain_capture_seed2026.json"
PREREG_REL = "manifests/h20/qwen25_7b_paired_effect_capture32_preregistration.json"
AUTHORITY_REL = "manifests/h20/qwen25_7b_paired_effect_s128_authority.json"
EXPERIMENT_NAME = "qwen25_7b_commit_retain_capture32_seed2026"
BRANCH = "h20/qwen25-7b-paired-effect-pipeline-20260821"
BASE_COMMIT = "51489768e339f7723fd7b617eba535a1dccc5486"
REQUIRED_ENV = (
    "MEMAGENT_CAPTURE32_WORK_ROOT",
    "MEMAGENT_CAPTURE32_REPO_DIR",
    "MEMAGENT_CAPTURE32_EXPECTED_COMMIT",
    "MEMAGENT_CAPTURE32_RUN_ID",
    "MEMAGENT_CAPTURE32_PHYSICAL_GPUS",
)
RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{1,31}")
FULL_SHA = re.compile(r"[0-9a-f]{40}")
CODE_OBJECTS = (
    "recurrent/impls/memory.py",
    "recurrent/utils.py",
    "recurrent/research/commit_retain_capture.py",
    "recurrent/research/gate_a_execution.py",
    "recurrent/research/paired_effect_credit.py",
    "recurrent/research/s128_hotpot_metrics.py",
    "recurrent/research/serialization_credit_pilots.py",
    "recurrent/research/stable_eval_identity.py",
    "recurrent/research/trajectory_seeding.py",
    "tools/h20/preflight_qwen25_7b_commit_retain.py",
    "tools/h20/preflight_qwen25_7b_serialization_credit.py",
    "tools/h20/preflight_qwen25_7b_s128_it.py",
    "tools/h20/preflight_qwen25_7b_stable_i4x2.py",
    "tools/h20/preflight_qwen25_7b_commit_retain_capture32.py",
    "tools/h20/run_qwen25_7b_commit_retain.py",
    "tools/h20/run_qwen25_7b_commit_retain_capture32.py",
    "scripts/h20/commit_retain_capture32_common.sh",
    "scripts/h20/preflight_qwen25_7b_commit_retain_capture32.sh",
    "scripts/h20/run_qwen25_7b_commit_retain_capture32.sh",
    BASE_MANIFEST_REL,
    MANIFEST_REL,
    "manifests/h20/qwen25_7b_commit_retain_capture32_commands.json",
    PREREG_REL,
    AUTHORITY_REL,
    "commit_retain_capture32_execution_ledger.schema.json",
    "commit_retain_capture32_provenance_anchor.schema.json",
    "docs/h20/commit_retain_capture32_freeze_20260821.md",
)
LEDGER_SCHEMA_REL = "commit_retain_capture32_execution_ledger.schema.json"
ANCHOR_SCHEMA_REL = "commit_retain_capture32_provenance_anchor.schema.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_schema_instance(schema_relative: str, value: Mapping[str, Any]) -> None:
    """Execute the frozen Draft 2020-12 schema; schemas are not documentation."""
    import jsonschema

    schema_path = REPO_ROOT / schema_relative
    if schema_path.is_symlink() or not schema_path.is_file() \
            or schema_path.resolve().parent != REPO_ROOT.resolve():
        raise ValueError(f"capture32 schema is missing/symlinked: {schema_relative}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    failures = sorted(
        validator.iter_errors(dict(value)),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if failures:
        details = "; ".join(
            f"{list(error.absolute_path)}: {error.message}" for error in failures
        )
        raise ValueError(f"capture32 {schema_relative} validation failed: {details}")


def _validate_supervisor_schema(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        _validate_schema_instance(LEDGER_SCHEMA_REL, record)


def _validate_anchor_schema(anchor: Mapping[str, Any]) -> None:
    _validate_schema_instance(ANCHOR_SCHEMA_REL, anchor)


def expected_git_commit() -> str:
    return os.environ[REQUIRED_ENV[2]]


def _merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        result = dict(base)
        for key, value in overlay.items():
            result[key] = _merge(result[key], value) if key in result else value
        return result
    return overlay


def _parse_gpu_pair(value: str) -> list[int]:
    if re.fullmatch(r"(0|[1-9][0-9]*),(0|[1-9][0-9]*)", value or "") is None:
        raise ValueError("MEMAGENT_CAPTURE32_PHYSICAL_GPUS must be two decimal indices N,M")
    indices = [int(item) for item in value.split(",")]
    if len(set(indices)) != 2:
        raise ValueError("capture32 requires two distinct physical GPUs")
    if indices != sorted(indices):
        raise ValueError("capture32 physical GPUs must be explicitly ascending; no silent reorder")
    return indices


def resolve_manifest_environment(
    value: Any, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    source = os.environ if environment is None else environment
    missing = [name for name in REQUIRED_ENV if not source.get(name)]
    if missing:
        raise ValueError(f"missing task-scoped capture32 bindings: {missing}")
    work_root = Path(str(source[REQUIRED_ENV[0]]))
    repo = Path(str(source[REQUIRED_ENV[1]]))
    commit = str(source[REQUIRED_ENV[2]])
    run_id = str(source[REQUIRED_ENV[3]])
    gpu_text = str(source[REQUIRED_ENV[4]])
    gpu_indices = _parse_gpu_pair(gpu_text)
    if not work_root.is_absolute() or not repo.is_absolute():
        raise ValueError("capture32 work/repository paths must be absolute")
    if FULL_SHA.fullmatch(commit) is None:
        raise ValueError("MEMAGENT_CAPTURE32_EXPECTED_COMMIT must be a full Git SHA")
    if RUN_ID.fullmatch(run_id) is None:
        raise ValueError("MEMAGENT_CAPTURE32_RUN_ID has an invalid format")
    replacements = {
        "${MEMAGENT_CAPTURE32_WORK_ROOT}": str(work_root),
        "${MEMAGENT_CAPTURE32_REPO_DIR}": str(repo),
        "${MEMAGENT_CAPTURE32_RUN_ID}": run_id,
        "${MEMAGENT_CAPTURE32_PHYSICAL_GPUS}": gpu_text,
        # The inherited base model/stable-I fields use the old placeholders.
        "${MEMAGENT_COMMIT_RETAIN_WORK_ROOT}": str(work_root),
        "${MEMAGENT_COMMIT_RETAIN_REPO_DIR}": str(repo),
        "${MEMAGENT_COMMIT_RETAIN_RUN_ID}": run_id,
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
                raise ValueError(f"unresolved capture32 placeholder: {result}")
            return result
        return item

    resolved = resolve(value)
    resolved["gpu"]["physical_whitelist"] = gpu_indices
    resolved["gpu"]["visible_devices"] = gpu_text
    resolved["gpu"]["per_gpu_lock_paths"] = [
        resolved["gpu"]["per_gpu_lock_template"].format(physical_index=index)
        for index in gpu_indices
    ]
    return resolved


def load_manifest(
    path: str | Path = REPO_ROOT / MANIFEST_REL,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("extends_manifest") != BASE_MANIFEST_REL:
        raise ValueError("capture32 manifest must extend the reviewed capture manifest")
    raw = dict(raw)
    raw.pop("extends_manifest")
    base = json.loads((REPO_ROOT / BASE_MANIFEST_REL).read_text(encoding="utf-8"))
    return resolve_manifest_environment(_merge(base, raw), environment)


def _assert_regular_confined(path: Path, root: Path, *, must_exist: bool) -> None:
    lexical, lexical_root = path.absolute(), root.absolute()
    try:
        relative = lexical.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"capture32 path escapes authority root: {path}") from error
    cursor = lexical_root
    if cursor.is_symlink():
        raise ValueError(f"capture32 root is symlinked: {cursor}")
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"capture32 path traverses symlink: {path}")
        if not cursor.exists():
            break
    if must_exist and (not lexical.is_file() or lexical.resolve() != lexical):
        raise ValueError(f"capture32 artifact is not a canonical regular file: {path}")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("branch") != BRANCH or manifest.get("base_commit") != BASE_COMMIT:
        raise ValueError("capture32 Git branch/base contract drifted")
    if manifest.get("experiment_name") != EXPERIMENT_NAME:
        raise ValueError("capture32 experiment identity drifted")
    if manifest.get("preregistration") != PREREG_REL \
            or manifest.get("s128_authority") != AUTHORITY_REL:
        raise ValueError("capture32 preregistration/authority path drifted")
    scope = manifest.get("scope", {})
    required_scope = {
        "examples": 32, "folds": 4, "examples_per_fold": 8,
        "training": False, "trainer_attached": False, "actor_updates": 0,
        "optimizer_steps": 0, "paper_performance_result": False,
        "causal_effect_claim": False, "training_authorized": False,
        "capture4_may_fill_missing": False,
    }
    for key, expected in required_scope.items():
        if scope.get(key) != expected or type(scope.get(key)) is not type(expected):
            raise ValueError(f"scope.{key} differs from capture32-only contract")
    if manifest.get("model", {}).get("id") != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("capture32 model must be Qwen2.5-7B-Instruct")
    if manifest.get("data", {}).get("expected_rows") != 128:
        raise ValueError("capture32 must consume the fixed existing S128")
    selection = manifest["data"].get("capture32_selection", {})
    if selection != {
        "kind": "outcome_blind_writer_turn0_prompt_length_strata",
        "selected_sorted_positions": list(range(1, 128, 4)),
        "prior_capture4_positions_excluded": [15, 47, 79, 111],
        "required_pair_count": 32,
        "fold_count": 4,
        "expected_fold_sizes": [8, 8, 8, 8],
        "replacement": "forbidden",
        "run_stitching": "forbidden",
    }:
        raise ValueError("capture32 selection/fold/attrition contract drifted")
    gpu = manifest.get("gpu", {})
    indices = gpu.get("physical_whitelist")
    if indices != _parse_gpu_pair(gpu.get("visible_devices", "")) \
            or gpu.get("selection") != "explicit_environment_only" \
            or gpu.get("tensor_parallel_size") != 2 \
            or gpu.get("required_model_name_substring") != "NVIDIA H20" \
            or gpu.get("required_compute_mode") != "Default" \
            or gpu.get("allowed_mig_modes") != ["Disabled", "N/A", "[N/A]"]:
        raise ValueError("capture32 explicit dual-H20 contract drifted")
    if gpu.get("per_gpu_lock_paths") != [
        f"{manifest['work_root']}/locks/memagent_h20_gpu_{index}.lock"
        for index in indices
    ]:
        raise ValueError("capture32 per-device lock paths drifted")
    intervention = manifest.get("intervention", {})
    if intervention.get("examples") != 32 \
            or intervention.get("arm_execution_order") != ["COMMIT", "RETAIN"] \
            or intervention.get("candidate_generation_count_per_pair") != 1 \
            or intervention.get("trainer_attached") is not False \
            or intervention.get("actor_updates") != 0:
        raise ValueError("capture32 intervention contract drifted")
    backend = manifest.get("backend", {})
    for key, expected in {
        "name": "vllm", "required_version": "0.8.2", "strict_vllm": True,
        "VLLM_USE_V1": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "allow_huggingface_generation_fallback": False,
        "enable_prefix_caching": False, "max_num_seqs": 1,
    }.items():
        if backend.get(key) != expected:
            raise ValueError(f"backend.{key} drifted")
    log_root = Path(manifest["paths"]["log_root"])
    expected_log = Path(manifest["work_root"]) / "logs" / \
        "commit_retain_capture32_frozen_20260821" / manifest["run_id"]
    if log_root != expected_log:
        raise ValueError("capture32 run root drifted")
    for path in manifest["paths"].values():
        path_obj = Path(path)
        root = Path(manifest["work_root"])
        _assert_regular_confined(path_obj, root, must_exist=False)


def _load_preregistration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    prereg_path = Path(manifest["repository"]) / PREREG_REL
    authority_path = Path(manifest["repository"]) / AUTHORITY_REL
    for path, expected in ((prereg_path, REPO_ROOT / PREREG_REL),
                           (authority_path, REPO_ROOT / AUTHORITY_REL)):
        if path.is_symlink() or path.resolve() != expected.resolve() or not path.is_file():
            raise ValueError(f"capture32 Git authority path drifted: {path}")
    prereg = validate_capture32_preregistration(
        json.loads(prereg_path.read_text(encoding="utf-8"))
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    return validate_capture32_authority_binding(prereg, authority)


def _validate_gpu_identity(
    devices: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> list[dict[str, Any]]:
    normalized = [dict(item) for item in devices]
    if [item.get("physical_index") for item in normalized] != list(indices) \
            or len({item.get("uuid") for item in normalized}) != 2 \
            or any("NVIDIA H20" not in str(item.get("name", "")) for item in normalized) \
            or any(item.get("compute_mode") != "Default" for item in normalized) \
            or any(item.get("mig_mode") not in {"Disabled", "N/A", "[N/A]"}
                   for item in normalized):
        raise ValueError(f"selected devices are not the exact two H20s: {normalized}")
    return normalized


def _gpu_identity(indices: Sequence[int]) -> list[dict[str, Any]]:
    executable = Path("/usr/bin/nvidia-smi")
    if not executable.is_file():
        raise ValueError("/usr/bin/nvidia-smi is required; PATH lookup is forbidden")
    result = subprocess.run(
        [str(executable), "-i", ",".join(map(str, indices)),
         "--query-gpu=index,uuid,pci.bus_id,name,compute_mode,mig.mode.current",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"cannot identify capture32 GPUs: {result.stderr.strip()}")
    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise ValueError(f"unexpected nvidia-smi identity row: {line}")
        devices.append({
            "physical_index": int(parts[0]), "uuid": parts[1],
            "pci_bus_id": parts[2], "name": parts[3],
            "compute_mode": parts[4], "mig_mode": parts[5],
        })
    return _validate_gpu_identity(devices, indices)


def _process_identity(pid: int) -> dict[str, Any]:
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    stat_fields = Path(f"/proc/{pid}/stat").read_text().split()
    return {"pid": pid, "boot_id": boot_id, "process_start_ticks": int(stat_fields[21])}


def capture_lock_holder_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    fd_text = os.environ.get("MEMAGENT_CAPTURE32_LOCK_FDS", "")
    fd_items = fd_text.split(",") if fd_text else []
    expected_paths = [*manifest["gpu"]["per_gpu_lock_paths"]]
    legacy_paths = []
    selected = set(manifest["gpu"]["physical_whitelist"])
    for pair in ((4, 5), (6, 7)):
        if selected.intersection(pair):
            legacy_paths.append(
                f"{manifest['work_root']}/locks/memagent_gate_a_gpu_{pair[0]}_{pair[1]}.lock"
            )
    expected_paths.extend(legacy_paths)
    if len(fd_items) != len(expected_paths) or any(not item.isdecimal() for item in fd_items):
        raise ValueError("capture32 inherited lock FD inventory is incomplete")
    locks = []
    seen_inodes: set[tuple[int, int]] = set()
    for path_text, fd_text_item in zip(expected_paths, fd_items):
        path = Path(path_text)
        fd = int(fd_text_item)
        fd_stat = os.fstat(fd)
        if not stat.S_ISREG(fd_stat.st_mode) or fd_stat.st_nlink != 1:
            raise ValueError(f"capture32 lock FD is not a single-link regular file: {path}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"capture32 lock path is missing or symlinked: {path}")
        path_stat = path.stat()
        identity = (fd_stat.st_dev, fd_stat.st_ino)
        if identity != (path_stat.st_dev, path_stat.st_ino) or identity in seen_inodes:
            raise ValueError(f"capture32 lock inode drifted or duplicated: {path}")
        if fd_stat.st_uid != os.getuid():
            raise ValueError(f"capture32 lock is not owned by current uid: {path}")
        seen_inodes.add(identity)
        # LOCK_EX|LOCK_NB on a duplicate description must fail while the shell
        # retains the actual lock; opening a new FD is the independent proof.
        probe = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError(f"capture32 lock is not held exclusively: {path}")
        finally:
            os.close(probe)
        locks.append({
            "path": str(path.resolve()), "fd": fd, "device": fd_stat.st_dev,
            "inode": fd_stat.st_ino, "owner_uid": fd_stat.st_uid,
        })
    receipt = {
        "schema": "memagent.capture32.lock-holder-receipt.v1",
        "physical_gpu_indices": manifest["gpu"]["physical_whitelist"],
        "locks": locks,
        "holder": _process_identity(os.getppid()),
    }
    receipt["gpu_lock_binding_sha256"] = canonical_sha256({
        "physical_gpu_indices": receipt["physical_gpu_indices"],
        "locks": [{key: item[key] for key in ("path", "device", "inode", "owner_uid")}
                  for item in locks],
    })
    receipt["lock_holder_receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _execution_frozen_pairs(
    manifest: Mapping[str, Any], prereg: Mapping[str, Any], tokenizer: Any
) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    rows = parquet.read_table(
        manifest["data"]["validation"],
        columns=["prompt", "context", "reward_model", "extra_info"],
    ).to_pylist()
    chunk_size = int(manifest["recurrent"]["chunk_size"])
    no_memory = list(tokenizer.encode(
        manifest["recurrent"]["no_memory_text"], add_special_tokens=False
    ))
    frozen = []
    for item in prereg["selected_inventory"]:
        source = rows[int(item["raw_row_position"])]
        question = str(source["prompt"][0]["content"])
        context = str(source["context"])
        ground_truth = [str(value) for value in source["reward_model"]["ground_truth"]]
        question_ids = list(tokenizer.encode(question, add_special_tokens=False))
        context_ids = center_truncate_token_ids(
            list(tokenizer.encode(context, add_special_tokens=False)),
            int(manifest["recurrent"]["max_context_tokens"]),
        )
        chunks = [context_ids[offset:offset + chunk_size]
                  for offset in range(0, len(context_ids), chunk_size)]
        checks = {
            "source_question_hash": hashlib.sha256(question.encode()).hexdigest(),
            "source_context_hash": hashlib.sha256(context.encode()).hexdigest(),
            "ground_truth_hash": canonical_sha256(ground_truth),
            "question_token_ids_sha256": canonical_sha256(question_ids),
            "context_token_ids_sha256": canonical_sha256(context_ids),
        }
        if any(item[key] != value for key, value in checks.items()):
            raise ValueError(f"capture32 source/token identity drifted: {item['stable_example_id']}")
        turn = int(item["intervention_writer_turn"])
        total = int(item["total_writer_turns"])
        if len(chunks) != total or not 0 < turn < total - 1:
            raise ValueError("capture32 writer horizon/timepoint drifted")
        frozen.append({
            **dict(item),
            "chunk_token_ids_sha256": [canonical_sha256(chunk) for chunk in chunks],
            "candidate_chunk_token_ids_sha256": canonical_sha256(chunks[turn]),
            "future_chunk_token_ids_sha256": [
                canonical_sha256(chunk) for chunk in chunks[turn + 1:]
            ],
            "no_memory_token_ids_sha256": canonical_sha256(no_memory),
            "expected_pair_generate_calls": turn + 1 + 2 * (total - turn),
        })
    if len(frozen) != CAPTURE32_COUNT or len({x["stable_write_id"] for x in frozen}) != 32:
        raise ValueError("capture32 execution freeze is not exact 32 unique stable writes")
    return frozen


def expected_pair_binding(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any], tokenizer: Any
) -> dict[str, Any]:
    return dict(resolved["expected_pair_binding"])


def _current_binding(manifest: Mapping[str, Any], resolved: Mapping[str, Any], *, full_model_sha: bool) -> str:
    lightweight = capture_lightweight_current_binding(manifest)
    if canonical_sha256(lightweight) != resolved["lightweight_current_binding_sha256"]:
        raise ValueError("capture32 current Git/model-stat/data/runtime binding differs from P0")
    lock_receipt = capture_lock_holder_receipt(manifest)
    if lock_receipt["gpu_lock_binding_sha256"] != resolved["gpu_lock_binding_sha256"]:
        raise ValueError("capture32 lock inode binding differs from P0")
    if full_model_sha:
        root = Path(manifest["model"]["path"])
        actual = [{"path": item["path"], "size": (root / item["path"]).stat().st_size,
                   "sha256": sha256_file(root / item["path"])}
                  for item in manifest["model"]["files"]]
        if actual != manifest["model"]["files"]:
            raise ValueError("capture32 full model inventory drifted")
    current = {
        "schema": "memagent.commit-retain.capture32-current-binding.v1",
        "run_id": manifest["run_id"], "git_commit": expected_git_commit(),
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "expected_pair_binding_sha256": canonical_sha256(resolved["expected_pair_binding"]),
        "physical_gpu_whitelist": manifest["gpu"]["physical_whitelist"],
        "visible_devices": manifest["gpu"]["visible_devices"],
    }
    current_sha = canonical_sha256(current)
    if current != resolved["current_binding"] or current_sha != resolved["current_binding_sha256"]:
        raise ValueError("capture32 canonical current binding differs from P0")
    return current_sha


def _code_hashes(repo: Path) -> dict[str, str]:
    missing = [name for name in CODE_OBJECTS if not (repo / name).is_file()]
    if missing:
        raise ValueError(f"capture32 code-object inventory missing: {missing}")
    return {name: sha256_file(repo / name) for name in CODE_OBJECTS}


def run_preflight(manifest_path: Path, *, check_runtime: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    failures: list[str] = []
    if not check_runtime:
        failures.append("formal capture32 P0 requires --check-runtime")
    try:
        manifest = load_manifest(manifest_path)
        _validate_manifest(manifest)
        prereg = _load_preregistration(manifest)
        lock_receipt = capture_lock_holder_receipt(manifest)
    except Exception as error:
        return {"gate": "P0", "status": "FAIL", "decision": "CAPTURE32_NO_GO:P0",
                "failures": [str(error)], "evidence": {}}, None
    repo = Path(manifest["repository"]).resolve()
    commit = expected_git_commit()
    evidence: dict[str, Any] = {"expected_git_commit": commit, "run_id": manifest["run_id"]}
    try:
        branch = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip()
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
        if repo != REPO_ROOT.resolve() or branch != BRANCH or head != commit or dirty:
            raise ValueError(f"capture32 Git closure failed: repo={repo}, branch={branch}, head={head}, dirty={bool(dirty)}")
        if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE_COMMIT, head]).returncode:
            raise ValueError("capture32 base commit is not an ancestor")
        if subprocess.check_output(["git", "-C", str(repo), "diff", "--name-only", BASE_COMMIT, head, "--", "sources"], text=True).strip():
            raise ValueError("capture32 branch changed sources/")
        code_hashes = _code_hashes(repo)
    except Exception as error:
        failures.append(str(error))
        code_hashes = {}
    indices = manifest["gpu"]["physical_whitelist"]
    visible = manifest["gpu"]["visible_devices"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
        failures.append(f"CUDA_VISIBLE_DEVICES must be exactly {visible}")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" \
            or os.environ.get("VLLM_USE_V1") != "0" \
            or os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        failures.append("capture32 CUDA/vLLM spawn environment drifted")
    model_root = Path(manifest["model"]["path"])
    actual_model = []
    for item in manifest["model"]["files"]:
        path = model_root / item["path"]
        if not path.is_file() or path.is_symlink():
            failures.append(f"capture32 model file missing/symlinked: {item['path']}")
            continue
        actual = {"path": item["path"], "size": path.stat().st_size, "sha256": sha256_file(path)}
        actual_model.append(actual)
        if actual != item:
            failures.append(f"capture32 model file drifted: {item['path']}")
    if model_root.is_dir() and _model_loading_paths(model_root) != sorted(x["path"] for x in manifest["model"]["files"]):
        failures.append("capture32 model loading inventory drifted")
    data_path = Path(manifest["data"]["validation"])
    if not data_path.is_file() or data_path.is_symlink() \
            or sha256_file(data_path) != manifest["data"]["validation_sha256"]:
        failures.append("capture32 fixed S128 parquet missing/symlinked/drifted")
    runtime_versions: dict[str, Any] = {}
    worker_binding: dict[str, Any] = {}
    devices: list[dict[str, Any]] = []
    if check_runtime and not failures:
        try:
            runtime_versions = _runtime_versions(Path(manifest["python"]), repo)
            if runtime_versions.get("vllm") != "0.8.2":
                raise ValueError("capture32 requires vLLM 0.8.2")
            worker_binding = _worker_multiprocessing_runtime_binding(Path(manifest["python"]), repo)
            devices = _gpu_identity(indices)
        except Exception as error:
            failures.append(str(error))
    resolved: dict[str, Any] | None = None
    if not failures:
        try:
            from transformers import AutoTokenizer
            from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
            from recurrent.utils import TokenTemplate, chat_template
            import pyarrow.parquet as parquet

            tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True, local_files_only=True)
            chat = chat_template(tokenizer)
            writer_text, reader_text = chat.format(message=TEMPLATE), chat.format(message=TEMPLATE_FINAL_BOXED)
            writer = TokenTemplate(writer_text, tokenizer)
            authority = json.loads((repo / AUTHORITY_REL).read_text())
            rows = parquet.read_table(data_path, columns=["prompt", "context", "reward_model", "extra_info"]).to_pylist()
            replay = recompute_capture32_source_evidence(
                parquet_rows=rows, authority=authority, tokenizer=tokenizer,
                writer_prompt_builder=lambda question, memory, chunk: writer.format(prompt=question, memory=memory, chunk=chunk).tolist(),
                no_memory_text=manifest["recurrent"]["no_memory_text"],
                max_context_tokens=int(manifest["recurrent"]["max_context_tokens"]),
                chunk_size=int(manifest["recurrent"]["chunk_size"]),
                base_seed=int(manifest["backend"]["engine_seed"]),
            )
            if canonical_json(replay["selected_inventory"]) != canonical_json(prereg["selected_inventory"]) \
                    or replay["full_population_ranking_sha256"] != prereg["selection"]["full_population_ranking_sha256"]:
                raise ValueError("capture32 raw S128/tokenizer replay differs from preregistration")
            enriched = _execution_frozen_pairs(manifest, prereg, tokenizer)
            model_sha = canonical_sha256(actual_model)
            tokenizer_sha = canonical_sha256([x for x in actual_model if x["path"] in {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}])
            if model_sha != prereg["source"]["model_file_manifest_sha256"] \
                    or tokenizer_sha != prereg["source"]["tokenizer_manifest_sha256"]:
                raise ValueError("capture32 model/tokenizer differs from preregistration")
            engine_config = {**dict(manifest["backend"]), "physical_gpu_whitelist": indices,
                             "visible_devices": visible, "tensor_parallel_size": 2,
                             "one_prompt_per_generate_call": True}
            expected_calls = sum(x["expected_pair_generate_calls"] for x in enriched)
            expected_pair = {
                "writer_checkpoint_sha256": model_sha, "reader_checkpoint_sha256": model_sha,
                "writer_prompt_template_sha256": hashlib.sha256(writer_text.encode()).hexdigest(),
                "reader_prompt_template_sha256": hashlib.sha256(reader_text.encode()).hexdigest(),
                "writer_decode": manifest["intervention"]["writer_decode"],
                "reader_decode": manifest["intervention"]["reader_decode"],
                "physical_gpu_whitelist": indices, "visible_devices": visible,
                "physical_gpu_identity": devices, "engine_config_sha256": canonical_sha256(engine_config),
                "worker_multiproc_method": "spawn", "vllm_observed_worker_multiproc_method": "spawn",
                "multiprocessing_context_method": "spawn",
                "parent_cuda_initialization_policy": "record_observed_spawn_required",
                "global_generate_call_count": expected_calls, "eos_token_id": int(tokenizer.eos_token_id),
                "gpu_lock_binding_sha256": lock_receipt["gpu_lock_binding_sha256"],
            }
            pair_sha = canonical_sha256(expected_pair)
            code_combined = canonical_sha256(code_hashes)
            execution = {
                "schema": "memagent.commit-retain.capture32-execution-binding.v1",
                "run_id": manifest["run_id"], "git_commit": commit,
                "preregistration_sha256": prereg["preregistration_sha256"],
                "eval_manifest_hash": prereg["source"]["eval_manifest_hash"],
                "selected_inventory_sha256": prereg["inventory"]["selected_inventory_sha256"],
                "fold_membership_sha256": prereg["folds"]["membership_sha256"],
                "execution_code_combined_sha256": code_combined,
                "expected_pair_binding_sha256": pair_sha,
                "physical_gpu_whitelist": indices, "visible_devices": visible,
                "gpu_lock_binding_sha256": lock_receipt["gpu_lock_binding_sha256"],
                "rollout_backend": "strict_vllm_0.8.2",
            }
            runtime = {
                "schema": "memagent.commit-retain.capture32-runtime-binding.v1",
                "run_id": manifest["run_id"], "git_commit": commit,
                "expected_pair_binding_sha256": pair_sha,
                "physical_gpu_whitelist": indices, "visible_devices": visible,
                "physical_gpu_identity": devices, "rollout_backend": "strict_vllm_0.8.2",
                "engine_config_sha256": expected_pair["engine_config_sha256"],
                "worker_multiproc_method": "spawn", "vllm_observed_worker_multiproc_method": "spawn",
                "multiprocessing_context_method": "spawn",
                "parent_cuda_initialization_policy": "record_observed_spawn_required",
                "writer_checkpoint_sha256": model_sha, "reader_checkpoint_sha256": model_sha,
                "model_file_manifest_sha256": model_sha, "tokenizer_manifest_sha256": tokenizer_sha,
                "gpu_lock_binding_sha256": lock_receipt["gpu_lock_binding_sha256"],
            }
            execution_sha, runtime_sha = canonical_sha256(execution), canonical_sha256(runtime)
            current = {
                "schema": "memagent.commit-retain.capture32-current-binding.v1",
                "run_id": manifest["run_id"], "git_commit": commit,
                "execution_binding_sha256": execution_sha, "runtime_binding_sha256": runtime_sha,
                "expected_pair_binding_sha256": pair_sha,
                "physical_gpu_whitelist": indices, "visible_devices": visible,
            }
            lightweight = capture_lightweight_current_binding(manifest)
            resolved = {
                "schema": "memagent.commit-retain.capture32-resolved.v2",
                "run_id": manifest["run_id"], "git_commit": commit,
                "preregistration_sha256": prereg["preregistration_sha256"],
                "source_validation_sha256": prereg["source"]["validation_sha256"],
                "eval_manifest_hash": prereg["source"]["eval_manifest_hash"],
                "base_model_id": prereg["source"]["base_model_id"],
                "base_model_revision": prereg["source"]["base_model_revision"],
                "model_file_manifest_sha256": model_sha, "tokenizer_manifest_sha256": tokenizer_sha,
                "s128_authority_file_sha256": prereg["source"]["s128_authority_file_sha256"],
                "s128_authority_sha256": prereg["source"]["s128_authority_sha256"],
                "rollout_backend": "strict_vllm_0.8.2",
                "physical_gpu_whitelist": indices, "visible_devices": visible,
                "selected_inventory_sha256": prereg["inventory"]["selected_inventory_sha256"],
                "full_population_ranking_sha256": prereg["selection"]["full_population_ranking_sha256"],
                "fold_membership_sha256": prereg["folds"]["membership_sha256"],
                "frozen_pairs": prereg["selected_inventory"],
                "execution_binding": execution, "execution_binding_sha256": execution_sha,
                "runtime_binding": runtime, "runtime_binding_sha256": runtime_sha,
                "current_binding": current, "current_binding_sha256": canonical_sha256(current),
                "expected_pair_binding": expected_pair,
                "execution_code_sha256": code_hashes, "execution_code_combined_sha256": code_combined,
                "gpu_lock_binding": {key: lock_receipt[key] for key in ("schema", "physical_gpu_indices", "locks", "gpu_lock_binding_sha256")},
                "gpu_lock_binding_sha256": lock_receipt["gpu_lock_binding_sha256"],
                "lightweight_current_binding": lightweight,
                "lightweight_current_binding_sha256": canonical_sha256(lightweight),
                "expected_global_generate_call_count": expected_calls,
            }
            evidence.update({"git_commit": commit, "git_branch": BRANCH,
                             "execution_code_sha256": code_hashes,
                             "physical_gpu_identity": devices,
                             "gpu_lock_binding_sha256": lock_receipt["gpu_lock_binding_sha256"],
                             "expected_pair_count": 32, "expected_global_generate_call_count": expected_calls})
        except Exception as error:
            failures.append(f"cannot freeze capture32 binding: {error}")
            resolved = None
    status = "PASS" if not failures and resolved is not None else "FAIL"
    return {"gate": "P0", "status": status,
            "decision": "COMMIT_RETAIN_CAPTURE32_P0_PASS" if status == "PASS" else "CAPTURE32_NO_GO:P0",
            "failures": failures, "evidence": evidence, "scope": manifest["scope"]}, resolved


def _external_prereg_anchor(manifest: Mapping[str, Any], p0_path: Path, resolved_path: Path) -> dict[str, Any]:
    anchor = {
        "schema": "memagent.commit-retain.capture32-provenance-anchor.v1",
        "anchor_kind": "PRE_GENERATION_LOCAL_EXPORT_CANDIDATE",
        "trust_status": "PENDING_EXTERNAL_SIGNATURE",
        "run_id": manifest["run_id"], "git_commit": expected_git_commit(),
        "recorded_at": utc_now(),
        "preregistration": str((REPO_ROOT / PREREG_REL).resolve()),
        "preregistration_sha256": sha256_file(REPO_ROOT / PREREG_REL),
        "p0_certificate": str(p0_path.resolve()), "p0_certificate_sha256": sha256_file(p0_path),
        "resolved_manifest": str(resolved_path.resolve()), "resolved_manifest_sha256": sha256_file(resolved_path),
        "training_authorized": False, "method_selected": False,
    }
    anchor["anchor_payload_sha256"] = canonical_sha256(anchor)
    return anchor


def write_preflight(manifest_path: Path, *, check_runtime: bool) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    paths = manifest["paths"]
    targets = [Path(paths[key]) for key in ("p0_certificate", "resolved_manifest", "execution_ledger", "external_preregistration_anchor")]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("refuse to overwrite capture32 P0/ledger/external anchor")
    report, resolved = run_preflight(manifest_path, check_runtime=check_runtime)
    p0_path, resolved_path, ledger_path, anchor_path = targets
    if report["status"] != "PASS" or resolved is None:
        return report
    write_json_exclusive(resolved_path, resolved)
    p0 = {
        "schema": "memagent.commit-retain.capture32-p0.v2", "status": "PASS",
        "decision": "COMMIT_RETAIN_CAPTURE32_P0_PASS", "run_id": manifest["run_id"],
        "git_commit": expected_git_commit(),
        "preregistration": str((REPO_ROOT / PREREG_REL).resolve()),
        "preregistration_file_sha256": sha256_file(REPO_ROOT / PREREG_REL),
        "preregistration_sha256": resolved["preregistration_sha256"],
        "resolved_manifest": str(resolved_path.resolve()), "resolved_manifest_sha256": sha256_file(resolved_path),
        "capture_ledger": str(Path(paths["capture_ledger"]).resolve()),
        "final_report": str(Path(paths["final_report"]).resolve()),
        "physical_gpu_whitelist": manifest["gpu"]["physical_whitelist"],
        "visible_devices": manifest["gpu"]["visible_devices"],
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "commitment_frozen_before_first_generate": True,
        "training_authorized": False, "method_selected": False, "failures": [],
    }
    write_json_exclusive(p0_path, p0)
    anchor = _external_prereg_anchor(manifest, p0_path, resolved_path)
    _validate_anchor_schema(anchor)
    write_json_exclusive(anchor_path, anchor)
    append_jsonl(ledger_path, {
        "record_type": "s0_preflight", "experiment_name": EXPERIMENT_NAME,
        "git_commit": expected_git_commit(), "run_id": manifest["run_id"], "recorded_at": utc_now(),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": resolved["current_binding_sha256"],
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "artifact": str(p0_path.resolve()), "artifact_sha256": sha256_file(p0_path),
        "resolved_manifest": str(resolved_path.resolve()), "resolved_manifest_sha256": sha256_file(resolved_path),
        "external_preregistration_anchor": str(anchor_path.resolve()),
        "external_preregistration_anchor_sha256": sha256_file(anchor_path),
        "status": "PASS", "decision": "COMMIT_RETAIN_CAPTURE32_P0_PASS",
        "training_authorized": False, "method_selected": False,
    })
    validate_p0(manifest_path)
    return p0


def _validate_pre_generation_anchor(
    manifest: Mapping[str, Any], *, p0_path: Path, resolved_path: Path
) -> dict[str, Any]:
    anchor_path = Path(manifest["paths"]["external_preregistration_anchor"])
    _assert_regular_confined(anchor_path, Path(manifest["work_root"]), must_exist=True)
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    _validate_anchor_schema(anchor)
    unsigned = dict(anchor)
    digest = unsigned.pop("anchor_payload_sha256", None)
    expected_fields = {
        "schema", "anchor_kind", "trust_status", "run_id", "git_commit",
        "recorded_at", "preregistration", "preregistration_sha256",
        "p0_certificate", "p0_certificate_sha256", "resolved_manifest",
        "resolved_manifest_sha256", "training_authorized", "method_selected",
        "anchor_payload_sha256",
    }
    if set(anchor) != expected_fields or digest != canonical_sha256(unsigned) \
            or anchor.get("schema") != "memagent.commit-retain.capture32-provenance-anchor.v1" \
            or anchor.get("anchor_kind") != "PRE_GENERATION_LOCAL_EXPORT_CANDIDATE" \
            or anchor.get("trust_status") != "PENDING_EXTERNAL_SIGNATURE" \
            or anchor.get("run_id") != manifest["run_id"] \
            or anchor.get("git_commit") != expected_git_commit() \
            or anchor.get("preregistration") != str((REPO_ROOT / PREREG_REL).resolve()) \
            or anchor.get("preregistration_sha256") != sha256_file(REPO_ROOT / PREREG_REL) \
            or anchor.get("p0_certificate") != str(p0_path.resolve()) \
            or anchor.get("p0_certificate_sha256") != sha256_file(p0_path) \
            or anchor.get("resolved_manifest") != str(resolved_path.resolve()) \
            or anchor.get("resolved_manifest_sha256") != sha256_file(resolved_path) \
            or anchor.get("training_authorized") is not False \
            or anchor.get("method_selected") is not False:
        raise ValueError("capture32 pre-generation provenance anchor failed")
    return anchor


def _validate_supervisor_common(
    record: Mapping[str, Any], *, record_type: str, decision: str,
    manifest: Mapping[str, Any], resolved: Mapping[str, Any], artifact: Path,
) -> None:
    if record.get("record_type") != record_type \
            or record.get("experiment_name") != EXPERIMENT_NAME \
            or record.get("git_commit") != expected_git_commit() \
            or record.get("run_id") != manifest["run_id"] \
            or record.get("eval_manifest_hash") != resolved["eval_manifest_hash"] \
            or record.get("execution_binding_sha256") != resolved["execution_binding_sha256"] \
            or record.get("runtime_binding_sha256") != resolved["runtime_binding_sha256"] \
            or record.get("current_binding_sha256") != resolved["current_binding_sha256"] \
            or record.get("gpu_lock_binding_sha256") != resolved["gpu_lock_binding_sha256"] \
            or record.get("artifact") != str(artifact.resolve()) \
            or record.get("artifact_sha256") != sha256_file(artifact) \
            or record.get("status") != "PASS" \
            or record.get("decision") != decision \
            or record.get("training_authorized") is not False \
            or record.get("method_selected") is not False:
        raise ValueError(f"capture32 canonical supervisor {record_type} failed")


def validate_p0(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    _validate_manifest(manifest)
    paths = manifest["paths"]
    p0_path, resolved_path, ledger_path = (Path(paths[key]) for key in ("p0_certificate", "resolved_manifest", "execution_ledger"))
    for path in (p0_path, resolved_path, ledger_path, Path(paths["external_preregistration_anchor"])):
        _assert_regular_confined(path, Path(manifest["work_root"]), must_exist=True)
    p0 = json.loads(p0_path.read_text())
    resolved = json.loads(resolved_path.read_text())
    records = read_jsonl(ledger_path)
    if validate_jsonl_chain(records) or not records or records[0].get("record_type") != "s0_preflight":
        raise ValueError("capture32 P0 supervisor chain failed")
    _validate_supervisor_schema(records)
    p0_fields = {
        "schema", "status", "decision", "run_id", "git_commit",
        "preregistration", "preregistration_file_sha256",
        "preregistration_sha256", "resolved_manifest",
        "resolved_manifest_sha256", "capture_ledger", "final_report",
        "physical_gpu_whitelist", "visible_devices", "gpu_lock_binding_sha256",
        "commitment_frozen_before_first_generate", "training_authorized",
        "method_selected", "failures",
    }
    if set(p0) != p0_fields or any((
            p0.get("schema") != "memagent.commit-retain.capture32-p0.v2",
            p0.get("status") != "PASS",
            p0.get("decision") != "COMMIT_RETAIN_CAPTURE32_P0_PASS",
            p0.get("run_id") != manifest["run_id"],
            p0.get("git_commit") != expected_git_commit(),
            p0.get("preregistration") != str((REPO_ROOT / PREREG_REL).resolve()),
            p0.get("preregistration_file_sha256") != sha256_file(REPO_ROOT / PREREG_REL),
            p0.get("resolved_manifest") != str(resolved_path.resolve()),
            p0.get("resolved_manifest_sha256") != sha256_file(resolved_path),
            p0.get("capture_ledger") != str(Path(paths["capture_ledger"]).resolve()),
            p0.get("final_report") != str(Path(paths["final_report"]).resolve()),
            p0.get("physical_gpu_whitelist") != manifest["gpu"]["physical_whitelist"],
            p0.get("visible_devices") != manifest["gpu"]["visible_devices"],
            p0.get("gpu_lock_binding_sha256") != resolved.get("gpu_lock_binding_sha256"),
            p0.get("commitment_frozen_before_first_generate") is not True,
            p0.get("training_authorized") is not False,
            p0.get("method_selected") is not False,
            p0.get("failures") != [],
    )):
        raise ValueError("capture32 P0 certificate binding failed")
    prereg = _load_preregistration(manifest)
    resolved_fields = {
        "schema", "run_id", "git_commit", "preregistration_sha256",
        "source_validation_sha256", "eval_manifest_hash", "base_model_id",
        "base_model_revision", "model_file_manifest_sha256",
        "tokenizer_manifest_sha256", "s128_authority_file_sha256",
        "s128_authority_sha256", "rollout_backend", "physical_gpu_whitelist",
        "visible_devices", "selected_inventory_sha256",
        "full_population_ranking_sha256", "fold_membership_sha256",
        "frozen_pairs", "execution_binding", "execution_binding_sha256",
        "runtime_binding", "runtime_binding_sha256", "current_binding",
        "current_binding_sha256", "expected_pair_binding",
        "execution_code_sha256", "execution_code_combined_sha256",
        "gpu_lock_binding", "gpu_lock_binding_sha256",
        "lightweight_current_binding", "lightweight_current_binding_sha256",
        "expected_global_generate_call_count",
    }
    code_hashes = _code_hashes(REPO_ROOT)
    if any(not isinstance(resolved.get(field), Mapping) for field in (
        "execution_binding", "runtime_binding", "current_binding",
        "expected_pair_binding", "gpu_lock_binding",
        "lightweight_current_binding",
    )):
        raise ValueError("capture32 resolved P0 structured binding is malformed")
    if set(resolved) != resolved_fields \
            or resolved.get("schema") != "memagent.commit-retain.capture32-resolved.v2" \
            or resolved.get("run_id") != manifest["run_id"] \
            or resolved.get("git_commit") != expected_git_commit() \
            or resolved.get("preregistration_sha256") != prereg["preregistration_sha256"] \
            or p0.get("preregistration_sha256") != prereg["preregistration_sha256"] \
            or resolved.get("source_validation_sha256") != prereg["source"]["validation_sha256"] \
            or resolved.get("eval_manifest_hash") != prereg["source"]["eval_manifest_hash"] \
            or resolved.get("selected_inventory_sha256") != prereg["inventory"]["selected_inventory_sha256"] \
            or resolved.get("full_population_ranking_sha256") != prereg["selection"]["full_population_ranking_sha256"] \
            or resolved.get("fold_membership_sha256") != prereg["folds"]["membership_sha256"] \
            or canonical_json(resolved.get("frozen_pairs")) != canonical_json(prereg["selected_inventory"]) \
            or resolved.get("physical_gpu_whitelist") != manifest["gpu"]["physical_whitelist"] \
            or resolved.get("visible_devices") != manifest["gpu"]["visible_devices"] \
            or resolved.get("rollout_backend") != "strict_vllm_0.8.2" \
            or resolved.get("expected_global_generate_call_count") != 353 \
            or resolved.get("execution_code_sha256") != code_hashes \
            or resolved.get("execution_code_combined_sha256") != canonical_sha256(code_hashes) \
            or resolved.get("execution_binding_sha256") != canonical_sha256(resolved.get("execution_binding")) \
            or resolved.get("runtime_binding_sha256") != canonical_sha256(resolved.get("runtime_binding")) \
            or resolved.get("current_binding_sha256") != canonical_sha256(resolved.get("current_binding")) \
            or canonical_sha256(resolved.get("expected_pair_binding")) != resolved.get("execution_binding", {}).get("expected_pair_binding_sha256"):
        raise ValueError("capture32 resolved P0 manifest failed independent reconstruction")
    live_lock = capture_lock_holder_receipt(manifest)
    expected_stored_lock = {
        key: live_lock[key] for key in (
            "schema", "physical_gpu_indices", "locks", "gpu_lock_binding_sha256"
        )
    }
    if canonical_json(resolved["gpu_lock_binding"]) != canonical_json(expected_stored_lock):
        raise ValueError("capture32 resolved lock receipt differs from live held locks")
    _validate_pre_generation_anchor(
        manifest, p0_path=p0_path, resolved_path=resolved_path
    )
    s0 = records[0]
    if s0.get("experiment_name") != EXPERIMENT_NAME \
            or s0.get("run_id") != manifest["run_id"] \
            or s0.get("git_commit") != expected_git_commit() \
            or s0.get("eval_manifest_hash") != resolved["eval_manifest_hash"] \
            or s0.get("execution_binding_sha256") != resolved["execution_binding_sha256"] \
            or s0.get("runtime_binding_sha256") != resolved["runtime_binding_sha256"] \
            or s0.get("current_binding_sha256") != resolved["current_binding_sha256"] \
            or s0.get("gpu_lock_binding_sha256") != resolved["gpu_lock_binding_sha256"] \
            or s0.get("artifact") != str(p0_path.resolve()) \
            or s0.get("artifact_sha256") != sha256_file(p0_path) \
            or s0.get("resolved_manifest") != str(resolved_path.resolve()) \
            or s0.get("resolved_manifest_sha256") != sha256_file(resolved_path) \
            or s0.get("external_preregistration_anchor") != str(Path(paths["external_preregistration_anchor"]).resolve()) \
            or s0.get("external_preregistration_anchor_sha256") != sha256_file(paths["external_preregistration_anchor"]) \
            or s0.get("status") != "PASS" \
            or s0.get("decision") != "COMMIT_RETAIN_CAPTURE32_P0_PASS" \
            or s0.get("training_authorized") is not False \
            or s0.get("method_selected") is not False:
        raise ValueError("capture32 canonical P0 supervisor record failed")
    _current_binding(manifest, resolved, full_model_sha=False)
    return p0, resolved


def issue_capture_credential(manifest_path: Path, *, output: Path, issuer_shell_pid: int) -> dict[str, Any]:
    if os.getppid() != issuer_shell_pid:
        raise ValueError("capture32 credential issuer must be direct shell child")
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    if output.resolve() != Path(manifest["paths"]["capture_credential"]).resolve():
        raise ValueError("capture32 credential path drifted")
    records = read_jsonl(manifest["paths"]["execution_ledger"])
    if [x.get("record_type") for x in records] != ["s0_preflight"]:
        raise ValueError("capture32 credential must immediately follow P0")
    holder = capture_lock_holder_receipt(manifest)
    credential = {
        "schema": "memagent.commit-retain.capture32-parent-credential.v1",
        "run_id": manifest["run_id"], "git_commit": expected_git_commit(),
        "child_kind": "single_engine_exact_32_pair_capture",
        "parent_identity": _process_identity(issuer_shell_pid), "issued_at": utc_now(),
        "nonce": secrets.token_hex(32), "single_use": True,
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": resolved["current_binding_sha256"],
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "lock_holder_receipt": holder,
        "training_authorized": False, "method_selected": False,
    }
    credential["parent_credential_id"] = canonical_sha256(credential)
    write_json_exclusive(output, credential)
    record = {
        "record_type": "capture_authorization", "experiment_name": EXPERIMENT_NAME,
        "git_commit": expected_git_commit(), "run_id": manifest["run_id"], "recorded_at": utc_now(),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": resolved["current_binding_sha256"],
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "lock_holder_receipt_sha256": holder["lock_holder_receipt_sha256"],
        "artifact": str(output.resolve()), "artifact_sha256": sha256_file(output),
        "parent_credential_id": credential["parent_credential_id"],
        "parent_issuer_pid": issuer_shell_pid, "status": "PASS",
        "decision": "COMMIT_RETAIN_CAPTURE32_CHILD_AUTHORIZED",
        "training_authorized": False, "method_selected": False,
    }
    append_jsonl(manifest["paths"]["execution_ledger"], record)
    _validate_supervisor_schema(read_jsonl(manifest["paths"]["execution_ledger"]))
    return record


def validate_capture_credential(
    credential_path: Path, *, manifest: Mapping[str, Any], resolved: Mapping[str, Any],
    current_binding_sha256: str, require_live_parent: bool,
) -> dict[str, Any]:
    if credential_path.resolve() != Path(manifest["paths"]["capture_credential"]).resolve() \
            or not credential_path.is_file() or credential_path.is_symlink():
        raise ValueError("capture32 credential missing/symlinked/path drifted")
    credential = json.loads(credential_path.read_text())
    unsigned = dict(credential); credential_id = unsigned.pop("parent_credential_id", None)
    credential_fields = {
        "schema", "run_id", "git_commit", "child_kind", "parent_identity",
        "issued_at", "nonce", "single_use", "execution_binding_sha256",
        "runtime_binding_sha256", "current_binding_sha256",
        "gpu_lock_binding_sha256", "lock_holder_receipt",
        "training_authorized", "method_selected", "parent_credential_id",
    }
    if set(credential) != credential_fields \
            or credential_id != canonical_sha256(unsigned) \
            or re.fullmatch(r"[0-9a-f]{64}", str(credential.get("nonce", ""))) is None:
        raise ValueError("capture32 credential digest failed")
    parent = credential.get("parent_identity", {})
    if require_live_parent and (_process_identity(os.getppid()) != parent):
        raise ValueError("capture32 live parent PID/boot/start identity differs")
    holder = capture_lock_holder_receipt(manifest)
    stored_holder = credential.get("lock_holder_receipt")
    if not isinstance(stored_holder, Mapping) or set(stored_holder) != {
        "schema", "physical_gpu_indices", "locks", "holder",
        "gpu_lock_binding_sha256", "lock_holder_receipt_sha256",
    } or holder != stored_holder:
        raise ValueError("capture32 live lock holder receipt differs from credential")
    for field, expected in {
        "schema": "memagent.commit-retain.capture32-parent-credential.v1",
        "run_id": manifest["run_id"], "git_commit": expected_git_commit(),
        "child_kind": "single_engine_exact_32_pair_capture", "single_use": True,
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": current_binding_sha256,
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "training_authorized": False, "method_selected": False,
    }.items():
        if credential.get(field) != expected:
            raise ValueError(f"capture32 credential.{field} binding failed")
    records = read_jsonl(manifest["paths"]["execution_ledger"])
    expected_prefix = ["s0_preflight", "capture_authorization"]
    if validate_jsonl_chain(records) or [x.get("record_type") for x in records[:2]] != expected_prefix:
        raise ValueError("capture32 authorization ledger prefix failed")
    authorization = records[1]
    _validate_supervisor_common(
        authorization,
        record_type="capture_authorization",
        decision="COMMIT_RETAIN_CAPTURE32_CHILD_AUTHORIZED",
        manifest=manifest,
        resolved=resolved,
        artifact=credential_path,
    )
    if authorization.get("artifact_sha256") != sha256_file(credential_path) \
            or authorization.get("parent_credential_id") != credential_id \
            or authorization.get("lock_holder_receipt_sha256") != holder[
                "lock_holder_receipt_sha256"
            ] \
            or authorization.get("parent_issuer_pid") != parent["pid"]:
        raise ValueError("capture32 authorization differs from credential")
    return {
        "parent_credential_id": credential_id,
        "parent_credential_sha256": sha256_file(credential_path),
        "parent_credential_path": str(credential_path.resolve()),
        "parent_identity": parent,
        "parent_issuer_pid": parent["pid"],
        "observed_parent_pid": parent["pid"],
        "parent_authorization_record_sha256": authorization["record_sha256"],
        "lock_holder_receipt_sha256": holder["lock_holder_receipt_sha256"],
        "gpu_lock_binding_sha256": holder["gpu_lock_binding_sha256"],
    }


def consume_credential_and_record_start(
    manifest_path: Path, *, credential_path: Path
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    current_sha = _current_binding(manifest, resolved, full_model_sha=True)
    evidence = validate_capture_credential(
        credential_path, manifest=manifest, resolved=resolved,
        current_binding_sha256=current_sha, require_live_parent=True,
    )
    consumption_path = credential_path.with_name("capture_child_consumed.json")
    consumption = {
        "schema": "memagent.commit-retain.capture32-credential-consumption.v1",
        "run_id": manifest["run_id"], "git_commit": expected_git_commit(),
        "parent_credential_id": evidence["parent_credential_id"],
        "child_identity": _process_identity(os.getpid()),
        "parent_identity": _process_identity(os.getppid()),
        "consumed_at": utc_now(), "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
    }
    consumption["credential_consumption_sha256"] = canonical_sha256(consumption)
    write_json_exclusive(consumption_path, consumption)
    append_jsonl(manifest["paths"]["execution_ledger"], {
        "record_type": "capture_started", "experiment_name": EXPERIMENT_NAME,
        "git_commit": expected_git_commit(), "run_id": manifest["run_id"], "recorded_at": utc_now(),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": current_sha,
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "lock_holder_receipt_sha256": evidence["lock_holder_receipt_sha256"],
        "artifact": str(consumption_path.resolve()), "artifact_sha256": sha256_file(consumption_path),
        "parent_credential_id": evidence["parent_credential_id"],
        "credential_consumption_sha256": consumption["credential_consumption_sha256"],
        "status": "PASS", "decision": "COMMIT_RETAIN_CAPTURE32_STARTED",
        "training_authorized": False, "method_selected": False,
    })
    _validate_supervisor_schema(read_jsonl(manifest["paths"]["execution_ledger"]))
    return {**evidence, "credential_consumption_path": str(consumption_path.resolve()),
            "credential_consumption_file_sha256": sha256_file(consumption_path),
            "credential_consumption_sha256": consumption["credential_consumption_sha256"]}


def _tokenizer(manifest: Mapping[str, Any]):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(manifest["model"]["path"], trust_remote_code=True, local_files_only=True)


def execution_frozen_pairs(manifest: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    return _execution_frozen_pairs(manifest, _load_preregistration(manifest), tokenizer)


def project_frozen_pair_eval_identity(
    frozen_pairs: Sequence[Mapping[str, Any]], eval_manifest_hash: str
) -> list[dict[str, Any]]:
    """Attach the authenticated manifest hash to execution-only pair identities.

    The preregistered inventory deliberately stores this hash once at the
    manifest level.  The generic COMMIT/RETAIN ledger validator requires it on
    every execution identity.  Projection is therefore explicit and rejects a
    conflicting row-level copy rather than silently overwriting it.
    """
    if not isinstance(eval_manifest_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", eval_manifest_hash
    ):
        raise ValueError("capture32 eval_manifest_hash is not a canonical SHA-256")
    projected: list[dict[str, Any]] = []
    for index, item in enumerate(frozen_pairs):
        row = dict(item)
        row_copy = row.get("eval_manifest_hash")
        if row_copy is not None and row_copy != eval_manifest_hash:
            raise ValueError(
                f"capture32 frozen row {index} eval_manifest_hash conflicts with P0"
            )
        row["eval_manifest_hash"] = eval_manifest_hash
        projected.append(row)
    return projected


def _expected_run_receipt(
    *, manifest: Mapping[str, Any], resolved: Mapping[str, Any], current_binding_sha256: str,
    capture_report: Mapping[str, Any], capture_path: Path,
) -> dict[str, Any]:
    execution = read_jsonl(capture_path)[0]["pair"]["execution"]
    receipt = {
        "schema": "memagent.commit-retain.capture32-run-receipt.v1",
        "run_id": manifest["run_id"], "git_commit": expected_git_commit(),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": current_binding_sha256,
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "capture_ledger": str(capture_path.resolve()), "capture_ledger_sha256": sha256_file(capture_path),
        "pair_count": capture_report["pair_count"], "pair_ids": capture_report["pair_ids"],
        "stable_write_ids": capture_report["stable_write_ids"],
        "generate_call_count": capture_report["generate_call_count"],
        "execution": execution, "training": capture_report["training"],
        "claim_boundary": capture_report["claim_boundary"],
    }
    receipt["run_receipt_id"] = canonical_sha256(receipt)
    return receipt


def validate_capture_artifacts(manifest_path: Path, *, require_supervisor_receipt: bool) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    current_sha = _current_binding(manifest, resolved, full_model_sha=False)
    capture_path, receipt_path = Path(manifest["paths"]["capture_ledger"]), Path(manifest["paths"]["capture_run_receipt"])
    for path in (capture_path, receipt_path):
        _assert_regular_confined(path, Path(manifest["paths"]["log_root"]), must_exist=True)
    tokenizer = _tokenizer(manifest)
    from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template
    chat = chat_template(tokenizer)
    writer, reader = TokenTemplate(chat.format(message=TEMPLATE), tokenizer), TokenTemplate(chat.format(message=TEMPLATE_FINAL_BOXED), tokenizer)
    frozen = execution_frozen_pairs(manifest, tokenizer)
    validation_frozen = project_frozen_pair_eval_identity(
        frozen, resolved["eval_manifest_hash"]
    )
    credential_path = Path(manifest["paths"]["capture_credential"])
    credential = validate_capture_credential(
        credential_path, manifest=manifest, resolved=resolved,
        current_binding_sha256=current_sha, require_live_parent=False,
    )
    consumption_path = credential_path.with_name("capture_child_consumed.json")
    _assert_regular_confined(
        consumption_path, Path(manifest["paths"]["log_root"]), must_exist=True
    )
    consumption = json.loads(consumption_path.read_text())
    if set(consumption) != {
        "schema", "run_id", "git_commit", "parent_credential_id",
        "child_identity", "parent_identity", "consumed_at",
        "gpu_lock_binding_sha256", "credential_consumption_sha256",
    }:
        raise ValueError("capture32 credential consumption fields drifted")
    for identity_name in ("child_identity", "parent_identity"):
        identity = consumption.get(identity_name)
        if not isinstance(identity, Mapping) or set(identity) != {
            "pid", "boot_id", "process_start_ticks"
        } or type(identity.get("pid")) is not int or identity["pid"] < 1 \
                or not isinstance(identity.get("boot_id"), str) \
                or not identity["boot_id"] \
                or type(identity.get("process_start_ticks")) is not int \
                or identity["process_start_ticks"] < 0:
            raise ValueError(
                f"capture32 credential consumption {identity_name} is invalid"
            )
    unsigned_consumption = dict(consumption)
    persisted_consumption_sha = unsigned_consumption.pop(
        "credential_consumption_sha256", None
    )
    if persisted_consumption_sha != canonical_sha256(unsigned_consumption) \
            or consumption.get("schema") != "memagent.commit-retain.capture32-credential-consumption.v1" \
            or consumption.get("run_id") != manifest["run_id"] \
            or consumption.get("git_commit") != expected_git_commit() \
            or consumption.get("parent_credential_id") != credential["parent_credential_id"] \
            or consumption.get("parent_identity") != credential.get("parent_identity") \
            or consumption.get("gpu_lock_binding_sha256") != resolved["gpu_lock_binding_sha256"]:
        raise ValueError("capture32 credential consumption receipt failed")
    runtime_expected = {
        **expected_pair_binding(manifest, resolved, tokenizer),
        "lock_holder_receipt_sha256": credential["lock_holder_receipt_sha256"],
        "credential_consumption_sha256": persisted_consumption_sha,
        "credential_consumption_file_sha256": sha256_file(consumption_path),
        "credential_consumption_path": str(consumption_path.resolve()),
    }
    report = validate_capture_ledger(
        read_jsonl(capture_path), frozen_pairs=validation_frozen,
        experiment_name=EXPERIMENT_NAME,
        git_commit=expected_git_commit(), run_id=manifest["run_id"],
        execution_binding_sha256=resolved["execution_binding_sha256"],
        runtime_binding_sha256=resolved["runtime_binding_sha256"],
        current_binding_sha256=current_sha,
        decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        writer_prompt_builder=lambda q, m, c: writer.format(prompt=q, memory=m, chunk=c).tolist(),
        reader_prompt_builder=lambda q, m: reader.format(prompt=q, memory=m).tolist(),
        expected_pair_binding=runtime_expected, expected_pair_count=32,
    )
    if report["pair_count"] != 32 or set(report["stable_write_ids"]) != {x["stable_write_id"] for x in frozen}:
        raise ValueError("capture32 attrition/substitution detected")
    expected_receipt = _expected_run_receipt(manifest=manifest, resolved=resolved,
        current_binding_sha256=current_sha, capture_report=report, capture_path=capture_path)
    if canonical_json(json.loads(receipt_path.read_text())) != canonical_json(expected_receipt):
        raise ValueError("capture32 run receipt does not reproduce")
    records = read_jsonl(manifest["paths"]["execution_ledger"])
    if validate_jsonl_chain(records) or len(records) < 3 \
            or [x.get("record_type") for x in records[:3]] != [
                "s0_preflight", "capture_authorization", "capture_started"
            ]:
        raise ValueError("capture32 supervisor start sequence failed")
    _validate_supervisor_schema(records)
    started = records[2]
    _validate_supervisor_common(
        started,
        record_type="capture_started",
        decision="COMMIT_RETAIN_CAPTURE32_STARTED",
        manifest=manifest,
        resolved=resolved,
        artifact=consumption_path,
    )
    if started.get("lock_holder_receipt_sha256") != credential[
        "lock_holder_receipt_sha256"
    ] or started.get("parent_credential_id") != credential["parent_credential_id"] \
            or started.get("credential_consumption_sha256") != persisted_consumption_sha:
        raise ValueError("capture32 supervisor start/credential receipt differs")
    execution = read_jsonl(capture_path)[0]["pair"]["execution"]
    if execution.get("process_pid") != consumption["child_identity"]["pid"] \
            or execution.get("parent_issuer_pid") != consumption["parent_identity"]["pid"] \
            or execution.get("observed_parent_pid") != consumption["parent_identity"]["pid"]:
        raise ValueError("capture32 child/parent process identity differs from capture")
    if require_supervisor_receipt:
        expected = ["s0_preflight", "capture_authorization", "capture_started", "capture_complete"]
        if validate_jsonl_chain(records) or [x.get("record_type") for x in records[:4]] != expected:
            raise ValueError("capture32 supervisor capture sequence failed")
        record = records[3]
        _validate_supervisor_common(
            record,
            record_type="capture_complete",
            decision="COMMIT_RETAIN_CAPTURE32_COMPLETE",
            manifest=manifest,
            resolved=resolved,
            artifact=capture_path,
        )
        if record.get("artifact_sha256") != sha256_file(capture_path) \
                or record.get("pair_count") != 32 \
                or record.get("pair_ids") != report["pair_ids"] \
                or record.get("stable_write_ids") != report["stable_write_ids"] \
                or record.get("generate_call_count") != 353 \
                or record.get("run_receipt") != str(receipt_path.resolve()) \
                or record.get("run_receipt_sha256") != sha256_file(receipt_path):
            raise ValueError("capture32 supervisor receipt differs from capture")
    return {**report, "git_commit": expected_git_commit(), "run_id": manifest["run_id"],
            "eval_manifest_hash": resolved["eval_manifest_hash"],
            "execution_binding_sha256": resolved["execution_binding_sha256"],
            "runtime_binding_sha256": resolved["runtime_binding_sha256"],
            "current_binding_sha256": current_sha,
            "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
            "capture_ledger": str(capture_path.resolve()), "capture_ledger_sha256": sha256_file(capture_path),
            "capture_run_receipt": str(receipt_path.resolve()), "capture_run_receipt_sha256": sha256_file(receipt_path)}


def build_final_audit_report(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    capture = validate_capture_artifacts(manifest_path, require_supervisor_receipt=True)
    p0_path, resolved_path = Path(manifest["paths"]["p0_certificate"]), Path(manifest["paths"]["resolved_manifest"])
    return {
        "schema": "memagent.commit-retain.capture32-final.v2", "status": "PASS",
        "decision": "COMMIT_RETAIN_CAPTURE32_LOCAL_AUDIT_COMPLETE_PROVENANCE_PENDING",
        "run_id": manifest["run_id"], "git_commit": expected_git_commit(),
        "eval_manifest_hash": capture["eval_manifest_hash"],
        "preregistration_sha256": _load_preregistration(manifest)["preregistration_sha256"],
        "p0_certificate": str(p0_path.resolve()), "p0_certificate_sha256": sha256_file(p0_path),
        "resolved_manifest": str(resolved_path.resolve()), "resolved_manifest_sha256": sha256_file(resolved_path),
        **{key: capture[key] for key in ("capture_ledger", "capture_ledger_sha256", "capture_run_receipt",
           "capture_run_receipt_sha256", "pair_count", "stable_write_ids", "pair_ids", "generate_call_count", "outcomes")},
        "gpu_lock_binding_sha256": capture["gpu_lock_binding_sha256"],
        "external_provenance_status": "PENDING_EXTERNAL_SIGNATURE",
        "training": {"trainer_attached": False, "actor_updates": 0, "optimizer_steps": 0},
        "claim_boundary": {"development_admissibility_only": True, "method_selected": False,
            "training_authorized": False, "paper_performance_result": False, "causal_effect_claim": False},
    }


def record_capture_complete(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    report = validate_capture_artifacts(manifest_path, require_supervisor_receipt=False)
    receipt = Path(manifest["paths"]["capture_run_receipt"])
    record = {
        "record_type": "capture_complete", "experiment_name": EXPERIMENT_NAME,
        "git_commit": expected_git_commit(), "run_id": manifest["run_id"], "recorded_at": utc_now(),
        "eval_manifest_hash": report["eval_manifest_hash"],
        "execution_binding_sha256": report["execution_binding_sha256"],
        "runtime_binding_sha256": report["runtime_binding_sha256"],
        "current_binding_sha256": report["current_binding_sha256"],
        "gpu_lock_binding_sha256": report["gpu_lock_binding_sha256"],
        "artifact": report["capture_ledger"], "artifact_sha256": report["capture_ledger_sha256"],
        "run_receipt": str(receipt.resolve()), "run_receipt_sha256": sha256_file(receipt),
        "pair_count": 32, "pair_ids": report["pair_ids"], "stable_write_ids": report["stable_write_ids"],
        "generate_call_count": report["generate_call_count"], "status": "PASS",
        "decision": "COMMIT_RETAIN_CAPTURE32_COMPLETE", "training_authorized": False, "method_selected": False,
    }
    append_jsonl(manifest["paths"]["execution_ledger"], record)
    _validate_supervisor_schema(read_jsonl(manifest["paths"]["execution_ledger"]))
    return record


def write_final_and_terminal_anchor(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    final_path, anchor_path = Path(manifest["paths"]["final_report"]), Path(manifest["paths"]["external_terminal_anchor"])
    if final_path.exists() or anchor_path.exists():
        raise FileExistsError("refuse to overwrite capture32 final/terminal anchor")
    report = build_final_audit_report(manifest_path)
    write_json_exclusive(final_path, report)
    append_jsonl(manifest["paths"]["execution_ledger"], {
        "record_type": "audit_result", "experiment_name": EXPERIMENT_NAME,
        "git_commit": expected_git_commit(), "run_id": manifest["run_id"], "recorded_at": utc_now(),
        "eval_manifest_hash": report["eval_manifest_hash"],
        "execution_binding_sha256": json.loads(Path(manifest["paths"]["resolved_manifest"]).read_text())["execution_binding_sha256"],
        "runtime_binding_sha256": json.loads(Path(manifest["paths"]["resolved_manifest"]).read_text())["runtime_binding_sha256"],
        "current_binding_sha256": json.loads(Path(manifest["paths"]["resolved_manifest"]).read_text())["current_binding_sha256"],
        "gpu_lock_binding_sha256": report["gpu_lock_binding_sha256"],
        "artifact": str(final_path.resolve()), "artifact_sha256": sha256_file(final_path),
        "pair_count": 32, "pair_ids": report["pair_ids"], "stable_write_ids": report["stable_write_ids"],
        "generate_call_count": report["generate_call_count"], "status": "PASS",
        "decision": report["decision"], "training_authorized": False, "method_selected": False,
    })
    ledger = Path(manifest["paths"]["execution_ledger"])
    anchor = {
        "schema": "memagent.commit-retain.capture32-provenance-anchor.v1",
        "anchor_kind": "POST_CAPTURE_LOCAL_EXPORT_CANDIDATE",
        "trust_status": "PENDING_EXTERNAL_SIGNATURE",
        "run_id": manifest["run_id"], "git_commit": expected_git_commit(), "recorded_at": utc_now(),
        "pre_generation_anchor": str(Path(manifest["paths"]["external_preregistration_anchor"]).resolve()),
        "pre_generation_anchor_sha256": sha256_file(manifest["paths"]["external_preregistration_anchor"]),
        "supervisor_ledger": str(ledger.resolve()), "supervisor_ledger_sha256": sha256_file(ledger),
        "supervisor_tail_sha256": read_jsonl(ledger)[-1]["record_sha256"],
        "capture_ledger": report["capture_ledger"], "capture_ledger_sha256": report["capture_ledger_sha256"],
        "capture_run_receipt": report["capture_run_receipt"],
        "capture_run_receipt_sha256": report["capture_run_receipt_sha256"],
        "final_report": str(final_path.resolve()), "final_report_sha256": sha256_file(final_path),
        "pair_ids_sha256": canonical_sha256(report["pair_ids"]),
        "stable_write_ids_sha256": canonical_sha256(report["stable_write_ids"]),
        "training_authorized": False, "method_selected": False,
    }
    anchor["anchor_payload_sha256"] = canonical_sha256(anchor)
    _validate_anchor_schema(anchor)
    write_json_exclusive(anchor_path, anchor)
    _validate_supervisor_schema(read_jsonl(ledger))
    return report


def verify_existing(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    report = build_final_audit_report(manifest_path)
    final_path = Path(manifest["paths"]["final_report"])
    if canonical_json(json.loads(final_path.read_text())) != canonical_json(report):
        raise ValueError("capture32 final report does not reproduce")
    ledger = read_jsonl(manifest["paths"]["execution_ledger"])
    if validate_jsonl_chain(ledger) or [x.get("record_type") for x in ledger] != [
        "s0_preflight", "capture_authorization", "capture_started", "capture_complete", "audit_result"
    ]:
        raise ValueError("capture32 finalized supervisor state machine failed")
    _validate_supervisor_schema(ledger)
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    p0_path = Path(manifest["paths"]["p0_certificate"])
    _validate_pre_generation_anchor(
        manifest, p0_path=p0_path, resolved_path=resolved_path
    )
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    audit = ledger[-1]
    if audit.get("experiment_name") != EXPERIMENT_NAME \
            or audit.get("git_commit") != expected_git_commit() \
            or audit.get("run_id") != manifest["run_id"] \
            or audit.get("eval_manifest_hash") != report["eval_manifest_hash"] \
            or audit.get("execution_binding_sha256") != resolved["execution_binding_sha256"] \
            or audit.get("runtime_binding_sha256") != resolved["runtime_binding_sha256"] \
            or audit.get("current_binding_sha256") != resolved["current_binding_sha256"] \
            or audit.get("gpu_lock_binding_sha256") != report["gpu_lock_binding_sha256"] \
            or audit.get("artifact") != str(final_path.resolve()) \
            or audit.get("artifact_sha256") != sha256_file(final_path) \
            or audit.get("pair_count") != 32 \
            or audit.get("pair_ids") != report["pair_ids"] \
            or audit.get("stable_write_ids") != report["stable_write_ids"] \
            or audit.get("generate_call_count") != 353 \
            or audit.get("status") != "PASS" \
            or audit.get("decision") != report["decision"] \
            or audit.get("training_authorized") is not False \
            or audit.get("method_selected") is not False:
        raise ValueError("capture32 final supervisor audit receipt failed")
    anchor_path = Path(manifest["paths"]["external_terminal_anchor"])
    _assert_regular_confined(anchor_path, Path(manifest["work_root"]), must_exist=True)
    anchor = json.loads(anchor_path.read_text())
    _validate_anchor_schema(anchor)
    unsigned = dict(anchor)
    digest = unsigned.pop("anchor_payload_sha256", None)
    expected_fields = {
        "schema", "anchor_kind", "trust_status", "run_id", "git_commit",
        "recorded_at", "pre_generation_anchor", "pre_generation_anchor_sha256",
        "supervisor_ledger", "supervisor_ledger_sha256",
        "supervisor_tail_sha256", "capture_ledger", "capture_ledger_sha256",
        "capture_run_receipt", "capture_run_receipt_sha256", "final_report",
        "final_report_sha256", "pair_ids_sha256", "stable_write_ids_sha256",
        "training_authorized", "method_selected", "anchor_payload_sha256",
    }
    pre_anchor = Path(manifest["paths"]["external_preregistration_anchor"])
    capture_path = Path(manifest["paths"]["capture_ledger"])
    receipt_path = Path(manifest["paths"]["capture_run_receipt"])
    if set(anchor) != expected_fields \
            or digest != canonical_sha256(unsigned) \
            or anchor.get("schema") != "memagent.commit-retain.capture32-provenance-anchor.v1" \
            or anchor.get("anchor_kind") != "POST_CAPTURE_LOCAL_EXPORT_CANDIDATE" \
            or anchor.get("trust_status") != "PENDING_EXTERNAL_SIGNATURE" \
            or anchor.get("run_id") != manifest["run_id"] \
            or anchor.get("git_commit") != expected_git_commit() \
            or anchor.get("pre_generation_anchor") != str(pre_anchor.resolve()) \
            or anchor.get("pre_generation_anchor_sha256") != sha256_file(pre_anchor) \
            or anchor.get("supervisor_ledger") != str(Path(manifest["paths"]["execution_ledger"]).resolve()) \
            or anchor.get("supervisor_ledger_sha256") != sha256_file(manifest["paths"]["execution_ledger"]) \
            or anchor.get("supervisor_tail_sha256") != ledger[-1]["record_sha256"] \
            or anchor.get("capture_ledger") != str(capture_path.resolve()) \
            or anchor.get("capture_ledger_sha256") != sha256_file(capture_path) \
            or anchor.get("capture_run_receipt") != str(receipt_path.resolve()) \
            or anchor.get("capture_run_receipt_sha256") != sha256_file(receipt_path) \
            or anchor.get("final_report") != str(final_path.resolve()) \
            or anchor.get("final_report_sha256") != sha256_file(final_path) \
            or anchor.get("pair_ids_sha256") != canonical_sha256(report["pair_ids"]) \
            or anchor.get("stable_write_ids_sha256") != canonical_sha256(report["stable_write_ids"]) \
            or anchor.get("training_authorized") is not False \
            or anchor.get("method_selected") is not False:
        raise ValueError("capture32 terminal external export anchor failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_REL)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--write-certificate", action="store_true")
    parser.add_argument("--validate-p0-prefix", action="store_true")
    parser.add_argument("--issue-capture-credential", type=Path)
    parser.add_argument("--issuer-shell-pid", type=int)
    parser.add_argument("--record-capture-complete", action="store_true")
    parser.add_argument("--write-final", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.issue_capture_credential:
        result = issue_capture_credential(args.manifest, output=args.issue_capture_credential,
                                          issuer_shell_pid=args.issuer_shell_pid or 0)
    elif args.validate_p0_prefix:
        validate_p0(args.manifest); result = {"status": "PASS", "decision": "CAPTURE32_P0_PREFIX_VALID"}
    elif args.record_capture_complete:
        result = record_capture_complete(args.manifest)
    elif args.write_final:
        result = write_final_and_terminal_anchor(args.manifest)
    elif args.verify_existing:
        result = verify_existing(args.manifest)
    elif args.write_certificate:
        result = write_preflight(args.manifest, check_runtime=args.check_runtime)
    else:
        result = run_preflight(args.manifest, check_runtime=args.check_runtime)[0]
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
