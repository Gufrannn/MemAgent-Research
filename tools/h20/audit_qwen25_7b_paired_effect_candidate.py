#!/usr/bin/env python3
"""Authenticate COMMIT/RETAIN evidence and build a paired-effect candidate.

This is an offline, read-only evidence consumer.  It never accepts a stored
candidate score, never attaches to the trainer, and never authorizes GPU work.
The four-pair capture exercises the pipeline but is permanently pilot-only.
Only the separately frozen, disjoint capture32 inventory may enter the
preregistered sample and predictive-signal gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))


def _preexisting_repo_python_objects() -> frozenset[str]:
    """Separate this auditor's imports from unrelated embedding-process state."""
    objects: set[str] = set()
    for module in tuple(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        try:
            relative = Path(origin).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            continue
        if relative.endswith(".py"):
            objects.add(relative)
    return frozenset(objects)


_PREEXISTING_REPO_PYTHON_OBJECTS = _preexisting_repo_python_objects()

import recurrent.research.commit_retain_capture as _capture_module  # noqa: E402
import recurrent.research.paired_effect_credit as _paired_module  # noqa: E402
import tools.h20.preflight_qwen25_7b_commit_retain as _capture_preflight_module  # noqa: E402
import tools.h20.preflight_qwen25_7b_commit_retain_capture32 as _capture32_preflight_module  # noqa: E402

from recurrent.research.commit_retain_capture import (  # noqa: E402
    ARMS,
    build_capture_envelope,
    canonical_json,
    canonical_sha256,
    validate_capture_ledger,
)
from recurrent.research.gate_a_execution import (  # noqa: E402
    sha256_file,
    validate_jsonl_chain,
)
from recurrent.research.paired_effect_credit import (  # noqa: E402
    CAPTURE32_COUNT,
    CAPTURE32_PREREG_SCHEMA,
    CANDIDATE_ID,
    EXPECTED_S128_AUTHORITY_FILE_SHA256,
    FEATURE_SCHEMA,
    S128_AUTHORITY_REL,
    TARGET_NAME,
    build_crossfit_bundle,
    order_and_validate_capture32_pairs,
    recompute_capture32_source_evidence,
    validate_capture32_authority_binding,
    validate_capture32_preregistration,
    validate_s128_authority,
    validate_crossfit_bundle,
)
from recurrent.research.stable_eval_identity import validate_resolved_manifest  # noqa: E402
from recurrent.research.serialization_credit_pilots import (  # noqa: E402
    read_jsonl,
    write_json_exclusive,
)
from tools.h20.preflight_qwen25_7b_commit_retain import (  # noqa: E402
    _code_objects as capture_code_objects,
    _expected_run_receipt,
    _gpu_profile,
    _model_loading_paths,
    _tokenizer,
    _validate_manifest as validate_capture_manifest,
    expected_pair_binding,
    experiment_name as capture_experiment_name,
    load_manifest as load_capture_manifest,
    validate_capture_credential,
    validate_p0,
)
from tools.h20.preflight_qwen25_7b_commit_retain_capture32 import (  # noqa: E402
    _validate_manifest as validate_capture32_producer_manifest,
    load_manifest as load_capture32_producer_manifest,
)


_EXPECTED_IMPORT_ORIGINS = {
    _capture_module: REPO_ROOT / "recurrent/research/commit_retain_capture.py",
    _paired_module: REPO_ROOT / "recurrent/research/paired_effect_credit.py",
    _capture_preflight_module: REPO_ROOT / "tools/h20/preflight_qwen25_7b_commit_retain.py",
    _capture32_preflight_module:
        REPO_ROOT / "tools/h20/preflight_qwen25_7b_commit_retain_capture32.py",
}
for _module, _expected_origin in _EXPECTED_IMPORT_ORIGINS.items():
    if Path(_module.__file__).resolve() != _expected_origin.resolve():
        raise ImportError(
            f"paired-effect audit imported {_module.__name__} outside REPO_ROOT"
        )


def _assert_loaded_repo_module_origins() -> None:
    for name, module in tuple(sys.modules.items()):
        if not (name == "recurrent" or name.startswith("recurrent.")
                or name == "tools.h20" or name.startswith("tools.h20.")):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        try:
            Path(origin).resolve().relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise ImportError(f"paired-effect audit imported {name} outside REPO_ROOT") from error


_assert_loaded_repo_module_origins()


MANIFEST_REL = "manifests/h20/qwen25_7b_paired_effect_candidate_seed2026.json"
CAPTURE32_PREREG_REL = (
    "manifests/h20/qwen25_7b_paired_effect_capture32_preregistration.json"
)
CAPTURE_SOURCE_BASE_COMMIT = "e019e7655046f34d368a82e7d5ea6d72c464ffc7"
REPORT_SCHEMA = "memagent.paired-effect.admissibility-report.v1"
REQUIRED_ENV = (
    "MEMAGENT_PAIRED_EFFECT_WORK_ROOT",
    "MEMAGENT_PAIRED_EFFECT_REPO_DIR",
    "MEMAGENT_PAIRED_EFFECT_EXPECTED_COMMIT",
    "MEMAGENT_PAIRED_EFFECT_RUN_ID",
    "MEMAGENT_PAIRED_EFFECT_CAPTURE_RUN_ID",
)
CAPTURE_ENV = (
    "MEMAGENT_COMMIT_RETAIN_WORK_ROOT",
    "MEMAGENT_COMMIT_RETAIN_REPO_DIR",
    "MEMAGENT_COMMIT_RETAIN_EXPECTED_COMMIT",
    "MEMAGENT_COMMIT_RETAIN_RUN_ID",
)
RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{1,31}")
FULL_SHA = re.compile(r"[0-9a-f]{40}")
PIPELINE_CODE_OBJECTS = (
    "recurrent/research/commit_retain_capture.py",
    "recurrent/research/gate_a_execution.py",
    "recurrent/research/paired_effect_credit.py",
    "recurrent/research/s128_hotpot_metrics.py",
    "recurrent/research/serialization_credit_pilots.py",
    "recurrent/research/stable_eval_identity.py",
    "recurrent/research/trajectory_seeding.py",
    "recurrent/impls/memory.py",
    "recurrent/utils.py",
    "tools/h20/audit_qwen25_7b_paired_effect_candidate.py",
    "tools/h20/preflight_qwen25_7b_commit_retain.py",
    "tools/h20/preflight_qwen25_7b_serialization_credit.py",
    "tools/h20/preflight_qwen25_7b_s128_it.py",
    "tools/h20/preflight_qwen25_7b_stable_i4x2.py",
    "tools/h20/preflight_qwen25_7b_commit_retain_capture32.py",
    "tools/h20/run_qwen25_7b_commit_retain_capture32.py",
    "scripts/h20/commit_retain_capture32_common.sh",
    "scripts/h20/preflight_qwen25_7b_commit_retain_capture32.sh",
    "scripts/h20/run_qwen25_7b_commit_retain_capture32.sh",
    "manifests/h20/qwen25_7b_commit_retain_capture32_seed2026.json",
    "manifests/h20/qwen25_7b_commit_retain_capture32_commands.json",
    "commit_retain_capture32_execution_ledger.schema.json",
    "commit_retain_capture32_provenance_anchor.schema.json",
    "docs/h20/commit_retain_capture32_freeze_20260821.md",
    MANIFEST_REL,
    CAPTURE32_PREREG_REL,
    S128_AUTHORITY_REL,
    "manifests/h20/qwen25_7b_stable_i4x2_seed2026.json",
    "paired_effect_admissibility_report.schema.json",
    "docs/h20/paired_effect_candidate_readiness_20260821.md",
)
CAPTURE32_EXECUTION_CODE_OBJECTS = (
    "recurrent/research/commit_retain_capture.py",
    "recurrent/research/gate_a_execution.py",
    "recurrent/research/s128_hotpot_metrics.py",
    "recurrent/research/serialization_credit_pilots.py",
    "recurrent/research/stable_eval_identity.py",
    "recurrent/research/trajectory_seeding.py",
    "recurrent/impls/memory.py",
    "recurrent/utils.py",
    "tools/h20/preflight_qwen25_7b_commit_retain.py",
    "tools/h20/preflight_qwen25_7b_serialization_credit.py",
    "tools/h20/preflight_qwen25_7b_s128_it.py",
    "tools/h20/preflight_qwen25_7b_stable_i4x2.py",
    "tools/h20/preflight_qwen25_7b_commit_retain_capture32.py",
    "tools/h20/run_qwen25_7b_commit_retain_capture32.py",
    "scripts/h20/commit_retain_capture32_common.sh",
    "scripts/h20/preflight_qwen25_7b_commit_retain_capture32.sh",
    "scripts/h20/run_qwen25_7b_commit_retain_capture32.sh",
    "manifests/h20/qwen25_7b_commit_retain_capture32_seed2026.json",
    "manifests/h20/qwen25_7b_commit_retain_capture32_commands.json",
    "commit_retain_capture32_execution_ledger.schema.json",
    "commit_retain_capture32_provenance_anchor.schema.json",
    "docs/h20/commit_retain_capture32_freeze_20260821.md",
    "manifests/h20/qwen25_7b_commit_retain_capture_seed2026.json",
    "recurrent/research/paired_effect_credit.py",
    "tools/h20/run_qwen25_7b_commit_retain.py",
    CAPTURE32_PREREG_REL,
    S128_AUTHORITY_REL,
)


def _loaded_pipeline_python_objects() -> tuple[str, ...]:
    """Snapshot repository Python dependencies imported by this auditor."""
    objects: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        if not (name.startswith("recurrent.research.") or name.startswith("tools.h20.")):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        try:
            relative = Path(origin).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            continue
        if relative.endswith(".py"):
            objects.add(relative)
    return tuple(sorted(objects - _PREEXISTING_REPO_PYTHON_OBJECTS))


LOADED_PIPELINE_PYTHON_OBJECTS = _loaded_pipeline_python_objects()
if not set(LOADED_PIPELINE_PYTHON_OBJECTS).issubset(PIPELINE_CODE_OBJECTS):
    raise ImportError(
        "paired-effect audit loaded unauthenticated repository modules: "
        f"{sorted(set(LOADED_PIPELINE_PYTHON_OBJECTS) - set(PIPELINE_CODE_OBJECTS))}"
    )


class CapturePendingError(RuntimeError):
    def __init__(self, missing: Sequence[str]):
        self.missing = list(missing)
        super().__init__(f"capture evidence is incomplete: {self.missing}")


class Capture32AttritionError(RuntimeError):
    """Raised once capture32 crossed P0 commitment but is not exact 32/32."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=not binary
    ).strip() if not binary else subprocess.check_output(
        ["git", "-C", str(repo), *args]
    )


def _git_blob(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"]
    )


def _is_sha(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _resolve_environment(
    raw: Mapping[str, Any], environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    source = os.environ if environment is None else environment
    missing = [name for name in REQUIRED_ENV if not source.get(name)]
    if missing:
        raise ValueError(f"missing task-scoped runtime bindings: {missing}")
    work_root = Path(source[REQUIRED_ENV[0]])
    repo = Path(source[REQUIRED_ENV[1]])
    expected_commit = source[REQUIRED_ENV[2]]
    run_id = source[REQUIRED_ENV[3]]
    capture_run_id = source[REQUIRED_ENV[4]]
    if not work_root.is_absolute() or not repo.is_absolute():
        raise ValueError("work root and repo dir must be absolute")
    if work_root.resolve() != work_root or repo.resolve() != repo:
        raise ValueError("work root and repo dir must be canonical real paths")
    if repo.resolve() != REPO_ROOT.resolve():
        raise ValueError("MEMAGENT_PAIRED_EFFECT_REPO_DIR differs from this checkout")
    if FULL_SHA.fullmatch(expected_commit) is None:
        raise ValueError("MEMAGENT_PAIRED_EFFECT_EXPECTED_COMMIT must be a full Git SHA")
    if RUN_ID.fullmatch(run_id) is None or RUN_ID.fullmatch(capture_run_id) is None:
        raise ValueError("paired-effect or capture run ID has an invalid format")
    replacements = {
        "${MEMAGENT_PAIRED_EFFECT_WORK_ROOT}": str(work_root),
        "${MEMAGENT_PAIRED_EFFECT_REPO_DIR}": str(repo),
        "${MEMAGENT_PAIRED_EFFECT_RUN_ID}": run_id,
        "${MEMAGENT_PAIRED_EFFECT_CAPTURE_RUN_ID}": capture_run_id,
    }

    def resolve(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: resolve(value) for key, value in item.items()}
        if isinstance(item, list):
            return [resolve(value) for value in item]
        if isinstance(item, str):
            value = item
            for placeholder, replacement in replacements.items():
                value = value.replace(placeholder, replacement)
            if "${" in value:
                raise ValueError(f"unresolved placeholder: {value}")
            return value
        return item

    result = resolve(dict(raw))
    result["work_root"] = str(work_root)
    result["repo_dir"] = str(repo)
    result["run_id"] = run_id
    result["capture_run_id"] = capture_run_id
    result["expected_pipeline_commit"] = expected_commit
    _validate_candidate_manifest(result)
    return result


def load_manifest(
    path: str | Path, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    manifest_path = Path(path)
    canonical = REPO_ROOT / MANIFEST_REL
    if manifest_path.is_symlink() or manifest_path.resolve() != canonical.resolve():
        raise ValueError("only the Git-bound paired-effect manifest is accepted")
    return _resolve_environment(
        json.loads(manifest_path.read_text(encoding="utf-8")), environment
    )


def load_s128_authority(manifest: Mapping[str, Any]) -> dict[str, Any]:
    authority_path = Path(manifest["repo_dir"]) / S128_AUTHORITY_REL
    expected_authority = REPO_ROOT / S128_AUTHORITY_REL
    if authority_path.is_symlink() or authority_path.resolve() != expected_authority.resolve() \
            or not authority_path.is_file():
        raise ValueError("S128 authority is missing, symlinked, or outside REPO_ROOT")
    if sha256_file(authority_path) != EXPECTED_S128_AUTHORITY_FILE_SHA256:
        raise ValueError("S128 authority file SHA differs from frozen preregistration")
    return validate_s128_authority(
        json.loads(authority_path.read_text(encoding="utf-8"))
    )


def load_capture32_preregistration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(manifest["repo_dir"]) / CAPTURE32_PREREG_REL
    expected = REPO_ROOT / CAPTURE32_PREREG_REL
    if path.is_symlink() or path.resolve() != expected.resolve() or not path.is_file():
        raise ValueError("capture32 preregistration is missing, symlinked, or outside REPO_ROOT")
    preregistration = validate_capture32_preregistration(
        json.loads(path.read_text(encoding="utf-8"))
    )
    authority = load_s128_authority(manifest)
    return validate_capture32_authority_binding(preregistration, authority)


def _authenticate_capture32_source_replay(
    *,
    manifest: Mapping[str, Any],
    capture_manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    tokenizer: Any,
    writer_prompt_builder: Any,
) -> dict[str, Any]:
    """Replay the exact raw S128/tokenizer selection before accepting outcomes."""
    authority = load_s128_authority(manifest)
    data_path = Path(capture_manifest["data"]["validation"])
    model_root = Path(capture_manifest["model"]["path"])
    stable_path = Path(capture_manifest["stable_identity_prerequisite"]["resolved_manifest"])
    for field, path in {
        "S128 parquet": data_path,
        "model root": model_root,
        "Stable-I resolved manifest": stable_path,
    }.items():
        if path.is_symlink() or path.resolve() != path or not path.exists():
            raise ValueError(f"capture32 {field} is missing, symlinked, or non-canonical")
    if sha256_file(data_path) != preregistration["source"]["validation_sha256"]:
        raise ValueError("capture32 raw S128 parquet SHA drifted")

    stable_resolved = validate_resolved_manifest(
        json.loads(stable_path.read_text(encoding="utf-8"))
    )
    if stable_resolved["eval_manifest_hash"] != authority["eval_manifest_hash"] \
            or canonical_json(stable_resolved["identity_payload"]) != canonical_json(
                authority["identity_payload"]
            ):
        raise ValueError("capture32 runtime Stable-I identity differs from authority")

    actual_model_files: list[dict[str, Any]] = []
    for expected in capture_manifest["model"]["files"]:
        path = model_root / expected["path"]
        if path.is_symlink() or not path.is_file() or path.resolve().parent != model_root:
            raise ValueError(f"capture32 model file is missing/symlinked: {expected['path']}")
        actual = {
            "path": expected["path"],
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if actual != expected:
            raise ValueError(f"capture32 model file drifted: {expected['path']}")
        actual_model_files.append(actual)
    if _model_loading_paths(model_root) != sorted(
        item["path"] for item in capture_manifest["model"]["files"]
    ):
        raise ValueError("capture32 model loading-relevant inventory has extras/missing")
    model_manifest_sha = canonical_sha256(actual_model_files)
    tokenizer_manifest_sha = canonical_sha256([
        item for item in actual_model_files if item["path"] in {
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        }
    ])
    if model_manifest_sha != preregistration["source"]["model_file_manifest_sha256"] \
            or tokenizer_manifest_sha != preregistration["source"][
                "tokenizer_manifest_sha256"
            ]:
        raise ValueError("capture32 runtime model/tokenizer inventory drifted")

    import pyarrow.parquet as parquet

    parquet_rows = parquet.read_table(
        data_path, columns=["prompt", "context", "reward_model", "extra_info"]
    ).to_pylist()
    replay = recompute_capture32_source_evidence(
        parquet_rows=parquet_rows,
        authority=authority,
        tokenizer=tokenizer,
        writer_prompt_builder=writer_prompt_builder,
        no_memory_text=capture_manifest["recurrent"]["no_memory_text"],
        max_context_tokens=int(capture_manifest["recurrent"]["max_context_tokens"]),
        chunk_size=int(capture_manifest["recurrent"]["chunk_size"]),
        base_seed=int(capture_manifest["backend"]["engine_seed"]),
    )
    expected_replay = {
        "full_population_ranking": preregistration["selection"][
            "full_population_ranking"
        ],
        "full_population_ranking_sha256": preregistration["selection"][
            "full_population_ranking_sha256"
        ],
        "selected_inventory": preregistration["selected_inventory"],
        "selected_inventory_sha256": preregistration["inventory"][
            "selected_inventory_sha256"
        ],
    }
    if canonical_json(replay) != canonical_json(expected_replay):
        raise ValueError("capture32 raw S128/tokenizer replay differs from preregistration")
    return {
        "validation_sha256": sha256_file(data_path),
        "eval_manifest_hash": stable_resolved["eval_manifest_hash"],
        "model_file_manifest_sha256": model_manifest_sha,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha,
        "authority_file_sha256": EXPECTED_S128_AUTHORITY_FILE_SHA256,
        "full_population_ranking_sha256": replay[
            "full_population_ranking_sha256"
        ],
        "selected_inventory_sha256": replay["selected_inventory_sha256"],
    }


def _validate_candidate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("paired-effect manifest identity drifted")
    if manifest.get("experiment_name") != "qwen25_7b_paired_effect_candidate_seed2026":
        raise ValueError("paired-effect experiment name drifted")
    if manifest.get("report_schema") != "paired_effect_admissibility_report.schema.json":
        raise ValueError("paired-effect report schema drifted")
    if manifest.get("pipeline_code_objects") != list(PIPELINE_CODE_OBJECTS):
        raise ValueError("paired-effect pipeline code-object inventory drifted")
    firewall = manifest.get("claim_boundary", {})
    expected_firewall = {
        "capture_analysis_only": True,
        "trainer_attached": False,
        "gpu_execution_authorized": False,
        "method_selected": False,
        "training_authorized": False,
        "paper_performance_result": False,
        "causal_effect_claim": False,
    }
    if firewall != expected_firewall:
        raise ValueError("paired-effect claim firewall drifted")
    source = manifest.get("source_capture", {})
    for field, expected in {
        "role": "pipeline_pilot4_only",
        "execution_profile": "gpu45",
        "manifest": "manifests/h20/qwen25_7b_commit_retain_capture_gpu45_seed2026.json",
        "expected_branch": "h20/qwen25-7b-commit-retain-capture-gpu45-20260821",
        "expected_git_commit": "85ba6d3b03874978ed5d9713d8a628ce37f0c478",
        "expected_experiment_name": "qwen25_7b_commit_retain_capture_gpu45_seed2026",
        "expected_physical_gpus": [4, 5],
        "expected_visible_devices": "4,5",
        "required_pair_count": 4,
    }.items():
        if source.get(field) != expected or type(source.get(field)) is not type(expected):
            raise ValueError(f"source_capture.{field} drifted")
    scorer = manifest.get("scorer", {})
    expected_scorer = {
        "kind": "standardized_ridge",
        "ridge": 1.0,
        "fold_count": 4,
        "fold_rule": "sorted_stable_example_id_round_robin_v1",
        "group_key": "stable_example_id",
        "outcome_hidden_for_scored_row": True,
        "deployment_model_role": "diagnostic_full_capture_fit",
        "accepts_persisted_score": False,
        "accepts_runtime_uuid": False,
    }
    if scorer != expected_scorer:
        raise ValueError("paired-effect scorer contract drifted")
    expected_outcome = {
        "target": TARGET_NAME,
        "formula": "COMMIT.final_reader.token_f1 - RETAIN.final_reader.token_f1",
        "required_contracts": [
            "same_materialized_candidate",
            "exact_prebranch_old_state",
            "commit_loads_candidate_state",
            "retain_loads_old_state",
            "same_future_chunks_and_horizon",
            "same_reader_checkpoint_and_decode",
            "same_future_writer_rng_and_reader_rng",
            "same_cache_contract",
            "same_configured_cost_budget",
            "single_strict_vllm_engine",
        ],
    }
    if manifest.get("paired_outcome") != expected_outcome:
        raise ValueError("paired-outcome causal contract drifted")
    capture32_root = (
        Path(manifest["work_root"])
        / "logs"
        / "commit_retain_capture32_frozen_20260821"
        / manifest["capture_run_id"]
    )
    expected_capture32 = {
        "status": "PREREGISTERED_NOT_YET_CAPTURED",
        "role": "development_admissibility_evidence",
        "preregistration": CAPTURE32_PREREG_REL,
        "producer_manifest": "manifests/h20/qwen25_7b_commit_retain_capture32_seed2026.json",
        "expected_experiment_name": "qwen25_7b_commit_retain_capture32_seed2026",
        "required_pair_count": 32,
        "output_root": str(capture32_root),
        "commitment_marker": str(capture32_root / "certificates" / "p0_preflight.json"),
        "resolved_manifest": str(
            capture32_root / "certificates" / "p0_resolved_manifest.json"
        ),
        "supervisor_ledger": str(
            capture32_root / "commit_retain_capture32_execution_ledger.jsonl"
        ),
        "capture_credential": str(
            capture32_root / "credentials" / "capture_child.json"
        ),
        "credential_consumption": str(
            capture32_root / "credentials" / "capture_child_consumed.json"
        ),
        "capture_ledger": str(
            capture32_root / "captures" / "commit_retain_pairs.jsonl"
        ),
        "capture_run_receipt": str(
            capture32_root / "captures" / "run_receipt.json"
        ),
        "final_report": str(
            capture32_root / "certificates" / "commit_retain_capture32_final_report.json"
        ),
        "external_preregistration_anchor": str(
            Path(manifest["work_root"]) / "provenance" / "commit_retain_capture32"
            / f"{manifest['capture_run_id']}.preregistration.json"
        ),
        "external_terminal_anchor": str(
            Path(manifest["work_root"]) / "provenance" / "commit_retain_capture32"
            / f"{manifest['capture_run_id']}.terminal.json"
        ),
        "partial_after_commitment_is_failure": True,
        "capture4_may_fill_missing": False,
    }
    if manifest.get("capture32") != expected_capture32:
        raise ValueError("capture32 preregistered artifact contract drifted")
    admissibility = manifest.get("admissibility", {})
    expected_admissibility = {
        "current_capture_is_pipeline_canary_only": True,
        "minimum_training_stable_examples": 32,
        "nontrivial_effect_epsilon": 0.01,
        "minimum_nontrivial_effect_examples": 8,
        "effect_bin_precision": 0.000001,
        "minimum_distinct_effect_bins": 3,
        "minimum_mean_absolute_effect": 0.02,
        "minimum_target_variance": 0.0001,
        "minimum_crossfit_mse_improvement_fraction": 0.05,
        "minimum_crossfit_pearson_correlation": 0.2,
        "minimum_folds_with_positive_mse_improvement": 3,
        "minimum_heldout_examples_per_fold": 8,
        "minimum_fit_examples_per_fold": 24,
        "pending_decision_missing_capture": "PAIRED_EFFECT_CAPTURE_PENDING",
        "pending_decision_pilot4_only": "PAIRED_EFFECT_CAPTURE4_PILOT_ONLY",
        "pending_decision_provenance": "PAIRED_EFFECT_CAPTURE_PROVENANCE_PENDING",
        "pending_decision_more_capture": "PAIRED_EFFECT_MORE_CAPTURE_REQUIRED",
        "evidence_ready_decision": "PAIRED_EFFECT_EVIDENCE_READY_FOR_EXTERNAL_REVIEW",
        "invalid_decision": "PAIRED_EFFECT_NO_GO:INVALID_CAPTURE_OR_PIPELINE",
        "evidence_ready_still_selects_method": False,
        "evidence_ready_still_authorizes_training": False,
    }
    if admissibility != expected_admissibility:
        raise ValueError("paired-effect admissibility thresholds/decisions drifted")
    expected_anchor = {
        "status": "LOCAL_EXPORT_PENDING_EXTERNAL_SIGNATURE",
        "required": True,
        "path": str(
            Path(manifest["work_root"]) / "provenance" / "commit_retain_capture32"
            / f"{manifest['capture_run_id']}.terminal.json"
        ),
        "expected_anchor_git_commit": None,
        "expected_anchor_file_sha256": None,
        "required_artifact_hashes": [
            "preregistration", "p0_certificate", "resolved_manifest",
            "capture_ledger", "final_report",
        ],
        "required_provenance": "externally_reviewed_immutable_or_signed_anchor",
    }
    if manifest.get("external_capture_anchor") != expected_anchor:
        raise ValueError("external capture anchor contract drifted")
    safety = manifest.get("credit_safety", {})
    if safety != {
        "route_only_within_same_uid_exact_qa_ties": True,
        "all_rows_must_be_eligible": True,
        "all_exact_correct_groups_are_protected": True,
        "center_scores_within_uid": True,
        "bonus_semantics": "trajectory_total",
        "writer_token_distribution": "uniform_over_all_valid_nonfinal_writer_tokens",
        "final_rows_bitwise_unchanged": True,
        "non_target_rows_bitwise_unchanged": True,
        "lambda_requires_separate_training_preregistration": True,
    }:
        raise ValueError("writer-token normalization contract drifted")
    expected_capture_root = (
        Path(manifest["work_root"])
        / "logs"
        / "commit_retain_capture_gpu45_frozen_20260821"
        / manifest["capture_run_id"]
    )
    if Path(source.get("output_root", "")) != expected_capture_root:
        raise ValueError("capture root differs from the frozen GPU45 root")
    expected_capture_paths = {
        "p0_certificate": expected_capture_root / "certificates" / "p0_preflight.json",
        "resolved_manifest": expected_capture_root / "certificates" / "p0_resolved_manifest.json",
        "supervisor_ledger": expected_capture_root / "commit_retain_capture_execution_ledger.jsonl",
        "capture_credential": expected_capture_root / "credentials" / "capture_child.json",
        "capture_ledger": expected_capture_root / "captures" / "commit_retain_pairs.jsonl",
        "capture_run_receipt": expected_capture_root / "captures" / "run_receipt.json",
        "final_report": expected_capture_root / "certificates" / "commit_retain_capture_final_report.json",
    }
    for field, expected in expected_capture_paths.items():
        if Path(source.get(field, "")) != expected:
            raise ValueError(f"source_capture.{field} path drifted")
    expected_output_root = (
        Path(manifest["work_root"])
        / "logs"
        / "paired_effect_candidate_frozen_20260821"
        / manifest["run_id"]
    )
    if Path(manifest.get("paths", {}).get("output_root", "")) != expected_output_root:
        raise ValueError("paired-effect output root drifted")
    if manifest.get("paths") != {
        "output_root": str(expected_output_root),
        "bundle": str(expected_output_root / "artifacts" / "paired_effect_crossfit_bundle.json"),
        "report": str(expected_output_root / "certificates" / "paired_effect_admissibility_report.json"),
    }:
        raise ValueError("paired-effect output artifact paths drifted")
    prereg = load_capture32_preregistration(manifest)
    if any((
        prereg["scorer"]["kind"] != scorer["kind"],
        prereg["scorer"]["ridge"] != scorer["ridge"],
        prereg["scorer"]["fold_count"] != scorer["fold_count"],
        prereg["scorer"]["fold_rule"] != scorer["fold_rule"],
        prereg["scorer"]["outcome_hidden_for_scored_row"]
        is not scorer["outcome_hidden_for_scored_row"],
        prereg["scorer"]["feature_schema"] != list(FEATURE_SCHEMA),
        prereg["scorer"]["baseline"] != "fit_fold_target_mean",
        prereg["scorer"]["standardization"]
        != "fit_fold_population_mean_and_std_ddof0",
    )):
        raise ValueError("candidate manifest and capture32 scorer disagree")
    threshold_key_map = {
        "required_unique_stable_examples": "minimum_training_stable_examples",
    }
    if prereg["admissibility"] != {
        key: admissibility[threshold_key_map.get(key, key)]
        for key in prereg["admissibility"]
    }:
        raise ValueError("candidate manifest and capture32 thresholds disagree")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _assert_confined_regular_file(path: Path, root: Path) -> None:
    lexical = path.absolute()
    root_lexical = root.absolute()
    if not _path_is_within(lexical, root_lexical):
        raise ValueError(f"artifact path escapes capture root: {path}")
    if lexical.is_symlink() or root_lexical.is_symlink():
        raise ValueError(f"symlinked capture evidence is forbidden: {path}")
    relative = lexical.relative_to(root_lexical)
    cursor = root_lexical
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"symlinked capture evidence is forbidden: {path}")
    if lexical.resolve() != lexical or not lexical.is_file():
        raise ValueError(f"capture evidence is not a canonical regular file: {path}")


def _assert_confined_new_output(path: Path, root: Path) -> None:
    lexical = path.absolute()
    root_lexical = root.absolute()
    if not _path_is_within(lexical, root_lexical) or root_lexical.is_symlink():
        raise ValueError(f"output path escapes or symlinks work root: {path}")
    cursor = root_lexical
    for part in lexical.relative_to(root_lexical).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"output path traverses a symlink: {path}")
        if not cursor.exists():
            # Later children cannot exist below the first missing directory.
            break


def _authenticate_pipeline_git(manifest: Mapping[str, Any]) -> dict[str, Any]:
    repo = Path(manifest["repo_dir"])
    commit = manifest["expected_pipeline_commit"]
    try:
        _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise ValueError("pipeline expected commit is not a Git commit object") from error
    if _git(repo, "rev-parse", "HEAD") != commit:
        raise ValueError("pipeline checkout HEAD differs from expected commit")
    hashes: dict[str, str] = {}
    for relative in manifest["pipeline_code_objects"]:
        path = repo / relative
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise ValueError(f"pipeline code object is missing/symlinked: {relative}")
        committed = _git_blob(repo, commit, relative)
        actual = path.read_bytes()
        if actual != committed:
            raise ValueError(f"pipeline code object differs from Git: {relative}")
        hashes[relative] = _sha256_bytes(committed)
    return {
        "git_commit": commit,
        "code_sha256": hashes,
        "code_combined_sha256": canonical_sha256(hashes),
    }


def _authenticate_capture_git(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any], capture_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    repo = Path(manifest["repo_dir"])
    source = manifest["source_capture"]
    commit = source["expected_git_commit"]
    try:
        _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise ValueError("capture expected commit is not a Git commit object") from error
    base = str(capture_manifest["base_commit"])
    try:
        subprocess.check_call(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", base, commit],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("capture commit is not descended from its frozen base") from error
    changed_sources = str(_git(repo, "diff", "--name-only", base, commit, "--", "sources"))
    if changed_sources:
        raise ValueError("capture commit changed sources/")
    expected_keys = set(capture_code_objects(capture_manifest))
    persisted = resolved.get("execution_binding", {}).get("execution_code_sha256")
    if not isinstance(persisted, Mapping) or set(persisted) != expected_keys:
        raise ValueError("capture execution code-hash inventory is incomplete or has extras")
    recomputed: dict[str, str] = {}
    for relative in sorted(expected_keys):
        blob = _git_blob(repo, commit, relative)
        digest = _sha256_bytes(blob)
        if persisted.get(relative) != digest:
            raise ValueError(f"capture code hash is not its Git blob: {relative}")
        current = repo / relative
        if current.is_symlink() or not current.is_file() or current.read_bytes() != blob:
            raise ValueError(f"loaded capture validator differs from authenticated Git: {relative}")
        recomputed[relative] = digest
    combined = canonical_sha256(dict(persisted))
    if resolved["execution_binding"].get("execution_code_combined_sha256") != combined:
        raise ValueError("capture combined execution-code digest mismatch")
    return {
        "git_commit": commit,
        "git_base_commit": base,
        "execution_code_sha256": recomputed,
        "execution_code_combined_sha256": combined,
        "sources_unchanged": True,
    }


@contextmanager
def _capture_environment(manifest: Mapping[str, Any]) -> Iterator[None]:
    source = manifest["source_capture"]
    updates = {
        CAPTURE_ENV[0]: manifest["work_root"],
        CAPTURE_ENV[1]: manifest["repo_dir"],
        CAPTURE_ENV[2]: source["expected_git_commit"],
        CAPTURE_ENV[3]: manifest["capture_run_id"],
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _capture_paths(manifest: Mapping[str, Any]) -> dict[str, Path]:
    source = manifest["source_capture"]
    return {
        "p0_certificate": Path(source["p0_certificate"]),
        "resolved_manifest": Path(source["resolved_manifest"]),
        "supervisor_ledger": Path(source["supervisor_ledger"]),
        "capture_credential": Path(source["capture_credential"]),
        "capture_ledger": Path(source["capture_ledger"]),
        "capture_run_receipt": Path(source["capture_run_receipt"]),
        "final_report": Path(source["final_report"]),
    }


def _capture32_paths(manifest: Mapping[str, Any]) -> dict[str, Path]:
    source = manifest["capture32"]
    return {
        "p0_certificate": Path(source["commitment_marker"]),
        "resolved_manifest": Path(source["resolved_manifest"]),
        "supervisor_ledger": Path(source["supervisor_ledger"]),
        "capture_credential": Path(source["capture_credential"]),
        "credential_consumption": Path(source["credential_consumption"]),
        "capture_ledger": Path(source["capture_ledger"]),
        "capture_run_receipt": Path(source["capture_run_receipt"]),
        "final_report": Path(source["final_report"]),
        "external_preregistration_anchor": Path(
            source["external_preregistration_anchor"]
        ),
        "external_terminal_anchor": Path(source["external_terminal_anchor"]),
    }


def _capture32_artifact_state(manifest: Mapping[str, Any]) -> tuple[str, dict[str, Path]]:
    paths = _capture32_paths(manifest)
    # A dangling symlink is still an attempted artifact and must not be
    # mistaken for a clean NOT_STARTED state.
    present = {
        name for name, path in paths.items() if path.exists() or path.is_symlink()
    }
    if not present:
        return "NOT_STARTED", paths
    missing = sorted(set(paths) - present)
    if "p0_certificate" not in present:
        raise Capture32AttritionError(
            "capture32 artifacts appeared without the preregistered P0 commitment"
        )
    if missing:
        raise Capture32AttritionError(
            f"capture32 attrition after P0 commitment; missing={missing}"
        )
    root = Path(manifest["capture32"]["output_root"])
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("capture32 run root is symlinked or non-canonical")
    external_names = {
        "external_preregistration_anchor", "external_terminal_anchor"
    }
    for name, path in paths.items():
        authority_root = (
            Path(manifest["work_root"]) / "provenance" / "commit_retain_capture32"
            if name in external_names else root
        )
        _assert_confined_regular_file(path, authority_root)
    return "COMPLETE", paths


def _validate_capture32_runtime_bindings(
    resolved: Mapping[str, Any], *, preregistration: Mapping[str, Any]
) -> dict[str, str]:
    """Close P0, execution, runtime, current, and per-pair GPU bindings."""
    expected_pair = resolved.get("expected_pair_binding")
    execution = resolved.get("execution_binding")
    runtime = resolved.get("runtime_binding")
    current = resolved.get("current_binding")
    if not all(isinstance(item, Mapping) for item in (
        expected_pair, execution, runtime, current
    )):
        raise ValueError("capture32 structured runtime bindings are missing")
    physical = resolved.get("physical_gpu_whitelist")
    visible = resolved.get("visible_devices")
    if not isinstance(physical, list) or len(physical) != 2 \
            or any(type(item) is not int or item < 0 for item in physical) \
            or physical != sorted(set(physical)) \
            or visible != ",".join(str(item) for item in physical):
        raise ValueError("capture32 P0 GPU pair is not explicit/distinct/ascending")
    gpu_lock = resolved.get("gpu_lock_binding")
    lock_sha = resolved.get("gpu_lock_binding_sha256")
    if not isinstance(gpu_lock, Mapping) or set(gpu_lock) != {
        "schema", "physical_gpu_indices", "locks", "gpu_lock_binding_sha256",
    } or gpu_lock.get("schema") != "memagent.capture32.lock-holder-receipt.v1" \
            or gpu_lock.get("physical_gpu_indices") != physical \
            or gpu_lock.get("gpu_lock_binding_sha256") != lock_sha:
        raise ValueError("capture32 GPU lock binding structure drifted")
    locks = gpu_lock.get("locks")
    if not isinstance(locks, list) or len(locks) < 2 \
            or any(not isinstance(item, Mapping) or set(item) != {
                "path", "fd", "device", "inode", "owner_uid",
            } for item in locks):
        raise ValueError("capture32 GPU lock inventory is incomplete")
    recomputed_lock_sha = canonical_sha256({
        "physical_gpu_indices": physical,
        "locks": [
            {key: item[key] for key in ("path", "device", "inode", "owner_uid")}
            for item in locks
        ],
    })
    if lock_sha != recomputed_lock_sha:
        raise ValueError("capture32 GPU lock binding digest mismatch")
    pair_sha = canonical_sha256(expected_pair)
    execution_sha = canonical_sha256(execution)
    runtime_sha = canonical_sha256(runtime)
    if resolved.get("execution_binding_sha256") != execution_sha \
            or resolved.get("runtime_binding_sha256") != runtime_sha:
        raise ValueError("capture32 execution/runtime binding digest mismatch")
    expected_current = {
        "schema": "memagent.commit-retain.capture32-current-binding.v1",
        "run_id": resolved.get("run_id"),
        "git_commit": resolved.get("git_commit"),
        "execution_binding_sha256": execution_sha,
        "runtime_binding_sha256": runtime_sha,
        "expected_pair_binding_sha256": pair_sha,
        "physical_gpu_whitelist": physical,
        "visible_devices": visible,
    }
    if dict(current) != expected_current \
            or resolved.get("current_binding_sha256") != canonical_sha256(current):
        raise ValueError("capture32 current binding is not canonical")
    expected_execution = {
        "schema": "memagent.commit-retain.capture32-execution-binding.v1",
        "run_id": resolved.get("run_id"),
        "git_commit": resolved.get("git_commit"),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "eval_manifest_hash": preregistration["source"]["eval_manifest_hash"],
        "selected_inventory_sha256": preregistration["inventory"][
            "selected_inventory_sha256"
        ],
        "fold_membership_sha256": preregistration["folds"]["membership_sha256"],
        "execution_code_combined_sha256": resolved.get(
            "execution_code_combined_sha256"
        ),
        "expected_pair_binding_sha256": pair_sha,
        "physical_gpu_whitelist": physical,
        "visible_devices": visible,
        "gpu_lock_binding_sha256": resolved.get("gpu_lock_binding_sha256"),
        "rollout_backend": "strict_vllm_0.8.2",
    }
    if dict(execution) != expected_execution:
        raise ValueError("capture32 execution binding structure drifted")
    runtime_fields = {
        "schema", "run_id", "git_commit", "expected_pair_binding_sha256",
        "physical_gpu_whitelist", "visible_devices", "physical_gpu_identity",
        "rollout_backend", "engine_config_sha256", "worker_multiproc_method",
        "vllm_observed_worker_multiproc_method", "multiprocessing_context_method",
        "parent_cuda_initialization_policy", "writer_checkpoint_sha256",
        "reader_checkpoint_sha256", "model_file_manifest_sha256",
        "tokenizer_manifest_sha256", "gpu_lock_binding_sha256",
    }
    if set(runtime) != runtime_fields:
        raise ValueError("capture32 runtime binding fields drifted")
    shared_runtime = {
        "run_id": resolved.get("run_id"),
        "git_commit": resolved.get("git_commit"),
        "expected_pair_binding_sha256": pair_sha,
        "physical_gpu_whitelist": physical,
        "visible_devices": visible,
        "rollout_backend": "strict_vllm_0.8.2",
        "worker_multiproc_method": "spawn",
        "vllm_observed_worker_multiproc_method": "spawn",
        "multiprocessing_context_method": "spawn",
        "parent_cuda_initialization_policy": "record_observed_spawn_required",
        "model_file_manifest_sha256": preregistration["source"][
            "model_file_manifest_sha256"
        ],
        "tokenizer_manifest_sha256": preregistration["source"][
            "tokenizer_manifest_sha256"
        ],
        "gpu_lock_binding_sha256": resolved.get("gpu_lock_binding_sha256"),
    }
    if runtime.get("schema") != "memagent.commit-retain.capture32-runtime-binding.v1" \
            or any(canonical_json(runtime.get(field)) != canonical_json(expected)
                   for field, expected in shared_runtime.items()) \
            or any(not _is_sha(runtime.get(field)) for field in (
                "engine_config_sha256", "writer_checkpoint_sha256",
                "reader_checkpoint_sha256",
            )) \
            or runtime.get("writer_checkpoint_sha256") != runtime.get(
                "reader_checkpoint_sha256"
            ):
        raise ValueError("capture32 runtime binding values drifted")
    for field in (
        "physical_gpu_whitelist", "visible_devices", "physical_gpu_identity",
        "engine_config_sha256", "worker_multiproc_method",
        "vllm_observed_worker_multiproc_method", "multiprocessing_context_method",
        "parent_cuda_initialization_policy", "writer_checkpoint_sha256",
        "reader_checkpoint_sha256",
    ):
        pair_field = (
            expected_pair.get(field)
            if field not in {"writer_checkpoint_sha256", "reader_checkpoint_sha256"}
            else expected_pair.get(field)
        )
        if canonical_json(pair_field) != canonical_json(runtime.get(field)):
            raise ValueError(f"capture32 pair/runtime split binding: {field}")
    identities = runtime.get("physical_gpu_identity")
    if not isinstance(identities, list) or len(identities) != 2 \
            or any(not isinstance(item, Mapping) or set(item) != {
                "physical_index", "uuid", "pci_bus_id", "name",
                "compute_mode", "mig_mode",
            } for item in identities) \
            or [item["physical_index"] for item in identities] != physical \
            or len({item["uuid"] for item in identities}) != 2 \
            or any("NVIDIA H20" not in item["name"] for item in identities) \
            or any(item["compute_mode"] != "Default" for item in identities) \
            or any(item["mig_mode"] not in {"Disabled", "N/A", "[N/A]"}
                   for item in identities):
        raise ValueError("capture32 physical GPU identity must contain two devices")
    expected_pair_fields = {
        "writer_checkpoint_sha256", "reader_checkpoint_sha256",
        "writer_prompt_template_sha256", "reader_prompt_template_sha256",
        "writer_decode", "reader_decode", "physical_gpu_whitelist",
        "visible_devices", "physical_gpu_identity", "engine_config_sha256",
        "worker_multiproc_method", "vllm_observed_worker_multiproc_method",
        "multiprocessing_context_method", "parent_cuda_initialization_policy",
        "global_generate_call_count", "eos_token_id",
        "gpu_lock_binding_sha256",
    }
    expected_calls = sum(
        int(row["intervention_writer_turn"])
        + 1
        + 2 * (
            int(row["total_writer_turns"])
            - int(row["intervention_writer_turn"])
        )
        for row in preregistration["selected_inventory"]
    )
    if set(expected_pair) != expected_pair_fields \
            or expected_pair.get("global_generate_call_count") != expected_calls \
            or expected_pair.get("gpu_lock_binding_sha256") != lock_sha:
        raise ValueError("capture32 expected pair binding is incomplete or drifted")
    if expected_pair.get("physical_gpu_whitelist") != physical \
            or expected_pair.get("visible_devices") != visible \
            or expected_pair.get("gpu_lock_binding_sha256") != resolved.get(
                "gpu_lock_binding_sha256"
            ):
        raise ValueError("capture32 P0/runtime/pair GPU/lock binding is split")
    return {
        "expected_pair_binding_sha256": pair_sha,
        "execution_binding_sha256": execution_sha,
        "runtime_binding_sha256": runtime_sha,
        "current_binding_sha256": canonical_sha256(current),
    }


def _require_capture_files(manifest: Mapping[str, Any]) -> dict[str, Path]:
    paths = _capture_paths(manifest)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise CapturePendingError(missing)
    root = Path(manifest["source_capture"]["output_root"])
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("capture run root is symlinked or non-canonical")
    for path in paths.values():
        _assert_confined_regular_file(path, root)
    return paths


def _capture_manifest_environment(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        CAPTURE_ENV[0]: manifest["work_root"],
        CAPTURE_ENV[1]: manifest["repo_dir"],
        CAPTURE_ENV[2]: manifest["source_capture"]["expected_git_commit"],
        CAPTURE_ENV[3]: manifest["capture_run_id"],
    }


def _validate_supervisor_terminal(
    *,
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    capture_manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    paths: Mapping[str, Path],
    capture_report: Mapping[str, Any],
    final_report: Mapping[str, Any],
) -> None:
    failures = validate_jsonl_chain(list(records))
    if failures or len(records) != 4:
        raise ValueError(f"capture supervisor is not one finalized four-record chain: {failures}")
    if [record.get("record_type") for record in records] != [
        "s0_preflight", "capture_authorization", "capture_complete", "audit_result"
    ]:
        raise ValueError("capture supervisor terminal order drifted")
    common = {
        "experiment_name": capture_experiment_name(capture_manifest),
        "git_commit": manifest["source_capture"]["expected_git_commit"],
        "run_id": manifest["capture_run_id"],
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": resolved["lightweight_current_binding_sha256"],
        "training_authorized": False,
        "method_selected": False,
    }
    for index, record in enumerate(records):
        for field, expected in common.items():
            if record.get(field) != expected:
                raise ValueError(f"supervisor[{index}].{field} binding mismatch")
    capture_allowed = {
        "record_type", "experiment_name", "git_commit", "run_id", "recorded_at",
        "eval_manifest_hash", "execution_binding_sha256", "runtime_binding_sha256",
        "current_binding_sha256", "artifact", "artifact_sha256", "training_authorized",
        "method_selected", "status", "decision", "pair_count", "pair_ids",
        "stable_write_ids", "generate_call_count", "run_receipt",
        "run_receipt_sha256", "record_index", "previous_record_sha256", "record_sha256",
    }
    capture_record = records[2]
    if set(capture_record) != capture_allowed:
        raise ValueError("capture_complete supervisor record has handcrafted fields")
    if not all((
        capture_record["artifact"] == str(paths["capture_ledger"].resolve()),
        capture_record["artifact_sha256"] == sha256_file(paths["capture_ledger"]),
        capture_record["run_receipt"] == str(paths["capture_run_receipt"].resolve()),
        capture_record["run_receipt_sha256"] == sha256_file(paths["capture_run_receipt"]),
        capture_record["pair_count"] == capture_report["pair_count"],
        capture_record["pair_ids"] == capture_report["pair_ids"],
        capture_record["stable_write_ids"] == capture_report["stable_write_ids"],
        capture_record["generate_call_count"] == capture_report["generate_call_count"],
        capture_record["status"] == "PASS",
        capture_record["decision"] == "COMMIT_RETAIN_CAPTURE_COMPLETE",
    )):
        raise ValueError("capture_complete supervisor record differs from raw evidence")
    audit_allowed = {
        "record_type", "experiment_name", "git_commit", "run_id", "recorded_at",
        "eval_manifest_hash", "execution_binding_sha256", "runtime_binding_sha256",
        "current_binding_sha256", "artifact", "artifact_sha256",
        "training_authorized", "method_selected", "status", "decision", "pair_count",
        "pair_ids", "stable_write_ids", "generate_call_count", "record_index",
        "previous_record_sha256", "record_sha256",
    }
    audit = records[3]
    if set(audit) != audit_allowed:
        raise ValueError("audit_result supervisor record has handcrafted fields")
    if not all((
        audit["artifact"] == str(paths["final_report"].resolve()),
        audit["artifact_sha256"] == sha256_file(paths["final_report"]),
        audit["status"] == final_report["status"],
        audit["decision"] == final_report["decision"],
        audit["pair_count"] == capture_report["pair_count"],
        audit["pair_ids"] == capture_report["pair_ids"],
        audit["stable_write_ids"] == capture_report["stable_write_ids"],
        audit["generate_call_count"] == capture_report["generate_call_count"],
    )):
        raise ValueError("audit_result supervisor record differs from final/raw evidence")


def _contract_evidence(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for pair in pairs:
        shared = pair["shared_contract"]
        commit = pair["arms"]["COMMIT"]
        retain = pair["arms"]["RETAIN"]
        evidence.append({
            "pair_id": pair["pair_id"],
            "stable_example_id": pair["stable_example_id"],
            "stable_write_id": pair["stable_write_id"],
            "candidate_generation_count": pair["candidate_generation_count"],
            "candidate_materialized_before_arm_start": pair[
                "candidate_materialized_before_arm_start"
            ],
            "old_state_bytes_sha256": pair["old_state"]["bytes_sha256"],
            "candidate_state_bytes_sha256": pair["candidate"]["state_after"][
                "bytes_sha256"
            ],
            "commit_initial_state_bytes_sha256": commit[
                "initial_loaded_state_receipt"
            ]["state"]["bytes_sha256"],
            "retain_initial_state_bytes_sha256": retain[
                "initial_loaded_state_receipt"
            ]["state"]["bytes_sha256"],
            "shared_contract_sha256": pair["shared_contract_sha256"],
            "future_chunks_sha256": shared["future_chunks_sha256"],
            "horizon_sha256": canonical_sha256(shared["horizon"]),
            "reader_checkpoint_sha256": shared["reader_checkpoint_sha256"],
            "reader_decode_sha256": canonical_sha256(shared["reader_decode"]),
            "future_writer_rng_sha256": canonical_sha256(
                shared["future_writer_request_seeds"]
            ),
            "reader_request_seed": shared["reader_request_seed"],
            "cache_contract_sha256": canonical_sha256(shared["cache_contract"]),
            "cost_contract_sha256": canonical_sha256(shared["cost_contract"]),
            "commit_actual_cost_sha256": canonical_sha256(
                commit["actual_cost_receipt"]
            ),
            "retain_actual_cost_sha256": canonical_sha256(
                retain["actual_cost_receipt"]
            ),
            "contract_validation": "RECOMPUTED_PASS",
        })
    return evidence


def external_capture_anchor_state(
    manifest: Mapping[str, Any], capture: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the directory-external provenance gate.

    The initial closure intentionally has no trusted anchor value.  Enabling
    this gate requires a later reviewed code/manifest change that freezes the
    real run ID and terminal artifact digests (or a verifiable signature).
    Merely placing another self-consistent JSON file beside the run can never
    change this result.
    """
    anchor = manifest["external_capture_anchor"]
    if anchor.get("status") != "FROZEN_VERIFIED" \
            or anchor.get("expected_anchor_git_commit") is None \
            or anchor.get("expected_anchor_file_sha256") is None:
        return {
            "status": "PENDING",
            "decision": "EXTERNAL_CAPTURE_ANCHOR_NOT_FROZEN",
            "anchor_path": anchor["path"],
            "capture_run_id": manifest["capture_run_id"],
            "internally_recomputed_final_report_sha256": capture["artifacts"][
                "final_report"
            ]["sha256"],
            "reason": (
                "Internal SHA chains prove consistency, not H20 provenance. A later "
                "reviewed commit must freeze this run and its terminal digests."
            ),
        }
    # This branch is unreachable while the local export lacks an externally
    # reviewed immutable digest/signature.
    # It deliberately fails until the later anchor-format/signature validator
    # is reviewed together with the literal anchor values.
    raise ValueError("external capture anchor claims frozen without a reviewed validator")


def authenticate_capture(
    manifest: Mapping[str, Any], *, tokenizer_factory: Any = _tokenizer
) -> dict[str, Any]:
    paths = _require_capture_files(manifest)
    capture_manifest_path = Path(manifest["repo_dir"]) / manifest["source_capture"]["manifest"]
    capture_manifest = load_capture_manifest(
        capture_manifest_path, _capture_manifest_environment(manifest)
    )
    validate_capture_manifest(capture_manifest)
    profile = _gpu_profile(capture_manifest)
    if profile["name"] != "gpu45" or capture_experiment_name(capture_manifest) != manifest[
        "source_capture"
    ]["expected_experiment_name"]:
        raise ValueError("capture profile or experiment identity mismatch")
    if {key: str(value) for key, value in paths.items()} != {
        "p0_certificate": capture_manifest["paths"]["p0_certificate"],
        "resolved_manifest": capture_manifest["paths"]["resolved_manifest"],
        "supervisor_ledger": capture_manifest["paths"]["execution_ledger"],
        "capture_credential": capture_manifest["paths"]["capture_credential"],
        "capture_ledger": capture_manifest["paths"]["capture_ledger"],
        "capture_run_receipt": capture_manifest["paths"]["capture_run_receipt"],
        "final_report": capture_manifest["paths"]["final_report"],
    }:
        raise ValueError("candidate manifest and source capture paths disagree")
    with _capture_environment(manifest):
        p0, resolved = validate_p0(capture_manifest_path)
        git_evidence = _authenticate_capture_git(manifest, resolved, capture_manifest)
        if p0.get("failures") != []:
            raise ValueError("capture P0 contains failures")
        tokenizer = tokenizer_factory(capture_manifest)
        from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
        from recurrent.utils import TokenTemplate, chat_template

        _assert_loaded_repo_module_origins()

        chat = chat_template(tokenizer)
        writer_template = TokenTemplate(chat.format(message=TEMPLATE), tokenizer)
        reader_template = TokenTemplate(chat.format(message=TEMPLATE_FINAL_BOXED), tokenizer)
        records = read_jsonl(paths["capture_ledger"])
        capture_report = validate_capture_ledger(
            records,
            frozen_pairs=resolved["frozen_pairs"],
            experiment_name=capture_experiment_name(capture_manifest),
            git_commit=manifest["source_capture"]["expected_git_commit"],
            run_id=manifest["capture_run_id"],
            execution_binding_sha256=resolved["execution_binding_sha256"],
            runtime_binding_sha256=resolved["runtime_binding_sha256"],
            current_binding_sha256=resolved["lightweight_current_binding_sha256"],
            decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
            writer_prompt_builder=lambda question, memory, chunk: writer_template.format(
                prompt=question, memory=memory, chunk=chunk
            ).tolist(),
            reader_prompt_builder=lambda question, memory: reader_template.format(
                prompt=question, memory=memory
            ).tolist(),
            expected_pair_binding=expected_pair_binding(
                capture_manifest, resolved, tokenizer
            ),
        )
        credential = validate_capture_credential(
            paths["capture_credential"],
            manifest=capture_manifest,
            resolved=resolved,
            current_binding_sha256=resolved["lightweight_current_binding_sha256"],
            require_live_parent=False,
        )
        pairs = [record["pair"] for record in records]
        if any(
            execution.get(field) != expected
            for pair in pairs
            for execution in [pair["execution"]]
            for field, expected in credential.items()
        ):
            raise ValueError("pair execution credential evidence mismatch")
        if any(pair["execution"]["physical_gpu_whitelist"] != [4, 5] for pair in pairs):
            raise ValueError("capture pair is not bound to physical GPU4-5")
        expected_receipt = _expected_run_receipt(
            manifest=capture_manifest,
            resolved=resolved,
            current_binding_sha256=resolved["lightweight_current_binding_sha256"],
            capture_report=capture_report,
            capture_path=paths["capture_ledger"],
        )
        actual_receipt = json.loads(paths["capture_run_receipt"].read_text(encoding="utf-8"))
        if canonical_json(actual_receipt) != canonical_json(expected_receipt):
            raise ValueError("capture run receipt does not independently reproduce")
        supervisor = read_jsonl(paths["supervisor_ledger"])
        prefix = supervisor[:3]
        expected_final = {
            "schema": "memagent.commit-retain.capture-final-audit.v1",
            "status": "PASS",
            "decision": "COMMIT_RETAIN_CAPTURE_AUDIT_COMPLETE",
            "git_commit": manifest["source_capture"]["expected_git_commit"],
            "run_id": manifest["capture_run_id"],
            "eval_manifest_hash": resolved["eval_manifest_hash"],
            "execution_binding_sha256": resolved["execution_binding_sha256"],
            "runtime_binding_sha256": resolved["runtime_binding_sha256"],
            "current_binding_sha256": resolved["lightweight_current_binding_sha256"],
            "capture_ledger": str(paths["capture_ledger"].resolve()),
            "capture_ledger_sha256": sha256_file(paths["capture_ledger"]),
            "capture_run_receipt": str(paths["capture_run_receipt"].resolve()),
            "capture_run_receipt_sha256": sha256_file(paths["capture_run_receipt"]),
            **{key: capture_report[key] for key in (
                "pair_count", "stable_write_ids", "pair_ids", "generate_call_count",
                "outcomes", "training", "claim_boundary",
            )},
            "supervisor_prefix_record_count": 3,
            "supervisor_prefix_sha256": canonical_sha256(prefix),
            "native_memory_interface_evidence": resolved["execution_binding"][
                "native_memory_interface_evidence"
            ],
        }
        actual_final = json.loads(paths["final_report"].read_text(encoding="utf-8"))
        if set(actual_final) != set(expected_final) or canonical_json(actual_final) != canonical_json(
            expected_final
        ):
            raise ValueError("capture final report does not independently reproduce")
        _validate_supervisor_terminal(
            records=supervisor,
            manifest=manifest,
            capture_manifest=capture_manifest,
            resolved=resolved,
            paths=paths,
            capture_report=capture_report,
            final_report=actual_final,
        )
    return {
        "capture_role": "pipeline_pilot4_only",
        "pairs": pairs,
        "contract_evidence": _contract_evidence(pairs),
        "capture_report": capture_report,
        "git_evidence": git_evidence,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "resolved_binding": {
            "eval_manifest_hash": resolved["eval_manifest_hash"],
            "execution_binding_sha256": resolved["execution_binding_sha256"],
            "runtime_binding_sha256": resolved["runtime_binding_sha256"],
            "current_binding_sha256": resolved["lightweight_current_binding_sha256"],
            "physical_gpu_identity": resolved["runtime_binding"]["physical_gpu_identity"],
        },
        "credential": credential,
    }


def _authenticate_capture32_git(
    *, repo: Path, git_commit: str, execution_code_sha256: Mapping[str, Any]
) -> dict[str, Any]:
    if FULL_SHA.fullmatch(git_commit) is None:
        raise ValueError("capture32 Git commit is not a full SHA")
    try:
        _git(repo, "cat-file", "-e", f"{git_commit}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise ValueError("capture32 Git commit is not a real commit object") from error
    try:
        subprocess.check_call(
            [
                "git", "-C", str(repo), "merge-base", "--is-ancestor",
                CAPTURE_SOURCE_BASE_COMMIT, git_commit,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("capture32 Git commit is not descended from the frozen base") from error
    if str(_git(repo, "diff", "--name-only", CAPTURE_SOURCE_BASE_COMMIT,
                git_commit, "--", "sources")):
        raise ValueError("capture32 Git commit changed sources/")
    required = set(CAPTURE32_EXECUTION_CODE_OBJECTS)
    if not isinstance(execution_code_sha256, Mapping) \
            or set(execution_code_sha256) != required:
        raise ValueError("capture32 execution code inventory is incomplete or has extras")
    recomputed: dict[str, str] = {}
    for relative, persisted in sorted(execution_code_sha256.items()):
        if not isinstance(relative, str) or relative.startswith("/") \
                or ".." in Path(relative).parts:
            raise ValueError("capture32 execution code path escapes repository")
        blob = _git_blob(repo, git_commit, relative)
        digest = _sha256_bytes(blob)
        if persisted != digest:
            raise ValueError(f"capture32 code hash is not its Git blob: {relative}")
        current = repo / relative
        if current.is_symlink() or not current.is_file() or current.read_bytes() != blob:
            raise ValueError(
                f"loaded capture32 execution object differs from authenticated Git: {relative}"
            )
        recomputed[relative] = digest
    return {
        "git_commit": git_commit,
        "git_base_commit": CAPTURE_SOURCE_BASE_COMMIT,
        "execution_code_sha256": recomputed,
        "execution_code_combined_sha256": canonical_sha256(recomputed),
        "sources_unchanged": True,
    }


def _validate_capture32_structural_completeness(
    records: Sequence[Mapping[str, Any]], preregistration: Mapping[str, Any]
) -> None:
    normalized = [dict(record) for record in records]
    if len(normalized) != CAPTURE32_COUNT:
        raise Capture32AttritionError(
            f"capture32 committed ledger has {len(normalized)} pairs, expected 32"
        )
    expected_rows = {
        row["stable_write_id"]: row for row in preregistration["selected_inventory"]
    }
    for index, envelope in enumerate(normalized):
        if not isinstance(envelope, Mapping) or "pair" not in envelope \
                or not isinstance(envelope.get("pair"), Mapping):
            raise Capture32AttritionError(f"capture32 envelope {index} is missing pair")
        pair = envelope["pair"]
        stable_write_id = envelope.get("stable_write_id", pair.get("stable_write_id"))
        frozen = expected_rows.get(stable_write_id)
        if frozen is None:
            # A present but substituted identity is invalid, not attrition.
            continue
        prefix = pair.get("prefix_turns")
        if not isinstance(prefix, list) or len(prefix) < int(
            frozen["intervention_writer_turn"]
        ):
            raise Capture32AttritionError(
                f"capture32 pair {stable_write_id} is missing prefix turns"
            )
        if not isinstance(pair.get("candidate"), Mapping):
            raise Capture32AttritionError(
                f"capture32 pair {stable_write_id} is missing candidate turn"
            )
        arms = pair.get("arms")
        if not isinstance(arms, Mapping):
            raise Capture32AttritionError(
                f"capture32 pair {stable_write_id} is missing arms"
            )
        expected_future = int(frozen["total_writer_turns"]) - int(
            frozen["intervention_writer_turn"]
        ) - 1
        for arm in ARMS:
            arm_record = arms.get(arm)
            if not isinstance(arm_record, Mapping):
                raise Capture32AttritionError(
                    f"capture32 pair {stable_write_id} is missing {arm} arm"
                )
            future = arm_record.get("future_turns")
            if not isinstance(future, list) or len(future) < expected_future:
                raise Capture32AttritionError(
                    f"capture32 pair {stable_write_id} is missing {arm} future turns"
                )
            reader = arm_record.get("final_reader")
            outcome = reader.get("outcome") if isinstance(reader, Mapping) else None
            if not isinstance(outcome, Mapping) or not {
                "prediction", "extraction_route", "format_success", "exact_match",
                "token_f1", "sub_exact_match",
            }.issubset(outcome):
                raise Capture32AttritionError(
                    f"capture32 pair {stable_write_id} is missing {arm} reader outcome"
                )


def _validate_capture32_envelopes(
    records: Sequence[Mapping[str, Any]], *, preregistration: Mapping[str, Any],
    run_id: str, git_commit: str, resolved: Mapping[str, Any], decoder: Any,
    writer_prompt_builder: Any, reader_prompt_builder: Any,
    execution_evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_capture32_runtime_bindings(
        resolved, preregistration=preregistration
    )
    normalized = [dict(record) for record in records]
    _validate_capture32_structural_completeness(normalized, preregistration)
    chain_failures = validate_jsonl_chain(normalized)
    if chain_failures:
        raise ValueError(f"capture32 ledger hash chain failed: {chain_failures}")
    allowed = {
        "record_type", "experiment_name", "git_commit", "run_id",
        "eval_manifest_hash", "execution_binding_sha256", "runtime_binding_sha256",
        "current_binding_sha256", "stable_example_id", "stable_root_id",
        "stable_write_id", "pair_id", "pair", "training_authorized",
        "method_selected", "record_index", "previous_record_sha256", "record_sha256",
    }
    raw_pairs: list[Mapping[str, Any]] = []
    for index, envelope in enumerate(normalized):
        if set(envelope) != allowed:
            raise ValueError(f"capture32 envelope {index} fields drifted")
        for field, expected in {
            "record_type": "commit_retain_pair_capture",
            "experiment_name": "qwen25_7b_commit_retain_capture32_seed2026",
            "git_commit": git_commit,
            "run_id": run_id,
            "eval_manifest_hash": preregistration["source"]["eval_manifest_hash"],
            "execution_binding_sha256": resolved["execution_binding_sha256"],
            "runtime_binding_sha256": resolved["runtime_binding_sha256"],
            "current_binding_sha256": resolved["current_binding_sha256"],
            "training_authorized": False,
            "method_selected": False,
        }.items():
            if envelope.get(field) != expected:
                raise ValueError(f"capture32 envelope {index}.{field} mismatch")
        raw_pairs.append(envelope["pair"])
    pairs = order_and_validate_capture32_pairs(
        raw_pairs, preregistration, decoder=decoder
    )
    observed_by_write = {pair["stable_write_id"]: pair for pair in pairs}
    for index, envelope in enumerate(normalized):
        pair = observed_by_write.get(envelope["stable_write_id"])
        if pair is None or envelope["stable_example_id"] != pair["stable_example_id"] \
                or envelope["stable_root_id"] != pair["stable_root_id"] \
                or envelope["pair_id"] != pair["pair_id"]:
            raise ValueError(f"capture32 envelope {index} identity differs from pair")
        canonical = build_capture_envelope(
            pair,
            experiment_name="qwen25_7b_commit_retain_capture32_seed2026",
            git_commit=git_commit,
            run_id=run_id,
            execution_binding_sha256=resolved["execution_binding_sha256"],
            runtime_binding_sha256=resolved["runtime_binding_sha256"],
            current_binding_sha256=resolved["current_binding_sha256"],
        )
        if any(canonical_json(envelope.get(field)) != canonical_json(expected)
               for field, expected in canonical.items()):
            raise ValueError(f"capture32 envelope {index} is non-canonical")

    expected_binding = resolved["expected_pair_binding"]
    shared_fields = (
        "writer_checkpoint_sha256", "reader_checkpoint_sha256",
        "writer_prompt_template_sha256", "reader_prompt_template_sha256",
        "writer_decode", "reader_decode",
    )
    execution_fields = (
        "physical_gpu_whitelist", "visible_devices", "physical_gpu_identity",
        "engine_config_sha256", "worker_multiproc_method",
        "vllm_observed_worker_multiproc_method", "multiprocessing_context_method",
        "parent_cuda_initialization_policy", "global_generate_call_count",
        "gpu_lock_binding_sha256",
    )
    for pair in pairs:
        for field in shared_fields:
            if canonical_json(pair["shared_contract"].get(field)) != canonical_json(
                expected_binding.get(field)
            ):
                raise ValueError(f"capture32 pair shared binding differs: {field}")
        for field in execution_fields:
            if canonical_json(pair["execution"].get(field)) != canonical_json(
                expected_binding.get(field)
            ):
                raise ValueError(f"capture32 pair execution binding differs: {field}")
        for field, expected in execution_evidence.items():
            if canonical_json(pair["execution"].get(field)) != canonical_json(expected):
                raise ValueError(
                    f"capture32 pair execution credential binding differs: {field}"
                )
        question = list(pair["question_token_ids"])
        writers = [*pair["prefix_turns"], pair["candidate"]]
        for arm in ARMS:
            writers.extend(pair["arms"][arm]["future_turns"])
        for generation in writers:
            prompt = generation["prompt"]
            expected_prompt = list(writer_prompt_builder(
                question,
                list(prompt["loaded_state_receipt"]["state"]["token_ids"]),
                list(prompt["chunk_token_ids"]),
            ))
            if prompt["token_ids"] != expected_prompt:
                raise ValueError("capture32 writer prompt does not reproduce")
        for arm in ARMS:
            final = pair["arms"][arm]["final_reader"]
            prompt = final["prompt"]
            expected_prompt = list(reader_prompt_builder(
                question, list(prompt["loaded_state_receipt"]["state"]["token_ids"])
            ))
            if prompt["token_ids"] != expected_prompt:
                raise ValueError("capture32 reader prompt does not reproduce")

    process_fields = (
        "engine_id", "cache_namespace", "process_instance_uuid", "process_pid",
        "physical_gpu_whitelist", "visible_devices", "physical_gpu_identity",
        "global_generate_call_count", "parent_credential_id", "engine_config_sha256",
        "worker_multiproc_method", "vllm_observed_worker_multiproc_method",
        "multiprocessing_context_method", "parent_cuda_initialized_before_engine",
        "parent_cuda_initialization_policy", "parent_credential_sha256",
        "parent_credential_path", "parent_issuer_pid", "observed_parent_pid",
        "parent_authorization_record_sha256", "gpu_lock_binding_sha256",
        "lock_holder_receipt_sha256", "credential_consumption_sha256",
        "credential_consumption_file_sha256", "credential_consumption_path",
    )
    executions = [pair["execution"] for pair in pairs]
    for field in process_fields:
        if len({canonical_json(execution[field]) for execution in executions}) != 1:
            raise ValueError(f"capture32 pairs do not share one process/engine {field}")
    call_indices = [index for pair in pairs for index in pair["pair_generate_call_indices"]]
    if call_indices != list(range(1, len(call_indices) + 1)) \
            or executions[0]["global_generate_call_count"] != len(call_indices) \
            or expected_binding.get("global_generate_call_count") != len(call_indices):
        raise ValueError("capture32 global generate-call sequence is not exact/contiguous")
    report = {
        "pair_count": CAPTURE32_COUNT,
        "stable_example_ids": [pair["stable_example_id"] for pair in pairs],
        "stable_root_ids": [pair["stable_root_id"] for pair in pairs],
        "stable_write_ids": [pair["stable_write_id"] for pair in pairs],
        "pair_ids": [pair["pair_id"] for pair in pairs],
        "generate_call_count": len(call_indices),
        "outcomes": {
            pair["stable_write_id"]: {
                arm: pair["arms"][arm]["final_reader"]["outcome"] for arm in ARMS
            }
            for pair in pairs
        },
    }
    return pairs, report


def _capture32_process_identity(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "pid", "boot_id", "process_start_ticks",
    } or type(value.get("pid")) is not int or value["pid"] < 1 \
            or not isinstance(value.get("boot_id"), str) or not value["boot_id"] \
            or type(value.get("process_start_ticks")) is not int \
            or value["process_start_ticks"] < 0:
        raise ValueError(f"capture32 {field} process identity is invalid")
    return dict(value)


def _validate_capture32_persisted_control_plane(
    *, manifest: Mapping[str, Any], capture_manifest: Mapping[str, Any],
    paths: Mapping[str, Path], p0: Mapping[str, Any], resolved: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently authenticate the stored supervisor/lock/credential chain."""
    run_id = str(p0["run_id"])
    git_commit = str(p0["git_commit"])
    physical = list(p0["physical_gpu_whitelist"])
    visible = str(p0["visible_devices"])
    lock_binding = resolved["gpu_lock_binding"]
    expected_lock_paths = list(capture_manifest["gpu"]["per_gpu_lock_paths"])
    selected = set(physical)
    for pair in ((4, 5), (6, 7)):
        if selected.intersection(pair):
            expected_lock_paths.append(
                f"{manifest['work_root']}/locks/memagent_gate_a_gpu_{pair[0]}_{pair[1]}.lock"
            )
    locks = lock_binding["locks"]
    if [item.get("path") for item in locks] != expected_lock_paths \
            or len({item.get("fd") for item in locks}) != len(locks) \
            or any(type(item.get(field)) is not int or item[field] < minimum
                   for item in locks
                   for field, minimum in (
                       ("fd", 0), ("device", 0), ("inode", 1), ("owner_uid", 0)
                   )):
        raise ValueError("capture32 persisted per-device/legacy lock inventory drifted")

    credential = json.loads(paths["capture_credential"].read_text(encoding="utf-8"))
    credential_fields = {
        "schema", "run_id", "git_commit", "child_kind", "parent_identity",
        "issued_at", "nonce", "single_use", "execution_binding_sha256",
        "runtime_binding_sha256", "current_binding_sha256",
        "gpu_lock_binding_sha256", "lock_holder_receipt",
        "training_authorized", "method_selected", "parent_credential_id",
    }
    unsigned_credential = dict(credential)
    credential_id = unsigned_credential.pop("parent_credential_id", None)
    if set(credential) != credential_fields \
            or credential_id != canonical_sha256(unsigned_credential) \
            or credential.get("schema") != "memagent.commit-retain.capture32-parent-credential.v1" \
            or credential.get("run_id") != run_id \
            or credential.get("git_commit") != git_commit \
            or credential.get("child_kind") != "single_engine_exact_32_pair_capture" \
            or credential.get("single_use") is not True \
            or credential.get("execution_binding_sha256") != resolved["execution_binding_sha256"] \
            or credential.get("runtime_binding_sha256") != resolved["runtime_binding_sha256"] \
            or credential.get("current_binding_sha256") != resolved["current_binding_sha256"] \
            or credential.get("gpu_lock_binding_sha256") != resolved["gpu_lock_binding_sha256"] \
            or credential.get("training_authorized") is not False \
            or credential.get("method_selected") is not False \
            or re.fullmatch(r"[0-9a-f]{64}", str(credential.get("nonce", ""))) is None:
        raise ValueError("capture32 persisted parent credential is non-canonical")
    parent = _capture32_process_identity(
        credential.get("parent_identity"), "credential parent"
    )
    holder = credential.get("lock_holder_receipt")
    if not isinstance(holder, Mapping) or set(holder) != {
        "schema", "physical_gpu_indices", "locks", "holder",
        "gpu_lock_binding_sha256", "lock_holder_receipt_sha256",
    }:
        raise ValueError("capture32 credential lock-holder receipt is malformed")
    unsigned_holder = dict(holder)
    holder_sha = unsigned_holder.pop("lock_holder_receipt_sha256", None)
    if holder_sha != canonical_sha256(unsigned_holder) \
            or holder.get("schema") != "memagent.capture32.lock-holder-receipt.v1" \
            or holder.get("physical_gpu_indices") != physical \
            or canonical_json({
                key: holder[key] for key in (
                    "schema", "physical_gpu_indices", "locks",
                    "gpu_lock_binding_sha256",
                )
            }) != canonical_json(lock_binding) \
            or holder.get("holder") != parent:
        raise ValueError("capture32 lock-holder receipt differs from P0 lock binding")

    consumption = json.loads(paths["credential_consumption"].read_text(encoding="utf-8"))
    consumption_fields = {
        "schema", "run_id", "git_commit", "parent_credential_id",
        "child_identity", "parent_identity", "consumed_at",
        "gpu_lock_binding_sha256", "credential_consumption_sha256",
    }
    unsigned_consumption = dict(consumption)
    consumption_sha = unsigned_consumption.pop("credential_consumption_sha256", None)
    if set(consumption) != consumption_fields \
            or consumption_sha != canonical_sha256(unsigned_consumption) \
            or consumption.get("schema") != "memagent.commit-retain.capture32-credential-consumption.v1" \
            or consumption.get("run_id") != run_id \
            or consumption.get("git_commit") != git_commit \
            or consumption.get("parent_credential_id") != credential_id \
            or consumption.get("parent_identity") != parent \
            or consumption.get("gpu_lock_binding_sha256") != resolved["gpu_lock_binding_sha256"]:
        raise ValueError("capture32 credential-consumption receipt is non-canonical")
    child = _capture32_process_identity(
        consumption.get("child_identity"), "credential child"
    )

    supervisor = read_jsonl(paths["supervisor_ledger"])
    record_types = [
        "s0_preflight", "capture_authorization", "capture_started",
        "capture_complete", "audit_result",
    ]
    if validate_jsonl_chain(supervisor) or [
        item.get("record_type") for item in supervisor
    ] != record_types:
        raise ValueError("capture32 supervisor state machine/hash chain failed")
    _capture32_preflight_module._validate_supervisor_schema(supervisor)
    base_fields = {
        "record_type", "experiment_name", "git_commit", "run_id", "recorded_at",
        "eval_manifest_hash", "execution_binding_sha256",
        "runtime_binding_sha256", "current_binding_sha256",
        "gpu_lock_binding_sha256", "artifact", "artifact_sha256", "status",
        "decision", "training_authorized", "method_selected", "record_index",
        "previous_record_sha256", "record_sha256",
    }
    extras = {
        "s0_preflight": {
            "resolved_manifest", "resolved_manifest_sha256",
            "external_preregistration_anchor",
            "external_preregistration_anchor_sha256",
        },
        "capture_authorization": {
            "lock_holder_receipt_sha256", "parent_credential_id",
            "parent_issuer_pid",
        },
        "capture_started": {
            "lock_holder_receipt_sha256", "parent_credential_id",
            "credential_consumption_sha256",
        },
        "capture_complete": {
            "run_receipt", "run_receipt_sha256", "pair_count", "pair_ids",
            "stable_write_ids", "generate_call_count",
        },
        "audit_result": {
            "pair_count", "pair_ids", "stable_write_ids", "generate_call_count",
        },
    }
    decisions = {
        "s0_preflight": "COMMIT_RETAIN_CAPTURE32_P0_PASS",
        "capture_authorization": "COMMIT_RETAIN_CAPTURE32_CHILD_AUTHORIZED",
        "capture_started": "COMMIT_RETAIN_CAPTURE32_STARTED",
        "capture_complete": "COMMIT_RETAIN_CAPTURE32_COMPLETE",
        "audit_result": "COMMIT_RETAIN_CAPTURE32_LOCAL_AUDIT_COMPLETE_PROVENANCE_PENDING",
    }
    artifact_paths = {
        "s0_preflight": paths["p0_certificate"],
        "capture_authorization": paths["capture_credential"],
        "capture_started": paths["credential_consumption"],
        "capture_complete": paths["capture_ledger"],
        "audit_result": paths["final_report"],
    }
    for record_type, record in zip(record_types, supervisor):
        artifact_path = artifact_paths[record_type]
        if set(record) != base_fields | extras[record_type] \
                or record.get("experiment_name") != "qwen25_7b_commit_retain_capture32_seed2026" \
                or record.get("git_commit") != git_commit \
                or record.get("run_id") != run_id \
                or record.get("eval_manifest_hash") != resolved["eval_manifest_hash"] \
                or record.get("execution_binding_sha256") != resolved["execution_binding_sha256"] \
                or record.get("runtime_binding_sha256") != resolved["runtime_binding_sha256"] \
                or record.get("current_binding_sha256") != resolved["current_binding_sha256"] \
                or record.get("gpu_lock_binding_sha256") != resolved["gpu_lock_binding_sha256"] \
                or record.get("artifact") != str(artifact_path.resolve()) \
                or record.get("artifact_sha256") != sha256_file(artifact_path) \
                or record.get("status") != "PASS" \
                or record.get("decision") != decisions[record_type] \
                or record.get("training_authorized") is not False \
                or record.get("method_selected") is not False:
            raise ValueError(f"capture32 supervisor {record_type} record drifted")
    if supervisor[0].get("resolved_manifest") != str(paths["resolved_manifest"].resolve()) \
            or supervisor[0].get("resolved_manifest_sha256") != sha256_file(paths["resolved_manifest"]) \
            or supervisor[0].get("external_preregistration_anchor") != str(
                paths["external_preregistration_anchor"].resolve()
            ) \
            or supervisor[0].get("external_preregistration_anchor_sha256") != sha256_file(
                paths["external_preregistration_anchor"]
            ):
        raise ValueError("capture32 P0 supervisor artifact bindings drifted")
    if supervisor[1].get("lock_holder_receipt_sha256") != holder_sha \
            or supervisor[1].get("parent_credential_id") != credential_id \
            or supervisor[1].get("parent_issuer_pid") != parent["pid"] \
            or supervisor[2].get("lock_holder_receipt_sha256") != holder_sha \
            or supervisor[2].get("parent_credential_id") != credential_id \
            or supervisor[2].get("credential_consumption_sha256") != consumption_sha:
        raise ValueError("capture32 supervisor credential transition drifted")
    return {
        "parent_credential_id": credential_id,
        "parent_credential_sha256": sha256_file(paths["capture_credential"]),
        "parent_credential_path": str(paths["capture_credential"].resolve()),
        "parent_issuer_pid": parent["pid"],
        "observed_parent_pid": parent["pid"],
        "parent_authorization_record_sha256": supervisor[1]["record_sha256"],
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "lock_holder_receipt_sha256": holder_sha,
        "credential_consumption_sha256": consumption_sha,
        "credential_consumption_file_sha256": sha256_file(
            paths["credential_consumption"]
        ),
        "credential_consumption_path": str(
            paths["credential_consumption"].resolve()
        ),
        "child_identity": child,
        "supervisor": supervisor,
    }


def authenticate_capture32(
    manifest: Mapping[str, Any], *, tokenizer_factory: Any = _tokenizer
) -> dict[str, Any]:
    state, paths = _capture32_artifact_state(manifest)
    if state == "NOT_STARTED":
        raise CapturePendingError(["capture32_not_started"])
    prereg = load_capture32_preregistration(manifest)
    p0 = json.loads(paths["p0_certificate"].read_text(encoding="utf-8"))
    resolved = json.loads(paths["resolved_manifest"].read_text(encoding="utf-8"))
    required_p0 = {
        "schema", "status", "decision", "run_id", "git_commit",
        "preregistration", "preregistration_file_sha256",
        "preregistration_sha256", "resolved_manifest", "resolved_manifest_sha256",
        "capture_ledger", "final_report", "physical_gpu_whitelist",
        "visible_devices", "gpu_lock_binding_sha256",
        "commitment_frozen_before_first_generate",
        "training_authorized", "method_selected", "failures",
    }
    prereg_path = Path(manifest["repo_dir"]) / CAPTURE32_PREREG_REL
    p0_physical = p0.get("physical_gpu_whitelist")
    physical_ok = isinstance(p0_physical, list) \
        and len(p0_physical) == 2 \
        and all(type(item) is int and item >= 0 for item in p0_physical) \
        and p0_physical == sorted(set(p0_physical))
    if set(p0) != required_p0 or any((
        p0.get("schema") != "memagent.commit-retain.capture32-p0.v2",
        p0.get("status") != "PASS",
        p0.get("decision") != "COMMIT_RETAIN_CAPTURE32_P0_PASS",
        p0.get("run_id") != manifest["capture_run_id"],
        p0.get("preregistration") != str(prereg_path.resolve()),
        p0.get("preregistration_file_sha256") != sha256_file(prereg_path),
        p0.get("preregistration_sha256") != prereg["preregistration_sha256"],
        p0.get("resolved_manifest") != str(paths["resolved_manifest"].resolve()),
        p0.get("resolved_manifest_sha256") != sha256_file(paths["resolved_manifest"]),
        p0.get("capture_ledger") != str(paths["capture_ledger"].resolve()),
        p0.get("final_report") != str(paths["final_report"].resolve()),
        not physical_ok,
        p0.get("visible_devices") != (
            ",".join(str(item) for item in p0_physical) if physical_ok else None
        ),
        not _is_sha(p0.get("gpu_lock_binding_sha256")),
        p0.get("commitment_frozen_before_first_generate") is not True,
        p0.get("training_authorized") is not False,
        p0.get("method_selected") is not False,
        p0.get("failures") != [],
    )):
        raise ValueError("capture32 P0 commitment does not exactly bind preregistration/artifacts")
    required_resolved = {
        "schema", "run_id", "git_commit", "preregistration_sha256",
        "source_validation_sha256", "eval_manifest_hash", "base_model_id",
        "base_model_revision", "model_file_manifest_sha256",
        "tokenizer_manifest_sha256", "s128_authority_file_sha256",
        "s128_authority_sha256", "rollout_backend",
        "physical_gpu_whitelist", "visible_devices",
        "selected_inventory_sha256", "full_population_ranking_sha256",
        "fold_membership_sha256", "frozen_pairs", "execution_binding_sha256",
        "runtime_binding_sha256", "current_binding_sha256", "execution_binding",
        "runtime_binding", "current_binding", "expected_pair_binding",
        "execution_code_sha256", "execution_code_combined_sha256",
        "gpu_lock_binding", "gpu_lock_binding_sha256",
        "lightweight_current_binding", "lightweight_current_binding_sha256",
        "expected_global_generate_call_count",
    }
    if set(resolved) != required_resolved \
            or resolved.get("schema") != "memagent.commit-retain.capture32-resolved.v2" \
            or resolved.get("run_id") != manifest["capture_run_id"] \
            or resolved.get("git_commit") != p0.get("git_commit") \
            or resolved.get("preregistration_sha256") != prereg["preregistration_sha256"] \
            or resolved.get("source_validation_sha256") != prereg["source"][
                "validation_sha256"
            ] \
            or resolved.get("eval_manifest_hash") != prereg["source"][
                "eval_manifest_hash"
            ] \
            or resolved.get("base_model_id") != prereg["source"]["base_model_id"] \
            or resolved.get("base_model_revision") != prereg["source"][
                "base_model_revision"
            ] \
            or resolved.get("model_file_manifest_sha256") != prereg["source"][
                "model_file_manifest_sha256"
            ] \
            or resolved.get("tokenizer_manifest_sha256") != prereg["source"][
                "tokenizer_manifest_sha256"
            ] \
            or resolved.get("s128_authority_file_sha256") != prereg["source"][
                "s128_authority_file_sha256"
            ] \
            or resolved.get("s128_authority_sha256") != prereg["source"][
                "s128_authority_sha256"
            ] \
            or resolved.get("rollout_backend") != "strict_vllm_0.8.2" \
            or resolved.get("physical_gpu_whitelist") != p0.get(
                "physical_gpu_whitelist"
            ) \
            or resolved.get("visible_devices") != p0.get("visible_devices") \
            or resolved.get("gpu_lock_binding_sha256") != p0.get(
                "gpu_lock_binding_sha256"
            ) \
            or resolved.get("selected_inventory_sha256") != prereg["inventory"][
                "selected_inventory_sha256"
            ] \
            or resolved.get("full_population_ranking_sha256") != prereg["selection"][
                "full_population_ranking_sha256"
            ] \
            or resolved.get("fold_membership_sha256") != prereg["folds"][
                "membership_sha256"
            ] \
            or canonical_json(resolved.get("frozen_pairs")) != canonical_json(
                prereg["selected_inventory"]
            ) \
            or resolved.get("expected_global_generate_call_count") != 353:
        raise ValueError("capture32 resolved manifest differs from preregistration")
    for field in (
        "execution_binding_sha256", "runtime_binding_sha256", "current_binding_sha256",
        "execution_code_combined_sha256", "gpu_lock_binding_sha256",
        "lightweight_current_binding_sha256",
    ):
        if not _is_sha(resolved.get(field)):
            raise ValueError(f"capture32 resolved {field} is invalid")
    git_evidence = _authenticate_capture32_git(
        repo=Path(manifest["repo_dir"]),
        git_commit=str(p0["git_commit"]),
        execution_code_sha256=resolved["execution_code_sha256"],
    )
    if resolved["execution_code_combined_sha256"] != git_evidence[
        "execution_code_combined_sha256"
    ]:
        raise ValueError("capture32 combined code inventory digest mismatch")
    _validate_capture32_runtime_bindings(resolved, preregistration=prereg)

    producer_manifest_path = (
        Path(manifest["repo_dir"]) / manifest["capture32"]["producer_manifest"]
    )
    if producer_manifest_path.is_symlink() \
            or producer_manifest_path.resolve() != (
                REPO_ROOT / "manifests/h20/qwen25_7b_commit_retain_capture32_seed2026.json"
            ).resolve():
        raise ValueError("capture32 producer manifest path escaped frozen Git closure")
    capture_manifest = load_capture32_producer_manifest(
        producer_manifest_path,
        {
            "MEMAGENT_CAPTURE32_WORK_ROOT": str(manifest["work_root"]),
            "MEMAGENT_CAPTURE32_REPO_DIR": str(manifest["repo_dir"]),
            "MEMAGENT_CAPTURE32_EXPECTED_COMMIT": str(p0["git_commit"]),
            "MEMAGENT_CAPTURE32_RUN_ID": str(manifest["capture_run_id"]),
            "MEMAGENT_CAPTURE32_PHYSICAL_GPUS": str(p0["visible_devices"]),
        },
    )
    validate_capture32_producer_manifest(capture_manifest)
    producer_path_bindings = {
        "p0_certificate": "p0_certificate",
        "resolved_manifest": "resolved_manifest",
        "supervisor_ledger": "execution_ledger",
        "capture_credential": "capture_credential",
        "capture_ledger": "capture_ledger",
        "capture_run_receipt": "capture_run_receipt",
        "final_report": "final_report",
        "external_preregistration_anchor": "external_preregistration_anchor",
        "external_terminal_anchor": "external_terminal_anchor",
    }
    for consumer_name, producer_name in producer_path_bindings.items():
        if paths[consumer_name].resolve() != Path(
            capture_manifest["paths"][producer_name]
        ).resolve():
            raise ValueError(f"capture32 producer/consumer path split: {consumer_name}")
    if paths["credential_consumption"].resolve() != Path(
        capture_manifest["paths"]["capture_credential"]
    ).with_name("capture_child_consumed.json").resolve():
        raise ValueError("capture32 producer/consumer credential-consumption path split")
    tokenizer = tokenizer_factory(capture_manifest)
    from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template

    _assert_loaded_repo_module_origins()
    chat = chat_template(tokenizer)
    writer_template_text = chat.format(message=TEMPLATE)
    reader_template_text = chat.format(message=TEMPLATE_FINAL_BOXED)
    writer_template = TokenTemplate(writer_template_text, tokenizer)
    reader_template = TokenTemplate(reader_template_text, tokenizer)
    writer_prompt_builder = lambda question, memory, chunk: writer_template.format(
        prompt=question, memory=memory, chunk=chunk
    ).tolist()
    source_replay = _authenticate_capture32_source_replay(
        manifest=manifest,
        capture_manifest=capture_manifest,
        preregistration=prereg,
        tokenizer=tokenizer,
        writer_prompt_builder=writer_prompt_builder,
    )
    binding = resolved["expected_pair_binding"]
    physical = list(p0["physical_gpu_whitelist"])
    visible = str(p0["visible_devices"])
    engine_config = {
        **dict(capture_manifest["backend"]),
        "physical_gpu_whitelist": physical,
        "visible_devices": visible,
        "tensor_parallel_size": 2,
        "one_prompt_per_generate_call": True,
    }
    expected_generate_calls = sum(
        int(row["intervention_writer_turn"])
        + 1
        + 2 * (
            int(row["total_writer_turns"])
            - int(row["intervention_writer_turn"])
        )
        for row in prereg["selected_inventory"]
    )
    if expected_generate_calls != 353 \
            or resolved.get("expected_global_generate_call_count") != 353:
        raise ValueError("capture32 frozen exact 32/353 schedule drifted")
    static_binding = {
        "writer_checkpoint_sha256": prereg["source"]["model_file_manifest_sha256"],
        "reader_checkpoint_sha256": prereg["source"]["model_file_manifest_sha256"],
        "writer_prompt_template_sha256": hashlib.sha256(
            writer_template_text.encode("utf-8")
        ).hexdigest(),
        "reader_prompt_template_sha256": hashlib.sha256(
            reader_template_text.encode("utf-8")
        ).hexdigest(),
        "writer_decode": capture_manifest["intervention"]["writer_decode"],
        "reader_decode": capture_manifest["intervention"]["reader_decode"],
        "physical_gpu_whitelist": physical,
        "visible_devices": visible,
        "worker_multiproc_method": "spawn",
        "vllm_observed_worker_multiproc_method": "spawn",
        "multiprocessing_context_method": "spawn",
        "parent_cuda_initialization_policy": "record_observed_spawn_required",
        "engine_config_sha256": canonical_sha256(engine_config),
        "global_generate_call_count": expected_generate_calls,
        "eos_token_id": int(tokenizer.eos_token_id),
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
    }
    binding_fields = {
        "writer_checkpoint_sha256", "reader_checkpoint_sha256",
        "writer_prompt_template_sha256", "reader_prompt_template_sha256",
        "writer_decode", "reader_decode", "physical_gpu_whitelist",
        "visible_devices", "physical_gpu_identity", "engine_config_sha256",
        "worker_multiproc_method", "vllm_observed_worker_multiproc_method",
        "multiprocessing_context_method", "parent_cuda_initialization_policy",
        "global_generate_call_count", "eos_token_id",
        "gpu_lock_binding_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != binding_fields or any(
        canonical_json(binding.get(field)) != canonical_json(expected)
        for field, expected in static_binding.items()
    ) or any(not _is_sha(binding.get(field)) for field in (
        "writer_checkpoint_sha256", "reader_checkpoint_sha256",
        "writer_prompt_template_sha256", "reader_prompt_template_sha256",
        "engine_config_sha256",
    )) or binding.get("writer_checkpoint_sha256") != binding.get(
        "reader_checkpoint_sha256"
    ):
        raise ValueError("capture32 strict-vLLM/model/decode binding drifted")
    control = _validate_capture32_persisted_control_plane(
        manifest=manifest,
        capture_manifest=capture_manifest,
        paths=paths,
        p0=p0,
        resolved=resolved,
    )
    execution_evidence = {
        key: control[key] for key in (
            "parent_credential_id", "parent_credential_sha256",
            "parent_credential_path", "parent_issuer_pid", "observed_parent_pid",
            "parent_authorization_record_sha256", "gpu_lock_binding_sha256",
            "lock_holder_receipt_sha256", "credential_consumption_sha256",
            "credential_consumption_file_sha256", "credential_consumption_path",
        )
    }
    records = read_jsonl(paths["capture_ledger"])
    pairs, capture_report = _validate_capture32_envelopes(
        records,
        preregistration=prereg,
        run_id=manifest["capture_run_id"],
        git_commit=str(p0["git_commit"]),
        resolved=resolved,
        decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        writer_prompt_builder=writer_prompt_builder,
        reader_prompt_builder=lambda question, memory: reader_template.format(
            prompt=question, memory=memory
        ).tolist(),
        execution_evidence=execution_evidence,
    )
    first_execution = pairs[0]["execution"]
    if first_execution.get("process_pid") != control["child_identity"]["pid"]:
        raise ValueError("capture32 credential child process differs from vLLM producer")
    claim_boundary = {
        "capture_and_audit_only": True,
        "method_selected": False,
        "training_authorized": False,
        "causal_or_performance_claim": False,
    }
    training = {"trainer_attached": False, "actor_updates": 0, "optimizer_steps": 0}
    expected_receipt_unsigned = {
        "schema": "memagent.commit-retain.capture32-run-receipt.v1",
        "run_id": manifest["capture_run_id"],
        "git_commit": p0["git_commit"],
        "eval_manifest_hash": prereg["source"]["eval_manifest_hash"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "current_binding_sha256": resolved["current_binding_sha256"],
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "capture_ledger": str(paths["capture_ledger"].resolve()),
        "capture_ledger_sha256": sha256_file(paths["capture_ledger"]),
        "pair_count": 32,
        "pair_ids": capture_report["pair_ids"],
        "stable_write_ids": capture_report["stable_write_ids"],
        "generate_call_count": 353,
        "execution": first_execution,
        "training": training,
        "claim_boundary": claim_boundary,
    }
    expected_receipt = {
        **expected_receipt_unsigned,
        "run_receipt_id": canonical_sha256(expected_receipt_unsigned),
    }
    actual_receipt = json.loads(
        paths["capture_run_receipt"].read_text(encoding="utf-8")
    )
    if canonical_json(actual_receipt) != canonical_json(expected_receipt):
        raise ValueError("capture32 run receipt does not independently reproduce")
    complete, audit = control["supervisor"][3], control["supervisor"][4]
    for record, is_complete in ((complete, True), (audit, False)):
        if record.get("pair_count") != 32 \
                or record.get("pair_ids") != capture_report["pair_ids"] \
                or record.get("stable_write_ids") != capture_report["stable_write_ids"] \
                or record.get("generate_call_count") != 353:
            raise ValueError("capture32 supervisor exact inventory receipt drifted")
        if is_complete and (
            record.get("run_receipt") != str(paths["capture_run_receipt"].resolve())
            or record.get("run_receipt_sha256") != sha256_file(
                paths["capture_run_receipt"]
            )
        ):
            raise ValueError("capture32 supervisor run-receipt binding drifted")
    expected_final = {
        "schema": "memagent.commit-retain.capture32-final.v2",
        "status": "PASS",
        "decision": "COMMIT_RETAIN_CAPTURE32_LOCAL_AUDIT_COMPLETE_PROVENANCE_PENDING",
        "run_id": manifest["capture_run_id"],
        "git_commit": p0["git_commit"],
        "eval_manifest_hash": prereg["source"]["eval_manifest_hash"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "p0_certificate": str(paths["p0_certificate"].resolve()),
        "p0_certificate_sha256": sha256_file(paths["p0_certificate"]),
        "resolved_manifest": str(paths["resolved_manifest"].resolve()),
        "resolved_manifest_sha256": sha256_file(paths["resolved_manifest"]),
        "capture_ledger": str(paths["capture_ledger"].resolve()),
        "capture_ledger_sha256": sha256_file(paths["capture_ledger"]),
        "capture_run_receipt": str(paths["capture_run_receipt"].resolve()),
        "capture_run_receipt_sha256": sha256_file(paths["capture_run_receipt"]),
        **{
            key: capture_report[key] for key in (
                "pair_count", "stable_write_ids", "pair_ids",
                "generate_call_count", "outcomes",
            )
        },
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "external_provenance_status": "PENDING_EXTERNAL_SIGNATURE",
        "training": training,
        "claim_boundary": {
            "development_admissibility_only": True,
            "method_selected": False,
            "training_authorized": False,
            "paper_performance_result": False,
            "causal_effect_claim": False,
        },
    }
    actual_final = json.loads(paths["final_report"].read_text(encoding="utf-8"))
    if set(actual_final) != set(expected_final) \
            or canonical_json(actual_final) != canonical_json(expected_final):
        raise ValueError("capture32 final report does not independently reproduce")
    prereg_anchor = json.loads(
        paths["external_preregistration_anchor"].read_text(encoding="utf-8")
    )
    _capture32_preflight_module._validate_anchor_schema(prereg_anchor)
    unsigned_prereg_anchor = dict(prereg_anchor)
    prereg_anchor_digest = unsigned_prereg_anchor.pop("anchor_payload_sha256", None)
    expected_prereg_anchor_fields = {
        "schema", "anchor_kind", "trust_status", "run_id", "git_commit",
        "recorded_at", "preregistration", "preregistration_sha256",
        "p0_certificate", "p0_certificate_sha256", "resolved_manifest",
        "resolved_manifest_sha256", "training_authorized", "method_selected",
        "anchor_payload_sha256",
    }
    if set(prereg_anchor) != expected_prereg_anchor_fields \
            or prereg_anchor_digest != canonical_sha256(unsigned_prereg_anchor) \
            or prereg_anchor.get("schema") != "memagent.commit-retain.capture32-provenance-anchor.v1" \
            or prereg_anchor.get("anchor_kind") != "PRE_GENERATION_LOCAL_EXPORT_CANDIDATE" \
            or prereg_anchor.get("trust_status") != "PENDING_EXTERNAL_SIGNATURE" \
            or prereg_anchor.get("run_id") != manifest["capture_run_id"] \
            or prereg_anchor.get("git_commit") != p0["git_commit"] \
            or prereg_anchor.get("preregistration") != str(prereg_path.resolve()) \
            or prereg_anchor.get("preregistration_sha256") != sha256_file(prereg_path) \
            or prereg_anchor.get("p0_certificate") != str(paths["p0_certificate"].resolve()) \
            or prereg_anchor.get("p0_certificate_sha256") != sha256_file(paths["p0_certificate"]) \
            or prereg_anchor.get("resolved_manifest") != str(paths["resolved_manifest"].resolve()) \
            or prereg_anchor.get("resolved_manifest_sha256") != sha256_file(paths["resolved_manifest"]) \
            or prereg_anchor.get("training_authorized") is not False \
            or prereg_anchor.get("method_selected") is not False:
        raise ValueError("capture32 pre-generation local export anchor drifted")
    terminal_anchor = json.loads(
        paths["external_terminal_anchor"].read_text(encoding="utf-8")
    )
    _capture32_preflight_module._validate_anchor_schema(terminal_anchor)
    unsigned_terminal_anchor = dict(terminal_anchor)
    terminal_anchor_digest = unsigned_terminal_anchor.pop(
        "anchor_payload_sha256", None
    )
    expected_terminal_fields = {
        "schema", "anchor_kind", "trust_status", "run_id", "git_commit",
        "recorded_at", "pre_generation_anchor", "pre_generation_anchor_sha256",
        "supervisor_ledger", "supervisor_ledger_sha256",
        "supervisor_tail_sha256", "capture_ledger", "capture_ledger_sha256",
        "capture_run_receipt", "capture_run_receipt_sha256", "final_report",
        "final_report_sha256", "pair_ids_sha256", "stable_write_ids_sha256",
        "training_authorized", "method_selected", "anchor_payload_sha256",
    }
    supervisor = control["supervisor"]
    if set(terminal_anchor) != expected_terminal_fields \
            or terminal_anchor_digest != canonical_sha256(unsigned_terminal_anchor) \
            or terminal_anchor.get("schema") != "memagent.commit-retain.capture32-provenance-anchor.v1" \
            or terminal_anchor.get("anchor_kind") != "POST_CAPTURE_LOCAL_EXPORT_CANDIDATE" \
            or terminal_anchor.get("trust_status") != "PENDING_EXTERNAL_SIGNATURE" \
            or terminal_anchor.get("run_id") != manifest["capture_run_id"] \
            or terminal_anchor.get("git_commit") != p0["git_commit"] \
            or terminal_anchor.get("pre_generation_anchor") != str(paths["external_preregistration_anchor"].resolve()) \
            or terminal_anchor.get("pre_generation_anchor_sha256") != sha256_file(paths["external_preregistration_anchor"]) \
            or terminal_anchor.get("supervisor_ledger") != str(paths["supervisor_ledger"].resolve()) \
            or terminal_anchor.get("supervisor_ledger_sha256") != sha256_file(paths["supervisor_ledger"]) \
            or terminal_anchor.get("supervisor_tail_sha256") != supervisor[-1]["record_sha256"] \
            or terminal_anchor.get("capture_ledger") != str(paths["capture_ledger"].resolve()) \
            or terminal_anchor.get("capture_ledger_sha256") != sha256_file(paths["capture_ledger"]) \
            or terminal_anchor.get("capture_run_receipt") != str(paths["capture_run_receipt"].resolve()) \
            or terminal_anchor.get("capture_run_receipt_sha256") != sha256_file(paths["capture_run_receipt"]) \
            or terminal_anchor.get("final_report") != str(paths["final_report"].resolve()) \
            or terminal_anchor.get("final_report_sha256") != sha256_file(paths["final_report"]) \
            or terminal_anchor.get("pair_ids_sha256") != canonical_sha256(capture_report["pair_ids"]) \
            or terminal_anchor.get("stable_write_ids_sha256") != canonical_sha256(capture_report["stable_write_ids"]) \
            or terminal_anchor.get("training_authorized") is not False \
            or terminal_anchor.get("method_selected") is not False:
        raise ValueError("capture32 terminal local export anchor drifted")
    return {
        "capture_role": "preregistered_capture32",
        # These values have already been closed across P0, resolved/runtime/
        # current bindings, every pair record, and the persisted control
        # plane above.  Carry them into the consumer report so the report
        # cannot silently relabel a dynamic two-H20 run as the legacy gpu45
        # profile.
        "physical_gpu_whitelist": list(p0["physical_gpu_whitelist"]),
        "visible_devices": str(p0["visible_devices"]),
        "pairs": pairs,
        "contract_evidence": _contract_evidence(pairs),
        "capture_report": capture_report,
        "git_evidence": git_evidence,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "resolved_binding": {
            field: resolved[field] for field in (
                "execution_binding_sha256", "runtime_binding_sha256",
                "current_binding_sha256",
            )
        },
        "credential": {"p0_commitment_sha256": sha256_file(paths["p0_certificate"])},
        "preregistration": prereg,
        "source_replay": source_replay,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    mean = _mean(values)
    return _mean([(value - mean) ** 2 for value in values])


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_mean, right_mean = _mean(left), _mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0 else None


def _rankdata(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in ordered[cursor:end]:
            ranks[position] = rank
        cursor = end
    return ranks


def crossfit_diagnostics(
    bundle: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_id = {str(row["stable_example_id"]): row for row in observations}
    scores = [float(row["paired_effect_score"]) for row in bundle["scores"]]
    targets = [float(by_id[str(row["stable_example_id"])]["paired_effect_target"])
               for row in bundle["scores"]]
    baseline: list[float] = []
    for score in bundle["scores"]:
        model = bundle["models"][str(score["score_fold"])]
        fit_targets = [
            float(by_id[stable_id]["paired_effect_target"])
            for stable_id in model["fit_stable_example_ids"]
        ]
        baseline.append(_mean(fit_targets))
    mse = _mean([(score - target) ** 2 for score, target in zip(scores, targets)])
    baseline_mse = _mean([(score - target) ** 2 for score, target in zip(baseline, targets)])
    improvement = (baseline_mse - mse) / baseline_mse if baseline_mse > 0 else None
    epsilon = 0.01
    precision = 0.000001
    per_fold: dict[str, Any] = {}
    for fold in range(int(bundle["fold_count"])):
        indices = [index for index, row in enumerate(bundle["scores"])
                   if int(row["score_fold"]) == fold]
        fold_scores = [scores[index] for index in indices]
        fold_targets = [targets[index] for index in indices]
        fold_baseline = [baseline[index] for index in indices]
        fold_mse = _mean([(score - target) ** 2
                          for score, target in zip(fold_scores, fold_targets)])
        fold_baseline_mse = _mean([(score - target) ** 2
                                   for score, target in zip(fold_baseline, fold_targets)])
        per_fold[str(fold)] = {
            "heldout_count": len(indices),
            "fit_count": len(bundle["models"][str(fold)]["fit_stable_example_ids"]),
            "target_mean": _mean(fold_targets),
            "target_variance": _variance(fold_targets),
            "crossfit_mse": fold_mse,
            "fold_mean_baseline_mse": fold_baseline_mse,
            "mse_improvement_fraction": (
                (fold_baseline_mse - fold_mse) / fold_baseline_mse
                if fold_baseline_mse > 0 else None
            ),
        }
    score_variance = _variance(scores)
    target_mean = _mean(targets)
    score_mean = _mean(scores)
    covariance = _mean([
        (score - score_mean) * (target - target_mean)
        for score, target in zip(scores, targets)
    ])
    calibration_slope = covariance / score_variance if score_variance > 0 else None
    calibration_intercept = (
        target_mean - calibration_slope * score_mean
        if calibration_slope is not None else None
    )
    return {
        "stable_example_count": len(targets),
        "target_mean": target_mean,
        "target_median": _median(targets),
        "target_min": min(targets),
        "target_max": max(targets),
        "target_variance": _variance(targets),
        "mean_absolute_effect": _mean([abs(value) for value in targets]),
        "median_absolute_effect": _median([abs(value) for value in targets]),
        "nontrivial_effect_epsilon": epsilon,
        "nontrivial_effect_count": sum(abs(value) >= epsilon for value in targets),
        "positive_effect_count": sum(value >= epsilon for value in targets),
        "near_zero_effect_count": sum(abs(value) < epsilon for value in targets),
        "negative_effect_count": sum(value <= -epsilon for value in targets),
        "effect_bin_precision": precision,
        "distinct_effect_bin_count": len({round(value / precision) for value in targets}),
        "crossfit_mse": mse,
        "fold_mean_baseline_mse": baseline_mse,
        "crossfit_mse_improvement_fraction": improvement,
        "crossfit_pearson_correlation": _pearson(scores, targets),
        "crossfit_spearman_correlation": _pearson(
            _rankdata(scores), _rankdata(targets)
        ),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "score_mean": score_mean,
        "score_variance": score_variance,
        "score_min": min(scores),
        "score_max": max(scores),
        "nonfinite_score_count": sum(not math.isfinite(value) for value in scores),
        "folds_with_positive_mse_improvement": sum(
            row["mse_improvement_fraction"] is not None
            and row["mse_improvement_fraction"] > 0
            for row in per_fold.values()
        ),
        "per_fold": per_fold,
        "descriptive_only_until_minimum_sample": True,
    }


def _readiness(
    manifest: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    contract = manifest["admissibility"]
    enough = diagnostics["stable_example_count"] >= contract[
        "minimum_training_stable_examples"
    ]
    gates: dict[str, Any] = {
        "minimum_unique_stable_examples": {
            "status": "PASS" if enough else "PENDING",
            "observed": diagnostics["stable_example_count"],
            "required": contract["minimum_training_stable_examples"],
        },
        "predictive_signal": {"status": "NOT_APPLICABLE", "failures": []},
    }
    if not enough:
        return "PENDING", contract["pending_decision_more_capture"], gates
    failures: list[str] = []
    checks = (
        (diagnostics["nontrivial_effect_epsilon"] == contract["nontrivial_effect_epsilon"],
         "paired-effect epsilon differs from preregistration"),
        (diagnostics["nontrivial_effect_count"] >= contract[
            "minimum_nontrivial_effect_examples"
        ], "too few nontrivial paired effects"),
        (diagnostics["effect_bin_precision"] == contract["effect_bin_precision"],
         "paired-effect bin precision differs from preregistration"),
        (diagnostics["distinct_effect_bin_count"] >= contract[
            "minimum_distinct_effect_bins"
        ], "too few distinct paired-effect bins"),
        (diagnostics["mean_absolute_effect"] >= contract["minimum_mean_absolute_effect"],
         "mean absolute paired effect is too small"),
        (diagnostics["target_variance"] >= contract["minimum_target_variance"],
         "paired-effect target variance is too small"),
        (diagnostics["crossfit_mse_improvement_fraction"] is not None and
         diagnostics["crossfit_mse_improvement_fraction"] >=
         contract["minimum_crossfit_mse_improvement_fraction"],
         "crossfit scorer does not improve over its fold-mean baseline"),
        (diagnostics["crossfit_pearson_correlation"] is not None and
         diagnostics["crossfit_pearson_correlation"] >=
         contract["minimum_crossfit_pearson_correlation"],
         "crossfit score/target correlation is not positive"),
        (diagnostics["folds_with_positive_mse_improvement"] >= contract[
            "minimum_folds_with_positive_mse_improvement"
        ], "too few folds improve over the fit-fold mean"),
        (all(row["heldout_count"] == contract["minimum_heldout_examples_per_fold"]
             and row["fit_count"] == contract["minimum_fit_examples_per_fold"]
             for row in diagnostics["per_fold"].values()),
         "crossfit fold sizes differ from preregistered 8 heldout/24 fit"),
        (diagnostics["nonfinite_score_count"] == 0,
         "crossfit contains non-finite scores"),
    )
    failures.extend(message for passed, message in checks if not passed)
    gates["predictive_signal"] = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    if failures:
        return "FAIL", "PAIRED_EFFECT_NO_GO:SCORER_ADMISSIBILITY", gates
    return "PASS", contract["evidence_ready_decision"], gates


def build_report(
    manifest: Mapping[str, Any], *, verify_pipeline_git: bool = True,
    artifact_written: bool = False, tokenizer_factory: Any = _tokenizer,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    pipeline = _authenticate_pipeline_git(manifest) if verify_pipeline_git else {
        "git_commit": manifest["expected_pipeline_commit"],
        "code_sha256": {},
        "code_combined_sha256": None,
    }
    manifest_path = Path(manifest["repo_dir"]) / MANIFEST_REL
    schema_path = Path(manifest["repo_dir"]) / manifest["report_schema"]
    preregistration_path = Path(manifest["repo_dir"]) / CAPTURE32_PREREG_REL
    preregistration = load_capture32_preregistration(manifest)
    capture32_source_base = {
        "capture_role": "preregistered_capture32",
        "capture_run_id": manifest["capture_run_id"],
        "expected_profile": "dynamic_explicit_two_h20",
        "expected_root": manifest["capture32"]["output_root"],
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_file_sha256": sha256_file(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "required_pair_count": CAPTURE32_COUNT,
    }
    pilot4_source_base = {
        "capture_role": "pipeline_pilot4_only",
        "capture_run_id": manifest["capture_run_id"],
        "expected_git_commit": manifest["source_capture"]["expected_git_commit"],
        "expected_profile": "gpu45",
        "expected_root": manifest["source_capture"]["output_root"],
        "required_pair_count": 4,
    }
    common = {
        "schema": REPORT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "run_id": manifest["run_id"],
        "pipeline": {
            **pipeline,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "report_schema": str(schema_path.resolve()),
            "report_schema_sha256": sha256_file(schema_path),
            "capture32_preregistration": str(preregistration_path.resolve()),
            "capture32_preregistration_file_sha256": sha256_file(
                preregistration_path
            ),
            "capture32_preregistration_sha256": preregistration[
                "preregistration_sha256"
            ],
        },
        "claim_boundary": manifest["claim_boundary"],
        "training_authorized": False,
        "method_selected": False,
        "provenance_limitation": (
            "Internal SHA chains and reviewed Git blobs are tamper-detecting but are not an "
            "external cryptographic signature or hardware attestation."
        ),
        "failures": [],
    }

    def capture_failure(
        error: Exception, *, source: Mapping[str, Any], decision: str,
        inventory_status: str = "FAIL",
    ) -> tuple[dict[str, Any], None]:
        message = str(error)
        return ({
            **common,
            "status": "FAIL",
            "decision": decision,
            "source_capture": dict(source),
            "gates": {
                "capture32_preregistration": {"status": "PASS"},
                "capture32_inventory": {
                    "status": inventory_status, "failures": [message]
                },
                "capture_authentication": {"status": "FAIL", "failures": [message]},
                "paired_outcome_recomputation": {"status": "NOT_RUN"},
                "outcome_hidden_crossfit": {"status": "NOT_RUN"},
                "training_evidence": {"status": "NOT_RUN"},
            },
            "crossfit_bundle": None,
            "failures": [message],
        }, None)

    capture_role: str
    try:
        capture = authenticate_capture32(
            manifest, tokenizer_factory=tokenizer_factory
        )
        capture_role = "preregistered_capture32"
        physical = capture.get("physical_gpu_whitelist")
        visible = capture.get("visible_devices")
        if not isinstance(physical, list) or len(physical) != 2 \
                or physical != sorted(set(physical)) \
                or any(type(item) is not int or item < 0 for item in physical) \
                or visible != ",".join(str(item) for item in physical):
            raise ValueError(
                "capture32 authenticated result omitted its dynamic two-H20 binding"
            )
        source_base = {
            **capture32_source_base,
            "physical_gpu_whitelist": physical,
            "visible_devices": visible,
        }
    except Capture32AttritionError as error:
        return capture_failure(
            error,
            source=capture32_source_base,
            decision="PAIRED_EFFECT_NO_GO:CAPTURE32_ATTRITION",
        )
    except CapturePendingError as capture32_pending:
        # capture4 is retained only as a pipeline canary.  It may be audited
        # when capture32 has not started, but it can never satisfy or pad the
        # preregistered 32-example evidence inventory.
        try:
            capture = authenticate_capture(
                manifest, tokenizer_factory=tokenizer_factory
            )
            capture_role = "pipeline_pilot4_only"
            source_base = pilot4_source_base
        except CapturePendingError as pilot4_pending:
            missing = sorted(set(capture32_pending.missing + pilot4_pending.missing))
            report = {
                **common,
                "status": "PENDING",
                "decision": manifest["admissibility"]["pending_decision_missing_capture"],
                "source_capture": {
                    **capture32_source_base,
                    "missing_artifacts": missing,
                    "pilot4_missing_artifacts": pilot4_pending.missing,
                },
                "gates": {
                    "capture32_preregistration": {"status": "PASS"},
                    "capture32_inventory": {
                        "status": "PENDING", "observed": 0, "required": CAPTURE32_COUNT
                    },
                    "capture_authentication": {
                        "status": "PENDING", "missing_artifacts": missing
                    },
                    "paired_outcome_recomputation": {"status": "NOT_RUN"},
                    "outcome_hidden_crossfit": {"status": "NOT_RUN"},
                    "training_evidence": {"status": "NOT_RUN"},
                },
                "crossfit_bundle": None,
            }
            return report, None
        except Exception as error:
            return capture_failure(
                error,
                source=pilot4_source_base,
                decision=manifest["admissibility"]["invalid_decision"],
                inventory_status="PENDING",
            )
    except Exception as error:
        return capture_failure(
            error,
            source=capture32_source_base,
            decision=manifest["admissibility"]["invalid_decision"],
        )

    pairs = capture.pop("pairs")
    capture.pop("preregistration", None)
    try:
        bundle, observations = build_crossfit_bundle(
            pairs,
            fold_count=manifest["scorer"]["fold_count"],
            ridge=manifest["scorer"]["ridge"],
            capture32_preregistration=(
                preregistration if capture_role == "preregistered_capture32" else None
            ),
        )
        expected_fold_assignments = None
        if capture_role == "preregistered_capture32":
            expected_fold_assignments = {
                row["stable_example_id"]: row["crossfit_fold"]
                for row in preregistration["selected_inventory"]
            }
        validate_crossfit_bundle(
            bundle, observations,
            expected_fold_assignments=expected_fold_assignments,
        )
        diagnostics = crossfit_diagnostics(bundle, observations)
        readiness_status, readiness_decision, readiness_gates = _readiness(
            manifest, diagnostics
        )
    except Exception as error:
        return capture_failure(
            error,
            source={**source_base, **capture},
            decision=manifest["admissibility"]["invalid_decision"],
        )

    if capture_role == "pipeline_pilot4_only":
        readiness_gates = {
            "minimum_unique_stable_examples": {
                "status": "PENDING",
                "observed": 0,
                "required": CAPTURE32_COUNT,
                "pilot4_observed_but_excluded": len(observations),
            },
            "predictive_signal": {
                "status": "NOT_APPLICABLE",
                "failures": [],
                "reason": "capture4 is pipeline-only and excluded from capture32",
            },
        }
        anchor = {
            "status": "NOT_APPLICABLE",
            "decision": "CAPTURE4_IS_PIPELINE_PILOT_ONLY",
        }
        status = "PENDING"
        decision = manifest["admissibility"]["pending_decision_pilot4_only"]
    else:
        anchor = external_capture_anchor_state(manifest, capture)
        if readiness_status == "FAIL":
            # A scientific/admissibility failure is stronger than an absent
            # provenance anchor and must stay visible at the top level.
            status, decision = readiness_status, readiness_decision
        elif anchor["status"] != "PASS":
            status = "PENDING"
            decision = manifest["admissibility"]["pending_decision_provenance"]
        else:
            # v1 deliberately cannot emit PASS.  A future reviewed schema can
            # promote this external-review-ready state without changing the
            # frozen scorer result.
            status = "PENDING"
            decision = readiness_decision
    readiness_failures = readiness_gates["predictive_signal"].get("failures", [])
    bundle_path = Path(manifest["paths"]["bundle"])
    bundle_file_sha = _sha256_bytes(_json_bytes(bundle))
    report = {
        **common,
        "status": status,
        "decision": decision,
        "source_capture": {
            **source_base,
            **capture,
        },
        "paired_outcomes": {
            "target_name": TARGET_NAME,
            "observation_count": len(observations),
            "observations_sha256": canonical_sha256(observations),
            "observations": observations,
        },
        "gates": {
            "capture32_preregistration": {"status": "PASS"},
            "capture32_inventory": {
                "status": (
                    "PASS" if capture_role == "preregistered_capture32" else "PENDING"
                ),
                "observed": len(observations) if capture_role == "preregistered_capture32" else 0,
                "required": CAPTURE32_COUNT,
                "capture4_observed_but_excluded": (
                    len(observations) if capture_role == "pipeline_pilot4_only" else 0
                ),
            },
            "capture_internal_consistency": {"status": "PASS"},
            "capture_external_provenance": anchor,
            "capture_authentication": {
                "status": (
                    "PASS" if capture_role == "pipeline_pilot4_only"
                    else ("PASS" if anchor["status"] == "PASS" else "PENDING")
                )
            },
            "paired_outcome_recomputation": {"status": "PASS"},
            "outcome_hidden_crossfit": {
                "status": "PASS",
                "group_key": "stable_example_id",
                "heldout_membership_exclusion_recomputed": True,
                "feature_schema": list(FEATURE_SCHEMA),
                "accepts_persisted_score": False,
                "accepts_runtime_uuid": False,
            },
            "training_evidence": readiness_gates,
        },
        "crossfit_bundle": {
            "path": str(bundle_path),
            "bundle_sha256": bundle["bundle_sha256"],
            "file_sha256": bundle_file_sha,
            "written": artifact_written,
            "diagnostics": diagnostics,
        },
        "failures": list(readiness_failures) if status == "FAIL" else [],
    }
    return report, bundle


def _write_outputs(
    manifest: Mapping[str, Any], report: Mapping[str, Any], bundle: Mapping[str, Any] | None
) -> None:
    output_root = Path(manifest["paths"]["output_root"])
    work_root = Path(manifest["work_root"])
    report_path = Path(manifest["paths"]["report"])
    bundle_path = Path(manifest["paths"]["bundle"])
    for path in (output_root, report_path, bundle_path):
        _assert_confined_new_output(path, work_root)
    if bundle is not None:
        write_json_exclusive(bundle_path, bundle)
        if sha256_file(bundle_path) != report["crossfit_bundle"]["file_sha256"]:
            raise AssertionError("written crossfit bundle digest mismatch")
    write_json_exclusive(report_path, report)


def validate_report_shape(report: Mapping[str, Any]) -> None:
    required = {
        "schema", "status", "decision", "candidate_id", "run_id", "pipeline",
        "source_capture", "gates", "crossfit_bundle", "claim_boundary",
        "training_authorized", "method_selected", "provenance_limitation", "failures",
    }
    if not required.issubset(report):
        raise ValueError(f"paired-effect report is missing fields: {sorted(required - set(report))}")
    if report["schema"] != REPORT_SCHEMA or report["candidate_id"] != CANDIDATE_ID:
        raise ValueError("paired-effect report identity drifted")
    if report["status"] not in {"PASS", "PENDING", "FAIL"}:
        raise ValueError("paired-effect report status is invalid")
    if report["status"] == "PASS":
        raise ValueError(
            "paired-effect report v1 forbids PASS until an external anchor validator is frozen"
        )
    if report["training_authorized"] is not False or report["method_selected"] is not False:
        raise ValueError("paired-effect report crossed the claim firewall")
    pipeline = report["pipeline"]
    pipeline_allowed = {
        "git_commit", "code_sha256", "code_combined_sha256", "manifest",
        "manifest_sha256", "report_schema", "report_schema_sha256",
        "capture32_preregistration", "capture32_preregistration_file_sha256",
        "capture32_preregistration_sha256",
    }
    if set(pipeline) != pipeline_allowed \
            or not FULL_SHA.fullmatch(str(pipeline.get("git_commit", ""))):
        raise ValueError("paired-effect pipeline binding is incomplete")
    code_hashes = pipeline.get("code_sha256")
    if not isinstance(code_hashes, Mapping) \
            or set(code_hashes) != set(PIPELINE_CODE_OBJECTS) \
            or any(not _is_sha(value) for value in code_hashes.values()) \
            or pipeline.get("code_combined_sha256") != canonical_sha256(dict(code_hashes)):
        raise ValueError("paired-effect pipeline code inventory is not authenticated")
    if not _is_sha(pipeline.get("manifest_sha256")) \
            or not _is_sha(pipeline.get("report_schema_sha256")) \
            or not _is_sha(pipeline.get("capture32_preregistration_file_sha256")) \
            or not _is_sha(pipeline.get("capture32_preregistration_sha256")):
        raise ValueError("paired-effect pipeline artifact digest is invalid")
    decisions = {
        "PASS": {"PAIRED_EFFECT_EVIDENCE_READY_FOR_EXTERNAL_REVIEW"},
        "PENDING": {
            "PAIRED_EFFECT_CAPTURE_PENDING",
            "PAIRED_EFFECT_CAPTURE4_PILOT_ONLY",
            "PAIRED_EFFECT_CAPTURE_PROVENANCE_PENDING",
            "PAIRED_EFFECT_MORE_CAPTURE_REQUIRED",
            "PAIRED_EFFECT_EVIDENCE_READY_FOR_EXTERNAL_REVIEW",
        },
        "FAIL": {
            "PAIRED_EFFECT_NO_GO:INVALID_CAPTURE_OR_PIPELINE",
            "PAIRED_EFFECT_NO_GO:CAPTURE32_ATTRITION",
            "PAIRED_EFFECT_NO_GO:SCORER_ADMISSIBILITY",
        },
    }
    if report["decision"] not in decisions[report["status"]]:
        raise ValueError("paired-effect report status/decision mismatch")
    status = report["status"]
    gates = report.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("paired-effect report gates are missing")
    if status == "PASS":  # Unreachable in report v1; retained as the v2 review checklist.
        required_gate_status = {
            "capture_internal_consistency": "PASS",
            "capture_external_provenance": "PASS",
            "capture_authentication": "PASS",
            "paired_outcome_recomputation": "PASS",
            "outcome_hidden_crossfit": "PASS",
        }
        if any(not isinstance(gates.get(name), Mapping)
               or gates[name].get("status") != expected
               for name, expected in required_gate_status.items()):
            raise ValueError("paired-effect PASS has a non-PASS evidence gate")
        training = gates.get("training_evidence")
        if not isinstance(training, Mapping) \
                or training.get("minimum_unique_stable_examples", {}).get("status") != "PASS" \
                or training.get("predictive_signal", {}).get("status") != "PASS":
            raise ValueError("paired-effect PASS has not passed training-evidence gates")
        outcomes = report.get("paired_outcomes")
        if not isinstance(outcomes, Mapping) or outcomes.get("target_name") != TARGET_NAME:
            raise ValueError("paired-effect PASS has no recomputed paired outcomes")
        observations = outcomes.get("observations")
        count = outcomes.get("observation_count")
        if type(count) is not int or count < 32 or not isinstance(observations, list) \
                or len(observations) != count \
                or len({row.get("stable_example_id") for row in observations}) != count \
                or len({row.get("stable_write_id") for row in observations}) != count \
                or len({row.get("pair_id") for row in observations}) != count \
                or outcomes.get("observations_sha256") != canonical_sha256(observations):
            raise ValueError("paired-effect PASS observation inventory is incomplete/duplicated")
        bundle = report.get("crossfit_bundle")
        if not isinstance(bundle, Mapping) \
                or not _is_sha(bundle.get("bundle_sha256")) \
                or not _is_sha(bundle.get("file_sha256")) \
                or bundle.get("diagnostics", {}).get("stable_example_count") != count:
            raise ValueError("paired-effect PASS crossfit bundle is incomplete")
        source = report.get("source_capture")
        artifacts = source.get("artifacts") if isinstance(source, Mapping) else None
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(_capture_paths_from_names()):
            raise ValueError("paired-effect PASS source artifact inventory is incomplete")
        if any(not isinstance(value, Mapping) or not _is_sha(value.get("sha256"))
               for value in artifacts.values()):
            raise ValueError("paired-effect PASS source artifact digest is invalid")
        if report.get("failures") != []:
            raise ValueError("paired-effect PASS contains failures")
    elif status == "PENDING":
        if report["decision"] == "PAIRED_EFFECT_CAPTURE_PENDING":
            if gates.get("capture32_preregistration", {}).get("status") != "PASS" \
                    or gates.get("capture32_inventory", {}).get("status") != "PENDING" \
                    or gates.get("capture_authentication", {}).get("status") != "PENDING" \
                    or report.get("crossfit_bundle") is not None:
                raise ValueError("missing-capture PENDING has contradictory evidence")
        elif report["decision"] == "PAIRED_EFFECT_CAPTURE4_PILOT_ONLY":
            source = report.get("source_capture", {})
            if source.get("capture_role") != "pipeline_pilot4_only" \
                    or gates.get("capture32_inventory", {}).get("status") != "PENDING" \
                    or gates.get("capture_authentication", {}).get("status") != "PASS" \
                    or not isinstance(report.get("crossfit_bundle"), Mapping):
                raise ValueError("capture4 pilot-only PENDING has contradictory evidence")
        elif report["decision"] == "PAIRED_EFFECT_CAPTURE_PROVENANCE_PENDING":
            if report.get("source_capture", {}).get("capture_role") \
                    != "preregistered_capture32" \
                    or gates.get("capture32_inventory", {}).get("status") != "PASS" \
                    or gates.get("capture_internal_consistency", {}).get("status") != "PASS" \
                    or gates.get("capture_external_provenance", {}).get("status") != "PENDING" \
                    or gates.get("capture_authentication", {}).get("status") != "PENDING":
                raise ValueError("provenance PENDING has contradictory capture gates")
        elif report["decision"] == "PAIRED_EFFECT_EVIDENCE_READY_FOR_EXTERNAL_REVIEW":
            training = gates.get("training_evidence", {})
            if gates.get("capture32_inventory", {}).get("status") != "PASS" \
                    or gates.get("capture_external_provenance", {}).get("status") != "PASS" \
                    or gates.get("capture_authentication", {}).get("status") != "PASS" \
                    or training.get("minimum_unique_stable_examples", {}).get("status") != "PASS" \
                    or training.get("predictive_signal", {}).get("status") != "PASS":
                raise ValueError("external-review-ready PENDING lacks passed evidence")
        elif gates.get("capture_authentication", {}).get("status") != "PASS":
            raise ValueError("more-capture PENDING lacks authenticated capture evidence")
    elif report["decision"] == "PAIRED_EFFECT_NO_GO:INVALID_CAPTURE_OR_PIPELINE":
        if gates.get("capture_authentication", {}).get("status") != "FAIL" \
                or not report.get("failures"):
            raise ValueError("invalid-capture FAIL lacks a concrete capture failure")
    elif report["decision"] == "PAIRED_EFFECT_NO_GO:CAPTURE32_ATTRITION":
        if gates.get("capture32_inventory", {}).get("status") != "FAIL" \
                or gates.get("capture_authentication", {}).get("status") != "FAIL" \
                or report.get("crossfit_bundle") is not None \
                or not report.get("failures"):
            raise ValueError("capture32 attrition FAIL lacks exact inventory failure")
    else:
        training = gates.get("training_evidence", {})
        internal = gates.get("capture_internal_consistency", {}).get("status")
        external = gates.get("capture_external_provenance", {}).get("status")
        authentication = gates.get("capture_authentication", {}).get("status")
        capture_state_is_consistent = internal == "PASS" and (
            (external == "PASS" and authentication == "PASS")
            or (external == "PENDING" and authentication == "PENDING")
        )
        if report.get("source_capture", {}).get("capture_role") \
                != "preregistered_capture32" \
                or gates.get("capture32_inventory", {}).get("status") != "PASS" \
                or not capture_state_is_consistent \
                or gates.get("paired_outcome_recomputation", {}).get("status") != "PASS" \
                or gates.get("outcome_hidden_crossfit", {}).get("status") != "PASS" \
                or not isinstance(report.get("crossfit_bundle"), Mapping) \
                or training.get("predictive_signal", {}).get("status") != "FAIL" \
                or not report.get("failures"):
            raise ValueError("scorer-admissibility FAIL lacks its recomputed capture failure")


def _capture_paths_from_names() -> tuple[str, ...]:
    return (
        "p0_certificate", "resolved_manifest", "supervisor_ledger",
        "capture_credential", "capture_ledger", "capture_run_receipt", "final_report",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_REL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    report, bundle = build_report(
        manifest, artifact_written=args.write or args.verify_existing
    )
    validate_report_shape(report)
    if args.write:
        _write_outputs(manifest, report, bundle)
    elif args.verify_existing:
        report_path = Path(manifest["paths"]["report"])
        actual_report = json.loads(report_path.read_text(encoding="utf-8"))
        if canonical_json(actual_report) != canonical_json(report):
            raise ValueError("existing paired-effect report does not reproduce")
        if bundle is not None:
            actual_bundle = json.loads(Path(manifest["paths"]["bundle"]).read_text(encoding="utf-8"))
            if canonical_json(actual_bundle) != canonical_json(bundle):
                raise ValueError("existing paired-effect bundle does not reproduce")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return {"PASS": 0, "PENDING": 2, "FAIL": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
