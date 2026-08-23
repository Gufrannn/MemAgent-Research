"""Frozen helpers for the sealed RWWPO-2 confirmation evaluation."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from recurrent.research.stable_eval_identity import canonical_sha256


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_report(path: str | Path, *, decision: str, commit: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"report is missing or a symlink: {path}")
    row = json.loads(source.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != decision or row.get("git_commit") != commit:
        raise ValueError(f"invalid authenticated report: {path}")
    return {**row, "report_sha256": declared}


def generation_protocol_projection(
    config: Mapping[str, Any], *, repo: str | Path,
    confirmation_data_sha256: str, model: Mapping[str, Any],
) -> dict[str, Any]:
    """Project generation-affecting config with repository-path normalization.

    Absolute evidence/output/checkpoint paths never enter the protocol identity.
    The custom reward source is represented by a repository-relative path and
    its bytes, preventing checkout-location drift without discarding code identity.
    """
    root = Path(repo).resolve()
    reward_path = Path(str(config["custom_reward_function"]["path"])).resolve()
    if root not in reward_path.parents or not reward_path.is_file():
        raise ValueError("custom reward path is outside the exact repository")
    data = config["data"]
    rollout = config["actor_rollout_ref"]["rollout"]
    model_config = config["actor_rollout_ref"]["model"]
    recurrent = config["recurrent"]
    memory_config = recurrent["memory"]["config"]
    effective_prompt_limit = int(memory_config["max_chunks"]) * int(
        memory_config["chunk_size"]
    )
    return {
        "schema_version": "rwwpo2-confirmation-generation-protocol-v1",
        "recurrent": recurrent,
        "data": {
            "validation_sha256": str(confirmation_data_sha256),
            # MemoryDataset deliberately mutates the shared Hydra data config
            # during construction.  Bind both sides of that transition so the
            # producer cannot certify only the pre-construction value.
            "hydra_pre_dataset_max_prompt_length": int(data["max_prompt_length"]),
            "memory_dataset_effective_max_prompt_length": effective_prompt_limit,
            **{key: data[key] for key in (
                "shuffle", "filter_overlong_prompts",
                "filter_overlong_prompts_workers", "dataloader_num_workers",
                "include_source_order_index", "truncation", "context_key",
                "val_max_samples", "max_response_length",
            )},
        },
        "model": {
            "id": model["id"], "revision": model["revision"],
            "use_remove_padding": model_config["use_remove_padding"],
        },
        "rollout": {key: rollout[key] for key in (
            "name", "mode", "n", "tensor_model_parallel_size",
            "dtype", "load_format", "ignore_eos", "enforce_eager",
            "free_cache_engine", "gpu_memory_utilization", "use_fire_sampling",
            "max_num_batched_tokens", "max_num_seqs", "val_kwargs",
        )},
        "reward_manager": config["reward_model"]["reward_manager"],
        "custom_reward_function": {
            "repository_relative_path": str(reward_path.relative_to(root)),
            "path_sha256": sha256_file(reward_path),
            "name": config["custom_reward_function"]["name"],
            "reward_kwargs": config["custom_reward_function"].get("reward_kwargs", {}),
        },
    }


def protocol_sha256(*args, **kwargs) -> str:
    return canonical_sha256(generation_protocol_projection(*args, **kwargs))


def one_sided_exact_paired_sign_flip(centered_differences: Sequence[float]) -> float:
    """Exact one-sided sign-flip p-value, retaining zeros in all assignments."""
    values = tuple(float(value) for value in centered_differences)
    if len(values) != 8:
        raise ValueError("RWWPO-2 confirmation requires exactly eight paired seeds")
    observed = sum(values) / len(values)
    null_statistics = (
        sum(sign * value for sign, value in zip(signs, values)) / len(values)
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    )
    # Inclusive tail makes ties (including centered zero differences) explicit.
    return sum(statistic >= observed - 1e-15 for statistic in null_statistics) / 256.0


def holm_two_test_decisions(pvalues: Mapping[str, float], *, alpha: float = 0.05) -> dict:
    if set(pvalues) != {"B-D", "B-E"}:
        raise ValueError("Holm family must be exactly the two co-primary contrasts")
    ordered = sorted((float(value), name) for name, value in pvalues.items())
    first_pass = ordered[0][0] <= float(alpha) / 2.0
    second_pass = first_pass and ordered[1][0] <= float(alpha)
    return {
        ordered[0][1]: {
            "raw_p": ordered[0][0], "holm_threshold": float(alpha) / 2.0,
            "reject": first_pass,
        },
        ordered[1][1]: {
            "raw_p": ordered[1][0], "holm_threshold": float(alpha),
            "reject": second_pass,
        },
    }
