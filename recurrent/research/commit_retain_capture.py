"""Fail-closed contracts for the base-I COMMIT(C) versus RETAIN(old) capture.

The native :class:`recurrent.impls.memory.MemoryAgent` always replaces its
memory in ``update``.  This module is the deliberately small, trainer-free
intervention adapter used by the H20 capture runner: a writer candidate is
generated once, materialized as canonical bytes, and then two continuations
load either those exact bytes (COMMIT) or the exact pre-write bytes (RETAIN).

Nothing in this module selects a method, computes a training reward, or updates
an actor.  Persisted status flags are never trusted; the auditor rebuilds every
derived identifier, digest, state transition, outcome, and cost receipt.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from recurrent.research.gate_a_execution import validate_jsonl_chain
from recurrent.research.s128_hotpot_metrics import score_terminal_output
from recurrent.research.trajectory_seeding import derive_turn_request_seeds


PAIR_SCHEMA = "memagent.commit-retain.capture-pair.v1"
STATE_ENCODING = "memagent.state-token-u32le.v1"
CAPTURE_RECORD_TYPE = "commit_retain_pair_capture"
ARMS = ("COMMIT", "RETAIN")
SHARED_ARM = "SHARED"
STABLE_FIELDS = (
    "example_id",
    "semantic_dataset_index",
    "source_order_index",
    "raw_row_position",
    "production_effective_position",
    "eval_manifest_hash",
    "source_question_hash",
    "source_context_hash",
    "ground_truth_hash",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def require_int(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be a JSON integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return value


def _tokens(value: Any, field: str, *, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a token-id array")
    result = [
        require_int(item, f"{field}[{index}]", minimum=0, maximum=(1 << 32) - 1)
        for index, item in enumerate(value)
    ]
    if not allow_empty and not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _sampling(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    result = dict(value)
    allowed = {
        "temperature", "top_p", "top_k", "min_p", "n", "best_of", "max_tokens",
    }
    extra = sorted(set(result) - allowed)
    if extra:
        raise ValueError(f"{field} contains unfrozen keys: {extra}")
    required = {"temperature", "top_p", "top_k", "min_p", "n", "best_of", "max_tokens"}
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"{field} is missing keys: {missing}")
    for name in ("temperature", "top_p", "min_p"):
        raw = result[name]
        if type(raw) not in (int, float) or not math.isfinite(float(raw)):
            raise ValueError(f"{field}.{name} must be finite")
        result[name] = float(raw)
    if result["temperature"] < 0 or not 0 <= result["top_p"] <= 1 or not 0 <= result["min_p"] <= 1:
        raise ValueError(f"{field} probability/temperature bounds failed")
    result["top_k"] = require_int(result["top_k"], f"{field}.top_k", minimum=-1)
    result["n"] = require_int(result["n"], f"{field}.n", minimum=1)
    result["best_of"] = require_int(result["best_of"], f"{field}.best_of", minimum=1)
    result["max_tokens"] = require_int(
        result["max_tokens"], f"{field}.max_tokens", minimum=1
    )
    if result["n"] != 1 or result["best_of"] != 1:
        raise ValueError(f"{field} must be strict single-output decoding")
    return result


def build_state_blob(value: Mapping[str, Any] | Sequence[int]) -> dict[str, Any]:
    """Return the only admissible byte representation of a memory state.

    Token IDs are unsigned 32-bit little-endian integers.  This makes an exact
    RETAIN check independent of Python container identity, tensor dtype, JSON
    whitespace, or a later tokenizer decode/encode round trip.
    """
    raw_ids: Any = value.get("token_ids") if isinstance(value, Mapping) else list(value)
    ids = _tokens(raw_ids, "state.token_ids")
    encoded = b"".join(struct.pack("<I", token_id) for token_id in ids)
    return {
        "encoding": STATE_ENCODING,
        "token_count": len(ids),
        "token_ids": ids,
        "token_ids_sha256": canonical_sha256(ids),
        "byte_length": len(encoded),
        "bytes_b64": base64.b64encode(encoded).decode("ascii"),
        "bytes_sha256": _sha256_bytes(encoded),
    }


def validate_state_blob(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("state blob must be an object")
    rebuilt = build_state_blob(value)
    if canonical_json(rebuilt) != canonical_json(dict(value)):
        raise ValueError("state blob bytes/token digests are non-canonical or stale")
    try:
        decoded = base64.b64decode(value["bytes_b64"], validate=True)
    except Exception as error:
        raise ValueError("state bytes are not strict base64") from error
    if base64.b64encode(decoded).decode("ascii") != value["bytes_b64"]:
        raise ValueError("state bytes use a non-canonical base64 spelling")
    return rebuilt


def _stable_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in STABLE_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"stable identity is missing: {missing}")
    result = {field: payload[field] for field in STABLE_FIELDS}
    result["example_id"] = str(result["example_id"])
    for field in (
        "semantic_dataset_index", "source_order_index", "raw_row_position",
        "production_effective_position",
    ):
        result[field] = require_int(result[field], field, minimum=0)
    if result["example_id"] != str(result["semantic_dataset_index"]):
        raise ValueError("example_id differs from semantic_dataset_index")
    for field in (
        "eval_manifest_hash", "source_question_hash", "source_context_hash",
        "ground_truth_hash",
    ):
        result[field] = require_sha256(result[field], field)
    return result


def stable_capture_ids(
    identity: Mapping[str, Any], *, trajectory_seed: int, writer_turn: int
) -> dict[str, str]:
    checked = _stable_identity(identity)
    seed = require_int(trajectory_seed, "trajectory_seed", minimum=0)
    turn = require_int(writer_turn, "writer_turn", minimum=0)
    stable_example_id = canonical_sha256(
        {
            "namespace": "memagent.commit-retain.example.v1",
            **checked,
        }
    )
    stable_root_id = canonical_sha256(
        {
            "namespace": "memagent.commit-retain.root.v1",
            "stable_example_id": stable_example_id,
            "trajectory_seed": seed,
            "replica_id": 0,
        }
    )
    stable_write_id = canonical_sha256(
        {
            "namespace": "memagent.commit-retain.write.v1",
            "stable_root_id": stable_root_id,
            "writer_turn": turn,
        }
    )
    return {
        "stable_example_id": stable_example_id,
        "stable_root_id": stable_root_id,
        "stable_write_id": stable_write_id,
    }


def stable_turn_id(
    *, stable_write_id: str, phase: str, arm: str, writer_turn: int
) -> str:
    require_sha256(stable_write_id, "stable_write_id")
    if phase not in {"prefix_writer", "candidate_writer", "future_writer", "final_reader"}:
        raise ValueError(f"unsupported capture phase: {phase}")
    if arm not in {*ARMS, SHARED_ARM}:
        raise ValueError(f"unsupported arm: {arm}")
    if phase in {"prefix_writer", "candidate_writer"} and arm != SHARED_ARM:
        raise ValueError("prefix/candidate turns must be shared before branching")
    if phase in {"future_writer", "final_reader"} and arm not in ARMS:
        raise ValueError("post-branch turns must name a real arm")
    turn = require_int(writer_turn, "writer_turn", minimum=0)
    return canonical_sha256(
        {
            "namespace": "memagent.commit-retain.turn.v1",
            "stable_write_id": stable_write_id,
            "phase": phase,
            "arm": arm,
            "writer_turn": turn,
        }
    )


def _loaded_state_receipt(
    value: Mapping[str, Any], *, field: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    source_role = value.get("source_role")
    if source_role not in {
        "no_memory", "previous_prefix_output", "old_state", "candidate",
        "previous_future_output",
    }:
        raise ValueError(f"{field}.source_role is invalid")
    source_turn_id = value.get("source_turn_id")
    if source_role == "no_memory":
        if source_turn_id is not None:
            raise ValueError(f"{field}.source_turn_id must be null for no_memory")
    else:
        source_turn_id = require_sha256(source_turn_id, f"{field}.source_turn_id")
    return {
        "source_role": source_role,
        "source_turn_id": source_turn_id,
        "state": build_state_blob(value.get("state", {})),
    }


def _prompt_receipt(
    value: Mapping[str, Any], *, field: str, writer: bool
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    text = value.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{field}.text must be non-empty")
    ids = _tokens(value.get("token_ids"), f"{field}.token_ids", allow_empty=False)
    result = {
        "text": text,
        "text_sha256": _sha256_text(text),
        "token_ids": ids,
        "token_ids_sha256": canonical_sha256(ids),
        "template_sha256": require_sha256(
            value.get("template_sha256"), f"{field}.template_sha256"
        ),
        "checkpoint_sha256": require_sha256(
            value.get("checkpoint_sha256"), f"{field}.checkpoint_sha256"
        ),
        "loaded_state_receipt": _loaded_state_receipt(
            value.get("loaded_state_receipt", {}),
            field=f"{field}.loaded_state_receipt",
        ),
    }
    if writer:
        chunk_ids = _tokens(value.get("chunk_token_ids"), f"{field}.chunk_token_ids", allow_empty=False)
        result.update(
            chunk_token_ids=chunk_ids,
            chunk_token_ids_sha256=canonical_sha256(chunk_ids),
        )
    elif "chunk_token_ids" in value or "chunk_token_ids_sha256" in value:
        raise ValueError(f"{field} final reader prompt must not contain a context chunk")
    return result


def _writer_generation(
    value: Mapping[str, Any],
    *,
    field: str,
    stable_write_id_value: str,
    expected_phase: str,
    expected_arm: str,
    expected_turn: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    raw_ids = _tokens(value.get("raw_completion_token_ids"), f"{field}.raw_completion_token_ids")
    eos_id = require_int(value.get("eos_token_id"), f"{field}.eos_token_id", minimum=0)
    removed_positions = [index for index, token_id in enumerate(raw_ids) if token_id == eos_id]
    if value.get("eos_token_positions_removed") != removed_positions:
        raise ValueError(f"{field} EOS removal positions differ from native unpad semantics")
    if value.get("eos_removal_semantics") != "remove_all_eos_matching_native_unpad":
        raise ValueError(f"{field} EOS removal semantics drifted")
    expected_state_ids = [token_id for token_id in raw_ids if token_id != eos_id]
    state_after = build_state_blob(value.get("state_after", {}))
    if state_after["token_ids"] != expected_state_ids:
        raise ValueError(f"{field} state_after is not the exact post-EOS writer output")
    output_text = value.get("output_text")
    if not isinstance(output_text, str):
        raise ValueError(f"{field}.output_text must be text")
    request_seed = require_int(value.get("request_seed"), f"{field}.request_seed", minimum=0)
    configured_seed = require_int(
        value.get("configured_request_seed"), f"{field}.configured_request_seed", minimum=0
    )
    actual_seed = require_int(
        value.get("actual_request_seed"), f"{field}.actual_request_seed", minimum=0
    )
    if configured_seed != request_seed or actual_seed != request_seed:
        raise ValueError(f"{field} configured/actual RNG seed mismatch")
    expected_id = stable_turn_id(
        stable_write_id=stable_write_id_value,
        phase=expected_phase,
        arm=expected_arm,
        writer_turn=expected_turn,
    )
    return {
        "phase": expected_phase,
        "arm": expected_arm,
        "writer_turn": expected_turn,
        "stable_turn_id": expected_id,
        "prompt": _prompt_receipt(value.get("prompt", {}), field=f"{field}.prompt", writer=True),
        "sampling_params": _sampling(value.get("sampling_params"), f"{field}.sampling_params"),
        "request_seed": request_seed,
        "configured_request_seed": configured_seed,
        "actual_request_seed": actual_seed,
        "generate_call_index": require_int(
            value.get("generate_call_index"), f"{field}.generate_call_index", minimum=1
        ),
        "raw_completion_token_ids": raw_ids,
        "raw_completion_token_ids_sha256": canonical_sha256(raw_ids),
        "eos_token_id": eos_id,
        "eos_token_positions_removed": removed_positions,
        "eos_removal_semantics": "remove_all_eos_matching_native_unpad",
        "state_after": state_after,
        "output_text": output_text,
        "output_text_sha256": _sha256_text(output_text),
    }


def _final_generation(
    value: Mapping[str, Any],
    *,
    field: str,
    stable_write_id_value: str,
    arm: str,
    total_writer_turns: int,
    ground_truth: list[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    ids = _tokens(value.get("output_token_ids"), f"{field}.output_token_ids")
    text = value.get("output_text")
    if not isinstance(text, str):
        raise ValueError(f"{field}.output_text must be text")
    request_seed = require_int(value.get("request_seed"), f"{field}.request_seed", minimum=0)
    configured_seed = require_int(
        value.get("configured_request_seed"), f"{field}.configured_request_seed", minimum=0
    )
    actual_seed = require_int(
        value.get("actual_request_seed"), f"{field}.actual_request_seed", minimum=0
    )
    if request_seed != configured_seed or request_seed != actual_seed:
        raise ValueError(f"{field} configured/actual RNG seed mismatch")
    outcome = score_terminal_output(text, ground_truth)
    return {
        "phase": "final_reader",
        "arm": arm,
        "writer_turn": total_writer_turns,
        "stable_turn_id": stable_turn_id(
            stable_write_id=stable_write_id_value,
            phase="final_reader",
            arm=arm,
            writer_turn=total_writer_turns,
        ),
        "prompt": _prompt_receipt(value.get("prompt", {}), field=f"{field}.prompt", writer=False),
        "sampling_params": _sampling(value.get("sampling_params"), f"{field}.sampling_params"),
        "request_seed": request_seed,
        "configured_request_seed": configured_seed,
        "actual_request_seed": actual_seed,
        "generate_call_index": require_int(
            value.get("generate_call_index"), f"{field}.generate_call_index", minimum=1
        ),
        "output_token_ids": ids,
        "output_token_ids_sha256": canonical_sha256(ids),
        "output_text": text,
        "output_text_sha256": _sha256_text(text),
        "outcome": outcome,
    }


def _shared_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("shared_contract must be an object")
    writer_turn = require_int(value.get("intervention_writer_turn"), "shared_contract.intervention_writer_turn", minimum=1)
    total = require_int(value.get("total_writer_turns"), "shared_contract.total_writer_turns", minimum=2)
    if writer_turn >= total - 0:
        raise ValueError("intervention writer turn must precede the final reader")
    expected_future_turns = list(range(writer_turn + 1, total))
    if not expected_future_turns:
        raise ValueError("capture requires at least one shared future writer chunk")
    raw_chunks = value.get("future_chunks")
    if not isinstance(raw_chunks, list) or len(raw_chunks) != len(expected_future_turns):
        raise ValueError("shared future chunk inventory is incomplete")
    chunks: list[dict[str, Any]] = []
    for expected_turn, raw in zip(expected_future_turns, raw_chunks):
        if not isinstance(raw, Mapping):
            raise ValueError("future chunk receipt must be an object")
        if require_int(raw.get("writer_turn"), "future_chunk.writer_turn", minimum=0) != expected_turn:
            raise ValueError("future chunk writer turns are not contiguous")
        ids = _tokens(raw.get("token_ids"), "future_chunk.token_ids", allow_empty=False)
        chunks.append(
            {
                "writer_turn": expected_turn,
                "token_ids": ids,
                "token_ids_sha256": canonical_sha256(ids),
            }
        )
    writer_decode = _sampling(value.get("writer_decode"), "shared_contract.writer_decode")
    reader_decode = _sampling(value.get("reader_decode"), "shared_contract.reader_decode")
    if reader_decode["temperature"] != 0.0:
        raise ValueError("capture final reader must be deterministic temperature zero")
    trajectory_seed = require_int(value.get("trajectory_seed"), "shared_contract.trajectory_seed", minimum=0)
    expected_writer_seeds = [
        {
            "writer_turn": turn,
            "request_seed": derive_turn_request_seeds([trajectory_seed], [0], turn)[0],
        }
        for turn in expected_future_turns
    ]
    reader_seed = derive_turn_request_seeds([trajectory_seed], [0], total)[0]
    cache_contract = {
        "enable_prefix_caching": False,
        "max_num_seqs": 1,
        "one_prompt_per_generate_call": True,
        "kv_state_reuse_across_generate_calls": False,
        "same_engine_for_both_arms": True,
    }
    if canonical_json(value.get("cache_contract")) != canonical_json(cache_contract):
        raise ValueError("shared cache contract drifted")
    horizon = {
        "future_writer_turns": expected_future_turns,
        "future_writer_calls_per_arm": len(expected_future_turns),
        "final_reader_calls_per_arm": 1,
        "terminal_writer_turn": total - 1,
    }
    if canonical_json(value.get("horizon")) != canonical_json(horizon):
        raise ValueError("shared future horizon drifted")
    cost = {
        "shared_candidate_generation_calls": 1,
        "per_arm_writer_generation_calls": len(expected_future_turns),
        "per_arm_reader_generation_calls": 1,
        "per_arm_total_generation_calls": len(expected_future_turns) + 1,
        "per_arm_writer_max_tokens": writer_decode["max_tokens"],
        "per_arm_reader_max_tokens": reader_decode["max_tokens"],
        "budgets_identical_by_design": True,
        "realized_token_counts_are_measured_not_forced_equal": True,
    }
    if canonical_json(value.get("cost_contract")) != canonical_json(cost):
        raise ValueError("shared cost contract drifted")
    result = {
        "intervention_writer_turn": writer_turn,
        "total_writer_turns": total,
        "trajectory_seed": trajectory_seed,
        "future_chunks": chunks,
        "future_chunks_sha256": canonical_sha256(chunks),
        "horizon": horizon,
        "writer_checkpoint_sha256": require_sha256(
            value.get("writer_checkpoint_sha256"), "shared_contract.writer_checkpoint_sha256"
        ),
        "reader_checkpoint_sha256": require_sha256(
            value.get("reader_checkpoint_sha256"), "shared_contract.reader_checkpoint_sha256"
        ),
        "writer_prompt_template_sha256": require_sha256(
            value.get("writer_prompt_template_sha256"),
            "shared_contract.writer_prompt_template_sha256",
        ),
        "reader_prompt_template_sha256": require_sha256(
            value.get("reader_prompt_template_sha256"),
            "shared_contract.reader_prompt_template_sha256",
        ),
        "writer_decode": writer_decode,
        "reader_decode": reader_decode,
        "future_writer_request_seeds": expected_writer_seeds,
        "reader_request_seed": reader_seed,
        "cache_contract": cache_contract,
        "cost_contract": cost,
    }
    return result


def _execution(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("execution must be an object")
    fixed = {
        "backend": "vllm",
        "vllm_version": "0.8.2",
        "strict_vllm": True,
        "tensor_parallel_size": 2,
        "cuda_device_order": "PCI_BUS_ID",
        "worker_multiproc_method": "spawn",
        "vllm_observed_worker_multiproc_method": "spawn",
        "multiprocessing_context_method": "spawn",
        "parent_cuda_initialization_policy": "record_observed_spawn_required",
        "prefix_cache_enabled": False,
        "max_num_seqs": 1,
        "one_prompt_per_generate_call": True,
        "engine_construction_count": 1,
        "full_model_sha_verified_at_capture_start": True,
        "trainer_attached": False,
        "actor_training_calls": 0,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected or type(value.get(field)) is not type(expected):
            raise ValueError(f"execution.{field} differs from strict capture contract")
    physical_whitelist = value.get("physical_gpu_whitelist")
    visible_devices = value.get("visible_devices")
    allowed_gpu_bindings = {
        (4, 5): "4,5",
        (6, 7): "6,7",
    }
    if (
        not isinstance(physical_whitelist, list)
        or tuple(physical_whitelist) not in allowed_gpu_bindings
        or visible_devices != allowed_gpu_bindings[tuple(physical_whitelist)]
    ):
        raise ValueError("execution physical/visible GPU binding is not preregistered")
    parent_cuda_initialized = value.get("parent_cuda_initialized_before_engine")
    if type(parent_cuda_initialized) is not bool:
        raise ValueError("execution parent CUDA initialization observation is not boolean")
    identities = value.get("physical_gpu_identity")
    if (
        not isinstance(identities, list)
        or len(identities) != 2
        or any(not isinstance(item, str) or not item for item in identities)
    ):
        raise ValueError("execution.physical_gpu_identity must bind two devices")
    try:
        identity_indices = [int(item.split(",", 1)[0].strip()) for item in identities]
    except (ValueError, IndexError) as error:
        raise ValueError(
            "execution.physical_gpu_identity has invalid physical indices"
        ) from error
    if identity_indices != physical_whitelist:
        raise ValueError(
            "execution.physical_gpu_identity indices differ from physical whitelist"
        )
    process_uuid = str(value.get("process_instance_uuid"))
    try:
        uuid.UUID(process_uuid)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("execution.process_instance_uuid is invalid") from error
    result = {
        **fixed,
        "physical_gpu_whitelist": physical_whitelist,
        "visible_devices": visible_devices,
        "parent_cuda_initialized_before_engine": parent_cuda_initialized,
        "physical_gpu_identity": identities,
        "engine_id": str(value.get("engine_id")),
        "cache_namespace": str(value.get("cache_namespace")),
        "process_instance_uuid": process_uuid,
        "process_pid": require_int(value.get("process_pid"), "execution.process_pid", minimum=1),
        "global_generate_call_count": require_int(
            value.get("global_generate_call_count"),
            "execution.global_generate_call_count",
            minimum=1,
        ),
        "engine_config_sha256": require_sha256(
            value.get("engine_config_sha256"), "execution.engine_config_sha256"
        ),
        "parent_credential_id": require_sha256(
            value.get("parent_credential_id"), "execution.parent_credential_id"
        ),
        "parent_credential_sha256": require_sha256(
            value.get("parent_credential_sha256"), "execution.parent_credential_sha256"
        ),
        "parent_credential_path": str(value.get("parent_credential_path")),
        "parent_issuer_pid": require_int(
            value.get("parent_issuer_pid"), "execution.parent_issuer_pid", minimum=1
        ),
        "observed_parent_pid": require_int(
            value.get("observed_parent_pid"), "execution.observed_parent_pid", minimum=1
        ),
        "parent_authorization_record_sha256": require_sha256(
            value.get("parent_authorization_record_sha256"),
            "execution.parent_authorization_record_sha256",
        ),
    }
    if not result["engine_id"] or not result["cache_namespace"]:
        raise ValueError("execution engine/cache identity is empty")
    if not result["parent_credential_path"] or result["parent_credential_path"] == "None":
        raise ValueError("execution parent credential path is empty")
    if result["observed_parent_pid"] != result["parent_issuer_pid"]:
        raise ValueError("execution parent process credential mismatch")
    return result


def _assert_state_equal(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    if canonical_json(left) != canonical_json(right):
        raise ValueError(f"{label} state bytes/tokens differ")


def _assert_prompt_contract(
    generation: Mapping[str, Any],
    *,
    loaded_state: Mapping[str, Any],
    checkpoint_sha256: str,
    template_sha256: str,
    sampling: Mapping[str, Any],
    seed: int,
    chunk: Mapping[str, Any] | None,
    label: str,
) -> None:
    prompt = generation["prompt"]
    _assert_state_equal(prompt["loaded_state_receipt"]["state"], loaded_state, label)
    if prompt["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"{label} checkpoint drift")
    if prompt["template_sha256"] != template_sha256:
        raise ValueError(f"{label} prompt template drift")
    if generation["sampling_params"] != sampling:
        raise ValueError(f"{label} decode drift")
    if generation["request_seed"] != seed:
        raise ValueError(f"{label} RNG drift")
    if chunk is not None:
        if prompt["chunk_token_ids"] != chunk["token_ids"]:
            raise ValueError(f"{label} future chunk drift")
        if prompt["chunk_token_ids_sha256"] != chunk["token_ids_sha256"]:
            raise ValueError(f"{label} future chunk hash drift")


def _actual_cost(future: list[dict[str, Any]], final: Mapping[str, Any]) -> dict[str, Any]:
    generations = [*future, final]
    return {
        "writer_generation_calls": len(future),
        "reader_generation_calls": 1,
        "total_generation_calls": len(generations),
        "writer_configured_token_budget": sum(
            int(item["sampling_params"]["max_tokens"]) for item in future
        ),
        "reader_configured_token_budget": int(final["sampling_params"]["max_tokens"]),
        "prompt_tokens_total": sum(len(item["prompt"]["token_ids"]) for item in generations),
        "completion_tokens_total": sum(
            len(item["raw_completion_token_ids"]) for item in future
        ) + len(final["output_token_ids"]),
        "post_eos_state_tokens_total": sum(
            len(item["state_after"]["token_ids"]) for item in future
        ),
        "shared_candidate_cost_excluded": True,
    }


def build_pair_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and substantively validate one paired intervention capture."""
    if not isinstance(payload, Mapping):
        raise ValueError("pair payload must be an object")
    identity = _stable_identity(payload)
    trajectory_seed = require_int(payload.get("trajectory_seed"), "trajectory_seed", minimum=0)
    intervention = require_int(
        payload.get("intervention_writer_turn"), "intervention_writer_turn", minimum=1
    )
    total_turns = require_int(payload.get("total_writer_turns"), "total_writer_turns", minimum=2)
    ids = stable_capture_ids(identity, trajectory_seed=trajectory_seed, writer_turn=intervention)
    ground_truth_raw = payload.get("ground_truth")
    if not isinstance(ground_truth_raw, list) or not ground_truth_raw:
        raise ValueError("ground_truth must be a non-empty array")
    ground_truth = [str(item) for item in ground_truth_raw]
    if canonical_sha256(ground_truth) != identity["ground_truth_hash"]:
        raise ValueError("ground truth differs from stable identity")
    question_ids = _tokens(payload.get("question_token_ids"), "question_token_ids", allow_empty=False)
    no_memory = build_state_blob(payload.get("no_memory_state", {}))
    shared = _shared_contract(payload.get("shared_contract", {}))
    if (
        shared["trajectory_seed"] != trajectory_seed
        or shared["intervention_writer_turn"] != intervention
        or shared["total_writer_turns"] != total_turns
    ):
        raise ValueError("pair schedule differs from shared contract")

    prefix_raw = payload.get("prefix_turns")
    if not isinstance(prefix_raw, list) or len(prefix_raw) != intervention:
        raise ValueError("prefix writer ledger is incomplete")
    prefix: list[dict[str, Any]] = []
    previous_state = no_memory
    previous_turn_id: str | None = None
    call_indices: list[int] = []
    for turn, raw in enumerate(prefix_raw):
        generation = _writer_generation(
            raw,
            field=f"prefix_turns[{turn}]",
            stable_write_id_value=ids["stable_write_id"],
            expected_phase="prefix_writer",
            expected_arm=SHARED_ARM,
            expected_turn=turn,
        )
        expected_seed = derive_turn_request_seeds([trajectory_seed], [0], turn)[0]
        _assert_prompt_contract(
            generation,
            loaded_state=previous_state,
            checkpoint_sha256=shared["writer_checkpoint_sha256"],
            template_sha256=shared["writer_prompt_template_sha256"],
            sampling=shared["writer_decode"],
            seed=expected_seed,
            chunk=None,
            label=f"prefix turn {turn}",
        )
        receipt = generation["prompt"]["loaded_state_receipt"]
        expected_role = "no_memory" if turn == 0 else "previous_prefix_output"
        if receipt["source_role"] != expected_role or receipt["source_turn_id"] != previous_turn_id:
            raise ValueError(f"prefix turn {turn} loaded-state provenance drift")
        prefix.append(generation)
        previous_state = generation["state_after"]
        previous_turn_id = generation["stable_turn_id"]
        call_indices.append(generation["generate_call_index"])
    old_state = build_state_blob(payload.get("old_state", {}))
    _assert_state_equal(old_state, previous_state, "old-state/prefix terminal")

    candidate = _writer_generation(
        payload.get("candidate", {}),
        field="candidate",
        stable_write_id_value=ids["stable_write_id"],
        expected_phase="candidate_writer",
        expected_arm=SHARED_ARM,
        expected_turn=intervention,
    )
    candidate_seed = derive_turn_request_seeds([trajectory_seed], [0], intervention)[0]
    _assert_prompt_contract(
        candidate,
        loaded_state=old_state,
        checkpoint_sha256=shared["writer_checkpoint_sha256"],
        template_sha256=shared["writer_prompt_template_sha256"],
        sampling=shared["writer_decode"],
        seed=candidate_seed,
        chunk=None,
        label="shared candidate",
    )
    candidate_loaded = candidate["prompt"]["loaded_state_receipt"]
    if candidate_loaded["source_role"] != "old_state" or candidate_loaded["source_turn_id"] != previous_turn_id:
        raise ValueError("candidate did not load the exact old state")
    candidate_state = candidate["state_after"]
    if candidate_state["bytes_sha256"] == old_state["bytes_sha256"]:
        raise ValueError("candidate bytes equal old-state bytes; contrast is non-operative")
    call_indices.append(candidate["generate_call_index"])

    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, Mapping) or set(raw_arms) != set(ARMS):
        raise ValueError("paired capture must contain exactly COMMIT and RETAIN arms")
    arms: dict[str, dict[str, Any]] = {}
    chunks_by_turn = {item["writer_turn"]: item for item in shared["future_chunks"]}
    seeds_by_turn = {
        item["writer_turn"]: item["request_seed"]
        for item in shared["future_writer_request_seeds"]
    }
    for arm in ARMS:
        raw_arm = raw_arms[arm]
        if not isinstance(raw_arm, Mapping):
            raise ValueError(f"arm {arm} must be an object")
        source_state = candidate_state if arm == "COMMIT" else old_state
        source_role = "candidate" if arm == "COMMIT" else "old_state"
        source_turn = candidate["stable_turn_id"] if arm == "COMMIT" else previous_turn_id
        initial = _loaded_state_receipt(
            raw_arm.get("initial_loaded_state_receipt", {}),
            field=f"arms.{arm}.initial_loaded_state_receipt",
        )
        if initial["source_role"] != source_role or initial["source_turn_id"] != source_turn:
            raise ValueError(f"arm {arm} initial source role/turn is wrong (arm swap)")
        _assert_state_equal(initial["state"], source_state, f"arm {arm} initial load")
        future_raw = raw_arm.get("future_turns")
        expected_turns = shared["horizon"]["future_writer_turns"]
        if not isinstance(future_raw, list) or len(future_raw) != len(expected_turns):
            raise ValueError(f"arm {arm} future-turn attrition")
        future: list[dict[str, Any]] = []
        loaded_state = source_state
        loaded_turn_id = source_turn
        for offset, (turn, raw) in enumerate(zip(expected_turns, future_raw)):
            generation = _writer_generation(
                raw,
                field=f"arms.{arm}.future_turns[{offset}]",
                stable_write_id_value=ids["stable_write_id"],
                expected_phase="future_writer",
                expected_arm=arm,
                expected_turn=turn,
            )
            _assert_prompt_contract(
                generation,
                loaded_state=loaded_state,
                checkpoint_sha256=shared["writer_checkpoint_sha256"],
                template_sha256=shared["writer_prompt_template_sha256"],
                sampling=shared["writer_decode"],
                seed=seeds_by_turn[turn],
                chunk=chunks_by_turn[turn],
                label=f"arm {arm} future turn {turn}",
            )
            receipt = generation["prompt"]["loaded_state_receipt"]
            expected_role = source_role if offset == 0 else "previous_future_output"
            if receipt["source_role"] != expected_role or receipt["source_turn_id"] != loaded_turn_id:
                raise ValueError(f"arm {arm} future turn {turn} state provenance drift")
            future.append(generation)
            loaded_state = generation["state_after"]
            loaded_turn_id = generation["stable_turn_id"]
            call_indices.append(generation["generate_call_index"])
        final = _final_generation(
            raw_arm.get("final_reader", {}),
            field=f"arms.{arm}.final_reader",
            stable_write_id_value=ids["stable_write_id"],
            arm=arm,
            total_writer_turns=total_turns,
            ground_truth=ground_truth,
        )
        _assert_prompt_contract(
            final,
            loaded_state=loaded_state,
            checkpoint_sha256=shared["reader_checkpoint_sha256"],
            template_sha256=shared["reader_prompt_template_sha256"],
            sampling=shared["reader_decode"],
            seed=shared["reader_request_seed"],
            chunk=None,
            label=f"arm {arm} final reader",
        )
        final_loaded = final["prompt"]["loaded_state_receipt"]
        if final_loaded["source_role"] != "previous_future_output" or final_loaded["source_turn_id"] != loaded_turn_id:
            raise ValueError(f"arm {arm} final reader loaded-state provenance drift")
        call_indices.append(final["generate_call_index"])
        cost = _actual_cost(future, final)
        expected_cost = shared["cost_contract"]
        if (
            cost["writer_generation_calls"] != expected_cost["per_arm_writer_generation_calls"]
            or cost["reader_generation_calls"] != expected_cost["per_arm_reader_generation_calls"]
            or cost["total_generation_calls"] != expected_cost["per_arm_total_generation_calls"]
            or cost["writer_configured_token_budget"]
            != expected_cost["per_arm_writer_generation_calls"]
            * expected_cost["per_arm_writer_max_tokens"]
            or cost["reader_configured_token_budget"]
            != expected_cost["per_arm_reader_max_tokens"]
        ):
            raise ValueError(f"arm {arm} cost receipt differs from the shared budget")
        arms[arm] = {
            "arm": arm,
            "initial_loaded_state_receipt": initial,
            "future_turns": future,
            "final_reader": final,
            "actual_cost_receipt": cost,
        }

    candidate_index = candidate["generate_call_index"]
    post_branch_indices = [
        item["generate_call_index"]
        for arm in ARMS
        for item in [*arms[arm]["future_turns"], arms[arm]["final_reader"]]
    ]
    if not post_branch_indices or candidate_index >= min(post_branch_indices):
        raise ValueError("candidate was not materialized before either arm started")
    if len(call_indices) != len(set(call_indices)):
        raise ValueError("generate-call indices are not unique")
    if call_indices != sorted(call_indices):
        raise ValueError("pair generation receipts are not in the frozen COMMIT-then-RETAIN order")
    expected_pair_calls = intervention + 1 + 2 * (total_turns - intervention)
    if len(call_indices) != expected_pair_calls:
        raise ValueError("pair generate-call count differs from the frozen horizon")

    execution = _execution(payload.get("execution", {}))
    if max(call_indices) > execution["global_generate_call_count"]:
        raise ValueError("pair call index exceeds process generate-call count")
    record = {
        "schema": PAIR_SCHEMA,
        "record_type": "commit_retain_pair",
        **identity,
        **ids,
        "trajectory_seed": trajectory_seed,
        "intervention_writer_turn": intervention,
        "total_writer_turns": total_turns,
        "question_token_ids": question_ids,
        "question_token_ids_sha256": canonical_sha256(question_ids),
        "ground_truth": ground_truth,
        "no_memory_state": no_memory,
        "prefix_turns": prefix,
        "old_state": old_state,
        "candidate_generation_count": 1,
        "candidate_materialized_before_arm_start": True,
        "candidate": candidate,
        "shared_contract": shared,
        "shared_contract_sha256": canonical_sha256(shared),
        "arm_execution_order": list(ARMS),
        "arms": arms,
        "pair_generate_call_indices": call_indices,
        "pair_generate_call_count": len(call_indices),
        "execution": execution,
        "training": {
            "trainer_attached": False,
            "actor_updates": 0,
            "optimizer_steps": 0,
            "base_i_checkpoint_only": True,
        },
        "claim_boundary": {
            "capture_and_audit_only": True,
            "method_selected": False,
            "training_authorized": False,
            "causal_or_performance_claim": False,
        },
    }
    record["pair_id"] = canonical_sha256(record)
    return record


def _decode_checks(record: Mapping[str, Any], decoder: Callable[[list[int]], str]) -> None:
    generations: list[Mapping[str, Any]] = [*record["prefix_turns"], record["candidate"]]
    for arm in ARMS:
        generations.extend(record["arms"][arm]["future_turns"])
        final = record["arms"][arm]["final_reader"]
        if decoder(list(final["output_token_ids"])) != final["output_text"]:
            raise ValueError(f"arm {arm} final text differs from tokenizer decode")
        if decoder(list(final["prompt"]["token_ids"])) != final["prompt"]["text"]:
            raise ValueError(f"arm {arm} final prompt text differs from tokenizer decode")
    for generation in generations:
        if decoder(list(generation["state_after"]["token_ids"])) != generation["output_text"]:
            raise ValueError(
                f"{generation['stable_turn_id']} writer text differs from tokenizer decode"
            )
        if decoder(list(generation["prompt"]["token_ids"])) != generation["prompt"]["text"]:
            raise ValueError(
                f"{generation['stable_turn_id']} writer prompt text differs from tokenizer decode"
            )


def validate_pair_record(
    record: Mapping[str, Any], *, decoder: Callable[[list[int]], str] | None = None
) -> dict[str, Any]:
    rebuilt = build_pair_record(record)
    if canonical_json(rebuilt) != canonical_json(dict(record)):
        raise ValueError("pair record is non-canonical, has extra fields, or has stale evidence")
    if decoder is not None:
        _decode_checks(rebuilt, decoder)
    return rebuilt


def build_capture_envelope(
    pair: Mapping[str, Any],
    *,
    experiment_name: str,
    git_commit: str,
    run_id: str,
    execution_binding_sha256: str,
    runtime_binding_sha256: str,
    current_binding_sha256: str,
) -> dict[str, Any]:
    checked = validate_pair_record(pair)
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError("experiment_name is empty")
    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ValueError("git_commit must be a full SHA")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,31}", run_id) is None:
        raise ValueError("run_id is invalid")
    return {
        "record_type": CAPTURE_RECORD_TYPE,
        "experiment_name": experiment_name,
        "git_commit": git_commit,
        "run_id": run_id,
        "eval_manifest_hash": checked["eval_manifest_hash"],
        "execution_binding_sha256": require_sha256(
            execution_binding_sha256, "execution_binding_sha256"
        ),
        "runtime_binding_sha256": require_sha256(
            runtime_binding_sha256, "runtime_binding_sha256"
        ),
        "current_binding_sha256": require_sha256(
            current_binding_sha256, "current_binding_sha256"
        ),
        "stable_example_id": checked["stable_example_id"],
        "stable_root_id": checked["stable_root_id"],
        "stable_write_id": checked["stable_write_id"],
        "pair_id": checked["pair_id"],
        "pair": checked,
        "training_authorized": False,
        "method_selected": False,
    }


def validate_capture_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    frozen_pairs: Sequence[Mapping[str, Any]],
    experiment_name: str,
    git_commit: str,
    run_id: str,
    execution_binding_sha256: str,
    runtime_binding_sha256: str,
    current_binding_sha256: str,
    decoder: Callable[[list[int]], str] | None = None,
    writer_prompt_builder: Callable[[list[int], list[int], list[int]], list[int]] | None = None,
    reader_prompt_builder: Callable[[list[int], list[int]], list[int]] | None = None,
    expected_pair_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute a complete four-pair artifact; persisted PASS flags are forbidden."""
    normalized_records = [dict(item) for item in records]
    if (writer_prompt_builder is None) != (reader_prompt_builder is None):
        raise ValueError("writer and reader prompt reconstruction callbacks must be paired")
    chain_failures = validate_jsonl_chain(normalized_records)
    if chain_failures:
        raise ValueError(f"capture ledger hash chain failed: {chain_failures}")
    if len(normalized_records) != 4 or len(frozen_pairs) != 4:
        raise ValueError("capture attrition: exactly four frozen pairs are required")
    expected_by_write: dict[str, dict[str, Any]] = {}
    for item in frozen_pairs:
        identity = _stable_identity(item)
        trajectory_seed = require_int(item.get("trajectory_seed"), "frozen.trajectory_seed", minimum=0)
        writer_turn = require_int(item.get("intervention_writer_turn"), "frozen.intervention_writer_turn", minimum=1)
        total = require_int(item.get("total_writer_turns"), "frozen.total_writer_turns", minimum=2)
        ids = stable_capture_ids(identity, trajectory_seed=trajectory_seed, writer_turn=writer_turn)
        expected_by_write[ids["stable_write_id"]] = {
            **dict(item),
            **identity,
            **ids,
            "trajectory_seed": trajectory_seed,
            "intervention_writer_turn": writer_turn,
            "total_writer_turns": total,
        }
    if len(expected_by_write) != 4:
        raise ValueError("frozen stable-write inventory is not unique")

    pairs: list[dict[str, Any]] = []
    envelope_allowed = {
        "record_type", "experiment_name", "git_commit", "run_id", "eval_manifest_hash",
        "execution_binding_sha256", "runtime_binding_sha256", "current_binding_sha256",
        "stable_example_id", "stable_root_id", "stable_write_id", "pair_id", "pair",
        "training_authorized", "method_selected", "record_index",
        "previous_record_sha256", "record_sha256",
    }
    seen_writes: set[str] = set()
    for index, envelope in enumerate(normalized_records):
        extra = sorted(set(envelope) - envelope_allowed)
        if extra:
            raise ValueError(f"capture envelope {index} contains handcrafted fields: {extra}")
        expected_common = {
            "record_type": CAPTURE_RECORD_TYPE,
            "experiment_name": experiment_name,
            "git_commit": git_commit,
            "run_id": run_id,
            "execution_binding_sha256": execution_binding_sha256,
            "runtime_binding_sha256": runtime_binding_sha256,
            "current_binding_sha256": current_binding_sha256,
            "training_authorized": False,
            "method_selected": False,
        }
        for field, expected in expected_common.items():
            if envelope.get(field) != expected:
                raise ValueError(f"capture envelope {index} {field} mismatch")
        pair = validate_pair_record(envelope.get("pair", {}), decoder=decoder)
        if expected_pair_binding is not None:
            expected_shared = {
                field: expected_pair_binding[field]
                for field in (
                    "writer_checkpoint_sha256", "reader_checkpoint_sha256",
                    "writer_prompt_template_sha256", "reader_prompt_template_sha256",
                    "writer_decode", "reader_decode",
                )
            }
            for field, expected in expected_shared.items():
                if canonical_json(pair["shared_contract"].get(field)) != canonical_json(expected):
                    raise ValueError(f"pair shared contract differs from P0 {field}")
            for field in (
                "physical_gpu_whitelist", "visible_devices",
                "physical_gpu_identity", "engine_config_sha256",
                "global_generate_call_count", "worker_multiproc_method",
                "vllm_observed_worker_multiproc_method",
                "multiprocessing_context_method",
                "parent_cuda_initialization_policy",
            ):
                if canonical_json(pair["execution"].get(field)) != canonical_json(
                    expected_pair_binding[field]
                ):
                    raise ValueError(f"pair execution differs from P0 {field}")
            expected_eos = require_int(
                expected_pair_binding.get("eos_token_id"),
                "expected_pair_binding.eos_token_id",
                minimum=0,
            )
            writer_generations_for_eos = [*pair["prefix_turns"], pair["candidate"]]
            for arm in ARMS:
                writer_generations_for_eos.extend(pair["arms"][arm]["future_turns"])
            if any(item["eos_token_id"] != expected_eos for item in writer_generations_for_eos):
                raise ValueError("writer EOS token ID differs from P0 tokenizer")
        if writer_prompt_builder is not None and reader_prompt_builder is not None:
            question = list(pair["question_token_ids"])
            writer_generations = [*pair["prefix_turns"], pair["candidate"]]
            for arm in ARMS:
                writer_generations.extend(pair["arms"][arm]["future_turns"])
            for generation in writer_generations:
                prompt = generation["prompt"]
                expected_prompt = list(
                    writer_prompt_builder(
                        question,
                        list(prompt["loaded_state_receipt"]["state"]["token_ids"]),
                        list(prompt["chunk_token_ids"]),
                    )
                )
                if prompt["token_ids"] != expected_prompt:
                    raise ValueError(
                        f"writer prompt was not reconstructed from exact loaded bytes: "
                        f"{generation['stable_turn_id']}"
                    )
            for arm in ARMS:
                final = pair["arms"][arm]["final_reader"]
                prompt = final["prompt"]
                expected_prompt = list(
                    reader_prompt_builder(
                        question,
                        list(prompt["loaded_state_receipt"]["state"]["token_ids"]),
                    )
                )
                if prompt["token_ids"] != expected_prompt:
                    raise ValueError(
                        f"reader prompt was not reconstructed from exact loaded bytes: "
                        f"{final['stable_turn_id']}"
                    )
        canonical_envelope = build_capture_envelope(
            pair,
            experiment_name=experiment_name,
            git_commit=git_commit,
            run_id=run_id,
            execution_binding_sha256=execution_binding_sha256,
            runtime_binding_sha256=runtime_binding_sha256,
            current_binding_sha256=current_binding_sha256,
        )
        for field, expected in canonical_envelope.items():
            if canonical_json(envelope.get(field)) != canonical_json(expected):
                raise ValueError(f"capture envelope {index} canonical {field} mismatch")
        frozen = expected_by_write.get(pair["stable_write_id"])
        if frozen is None or pair["stable_write_id"] in seen_writes:
            raise ValueError("capture has an unexpected or duplicate stable write")
        for field in (*STABLE_FIELDS, "stable_example_id", "stable_root_id", "stable_write_id", "trajectory_seed", "intervention_writer_turn", "total_writer_turns"):
            if pair[field] != frozen[field]:
                raise ValueError(f"capture pair differs from P0-frozen {field}")
        if pair["question_token_ids_sha256"] != frozen.get("question_token_ids_sha256"):
            raise ValueError("capture question tokens differ from P0")
        if pair["no_memory_state"]["token_ids_sha256"] != frozen.get(
            "no_memory_token_ids_sha256"
        ):
            raise ValueError("capture no-memory sentinel differs from P0")
        all_writer_generations = [*pair["prefix_turns"], pair["candidate"]]
        observed_prefix_candidate_chunks = [
            item["prompt"]["chunk_token_ids_sha256"] for item in all_writer_generations
        ]
        frozen_chunk_hashes = frozen.get("chunk_token_ids_sha256")
        if (
            not isinstance(frozen_chunk_hashes, list)
            or observed_prefix_candidate_chunks
            != frozen_chunk_hashes[: pair["intervention_writer_turn"] + 1]
        ):
            raise ValueError("capture prefix/candidate chunks differ from P0")
        observed_future_hashes = [
            item["token_ids_sha256"] for item in pair["shared_contract"]["future_chunks"]
        ]
        if observed_future_hashes != frozen.get("future_chunk_token_ids_sha256"):
            raise ValueError("capture shared future chunks differ from P0")
        reconstructed_context = [
            token
            for generation in all_writer_generations
            for token in generation["prompt"]["chunk_token_ids"]
        ] + [
            token
            for chunk in pair["shared_contract"]["future_chunks"]
            for token in chunk["token_ids"]
        ]
        if canonical_sha256(reconstructed_context) != frozen.get("context_token_ids_sha256"):
            raise ValueError("capture reconstructed context tokens differ from P0")
        if pair["prefix_turns"][0]["prompt"]["token_ids_sha256"] != frozen.get(
            "writer_turn0_prompt_token_sha256"
        ):
            raise ValueError("capture writer turn-0 prompt differs from P0")
        if pair["pair_generate_call_count"] != frozen.get("expected_pair_generate_calls"):
            raise ValueError("capture pair call count differs from P0")
        seen_writes.add(pair["stable_write_id"])
        pairs.append(pair)
    if seen_writes != set(expected_by_write):
        raise ValueError("capture attrition: frozen stable-write inventory is incomplete")

    executions = [pair["execution"] for pair in pairs]
    process_fields = (
        "engine_id", "cache_namespace", "process_instance_uuid", "process_pid",
        "physical_gpu_whitelist", "visible_devices", "physical_gpu_identity",
        "global_generate_call_count", "parent_credential_id",
        "worker_multiproc_method", "vllm_observed_worker_multiproc_method",
        "multiprocessing_context_method", "parent_cuda_initialized_before_engine",
        "parent_cuda_initialization_policy",
        "parent_credential_sha256", "parent_credential_path", "parent_issuer_pid",
        "observed_parent_pid",
        "parent_authorization_record_sha256", "engine_config_sha256",
    )
    for field in process_fields:
        if len({canonical_json(item[field]) for item in executions}) != 1:
            raise ValueError(f"capture pairs do not share one process/engine {field}")
    call_indices = [index for pair in pairs for index in pair["pair_generate_call_indices"]]
    if call_indices != list(range(1, len(call_indices) + 1)):
        raise ValueError("global generate-call ledger is not contiguous in capture order")
    if executions[0]["global_generate_call_count"] != len(call_indices):
        raise ValueError("global generate-call count differs from pair receipts")

    outcomes = {
        pair["stable_write_id"]: {
            arm: pair["arms"][arm]["final_reader"]["outcome"] for arm in ARMS
        }
        for pair in pairs
    }
    return {
        "status": "PASS",
        "decision": "COMMIT_RETAIN_CAPTURE_AUDIT_COMPLETE",
        "pair_count": 4,
        "stable_write_ids": [pair["stable_write_id"] for pair in pairs],
        "pair_ids": [pair["pair_id"] for pair in pairs],
        "generate_call_count": len(call_indices),
        "outcomes": outcomes,
        "training": {"trainer_attached": False, "actor_updates": 0, "optimizer_steps": 0},
        "claim_boundary": {
            "capture_and_audit_only": True,
            "method_selected": False,
            "training_authorized": False,
            "causal_or_performance_claim": False,
        },
    }
