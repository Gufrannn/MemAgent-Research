"""Frozen, engine-agnostic contracts for SMSB4 and Tetrad4.

The GPU executors live under ``tools/h20``.  This module only normalizes and
validates append-only evidence.  It is intentionally independent of PPO,
reward computation, and actor updates.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SMSB_SCHEMA = "memagent.smsb.capture-replay.v3"
TETRAD_SCHEMA = "memagent.tetrad.pilot4.v2"
PROVENANCE_HASHES = ("model", "tokenizer", "config", "code")
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
SMSB_REGIMES = ("temperature_zero", "matched_seed", "independent_seed")
TETRAD_ROLES = ("generated", "empty", "irrelevant", "shuffle", "gold")
FROZEN_VLLM_VERSION = "0.8.2"
PARENT_RECEIPT_SCHEMA = "memagent.serialization-credit.parent-launch-receipt.v2"


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_authority_mac(
    authority_secret: bytes, domain: str, payload: Mapping[str, Any]
) -> str:
    if not isinstance(authority_secret, bytes) or len(authority_secret) < 32:
        raise ValueError("parent authority secret must contain at least 32 bytes")
    if not isinstance(domain, str) or not domain:
        raise ValueError("parent authority MAC domain is empty")
    message = domain.encode("ascii") + b"\0" + canonical_json(dict(payload)).encode(
        "utf-8"
    )
    return hmac.new(authority_secret, message, hashlib.sha256).hexdigest()


def build_parent_launch_receipt(
    payload: Mapping[str, Any], authority_secret: bytes
) -> dict[str, Any]:
    receipt = dict(payload)
    if receipt.get("schema") != PARENT_RECEIPT_SCHEMA:
        raise ValueError("parent receipt schema differs")
    if receipt.get("record_type") != "parent_launch_receipt":
        raise ValueError("parent receipt record type differs")
    if receipt.get("child_kind") not in {
        "smsb_capture",
        "smsb_replay",
        "tetrad_replay",
    }:
        raise ValueError("parent receipt child kind is unsupported")
    for field in ("child_identity", "artifact", "stdout_artifact"):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            raise ValueError(f"parent receipt {field} is empty")
    for field in (
        "credential_id",
        "credential_mac",
        "credential_sha256",
        "artifact_sha256",
        "artifact_canonical_sha256",
        "stdout_artifact_sha256",
        "runner_argv_sha256",
        "runner_code_sha256",
        "current_binding_sha256",
        "runtime_binding_sha256",
        "execution_binding_sha256",
        "pre_child_model_manifest_sha256",
        "post_child_model_manifest_sha256",
        "post_child_current_binding_sha256",
        "pre_child_physical_gpu_identity_sha256",
        "post_child_physical_gpu_identity_sha256",
        "authority_secret_sha256",
    ):
        require_sha256(receipt.get(field), f"receipt.{field}")
    for field in ("parent_launcher_pid", "child_pid", "observed_child_ppid"):
        require_int(receipt.get(field), f"receipt.{field}", minimum=1)
    if receipt["observed_child_ppid"] != receipt["parent_launcher_pid"]:
        raise ValueError("parent receipt did not observe itself as child PPID")
    if require_int(receipt.get("child_exit_code"), "receipt.child_exit_code") != 0:
        raise ValueError("parent receipt child exit code is not zero")
    if receipt.get("parent_observed_launch") is not True:
        raise ValueError("parent receipt lacks parent-observed launch evidence")
    if receipt.get("pre_child_full_model_sha_verified") is not True:
        raise ValueError("parent receipt lacks child-start full model SHA evidence")
    if receipt.get("post_child_full_model_sha_verified") is not True:
        raise ValueError("parent receipt lacks parent post-child full model SHA evidence")
    for field in (
        "pre_child_physical_gpu_identity",
        "post_child_physical_gpu_identity",
    ):
        identities = receipt.get(field)
        if (
            not isinstance(identities, list)
            or len(identities) != 2
            or any(not isinstance(value, str) or not value for value in identities)
        ):
            raise ValueError(f"parent receipt {field} must contain two devices")
        if receipt.get(f"{field}_sha256") != canonical_sha256(identities):
            raise ValueError(f"parent receipt {field} digest differs")
    if receipt.get("post_child_cuda_device_order") != "PCI_BUS_ID":
        raise ValueError("parent receipt post-child CUDA_DEVICE_ORDER differs")
    if (
        receipt.get("post_child_model_manifest_sha256")
        != receipt.get("pre_child_model_manifest_sha256")
        or receipt.get("post_child_current_binding_sha256")
        != receipt.get("current_binding_sha256")
        or receipt.get("post_child_physical_gpu_identity")
        != receipt.get("pre_child_physical_gpu_identity")
    ):
        raise ValueError("parent receipt post-child model/GPU differs from pre-child")
    if receipt.get("training_authorized") is not False:
        raise ValueError("parent receipt improperly authorizes training")
    unsigned = dict(receipt)
    unsigned.pop("receipt_id", None)
    unsigned.pop("receipt_mac", None)
    receipt_id = canonical_sha256(unsigned)
    signed = {**unsigned, "receipt_id": receipt_id}
    signed["receipt_mac"] = parent_authority_mac(
        authority_secret, "parent-launch-receipt-v2", signed
    )
    return signed


def validate_parent_launch_receipt(
    receipt: Mapping[str, Any],
    *,
    authority_secret: bytes,
    artifact_payload: Any,
    child_evidence: Mapping[str, Any],
    child_kind: str,
    child_identity: str,
) -> dict[str, Any]:
    raw = dict(receipt)
    receipt_id = require_sha256(raw.get("receipt_id"), "receipt.receipt_id")
    receipt_mac = require_sha256(raw.get("receipt_mac"), "receipt.receipt_mac")
    unsigned = dict(raw)
    unsigned.pop("receipt_id")
    unsigned.pop("receipt_mac")
    rebuilt = build_parent_launch_receipt(unsigned, authority_secret)
    if not hmac.compare_digest(receipt_id, rebuilt["receipt_id"]) or not hmac.compare_digest(
        receipt_mac, rebuilt["receipt_mac"]
    ):
        raise ValueError("parent launch receipt signature differs")
    if raw.get("authority_secret_sha256") != hashlib.sha256(
        authority_secret
    ).hexdigest():
        raise ValueError("parent launch receipt authority binding differs")
    if raw.get("child_kind") != child_kind or raw.get("child_identity") != child_identity:
        raise ValueError("parent launch receipt child identity differs")
    if raw.get("artifact_canonical_sha256") != canonical_sha256(artifact_payload):
        raise ValueError("parent launch receipt canonical artifact digest differs")
    expected_cross_binding = {
        "child_pid": child_evidence.get("process_pid"),
        "observed_child_ppid": child_evidence.get("observed_parent_pid"),
        "process_instance_uuid": child_evidence.get("process_instance_uuid"),
        "credential_id": child_evidence.get("parent_credential_id"),
        "credential_mac": child_evidence.get("parent_credential_mac"),
        "credential_sha256": child_evidence.get("parent_credential_sha256"),
        "current_binding_sha256": child_evidence.get("current_binding_sha256"),
        "runtime_binding_sha256": child_evidence.get("runtime_binding_sha256"),
        "execution_binding_sha256": child_evidence.get(
            "execution_binding_sha256",
            child_evidence.get("engine_config_sha256"),
        ),
        "pre_child_model_manifest_sha256": child_evidence.get(
            "model_manifest_sha256"
        ),
        "pre_child_full_model_sha_verified": child_evidence.get(
            "full_model_sha_verified_at_child_start"
        ),
        "pre_child_physical_gpu_identity": child_evidence.get(
            "physical_gpu_identity"
        ),
        "pre_child_physical_gpu_identity_sha256": canonical_sha256(
            child_evidence.get("physical_gpu_identity")
        ),
    }
    for field, value in expected_cross_binding.items():
        if raw.get(field) != value:
            raise ValueError(f"parent launch receipt/result differs at {field}")
    return rebuilt


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_exclusive(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(dict(row)) + "\n")


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def require_int(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Accept JSON integers only; bool, float, and numeric strings are evidence drift."""
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer (not bool/float/string)")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return value


def require_finite_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Accept finite JSON numbers only; bool and numeric strings are forbidden."""
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be a finite number (not bool/string)")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return result


def validate_sampling_params(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    result = dict(value)
    for name in ("temperature", "top_p", "min_p"):
        if name in result:
            result[name] = require_finite_number(
                result[name],
                f"{field}.{name}",
                minimum=0.0,
                maximum=1.0 if name in ("top_p", "min_p") else None,
            )
    for name, minimum in (("top_k", -1), ("n", 1), ("best_of", 1), ("max_tokens", 1)):
        if name in result:
            result[name] = require_int(result[name], f"{field}.{name}", minimum=minimum)
    if "seed" in result:
        result["seed"] = require_int(result["seed"], f"{field}.seed", minimum=0)
    if "do_sample" in result and type(result["do_sample"]) is not bool:
        raise ValueError(f"{field}.do_sample must be a boolean")
    return result


def token_ids(value: Iterable[Any], field: str, *, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON token-id array")
    result = [require_int(item, f"{field}[{index}]", minimum=0) for index, item in enumerate(value)]
    if not allow_empty and not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def center_truncate_token_ids(value: list[int], maximum: Any) -> list[int]:
    """Mirror ``verl_F.postprocess_data(..., truncation='center')`` for even limits."""
    ids = token_ids(value, "context_token_ids")
    limit = require_int(maximum, "max_context_tokens", minimum=1)
    if len(ids) <= limit:
        return list(ids)
    half = limit // 2
    return ids[:half] + ids[-half:]


def validate_single_request_token_budget(
    prompt_token_ids: list[int],
    max_tokens: Any,
    *,
    max_model_len: Any,
    max_num_batched_tokens: Any,
) -> dict[str, int]:
    """Fail before vLLM when one prompt plus its output budget exceeds capacity."""
    prompt = token_ids(
        prompt_token_ids, "generation.prompt_token_ids", allow_empty=False
    )
    output_limit = require_int(max_tokens, "generation.max_tokens", minimum=1)
    model_limit = require_int(max_model_len, "backend.max_model_len", minimum=1)
    batched_limit = require_int(
        max_num_batched_tokens,
        "backend.max_num_batched_tokens",
        minimum=1,
    )
    capacity = min(model_limit, batched_limit)
    total = len(prompt) + output_limit
    if total > capacity:
        raise ValueError(
            "single-request token budget exceeds frozen vLLM capacity: "
            f"prompt_tokens={len(prompt)}, max_tokens={output_limit}, "
            f"total={total}, max_model_len={model_limit}, "
            f"max_num_batched_tokens={batched_limit}, capacity={capacity}"
        )
    return {
        "prompt_tokens": len(prompt),
        "max_tokens": output_limit,
        "total_tokens": total,
        "capacity_tokens": capacity,
    }


def _stable_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in STABLE_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"stable identity is missing fields: {missing}")
    identity = {field: payload[field] for field in STABLE_FIELDS}
    for field in (
        "semantic_dataset_index",
        "source_order_index",
        "raw_row_position",
        "production_effective_position",
    ):
        identity[field] = require_int(identity[field], field, minimum=0)
    if not isinstance(identity["example_id"], str) or not identity["example_id"]:
        raise ValueError("example_id must be a non-empty string")
    if identity["example_id"] != str(identity["semantic_dataset_index"]):
        raise ValueError("example_id differs from semantic_dataset_index")
    for field in (
        "eval_manifest_hash",
        "source_question_hash",
        "source_context_hash",
        "ground_truth_hash",
    ):
        identity[field] = require_sha256(identity[field], field)
    return identity


def build_capture_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one recurrent capture and bind exact prompt/output tokens."""
    required = {
        *STABLE_FIELDS,
        "experiment_id",
        "engine_id",
        "cache_namespace",
        "memory_ledger",
        "question_token_ids",
        "final_memory_token_ids",
        "final_prompt_token_ids",
        "answer_token_ids",
        "temperature_zero_control_answer_token_ids",
        "sampling_params",
        "trajectory_seed",
        "request_seed",
        "final_stochastic_request_seed",
        "final_control_request_seed",
        "final_stochastic_configured_request_seed",
        "final_stochastic_actual_request_seed",
        "final_control_configured_request_seed",
        "final_control_actual_request_seed",
        "final_stochastic_generate_call_index",
        "final_control_generate_call_index",
        "hashes",
        "vllm_version",
        "updater_calls",
        "prompt_template_sha256",
        "runtime_binding_sha256",
        "engine_config_sha256",
        "current_binding_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"capture missing fields: {missing}")
    identity = _stable_identity(payload)
    hashes = dict(payload["hashes"])
    absent_hashes = [field for field in PROVENANCE_HASHES if not hashes.get(field)]
    if absent_hashes:
        raise ValueError(f"capture missing provenance hashes: {absent_hashes}")
    for field in PROVENANCE_HASHES:
        hashes[field] = require_sha256(hashes[field], f"hashes.{field}")
    ledger: list[dict[str, Any]] = []
    trajectory_seed = require_int(payload["trajectory_seed"], "trajectory_seed", minimum=0)
    from recurrent.research.trajectory_seeding import derive_turn_request_seeds

    for expected_turn, raw in enumerate(payload["memory_ledger"]):
        turn = require_int(raw.get("turn"), f"memory_ledger[{expected_turn}].turn", minimum=0)
        if turn != expected_turn:
            raise ValueError("memory ledger turns must be contiguous from zero")
        request_seed = require_int(
            raw.get("request_seed"), f"memory_ledger[{turn}].request_seed", minimum=0
        )
        configured_seed = require_int(
            raw.get("configured_request_seed"),
            f"memory_ledger[{turn}].configured_request_seed",
            minimum=0,
        )
        actual_seed = require_int(
            raw.get("actual_request_seed"),
            f"memory_ledger[{turn}].actual_request_seed",
            minimum=0,
        )
        expected_seed = derive_turn_request_seeds([trajectory_seed], [0], turn)[0]
        if request_seed != expected_seed or configured_seed != expected_seed or actual_seed != expected_seed:
            raise ValueError(f"memory ledger turn {turn} request seed differs from trajectory seed")
        generate_call_index = require_int(
            raw.get("generate_call_index"),
            f"memory_ledger[{turn}].generate_call_index",
            minimum=1,
        )
        writer_prompt_ids = token_ids(
            raw.get("writer_prompt_token_ids"),
            f"memory_ledger[{turn}].writer_prompt_token_ids",
            allow_empty=False,
        )
        chunk_start = require_int(
            raw.get("chunk_start"),
            f"memory_ledger[{turn}].chunk_start",
            minimum=0,
        )
        chunk_end = require_int(
            raw.get("chunk_end"),
            f"memory_ledger[{turn}].chunk_end",
            minimum=1,
        )
        chunk_ids = token_ids(
            raw.get("chunk_token_ids"),
            f"memory_ledger[{turn}].chunk_token_ids",
            allow_empty=False,
        )
        input_memory_ids = token_ids(
            raw.get("input_memory_token_ids"),
            f"memory_ledger[{turn}].input_memory_token_ids",
            allow_empty=True,
        )
        if chunk_end <= chunk_start or chunk_end - chunk_start != len(chunk_ids):
            raise ValueError(f"memory ledger turn {turn} chunk span differs from IDs")
        if turn == 0:
            if chunk_start != 0:
                raise ValueError("memory ledger first chunk must start at zero")
        else:
            if chunk_start != ledger[-1]["chunk_end"]:
                raise ValueError("memory ledger chunk spans must be contiguous")
            if input_memory_ids != ledger[-1]["token_ids"]:
                raise ValueError(
                    f"memory ledger turn {turn} input memory differs from prior output"
                )
        derived_fields = {
            "writer_prompt_token_ids_sha256": canonical_sha256(writer_prompt_ids),
            "writer_prompt_token_length": len(writer_prompt_ids),
            "chunk_token_ids_sha256": canonical_sha256(chunk_ids),
            "chunk_token_length": len(chunk_ids),
            "input_memory_token_ids_sha256": canonical_sha256(input_memory_ids),
            "input_memory_token_length": len(input_memory_ids),
        }
        for field, expected in derived_fields.items():
            supplied = raw.get(field)
            if field.endswith("_sha256"):
                require_sha256(supplied, f"memory_ledger[{turn}].{field}")
            else:
                require_int(
                    supplied,
                    f"memory_ledger[{turn}].{field}",
                    minimum=(0 if field == "input_memory_token_length" else 1),
                )
            if supplied != expected:
                raise ValueError(
                    f"memory ledger turn {turn} {field} differs from token IDs"
                )
        text = str(raw["text"])
        ids = token_ids(raw["token_ids"], f"memory_ledger[{turn}].token_ids")
        ledger.append(
            {
                "turn": turn,
                "writer_prompt_token_ids": writer_prompt_ids,
                "writer_prompt_token_ids_sha256": derived_fields[
                    "writer_prompt_token_ids_sha256"
                ],
                "writer_prompt_token_length": len(writer_prompt_ids),
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "chunk_token_ids": chunk_ids,
                "chunk_token_ids_sha256": derived_fields[
                    "chunk_token_ids_sha256"
                ],
                "chunk_token_length": len(chunk_ids),
                "input_memory_token_ids": input_memory_ids,
                "input_memory_token_ids_sha256": derived_fields[
                    "input_memory_token_ids_sha256"
                ],
                "input_memory_token_length": len(input_memory_ids),
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_ids": ids,
                "token_ids_sha256": canonical_sha256(ids),
                "request_seed": request_seed,
                "configured_request_seed": configured_seed,
                "actual_request_seed": actual_seed,
                "generate_call_index": generate_call_index,
            }
        )
    updater_calls = require_int(payload["updater_calls"], "updater_calls", minimum=1)
    if updater_calls != len(ledger) or updater_calls < 1:
        raise ValueError("capture updater count differs from the memory ledger")
    if [row["generate_call_index"] for row in ledger] != list(
        range(ledger[0]["generate_call_index"], ledger[0]["generate_call_index"] + updater_calls)
    ):
        raise ValueError("writer generate-call indices are not contiguous")
    final_stochastic_seed = require_int(
        payload["final_stochastic_request_seed"], "final_stochastic_request_seed", minimum=0
    )
    final_control_seed = require_int(
        payload["final_control_request_seed"], "final_control_request_seed", minimum=0
    )
    expected_final_seed = derive_turn_request_seeds([trajectory_seed], [0], updater_calls)[0]
    if final_stochastic_seed != expected_final_seed or final_control_seed != expected_final_seed:
        raise ValueError("final stochastic/control request seed differs from trajectory seed")
    final_seed_fields = (
        "final_stochastic_configured_request_seed",
        "final_stochastic_actual_request_seed",
        "final_control_configured_request_seed",
        "final_control_actual_request_seed",
    )
    if any(
        require_int(payload[field], field, minimum=0) != expected_final_seed
        for field in final_seed_fields
    ):
        raise ValueError("configured/actual final request seed differs from trajectory seed")
    if require_int(payload["request_seed"], "request_seed", minimum=0) != final_stochastic_seed:
        raise ValueError("legacy request_seed alias differs from final stochastic seed")
    final_stochastic_call = require_int(
        payload["final_stochastic_generate_call_index"],
        "final_stochastic_generate_call_index",
        minimum=1,
    )
    final_control_call = require_int(
        payload["final_control_generate_call_index"],
        "final_control_generate_call_index",
        minimum=1,
    )
    if final_stochastic_call != ledger[-1]["generate_call_index"] + 1:
        raise ValueError("final stochastic call is not immediately after the writer calls")
    if final_control_call != final_stochastic_call + 1:
        raise ValueError("final control must be a separate generate call after stochastic final")
    prompt_ids = token_ids(payload["final_prompt_token_ids"], "final_prompt_token_ids", allow_empty=False)
    final_memory_ids = token_ids(payload["final_memory_token_ids"], "final_memory_token_ids")
    if final_memory_ids != ledger[-1]["token_ids"]:
        raise ValueError("final memory differs from the last writer output")
    stochastic_ids = token_ids(payload["answer_token_ids"], "answer_token_ids")
    deterministic_ids = token_ids(
        payload["temperature_zero_control_answer_token_ids"],
        "temperature_zero_control_answer_token_ids",
    )
    execution = dict(payload.get("execution", {}))
    if execution.get("strict_vllm") is not True:
        raise ValueError("capture did not certify strict vLLM")
    if require_int(execution.get("tensor_parallel_size"), "execution.tensor_parallel_size") != 2:
        raise ValueError("capture tensor_parallel_size must be 2")
    physical_gpu_whitelist = execution.get("physical_gpu_whitelist")
    if (
        not isinstance(physical_gpu_whitelist, list)
        or len(physical_gpu_whitelist) != 2
        or any(type(value) is not int or value < 0 for value in physical_gpu_whitelist)
        or physical_gpu_whitelist[0] >= physical_gpu_whitelist[1]
    ):
        raise ValueError("capture GPU whitelist must be an ascending distinct pair")
    physical_gpu_identity = execution.get("physical_gpu_identity")
    if (
        not isinstance(physical_gpu_identity, list)
        or len(physical_gpu_identity) != 2
        or any(not isinstance(value, str) or not value for value in physical_gpu_identity)
    ):
        raise ValueError("capture physical GPU identity must contain two concrete devices")
    if execution.get("cuda_device_order") != "PCI_BUS_ID":
        raise ValueError("capture CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if execution.get("prefix_cache_enabled") is not False:
        raise ValueError("capture prefix cache must be disabled")
    if execution.get("full_model_sha_verified_at_capture_start") is not True:
        raise ValueError("capture start did not certify a full model SHA verification")
    if execution.get("full_model_sha_verified_at_child_start") is not True:
        raise ValueError("capture child start did not certify a full model SHA verification")
    if require_sha256(
        execution.get("model_manifest_sha256"),
        "execution.model_manifest_sha256",
    ) != hashes["model"]:
        raise ValueError("capture execution model manifest differs from provenance")
    if not isinstance(execution.get("process_instance_uuid"), str) or not execution[
        "process_instance_uuid"
    ]:
        raise ValueError("capture process instance UUID is missing")
    try:
        import uuid

        uuid.UUID(str(execution.get("process_instance_uuid")))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("capture process instance UUID is invalid") from error
    require_int(execution.get("process_pid"), "execution.process_pid", minimum=1)
    if require_int(
        execution.get("engine_construction_count"), "execution.engine_construction_count", minimum=0
    ) != 1:
        raise ValueError("capture must construct exactly one engine")
    process_generate_calls = require_int(
        execution.get("generate_call_count"), "execution.generate_call_count", minimum=1
    )
    if final_control_call > process_generate_calls:
        raise ValueError("capture generate-call count is inconsistent")
    record = {
        "schema": SMSB_SCHEMA,
        "record_type": "capture",
        **identity,
        "experiment_id": str(payload["experiment_id"]),
        "engine_id": str(payload["engine_id"]),
        "cache_namespace": str(payload["cache_namespace"]),
        "memory_ledger": ledger,
        "question_token_ids": token_ids(payload["question_token_ids"], "question_token_ids"),
        "final_memory_token_ids": final_memory_ids,
        "prompt_construction": "recurrent.TokenTemplate.token_concat.v1",
        "prompt_template_sha256": require_sha256(
            payload["prompt_template_sha256"], "prompt_template_sha256"
        ),
        "final_prompt_token_ids": prompt_ids,
        "final_prompt_token_ids_sha256": canonical_sha256(prompt_ids),
        "answer_token_ids": stochastic_ids,
        "answer_token_ids_sha256": canonical_sha256(stochastic_ids),
        "temperature_zero_control_answer_token_ids": deterministic_ids,
        "temperature_zero_control_answer_token_ids_sha256": canonical_sha256(
            deterministic_ids
        ),
        "sampling_params": validate_sampling_params(payload["sampling_params"], "sampling_params"),
        "trajectory_seed": trajectory_seed,
        "request_seed": final_stochastic_seed,
        "final_stochastic_request_seed": final_stochastic_seed,
        "final_control_request_seed": final_control_seed,
        "final_stochastic_configured_request_seed": final_stochastic_seed,
        "final_stochastic_actual_request_seed": final_stochastic_seed,
        "final_control_configured_request_seed": final_control_seed,
        "final_control_actual_request_seed": final_control_seed,
        "final_stochastic_generate_call_index": final_stochastic_call,
        "final_control_generate_call_index": final_control_call,
        "hashes": hashes,
        "vllm_version": str(payload["vllm_version"]),
        "updater_calls": updater_calls,
        "runtime_binding_sha256": require_sha256(
            payload["runtime_binding_sha256"], "runtime_binding_sha256"
        ),
        "engine_config_sha256": require_sha256(
            payload["engine_config_sha256"], "engine_config_sha256"
        ),
        "current_binding_sha256": require_sha256(
            payload["current_binding_sha256"], "current_binding_sha256"
        ),
        "execution": execution,
        "ground_truth": list(payload.get("ground_truth", [])),
    }
    record["capture_id"] = canonical_sha256(record)
    return record


def validate_capture_record(record: Mapping[str, Any]) -> dict[str, Any]:
    rebuilt = build_capture_record(record)
    if canonical_json(rebuilt) != canonical_json(dict(record)):
        raise ValueError("capture record is not canonical or its capture_id is stale")
    return rebuilt


def build_replay_request(
    capture: Mapping[str, Any],
    regime: str,
    *,
    replay_engine_id: str,
    replay_cache_namespace: str,
    independent_seed: int | None = None,
) -> dict[str, Any]:
    if regime not in SMSB_REGIMES:
        raise ValueError(f"unknown SMSB replay regime: {regime}")
    if replay_engine_id == capture["engine_id"]:
        raise ValueError("fresh replay reused the capture engine")
    if replay_cache_namespace == capture["cache_namespace"]:
        raise ValueError("fresh replay reused the capture cache namespace")
    params = validate_sampling_params(capture["sampling_params"], "capture.sampling_params")
    seed = require_int(capture["request_seed"], "capture.request_seed", minimum=0)
    if regime == "temperature_zero":
        params.update(temperature=0.0, do_sample=False)
    elif regime == "independent_seed":
        if independent_seed is None:
            raise ValueError("independent replay seed is required")
        independent_seed = require_int(independent_seed, "independent_seed", minimum=0)
        if independent_seed == seed:
            raise ValueError("independent replay seed must differ from capture seed")
        seed = independent_seed
    request = {
        "schema": SMSB_SCHEMA,
        "record_type": "replay_request",
        **{field: capture[field] for field in STABLE_FIELDS},
        "capture_id": capture["capture_id"],
        "experiment_id": capture["experiment_id"],
        "regime": regime,
        "engine_id": str(replay_engine_id),
        "cache_namespace": str(replay_cache_namespace),
        "prefix_caching": False,
        "single_request_only": True,
        "max_num_seqs": 1,
        "recurrent_updater_calls": 0,
        "context_or_chunks_visible": False,
        "question_token_ids": capture["question_token_ids"],
        "final_memory_token_ids": capture["final_memory_token_ids"],
        "prompt_construction": capture["prompt_construction"],
        "prompt_template_sha256": capture["prompt_template_sha256"],
        "expected_prompt_token_ids": capture["final_prompt_token_ids"],
        "expected_prompt_token_ids_sha256": capture["final_prompt_token_ids_sha256"],
        "sampling_params": params,
        "request_seed": seed,
        "hashes": capture["hashes"],
        "vllm_version": capture["vllm_version"],
        "runtime_binding_sha256": capture["runtime_binding_sha256"],
        "engine_config_sha256": capture["engine_config_sha256"],
        "current_binding_sha256": capture["current_binding_sha256"],
    }
    request["request_id"] = canonical_sha256(request)
    return request


def validate_replay(
    capture: Mapping[str, Any],
    request: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        regime_value = request.get("regime")
        expected_request = build_replay_request(
            capture,
            regime_value,
            replay_engine_id=request.get("engine_id"),
            replay_cache_namespace=request.get("cache_namespace"),
            independent_seed=(
                request.get("request_seed")
                if regime_value == "independent_seed"
                else None
            ),
        )
        if canonical_json(expected_request) != canonical_json(dict(request)):
            errors.append("replay_request_not_canonical")
    except Exception as error:
        errors.append(f"replay_request_invalid:{error}")
    if request.get("capture_id") != capture.get("capture_id"):
        errors.append("capture_id_mismatch")
    if result.get("request_id") != request.get("request_id"):
        errors.append("request_id_mismatch")
    if result.get("engine_id") != request.get("engine_id"):
        errors.append("engine_id_mismatch")
    if request.get("engine_id") == capture.get("engine_id") or result.get("fresh_engine_verified") is not True:
        errors.append("fresh_engine_not_verified")
    if request.get("cache_namespace") == capture.get("cache_namespace") or result.get("cache_isolation_verified") is not True:
        errors.append("cache_isolation_not_verified")
    if request.get("prefix_caching") is not False or result.get("prefix_cache_enabled") is not False:
        errors.append("prefix_cache_not_disabled")
    if request.get("single_request_only") is not True or result.get("single_request_execution_verified") is not True:
        errors.append("single_request_not_verified")
    if type(request.get("max_num_seqs")) is not int or request.get("max_num_seqs") != 1:
        errors.append("request_max_num_seqs_not_one_integer")
    if type(result.get("max_num_seqs")) is not int or result.get("max_num_seqs") != 1:
        errors.append("max_num_seqs_not_one")
    if (
        type(request.get("recurrent_updater_calls")) is not int
        or request.get("recurrent_updater_calls") != 0
        or type(result.get("observed_updater_calls")) is not int
        or result.get("observed_updater_calls") != 0
    ):
        errors.append("updater_was_called")
    if request.get("context_or_chunks_visible") is not False or result.get("context_or_chunks_visible") is not False:
        errors.append("context_or_chunk_leak")
    if result.get("prompt_reconstructed_from_serialized_state") is not True:
        errors.append("prompt_not_reconstructed_from_serialized_state")
    for field in (
        "hashes", "vllm_version", "runtime_binding_sha256", "engine_config_sha256",
        "current_binding_sha256",
    ):
        if request.get(field) != capture.get(field) or result.get(field) != capture.get(field):
            errors.append(f"{field}_mismatch")
    try:
        import uuid

        if not isinstance(result.get("process_instance_uuid"), str):
            raise ValueError
        uuid.UUID(result["process_instance_uuid"])
    except (TypeError, ValueError, AttributeError):
        errors.append("process_instance_uuid_invalid")
    if type(result.get("process_pid")) is not int or result.get("process_pid", 0) < 1:
        errors.append("process_pid_invalid")
    if type(result.get("engine_construction_count")) is not int or result.get("engine_construction_count") != 1:
        errors.append("engine_construction_count_not_one")
    if type(result.get("generate_call_count")) is not int or result.get("generate_call_count") != 1:
        errors.append("generate_call_count_not_one")
    for field in ("parent_credential_id", "parent_credential_sha256"):
        try:
            require_sha256(result.get(field), f"result.{field}")
        except ValueError:
            errors.append(f"{field}_invalid")
    if (
        type(result.get("parent_issuer_pid")) is not int
        or result.get("parent_issuer_pid", 0) < 1
        or type(result.get("observed_parent_pid")) is not int
        or result.get("observed_parent_pid") != result.get("parent_issuer_pid")
    ):
        errors.append("parent_process_credential_mismatch")
    replay_prompt = token_ids(result.get("prompt_token_ids", []), "result.prompt_token_ids")
    if result.get("prompt_token_ids_sha256") != canonical_sha256(replay_prompt):
        errors.append("actual_prompt_token_hash_mismatch")
    if request.get("expected_prompt_token_ids_sha256") != canonical_sha256(replay_prompt):
        errors.append("request_prompt_token_hash_mismatch")
    l0 = replay_prompt == capture.get("final_prompt_token_ids")
    if not l0:
        errors.append("L0_prompt_token_mismatch")
    replay_answer = token_ids(result.get("answer_token_ids", []), "result.answer_token_ids")
    if result.get("answer_token_ids_sha256") != canonical_sha256(replay_answer):
        errors.append("actual_answer_token_hash_mismatch")
    regime = str(request.get("regime"))
    try:
        validate_sampling_params(request.get("sampling_params"), "request.sampling_params")
        request_seed = require_int(
            request.get("request_seed"), "request.request_seed", minimum=0
        )
        configured_seed = require_int(
            result.get("configured_request_seed"),
            "result.configured_request_seed",
            minimum=0,
        )
        actual_seed = require_int(
            result.get("actual_request_seed"),
            "result.actual_request_seed",
            minimum=0,
        )
        if configured_seed != request_seed or actual_seed != request_seed:
            errors.append("configured_or_actual_request_seed_mismatch")
    except ValueError as error:
        errors.append(f"sampling_or_seed_invalid:{error}")
    if type(result.get("tensor_parallel_size")) is not int or result.get("tensor_parallel_size") != 2:
        errors.append("execution_tensor_parallel_mismatch")
    physical_gpus = result.get("physical_gpu_whitelist")
    if (
        not isinstance(physical_gpus, list)
        or any(type(value) is not int for value in physical_gpus)
        or len(physical_gpus) != 2
        or physical_gpus[0] >= physical_gpus[1]
        or physical_gpus != capture.get("execution", {}).get("physical_gpu_whitelist")
    ):
        errors.append("execution_gpu_contract_mismatch")
    if (
        result.get("physical_gpu_identity")
        != capture.get("execution", {}).get("physical_gpu_identity")
        or result.get("cuda_device_order") != "PCI_BUS_ID"
    ):
        errors.append("execution_gpu_identity_mismatch")
    if result.get("full_model_sha_verified_at_child_start") is not True:
        errors.append("child_full_model_sha_not_verified")
    if result.get("model_manifest_sha256") != capture.get("hashes", {}).get("model"):
        errors.append("child_model_manifest_mismatch")
    l1 = (
        bool(l0 and replay_answer == capture.get("temperature_zero_control_answer_token_ids"))
        if regime == "temperature_zero"
        else None
    )
    l2 = (
        bool(l0 and replay_answer == capture.get("answer_token_ids"))
        if regime == "matched_seed"
        else None
    )
    if regime == "temperature_zero" and l1 is not True:
        errors.append("L1_deterministic_answer_mismatch")
    warnings: list[str] = []
    if regime == "matched_seed" and l2 is not True:
        warnings.append("L2_matched_seed_answer_mismatch")
    execution_valid = not errors
    return {
        "schema": SMSB_SCHEMA,
        "record_type": "validation",
        "capture_id": capture.get("capture_id"),
        "request_id": request.get("request_id"),
        "example_id": capture.get("example_id"),
        "regime": regime,
        "L0_prompt_identity": l0,
        "L1_deterministic_answer_identity": l1,
        "L2_matched_seed_answer_identity": l2,
        "answer_identity_descriptive_only": (
            replay_answer == capture.get("answer_token_ids")
            if regime == "independent_seed"
            else None
        ),
        "execution_valid": execution_valid,
        "E_det_gate_pass": l1 if regime == "temperature_zero" else None,
        "L2_report_only": regime == "matched_seed",
        "errors": errors,
        "diagnostic_warnings": warnings,
    }


def summarize_smsb_pilot(
    captures: Sequence[Mapping[str, Any]],
    replay_payloads: Sequence[Mapping[str, Any]],
    *,
    expected_examples: int = 4,
    capture_receipt: Mapping[str, Any] | None = None,
    replay_receipts: Sequence[Mapping[str, Any]] | None = None,
    authority_secret: bytes | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    canonical_captures: list[Mapping[str, Any]] = []
    for index, capture in enumerate(captures):
        try:
            canonical_captures.append(validate_capture_record(capture))
        except Exception as error:
            errors.append(f"capture_invalid:{index}:{error}")
    captures_by_id = {str(row.get("example_id")): row for row in captures}
    if len(captures) != expected_examples or len(captures_by_id) != expected_examples:
        errors.append("capture_bijection_failed")
    if authority_secret is None or capture_receipt is None:
        errors.append("parent_receipt_capture_missing")
    elif canonical_captures:
        try:
            capture_execution = canonical_captures[0]["execution"]
            if any(
                canonical_json(row["execution"]) != canonical_json(capture_execution)
                for row in canonical_captures[1:]
            ):
                raise ValueError("capture rows do not share one process execution")
            validate_parent_launch_receipt(
                capture_receipt,
                authority_secret=authority_secret,
                artifact_payload=list(canonical_captures),
                child_evidence=capture_execution,
                child_kind="smsb_capture",
                child_identity="capture4",
            )
        except Exception as error:
            errors.append(f"parent_receipt_capture_invalid:{error}")
    if canonical_captures:
        capture_processes = {
            row.get("execution", {}).get("process_instance_uuid") for row in canonical_captures
        }
        capture_engines = {row.get("engine_id") for row in canonical_captures}
        process_generate_counts = {
            row.get("execution", {}).get("generate_call_count") for row in canonical_captures
        }
        call_indices = sorted(
            [
                turn["generate_call_index"]
                for row in canonical_captures
                for turn in row["memory_ledger"]
            ]
            + [
                call
                for row in canonical_captures
                for call in (
                    row["final_stochastic_generate_call_index"],
                    row["final_control_generate_call_index"],
                )
            ]
        )
        if len(capture_processes) != 1 or len(capture_engines) != 1:
            errors.append("capture_not_one_process_one_engine")
        if len(process_generate_counts) != 1:
            errors.append("capture_generate_count_inconsistent")
        else:
            process_generate_count = next(iter(process_generate_counts))
            if type(process_generate_count) is not int or call_indices != list(
                range(1, process_generate_count + 1)
            ):
                errors.append("capture_generate_call_schedule_not_bijective")
    expected_replays = expected_examples * len(SMSB_REGIMES)
    if len(replay_payloads) != expected_replays:
        errors.append(f"replay_count:{len(replay_payloads)}:expected_{expected_replays}")
    receipts_by_identity: dict[str, Mapping[str, Any]] = {}
    if replay_receipts is None or authority_secret is None:
        errors.append("parent_receipt_replays_missing")
    else:
        receipts_by_identity = {
            str(receipt.get("child_identity")): receipt for receipt in replay_receipts
        }
        if (
            len(replay_receipts) != expected_replays
            or len(receipts_by_identity) != expected_replays
        ):
            errors.append("parent_receipt_replay_bijection_failed")
        parent_launcher_pids = [
            capture_receipt.get("parent_launcher_pid")
            if capture_receipt is not None
            else None,
            *(receipt.get("parent_launcher_pid") for receipt in replay_receipts),
        ]
        if (
            any(type(value) is not int or value < 1 for value in parent_launcher_pids)
            or len(set(parent_launcher_pids)) != expected_replays + 1
        ):
            errors.append("parent_launcher_pid_not_unique")
    engine_ids = [item.get("result", {}).get("engine_id") for item in replay_payloads]
    cache_ids = [item.get("result", {}).get("cache_namespace") for item in replay_payloads]
    process_uuids = [
        item.get("result", {}).get("process_instance_uuid") for item in replay_payloads
    ]
    process_pids = [item.get("result", {}).get("process_pid") for item in replay_payloads]
    credential_ids = [
        item.get("result", {}).get("parent_credential_id") for item in replay_payloads
    ]
    if len(set(engine_ids)) != len(engine_ids):
        errors.append("replay_engine_not_unique")
    if len(set(cache_ids)) != len(cache_ids):
        errors.append("replay_cache_not_unique")
    if None in process_uuids or len(set(process_uuids)) != len(process_uuids):
        errors.append("replay_process_instance_not_unique")
    if None in process_pids or len(set(process_pids)) != len(process_pids):
        errors.append("replay_process_pid_not_unique")
    if None in credential_ids or len(set(credential_ids)) != len(credential_ids):
        errors.append("replay_parent_credential_not_unique")
    capture_pids = {
        row.get("execution", {}).get("process_pid") for row in canonical_captures
    }
    if capture_pids & set(process_pids):
        errors.append("capture_process_pid_reused")
    if set(engine_ids) & {row.get("engine_id") for row in captures}:
        errors.append("capture_engine_reused")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    l2_values: list[bool] = []
    for payload in replay_payloads:
        capture_id = payload.get("capture_id")
        matching_captures = [
            row for row in canonical_captures if row.get("capture_id") == capture_id
        ]
        if len(matching_captures) != 1:
            errors.append(f"replay_capture_id_invalid:{capture_id}")
            continue
        try:
            validation = validate_replay(
                matching_captures[0], payload.get("request", {}), payload.get("result", {})
            )
        except Exception as error:
            errors.append(f"replay_revalidation_failed:{capture_id}:{error}")
            continue
        if canonical_json(validation) != canonical_json(payload.get("validation", {})):
            errors.append(f"persisted_replay_validation_mismatch:{capture_id}")
        key = (str(validation.get("example_id", "")), str(validation.get("regime", "")))
        child_identity = f"{key[0]}::{key[1]}"
        receipt = receipts_by_identity.get(child_identity)
        if receipt is None:
            errors.append(f"parent_receipt_replay_missing:{child_identity}")
        else:
            try:
                validate_parent_launch_receipt(
                    receipt,
                    authority_secret=authority_secret,
                    artifact_payload=payload,
                    child_evidence=payload.get("result", {}),
                    child_kind="smsb_replay",
                    child_identity=child_identity,
                )
            except Exception as error:
                errors.append(
                    f"parent_receipt_replay_invalid:{child_identity}:{error}"
                )
        if key in by_key:
            errors.append(f"duplicate_replay:{key[0]}:{key[1]}")
        by_key[key] = validation
    for example_id in sorted(captures_by_id):
        for regime in SMSB_REGIMES:
            validation = by_key.get((example_id, regime))
            if validation is None:
                errors.append(f"missing_replay:{example_id}:{regime}")
                continue
            if validation.get("execution_valid") is not True:
                errors.append(f"execution_invalid:{example_id}:{regime}")
            if regime == "temperature_zero" and validation.get("E_det_gate_pass") is not True:
                errors.append(f"E_det_fail:{example_id}")
            if regime == "matched_seed":
                l2_values.append(validation.get("L2_matched_seed_answer_identity") is True)
    gate_prefixes = (
        "capture_",
        "replay_count",
        "replay_engine",
        "replay_cache",
        "replay_process",
        "replay_parent_credential",
        "capture_engine",
        "capture_process_pid",
        "replay_capture_id",
        "replay_revalidation",
        "persisted_replay_validation",
        "parent_receipt",
        "parent_launcher_pid",
        "duplicate_replay",
        "missing_replay",
        "execution_invalid",
        "E_det_fail",
    )
    e_det = not any(error.startswith(gate_prefixes) for error in errors)
    return {
        "schema": "memagent.smsb.pilot4.adjudication.v2",
        "status": "PASS" if e_det else "FAIL",
        "decision": "PASS_E_DET_SINGLE_REQUEST" if e_det else "NO_GO_SMSB",
        "capture_count": len(captures),
        "replay_count": len(replay_payloads),
        "observed_examples": len(captures_by_id),
        "E_det_pass": e_det,
        "L2_exact_count_report_only": sum(l2_values),
        "L2_total_report_only": len(l2_values),
        "L2_exact_rate_report_only": (
            sum(l2_values) / len(l2_values) if l2_values else None
        ),
        "L2_report_only": True,
        "single_request_only": True,
        "batched_executor_forbidden": True,
        "training_updates": 0,
        "errors": sorted(set(errors)),
        "claim_boundary": (
            "fixed_checkpoint_protocol_template_temperature_zero_reader_execution_closure_only"
        ),
    }


def split_documents(context: str) -> list[dict[str, Any]]:
    matches = list(re.finditer(r"Document\s+(\d+):\n", context))
    documents: list[dict[str, Any]] = []
    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(context)
        body = context[start:end].strip()
        title, _, text = body.partition("\n")
        documents.append(
            {
                "number": int(match.group(1)),
                "title": title.strip(),
                "text": text.strip(),
            }
        )
    if not documents:
        raise ValueError("Hotpot context contains no numbered documents")
    return documents


_STOP_WORDS = {
    "what", "which", "when", "where", "with", "that", "this", "from", "until",
    "were", "was", "are", "had", "have", "into", "about", "whose", "included",
    "album", "company", "american", "party", "power", "year", "sold", "based",
}


def content_words(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 4 and token not in _STOP_WORDS
    }


def best_length_derangement(
    lengths: Mapping[str, int], *, maximum_caliper: int
) -> dict[str, str]:
    maximum_caliper = require_int(maximum_caliper, "maximum_caliper", minimum=0)
    normalized_lengths = {
        key: require_int(value, f"lengths.{key}", minimum=0) for key, value in lengths.items()
    }
    ids = sorted(normalized_lengths)
    candidates: list[tuple[int, tuple[str, ...]]] = []
    for permutation in itertools.permutations(ids):
        if any(target == donor for target, donor in zip(ids, permutation)):
            continue
        deltas = [
            abs(normalized_lengths[target] - normalized_lengths[donor])
            for target, donor in zip(ids, permutation)
        ]
        if max(deltas, default=0) > maximum_caliper:
            continue
        candidates.append((sum(deltas), permutation))
    if not candidates:
        raise ValueError("frozen pilot has no perfect derangement within the token-length caliper")
    _, chosen = min(candidates)
    return dict(zip(ids, chosen))


def _validate_authoring_row(row: Mapping[str, Any]) -> None:
    required = {
        *STABLE_FIELDS,
        "question_token_ids",
        "ground_truth",
        "question_type",
        "answer_type",
        "checkpoint_hash",
        "model_hash",
        "tokenizer_hash",
        "hashes",
        "vllm_version",
        "runtime_binding_sha256",
        "engine_config_sha256",
        "current_binding_sha256",
        "prompt_protocol_hash",
        "prompt_outside_memory_span_hash",
        "physical_gpu_identity",
        "cuda_device_order",
        "generated",
        "empty",
        "irrelevant",
        "gold",
        "shuffle_approved_donor_ids",
        "shuffle_memory_token_delta",
        "generated_memory_token_length",
        "gold_memory_token_length",
        "irrelevant_memory_token_length",
        "full_model_sha_verified_at_tetrad_start",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Tetrad authoring row missing fields: {missing}")
    _stable_identity(row)
    if row.get("full_model_sha_verified_at_tetrad_start") is not True:
        raise ValueError("Tetrad start did not certify a full model SHA verification")
    for field in (
        "checkpoint_hash", "model_hash", "tokenizer_hash", "runtime_binding_sha256",
        "engine_config_sha256", "current_binding_sha256", "prompt_protocol_hash",
        "prompt_outside_memory_span_hash",
    ):
        require_sha256(row[field], field)
    hashes = row.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("Tetrad authoring hashes must be an object")
    for field in PROVENANCE_HASHES:
        require_sha256(hashes.get(field), f"hashes.{field}")
    if row.get("vllm_version") != FROZEN_VLLM_VERSION:
        raise ValueError(f"Tetrad authoring requires vLLM {FROZEN_VLLM_VERSION}")
    if (
        not isinstance(row.get("physical_gpu_identity"), list)
        or len(row["physical_gpu_identity"]) != 2
        or any(not isinstance(value, str) or not value for value in row["physical_gpu_identity"])
        or row.get("cuda_device_order") != "PCI_BUS_ID"
    ):
        raise ValueError("Tetrad authoring physical GPU identity/order is invalid")
    token_ids(row.get("question_token_ids"), "question_token_ids", allow_empty=False)
    ground_truth = row.get("ground_truth")
    if (
        not isinstance(ground_truth, list)
        or not ground_truth
        or any(not isinstance(answer, str) or not answer.strip() for answer in ground_truth)
    ):
        raise ValueError("Tetrad authoring ground_truth must contain non-empty strings")
    state_token_ids: dict[str, list[int]] = {}
    for role in ("generated", "empty", "irrelevant", "gold"):
        state = row[role]
        if not isinstance(state, Mapping):
            raise ValueError(f"{role} state must be an object")
        if not isinstance(state.get("state_id"), str) or not state["state_id"]:
            raise ValueError(f"{role} state_id must be a non-empty string")
        if state.get("validity_status") != "pass":
            raise ValueError(f"{role} state validity did not pass")
        state_token_ids[role] = token_ids(
            state.get("memory_token_ids", []),
            f"{role}.memory_token_ids",
            allow_empty=role == "empty",
        )
    if state_token_ids["empty"]:
        raise ValueError("empty Tetrad state must contain zero tokens")
    if len(state_token_ids["irrelevant"]) != len(state_token_ids["generated"]):
        raise ValueError("irrelevant state is not token-length matched to generated state")
    for field, role in (
        ("generated_memory_token_length", "generated"),
        ("gold_memory_token_length", "gold"),
        ("irrelevant_memory_token_length", "irrelevant"),
    ):
        if require_int(row.get(field), field, minimum=0) != len(state_token_ids[role]):
            raise ValueError(f"{field} differs from the actual token array")
    approved = row.get("shuffle_approved_donor_ids")
    if (
        not isinstance(approved, list)
        or len(approved) != 1
        or not isinstance(approved[0], str)
        or not approved[0]
        or approved[0] == str(row["example_id"])
    ):
        raise ValueError("shuffle_approved_donor_ids must name one distinct example")
    require_int(row.get("shuffle_memory_token_delta"), "shuffle_memory_token_delta", minimum=0)
    if row["generated"].get("smsb_status") != "pass":
        raise ValueError("generated state does not carry SMSB PASS")
    if row["irrelevant"].get("support_answer_bridge_leakage_audit") != "pass":
        raise ValueError("irrelevant state leakage audit did not pass")
    if row["irrelevant"].get("length_match_audit") != "pass":
        raise ValueError("irrelevant state length audit did not pass")
    if row["gold"].get("canonical_authoring_audit") != "pass":
        raise ValueError("gold state authoring audit did not pass")


def build_tetrad_requests(
    authoring_rows: Sequence[Mapping[str, Any]],
    *,
    matching: Mapping[str, str],
    base_seed: int,
    prompt_builder: Any,
    prompt_template_sha256: str,
    capture_prompt_ids: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    """Build the authoritative five-state deterministic construction pilot."""
    base_seed = require_int(base_seed, "base_seed", minimum=0)
    if len(authoring_rows) != 4:
        raise ValueError("Tetrad pilot requires exactly four authoring rows")
    for row in authoring_rows:
        _validate_authoring_row(row)
    by_id = {str(row["example_id"]): row for row in authoring_rows}
    if len(by_id) != 4 or set(matching) != set(by_id) or set(matching.values()) != set(by_id):
        raise ValueError("shuffle matching is not a four-example bijection")
    if any(target == donor for target, donor in matching.items()):
        raise ValueError("shuffle matching contains a self-match")
    matching_hash = canonical_sha256(dict(sorted(matching.items())))
    requests: list[dict[str, Any]] = []
    for example_id in sorted(by_id):
        row = by_id[example_id]
        donor_id = matching[example_id]
        donor = by_id[donor_id]
        if row["shuffle_approved_donor_ids"] != [donor_id]:
            raise ValueError(f"shuffle matching differs from approved donor for {example_id}")
        observed_delta = abs(
            len(row["generated"]["memory_token_ids"])
            - len(donor["generated"]["memory_token_ids"])
        )
        if row["shuffle_memory_token_delta"] != observed_delta:
            raise ValueError(f"shuffle token-length delta differs for {example_id}")
        role_states = {
            "generated": (example_id, row["generated"]),
            "empty": (example_id, row["empty"]),
            "irrelevant": (example_id, row["irrelevant"]),
            "shuffle": (donor_id, donor["generated"]),
            "gold": (example_id, row["gold"]),
        }
        seed_payload = f"{base_seed}:{example_id}:deterministic".encode("utf-8")
        request_seed = int.from_bytes(
            hashlib.blake2b(seed_payload, digest_size=8).digest(), "little"
        ) % (2**63 - 1)
        for role in TETRAD_ROLES:
            source_id, state = role_states[role]
            memory_ids = token_ids(state["memory_token_ids"], f"{role}.memory_token_ids")
            prompt_ids = token_ids(
                prompt_builder(row["question_token_ids"], memory_ids),
                "expected_prompt_token_ids",
                allow_empty=False,
            )
            if role == "generated" and prompt_ids != list(capture_prompt_ids[example_id]):
                raise ValueError(f"generated-state prompt differs from SMSB L0 for {example_id}")
            request_id = f"{example_id}::deterministic::{role}"
            requests.append(
                {
                    "schema": TETRAD_SCHEMA,
                    "request_id": request_id,
                    **{field: row[field] for field in STABLE_FIELDS},
                    "state_role": role,
                    "memory_source_example_id": source_id,
                    "state_id": state["state_id"],
                    "memory_token_ids": memory_ids,
                    "memory_token_hash": canonical_sha256(memory_ids),
                    "memory_span_token_length": len(memory_ids),
                    "question_token_ids": list(row["question_token_ids"]),
                    "ground_truth": list(row["ground_truth"]),
                    "question_type": row["question_type"],
                    "answer_type": row["answer_type"],
                    "checkpoint_hash": row["checkpoint_hash"],
                    "model_hash": row["model_hash"],
                    "tokenizer_hash": row["tokenizer_hash"],
                    "prompt_protocol_hash": row["prompt_protocol_hash"],
                    "prompt_outside_memory_span_hash": row[
                        "prompt_outside_memory_span_hash"
                    ],
                    "physical_gpu_identity": list(row["physical_gpu_identity"]),
                    "cuda_device_order": row["cuda_device_order"],
                    "hashes": dict(row["hashes"]),
                    "vllm_version": str(row["vllm_version"]),
                    "runtime_binding_sha256": require_sha256(
                        row["runtime_binding_sha256"], "runtime_binding_sha256"
                    ),
                    "engine_config_sha256": require_sha256(
                        row["engine_config_sha256"], "engine_config_sha256"
                    ),
                    "current_binding_sha256": require_sha256(
                        row["current_binding_sha256"], "current_binding_sha256"
                    ),
                    "prompt_construction": "recurrent.TokenTemplate.token_concat.v1",
                    "prompt_template_sha256": require_sha256(
                        prompt_template_sha256, "prompt_template_sha256"
                    ),
                    "expected_prompt_token_ids": prompt_ids,
                    "expected_prompt_token_sha256": canonical_sha256(prompt_ids),
                    "decoder_regime": "deterministic",
                    "request_seed": request_seed,
                    "seed_semantics": "deterministic_explicit",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "cache_namespace": canonical_sha256(
                        {"request_id": request_id, "cache": "isolated"}
                    ),
                    "prefix_cache_allowed": False,
                    "recurrent_updater_calls": 0,
                    "fresh_engine_required": True,
                    "single_request_only": True,
                    "max_num_seqs": 1,
                    "append_only_output_required": True,
                    "shuffle_derangement_manifest_hash": matching_hash,
                    "smsb_L0_source_prompt_identity": (
                        True if role == "generated" else "not_applicable_replacement_arm"
                    ),
                    "construction_only_pilot": True,
                    "effects_reportable": False,
                }
            )
    validate_tetrad_manifest(requests)
    return requests


def validate_tetrad_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 20:
        raise ValueError(f"Tetrad pilot requires 20 requests, got {len(rows)}")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    request_ids: set[str] = set()
    cache_ids: set[str] = set()
    matching: dict[str, str] = {}
    matching_hashes: set[str] = set()
    for row in rows:
        required = {
            *STABLE_FIELDS,
            "request_id", "example_id", "state_role", "memory_source_example_id",
            "memory_token_ids", "memory_token_hash", "checkpoint_hash", "model_hash",
            "tokenizer_hash", "prompt_protocol_hash", "prompt_outside_memory_span_hash",
            "question_token_ids", "ground_truth", "prompt_template_sha256",
            "expected_prompt_token_ids", "expected_prompt_token_sha256", "decoder_regime",
            "request_seed", "cache_namespace", "prefix_cache_allowed",
            "recurrent_updater_calls", "fresh_engine_required", "single_request_only",
            "max_num_seqs", "append_only_output_required",
            "shuffle_derangement_manifest_hash", "effects_reportable", "hashes",
            "vllm_version", "runtime_binding_sha256", "engine_config_sha256",
            "current_binding_sha256",
            "physical_gpu_identity", "cuda_device_order",
            "prompt_construction", "seed_semantics", "construction_only_pilot",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Tetrad request missing fields: {missing}")
        _stable_identity(row)
        token_ids(row["question_token_ids"], "question_token_ids", allow_empty=False)
        if (
            not isinstance(row["ground_truth"], list)
            or not row["ground_truth"]
            or any(not isinstance(value, str) or not value.strip() for value in row["ground_truth"])
            or canonical_sha256(row["ground_truth"]) != row["ground_truth_hash"]
        ):
            raise ValueError("Tetrad request ground truth/hash binding is invalid")
        request_id = str(row["request_id"])
        if request_id in request_ids:
            raise ValueError("duplicate Tetrad request_id")
        request_ids.add(request_id)
        cache = str(row["cache_namespace"])
        if cache in cache_ids:
            raise ValueError("Tetrad cache namespaces must be unique")
        cache_ids.add(cache)
        role = str(row["state_role"])
        if role not in TETRAD_ROLES:
            raise ValueError(f"unsupported Tetrad role: {role}")
        if any(
            (
                row["decoder_regime"] != "deterministic",
                require_finite_number(row.get("temperature"), "temperature") != 0.0,
                require_finite_number(row.get("top_p"), "top_p", minimum=0.0, maximum=1.0) != 1.0,
                require_int(row.get("top_k"), "top_k", minimum=-1) != -1,
                row["prefix_cache_allowed"] is not False,
                require_int(row["recurrent_updater_calls"], "recurrent_updater_calls", minimum=0) != 0,
                row["fresh_engine_required"] is not True,
                row["single_request_only"] is not True,
                require_int(row["max_num_seqs"], "max_num_seqs", minimum=1) != 1,
                row["append_only_output_required"] is not True,
                row["effects_reportable"] is not False,
                row["construction_only_pilot"] is not True,
                row["prompt_construction"]
                != "recurrent.TokenTemplate.token_concat.v1",
                row["seed_semantics"] != "deterministic_explicit",
            )
        ):
            raise ValueError("Tetrad execution isolation contract drifted")
        ids = token_ids(row["memory_token_ids"], "memory_token_ids")
        if require_int(
            row.get("memory_span_token_length"), "memory_span_token_length", minimum=0
        ) != len(ids):
            raise ValueError("Tetrad memory length mismatch")
        if canonical_sha256(ids) != row["memory_token_hash"]:
            raise ValueError("Tetrad memory hash mismatch")
        prompt_ids = token_ids(row["expected_prompt_token_ids"], "expected_prompt_token_ids")
        if canonical_sha256(prompt_ids) != row["expected_prompt_token_sha256"]:
            raise ValueError("Tetrad prompt hash mismatch")
        require_int(row["request_seed"], "request_seed", minimum=0)
        if row.get("vllm_version") != FROZEN_VLLM_VERSION:
            raise ValueError(f"Tetrad requires vLLM {FROZEN_VLLM_VERSION}")
        if (
            not isinstance(row.get("physical_gpu_identity"), list)
            or len(row["physical_gpu_identity"]) != 2
            or any(
                not isinstance(value, str) or not value
                for value in row["physical_gpu_identity"]
            )
            or row.get("cuda_device_order") != "PCI_BUS_ID"
        ):
            raise ValueError("Tetrad request physical GPU identity/order is invalid")
        for field in (
            "checkpoint_hash", "model_hash", "tokenizer_hash", "prompt_protocol_hash",
            "prompt_outside_memory_span_hash", "runtime_binding_sha256",
            "engine_config_sha256", "current_binding_sha256",
        ):
            require_sha256(row[field], field)
        if not isinstance(row.get("hashes"), Mapping):
            raise ValueError("Tetrad hashes must be an object")
        for field in PROVENANCE_HASHES:
            require_sha256(row["hashes"].get(field), f"hashes.{field}")
        groups[str(row["example_id"])].append(row)
        matching_hashes.add(str(row["shuffle_derangement_manifest_hash"]))
        if role == "shuffle":
            matching[str(row["example_id"])] = str(row["memory_source_example_id"])
        elif str(row["memory_source_example_id"]) != str(row["example_id"]):
            raise ValueError("non-shuffle role uses another example's state")
    if len(groups) != 4:
        raise ValueError("Tetrad pilot must contain four examples")
    same_fields = (
        "checkpoint_hash", "model_hash", "tokenizer_hash", "prompt_protocol_hash",
        "prompt_outside_memory_span_hash", "decoder_regime", "request_seed",
        "shuffle_derangement_manifest_hash", "hashes", "vllm_version",
        "runtime_binding_sha256", "engine_config_sha256", "current_binding_sha256",
        "physical_gpu_identity", "cuda_device_order", "ground_truth",
    )
    for example_id, group in groups.items():
        if len(group) != 5 or {str(row["state_role"]) for row in group} != set(TETRAD_ROLES):
            raise ValueError(f"{example_id} does not contain exactly five roles")
        for field in same_fields:
            if len({canonical_json(row[field]) for row in group}) != 1:
                raise ValueError(f"{example_id} differs in frozen field {field}")
        non_memory_projection = {
            canonical_sha256(
                {
                    "question_token_ids": row["question_token_ids"],
                    "prompt_outside_memory_span_hash": row[
                        "prompt_outside_memory_span_hash"
                    ],
                    "prompt_template_sha256": row["prompt_template_sha256"],
                }
            )
            for row in group
        }
        if len(non_memory_projection) != 1:
            raise ValueError(f"{example_id} differs outside the memory intervention span")
    if len(matching_hashes) != 1:
        raise ValueError("Tetrad requests do not share one frozen derangement")
    if set(matching) != set(groups) or set(matching.values()) != set(groups):
        raise ValueError("shuffle donors do not form a one-to-one derangement")
    if any(target == donor for target, donor in matching.items()):
        raise ValueError("shuffle donor self-match")
    return {
        "status": "PASS",
        "decision": "TETRAD_REQUEST_MANIFEST_GATE_PASS",
        "request_count": len(rows),
        "example_count": len(groups),
        "roles_per_example": 5,
        "unique_cache_namespace_count": len(cache_ids),
        "shuffle_derangement_valid": True,
        "effects_reportable": False,
        "claim_boundary": "construction_only_no_fresh_engine_behavior_or_utilization_effect",
    }


def adjudicate_tetrad_pilot(
    manifest_rows: Sequence[Mapping[str, Any]],
    authoring_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    answer_decoder: Any,
    parent_receipts: Sequence[Mapping[str, Any]] | None = None,
    authority_secret: bytes | None = None,
    competence_score_threshold: float = 0.5,
    competence_rate_floor: float = 0.75,
) -> dict[str, Any]:
    competence_score_threshold = require_finite_number(
        competence_score_threshold,
        "competence_score_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    competence_rate_floor = require_finite_number(
        competence_rate_floor,
        "competence_rate_floor",
        minimum=0.0,
        maximum=1.0,
    )
    manifest_gate = validate_tetrad_manifest(manifest_rows)
    if len(authoring_rows) != 4:
        raise ValueError("Tetrad authoring ledger must have four rows")
    for row in authoring_rows:
        _validate_authoring_row(row)
    if len(result_rows) != 20:
        raise ValueError(f"Tetrad pilot requires 20 execution results, got {len(result_rows)}")
    if authority_secret is None or parent_receipts is None:
        raise ValueError("Tetrad requires authenticated parent launch receipts")
    receipts_by_identity = {
        str(receipt.get("child_identity")): receipt for receipt in parent_receipts
    }
    if len(parent_receipts) != 20 or len(receipts_by_identity) != 20:
        raise ValueError("Tetrad parent receipt/request bijection failed")
    parent_launcher_pids = [
        receipt.get("parent_launcher_pid") for receipt in parent_receipts
    ]
    if (
        any(type(value) is not int or value < 1 for value in parent_launcher_pids)
        or len(set(parent_launcher_pids)) != 20
    ):
        raise ValueError(
            "each Tetrad request must use a distinct parent supervisor PID"
        )
    manifest_by_id = {str(row["request_id"]): row for row in manifest_rows}
    results_by_id = {str(row.get("request_id")): row for row in result_rows}
    if len(results_by_id) != 20 or set(results_by_id) != set(manifest_by_id):
        raise ValueError("Tetrad result/manifest request bijection failed")
    if len({row.get("engine_id") for row in result_rows}) != 20:
        raise ValueError("each Tetrad request must use a distinct fresh engine")
    if len({row.get("cache_namespace") for row in result_rows}) != 20:
        raise ValueError("each Tetrad request must use a distinct cache namespace")
    process_uuids = [row.get("process_instance_uuid") for row in result_rows]
    if None in process_uuids or len(set(process_uuids)) != 20:
        raise ValueError("each Tetrad request must use a distinct Python process instance")
    try:
        import uuid

        for value in process_uuids:
            if not isinstance(value, str):
                raise ValueError
            uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("Tetrad process instance UUID is invalid") from error
    process_pids = [row.get("process_pid") for row in result_rows]
    if (
        any(type(value) is not int or value < 1 for value in process_pids)
        or len(set(process_pids)) != 20
    ):
        raise ValueError("each Tetrad request must use a distinct child process PID")
    credential_ids = [row.get("parent_credential_id") for row in result_rows]
    if (
        any(not isinstance(value, str) for value in credential_ids)
        or len(set(credential_ids)) != 20
    ):
        raise ValueError("each Tetrad request must use a distinct parent credential")
    if not callable(answer_decoder):
        raise ValueError("Tetrad adjudication requires a token-native answer decoder")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    verified_scores: dict[str, float] = {}
    for request_id, result in results_by_id.items():
        request = manifest_by_id[request_id]
        receipt = receipts_by_identity.get(request_id)
        if receipt is None:
            raise ValueError(f"Tetrad parent receipt is missing for {request_id}")
        validate_parent_launch_receipt(
            receipt,
            authority_secret=authority_secret,
            artifact_payload=result,
            child_evidence=result,
            child_kind="tetrad_replay",
            child_identity=request_id,
        )
        prompt_ids = token_ids(result.get("prompt_token_ids", []), "result.prompt_token_ids")
        answer_ids = token_ids(result.get("answer_token_ids", []), "result.answer_token_ids")
        physical_gpus = result.get("physical_gpu_whitelist")
        hard = all(
            (
                result.get("fresh_engine_verified") is True,
                result.get("single_request_execution_verified") is True,
                type(result.get("tensor_parallel_size")) is int
                and result.get("tensor_parallel_size") == 2,
                isinstance(physical_gpus, list)
                and all(type(value) is int for value in physical_gpus)
                and len(physical_gpus) == 2
                and physical_gpus[0] < physical_gpus[1],
                type(result.get("max_num_seqs")) is int
                and result.get("max_num_seqs") == 1,
                result.get("prefix_cache_enabled") is False,
                type(result.get("observed_updater_calls")) is int
                and result.get("observed_updater_calls") == 0,
                result.get("context_or_chunks_visible") is False,
                result.get("prompt_reconstructed_from_question_and_serialized_memory") is True,
                result.get("prompt_token_sha256") == request["expected_prompt_token_sha256"],
                result.get("example_id") == request["example_id"],
                result.get("state_role") == request["state_role"],
                result.get("cache_namespace") == request["cache_namespace"],
                result.get("construction_only_pilot") is True,
                isinstance(result.get("process_instance_uuid"), str),
                type(result.get("process_pid")) is int and result.get("process_pid") > 0,
                type(result.get("engine_construction_count")) is int
                and result.get("engine_construction_count") == 1,
                type(result.get("generate_call_count")) is int
                and result.get("generate_call_count") == 1,
                isinstance(result.get("parent_credential_id"), str)
                and re.fullmatch(r"[0-9a-f]{64}", result["parent_credential_id"])
                is not None,
                isinstance(result.get("parent_credential_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", result["parent_credential_sha256"])
                is not None,
                type(result.get("parent_issuer_pid")) is int
                and result.get("parent_issuer_pid") > 0,
                type(result.get("observed_parent_pid")) is int
                and result.get("observed_parent_pid") == result.get("parent_issuer_pid"),
                type(result.get("configured_request_seed")) is int
                and result.get("configured_request_seed") == request["request_seed"],
                type(result.get("actual_request_seed")) is int
                and result.get("actual_request_seed") == request["request_seed"],
                result.get("vllm_version") == request["vllm_version"],
                result.get("hashes") == request["hashes"],
                result.get("runtime_binding_sha256") == request["runtime_binding_sha256"],
                result.get("engine_config_sha256") == request["engine_config_sha256"],
                result.get("current_binding_sha256") == request["current_binding_sha256"],
                result.get("physical_gpu_identity") == request["physical_gpu_identity"],
                result.get("cuda_device_order") == request["cuda_device_order"],
                result.get("full_model_sha_verified_at_child_start") is True,
                result.get("model_manifest_sha256") == request["hashes"]["model"],
                prompt_ids == request["expected_prompt_token_ids"],
                canonical_sha256(prompt_ids)
                == request["expected_prompt_token_sha256"],
                result.get("prompt_token_sha256")
                == canonical_sha256(prompt_ids),
                result.get("prompt_token_ids_sha256")
                == canonical_sha256(prompt_ids),
                result.get("answer_token_ids_sha256")
                == canonical_sha256(answer_ids),
                result.get("effects_reportable") is False,
            )
        )
        if not hard:
            raise ValueError(f"Tetrad execution certificate failed for {request_id}")
        decoded_answer = answer_decoder(answer_ids)
        if not isinstance(decoded_answer, str) or decoded_answer != result.get("answer_text"):
            raise ValueError(f"Tetrad answer text/token identity failed for {request_id}")
        from recurrent.research.s128_hotpot_metrics import score_terminal_output

        recomputed_metrics = score_terminal_output(decoded_answer, request["ground_truth"])
        score = require_finite_number(
            result.get("score"), f"{request_id}.score", minimum=0.0, maximum=1.0
        )
        exact_match = require_finite_number(
            result.get("exact_match"), f"{request_id}.exact_match", minimum=0.0, maximum=1.0
        )
        format_success = require_finite_number(
            result.get("format_success"),
            f"{request_id}.format_success",
            minimum=0.0,
            maximum=1.0,
        )
        if (
            score != float(recomputed_metrics["token_f1"])
            or exact_match != float(recomputed_metrics["exact_match"])
            or format_success != float(recomputed_metrics["format_success"])
            or result.get("extraction_route") != recomputed_metrics["extraction_route"]
        ):
            raise ValueError(f"Tetrad score differs from independent recomputation for {request_id}")
        verified_scores[request_id] = score
        grouped[str(result["example_id"])][str(result["state_role"])] = result
    if len(grouped) != 4 or any(set(group) != set(TETRAD_ROLES) for group in grouped.values()):
        raise ValueError("Tetrad execution role coverage is incomplete")
    competent = [
        example_id
        for example_id, group in grouped.items()
        if verified_scores[str(group["gold"]["request_id"])]
        >= competence_score_threshold
    ]
    competence_rate = len(competent) / 4
    passed = competence_rate >= competence_rate_floor
    return {
        "schema": "memagent.tetrad.pilot4.construction-adjudication.v2",
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "TETRAD_PILOT4_CONSTRUCTION_GATE_PASS"
            if passed
            else "NO_GO_TETRAD_CANONICAL_COMPETENCE"
        ),
        "manifest_gate": manifest_gate["decision"],
        "example_count": 4,
        "request_count": 20,
        "fresh_engine_count": 20,
        "single_request_only": True,
        "batched_executor_forbidden": True,
        "canonical_competence_threshold_f1": competence_score_threshold,
        "canonical_competence_rate_floor": competence_rate_floor,
        "canonical_competent_count": len(competent),
        "canonical_competence_rate": competence_rate,
        "canonical_gold_scores_by_example": {
            example_id: verified_scores[str(group["gold"]["request_id"])]
            for example_id, group in sorted(grouped.items())
        },
        "effects_reportable": False,
        "training_authorized": False,
        "method_selection_status": "PENDING_EVIDENCE_NO_SELECTION",
        "next_gate": "TETRAD_AUDIT32_NOT_RUN_BY_THIS_PILOT",
        "claim_boundary": "construction_and_execution_only_no_utilization_effect",
    }
