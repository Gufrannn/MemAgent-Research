"""MIC-v2 action-boundary credit and fixed-slot PPO primitives.

This module is deliberately independent from :mod:`recurrent.research.mic`,
which implements the historical MIC-v1 post-to-post pilot.  MIC-v2 treats a
complete writer sequence as the policy action and separates exogenous chunk
arrival from memory materialization.  All public constructors are fail-closed:
unknown state fields, non-nested histories, future/outcome fields, mutable
credits, or realized-token denominators are rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "memagent.mic.v2"
CONTRACT_SHA256 = "6dc99a253c0f3a39230ba93688c6270cb80efdd097774d301f5522dde9c4b3e4"
FOLD_RULE = "sha256_content_root_namespace_modulo_v2"
T_MAX = 8
GROUP_SIZE = 4
MATERIALIZATION_PARSER_VERSION = "memagent-unpad-remove-pad-and-eos-v1"

FORBIDDEN_STATE_KEYS = frozenset({
    "answer", "answer_text", "current_outcome", "dense_reward", "em",
    "exact_match", "final_answer", "future_chunk", "future_chunks", "gold",
    "gold_answer", "ground_truth", "label", "logprob", "next_chunk",
    "outcome", "raw_writer_completion", "reference_answer", "reward",
    "score", "token_f1", "writer_cache", "writer_hidden_state",
})

PUBLIC_METADATA_KEYS = frozenset({
    "arrived_context_token_count", "chunk_schedule_id", "exogenous_termination",
    "forced_truncation", "policy_termination", "prior_active_turn_count",
})

BOUNDARY_KEYS = frozenset({
    "schema", "phase", "content_root_id", "stable_example_id", "trajectory_id",
    "turn_index", "question", "arrived_chunks", "materialized_memory_history",
    "current_memory", "public_metadata", "state_sha256",
})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampled_policy_mask_receipt(
    *, token_ids: Sequence[int], termination: str,
    termination_token_ids: Sequence[int], token_width: int,
) -> dict[str, Any]:
    """Bind an unpadded vLLM completion to the actor's sampled-token mask.

    Training pads responses to ``token_width`` and includes the first sampled
    terminator in the policy mask.  Everything after that terminator is padding.
    Forced truncation contains no terminator and must fill the declared width.
    The compact receipt hashes the exact boolean prefix mask without storing it.
    """
    tokens = list(token_ids)
    terminators = list(termination_token_ids)
    if token_width < 1 or not terminators or len(set(terminators)) != len(terminators):
        raise ValueError("MIC_V2_NO_GO: sampled-mask authority is invalid")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0
           for token in tokens + terminators):
        raise ValueError("MIC_V2_NO_GO: sampled-mask token IDs are invalid")
    if not 0 < len(tokens) <= token_width:
        raise ValueError("MIC_V2_NO_GO: active completion length is invalid")
    if termination == "sampled_eos":
        if tokens[-1] not in terminators or any(
            token in terminators for token in tokens[:-1]
        ):
            raise ValueError("MIC_V2_NO_GO: sampled terminator is not first and final")
    elif termination == "forced_truncation":
        if len(tokens) != token_width or any(token in terminators for token in tokens):
            raise ValueError("MIC_V2_NO_GO: forced truncation has a terminator or wrong width")
    else:
        raise ValueError("MIC_V2_NO_GO: active completion termination is invalid")
    mask = [True] * len(tokens) + [False] * (token_width - len(tokens))
    return {
        "sampled_mask_width": token_width,
        "sampled_mask_true_count": len(tokens),
        "sampled_mask_sha256": sha256_json(mask),
    }


def materialized_memory_receipt(
    *, token_ids: Sequence[int], termination_token_ids: Sequence[int],
    content_root_id: str, trajectory_seed: int, turn_index: int,
    arrived_chunk_token_sha256: Sequence[str],
    prior_memory_token_sha256: Sequence[str],
) -> tuple[list[int], dict[str, Any]]:
    """Reconstruct MemoryAgent's declared post-write afterstate from raw tokens."""
    tokens = list(token_ids)
    terminators = set(termination_token_ids)
    is_hex64 = lambda value: (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
    if not is_hex64(content_root_id) \
            or isinstance(trajectory_seed, bool) or not isinstance(trajectory_seed, int) \
            or trajectory_seed < 0 \
            or isinstance(turn_index, bool) or not isinstance(turn_index, int) \
            or not 0 <= turn_index < T_MAX:
        raise ValueError("MIC_V2_NO_GO: materialization identity is invalid")
    if len(arrived_chunk_token_sha256) != turn_index + 1 \
            or len(prior_memory_token_sha256) != turn_index:
        raise ValueError("MIC_V2_NO_GO: afterstate history is not phase-nested")
    digests = list(arrived_chunk_token_sha256) + list(prior_memory_token_sha256)
    if any(not is_hex64(value) for value in digests):
        raise ValueError("MIC_V2_NO_GO: afterstate history digest is invalid")
    parsed = [token for token in tokens if token not in terminators]
    parsed_sha256 = sha256_json(parsed)
    afterstate = {
        "schema": "memagent.mic.v2.materialized-afterstate-v1",
        "content_root_id": content_root_id,
        "trajectory_seed": trajectory_seed,
        "turn_index": turn_index,
        "arrived_chunk_token_sha256": list(arrived_chunk_token_sha256),
        "materialized_memory_token_sha256": [
            *prior_memory_token_sha256, parsed_sha256,
        ],
        "current_memory_token_sha256": parsed_sha256,
        "parser_version": MATERIALIZATION_PARSER_VERSION,
    }
    return parsed, {
        "parsed_memory_token_ids": parsed,
        "parsed_memory_sha256": parsed_sha256,
        "parser_version": MATERIALIZATION_PARSER_VERSION,
        "afterstate_sha256": sha256_json(afterstate),
    }


def canonical_content_root(question: str, all_source_chunks: Sequence[str]) -> str:
    if not isinstance(question, str) or not question \
            or not isinstance(all_source_chunks, Sequence) \
            or isinstance(all_source_chunks, (str, bytes)) \
            or not all_source_chunks \
            or any(not isinstance(chunk, str) for chunk in all_source_chunks):
        raise ValueError("MIC_V2_NO_GO: canonical content identity inputs are invalid")
    return sha256_json({"question": question, "source_chunks": list(all_source_chunks)})


def validate_content_root(
    content_root_id: str, question: str, all_source_chunks: Sequence[str],
) -> None:
    if content_root_id != canonical_content_root(question, all_source_chunks):
        raise ValueError("MIC_V2_NO_GO: content root does not bind canonical source content")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"MIC_V2_NO_GO: {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"MIC_V2_NO_GO: {name} must be finite")
    return result


def _scan_forbidden(value: Any, path: str = "state") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_STATE_KEYS:
                raise ValueError(f"MIC_V2_NO_GO: forbidden state field {path}.{key}")
            _scan_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_forbidden(child, f"{path}[{index}]")


def validate_boundary_state(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one declared pre-write or post-write information state."""
    if not isinstance(record, Mapping):
        raise ValueError("MIC_V2_NO_GO: boundary state must be a mapping")
    _scan_forbidden(record)
    unknown = set(record) - BOUNDARY_KEYS
    if unknown:
        raise ValueError(f"MIC_V2_NO_GO: unknown boundary fields {sorted(unknown)}")
    required = BOUNDARY_KEYS - {"state_sha256"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"MIC_V2_NO_GO: missing boundary fields {sorted(missing)}")
    row = dict(record)
    if row["schema"] != SCHEMA:
        raise ValueError("MIC_V2_NO_GO: boundary schema mismatch")
    if row["phase"] not in ("pre_write", "post_write"):
        raise ValueError("MIC_V2_NO_GO: phase must be pre_write or post_write")
    for key in ("content_root_id", "stable_example_id", "trajectory_id", "question"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"MIC_V2_NO_GO: {key} must be non-empty text")
    if isinstance(row["turn_index"], bool) or not isinstance(row["turn_index"], int) \
            or row["turn_index"] < 1 or row["turn_index"] > T_MAX:
        raise ValueError(f"MIC_V2_NO_GO: turn_index must be in [1,{T_MAX}]")
    for key in ("arrived_chunks", "materialized_memory_history"):
        if not isinstance(row[key], list) or any(not isinstance(item, str) for item in row[key]):
            raise ValueError(f"MIC_V2_NO_GO: {key} must be a text list")
    if len(row["arrived_chunks"]) != row["turn_index"]:
        raise ValueError("MIC_V2_NO_GO: arrived history must end at the current turn")
    expected_memories = row["turn_index"] - (1 if row["phase"] == "pre_write" else 0)
    if len(row["materialized_memory_history"]) != expected_memories:
        raise ValueError("MIC_V2_NO_GO: materialized history is not phase-nested")
    if not isinstance(row["current_memory"], str):
        raise ValueError("MIC_V2_NO_GO: current_memory must be text")
    expected_current = row["materialized_memory_history"][-1] \
        if row["materialized_memory_history"] else ""
    if row["current_memory"] != expected_current:
        raise ValueError("MIC_V2_NO_GO: current_memory differs from history tail")
    if not isinstance(row["public_metadata"], Mapping):
        raise ValueError("MIC_V2_NO_GO: public_metadata must be a mapping")
    metadata_unknown = set(row["public_metadata"]) - PUBLIC_METADATA_KEYS
    if metadata_unknown:
        raise ValueError(
            f"MIC_V2_NO_GO: unknown public metadata fields {sorted(metadata_unknown)}"
        )
    for key, value in row["public_metadata"].items():
        if key in ("arrived_context_token_count", "prior_active_turn_count"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"MIC_V2_NO_GO: {key} must be a non-negative integer")
        elif key == "chunk_schedule_id":
            if not isinstance(value, str) or not value:
                raise ValueError("MIC_V2_NO_GO: chunk_schedule_id must be non-empty text")
        elif not isinstance(value, bool):
            raise ValueError(f"MIC_V2_NO_GO: {key} must be boolean")
    payload = {key: row[key] for key in sorted(required)}
    expected_hash = sha256_json(payload)
    supplied_hash = row.get("state_sha256")
    if supplied_hash is not None and supplied_hash != expected_hash:
        raise ValueError("MIC_V2_NO_GO: boundary state digest mismatch")
    row["state_sha256"] = expected_hash
    return row


def validate_boundary_pair(pre: Mapping[str, Any], post: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the action boundary to differ only by the new materialized memory."""
    before, after = validate_boundary_state(pre), validate_boundary_state(post)
    if before["phase"] != "pre_write" or after["phase"] != "post_write":
        raise ValueError("MIC_V2_NO_GO: boundary pair phases are reversed")
    invariant = (
        "content_root_id", "stable_example_id", "trajectory_id", "turn_index",
        "question", "arrived_chunks", "public_metadata",
    )
    for key in invariant:
        if before[key] != after[key]:
            raise ValueError(f"MIC_V2_NO_GO: boundary invariant drifted: {key}")
    if after["materialized_memory_history"][:-1] != before["materialized_memory_history"]:
        raise ValueError("MIC_V2_NO_GO: post-write history does not extend pre-write history")
    return before, after


def stable_fold(content_root_id: str, namespace: str, fold_count: int) -> int:
    if not isinstance(content_root_id, str) or not content_root_id:
        raise ValueError("MIC_V2_NO_GO: content_root_id must be non-empty")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("MIC_V2_NO_GO: fold namespace must be non-empty")
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError("MIC_V2_NO_GO: fold_count must be at least two")
    payload = canonical_json([content_root_id, namespace]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % fold_count


def _hash_mod(parts: Sequence[Any], modulus: int) -> int:
    if modulus < 1:
        raise ValueError("MIC_V2_NO_GO: hash modulus must be positive")
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def stable_fold_assignments(root_ids: Sequence[str], namespace: str, fold_count: int) -> dict[str, int]:
    unique = set(root_ids)
    if len(unique) < fold_count:
        raise ValueError("MIC_V2_NO_GO: fewer content roots than folds")
    return {root: stable_fold(root, namespace, fold_count) for root in sorted(unique)}


def group_centered_broadcast(returns: Sequence[float]) -> np.ndarray:
    values = np.asarray([_finite(value, "return") for value in returns], dtype=np.float64)
    if values.shape != (GROUP_SIZE,):
        raise ValueError(f"MIC_V2_NO_GO: exactly {GROUP_SIZE} replicas are required")
    return values - values.mean()


def sibling_reconstruction(returns: Sequence[float]) -> np.ndarray:
    values = np.asarray([_finite(value, "return") for value in returns], dtype=np.float64)
    if values.shape != (GROUP_SIZE,):
        raise ValueError(f"MIC_V2_NO_GO: exactly {GROUP_SIZE} replicas are required")
    c_g = (GROUP_SIZE - 1.0) / GROUP_SIZE
    return np.asarray([
        c_g * (value - np.delete(values, index).mean())
        for index, value in enumerate(values)
    ], dtype=np.float64)


def mechanism_cell_credits(
    *, returns: Sequence[float], mic_write_deltas: Sequence[float],
    postpost_deltas: Sequence[float], gate: Sequence[int],
) -> dict[str, dict[str, np.ndarray]]:
    """Construct the four primary cells from one exact role-routing producer."""
    broadcast = group_centered_broadcast(returns)
    mic = np.asarray([_finite(value, "mic_write_delta") for value in mic_write_deltas],
                     dtype=np.float64)
    postpost = np.asarray([_finite(value, "postpost_delta") for value in postpost_deltas],
                          dtype=np.float64)
    gate_values = np.asarray(gate)
    if mic.shape != broadcast.shape or postpost.shape != broadcast.shape \
            or gate_values.shape != broadcast.shape:
        raise ValueError("MIC_V2_NO_GO: mechanism credit rows are misaligned")
    if any(isinstance(value, bool) is False and not isinstance(value, (int, np.integer))
           for value in gate):
        raise ValueError("MIC_V2_NO_GO: gate must contain binary integers")
    if not np.all((gate_values == 0) | (gate_values == 1)):
        raise ValueError("MIC_V2_NO_GO: gate must contain only zero or one")
    c_g = (GROUP_SIZE - 1.0) / GROUP_SIZE
    mic_scaled = c_g * mic
    postpost_scaled = c_g * postpost
    gated = (1.0 - gate_values) * broadcast + gate_values * mic_scaled
    result = {
        "Broadcast-Ghost": {"writer": broadcast.copy(), "answer": broadcast.copy()},
        "PostPost-Ghost": {"writer": postpost_scaled, "answer": broadcast.copy()},
        "MIC-core": {"writer": mic_scaled, "answer": broadcast.copy()},
        "MIC-Gated": {"writer": gated, "answer": broadcast.copy()},
    }
    for row in result.values():
        row["writer"].setflags(write=False)
        row["answer"].setflags(write=False)
    return result


def expand_role_credits(
    *, writer_sequence_credits: Any, answer_sequence_credits: Any,
    writer_mask: Any, answer_mask: Any,
) -> Any:
    """Expand detached sequence scalars over exact disjoint sampled-token masks."""
    import torch

    values = (writer_sequence_credits, answer_sequence_credits, writer_mask, answer_mask)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError("MIC_V2_NO_GO: role routing inputs must be torch tensors")
    if writer_mask.dtype != torch.bool or answer_mask.dtype != torch.bool \
            or writer_mask.shape != answer_mask.shape:
        raise ValueError("MIC_V2_NO_GO: role routing masks are invalid")
    rows = writer_mask.shape[0]
    if writer_sequence_credits.shape != (rows,) or answer_sequence_credits.shape != (rows,):
        raise ValueError("MIC_V2_NO_GO: sequence credit rows are misaligned")
    if writer_sequence_credits.requires_grad or answer_sequence_credits.requires_grad:
        raise ValueError("MIC_V2_NO_GO: role sequence credits are not detached")
    writer_rows = torch.any(writer_mask, dim=-1)
    answer_rows = torch.any(answer_mask, dim=-1)
    if torch.any(writer_mask & answer_mask) or torch.any(writer_rows & answer_rows):
        raise ValueError("MIC_V2_NO_GO: writer and answer roles overlap")
    if torch.any(writer_sequence_credits.masked_select(~writer_rows) != 0) \
            or torch.any(answer_sequence_credits.masked_select(~answer_rows) != 0):
        raise ValueError("MIC_V2_NO_GO: role credit is present on the wrong action row")
    output = (
        writer_sequence_credits.unsqueeze(-1) * writer_mask
        + answer_sequence_credits.unsqueeze(-1) * answer_mask
    ).detach()
    return output


def scheduled_slot_count(root_count: int, group_size: int = GROUP_SIZE, t_max: int = T_MAX) -> int:
    values = (root_count, group_size, t_max)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("MIC_V2_NO_GO: slot dimensions must be positive integers")
    return root_count * group_size * (t_max + 1)


def sampled_token_masks(
    *, sampled_lengths: Sequence[int], roles: Sequence[str], token_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact sampled-token masks; inactive slots have length zero.

    A sampled EOS is part of ``sampled_lengths``.  A parser delimiter or a
    fictitious EOS after forced truncation is not.  This producer therefore
    never infers a token from a termination label.
    """
    if len(sampled_lengths) != len(roles) or token_width < 1:
        raise ValueError("MIC_V2_NO_GO: sampled slot metadata is misaligned")
    sampled = np.zeros((len(roles), token_width), dtype=np.bool_)
    writer = np.zeros_like(sampled)
    answer = np.zeros_like(sampled)
    for index, (length, role) in enumerate(zip(sampled_lengths, roles)):
        if isinstance(length, bool) or not isinstance(length, int) or length < 0 or length > token_width:
            raise ValueError("MIC_V2_NO_GO: sampled length is outside token storage")
        if role not in ("writer", "answer", "inactive"):
            raise ValueError("MIC_V2_NO_GO: unknown action-slot role")
        if role == "inactive" and length:
            raise ValueError("MIC_V2_NO_GO: inactive slot contains a fictitious action")
        sampled[index, :length] = True
        if role == "writer":
            writer[index, :length] = True
        elif role == "answer":
            answer[index, :length] = True
    return sampled, writer, answer


def action_slot_receipts(
    *, slots: Sequence[Mapping[str, Any]], token_width: int, eos_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Bind sampled tokens to role and termination semantics without fake EOS."""
    lengths, roles, receipts = [], [], []
    allowed_termination = {
        "sampled_eos", "forced_truncation", "policy_termination",
        "exogenous_termination",
    }
    for slot_index, slot in enumerate(slots):
        if set(slot) != {"role", "sampled_token_ids", "termination"}:
            raise ValueError("MIC_V2_NO_GO: action-slot receipt schema drifted")
        role = slot["role"]
        token_ids = slot["sampled_token_ids"]
        termination = slot["termination"]
        if not isinstance(token_ids, list) or any(
            isinstance(token, bool) or not isinstance(token, int) for token in token_ids
        ):
            raise ValueError("MIC_V2_NO_GO: sampled token IDs are invalid")
        if termination not in allowed_termination:
            raise ValueError("MIC_V2_NO_GO: unknown termination reason")
        if role == "inactive":
            if token_ids or termination not in ("policy_termination", "exogenous_termination"):
                raise ValueError("MIC_V2_NO_GO: inactive slot has a fictitious action")
        elif termination in ("policy_termination", "exogenous_termination"):
            raise ValueError("MIC_V2_NO_GO: absent-slot termination labels an active action")
        elif termination == "sampled_eos":
            if not token_ids or token_ids[-1] != eos_token_id:
                raise ValueError("MIC_V2_NO_GO: sampled EOS receipt has no terminal EOS token")
        elif termination == "forced_truncation" and token_ids and token_ids[-1] == eos_token_id:
            raise ValueError("MIC_V2_NO_GO: forced truncation inserted a fictitious EOS")
        lengths.append(len(token_ids))
        roles.append(role)
        receipt = {
            "slot_index": slot_index, "role": role, "termination": termination,
            "sampled_token_count": len(token_ids),
            "sampled_token_sha256": sha256_json(token_ids),
            "sampled_eos_counted": bool(token_ids and token_ids[-1] == eos_token_id),
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        receipts.append(receipt)
    sampled, writer, answer = sampled_token_masks(
        sampled_lengths=lengths, roles=roles, token_width=token_width,
    )
    return sampled, writer, answer, receipts


@dataclass(frozen=True)
class SlotLossReceipt:
    world_size: int
    global_scheduled_slots: int
    local_scheduled_slots: int
    active_writer_slots: int
    active_answer_slots: int
    sampled_writer_tokens: int
    sampled_answer_tokens: int
    reference_length: float
    reduction_mode: str
    pre_ddp_scale: float
    local_pg_numerator: float
    local_kl_numerator: float
    local_entropy_numerator: float
    local_loss: float

    def as_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["schema"] = SCHEMA
        result["receipt_sha256"] = sha256_json(result)
        return result


def fixed_slot_actor_loss(
    *,
    old_log_prob: Any,
    log_prob: Any,
    credits: Any,
    sampled_mask: Any,
    writer_mask: Any,
    answer_mask: Any,
    global_scheduled_slots: int,
    local_scheduled_slots: int,
    reference_length: float,
    clip_low: float,
    clip_high: float,
    world_size: int = 1,
    kl_token: Any | None = None,
    entropy_token: Any | None = None,
    beta_kl: float = 0.0,
    beta_entropy: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Return MIC-v2's globally fixed-slot actor loss and an audit receipt.

    ``credits`` must be detached.  DDP is assumed to average rank gradients;
    hence each equal logical shard receives the ``world_size`` multiplier.
    Realized token counts are recorded but never enter the denominator.
    """
    import torch

    tensors = (old_log_prob, log_prob, credits, sampled_mask, writer_mask, answer_mask)
    if any(not isinstance(item, torch.Tensor) for item in tensors):
        raise ValueError("MIC_V2_NO_GO: slot loss inputs must be torch tensors")
    if any(item.shape != log_prob.shape for item in tensors):
        raise ValueError("MIC_V2_NO_GO: slot loss tensor shapes differ")
    if old_log_prob.requires_grad:
        raise ValueError("MIC_V2_NO_GO: old log probabilities are not detached")
    if credits.requires_grad:
        raise ValueError("MIC_V2_NO_GO: actor credits are not detached")
    if sampled_mask.dtype != torch.bool or writer_mask.dtype != torch.bool \
            or answer_mask.dtype != torch.bool:
        raise ValueError("MIC_V2_NO_GO: action masks must be boolean")
    if not torch.isfinite(old_log_prob).all() or not torch.isfinite(log_prob).all() \
            or not torch.isfinite(credits).all():
        raise ValueError("MIC_V2_NO_GO: non-finite actor loss input")
    sampled = sampled_mask.bool()
    writer = writer_mask.bool()
    answer = answer_mask.bool()
    if torch.any(writer & answer) or not torch.equal(writer | answer, sampled):
        raise ValueError("MIC_V2_NO_GO: writer/answer masks do not partition sampled tokens")
    if torch.any(torch.any(writer, dim=-1) & torch.any(answer, dim=-1)):
        raise ValueError("MIC_V2_NO_GO: one action row spans writer and answer roles")
    if torch.any(credits.masked_select(~sampled) != 0):
        raise ValueError("MIC_V2_NO_GO: inactive/padding token carries actor credit")
    for row_index in range(credits.shape[0]):
        active_credit = credits[row_index].masked_select(sampled[row_index])
        if active_credit.numel() and not torch.equal(
            active_credit, active_credit[0].expand_as(active_credit)
        ):
            raise ValueError("MIC_V2_NO_GO: credit varies inside a sampled action sequence")
    if global_scheduled_slots < 1 or local_scheduled_slots < 0 or world_size < 1:
        raise ValueError("MIC_V2_NO_GO: invalid scheduled-slot receipt")
    if local_scheduled_slots * world_size != global_scheduled_slots:
        raise ValueError("MIC_V2_NO_GO: equal-shard scheduled slots do not reconstruct global slots")
    reference = _finite(reference_length, "reference_length")
    if reference <= 0:
        raise ValueError("MIC_V2_NO_GO: reference_length must be positive")
    low, high = _finite(clip_low, "clip_low"), _finite(clip_high, "clip_high")
    if low < 0 or high < 0:
        raise ValueError("MIC_V2_NO_GO: clip bounds must be non-negative")

    log_ratio = torch.where(sampled, log_prob - old_log_prob, torch.zeros_like(log_prob))
    ratio = torch.exp(log_ratio)
    unclipped = ratio * credits
    clipped = torch.clamp(ratio, 1.0 - low, 1.0 + high) * credits
    surrogate = torch.minimum(unclipped, clipped)
    pg_numerator = (surrogate * sampled).sum()
    kl_numerator = torch.zeros((), dtype=log_prob.dtype, device=log_prob.device)
    entropy_numerator = torch.zeros((), dtype=log_prob.dtype, device=log_prob.device)
    if kl_token is not None:
        if kl_token.shape != log_prob.shape:
            raise ValueError("MIC_V2_NO_GO: KL tensor shape differs")
        if not torch.isfinite(kl_token.masked_select(sampled)).all():
            raise ValueError("MIC_V2_NO_GO: active KL token is non-finite")
        kl_numerator = torch.where(sampled, kl_token, torch.zeros_like(kl_token)).sum()
    elif beta_kl:
        raise ValueError("MIC_V2_NO_GO: nonzero beta_kl without KL tokens")
    if entropy_token is not None:
        if entropy_token.shape != log_prob.shape:
            raise ValueError("MIC_V2_NO_GO: entropy tensor shape differs")
        if not torch.isfinite(entropy_token.masked_select(sampled)).all():
            raise ValueError("MIC_V2_NO_GO: active entropy token is non-finite")
        entropy_numerator = torch.where(
            sampled, entropy_token, torch.zeros_like(entropy_token)
        ).sum()
    elif beta_entropy:
        raise ValueError("MIC_V2_NO_GO: nonzero beta_entropy without entropy tokens")

    scale = float(world_size) / (float(global_scheduled_slots) * reference)
    loss = scale * (-pg_numerator + beta_kl * kl_numerator - beta_entropy * entropy_numerator)
    active_writer_rows = int(torch.any(writer, dim=-1).sum().item())
    active_answer_rows = int(torch.any(answer, dim=-1).sum().item())
    receipt = SlotLossReceipt(
        world_size=world_size,
        global_scheduled_slots=global_scheduled_slots,
        local_scheduled_slots=local_scheduled_slots,
        active_writer_slots=active_writer_rows,
        active_answer_slots=active_answer_rows,
        sampled_writer_tokens=int(writer.sum().item()),
        sampled_answer_tokens=int(answer.sum().item()),
        reference_length=reference,
        reduction_mode="ddp_mean_equal_logical_shards",
        pre_ddp_scale=scale,
        local_pg_numerator=float(pg_numerator.detach().cpu().item()),
        local_kl_numerator=float(kl_numerator.detach().cpu().item()),
        local_entropy_numerator=float(entropy_numerator.detach().cpu().item()),
        local_loss=float(loss.detach().cpu().item()),
    ).as_dict()
    return loss, receipt


def seal_credit_bundle(
    *, block_id: str, behavior_checkpoint_sha256: str,
    fold_receipts: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not block_id or len(behavior_checkpoint_sha256) != 64:
        raise ValueError("MIC_V2_NO_GO: incomplete block/checkpoint identity")
    normalized = []
    for row in rows:
        required = {"content_root_id", "trajectory_id", "turn_index", "writer_credit", "answer_credit"}
        if set(row) != required:
            raise ValueError("MIC_V2_NO_GO: credit row schema drifted")
        item = dict(row)
        item["writer_credit"] = _finite(item["writer_credit"], "writer_credit")
        item["answer_credit"] = _finite(item["answer_credit"], "answer_credit")
        if item["turn_index"] < 1 or item["turn_index"] > T_MAX:
            raise ValueError("MIC_V2_NO_GO: credit turn outside slot program")
        normalized.append(item)
    normalized.sort(key=lambda row: (row["content_root_id"], row["trajectory_id"], row["turn_index"]))
    payload = {
        "schema": SCHEMA,
        "kind": "sealed_credit_bundle",
        "block_id": block_id,
        "behavior_checkpoint_sha256": behavior_checkpoint_sha256,
        "fold_receipts": [dict(row) for row in fold_receipts],
        "rows": normalized,
    }
    payload["bundle_sha256"] = sha256_json(payload)
    return payload


def verify_sealed_credit_bundle(bundle: Mapping[str, Any], expected_sha256: str) -> None:
    payload = dict(bundle)
    digest = payload.pop("bundle_sha256", None)
    if digest != expected_sha256 or digest != sha256_json(payload):
        raise ValueError("MIC_V2_NO_GO: sealed credit bundle changed")


def sparse_branch_schedule(content_root_id: str, experiment_seed: int) -> dict[str, Any]:
    replica = stable_fold(content_root_id, "branch-replica", GROUP_SIZE)
    turn = 1 + _hash_mod([content_root_id, replica, "branch-turn"], T_MAX)
    arms = [sha256_json([experiment_seed, content_root_id, replica, turn, "writer-arm", arm])
            for arm in (0, 1)]
    return {"replica": replica, "turn": turn, "writer_arm_keys": arms}


def sparse_future_counter_key(
    *, experiment_seed: int, content_root_id: str, replica: int, turn: int,
    future_turn: int, role: str, future_seed_index: int,
) -> str:
    if role not in ("writer", "answer") or future_seed_index < 0:
        raise ValueError("MIC_V2_NO_GO: invalid SparseBranch future counter coordinate")
    if future_turn <= turn or future_turn > T_MAX + 1 \
            or ((role == "answer") != (future_turn == T_MAX + 1)):
        raise ValueError("MIC_V2_NO_GO: SparseBranch future role/turn coordinate drifted")
    return sha256_json([
        experiment_seed, content_root_id, replica, turn,
        future_turn, role, future_seed_index,
    ])


def sparse_branch_accounting(
    *, trunk_tokens: int, arm_writer_tokens: Sequence[int], continuation_tokens: Sequence[int],
    leaf_returns: Sequence[float], other_replica_returns: Sequence[float],
    model_forward_tokens: int, model_backward_tokens: int,
    h20_seconds: float, wall_seconds: float, active: bool,
) -> dict[str, Any]:
    if len(arm_writer_tokens) != 2 or len(continuation_tokens) != 2:
        raise ValueError("MIC_V2_NO_GO: SparseBranch requires exactly two leaves")
    counts = [trunk_tokens, *arm_writer_tokens, *continuation_tokens,
              model_forward_tokens, model_backward_tokens]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("MIC_V2_NO_GO: branch token counts must be non-negative integers")
    physical = trunk_tokens + sum(arm_writer_tokens) + sum(continuation_tokens)
    actor_weighted = trunk_tokens + 0.5 * (sum(arm_writer_tokens) + sum(continuation_tokens))
    c_g = (GROUP_SIZE - 1.0) / GROUP_SIZE
    if active:
        if len(leaf_returns) != 2 or len(other_replica_returns) != GROUP_SIZE - 1:
            raise ValueError("MIC_V2_NO_GO: active SparseBranch return groups are incomplete")
        returns = [_finite(value, "leaf_return") for value in leaf_returns]
        other_returns = [_finite(value, "other_replica_return")
                         for value in other_replica_returns]
        pair_credits = [c_g * (returns[0] - returns[1]), c_g * (returns[1] - returns[0])]
        other_mean = math.fsum(other_returns) / len(other_returns)
        branch_advantages = [c_g * (value - other_mean) for value in returns]
        trunk_credit = math.fsum(branch_advantages) / 2.0
    else:
        if leaf_returns or other_replica_returns or any(arm_writer_tokens) \
                or any(continuation_tokens):
            raise ValueError("MIC_V2_NO_GO: inactive SparseBranch launched an outcome-based fallback")
        pair_credits = [0.0, 0.0]
        branch_advantages = [0.0, 0.0]
        trunk_credit = 0.0
    result = {
        "schema": SCHEMA, "kind": "sparse_branch_accounting",
        "scheduled_boundary_active": active,
        "correction_applied": active,
        "outcome_based_fallback": False,
        "trunk_tokens_once": trunk_tokens,
        "arm_writer_tokens": list(arm_writer_tokens),
        "continuation_tokens": list(continuation_tokens),
        "physical_model_tokens": physical,
        "actor_weighted_tokens": actor_weighted,
        "pair_credits": pair_credits,
        "branch_advantages": branch_advantages,
        "trunk_credit": trunk_credit,
        "downstream_leaf_credits": branch_advantages,
        "c_g": c_g,
        "model_forward_tokens": model_forward_tokens,
        "model_backward_tokens": model_backward_tokens,
        "h20_seconds": _finite(h20_seconds, "h20_seconds"),
        "wall_seconds": _finite(wall_seconds, "wall_seconds"),
    }
    if result["h20_seconds"] < 0 or result["wall_seconds"] < 0:
        raise ValueError("MIC_V2_NO_GO: SparseBranch time accounting is negative")
    if model_forward_tokens != physical or model_backward_tokens != physical:
        raise ValueError("MIC_V2_NO_GO: SparseBranch compute ledger does not reconstruct")
    result["receipt_sha256"] = sha256_json(result)
    return result


def full_branch_block_schedule(
    *, block_id: str, content_root_ids: Sequence[str], experiment_seed: int,
    t_max: int = T_MAX,
) -> dict[str, Any]:
    """Freeze the exact lowest-hash 25% FullBranch root--turn positions."""
    roots = sorted(set(content_root_ids))
    if not block_id or not roots or len(roots) != len(content_root_ids):
        raise ValueError("MIC_V2_NO_GO: FullBranch block/root identity is incomplete or duplicated")
    if t_max < 1:
        raise ValueError("MIC_V2_NO_GO: FullBranch t_max must be positive")
    candidates = []
    for root in roots:
        for turn in range(1, t_max + 1):
            selection_hash = sha256_json([block_id, root, turn, "logo-local-select"])
            candidates.append((selection_hash, root, turn))
    candidates.sort()
    if len(candidates) % 4:
        raise ValueError("MIC_V2_NO_GO: FullBranch candidate count is not divisible by four")
    selected_positions = candidates[:len(candidates) // 4]
    records = []
    for selection_hash, root, turn in selected_positions:
        anchor = _hash_mod([block_id, root, turn, "logo-anchor"], GROUP_SIZE)
        arms = [sha256_json([
            experiment_seed, block_id, root, turn, "logo-writer-arm", arm,
        ]) for arm in range(4)]
        records.append({
            "content_root_id": root, "turn_index": turn,
            "selection_hash": selection_hash, "anchor_replica": anchor,
            "writer_arm_keys": arms,
        })
    result = {
        "schema": SCHEMA, "kind": "full_branch_block_schedule",
        "block_id": block_id, "experiment_seed": experiment_seed,
        "root_count": len(roots), "t_max": t_max,
        "candidate_count": len(candidates), "selected_count": len(records),
        "records": records,
    }
    result["schedule_sha256"] = sha256_json(result)
    return result


def full_branch_future_counter_key(
    *, experiment_seed: int, block_id: str, content_root_id: str,
    turn: int, arm: int, future_turn: int, role: str,
) -> str:
    if not block_id or arm not in range(4) or role not in ("writer", "answer"):
        raise ValueError("MIC_V2_NO_GO: invalid FullBranch future counter coordinate")
    if future_turn <= turn or future_turn > T_MAX + 1:
        raise ValueError("MIC_V2_NO_GO: FullBranch future turn is outside continuation")
    if (role == "answer") != (future_turn == T_MAX + 1):
        raise ValueError("MIC_V2_NO_GO: FullBranch future role/turn coordinate drifted")
    return sha256_json([
        experiment_seed, block_id, content_root_id, turn,
        "logo-future", arm, future_turn, role,
    ])


def bind_full_branch_arm_states(
    *, pre_states: Sequence[Mapping[str, Any]], schedule_record: Mapping[str, Any],
    global_trajectory_ids_by_replica: Sequence[str],
) -> dict[str, Any]:
    """Bind four independently reconstructed local arms to the exact same X-minus."""
    required_schedule = {
        "content_root_id", "turn_index", "selection_hash", "anchor_replica",
        "writer_arm_keys",
    }
    if set(schedule_record) != required_schedule or len(pre_states) != 4:
        raise ValueError("MIC_V2_NO_GO: FullBranch arm binding schema drifted")
    if len(schedule_record["writer_arm_keys"]) != 4 \
            or len(set(schedule_record["writer_arm_keys"])) != 4:
        raise ValueError("MIC_V2_NO_GO: FullBranch writer arm keys are incomplete")
    if len(global_trajectory_ids_by_replica) != GROUP_SIZE \
            or len(set(global_trajectory_ids_by_replica)) != GROUP_SIZE \
            or any(not isinstance(value, str) or not value
                   for value in global_trajectory_ids_by_replica):
        raise ValueError("MIC_V2_NO_GO: global replica trajectory mapping is invalid")
    anchor_replica = schedule_record["anchor_replica"]
    if isinstance(anchor_replica, bool) or not isinstance(anchor_replica, int) \
            or anchor_replica < 0 or anchor_replica >= GROUP_SIZE:
        raise ValueError("MIC_V2_NO_GO: FullBranch anchor replica is invalid")
    anchor_trajectory_id = global_trajectory_ids_by_replica[anchor_replica]
    checked = [validate_boundary_state(state) for state in pre_states]
    for state in checked:
        if state["phase"] != "pre_write" \
                or state["content_root_id"] != schedule_record["content_root_id"] \
                or state["turn_index"] != schedule_record["turn_index"]:
            raise ValueError("MIC_V2_NO_GO: FullBranch arm restored the wrong boundary")
        if not anchor_trajectory_id or state["trajectory_id"] != anchor_trajectory_id:
            raise ValueError("MIC_V2_NO_GO: FullBranch arm is not bound to anchor replica")
    hashes = [state["state_sha256"] for state in checked]
    if len(set(hashes)) != 1:
        raise ValueError("MIC_V2_NO_GO: FullBranch arms do not share exact X-minus")
    result = {
        "schema": SCHEMA, "kind": "full_branch_arm_state_binding",
        "content_root_id": schedule_record["content_root_id"],
        "turn_index": schedule_record["turn_index"],
        "anchor_replica": schedule_record["anchor_replica"],
        "anchor_trajectory_id": anchor_trajectory_id,
        "global_trajectory_mapping_sha256": sha256_json(list(global_trajectory_ids_by_replica)),
        "pre_state_sha256": hashes[0],
        "writer_arm_keys": list(schedule_record["writer_arm_keys"]),
    }
    result["receipt_sha256"] = sha256_json(result)
    return result


def full_branch_matched_slot_count(
    *, root_count: int, selected_scheduled_boundaries: int,
    group_size: int = GROUP_SIZE, t_max: int = T_MAX,
) -> int:
    if isinstance(selected_scheduled_boundaries, bool) \
            or not isinstance(selected_scheduled_boundaries, int) \
            or selected_scheduled_boundaries < 0:
        raise ValueError("MIC_V2_NO_GO: selected local boundary count is invalid")
    return scheduled_slot_count(root_count, group_size, t_max) + 4 * selected_scheduled_boundaries


def full_branch_accounting(
    *, root_count: int, selected_scheduled_boundaries: int,
    selected_active_boundaries: int,
    global_actor_tokens: int, local_writer_actor_tokens: int,
    reward_continuation_tokens: int, terminal_continuations: int,
    model_forward_tokens: int, model_backward_tokens: int,
    h20_seconds: float, wall_seconds: float,
) -> dict[str, Any]:
    counts = (global_actor_tokens, local_writer_actor_tokens, reward_continuation_tokens,
              terminal_continuations, model_forward_tokens, model_backward_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("MIC_V2_NO_GO: FullBranch token counts must be non-negative integers")
    if selected_active_boundaries < 0 \
            or selected_active_boundaries > selected_scheduled_boundaries:
        raise ValueError("MIC_V2_NO_GO: FullBranch active boundaries exceed frozen schedule")
    result = {
        "schema": SCHEMA, "kind": "full_branch_accounting",
        "fixed_actor_slots": full_branch_matched_slot_count(
            root_count=root_count,
            selected_scheduled_boundaries=selected_scheduled_boundaries,
        ),
        "selected_scheduled_boundaries": selected_scheduled_boundaries,
        "selected_active_boundaries": selected_active_boundaries,
        "selected_inactive_boundaries": (
            selected_scheduled_boundaries - selected_active_boundaries
        ),
        "global_actor_tokens": global_actor_tokens,
        "local_writer_actor_tokens": local_writer_actor_tokens,
        "reward_continuation_tokens": reward_continuation_tokens,
        "terminal_continuations": terminal_continuations,
        "expected_terminal_continuations": 4 * selected_active_boundaries,
        "actor_regularizer_tokens": global_actor_tokens + local_writer_actor_tokens,
        "physical_model_tokens": (
            global_actor_tokens + local_writer_actor_tokens + reward_continuation_tokens
        ),
        "model_forward_tokens": model_forward_tokens,
        "model_backward_tokens": model_backward_tokens,
        "h20_seconds": _finite(h20_seconds, "h20_seconds"),
        "wall_seconds": _finite(wall_seconds, "wall_seconds"),
    }
    if result["h20_seconds"] < 0 or result["wall_seconds"] < 0:
        raise ValueError("MIC_V2_NO_GO: FullBranch time accounting is negative")
    if terminal_continuations != result["expected_terminal_continuations"]:
        raise ValueError("MIC_V2_NO_GO: FullBranch terminal continuations are incomplete")
    if model_forward_tokens != result["physical_model_tokens"] \
            or model_backward_tokens != result["actor_regularizer_tokens"]:
        raise ValueError("MIC_V2_NO_GO: FullBranch compute ledger does not reconstruct")
    result["receipt_sha256"] = sha256_json(result)
    return result


def standardized_group_credit(returns: Sequence[float], epsilon: float = 1e-6) -> np.ndarray:
    values = np.asarray([_finite(value, "return") for value in returns], dtype=np.float64)
    if len(values) < 2:
        raise ValueError("MIC_V2_NO_GO: standardized group needs at least two returns")
    centered = values - values.mean()
    sample_std = float(values.std(ddof=1))
    if sample_std == 0.0:
        return np.zeros_like(values)
    return centered / (sample_std + _finite(epsilon, "epsilon"))


def logo_port_sequence_loss(
    *, old_log_prob: Any, log_prob: Any, credits: Any, sampled_mask: Any,
) -> Any:
    """Delivery-faithful LoGo-style geometric-ratio dual-clipped PG loss."""
    import torch

    if any(not isinstance(item, torch.Tensor)
           for item in (old_log_prob, log_prob, credits, sampled_mask)):
        raise ValueError("MIC_V2_NO_GO: LoGo port inputs must be torch tensors")
    if old_log_prob.shape != log_prob.shape or sampled_mask.shape != log_prob.shape:
        raise ValueError("MIC_V2_NO_GO: LoGo port token shapes differ")
    if sampled_mask.dtype != torch.bool:
        raise ValueError("MIC_V2_NO_GO: LoGo port sampled mask must be boolean")
    if old_log_prob.requires_grad:
        raise ValueError("MIC_V2_NO_GO: LoGo port old log probabilities are not detached")
    if credits.ndim != 1 or credits.shape[0] != log_prob.shape[0] or credits.requires_grad:
        raise ValueError("MIC_V2_NO_GO: LoGo port credits must be detached sequence scalars")
    if not torch.isfinite(credits).all():
        raise ValueError("MIC_V2_NO_GO: LoGo port credit is non-finite")
    mask = sampled_mask
    if not torch.isfinite(old_log_prob.masked_select(mask)).all() \
            or not torch.isfinite(log_prob.masked_select(mask)).all():
        raise ValueError("MIC_V2_NO_GO: active LoGo log probability is non-finite")
    lengths = mask.sum(dim=-1)
    valid = lengths > 0
    if not torch.any(valid):
        raise ValueError("MIC_V2_NO_GO: LoGo port has no realized action sequence")
    log_ratio = torch.where(mask, log_prob - old_log_prob, torch.zeros_like(log_prob))
    mean_log_ratio = log_ratio.sum(dim=-1) / lengths.clamp_min(1)
    ratio = torch.exp(mean_log_ratio)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    ordinary = torch.maximum(-ratio * credits, -clipped * credits)
    negative = torch.minimum(-3.0 * credits, ordinary)
    per_sequence = torch.where(credits < 0, negative, ordinary)
    return per_sequence[valid].mean()


def logo_port_actor_loss(
    *, old_log_prob: Any, log_prob: Any, credits: Any, sampled_mask: Any,
    kl_token: Any, entropy_token: Any, beta_kl: float, beta_entropy: float,
) -> tuple[Any, dict[str, Any]]:
    """Complete LoGo-Port objective; reward-only continuation tokens stay masked."""
    import torch

    pg = logo_port_sequence_loss(
        old_log_prob=old_log_prob, log_prob=log_prob,
        credits=credits, sampled_mask=sampled_mask,
    )
    if kl_token.shape != log_prob.shape or entropy_token.shape != log_prob.shape:
        raise ValueError("MIC_V2_NO_GO: LoGo regularizer token shapes differ")
    mask = sampled_mask.bool()
    token_count = mask.sum()
    if int(token_count.item()) < 1:
        raise ValueError("MIC_V2_NO_GO: LoGo regularizer set is empty")
    if not torch.isfinite(kl_token.masked_select(mask)).all() \
            or not torch.isfinite(entropy_token.masked_select(mask)).all():
        raise ValueError("MIC_V2_NO_GO: active LoGo regularizer is non-finite")
    kl_mean = torch.where(mask, kl_token, torch.zeros_like(kl_token)).sum() / token_count
    entropy_mean = torch.where(mask, entropy_token, torch.zeros_like(entropy_token)).sum() / token_count
    loss = pg + _finite(beta_kl, "beta_kl") * kl_mean \
        - _finite(beta_entropy, "beta_entropy") * entropy_mean
    receipt = {
        "schema": SCHEMA, "kind": "logo_port_actor_loss",
        "valid_action_sequences": int(torch.any(mask, dim=-1).sum().item()),
        "valid_actor_tokens": int(token_count.item()),
        "pg_loss": float(pg.detach().cpu().item()),
        "kl_mean": float(kl_mean.detach().cpu().item()),
        "entropy_mean": float(entropy_mean.detach().cpu().item()),
        "beta_kl": float(beta_kl), "beta_entropy": float(beta_entropy),
        "actor_loss": float(loss.detach().cpu().item()),
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return loss, receipt


def oracle_boundary_decomposition(
    *, initial_post_value: float, pre_values: Sequence[float], post_values: Sequence[float], outcome: float,
) -> dict[str, Any]:
    if not pre_values or len(pre_values) != len(post_values):
        raise ValueError("MIC_V2_NO_GO: oracle boundary sequences differ")
    v0 = _finite(initial_post_value, "initial_post_value")
    pre = [_finite(value, "pre_value") for value in pre_values]
    post = [_finite(value, "post_value") for value in post_values]
    terminal = _finite(outcome, "outcome")
    chunks, writes = [], []
    previous = v0
    for before, after in zip(pre, post):
        chunks.append(before - previous)
        writes.append(after - before)
        previous = after
    residual = terminal - post[-1]
    closure = v0 + math.fsum(chunks) + math.fsum(writes) + residual - terminal
    return {"initial_post_value": v0, "chunk_credits": chunks, "writer_credits": writes,
            "answer_residual": residual, "outcome": terminal, "closure_error": closure}


def enumerate_oracle_toy_mdp() -> dict[str, Any]:
    """Enumerate a two-turn MDP with separate chunk and writer transitions."""
    rows = []
    p_c1 = 0.45
    p_future_noise = 0.25
    for chunk1 in (0, 1):
        p1 = p_c1 if chunk1 else 1.0 - p_c1
        p_action1 = 0.30 + 0.25 * chunk1
        for action1 in (0, 1):
            pa1 = p_action1 if action1 else 1.0 - p_action1
            p_chunk2 = 0.30 + 0.40 * chunk1
            for chunk2 in (0, 1):
                p2 = p_chunk2 if chunk2 else 1.0 - p_chunk2
                p_action2 = 0.20 + 0.20 * action1 + 0.20 * chunk2
                for action2 in (0, 1):
                    pa2 = p_action2 if action2 else 1.0 - p_action2
                    for future_noise in (0, 1):
                        pu = p_future_noise if future_noise else 1.0 - p_future_noise
                        reward = (
                            0.05 + 0.15 * chunk1 + 0.25 * action1
                            + 0.10 * chunk2 + 0.30 * action2
                            + 0.15 * future_noise
                        )
                        rows.append({
                            "chunk1": chunk1, "action1": action1,
                            "chunk2": chunk2, "action2": action2,
                            "future_noise": future_noise, "reward": reward,
                            "probability": p1 * pa1 * p2 * pa2 * pu,
                        })
    total_probability = math.fsum(row["probability"] for row in rows)
    if abs(total_probability - 1.0) > 1e-15:
        raise RuntimeError("MIC_V2_NO_GO: oracle toy probability table does not close")

    def conditional_value(**conditions: int) -> float:
        selected = [row for row in rows if all(row[key] == value for key, value in conditions.items())]
        mass = math.fsum(row["probability"] for row in selected)
        if mass <= 0:
            raise RuntimeError("MIC_V2_NO_GO: empty oracle condition")
        return math.fsum(row["probability"] * row["reward"] for row in selected) / mass

    chosen = {"chunk1": 1, "action1": 0, "chunk2": 1, "action2": 1, "future_noise": 0}
    chosen_row = next(row for row in rows if all(row[key] == value for key, value in chosen.items()))
    initial = conditional_value()
    pre_values = [
        conditional_value(chunk1=chosen["chunk1"]),
        conditional_value(chunk1=chosen["chunk1"], action1=chosen["action1"],
                          chunk2=chosen["chunk2"]),
    ]
    post_values = [
        conditional_value(chunk1=chosen["chunk1"], action1=chosen["action1"]),
        conditional_value(chunk1=chosen["chunk1"], action1=chosen["action1"],
                          chunk2=chosen["chunk2"], action2=chosen["action2"]),
    ]
    decomposition = oracle_boundary_decomposition(
        initial_post_value=initial, pre_values=pre_values,
        post_values=post_values, outcome=chosen_row["reward"],
    )
    q_action1 = [conditional_value(chunk1=1, action1=action) for action in (0, 1)]
    condition_mass = math.fsum(row["probability"] for row in rows if row["chunk1"] == 1)
    direct_terminal_score_gradient = math.fsum(
        row["probability"] / condition_mass
        * (row["action1"] - 0.55) * row["reward"]
        for row in rows if row["chunk1"] == 1
    )
    return {
        "schema": SCHEMA, "kind": "enumerated_oracle_toy_mdp",
        "joint_row_count": len(rows), "joint_probability": total_probability,
        "joint_table_sha256": sha256_json(rows), "chosen_trajectory": chosen,
        "pre_values": pre_values, "post_values": post_values,
        "decomposition": decomposition,
        "action1_probability_given_chunk1": 0.55,
        "action1_q_values_given_chunk1": q_action1,
        "action1_direct_terminal_score_gradient": direct_terminal_score_gradient,
    }


def write_json_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
