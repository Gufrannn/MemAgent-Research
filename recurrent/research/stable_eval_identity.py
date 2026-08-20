"""Stable, auditable identities for evaluation rollouts.

This module is intentionally independent from reward computation.  The stable
row key is derived from a frozen evaluation manifest and never from a runtime
UUID, output order, or batch-local group number.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from recurrent.research.trajectory_seeding import MAX_TORCH_SEED


STABLE_KEY_FIELDS = ("eval_manifest_hash", "example_id", "replica_id")
MANIFEST_ROW_FIELDS = (
    "example_id",
    "semantic_dataset_index",
    "source_order_index",
    "raw_row_position",
    "production_effective_position",
    "context_token_count",
    "source_question_hash",
    "source_context_hash",
    "ground_truth_hash",
)
OUTPUT_IDENTITY_FIELDS = (
    "interface_id",
    "attempt_id",
    *MANIFEST_ROW_FIELDS,
    "eval_manifest_hash",
    "replica_id",
    "source_repeated_row",
    "trajectory_seed",
    "trajectory_id",
    "runtime_sample_uuid",
)
AUDIT_ONLY_NON_TENSOR_FIELDS = (
    *OUTPUT_IDENTITY_FIELDS,
    "active_sample_index",
    "request_seed",
    "configured_request_seed",
    "rollout_request_seed",
    "request_prompt_token_sha256",
    "returned_prompt_token_sha256",
    "rollout_worker_rank",
    "is_final",
)
TURN_LEDGER_NON_TENSOR_FIELDS = AUDIT_ONLY_NON_TENSOR_FIELDS


def trajectory_turn_record_from_columns(
    columns: Mapping[str, Sequence[Any]],
    *,
    row: int,
    trajectory_turn: int,
    response_token_sha256: str,
) -> dict[str, Any]:
    """Serialize one turn using the same complete identity contract as terminal output."""
    missing = [field for field in TURN_LEDGER_NON_TENSOR_FIELDS if field not in columns]
    if missing:
        raise ValueError(f"stable evaluation turn ledger is missing row fields: {missing}")
    lengths = {field: len(columns[field]) for field in TURN_LEDGER_NON_TENSOR_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"stable evaluation turn ledger columns are not row-aligned: {lengths}")
    row_count = next(iter(lengths.values()))
    if row < 0 or row >= row_count:
        raise IndexError(f"stable evaluation turn ledger row {row} outside [0, {row_count})")

    def json_scalar(value: Any) -> Any:
        item = getattr(value, "item", None)
        return item() if callable(item) else value

    return {
        "record_type": "trajectory_turn",
        **{
            field: json_scalar(columns[field][row])
            for field in TURN_LEDGER_NON_TENSOR_FIELDS
        },
        "trajectory_turn": int(trajectory_turn),
        "response_token_sha256": str(response_token_sha256),
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using the canonical JSON form used by this gate."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stable_eval_runtime_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the fully resolved Hydra job config without its self-hash field."""
    if not isinstance(config, Mapping):
        raise TypeError("resolved stable-evaluation runtime config must be a mapping")
    payload = deepcopy(dict(config))
    trainer = payload.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("resolved stable-evaluation runtime config lacks trainer")
    eval_identity = trainer.get("eval_identity")
    if not isinstance(eval_identity, dict):
        raise ValueError("resolved runtime config lacks trainer.eval_identity")
    eval_identity.pop("expected_runtime_config_sha256", None)
    return canonical_sha256(payload)


def sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected text for SHA-256, got {type(value).__name__}")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evaluation_trajectory_seed(
    *,
    base_seed: int,
    eval_manifest_hash: str,
    example_id: str,
    source_order_index: int,
    replica_id: int,
) -> int:
    """Derive a batch-independent seed for one evaluation trajectory."""
    if len(eval_manifest_hash) != 64:
        raise ValueError("eval_manifest_hash must be a 64-character SHA-256")
    payload = canonical_json_bytes(
        [
            "memagent-stable-eval-trajectory-v1",
            int(base_seed),
            eval_manifest_hash,
            str(example_id),
            int(source_order_index),
            int(replica_id),
        ]
    )
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % MAX_TORCH_SEED


def stable_trajectory_id(
    *, eval_manifest_hash: str, example_id: str, replica_id: int, trajectory_seed: int
) -> str:
    return canonical_sha256(
        {
            "namespace": "memagent-stable-eval-trajectory-id-v1",
            "eval_manifest_hash": eval_manifest_hash,
            "example_id": str(example_id),
            "replica_id": int(replica_id),
            "trajectory_seed": int(trajectory_seed),
        }
    )


def _require_sha256(value: object, field: str) -> str:
    value = str(value)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def validate_resolved_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an S0-resolved stable evaluation manifest."""
    if not isinstance(manifest, Mapping):
        raise TypeError("resolved evaluation manifest must be a mapping")
    payload = manifest.get("identity_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("resolved evaluation manifest is missing identity_payload")
    expected_hash = _require_sha256(manifest.get("eval_manifest_hash"), "eval_manifest_hash")
    actual_hash = canonical_sha256(payload)
    if actual_hash != expected_hash:
        raise ValueError(
            "resolved evaluation manifest hash mismatch: "
            f"declared={expected_hash}, computed={actual_hash}"
        )

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("identity_payload.rows must be a non-empty list")
    normalized_rows: list[dict[str, Any]] = []
    seen_example_ids: set[str] = set()
    for expected_order, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"manifest row {expected_order} must be a mapping")
        missing = [field for field in MANIFEST_ROW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"manifest row {expected_order} is missing fields: {missing}")
        normalized = dict(row)
        normalized["example_id"] = str(row["example_id"])
        for field in (
            "semantic_dataset_index",
            "source_order_index",
            "raw_row_position",
            "production_effective_position",
            "context_token_count",
        ):
            normalized[field] = int(row[field])
        if normalized["context_token_count"] < 1:
            raise ValueError(
                f"context_token_count must be positive at row {expected_order}"
            )
        if normalized["source_order_index"] != expected_order:
            raise ValueError(
                "source_order_index must be contiguous in frozen order: "
                f"row={expected_order}, value={normalized['source_order_index']}"
            )
        if normalized["production_effective_position"] != expected_order:
            raise ValueError(
                "production_effective_position must equal frozen source order: "
                f"row={expected_order}, value={normalized['production_effective_position']}"
            )
        if normalized["example_id"] != str(normalized["semantic_dataset_index"]):
            raise ValueError(
                "example_id must be the string form of semantic_dataset_index: "
                f"row={expected_order}, example_id={normalized['example_id']}, "
                f"semantic_dataset_index={normalized['semantic_dataset_index']}"
            )
        for field in ("source_question_hash", "source_context_hash", "ground_truth_hash"):
            normalized[field] = _require_sha256(row[field], field)
        if normalized["example_id"] in seen_example_ids:
            raise ValueError(f"duplicate example_id in resolved manifest: {normalized['example_id']}")
        seen_example_ids.add(normalized["example_id"])
        normalized_rows.append(normalized)

    raw_positions = [row["raw_row_position"] for row in normalized_rows]
    if raw_positions != sorted(set(raw_positions)):
        raise ValueError(
            "raw_row_position must be unique and strictly increasing in frozen source order"
        )

    normalized_payload = dict(payload)
    normalized_payload["rows"] = normalized_rows
    normalized_manifest = dict(manifest)
    normalized_manifest["identity_payload"] = normalized_payload
    normalized_manifest["eval_manifest_hash"] = expected_hash
    return normalized_manifest


def load_resolved_manifest(path: str | Path, *, expected_hash: str | None = None) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = validate_resolved_manifest(json.load(stream))
    if expected_hash is not None and manifest["eval_manifest_hash"] != _require_sha256(
        expected_hash, "expected eval manifest hash"
    ):
        raise ValueError(
            "runtime evaluation manifest differs from the frozen expected hash: "
            f"expected={expected_hash}, actual={manifest['eval_manifest_hash']}"
        )
    return manifest


def manifest_rows_by_identity(manifest: Mapping[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    checked = validate_resolved_manifest(manifest)
    rows = checked["identity_payload"]["rows"]
    return {(int(row["source_order_index"]), str(row["example_id"])): row for row in rows}


def build_stable_eval_identities(
    *,
    semantic_indices: Sequence[object],
    source_order_indices: Sequence[object],
    replicas: int,
    base_seed: int,
    interface_id: str,
    attempt_id: str,
    resolved_manifest: Mapping[str, Any],
    runtime_uuid_factory: Callable[[], object] = uuid4,
) -> list[dict[str, Any]]:
    """Build identities for a repeat-interleaved validation batch.

    ``semantic_indices`` and ``source_order_indices`` are the already repeated,
    row-aligned columns captured before generation fields are popped.
    """
    if replicas < 1:
        raise ValueError("replicas must be positive")
    if len(semantic_indices) != len(source_order_indices) or not semantic_indices:
        raise ValueError("semantic and source-order identity columns must be non-empty and row-aligned")
    if len(semantic_indices) % replicas:
        raise ValueError("repeated validation batch size must be divisible by replicas")
    if not interface_id or not attempt_id:
        raise ValueError("interface_id and attempt_id are required")

    checked = validate_resolved_manifest(resolved_manifest)
    manifest_hash = checked["eval_manifest_hash"]
    frozen = manifest_rows_by_identity(checked)
    occurrence: defaultdict[tuple[int, str], int] = defaultdict(int)
    identities: list[dict[str, Any]] = []
    runtime_uuids: set[str] = set()

    for semantic_value, order_value in zip(semantic_indices, source_order_indices):
        semantic_index = int(semantic_value)
        source_order_index = int(order_value)
        example_id = str(semantic_index)
        key = (source_order_index, example_id)
        if key not in frozen:
            raise ValueError(
                "validation row is not present in the frozen resolved manifest: "
                f"source_order_index={source_order_index}, example_id={example_id}"
            )
        replica_id = occurrence[key]
        occurrence[key] += 1
        if replica_id >= replicas:
            raise ValueError(f"too many repeated rows for frozen example {key}")

        trajectory_seed = evaluation_trajectory_seed(
            base_seed=base_seed,
            eval_manifest_hash=manifest_hash,
            example_id=example_id,
            source_order_index=source_order_index,
            replica_id=replica_id,
        )
        runtime_uuid = str(runtime_uuid_factory())
        if not runtime_uuid or runtime_uuid in runtime_uuids:
            raise ValueError("runtime_sample_uuid must be non-empty and unique within an attempt")
        runtime_uuids.add(runtime_uuid)
        frozen_row = frozen[key]
        identity = {
            "interface_id": str(interface_id),
            "attempt_id": str(attempt_id),
            **{field: frozen_row[field] for field in MANIFEST_ROW_FIELDS},
            "eval_manifest_hash": manifest_hash,
            "replica_id": replica_id,
            "source_repeated_row": source_order_index * replicas + replica_id,
            "trajectory_seed": trajectory_seed,
            "trajectory_id": stable_trajectory_id(
                eval_manifest_hash=manifest_hash,
                example_id=example_id,
                replica_id=replica_id,
                trajectory_seed=trajectory_seed,
            ),
            "runtime_sample_uuid": runtime_uuid,
        }
        identities.append(identity)

    bad_counts = {key: count for key, count in occurrence.items() if count != replicas}
    if bad_counts:
        raise ValueError(f"each frozen example must have exactly {replicas} replicas: {bad_counts}")
    if len({row["trajectory_seed"] for row in identities}) != len(identities):
        raise ValueError("evaluation trajectory seed collision")
    if len({row["trajectory_id"] for row in identities}) != len(identities):
        raise ValueError("evaluation trajectory ID collision")
    return identities


def identity_rows_to_columns(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    if not rows:
        raise ValueError("identity rows must be non-empty")
    missing_by_row = {
        row_index: [field for field in OUTPUT_IDENTITY_FIELDS if field not in row]
        for row_index, row in enumerate(rows)
    }
    missing_by_row = {row: fields for row, fields in missing_by_row.items() if fields}
    if missing_by_row:
        raise ValueError(f"identity rows are missing required fields: {missing_by_row}")
    return {field: [row[field] for row in rows] for field in OUTPUT_IDENTITY_FIELDS}


def identity_columns_to_rows(columns: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    missing = [field for field in OUTPUT_IDENTITY_FIELDS if field not in columns]
    if missing:
        raise ValueError(f"identity columns are missing required fields: {missing}")
    lengths = {field: len(columns[field]) for field in OUTPUT_IDENTITY_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"identity columns are not row-aligned: {lengths}")
    count = next(iter(lengths.values()))
    return [{field: columns[field][row] for field in OUTPUT_IDENTITY_FIELDS} for row in range(count)]


def detach_identity_columns_for_metrics(
    non_tensor_batch: dict[str, Any],
    tensor_batch: Any | None = None,
) -> dict[str, list[Any]]:
    """Remove audit-only columns before reward/metric processing.

    The returned copy is used only by the append-only output writer and audit.
    Mutating ``non_tensor_batch`` prevents identity instrumentation from
    becoming an accidental reward input.
    """
    missing = [field for field in OUTPUT_IDENTITY_FIELDS if field not in non_tensor_batch]
    if missing:
        raise ValueError(f"terminal validation output lost stable identity fields: {missing}")
    identities = {
        field: (
            non_tensor_batch[field].tolist()
            if hasattr(non_tensor_batch[field], "tolist")
            else list(non_tensor_batch[field])
        )
        for field in OUTPUT_IDENTITY_FIELDS
    }
    for field in AUDIT_ONLY_NON_TENSOR_FIELDS:
        non_tensor_batch.pop(field, None)
    if tensor_batch is not None:
        # This tensor is introduced solely for the per-turn audit ledger.  It
        # must not expand the schema observed by reward or metric functions.
        tensor_batch.pop("trajectory_turn", None)
    return identities


def detach_audit_meta_for_metrics(*meta_info_mappings: dict[str, Any]) -> None:
    """Remove strict-evaluation control state before reward/metric union."""
    for meta_info in meta_info_mappings:
        for field in (
            "strict_eval_identity",
            "stable_eval_identity",
            "trajectory_base_seeds",
            "trajectory_seed_turn",
            "request_seeds",
        ):
            meta_info.pop(field, None)


def stable_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["eval_manifest_hash"]),
        str(row["example_id"]),
        int(row["replica_id"]),
    )


def validate_attempt_identity_rows(
    rows: Sequence[Mapping[str, Any]], *, examples: int, replicas: int
) -> None:
    expected = examples * replicas
    if len(rows) != expected:
        raise ValueError(f"attempt must contain exactly {expected} terminal rows, got {len(rows)}")
    keys = [stable_key(row) for row in rows]
    if len(set(keys)) != expected:
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        raise ValueError(f"duplicate stable evaluation keys: {duplicates}")
    order_counts = Counter(int(row["source_order_index"]) for row in rows)
    if order_counts != Counter({index: replicas for index in range(examples)}):
        raise ValueError(f"unexpected source-order/replica counts: {dict(order_counts)}")
    for source_order_index in range(examples):
        group_rows = [
            row for row in rows if int(row["source_order_index"]) == source_order_index
        ]
        replica_ids = {int(row["replica_id"]) for row in group_rows}
        if replica_ids != set(range(replicas)):
            raise ValueError(
                f"source_order_index {source_order_index} has invalid replica IDs: "
                f"{sorted(replica_ids)} != {list(range(replicas))}"
            )
        for row in group_rows:
            expected_repeated_row = source_order_index * replicas + int(row["replica_id"])
            if int(row["source_repeated_row"]) != expected_repeated_row:
                raise ValueError(
                    "source_repeated_row is inconsistent with source order and replica: "
                    f"{row['source_repeated_row']} != {expected_repeated_row}"
                )
            if str(row["example_id"]) != str(int(row["semantic_dataset_index"])):
                raise ValueError("terminal semantic dataset identity is inconsistent")
            expected_trajectory_id = stable_trajectory_id(
                eval_manifest_hash=str(row["eval_manifest_hash"]),
                example_id=str(row["example_id"]),
                replica_id=int(row["replica_id"]),
                trajectory_seed=int(row["trajectory_seed"]),
            )
            if str(row["trajectory_id"]) != expected_trajectory_id:
                raise ValueError("terminal trajectory_id is not reconstructible")
    if len({str(row["attempt_id"]) for row in rows}) != 1:
        raise ValueError("attempt_id must be constant within an attempt")
    if len({str(row["interface_id"]) for row in rows}) != 1:
        raise ValueError("interface_id must be constant within an attempt")
    if len({str(row["eval_manifest_hash"]) for row in rows}) != 1:
        raise ValueError("eval_manifest_hash must be constant within an attempt")
    if len({str(row["runtime_sample_uuid"]) for row in rows}) != expected:
        raise ValueError("runtime_sample_uuid must be unique within an attempt")
    if len({int(row["trajectory_seed"]) for row in rows}) != expected:
        raise ValueError("trajectory_seed must be unique within an attempt")


def validate_repeated_attempts(attempts: Sequence[Sequence[Mapping[str, Any]]]) -> None:
    if len(attempts) < 2:
        raise ValueError("at least two preregistered attempts are required")
    stable_fields = tuple(field for field in OUTPUT_IDENTITY_FIELDS if field not in {"attempt_id", "runtime_sample_uuid"})
    reference = {
        stable_key(row): tuple(row[field] for field in stable_fields)
        for row in attempts[0]
    }
    all_runtime: set[str] = set()
    for attempt_index, rows in enumerate(attempts):
        current = {stable_key(row): tuple(row[field] for field in stable_fields) for row in rows}
        if current != reference:
            raise ValueError(f"stable evaluation fields changed in attempt {attempt_index}")
        runtime = {str(row["runtime_sample_uuid"]) for row in rows}
        if len(runtime) != len(rows) or runtime & all_runtime:
            raise ValueError("runtime UUIDs must be unique within and across preregistered attempts")
        all_runtime.update(runtime)


def rows_from_columns_at_indices(
    columns: Mapping[str, Sequence[Any]], indices: Iterable[int]
) -> list[dict[str, Any]]:
    """Index identity columns with the exact same final-output permutation."""
    rows = identity_columns_to_rows(columns)
    result = []
    for index in indices:
        index = int(index)
        if index < 0 or index >= len(rows):
            raise IndexError(f"identity index out of range: {index}")
        result.append(rows[index])
    return result


def validate_request_seed_echo(
    expected: Sequence[object], echoed: Sequence[object], worker_ranks: Sequence[object]
) -> None:
    expected_values = [int(value) for value in expected]
    echoed_values = [int(value) for value in echoed]
    ranks = [int(value) for value in worker_ranks]
    if len(expected_values) != len(echoed_values) or len(expected_values) != len(ranks):
        raise ValueError(
            "vLLM request-seed evidence is not row-aligned: "
            f"expected={len(expected_values)}, echoed={len(echoed_values)}, workers={len(ranks)}"
        )
    if expected_values != echoed_values:
        mismatches = [
            (row, requested, actual)
            for row, (requested, actual) in enumerate(zip(expected_values, echoed_values))
            if requested != actual
        ]
        raise ValueError(f"vLLM request seed echo mismatch: {mismatches}")
    if any(rank < 0 for rank in ranks):
        raise ValueError(f"vLLM worker ranks must be non-negative: {ranks}")


def validate_configured_request_binding(
    expected_seeds: Sequence[object],
    configured_seeds: Sequence[object],
    requested_prompt_hashes: Sequence[object],
    returned_prompt_hashes: Sequence[object],
    worker_ranks: Sequence[object],
) -> None:
    """Validate driver seed configuration and the actual vLLM prompt return.

    vLLM does not echo ``SamplingParams.seed`` in ``RequestOutput``.  We
    therefore name the seed evidence accurately (configured, not echoed) and
    bind it to an input/returned prompt-token digest checked after generation.
    """
    expected = [int(value) for value in expected_seeds]
    configured = [int(value) for value in configured_seeds]
    requested = [str(value) for value in requested_prompt_hashes]
    returned = [str(value) for value in returned_prompt_hashes]
    ranks = [int(value) for value in worker_ranks]
    lengths = {
        "expected": len(expected),
        "configured": len(configured),
        "requested_prompts": len(requested),
        "returned_prompts": len(returned),
        "workers": len(ranks),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"vLLM request binding is not row-aligned: {lengths}")
    if expected != configured:
        raise ValueError(
            f"configured vLLM request seeds differ from driver seeds: {configured} != {expected}"
        )
    if requested != returned:
        mismatches = [
            (row, before, after)
            for row, (before, after) in enumerate(zip(requested, returned))
            if before != after
        ]
        raise ValueError(f"vLLM returned prompt-token binding mismatch: {mismatches}")
    for name, values in (
        ("requested", requested),
        ("returned", returned),
    ):
        invalid = [value for value in values if len(value) != 64 or any(c not in "0123456789abcdef" for c in value)]
        if invalid:
            raise ValueError(f"{name} prompt-token hashes are invalid: {invalid}")
    if any(rank < 0 for rank in ranks):
        raise ValueError(f"vLLM worker ranks must be non-negative: {ranks}")
