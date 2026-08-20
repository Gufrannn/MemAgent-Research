#!/usr/bin/env python3
"""Audit the two preregistered recurrent-I stable-identity attempts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import append_jsonl, validate_jsonl_chain
from recurrent.research.stable_eval_identity import (
    MANIFEST_ROW_FIELDS,
    OUTPUT_IDENTITY_FIELDS,
    canonical_sha256,
    evaluation_trajectory_seed,
    stable_key,
    stable_trajectory_id,
    validate_attempt_identity_rows,
    validate_repeated_attempts,
    validate_resolved_manifest,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds
from tools.h20.preflight_qwen25_7b_stable_i4x2 import (
    EXPECTED_BRANCH,
    build_execution_binding,
    freeze_trainer_configuration,
    git,
    load_manifest,
    sha256_file,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _json_type_matches(value: object, expected: str) -> bool:
    return {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


def _is_json_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_ledger_schema(records: list[dict[str, Any]], schema: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})
    record_types = set(properties.get("record_type", {}).get("enum", []))
    for index, record in enumerate(records):
        missing = sorted(required - record.keys())
        if missing:
            failures.append(f"ledger record {index} is missing required fields {missing}")
        if record.get("record_type") not in record_types:
            failures.append(
                f"ledger record {index} has unknown record type {record.get('record_type')!r}"
            )
        for name, value in record.items():
            rule = properties.get(name)
            if not isinstance(rule, Mapping):
                continue
            allowed = rule.get("type")
            if isinstance(allowed, str):
                allowed = [allowed]
            if allowed and not any(_json_type_matches(value, item) for item in allowed):
                failures.append(f"ledger record {index} field {name} has invalid type")
                continue
            if "enum" in rule and value not in rule["enum"]:
                failures.append(f"ledger record {index} field {name} is outside its enum")
            if isinstance(value, str) and rule.get("pattern"):
                if re.fullmatch(str(rule["pattern"]), value) is None:
                    failures.append(f"ledger record {index} field {name} violates its pattern")
            if isinstance(value, str) and rule.get("format") == "date-time":
                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    failures.append(f"ledger record {index} field {name} is not date-time")
            if isinstance(value, (int, float)) and "minimum" in rule:
                if value < rule["minimum"]:
                    failures.append(f"ledger record {index} field {name} is below its minimum")
    return failures


def _identity_rows(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {field: record[field] for field in OUTPUT_IDENTITY_FIELDS if field in record}
        for record in records
    ]


def compare_deterministic_attempt_evidence(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    for field in (
        "terminal_result_by_stable_key",
        "turn_path_by_stable_key",
    ):
        if first.get(field) != second.get(field):
            failures.append(
                f"cross-attempt deterministic recurrent evidence changed: {field}"
            )
    return failures


def audit_attempt(
    *,
    attempt_id: str,
    attempt_root: Path,
    resolved_manifest: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    failures: list[str] = []
    evidence: dict[str, Any] = {"attempt_root": str(attempt_root)}
    terminal_path = attempt_root / "terminal/0.jsonl"
    turn_path = attempt_root / "trajectory_turns.jsonl"
    summary_path = attempt_root / "execution_summary.json"
    log_path = attempt_root / "run.log"
    for name, path in {
        "terminal output": terminal_path,
        "turn ledger": turn_path,
        "execution summary": summary_path,
        "run log": log_path,
    }.items():
        if not path.is_file():
            failures.append(f"{attempt_id} missing {name}: {path}")
    if failures:
        return [], failures, evidence

    terminal_records = read_jsonl(terminal_path)
    terminal_rows = _identity_rows(terminal_records)
    evidence["terminal_row_count"] = len(terminal_records)
    evidence["terminal_sha256"] = sha256_file(terminal_path)
    terminal_order = [
        int(record.get("source_repeated_row", -1)) for record in terminal_records
    ]
    if terminal_order != list(range(8)):
        failures.append(
            f"{attempt_id} terminal output order {terminal_order} != source repeated order 0..7"
        )
    try:
        validate_attempt_identity_rows(terminal_rows, examples=4, replicas=2)
    except Exception as error:
        failures.append(f"{attempt_id} terminal identity inventory failed: {error}")

    checked = validate_resolved_manifest(resolved_manifest)
    manifest_hash = checked["eval_manifest_hash"]
    frozen_rows = {
        (int(row["source_order_index"]), str(row["example_id"])): row
        for row in checked["identity_payload"]["rows"]
    }
    expected_orders = {0, 1, 2, 3}
    for row_number, row in enumerate(terminal_records):
        missing_result_fields = sorted(
            {
                "output",
                "score",
                "step",
                "terminal_response_token_sha256",
            }
            - row.keys()
        )
        if missing_result_fields:
            failures.append(
                f"{attempt_id} terminal row {row_number} missing result fields "
                f"{missing_result_fields}"
            )
        if not isinstance(row.get("output"), str):
            failures.append(f"{attempt_id} terminal row {row_number} output is not text")
        score = row.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            failures.append(f"{attempt_id} terminal row {row_number} score is not finite")
        step = row.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step != 0:
            failures.append(f"{attempt_id} terminal row {row_number} step is not frozen at zero")
        terminal_integer_fields = (
            "semantic_dataset_index",
            "source_order_index",
            "raw_row_position",
            "production_effective_position",
            "context_token_count",
            "replica_id",
            "source_repeated_row",
            "trajectory_seed",
        )
        invalid_terminal_integers = [
            field for field in terminal_integer_fields
            if field in row and not _is_json_integer(row[field])
        ]
        if invalid_terminal_integers:
            failures.append(
                f"{attempt_id} terminal row {row_number} has non-integer identity fields "
                f"{invalid_terminal_integers}"
            )
        missing = [field for field in OUTPUT_IDENTITY_FIELDS if field not in row]
        if missing:
            failures.append(f"{attempt_id} terminal row {row_number} missing fields {missing}")
            continue
        if row["attempt_id"] != attempt_id or row["interface_id"] != "I":
            failures.append(f"{attempt_id} terminal row {row_number} has wrong interface/attempt")
        if row["eval_manifest_hash"] != manifest_hash:
            failures.append(f"{attempt_id} terminal row {row_number} has wrong manifest hash")
        order = int(row["source_order_index"])
        example_id = str(row["example_id"])
        key = (order, example_id)
        if order not in expected_orders or key not in frozen_rows:
            failures.append(f"{attempt_id} terminal row {row_number} is outside frozen canary: {key}")
            continue
        expected = frozen_rows[key]
        for field in MANIFEST_ROW_FIELDS:
            if row[field] != expected[field]:
                failures.append(
                    f"{attempt_id} terminal row {row_number} field {field} changed from P0"
                )
        replica = int(row["replica_id"])
        seed = evaluation_trajectory_seed(
            base_seed=int(manifest["evaluation"]["base_seed"]),
            eval_manifest_hash=manifest_hash,
            example_id=example_id,
            source_order_index=order,
            replica_id=replica,
        )
        if int(row["trajectory_seed"]) != seed:
            failures.append(f"{attempt_id} terminal row {row_number} seed is not reconstructable")
        expected_trajectory_id = stable_trajectory_id(
            eval_manifest_hash=manifest_hash,
            example_id=example_id,
            replica_id=replica,
            trajectory_seed=seed,
        )
        if row["trajectory_id"] != expected_trajectory_id:
            failures.append(f"{attempt_id} terminal row {row_number} trajectory ID drifted")
        if int(row["source_repeated_row"]) != order * 2 + replica:
            failures.append(f"{attempt_id} terminal row {row_number} repeated-row index drifted")
        token_digest = terminal_records[row_number].get(
            "terminal_response_token_sha256", ""
        )
        if re.fullmatch(r"[0-9a-f]{64}", str(token_digest)) is None:
            failures.append(
                f"{attempt_id} terminal row {row_number} has invalid response-token digest"
            )

    terminal_by_key = {stable_key(row): row for row in terminal_records if not any(
        field not in row for field in OUTPUT_IDENTITY_FIELDS
    )}
    turn_records = read_jsonl(turn_path)
    evidence["turn_row_count"] = len(turn_records)
    evidence["turn_ledger_sha256"] = sha256_file(turn_path)
    required_turn_fields = {
        "record_type",
        *OUTPUT_IDENTITY_FIELDS,
        "active_sample_index",
        "request_seed",
        "configured_request_seed",
        "rollout_request_seed",
        "request_prompt_token_sha256",
        "returned_prompt_token_sha256",
        "rollout_worker_rank",
        "is_final",
        "trajectory_turn",
        "response_token_sha256",
    }
    seen_turns: set[tuple[tuple[str, str, int], int]] = set()
    turns_by_key: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    worker_ranks: set[int] = set()
    for turn_number, turn in enumerate(turn_records):
        missing = sorted(required_turn_fields - turn.keys())
        if missing:
            failures.append(f"{attempt_id} turn row {turn_number} missing fields {missing}")
            continue
        if turn["record_type"] != "trajectory_turn":
            failures.append(f"{attempt_id} turn row {turn_number} has wrong record type")
        turn_integer_fields = (
            "semantic_dataset_index",
            "source_order_index",
            "raw_row_position",
            "production_effective_position",
            "context_token_count",
            "replica_id",
            "source_repeated_row",
            "trajectory_seed",
            "active_sample_index",
            "request_seed",
            "configured_request_seed",
            "rollout_request_seed",
            "rollout_worker_rank",
            "trajectory_turn",
        )
        invalid_turn_integers = [
            field for field in turn_integer_fields
            if not _is_json_integer(turn[field])
        ]
        if invalid_turn_integers:
            failures.append(
                f"{attempt_id} turn row {turn_number} has non-integer fields "
                f"{invalid_turn_integers}"
            )
            continue
        key = stable_key(turn)
        terminal = terminal_by_key.get(key)
        if terminal is None:
            failures.append(f"{attempt_id} turn row {turn_number} has no terminal stable key")
            continue
        for field in OUTPUT_IDENTITY_FIELDS:
            if turn[field] != terminal[field]:
                failures.append(
                    f"{attempt_id} turn row {turn_number} field {field} differs from terminal row"
                )
        recurrent_turn = int(turn["trajectory_turn"])
        unique_turn = (key, recurrent_turn)
        if unique_turn in seen_turns:
            failures.append(f"{attempt_id} duplicate trajectory turn {unique_turn}")
        seen_turns.add(unique_turn)
        turns_by_key[key].append(turn)
        expected_request = derive_turn_request_seeds(
            [int(turn["trajectory_seed"])], [0], recurrent_turn
        )[0]
        requested = int(turn["request_seed"])
        configured = int(turn["configured_request_seed"])
        rollout_alias = int(turn["rollout_request_seed"])
        if (
            requested != expected_request
            or configured != requested
            or rollout_alias != configured
        ):
            failures.append(
                f"{attempt_id} configured request seed mismatch at {unique_turn}: "
                f"expected={expected_request}, requested={requested}, configured={configured}, "
                f"rollout_alias={rollout_alias}"
            )
        requested_prompt = str(turn["request_prompt_token_sha256"])
        returned_prompt = str(turn["returned_prompt_token_sha256"])
        if requested_prompt != returned_prompt:
            failures.append(
                f"{attempt_id} vLLM prompt-token return mismatch at {unique_turn}"
            )
        for label, digest in (
            ("requested", requested_prompt),
            ("returned", returned_prompt),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                failures.append(
                    f"{attempt_id} {label} prompt digest is invalid at {unique_turn}"
                )
        rank = int(turn["rollout_worker_rank"])
        worker_ranks.add(rank)
        if rank not in (0, 1):
            failures.append(f"{attempt_id} turn row {turn_number} has invalid worker rank {rank}")
        if int(turn["active_sample_index"]) != int(turn["source_repeated_row"]):
            failures.append(f"{attempt_id} turn row {turn_number} lost repeated-row alignment")
        if not isinstance(turn["is_final"], bool):
            failures.append(f"{attempt_id} turn row {turn_number} is_final is not boolean")
        if re.fullmatch(r"[0-9a-f]{64}", str(turn["response_token_sha256"])) is None:
            failures.append(f"{attempt_id} turn row {turn_number} has invalid response digest")

    if worker_ranks != {0, 1}:
        failures.append(f"{attempt_id} vLLM worker-rank coverage {sorted(worker_ranks)} != [0, 1]")
    if set(turns_by_key) != set(terminal_by_key):
        failures.append(f"{attempt_id} turn/terminal stable-key coverage differs")
    turn_schedule = checked.get("execution_binding", {}).get(
        "canary_turn_schedule", {}
    )
    active_turn_count_by_order = turn_schedule.get(
        "active_turn_count_by_source_order", {}
    )
    expected_final_turn = int(turn_schedule.get("shared_final_turn", -1))
    for key, rows in turns_by_key.items():
        final_turns = sorted(int(row["trajectory_turn"]) for row in rows if row["is_final"])
        active_turns = sorted(int(row["trajectory_turn"]) for row in rows if not row["is_final"])
        if len(final_turns) != 1:
            failures.append(f"{attempt_id} trajectory {key} must have one final turn: {final_turns}")
        else:
            final_turn = final_turns[0]
            source_order = int(terminal_by_key[key]["source_order_index"])
            expected_active_count = int(
                active_turn_count_by_order.get(str(source_order), -1)
            )
            expected_active_turns = list(range(expected_active_count))
            if active_turns != expected_active_turns:
                failures.append(
                    f"{attempt_id} trajectory {key} active turns differ from frozen context schedule: "
                    f"{active_turns} != {expected_active_turns}"
                )
            if final_turn != expected_final_turn:
                failures.append(
                    f"{attempt_id} trajectory {key} final turn {final_turn} != "
                    f"frozen shared final turn {expected_final_turn}"
                )
            final_record = next(row for row in rows if row["is_final"])
            terminal_digest = terminal_by_key[key].get(
                "terminal_response_token_sha256"
            )
            if final_record["response_token_sha256"] != terminal_digest:
                failures.append(
                    f"{attempt_id} trajectory {key} final response digest is not bound "
                    "to its terminal output row"
                )

    def key_text(row: Mapping[str, Any]) -> str:
        return json.dumps(stable_key(row), separators=(",", ":"))

    evidence["terminal_result_by_stable_key"] = {
        key_text(row): {
            "terminal_response_token_sha256": row.get(
                "terminal_response_token_sha256"
            ),
            "output": row.get("output"),
            "score": row.get("score"),
        }
        for row in terminal_records
        if not any(field not in row for field in OUTPUT_IDENTITY_FIELDS)
    }
    evidence["turn_path_by_stable_key"] = {
        f"{key_text(row)}:{int(row['trajectory_turn'])}": {
            "response_token_sha256": row.get("response_token_sha256"),
            "is_final": row.get("is_final"),
            "request_seed": row.get("request_seed"),
            "configured_request_seed": row.get("configured_request_seed"),
            "rollout_request_seed": row.get("rollout_request_seed"),
            "request_prompt_token_sha256": row.get(
                "request_prompt_token_sha256"
            ),
            "returned_prompt_token_sha256": row.get(
                "returned_prompt_token_sha256"
            ),
            "active_sample_index": row.get("active_sample_index"),
        }
        for row in turn_records
        if not any(field not in row for field in required_turn_fields)
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence["execution_summary_sha256"] = sha256_file(summary_path)
    expected_runtime_config_sha = (
        checked.get("execution_binding", {})
        .get("trainer_configuration", {})
        .get("attempts", {})
        .get(attempt_id, {})
        .get("resolved_config_sha256")
    )
    if re.fullmatch(r"[0-9a-f]{64}", str(expected_runtime_config_sha or "")) is None:
        failures.append(f"{attempt_id} P0 lacks a frozen resolved Hydra config hash")
    expected_summary_fields = {
        "record_type": "execution_summary",
        "interface_id": "I",
        "attempt_id": attempt_id,
        "eval_manifest_hash": manifest_hash,
        "resolved_runtime_config_sha256": expected_runtime_config_sha,
        "global_step": 0,
        "actor_update_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_save_calls": 0,
        "resume_mode": "disable",
        "validation_only": True,
    }
    for field in (
        "global_step",
        "actor_update_calls",
        "optimizer_step_calls",
        "checkpoint_save_calls",
    ):
        if not _is_json_integer(summary.get(field)):
            failures.append(f"{attempt_id} execution summary field {field} is not an integer")
    if not isinstance(summary.get("validation_only"), bool):
        failures.append(f"{attempt_id} execution summary validation_only is not boolean")
    for field, expected_value in expected_summary_fields.items():
        if summary.get(field) != expected_value:
            failures.append(
                f"{attempt_id} execution summary field {field} "
                f"{summary.get(field)!r} != {expected_value!r}"
            )
    snapshot_fields = (
        "actor_master_sampled_tensor_digest",
        "actor_rollout_sampled_tensor_digest",
        "vllm_sampled_tensor_digest",
        "worker_ranks",
        "worker_evidence",
    )
    before = summary.get("weight_snapshot_before")
    after = summary.get("weight_snapshot_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        failures.append(f"{attempt_id} is missing before/after read-only weight snapshots")
    else:
        if before.get("sync_kind") != "stable_eval_before":
            failures.append(f"{attempt_id} before snapshot has wrong sync kind")
        if after.get("sync_kind") != "stable_eval_after":
            failures.append(f"{attempt_id} after snapshot has wrong sync kind")
        for field in snapshot_fields:
            if before.get(field) != after.get(field):
                failures.append(f"{attempt_id} weight/optimizer evidence changed: {field}")
        required_ranks = list(manifest["weight_snapshot"]["required_worker_ranks"])
        if before.get("worker_ranks") != required_ranks or after.get("worker_ranks") != required_ranks:
            failures.append(f"{attempt_id} sampled-weight worker ranks are not {required_ranks}")
        digest_fields = snapshot_fields[:3]
        for phase, snapshot in (("before", before), ("after", after)):
            for field in digest_fields:
                if re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get(field, ""))) is None:
                    failures.append(f"{attempt_id} {phase} snapshot has invalid {field}")
            if snapshot.get("actor_rollout_sampled_tensor_digest") != snapshot.get(
                "vllm_sampled_tensor_digest"
            ):
                failures.append(f"{attempt_id} {phase} actor-effective/vLLM digest mismatch")
            workers = snapshot.get("worker_evidence")
            if not isinstance(workers, list) or len(workers) != 2:
                failures.append(f"{attempt_id} {phase} snapshot lacks two worker evidence rows")
                continue
            for rank, worker in zip(required_ranks, workers):
                if not isinstance(worker, dict):
                    failures.append(f"{attempt_id} {phase} worker {rank} evidence is not an object")
                    continue
                for integer_field in (
                    "optimizer_state_entry_count",
                    "optimizer_step_entry_count",
                    "lr_scheduler_last_epoch",
                    "loaded_parameter_count",
                    "model_parameter_count",
                ):
                    if not _is_json_integer(worker.get(integer_field)):
                        failures.append(
                            f"{attempt_id} {phase} worker {rank} field "
                            f"{integer_field} is not an integer"
                        )
                if worker.get("optimizer_state_entry_count") != 0:
                    failures.append(f"{attempt_id} {phase} worker {rank} optimizer state is non-empty")
                if worker.get("optimizer_step_entry_count") != 0:
                    failures.append(f"{attempt_id} {phase} worker {rank} has optimizer steps")
                if worker.get("optimizer_step_min") is not None or worker.get("optimizer_step_max") is not None:
                    failures.append(f"{attempt_id} {phase} worker {rank} has optimizer step bounds")
                if worker.get("optimizer_step_histogram") != {}:
                    failures.append(f"{attempt_id} {phase} worker {rank} optimizer histogram is non-empty")
                expected_loaded = int(manifest["weight_snapshot"]["expected_loaded_parameter_count"])
                if worker.get("loaded_parameter_count") != expected_loaded or worker.get(
                    "model_parameter_count"
                ) != expected_loaded:
                    failures.append(f"{attempt_id} {phase} worker {rank} load coverage is incomplete")
                if worker.get("loaded_parameter_names_sha256") != worker.get(
                    "model_parameter_names_sha256"
                ):
                    failures.append(f"{attempt_id} {phase} worker {rank} parameter-name digest mismatch")
                for name_digest_field in (
                    "loaded_parameter_names_sha256",
                    "model_parameter_names_sha256",
                ):
                    if re.fullmatch(
                        r"[0-9a-f]{64}", str(worker.get(name_digest_field, ""))
                    ) is None:
                        failures.append(
                            f"{attempt_id} {phase} worker {rank} has invalid "
                            f"{name_digest_field}"
                        )
                if worker.get("weight_transfer_format") != manifest["weight_snapshot"][
                    "transfer_format"
                ]:
                    failures.append(f"{attempt_id} {phase} worker {rank} transfer format drifted")
                if sorted(worker.get("audited_loaded_parameters") or []) != sorted(
                    manifest["weight_snapshot"]["parameter_names"]
                ):
                    failures.append(f"{attempt_id} {phase} worker {rank} sampled coverage drifted")
                if sorted((worker.get("sampled_parameter_dtypes") or {}).keys()) != sorted(
                    manifest["weight_snapshot"]["parameter_names"]
                ):
                    failures.append(f"{attempt_id} {phase} worker {rank} dtype coverage drifted")
                elif set(worker["sampled_parameter_dtypes"].values()) != {
                    manifest["weight_snapshot"]["expected_sampled_parameter_dtype"]
                }:
                    failures.append(
                        f"{attempt_id} {phase} worker {rank} sampled dtype values drifted"
                    )
        if before.get("vllm_pre_sync_sampled_tensor_digest") is not None:
            failures.append(
                f"{attempt_id} before snapshot pre-sync digest must be not-applicable"
            )
        after_pre_sync = after.get("vllm_pre_sync_sampled_tensor_digest")
        if re.fullmatch(r"[0-9a-f]{64}", str(after_pre_sync or "")) is None:
            failures.append(f"{attempt_id} after snapshot has invalid pre-sync vLLM digest")
        elif after_pre_sync != before.get("vllm_sampled_tensor_digest"):
            failures.append(
                f"{attempt_id} validation-time vLLM drift was hidden by final sync"
            )
        evidence["weight_snapshot_before"] = before
        evidence["weight_snapshot_after"] = after

    forbidden_files = sorted(
        str(path.relative_to(attempt_root))
        for path in attempt_root.rglob("*")
        if path.is_file() and (
            path.suffix == ".pt"
            or any(part.startswith("global_step_") for part in path.parts)
            or path.name.startswith(("optim_", "model_world_size_", "extra_state_"))
        )
    )
    evidence["forbidden_checkpoint_files"] = forbidden_files
    if forbidden_files:
        failures.append(f"{attempt_id} created forbidden checkpoint artifacts: {forbidden_files}")
    return terminal_rows, failures, evidence


def audit_execution_ledger(
    records: list[dict[str, Any]], *, manifest: Mapping[str, Any], run_id: str, commit: str,
    eval_manifest_hash: str, execution_binding_sha256: str,
    runtime_binding_sha256: str, schema: Mapping[str, Any]
) -> list[str]:
    failures = validate_jsonl_chain(records)
    failures.extend(validate_ledger_schema(records, schema))
    expected_sequence = [
        ("s0_preflight", None),
        ("run_start", "repeat_a"),
        ("run_finish", "repeat_a"),
        ("run_start", "repeat_b"),
        ("run_finish", "repeat_b"),
    ]
    if records and records[-1].get("record_type") == "audit_result":
        expected_sequence.append(("audit_result", None))
    actual_sequence = [
        (record.get("record_type"), record.get("attempt_id")) for record in records
    ]
    if actual_sequence != expected_sequence:
        failures.append(
            f"execution ledger order {actual_sequence} != preregistered {expected_sequence}"
        )
    for index, record in enumerate(records):
        if record.get("run_id") != run_id:
            failures.append(f"ledger record {index} run_id changed")
        if record.get("git_commit") != commit:
            failures.append(f"ledger record {index} Git commit changed")
        if record.get("eval_manifest_hash") != eval_manifest_hash:
            failures.append(f"ledger record {index} evaluation manifest hash changed")
        if record.get("execution_binding_sha256") != execution_binding_sha256:
            failures.append(f"ledger record {index} execution/model binding changed")
        if record.get("runtime_binding_sha256") != runtime_binding_sha256:
            failures.append(f"ledger record {index} mutable runtime binding changed")
    if records:
        p0_path = Path(manifest["paths"]["p0_certificate"])
        if records[0].get("artifact") != str(p0_path):
            failures.append("P0 ledger artifact path changed")
        if not p0_path.is_file() or records[0].get("artifact_sha256") != sha256_file(p0_path):
            failures.append("P0 certificate is not protected by the execution ledger")
    if records and records[-1].get("record_type") == "audit_result":
        audit_record = records[-1]
        if audit_record.get("status") != "PASS" or audit_record.get("decision") != (
            "I_RECURRENT_IDENTITY_CANARY_PASS"
        ):
            failures.append("persisted audit_result is not the frozen PASS decision")
    finish_records = {
        record.get("attempt_id"): record
        for record in records if record.get("record_type") == "run_finish"
    }
    start_records = {
        record.get("attempt_id"): record
        for record in records if record.get("record_type") == "run_start"
    }
    required_artifacts = {
        "terminal/0.jsonl",
        "trajectory_turns.jsonl",
        "execution_summary.json",
        "run.log",
    }
    for attempt_id in manifest["evaluation"]["attempts"]:
        start = start_records.get(attempt_id)
        if start is not None and (
            start.get("status") != "PASS" or start.get("artifacts") != {}
        ):
            failures.append(
                f"{attempt_id} run_start must be PASS with an empty artifact inventory"
            )
        record = finish_records.get(attempt_id)
        if not record:
            continue
        artifacts = record.get("artifacts")
        if record.get("status") != "PASS" or not isinstance(artifacts, dict):
            failures.append(f"{attempt_id} run_finish is not a PASS artifact inventory")
            continue
        actual_artifacts = set(artifacts)
        if actual_artifacts != required_artifacts:
            failures.append(
                f"{attempt_id} run_finish artifact keys {sorted(actual_artifacts)} "
                f"!= {sorted(required_artifacts)}"
            )
        root = Path(manifest["paths"][attempt_id])
        for relative, item in artifacts.items():
            path = root / relative
            if not path.is_file():
                failures.append(f"ledger-frozen artifact is missing: {path}")
            elif sha256_file(path) != item.get("sha256") or path.stat().st_size != int(
                item.get("size", -1)
            ):
                failures.append(f"ledger-frozen artifact changed: {path}")
    if records and records[-1].get("record_type") == "audit_result":
        audit_record = records[-1]
        report_path = Path(str(audit_record.get("artifact", "")))
        if not report_path.is_file():
            failures.append(f"ledger-frozen final report is missing: {report_path}")
        elif sha256_file(report_path) != audit_record.get("artifact_sha256"):
            failures.append(f"ledger-frozen final report changed: {report_path}")
    return failures


def run_audit(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    repo = Path(manifest["repository"])
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    status = git(repo, "status", "--porcelain")
    expected_commit = os.environ["MEMAGENT_STABLE_I_EXPECTED_COMMIT"]
    evidence.update(git_commit=head, branch=branch, worktree_clean=not status)
    if head != expected_commit:
        failures.append(f"HEAD/expected commit mismatch: {head} != {expected_commit}")
    if branch != EXPECTED_BRANCH:
        failures.append(f"branch mismatch: {branch} != {EXPECTED_BRANCH}")
    if status:
        failures.append(f"worktree is dirty: {status.splitlines()}")

    p0_path = Path(manifest["paths"]["p0_certificate"])
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    if not p0_path.is_file() or not resolved_path.is_file():
        return {
            "status": "FAIL",
            "decision": "STABLE_I_NO_GO:AUDIT",
            "failures": [*failures, "P0 certificate/resolved manifest is missing"],
            "evidence": evidence,
        }
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    resolved = validate_resolved_manifest(json.loads(resolved_path.read_text(encoding="utf-8")))
    if p0.get("status") != "PASS":
        failures.append("P0 certificate is not PASS")
    if p0.get("evidence", {}).get("resolved_manifest_sha256") != sha256_file(resolved_path):
        failures.append("resolved identity manifest changed after P0")
    if p0.get("evidence", {}).get("eval_manifest_hash") != resolved["eval_manifest_hash"]:
        failures.append("P0 and resolved evaluation manifest hashes differ")
    identity_rows = resolved["identity_payload"]["rows"]
    if len(identity_rows) != 128:
        failures.append(f"resolved manifest does not freeze all 128 existing rows: {len(identity_rows)}")
    if resolved.get("canary") != {
        "source_order_indices": [0, 1, 2, 3],
        "examples": 4,
        "replicas": 2,
        "attempts": ["repeat_a", "repeat_b"],
    }:
        failures.append("resolved canary positions are not the preregistered first four rows")
    expected_execution_binding = build_execution_binding(
        manifest,
        repo=repo,
        git_commit=head,
        rows=identity_rows,
        trainer_configuration=freeze_trainer_configuration(
            manifest,
            repo=repo,
            eval_manifest_hash=resolved["eval_manifest_hash"],
        ),
    )
    if resolved.get("execution_binding") != expected_execution_binding:
        failures.append("I canary execution/model binding changed after P0")
    forbidden_identity_keys = {
        "interface_id",
        "checkpoint_inventory_hash",
        "checkpoint_inventory_sha256",
        "model_artifact",
        "git_commit",
        "recurrent",
        "canary_source_order_indices",
    }
    leaked_identity_keys = sorted(forbidden_identity_keys & resolved["identity_payload"].keys())
    if leaked_identity_keys:
        failures.append(
            f"stable join hash contains run/interface-specific keys: {leaked_identity_keys}"
        )

    attempts: list[list[dict[str, Any]]] = []
    attempt_evidence: dict[str, Any] = {}
    for attempt_id in manifest["evaluation"]["attempts"]:
        rows, attempt_failures, details = audit_attempt(
            attempt_id=attempt_id,
            attempt_root=Path(manifest["paths"][attempt_id]),
            resolved_manifest=resolved,
            manifest=manifest,
        )
        attempts.append(rows)
        failures.extend(attempt_failures)
        attempt_evidence[attempt_id] = details
    if all(attempts):
        try:
            validate_repeated_attempts(attempts)
        except Exception as error:
            failures.append(f"cross-attempt stable identity failed: {error}")
    else:
        failures.append("both preregistered attempts must have terminal identity rows")
    if all(
        attempt_evidence.get(attempt_id, {}).get("terminal_result_by_stable_key")
        for attempt_id in manifest["evaluation"]["attempts"]
    ):
        failures.extend(
            compare_deterministic_attempt_evidence(
                attempt_evidence["repeat_a"], attempt_evidence["repeat_b"]
            )
        )
    evidence["attempts"] = attempt_evidence
    if all(
        isinstance(attempt_evidence.get(attempt_id, {}).get("weight_snapshot_before"), dict)
        for attempt_id in manifest["evaluation"]["attempts"]
    ):
        first = attempt_evidence["repeat_a"]["weight_snapshot_before"]
        second = attempt_evidence["repeat_b"]["weight_snapshot_before"]
        for field in (
            "actor_master_sampled_tensor_digest",
            "actor_rollout_sampled_tensor_digest",
            "vllm_sampled_tensor_digest",
            "worker_ranks",
            "worker_evidence",
        ):
            if first.get(field) != second.get(field):
                failures.append(f"fresh standard-model state differs across attempts: {field}")

    ledger_path = Path(manifest["paths"]["execution_ledger"])
    if not ledger_path.is_file():
        failures.append("append-only stable identity execution ledger is missing")
    else:
        records = read_jsonl(ledger_path)
        schema = json.loads(
            (repo / manifest["ledger_schema"]).read_text(encoding="utf-8")
        )
        failures.extend(
            audit_execution_ledger(
                records,
                manifest=manifest,
                run_id=str(p0.get("evidence", {}).get("run_id", "")),
                commit=head,
                eval_manifest_hash=resolved["eval_manifest_hash"],
                execution_binding_sha256=canonical_sha256(
                    resolved["execution_binding"]
                ),
                runtime_binding_sha256=str(
                    p0.get("evidence", {}).get("runtime_binding_sha256", "")
                ),
                schema=schema,
            )
        )
        evidence["execution_ledger_sha256"] = sha256_file(ledger_path)
        evidence["execution_ledger_records"] = len(records)

    evidence["eval_manifest_hash"] = resolved["eval_manifest_hash"]
    evidence["existing_s128_rows_frozen"] = len(identity_rows)
    evidence["canary_shape"] = {
        "examples": 4,
        "replicas_per_example": 2,
        "attempts": 2,
        "terminal_rows_per_attempt": 8,
    }
    return {
        "status": "PASS" if not failures else "FAIL",
        "decision": (
            "I_RECURRENT_IDENTITY_CANARY_PASS"
            if not failures else "STABLE_I_NO_GO:AUDIT"
        ),
        "scope": {
            "interface": "I",
            "identity_transport_only": True,
            "paper_performance_result": False,
            "five_interface_gate": False,
            "q_g_r_t_status": "NOT_RUN_BY_THIS_CANARY",
            "stable_key_domain": "interface-neutral and reusable by future Q/G/R/I/T runs",
            "source_dataset": "existing fixed HotpotQA S128; no new dataset",
        },
        "failures": failures,
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    try:
        result = run_audit(args.manifest)
        manifest = load_manifest(args.manifest)
    except Exception as error:
        result = {
            "status": "FAIL",
            "decision": "STABLE_I_NO_GO:AUDIT",
            "scope": {"paper_performance_result": False, "five_interface_gate": False},
            "failures": [str(error)],
            "evidence": {},
        }
        manifest = None

    if args.write_report and manifest is not None:
        report = Path(manifest["paths"]["final_report"])
        if report.exists():
            raise SystemExit(f"refusing to overwrite append-only audit report: {report}")
        ledger = Path(manifest["paths"]["execution_ledger"])
        records = read_jsonl(ledger)
        if any(record.get("record_type") == "audit_result" for record in records):
            raise SystemExit("refusing to append a second audit_result record")
        p0 = json.loads(Path(manifest["paths"]["p0_certificate"]).read_text(encoding="utf-8"))
        resolved = validate_resolved_manifest(
            json.loads(Path(manifest["paths"]["resolved_manifest"]).read_text(encoding="utf-8"))
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        append_jsonl(
            ledger,
            {
                "record_type": "audit_result",
                "experiment_name": "qwen25_7b_h20_2gpu_stable_i4x2_audit_seed2026_20260821",
                "git_commit": p0["evidence"]["git_commit"],
                "run_id": p0["evidence"]["run_id"],
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "eval_manifest_hash": resolved["eval_manifest_hash"],
                "execution_binding_sha256": canonical_sha256(
                    resolved["execution_binding"]
                ),
                "runtime_binding_sha256": p0["evidence"][
                    "runtime_binding_sha256"
                ],
                "attempt_id": None,
                "status": result["status"],
                "decision": result["decision"],
                "artifact": str(report),
                "artifact_sha256": sha256_file(report),
                "row_count": 16,
            },
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
