#!/usr/bin/env python3
"""Read-only audit and certificate writer for fixed-S128 I/T25 evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import append_jsonl, validate_jsonl_chain
from recurrent.research.s128_hotpot_metrics import (
    paired_descriptive_summary,
    score_terminal_output,
    summarize_fixed_s128,
)
from recurrent.research.stable_eval_identity import (
    MANIFEST_ROW_FIELDS,
    OUTPUT_IDENTITY_FIELDS,
    canonical_sha256,
    evaluation_trajectory_seed,
    stable_key,
    stable_trajectory_id,
    validate_attempt_identity_rows,
    validate_resolved_manifest,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds
from tools.h20.preflight_qwen25_7b_s128_it import (
    EXPECTED_BRANCH,
    EXPECTED_INTERFACES,
    _attempt_id,
    _checkpoint_contract,
    _expected_step,
    _load_parquet_rows,
    _stable_canary_contract,
    audit_code_commit,
    build_execution_binding,
    freeze_trainer_configuration,
    git,
    load_manifest,
    sha256_file,
)
from tools.h20.audit_qwen25_7b_stable_i4x2 import validate_ledger_schema


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha(value: object) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is not None


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


STRICT_IDENTITY_INTEGER_FIELDS = (
    "semantic_dataset_index",
    "source_order_index",
    "raw_row_position",
    "production_effective_position",
    "context_token_count",
    "replica_id",
    "source_repeated_row",
    "trajectory_seed",
)


def _strict_integer_failures(
    row: Mapping[str, Any], fields: tuple[str, ...], *, label: str
) -> list[str]:
    return [
        f"{label} field {field} must be a JSON integer, got {row.get(field)!r}"
        for field in fields
        if not _integer(row.get(field))
    ]


def _strict_turn_type_failures(row: Mapping[str, Any], *, label: str) -> list[str]:
    failures = _strict_integer_failures(
        row,
        (*STRICT_IDENTITY_INTEGER_FIELDS, "active_sample_index", "request_seed",
         "configured_request_seed", "rollout_request_seed", "rollout_worker_rank",
         "trajectory_turn"),
        label=label,
    )
    if row.get("record_type") != "trajectory_turn":
        failures.append(f"{label} record_type must be exact trajectory_turn")
    if type(row.get("is_final")) is not bool:
        failures.append(f"{label} is_final must be a JSON boolean")
    return failures


def _register_unique_turn(
    seen: set[tuple[tuple[str, str, int], int]],
    key: tuple[str, str, int],
    turn_number: int,
    *,
    label: str,
) -> list[str]:
    key_turn = (key, turn_number)
    if key_turn in seen:
        return [f"{label} duplicate trajectory turn for stable key {key}: {turn_number}"]
    seen.add(key_turn)
    return []


def _audit_t25_training_digest(
    snapshot: object, checkpoint: Mapping[str, Any]
) -> list[str]:
    expected = checkpoint.get("training_effective_actor_vllm_digest")
    if not isinstance(snapshot, Mapping) or not all(
        snapshot.get(field) == expected
        for field in (
            "actor_rollout_sampled_tensor_digest",
            "vllm_sampled_tensor_digest",
        )
    ):
        return [
            "T25 loaded actor-effective/vLLM digest does not equal the training "
            f"version-25 attestation: expected={expected}"
        ]
    return []


def _ground_truth_by_source_order(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any]
) -> dict[int, object]:
    raw_rows = _load_parquet_rows(Path(manifest["data"]["validation"]))
    frozen = resolved["identity_payload"]["rows"]
    result: dict[int, object] = {}
    for row in frozen:
        raw_position = int(row["raw_row_position"])
        source_order = int(row["source_order_index"])
        reward_model = raw_rows[raw_position].get("reward_model")
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)
        if not isinstance(reward_model, Mapping) or "ground_truth" not in reward_model:
            raise ValueError(f"S128 row {source_order} lacks parquet ground truth")
        ground_truth = reward_model["ground_truth"]
        if canonical_sha256(ground_truth) != row["ground_truth_hash"]:
            raise ValueError(f"S128 row {source_order} ground truth differs from P0 hash")
        result[source_order] = ground_truth
    if set(result) != set(range(128)):
        raise ValueError("parquet ground-truth coverage is not source order 0..127")
    return result


def _audit_weight_snapshot(snapshot: object, label: str) -> list[str]:
    failures: list[str] = []
    if not isinstance(snapshot, Mapping):
        return [f"{label} weight snapshot is missing"]
    if snapshot.get("worker_ranks") != [0, 1]:
        failures.append(f"{label} weight snapshot worker ranks are not [0,1]")
    for field in (
        "actor_master_sampled_tensor_digest",
        "actor_rollout_sampled_tensor_digest",
        "vllm_sampled_tensor_digest",
    ):
        if not _sha(snapshot.get(field)):
            failures.append(f"{label} snapshot has invalid {field}")
    if snapshot.get("actor_rollout_sampled_tensor_digest") != snapshot.get(
        "vllm_sampled_tensor_digest"
    ):
        failures.append(f"{label} actor-effective/vLLM digests do not close")
    workers = snapshot.get("worker_evidence")
    if not isinstance(workers, list) or len(workers) != 2:
        failures.append(f"{label} snapshot lacks two worker evidence rows")
        return failures
    for index, worker in enumerate(workers):
        if worker.get("optimizer_state_entry_count") != 0:
            failures.append(f"{label} worker {index} optimizer is not empty")
        if worker.get("optimizer_step_entry_count") != 0:
            failures.append(f"{label} worker {index} has optimizer steps")
        if worker.get("loaded_parameter_count") != worker.get("model_parameter_count"):
            failures.append(f"{label} worker {index} did not sync every vLLM parameter")
        if worker.get("loaded_parameter_count") != 199:
            failures.append(f"{label} worker {index} loaded parameter count is not 199")
        if worker.get("loaded_parameter_names_sha256") != worker.get(
            "model_parameter_names_sha256"
        ):
            failures.append(f"{label} worker {index} loaded/model name digests differ")
        dtypes = worker.get("sampled_parameter_dtypes")
        if not isinstance(dtypes, Mapping) or set(dtypes.values()) != {"torch.bfloat16"}:
            failures.append(f"{label} worker {index} sampled dtype is not exact bfloat16")
    return failures


def _audit_actor_only_load(
    summary: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    acks = summary.get("actor_checkpoint_load_acks")
    if not isinstance(acks, list) or len(acks) != 2:
        return ["T25 summary lacks two actor-only checkpoint load acknowledgements"]
    frozen_by_rank = {
        int(re.search(r"rank_(\d+)\.pt$", item["path"]).group(1)): item
        for item in checkpoint["actor_model_shards"]
    }
    if sorted(int(ack.get("rank", -1)) for ack in acks) != [0, 1]:
        failures.append("T25 actor-only acknowledgement ranks are not [0,1]")
    for ack in acks:
        rank = int(ack.get("rank", -1))
        frozen = frozen_by_rank.get(rank)
        if frozen is None:
            failures.append(f"T25 actor-only ack has unexpected rank {rank}")
            continue
        if ack.get("world_size") != 2 or ack.get("model_loaded") is not True:
            failures.append(f"T25 rank {rank} model/world-size acknowledgement is invalid")
        if Path(str(ack.get("model_shard_path", ""))).resolve() != (
            Path(checkpoint["path"]) / frozen["path"]
        ).resolve():
            failures.append(f"T25 rank {rank} loaded the wrong actor shard path")
        if ack.get("model_shard_size") != frozen["size"]:
            failures.append(f"T25 rank {rank} actor shard size differs from P0")
        if ack.get("model_shard_sha256") != frozen["sha256"]:
            failures.append(f"T25 rank {rank} actor shard SHA differs from P0")
        for field in (
            "optimizer_loaded", "lr_scheduler_loaded", "rng_loaded", "dataloader_loaded"
        ):
            if ack.get(field) is not False:
                failures.append(f"T25 rank {rank} forbidden state flag {field} is not false")
        if ack.get("optimizer_state_entry_count_before") != 0 or ack.get(
            "optimizer_state_entry_count_after"
        ) != 0:
            failures.append(f"T25 rank {rank} optimizer was not empty throughout load")
        if ack.get("lr_scheduler_last_epoch_before") != ack.get(
            "lr_scheduler_last_epoch_after"
        ):
            failures.append(f"T25 rank {rank} LR scheduler changed during actor-only load")
    return failures


def audit_interface(
    *, interface_id: str, manifest: Mapping[str, Any],
    resolved: Mapping[str, Any], ground_truth: Mapping[int, object],
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    root = Path(manifest["paths"][interface_id])
    step = _expected_step(interface_id)
    terminal_path = root / f"terminal/{step}.jsonl"
    turns_path = root / "trajectory_turns.jsonl"
    summary_path = root / "execution_summary.json"
    log_path = root / "run.log"
    for name, path in {
        "terminal": terminal_path, "turn ledger": turns_path,
        "execution summary": summary_path, "run log": log_path,
    }.items():
        if not path.is_file():
            failures.append(f"{interface_id} missing {name}: {path}")
    if failures:
        return failures, evidence, []

    terminal = read_jsonl(terminal_path)
    evidence.update(
        terminal_sha256=sha256_file(terminal_path), terminal_rows=len(terminal),
        turn_ledger_sha256=sha256_file(turns_path),
        execution_summary_sha256=sha256_file(summary_path),
        run_log_sha256=sha256_file(log_path),
    )
    if len(terminal) != 128:
        failures.append(f"{interface_id} terminal denominator {len(terminal)} != 128")
    if [row.get("source_repeated_row") for row in terminal] != list(range(128)):
        failures.append(f"{interface_id} terminal order is not exact source order 0..127")
    identity_rows = [
        {field: row[field] for field in OUTPUT_IDENTITY_FIELDS if field in row}
        for row in terminal
    ]
    try:
        validate_attempt_identity_rows(identity_rows, examples=128, replicas=1)
    except Exception as error:
        failures.append(f"{interface_id} stable identity inventory failed: {error}")

    frozen = {
        (int(row["source_order_index"]), str(row["example_id"])): row
        for row in resolved["identity_payload"]["rows"]
    }
    attempt_id = _attempt_id(manifest, interface_id)
    metric_rows: list[dict[str, Any]] = []
    terminal_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, row in enumerate(terminal):
        missing = sorted(set(OUTPUT_IDENTITY_FIELDS) - row.keys())
        if missing:
            failures.append(f"{interface_id} terminal row {index} missing identity {missing}")
            continue
        type_failures = _strict_integer_failures(
            row, (*STRICT_IDENTITY_INTEGER_FIELDS, "step"),
            label=f"{interface_id} terminal row {index}",
        )
        if type_failures:
            failures.extend(type_failures)
            continue
        if row.get("interface_id") != interface_id or row.get("attempt_id") != attempt_id:
            failures.append(f"{interface_id} terminal row {index} has wrong interface/attempt")
        if row.get("eval_manifest_hash") != resolved["eval_manifest_hash"]:
            failures.append(f"{interface_id} terminal row {index} has wrong manifest hash")
        if row.get("step") != step or not _integer(row.get("step")):
            failures.append(f"{interface_id} terminal row {index} has wrong global step")
        order = int(row["source_order_index"])
        key = (order, str(row["example_id"]))
        if key not in frozen:
            failures.append(f"{interface_id} terminal row {index} is outside S128")
            continue
        for field in MANIFEST_ROW_FIELDS:
            if row[field] != frozen[key][field]:
                failures.append(f"{interface_id} terminal row {index} changed {field}")
        seed = evaluation_trajectory_seed(
            base_seed=int(manifest["evaluation"]["base_seed"]),
            eval_manifest_hash=resolved["eval_manifest_hash"],
            example_id=str(row["example_id"]), source_order_index=order, replica_id=0,
        )
        if row["trajectory_seed"] != seed:
            failures.append(f"{interface_id} terminal row {index} seed is not reconstructable")
        if row["trajectory_id"] != stable_trajectory_id(
            eval_manifest_hash=resolved["eval_manifest_hash"],
            example_id=str(row["example_id"]), replica_id=0, trajectory_seed=seed,
        ):
            failures.append(f"{interface_id} terminal row {index} trajectory ID changed")
        if row["replica_id"] != 0 or row["source_repeated_row"] != order:
            failures.append(f"{interface_id} terminal row {index} is not n=1 aligned")
        if not isinstance(row.get("output"), str):
            failures.append(f"{interface_id} terminal row {index} output is not text")
            output = ""
        else:
            output = row["output"]
        rollout_score = row.get("score")
        if (
            not isinstance(rollout_score, (int, float))
            or isinstance(rollout_score, bool)
            or not math.isfinite(float(rollout_score))
        ):
            failures.append(
                f"{interface_id} terminal row {index} has a non-finite rollout training score"
            )
        if not _sha(row.get("terminal_response_token_sha256")):
            failures.append(f"{interface_id} terminal row {index} response digest is invalid")
        scored = score_terminal_output(output, ground_truth[order])
        metric_rows.append({
            "stable_key": json.dumps(stable_key(row), separators=(",", ":")),
            "source_order_index": order,
            "eval_manifest_hash": row["eval_manifest_hash"],
            "example_id": row["example_id"],
            "replica_id": row["replica_id"],
            "trajectory_seed": row["trajectory_seed"],
            "trajectory_id": row["trajectory_id"],
            **scored,
        })
        terminal_by_key[stable_key(row)] = row

    if len(metric_rows) == 128:
        evidence["metrics"] = summarize_fixed_s128(metric_rows)
        evidence["independent_metric_rows_sha256"] = canonical_sha256(metric_rows)
        evidence["extraction_routes"] = {
            route: sum(row["extraction_route"] == route for row in metric_rows)
            for route in ("boxed", "explicit", "last_line")
        }
    else:
        failures.append(f"{interface_id} independently scored rows {len(metric_rows)} != 128")

    turns = read_jsonl(turns_path)
    required_turn = {
        "record_type", *OUTPUT_IDENTITY_FIELDS, "active_sample_index",
        "request_seed", "configured_request_seed", "rollout_request_seed",
        "request_prompt_token_sha256", "returned_prompt_token_sha256",
        "rollout_worker_rank", "is_final", "trajectory_turn", "response_token_sha256",
    }
    turns_by_key: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    worker_ranks: set[int] = set()
    seen_key_turns: set[tuple[tuple[str, str, int], int]] = set()
    for index, turn in enumerate(turns):
        missing = sorted(required_turn - turn.keys())
        if missing:
            failures.append(f"{interface_id} turn row {index} missing {missing}")
            continue
        type_failures = _strict_turn_type_failures(
            turn, label=f"{interface_id} turn row {index}"
        )
        if type_failures:
            failures.extend(type_failures)
            continue
        key = stable_key(turn)
        terminal_row = terminal_by_key.get(key)
        if terminal_row is None:
            failures.append(f"{interface_id} turn row {index} lacks terminal stable key")
            continue
        for field in OUTPUT_IDENTITY_FIELDS:
            if turn[field] != terminal_row[field]:
                failures.append(f"{interface_id} turn row {index} changed {field}")
        turn_number = int(turn["trajectory_turn"])
        duplicate_failures = _register_unique_turn(
            seen_key_turns,
            key,
            turn_number,
            label=f"{interface_id} turn row {index}",
        )
        if duplicate_failures:
            failures.extend(duplicate_failures)
            continue
        expected_seed = derive_turn_request_seeds(
            [int(turn["trajectory_seed"])], [0], turn_number
        )[0]
        if not (
            turn["request_seed"] == expected_seed
            and turn["configured_request_seed"] == expected_seed
            and turn["rollout_request_seed"] == expected_seed
        ):
            failures.append(f"{interface_id} turn row {index} request seed chain differs")
        if turn["request_prompt_token_sha256"] != turn["returned_prompt_token_sha256"]:
            failures.append(f"{interface_id} turn row {index} vLLM prompt binding differs")
        for field in (
            "request_prompt_token_sha256", "returned_prompt_token_sha256",
            "response_token_sha256",
        ):
            if not _sha(turn[field]):
                failures.append(f"{interface_id} turn row {index} invalid {field}")
        rank = int(turn["rollout_worker_rank"])
        worker_ranks.add(rank)
        if rank not in (0, 1):
            failures.append(f"{interface_id} turn row {index} invalid worker rank {rank}")
        if turn["active_sample_index"] != turn["source_repeated_row"]:
            failures.append(f"{interface_id} turn row {index} lost row alignment")
        turns_by_key[key].append(turn)
    if worker_ranks != {0, 1}:
        failures.append(f"{interface_id} vLLM worker coverage {sorted(worker_ranks)} != [0,1]")
    if set(turns_by_key) != set(terminal_by_key):
        failures.append(f"{interface_id} turn/terminal stable-key coverage differs")
    schedule = resolved["execution_binding"]["all_s128_turn_schedule"]
    shared_final = int(schedule["shared_final_turn"])
    active_by_order = schedule["active_turn_count_by_source_order"]
    for key, rows in turns_by_key.items():
        order = int(terminal_by_key[key]["source_order_index"])
        active_turns = sorted(int(row["trajectory_turn"]) for row in rows if not row["is_final"])
        final_turns = sorted(int(row["trajectory_turn"]) for row in rows if row["is_final"])
        expected_active = list(range(int(active_by_order[str(order)])))
        if active_turns != expected_active or final_turns != [shared_final]:
            failures.append(
                f"{interface_id} trajectory {key} turn schedule differs: "
                f"active={active_turns}, final={final_turns}"
            )
        if final_turns:
            final = next(row for row in rows if row["is_final"])
            if final["response_token_sha256"] != terminal_by_key[key]["terminal_response_token_sha256"]:
                failures.append(f"{interface_id} trajectory {key} final/terminal digest differs")
    evidence["initial_prompt_sha256_by_stable_key"] = {
        json.dumps(key, separators=(",", ":")): next(
            (row["request_prompt_token_sha256"] for row in rows if row["trajectory_turn"] == 0),
            None,
        )
        for key, rows in turns_by_key.items()
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trainer_binding = resolved["execution_binding"]["trainer_configuration"]["interfaces"][interface_id]
    expected_summary = {
        "record_type": "execution_summary",
        "interface_id": interface_id,
        "attempt_id": attempt_id,
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "resolved_runtime_config_sha256": trainer_binding["resolved_config_sha256"],
        "global_step": step,
        "actor_update_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_save_calls": 0,
        "validation_only": True,
        "resume_mode": "disable" if interface_id == "I" else "actor_only_eval",
        "weight_source": "base_model" if interface_id == "I" else "actor_checkpoint",
        "checkpoint_load_mode": "none" if interface_id == "I" else "actor_only",
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            failures.append(
                f"{interface_id} summary {field}={summary.get(field)!r} != {expected!r}"
            )
    expected_source = None if interface_id == "I" else str(Path(manifest["training_anchor"]["checkpoint"]).resolve())
    if summary.get("checkpoint_source") != expected_source:
        failures.append(f"{interface_id} summary checkpoint source differs")
    before, after = summary.get("weight_snapshot_before"), summary.get("weight_snapshot_after")
    failures.extend(_audit_weight_snapshot(before, f"{interface_id}/before"))
    failures.extend(_audit_weight_snapshot(after, f"{interface_id}/after"))
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for field in (
            "actor_master_sampled_tensor_digest", "actor_rollout_sampled_tensor_digest",
            "vllm_sampled_tensor_digest", "worker_ranks", "worker_evidence",
        ):
            if before.get(field) != after.get(field):
                failures.append(f"{interface_id} validation mutated weight state field {field}")
        if after.get("vllm_pre_sync_sampled_tensor_digest") != before.get(
            "vllm_sampled_tensor_digest"
        ):
            failures.append(f"{interface_id} vLLM drifted before final read-only sync")
        evidence["weight_snapshot_before"] = before
    checkpoint = resolved["execution_binding"]["model_artifacts"]["T25"]
    if interface_id == "I":
        if summary.get("actor_checkpoint_load_acks") != []:
            failures.append("I unexpectedly contains checkpoint load acknowledgements")
    else:
        failures.extend(_audit_actor_only_load(summary, checkpoint))
        failures.extend(_audit_t25_training_digest(before, checkpoint))
    return failures, evidence, metric_rows


def _audit_ledger(
    records: list[dict[str, Any]], *, manifest: Mapping[str, Any],
    resolved: Mapping[str, Any], p0: Mapping[str, Any],
    expected_audit_code_commit: str | None = None,
) -> list[str]:
    failures = validate_jsonl_chain(records)
    schema = json.loads(
        (Path(manifest["repository"]) / manifest["ledger_schema"]).read_text(
            encoding="utf-8"
        )
    )
    failures.extend(validate_ledger_schema(records, schema))
    expected_prefix = [
        ("s0_preflight", None),
        ("interface_start", "I"), ("interface_finish", "I"),
        ("interface_start", "T25"), ("interface_finish", "T25"),
    ]
    actual = [(row.get("record_type"), row.get("interface_id")) for row in records]
    if actual not in (expected_prefix, [*expected_prefix, ("audit_result", None)]):
        failures.append(f"ledger sequence differs from frozen sequence: {actual}")
    expected_experiments = {
        0: "qwen25_7b_s128_it_p0_seed2026_20260821",
        1: "qwen25_7b_s128_i_base_seed2026_20260821",
        2: "qwen25_7b_s128_i_base_seed2026_20260821",
        3: "qwen25_7b_s128_t25_corrected_original_style_seed2026_20260821",
        4: "qwen25_7b_s128_t25_corrected_original_style_seed2026_20260821",
        5: "qwen25_7b_s128_it_audit_seed2026_20260821",
    }
    execution_sha = canonical_sha256(resolved["execution_binding"])
    runtime_sha = p0["evidence"].get("runtime_binding_sha256")
    for index, row in enumerate(records):
        if row.get("git_commit") != p0["evidence"].get("git_commit"):
            failures.append(f"ledger record {index} Git commit differs")
        if row.get("run_id") != p0["evidence"].get("run_id"):
            failures.append(f"ledger record {index} run ID differs")
        if row.get("eval_manifest_hash") != resolved["eval_manifest_hash"]:
            failures.append(f"ledger record {index} eval hash differs")
        if row.get("execution_binding_sha256") != execution_sha:
            failures.append(f"ledger record {index} execution binding differs")
        if row.get("runtime_binding_sha256") != runtime_sha:
            failures.append(f"ledger record {index} runtime binding differs")
        if row.get("status") != "PASS":
            failures.append(f"ledger record {index} status is not PASS")
        if row.get("experiment_name") != expected_experiments.get(index):
            failures.append(f"ledger record {index} experiment name differs")
        if index >= 2 and expected_audit_code_commit is not None:
            if row.get("audit_code_commit") != expected_audit_code_commit:
                failures.append(f"ledger record {index} audit-code commit differs")
    if records:
        p0_path = Path(manifest["paths"]["p0_certificate"])
        p0_record = records[0]
        if (
            p0_record.get("interface_id") is not None
            or p0_record.get("row_count") != 128
            or Path(str(p0_record.get("artifact", ""))).resolve() != p0_path.resolve()
            or not p0_path.is_file()
            or p0_record.get("artifact_sha256") != sha256_file(p0_path)
        ):
            failures.append("ledger P0 record does not authenticate the exact 128-row certificate")
    for start_index in (1, 3):
        if len(records) > start_index and records[start_index].get("artifacts") != {}:
            failures.append(f"ledger interface start record {start_index} artifacts are not empty")
    for interface_id, finish_index in (("I", 2), ("T25", 4)):
        if len(records) <= finish_index:
            continue
        root = Path(manifest["paths"][interface_id])
        artifacts = records[finish_index].get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            f"terminal/{_expected_step(interface_id)}.jsonl",
            "trajectory_turns.jsonl", "execution_summary.json", "run.log",
        }:
            failures.append(f"ledger {interface_id} artifact inventory is not exact")
            continue
        for relative, frozen in artifacts.items():
            path = root / relative
            if not path.is_file() or path.stat().st_size != frozen.get("size") or sha256_file(path) != frozen.get("sha256"):
                failures.append(f"ledger-frozen {interface_id} artifact changed: {relative}")
    if actual and actual[-1] == ("audit_result", None):
        report = Path(manifest["paths"]["final_report"])
        if (
            records[-1].get("decision") != "S128_IT_PERFORMANCE_PASS"
            or records[-1].get("row_count") != 256
            or Path(str(records[-1].get("artifact", ""))).resolve() != report.resolve()
            or not report.is_file()
            or sha256_file(report) != records[-1].get("artifact_sha256")
        ):
            failures.append("ledger-frozen final report changed")
    return failures


def run_audit(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    repo = Path(manifest["repository"])
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    status = git(repo, "status", "--porcelain")
    expected_commit = os.environ["MEMAGENT_S128_IT_EXPECTED_COMMIT"]
    try:
        audit_commit = audit_code_commit()
        evidence["audit_code_commit"] = audit_commit
    except Exception as error:
        failures.append(f"audit-code binding failed: {error}")
        audit_commit = None
    if head != expected_commit or branch != EXPECTED_BRANCH or status:
        failures.append(
            f"Git binding differs: head={head}, expected={expected_commit}, branch={branch}, dirty={bool(status)}"
        )
    p0_path = Path(manifest["paths"]["p0_certificate"])
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    if not p0_path.is_file() or not resolved_path.is_file():
        return {
            "status": "FAIL", "decision": "S128_IT_NO_GO:AUDIT",
            "failures": [*failures, "P0 certificate/resolved manifest is missing"],
            "evidence": evidence,
        }
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    resolved = validate_resolved_manifest(json.loads(resolved_path.read_text(encoding="utf-8")))
    if p0.get("status") != "PASS":
        failures.append("P0 is not PASS")
    if p0.get("evidence", {}).get("resolved_manifest_sha256") != sha256_file(resolved_path):
        failures.append("resolved manifest changed after P0")
    rows = resolved["identity_payload"]["rows"]
    if len(rows) != 128 or resolved.get("cohort") != {
        "source_order_indices": list(range(128)), "examples": 128,
        "replicas": 1, "interfaces": ["I", "T25"],
    }:
        failures.append("resolved cohort is not the complete fixed S128 n=1 I/T cohort")
    try:
        canary = _stable_canary_contract(
            manifest, expected_eval_manifest_hash=resolved["eval_manifest_hash"]
        )
        if resolved["execution_binding"].get("stable_identity_canary_prerequisite") != canary:
            failures.append("stable-I canary binding changed after P0")
    except Exception as error:
        failures.append(f"stable-I canary prerequisite failed: {error}")
    try:
        checkpoint = _checkpoint_contract(manifest, expected_git_commit=expected_commit)
        if resolved["execution_binding"]["model_artifacts"]["T25"] != checkpoint:
            failures.append("T25 checkpoint inventory/report changed after P0")
    except Exception as error:
        failures.append(f"T25 checkpoint audit failed: {error}")
        checkpoint = None
    if checkpoint is not None:
        try:
            trainer = freeze_trainer_configuration(
                manifest, repo=repo, eval_manifest_hash=resolved["eval_manifest_hash"]
            )
            expected_execution = build_execution_binding(
                manifest, repo=repo, rows=rows, checkpoint=checkpoint,
                stable_canary=resolved["execution_binding"]["stable_identity_canary_prerequisite"],
                trainer_configuration=trainer,
            )
            if resolved["execution_binding"] != expected_execution:
                failures.append("resolved executable/protocol binding changed after P0")
        except Exception as error:
            failures.append(f"cannot reconstruct executable/protocol binding: {error}")
    try:
        ground_truth = _ground_truth_by_source_order(manifest, resolved)
    except Exception as error:
        failures.append(f"cannot bind parquet ground truth: {error}")
        ground_truth = {}

    all_metrics: dict[str, list[dict[str, Any]]] = {}
    interface_evidence: dict[str, Any] = {}
    for interface_id in EXPECTED_INTERFACES:
        if len(ground_truth) == 128:
            interface_failures, details, metric_rows = audit_interface(
                interface_id=interface_id, manifest=manifest,
                resolved=resolved, ground_truth=ground_truth,
            )
            failures.extend(interface_failures)
            interface_evidence[interface_id] = details
            all_metrics[interface_id] = metric_rows
    evidence["interfaces"] = interface_evidence
    if all(len(all_metrics.get(name, [])) == 128 for name in EXPECTED_INTERFACES):
        i_by_key = {row["stable_key"]: row for row in all_metrics["I"]}
        t_by_key = {row["stable_key"]: row for row in all_metrics["T25"]}
        if set(i_by_key) != set(t_by_key):
            failures.append("I/T stable-key coverage differs")
        for key in set(i_by_key) & set(t_by_key):
            for field in ("eval_manifest_hash", "example_id", "source_order_index", "replica_id", "trajectory_seed", "trajectory_id"):
                if i_by_key[key][field] != t_by_key[key][field]:
                    failures.append(f"I/T stable identity differs at {key}: {field}")
        evidence["paired_descriptive_t25_minus_i"] = paired_descriptive_summary(
            all_metrics["I"], all_metrics["T25"]
        )
        i_prompts = interface_evidence["I"].get("initial_prompt_sha256_by_stable_key")
        t_prompts = interface_evidence["T25"].get("initial_prompt_sha256_by_stable_key")
        if i_prompts != t_prompts:
            failures.append("I/T initial recurrent prompt-token digests differ")
        i_weight = interface_evidence["I"].get("weight_snapshot_before", {})
        t_weight = interface_evidence["T25"].get("weight_snapshot_before", {})
        if i_weight.get("actor_master_sampled_tensor_digest") == t_weight.get(
            "actor_master_sampled_tensor_digest"
        ):
            failures.append("T25 sampled actor digest did not differ from base I")

    ledger = Path(manifest["paths"]["execution_ledger"])
    if not ledger.is_file():
        failures.append("append-only S128 I/T ledger is missing")
    else:
        records = read_jsonl(ledger)
        failures.extend(_audit_ledger(
            records, manifest=manifest, resolved=resolved, p0=p0,
            expected_audit_code_commit=audit_commit,
        ))
        evidence["execution_ledger_sha256"] = sha256_file(ledger)
        evidence["execution_ledger_records"] = len(records)
    evidence.update(
        eval_manifest_hash=resolved["eval_manifest_hash"],
        existing_fixed_s128_rows=128,
        shared_generation_protocol_sha256=resolved["execution_binding"]["trainer_configuration"]["shared_generation_protocol_sha256"],
        metric_code_sha256=sha256_file(repo / "recurrent/research/s128_hotpot_metrics.py"),
        rollout_dense_training_reward_fields_ignored_for_performance=True,
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "decision": "S128_IT_PERFORMANCE_PASS" if not failures else "S128_IT_NO_GO:AUDIT",
        "primary_metrics": {
            name: interface_evidence.get(name, {}).get("metrics")
            for name in EXPECTED_INTERFACES
        },
        "paired_descriptive_t25_minus_i": evidence.get("paired_descriptive_t25_minus_i"),
        "claim_boundaries": {
            "comparison": "same-protocol base recurrent-I versus corrected Original-style 2-GPU pilot T25",
            "t25_minus_i_meaning": "total change from base through 25 corrected Original-style updates; current continuation adds steps 4-25 (22 updates)",
            "causal": False,
            "population_inference": False,
            "dataset": "curated fixed memory challenge from original project question-only filter then head(128), not random HotpotQA dev",
            "not_original_paper_7b_reproduction": True,
            "published_R": "historical reference only; not rerun and not directly paired because execution protocols differ",
            "dense_training_reward": "ignored for performance; normalized EM/F1 independently recomputed",
            "historical_sub_em": "diagnostic only",
            "future_method_contrast": "method25-Original25 isolates the replacement for updates 4-25; method25-I would include shared first three updates",
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
            "status": "FAIL", "decision": "S128_IT_NO_GO:AUDIT",
            "failures": [str(error)], "evidence": {},
        }
        manifest = None
    if args.write_report and manifest is not None:
        if result["status"] != "PASS":
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
        report = Path(manifest["paths"]["final_report"])
        ledger = Path(manifest["paths"]["execution_ledger"])
        if report.exists():
            raise SystemExit(f"refusing to overwrite append-only report: {report}")
        records = read_jsonl(ledger)
        if any(row.get("record_type") == "audit_result" for row in records):
            raise SystemExit("refusing to append a second audit_result")
        p0 = json.loads(Path(manifest["paths"]["p0_certificate"]).read_text(encoding="utf-8"))
        resolved = validate_resolved_manifest(
            json.loads(Path(manifest["paths"]["resolved_manifest"]).read_text(encoding="utf-8"))
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        append_jsonl(ledger, {
            "record_type": "audit_result",
            "experiment_name": "qwen25_7b_s128_it_audit_seed2026_20260821",
            "git_commit": p0["evidence"]["git_commit"],
            "run_id": p0["evidence"]["run_id"],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "eval_manifest_hash": resolved["eval_manifest_hash"],
            "execution_binding_sha256": canonical_sha256(resolved["execution_binding"]),
            "runtime_binding_sha256": p0["evidence"]["runtime_binding_sha256"],
            "interface_id": None,
            "status": "PASS",
            "decision": result["decision"],
            "artifact": str(report),
            "artifact_sha256": sha256_file(report),
            "row_count": 256,
            "audit_code_commit": audit_code_commit(),
        })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
