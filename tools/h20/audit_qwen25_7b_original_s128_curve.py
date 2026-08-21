#!/usr/bin/env python3
"""Read-only audit/report writer for the fixed-S128 Original learning curve."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
    validate_actor_only_checkpoint_acknowledgements,
    validate_attempt_identity_rows,
    validate_resolved_manifest,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds
from tools.h20.audit_qwen25_7b_s128_it import (
    STRICT_IDENTITY_INTEGER_FIELDS,
    _audit_weight_snapshot,
    _register_unique_turn,
    _strict_integer_failures,
    _strict_turn_type_failures,
    read_jsonl,
    validate_ledger_schema,
)
from tools.h20.preflight_qwen25_7b_original_s128_curve import (
    EXPECTED_BRANCH,
    EXPECTED_EVAL_HASH,
    INTERFACES,
    _artifact_inventory,
    _attempt_id,
    _experiment_name,
    _load_inherited_contract,
    _stable_canary_contract,
    _step,
    build_execution_binding,
    freeze_trainer_configuration,
    git,
    load_manifest,
    sha256_file,
    validate_prior_import,
    validate_training_source,
)
from tools.h20.preflight_qwen25_7b_stable_i4x2 import _load_parquet_rows


def _sha(value: object) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is not None


def ground_truth_by_source_order(
    inherited: Mapping[str, Any], resolved: Mapping[str, Any]
) -> dict[int, object]:
    raw_rows = _load_parquet_rows(Path(inherited["data"]["validation"]))
    result: dict[int, object] = {}
    for row in resolved["identity_payload"]["rows"]:
        raw_position = int(row["raw_row_position"])
        order = int(row["source_order_index"])
        reward_model = raw_rows[raw_position].get("reward_model")
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)
        if not isinstance(reward_model, Mapping) or "ground_truth" not in reward_model:
            raise ValueError(f"S128 row {order} lacks parquet ground truth")
        ground_truth = reward_model["ground_truth"]
        if canonical_sha256(ground_truth) != row["ground_truth_hash"]:
            raise ValueError(f"S128 row {order} ground truth differs from P0")
        result[order] = ground_truth
    if set(result) != set(range(128)):
        raise ValueError("parquet ground truth does not cover exact source order 0..127")
    return result


def _expected_artifact_identity(
    interface_id: str, plan: Mapping[str, Any], resolved: Mapping[str, Any]
) -> tuple[str, str, Mapping[str, Any]]:
    artifact_interface = str(plan["source_interface"])
    artifact_attempt = str(plan["source_attempt"])
    if plan["mode"] == "import":
        source_resolved = resolved["execution_binding"]["prior_s128_it_import"].get(
            "resolved"
        )
        # The bulky source resolved manifest is deliberately not duplicated in
        # the execution binding. Load the exact P0-frozen path instead.
        source_path = resolved["execution_binding"]["prior_s128_it_import"][
            "resolved_manifest"
        ]
        source_resolved = validate_resolved_manifest(
            json.loads(Path(source_path).read_text(encoding="utf-8"))
        )
        binding = source_resolved["execution_binding"]["trainer_configuration"][
            "interfaces"
        ][artifact_interface]
    else:
        binding = resolved["execution_binding"]["trainer_configuration"]["interfaces"][
            interface_id
        ]
    return artifact_interface, artifact_attempt, binding


def _audit_actor_only_load(
    summary: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> list[str]:
    acknowledgements = summary.get("actor_checkpoint_load_acks")
    if not isinstance(acknowledgements, list):
        return ["actor checkpoint summary lacks load acknowledgements"]
    try:
        validate_actor_only_checkpoint_acknowledgements(
            acknowledgements,
            checkpoint["actor_model_shards"],
            global_step_folder=checkpoint["path"],
            world_size=2,
        )
    except Exception as error:
        return [f"actor-only checkpoint acknowledgement failed: {error}"]
    return []


def audit_interface(
    *, interface_id: str, plan: Mapping[str, Any], manifest: Mapping[str, Any],
    resolved: Mapping[str, Any], ground_truth: Mapping[int, object],
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    root = Path(plan["root"])
    step = int(plan["global_step"])
    terminal_path = root / f"terminal/{step}.jsonl"
    turns_path = root / "trajectory_turns.jsonl"
    summary_path = root / "execution_summary.json"
    log_path = root / "run.log"
    for label, path in (
        ("terminal", terminal_path), ("turn ledger", turns_path),
        ("execution summary", summary_path), ("run log", log_path),
    ):
        if not path.is_file():
            failures.append(f"{interface_id} missing {label}: {path}")
    if failures:
        return failures, evidence, []
    frozen_artifacts = _artifact_inventory(root, step=step)
    if plan["mode"] == "import" and frozen_artifacts != plan.get("artifacts"):
        failures.append(f"{interface_id} imported artifacts changed from P0")
    evidence.update(
        mode=plan["mode"], root=str(root), artifacts=frozen_artifacts,
        terminal_sha256=sha256_file(terminal_path),
        turn_ledger_sha256=sha256_file(turns_path),
        execution_summary_sha256=sha256_file(summary_path),
        run_log_sha256=sha256_file(log_path),
    )
    artifact_interface, artifact_attempt, trainer_binding = _expected_artifact_identity(
        interface_id, plan, resolved
    )
    terminal = read_jsonl(terminal_path)
    if len(terminal) != 128:
        failures.append(f"{interface_id} terminal denominator {len(terminal)} != 128")
    if [row.get("source_repeated_row") for row in terminal] != list(range(128)):
        failures.append(f"{interface_id} terminal order is not exact 0..127")
    identities = [
        {field: row[field] for field in OUTPUT_IDENTITY_FIELDS if field in row}
        for row in terminal
    ]
    try:
        validate_attempt_identity_rows(identities, examples=128, replicas=1)
    except Exception as error:
        failures.append(f"{interface_id} stable identity inventory failed: {error}")
    frozen_rows = {
        (int(row["source_order_index"]), str(row["example_id"])): row
        for row in resolved["identity_payload"]["rows"]
    }
    metric_rows: list[dict[str, Any]] = []
    terminal_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, row in enumerate(terminal):
        missing = sorted(set(OUTPUT_IDENTITY_FIELDS) - row.keys())
        if missing:
            failures.append(f"{interface_id} terminal row {index} missing {missing}")
            continue
        type_failures = _strict_integer_failures(
            row, (*STRICT_IDENTITY_INTEGER_FIELDS, "step"),
            label=f"{interface_id} terminal row {index}",
        )
        if type_failures:
            failures.extend(type_failures)
            continue
        if row["interface_id"] != artifact_interface or row["attempt_id"] != artifact_attempt:
            failures.append(f"{interface_id} terminal row {index} interface/attempt differs")
        if row["eval_manifest_hash"] != EXPECTED_EVAL_HASH:
            failures.append(f"{interface_id} terminal row {index} eval hash differs")
        if row["step"] != step:
            failures.append(f"{interface_id} terminal row {index} global step differs")
        order = int(row["source_order_index"])
        frozen_key = (order, str(row["example_id"]))
        if frozen_key not in frozen_rows:
            failures.append(f"{interface_id} terminal row {index} is outside S128")
            continue
        for field in MANIFEST_ROW_FIELDS:
            if row[field] != frozen_rows[frozen_key][field]:
                failures.append(f"{interface_id} terminal row {index} changed {field}")
        seed = evaluation_trajectory_seed(
            base_seed=int(manifest["evaluation"]["base_seed"]),
            eval_manifest_hash=EXPECTED_EVAL_HASH,
            example_id=str(row["example_id"]), source_order_index=order, replica_id=0,
        )
        if row["trajectory_seed"] != seed:
            failures.append(f"{interface_id} terminal row {index} seed differs")
        if row["trajectory_id"] != stable_trajectory_id(
            eval_manifest_hash=EXPECTED_EVAL_HASH,
            example_id=str(row["example_id"]), replica_id=0, trajectory_seed=seed,
        ):
            failures.append(f"{interface_id} terminal row {index} trajectory ID differs")
        if row["replica_id"] != 0 or row["source_repeated_row"] != order:
            failures.append(f"{interface_id} terminal row {index} is not n=1 aligned")
        output = row.get("output")
        if not isinstance(output, str):
            failures.append(f"{interface_id} terminal row {index} output is not text")
            output = ""
        rollout_score = row.get("score")
        if (
            not isinstance(rollout_score, (int, float))
            or isinstance(rollout_score, bool)
            or not math.isfinite(float(rollout_score))
        ):
            failures.append(f"{interface_id} terminal row {index} rollout score is non-finite")
        if not _sha(row.get("terminal_response_token_sha256")):
            failures.append(f"{interface_id} terminal row {index} response digest is invalid")
        scored = score_terminal_output(output, ground_truth[order])
        metric_rows.append({
            "stable_key": json.dumps(stable_key(row), separators=(",", ":")),
            "source_order_index": order,
            "eval_manifest_hash": row["eval_manifest_hash"],
            "example_id": row["example_id"], "replica_id": row["replica_id"],
            "trajectory_seed": row["trajectory_seed"],
            "trajectory_id": row["trajectory_id"], **scored,
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
        duplicates = _register_unique_turn(
            seen_key_turns, key, turn_number,
            label=f"{interface_id} turn row {index}",
        )
        if duplicates:
            failures.extend(duplicates)
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
        active = sorted(int(row["trajectory_turn"]) for row in rows if not row["is_final"])
        final = sorted(int(row["trajectory_turn"]) for row in rows if row["is_final"])
        expected_active = list(range(int(active_by_order[str(order)])))
        if active != expected_active or final != [shared_final]:
            failures.append(
                f"{interface_id} trajectory {key} turn schedule differs: "
                f"active={active}, final={final}"
            )
        if final:
            final_row = next(row for row in rows if row["is_final"])
            if final_row["response_token_sha256"] != terminal_by_key[key][
                "terminal_response_token_sha256"
            ]:
                failures.append(f"{interface_id} final/terminal response digest differs")
    evidence["initial_prompt_sha256_by_stable_key"] = {
        json.dumps(key, separators=(",", ":")): next(
            (row["request_prompt_token_sha256"] for row in rows if row["trajectory_turn"] == 0),
            None,
        )
        for key, rows in turns_by_key.items()
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary = {
        "record_type": "execution_summary",
        "interface_id": artifact_interface,
        "attempt_id": artifact_attempt,
        "eval_manifest_hash": EXPECTED_EVAL_HASH,
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
    checkpoint = resolved["execution_binding"]["model_artifacts"].get(interface_id)
    expected_source = None if interface_id == "I" else str(Path(checkpoint["path"]).resolve())
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
                failures.append(f"{interface_id} validation mutated {field}")
        if after.get("vllm_pre_sync_sampled_tensor_digest") != before.get(
            "vllm_sampled_tensor_digest"
        ):
            failures.append(f"{interface_id} vLLM drifted before final read-only sync")
        evidence["weight_snapshot_before"] = before
    if interface_id == "I":
        if summary.get("actor_checkpoint_load_acks") != []:
            failures.append("I unexpectedly loaded an actor checkpoint")
    else:
        failures.extend(_audit_actor_only_load(summary, checkpoint))
        expected_digest = checkpoint["training_effective_actor_vllm_digest"]
        expected_master = checkpoint[
            "training_actor_master_sampled_tensor_digest"
        ]
        expected_names = checkpoint["training_loaded_parameter_names_sha256"]
        if not isinstance(before, Mapping) or (
            before.get("actor_rollout_sampled_tensor_digest") != expected_digest
            or before.get("vllm_sampled_tensor_digest") != expected_digest
            or before.get("actor_master_sampled_tensor_digest") != expected_master
            or any(
                worker.get("loaded_parameter_names_sha256") != expected_names
                for worker in before.get("worker_evidence", [])
            )
        ):
            failures.append(
                f"{interface_id} master/effective/vLLM/name digests do not equal "
                f"training version {step}"
            )
    return failures, evidence, metric_rows


def build_curve_summary(
    metrics: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    for interface_id in INTERFACES:
        if len(metrics.get(interface_id, ())) != 128:
            raise ValueError(f"curve interface {interface_id} does not have 128 metric rows")
    points = {
        interface_id: summarize_fixed_s128(metrics[interface_id])
        for interface_id in INTERFACES
    }
    def paired(left: str, right: str) -> dict[str, Any]:
        result = paired_descriptive_summary(metrics[left], metrics[right])
        result["estimand"] = (
            f"{right} minus {left} on these same curated fixed 128 examples"
        )
        return result

    versus_i = {interface_id: paired("I", interface_id) for interface_id in INTERFACES[1:]}
    consecutive: dict[str, Any] = {}
    for left, right in zip(INTERFACES, INTERFACES[1:]):
        consecutive[f"{right}_minus_{left}"] = paired(left, right)
    return {
        "order": list(INTERFACES),
        "points": points,
        "paired_descriptive_vs_I": versus_i,
        "paired_descriptive_consecutive": consecutive,
        "causal": False,
        "population_inference": False,
        "monotonicity_required": False,
    }


def _audit_ledger(
    records: list[dict[str, Any]], *, manifest: Mapping[str, Any],
    resolved: Mapping[str, Any], p0: Mapping[str, Any],
) -> list[str]:
    failures = validate_jsonl_chain(records)
    schema = json.loads(
        (Path(manifest["repository"]) / manifest["ledger_schema"]).read_text(
            encoding="utf-8"
        )
    )
    failures.extend(validate_ledger_schema(records, schema))
    plan = resolved["execution_binding"]["interface_plan"]
    expected = [("p0_preflight", None)]
    for interface_id in INTERFACES:
        if plan[interface_id]["mode"] == "import":
            expected.append(("source_import", interface_id))
        else:
            expected.extend(
                (("interface_start", interface_id), ("interface_finish", interface_id))
            )
    if records and records[-1].get("record_type") == "audit_result":
        expected.append(("audit_result", None))
    actual = [(row.get("record_type"), row.get("interface_id")) for row in records]
    if actual != expected:
        failures.append(f"curve ledger sequence differs: {actual} != {expected}")
    execution_sha = canonical_sha256(resolved["execution_binding"])
    runtime_sha = p0["evidence"].get("runtime_binding_sha256")
    for index, row in enumerate(records):
        if row.get("git_commit") != p0["evidence"].get("git_commit"):
            failures.append(f"ledger record {index} Git commit differs")
        if row.get("run_id") != p0["evidence"].get("run_id"):
            failures.append(f"ledger record {index} run ID differs")
        if row.get("eval_manifest_hash") != EXPECTED_EVAL_HASH:
            failures.append(f"ledger record {index} eval hash differs")
        if row.get("execution_binding_sha256") != execution_sha:
            failures.append(f"ledger record {index} execution binding differs")
        if row.get("runtime_binding_sha256") != runtime_sha:
            failures.append(f"ledger record {index} runtime binding differs")
        if row.get("status") != "PASS":
            failures.append(f"ledger record {index} status is not PASS")
    if records:
        p0_path = Path(manifest["paths"]["p0_certificate"])
        first = records[0]
        if (
            first.get("experiment_name") != "qwen25_7b_original_s128_curve_p0_seed2026_20260821"
            or first.get("row_count") != 128
            or Path(str(first.get("artifact", ""))).resolve() != p0_path.resolve()
            or not p0_path.is_file()
            or first.get("artifact_sha256") != sha256_file(p0_path)
        ):
            failures.append("curve P0 ledger record does not authenticate certificate")
    cursor = 1
    prior = resolved["execution_binding"]["prior_s128_it_import"]
    for interface_id in INTERFACES:
        interface_plan = plan[interface_id]
        try:
            current = _artifact_inventory(
                Path(interface_plan["root"]),
                step=int(interface_plan["global_step"]),
            )
        except Exception as error:
            failures.append(f"curve ledger {interface_id} artifact audit failed: {error}")
            current = {}
        if interface_plan["mode"] == "import":
            if len(records) <= cursor:
                cursor += 1
                continue
            imported = records[cursor]
            expected_source = {
                "source_p0_certificate_sha256": prior.get("p0_certificate_sha256"),
                "source_resolved_manifest_sha256": prior.get(
                    "resolved_manifest_sha256"
                ),
                "source_final_report_sha256": prior.get("final_report_sha256"),
                "source_execution_ledger_sha256": prior.get(
                    "execution_ledger_sha256"
                ),
                "source_execution_ledger_tail_sha256": prior.get(
                    "execution_ledger_tail_sha256"
                ),
            }
            if (
                imported.get("experiment_name") != _experiment_name(interface_id)
                or imported.get("mode") != "import"
                or imported.get("source_interface")
                != interface_plan["source_interface"]
                or imported.get("source_attempt") != interface_plan["source_attempt"]
                or imported.get("source_root") != interface_plan["root"]
                or imported.get("artifacts") != current
                or current != interface_plan.get("artifacts")
                or any(imported.get(key) != value for key, value in expected_source.items())
            ):
                failures.append(
                    f"curve ledger {interface_id} source-import identity differs"
                )
            cursor += 1
            continue
        if len(records) <= cursor + 1:
            cursor += 2
            continue
        start, finish = records[cursor], records[cursor + 1]
        if (
            start.get("experiment_name") != _experiment_name(interface_id)
            or finish.get("experiment_name") != _experiment_name(interface_id)
            or start.get("mode") != "run"
            or finish.get("mode") != "run"
            or start.get("artifacts") != {}
            or finish.get("artifacts") != current
        ):
            failures.append(f"curve ledger {interface_id} start/finish identity differs")
        cursor += 2
    if records and records[-1].get("record_type") == "audit_result":
        report = Path(manifest["paths"]["final_report"])
        tail = records[-1]
        if (
            tail.get("experiment_name") != "qwen25_7b_original_s128_curve_audit_seed2026_20260821"
            or tail.get("decision") != "ORIGINAL_S128_CURVE_PASS"
            or tail.get("row_count") != 768
            or Path(str(tail.get("artifact", ""))).resolve() != report.resolve()
            or not report.is_file()
            or tail.get("artifact_sha256") != sha256_file(report)
        ):
            failures.append("curve final audit ledger record changed")
    return failures


def run_audit(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    repo = Path(manifest["repository"])
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    expected_commit = os.environ["MEMAGENT_ORIGINAL_CURVE_EXPECTED_COMMIT"]
    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    status = git(repo, "status", "--porcelain")
    if head != expected_commit or branch != EXPECTED_BRANCH or status:
        failures.append(
            f"Git binding differs: head={head}, expected={expected_commit}, "
            f"branch={branch}, dirty={bool(status)}"
        )
    p0_path = Path(manifest["paths"]["p0_certificate"])
    resolved_path = Path(manifest["paths"]["resolved_manifest"])
    if not p0_path.is_file() or not resolved_path.is_file():
        return {
            "status": "FAIL", "decision": "ORIGINAL_S128_CURVE_NO_GO:AUDIT",
            "failures": [*failures, "P0 certificate/resolved manifest is missing"],
            "evidence": evidence,
        }
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    resolved = validate_resolved_manifest(
        json.loads(resolved_path.read_text(encoding="utf-8"))
    )
    if p0.get("status") != "PASS":
        failures.append("curve P0 is not PASS")
    if p0.get("evidence", {}).get("resolved_manifest_sha256") != sha256_file(resolved_path):
        failures.append("curve resolved manifest changed after P0")
    if resolved.get("eval_manifest_hash") != EXPECTED_EVAL_HASH:
        failures.append("curve eval manifest hash changed")
    if resolved.get("cohort") != {
        "source_order_indices": list(range(128)), "examples": 128,
        "replicas": 1, "interfaces": list(INTERFACES),
    }:
        failures.append("curve cohort is not exact six-interface fixed S128")
    inherited = None
    stable = None
    training = None
    prior = None
    try:
        inherited, _ = _load_inherited_contract(manifest, repo)
        stable = _stable_canary_contract(
            manifest, expected_eval_manifest_hash=EXPECTED_EVAL_HASH
        )
        training = validate_training_source(manifest, stable)
        prior = validate_prior_import(
            manifest, expected_eval_manifest_hash=EXPECTED_EVAL_HASH
        )
        trainer = freeze_trainer_configuration(
            manifest, inherited, repo=repo, eval_manifest_hash=EXPECTED_EVAL_HASH
        )
        expected_execution = build_execution_binding(
            manifest, inherited, repo=repo,
            rows=resolved["identity_payload"]["rows"], stable_canary=stable,
            training=training, prior=prior, trainer_configuration=trainer,
        )
        if expected_execution != resolved["execution_binding"]:
            failures.append("curve executable/provenance binding changed after P0")
    except Exception as error:
        failures.append(f"cannot reconstruct curve prerequisites: {error}")
    try:
        ground_truth = ground_truth_by_source_order(inherited, resolved)
    except Exception as error:
        failures.append(f"cannot bind parquet ground truth: {error}")
        ground_truth = {}

    metrics: dict[str, list[dict[str, Any]]] = {}
    interface_evidence: dict[str, Any] = {}
    plan = resolved["execution_binding"]["interface_plan"]
    if len(ground_truth) == 128:
        for interface_id in INTERFACES:
            interface_failures, details, rows = audit_interface(
                interface_id=interface_id, plan=plan[interface_id],
                manifest=manifest, resolved=resolved, ground_truth=ground_truth,
            )
            failures.extend(interface_failures)
            interface_evidence[interface_id] = details
            metrics[interface_id] = rows
    evidence["interfaces"] = interface_evidence
    if all(len(metrics.get(name, [])) == 128 for name in INTERFACES):
        reference = {row["stable_key"]: row for row in metrics["I"]}
        reference_prompts = interface_evidence["I"].get(
            "initial_prompt_sha256_by_stable_key"
        )
        for interface_id in INTERFACES[1:]:
            current = {row["stable_key"]: row for row in metrics[interface_id]}
            if set(current) != set(reference):
                failures.append(f"{interface_id}/I stable-key coverage differs")
            for key in set(current) & set(reference):
                for field in (
                    "eval_manifest_hash", "example_id", "source_order_index",
                    "replica_id", "trajectory_seed", "trajectory_id",
                ):
                    if current[key][field] != reference[key][field]:
                        failures.append(f"{interface_id}/I stable identity differs: {key}/{field}")
            if interface_evidence[interface_id].get(
                "initial_prompt_sha256_by_stable_key"
            ) != reference_prompts:
                failures.append(f"{interface_id}/I initial recurrent prompt digests differ")
        evidence["curve"] = build_curve_summary(metrics)
        evidence["metric_rows_sha256"] = {
            name: canonical_sha256(metrics[name]) for name in INTERFACES
        }
        if bool(prior and prior.get("available")):
            source_report = json.loads(
                Path(prior["final_report"]).read_text(encoding="utf-8")
            )
            source_interfaces = source_report.get("evidence", {}).get(
                "interfaces", {}
            )
            for target, source_name in (("I", "I"), ("Original25", "T25")):
                if source_report.get("primary_metrics", {}).get(source_name) != (
                    interface_evidence[target].get("metrics")
                ):
                    failures.append(
                        f"imported {target} metrics differ from its authenticated source report"
                    )
                if source_interfaces.get(source_name, {}).get(
                    "independent_metric_rows_sha256"
                ) != evidence["metric_rows_sha256"][target]:
                    failures.append(
                        f"imported {target} independently scored rows differ from source"
                    )
            expected_pair = paired_descriptive_summary(
                metrics["I"], metrics["Original25"]
            )
            if source_report.get("paired_descriptive_t25_minus_i") != expected_pair:
                failures.append(
                    "imported I/Original25 paired summary differs from source report"
                )

    ledger = Path(manifest["paths"]["execution_ledger"])
    if not ledger.is_file():
        failures.append("curve append-only execution ledger is missing")
    else:
        records = read_jsonl(ledger)
        failures.extend(_audit_ledger(records, manifest=manifest, resolved=resolved, p0=p0))
        evidence["execution_ledger_sha256"] = sha256_file(ledger)
        evidence["execution_ledger_records"] = len(records)
    evidence.update(
        eval_manifest_hash=EXPECTED_EVAL_HASH,
        fixed_s128_rows=128,
        metric_code_sha256=sha256_file(
            repo / "recurrent/research/s128_hotpot_metrics.py"
        ),
        dense_training_reward_ignored_for_performance=True,
        imported_interfaces=[
            name for name in INTERFACES if plan[name]["mode"] == "import"
        ],
        executed_interfaces=[
            name for name in INTERFACES if plan[name]["mode"] == "run"
        ],
    )
    curve = evidence.get("curve")
    return {
        "status": "PASS" if not failures else "FAIL",
        "decision": "ORIGINAL_S128_CURVE_PASS" if not failures else "ORIGINAL_S128_CURVE_NO_GO:AUDIT",
        "curve": curve,
        "claim_boundaries": {
            "comparison": "same-protocol recurrent-I and corrected Original-style checkpoints 5/10/15/20/25",
            "causal": False, "population_inference": False,
            "monotonic_improvement_required": False,
            "dataset": "curated fixed S128 from the existing project question-only filter then head(128); not random HotpotQA dev",
            "not_original_paper_7b_reproduction": True,
            "published_R": "historical reference only; not rerun",
            "dense_training_reward": "ignored; EM/F1 independently recomputed from terminal text and parquet ground truth",
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
            "status": "FAIL", "decision": "ORIGINAL_S128_CURVE_NO_GO:AUDIT",
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
            raise SystemExit(f"refusing to overwrite append-only curve report: {report}")
        records = read_jsonl(ledger)
        if any(row.get("record_type") == "audit_result" for row in records):
            raise SystemExit("refusing to append a second curve audit_result")
        p0 = json.loads(Path(manifest["paths"]["p0_certificate"]).read_text(encoding="utf-8"))
        resolved = validate_resolved_manifest(
            json.loads(Path(manifest["paths"]["resolved_manifest"]).read_text(encoding="utf-8"))
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        append_jsonl(ledger, {
            "record_type": "audit_result",
            "experiment_name": "qwen25_7b_original_s128_curve_audit_seed2026_20260821",
            "git_commit": p0["evidence"]["git_commit"],
            "run_id": p0["evidence"]["run_id"],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "eval_manifest_hash": EXPECTED_EVAL_HASH,
            "execution_binding_sha256": canonical_sha256(resolved["execution_binding"]),
            "runtime_binding_sha256": p0["evidence"]["runtime_binding_sha256"],
            "interface_id": None, "mode": None, "status": "PASS",
            "decision": result["decision"], "artifact": str(report),
            "artifact_sha256": sha256_file(report), "row_count": 768,
        })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
