#!/usr/bin/env python3
"""Audit the corrected Original-style 2-GPU step3-to-T25 pilot without rollout."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import (
    append_jsonl,
    checkpoint_inventory,
    sha256_file,
    validate_jsonl_chain,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds
from tools.h20.audit_qwen25_7b_gatea import (
    audit_seeds,
    audit_sync,
    component_inventory,
    read_jsonl,
    validate_ledger_schema,
)
from tools.h20.preflight_qwen25_7b_original_t25 import (
    canonical_sha256,
    load_manifest,
)


EXPECTED_BRANCH = "h20/qwen25-7b-original-t25-s128-frozen-20260821"
REQUIRED_METRIC_FRAGMENTS = ("grad_norm", "pg_loss", "rewards/", "advantages/")
REQUIRED_RNG_STATE_KEYS = {"cpu", "cuda", "numpy", "random"}


def _failures_for_resume_state_acks(load_acks: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    if sorted(int(ack.get("rank", -1)) for ack in load_acks) != [0, 1]:
        failures.append("resume_load does not contain rank0/rank1 actor acknowledgements")
    observed_rng_keys: list[list[str]] = []
    for ack in load_acks:
        if not all(
            ack.get(name)
            for name in ("model_loaded", "optimizer_loaded", "extra_loaded")
        ):
            failures.append(f"resume actor load acknowledgement is incomplete: {ack}")
        rng_keys = sorted(str(key) for key in (ack.get("rng_state_keys") or []))
        observed_rng_keys.append(rng_keys)
        if (
            ack.get("rng_restored") is not True
            or rng_keys != sorted(REQUIRED_RNG_STATE_KEYS)
            or ack.get("lr_scheduler_loaded") is not True
        ):
            failures.append(
                f"resume actor did not restore complete scheduler/RNG state: {ack}"
            )
        if ack.get("optimizer_step_max") != 3 or ack.get("lr_scheduler_last_epoch") != 3:
            failures.append(f"resume optimizer/scheduler did not load at step3: {ack}")
    if len(observed_rng_keys) == 2 and observed_rng_keys[0] != observed_rng_keys[1]:
        failures.append(
            f"resume ranks disagree on RNG state components: {observed_rng_keys}"
        )
    return failures


def _failures_for_sync_optimizer(
    records: list[dict[str, Any]], *, versions: list[int], ranks: list[int]
) -> list[str]:
    failures: list[str] = []
    state_counts_by_rank: dict[int, set[int]] = defaultdict(set)
    for version in versions:
        kind = "resume_loaded" if version == 3 else "post_actor_update"
        rows = [
            row
            for row in records
            if row.get("record_type") == "weight_sync_ack"
            and int(row.get("actor_version", -1)) == version
            and row.get("sync_kind") == kind
        ]
        by_rank = {int(row.get("vllm_worker_rank", -1)): row for row in rows}
        for rank in ranks:
            row = by_rank.get(rank)
            if row is None:
                continue
            step_max = row.get("optimizer_step_max")
            scheduler_epoch = row.get("lr_scheduler_last_epoch")
            if step_max != version:
                failures.append(
                    f"rank {rank} optimizer max at actor version {version} is {step_max}, expected {version}"
                )
            if scheduler_epoch != version:
                failures.append(
                    f"rank {rank} scheduler epoch at actor version {version} is "
                    f"{scheduler_epoch}, expected {version}"
                )
            state_count = int(row.get("optimizer_state_entry_count", 0))
            step_count = int(row.get("optimizer_step_entry_count", 0))
            histogram = row.get("optimizer_step_histogram")
            if (
                state_count < 1
                or step_count != state_count
                or not isinstance(histogram, dict)
                or sum(int(value) for value in histogram.values()) != step_count
            ):
                failures.append(
                    f"rank {rank} actor version {version} has incomplete optimizer evidence"
                )
            state_counts_by_rank[rank].add(state_count)
    for rank in ranks:
        if len(state_counts_by_rank[rank]) != 1:
            failures.append(
                f"rank {rank} optimizer state-entry count changed across step3..25: "
                f"{sorted(state_counts_by_rank[rank])}"
            )
    return failures


def _audit_exact_turn_schedule(
    seed_records: list[dict[str, Any]], *, active_turn_counts: list[int], rollout_n: int
) -> list[str]:
    """Bind every recurrent turn to the P0-frozen context-token schedule."""
    failures: list[str] = []
    base_records = [
        row
        for row in seed_records
        if row.get("record_type", "trajectory_seed") == "trajectory_seed"
    ]
    turn_records = [
        row for row in seed_records if row.get("record_type") == "trajectory_turn_seed"
    ]
    base_by_key = {
        (int(row.get("global_step", -1)), int(row.get("row", -1))): row
        for row in base_records
    }
    turns_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in turn_records:
        turns_by_key[
            (int(row.get("global_step", -1)), int(row.get("sample_index", -1)))
        ].append(row)
    for step in range(4, 26):
        group_active_counts = [
            int(active_turn_counts[(step - 1) * 4 + group]) for group in range(4)
        ]
        expected_final_turn = max(group_active_counts)
        for sample_index in range(8):
            group, replica = divmod(sample_index, rollout_n)
            base = base_by_key.get((step, sample_index))
            if base is None:
                failures.append(f"step {step} sample {sample_index} has no base trajectory")
                continue
            rows = turns_by_key.get((step, sample_index), [])
            active = sorted(
                int(row.get("turn", -1)) for row in rows if row.get("is_final") is False
            )
            final = sorted(
                int(row.get("turn", -1)) for row in rows if row.get("is_final") is True
            )
            expected_active = list(range(group_active_counts[group]))
            if active != expected_active:
                failures.append(
                    f"step {step} group {group} replica {replica} active turns "
                    f"{active} != P0 context schedule {expected_active}"
                )
            if final != [expected_final_turn]:
                failures.append(
                    f"step {step} group {group} replica {replica} final turn "
                    f"{final} != shared final {[expected_final_turn]}"
                )
            for row in rows:
                turn = int(row.get("turn", -1))
                expected_seed = derive_turn_request_seeds(
                    [int(base["trajectory_seed"])], [0], turn
                )[0]
                if int(row.get("request_seed", -1)) != expected_seed:
                    failures.append(
                        f"step {step} sample {sample_index} turn {turn} request seed drifted"
                    )
                if (
                    row.get("uid") != base.get("uid")
                    or int(row.get("dataset_index", -1))
                    != int(base.get("dataset_index", -2))
                    or int(row.get("group", -1)) != group
                    or int(row.get("replica", -1)) != replica
                ):
                    failures.append(
                        f"step {step} sample {sample_index} turn identity is misaligned"
                    )
    return failures


def _complete_checkpoint_steps(output: Path, world_size: int) -> tuple[list[int], dict[int, list[dict]]]:
    complete: list[int] = []
    inventories: dict[int, list[dict]] = {}
    for path in sorted(output.glob("global_step_*")):
        if not path.is_dir() or re.fullmatch(r"global_step_(\d+)", path.name) is None:
            continue
        step = int(path.name.rsplit("_", 1)[-1])
        inventory, missing = component_inventory(path, world_size)
        if not missing:
            complete.append(step)
            inventories[step] = inventory
    return sorted(complete), inventories


def _checkpoint_anchor_evidence(
    output: Path, inventories: Mapping[int, list[dict]], steps: list[int]
) -> list[dict[str, Any]]:
    return [
        {
            "path": str(output / f"global_step_{step}"),
            "global_step": step,
            "inventory": inventories.get(step, []),
            "inventory_sha256": canonical_sha256(inventories.get(step, [])),
        }
        for step in steps
    ]


def _failures_for_persisted_anchor_record(
    record: Mapping[str, Any], expected_anchors: list[dict[str, Any]]
) -> list[str]:
    failures: list[str] = []
    if record.get("checkpoint_anchors") != expected_anchors:
        failures.append("persisted checkpoint-anchor inventory map changed")
    if record.get("checkpoint_anchors_sha256") != canonical_sha256(
        expected_anchors
    ):
        failures.append("persisted checkpoint-anchor map SHA-256 changed")
    return failures


def run_audit(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(manifest_path.resolve())
    failures: list[str] = []
    audit_code_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    audit_code_status = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if audit_code_status:
        failures.append(
            f"audit code worktree is dirty: {audit_code_status.splitlines()}"
        )
    repo = Path(manifest["repository"])
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(repo), "branch", "--show-current"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()
    expected_commit = os.environ.get("MEMAGENT_T25_EXPECTED_COMMIT", "")
    if head != expected_commit:
        failures.append(f"Git commit differs from explicit binding: {head} != {expected_commit}")
    if branch != EXPECTED_BRANCH:
        failures.append(f"branch mismatch: {branch} != {EXPECTED_BRANCH}")
    if status:
        failures.append(f"worktree is dirty: {status.splitlines()}")

    paths = manifest["paths"]
    p0_path = Path(paths["p0_certificate"])
    if not p0_path.is_file():
        p0: dict[str, Any] = {}
        failures.append("T25 P0 certificate is missing")
    else:
        p0 = json.loads(p0_path.read_text(encoding="utf-8"))
        if p0.get("status") != "PASS" or p0.get("decision") != "T25_P0_PASS":
            failures.append("T25 P0 certificate is not PASS")
    p0_evidence = p0.get("evidence", {})
    if p0_evidence.get("git_commit") != head or p0_evidence.get("expected_git_commit") != head:
        failures.append("T25 P0 is not bound to the current exact commit")
    run_id = p0_evidence.get("run_id")
    if re.fullmatch(r"[0-9a-f]{32}", str(run_id or "")) is None:
        failures.append("T25 P0 run ID is invalid")

    source_step = Path(manifest["source_gate_a"]["checkpoint"])
    source_inventory, source_missing = component_inventory(
        source_step, int(manifest["gpu"]["world_size"])
    )
    if source_missing:
        failures.append(f"source Gate A step3 is incomplete after training: {source_missing}")
    if source_inventory != p0_evidence.get("source_step3_inventory"):
        failures.append("source Gate A step3 inventory changed between P0 and final audit")
    if canonical_sha256(source_inventory) != p0_evidence.get("source_step3_inventory_sha256"):
        failures.append("source Gate A step3 canonical inventory hash changed")
    source_data = next((item for item in source_inventory if item["path"] == "data.pt"), None)

    stable_spec = manifest["stable_identity_prerequisite"]
    for label, path_key, evidence_key in (
        ("P0 certificate", "p0_certificate", "stable_i_p0_certificate_sha256"),
        ("final report", "final_report", "stable_i_final_report_sha256"),
        ("resolved manifest", "resolved_manifest", "stable_i_resolved_manifest_sha256"),
        ("execution ledger", "execution_ledger", "stable_i_execution_ledger_sha256"),
    ):
        stable_path = Path(stable_spec[path_key])
        if not stable_path.is_file() or sha256_file(stable_path) != p0_evidence.get(
            evidence_key
        ):
            failures.append(f"stable-I prerequisite {label} changed after T25 P0")

    ledger_path = Path(paths["execution_ledger"])
    records = read_jsonl(ledger_path)
    failures.extend(validate_jsonl_chain(records))
    schema = json.loads((repo / manifest["ledger_schema"]).read_text(encoding="utf-8"))
    failures.extend(validate_ledger_schema(records, schema))
    suffix_present = len(records) >= 2 and [
        records[-2].get("record_type"), records[-1].get("record_type")
    ] == ["checkpoint_inventory", "audit_result"]
    training_records = records[:-2] if suffix_present else records
    for index, record in enumerate(records):
        if record.get("experiment_name") != manifest["experiment_name"]:
            failures.append(f"ledger record {index} experiment identity changed")
        if record.get("git_commit") != head:
            failures.append(f"ledger record {index} Git commit changed")
        if record.get("run_id") != run_id:
            failures.append(f"ledger record {index} run ID changed")

    expected_record_count = 1 + 1 + 1 + 3 + 22 * 5
    if len(training_records) != expected_record_count:
        failures.append(
            f"training ledger record count {len(training_records)} != {expected_record_count}"
        )
    if not training_records or training_records[0].get("record_type") != "p0_preflight":
        failures.append("ledger does not start with the unique P0 record")
    elif (
        Path(str(training_records[0].get("artifact", ""))).resolve() != p0_path.resolve()
        or training_records[0].get("artifact_sha256") != sha256_file(p0_path)
    ):
        failures.append("P0 certificate changed after its hash-chained ledger record")
    runtime_rows = [row for row in training_records if row.get("record_type") == "runtime_config"]
    if len(runtime_rows) != 1:
        failures.append(f"expected one runtime_config record, found {len(runtime_rows)}")
    else:
        runtime = runtime_rows[0]
        if (
            runtime.get("resolved_config_sha256")
            != p0_evidence.get("resolved_trainer_config_sha256")
            or runtime.get("override_argv_sha256")
            != p0_evidence.get("trainer_override_argv_sha256")
        ):
            failures.append("runtime Hydra config/argv hashes differ from P0")

    resume_rows = [row for row in training_records if row.get("record_type") == "resume_load"]
    if len(resume_rows) != 1:
        failures.append(f"expected one resume_load record, found {len(resume_rows)}")
    else:
        resume = resume_rows[0]
        if (
            int(resume.get("global_step", -1)) != 3
            or Path(str(resume.get("resume_source", ""))).resolve() != source_step.resolve()
            or not source_data
            or resume.get("data_sha256") != source_data["sha256"]
            or resume.get("data_loaded") is not True
        ):
            failures.append("resume_load is not bound to the exact Gate A step3 data cursor")
        load_acks = resume.get("actor_load_worker_acks") or []
        failures.extend(_failures_for_resume_state_acks(load_acks))

    versions = list(range(3, 26))
    required_syncs = [
        (manifest["experiment_name"], version, "resume_loaded" if version == 3 else "post_actor_update")
        for version in versions
    ]
    sync_ok, sync_failures, version_digests = audit_sync(
        training_records,
        versions,
        manifest["weight_sync"]["required_worker_ranks"],
        required_syncs,
        manifest["weight_sync"]["parameter_names"],
        manifest["weight_sync"]["transfer_format"],
        manifest["weight_sync"]["expected_loaded_parameter_count"],
    )
    if not sync_ok:
        failures.extend(sync_failures)
    failures.extend(
        _failures_for_sync_optimizer(
            training_records,
            versions=versions,
            ranks=manifest["weight_sync"]["required_worker_ranks"],
        )
    )
    if version_digests.get(3) != manifest["source_gate_a"]["required_version_3_digest"]:
        failures.append("resume-loaded actor/vLLM digest is not Gate A r5 version3")
    if version_digests.get(25) == version_digests.get(3):
        failures.append("effective sampled actor digest did not change from version3 to version25")

    rollout_rows = [row for row in training_records if row.get("record_type") == "rollout_start"]
    if len(rollout_rows) != 22:
        failures.append(f"rollout_start count {len(rollout_rows)} != 22")
    for step in range(4, 26):
        rows = [row for row in rollout_rows if int(row.get("global_step", -1)) == step]
        if len(rows) != 1:
            failures.append(f"step {step} does not have exactly one rollout_start")
        elif (
            int(rows[0].get("actor_version", -1)) != step - 1
            or rows[0].get("sampled_tensor_digest") != version_digests.get(step - 1)
        ):
            failures.append(f"step {step} rollout did not use synchronized actor version {step - 1}")

    signal_rows = [row for row in training_records if row.get("record_type") == "execution_signal"]
    if sorted(int(row.get("global_step", -1)) for row in signal_rows) != list(range(4, 26)):
        failures.append("execution signals do not cover exactly step4..25")
    signal_summary: dict[str, dict[str, float | None]] = {}
    for step in range(4, 26):
        rows = [row for row in signal_rows if int(row.get("global_step", -1)) == step]
        if len(rows) != 1:
            continue
        row = rows[0]
        metrics = row.get("metrics") or {}
        if int(row.get("actor_version", -1)) != step:
            failures.append(f"step {step} execution signal actor version drifted")
        if row.get("nonfinite_metric_names") != []:
            failures.append(f"step {step} contains non-finite metric names")
        if any(not math.isfinite(float(value)) for value in metrics.values()):
            failures.append(f"step {step} contains a non-finite execution metric")
        missing_families = [
            fragment
            for fragment in REQUIRED_METRIC_FRAGMENTS
            if not any(fragment in key for key in metrics)
        ]
        if missing_families:
            failures.append(f"step {step} misses metric families {missing_families}")
        signal_summary[str(step)] = {
            "grad_norm": metrics.get("actor/grad_norm"),
            "pg_loss": metrics.get("actor/pg_loss"),
            "reward_mean": metrics.get("critic/rewards/mean"),
            "advantage_min": metrics.get("critic/advantages/min"),
            "advantage_max": metrics.get("critic/advantages/max"),
        }
    if not any(abs(float(item.get("grad_norm") or 0.0)) > 0 for item in signal_summary.values()):
        failures.append("all 22 updates have zero/missing grad norm")

    # Enforce the semantic record order while allowing rank-ack order to vary.
    if training_records:
        positions = {id(row): index for index, row in enumerate(training_records)}
        last_position = -1
        ordered_groups: list[list[dict[str, Any]]] = [
            [training_records[0]], runtime_rows, resume_rows
        ]
        for version in versions:
            kind = "resume_loaded" if version == 3 else "post_actor_update"
            if version > 3:
                ordered_groups.append([
                    row for row in rollout_rows if int(row.get("global_step", -1)) == version
                ])
            ordered_groups.append([
                row for row in training_records
                if row.get("record_type") == "weight_sync_ack"
                and int(row.get("actor_version", -1)) == version
                and row.get("sync_kind") == kind
            ])
            ordered_groups.append([
                row for row in training_records
                if row.get("record_type") == "weight_sync_summary"
                and int(row.get("actor_version", -1)) == version
                and row.get("sync_kind") == kind
            ])
            if version > 3:
                ordered_groups.append([
                    row for row in signal_rows if int(row.get("global_step", -1)) == version
                ])
        for group in ordered_groups:
            if not group:
                continue
            group_positions = sorted(positions[id(row)] for row in group)
            if group_positions[0] <= last_position:
                failures.append("ledger resume/rollout/update/sync/signal semantic order is invalid")
                break
            last_position = group_positions[-1]

    seed_path = Path(paths["output"]) / "rollout_seed_audit.jsonl"
    seed_records = read_jsonl(seed_path)
    expected_cursor = p0_evidence.get("train_cursor_semantic_indices_0_to_99") or []
    seed_ok, seed_failures = audit_seeds(
        seed_records,
        int(manifest["training"]["seed"]),
        int(manifest["training"]["rollout_n"]),
        expected_steps=list(range(4, 26)),
        expected_batch_size=8,
        expected_dataset_cursor=expected_cursor,
    )
    if not seed_ok:
        failures.extend(seed_failures)
    active_turn_counts = p0_evidence.get("train_cursor_active_turn_counts_0_to_99") or []
    if len(active_turn_counts) != 100:
        failures.append("P0 does not freeze 100 context-derived active-turn counts")
    else:
        failures.extend(
            _audit_exact_turn_schedule(
                seed_records,
                active_turn_counts=[int(value) for value in active_turn_counts],
                rollout_n=int(manifest["training"]["rollout_n"]),
            )
        )
    base_trajectory_count = sum(
        row.get("record_type", "trajectory_seed") == "trajectory_seed"
        for row in seed_records
    )
    if base_trajectory_count != 176:
        failures.append(f"base trajectory count {base_trajectory_count} != 176")

    output = Path(paths["output"])
    complete_steps, complete_inventories = _complete_checkpoint_steps(
        output, int(manifest["gpu"]["world_size"])
    )
    expected_complete = manifest["training"]["expected_retained_complete_actor_checkpoints"]
    if complete_steps != expected_complete:
        failures.append(
            f"retained complete actor checkpoints {complete_steps} != {expected_complete}"
        )
    data_only_orphan_steps = sorted(
        int(path.parent.name.rsplit("_", 1)[-1])
        for path in output.glob("global_step_*/data.pt")
        if int(path.parent.name.rsplit("_", 1)[-1]) not in complete_steps
    )
    if data_only_orphan_steps:
        failures.append(
            f"all five preregistered anchors must retain actor state; data-only remnants: "
            f"{data_only_orphan_steps}"
        )
    checkpoint_anchors = _checkpoint_anchor_evidence(
        output, complete_inventories, expected_complete
    )
    checkpoint_anchors_sha = canonical_sha256(checkpoint_anchors)
    latest_path = output / "latest_checkpointed_iteration.txt"
    if not latest_path.is_file() or latest_path.read_text(encoding="utf-8").strip() != "25":
        failures.append("latest checkpoint marker is not exactly 25")
    step25_path = Path(paths["step25"])
    step25_inventory = complete_inventories.get(25, [])
    step25_inventory_sha = canonical_sha256(step25_inventory)
    step25_data = next((item for item in step25_inventory if item["path"] == "data.pt"), None)
    if not step25_data or not source_data or step25_data["sha256"] == source_data["sha256"]:
        failures.append("step25 dataloader cursor did not change from source step3")

    p0_source_bytes = int(p0_evidence.get("source_step3_inventory_bytes", -1))
    if p0_source_bytes != sum(int(item["size"]) for item in source_inventory):
        failures.append("source step3 byte inventory changed after P0")

    status_value = "PASS" if not failures else "FAIL"
    report = {
        "schema_version": 1,
        "phase": "final",
        "status": status_value,
        "decision": "ORIGINAL_T25_PASS" if not failures else "ORIGINAL_T25_NO_GO:AUDIT",
        "failures": failures,
        "study_label": "corrected Original-style 2-GPU pilot",
        "not_original_paper_7b_reproduction": True,
        "git_commit": head,
        "audit_code_commit": audit_code_commit,
        "audit_code_worktree_clean": not bool(audit_code_status),
        "branch": branch,
        "experiment_name": manifest["experiment_name"],
        "source_gate_a_commit": manifest["source_gate_a"]["commit"],
        "source_gate_a_report_sha": p0_evidence.get("source_gate_a_report_sha256"),
        "source_gate_a": {
            "commit": manifest["source_gate_a"]["commit"],
            "report_path": manifest["source_gate_a"]["final_report"],
            "report_sha256": p0_evidence.get("source_gate_a_report_sha256"),
            "checkpoint_path": str(source_step),
            "global_step": 3,
            "inventory": source_inventory,
            "inventory_sha256": canonical_sha256(source_inventory),
            "effective_actor_vllm_digest": version_digests.get(3),
        },
        "stable_identity_prerequisite": {
            "status": "PASS",
            "decision": "I_RECURRENT_IDENTITY_CANARY_PASS",
            "commit": manifest["stable_identity_prerequisite"]["commit"],
            "report_sha256": p0_evidence.get("stable_i_final_report_sha256"),
            "eval_manifest_hash": p0_evidence.get("stable_i_eval_manifest_hash"),
            "execution_ledger_sha256": p0_evidence.get(
                "stable_i_execution_ledger_sha256"
            ),
        },
        "resume_state_continuity": {
            "source_extra_state": p0_evidence.get("source_step3_extra_state"),
            "actor_load_worker_acks": (
                resume_rows[0].get("actor_load_worker_acks")
                if len(resume_rows) == 1
                else []
            ),
            "dataloader_data_sha256": source_data.get("sha256") if source_data else None,
        },
        "training": {
            "source_step": 3,
            "first_update_step": 4,
            "target_step": 25,
            "update_steps": list(range(4, 26)),
            "updates_in_this_resume": 22,
            "t25_total_corrected_original_updates_from_base": 25,
            "rollout_trajectories_in_this_resume": 176,
            "production_cursor_positions": list(range(12, 100)),
            "production_prompt_count": 88,
            "primary_scientific_endpoint": 25,
            "secondary_learning_curve_anchor_steps": [5, 10, 15, 20],
            "technical_save_steps": [5, 10, 15, 20, 25],
            "retained_complete_checkpoints": complete_steps,
            "data_only_remnant_steps": data_only_orphan_steps,
        },
        "comparison_semantics": {
            "T25_minus_I_observed_change": "descriptive change on the same frozen S128 after 25 corrected Original-style updates; not a population or causal effect",
            "this_execution": "explicit resume from passed step3 and exactly 22 new updates, step4 through step25",
            "future_method_same_budget_paired_comparison": "method25 minus Original25 on the fixed evaluation set, with both starting from the same Gate A step3 warm-start and receiving the same 22-update budget; descriptive, not causal",
        },
        "runtime_configuration": {
            "override_argv_sha256": p0_evidence.get("trainer_override_argv_sha256"),
            "resolved_config_sha256": p0_evidence.get("resolved_trainer_config_sha256"),
            "gate_a_reference_resolved_config_sha256": p0_evidence.get(
                "gate_a_reference_resolved_config_sha256"
            ),
            "allowed_differences": p0_evidence.get(
                "allowed_gate_a_to_t25_config_differences"
            ),
        },
        "data_cursor": {
            "source_consumed_positions": list(range(12)),
            "continuation_positions": list(range(12, 100)),
            "continuation_semantic_indices": expected_cursor[12:100],
            "context_token_counts": (p0_evidence.get("train_cursor_context_token_counts_0_to_99") or [])[12:100],
            "active_turn_counts": active_turn_counts[12:100],
        },
        "weight_sync": {
            "required_worker_ranks": [0, 1],
            "versions": versions,
            "version_digests": {str(key): value for key, value in version_digests.items()},
            "version3_to_version25_changed": version_digests.get(3) != version_digests.get(25),
        },
        "execution_signals": signal_summary,
        "step25_checkpoint": {
            "path": str(step25_path),
            "global_step": 25,
            "inventory": step25_inventory,
            "inventory_sha256": step25_inventory_sha,
        },
        "checkpoint_anchors": checkpoint_anchors,
        "checkpoint_anchors_sha256": checkpoint_anchors_sha,
        "source_step3_inventory_unchanged": source_inventory == p0_evidence.get(
            "source_step3_inventory"
        ),
        "p0_certificate_sha256": sha256_file(p0_path) if p0_path.is_file() else None,
        "execution_ledger": {
            "path": str(ledger_path),
            "training_prefix_record_count": len(training_records),
            "training_prefix_sha256": canonical_sha256(training_records),
            "training_prefix_tail_sha256": (
                training_records[-1].get("record_sha256") if training_records else None
            ),
        },
    }
    if suffix_present:
        checkpoint_record, audit_record = records[-2:]
        final_path = Path(paths["final_report"])
        if (
            checkpoint_record.get("global_step") != 25
            or checkpoint_record.get("inventory") != step25_inventory
            or checkpoint_record.get("inventory_sha256") != step25_inventory_sha
            or checkpoint_record.get("audit_code_commit") != audit_code_commit
        ):
            failures.append("persisted step25 inventory ledger suffix changed")
        failures.extend(
            _failures_for_persisted_anchor_record(
                checkpoint_record, checkpoint_anchors
            )
        )
        if (
            audit_record.get("status") != "PASS"
            or audit_record.get("decision") != "ORIGINAL_T25_PASS"
            or Path(str(audit_record.get("report", ""))).resolve() != final_path.resolve()
            or not final_path.is_file()
            or audit_record.get("report_sha256") != sha256_file(final_path)
            or audit_record.get("audit_code_commit") != audit_code_commit
        ):
            failures.append("persisted T25 final audit ledger suffix changed")
        if failures:
            report["status"] = "FAIL"
            report["decision"] = "ORIGINAL_T25_NO_GO:AUDIT"
            report["failures"] = failures
    return report, manifest


def persist_report(report: dict[str, Any], manifest: Mapping[str, Any]) -> None:
    if report["status"] != "PASS":
        raise ValueError("refusing to persist a failed T25 final report")
    report_path = Path(manifest["paths"]["final_report"])
    ledger_path = Path(manifest["paths"]["execution_ledger"])
    if report_path.exists():
        raise ValueError(f"refusing to overwrite final report: {report_path}")
    records = read_jsonl(ledger_path)
    if any(row.get("record_type") in {"checkpoint_inventory", "audit_result"} for row in records):
        raise ValueError("refusing to append a second final-audit ledger suffix")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    identity = {
        "experiment_name": manifest["experiment_name"],
        "git_commit": report["git_commit"],
        "audit_code_commit": report["audit_code_commit"],
        "run_id": json.loads(
            Path(manifest["paths"]["p0_certificate"]).read_text(encoding="utf-8")
        )["evidence"]["run_id"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint = report["step25_checkpoint"]
    checkpoint_anchors = report["checkpoint_anchors"]
    append_jsonl(
        ledger_path,
        {
            **identity,
            "record_type": "checkpoint_inventory",
            "global_step": 25,
            "inventory": checkpoint["inventory"],
            "inventory_sha256": checkpoint["inventory_sha256"],
            "checkpoint_anchors": checkpoint_anchors,
            "checkpoint_anchors_sha256": canonical_sha256(checkpoint_anchors),
        },
    )
    append_jsonl(
        ledger_path,
        {
            **identity,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "record_type": "audit_result",
            "status": "PASS",
            "decision": "ORIGINAL_T25_PASS",
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    try:
        report, manifest = run_audit(args.manifest)
        if args.write_report and report.get("status") == "PASS":
            persist_report(report, manifest)
    except Exception as error:
        report = {
            "status": "FAIL",
            "decision": "ORIGINAL_T25_NO_GO:AUDIT",
            "failures": [str(error)],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
