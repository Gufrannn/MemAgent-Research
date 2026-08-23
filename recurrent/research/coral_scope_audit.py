"""Pure helpers for CORAL budget and data-scope audits.

The overlap identity deliberately excludes split-local row numbers.  A row is
identified by the SHA-256 pair of the exact question text and context text that
the recurrent actor consumes.  This catches direct example reuse even when a
dataset has been reordered or assigned new integer indices.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from collections.abc import Sequence

from recurrent.research.cosi import (
    canonical_sha256, checkpoint_sha256, sha256_file, validate_ledger,
)
from recurrent.research.gate_a_execution import validate_jsonl_chain
from recurrent.research.stable_eval_identity import validate_resolved_manifest


def sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("CORAL_SCOPE_NO_GO: identity text must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def question_text(prompt: Any) -> str:
    if not isinstance(prompt, list) or not prompt or not isinstance(prompt[0], Mapping):
        raise ValueError("CORAL_SCOPE_NO_GO: prompt must start with a chat message")
    first = prompt[0]
    if first.get("role") != "user" or not isinstance(first.get("content"), str):
        raise ValueError("CORAL_SCOPE_NO_GO: prompt[0] must be user text")
    return str(first["content"])


def content_identity_from_hashes(question_sha256: str, context_sha256: str) -> str:
    for name, value in (
        ("question_sha256", question_sha256), ("context_sha256", context_sha256)
    ):
        if not isinstance(value, str) or len(value) != 64 \
                or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"CORAL_SCOPE_NO_GO: invalid {name}")
    return canonical_sha256({
        "source_question_sha256": question_sha256,
        "source_context_sha256": context_sha256,
    })


def parquet_row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError("CORAL_SCOPE_NO_GO: parquet row must be a mapping")
    question = question_text(row.get("prompt"))
    context = row.get("context")
    if not isinstance(context, str):
        raise TypeError("CORAL_SCOPE_NO_GO: context must be text")
    q_hash = sha256_text(question)
    c_hash = sha256_text(context)
    extra = row.get("extra_info")
    semantic_index = None
    if isinstance(extra, Mapping) and "index" in extra:
        semantic_index = str(int(extra["index"]))
    return {
        "split_local_semantic_index": semantic_index,
        "source_question_sha256": q_hash,
        "source_context_sha256": c_hash,
        "content_identity_sha256": content_identity_from_hashes(q_hash, c_hash),
    }


def stable_row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    q_hash = row.get("source_question_hash")
    c_hash = row.get("source_context_hash")
    return {
        "split_local_semantic_index": str(row.get("example_id")),
        "source_question_sha256": q_hash,
        "source_context_sha256": c_hash,
        "content_identity_sha256": content_identity_from_hashes(q_hash, c_hash),
    }


def identity_inventory(
    identities: Iterable[Mapping[str, Any]], *, require_unique_content: bool = False
) -> dict[str, Any]:
    rows = [dict(row) for row in identities]
    content = [str(row["content_identity_sha256"]) for row in rows]
    questions = [str(row["source_question_sha256"]) for row in rows]
    contexts = [str(row["source_context_sha256"]) for row in rows]
    if require_unique_content and len(content) != len(set(content)):
        raise ValueError("CORAL_SCOPE_NO_GO: duplicate canonical content identity")
    return {
        "row_count": len(rows),
        "unique_content_identity_count": len(set(content)),
        "duplicate_content_row_count": len(content) - len(set(content)),
        "content_identity_inventory_sha256": canonical_sha256(sorted(content)),
        "question_inventory_sha256": canonical_sha256(sorted(questions)),
        "context_inventory_sha256": canonical_sha256(sorted(contexts)),
        "content_identities": set(content),
        "question_identities": set(questions),
        "context_identities": set(contexts),
    }


def overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    content = sorted(left["content_identities"] & right["content_identities"])
    questions = sorted(left["question_identities"] & right["question_identities"])
    contexts = sorted(left["context_identities"] & right["context_identities"])
    return {
        "canonical_content_pair_count": len(content),
        "question_hash_count": len(questions),
        "context_hash_count": len(contexts),
        "canonical_content_pair_inventory_sha256": canonical_sha256(content),
        "matching_content_pair_sha256": content,
    }


def recompute_scope_evidence(
    train_path: str | Path, s128_path: str | Path, stable_resolved_path: str | Path
) -> dict[str, Any]:
    """Independently recompute outcome-free inventories used by certificate validators."""
    import pyarrow.parquet as parquet

    columns = ["prompt", "context", "extra_info"]
    train_rows = parquet.read_table(Path(train_path), columns=columns).to_pylist()
    s128_rows = parquet.read_table(Path(s128_path), columns=columns).to_pylist()
    stable_raw = json.loads(Path(stable_resolved_path).read_text(encoding="utf-8"))
    stable = validate_resolved_manifest(stable_raw)
    train_inventory = identity_inventory(parquet_row_identity(row) for row in train_rows)
    s128_inventory = identity_inventory(
        (parquet_row_identity(row) for row in s128_rows), require_unique_content=True
    )
    stable_inventory = identity_inventory(
        (stable_row_identity(row) for row in stable["identity_payload"]["rows"]),
        require_unique_content=True,
    )
    for key in ("content_identities", "question_identities", "context_identities"):
        if s128_inventory[key] != stable_inventory[key]:
            raise ValueError("CORAL_SCOPE_NO_GO: stable identities/S128 parquet drift")
    empty = identity_inventory([])
    selection = dict(train_inventory)
    for key in ("content_identities", "question_identities", "context_identities"):
        selection[key] = train_inventory[key] | stable_inventory[key]
    public = lambda value: {
        key: child for key, child in value.items() if not isinstance(child, set)
    }
    return {
        "actor_inventory": public(train_inventory),
        "s128_inventory": public(stable_inventory),
        "actor_overlap": overlap(train_inventory, stable_inventory),
        "critic_overlap": overlap(empty, stable_inventory),
        "selection_overlap": overlap(selection, stable_inventory),
        "eval_manifest_hash": stable["eval_manifest_hash"],
    }


def static_budget(manifest: Mapping[str, Any]) -> dict[str, Any]:
    training = manifest["training"]
    exposure = manifest["role_exposure"]
    updates = int(exposure["primary_updates"])
    groups = int(training["train_batch_size"])
    rollout_n = int(training["rollout_n"])
    mini = int(training["ppo_mini_batch_size"])
    epochs = int(training["ppo_epochs"])
    if updates != 25 or groups != 4 or rollout_n != 2 or mini != 4 or epochs != 1:
        raise ValueError("CORAL_SCOPE_NO_GO: frozen pilot budget drifted")
    if groups % mini:
        raise ValueError("CORAL_SCOPE_NO_GO: prompt minibatch does not divide batch")
    if manifest["protocol"]["advantage_estimator"] != "grpo":
        raise ValueError("CORAL_SCOPE_NO_GO: expected critic-free GRPO")
    return {
        "classification": "single_seed_early_budget_pilot_not_convergence",
        "actor_optimizer_updates_planned": updates,
        "prompt_groups_per_update": groups,
        "prompt_groups_planned": updates * groups,
        "trajectories_per_group": rollout_n,
        "sampled_training_trajectories_planned": updates * groups * rollout_n,
        "ppo_epochs_per_update": epochs,
        "prompt_minibatches_per_epoch_per_update": groups // mini,
        "memory_writer_active_updates": int(exposure["memory_writer_updates"]),
        "terminal_answer_active_updates": int(exposure["terminal_answer_updates"]),
        "critic_fit_optimizer_updates": 0,
        "prior_or_reference_fit_optimizer_updates": 0,
        "auxiliary_fit_optimizer_updates": 0,
        "early_stopping_evaluations": 0,
        "runtime_only_fields": [
            "materialized_memory_writer_turns",
            "materialized_memory_writer_tokens",
            "terminal_answer_tokens",
            "dynamic_microbatch_sections",
            "generated_tokens",
            "forward_backward_flops",
            "wall_time",
        ],
    }


def actual_budget(
    run_root: str | Path | None,
    training_root: str | Path | None,
    *,
    expected_commit: str,
    expected_manifest_sha256: str | None = None,
    p0_t5_path: str | Path | None = None,
    expected_p0_t5_file_sha256: str | None = None,
    expected_dataset_cursor: Sequence[int] | None = None,
    expected_gpu_pair: Sequence[int] | None = None,
    expected_gate_hashes: Mapping[str, str] | None = None,
    expected_original_resolved_sha256: str | None = None,
    expected_s128_resolved_sha256: str | None = None,
    expected_weight_sync_parameters: Sequence[str] | None = None,
    expected_weight_transfer_format: str | None = None,
    expected_loaded_parameter_count: int | None = None,
) -> dict[str, Any]:
    """Derive completed budget from optimizer, trajectory, and checkpoint evidence."""
    if run_root is None and training_root is None:
        return {
            "status": "PENDING_ACTUAL_T25_LEDGER",
            "reason": "no completed Method-T25 run was supplied",
        }
    if run_root is None or training_root is None:
        raise ValueError("CORAL_SCOPE_NO_GO: actual budget needs both run roots")
    run = Path(run_root).resolve()
    training = Path(training_root).resolve()
    coral_path = run / "coral_execution_ledger.jsonl"
    coral_rows = validate_ledger(coral_path)
    updates = [row["payload"] for row in coral_rows
               if row["payload"].get("event") == "coral_role_update"]
    steps = [int(row.get("global_step", -1)) for row in updates]
    if steps != list(range(1, 26)):
        return {
            "status": "PENDING_INCOMPLETE_T25_LEDGER",
            "observed_update_steps": steps,
            "reason": "actual counts are not promoted from an incomplete run",
        }
    if [int(row.get("actor_update_calls", -1)) for row in updates] != list(range(1, 26)):
        raise ValueError("CORAL_SCOPE_NO_GO: actor update-call inventory")
    expected_phases = ["memory_writer" if step % 2 else "terminal_answer"
                       for step in range(1, 26)]
    if [row.get("phase") for row in updates] != expected_phases:
        raise ValueError("CORAL_SCOPE_NO_GO: actual role schedule")

    gate_path = run / "gate_a_execution_ledger.jsonl"
    gate_rows = [json.loads(line) for line in gate_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if validate_jsonl_chain(gate_rows):
        raise ValueError("CORAL_SCOPE_NO_GO: Gate-A ledger chain")
    relevant = [row for row in gate_rows if row.get("record_type") in {
        "weight_sync_ack", "weight_sync_summary",
    } and row.get("sync_kind") == "post_actor_update"]
    if any(row.get("git_commit") != expected_commit for row in relevant):
        raise ValueError("CORAL_SCOPE_NO_GO: optimizer evidence commit drift")
    summaries = [row for row in relevant if row.get("record_type") == "weight_sync_summary"]
    if [int(row.get("global_step", -1)) for row in summaries] != list(range(1, 26)):
        raise ValueError("CORAL_SCOPE_NO_GO: exact post-update sync inventory")
    acknowledgements = [row for row in relevant if row.get("record_type") == "weight_sync_ack"]
    if len(acknowledgements) != 50:
        raise ValueError("CORAL_SCOPE_NO_GO: rank-complete optimizer acknowledgements")
    state_counts: dict[int, set[int]] = {0: set(), 1: set()}
    for step, summary in enumerate(summaries, start=1):
        if summary.get("worker_ranks") != [0, 1] \
                or int(summary.get("actor_version", -1)) != step:
            raise ValueError("CORAL_SCOPE_NO_GO: optimizer summary identity")
        step_acks = [row for row in acknowledgements
                     if int(row.get("global_step", -1)) == step]
        if sorted(int(row.get("vllm_worker_rank", -1)) for row in step_acks) != [0, 1]:
            raise ValueError("CORAL_SCOPE_NO_GO: optimizer rank inventory")
        for ack in step_acks:
            rank = int(ack["vllm_worker_rank"])
            state_count = ack.get("optimizer_state_entry_count")
            step_count = ack.get("optimizer_step_entry_count")
            histogram = ack.get("optimizer_step_histogram")
            if type(state_count) is not int or state_count < 1 \
                    or type(step_count) is not int or step_count != state_count \
                    or ack.get("optimizer_step_min") != step \
                    or ack.get("optimizer_step_max") != step \
                    or ack.get("lr_scheduler_last_epoch") != step \
                    or histogram != {str(step): state_count}:
                raise ValueError("CORAL_SCOPE_NO_GO: optimizer did not advance exactly once")
            state_counts[rank].add(state_count)
            if ack.get("actor_rollout_sampled_tensor_digest") \
                    != summary.get("sampled_tensor_digest"):
                raise ValueError("CORAL_SCOPE_NO_GO: optimizer/sync digest mismatch")
    if any(len(values) != 1 for values in state_counts.values()):
        raise ValueError("CORAL_SCOPE_NO_GO: optimizer state inventory drift")
    from tools.h20.audit_qwen25_7b_gatea import audit_sync
    if not isinstance(expected_weight_sync_parameters, Sequence) \
            or isinstance(expected_weight_sync_parameters, (str, bytes)) \
            or len(expected_weight_sync_parameters) < 1 \
            or any(not isinstance(value, str) or not value
                   for value in expected_weight_sync_parameters) \
            or len(set(expected_weight_sync_parameters)) \
            != len(expected_weight_sync_parameters) \
            or expected_weight_transfer_format != "dtensor" \
            or type(expected_loaded_parameter_count) is not int \
            or expected_loaded_parameter_count < len(expected_weight_sync_parameters):
        raise ValueError("CORAL_SCOPE_NO_GO: authenticated weight-sync contract required")
    sync_ok, sync_failures, _ = audit_sync(
        gate_rows, list(range(1, 26)), [0, 1],
        required_syncs=[("", step, "post_actor_update") for step in range(1, 26)],
        required_parameters=list(expected_weight_sync_parameters),
        required_transfer_format=expected_weight_transfer_format,
        expected_loaded_parameter_count=expected_loaded_parameter_count,
    )
    if not sync_ok:
        raise ValueError(
            "CORAL_SCOPE_NO_GO: complete Gate-A sync audit: "
            + "; ".join(sync_failures)
        )

    seed_path = training / "rollout_seed_audit.jsonl"
    seed_rows = [json.loads(line) for line in seed_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    trajectories = [row for row in seed_rows if row.get("record_type") == "trajectory_seed"]
    turns = [row for row in seed_rows if row.get("record_type") == "trajectory_turn_seed"]
    expected_pairs = {(step, row) for step in range(1, 26) for row in range(8)}
    actual_pairs = {(int(row["global_step"]), int(row["row"])) for row in trajectories}
    if actual_pairs != expected_pairs or len(trajectories) != 200:
        raise ValueError("CORAL_SCOPE_NO_GO: trajectory inventory is not exact 25x8")
    if expected_dataset_cursor is None \
            or len(expected_dataset_cursor) != 100 \
            or any(type(value) is not int for value in expected_dataset_cursor):
        raise ValueError("CORAL_SCOPE_NO_GO: authenticated Original cursor required")
    cursor = list(expected_dataset_cursor)
    from recurrent.research.trajectory_seeding import stable_training_group_id
    for step in range(1, 26):
        step_rows = {int(row.get("row", -1)): row for row in trajectories
                     if int(row.get("global_step", -1)) == step}
        for row in step_rows.values():
            dataset_index = int(row.get("dataset_index", -1))
            if row.get("base_seed") != 2026 \
                    or row.get("uid") != stable_training_group_id(
                        base_seed=2026, global_step=step,
                        dataset_index=dataset_index,
                    ):
                raise ValueError("CORAL_SCOPE_NO_GO: trajectory stable uid/base seed")
        for group in range(4):
            indices = {
                int(step_rows[row].get("dataset_index", -1))
                for row in (group * 2, group * 2 + 1) if row in step_rows
            }
            expected_index = cursor[(step - 1) * 4 + group]
            if indices != {expected_index}:
                raise ValueError("CORAL_SCOPE_NO_GO: trajectory dataset/group identity")
    from tools.h20.audit_qwen25_7b_gatea import audit_seeds, component_inventory
    seeds_ok, seed_failures = audit_seeds(
        seed_rows, 2026, 2, expected_steps=list(range(1, 26)),
        expected_batch_size=8, expected_dataset_cursor=cursor,
    )
    if not seeds_ok:
        raise ValueError(
            "CORAL_SCOPE_NO_GO: trajectory seed/identity audit: "
            + "; ".join(seed_failures)
        )
    terminal_turns = [row for row in turns if row.get("is_final") is True]
    writer_turns = [row for row in turns if row.get("is_final") is False]
    turn_keys = [
        (int(row.get("global_step", -1)), int(row.get("row", -1)),
         int(row.get("turn", -1)), bool(row.get("is_final")))
        for row in turns
    ]
    if len(turn_keys) != len(set(turn_keys)) \
            or any((step, row) not in expected_pairs or turn < 0
                   for step, row, turn, _ in turn_keys):
        raise ValueError("CORAL_SCOPE_NO_GO: duplicate/out-of-scope recurrent turn")
    terminal_pairs = {(int(row["global_step"]), int(row["row"])) for row in terminal_turns}
    if terminal_pairs != expected_pairs or len(terminal_turns) != 200:
        raise ValueError("CORAL_SCOPE_NO_GO: every trajectory needs one terminal turn")

    anchors = [5, 10, 15, 20, 25]
    anchor_hashes = {}
    for step in anchors:
        checkpoint = training / f"global_step_{step}"
        _, missing = component_inventory(checkpoint, 2)
        if missing:
            raise ValueError(
                f"CORAL_SCOPE_NO_GO: incomplete anchor checkpoint {step}: {missing}"
            )
        anchor_hashes[str(step)] = checkpoint_sha256(checkpoint)
    if p0_t5_path is None:
        raise ValueError("CORAL_SCOPE_NO_GO: completed budget requires T5 P0 binding")
    p0_path = Path(p0_t5_path).resolve()
    if expected_p0_t5_file_sha256 is None \
            or sha256_file(p0_path) != expected_p0_t5_file_sha256:
        raise ValueError("CORAL_SCOPE_NO_GO: external T5 P0 file SHA binding")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    p0_fields = {
        "schema", "status", "decision", "stage", "git_commit", "manifest_sha256",
        "original_resolved_manifest_sha256", "original_p0_certificate_sha256",
        "original_training_final_sha256", "original_training_ledger_sha256",
        "s128_resolved_manifest_sha256", "s128_final_sha256", "s128_ledger_sha256",
        "evidence_authority_sha256", "fresh_base_model_tokenizer_inventory_sha256",
        "original_protocol_comparison_sha256", "original_protocol_compared_leaves",
        "resolved_config_comparison_sha256", "method_nonwhitelist_config_sha256",
        "resolved_config_comparison", "gpu_pair", "gate_hashes", "report_sha256",
    }
    p0_unsigned = {key: value for key, value in p0.items() if key != "report_sha256"}
    sha_fields = p0_fields - {
        "schema", "status", "decision", "stage", "git_commit",
        "original_protocol_compared_leaves", "resolved_config_comparison", "gpu_pair",
        "gate_hashes", "report_sha256",
    }
    if not isinstance(p0, Mapping) or set(p0) != p0_fields \
            or p0.get("schema") != "memagent.cosi.preflight.v4" \
            or p0.get("status") != "PASS" or p0.get("decision") != "COSI_T5_P0_PASS" \
            or p0.get("stage") != "t5" or p0.get("git_commit") != expected_commit \
            or (expected_manifest_sha256 is not None
                and p0.get("manifest_sha256") != expected_manifest_sha256) \
            or any(not isinstance(p0.get(field), str)
                   or len(p0[field]) != 64
                   or any(char not in "0123456789abcdef" for char in p0[field])
                   for field in sha_fields) \
            or not isinstance(p0.get("original_protocol_compared_leaves"), list) \
            or not p0["original_protocol_compared_leaves"] \
            or not isinstance(p0.get("resolved_config_comparison"), Mapping) \
            or p0.get("resolved_config_comparison_sha256") \
            != canonical_sha256(p0["resolved_config_comparison"]) \
            or p0.get("method_nonwhitelist_config_sha256") \
            != p0["resolved_config_comparison"].get(
                "method_nonwhitelist_config_sha256"
            ) \
            or not isinstance(p0.get("gpu_pair"), list) or len(p0["gpu_pair"]) != 2 \
            or any(type(gpu) is not int or gpu < 0 for gpu in p0["gpu_pair"]) \
            or p0["gpu_pair"] != sorted(set(p0["gpu_pair"])) \
            or not isinstance(p0.get("gate_hashes"), Mapping) \
            or set(p0["gate_hashes"]) != {"paper", "e0", "e1", "baseline", "scope"} \
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in p0["gate_hashes"].values()) \
            or p0.get("report_sha256") != canonical_sha256(p0_unsigned):
        raise ValueError("CORAL_SCOPE_NO_GO: complete T5 P0 binding")
    if expected_gpu_pair is not None and p0["gpu_pair"] != list(expected_gpu_pair):
        raise ValueError("CORAL_SCOPE_NO_GO: T5 P0 GPU pair drift")
    if expected_gate_hashes is not None and p0["gate_hashes"] != dict(expected_gate_hashes):
        raise ValueError("CORAL_SCOPE_NO_GO: T5 P0 gate projection drift")
    if expected_original_resolved_sha256 is not None \
            and p0["original_resolved_manifest_sha256"] \
            != expected_original_resolved_sha256:
        raise ValueError("CORAL_SCOPE_NO_GO: T5 P0 Original authority drift")
    if expected_s128_resolved_sha256 is not None \
            and p0["s128_resolved_manifest_sha256"] != expected_s128_resolved_sha256:
        raise ValueError("CORAL_SCOPE_NO_GO: T5 P0 S128 authority drift")

    writer_tokens = sum(int(row["active_tokens"] if row["phase"] == "memory_writer"
                            else row["inactive_tokens"]) for row in updates)
    terminal_tokens = sum(int(row["inactive_tokens"] if row["phase"] == "memory_writer"
                              else row["active_tokens"]) for row in updates)
    return {
        "status": "COMPLETE_ACTUAL_T25_BUDGET",
        "classification": "single_seed_early_budget_pilot_not_convergence",
        "git_commit": expected_commit,
        "run_root": str(run),
        "training_root": str(training),
        "actor_optimizer_updates": 25,
        "optimizer_steps_per_update": 1,
        "ppo_epochs_per_update": 1,
        "prompt_minibatches_per_epoch_per_update": 1,
        "optimizer_rank_acknowledgements": 50,
        "optimizer_state_entry_count_by_rank": {
            str(rank): next(iter(values)) for rank, values in state_counts.items()
        },
        "prompt_groups": 100,
        "sampled_training_trajectories": 200,
        "authenticated_dataset_cursor_sha256": canonical_sha256(cursor),
        "materialized_memory_writer_turns": len(writer_turns),
        "terminal_answer_turns": 200,
        "materialized_memory_writer_tokens": writer_tokens,
        "terminal_answer_tokens": terminal_tokens,
        "post_actor_weight_syncs": 25,
        "critic_fit_optimizer_updates": 0,
        "prior_or_reference_fit_optimizer_updates": 0,
        "auxiliary_fit_optimizer_updates": 0,
        "early_stopping_evaluations": 0,
        "anchor_checkpoint_sha256": anchor_hashes,
        "t5_p0_path": str(p0_path),
        "t5_p0_file_sha256": sha256_file(p0_path),
        "coral_ledger_sha256": sha256_file(coral_path),
        "gate_a_ledger_sha256": sha256_file(gate_path),
        "rollout_seed_audit_sha256": sha256_file(seed_path),
    }


def validate_scope_report(
    report: Mapping[str, Any], *, expected_commit: str,
    expected_manifest_path: str, expected_manifest_sha256: str,
    expected_repo: str, expected_work_root: str,
    expected_train_sha256: str, expected_s128_parquet_sha256: str,
    expected_s128_resolved_path: str, expected_s128_resolved_sha256: str,
    expected_eval_manifest_hash: str,
) -> dict[str, Any]:
    """Validate the security-critical conclusions of a scope certificate."""
    if not isinstance(report, Mapping):
        raise TypeError("CORAL_SCOPE_NO_GO: scope report must be a mapping")
    exact_fields = {
        "schema", "status", "decision", "git_commit", "repository",
        "method_manifest", "static_training_budget", "actual_training_budget",
        "data_roles", "set_intersections", "adaptive_benchmark_classification",
        "report_sha256",
    }
    if set(report) != exact_fields:
        raise ValueError("CORAL_SCOPE_NO_GO: scope report fields drifted")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("schema") != "memagent.coral.scientific-scope-audit.v1" \
            or report.get("status") != "PASS" \
            or report.get("decision") \
            != "CORAL_SCOPE_DIRECT_LEAKAGE_CLEAR_S128_ADAPTIVE_DEV_ONLY" \
            or report.get("git_commit") != expected_commit \
            or report.get("report_sha256") != canonical_sha256(unsigned):
        raise ValueError("CORAL_SCOPE_NO_GO: scope report authentication")
    method_manifest = report.get("method_manifest")
    if not isinstance(method_manifest, Mapping) or set(method_manifest) != {"path", "sha256"} \
            or method_manifest.get("path") != expected_manifest_path \
            or method_manifest.get("sha256") != expected_manifest_sha256 \
            or report.get("repository") != expected_repo:
        raise ValueError("CORAL_SCOPE_NO_GO: source/manifest binding")
    roles = report.get("data_roles")
    expected_roles = {
        "actor_training", "critic_fit", "prior_or_reference_fit", "auxiliary_fit",
        "hyperparameter_and_method_selection", "early_stopping", "fixed_s128",
    }
    if not isinstance(roles, Mapping) or set(roles) != expected_roles:
        raise ValueError("CORAL_SCOPE_NO_GO: data-role inventory")
    actor = roles["actor_training"]
    actor_fields = {
        "path", "parquet_sha256", "resolved_row_manifest", "row_count",
        "unique_content_identity_count", "duplicate_content_row_count",
        "content_identity_inventory_sha256", "question_inventory_sha256",
        "context_inventory_sha256",
    }
    if not isinstance(actor, Mapping) or set(actor) != actor_fields \
            or actor.get("path") \
            != str(Path(expected_work_root).resolve()
                   / "datasets/hotpotqa/hotpotqa_train_32k.parquet") \
            or actor.get("parquet_sha256") != expected_train_sha256 \
            or actor.get("resolved_row_manifest") \
            != "derived_read_only_content_inventory_in_this_report" \
            or type(actor.get("row_count")) is not int or actor["row_count"] < 1 \
            or type(actor.get("unique_content_identity_count")) is not int \
            or not 1 <= actor["unique_content_identity_count"] <= actor["row_count"] \
            or actor.get("duplicate_content_row_count") \
            != actor["row_count"] - actor["unique_content_identity_count"]:
        raise ValueError("CORAL_SCOPE_NO_GO: actor data binding")
    fixed = roles["fixed_s128"]
    fixed_fields = {
        "parquet_path", "parquet_sha256", "resolved_manifest_path",
        "resolved_manifest_file_sha256", "eval_manifest_hash", "row_count",
        "unique_content_identity_count", "duplicate_content_row_count",
        "content_identity_inventory_sha256", "question_inventory_sha256",
        "context_inventory_sha256",
    }
    if not isinstance(fixed, Mapping) or set(fixed) != fixed_fields \
            or fixed.get("parquet_path") \
            != str(Path(expected_work_root).resolve()
                   / "datasets/hotpotqa/hotpotqa_dev.parquet") \
            or fixed.get("parquet_sha256") != expected_s128_parquet_sha256 \
            or fixed.get("resolved_manifest_path") != expected_s128_resolved_path \
            or fixed.get("resolved_manifest_file_sha256") != expected_s128_resolved_sha256 \
            or fixed.get("eval_manifest_hash") != expected_eval_manifest_hash \
            or fixed.get("row_count") != 128 \
            or fixed.get("unique_content_identity_count") != 128 \
            or fixed.get("duplicate_content_row_count") != 0:
        raise ValueError("CORAL_SCOPE_NO_GO: S128 authority binding")
    for name in ("content_identity_inventory_sha256", "question_inventory_sha256",
                 "context_inventory_sha256"):
        for bound in (actor, fixed):
            value = bound.get(name)
            if not isinstance(value, str) or len(value) != 64 \
                    or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("CORAL_SCOPE_NO_GO: identity inventory SHA")
    train_path = Path(expected_work_root).resolve() \
        / "datasets/hotpotqa/hotpotqa_train_32k.parquet"
    s128_path = Path(expected_work_root).resolve() \
        / "datasets/hotpotqa/hotpotqa_dev.parquet"
    if sha256_file(train_path) != expected_train_sha256 \
            or sha256_file(s128_path) != expected_s128_parquet_sha256 \
            or sha256_file(expected_s128_resolved_path) != expected_s128_resolved_sha256:
        raise ValueError("CORAL_SCOPE_NO_GO: live authority file SHA mismatch")
    recomputed = recompute_scope_evidence(
        train_path, s128_path, expected_s128_resolved_path
    )
    actor_report_inventory = {key: actor[key] for key in recomputed["actor_inventory"]}
    fixed_report_inventory = {key: fixed[key] for key in recomputed["s128_inventory"]}
    if actor_report_inventory != recomputed["actor_inventory"] \
            or fixed_report_inventory != recomputed["s128_inventory"] \
            or recomputed["eval_manifest_hash"] != expected_eval_manifest_hash:
        raise ValueError("CORAL_SCOPE_NO_GO: independently recomputed inventory mismatch")
    if roles["critic_fit"] != {"status": "EMPTY_GRPO_HAS_NO_CRITIC", "row_count": 0} \
            or roles["prior_or_reference_fit"] \
            != {"status": "EMPTY_REFERENCE_POLICY_IS_FROZEN_NOT_FIT", "row_count": 0} \
            or roles["auxiliary_fit"] \
            != {"status": "EMPTY_NO_LEARNED_AUXILIARY", "row_count": 0} \
            or roles["early_stopping"] \
            != {"status": "EMPTY_TEST_FREQ_MINUS_ONE_T5_HEALTH_HAS_NO_BENCHMARK",
                "row_count": 0}:
        raise ValueError("CORAL_SCOPE_NO_GO: fit/early-stop data roles")
    selection = roles["hyperparameter_and_method_selection"]
    if not isinstance(selection, Mapping) \
            or selection.get("s128_is_in_selection_domain") is not True \
            or selection.get("components") != [
                "E0 synthetic recurrent MDP (no HotpotQA rows)",
                "E1 mechanism roots from authenticated actor-training parquet",
                "previously observed Original/Capture32 fixed-S128 facts",
            ]:
        raise ValueError("CORAL_SCOPE_NO_GO: selection-data declaration")
    intersections = report.get("set_intersections")
    expected_intersection_names = {
        "actor_train_intersection_s128", "critic_fit_intersection_s128",
        "selection_intersection_s128",
    }
    overlap_fields = {
        "canonical_content_pair_count", "question_hash_count", "context_hash_count",
        "canonical_content_pair_inventory_sha256", "matching_content_pair_sha256",
    }
    if not isinstance(intersections, Mapping) \
            or set(intersections) != expected_intersection_names:
        raise ValueError("CORAL_SCOPE_NO_GO: scope intersections missing")
    expected = {
        "actor_train_intersection_s128": 0,
        "critic_fit_intersection_s128": 0,
        "selection_intersection_s128": 128,
    }
    for name, count in expected.items():
        row = intersections.get(name)
        if not isinstance(row, Mapping) or set(row) != overlap_fields \
                or type(row.get("canonical_content_pair_count")) is not int \
                or row["canonical_content_pair_count"] != count \
                or not isinstance(row.get("matching_content_pair_sha256"), list) \
                or len(row["matching_content_pair_sha256"]) != count \
                or row.get("canonical_content_pair_inventory_sha256") \
                != canonical_sha256(row["matching_content_pair_sha256"]):
            raise ValueError(f"CORAL_SCOPE_NO_GO: invalid intersection {name}")
    actor = intersections["actor_train_intersection_s128"]
    if actor.get("question_hash_count") != 0 or actor.get("context_hash_count") != 0:
        raise ValueError("CORAL_SCOPE_NO_GO: partial actor/S128 content overlap")
    if intersections["actor_train_intersection_s128"] != recomputed["actor_overlap"] \
            or intersections["critic_fit_intersection_s128"] != recomputed["critic_overlap"] \
            or intersections["selection_intersection_s128"] \
            != recomputed["selection_overlap"]:
        raise ValueError("CORAL_SCOPE_NO_GO: independently recomputed overlap mismatch")
    adaptive = report.get("adaptive_benchmark_classification")
    if not isinstance(adaptive, Mapping) or set(adaptive) != {
        "fixed_s128", "reason", "untouched_confirmation_required",
    } \
            or adaptive.get("fixed_s128") != "DEVELOPMENT_SCREEN_NOT_BLIND_FINAL_TEST" \
            or not isinstance(adaptive.get("reason"), str) or not adaptive["reason"] \
            or adaptive.get("untouched_confirmation_required") is not True:
        raise ValueError("CORAL_SCOPE_NO_GO: adaptive benchmark classification")
    budget = report.get("static_training_budget")
    bound_manifest = json.loads(Path(expected_manifest_path).read_text(encoding="utf-8"))
    if sha256_file(expected_manifest_path) != expected_manifest_sha256 \
            or budget != static_budget(bound_manifest):
        raise ValueError("CORAL_SCOPE_NO_GO: training-budget classification")
    if report.get("actual_training_budget") != {
        "status": "PENDING_ACTUAL_T25_LEDGER",
        "reason": "no completed Method-T25 run was supplied",
    }:
        raise ValueError("CORAL_SCOPE_NO_GO: pretraining scope cannot claim actual updates")
    return dict(report)
