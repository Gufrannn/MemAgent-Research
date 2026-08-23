#!/usr/bin/env python3
"""Read-only scientific budget, overlap, and adaptive-use audit for MIC.

This audit intentionally hashes row contents.  A train/dev filename split is
not accepted as evidence that the underlying examples are disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from recurrent.research.mic import CriticCheckpoint, canonical_json, sha256_file, sha256_json
from recurrent.research.stable_eval_identity import canonical_sha256, validate_resolved_manifest
from tools.h20.audit_qwen25_7b_gatea import audit_seeds
from tools.h20.mic_pipeline import audit as replay_mic_training_audit


SCHEMA = "memagent.mic.scientific-evidence.v1"
ANCHORS = (5, 10, 15, 20, 25)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8"
    ).splitlines() if line]


def _git_blob(commit: str, relative_path: str) -> tuple[str, str]:
    value = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{commit}:{relative_path}"],
        text=True,
    )
    return value, hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    item = getattr(value, "item", None)
    return _json_native(item()) if callable(item) else value


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return [_json_native(row) for row in parquet.read_table(
        path, columns=["prompt", "context", "reward_model", "extra_info"]
    ).to_pylist()]


def _question(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("MIC_NO_GO: parquet row prompt is not a non-empty message list")
    first = prompt[0]
    if not isinstance(first, Mapping) or first.get("role") != "user" \
            or not isinstance(first.get("content"), str):
        raise ValueError("MIC_NO_GO: parquet row has no canonical user question")
    return str(first["content"])


def _row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    question = _question(row)
    context = row.get("context")
    reward_model = row.get("reward_model")
    extra_info = row.get("extra_info")
    if not isinstance(context, str) or not isinstance(reward_model, Mapping) \
            or "ground_truth" not in reward_model or not isinstance(extra_info, Mapping) \
            or isinstance(extra_info.get("index"), bool) \
            or not isinstance(extra_info.get("index"), int):
        raise ValueError("MIC_NO_GO: parquet scientific identity schema drifted")
    q_sha = hashlib.sha256(question.encode("utf-8")).hexdigest()
    c_sha = hashlib.sha256(context.encode("utf-8")).hexdigest()
    gt_sha = canonical_sha256(reward_model["ground_truth"])
    root = canonical_sha256({"source_question_hash": q_sha, "source_context_hash": c_sha})
    example = canonical_sha256({"content_root_sha256": root, "ground_truth_hash": gt_sha})
    return {
        "semantic_dataset_index": int(extra_info["index"]),
        "source_question_hash": q_sha,
        "source_context_hash": c_sha,
        "ground_truth_hash": gt_sha,
        "content_root_sha256": root,
        "content_example_sha256": example,
    }


def _validate_chain(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous = "0" * 64
    checked = []
    for sequence, source in enumerate(rows):
        row = dict(source)
        digest = row.pop("entry_sha256", None)
        if row.get("sequence") != sequence or row.get("previous_entry_sha256") != previous \
                or digest != sha256_json(row):
            raise ValueError("MIC_NO_GO: MIC training ledger chain is corrupt")
        previous = str(digest)
        row["entry_sha256"] = digest
        checked.append(row)
    return checked


def _ensure_unique_index(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    counts = Counter(int(row["semantic_dataset_index"]) for row in rows)
    duplicates = sorted(index for index, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(f"MIC_NO_GO: {label} semantic indices are not unique: {duplicates[:8]}")
    return {int(row["semantic_dataset_index"]): row for row in rows}


def _intersection(left: list[dict[str, Any]], right: list[dict[str, Any]], field: str) -> dict[str, Any]:
    left_set = {str(row[field]) for row in left}
    right_set = {str(row[field]) for row in right}
    overlap = sorted(left_set & right_set)
    return {
        "field": field,
        "left_unique": len(left_set),
        "right_unique": len(right_set),
        "intersection_count": len(overlap),
        "intersection_inventory_sha256": canonical_sha256(overlap),
        "intersection_values": overlap,
    }


def _validate_seed_producer_bindings(seed_rows: list[dict[str, Any]]) -> None:
    for row in seed_rows:
        expected_uid = hashlib.sha256(json.dumps({
            "namespace": "memagent-mic-prompt-group-v1",
            "global_step": int(row["global_step"]),
            "dataset_index": int(row["dataset_index"]),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if row.get("uid") != expected_uid or int(row.get("base_seed", -1)) != 2026:
            raise ValueError("MIC_NO_GO: rollout UID/base seed differs from producer contract")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    p0_path = Path(args.p0).resolve()
    e0_path = Path(args.e0).resolve()
    train_path = Path(args.train_parquet).resolve()
    s128_path = Path(args.s128_parquet).resolve()
    resolved_path = Path(args.s128_resolved).resolve()
    ledger_path = Path(args.mic_ledger).resolve()
    weight_ledger_path = Path(args.weight_ledger).resolve()
    seed_path = Path(args.rollout_seed_audit).resolve()
    critic_path = Path(args.critic_checkpoint).resolve()
    disclosure_path = Path(args.adaptive_disclosure).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"MIC_NO_GO: refusing to overwrite {output_path}")

    manifest = _read_json(manifest_path)
    p0 = _read_json(p0_path)
    e0 = _read_json(e0_path)
    if p0.get("status") != "PASS" or p0.get("decision") != "MIC_P0_PASS" \
            or p0.get("run_id") != args.run_id \
            or not re.fullmatch(r"[0-9a-f]{40}", str(p0.get("git_commit", ""))):
        raise ValueError("MIC_NO_GO: training P0 identity is not authoritative")
    if p0.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("MIC_NO_GO: frozen MIC manifest differs from training P0")
    if e0.get("status") != "PASS" or e0.get("decision") != "MIC_E0_PASS":
        raise ValueError("MIC_NO_GO: E0 synthetic-selection gate is not authoritative")
    training_commit = str(p0["git_commit"])
    trainer_config_source, trainer_config_sha = _git_blob(
        training_commit, "verl/trainer/config/ppo_trainer.yaml"
    )
    launcher_source, launcher_sha = _git_blob(
        training_commit, "experiments/7b_gate_a/run_gate_a.sh"
    )
    if not re.search(r"^\s+ppo_epochs:\s+1\s*$", trainer_config_source, re.MULTILINE) \
            or "trainer.val_before_train=False" not in launcher_source \
            or "trainer.test_freq=-1" not in launcher_source \
            or '"trainer.total_training_steps=$TOTAL_STEPS"' not in launcher_source:
        raise ValueError("MIC_NO_GO: training-commit budget/selection source contract drifted")
    if manifest.get("training", {}).get("train_batch_size") != 4 \
            or manifest.get("training", {}).get("rollout_n") != 2 \
            or manifest.get("training", {}).get("ppo_mini_batch_size") != 4 \
            or manifest.get("training", {}).get("anchors") != list(ANCHORS):
        raise ValueError("MIC_NO_GO: frozen MIC budget manifest drifted")

    resolved = validate_resolved_manifest(_read_json(resolved_path))
    curve_authority_path = REPO / "manifests/h20/qwen25_7b_mic_original_curve_authority.json"
    curve_authority = _read_json(curve_authority_path)
    if Path(str(curve_authority.get("curve_resolved_path", ""))).resolve() != resolved_path \
            or curve_authority.get("curve_resolved_sha256") != sha256_file(resolved_path):
        raise ValueError("MIC_NO_GO: fixed-S128 resolved manifest lacks release authority")
    expected_s128_sha = resolved["identity_payload"].get("source_dataset", {}).get(
        "parquet_sha256"
    )
    if expected_s128_sha != sha256_file(s128_path):
        raise ValueError("MIC_NO_GO: fixed-S128 parquet differs from resolved identity manifest")
    frozen_rows = resolved["identity_payload"]["rows"]
    if len(frozen_rows) != 128:
        raise ValueError("MIC_NO_GO: fixed-S128 identity coverage is not 128")

    train_raw = _load_parquet(train_path)
    s128_raw = _load_parquet(s128_path)
    if len(s128_raw) != 128:
        raise ValueError("MIC_NO_GO: fixed-S128 parquet row coverage is not 128")
    train_rows = [_row_identity(row) for row in train_raw]
    s128_rows = [_row_identity(row) for row in s128_raw]
    train_by_index = _ensure_unique_index(train_rows, "actor-train")
    original_resolved_path = Path(str(p0.get("original_resolved_manifest", ""))).resolve()
    if not original_resolved_path.is_file() \
            or sha256_file(original_resolved_path) != p0.get(
                "original_resolved_manifest_sha256"
            ):
        raise ValueError("MIC_NO_GO: accepted Original resolved protocol authority differs")
    original_resolved = _read_json(original_resolved_path)
    original_data = original_resolved.get("data", {})
    if Path(str(original_data.get("train", ""))).resolve() != train_path \
            or original_data.get("train_sha256") != sha256_file(train_path) \
            or Path(str(original_data.get("validation", ""))).resolve() != s128_path \
            or original_data.get("validation_sha256") != sha256_file(s128_path):
        raise ValueError("MIC_NO_GO: actor/S128 data files differ from accepted protocol authority")
    original_final_path = Path(
        manifest["certified_read_only_sources"]["original_t25_training"]["final_report"]
    ).resolve()
    expected_original_final_sha = manifest[
        "certified_read_only_sources"
    ]["original_t25_training"]["final_report_sha256"]
    original_p0_path = original_resolved_path.parent / "p0_preflight.json"
    if not original_final_path.is_file() or sha256_file(original_final_path) != \
            expected_original_final_sha or not original_p0_path.is_file():
        raise ValueError("MIC_NO_GO: accepted Original cursor authority is absent")
    original_final = _read_json(original_final_path)
    original_p0 = _read_json(original_p0_path)
    if original_final.get("status") != "PASS" \
            or original_final.get("decision") != "ORIGINAL_T25_PASS" \
            or original_final.get("p0_certificate_sha256") != sha256_file(
                original_p0_path
            ) \
            or original_p0.get("status") != "PASS" \
            or original_p0.get("decision") != "T25_P0_PASS":
        raise ValueError("MIC_NO_GO: accepted Original cursor P0 is not authenticated")
    expected_cursor = original_p0.get("evidence", {}).get(
        "train_cursor_semantic_indices_0_to_99"
    )
    if not isinstance(expected_cursor, list) or len(expected_cursor) != 100:
        raise ValueError("MIC_NO_GO: frozen 100-group semantic cursor is absent")
    for frozen in frozen_rows:
        raw = int(frozen["raw_row_position"])
        actual = s128_rows[raw]
        if actual["semantic_dataset_index"] != int(frozen["semantic_dataset_index"]) \
                or actual["source_question_hash"] != frozen["source_question_hash"] \
                or actual["source_context_hash"] != frozen["source_context_hash"] \
                or actual["ground_truth_hash"] != frozen["ground_truth_hash"]:
            raise ValueError("MIC_NO_GO: fixed-S128 content/identity manifest mismatch")

    all_seed_rows = _read_jsonl(seed_path)
    seed_ok, seed_failures = audit_seeds(
        all_seed_rows, 2026, 2, expected_steps=list(range(1, 26)),
        expected_batch_size=8, expected_dataset_cursor=expected_cursor,
    )
    if not seed_ok:
        raise ValueError("MIC_NO_GO: rollout seed/cursor audit failed: " + "; ".join(
            seed_failures
        ))
    seed_rows = [row for row in all_seed_rows
                 if row.get("record_type") == "trajectory_seed"]
    if len(seed_rows) != 25 * 4 * 2 \
            or sorted({int(row["global_step"]) for row in seed_rows}) != list(range(1, 26)):
        raise ValueError("MIC_NO_GO: rollout seed audit does not cover 25x4x2 trajectories")
    _validate_seed_producer_bindings(seed_rows)
    trajectory_ids = [f"{row['uid']}:{int(row['trajectory_seed'])}" for row in seed_rows]
    if len(set(trajectory_ids)) != 200:
        raise ValueError("MIC_NO_GO: rollout trajectory identity is not unique")
    actor_indices = [int(row["dataset_index"]) for row in seed_rows]
    missing_indices = sorted(set(actor_indices) - set(train_by_index))
    if missing_indices:
        raise ValueError(f"MIC_NO_GO: rollout indices absent from train parquet: {missing_indices[:8]}")
    critic_fit_rows = [train_by_index[index] for index in sorted(set(actor_indices))]

    ledger = _validate_chain(_read_jsonl(ledger_path))
    deliveries = [row for row in ledger if row.get("record_type") == "mic_advantage_delivery"]
    gradients = [row for row in ledger if row.get("record_type") == "mic_actual_gradient_delivery"]
    if [int(row["global_step"]) for row in deliveries] != list(range(1, 26)) \
            or [int(row["global_step"]) for row in gradients] != list(range(1, 26)):
        raise ValueError("MIC_NO_GO: update-level MIC delivery/gradient coverage is not 1..25")
    ledger_trajectory_ids = [str(value) for row in deliveries
                             for value in row.get("current_trajectory_ids", [])]
    if ledger_trajectory_ids != trajectory_ids:
        raise ValueError("MIC_NO_GO: seed audit and critic trajectory identities differ")

    critic = CriticCheckpoint.read(critic_path, expected_actor_commit=training_commit)
    if critic_path != Path(str(deliveries[-1]["critic_checkpoint"])).resolve() \
            or critic["checkpoint_sha256"] != deliveries[-1]["critic_checkpoint_sha256"]:
        raise ValueError("MIC_NO_GO: supplied T25 critic is not the final training-ledger checkpoint")
    payload = critic.get("critic_payload", {})
    outcomes = payload.get("history_outcomes", {})
    states = payload.get("history_states", [])
    if set(outcomes) != set(trajectory_ids) or len(outcomes) != 200:
        raise ValueError("MIC_NO_GO: T25 critic outcome inventory differs from 200 rollouts")
    writer_states = [row for row in states if not bool(row.get("is_prewrite"))]
    prewrite_states = [row for row in states if bool(row.get("is_prewrite"))]
    if len(prewrite_states) != 200:
        raise ValueError("MIC_NO_GO: T25 critic prewrite coverage differs from trajectories")
    per_trajectory_turns = Counter(str(row["trajectory_id"]) for row in writer_states)
    if set(per_trajectory_turns) != set(trajectory_ids) \
            or any(count < 1 or count > 8 for count in per_trajectory_turns.values()):
        raise ValueError("MIC_NO_GO: writer-turn history coverage is invalid")

    fold_solves = 0
    for row in deliveries:
        checkpoint = CriticCheckpoint.read(
            row["critic_checkpoint"], expected_actor_commit=training_commit
        )
        if checkpoint["checkpoint_sha256"] != row["critic_checkpoint_sha256"]:
            raise ValueError("MIC_NO_GO: critic checkpoint SHA binding differs")
        fold_solves += len(checkpoint.get("critic_payload", {}).get("oof", {}).get(
            "receipts", []
        ))

    writer_tokens = sum(int(row["delivery"]["writer_active_tokens"]) for row in deliveries)
    answer_tokens = sum(int(row["delivery"]["answer_active_tokens"]) for row in deliveries)
    train_vs_s128_root = _intersection(train_rows, s128_rows, "content_root_sha256")
    train_vs_s128_example = _intersection(train_rows, s128_rows, "content_example_sha256")
    critic_vs_s128_root = _intersection(critic_fit_rows, s128_rows, "content_root_sha256")
    critic_vs_s128_example = _intersection(critic_fit_rows, s128_rows, "content_example_sha256")
    actor_consumed_vs_s128_root = dict(critic_vs_s128_root)
    actor_consumed_vs_s128_example = dict(critic_vs_s128_example)
    direct_overlap_count = max(
        actor_consumed_vs_s128_root["intersection_count"],
        actor_consumed_vs_s128_example["intersection_count"],
        critic_vs_s128_root["intersection_count"],
        critic_vs_s128_example["intersection_count"],
    )

    disclosure = _read_json(disclosure_path)
    if disclosure.get("schema") != "memagent.mic.adaptive-use-disclosure.v1" \
            or disclosure.get("s128_rows_exposed") != 128 \
            or disclosure.get("classification") != "ADAPTIVE_DEVELOPMENT_BENCHMARK":
        raise ValueError("MIC_NO_GO: adaptive-use disclosure is absent or incomplete")
    selection_overlap = {
        "automated_early_stopping_or_checkpoint_selection_rows": 0,
        "human_adaptive_exposure_intersection_count": 128,
        "interpretation": (
            "No validation-driven optimizer stop/checkpoint selection occurred, but all S128 "
            "anchor aggregates were inspected; S128 is development evidence, not a blind final test."
        ),
    }

    health_files = []
    health_values = []
    for step, path_text in zip(ANCHORS, args.health_audits):
        path = Path(path_text).resolve()
        value = _read_json(path)
        if value.get("status") != "PASS" or value.get("decision") != f"MIC_T{step}_AUDIT_PASS":
            raise ValueError(f"MIC_NO_GO: T{step} training health audit is not PASS")
        health_files.append({"step": step, "path": str(path), "sha256": sha256_file(path)})
        health_values.append(value)
    paper_review_path = Path(args.paper_review).resolve()
    replayed_t25 = replay_mic_training_audit(argparse.Namespace(
        p0=str(p0_path), e0=str(e0_path), paper_review=str(paper_review_path),
        ledger=str(ledger_path), weight_ledger=str(weight_ledger_path),
        target_step=25, output=None,
    ))
    recorded_t25 = dict(health_values[-1])
    if p0.get("requires_training_checkpoint_inventory") is not True \
            and "checkpoint_inventory_steps" not in recorded_t25:
        # Explicit migration for the one legacy runtimefix3 health schema.  New
        # P0 runs require the field and cannot take this compatibility branch.
        recorded_t25["checkpoint_inventory_steps"] = []
    if canonical_json(replayed_t25) != canonical_json(recorded_t25):
        raise ValueError("MIC_NO_GO: T25 health certificate differs from full ledger/weight replay")

    result = {
        "schema": SCHEMA,
        "status": "FAIL" if direct_overlap_count else "PASS",
        "decision": (
            "MIC_DIRECT_DATA_LEAKAGE_NO_GO" if direct_overlap_count
            else "MIC_SCIENTIFIC_EVIDENCE_AUDIT_PASS_WITH_ADAPTIVE_S128"
        ),
        "claim_boundary": {
            "t25_budget_classification": "EARLY_BUDGET_SINGLE_SEED_PILOT",
            "convergence_or_sufficient_training_claim_allowed": False,
            "performance_scope": "paired descriptive same-budget development-benchmark comparison",
            "blind_final_test_claim_allowed": False,
        },
        "training_budget": {
            "optimizer_updates": 25,
            "rollout_groups": 25 * 4,
            "trajectories": len(trajectory_ids),
            "unique_actor_source_examples": len(set(actor_indices)),
            "writer_turns": len(writer_states),
            "writer_active_tokens": writer_tokens,
            "answer_active_tokens": answer_tokens,
            "ppo_epochs_per_update": 1,
            "global_ppo_minibatches_per_update": 1,
            "global_ppo_minibatches_total": 25,
            "gradient_delivery_updates": len(gradients),
            "critic_optimizer_updates": 0,
            "critic_closed_form_cumulative_refits": len(deliveries),
            "critic_fold_ridge_solves": fold_solves,
            "separate_prior_or_auxiliary_fit_updates": 0,
            "writer_turns_per_trajectory": {
                "minimum": min(per_trajectory_turns.values()),
                "maximum": max(per_trajectory_turns.values()),
                "mean": sum(per_trajectory_turns.values()) / len(per_trajectory_turns),
            },
        },
        "data_usage": {
            "actor_training": {
                "path": str(train_path), "sha256": sha256_file(train_path),
                "rows": len(train_rows),
                "used_semantic_index_inventory_sha256": canonical_sha256(sorted(actor_indices)),
            },
            "critic_fit": {
                "source": "exact actor on-policy trajectory subset; cumulative OOF ridge",
                "unique_source_rows": len(critic_fit_rows),
                "trajectory_count": len(outcomes),
                "critic_checkpoint": str(critic_path),
                "critic_checkpoint_file_sha256": sha256_file(critic_path),
                "critic_checkpoint_payload_sha256": critic["checkpoint_sha256"],
            },
            "prior_or_auxiliary_training": None,
            "hyperparameter_selection": {
                "source": "frozen manifest plus E0 synthetic toy; no in-run S128 metric selection",
                "e0_path": str(e0_path),
                "e0_sha256": sha256_file(e0_path),
                "pre_run_human_adaptive_history": "PENDING_EXTERNAL_LAB_NOTE",
            },
            "early_stopping": {
                "enabled": False,
                "validation_before_training": False,
                "validation_frequency": -1,
                "fixed_total_updates": 25,
            },
            "fixed_s128": {
                "path": str(s128_path), "sha256": sha256_file(s128_path),
                "resolved_manifest": str(resolved_path),
                "resolved_manifest_sha256": sha256_file(resolved_path),
                "eval_manifest_hash": resolved["eval_manifest_hash"], "rows": 128,
            },
            "accepted_original_protocol": {
                "path": str(original_resolved_path),
                "sha256": sha256_file(original_resolved_path),
            },
            "actor_cursor_authority": {
                "original_p0_path": str(original_p0_path),
                "original_p0_sha256": sha256_file(original_p0_path),
                "semantic_cursor_inventory_sha256": canonical_sha256(expected_cursor),
            },
        },
        "content_overlap": {
            "train_intersection_s128_root": train_vs_s128_root,
            "train_intersection_s128_example": train_vs_s128_example,
            "actor_consumed_intersection_s128_root": actor_consumed_vs_s128_root,
            "actor_consumed_intersection_s128_example": actor_consumed_vs_s128_example,
            "critic_fit_intersection_s128_root": critic_vs_s128_root,
            "critic_fit_intersection_s128_example": critic_vs_s128_example,
            "direct_leakage_gate_scope": (
                "actual 100 consumed actor groups / 200 critic-fit trajectories; full 32k "
                "source-corpus overlap is reported separately as pool risk"
            ),
            "selection_intersection_s128": selection_overlap,
        },
        "adaptive_use": {
            "classification": "S128_DOWNGRADED_TO_DEVELOPMENT_BENCHMARK",
            "disclosure_path": str(disclosure_path),
            "disclosure_sha256": sha256_file(disclosure_path),
            "next_test_required": "sealed untouched held-out set, one-shot T25 reveal",
        },
        "evidence_files": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "curve_authority": {"path": str(curve_authority_path.resolve()),
                                "sha256": sha256_file(curve_authority_path)},
            "training_p0": {"path": str(p0_path), "sha256": sha256_file(p0_path),
                            "git_commit": training_commit},
            "e0": {"path": str(e0_path), "sha256": sha256_file(e0_path)},
            "mic_ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
            "weight_sync_ledger": {"path": str(weight_ledger_path),
                                   "sha256": sha256_file(weight_ledger_path)},
            "paper_review": {"path": str(paper_review_path),
                             "sha256": sha256_file(paper_review_path)},
            "rollout_seed_audit": {"path": str(seed_path), "sha256": sha256_file(seed_path)},
            "training_source": {
                "git_commit": training_commit,
                "trainer_config_path": "verl/trainer/config/ppo_trainer.yaml",
                "trainer_config_blob_sha256": trainer_config_sha,
                "launcher_path": "experiments/7b_gate_a/run_gate_a.sh",
                "launcher_blob_sha256": launcher_sha,
            },
            "health_audits": health_files,
        },
        "remaining_blockers": [
            "E1 mechanism feasibility has not been certified on the completed on-policy bundle.",
            "Single seed and 25 optimizer updates do not establish convergence or robustness.",
            "S128 is adaptively exposed and cannot serve as a blind final test.",
            "An independent sealed held-out set and multi-seed confirmation remain required.",
            "Pre-run human hyperparameter provenance remains PENDING_EXTERNAL_LAB_NOTE.",
        ],
    }
    for value in (
            result["training_budget"]["writer_turns_per_trajectory"]["mean"],):
        if not math.isfinite(float(value)):
            raise ValueError("MIC_NO_GO: non-finite scientific budget statistic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--p0", required=True)
    parser.add_argument("--e0", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--s128-parquet", required=True)
    parser.add_argument("--s128-resolved", required=True)
    parser.add_argument("--mic-ledger", required=True)
    parser.add_argument("--weight-ledger", required=True)
    parser.add_argument("--rollout-seed-audit", required=True)
    parser.add_argument("--critic-checkpoint", required=True)
    parser.add_argument("--health-audits", nargs=5, required=True)
    parser.add_argument("--paper-review", required=True)
    parser.add_argument("--adaptive-disclosure", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = audit(args)
        print(canonical_json({
            "status": report["status"], "decision": report["decision"],
            "training_budget": report["training_budget"],
            "content_overlap": report["content_overlap"],
            "output": str(Path(args.output).resolve()),
        }))
        if report["status"] != "PASS":
            raise SystemExit(report["decision"])
    except Exception as exc:
        raise SystemExit(f"MIC_NO_GO:{type(exc).__name__}:{exc}") from exc


if __name__ == "__main__":
    main()
