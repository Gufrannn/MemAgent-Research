#!/usr/bin/env python3
"""Independently join and score one sealed RWWPO-2 confirmation opening."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.gate_a_execution import checkpoint_inventory
from recurrent.research.rwwpo2_confirmation import sha256_file
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_hotpot_metrics
from recurrent.research.stable_eval_identity import (
    MANIFEST_ROW_FIELDS, OUTPUT_IDENTITY_FIELDS, canonical_sha256,
    evaluation_trajectory_seed, sha256_text, stable_key, stable_trajectory_id,
    validate_actor_only_checkpoint_acknowledgements,
    validate_attempt_identity_rows, validate_resolved_manifest,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds
from tools.h20.preflight_qwen25_7b_stable_i4x2 import _load_parquet_rows
from tools.h20.preflight_rwwpo2_confirmation import expected_configuration


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def mapping(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("source mapping is malformed")
    return value


def question_text(prompt) -> str:
    prompt = json.loads(prompt) if isinstance(prompt, str) else prompt
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("source prompt is malformed")
    first = mapping(prompt[0])
    if first.get("role") != "user" or not isinstance(first.get("content"), str):
        raise ValueError("source question is malformed")
    return first["content"]


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "eval-root", "resolved-manifest", "resolved-manifest-sha256",
        "validation", "checkpoint", "model", "interface-id", "attempt-id",
        "expected-commit", "metric-rows-output", "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:checkout")
    if Path(args.resolved_manifest).is_symlink() \
            or Path(args.validation).is_symlink() \
            or Path(args.checkpoint).is_symlink() \
            or Path(args.eval_root).is_symlink():
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:source symlink")
    resolved_path = Path(args.resolved_manifest).resolve()
    if sha256_file(resolved_path) != args.resolved_manifest_sha256:
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:resolved SHA")
    resolved = validate_resolved_manifest(json.loads(resolved_path.read_text()))
    binding = resolved.get("execution_binding", {})
    if binding.get("interface_id") != args.interface_id:
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:interface binding")
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint_inventory(checkpoint) != binding.get("target_checkpoint_inventory"):
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:checkpoint inventory")
    try:
        _, runtime_sha, protocol_sha, protocol = expected_configuration(args, resolved)
    except (KeyError, ValueError) as error:
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:" + str(error)) from error
    trainer = binding.get("trainer_configuration", {})
    if trainer.get("resolved_runtime_config_sha256") != runtime_sha \
            or trainer.get("generation_protocol_sha256") != protocol_sha \
            or trainer.get("generation_protocol") != protocol:
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:runtime protocol")

    root = Path(args.eval_root).resolve()
    terminal_path = root / "terminal/400.jsonl"
    turns_path = root / "trajectory_turns.jsonl"
    summary_path = root / "execution_summary.json"
    log_path = root / "run.log"
    if any(not path.is_file() or path.is_symlink()
           for path in (terminal_path, turns_path, summary_path, log_path)):
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:missing/symlink artifact")
    frozen_rows = resolved["identity_payload"]["rows"]
    examples = len(frozen_rows)
    terminal = read_jsonl(terminal_path)
    try:
        validate_attempt_identity_rows(terminal, examples=examples, replicas=1)
    except Exception as error:
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:terminal identity:" + str(error)) from error
    if [int(row.get("source_order_index", -1)) for row in terminal] != list(range(examples)):
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:terminal order")
    frozen_by_key = {
        (int(row["source_order_index"]), str(row["example_id"])): row
        for row in frozen_rows
    }
    raw_rows = _load_parquet_rows(Path(args.validation).resolve())
    truth = {}
    for frozen in frozen_rows:
        order = int(frozen["source_order_index"])
        raw = raw_rows[int(frozen["raw_row_position"])]
        reward = mapping(raw["reward_model"])
        ground_truth = reward.get("ground_truth")
        if sha256_text(question_text(raw["prompt"])) != frozen["source_question_hash"] \
                or sha256_text(str(raw["context"])) != frozen["source_context_hash"] \
                or canonical_sha256(ground_truth) != frozen["ground_truth_hash"]:
            raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:parquet stable join")
        truth[order] = ground_truth
    metric_rows = []
    terminal_by_key = {}
    base_seed = int(resolved["confirmation_binding"]["generation_seed"])
    for row in terminal:
        order = int(row["source_order_index"])
        key = (order, str(row["example_id"]))
        frozen = frozen_by_key.get(key)
        if frozen is None or any(row.get(field) != frozen[field]
                                 for field in MANIFEST_ROW_FIELDS):
            raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:frozen row identity")
        seed = evaluation_trajectory_seed(
            base_seed=base_seed, eval_manifest_hash=resolved["eval_manifest_hash"],
            example_id=str(row["example_id"]), source_order_index=order, replica_id=0,
        )
        if row.get("interface_id") != args.interface_id \
                or row.get("attempt_id") != args.attempt_id \
                or row.get("eval_manifest_hash") != resolved["eval_manifest_hash"] \
                or int(row.get("step", -1)) != 400 \
                or int(row.get("trajectory_seed", -1)) != seed \
                or row.get("trajectory_id") != stable_trajectory_id(
                    eval_manifest_hash=resolved["eval_manifest_hash"],
                    example_id=str(row["example_id"]), replica_id=0,
                    trajectory_seed=seed,
                ) or not isinstance(row.get("output"), str):
            raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:terminal execution identity")
        scored = score_terminal_output(row["output"], truth[order])
        metric_rows.append({
            "stable_key": json.dumps(stable_key(row), separators=(",", ":")),
            "source_order_index": order,
            "example_id": str(row["example_id"]),
            **{name: scored[name] for name in (
                "exact_match", "token_f1", "precision", "recall",
                "format_success", "sub_exact_match", "extraction_route",
            )},
        })
        terminal_by_key[stable_key(row)] = row
    aggregates = summarize_hotpot_metrics(metric_rows, expected_denominator=examples)

    turns = read_jsonl(turns_path)
    turns_by_key = defaultdict(list)
    workers = set()
    seen = set()
    for turn in turns:
        missing = set(OUTPUT_IDENTITY_FIELDS) - set(turn)
        key = stable_key(turn) if not missing else None
        terminal_row = terminal_by_key.get(key)
        turn_id = (key, int(turn.get("trajectory_turn", -1)))
        if missing or terminal_row is None or turn_id in seen \
                or any(turn.get(field) != terminal_row.get(field)
                       for field in OUTPUT_IDENTITY_FIELDS):
            raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:turn identity")
        seen.add(turn_id)
        expected_seed = derive_turn_request_seeds(
            [int(turn["trajectory_seed"])], [0], int(turn["trajectory_turn"])
        )[0]
        if any(int(turn.get(field, -1)) != expected_seed for field in (
                "request_seed", "configured_request_seed", "rollout_request_seed")) \
                or turn.get("request_prompt_token_sha256") != turn.get(
                    "returned_prompt_token_sha256"):
            raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:turn seed/prompt binding")
        workers.add(int(turn.get("rollout_worker_rank", -1)))
        turns_by_key[key].append(turn)
    if workers != {0, 1} or set(turns_by_key) != set(terminal_by_key):
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:turn coverage")
    chunk_size = int(protocol["recurrent"]["memory"]["config"]["chunk_size"])
    shared_final = max(
        math.ceil(int(row["context_token_count"]) / chunk_size) for row in frozen_rows
    )
    for key, rows in turns_by_key.items():
        order = int(terminal_by_key[key]["source_order_index"])
        active_count = math.ceil(int(frozen_rows[order]["context_token_count"]) / chunk_size)
        active = sorted(int(row["trajectory_turn"]) for row in rows if not row["is_final"])
        final = [row for row in rows if row["is_final"]]
        if active != list(range(active_count)) or len(final) != 1 \
                or int(final[0]["trajectory_turn"]) != shared_final \
                or final[0].get("response_token_sha256") != terminal_by_key[key].get(
                    "terminal_response_token_sha256"):
            raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:turn schedule")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frozen_artifact = binding["model_artifacts"][args.interface_id]
    try:
        validate_actor_only_checkpoint_acknowledgements(
            summary.get("actor_checkpoint_load_acks", []),
            frozen_artifact["actor_model_shards"],
            global_step_folder=checkpoint, world_size=2,
        )
    except Exception as error:
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:checkpoint acks:" + str(error)) from error
    expected_summary = {
        "interface_id": args.interface_id, "attempt_id": args.attempt_id,
        "eval_manifest_hash": resolved["eval_manifest_hash"], "global_step": 400,
        "actor_update_calls": 0, "optimizer_step_calls": 0,
        "checkpoint_save_calls": 0, "resume_mode": "actor_only_eval",
        "weight_source": "actor_checkpoint", "checkpoint_load_mode": "actor_only",
        "checkpoint_source": str(checkpoint), "validation_only": True,
        "resolved_runtime_config_sha256": runtime_sha,
        "hydra_pre_dataset_max_prompt_length": protocol["data"][
            "hydra_pre_dataset_max_prompt_length"
        ],
        "memory_dataset_effective_max_prompt_length": protocol["data"][
            "memory_dataset_effective_max_prompt_length"
        ],
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:execution summary")
    before, after = summary.get("weight_snapshot_before"), summary.get("weight_snapshot_after")
    stable_fields = (
        "actor_master_sampled_tensor_digest", "actor_rollout_sampled_tensor_digest",
        "vllm_sampled_tensor_digest", "worker_ranks", "worker_evidence",
    )
    if not isinstance(before, dict) or not isinstance(after, dict) \
            or any(before.get(field) != after.get(field) for field in stable_fields) \
            or before.get("actor_rollout_sampled_tensor_digest") != before.get(
                "vllm_sampled_tensor_digest"):
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:weight mutation/sync")

    metric_output = Path(args.metric_rows_output)
    report_output = Path(args.output)
    if metric_output.exists() or report_output.exists() \
            or metric_output.is_symlink() or report_output.is_symlink():
        raise SystemExit("RWWPO2_CONFIRM_AUDIT_NO_GO:append-only audit output")
    metric_output.parent.mkdir(parents=True, exist_ok=True)
    with metric_output.open("x", encoding="utf-8") as stream:
        for row in metric_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "schema_version": "rwwpo2-confirmation-eval-audit-v1",
        "status": "PASS", "decision": "RWWPO2_CONFIRMATION_EVAL_PASS",
        "git_commit": head, "interface_id": args.interface_id,
        "cell": binding["cell"], "experiment_seed": binding["experiment_seed"],
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "resolved_manifest_path": str(resolved_path),
        "resolved_manifest_sha256": args.resolved_manifest_sha256,
        "confirmation_data_sha256": resolved["confirmation_binding"][
            "confirmation_data_sha256"],
        "generation_protocol_sha256": protocol_sha,
        "checkpoint_inventory_sha256": binding["target_checkpoint_inventory_sha256"],
        "training_attempt_audit_sha256": binding["training_attempt_audit_sha256"],
        "eval_root": str(root),
        "checkpoint_path": str(checkpoint),
        "terminal_sha256": sha256_file(terminal_path),
        "turn_ledger_sha256": sha256_file(turns_path),
        "execution_summary_sha256": sha256_file(summary_path),
        "run_log_sha256": sha256_file(log_path),
        "metric_rows_path": str(metric_output.resolve()),
        "metric_rows_sha256": sha256_file(metric_output),
        "metric_rows_canonical_sha256": canonical_sha256(metric_rows),
        "stable_key_inventory_sha256": canonical_sha256(
            [row["stable_key"] for row in metric_rows]
        ),
        "metrics": aggregates,
        "primary_metric": "macro_token_f1",
        "opened_confirmation": True,
        "training_or_selection_mutation": False,
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    report_output.parent.mkdir(parents=True, exist_ok=True)
    with report_output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "interface_id": args.interface_id, "metrics": aggregates,
        "output": str(report_output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
