#!/usr/bin/env python3
"""Read-only budget, overlap, and adaptive-use audit for TF-RWWPO.

This audit deliberately distinguishes three questions that are easy to blur:

* which source examples can be consumed by the actor;
* whether any of those examples share content with fixed S128; and
* whether S128 was used adaptively even when it never entered an optimizer.

No answer text is emitted.  Only counts and hashes leave this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.stable_eval_identity import canonical_sha256, sha256_text


EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"
EXPECTED_TRAIN_SHA256 = "798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8"
EXPECTED_S128_DATA_SHA256 = "54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6"
EXPECTED_S128_RESOLVED_SHA256 = "6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411"
EXPECTED_S128_MANIFEST_HASH = "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a"
EXPECTED_S128_SIZE = 128
ADAPTIVE_USE_EVIDENCE = "docs/papers/tf_rwwpo_revision_20260822.md"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return dict(value)


def _prompt(value: object) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError("prompt must be a non-empty chat-message list")
    return [_mapping(item, "prompt item") for item in value]


def source_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return outcome-free identifiers plus hashes; never return source strings."""
    prompt = _prompt(row.get("prompt"))
    first = prompt[0]
    if first.get("role") != "user" or not isinstance(first.get("content"), str):
        raise ValueError("prompt[0] must be a user message with string content")
    context = row.get("context")
    if not isinstance(context, str):
        raise TypeError("context must be text")
    reward = _mapping(row.get("reward_model"), "reward_model")
    if "ground_truth" not in reward:
        raise ValueError("reward_model.ground_truth is missing")
    extra = _mapping(row.get("extra_info"), "extra_info")
    question_hash = sha256_text(str(first["content"]))
    context_hash = sha256_text(context)
    ground_truth_hash = canonical_sha256(reward["ground_truth"])
    root_key = canonical_sha256([question_hash, context_hash])
    content_key = canonical_sha256([question_hash, context_hash, ground_truth_hash])
    return {
        "semantic_dataset_index": int(extra["index"]),
        "question_hash": question_hash,
        "context_hash": context_hash,
        "ground_truth_hash": ground_truth_hash,
        "root_key": root_key,
        "content_key": content_key,
        "prompt": prompt,
    }


def s128_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    question_hash = str(row["source_question_hash"])
    context_hash = str(row["source_context_hash"])
    ground_truth_hash = str(row["ground_truth_hash"])
    for value in (question_hash, context_hash, ground_truth_hash):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("malformed S128 content hash")
    return {
        "semantic_dataset_index": int(row["semantic_dataset_index"]),
        "root_key": canonical_sha256([question_hash, context_hash]),
        "content_key": canonical_sha256([question_hash, context_hash, ground_truth_hash]),
    }


def overlap_counts(
    actor_rows: Iterable[Mapping[str, Any]], s128_rows: Iterable[Mapping[str, Any]]
) -> dict[str, int]:
    actor = list(actor_rows)
    s128 = list(s128_rows)
    actor_content = {str(row["content_key"]) for row in actor}
    actor_roots = {str(row["root_key"]) for row in actor}
    actor_semantic = {int(row["semantic_dataset_index"]) for row in actor}
    s128_content = {str(row["content_key"]) for row in s128}
    s128_roots = {str(row["root_key"]) for row in s128}
    s128_semantic = {int(row["semantic_dataset_index"]) for row in s128}
    return {
        "actor_unique_content_keys": len(actor_content),
        "actor_unique_root_keys": len(actor_roots),
        "s128_unique_content_keys": len(s128_content),
        "s128_unique_root_keys": len(s128_roots),
        "train_intersect_s128_content": len(actor_content & s128_content),
        "train_intersect_s128_root": len(actor_roots & s128_roots),
        # Semantic indices are dataset-local in the source parquet.  Retain this
        # only as a diagnostic; content/root hashes decide leakage.
        "train_intersect_s128_dataset_local_semantic_id_diagnostic": len(
            actor_semantic & s128_semantic
        ),
    }


def _read_parquet(path: Path) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(
        batch_size=256,
        columns=["prompt", "context", "reward_model", "extra_info"],
    ):
        yield from batch.to_pylist()


def _actor_consumed_rows(
    rows: Iterable[Mapping[str, Any]], *, tokenizer_root: Path, target_prompt_groups: int,
    effective_max_prompt_length: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    consumed: list[dict[str, Any]] = []
    source_seen = 0
    filtered = 0
    for row in rows:
        source_seen += 1
        identity = source_identity(row)
        prompt_length = len(
            tokenizer.apply_chat_template(identity.pop("prompt"), add_generation_prompt=True)
        )
        if prompt_length > effective_max_prompt_length:
            filtered += 1
            continue
        consumed.append(identity)
        if len(consumed) == target_prompt_groups:
            break
    if len(consumed) != target_prompt_groups:
        raise ValueError(
            f"only {len(consumed)} eligible actor rows found; expected {target_prompt_groups}"
        )
    return consumed, {
        "source_rows_scanned_until_budget_closed": source_seen,
        "overlong_rows_skipped_before_budget_closed": filtered,
        "consumed_prompt_groups": len(consumed),
    }


def _load_s128(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = json.loads(path.read_text(encoding="utf-8"))
    if resolved.get("eval_manifest_hash") != EXPECTED_S128_MANIFEST_HASH:
        raise ValueError("S128 eval_manifest_hash drift")
    payload = resolved.get("identity_payload")
    if not isinstance(payload, Mapping) or canonical_sha256(payload) != EXPECTED_S128_MANIFEST_HASH:
        raise ValueError("S128 identity payload self-hash mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_S128_SIZE:
        raise ValueError("S128 resolved manifest is not exactly 128 rows")
    identities = [s128_identity(row) for row in rows]
    if len({row["content_key"] for row in identities}) != EXPECTED_S128_SIZE:
        raise ValueError("S128 contains duplicate canonical content keys")
    return resolved, identities


def _load_actual_budget(
    ledger_dir: Path, seed_audit_path: Path, execution_ledger_path: Path, *, expected_commit: str
) -> dict[str, Any]:
    from tools.h20.audit_rwwpo_actual_loss import audit
    from recurrent.research.gate_a_execution import validate_jsonl_chain

    ledgers = sorted(ledger_dir.glob("actual_loss_rank*.jsonl"))
    summary = audit(ledgers, require_method=True)
    execution_rows = [
        json.loads(line)
        for line in execution_ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = validate_jsonl_chain(execution_rows)
    if failures or not execution_rows \
            or any(row.get("git_commit") != expected_commit for row in execution_rows):
        raise ValueError("execution ledger chain/commit mismatch")
    seeds = [
        json.loads(line)
        for line in seed_audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    turns = [row for row in seeds if row.get("record_type") == "trajectory_turn_seed"]
    groups = {
        (int(row["global_step"]), str(row["stable_example_id"])) for row in turns
    }
    trajectories = {
        (int(row["global_step"]), str(row["trajectory_id"])) for row in turns
    }
    recurrent_rows = {
        (int(row["global_step"]), str(row["trajectory_id"]), int(row["turn"]))
        for row in turns
    }
    writer_turns: set[tuple[int, int, int]] = set()
    writer_tokens = answer_tokens = response_tokens = 0
    transaction_keys: set[tuple[str, int, int, int]] = set()
    steps: set[int] = set()
    epochs: set[int] = set()
    minibatches: set[int] = set()
    # Stream a second time after the independent validator returns.  Actual-loss
    # rows contain token matrices, so retaining the full T25 ledger in memory is
    # unnecessary and can itself make a read-only audit operationally unsafe.
    for path in ledgers:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                step = int(row["global_step"])
                epoch = int(row["epoch"])
                minibatch = int(row["minibatch"])
                steps.add(step)
                epochs.add(epoch)
                minibatches.add(minibatch)
                transaction_keys.add((str(row["attempt_id"]), step, epoch, minibatch))
                for trajectory_hash, turn, writer_mask, answer_mask, response_mask in zip(
                    row["trajectory_identity_hash"], row["trajectory_turn"], row["writer_mask"],
                    row["answer_mask"], row["response_mask"],
                ):
                    writer_count = sum(bool(value) for value in writer_mask)
                    answer_count = sum(bool(value) for value in answer_mask)
                    response_count = sum(bool(value) for value in response_mask)
                    writer_tokens += writer_count
                    answer_tokens += answer_count
                    response_tokens += response_count
                    if writer_count:
                        writer_turns.add((step, int(trajectory_hash), int(turn)))
    return {
        "status": "PASS",
        "actual_loss_audit_decision": summary["decision"],
        "completed_global_steps": sorted(steps),
        "rollout_groups": len(groups),
        "trajectories": len(trajectories),
        "recurrent_trajectory_turn_rows": len(recurrent_rows),
        "writer_turns": len(writer_turns),
        "writer_tokens": writer_tokens,
        "final_answer_tokens": answer_tokens,
        "active_response_tokens": response_tokens,
        "global_optimizer_proposals": len(transaction_keys),
        "nonzero_committed_updates": int(summary["nonzero_commit_count"]),
        "ppo_epoch_indices": sorted(epochs),
        "optimizer_minibatch_indices": sorted(minibatches),
        "critic_optimizer_updates": 0,
        "auxiliary_fit_updates": 0,
        "actual_loss_ledger_sha256": {path.name: sha256_file(path) for path in ledgers},
        "rollout_seed_audit_sha256": sha256_file(seed_audit_path),
        "execution_ledger_sha256": sha256_file(execution_ledger_path),
    }


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--tokenizer-root", required=True)
    parser.add_argument("--s128-data", required=True)
    parser.add_argument("--s128-resolved", required=True)
    parser.add_argument("--s128-resolved-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--actual-ledger-dir")
    parser.add_argument("--rollout-seed-audit")
    parser.add_argument("--execution-ledger")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if _git("rev-parse", "HEAD") != args.expected_commit or _git("status", "--porcelain"):
        raise SystemExit("TF_RWWPO_BUDGET_LEAKAGE_NO_GO:checkout")
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise SystemExit("TF_RWWPO_BUDGET_LEAKAGE_NO_GO:branch")
    manifest_path = Path(args.manifest).resolve()
    train_path = Path(args.train).resolve()
    tokenizer_root = Path(args.tokenizer_root).resolve()
    s128_data_path = Path(args.s128_data).resolve()
    s128_path = Path(args.s128_resolved).resolve()
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA256:
        raise SystemExit("TF_RWWPO_BUDGET_LEAKAGE_NO_GO:train SHA")
    if sha256_file(s128_data_path) != EXPECTED_S128_DATA_SHA256:
        raise SystemExit("TF_RWWPO_BUDGET_LEAKAGE_NO_GO:S128 data SHA")
    if args.s128_resolved_sha256 != EXPECTED_S128_RESOLVED_SHA256 \
            or sha256_file(s128_path) != EXPECTED_S128_RESOLVED_SHA256:
        raise SystemExit("TF_RWWPO_BUDGET_LEAKAGE_NO_GO:S128 resolved SHA")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training = manifest["training"]
    if manifest["data"]["train_sha256"] != EXPECTED_TRAIN_SHA256:
        raise SystemExit("TF_RWWPO_BUDGET_LEAKAGE_NO_GO:manifest train identity")
    steps = max(int(value) for value in training["target_steps"])
    batch_size = int(training["train_batch_size"])
    rollout_n = int(training["rollout_n"])
    mini_batch_size = int(training["ppo_mini_batch_size"])
    ppo_epochs = int(training["ppo_epochs"])
    effective_prompt_filter = int(training["max_chunks"]) * int(training["chunk_size"])
    if ppo_epochs != 1 or int(training["critic_optimizer_updates"]) != 0 \
            or int(training["auxiliary_fit_updates"]) != 0:
        raise SystemExit("TF_RWWPO_BUDGET_LEAKAGE_NO_GO:optimizer budget drift")
    if int(training["runtime_effective_prompt_filter_length"]) != effective_prompt_filter:
        raise SystemExit("TF_RWWPO_BUDGET_LEAKAGE_NO_GO:effective prompt filter drift")
    target_prompt_groups = steps * batch_size
    source_rows = _read_parquet(train_path)
    actor_rows, scan = _actor_consumed_rows(
        source_rows,
        tokenizer_root=tokenizer_root,
        target_prompt_groups=target_prompt_groups,
        # MemoryDataset mutates 8192 to max_chunks * chunk_size before the
        # base dataset applies its filter.  This is recorded, not hidden.
        effective_max_prompt_length=effective_prompt_filter,
    )
    _, s128_rows = _load_s128(s128_path)
    overlaps = overlap_counts(actor_rows, s128_rows)
    direct_leakage = (
        overlaps["train_intersect_s128_content"] > 0
        or overlaps["train_intersect_s128_root"] > 0
    )

    actual_flags = [
        bool(args.actual_ledger_dir), bool(args.rollout_seed_audit), bool(args.execution_ledger)
    ]
    actual_requested = any(actual_flags)
    if actual_requested and not all(actual_flags):
        raise SystemExit("TF_RWWPO_BUDGET_LEAKAGE_NO_GO:incomplete actual-budget inputs")
    actual_budget: dict[str, Any]
    if actual_requested:
        actual_budget = _load_actual_budget(
            Path(args.actual_ledger_dir).resolve(),
            Path(args.rollout_seed_audit).resolve(),
            Path(args.execution_ledger).resolve(),
            expected_commit=args.expected_commit,
        )
        expected_steps = list(range(1, steps + 1))
        actual_closure = (
            actual_budget["completed_global_steps"] == expected_steps
            and actual_budget["rollout_groups"] == target_prompt_groups
            and actual_budget["trajectories"] == target_prompt_groups * rollout_n
            and actual_budget["global_optimizer_proposals"]
            == steps * ppo_epochs * (batch_size // mini_batch_size)
            and actual_budget["ppo_epoch_indices"] == [0]
            and actual_budget["optimizer_minibatch_indices"] == [0]
        )
        actual_budget["t25_budget_closure"] = "PASS" if actual_closure else "FAIL"
        if not actual_closure:
            raise SystemExit("TF_RWWPO_BUDGET_LEAKAGE_NO_GO:T25 actual budget closure")
    else:
        actual_budget = {
            "status": "PENDING_TF_RUN",
            "writer_turns": None,
            "writer_tokens": None,
            "final_answer_tokens": None,
            "nonzero_committed_updates": None,
        }

    report: dict[str, Any] = {
        "schema_version": "tf-rwwpo-budget-leakage-v1",
        "status": "NO_GO" if direct_leakage else "PASS",
        "decision": (
            "TF_RWWPO_DIRECT_DATA_LEAKAGE_DETECTED"
            if direct_leakage else "TF_RWWPO_BUDGET_LEAKAGE_AUDIT_PASS"
        ),
        "git_commit": args.expected_commit,
        "scientific_scope": {
            "t25_classification": "EARLY_BUDGET_PILOT_NOT_CONVERGENCE",
            "convergence_claim_authorized": False,
            "sufficient_training_claim_authorized": False,
            "single_seed": 2026,
        },
        "static_budget": {
            "target_global_steps": steps,
            "prompt_groups_per_step": batch_size,
            "rollout_trajectories_per_prompt": rollout_n,
            "maximum_prompt_groups": target_prompt_groups,
            "maximum_trajectories": target_prompt_groups * rollout_n,
            "ppo_epochs_per_step": ppo_epochs,
            "global_optimizer_minibatches_per_epoch": batch_size // mini_batch_size,
            "maximum_global_optimizer_proposals": steps * ppo_epochs * (batch_size // mini_batch_size),
            "critic_optimizer_updates": 0,
            "auxiliary_fit_updates": 0,
            "critic_reason": "GRPO sets use_critic=False",
            "auxiliary_reason": "reward_model.enable=False; dense reward is a fixed scorer, not a fit model",
        },
        "effective_data_contract": {
            "manifest_max_prompt_length": int(training["max_prompt_length"]),
            "memory_dataset_filter_max_prompt_length": effective_prompt_filter,
            "mutation_site": "recurrent/impls/memory.py:62",
            "filter_contract_note": "MemoryDataset mutates the Hydra data config before RLHFDataset filtering",
            "shuffle": False,
            "sampler": "SequentialSampler",
            **scan,
        },
        "data_sources": {
            "actor_train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "critic_fit": {"path": None, "sha256": None, "updates": 0},
            "reference_policy_fit": {"path": None, "sha256": None, "updates": 0},
            "auxiliary_fit": {"path": None, "sha256": None, "updates": 0},
            "hyperparameter_and_controller_selection": {
                "path": str(s128_path),
                "sha256": sha256_file(s128_path),
                "adaptive_use": True,
                "basis": ADAPTIVE_USE_EVIDENCE,
                "threshold_selection_provenance": "PENDING_DOCUMENTARY_EVIDENCE",
            },
            "early_stopping": {
                "dataset": None,
                "performance_based": False,
                "test_freq": -1,
                "val_before_train": False,
                "t5_gate_uses": "on-policy mechanism/health ledger only",
            },
            "fixed_s128": {
                "data_path": str(s128_data_path),
                "data_sha256": sha256_file(s128_data_path),
                "path": str(s128_path),
                "sha256": sha256_file(s128_path),
                "eval_manifest_hash": EXPECTED_S128_MANIFEST_HASH,
                "rows": EXPECTED_S128_SIZE,
                "classification": "ADAPTIVE_DEVELOPMENT_BENCHMARK_NOT_BLIND_FINAL_TEST",
            },
        },
        "intersections": {
            **overlaps,
            "critic_fit_intersect_s128": 0,
            "auxiliary_fit_intersect_s128": 0,
            "selection_intersect_s128": EXPECTED_S128_SIZE,
            "selection_intersection_reason": "all 128 fixed-S128 rows were evaluated before the controller pivot",
            "content_and_root_hashes_are_decisive": True,
        },
        "actual_budget": actual_budget,
        "adaptive_benchmark_risk": {
            "status": "CONFIRMED",
            "s128_results_viewed_at_anchors": [5, 10, 15, 20, 25],
            "controller_form_changed_after_viewing": True,
            "blind_final_test_claim_authorized": False,
        },
        "heldout_confirmation_requirement": {
            "status": "PENDING_PREREGISTRATION_AND_MATERIALIZATION",
            "minimum_conditions": [
                "exclude every actor-train and fixed-S128 root/content key",
                "commit deterministic sample rule and salt before generation",
                "freeze code, controller, thresholds, and anchor rule before access",
                "run once after all development decisions; no tuning or checkpoint selection",
            ],
        },
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "tokenizer_root": str(tokenizer_root),
            "s128_data": str(s128_data_path),
            "s128_resolved": str(s128_path),
            "code_sha256": {
                path: sha256_file(REPO_ROOT / path)
                for path in (
                    "tools/h20/audit_tf_rwwpo_budget_leakage.py",
                    "recurrent/impls/memory.py",
                    "verl/utils/dataset/rl_dataset.py",
                    "verl/trainer/ppo/ray_trainer.py",
                    "verl/workers/actor/dp_actor.py",
                    "experiments/7b_gate_a/run_gate_a.sh",
                    "scripts/h20/run_qwen25_7b_rwwpo.sh",
                )
            },
        },
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(payload).hexdigest()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "decision": report["decision"],
        "output": str(output), "report_sha256": report["report_sha256"],
        "intersections": report["intersections"],
    }, sort_keys=True))
    if direct_leakage:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
