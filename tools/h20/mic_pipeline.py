#!/usr/bin/env python3
"""Fail-closed P0/E0/E1, baseline import, and read-only MIC audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from recurrent.research.mic import (  # noqa: E402
    FORBIDDEN_KEYS, SCHEMA, calibration_report, canonical_json,
    cross_fitted_values, innovation_ledger, sha256_file, sha256_json,
    stable_fold_assignments, validate_admissible_state,
)
from recurrent.research.s128_hotpot_metrics import (  # noqa: E402
    score_terminal_output, summarize_fixed_s128,
)

BRANCH = "h20/qwen25-7b-mic-t25-frozen-20260822"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"MIC_NO_GO: refusing to overwrite {target}")
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def _nested(value: dict[str, Any], dotted: str) -> Any:
    cursor: Any = value
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(dotted)
        cursor = cursor[part]
    return cursor


def _leaf_map(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_leaf_map(value[key], child))
        return result
    return {prefix: value}


def validate_full_resolved_parity(method_manifest: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    parity = inventory.get("resolved_config_parity", {})
    original_path = Path(expand(parity.get("original", "")))
    method_path = Path(expand(parity.get("method", "")))
    if not original_path.is_file() or not method_path.is_file() \
            or sha256_file(original_path) != parity.get("original_sha256") \
            or sha256_file(method_path) != parity.get("method_sha256"):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: resolved config parity evidence")
    original, method = _leaf_map(read_json(original_path)), _leaf_map(read_json(method_path))
    allowed = tuple(method_manifest["only_allowed_scientific_differences"])
    differences = sorted(path for path in set(original) | set(method)
                         if original.get(path) != method.get(path))
    illegal = [path for path in differences if not any(
        path == prefix or path.startswith(prefix + ".") for prefix in allowed
    )]
    if illegal or not any(path.startswith("algorithm.mic") for path in differences):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: illegal resolved config diff "
                         + canonical_json(illegal))
    return {"difference_paths": differences, "difference_sha256": sha256_json(differences)}


def validate_original_parity(method: dict[str, Any], original: dict[str, Any]) -> None:
    translations = {
        "model.id": "training.model_id", "model.revision": "training.model_revision",
        "training.seed": "training.seed", "training.trajectory_seed_mode": "training.trajectory_seed_mode",
        "training.train_batch_size": "training.train_batch_size",
        "training.rollout_n": "training.rollout_n",
        "training.ppo_mini_batch_size": "training.ppo_mini_batch_size",
        "training.chunk_size": "training.chunk_size", "training.max_chunks": "training.max_chunks",
        "training.max_prompt_length": "training.max_prompt_length",
        "training.max_response_length": "training.max_response_length",
        "training.actor_learning_rate": "training.actor_learning_rate",
        "training.actor_lr_warmup_steps": "training.actor_lr_warmup_steps",
        "training.clip_ratio_high": "training.clip_ratio_high",
        "training.entropy_coefficient": "training.entropy_coefficient",
        "training.kl_loss_coefficient": "training.kl_loss_coefficient",
        "backend.reward_manager": "training.reward_manager",
        "gpu.world_size": "training.world_size", "gpu.fsdp_size": "training.fsdp_size",
        "gpu.tensor_parallel_size": "training.tensor_parallel_size",
    }
    failures = []
    for original_path in method["original_manifest_equal_paths"]:
        method_path = translations[original_path]
        try:
            old, new = _nested(original, original_path), _nested(method, method_path)
        except KeyError:
            failures.append(f"missing parity field {original_path}/{method_path}")
            continue
        if old != new:
            failures.append(f"{original_path}: Original={old!r}, MIC={new!r}")
    if failures:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: " + "; ".join(failures))


def parse_gpu_pair(value: str) -> tuple[int, int]:
    if re.fullmatch(r"[0-9]+,[0-9]+", value or "") is None:
        raise ValueError("MIC_NO_GO: GPU pair must be explicit N,M")
    pair = tuple(int(item) for item in value.split(","))
    if pair[0] >= pair[1]:
        raise ValueError("MIC_NO_GO: GPU pair must be distinct canonical ascending")
    return pair


def p0(args: argparse.Namespace) -> dict[str, Any]:
    manifest = expand(read_json(args.manifest))
    missing = [key for key in manifest["runtime"]["required_environment"] if not os.environ.get(key)]
    if missing:
        raise ValueError(f"MIC_NO_GO: required environment missing {missing}")
    expected = os.environ["MEMAGENT_MIC_EXPECTED_COMMIT"]
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None or git("rev-parse", "HEAD") != expected:
        raise ValueError("MIC_NO_GO: exact Git commit mismatch")
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("MIC_NO_GO: wrong MIC branch")
    if git("status", "--porcelain"):
        raise ValueError("MIC_NO_GO: dirty worktree")
    if Path(os.environ["MEMAGENT_MIC_REPO_DIR"]).resolve() != REPO:
        raise ValueError("MIC_NO_GO: invoked checkout differs from declared repository")
    pair = parse_gpu_pair(os.environ["MEMAGENT_MIC_GPU_PAIR"])
    run_id = os.environ["MEMAGENT_MIC_RUN_ID"]
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{7,63}", run_id) is None:
        raise ValueError("MIC_NO_GO: run ID must be explicit stable text, not UUID")
    original_path = Path(os.environ["MEMAGENT_MIC_ORIGINAL_RESOLVED_MANIFEST"])
    curve_path = Path(os.environ["MEMAGENT_MIC_ORIGINAL_CURVE_REPORT"])
    if not original_path.is_file() or not curve_path.is_file():
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: certified protocol/curve is absent")
    original = read_json(original_path)
    validate_original_parity(manifest, original)
    curve = read_json(curve_path)
    if curve.get("status") != "PASS" or curve.get("decision") != "ORIGINAL_S128_CURVE_PASS":
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: certified curve is not PASS")
    certified = manifest["certified_read_only_sources"]
    expected_curve = Path(certified["original_s128_curve"]["final_report"])
    if curve_path.resolve() != expected_curve.resolve():
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: unexpected curve authority path")
    expected_original = Path(certified["original_t25_training"]["resolved"])
    if original_path.resolve() != expected_original.resolve():
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: unexpected Original protocol path")
    training_report = Path(certified["original_t25_training"]["final_report"])
    if not training_report.is_file() or sha256_file(training_report) != \
            certified["original_t25_training"]["final_report_sha256"]:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: Original training authority")
    evidence = {
        "schema": SCHEMA, "status": "PASS", "decision": "MIC_P0_PASS",
        "git_commit": expected, "branch": BRANCH, "run_id": run_id,
        "gpu_pair": list(pair), "manifest_sha256": sha256_file(args.manifest),
        "original_resolved_manifest": str(original_path.resolve()),
        "original_resolved_manifest_sha256": sha256_file(original_path),
        "original_curve_report": str(curve_path.resolve()),
        "original_curve_report_sha256": sha256_file(curve_path),
        "baseline_inventory_deferred_to_evaluation": True,
        "original_internal_trajectory_required": False,
    }
    if args.check_runtime:
        result = subprocess.run([
            "nvidia-smi", "-i", os.environ["MEMAGENT_MIC_GPU_PAIR"],
            "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, check=True)
        if result.stdout.strip():
            raise ValueError("MIC_NO_GO: declared GPUs are occupied; no process was changed")
    write_new(args.output, evidence)
    return evidence


def e0(args: argparse.Namespace) -> dict[str, Any]:
    # Eight roots, two terminal branches. State text perfectly identifies the
    # analytic conditional value while turn/length alone cannot.
    states, outcomes = [], {}
    oracle_rows = []
    for index in range(8):
        trajectory = f"toy-{index}"
        root = f"root-{index}"
        good = index % 2 == 0
        reward = 1.0 if good else -1.0
        outcomes[trajectory] = reward
        base = {
            "stable_example_id": f"example-{index}", "stable_root_id": root,
            "trajectory_id": trajectory, "question": "retain the key",
        }
        states.append({**base, "turn_index": 0, "visible_chunks": [],
                       "materialized_memory": "", "materialized_memory_history": [],
                       "is_prewrite": True})
        states.append({**base, "turn_index": 1, "visible_chunks": ["key good" if good else "key bad"],
                       "materialized_memory": "good" if good else "bad",
                       "materialized_memory_history": ["good" if good else "bad"],
                       "is_prewrite": False})
        oracle_rows.append({"trajectory_id": trajectory, "v0": 0.0,
                            "values": [reward], "writer_innovations": [reward],
                            "answer_residual": 0.0, "outcome": reward, "fold": index % 4,
                            "stable_root_id": root, "closure_error": 0.0})
    oracle = {"trajectories": oracle_rows}
    max_error = max(abs(row["v0"] + math.fsum(row["writer_innovations"])
                        + row["answer_residual"] - row["outcome"]) for row in oracle_rows)
    if max_error > 1e-12:
        raise ValueError("MIC_NO_GO: oracle closure failed")
    # Adversarial taints must be rejected.
    taint_rejections = []
    for key in sorted(FORBIDDEN_KEYS):
        tainted = dict(states[0]); tainted[key] = "forbidden"
        try:
            validate_admissible_state(tainted)
        except ValueError:
            taint_rejections.append(key)
    if set(taint_rejections) != set(FORBIDDEN_KEYS):
        raise ValueError("MIC_NO_GO: source firewall did not reject every taint")
    folds = stable_fold_assignments([row["stable_root_id"] for row in states], 4)
    fitted = cross_fitted_values(states, outcomes, fold_count=4, alpha=0.01, dimension=64)
    fitted_ledger = innovation_ledger(fitted, outcomes)
    fitted_report = calibration_report(fitted_ledger)
    restricted = cross_fitted_values(
        states, outcomes, fold_count=4, alpha=0.01, dimension=64, restricted=True
    )
    restricted_report = calibration_report(innovation_ledger(restricted, outcomes))
    shuffled_values = [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
    shuffled_outcomes = dict(zip(sorted(outcomes), shuffled_values))
    shuffled = cross_fitted_values(states, shuffled_outcomes, fold_count=4,
                                   alpha=0.01, dimension=64)
    shuffled_report = calibration_report(innovation_ledger(shuffled, shuffled_outcomes))
    if fitted_report["mse"] >= restricted_report["mse"]:
        raise ValueError("MIC_NO_GO: oracle fitted critic does not beat turn/length")
    if fitted_ledger["maximum_closure_error"] > 1e-12:
        raise ValueError("MIC_NO_GO: fitted oracle ledger does not close")
    if shuffled_report["mse"] <= fitted_report["mse"]:
        raise ValueError("MIC_NO_GO: shuffle control retained oracle predictability")
    report = {
        "schema": SCHEMA, "status": "PASS", "decision": "MIC_E0_PASS",
        "oracle_maximum_closure_error": max_error, "oracle_trajectory_count": 8,
        "fold_assignments_sha256": sha256_json(folds),
        "taint_rejections": taint_rejections,
        "oracle_calibration": calibration_report(oracle),
        "fitted_calibration": fitted_report,
        "restricted_calibration": restricted_report,
        "shuffle_calibration": shuffled_report,
        "fitted_oof_bundle_sha256": fitted["bundle_sha256"],
        "fitted_maximum_closure_error": fitted_ledger["maximum_closure_error"],
    }
    write_new(args.output, report)
    return report


def e1(args: argparse.Namespace) -> dict[str, Any]:
    bundle_path = Path(os.environ["MEMAGENT_MIC_E1_BUNDLE"])
    inventory = read_json(os.environ["MEMAGENT_MIC_BASELINE_INVENTORY"])
    frozen = inventory.get("e1_bundle", {})
    if Path(expand(frozen.get("path", ""))).resolve() != bundle_path.resolve() \
            or frozen.get("sha256") != sha256_file(bundle_path) \
            or frozen.get("source_kind") != "frozen_corrected_original_trajectories" \
            or re.fullmatch(r"[0-9a-f]{40}", str(frozen.get("original_commit", ""))) is None:
        raise ValueError("MIC_NO_GO: E1 is not authenticated frozen Original evidence")
    bundle = read_json(bundle_path)
    states, outcomes = bundle.get("states"), bundle.get("outcomes")
    if not isinstance(states, list) or not isinstance(outcomes, dict):
        raise ValueError("MIC_NO_GO: E1 bundle requires states and outcomes")
    full = cross_fitted_values(states, outcomes, fold_count=args.fold_count,
                               alpha=args.alpha, dimension=args.dimension)
    restricted = cross_fitted_values(states, outcomes, fold_count=args.fold_count,
                                     alpha=args.alpha, dimension=args.dimension, restricted=True)
    full_ledger = innovation_ledger(full, outcomes)
    restricted_ledger = innovation_ledger(restricted, outcomes)
    full_report = calibration_report(full_ledger)
    restricted_report = calibration_report(restricted_ledger)
    improvement = ((restricted_report["mse"] - full_report["mse"])
                   / max(restricted_report["mse"], 1e-15))
    failures = []
    if improvement < args.minimum_mse_improvement:
        failures.append(f"OOF MSE improvement {improvement:.6g} below {args.minimum_mse_improvement}")
    if full_report["writer_innovation_variance"] <= 0:
        failures.append("writer innovations are degenerate")
    innovation_scale = math.sqrt(max(full_report["writer_innovation_variance"], 1e-15))
    standardized_mean = abs(full_report["writer_innovation_mean"]) / innovation_scale
    if not (args.minimum_calibration_slope <= full_report["calibration_slope"]
            <= args.maximum_calibration_slope):
        failures.append(f"calibration slope failed: {full_report['calibration_slope']:.6g}")
    residual_share = full_report["answer_residual_variance"] / max(
        full_report["answer_residual_variance"] + full_report["writer_innovation_variance"], 1e-15
    )
    if residual_share >= args.maximum_answer_residual_share:
        failures.append(f"answer residual absorbs signal: share={residual_share:.6g}")
    roots = sorted({validate_admissible_state(row)["stable_root_id"] for row in states})
    fold_counts = {str(fold): sum(
        validate_admissible_state(row)["stable_root_id"] in {
            root for root in roots if stable_fold_assignments(roots, args.fold_count)[root] == fold
        } for row in states
    ) for fold in range(args.fold_count)}
    turn_counts: dict[str, int] = {}
    turn_means: dict[str, float] = {}
    conditional_tests = []
    for turn in sorted({index for row in full_ledger["trajectories"]
                        for index in range(1, len(row["writer_innovations"]) + 1)}):
        values = [row["writer_innovations"][turn - 1] for row in full_ledger["trajectories"]
                  if len(row["writer_innovations"]) >= turn]
        turn_counts[str(turn)] = len(values)
        turn_means[str(turn)] = float(sum(values) / len(values))
    # Finite-sample martingale diagnostics condition on fold, turn, and a
    # preregistered sign bin of V_{t-1}; the maximum Studentized mean is gated.
    for trajectory in full_ledger["trajectories"]:
        prior_values = [trajectory["v0"], *trajectory["values"][:-1]]
        for turn, (prior, innovation) in enumerate(zip(
                prior_values, trajectory["writer_innovations"]), start=1):
            conditional_tests.append({
                "fold": int(trajectory["fold"]), "turn": turn,
                "prior_value_bin": "nonnegative" if prior >= 0 else "negative",
                "innovation": float(innovation),
            })
    grouped_tests = []
    group_keys = sorted({(row["fold"], row["turn"], row["prior_value_bin"])
                         for row in conditional_tests})
    for fold, turn, value_bin in group_keys:
        values = [row["innovation"] for row in conditional_tests
                  if (row["fold"], row["turn"], row["prior_value_bin"])
                  == (fold, turn, value_bin)]
        if len(values) < args.minimum_conditional_group_size:
            failures.append(
                f"undersized occupied conditional cell fold={fold},turn={turn},"
                f"bin={value_bin},n={len(values)}"
            )
            continue
        mean = float(sum(values) / len(values))
        variance = float(sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1))
        standard_error = math.sqrt(variance / len(values))
        statistic = abs(mean) / max(standard_error, 1e-12)
        grouped_tests.append({"fold": fold, "turn": turn, "prior_value_bin": value_bin,
                              "count": len(values), "mean": mean,
                              "standard_error": standard_error, "absolute_t": statistic})
    if not grouped_tests:
        failures.append("no conditional martingale group has preregistered coverage")
    elif max(row["absolute_t"] for row in grouped_tests) > args.maximum_conditional_t:
        failures.append("fold/turn/prior-value conditional martingale mean test failed")
    if any(count == 0 for count in fold_counts.values()):
        failures.append(f"empty stable fold coverage: {fold_counts}")
    if any(count < args.minimum_turn_coverage for count in turn_counts.values()):
        failures.append(f"insufficient turn coverage: {turn_counts}")
    report = {
        "schema": SCHEMA, "status": "PASS" if not failures else "FAIL",
        "decision": "MIC_E1_PASS" if not failures else "MIC_NO_GO_E1",
        "source_bundle_sha256": sha256_file(os.environ["MEMAGENT_MIC_E1_BUNDLE"]),
        "full": full_report, "turn_length": restricted_report,
        "mse_improvement_fraction": improvement, "answer_residual_variance_share": residual_share,
        "standardized_innovation_mean": standardized_mean,
        "maximum_closure_error": full_ledger["maximum_closure_error"],
        "oof_bundle_sha256": full["bundle_sha256"], "failures": failures,
        "stable_root_inventory_sha256": sha256_json(roots),
        "fold_state_counts": fold_counts, "turn_counts": turn_counts,
        "turn_innovation_means": turn_means,
        "conditional_martingale_tests": grouped_tests,
        "original_commit": frozen["original_commit"],
    }
    write_new(args.output, report)
    if failures:
        raise ValueError("MIC_NO_GO_E1: " + "; ".join(failures))
    return report


def import_baseline(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path = Path(os.environ.get("MEMAGENT_MIC_BASELINE_INVENTORY", ""))
    authority_sha = os.environ.get("MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256", "")
    if not inventory_path.is_file() or re.fullmatch(r"[0-9a-f]{64}", authority_sha) is None \
            or sha256_file(inventory_path) != authority_sha:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: evaluation inventory authority")
    inventory = read_json(inventory_path)
    for item in inventory.get("files", []):
        path = Path(expand(item["path"]))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {path}")
    validate_full_resolved_parity(expand(read_json(args.manifest)), inventory)
    interfaces = {}
    frozen_keys = None
    for item in inventory.get("prediction_files", []):
        path = Path(expand(item["path"]))
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != 128 or len({str(row["stable_key"]) for row in rows}) != 128:
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: stable S128 coverage")
        if [int(row.get("source_order_index", -1)) for row in rows] != list(range(128)):
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: source order is not 0..127")
        if any(any(re.fullmatch(r"[0-9a-f]{64}", str(row.get(key, ""))) is None
                   for key in ("source_question_sha256", "source_context_sha256")) for row in rows):
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: source identity hash absent")
        keys = [str(row["stable_key"]) for row in rows]
        if frozen_keys is None:
            frozen_keys = keys
        elif keys != frozen_keys:
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: anchor stable identities differ")
        scored = [score_terminal_output(row["output"], row["ground_truth"]) for row in rows]
        aggregate = summarize_fixed_s128(scored)
        expected = item.get("expected_aggregate")
        if expected and any(abs(float(aggregate[key]) - float(expected[key])) > 1e-12
                            for key in ("normalized_exact_match", "token_f1", "format_success")):
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: independently recomputed aggregate")
        interfaces[item["interface"]] = {
            "sha256": item["sha256"], "aggregate": aggregate,
            "stable_key_inventory_sha256": sha256_json(keys),
            "source_hash_inventory_sha256": sha256_json([
                [row["source_question_sha256"], row["source_context_sha256"]] for row in rows
            ]),
        }
    required = {"I", "Original5", "Original10", "Original15", "Original20", "Original25"}
    if set(interfaces) != required:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: all six baseline interfaces required")
    report = {"schema": SCHEMA, "status": "PASS", "decision": "MIC_BASELINE_IMPORT_PASS",
              "inventory_sha256": sha256_file(inventory_path),
              "interfaces": interfaces}
    write_new(args.output, report)
    return report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from recurrent.research.mic import CriticCheckpoint
    from recurrent.research.gate_a_execution import validate_jsonl_chain
    required = {
        "p0": "MIC_P0_PASS", "e0": "MIC_E0_PASS",
        "paper_review": "MIC_PAPER_REVIEW_GO",
    }
    checked = {}
    for name, decision in required.items():
        path = Path(getattr(args, name))
        value = read_json(path)
        if value.get("status") != "PASS" or value.get("decision") != decision:
            raise ValueError(f"MIC_NO_GO: {name} gate is not {decision}")
        checked[name] = sha256_file(path)
    ledger_path = Path(args.ledger)
    lines = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    previous = "0" * 64
    previous_critic = None
    steps = []
    gradient_steps = []
    health_by_step: dict[int, dict[str, Any]] = {}
    for sequence, row in enumerate(lines):
        digest = row.pop("entry_sha256", None)
        if row.get("sequence") != sequence or row.get("previous_entry_sha256") != previous \
                or digest != sha256_json(row):
            raise ValueError("MIC_NO_GO: execution ledger chain corrupted")
        previous = digest
        if row.get("record_type") == "mic_advantage_delivery":
            step = int(row["global_step"])
            steps.append(step)
            checkpoint = Path(row["critic_checkpoint"])
            if not checkpoint.is_file():
                raise ValueError("MIC_NO_GO: critic checkpoint binding mismatch")
            checkpoint_payload = CriticCheckpoint.read(
                checkpoint, expected_actor_commit=read_json(args.p0)["git_commit"]
            )
            if checkpoint_payload["checkpoint_sha256"] != row["critic_checkpoint_sha256"] \
                    or checkpoint_payload["fold_bundle_sha256"] != row["oof_bundle_sha256"]:
                raise ValueError("MIC_NO_GO: critic checkpoint binding mismatch")
            if row.get("parent_critic_checkpoint_sha256") != previous_critic \
                    or checkpoint_payload["critic_payload"].get(
                        "parent_checkpoint_sha256"
                    ) != previous_critic:
                raise ValueError("MIC_NO_GO: critic resume chain mismatch")
            previous_critic = checkpoint_payload["checkpoint_sha256"]
            closure = float(row["maximum_closure_error"])
            if not math.isfinite(closure) or closure > 1e-12:
                raise ValueError("MIC_NO_GO: training closure drift")
            calibration = row.get("calibration", {})
            required_calibration = (
                "mse", "mae", "calibration_slope", "calibration_intercept",
                "writer_innovation_mean", "writer_innovation_variance",
                "answer_residual_variance",
            )
            try:
                calibration_values = {key: float(calibration[key]) for key in required_calibration}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("MIC_NO_GO: on-policy calibration metrics missing") from exc
            if any(not math.isfinite(value) for value in calibration_values.values()):
                raise ValueError("MIC_NO_GO: on-policy calibration is non-finite")
            writer_variance = calibration_values["writer_innovation_variance"]
            answer_variance = calibration_values["answer_residual_variance"]
            if writer_variance <= 1e-15:
                raise ValueError("MIC_NO_GO: writer innovation is degenerate")
            residual_share = answer_variance / max(writer_variance + answer_variance, 1e-15)
            if residual_share >= 0.999999:
                raise ValueError("MIC_NO_GO: answer residual absorbs all on-policy signal")
            delivery = row.get("delivery", {})
            if int(delivery.get("writer_active_tokens", 0)) <= 0 \
                    or int(delivery.get("answer_active_tokens", 0)) <= 0:
                raise ValueError("MIC_NO_GO: writer/answer advantage delivery inactive")
            health_by_step.setdefault(step, {}).update({
                "closure": closure, "calibration": calibration_values,
                "answer_residual_variance_share": residual_share,
                "advantage_delivery": {
                    "writer_active_tokens": int(delivery["writer_active_tokens"]),
                    "answer_active_tokens": int(delivery["answer_active_tokens"]),
                },
            })
        elif row.get("record_type") == "mic_actual_gradient_delivery":
            step = int(row["global_step"])
            gradient_steps.append(step)
            role_metrics = row.get("role_metrics", {})
            required_role_metrics = (
                "mic_gradient/writer_pg_loss", "mic_gradient/answer_pg_loss",
                "mic_gradient/writer_active_tokens", "mic_gradient/answer_active_tokens",
                "mic_gradient/writer_logprob_grad_l2", "mic_gradient/answer_logprob_grad_l2",
                "mic_gradient/writer_logprob_grad_abs_max",
                "mic_gradient/answer_logprob_grad_abs_max",
            )
            try:
                numeric_role_metrics = {key: float(role_metrics[key]) for key in required_role_metrics}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("MIC_NO_GO: actual role gradient metrics missing") from exc
            if any(not math.isfinite(value) for value in numeric_role_metrics.values()):
                raise ValueError("MIC_NO_GO: actual role gradient metric is non-finite")
            if numeric_role_metrics["mic_gradient/writer_active_tokens"] <= 0 \
                    or numeric_role_metrics["mic_gradient/answer_active_tokens"] <= 0:
                raise ValueError("MIC_NO_GO: actual role gradient tokens inactive")
            if numeric_role_metrics["mic_gradient/writer_logprob_grad_l2"] <= 0 \
                    or numeric_role_metrics["mic_gradient/writer_logprob_grad_abs_max"] <= 0:
                raise ValueError("MIC_NO_GO: actual writer gradient is zero")
            health_by_step.setdefault(step, {})["actual_role_gradient"] = numeric_role_metrics
    if args.target_step not in steps:
        raise ValueError(f"MIC_NO_GO: target step {args.target_step} absent from MIC ledger")
    if sorted(steps) != sorted(gradient_steps):
        raise ValueError("MIC_NO_GO: role-specific actual gradient ledger coverage differs")
    if set(health_by_step) != set(steps) \
            or any("actual_role_gradient" not in health_by_step[step] for step in steps):
        raise ValueError("MIC_NO_GO: on-policy mechanism health coverage differs")
    weight_rows = [json.loads(line) for line in Path(args.weight_ledger).read_text(
        encoding="utf-8").splitlines() if line]
    weight_failures = validate_jsonl_chain(weight_rows)
    if weight_failures:
        raise ValueError("MIC_NO_GO: weight-sync ledger chain corrupted: " + ";".join(weight_failures))
    sync_steps = []
    for row in weight_rows:
        if row.get("sync_kind") != "post_actor_update":
            continue
        sync_steps.append(int(row["global_step"]))
        if row.get("worker_ranks") != [0, 1] \
                or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sampled_tensor_digest", ""))) is None \
                or re.fullmatch(r"[0-9a-f]{64}", str(
                    row.get("actor_master_sampled_tensor_digest", "")
                )) is None:
            raise ValueError("MIC_NO_GO: actor/vLLM weight-sync summary mismatch")
    if sorted(set(steps)) != sorted(set(sync_steps)):
        raise ValueError("MIC_NO_GO: weight-sync step coverage differs from MIC updates")
    report = {"schema": SCHEMA, "status": "PASS", "decision": f"MIC_T{args.target_step}_AUDIT_PASS",
              "gate_sha256": checked, "ledger_tail_sha256": previous, "mic_steps": steps,
              "gradient_steps": gradient_steps, "on_policy_health": health_by_step}
    report["weight_sync_ledger_sha256"] = sha256_file(args.weight_ledger)
    if args.output:
        write_new(args.output, report)
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(args.predictions).read_text(
        encoding="utf-8").splitlines() if line]
    if len(rows) != 128:
        raise ValueError("MIC_NO_GO: method fixed-S128 denominator is not 128")
    keys = [str(row.get("stable_key", "")) for row in rows]
    if any(not key for key in keys) or len(set(keys)) != 128:
        raise ValueError("MIC_NO_GO: method fixed-S128 stable IDs are invalid")
    scored = [score_terminal_output(row["output"], row["ground_truth"]) for row in rows]
    aggregate = summarize_fixed_s128(scored)
    baseline = read_json(args.baseline)
    interface = f"Original{args.step}"
    original = baseline["interfaces"][interface]["aggregate"]
    if sha256_json(keys) != baseline["interfaces"][interface]["stable_key_inventory_sha256"]:
        raise ValueError("MIC_NO_GO: method S128 identities differ from frozen Original")
    f1_delta_pp = 100.0 * (float(aggregate["token_f1"]) - float(original["token_f1"]))
    decision = f"MIC_T{args.step}_EVAL_PASS"
    report = {
        "schema": SCHEMA, "status": "PASS",
        "decision": decision, "step": args.step,
        "prediction_sha256": sha256_file(args.predictions), "aggregate": aggregate,
        "original_aggregate": original, "token_f1_delta_pp": f1_delta_pp,
    }
    write_new(args.output, report)
    return report


def prepare_eval(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd

    generated = [json.loads(line) for line in Path(args.generations).read_text(
        encoding="utf-8").splitlines() if line]
    baseline = [json.loads(line) for line in Path(args.identity_source).read_text(
        encoding="utf-8").splitlines() if line]
    frame = pd.read_parquet(args.validation)
    if len(generated) != 128 or len(baseline) != 128 or len(frame) != 128:
        raise ValueError("MIC_NO_GO: evaluation producer inputs must each contain 128 rows")
    baseline.sort(key=lambda row: int(row.get("source_order_index", row.get("source_repeated_row", -1))))
    generated_by_key = {str(row.get("stable_key", "")): row for row in generated}
    baseline_keys = {str(row["stable_key"]) for row in baseline}
    if len(generated_by_key) != 128 or set(generated_by_key) != baseline_keys:
        raise ValueError("MIC_NO_GO: Method generation identity set differs from frozen S128")
    rows = []
    for index, identity in enumerate(baseline):
        generation = generated_by_key[str(identity["stable_key"])]
        if int(generation.get("source_order_index", -1)) != index \
                or generation.get("source_question_sha256") != identity.get("source_question_sha256") \
                or generation.get("source_context_sha256") != identity.get("source_context_sha256"):
            raise ValueError(f"MIC_NO_GO: Method generation identity binding mismatch at row {index}")
        source = frame.iloc[index].to_dict()
        reward_model = source.get("reward_model") if isinstance(source.get("reward_model"), dict) else {}
        extra_info = source.get("extra_info") if isinstance(source.get("extra_info"), dict) else {}
        ground_truth = reward_model.get("ground_truth", extra_info.get(
            "answer", source.get("answer", source.get("ground_truth"))
        ))
        if ground_truth is None:
            raise ValueError(f"MIC_NO_GO: validation row {index} has no ground truth")
        stable_key = identity["stable_key"]
        question = source.get("prompt", source.get("question", source.get("input")))
        if isinstance(question, list) and question and isinstance(question[0], dict):
            question = question[0].get("content")
        context = source.get("context")
        if hashlib.sha256(str(question).encode("utf-8")).hexdigest() \
                != identity.get("source_question_sha256") \
                or hashlib.sha256(str(context).encode("utf-8")).hexdigest() \
                != identity.get("source_context_sha256"):
            raise ValueError(f"MIC_NO_GO: frozen source identity mismatch at row {index}")
        rows.append({"source_order_index": index, "stable_key": stable_key,
                     "output": generation["output"], "ground_truth": ground_truth})
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"MIC_NO_GO: refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    return {"schema": SCHEMA, "status": "PASS", "decision": "MIC_S128_PREPARE_PASS",
            "rows": 128, "predictions_sha256": sha256_file(target)}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("p0", "e0", "e1", "import-baseline"):
        child = sub.add_parser(name)
        child.add_argument("--output", required=True)
        child.add_argument("--manifest", default=str(REPO / "manifests/h20/qwen25_7b_mic_seed2026.json"))
    sub.choices["p0"].add_argument("--check-runtime", action="store_true")
    sub.choices["e1"].add_argument("--fold-count", type=int, default=4)
    sub.choices["e1"].add_argument("--alpha", type=float, default=1.0)
    sub.choices["e1"].add_argument("--dimension", type=int, default=64)
    sub.choices["e1"].add_argument("--minimum-mse-improvement", type=float, default=0.05)
    sub.choices["e1"].add_argument("--maximum-answer-residual-share", type=float, default=0.95)
    sub.choices["e1"].add_argument("--maximum-standardized-innovation-mean", type=float, default=0.25)
    sub.choices["e1"].add_argument("--minimum-calibration-slope", type=float, default=0.2)
    sub.choices["e1"].add_argument("--maximum-calibration-slope", type=float, default=1.8)
    sub.choices["e1"].add_argument("--minimum-turn-coverage", type=int, default=8)
    sub.choices["e1"].add_argument("--minimum-conditional-group-size", type=int, default=8)
    sub.choices["e1"].add_argument("--maximum-conditional-t", type=float, default=3.5)
    child = sub.add_parser("audit")
    for name in ("p0", "e0", "paper-review", "ledger", "weight-ledger"):
        child.add_argument("--" + name, required=True)
    child.add_argument("--target-step", type=int, choices=(5, 10, 15, 20, 25), required=True)
    child.add_argument("--output")
    child = sub.add_parser("evaluate")
    child.add_argument("--predictions", required=True)
    child.add_argument("--baseline", required=True)
    child.add_argument("--step", type=int, choices=(5, 10, 15, 20, 25), required=True)
    child.add_argument("--output", required=True)
    child = sub.add_parser("prepare-eval")
    child.add_argument("--generations", required=True)
    child.add_argument("--validation", required=True)
    child.add_argument("--identity-source", required=True)
    child.add_argument("--output", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    functions = {"p0": p0, "e0": e0, "e1": e1, "import-baseline": import_baseline,
                 "audit": audit, "evaluate": evaluate, "prepare-eval": prepare_eval}
    try:
        result = functions[args.command](args)
    except Exception as exc:
        print(f"MIC_NO_GO:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
