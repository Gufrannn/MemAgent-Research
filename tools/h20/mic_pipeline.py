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
CHECKPOINT_AUTHORITY = REPO / "manifests/h20/qwen25_7b_mic_checkpoint_authority.json"
CURVE_AUTHORITY = REPO / "manifests/h20/qwen25_7b_mic_original_curve_authority.json"


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


def _verify_curve_authority(
        authority_path: str | Path, p0_value: dict[str, Any],
        report_path: str | Path, resolved_path: str | Path,
) -> dict[str, Any]:
    authority_path = Path(authority_path).resolve()
    report_path = Path(report_path).resolve()
    resolved_path = Path(resolved_path).resolve()
    authority = read_json(authority_path)
    if authority.get("schema") != SCHEMA \
            or authority.get("authority_kind") != "out_of_band_original_curve_sha256_pinned" \
            or authority.get("curve_report_path") != str(report_path) \
            or authority.get("curve_report_sha256") != sha256_file(report_path) \
            or authority.get("curve_report_sha256") != p0_value.get(
                "original_curve_report_sha256"
            ) \
            or authority.get("curve_resolved_path") != str(resolved_path) \
            or authority.get("curve_resolved_sha256") != sha256_file(resolved_path):
        raise ValueError(
            "ORIGINAL_BASELINE_PROTOCOL_MISMATCH: release-pinned curve authority"
        )
    return authority


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
        "requires_training_checkpoint_inventory": True,
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
    if not inventory_path.is_file() or re.fullmatch(r"[0-9a-f]{64}", authority_sha) is None:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: evaluation inventory authority")
    inventory = read_json(inventory_path)
    p0_value = read_json(args.p0)
    if p0_value.get("status") != "PASS" or p0_value.get("decision") != "MIC_P0_PASS" \
            or inventory.get("source_curve_report_sha256") != authority_sha \
            or p0_value.get("original_curve_report_sha256") != authority_sha \
            or Path(inventory.get("source_curve_report", "")).resolve() != Path(
                p0_value.get("original_curve_report", "")
            ).resolve():
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: certified curve trust root")
    from recurrent.research.stable_eval_identity import canonical_sha256, stable_key

    curve_report = read_json(inventory["source_curve_report"])
    curve_interfaces = curve_report.get("evidence", {}).get("interfaces", {})
    curve_resolved_path = Path(inventory.get("source_curve_resolved", ""))
    if not curve_resolved_path.is_file() \
            or sha256_file(curve_resolved_path) != inventory.get(
                "source_curve_resolved_sha256"
            ):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: curve identity authority")
    curve_resolved = read_json(curve_resolved_path)
    curve_authority_path = Path(
        getattr(args, "curve_authority", CURVE_AUTHORITY)
    ).resolve()
    _verify_curve_authority(
        curve_authority_path, p0_value,
        inventory["source_curve_report"], curve_resolved_path,
    )
    if inventory.get("curve_authority_path") != str(curve_authority_path) \
            or inventory.get("curve_authority_file_sha256") != sha256_file(
                curve_authority_path
            ):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: inventory curve authority")
    identity_payload = curve_resolved.get("identity_payload", {})
    if curve_resolved.get("eval_manifest_hash") != canonical_sha256(identity_payload):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: curve identity digest")
    shared_protocol_sha = curve_resolved.get("execution_binding", {}).get(
        "trainer_configuration", {}
    ).get("shared_generation_protocol_sha256", "")
    if re.fullmatch(r"[0-9a-f]{64}", str(shared_protocol_sha)) is None:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: generation protocol authority")
    reward_code_sha = curve_resolved.get("execution_binding", {}).get(
        "execution_code_sha256", {}
    ).get("recurrent/research/hotpotqa_dense_reward.py", "")
    if re.fullmatch(r"[0-9a-f]{64}", str(reward_code_sha)) is None:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: reward code authority")
    frozen_by_order = {
        int(row["source_order_index"]): row for row in identity_payload.get("rows", [])
    }
    if set(frozen_by_order) != set(range(128)):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: curve identity S128 coverage")
    for item in inventory.get("files", []):
        path = Path(expand(item["path"]))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {path}")
    interfaces = {}
    frozen_keys = None
    for item in inventory.get("prediction_files", []):
        interface = item["interface"]
        certified_interface = curve_interfaces.get(interface, {})
        certified_terminals = [
            evidence.get("sha256")
            for relative, evidence in certified_interface.get("artifacts", {}).items()
            if relative.startswith("terminal/") and relative.endswith(".jsonl")
        ]
        if len(certified_terminals) != 1 \
                or item.get("source_terminal_sha256") != certified_terminals[0] \
                or item.get("independent_metric_rows_sha256") != certified_interface.get(
                    "independent_metric_rows_sha256"
                ):
            raise ValueError(
                f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} final-report binding"
            )
        terminal_path = Path(item.get("source_terminal_path", ""))
        if not terminal_path.is_file() \
                or sha256_file(terminal_path) != item["source_terminal_sha256"]:
            raise ValueError(
                f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} terminal artifact"
            )
        terminal_rows = [json.loads(line) for line in terminal_path.read_text(
            encoding="utf-8"
        ).splitlines() if line]
        terminal_rows.sort(key=lambda row: int(row.get("source_order_index", -1)))
        if len(terminal_rows) != 128 or [int(row.get("source_order_index", -1))
                                        for row in terminal_rows] != list(range(128)):
            raise ValueError(
                f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} terminal S128 order"
            )
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
        metric_rows = []
        for order, (row, terminal) in enumerate(zip(rows, terminal_rows)):
            frozen = frozen_by_order[order]
            if row.get("output") != terminal.get("output") \
                    or int(row.get("raw_row_position", -1)) != int(
                        frozen["raw_row_position"]
                    ) \
                    or row.get("source_question_sha256") != frozen["source_question_hash"] \
                    or row.get("source_context_sha256") != frozen["source_context_hash"] \
                    or canonical_sha256(row.get("ground_truth")) != frozen["ground_truth_hash"]:
                raise ValueError(
                    f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} normalized row {order}"
                )
            expected_key = json.dumps(stable_key(row), separators=(",", ":"))
            if row.get("stable_key") != expected_key:
                raise ValueError(
                    f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} stable key row {order}"
                )
            metric_rows.append({
                "stable_key": expected_key, "source_order_index": order,
                "eval_manifest_hash": row["eval_manifest_hash"],
                "example_id": row["example_id"], "replica_id": row["replica_id"],
                "trajectory_seed": row["trajectory_seed"],
                "trajectory_id": row["trajectory_id"],
                **score_terminal_output(row["output"], row["ground_truth"]),
            })
        if canonical_sha256(metric_rows) != certified_interface.get(
            "independent_metric_rows_sha256"
        ):
            raise ValueError(
                f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} canonical metric rows"
            )
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
            "path": str(path.resolve()), "sha256": item["sha256"], "aggregate": aggregate,
            "stable_key_inventory_sha256": sha256_json(keys),
            "source_hash_inventory_sha256": sha256_json([
                [row["source_question_sha256"], row["source_context_sha256"]] for row in rows
            ]),
        }
    required = {"I", "Original5", "Original10", "Original15", "Original20", "Original25"}
    if set(interfaces) != required:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: all six baseline interfaces required")
    validation_items = [
        item for item in inventory.get("files", [])
        if Path(expand(item.get("path", ""))).resolve().suffix == ".parquet"
    ]
    if len(validation_items) != 1:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: validation authority")
    report = {"schema": SCHEMA, "status": "PASS", "decision": "MIC_BASELINE_IMPORT_PASS",
              "inventory_path": str(inventory_path.resolve()),
              "inventory_sha256": sha256_file(inventory_path),
              "curve_report_sha256": authority_sha,
              "curve_resolved_path": str(curve_resolved_path.resolve()),
              "curve_resolved_sha256": sha256_file(curve_resolved_path),
              "validation_path": str(Path(expand(validation_items[0]["path"])).resolve()),
              "validation_sha256": validation_items[0]["sha256"],
              "shared_generation_protocol_sha256": shared_protocol_sha,
              "original_reward_code_sha256": reward_code_sha,
              "interfaces": interfaces}
    if getattr(args, "verify_existing", False):
        existing = read_json(args.output)
        if sha256_json(existing) != sha256_json(report):
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: import certificate differs")
    else:
        write_new(args.output, report)
    return report


def _parquet_ground_truth(source: dict[str, Any], *, row: int) -> Any:
    reward_model = source.get("reward_model")
    if isinstance(reward_model, str):
        reward_model = json.loads(reward_model)
    if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
        raise ValueError(f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: row {row} has no ground truth")
    return _json_native(reward_model["ground_truth"])


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_native(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_native(item())
    return value


def _load_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return parquet.read_table(Path(path)).to_pylist()


def _actor_model_inventory(output_root: str | Path, steps: tuple[int, ...]) -> list[dict[str, Any]]:
    root = Path(output_root).resolve()
    records = []
    for step in steps:
        files = []
        for rank in (0, 1):
            path = root / f"global_step_{step}/actor/model_world_size_2_rank_{rank}.pt"
            if not path.is_file():
                raise ValueError(f"MIC_NO_GO: actor checkpoint shard absent: {path}")
            files.append({
                "path": f"global_step_{step}/actor/{path.name}",
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        records.append({"step": step, "files": files})
    return records


def _checkpoint_records_from_training_ledger(
        ledger_path: str | Path, output_root: str | Path,
        p0_value: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    ledger_path = Path(ledger_path).resolve()
    rows = [json.loads(line) for line in ledger_path.read_text(
        encoding="utf-8"
    ).splitlines() if line]
    previous = "0" * 64
    for sequence, original in enumerate(rows):
        row = dict(original)
        digest = row.pop("entry_sha256", None)
        if row.get("sequence") != sequence \
                or row.get("previous_entry_sha256") != previous \
                or digest != sha256_json(row):
            raise ValueError("MIC_NO_GO: checkpoint training ledger chain corrupted")
        previous = str(digest)
    output_root = Path(output_root).resolve()
    records = []
    for step in (5, 10, 15, 20, 25):
        matches = [
            row for row in rows
            if row.get("record_type") == "mic_actor_checkpoint_inventory"
            and int(row.get("global_step", -1)) == step
        ]
        if len(matches) != 1:
            raise ValueError(
                f"MIC_NO_GO: training-time checkpoint inventory cardinality T{step}"
            )
        row = matches[0]
        checkpoint = output_root / f"global_step_{step}"
        if Path(row.get("checkpoint_path", "")).resolve() != checkpoint \
                or row.get("git_commit") != p0_value.get("git_commit") \
                or row.get("run_id") != p0_value.get("run_id"):
            raise ValueError("MIC_NO_GO: training-time checkpoint inventory identity")
        files = []
        for item in row.get("model_shards", []):
            relative = str(item.get("path", ""))
            files.append({
                "path": f"global_step_{step}/{relative}",
                "size": item.get("size"), "sha256": item.get("sha256"),
            })
        if len(files) != 2 \
                or sha256_json(row.get("model_shards", [])) != row.get(
                    "model_shards_sha256"
                ):
            raise ValueError("MIC_NO_GO: training-time checkpoint shard inventory")
        records.append({"step": step, "files": files})
    current = _actor_model_inventory(output_root, (5, 10, 15, 20, 25))
    if records != current:
        raise ValueError("MIC_NO_GO: actor checkpoints differ from training-time inventory")
    return records, sha256_file(ledger_path)


def _checkpoint_authority_report(args: argparse.Namespace) -> dict[str, Any]:
    p0_value = read_json(args.p0)
    records = _actor_model_inventory(args.output_root, (5, 10, 15, 20, 25))
    report = {
        "schema": SCHEMA, "status": "PASS",
        "decision": "MIC_CHECKPOINT_AUTHORITY_PASS",
        "training_p0_sha256": sha256_file(args.p0),
        "output_root": str(Path(args.output_root).resolve()), "records": records,
    }
    if p0_value.get("requires_training_checkpoint_inventory") is True:
        ledger_records, ledger_sha = _checkpoint_records_from_training_ledger(
            args.ledger, args.output_root, p0_value,
        )
        if ledger_records != records:
            raise ValueError("MIC_NO_GO: training-ledger checkpoint authority mismatch")
        report.update({
            "authority_kind": "training_ledger_checkpoint_inventory",
            "training_ledger_path": str(Path(args.ledger).resolve()),
            "training_ledger_sha256": ledger_sha,
            "checkpoint_authority_sha256": sha256_json(records),
        })
        return report
    authority_path = Path(args.authority).resolve()
    authority = read_json(authority_path)
    expected = authority.get("checkpoint_authority_sha256", "")
    if authority.get("schema") != SCHEMA \
            or authority.get("authority_kind") != "out_of_band_operator_sha256_pinned" \
            or authority.get("training_git_commit") != p0_value.get("git_commit") \
            or authority.get("run_id") != p0_value.get("run_id") \
            or re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None:
        raise ValueError("MIC_NO_GO: legacy checkpoint authority is not release-pinned")
    if sha256_json(records) != expected:
        raise ValueError("MIC_NO_GO: actor checkpoint inventory differs from release authority")
    report.update({
        "authority_kind": "out_of_band_operator_sha256_pinned",
        "authority_path": str(authority_path),
        "authority_file_sha256": sha256_file(authority_path),
        "checkpoint_authority_sha256": expected,
    })
    return report


def materialize_checkpoint_authority(args: argparse.Namespace) -> dict[str, Any]:
    report = _checkpoint_authority_report(args)
    write_new(args.output, report)
    return report


def verify_checkpoint_authority(
        authority_path: str | Path, certificate_path: str | Path,
        output_root: str | Path, step: int, p0_path: str | Path,
        ledger_path: str | Path) -> dict[str, Any]:
    certificate_path = Path(certificate_path).resolve()
    expected_report = _checkpoint_authority_report(argparse.Namespace(
        authority=authority_path, p0=p0_path, output_root=output_root,
        ledger=ledger_path,
    ))
    certificate = read_json(certificate_path)
    if sha256_json(certificate) != sha256_json(expected_report):
        raise ValueError("MIC_NO_GO: checkpoint authority certificate mismatch")
    expected_rows = [row for row in certificate["records"] if int(row.get("step", -1)) == step]
    if len(expected_rows) != 1 \
            or _actor_model_inventory(output_root, (step,)) != expected_rows:
        raise ValueError("MIC_NO_GO: evaluated actor checkpoint differs from release authority")
    return certificate


def materialize_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize normalized S128 rows from final-report-authenticated terminals."""
    from recurrent.research.stable_eval_identity import canonical_sha256, stable_key

    p0_value = read_json(args.p0)
    report_path = Path(args.curve_report).resolve()
    resolved_path = Path(args.curve_resolved).resolve()
    validation_path = Path(args.validation).resolve()
    search_root = Path(args.search_root).resolve()
    output_root = Path(args.output_root).resolve()
    if p0_value.get("status") != "PASS" or p0_value.get("decision") != "MIC_P0_PASS" \
            or report_path != Path(p0_value.get("original_curve_report", "")).resolve() \
            or sha256_file(report_path) != p0_value.get("original_curve_report_sha256"):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: MIC P0 curve binding")
    report = read_json(report_path)
    resolved = read_json(resolved_path)
    if report.get("status") != "PASS" \
            or report.get("decision") != "ORIGINAL_S128_CURVE_PASS":
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: curve report is not PASS")
    _verify_curve_authority(
        getattr(args, "curve_authority", CURVE_AUTHORITY), p0_value,
        report_path, resolved_path,
    )
    identity_payload = resolved.get("identity_payload", {})
    frozen_rows = identity_payload.get("rows", [])
    if len(frozen_rows) != 128 or resolved.get("eval_manifest_hash") != canonical_sha256(
        identity_payload
    ):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: frozen S128 identity manifest")
    expected_parquet_sha = identity_payload.get("source_dataset", {}).get("parquet_sha256")
    if expected_parquet_sha != sha256_file(validation_path):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: frozen S128 parquet digest")
    frame = _load_parquet_rows(validation_path)
    if len(frame) != 128:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: validation parquet is not S128")
    frozen_by_order = {int(row["source_order_index"]): row for row in frozen_rows}
    if set(frozen_by_order) != set(range(128)):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: identity order is not 0..127")
    if output_root.exists() or Path(args.output).exists():
        raise FileExistsError("MIC_NO_GO: refusing to overwrite baseline materialization")

    interface_evidence = report.get("evidence", {}).get("interfaces", {})
    required = {"I", "Original5", "Original10", "Original15", "Original20", "Original25"}
    if set(interface_evidence) != required:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: curve interface inventory")
    terminal_targets: dict[str, str] = {}
    for interface, evidence in interface_evidence.items():
        terminals = [
            (relative, item.get("sha256"))
            for relative, item in evidence.get("artifacts", {}).items()
            if relative.startswith("terminal/") and relative.endswith(".jsonl")
        ]
        if len(terminals) != 1 or re.fullmatch(r"[0-9a-f]{64}", str(terminals[0][1])) is None:
            raise ValueError(
                f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} terminal authority"
            )
        terminal_targets[interface] = str(terminals[0][1])
    locations_by_digest: dict[str, list[Path]] = {
        digest: [] for digest in set(terminal_targets.values())
    }
    for path in search_root.rglob("*.jsonl"):
        if path.parent.name != "terminal":
            continue
        digest = sha256_file(path)
        if digest in locations_by_digest:
            locations_by_digest[digest].append(path.resolve())
    missing_locations = sorted(
        interface for interface, digest in terminal_targets.items()
        if not locations_by_digest[digest]
    )
    if missing_locations:
        raise ValueError(
            "ORIGINAL_BASELINE_PROTOCOL_MISMATCH: terminal SHA is absent for "
            + canonical_json(missing_locations)
        )
    matches = {
        interface: sorted(locations_by_digest[digest])[0]
        for interface, digest in terminal_targets.items()
    }

    output_root.mkdir(parents=True)
    prediction_files = []
    files = [
        {"path": str(report_path), "sha256": sha256_file(report_path)},
        {"path": str(resolved_path), "sha256": sha256_file(resolved_path)},
        {"path": str(validation_path), "sha256": sha256_file(validation_path)},
    ]
    reference_keys = None
    for interface in sorted(required):
        terminal_path = matches[interface]
        terminal_sha = sha256_file(terminal_path)
        terminal_rows = [json.loads(line) for line in terminal_path.read_text(
            encoding="utf-8"
        ).splitlines() if line]
        terminal_rows.sort(key=lambda row: int(row.get("source_order_index", -1)))
        if len(terminal_rows) != 128 or [int(row.get("source_order_index", -1))
                                        for row in terminal_rows] != list(range(128)):
            raise ValueError(
                f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} terminal order"
            )
        normalized_rows = []
        metric_rows = []
        for order, terminal in enumerate(terminal_rows):
            frozen = frozen_by_order[order]
            for field in (
                "example_id", "semantic_dataset_index", "source_order_index",
                "raw_row_position", "production_effective_position", "context_token_count",
                "source_question_hash", "source_context_hash", "ground_truth_hash",
                "eval_manifest_hash", "replica_id", "trajectory_seed", "trajectory_id",
            ):
                expected = resolved["eval_manifest_hash"] if field == "eval_manifest_hash" \
                    else frozen.get(field, terminal.get(field))
                if terminal.get(field) != expected:
                    raise ValueError(
                        f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} identity {field} row {order}"
                    )
            raw_position = int(frozen["raw_row_position"])
            source = frame[raw_position]
            ground_truth = _parquet_ground_truth(source, row=order)
            if canonical_sha256(ground_truth) != terminal.get("ground_truth_hash"):
                raise ValueError(
                    f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} ground truth row {order}"
                )
            key = json.dumps(stable_key(terminal), separators=(",", ":"))
            scored = score_terminal_output(terminal.get("output", ""), ground_truth)
            metric_rows.append({
                "stable_key": key, "source_order_index": order,
                "eval_manifest_hash": terminal["eval_manifest_hash"],
                "example_id": terminal["example_id"], "replica_id": terminal["replica_id"],
                "trajectory_seed": terminal["trajectory_seed"],
                "trajectory_id": terminal["trajectory_id"], **scored,
            })
            normalized_rows.append({
                "source_order_index": order, "raw_row_position": raw_position,
                "stable_key": key, "stable_example_id": str(terminal["example_id"]),
                "stable_root_id": key, "trajectory_seed": int(terminal["trajectory_seed"]),
                "eval_manifest_hash": terminal["eval_manifest_hash"],
                "example_id": terminal["example_id"], "replica_id": terminal["replica_id"],
                "trajectory_id": terminal["trajectory_id"],
                "source_question_sha256": terminal["source_question_hash"],
                "source_context_sha256": terminal["source_context_hash"],
                "output": terminal.get("output", ""), "ground_truth": ground_truth,
            })
        expected_metric_digest = interface_evidence[interface].get(
            "independent_metric_rows_sha256"
        )
        if canonical_sha256(metric_rows) != expected_metric_digest:
            raise ValueError(
                f"ORIGINAL_BASELINE_PROTOCOL_MISMATCH: {interface} metric-row digest"
            )
        keys = [row["stable_key"] for row in normalized_rows]
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: stable keys differ by anchor")
        target = output_root / f"{interface}.jsonl"
        target.write_text("".join(canonical_json(row) + "\n" for row in normalized_rows),
                          encoding="utf-8")
        prediction_files.append({
            "interface": interface, "path": str(target), "sha256": sha256_file(target),
            "source_terminal_path": str(terminal_path),
            "source_terminal_sha256": terminal_sha,
            "independent_metric_rows_sha256": expected_metric_digest,
            "expected_aggregate": summarize_fixed_s128(metric_rows),
        })
        files.append({"path": str(terminal_path), "sha256": terminal_sha})
    inventory = {
        "schema": SCHEMA, "kind": "certified_baseline_materialization",
        "source_curve_report": str(report_path),
        "source_curve_report_sha256": sha256_file(report_path),
        "source_curve_resolved": str(resolved_path),
        "source_curve_resolved_sha256": sha256_file(resolved_path),
        "curve_authority_path": str(Path(args.curve_authority).resolve()),
        "curve_authority_file_sha256": sha256_file(args.curve_authority),
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "files": files, "prediction_files": prediction_files,
    }
    write_new(args.output, inventory)
    return {"schema": SCHEMA, "status": "PASS",
            "decision": "MIC_BASELINE_MATERIALIZE_PASS",
            "inventory_sha256": sha256_file(args.output),
            "curve_authority_file_sha256": sha256_file(args.curve_authority),
            "interfaces": sorted(required)}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from recurrent.research.mic import (
        CriticCheckpoint, calibration_report, innovation_ledger,
        select_trajectory_ledger,
    )
    from recurrent.research.gate_a_execution import validate_jsonl_chain
    required = {
        "p0": "MIC_P0_PASS", "e0": "MIC_E0_PASS",
        "paper_review": "MIC_PAPER_REVIEW_GO",
    }
    checked = {}
    gate_values = {}
    for name, decision in required.items():
        path = Path(getattr(args, name))
        value = read_json(path)
        if value.get("status") != "PASS" or value.get("decision") != decision:
            raise ValueError(f"MIC_NO_GO: {name} gate is not {decision}")
        checked[name] = sha256_file(path)
        gate_values[name] = value
    ledger_path = Path(args.ledger)
    lines = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    previous = "0" * 64
    previous_critic = None
    steps = []
    gradient_steps = []
    checkpoint_steps = []
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
            critic_payload = checkpoint_payload.get("critic_payload", {})
            reconstructed_cumulative = innovation_ledger(
                critic_payload.get("oof", {}), critic_payload.get("history_outcomes", {}),
                tolerance=1e-12,
            )
            current_ids = row.get("current_trajectory_ids")
            if not isinstance(current_ids, list):
                raise ValueError("MIC_NO_GO: current trajectory IDs missing from ledger")
            reconstructed_current = select_trajectory_ledger(
                reconstructed_cumulative, [str(value) for value in current_ids]
            )
            if reconstructed_cumulative["ledger_sha256"] != \
                    row.get("cumulative_innovation_ledger_sha256") \
                    or reconstructed_current["ledger_sha256"] != \
                    row.get("innovation_ledger_sha256"):
                raise ValueError("MIC_NO_GO: innovation ledger digest binding mismatch")
            if sha256_json(calibration_report(reconstructed_current)) != \
                    sha256_json(row.get("calibration", {})):
                raise ValueError("MIC_NO_GO: on-policy calibration binding mismatch")
            closure = float(row["maximum_closure_error"])
            if not math.isfinite(closure) or closure > 1e-12 \
                    or closure != reconstructed_current["maximum_closure_error"]:
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
        elif row.get("record_type") == "mic_actor_checkpoint_inventory":
            step = int(row.get("global_step", -1))
            checkpoint = Path(row.get("checkpoint_path", ""))
            shards = row.get("model_shards", [])
            current = []
            for item in shards:
                path = checkpoint / str(item.get("path", ""))
                if not path.is_file():
                    raise ValueError("MIC_NO_GO: actor checkpoint inventory file absent")
                current.append({
                    "path": item["path"], "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
            if step not in (5, 10, 15, 20, 25) \
                    or len(current) != 2 or current != shards \
                    or sha256_json(current) != row.get("model_shards_sha256") \
                    or row.get("git_commit") != gate_values["p0"].get("git_commit") \
                    or row.get("run_id") != gate_values["p0"].get("run_id"):
                raise ValueError("MIC_NO_GO: actor checkpoint inventory binding mismatch")
            checkpoint_steps.append(step)
    if args.target_step not in steps:
        raise ValueError(f"MIC_NO_GO: target step {args.target_step} absent from MIC ledger")
    if sorted(steps) != sorted(gradient_steps):
        raise ValueError("MIC_NO_GO: role-specific actual gradient ledger coverage differs")
    if set(health_by_step) != set(steps) \
            or any("actual_role_gradient" not in health_by_step[step] for step in steps):
        raise ValueError("MIC_NO_GO: on-policy mechanism health coverage differs")
    expected_checkpoint_steps = [step for step in sorted(set(steps)) if step % 5 == 0]
    if checkpoint_steps and sorted(checkpoint_steps) != expected_checkpoint_steps:
        raise ValueError("MIC_NO_GO: actor checkpoint inventory anchor coverage differs")
    if gate_values["p0"].get("requires_training_checkpoint_inventory") is True \
            and sorted(checkpoint_steps) != expected_checkpoint_steps:
        raise ValueError("MIC_NO_GO: required training-time actor checkpoint inventory absent")
    weight_rows = [json.loads(line) for line in Path(args.weight_ledger).read_text(
        encoding="utf-8").splitlines() if line]
    weight_failures = validate_jsonl_chain(weight_rows)
    if weight_failures:
        raise ValueError("MIC_NO_GO: weight-sync ledger chain corrupted: " + ";".join(weight_failures))
    post_sync_rows = [
        row for row in weight_rows if row.get("sync_kind") == "post_actor_update"
    ]
    expected_weight_binding = {
        "git_commit": gate_values["p0"].get("git_commit"),
        "run_id": gate_values["p0"].get("run_id"),
    }
    if any(not value for value in expected_weight_binding.values()):
        raise ValueError("MIC_NO_GO: P0 weight-sync identity binding missing")
    for row in post_sync_rows:
        if any(row.get(key) != value for key, value in expected_weight_binding.items()):
            raise ValueError("MIC_NO_GO: actor/vLLM weight-sync run binding mismatch")
    unknown_post_sync_types = sorted({
        str(row.get("record_type")) for row in post_sync_rows
        if row.get("record_type") not in {"weight_sync_ack", "weight_sync_summary"}
    })
    if unknown_post_sync_types:
        raise ValueError(
            "MIC_NO_GO: unexpected actor/vLLM weight-sync record types: "
            + ",".join(unknown_post_sync_types)
        )
    ack_rows = [
        row for row in post_sync_rows if row.get("record_type") == "weight_sync_ack"
    ]
    summary_rows = [
        row for row in post_sync_rows if row.get("record_type") == "weight_sync_summary"
    ]
    sync_steps = [int(row["global_step"]) for row in summary_rows]
    ack_steps = [int(row["global_step"]) for row in ack_rows]
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    for step in sorted(set(steps)):
        step_acks = [row for row in ack_rows if int(row["global_step"]) == step]
        step_summaries = [row for row in summary_rows if int(row["global_step"]) == step]
        if len(step_acks) != 2 or len(step_summaries) != 1:
            raise ValueError(
                "MIC_NO_GO: actor/vLLM weight-sync evidence cardinality mismatch "
                f"at step {step}: acks={len(step_acks)}, summaries={len(step_summaries)}"
            )
        ranks = sorted(int(row.get("vllm_worker_rank", -1)) for row in step_acks)
        summary = step_summaries[0]
        if ranks != [0, 1] or summary.get("worker_ranks") != [0, 1]:
            raise ValueError(
                f"MIC_NO_GO: actor/vLLM weight-sync ranks mismatch at step {step}"
            )
        if int(summary.get("actor_version", -1)) != step:
            raise ValueError(
                f"MIC_NO_GO: actor/vLLM weight-sync summary version mismatch at step {step}"
            )
        effective_digests = set()
        master_digests = set()
        loaded_name_digests = set()
        loaded_counts = set()
        for ack in step_acks:
            if int(ack.get("actor_version", -1)) != step \
                    or int(ack.get("vllm_ack_version", -1)) != step:
                raise ValueError(
                    f"MIC_NO_GO: actor/vLLM weight-sync version mismatch at step {step}"
                )
            effective = str(ack.get("actor_rollout_sampled_tensor_digest", ""))
            master = str(ack.get("actor_master_sampled_tensor_digest", ""))
            vllm = str(ack.get("vllm_sampled_tensor_digest", ""))
            if sha_pattern.fullmatch(effective) is None \
                    or sha_pattern.fullmatch(master) is None \
                    or sha_pattern.fullmatch(vllm) is None \
                    or ack.get("actor_sampled_tensor_digest") != effective \
                    or vllm != effective:
                raise ValueError(
                    f"MIC_NO_GO: actor/vLLM weight-sync acknowledgement mismatch at step {step}"
                )
            if ack.get("weight_transfer_format") != "dtensor":
                raise ValueError(
                    f"MIC_NO_GO: actor/vLLM weight-sync transfer format mismatch at step {step}"
                )
            loaded_count = int(ack.get("loaded_parameter_count", -1))
            model_count = int(ack.get("model_parameter_count", -1))
            loaded_names = str(ack.get("loaded_parameter_names_sha256", ""))
            model_names = str(ack.get("model_parameter_names_sha256", ""))
            audited_parameters = sorted(ack.get("audited_loaded_parameters") or [])
            sampled_dtypes = sorted((ack.get("sampled_parameter_dtypes") or {}).keys())
            if loaded_count <= 0 or loaded_count != model_count \
                    or sha_pattern.fullmatch(loaded_names) is None \
                    or loaded_names != model_names \
                    or not audited_parameters or sampled_dtypes != audited_parameters:
                raise ValueError(
                    f"MIC_NO_GO: actor/vLLM weight-sync load coverage mismatch at step {step}"
                )
            effective_digests.add(effective)
            master_digests.add(master)
            loaded_name_digests.add(loaded_names)
            loaded_counts.add(loaded_count)
        if len(effective_digests) != 1 or len(master_digests) != 1 \
                or len(loaded_name_digests) != 1 or len(loaded_counts) != 1 \
                or summary.get("sampled_tensor_digest") not in effective_digests \
                or summary.get("actor_master_sampled_tensor_digest") not in master_digests:
            raise ValueError(
                f"MIC_NO_GO: actor/vLLM weight-sync summary mismatch at step {step}"
            )
    if sorted(set(steps)) != sorted(set(sync_steps)) \
            or sorted(steps * 2) != sorted(ack_steps):
        raise ValueError("MIC_NO_GO: weight-sync step coverage differs from MIC updates")
    report = {"schema": SCHEMA, "status": "PASS", "decision": f"MIC_T{args.target_step}_AUDIT_PASS",
              "gate_sha256": checked, "ledger_tail_sha256": previous, "mic_steps": steps,
              "gradient_steps": gradient_steps, "on_policy_health": health_by_step}
    report["checkpoint_inventory_steps"] = checkpoint_steps
    report["weight_sync_ledger_sha256"] = sha256_file(args.weight_ledger)
    if args.output:
        write_new(args.output, report)
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from recurrent.research.s128_hotpot_metrics import paired_descriptive_summary

    rows = [json.loads(line) for line in Path(args.predictions).read_text(
        encoding="utf-8").splitlines() if line]
    if len(rows) != 128:
        raise ValueError("MIC_NO_GO: method fixed-S128 denominator is not 128")
    keys = [str(row.get("stable_key", "")) for row in rows]
    if any(not key for key in keys) or len(set(keys)) != 128:
        raise ValueError("MIC_NO_GO: method fixed-S128 stable IDs are invalid")
    scored = [{"stable_key": row["stable_key"],
               **score_terminal_output(row["output"], row["ground_truth"])} for row in rows]
    aggregate = summarize_fixed_s128(scored)
    baseline = import_baseline(argparse.Namespace(
        p0=args.p0, manifest=str(REPO / "manifests/h20/qwen25_7b_mic_seed2026.json"),
        output=args.baseline, verify_existing=True,
    ))
    inventory_path = Path(os.environ.get("MEMAGENT_MIC_BASELINE_INVENTORY", ""))
    if baseline.get("status") != "PASS" \
            or baseline.get("decision") != "MIC_BASELINE_IMPORT_PASS" \
            or not inventory_path.is_file() \
            or sha256_file(inventory_path) != baseline.get("inventory_sha256"):
        raise ValueError("MIC_NO_GO: baseline import binding mismatch")
    interface = f"Original{args.step}"
    original = baseline["interfaces"][interface]["aggregate"]
    if sha256_json(keys) != baseline["interfaces"][interface]["stable_key_inventory_sha256"]:
        raise ValueError("MIC_NO_GO: method S128 identities differ from frozen Original")
    f1_delta_pp = 100.0 * (float(aggregate["token_f1"]) - float(original["token_f1"]))
    original_path = Path(baseline["interfaces"][interface]["path"])
    if sha256_file(original_path) != baseline["interfaces"][interface]["sha256"]:
        raise ValueError("MIC_NO_GO: frozen Original normalized rows changed")
    original_rows = [json.loads(line) for line in original_path.read_text(
        encoding="utf-8"
    ).splitlines() if line]
    original_scored = [{"stable_key": row["stable_key"],
                        **score_terminal_output(row["output"], row["ground_truth"])}
                       for row in original_rows]
    if sha256_json(summarize_fixed_s128(original_scored)) != sha256_json(original):
        raise ValueError("MIC_NO_GO: frozen Original aggregate changed")
    paired = paired_descriptive_summary(original_scored, scored)
    delta_pp = {
        name: 100.0 * (float(aggregate[name]) - float(original[name]))
        for name in ("normalized_exact_match", "token_f1", "format_success")
    }
    decision = f"MIC_T{args.step}_EVAL_PASS"
    evaluation_root = Path(args.predictions).resolve().parent
    report = {
        "schema": SCHEMA, "status": "PASS",
        "decision": decision, "step": args.step,
        "prediction_sha256": sha256_file(args.predictions), "aggregate": aggregate,
        "evaluation_root": str(evaluation_root),
        "original_aggregate": original, "token_f1_delta_pp": f1_delta_pp,
        "delta_percentage_points": delta_pp, "paired_descriptive": paired,
    }
    if getattr(args, "verify_existing", False):
        existing = read_json(args.output)
        if sha256_json(existing) != sha256_json(report):
            raise ValueError("MIC_NO_GO: evaluation certificate differs from recomputation")
    else:
        write_new(args.output, report)
    return report


def final_eval_audit(args: argparse.Namespace) -> dict[str, Any]:
    baseline = import_baseline(argparse.Namespace(
        p0=args.p0, manifest=str(REPO / "manifests/h20/qwen25_7b_mic_seed2026.json"),
        output=args.baseline, verify_existing=True,
    ))
    health_root = Path(args.health_root)
    eval_root = Path(args.eval_root)
    replayed_health = audit(argparse.Namespace(
        p0=args.p0, e0=args.e0, paper_review=args.paper_review,
        ledger=args.mic_ledger, weight_ledger=args.weight_ledger,
        target_step=25, output=None,
    ))
    if replayed_health.get("status") != "PASS" \
            or replayed_health.get("decision") != "MIC_T25_AUDIT_PASS":
        raise ValueError("MIC_NO_GO: final mechanism audit replay failed")
    curve = {}
    for step in (5, 10, 15, 20, 25):
        health_path = health_root / f"t{step}_audit.json"
        eval_path = health_root / f"t{step}_eval.json"
        health = read_json(health_path)
        existing_evaluation = read_json(eval_path)
        anchor_root = Path(existing_evaluation.get("evaluation_root", "")).resolve()
        expected_parent = (eval_root / f"eval_t{step}_attempts").resolve()
        if anchor_root.parent != expected_parent \
                or re.fullmatch(r"attempt_[0-9]{4}", anchor_root.name) is None:
            raise ValueError(f"MIC_NO_GO: final audit T{step} attempt binding")
        prepared = prepare_eval(argparse.Namespace(
            generations=str(anchor_root / "raw" / f"{step}.jsonl"),
            validation=baseline["validation_path"],
            identity_source=baseline["interfaces"][f"Original{step}"]["path"],
            execution_summary=str(anchor_root / "execution_summary.json"),
            training_audit=str(health_path), baseline=args.baseline, p0=args.p0,
            checkpoint=str(Path(args.output_root) / f"global_step_{step}"), step=step,
            checkpoint_authority=args.checkpoint_authority,
            checkpoint_authority_certificate=args.checkpoint_authority_certificate,
            output_root=args.output_root, weight_ledger=args.weight_ledger,
            mic_ledger=args.mic_ledger,
            output=str(anchor_root / "predictions.jsonl"),
            report=str(anchor_root / "prepare.json"), verify_existing=True,
        ))
        evaluation = evaluate(argparse.Namespace(
            predictions=str(anchor_root / "predictions.jsonl"),
            baseline=args.baseline, p0=args.p0, step=step, output=str(eval_path),
            verify_existing=True,
        ))
        if health.get("status") != "PASS" \
                or health.get("decision") != f"MIC_T{step}_AUDIT_PASS" \
                or evaluation.get("status") != "PASS" \
                or evaluation.get("decision") != f"MIC_T{step}_EVAL_PASS" \
                or int(evaluation.get("step", -1)) != step:
            raise ValueError(f"MIC_NO_GO: final audit T{step} evidence is incomplete")
        curve[str(step)] = {
            "method": evaluation["aggregate"],
            "original": evaluation["original_aggregate"],
            "delta_percentage_points": evaluation["delta_percentage_points"],
            "paired_descriptive": evaluation["paired_descriptive"],
            "prepare_sha256": sha256_file(anchor_root / "prepare.json"),
            "checkpoint_authority_sha256": prepared["checkpoint_authority_sha256"],
            "health_sha256": sha256_file(health_path),
            "evaluation_sha256": sha256_file(eval_path),
        }
    t25_delta = float(curve["25"]["delta_percentage_points"]["token_f1"])
    primary_pass = t25_delta >= 2.0
    report = {
        "schema": SCHEMA, "status": "PASS", "decision": "MIC_FINAL_AUDIT_PASS",
        "scientific_decision": (
            "MIC_PRIMARY_PERFORMANCE_CRITERION_PASS" if primary_pass
            else "MIC_PRIMARY_PERFORMANCE_CRITERION_NO_GO"
        ),
        "primary_criterion": "Method T25 token-F1 minus Original T25 >= 2 percentage points",
        "primary_criterion_pass": primary_pass,
        "t25_token_f1_delta_percentage_points": t25_delta,
        "alternative_criterion_status": (
            "NOT_EVALUABLE: certified Original writer-gradient variance was never collected"
        ),
        "curve": curve,
        "baseline_sha256": sha256_file(args.baseline),
        "checkpoint_authority_certificate_sha256": sha256_file(
            args.checkpoint_authority_certificate
        ),
        "replayed_mechanism_audit": replayed_health,
        "mechanism_status": "METHOD_ON_POLICY_HEALTH_PASS_E1_FEASIBILITY_STILL_PENDING",
        "claim_boundaries": {
            "causal": False, "population_inference": False,
            "dataset": "curated frozen S128", "original_rerun": False,
            "dense_training_reward_used_as_performance": False,
        },
    }
    write_new(args.output, report)
    return report


def _verify_training_weight_prefix(
        weight_ledger_path: str | Path, training_audit: dict[str, Any],
        step: int, weight_snapshot: dict[str, Any],
        expected_git_commit: str, expected_run_id: str) -> str:
    from recurrent.research.gate_a_execution import validate_jsonl_chain

    target_sha = str(training_audit.get("weight_sync_ledger_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", target_sha) is None:
        raise ValueError("MIC_NO_GO: training audit weight-ledger binding missing")
    digest = hashlib.sha256()
    prefix_lines: list[bytes] | None = None
    accumulated: list[bytes] = []
    for line in Path(weight_ledger_path).read_bytes().splitlines(keepends=True):
        accumulated.append(line)
        digest.update(line)
        if digest.hexdigest() == target_sha:
            prefix_lines = list(accumulated)
            break
    if prefix_lines is None:
        raise ValueError("MIC_NO_GO: authenticated training weight-ledger prefix absent")
    rows = [json.loads(line) for line in b"".join(prefix_lines).decode("utf-8").splitlines()
            if line]
    failures = validate_jsonl_chain(rows)
    if failures:
        raise ValueError("MIC_NO_GO: authenticated training weight-ledger prefix corrupted")
    if any(row.get("git_commit") != expected_git_commit \
           or row.get("run_id") != expected_run_id for row in rows):
        raise ValueError("MIC_NO_GO: authenticated training weight-ledger cross-run replay")
    summaries = [
        row for row in rows
        if row.get("record_type") == "weight_sync_summary"
        and row.get("sync_kind") == "post_actor_update"
        and int(row.get("global_step", -1)) == step
    ]
    if len(summaries) != 1:
        raise ValueError("MIC_NO_GO: target training weight-sync summary missing")
    target = summaries[0]
    if weight_snapshot.get("actor_master_sampled_tensor_digest") != target.get(
        "actor_master_sampled_tensor_digest"
    ) or weight_snapshot.get("actor_rollout_sampled_tensor_digest") != target.get(
        "sampled_tensor_digest"
    ) or weight_snapshot.get("vllm_sampled_tensor_digest") != target.get(
        "sampled_tensor_digest"
    ):
        raise ValueError("MIC_NO_GO: evaluated checkpoint differs from training weight-sync evidence")
    return target_sha


def prepare_eval(args: argparse.Namespace) -> dict[str, Any]:
    from recurrent.research.stable_eval_identity import canonical_sha256

    summary = read_json(args.execution_summary)
    checkpoint = Path(args.checkpoint).resolve()
    identity_path = Path(args.identity_source).resolve()
    training_audit_path = Path(args.training_audit).resolve()
    training_audit = read_json(training_audit_path)
    p0_value = read_json(args.p0)
    certified_baseline = import_baseline(argparse.Namespace(
        p0=args.p0, manifest=str(REPO / "manifests/h20/qwen25_7b_mic_seed2026.json"),
        output=args.baseline, verify_existing=True,
    ))
    validation_path = Path(args.validation).resolve()
    expected_identity_path = Path(
        certified_baseline["interfaces"][f"Original{args.step}"]["path"]
    ).resolve()
    protocol_evidence = summary.get("generation_protocol_evidence", {})
    reconstructed_reward_path = Path(
        protocol_evidence.get("original_protocol_reconstruction_path", "")
    ).resolve()
    expected_reward_path = (
        REPO / "recurrent/research/hotpotqa_dense_reward.py"
    ).resolve()
    if validation_path != Path(certified_baseline["validation_path"]).resolve() \
            or sha256_file(validation_path) != certified_baseline["validation_sha256"] \
            or identity_path != expected_identity_path:
        raise ValueError("MIC_NO_GO: certified S128 evaluation authority mismatch")
    if summary.get("schema") != "memagent.mic.eval.v1" \
            or summary.get("record_type") != "mic_read_only_execution_summary" \
            or int(summary.get("global_step", -1)) != int(args.step) \
            or summary.get("run_id") != os.environ.get("MEMAGENT_MIC_RUN_ID") \
            or summary.get("evaluation_git_commit") != git("rev-parse", "HEAD") \
            or Path(summary.get("checkpoint_source", "")).resolve() != checkpoint \
            or Path(summary.get("identity_path", "")).resolve() != identity_path \
            or summary.get("identity_sha256") != sha256_file(identity_path) \
            or Path(summary.get("generation_path", "")).resolve() != Path(
                args.generations
            ).resolve() \
            or summary.get("generation_sha256") != sha256_file(args.generations) \
            or summary.get("training_audit_sha256") != sha256_file(training_audit_path) \
            or protocol_evidence.get("method_generation_protocol_sha256") \
            != certified_baseline["shared_generation_protocol_sha256"] \
            or protocol_evidence.get("original_generation_protocol_sha256") \
            != certified_baseline["shared_generation_protocol_sha256"] \
            or protocol_evidence.get("projection_schema") \
            != "repository-relative-reward-code-sha256-v1" \
            or protocol_evidence.get("reward_code_sha256") \
            != certified_baseline["original_reward_code_sha256"] \
            or reconstructed_reward_path != expected_reward_path \
            or not reconstructed_reward_path.is_file() \
            or sha256_file(reconstructed_reward_path) \
            != certified_baseline["original_reward_code_sha256"] \
            or summary.get("actor_update_calls") != 0 \
            or summary.get("validation_only") is not True:
        raise ValueError("MIC_NO_GO: read-only evaluation summary binding mismatch")
    if training_audit.get("status") != "PASS" \
            or training_audit.get("decision") != f"MIC_T{args.step}_AUDIT_PASS" \
            or int(args.step) not in [int(value) for value in training_audit.get("mic_steps", [])] \
            or training_audit.get("gate_sha256", {}).get("p0") != sha256_file(args.p0):
        raise ValueError("MIC_NO_GO: training audit target binding mismatch")
    checkpoint_authority = verify_checkpoint_authority(
        args.checkpoint_authority, args.checkpoint_authority_certificate,
        args.output_root, int(args.step), args.p0, args.mic_ledger,
    )
    inventory = []
    for item in summary.get("checkpoint_inventory", []):
        path = checkpoint / str(item.get("path", ""))
        if not path.is_file():
            raise ValueError("MIC_NO_GO: evaluated checkpoint inventory is incomplete")
        inventory.append({"path": item["path"], "size": path.stat().st_size,
                          "sha256": sha256_file(path)})
    if inventory != summary.get("checkpoint_inventory") or len(inventory) != 2:
        raise ValueError("MIC_NO_GO: evaluated checkpoint inventory changed")
    acknowledgements = summary.get("actor_checkpoint_load_acks", [])
    if sorted(int(row.get("rank", -1)) for row in acknowledgements) != [0, 1] \
            or any(row.get("optimizer_loaded") is not False \
                   or row.get("lr_scheduler_loaded") is not False \
                   or row.get("rng_loaded") is not False \
                   or row.get("dataloader_loaded") is not False
                   for row in acknowledgements):
        raise ValueError("MIC_NO_GO: actor-only checkpoint load acknowledgement mismatch")
    before = summary.get("weight_snapshot_before", {})
    after = summary.get("weight_snapshot_after", {})
    for field in (
        "actor_master_sampled_tensor_digest", "actor_rollout_sampled_tensor_digest",
        "vllm_sampled_tensor_digest", "worker_ranks", "worker_evidence",
    ):
        if before.get(field) != after.get(field):
            raise ValueError("MIC_NO_GO: actor/vLLM state changed during evaluation")
    if after.get("vllm_pre_sync_sampled_tensor_digest") != before.get(
        "vllm_sampled_tensor_digest"
    ):
        raise ValueError("MIC_NO_GO: vLLM state drifted during evaluation")
    weight_prefix_sha = _verify_training_weight_prefix(
        args.weight_ledger, training_audit, int(args.step), before,
        str(p0_value.get("git_commit", "")), str(p0_value.get("run_id", "")),
    )

    generated = [json.loads(line) for line in Path(args.generations).read_text(
        encoding="utf-8").splitlines() if line]
    baseline = [json.loads(line) for line in Path(args.identity_source).read_text(
        encoding="utf-8").splitlines() if line]
    frame = _load_parquet_rows(args.validation)
    if len(generated) != 128 or len(baseline) != 128 or len(frame) < 128:
        raise ValueError("MIC_NO_GO: evaluation producer inputs lack S128 coverage")
    baseline.sort(key=lambda row: int(row.get("source_order_index", row.get("source_repeated_row", -1))))
    generated_by_key = {str(row.get("stable_key", "")): row for row in generated}
    baseline_keys = {str(row["stable_key"]) for row in baseline}
    if len(generated_by_key) != 128 or set(generated_by_key) != baseline_keys:
        raise ValueError("MIC_NO_GO: Method generation identity set differs from frozen S128")
    rows = []
    for index, identity in enumerate(baseline):
        generation = generated_by_key[str(identity["stable_key"])]
        if int(generation.get("source_order_index", -1)) != index \
                or int(generation.get("raw_row_position", -1)) != int(
                    identity.get("raw_row_position", -1)
                ) \
                or generation.get("source_question_sha256") != identity.get("source_question_sha256") \
                or generation.get("source_context_sha256") != identity.get("source_context_sha256"):
            raise ValueError(f"MIC_NO_GO: Method generation identity binding mismatch at row {index}")
        raw_position = int(identity["raw_row_position"])
        if raw_position < 0 or raw_position >= len(frame):
            raise ValueError(f"MIC_NO_GO: validation raw row {raw_position} is invalid")
        source = frame[raw_position]
        ground_truth = _parquet_ground_truth(source, row=index)
        if canonical_sha256(ground_truth) != canonical_sha256(identity.get("ground_truth")):
            raise ValueError(f"MIC_NO_GO: frozen ground truth mismatch at row {index}")
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
        rows.append({"source_order_index": index, "raw_row_position": raw_position,
                     "stable_key": stable_key,
                     "output": generation["output"], "ground_truth": ground_truth})
    target = Path(args.output).resolve()
    payload = "".join(canonical_json(row) + "\n" for row in rows)
    prediction_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    report = {
        "schema": SCHEMA, "status": "PASS", "decision": "MIC_S128_PREPARE_PASS",
        "step": int(args.step), "rows": 128,
        "predictions_path": str(target), "predictions_sha256": prediction_sha,
        "generations_path": str(Path(args.generations).resolve()),
        "generations_sha256": sha256_file(args.generations),
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "identity_path": str(identity_path), "identity_sha256": sha256_file(identity_path),
        "execution_summary_path": str(Path(args.execution_summary).resolve()),
        "execution_summary_sha256": sha256_file(args.execution_summary),
        "training_audit_path": str(training_audit_path),
        "training_audit_sha256": sha256_file(training_audit_path),
        "training_weight_ledger_path": str(Path(args.weight_ledger).resolve()),
        "training_weight_ledger_prefix_sha256": weight_prefix_sha,
        "checkpoint_path": str(checkpoint),
        "checkpoint_authority_sha256": checkpoint_authority[
            "checkpoint_authority_sha256"
        ],
        "generation_protocol_evidence": protocol_evidence,
    }
    report_path = Path(args.report).resolve()
    if getattr(args, "verify_existing", False):
        if not target.is_file() or sha256_file(target) != prediction_sha \
                or sha256_json(read_json(report_path)) != sha256_json(report):
            raise ValueError("MIC_NO_GO: prepared evaluation evidence differs from recomputation")
    else:
        if target.exists():
            raise FileExistsError(f"MIC_NO_GO: refusing to overwrite {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        write_new(report_path, report)
    return report


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
    sub.choices["import-baseline"].add_argument("--p0", required=True)
    sub.choices["import-baseline"].add_argument(
        "--curve-authority", default=str(CURVE_AUTHORITY)
    )
    sub.choices["import-baseline"].add_argument("--verify-existing", action="store_true")
    child = sub.add_parser("materialize-baseline")
    child.add_argument("--p0", required=True)
    child.add_argument("--curve-report", required=True)
    child.add_argument("--curve-resolved", required=True)
    child.add_argument("--search-root", required=True)
    child.add_argument("--validation", required=True)
    child.add_argument("--curve-authority", default=str(CURVE_AUTHORITY))
    child.add_argument("--output-root", required=True)
    child.add_argument("--output", required=True)
    child = sub.add_parser("materialize-checkpoint-authority")
    child.add_argument("--authority", default=str(CHECKPOINT_AUTHORITY))
    child.add_argument("--p0", required=True)
    child.add_argument("--output-root", required=True)
    child.add_argument("--ledger", required=True)
    child.add_argument("--output", required=True)
    child = sub.add_parser("audit")
    for name in ("p0", "e0", "paper-review", "ledger", "weight-ledger"):
        child.add_argument("--" + name, required=True)
    child.add_argument("--target-step", type=int, choices=(5, 10, 15, 20, 25), required=True)
    child.add_argument("--output")
    child = sub.add_parser("evaluate")
    child.add_argument("--predictions", required=True)
    child.add_argument("--baseline", required=True)
    child.add_argument("--p0", required=True)
    child.add_argument("--step", type=int, choices=(5, 10, 15, 20, 25), required=True)
    child.add_argument("--output", required=True)
    child.add_argument("--verify-existing", action="store_true")
    child = sub.add_parser("prepare-eval")
    child.add_argument("--generations", required=True)
    child.add_argument("--validation", required=True)
    child.add_argument("--identity-source", required=True)
    child.add_argument("--execution-summary", required=True)
    child.add_argument("--training-audit", required=True)
    child.add_argument("--baseline", required=True)
    child.add_argument("--p0", required=True)
    child.add_argument("--checkpoint", required=True)
    child.add_argument("--checkpoint-authority", default=str(CHECKPOINT_AUTHORITY))
    child.add_argument("--checkpoint-authority-certificate", required=True)
    child.add_argument("--output-root", required=True)
    child.add_argument("--weight-ledger", required=True)
    child.add_argument("--mic-ledger", required=True)
    child.add_argument("--step", type=int, choices=(5, 10, 15, 20, 25), required=True)
    child.add_argument("--output", required=True)
    child.add_argument("--report", required=True)
    child.add_argument("--verify-existing", action="store_true")
    child = sub.add_parser("final-eval-audit")
    child.add_argument("--baseline", required=True)
    child.add_argument("--p0", required=True)
    child.add_argument("--health-root", required=True)
    child.add_argument("--eval-root", required=True)
    child.add_argument("--output-root", required=True)
    child.add_argument("--checkpoint-authority", default=str(CHECKPOINT_AUTHORITY))
    child.add_argument("--checkpoint-authority-certificate", required=True)
    child.add_argument("--weight-ledger", required=True)
    child.add_argument("--mic-ledger", required=True)
    child.add_argument("--e0", required=True)
    child.add_argument("--paper-review", required=True)
    child.add_argument("--output", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    functions = {"p0": p0, "e0": e0, "e1": e1, "import-baseline": import_baseline,
                 "materialize-baseline": materialize_baseline,
                 "materialize-checkpoint-authority": materialize_checkpoint_authority,
                 "audit": audit, "evaluate": evaluate, "prepare-eval": prepare_eval,
                 "final-eval-audit": final_eval_audit}
    try:
        result = functions[args.command](args)
    except Exception as exc:
        print(f"MIC_NO_GO:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
