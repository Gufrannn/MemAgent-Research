#!/usr/bin/env python3
"""Fail-closed CPU preflight for one fresh or resumed RWWPO-2 cell/seed."""
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.h20.verify_rwwpo2_release_tests import verify_release_test_receipt


EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"
CELLS = {
    "D": ("original_tokenwise", "none", 400),
    "C": ("original_tokenwise", "feasible_backtracking", 400),
    "E": ("per_write_joint", "feasible_backtracking", 400),
    "B": ("whole_prefix", "feasible_backtracking", 400),
    "A": ("whole_prefix", "hard_rollback", 50),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(path: str, *, decision: str, commit: str) -> dict:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"invalid {decision} receipt path")
    row = json.loads(source.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != decision or row.get("git_commit") != commit:
        raise ValueError(f"invalid {decision} receipt")
    return {**row, "report_sha256": declared}


def source_training(original: dict) -> dict:
    return dict(original.get("training", original))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-contract", required=True)
    parser.add_argument("--resolved-contract-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--gpu-pair", required=True)
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--experiment-seed", type=int, required=True)
    parser.add_argument("--target-round", type=int, required=True)
    parser.add_argument("--phase", choices=("fresh", "resume"), required=True)
    parser.add_argument("--e0", required=True)
    parser.add_argument("--data-boundary-audit", required=True)
    parser.add_argument("--base-protocol-audit", required=True)
    parser.add_argument("--release-test-receipt", required=True)
    parser.add_argument("--release-test-receipt-sha256", required=True)
    parser.add_argument("--original-resolved-manifest", required=True)
    parser.add_argument("--original-resolved-sha256", required=True)
    parser.add_argument("--lineage-parent")
    parser.add_argument("--resume-round", type=int)
    parser.add_argument("--r50-program-gate")
    parser.add_argument("--r50-program-gate-sha256")
    parser.add_argument("--confirmation-seal")
    parser.add_argument("--confirmation-seal-sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or re.fullmatch(r"[0-9a-f]{40}", head) is None \
            or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:checkout")
    try:
        pair = [int(value) for value in args.gpu_pair.split(",")]
    except ValueError as error:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:GPU pair") from error
    if len(pair) != 2 or pair != sorted(set(pair)):
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:GPU pair must be canonical ascending")

    if Path(args.resolved_contract).is_symlink() \
            or Path(args.original_resolved_manifest).is_symlink():
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:source symlink")
    resolved_path = Path(args.resolved_contract).resolve()
    if sha256_file(resolved_path) != args.resolved_contract_sha256:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:resolved contract SHA")
    resolved = receipt(
        str(resolved_path), decision="RWWPO2_RESOLVED_CONTRACT_PASS", commit=head
    )
    manifest = resolved["manifest"]
    if manifest.get("program") != "RWWPO-2" or manifest.get("branch") != EXPECTED_BRANCH:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:program identity")
    if sha256_file(Path(manifest_path := resolved["source_manifest_path"])) != \
            resolved["source_manifest_sha256"]:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:source manifest drift")
    if Path(manifest_path).resolve() != (ROOT / "manifests/h20/"
            "qwen25_7b_rwwpo2_r400_k2_seed2026.json").resolve():
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:noncanonical manifest")
    if Path(manifest_path).is_symlink() \
            or Path(resolved.get("source_manifest_schema_path", "")).is_symlink():
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:manifest/schema symlink")
    schema_path = Path(resolved.get("source_manifest_schema_path", "")).resolve()
    if schema_path != (ROOT / "rwwpo2_experiment_manifest.schema.json").resolve() \
            or sha256_file(schema_path) != resolved.get("source_manifest_schema_sha256"):
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:manifest schema drift")
    try:
        release_tests = verify_release_test_receipt(
            args.release_test_receipt,
            receipt_sha256=args.release_test_receipt_sha256,
            expected_commit=head,
            manifest_path=manifest_path,
            manifest_sha256=resolved["source_manifest_sha256"],
            work_root=os.environ.get("RWWPO_WORK_ROOT", ""),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:release tests:" + str(error)) from error
    receipt(args.e0, decision="RWWPO2_E0_PASS", commit=head)
    data_audit = receipt(
        args.data_boundary_audit,
        decision="RWWPO2_DATA_BOUNDARY_AUDIT_PASS",
        commit=head,
    )
    if data_audit.get("direct_leakage") is not False \
            or data_audit.get("s128_role") != "ADAPTIVE_DEVELOPMENT_NOT_BLIND_FINAL":
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:data boundary")
    base_audit = receipt(
        args.base_protocol_audit,
        decision="RWWPO2_BASE_PROTOCOL_AUDIT_PASS",
        commit=head,
    )

    objective, controller, maximum_round = CELLS[args.cell]
    declared = manifest["method_cells"][args.cell]
    if declared["objective_variant"] != objective \
            or declared["controller_variant"] != controller:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:cell identity")
    if args.target_round not in (50, 400) or args.target_round > maximum_round:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:target round")
    approved_seeds = set(manifest["training"]["confirmatory_seed_values"])
    if args.experiment_seed not in approved_seeds:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:unregistered seed")
    if int(manifest["method"]["inner_transactions_per_round"]) != 2 \
            or int(manifest["method"]["optimizer_minibatches_per_inner_transaction"]) != 1 \
            or int(manifest["method"]["optimizer_steps_per_inner_transaction"]) != 1 \
            or int(manifest["training"]["ppo_epochs"]) != 2 \
            or int(manifest["training"]["ppo_mini_batch_size"]) != int(
                manifest["training"]["train_batch_size"]
            ):
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:K2 transaction mapping")
    expected_actual_loss = {
        "loss_agg_mode": "token-mean", "clip_ratio": 0.2,
        "clip_ratio_low": 0.2, "clip_ratio_high": 0.2,
        "clip_ratio_c": 3.0, "use_kl_loss": True,
        "kl_loss_type": "low_var_kl", "kl_loss_coefficient": 0.001,
        "entropy_coefficient": 0.0,
    }
    actual_loss_drift = {
        key: [manifest["training"].get(key), expected]
        for key, expected in expected_actual_loss.items()
        if manifest["training"].get(key) != expected
    }
    if actual_loss_drift:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:actual-loss contract:" +
                         json.dumps(actual_loss_drift, sort_keys=True))
    schedule = resolved["proposal_schedule"]
    if schedule != {"kind": "constant_with_linear_warmup", "base_lr": 1e-6,
                     "warmup_proposals": 2, "total_proposals": 800}:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:proposal schedule")
    thresholds = resolved["numeric_thresholds"]
    if set(thresholds) != {
        "tau_theta", "tau_logprob", "tau_gradient", "tau_coefficient"
    } \
            or any(not math.isfinite(float(value)) or float(value) <= 0
                   for value in thresholds.values()):
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:numeric thresholds")
    if float(resolved.get("behavior_coefficient_tolerance", -1)) != float(
            thresholds["tau_coefficient"]) \
            or float(resolved.get("behavior_gradient_tolerance", -1)) != float(
                thresholds["tau_gradient"]):
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:numeric tolerance binding")

    original_path = Path(args.original_resolved_manifest).resolve()
    if sha256_file(original_path) != args.original_resolved_sha256:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:Original resolved SHA")
    if base_audit.get("original_resolved_sha256") != args.original_resolved_sha256 \
            or base_audit.get("train_sha256") != manifest["data"]["train_sha256"] \
            or base_audit.get("model_id") != manifest["model"]["id"] \
            or base_audit.get("model_revision") != manifest["model"]["revision"]:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:base protocol receipt binding")
    original = source_training(json.loads(original_path.read_text(encoding="utf-8")))
    training = manifest["training"]
    aliases = {
        "train_batch_size": "train_batch_size",
        "rollout_n": "rollout_n",
        "ppo_mini_batch_size": "ppo_mini_batch_size",
        "chunk_size": "chunk_size",
        "max_chunks": "max_chunks",
        "max_prompt_length": "max_prompt_length",
        "max_response_length": "max_response_length",
        "learning_rate": "actor_learning_rate",
        "kl_loss_coefficient": "kl_loss_coefficient",
    }
    drift = {
        key: [training[key], original.get(source_key)]
        for key, source_key in aliases.items()
        if original.get(source_key) != training[key]
    }
    if drift:
        raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:Original protocol drift:" +
                         json.dumps(drift, sort_keys=True))

    parent = None
    if args.phase == "fresh":
        if args.lineage_parent is not None or args.resume_round not in (None, 0):
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:fresh lineage arguments")
        lineage_start_round = 1
    else:
        if not args.lineage_parent or args.resume_round is None \
                or args.resume_round <= 0 or args.resume_round % 10 != 0 \
                or args.resume_round >= args.target_round:
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:resume lineage arguments")
        parent = receipt(
            args.lineage_parent, decision="RWWPO2_LINEAGE_PARENT_PASS", commit=head
        )
        if parent.get("cell") != args.cell \
                or int(parent.get("experiment_seed", -1)) != args.experiment_seed \
                or int(parent.get("checkpoint_round", -1)) != args.resume_round \
                or parent.get("resolved_contract_file_sha256") != \
                args.resolved_contract_sha256 \
                or parent.get("resolved_contract_report_sha256") != \
                resolved["report_sha256"] \
                or parent.get("source_manifest_sha256") != \
                resolved["source_manifest_sha256"]:
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:lineage parent identity")
        lineage_start_round = args.resume_round + 1

    r50_gate = None
    confirmation_seal = None
    if args.target_round == 400:
        if args.cell == "A" or not args.r50_program_gate or not args.confirmation_seal \
                or not args.r50_program_gate_sha256 \
                or not args.confirmation_seal_sha256:
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:R400 requires R50 program gate")
        if Path(args.r50_program_gate).is_symlink() \
                or Path(args.confirmation_seal).is_symlink():
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:R400 evidence symlink")
        if sha256_file(Path(args.r50_program_gate).resolve()) != \
                args.r50_program_gate_sha256:
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:R50 gate file SHA")
        r50_gate = receipt(
            args.r50_program_gate,
            decision="RWWPO2_R50_MECHANISM_GATE_PASS",
            commit=head,
        )
        if r50_gate.get("s128_consumed") is not False \
                or r50_gate.get("program_version") != "rwwpo2-k2":
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:R50 gate identity")
        confirmation_seal = receipt(
            args.confirmation_seal,
            decision="RWWPO2_CONFIRMATION_SEAL_PASS",
            commit=head,
        )
        if sha256_file(Path(args.confirmation_seal).resolve()) != \
                args.confirmation_seal_sha256:
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:confirmation seal file SHA")
        overlaps = confirmation_seal.get("overlaps", {})
        duplicates = confirmation_seal.get("duplicates", {})
        if confirmation_seal.get("manifest_sha256") != resolved["source_manifest_sha256"] \
                or int(confirmation_seal.get("row_count", 0)) < int(
                    manifest["performance"]["confirmation_minimum_examples"]
                ) \
                or set(overlaps) != {
                    "actor_training_content", "actor_training_root",
                    "adaptive_s128_content", "adaptive_s128_root",
                    "capture32_content", "capture32_root",
                } \
                or any(int(value) != 0 for value in overlaps.values()) \
                or set(duplicates) != {"content", "root"} \
                or any(int(value) != 0 for value in duplicates.values()) \
                or confirmation_seal.get("opened_for_training_or_selection") is not False:
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:confirmation seal identity")
        from recurrent.research.stable_eval_identity import validate_resolved_manifest
        raw_sealed_resolved = Path(confirmation_seal.get(
            "resolved_identity_manifest_path", ""
        ))
        sealed_resolved = raw_sealed_resolved.resolve()
        if raw_sealed_resolved.is_symlink() or not sealed_resolved.is_file() \
                or sha256_file(sealed_resolved) != \
                confirmation_seal.get("resolved_identity_manifest_sha256"):
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:sealed resolved identity bytes")
        checked_confirmation = validate_resolved_manifest(json.loads(
            sealed_resolved.read_text(encoding="utf-8")
        ))
        if checked_confirmation["eval_manifest_hash"] != confirmation_seal.get(
                "eval_manifest_hash") \
                or checked_confirmation.get("confirmation_binding", {}).get(
                    "seal_id") != confirmation_seal.get("seal_id"):
            raise SystemExit("RWWPO2_PREFLIGHT_NO_GO:sealed resolved identity")

    report = {
        "schema_version": "rwwpo2-preflight-v1",
        "status": "PASS",
        "decision": "RWWPO2_PREFLIGHT_PASS",
        "git_commit": head,
        "gpu_pair": pair,
        "cell": args.cell,
        "objective_variant": objective,
        "controller_variant": controller,
        "experiment_seed": args.experiment_seed,
        "phase": args.phase,
        "target_round": args.target_round,
        "resume_round": args.resume_round,
        "lineage_start_round": lineage_start_round,
        "lineage_parent_report_sha256": None if parent is None else parent["report_sha256"],
        "numeric_thresholds": thresholds,
        "behavior_coefficient_tolerance": resolved["behavior_coefficient_tolerance"],
        "behavior_gradient_tolerance": resolved["behavior_gradient_tolerance"],
        "resolved_contract_file_sha256": args.resolved_contract_sha256,
        "resolved_contract_report_sha256": resolved["report_sha256"],
        "source_manifest_sha256": resolved["source_manifest_sha256"],
        "original_resolved_sha256": args.original_resolved_sha256,
        "base_protocol_audit_report_sha256": base_audit["report_sha256"],
        "release_test_receipt_file_sha256": args.release_test_receipt_sha256,
        "release_test_receipt_report_sha256": release_tests["report_sha256"],
        "s128_consumed_by_training": False,
        "r50_program_gate_report_sha256": (
            None if r50_gate is None else r50_gate["report_sha256"]
        ),
        "r50_program_gate_file_sha256": (
            None if r50_gate is None else args.r50_program_gate_sha256
        ),
        "confirmation_seal_report_sha256": (
            None if confirmation_seal is None else confirmation_seal["report_sha256"]
        ),
        "confirmation_seal_file_sha256": (
            None if confirmation_seal is None else args.confirmation_seal_sha256
        ),
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
