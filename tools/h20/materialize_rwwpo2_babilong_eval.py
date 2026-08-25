#!/usr/bin/env python3
"""Bind one existing RWWPO-2 checkpoint to a frozen BABILong partition."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.gate_a_execution import checkpoint_inventory
from recurrent.research.stable_eval_identity import canonical_sha256, validate_resolved_manifest
from recurrent.research.rwwpo2_confirmation import sha256_file, signed_report
from tools.h20.preflight_rwwpo2_babilong import expected_configuration


EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"


def authenticate_bundle_audit(path: Path, *, file_sha256: str, commit: str) -> dict:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != file_sha256:
        raise ValueError("bundle audit file SHA")
    report = json.loads(path.read_text(encoding="utf-8"))
    declared = report.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared != actual or report.get("status") != "PASS" \
            or report.get("decision") != "RWWPO2_BABILONG_BUNDLE_AUDIT_PASS" \
            or report.get("git_commit") != commit:
        raise ValueError("bundle audit authentication")
    return {**report, "report_sha256": declared}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "base-resolved", "base-resolved-sha256", "bundle-audit",
        "bundle-audit-sha256", "checkpoint", "validation", "model",
        "training-commit", "eval-root", "interface-id", "attempt-id",
        "expected-commit", "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cell", choices=("B", "D", "E"), required=True)
    parser.add_argument("--experiment-seed", type=int, required=True)
    parser.add_argument("--step", type=int, choices=(20, 50, 400), required=True)
    parser.add_argument("--training-attempt-audit")
    parser.add_argument("--training-attempt-audit-sha256")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:checkout")
    if not re.fullmatch(r"[0-9a-f]{40}", args.training_commit):
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:training commit")
    if args.experiment_seed not in range(2026, 2034):
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:experiment seed")
    output = Path(args.output)
    sources = tuple(Path(value) for value in (
        args.base_resolved, args.bundle_audit, args.checkpoint,
        args.validation, args.model,
    ))
    if any(path.is_symlink() for path in sources) or output.exists() or output.is_symlink() \
            or not Path(args.eval_root).is_absolute():
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:symlink/append-only")
    base_path = sources[0].resolve()
    if sha256_file(base_path) != args.base_resolved_sha256:
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:base resolved SHA")
    try:
        audit = authenticate_bundle_audit(
            sources[1].resolve(), file_sha256=args.bundle_audit_sha256, commit=head
        )
        base = validate_resolved_manifest(json.loads(base_path.read_text(encoding="utf-8")))
    except ValueError as error:
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:" + str(error)) from error
    binding = dict(base.get("babilong_binding", {}))
    length = binding.get("length")
    partition = binding.get("partition")
    suffix = "DEV" if partition == "development" else "CONFIRM"
    expected_interface = (
        f"RWWPO2_{args.cell}_seed{args.experiment_seed}_R{args.step}_"
        f"BABILONG_{str(length).upper()}_{suffix}"
    )
    if args.interface_id != expected_interface:
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:interface")
    if args.step == 400 and partition != "confirmation":
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:R400 requires confirmation")
    if args.step in (20, 50) and partition != "development":
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:pilot requires development")
    audited = {row["length"]: row for row in audit.get("outputs", [])}.get(length)
    if not isinstance(audited, dict) or audited.get("resolved_sha256") != args.base_resolved_sha256 \
            or audited.get("validation_sha256") != binding.get("validation_sha256"):
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:bundle audit binding")
    checkpoint = sources[2].resolve()
    validation = sources[3].resolve()
    if checkpoint.name != f"global_step_{args.step}" \
            or validation != Path(str(binding.get("validation_path", ""))).resolve() \
            or sha256_file(validation) != binding.get("validation_sha256"):
        raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:checkpoint/data")
    inventory = checkpoint_inventory(checkpoint)
    attempt_binding = {
        "required": args.step in (50, 400),
        "role": (
            "unaudited_low_budget_development_checkpoint"
            if args.step == 20 else "authenticated_training_attempt_endpoint"
        ),
        "audit_path": None, "audit_file_sha256": None,
        "audit_report_sha256": None,
    }
    if args.step in (50, 400):
        if not args.training_attempt_audit \
                or not re.fullmatch(r"[0-9a-f]{64}", str(
                    args.training_attempt_audit_sha256 or "")):
            raise SystemExit(
                "RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:attempt audit required"
            )
        raw_attempt = Path(args.training_attempt_audit)
        if raw_attempt.is_symlink() or not raw_attempt.is_file() \
                or sha256_file(raw_attempt) != args.training_attempt_audit_sha256:
            raise SystemExit(
                "RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:attempt audit file SHA"
            )
        try:
            attempt = signed_report(
                raw_attempt, decision=f"RWWPO2_R{args.step}_ATTEMPT_AUDIT_PASS",
                commit=args.training_commit,
            )
        except ValueError as error:
            raise SystemExit(
                "RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:" + str(error)
            ) from error
        expected_checkpoint = Path(str(attempt.get("output_root", ""))).resolve() \
            / f"global_step_{args.step}"
        if checkpoint != expected_checkpoint \
                or attempt.get("cell") != args.cell \
                or int(attempt.get("experiment_seed", -1)) != args.experiment_seed \
                or int(attempt.get("target_round", -1)) != args.step \
                or attempt.get("target_checkpoint_inventory") != inventory \
                or attempt.get("s128_consumed") is not False \
                or attempt.get("performance_evaluated") is not False:
            raise SystemExit(
                "RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:attempt/checkpoint binding"
            )
        attempt_binding = {
            "required": True, "role": "authenticated_training_attempt_endpoint",
            "audit_path": str(raw_attempt.resolve()),
            "audit_file_sha256": args.training_attempt_audit_sha256,
            "audit_report_sha256": attempt["report_sha256"],
        }
    elif args.training_attempt_audit or args.training_attempt_audit_sha256:
        raise SystemExit(
            "RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:T20 audit arguments forbidden"
        )
    shards = []
    for rank in (0, 1):
        relative = f"actor/model_world_size_2_rank_{rank}.pt"
        item = next((row for row in inventory if row["path"] == relative), None)
        if item is None:
            raise SystemExit("RWWPO2_BABILONG_EVAL_MATERIALIZE_NO_GO:actor shards")
        shards.append(item)
    resolved = dict(base)
    resolved["babilong_binding"] = {
        **binding,
        "interface_id": args.interface_id, "cell": args.cell,
        "experiment_seed": args.experiment_seed, "evaluation_step": args.step,
        "training_git_commit": args.training_commit,
        "training_attempt_evidence": attempt_binding,
        "bundle_audit_path": str(sources[1].resolve()),
        "bundle_audit_file_sha256": args.bundle_audit_sha256,
        "bundle_audit_report_sha256": audit["report_sha256"],
        "checkpoint_inventory": inventory,
        "checkpoint_inventory_sha256": canonical_sha256(inventory),
        "model_artifact": {
            "kind": "rwwpo2_actor_checkpoint_babilong_eval",
            "path": str(checkpoint), "global_step": args.step,
            "training_git_commit": args.training_commit,
            "actor_model_shards": shards, "fsdp_world_size": 2,
            "load_mode": "actor_only",
        },
    }
    provisional_args = SimpleNamespace(**{
        **vars(args), "resolved_manifest": str(output),
    })
    overrides, runtime_sha, protocol_sha, protocol = expected_configuration(
        provisional_args, resolved
    )
    resolved["babilong_binding"]["trainer_configuration"] = {
        "resolved_runtime_config_sha256": runtime_sha,
        "override_argv_sha256": canonical_sha256(overrides),
        "generation_protocol_sha256": protocol_sha,
        "generation_protocol": protocol,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(resolved, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": "RWWPO2_BABILONG_EVAL_MATERIALIZED",
        "interface_id": args.interface_id, "output": str(output.resolve()),
        "sha256": sha256_file(output), "eval_manifest_hash": resolved["eval_manifest_hash"],
        "generation_protocol_sha256": protocol_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
