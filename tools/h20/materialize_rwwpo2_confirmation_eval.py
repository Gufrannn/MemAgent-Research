#!/usr/bin/env python3
"""Bind one audited R400 checkpoint to the sealed confirmation identity."""
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
from recurrent.research.rwwpo2_confirmation import sha256_file, signed_report
from recurrent.research.stable_eval_identity import canonical_sha256, validate_resolved_manifest
from tools.h20.preflight_rwwpo2_confirmation import expected_configuration

EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "seal", "seal-sha256", "base-resolved", "base-resolved-sha256",
        "attempt-audit", "checkpoint", "validation", "model", "eval-root",
        "interface-id", "attempt-id", "expected-commit", "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cell", choices=("B", "D", "E"), required=True)
    parser.add_argument("--experiment-seed", type=int, choices=range(2026, 2034), required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:checkout")
    expected_interface = f"RWWPO2_{args.cell}_seed{args.experiment_seed}_R400"
    if args.interface_id != expected_interface:
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:interface")
    output = Path(args.output)
    if output.exists() or output.is_symlink() or not Path(args.eval_root).is_absolute():
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:append-only output")
    raw_seal = Path(args.seal)
    raw_base = Path(args.base_resolved)
    raw_attempt = Path(args.attempt_audit)
    raw_checkpoint = Path(args.checkpoint)
    raw_validation = Path(args.validation)
    if any(path.is_symlink() for path in (
            raw_seal, raw_base, raw_attempt, raw_checkpoint, raw_validation)):
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:source symlink")
    seal_path = raw_seal.resolve()
    base_path = raw_base.resolve()
    if sha256_file(seal_path) != args.seal_sha256 \
            or sha256_file(base_path) != args.base_resolved_sha256:
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:source SHA")
    try:
        seal = signed_report(
            seal_path, decision="RWWPO2_CONFIRMATION_SEAL_PASS", commit=head
        )
        attempt = signed_report(
            args.attempt_audit, decision="RWWPO2_R400_ATTEMPT_AUDIT_PASS", commit=head
        )
    except ValueError as error:
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:" + str(error)) from error
    if seal.get("resolved_identity_manifest_sha256") != args.base_resolved_sha256:
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:seal/resolved binding")
    if attempt.get("source_manifest_sha256") != seal.get("manifest_sha256") \
            or attempt.get("confirmation_seal_file_sha256") != args.seal_sha256 \
            or attempt.get("confirmation_seal_report_sha256") != seal.get(
                "report_sha256") \
            or attempt.get("s128_consumed") is not False \
            or attempt.get("performance_evaluated") is not False:
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:training preflight binding")
    base = validate_resolved_manifest(json.loads(base_path.read_text(encoding="utf-8")))
    if base["eval_manifest_hash"] != seal.get("eval_manifest_hash"):
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:identity hash")
    checkpoint = raw_checkpoint.resolve()
    expected_checkpoint = Path(attempt.get("output_root", "")).resolve() / "global_step_400"
    current_inventory = checkpoint_inventory(checkpoint)
    if checkpoint != expected_checkpoint or checkpoint.name != "global_step_400" \
            or attempt.get("cell") != args.cell \
            or int(attempt.get("experiment_seed", -1)) != args.experiment_seed \
            or int(attempt.get("target_round", -1)) != 400 \
            or current_inventory != attempt.get("target_checkpoint_inventory"):
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:training/checkpoint binding")
    if sha256_file(raw_validation.resolve()) != seal.get("confirmation_data_sha256"):
        raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:confirmation data")
    shards = []
    for rank in (0, 1):
        relative = f"actor/model_world_size_2_rank_{rank}.pt"
        item = next((row for row in current_inventory if row["path"] == relative), None)
        if item is None:
            raise SystemExit("RWWPO2_CONFIRM_MATERIALIZE_NO_GO:actor shards")
        shards.append(item)
    resolved = dict(base)
    resolved["execution_binding"] = {
        "interface_id": args.interface_id,
        "cell": args.cell,
        "experiment_seed": args.experiment_seed,
        "training_attempt_audit_path": str(raw_attempt.resolve()),
        "training_attempt_audit_sha256": sha256_file(args.attempt_audit),
        "training_attempt_audit_report_sha256": attempt["report_sha256"],
        "checkpoint_inventory_record_sha256": attempt[
            "checkpoint_inventory_record_sha256"
        ],
        "target_checkpoint_inventory": current_inventory,
        "target_checkpoint_inventory_sha256": canonical_sha256(current_inventory),
        "model_artifacts": {
            args.interface_id: {
                "kind": "authenticated_rwwpo2_r400_actor_checkpoint",
                "path": str(checkpoint), "global_step": 400,
                "actor_model_shards": shards, "fsdp_world_size": 2,
                "load_mode": "actor_only",
            }
        },
    }
    provisional_args = SimpleNamespace(**{
        **vars(args), "resolved_manifest": str(output),
    })
    overrides, runtime_sha, protocol_sha, protocol = expected_configuration(
        provisional_args, resolved
    )
    resolved["execution_binding"]["trainer_configuration"] = {
        "resolved_runtime_config_sha256": runtime_sha,
        "override_argv_sha256": canonical_sha256(overrides),
        "generation_protocol_sha256": protocol_sha,
        "generation_protocol": protocol,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(resolved, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": "RWWPO2_CONFIRMATION_EVAL_MATERIALIZED",
        "interface_id": args.interface_id, "output": str(output.resolve()),
        "sha256": sha256_file(output), "eval_manifest_hash": resolved["eval_manifest_hash"],
        "generation_protocol_sha256": protocol_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
