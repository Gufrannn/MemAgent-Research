#!/usr/bin/env python3
"""Freeze one validation-only CORAL checkpoint evaluation and emit Hydra argv."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.cosi import canonical_sha256, checkpoint_inventory, sha256_file
from recurrent.research.stable_eval_identity import (
    stable_eval_runtime_config_sha256,
    validate_resolved_manifest,
)
from tools.h20.preflight_qwen25_7b_stable_i4x2 import compose_resolved_trainer_config

BRANCH = "h20/qwen25-7b-cosi-t25-frozen-20260822"
DATA_SHA256 = "54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6"
DIGEST_PARAMETERS = "model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def actor_shards(checkpoint: Path):
    rows = checkpoint_inventory(checkpoint)
    shards = [row for row in rows if re.fullmatch(
        r"actor/model_world_size_2_rank_[01]\.pt", row["path"]
    )]
    if [row["path"] for row in shards] != [
        "actor/model_world_size_2_rank_0.pt", "actor/model_world_size_2_rank_1.pt"
    ]:
        raise ValueError("CORAL_S128_NO_GO: exact two-rank actor shards missing")
    return rows, shards


def render(*, step, checkpoint, output_root, resolved_path, eval_hash, runtime_hash):
    command = json.loads((ROOT / "manifests/h20/qwen25_7b_s128_it_commands.json").read_text())
    common = list(command["common_trainer_overrides"])
    interface = f"CORAL_T{step}"
    replacements = {
        "${VALIDATION_PATH}": str(Path(os.environ["MEMAGENT_COSI_WORK_ROOT"]) / "datasets/hotpotqa/hotpotqa_dev.parquet"),
        "${MODEL_PATH}": str(Path(os.environ["MEMAGENT_COSI_WORK_ROOT"]) / "models/Qwen2.5-7B-Instruct"),
        "${REPO_DIR}": str(ROOT),
        "${EXPERIMENT_NAME}": f"qwen25_7b_coral_fixed_s128_t{step}",
        "${INTERFACE_ID}": interface,
        "${ATTEMPT_ID}": f"coral_t{step}_primary",
        "${INTERFACE_ROOT}": str(output_root),
        "${TERMINAL_DIR}": str(output_root / "terminal"),
        "${RESOLVED_MANIFEST_PATH}": str(resolved_path),
        "${EVAL_MANIFEST_HASH}": eval_hash,
        "${TURN_LEDGER_PATH}": str(output_root / "trajectory_turns.jsonl"),
        "${EXECUTION_SUMMARY_PATH}": str(output_root / "execution_summary.json"),
        "${EXPECTED_RUNTIME_CONFIG_SHA256}": runtime_hash,
    }
    result = []
    for raw in common:
        value = raw
        for source, target in replacements.items():
            value = value.replace(source, target)
        if "${" in value:
            raise ValueError(f"CORAL_S128_NO_GO: unresolved override {value}")
        result.append(value)
    result.extend([
        "trainer.resume_mode=actor_only_eval",
        f"trainer.resume_from_path={checkpoint}",
        "+trainer.eval_identity.weight_source=actor_checkpoint",
        f"+trainer.eval_identity.expected_global_step={step}",
    ])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", type=int, choices=(5, 10, 15, 20, 25), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--emit-overrides", action="store_true")
    args = parser.parse_args()
    env = os.environ
    required = (
        "MEMAGENT_COSI_WORK_ROOT", "MEMAGENT_COSI_REPO_DIR",
        "MEMAGENT_COSI_EXPECTED_COMMIT", "MEMAGENT_COSI_S128_RESOLVED_MANIFEST",
        "MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256",
    )
    if any(not env.get(key) for key in required):
        raise ValueError("CORAL_S128_NO_GO: explicit environment binding missing")
    expected = env["MEMAGENT_COSI_EXPECTED_COMMIT"]
    if Path(env["MEMAGENT_COSI_REPO_DIR"]).resolve() != ROOT \
            or not re.fullmatch(r"[0-9a-f]{40}", expected) \
            or git("rev-parse", "HEAD") != expected \
            or git("branch", "--show-current") != BRANCH \
            or git("status", "--porcelain"):
        raise ValueError("CORAL_S128_NO_GO: checkout/commit/cleanliness binding")
    checkpoint = Path(args.checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    if checkpoint.name != f"global_step_{args.step}" or not checkpoint.is_dir():
        raise ValueError("CORAL_S128_NO_GO: checkpoint step/path")
    if output_root.exists():
        raise ValueError("CORAL_S128_NO_GO: append-only evaluation root exists")
    validation = Path(env["MEMAGENT_COSI_WORK_ROOT"]) / "datasets/hotpotqa/hotpotqa_dev.parquet"
    if sha256_file(validation) != DATA_SHA256:
        raise ValueError("CORAL_S128_NO_GO: fixed-S128 parquet drift")
    source_resolved_path = Path(env["MEMAGENT_COSI_S128_RESOLVED_MANIFEST"]).resolve()
    expected_source_sha = env["MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256"]
    if re.fullmatch(r"[0-9a-f]{64}", expected_source_sha) is None \
            or sha256_file(source_resolved_path) != expected_source_sha:
        raise ValueError("CORAL_S128_NO_GO: external S128 resolved-manifest SHA")
    source_resolved = json.loads(source_resolved_path.read_text())
    validate_resolved_manifest(source_resolved)
    inventory, shards = actor_shards(checkpoint)
    interface = f"CORAL_T{args.step}"
    resolved = json.loads(json.dumps(source_resolved))
    resolved.setdefault("execution_binding", {}).setdefault("model_artifacts", {})[interface] = {
        "kind": "coral_actor_checkpoint",
        "global_step": args.step,
        "git_commit": expected,
        "checkpoint_inventory_sha256": canonical_sha256(inventory),
        "actor_model_shards": shards,
    }
    resolved_path = output_root / "certificates/resolved_eval_manifest.json"
    eval_hash = resolved["eval_manifest_hash"]
    placeholder = render(
        step=args.step, checkpoint=checkpoint, output_root=output_root,
        resolved_path=resolved_path, eval_hash=eval_hash, runtime_hash="0" * 64,
    )
    runtime_hash = stable_eval_runtime_config_sha256(
        compose_resolved_trainer_config(ROOT, placeholder)
    )
    overrides = render(
        step=args.step, checkpoint=checkpoint, output_root=output_root,
        resolved_path=resolved_path, eval_hash=eval_hash, runtime_hash=runtime_hash,
    )
    if stable_eval_runtime_config_sha256(compose_resolved_trainer_config(ROOT, overrides)) != runtime_hash:
        raise ValueError("CORAL_S128_NO_GO: self-hashed config instability")
    report = {
        "schema": "memagent.coral.s128-preflight.v1", "status": "PASS",
        "decision": "CORAL_S128_P0_PASS", "step": args.step, "git_commit": expected,
        "checkpoint": str(checkpoint), "checkpoint_inventory_sha256": canonical_sha256(inventory),
        "eval_manifest_hash": eval_hash, "runtime_config_sha256": runtime_hash,
        "override_argv_sha256": canonical_sha256(overrides),
        "digest_parameters": DIGEST_PARAMETERS, "digest_samples": 256,
    }
    report["report_sha256"] = canonical_sha256(report)
    output_root.mkdir(parents=True)
    resolved_path.parent.mkdir(parents=True)
    resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    (output_root / "certificates/p0.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "certificates/overrides.json").write_text(
        json.dumps(overrides, indent=2) + "\n"
    )
    if args.emit_overrides:
        print("\n".join(overrides))
    else:
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
