#!/usr/bin/env python3
"""Render a commit-bound actor-only RWWPO-2 BABILong evaluation."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.gate_a_execution import checkpoint_inventory
from recurrent.research.rwwpo2_babilong import LENGTHS, MAX_CHUNKS
from recurrent.research.rwwpo2_confirmation import (
    generation_protocol_projection, sha256_file, signed_report,
)
from recurrent.research.stable_eval_identity import (
    canonical_sha256, stable_eval_runtime_config_sha256, validate_resolved_manifest,
)
from tools.h20.preflight_qwen25_7b_stable_i4x2 import compose_resolved_trainer_config


EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"
COMMANDS = ROOT / "manifests/h20/qwen25_7b_s128_it_commands.json"


def render_overrides(
    *, checkpoint: str, validation: str, model_path: str, resolved_manifest: str,
    eval_hash: str, eval_root: str, interface_id: str, attempt_id: str,
    examples: int, generation_seed: int, expected_runtime_sha256: str,
    step: int, length: str,
) -> list[str]:
    templates = json.loads(COMMANDS.read_text(encoding="utf-8"))["common_trainer_overrides"]
    replacements = {
        "${VALIDATION_PATH}": validation, "${MODEL_PATH}": model_path,
        "${REPO_DIR}": str(ROOT),
        "${EXPERIMENT_NAME}": f"rwwpo2_babilong_{interface_id}_{attempt_id}",
        "${INTERFACE_ID}": interface_id, "${ATTEMPT_ID}": attempt_id,
        "${INTERFACE_ROOT}": eval_root,
        "${TERMINAL_DIR}": str(Path(eval_root) / "terminal"),
        "${RESOLVED_MANIFEST_PATH}": resolved_manifest,
        "${EVAL_MANIFEST_HASH}": eval_hash,
        "${TURN_LEDGER_PATH}": str(Path(eval_root) / "trajectory_turns.jsonl"),
        "${EXECUTION_SUMMARY_PATH}": str(Path(eval_root) / "execution_summary.json"),
        "${EXPECTED_RUNTIME_CONFIG_SHA256}": expected_runtime_sha256,
        "${T25_CHECKPOINT}": checkpoint,
    }
    rendered = []
    for template in templates:
        value = str(template)
        for source, target in replacements.items():
            value = value.replace(source, str(target))
        if value == "recurrent.memory.config.max_chunks=8":
            value = f"recurrent.memory.config.max_chunks={MAX_CHUNKS[length]}"
        elif value == "+data.val_max_samples=128":
            value = f"+data.val_max_samples={int(examples)}"
        elif value == "+trainer.eval_identity.base_seed=2026":
            value = f"+trainer.eval_identity.base_seed={int(generation_seed)}"
        elif value == "+trainer.eval_identity.examples=128":
            value = f"+trainer.eval_identity.examples={int(examples)}"
        if "${" in value or "=" not in value:
            raise ValueError(f"unresolved BABILong override: {value}")
        rendered.append(value)
    rendered += [
        "trainer.resume_mode=actor_only_eval",
        f"trainer.resume_from_path={checkpoint}",
        "+trainer.eval_identity.weight_source=actor_checkpoint",
        f"+trainer.eval_identity.expected_global_step={int(step)}",
    ]
    return rendered


def expected_configuration(args, resolved: dict) -> tuple[list[str], str, str, dict]:
    binding = resolved["babilong_binding"]
    training_evidence = binding.get("training_attempt_evidence", {})
    step = int(binding["evaluation_step"])
    if step == 20:
        if training_evidence != {
            "required": False,
            "role": "unaudited_low_budget_development_checkpoint",
            "audit_path": None, "audit_file_sha256": None,
            "audit_report_sha256": None,
        }:
            raise ValueError("BABILong T20 evidence role")
    else:
        raw_attempt = Path(str(training_evidence.get("audit_path", "")))
        if training_evidence.get("required") is not True \
                or training_evidence.get("role") != "authenticated_training_attempt_endpoint" \
                or raw_attempt.is_symlink() or not raw_attempt.is_file() \
                or sha256_file(raw_attempt) != training_evidence.get("audit_file_sha256"):
            raise ValueError("BABILong training attempt file binding")
        attempt = signed_report(
            raw_attempt, decision=f"RWWPO2_R{step}_ATTEMPT_AUDIT_PASS",
            commit=str(binding["training_git_commit"]),
        )
        checkpoint = Path(args.checkpoint).resolve()
        if attempt.get("report_sha256") != training_evidence.get("audit_report_sha256") \
                or checkpoint != Path(str(attempt.get("output_root", ""))).resolve() \
                    / f"global_step_{step}" \
                or attempt.get("cell") != binding.get("cell") \
                or int(attempt.get("experiment_seed", -1)) != int(
                    binding.get("experiment_seed", -2)) \
                or int(attempt.get("target_round", -1)) != step \
                or attempt.get("target_checkpoint_inventory") != binding.get(
                    "checkpoint_inventory") \
                or attempt.get("s128_consumed") is not False \
                or attempt.get("performance_evaluated") is not False:
            raise ValueError("BABILong training attempt semantic binding")
    parameters = dict(
        checkpoint=args.checkpoint, validation=args.validation, model_path=args.model,
        resolved_manifest=args.resolved_manifest,
        eval_hash=resolved["eval_manifest_hash"], eval_root=args.eval_root,
        interface_id=args.interface_id, attempt_id=args.attempt_id,
        examples=len(resolved["identity_payload"]["rows"]),
        generation_seed=int(binding["generation_seed"]),
        step=step, length=str(binding["length"]),
    )
    provisional = render_overrides(expected_runtime_sha256="0" * 64, **parameters)
    runtime_sha = stable_eval_runtime_config_sha256(
        compose_resolved_trainer_config(ROOT, provisional)
    )
    final = render_overrides(expected_runtime_sha256=runtime_sha, **parameters)
    config = compose_resolved_trainer_config(ROOT, final)
    if stable_eval_runtime_config_sha256(config) != runtime_sha:
        raise ValueError("BABILong runtime self-hash is unstable")
    protocol = generation_protocol_projection(
        config, repo=ROOT,
        confirmation_data_sha256=binding["validation_sha256"],
        model=resolved["identity_payload"]["base_model_protocol"],
    )
    return final, runtime_sha, canonical_sha256(protocol), protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint", "validation", "model", "resolved-manifest",
        "resolved-manifest-sha256", "eval-root", "interface-id",
        "attempt-id", "expected-commit",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:checkout")
    if not re.fullmatch(r"RWWPO2_[BDE]_seed(202[6-9]|203[0-3])_R(20|50|400)_BABILONG_(32K|128K)_(DEV|CONFIRM)", args.interface_id):
        raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:interface")
    if not re.fullmatch(r"rwwpo2-babilong-eval-[a-z0-9_-]{8,80}", args.attempt_id):
        raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:attempt")
    paths = tuple(Path(value) for value in (
        args.resolved_manifest, args.validation, args.checkpoint, args.model,
    ))
    if any(path.is_symlink() for path in paths):
        raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:source symlink")
    resolved_path = paths[0].resolve()
    if sha256_file(resolved_path) != args.resolved_manifest_sha256:
        raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:resolved SHA")
    resolved = validate_resolved_manifest(json.loads(resolved_path.read_text(encoding="utf-8")))
    binding = resolved.get("babilong_binding", {})
    checkpoint = paths[2].resolve()
    validation = paths[1].resolve()
    artifact = binding.get("model_artifact", {})
    training_evidence = binding.get("training_attempt_evidence", {})
    if binding.get("interface_id") != args.interface_id \
            or Path(str(artifact.get("path", ""))).resolve() != checkpoint \
            or checkpoint.name != f"global_step_{binding.get('evaluation_step')}" \
            or checkpoint_inventory(checkpoint) != binding.get("checkpoint_inventory"):
        raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:checkpoint binding")
    if int(binding.get("evaluation_step", -1)) == 20:
        if training_evidence != {
            "required": False,
            "role": "unaudited_low_budget_development_checkpoint",
            "audit_path": None, "audit_file_sha256": None,
            "audit_report_sha256": None,
        }:
            raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:T20 evidence role")
    elif training_evidence.get("required") is not True \
            or training_evidence.get("role") != "authenticated_training_attempt_endpoint" \
            or any(not isinstance(training_evidence.get(field), str)
                   or not training_evidence[field]
                   for field in ("audit_path", "audit_file_sha256", "audit_report_sha256")):
        raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:training attempt evidence")
    if validation != Path(str(binding.get("validation_path", ""))).resolve() \
            or sha256_file(validation) != binding.get("validation_sha256"):
        raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:validation binding")
    try:
        overrides, runtime_sha, protocol_sha, protocol = expected_configuration(args, resolved)
    except (KeyError, ValueError) as error:
        raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:" + str(error)) from error
    trainer = binding.get("trainer_configuration", {})
    if trainer != {
        "resolved_runtime_config_sha256": runtime_sha,
        "override_argv_sha256": canonical_sha256(overrides),
        "generation_protocol_sha256": protocol_sha,
        "generation_protocol": protocol,
    }:
        raise SystemExit("RWWPO2_BABILONG_EVAL_NO_GO:runtime/protocol binding")
    print("\n".join(overrides))


if __name__ == "__main__":
    main()
