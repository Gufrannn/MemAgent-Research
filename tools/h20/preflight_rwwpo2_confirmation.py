#!/usr/bin/env python3
"""Render and verify one strict-vLLM RWWPO-2 confirmation evaluation."""
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
from recurrent.research.rwwpo2_confirmation import (
    generation_protocol_projection, sha256_file,
)
from recurrent.research.stable_eval_identity import (
    canonical_sha256, stable_eval_runtime_config_sha256,
    validate_resolved_manifest,
)
from tools.h20.preflight_qwen25_7b_stable_i4x2 import compose_resolved_trainer_config


EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"
COMMANDS = ROOT / "manifests/h20/qwen25_7b_s128_it_commands.json"


def render_overrides(*, checkpoint: str, validation: str, model_path: str,
                     resolved_manifest: str, eval_hash: str, eval_root: str,
                     interface_id: str, attempt_id: str, examples: int,
                     generation_seed: int, expected_runtime_sha256: str) -> list[str]:
    templates = json.loads(COMMANDS.read_text(encoding="utf-8"))[
        "common_trainer_overrides"
    ]
    replacements = {
        "${VALIDATION_PATH}": validation,
        "${MODEL_PATH}": model_path,
        "${REPO_DIR}": str(ROOT),
        "${EXPERIMENT_NAME}": f"rwwpo2_confirmation_{interface_id}_{attempt_id}",
        "${INTERFACE_ID}": interface_id,
        "${ATTEMPT_ID}": attempt_id,
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
        if value == "+data.val_max_samples=128":
            value = f"+data.val_max_samples={int(examples)}"
        elif value == "+trainer.eval_identity.base_seed=2026":
            value = f"+trainer.eval_identity.base_seed={int(generation_seed)}"
        elif value == "+trainer.eval_identity.examples=128":
            value = f"+trainer.eval_identity.examples={int(examples)}"
        if "${" in value or "=" not in value:
            raise ValueError(f"unresolved confirmation override: {value}")
        rendered.append(value)
    rendered += [
        "trainer.resume_mode=actor_only_eval",
        f"trainer.resume_from_path={checkpoint}",
        "+trainer.eval_identity.weight_source=actor_checkpoint",
        "+trainer.eval_identity.expected_global_step=400",
    ]
    return rendered


def expected_configuration(args, resolved: dict) -> tuple[list[str], str, str, dict]:
    binding = resolved["confirmation_binding"]
    provisional = render_overrides(
        checkpoint=args.checkpoint, validation=args.validation,
        model_path=args.model, resolved_manifest=args.resolved_manifest,
        eval_hash=resolved["eval_manifest_hash"], eval_root=args.eval_root,
        interface_id=args.interface_id, attempt_id=args.attempt_id,
        examples=len(resolved["identity_payload"]["rows"]),
        generation_seed=int(binding["generation_seed"]),
        expected_runtime_sha256="0" * 64,
    )
    runtime_sha = stable_eval_runtime_config_sha256(
        compose_resolved_trainer_config(ROOT, provisional)
    )
    final = render_overrides(
        checkpoint=args.checkpoint, validation=args.validation,
        model_path=args.model, resolved_manifest=args.resolved_manifest,
        eval_hash=resolved["eval_manifest_hash"], eval_root=args.eval_root,
        interface_id=args.interface_id, attempt_id=args.attempt_id,
        examples=len(resolved["identity_payload"]["rows"]),
        generation_seed=int(binding["generation_seed"]),
        expected_runtime_sha256=runtime_sha,
    )
    config = compose_resolved_trainer_config(ROOT, final)
    if stable_eval_runtime_config_sha256(config) != runtime_sha:
        raise ValueError("confirmation runtime self-hash is unstable")
    protocol = generation_protocol_projection(
        config, repo=ROOT,
        confirmation_data_sha256=binding["confirmation_data_sha256"],
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
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_CONFIRM_EVAL_NO_GO:checkout")
    if not re.fullmatch(r"RWWPO2_[BDE]_seed20(2[6-9]|3[0-3])_R400", args.interface_id):
        raise SystemExit("RWWPO2_CONFIRM_EVAL_NO_GO:interface")
    if not re.fullmatch(r"rwwpo2-confirm-eval-[a-z0-9_-]{8,63}", args.attempt_id):
        raise SystemExit("RWWPO2_CONFIRM_EVAL_NO_GO:attempt")
    if Path(args.resolved_manifest).is_symlink() \
            or Path(args.validation).is_symlink() \
            or Path(args.checkpoint).is_symlink():
        raise SystemExit("RWWPO2_CONFIRM_EVAL_NO_GO:source symlink")
    resolved_path = Path(args.resolved_manifest).resolve()
    if sha256_file(resolved_path) != args.resolved_manifest_sha256:
        raise SystemExit("RWWPO2_CONFIRM_EVAL_NO_GO:resolved SHA")
    resolved = validate_resolved_manifest(json.loads(resolved_path.read_text()))
    binding = resolved.get("execution_binding", {})
    frozen = binding.get("model_artifacts", {}).get(args.interface_id)
    checkpoint = Path(args.checkpoint).resolve()
    validation = Path(args.validation).resolve()
    if checkpoint.name != "global_step_400" or not isinstance(frozen, dict) \
            or Path(str(frozen.get("path", ""))).resolve() != checkpoint:
        raise SystemExit("RWWPO2_CONFIRM_EVAL_NO_GO:checkpoint binding")
    if checkpoint_inventory(checkpoint) != binding.get("target_checkpoint_inventory"):
        raise SystemExit("RWWPO2_CONFIRM_EVAL_NO_GO:checkpoint inventory")
    if sha256_file(validation) != resolved["confirmation_binding"][
            "confirmation_data_sha256"]:
        raise SystemExit("RWWPO2_CONFIRM_EVAL_NO_GO:confirmation data")
    try:
        overrides, runtime_sha, protocol_sha, protocol = expected_configuration(
            args, resolved
        )
    except (KeyError, ValueError) as error:
        raise SystemExit("RWWPO2_CONFIRM_EVAL_NO_GO:" + str(error)) from error
    trainer = binding.get("trainer_configuration", {})
    if trainer != {
        "resolved_runtime_config_sha256": runtime_sha,
        "override_argv_sha256": canonical_sha256(overrides),
        "generation_protocol_sha256": protocol_sha,
        "generation_protocol": protocol,
    }:
        raise SystemExit("RWWPO2_CONFIRM_EVAL_NO_GO:runtime/protocol binding")
    print("\n".join(overrides))


if __name__ == "__main__":
    main()
