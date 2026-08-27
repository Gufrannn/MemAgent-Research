#!/usr/bin/env python3
"""Authorize a resume-only source fix without changing the frozen algorithm.

The producer's resolved numeric contract remains authoritative.  This entry
proves that the consumer checkout changes only evidence/recovery machinery and
that every actor/objective/controller/training source is byte-identical.  The
receipt is intentionally generic across cells and checkpoints; the existing
lineage-parent audit binds each imported checkpoint separately.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.rwwpo_transaction import (
    RWWPO2_FSDP_PARAMETER_COMMIT_PRIMITIVE,
    RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS,
)
from tools.h20.audit_rwwpo2_numeric_oracle import (
    validate_fsdp_transaction_closure,
)


EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"
MANIFEST = "manifests/h20/qwen25_7b_rwwpo2_r400_k2_seed2026.json"
TRAINER = "verl/trainer/ppo/ray_trainer.py"
LAUNCHER = "scripts/h20/run_qwen25_7b_rwwpo2.sh"
PROTECTED_EXACT_SOURCES = (
    MANIFEST,
    "rwwpo2_experiment_manifest.schema.json",
    "rwwpo2_actual_loss_receipt.schema.json",
    "recurrent/research/rwwpo_transaction.py",
    "recurrent/research/rwwpo_ledger.py",
    "recurrent/research/actor_batch.py",
    "recurrent/research/hotpotqa_dense_reward.py",
    "verl/trainer/main_ppo.py",
    "verl/trainer/ppo/core_algos.py",
    "verl/workers/actor/dp_actor.py",
    "verl/utils/fsdp_utils.py",
    "verl/utils/torch_functional.py",
    "experiments/7b_gate_a/run_gate_a.sh",
    "scripts/h20/rwwpo2_common.sh",
    "tools/h20/calibrate_rwwpo2_numeric_oracle.py",
    "tools/h20/audit_rwwpo2_numeric_oracle.py",
    "tools/h20/materialize_rwwpo2_resolved_contract.py",
)
TRAINER_COMPATIBILITY_EXCLUSIONS = {
    "_load_checkpoint",
    "_prune_rwwpo2_recovery_roots",
}
ALLOWED_RECURRENT_CHANGES = {
    "recurrent/research/gate_a_execution.py",
    "recurrent/research/rwwpo2_babilong.py",
}
LAUNCHER_WIRING_BEGIN = "# BEGIN RWWPO2_CROSS_COMMIT_COMPATIBILITY_WIRING"
LAUNCHER_WIRING_END = "# END RWWPO2_CROSS_COMMIT_COMPATIBILITY_WIRING"
RECOVERY_PRUNE_CONTRACT = "scientific_anchor_aware_two_phase_v2"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT
    )


def signed_receipt(path: Path, *, decision: str, commit: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("resolved contract path")
    row = json.loads(path.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = sha256_bytes(json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode())
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != decision or row.get("git_commit") != commit:
        raise ValueError("resolved contract identity")
    return {**row, "report_sha256": declared}


def release_test_receipt(path: Path, *, expected_sha: str, commit: str) -> dict:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError("release-test receipt bytes")
    row = signed_receipt(
        path, decision="RWWPO2_RELEASE_TESTS_PASS", commit=commit
    )
    if not isinstance(row.get("runtime_environment"), dict) \
            or int(row.get("junit_summary", {}).get("tests", 0)) < 1:
        raise ValueError("release-test receipt semantics")
    return row


class _TrainerProjection(ast.NodeTransformer):
    def visit_ClassDef(self, node: ast.ClassDef):  # noqa: N802
        if node.name == "RayPPOTrainer":
            node.body = [
                child for child in node.body
                if not (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name in TRAINER_COMPATIBILITY_EXCLUSIONS
                )
            ]
        return self.generic_visit(node)


def trainer_projection(source: bytes) -> str:
    tree = ast.parse(source.decode("utf-8"))
    projected = _TrainerProjection().visit(tree)
    ast.fix_missing_locations(projected)
    return sha256_bytes(ast.dump(projected, include_attributes=False).encode())


def launcher_projection(source: bytes) -> str:
    """Remove only the additive, delimited compatibility receipt wiring."""
    text = source.decode("utf-8")
    begin_count = text.count(LAUNCHER_WIRING_BEGIN)
    end_count = text.count(LAUNCHER_WIRING_END)
    if begin_count == end_count == 0:
        return sha256_bytes(source)
    if begin_count != 1 or end_count != 1:
        raise ValueError("launcher compatibility marker count")
    before, remainder = text.split(LAUNCHER_WIRING_BEGIN, 1)
    _, after = remainder.split(LAUNCHER_WIRING_END, 1)
    if not after.startswith("\n"):
        raise ValueError("launcher compatibility marker boundary")
    return sha256_bytes((before + after[1:]).encode("utf-8"))


def allowed_changed_path(path: str) -> bool:
    if path == TRAINER or path in ALLOWED_RECURRENT_CHANGES \
            or path == "gate_a_execution_ledger.schema.json":
        return True
    return path.startswith(("docs/", "tests/", "tools/h20/", "scripts/h20/")) \
        or path == "manifests/h20/rwwpo2_babilong_pilot_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--producer-resolved-contract", required=True)
    parser.add_argument("--producer-resolved-sha256", required=True)
    parser.add_argument("--producer-release-test-receipt", required=True)
    parser.add_argument("--producer-release-test-receipt-sha256", required=True)
    parser.add_argument("--consumer-release-test-receipt", required=True)
    parser.add_argument("--consumer-release-test-receipt-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if re.fullmatch(r"[0-9a-f]{40}", args.producer_commit) is None \
            or re.fullmatch(r"[0-9a-f]{40}", args.consumer_commit) is None \
            or args.producer_commit == args.consumer_commit:
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:commit identity")
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    dirty = git("status", "--porcelain")
    if head != args.consumer_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:checkout")
    try:
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{args.producer_commit}^{{commit}}"],
            cwd=ROOT,
        )
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", args.producer_commit,
             args.consumer_commit],
            cwd=ROOT,
        )
    except subprocess.CalledProcessError as error:
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:producer ancestry") from error

    resolved_path = Path(args.producer_resolved_contract)
    if resolved_path.is_symlink() or not resolved_path.is_file() \
            or sha256_file(resolved_path) != args.producer_resolved_sha256:
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:producer contract bytes")
    try:
        resolved = signed_receipt(
            resolved_path, decision="RWWPO2_RESOLVED_CONTRACT_PASS",
            commit=args.producer_commit,
        )
        validate_fsdp_transaction_closure(
            resolved["fsdp_transaction_closure"],
            tau_logprob=float(resolved["numeric_thresholds"]["tau_logprob"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:producer contract") from error
    if resolved.get("fsdp_parameter_commit_primitive") != \
            RWWPO2_FSDP_PARAMETER_COMMIT_PRIMITIVE \
            or int(resolved.get("gradient_sketch_chunk_elements", -1)) != \
            RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS:
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:runtime numeric contract")
    try:
        producer_release = release_test_receipt(
            Path(args.producer_release_test_receipt),
            expected_sha=args.producer_release_test_receipt_sha256,
            commit=args.producer_commit,
        )
        consumer_release = release_test_receipt(
            Path(args.consumer_release_test_receipt),
            expected_sha=args.consumer_release_test_receipt_sha256,
            commit=args.consumer_commit,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:release tests") from error
    if producer_release["runtime_environment"] != consumer_release[
            "runtime_environment"]:
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:runtime environment drift")

    protected = {}
    for path in PROTECTED_EXACT_SOURCES:
        producer = git_bytes(args.producer_commit, path)
        consumer = git_bytes(args.consumer_commit, path)
        if producer != consumer:
            raise SystemExit(
                "RWWPO2_CROSS_COMMIT_NO_GO:algorithm source drift:" + path
            )
        protected[path] = sha256_bytes(producer)
    if protected[MANIFEST] != resolved.get("source_manifest_sha256"):
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:manifest contract drift")

    producer_trainer = git_bytes(args.producer_commit, TRAINER)
    consumer_trainer = git_bytes(args.consumer_commit, TRAINER)
    producer_projection = trainer_projection(producer_trainer)
    consumer_projection = trainer_projection(consumer_trainer)
    if producer_projection != consumer_projection:
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:trainer algorithm drift")
    consumer_trainer_text = consumer_trainer.decode("utf-8")
    for token in (
        "RWWPO2_RESUME_CROSS_COMMIT_COMPATIBILITY_DRIFT",
        '"rwwpo2_recovery_prune_intent"',
        '"rwwpo2_recovery_pruned"',
        "prune_intent_record_sha256",
        "scientific_anchor_required",
    ):
        if token not in consumer_trainer_text:
            raise SystemExit(
                "RWWPO2_CROSS_COMMIT_NO_GO:consumer recovery closure:" + token
            )
    try:
        producer_launcher_projection = launcher_projection(git_bytes(
            args.producer_commit, LAUNCHER
        ))
        consumer_launcher_projection = launcher_projection(git_bytes(
            args.consumer_commit, LAUNCHER
        ))
    except ValueError as error:
        raise SystemExit(
            "RWWPO2_CROSS_COMMIT_NO_GO:launcher compatibility markers"
        ) from error
    if producer_launcher_projection != consumer_launcher_projection:
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:launcher algorithm drift")

    changed_paths = sorted(filter(None, git(
        "diff", "--name-only", args.producer_commit, args.consumer_commit
    ).splitlines()))
    disallowed = [path for path in changed_paths if not allowed_changed_path(path)]
    if disallowed:
        raise SystemExit(
            "RWWPO2_CROSS_COMMIT_NO_GO:unapproved changed paths:" +
            json.dumps(disallowed, sort_keys=True)
        )

    report = {
        "schema_version": "rwwpo2-cross-commit-resume-v1",
        "status": "PASS",
        "decision": "RWWPO2_CROSS_COMMIT_RESUME_COMPATIBILITY_PASS",
        "git_commit": args.consumer_commit,
        "producer_git_commit": args.producer_commit,
        "consumer_git_commit": args.consumer_commit,
        "compatibility_scope": "producer_contract_continuity",
        "allowed_training_phases": ["fresh", "resume"],
        "producer_resolved_contract_reused": True,
        "consumer_numeric_contract_substitution_forbidden": True,
        "producer_resolved_contract_path": str(resolved_path.resolve()),
        "producer_resolved_contract_file_sha256": args.producer_resolved_sha256,
        "producer_resolved_contract_report_sha256": resolved["report_sha256"],
        "producer_release_test_receipt_file_sha256":
            args.producer_release_test_receipt_sha256,
        "producer_release_test_receipt_report_sha256":
            producer_release["report_sha256"],
        "consumer_release_test_receipt_file_sha256":
            args.consumer_release_test_receipt_sha256,
        "consumer_release_test_receipt_report_sha256":
            consumer_release["report_sha256"],
        "runtime_environment": producer_release["runtime_environment"],
        "source_manifest_sha256": resolved["source_manifest_sha256"],
        "numeric_thresholds": resolved["numeric_thresholds"],
        "proposal_schedule": resolved["proposal_schedule"],
        "fsdp_parameter_commit_primitive": resolved[
            "fsdp_parameter_commit_primitive"
        ],
        "protected_exact_source_sha256": protected,
        "trainer_projection_exclusions": sorted(
            TRAINER_COMPATIBILITY_EXCLUSIONS
        ),
        "producer_trainer_projection_sha256": producer_projection,
        "consumer_trainer_projection_sha256": consumer_projection,
        "launcher_projection_markers": [
            LAUNCHER_WIRING_BEGIN, LAUNCHER_WIRING_END,
        ],
        "recovery_prune_contract": RECOVERY_PRUNE_CONTRACT,
        "producer_launcher_projection_sha256":
            producer_launcher_projection,
        "consumer_launcher_projection_sha256":
            consumer_launcher_projection,
        "changed_paths": changed_paths,
        "algorithmic_source_or_contract_change": False,
    }
    raw = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    report["report_sha256"] = sha256_bytes(raw.encode())
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise SystemExit("RWWPO2_CROSS_COMMIT_NO_GO:output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "producer_git_commit": args.producer_commit,
        "consumer_git_commit": args.consumer_commit,
        "output": str(output.resolve()),
        "report_sha256": report["report_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
