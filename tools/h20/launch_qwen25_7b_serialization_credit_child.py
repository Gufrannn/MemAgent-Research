#!/usr/bin/env python3
"""Parent supervisor for one authenticated serialization-credit GPU child."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.serialization_credit_pilots import (  # noqa: E402
    PARENT_RECEIPT_SCHEMA,
    build_parent_launch_receipt,
    canonical_sha256,
    read_jsonl,
    sha256_file,
    write_json_exclusive,
)
from tools.h20.preflight_qwen25_7b_serialization_credit import (  # noqa: E402
    MANIFEST_REL,
    issue_child_credential,
    load_manifest,
    load_parent_authority_secret,
    record_stage,
    utc_now,
    validate_child_credential,
    validate_p0,
    verify_current_binding,
)


def _observed_ppid(child_pid: int) -> int:
    completed = subprocess.run(
        ["ps", "-o", "ppid=", "-p", str(child_pid)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"parent supervisor could not observe child PPID: {completed.stderr.strip()}"
        )
    value = completed.stdout.strip()
    if not value.isdigit():
        raise RuntimeError(f"parent supervisor observed invalid child PPID: {value!r}")
    return int(value)


def _artifact_payload(path: Path, child_kind: str) -> Any:
    if child_kind == "smsb_capture":
        return read_jsonl(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _child_evidence(payload: Any, child_kind: str) -> dict[str, Any]:
    if child_kind == "smsb_capture":
        if not isinstance(payload, list) or len(payload) != 4:
            raise ValueError("supervised SMSB capture did not emit four rows")
        executions = [row.get("execution") for row in payload]
        if any(not isinstance(value, dict) for value in executions):
            raise ValueError("supervised SMSB capture lacks execution evidence")
        if any(canonical_sha256(value) != canonical_sha256(executions[0]) for value in executions[1:]):
            raise ValueError("supervised SMSB capture rows differ in process evidence")
        return dict(executions[0])
    if child_kind == "smsb_replay":
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
            raise ValueError("supervised SMSB replay payload is malformed")
        return dict(payload["result"])
    if not isinstance(payload, dict):
        raise ValueError("supervised Tetrad result payload is malformed")
    return dict(payload)


def launch(
    manifest_path: Path,
    *,
    child_kind: str,
    child_identity: str,
    artifact: Path,
    credential: Path,
    receipt: Path,
    stdout_artifact: Path,
    runner_arguments: list[str],
    record_identity: dict[str, str | None],
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    current_binding_sha = verify_current_binding(
        manifest, resolved, full_model_sha=False
    )
    authority_secret = load_parent_authority_secret(manifest, resolved)
    log_root = Path(manifest["paths"]["log_root"]).resolve()
    for path in (artifact, credential, receipt, stdout_artifact):
        if not path.resolve().is_relative_to(log_root):
            raise ValueError(f"supervised output is outside the run root: {path}")
        if path.exists():
            raise FileExistsError(f"supervised append-only output exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    issued = issue_child_credential(
        manifest_path,
        output=credential,
        child_kind=child_kind,
        child_identity=child_identity,
        issuer_pid=os.getpid(),
    )
    runner = REPO_ROOT / "tools/h20/run_qwen25_7b_serialization_credit.py"
    argv = [
        manifest["python"],
        str(runner),
        "--manifest",
        str(manifest_path.resolve()),
        *runner_arguments,
        "--credential",
        str(credential.resolve()),
    ]
    descriptor = os.open(
        stdout_artifact, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output_stream:
            process = subprocess.Popen(
                argv,
                cwd=REPO_ROOT,
                env=dict(os.environ),
                stdin=subprocess.DEVNULL,
                stdout=output_stream,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            observed_ppid = _observed_ppid(process.pid)
            exit_code = process.wait()
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except Exception:
        raise
    if observed_ppid != os.getpid():
        raise RuntimeError(
            f"supervisor is not the observed child parent: {observed_ppid} != {os.getpid()}"
        )
    if exit_code != 0:
        raise RuntimeError(
            f"supervised child {child_identity} exited {exit_code}; see {stdout_artifact}"
        )
    if not artifact.is_file():
        raise RuntimeError("supervised child exited zero without its result artifact")

    payload = _artifact_payload(artifact, child_kind)
    evidence = _child_evidence(payload, child_kind)
    if evidence.get("process_pid") != process.pid:
        raise RuntimeError("child result PID differs from parent-observed PID")
    if evidence.get("observed_parent_pid") != observed_ppid:
        raise RuntimeError("child result PPID differs from parent-observed PPID")
    credential_evidence = validate_child_credential(
        credential,
        manifest=manifest,
        resolved=resolved,
        current_binding_sha=current_binding_sha,
        child_kind=child_kind,
        child_identity=child_identity,
        authority_secret=authority_secret,
        expected_issuer_pid=os.getpid(),
    )
    if any(evidence.get(key) != value for key, value in credential_evidence.items()):
        raise RuntimeError("child result differs from authenticated parent credential")

    receipt_payload = {
        "schema": PARENT_RECEIPT_SCHEMA,
        "record_type": "parent_launch_receipt",
        "recorded_at": utc_now(),
        "child_kind": child_kind,
        "child_identity": child_identity,
        "credential_id": issued["parent_credential_id"],
        "credential_mac": issued["parent_credential_mac"],
        "credential_sha256": sha256_file(credential),
        "parent_launcher_pid": os.getpid(),
        "child_pid": process.pid,
        "observed_child_ppid": observed_ppid,
        "child_exit_code": exit_code,
        "parent_observed_launch": True,
        "process_instance_uuid": evidence["process_instance_uuid"],
        "artifact": str(artifact.resolve()),
        "artifact_sha256": sha256_file(artifact),
        "artifact_canonical_sha256": canonical_sha256(payload),
        "stdout_artifact": str(stdout_artifact.resolve()),
        "stdout_artifact_sha256": sha256_file(stdout_artifact),
        "runner_argv_sha256": canonical_sha256(argv),
        "runner_code_sha256": sha256_file(runner),
        "current_binding_sha256": current_binding_sha,
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "authority_secret_sha256": resolved["parent_receipt_authority"][
            "secret_sha256"
        ],
        "training_authorized": False,
    }
    signed_receipt = build_parent_launch_receipt(
        receipt_payload, authority_secret
    )
    write_json_exclusive(receipt, signed_receipt)
    record_stage(
        manifest_path,
        record_type=("smsb_capture" if child_kind == "smsb_capture" else child_kind),
        artifact=artifact,
        example_id=record_identity.get("example_id"),
        regime=record_identity.get("regime"),
        request_id=record_identity.get("request_id"),
        state_role=record_identity.get("state_role"),
        parent_credential=credential,
        parent_receipt=receipt,
    )
    return {
        "status": "PASS",
        "decision": "PARENT_SUPERVISED_CHILD_PASS",
        "child_kind": child_kind,
        "child_identity": child_identity,
        "child_pid": process.pid,
        "receipt_id": signed_receipt["receipt_id"],
        "artifact": str(artifact.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_REL)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--credential", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stdout-artifact", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("capture-smsb")
    replay = subparsers.add_parser("replay-smsb")
    replay.add_argument("--captures", type=Path, required=True)
    replay.add_argument("--example-id", required=True)
    replay.add_argument(
        "--regime",
        choices=("temperature_zero", "matched_seed", "independent_seed"),
        required=True,
    )
    tetrad = subparsers.add_parser("run-tetrad-request")
    tetrad.add_argument("--tetrad-manifest", type=Path, required=True)
    tetrad.add_argument("--request-id", required=True)
    tetrad.add_argument("--example-id", required=True)
    tetrad.add_argument("--state-role", required=True)
    args = parser.parse_args()

    if args.command == "capture-smsb":
        report = launch(
            args.manifest,
            child_kind="smsb_capture",
            child_identity="capture4",
            artifact=args.artifact,
            credential=args.credential,
            receipt=args.receipt,
            stdout_artifact=args.stdout_artifact,
            runner_arguments=["capture-smsb", "--output", str(args.artifact)],
            record_identity={
                "example_id": None,
                "regime": None,
                "request_id": None,
                "state_role": None,
            },
        )
    elif args.command == "replay-smsb":
        report = launch(
            args.manifest,
            child_kind="smsb_replay",
            child_identity=f"{args.example_id}::{args.regime}",
            artifact=args.artifact,
            credential=args.credential,
            receipt=args.receipt,
            stdout_artifact=args.stdout_artifact,
            runner_arguments=[
                "replay-smsb",
                "--captures",
                str(args.captures),
                "--example-id",
                args.example_id,
                "--regime",
                args.regime,
                "--output",
                str(args.artifact),
            ],
            record_identity={
                "example_id": args.example_id,
                "regime": args.regime,
                "request_id": None,
                "state_role": None,
            },
        )
    else:
        report = launch(
            args.manifest,
            child_kind="tetrad_replay",
            child_identity=args.request_id,
            artifact=args.artifact,
            credential=args.credential,
            receipt=args.receipt,
            stdout_artifact=args.stdout_artifact,
            runner_arguments=[
                "run-tetrad-request",
                "--tetrad-manifest",
                str(args.tetrad_manifest),
                "--request-id",
                args.request_id,
                "--output",
                str(args.artifact),
            ],
            record_identity={
                "example_id": args.example_id,
                "regime": None,
                "request_id": args.request_id,
                "state_role": args.state_role,
            },
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
