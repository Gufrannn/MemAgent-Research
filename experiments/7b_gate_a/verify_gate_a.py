#!/usr/bin/env python3
"""Fail-closed verifier for the 7B 2->3 step infrastructure gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(step: Path) -> dict[str, list[Path]]:
    actor = step / "actor"
    return {
        "model": sorted(actor.glob("*model*.pt")),
        "optim": sorted(actor.glob("*optim*.pt")),
        "extra": sorted(actor.glob("*extra*.pt")),
        "data": [step / "data.pt"] if (step / "data.pt").is_file() else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fresh-log", type=Path, required=True)
    parser.add_argument("--resume-log", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    audit_path = args.run_dir / "rollout_seed_audit.jsonl"
    records = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()] if audit_path.is_file() else []
    if not records:
        failures.append("missing rollout_seed_audit.jsonl records")
    for step in sorted({int(row["global_step"]) for row in records}):
        step_rows = [row for row in records if int(row["global_step"]) == step]
        seeds = [int(row["trajectory_seed"]) for row in step_rows]
        if len(seeds) != len(set(seeds)):
            failures.append(f"trajectory seeds are not independent at global_step={step}")
        if any(row.get("mode") != "independent" for row in step_rows):
            failures.append(f"non-independent seed mode at global_step={step}")

    for step_number in (2, 3):
        files = checkpoint_files(args.run_dir / f"global_step_{step_number}")
        for kind, matches in files.items():
            if not matches:
                failures.append(f"global_step_{step_number} missing {kind} checkpoint component")

    fresh = args.fresh_log.read_text(errors="replace") if args.fresh_log.is_file() else ""
    resume = args.resume_log.read_text(errors="replace") if args.resume_log.is_file() else ""
    if not re.search(r"step:2\s+-", fresh):
        failures.append("fresh log has no completed step:2 metrics")
    if "Resuming from" not in resume or "global_step_2" not in resume:
        failures.append("resume log does not prove explicit load from global_step_2")
    if not re.search(r"step:3\s+-", resume):
        failures.append("resume log has no completed step:3 metrics")
    combined = fresh + "\n" + resume
    if "After sync model weights in sharding manager" not in combined and "vLLM load weights" not in combined:
        failures.append("logs do not expose FSDP->vLLM weight synchronization")
    if "actor/grad_norm:" not in combined:
        failures.append("logs do not expose actor backward/update")

    step2_models = checkpoint_files(args.run_dir / "global_step_2")["model"]
    step3_models = checkpoint_files(args.run_dir / "global_step_3")["model"]
    if step2_models and step3_models:
        hashes2 = [sha256(path) for path in step2_models]
        hashes3 = [sha256(path) for path in step3_models]
        if hashes2 == hashes3:
            failures.append("step2 and step3 actor model shard hashes are identical")

    result = {"gate_a_pass": not failures, "failures": failures, "seed_records": len(records)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
