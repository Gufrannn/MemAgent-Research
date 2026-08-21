#!/usr/bin/env python3
"""Read-only gates and append-only evidence utilities for PRD-MemRL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def certificate(decision: str, status: str, evidence: dict) -> dict:
    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": decision,
        "evidence": evidence,
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def command_e0(args: argparse.Namespace) -> int:
    import torch
    from recurrent.research.prd_memrl import PriorTaintError, ProjectedDual, conditional_rate_nats, validate_prior_record

    failures = []
    actor = torch.log(torch.tensor([[0.8], [0.8], [0.2], [0.2]], dtype=torch.float64))
    prior = torch.full_like(actor, -torch.log(torch.tensor(2.0, dtype=torch.float64)))
    _, bound = conditional_rate_nats(actor, prior, torch.ones_like(actor))
    mi = float(torch.log(torch.tensor(2.0)) + 0.2 * torch.log(torch.tensor(0.2)) + 0.8 * torch.log(torch.tensor(0.8)))
    if float(bound) + 1e-9 < mi:
        failures.append("synthetic KL upper bound failed")
    for field in ("new_evidence", "history_chunk", "gold", "future", "reward"):
        try:
            validate_prior_record({"previous_memory": "m", "turn_index": 0, field: "taint"})
            failures.append(f"taint accepted: {field}")
        except PriorTaintError:
            pass
    dual = ProjectedDual(0.1, 0.5)
    for _ in range(10):
        dual.step(0.2)
    rose = dual.value > 0
    for _ in range(20):
        dual.step(0.0)
    if not rose or dual.value != 0:
        failures.append("projected dual direction/recovery failed")
    result = certificate("PRD_E0_PASS" if not failures else "PRD_E0_NO_GO", "PASS" if not failures else "FAIL", {"failures": failures, "synthetic_mi_nats": mi, "variational_bound_nats": float(bound), "git_commit": git("rev-parse", "HEAD")})
    write_json_exclusive(Path(args.output), result)
    return 0 if not failures else 2


def command_e1(args: argparse.Namespace) -> int:
    from recurrent.research.prd_memrl import assert_rate_not_length

    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    required = {"stable_id", "writer_tokens", "actor_logprob_sum", "prior_logprob_sum"}
    failures = []
    if not rows or any(set(row) < required for row in rows):
        failures.append("missing exact E1 row fields")
    discordance = None
    if not failures:
        rates = [float(row["actor_logprob_sum"]) - float(row["prior_logprob_sum"]) for row in rows]
        try:
            discordance = assert_rate_not_length(rates, [int(row["writer_tokens"]) for row in rows])
        except ValueError as exc:
            failures.append(str(exc))
    result = certificate("PRD_E1_PASS" if not failures else "PRD_E1_NO_GO", "PASS" if not failures else "FAIL", {"failures": failures, "row_count": len(rows), "rate_length_discordance": discordance, "rows_sha256": sha256(Path(args.rows)), "git_commit": git("rev-parse", "HEAD")})
    write_json_exclusive(Path(args.output), result)
    return 0 if not failures else 3


def verify_pass(path: Path, decision: str, commit: str) -> None:
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS" or payload.get("decision") != decision or payload.get("evidence", {}).get("git_commit") != commit:
        raise SystemExit(f"PRD_NO_GO invalid {decision} certificate")


def command_preflight(args: argparse.Namespace) -> int:
    failures = []
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    if commit != args.expected_commit:
        failures.append("wrong exact commit")
    if branch != "h20/qwen25-7b-prd-memrl-t25-frozen-20260822":
        failures.append("wrong branch")
    if git("status", "--porcelain"):
        failures.append("dirty worktree")
    pair = args.gpu_pair.split(",")
    if len(pair) != 2 or any(not item.isdigit() for item in pair) or len(set(pair)) != 2 or list(map(int, pair)) != sorted(map(int, pair)):
        failures.append("GPU pair must be two distinct canonical ascending indices")
    for path, decision in ((Path(args.e0), "PRD_E0_PASS"), (Path(args.e1), "PRD_E1_PASS"), (Path(args.paper_review), "PRD_PAPER_REVIEW_GO")):
        if not path.is_file():
            failures.append(f"missing {decision}")
        else:
            try:
                verify_pass(path, decision, commit)
            except SystemExit as exc:
                failures.append(str(exc))
    manifest = ROOT / "manifests/h20/qwen25_7b_prd_memrl_seed2026.json"
    capacities = json.loads(manifest.read_text())["method"]["capacity_points_nats"]
    if len(set(capacities)) < 3 or capacities != sorted(capacities):
        failures.append("invalid capacity frontier")
    result = certificate("PRD_P0_PASS" if not failures else "PRD_P0_NO_GO", "PASS" if not failures else "FAIL", {"failures": failures, "git_commit": commit, "gpu_pair": args.gpu_pair, "manifest_sha256": sha256(manifest)})
    write_json_exclusive(Path(args.output), result)
    return 0 if not failures else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    e0 = sub.add_parser("e0"); e0.add_argument("--output", required=True); e0.set_defaults(func=command_e0)
    e1 = sub.add_parser("e1"); e1.add_argument("--rows", required=True); e1.add_argument("--output", required=True); e1.set_defaults(func=command_e1)
    p0 = sub.add_parser("preflight")
    for name in ("expected_commit", "gpu_pair", "e0", "e1", "paper_review", "output"):
        p0.add_argument("--" + name.replace("_", "-"), required=True)
    p0.set_defaults(func=command_preflight)
    return args.func(args) if (args := parser.parse_args()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
