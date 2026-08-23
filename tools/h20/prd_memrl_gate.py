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
    from recurrent.research.prd_memrl import PriorTaintError, ProjectedDual, load_prd_checkpoint, save_prd_checkpoint, validate_prior_record

    failures = []
    error = torch.tensor(0.2, dtype=torch.float64)
    mi_tensor = torch.log(torch.tensor(2.0, dtype=torch.float64)) + error * torch.log(error) + (1-error) * torch.log(1-error)
    # Exact expectation for a uniform-input binary symmetric channel and its
    # exact history-blind output marginal q(y)=1/2.
    bound = (1-error) * torch.log(2*(1-error)) + error * torch.log(2*error)
    mi = float(mi_tensor)
    if abs(float(bound) - mi) > 1e-12:
        failures.append("synthetic KL upper bound failed")
    if abs(float(bound - mi_tensor)) > 1e-12:
        failures.append("synthetic KL decomposition failed")
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
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        saved = save_prd_checkpoint(directory, actor_state={"w": torch.ones(1)}, prior_state={"w": torch.zeros(1)}, dual=dual)
        actor_state, prior_state, restored = load_prd_checkpoint(saved)
        if actor_state["w"].item() != 1 or prior_state["w"].item() != 0 or restored.state_dict() != dual.state_dict():
            failures.append("actor/prior/dual checkpoint roundtrip failed")
    result = certificate("PRD_E0_PASS" if not failures else "PRD_E0_NO_GO", "PASS" if not failures else "FAIL", {"failures": failures, "synthetic_mi_nats": mi, "variational_bound_nats": float(bound), "git_commit": git("rev-parse", "HEAD")})
    write_json_exclusive(Path(args.output), result)
    return 0 if not failures else 2


def command_e1(args: argparse.Namespace) -> int:
    from recurrent.research.prd_memrl import assert_rate_not_length

    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    required = {
        "stable_id", "turn_index", "writer_tokens", "actor_logprob_sum",
        "prior_logprob_sum", "unigram_logprob_sum", "turn_only_logprob_sum",
        "entropy_nats", "reference_kl_nats", "prior_context_sha256",
        "mutated_evidence_prior_context_sha256",
    }
    failures = []
    if not rows or any(set(row) < required for row in rows):
        failures.append("missing exact E1 row fields")
    discordance = None
    coding_gain = None
    controlled_r2 = None
    if not failures:
        import numpy as np
        rates = [float(row["actor_logprob_sum"]) - float(row["prior_logprob_sum"]) for row in rows]
        try:
            discordance = assert_rate_not_length(rates, [int(row["writer_tokens"]) for row in rows])
        except ValueError as exc:
            failures.append(str(exc))
        coding_gain = sum(float(row["prior_logprob_sum"]) - max(float(row["unigram_logprob_sum"]), float(row["turn_only_logprob_sum"])) for row in rows) / len(rows)
        if coding_gain <= 0:
            failures.append("legal learned prior has no coding gain over legal baselines")
        if any(row["prior_context_sha256"] != row["mutated_evidence_prior_context_sha256"] for row in rows):
            failures.append("dynamic evidence mutation changed history-blind prior context")
        design = np.asarray([[1.0, float(row["writer_tokens"]), float(row["turn_index"]), float(row["entropy_nats"]), float(row["reference_kl_nats"])] for row in rows])
        target = np.asarray(rates)
        prediction = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        denominator = float(((target-target.mean())**2).sum())
        controlled_r2 = 1.0 if denominator == 0 else 1.0-float(((target-prediction)**2).sum())/denominator
        if controlled_r2 >= 0.95:
            failures.append("conditional rate is explained by length/turn/entropy/reference-KL controls")
    result = certificate("PRD_E1_PASS" if not failures else "PRD_E1_NO_GO", "PASS" if not failures else "FAIL", {"failures": failures, "row_count": len(rows), "rate_length_discordance": discordance, "legal_prior_coding_gain_nats": coding_gain, "controlled_r2": controlled_r2, "rows_sha256": sha256(Path(args.rows)), "git_commit": git("rev-parse", "HEAD")})
    write_json_exclusive(Path(args.output), result)
    return 0 if not failures else 3


def verify_pass(path: Path, decision: str, commit: str) -> None:
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS" or payload.get("decision") != decision or payload.get("evidence", {}).get("git_commit") != commit:
        raise SystemExit(f"PRD_NO_GO invalid {decision} certificate")


def verify_data_overlap(path: Path, commit: str) -> dict:
    payload = json.loads(path.read_text())
    evidence, sources = payload.get("evidence", {}), payload.get("sources", {})
    train, fixed = sources.get("train", {}), sources.get("fixed_s128_validation", {})
    producer, intersections = payload.get("producer", {}), payload.get("intersections", {})
    expected_train = Path("/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_train_32k.parquet")
    expected_val = Path("/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_dev.parquet")
    expected_stable = Path("/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json")
    expected_producer = (ROOT / "tools/h20/audit_prd_data_overlap.py").resolve()
    checks = (
        payload.get("status") == "PASS", payload.get("decision") == "PRD_DATA_OVERLAP_PASS",
        evidence.get("git_commit") == commit, payload.get("failures") == [],
        train.get("path") == str(expected_train), fixed.get("path") == str(expected_val),
        fixed.get("resolved") == str(expected_stable),
        fixed.get("sha256") == "54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6",
        fixed.get("resolved_sha256") == "6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411",
        fixed.get("rows") == 128, isinstance(train.get("rows"), int) and train.get("rows", 0) > 128,
        intersections == {"train_pool_and_s128_content": 0, "train_pool_and_s128_root": 0,
                            "critic_fit_and_s128": 0, "selection_and_s128": 128},
        producer.get("path") == str(expected_producer), producer.get("sha256") == sha256(expected_producer),
    )
    if not all(checks):
        raise SystemExit("PRD_NO_GO invalid PRD_DATA_OVERLAP_PASS certificate")
    for candidate, recorded in ((expected_train, train.get("sha256")), (expected_val, fixed.get("sha256")),
                                (expected_stable, fixed.get("resolved_sha256"))):
        if not candidate.is_file() or candidate.is_symlink() or sha256(candidate) != recorded:
            raise SystemExit(f"PRD_NO_GO data identity drift after overlap audit: {candidate}")
    return payload


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
    for path, decision in ((Path(args.e0), "PRD_E0_PASS"), (Path(args.paper_review), "PRD_PAPER_REVIEW_GO")):
        if not path.is_file():
            failures.append(f"missing {decision}")
        else:
            try:
                verify_pass(path, decision, commit)
            except SystemExit as exc:
                failures.append(str(exc))
    overlap = None
    try:
        overlap = verify_data_overlap(Path(args.data_overlap), commit)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, SystemExit) as exc:
        failures.append(str(exc))
    manifest = ROOT / "manifests/h20/qwen25_7b_prd_memrl_seed2026.json"
    capacities = json.loads(manifest.read_text())["method"]["capacity_points_nats"]
    if len(set(capacities)) < 3 or capacities != sorted(capacities):
        failures.append("invalid capacity frontier")
    prior_root=Path(args.prior_model).resolve(); prior_files=[]
    try:
        prior_config=json.loads((prior_root/"config.json").read_text())
        if prior_config.get("hidden_size")!=896 or prior_config.get("num_hidden_layers")!=24:
            failures.append("prior is not the frozen Qwen2.5-0.5B architecture")
        for path in sorted(prior_root.iterdir()):
            if path.is_file() and (path.suffix in {".json",".safetensors",".txt"}):
                prior_files.append({"path":path.name,"size":path.stat().st_size,"sha256":sha256(path)})
        if not any(item["path"].endswith(".safetensors") for item in prior_files): failures.append("prior weights missing")
    except Exception as exc: failures.append(f"invalid prior model inventory: {exc}")
    base_root=Path(args.base_model).resolve(); base_files=[]; training_resolved=Path(args.original_training_resolved).resolve()
    try:
        config=json.loads((base_root/"config.json").read_text())
        if base_root!=Path("/data/cw/memagent_work/models/Qwen2.5-7B-Instruct") or config.get("hidden_size")!=3584 or config.get("num_hidden_layers")!=28:
            failures.append("base is not the frozen Qwen2.5-7B architecture/path")
        for path in sorted(base_root.iterdir()):
            if path.is_file() and path.suffix in {".json",".safetensors",".txt"}: base_files.append({"path":path.name,"size":path.stat().st_size,"sha256":sha256(path)})
        if not any(item["path"].endswith(".safetensors") for item in base_files): failures.append("base weights missing")
        resolved_text=training_resolved.read_text()
        if str(base_root) not in resolved_text: failures.append("Original training resolved does not bind the base path")
    except Exception as exc: failures.append(f"invalid base/original resolved binding: {exc}")
    overlap_binding = None if overlap is None else {"certificate_path": str(Path(args.data_overlap).resolve()), "certificate_sha256": sha256(Path(args.data_overlap)), "sources": overlap["sources"], "intersections": overlap["intersections"], "producer": overlap["producer"]}
    result = certificate("PRD_P0_PASS" if not failures else "PRD_P0_NO_GO", "PASS" if not failures else "FAIL", {"failures": failures, "git_commit": commit, "gpu_pair": args.gpu_pair, "manifest_sha256": sha256(manifest), "data_overlap": overlap_binding, "prior_model":{"id":"Qwen/Qwen2.5-0.5B-Instruct","revision":"c89bee90d9f811437d9735454613c35b4a3c4dc8","path":str(prior_root),"files":prior_files},"base_model":{"id":"Qwen2.5-7B-Instruct","revision":"a09a35458c702b33eeacc393d103063234e8bc28","path":str(base_root),"files":base_files},"original_training_resolved":{"path":str(training_resolved),"sha256":sha256(training_resolved)}})
    write_json_exclusive(Path(args.output), result)
    return 0 if not failures else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    e0 = sub.add_parser("e0"); e0.add_argument("--output", required=True); e0.set_defaults(func=command_e0)
    e1 = sub.add_parser("e1"); e1.add_argument("--rows", required=True); e1.add_argument("--output", required=True); e1.set_defaults(func=command_e1)
    p0 = sub.add_parser("preflight")
    for name in ("expected_commit", "gpu_pair", "e0", "paper_review", "data_overlap", "prior_model", "base_model", "original_training_resolved", "output"):
        p0.add_argument("--" + name.replace("_", "-"), required=True)
    p0.set_defaults(func=command_preflight)
    return args.func(args) if (args := parser.parse_args()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
