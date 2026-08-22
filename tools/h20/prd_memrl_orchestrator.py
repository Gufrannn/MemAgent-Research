#!/usr/bin/env python3
"""Fail-closed orchestration and audit for the PRD-MemRL H20 frontier.

This module never starts training.  It binds immutable inputs/outputs and validates
the artifacts produced by the production trainer before a later stage is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CAPACITIES = (128.0, 256.0, 512.0)
ANCHORS = (5, 10, 15, 20, 25)


def fail(message: str) -> None:
    raise SystemExit(f"PRD_NO_GO: {message}")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        fail(f"missing or symlinked evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def capacity_id(value: float) -> str:
    if value not in CAPACITIES:
        fail("capacity is outside frozen three-point frontier")
    return f"c{int(value)}"


def command_bind(args: argparse.Namespace) -> None:
    root = Path(args.run_root).resolve()
    if (root / "resolved_run.json").exists() or (root / "frontier").exists():
        fail("run was already bound; choose a new RUN_ID")
    baseline = load(Path(args.baseline).resolve())
    p0 = load(Path(args.p0).resolve())
    if baseline.get("status") != "PASS" or baseline.get("decision") != "PRD_ORIGINAL_BASELINE_IMPORT_PASS":
        fail("certified Original baseline import is not PASS")
    if baseline.get("stable_resolved_sha256") != "6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411":
        fail("baseline stable-S128 binding mismatch")
    if set(baseline.get("recomputed", {})) != {"0", "5", "10", "15", "20", "25"}:
        fail("baseline does not contain the complete six-anchor curve")
    if baseline.get("actual_loss_status") != "PENDING_ACTUAL_LOSS_LEDGER":
        fail("baseline actual-loss boundary missing")
    if baseline.get("original_training_final_report") != "/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/original_t25_final_report.json":
        fail("Original training report provenance mismatch")
    if baseline.get("original_training_final_report_sha256") != "33cab1eb09eefd89b7f764d0f2c6851eac5e58dc7c0a3d147c30ce05522c9040":
        fail("Original training report authentication mismatch")
    if not baseline.get("original_training_p0_sha256") or not baseline.get("original_training_resolved_sha256"):
        fail("Original training P0/protocol resolved manifest is unbound")
    if p0.get("status") != "PASS" or p0.get("decision") != "PRD_P0_PASS":
        fail("P0 is not PASS")
    if p0.get("evidence", {}).get("git_commit") != args.commit:
        fail("P0 commit drift")
    payload = {
        "schema_version": 1, "run_id": args.run_id, "git_commit": args.commit,
        "gpu_pair": args.gpu_pair, "fresh_base": True, "update1_enabled": True,
        "original_warm_start": False, "anchors": list(ANCHORS),
        "capacities_nats": list(CAPACITIES), "baseline_path": str(Path(args.baseline).resolve()),
        "baseline_sha256": digest(Path(args.baseline).resolve()),
        "p0_path": str(Path(args.p0).resolve()), "p0_sha256": digest(Path(args.p0).resolve()),
        "prior_model": p0["evidence"]["prior_model"],
    }
    exclusive(root / "resolved_run.json", payload)
    for cap in CAPACITIES:
        (root / "frontier" / capacity_id(cap)).mkdir(parents=True)


def validate_checkpoint(path: Path, step: int, cap: float, commit: str, run_id: str) -> dict:
    meta = load(path / "prd_checkpoint.json")
    required = {"actor", "actor_optimizer", "actor_scheduler", "prior", "prior_optimizer",
                "prior_scheduler", "dual", "rng", "global_step", "frontier_id", "weight_sync"}
    if set(meta.get("components", [])) != required:
        fail(f"checkpoint component set mismatch: {path}")
    if meta.get("global_step") != step or meta.get("frontier_id") != capacity_id(cap):
        fail(f"checkpoint step/frontier mismatch: {path}")
    if meta.get("git_commit") != commit or meta.get("run_id") != run_id:
        fail(f"checkpoint identity mismatch: {path}")
    if meta.get("method_active") is not True or meta.get("weight_sync", {}).get("verified") is not True:
        fail(f"method inactive or weight sync unverified: {path}")
    files = meta.get("files")
    if not isinstance(files, list) or not files:
        fail(f"checkpoint has no immutable file inventory: {path}")
    seen = set()
    for item in files:
        relative = item.get("path", "")
        candidate = (path / relative).resolve()
        if not relative or relative in seen or candidate.parent == path.parent or path.resolve() not in candidate.parents:
            fail(f"unsafe/duplicate checkpoint inventory path: {relative}")
        seen.add(relative)
        if not candidate.is_file() or candidate.is_symlink():
            fail(f"checkpoint inventory file missing/symlinked: {candidate}")
        if candidate.stat().st_size != item.get("size") or digest(candidate) != item.get("sha256"):
            fail(f"checkpoint inventory hash mismatch: {candidate}")
    required_prefixes = ("actor/", "prd_prior/")
    if any(not any(name.startswith(prefix) for name in seen) for prefix in required_prefixes):
        fail(f"checkpoint actor/prior inventories incomplete: {path}")
    required_patterns = (
        "actor/model_world_size_", "actor/optim_world_size_", "actor/extra_state_world_size_",
        "prd_prior/model_world_size_", "prd_prior/optim_world_size_", "prd_prior/extra_state_world_size_",
    )
    if any(not any(name.startswith(pattern) for name in seen) for pattern in required_patterns) or "data.pt" not in seen:
        fail(f"checkpoint optimizer/scheduler/RNG/dataloader inventory incomplete: {path}")
    actor_acks=meta.get("weight_sync",{}).get("actor")
    prior_acks=meta.get("weight_sync",{}).get("prior")
    if not isinstance(actor_acks,list) or not actor_acks or not isinstance(prior_acks,list) or not prior_acks:
        fail(f"checkpoint lacks actor/prior weight-sync acknowledgements: {path}")
    actor_ranks=sorted(int(ack["vllm_worker_rank"]) for ack in actor_acks)
    if actor_ranks != list(range(len(actor_acks))): fail(f"actor sync rank coverage mismatch: {path}")
    if len({ack["actor_master_sampled_tensor_digest"] for ack in actor_acks}) != 1 or len({ack["actor_rollout_sampled_tensor_digest"] for ack in actor_acks}) != 1:
        fail(f"actor sync digests disagree: {path}")
    prior_ranks=sorted(int(ack["rank"]) for ack in prior_acks)
    worlds={int(ack["world_size"]) for ack in prior_acks}
    if len(worlds)!=1 or prior_ranks!=list(range(worlds.pop())) or len({ack["global_parameter_moments_sha256"] for ack in prior_acks})!=1:
        fail(f"prior sync rank/digest disagreement: {path}")
    return meta


def command_stage(args: argparse.Namespace) -> None:
    run = load(Path(args.run_root) / "resolved_run.json")
    cap = float(args.capacity)
    cid = capacity_id(cap)
    if run["run_id"] != args.run_id or run["git_commit"] != args.commit:
        fail("resolved run identity drift")
    if args.stage == "full":
        if args.resume:
            fail("fresh T25 run forbids any resume/warm-start checkpoint")
        start, target = 0, 25
    else:
        if not args.resume:
            fail("continuation requires exact checkpoint")
        gate = load(Path(args.run_root) / "frontier" / cid / "t5_health.json")
        if gate.get("status") != "PASS" or gate.get("decision") != "PRD_T5_HEALTH_PASS":
            fail("recovery requires T5 health PASS")
        checkpoint = Path(args.resume).resolve()
        expected = (Path(args.run_root) / "frontier" / cid / "checkpoints" / "global_step_5").resolve()
        if checkpoint != expected:
            fail("resume path is not this run/capacity exact step 5")
        validate_checkpoint(checkpoint, 5, cap, args.commit, args.run_id)
        start, target = 5, 25
    output = Path(args.run_root) / "frontier" / cid
    launch = output / f"launch_{args.stage}.json"
    exclusive(launch, {"schema_version": 1, "run_id": args.run_id, "git_commit": args.commit,
        "frontier_id": cid, "capacity_nats": cap, "start_step": start, "target_step": target,
        "resume": str(Path(args.resume).resolve()) if args.resume else None,
        "output_root": str(output.resolve()), "gpu_pair": run["gpu_pair"],
        "prior_model": run["prior_model"],
        "fresh_base": args.stage == "full", "update1_enabled": True})
    print(launch)


def metric_rows(path: Path, expected_binding: dict | None = None) -> tuple[dict, set[str]]:
    from recurrent.research.s128_hotpot_metrics import score_terminal_output
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 128:
        fail(f"fixed-S128 requires exactly 128 rows: {path}")
    keys = [str(row.get("stable_key", "")) for row in rows]
    if "" in keys or len(set(keys)) != 128:
        fail(f"fixed-S128 stable identity failure: {path}")
    required = ("normalized_exact_match", "token_f1", "format_success")
    scored = []
    for row in rows:
        if expected_binding is not None and any(row.get(key) != value for key, value in expected_binding.items()):
            fail(f"raw terminal checkpoint/run binding mismatch: {path}")
        if "terminal_output" not in row or "ground_truth" not in row:
            fail(f"raw terminal output/ground truth missing: {path}")
        item = score_terminal_output(row["terminal_output"], row["ground_truth"])
        normalized = {"normalized_exact_match": item["exact_match"],
                      "token_f1": item["token_f1"], "format_success": item["format_success"]}
        if any(not 0 <= float(normalized[key]) <= 1 for key in required):
            fail(f"metric outside [0,1]: {path}")
        scored.append(normalized)
    return ({key: sum(float(row[key]) for row in scored) / 128 for key in required}, set(keys))


def command_evaluate(args: argparse.Namespace) -> None:
    cap = float(args.capacity); cid = capacity_id(cap); run_root = Path(args.run_root)
    run = load(run_root / "resolved_run.json")
    validate_checkpoint(run_root / "frontier" / cid / "checkpoints" / "global_step_25",
                        25, cap, run["git_commit"], run["run_id"])
    anchors = tuple(int(x) for x in args.anchors.split(","))
    if not anchors or len(set(anchors)) != len(anchors) or any(x not in ANCHORS for x in anchors):
        fail("evaluation anchors must be a unique subset of 5,10,15,20,25")
    domain = None
    existing_domains = []
    for old in (run_root / "frontier" / cid).glob("fixed_s128_anchor_*.json"):
        existing_domains.append(set(load(old).get("stable_keys", [])))
    for anchor in anchors:
        path = Path(args.input_template.format(anchor=anchor)).resolve()
        checkpoint = run_root / "frontier" / cid / "checkpoints" / f"global_step_{anchor}"
        validate_checkpoint(checkpoint, anchor, cap, run["git_commit"], run["run_id"])
        metadata_sha = digest(checkpoint / "prd_checkpoint.json")
        binding = {"run_id": run["run_id"], "git_commit": run["git_commit"], "frontier_id": cid,
                   "global_step": anchor, "checkpoint_metadata_sha256": metadata_sha}
        summary, keys = metric_rows(path, binding)
        if (domain is not None and keys != domain) or any(keys != old for old in existing_domains):
            fail("stable-ID cohort drift across anchors")
        domain = keys
        output = run_root / "frontier" / cid / f"fixed_s128_anchor_{anchor}.json"
        exclusive(output, {"schema_version": 1, "status": "PASS", "decision": "PRD_FIXED_S128_PASS",
            "capacity_nats": cap, "anchor": anchor, "metrics": summary,
            "rows_sha256": digest(path), "checkpoint_binding": binding, "stable_keys": sorted(keys)})


def command_audit(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root); run = load(run_root / "resolved_run.json")
    failures = []; comparisons = {}
    try:
        if digest(Path(run["baseline_path"])) != run["baseline_sha256"]: fail("baseline drift")
        for cap in CAPACITIES:
            cid = capacity_id(cap)
            health = load(run_root / "frontier" / cid / "t5_health.json")
            if health.get("status") != "PASS" or health.get("decision") != "PRD_T5_HEALTH_PASS": fail(f"T5 health is not PASS for {cid}")
            validate_checkpoint(run_root / "frontier" / cid / "checkpoints" / "global_step_25", 25, cap, run["git_commit"], run["run_id"])
            domains = []
            for anchor in ANCHORS:
                summary = load(run_root / "frontier" / cid / f"fixed_s128_anchor_{anchor}.json")
                checkpoint = run_root / "frontier" / cid / "checkpoints" / f"global_step_{anchor}"
                expected_binding = {"run_id": run["run_id"], "git_commit": run["git_commit"],
                    "frontier_id": cid, "global_step": anchor,
                    "checkpoint_metadata_sha256": digest(checkpoint / "prd_checkpoint.json")}
                if summary.get("checkpoint_binding") != expected_binding:
                    fail(f"fixed-S128 checkpoint binding mismatch for {cid} step {anchor}")
                domains.append(set(summary.get("stable_keys", [])))
                original = load(Path(run["baseline_path"]))["recomputed"][str(anchor)]
                method = summary["metrics"]
                comparisons.setdefault(cid, {})[str(anchor)] = {
                    "method": method, "original": original,
                    "method_minus_original": {key: float(method[key]) - float(original[key])
                        for key in ("normalized_exact_match", "token_f1", "format_success")},
                    "interpretation": "paired descriptive difference on the same frozen S128; not causal or population inference",
                }
            if not domains[0] or any(keys != domains[0] for keys in domains[1:]): fail("incomplete or drifting anchor cohort")
    except SystemExit as exc: failures.append(str(exc))
    ledger = Path(args.ledger)
    if not ledger.is_file(): failures.append("missing execution ledger")
    else:
        result = subprocess.run([sys.executable, str(ROOT/'tools/h20/prd_memrl_ledger.py'),
            "verify", "--ledger", str(ledger)], capture_output=True, text=True, check=False)
        if result.returncode: failures.append("ledger verification failed")
        else:
            records=[json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
            for cap in CAPACITIES:
                cid=capacity_id(cap)
                updates={int(r["payload"]["global_step"]) for r in records if r["event"]=="UPDATE" and r["payload"].get("frontier_id")==cid}
                checkpoints={int(r["payload"]["global_step"]) for r in records if r["event"]=="CHECKPOINT" and r["payload"].get("frontier_id")==cid}
                health=[r for r in records if r["event"]=="T5_HEALTH" and r["payload"].get("frontier_id")==cid]
                if not set(range(1,26)).issubset(updates): failures.append(f"incomplete UPDATE ledger for {cid}")
                if not set(ANCHORS).issubset(checkpoints): failures.append(f"incomplete CHECKPOINT ledger for {cid}")
                if len(health)!=1 or health[0]["payload"].get("decision")!="PRD_T5_HEALTH_PASS": failures.append(f"invalid T5 health ledger for {cid}")
                for record in records:
                    if record["event"]=="CHECKPOINT" and record["payload"].get("frontier_id")==cid:
                        step=int(record["payload"]["global_step"])
                        metadata=run_root/"frontier"/cid/"checkpoints"/f"global_step_{step}"/"prd_checkpoint.json"
                        if not metadata.is_file() or digest(metadata)!=record["payload"].get("metadata_sha256"):
                            failures.append(f"checkpoint ledger SHA mismatch for {cid} step {step}")
    output = Path(args.output)
    exclusive(output, {"schema_version": 1, "status": "PASS" if not failures else "FAIL",
        "decision": "PRD_FINAL_AUDIT_PASS" if not failures else "PRD_FINAL_AUDIT_NO_GO",
        "paired_descriptive_comparisons": comparisons, "failures": failures})
    output.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    if failures: raise SystemExit(8)


def main() -> None:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bind")
    for x in ("run-root", "run-id", "commit", "gpu-pair", "baseline", "p0"): b.add_argument("--"+x, required=True)
    b.set_defaults(func=command_bind)
    s = sub.add_parser("stage"); s.add_argument("stage", choices=("full", "continue"))
    for x in ("run-root", "run-id", "commit", "capacity"): s.add_argument("--"+x, required=True)
    s.add_argument("--resume"); s.set_defaults(func=command_stage)
    e = sub.add_parser("evaluate"); e.add_argument("--run-root", required=True); e.add_argument("--capacity", required=True); e.add_argument("--input-template", required=True); e.add_argument("--anchors", default="5,10,15,20,25"); e.set_defaults(func=command_evaluate)
    a = sub.add_parser("audit"); a.add_argument("--run-root", required=True); a.add_argument("--ledger", required=True); a.add_argument("--output", required=True); a.set_defaults(func=command_audit)
    args = p.parse_args(); args.func(args)


if __name__ == "__main__": main()
