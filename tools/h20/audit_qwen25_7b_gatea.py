#!/usr/bin/env python3
"""Generate P1 or final P0/P1/P2 + A1-A5 Gate A certificates without rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import checkpoint_inventory, load_frozen_manifest, sha256_file
from recurrent.research.trajectory_seeding import build_trajectory_seed_records, derive_turn_request_seeds


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def component_inventory(step_dir: Path, expected_world_size: int) -> tuple[list[dict], list[str]]:
    inventory = checkpoint_inventory(step_dir) if step_dir.is_dir() else []
    names = [item["path"] for item in inventory if item["size"] > 0]
    missing = []
    component_prefixes = {
        "model": "model",
        "optim": "optim",
        "extra": "extra_state",
    }
    expected_ranks = list(range(expected_world_size))
    for component, prefix in component_prefixes.items():
        pattern = re.compile(
            rf"^actor/{prefix}_world_size_{expected_world_size}_rank_(\d+)\.pt$"
        )
        ranks = sorted(int(match.group(1)) for name in names if (match := pattern.match(name)))
        if ranks != expected_ranks:
            missing.append(f"{component}_ranks_{ranks}")
    if "data.pt" not in names:
        missing.append("data")
    return inventory, missing


def audit_seeds(seed_records: list[dict], seed: int, rollout_n: int) -> tuple[bool, list[str]]:
    failures = []
    if not seed_records:
        return False, ["missing rollout seed records"]
    base_records = [row for row in seed_records if row.get("record_type", "trajectory_seed") == "trajectory_seed"]
    turn_records = [row for row in seed_records if row.get("record_type") == "trajectory_turn_seed"]
    if not base_records:
        return False, ["missing base trajectory seed records"]
    if not turn_records:
        failures.append("missing recurrent turn seed records")
    for step in sorted({int(row["global_step"]) for row in base_records}):
        rows = [row for row in base_records if int(row["global_step"]) == step]
        required = {"row", "group", "replica", "global_step", "trajectory_seed", "uid", "mode"}
        for index, row in enumerate(rows):
            missing = sorted(required - row.keys())
            if missing:
                failures.append(f"seed row {index} at step {step} missing identity fields {missing}")
        identity_keys = [
            (row.get("uid"), int(row.get("replica", -1)), int(row.get("row", -1))) for row in rows
        ]
        if len(identity_keys) != len(set(identity_keys)):
            failures.append(f"trajectory identity collision at step {step}")
        actual = [int(row["trajectory_seed"]) for row in rows]
        if len(actual) != len(set(actual)):
            failures.append(f"trajectory seed collision at step {step}")
        expected = build_trajectory_seed_records(
            base_seed=seed,
            global_step=step,
            batch_size=len(rows),
            rollout_n=rollout_n,
            mode="independent",
        )
        if actual != [int(row["trajectory_seed"]) for row in expected]:
            failures.append(f"seed schedule is not reconstructable at step {step}")

        base_by_row = {int(row["row"]): row for row in rows}
        step_turns = [row for row in turn_records if int(row["global_step"]) == step]
        turn_keys = [
            (row.get("uid"), int(row.get("replica", -1)), int(row.get("turn", -1)))
            for row in step_turns
        ]
        if len(turn_keys) != len(set(turn_keys)):
            failures.append(f"trajectory turn identity collision at step {step}")
        turns_by_row: dict[int, list[int]] = defaultdict(list)
        for row in step_turns:
            source_row = int(row.get("sample_index", -1))
            base = base_by_row.get(source_row)
            if base is None:
                failures.append(f"turn record references unknown sample_index {source_row} at step {step}")
                continue
            if (
                row.get("uid") != base.get("uid")
                or int(row.get("replica", -1)) != int(base["replica"])
                or int(row.get("trajectory_seed", -1)) != int(base["trajectory_seed"])
            ):
                failures.append(f"turn/base trajectory identity mismatch at step {step}, row {source_row}")
            turn = int(row.get("turn", -1))
            turns_by_row[source_row].append(turn)
            if turn < 0:
                failures.append(f"invalid recurrent turn at step {step}, row {source_row}: {turn}")
                continue
            expected_request_seed = derive_turn_request_seeds(
                [int(base["trajectory_seed"])], [0], turn
            )[0]
            if int(row.get("request_seed", -1)) != expected_request_seed:
                failures.append(f"request seed is not reconstructable at step {step}, row {source_row}, turn {turn}")
        for source_row in base_by_row:
            actual_turns = sorted(turns_by_row[source_row])
            if actual_turns != list(range(len(actual_turns))) or not actual_turns:
                failures.append(
                    f"trajectory turns are missing or non-contiguous at step {step}, "
                    f"row {source_row}: {actual_turns}"
                )
    return not failures, failures


def audit_sync(
    records: list[dict],
    required_versions: list[int],
    ranks: list[int],
    required_syncs: list[tuple[str, int, str]] | None = None,
) -> tuple[bool, list[str], dict[int, str]]:
    failures = []
    digests_by_version: dict[int, set[str]] = defaultdict(set)
    syncs = required_syncs or [("", version, "") for version in required_versions]
    for experiment, version, sync_kind in syncs:
        acks = [
            row for row in records
            if row.get("record_type") == "weight_sync_ack"
            and int(row.get("actor_version", -1)) == version
            and (not experiment or row.get("experiment_name") == experiment)
            and (not sync_kind or row.get("sync_kind") == sync_kind)
        ]
        actual_ranks = sorted({int(row["vllm_worker_rank"]) for row in acks})
        context = f"{experiment or '*'}:{version}:{sync_kind or '*'}"
        if actual_ranks != ranks:
            failures.append(f"sync {context} ack ranks {actual_ranks} != {ranks}")
        if len(acks) != len(ranks):
            failures.append(f"sync {context} expected {len(ranks)} unique acks, found {len(acks)}")
        for row in acks:
            if row["actor_sampled_tensor_digest"] != row["vllm_sampled_tensor_digest"]:
                failures.append(f"actor/vLLM digest mismatch at sync {context}, rank {row['vllm_worker_rank']}")
            if int(row["vllm_ack_version"]) != version:
                failures.append(f"ack version mismatch at actor version {version}")
            digests_by_version[version].add(row["actor_sampled_tensor_digest"])
        if len(digests_by_version[version]) != 1:
            failures.append(f"sync {context} has split or missing actor digests")
    collapsed = {version: next(iter(values)) for version, values in digests_by_version.items() if len(values) == 1}
    return not failures, failures, collapsed


def build_report(manifest: dict, phase: str) -> tuple[dict, list[dict]]:
    paths = manifest["paths"]
    ledger_path = Path(paths["execution_ledger"])
    ledger = read_jsonl(ledger_path)
    fresh_experiment = manifest["experiments"]["fresh"]
    resume_experiment = manifest["experiments"]["resume"]
    accepted_experiments = {fresh_experiment} | ({resume_experiment} if phase == "final" else set())
    execution_records = [row for row in ledger if row.get("experiment_name") in accepted_experiments]
    fresh_dir = Path(paths["fresh_output"])
    resume_dir = Path(paths["resume_output"])
    fresh_seeds = read_jsonl(fresh_dir / "rollout_seed_audit.jsonl")
    resume_seeds = read_jsonl(resume_dir / "rollout_seed_audit.jsonl")
    a1_ok, a1_failures = audit_seeds(
        fresh_seeds + (resume_seeds if phase == "final" else []),
        manifest["training"]["seed"],
        manifest["training"]["rollout_n"],
    )

    expected_world_size = int(manifest["gpu"]["world_size"])
    step2_inventory, step2_missing = component_inventory(
        fresh_dir / "global_step_2", expected_world_size
    )
    step3_inventory, step3_missing = (
        component_inventory(resume_dir / "global_step_3", expected_world_size)
        if phase == "final" else ([], [])
    )
    a2_failures = [f"step2 missing {item}" for item in step2_missing]
    if phase == "final":
        a2_failures.extend(f"step3 missing {item}" for item in step3_missing)
        p1_report_path = Path(paths["certificate_root"]) / "p1_audit_report.json"
        if not p1_report_path.is_file():
            a2_failures.append("missing immutable P1 audit report")
        else:
            frozen_step2 = json.loads(p1_report_path.read_text()).get("step2_inventory", [])
            if frozen_step2 != step2_inventory:
                a2_failures.append("step2 checkpoint inventory changed after P1 certification")

    a3_failures = []
    if phase == "final":
        loads = [
            row for row in execution_records
            if row.get("record_type") == "resume_load" and row.get("experiment_name") == resume_experiment
        ]
        expected_source = str(Path(paths["resume_source"]).resolve())
        if len(loads) != 1:
            a3_failures.append(f"expected exactly one resume_load record, found {len(loads)}")
        else:
            load = loads[0]
            if load.get("resume_source") != expected_source or int(load.get("global_step", -1)) != 2:
                a3_failures.append(f"resume source/global step drift: {load}")
            load_acks = load.get("actor_load_worker_acks") or []
            expected_ranks = manifest["weight_sync"]["required_worker_ranks"]
            actual_ranks = sorted(int(ack.get("rank", -1)) for ack in load_acks)
            if actual_ranks != expected_ranks:
                a3_failures.append(f"resume actor load ack ranks {actual_ranks} != {expected_ranks}")
            for ack in load_acks:
                if not all(ack.get(key) for key in ("model_loaded", "optimizer_loaded", "extra_loaded")):
                    a3_failures.append(f"incomplete actor checkpoint load ack: {ack}")
                if ack.get("optimizer_step_min") is None or int(ack["optimizer_step_min"]) < 1:
                    a3_failures.append(f"missing/non-positive loaded optimizer step: {ack}")
                if ack.get("lr_scheduler_last_epoch") is None or int(ack["lr_scheduler_last_epoch"]) < 1:
                    a3_failures.append(f"missing/non-positive loaded scheduler epoch: {ack}")
            data_item = next((item for item in step2_inventory if item["path"] == "data.pt"), None)
            if not data_item or load.get("data_sha256") != data_item["sha256"]:
                a3_failures.append("loaded data cursor SHA does not match P1 step2 data.pt")
            step3_data = next((item for item in step3_inventory if item["path"] == "data.pt"), None)
            if data_item and step3_data and data_item["sha256"] == step3_data["sha256"]:
                a3_failures.append("step3 data cursor did not advance from step2")

        resume_syncs = [
            row for row in execution_records
            if row.get("record_type") == "weight_sync_ack"
            and row.get("experiment_name") == resume_experiment
            and int(row.get("actor_version", -1)) in (2, 3)
        ]
        for rank in manifest["weight_sync"]["required_worker_ranks"]:
            loaded = next(
                (row for row in resume_syncs if int(row["vllm_worker_rank"]) == rank and int(row["actor_version"]) == 2),
                None,
            )
            updated = next(
                (row for row in resume_syncs if int(row["vllm_worker_rank"]) == rank and int(row["actor_version"]) == 3),
                None,
            )
            if loaded is None or updated is None:
                a3_failures.append(f"rank {rank} missing resume optimizer continuity evidence")
                continue
            loaded_step = loaded.get("optimizer_step_min")
            updated_step = updated.get("optimizer_step_min")
            loaded_epoch = loaded.get("lr_scheduler_last_epoch")
            updated_epoch = updated.get("lr_scheduler_last_epoch")
            if loaded_step is None or updated_step is None or int(updated_step) <= int(loaded_step):
                a3_failures.append(
                    f"rank {rank} optimizer step did not advance across resume: {loaded_step} -> {updated_step}"
                )
            if loaded_epoch is None or updated_epoch is None or int(updated_epoch) <= int(loaded_epoch):
                a3_failures.append(
                    f"rank {rank} scheduler epoch did not advance across resume: {loaded_epoch} -> {updated_epoch}"
                )

    required_versions = [0, 1, 2] + ([3] if phase == "final" else [])
    required_syncs = [
        (fresh_experiment, 0, "fresh_initial"),
        (fresh_experiment, 1, "post_actor_update"),
        (fresh_experiment, 2, "post_actor_update"),
    ]
    if phase == "final":
        required_syncs.extend([
            (resume_experiment, 2, "resume_loaded"),
            (resume_experiment, 3, "post_actor_update"),
        ])
    a4_ok, a4_failures, digests = audit_sync(
        execution_records,
        required_versions,
        manifest["weight_sync"]["required_worker_ranks"],
        required_syncs,
    )

    signal_steps = {
        int(row["global_step"]): row
        for row in execution_records
        if row.get("record_type") == "execution_signal"
    }
    required_signal_steps = [1, 2] + ([3] if phase == "final" else [])
    a5_failures = []
    signal_summary: dict[int, dict] = {}
    for step in required_signal_steps:
        record = signal_steps.get(step)
        if record is None:
            a5_failures.append(f"missing execution signal for step {step}")
            continue
        values = record.get("metrics", {}).values()
        if any(not math.isfinite(float(value)) for value in values):
            a5_failures.append(f"non-finite execution metric at step {step}")
        metrics = record.get("metrics", {})
        required_metric_fragments = ("grad_norm", "pg_loss", "rewards/", "advantages/")
        missing_fragments = [
            fragment for fragment in required_metric_fragments
            if not any(fragment in key for key in metrics)
        ]
        if missing_fragments:
            a5_failures.append(f"step {step} missing execution metric families {missing_fragments}")
        signal_summary[step] = {
            "grad_norm": metrics.get("actor/grad_norm"),
            "reward_mean": metrics.get("critic/rewards/mean"),
            "advantage_min": metrics.get("critic/advantages/min"),
            "advantage_max": metrics.get("critic/advantages/max"),
            "pg_loss": metrics.get("actor/pg_loss"),
        }
    fresh_signals = [signal_summary.get(step, {}) for step in (1, 2)]
    if not any(abs(float(row.get("grad_norm") or 0.0)) > 0 for row in fresh_signals):
        a5_failures.append("both fresh steps have zero or missing grad norm")
    if not any(
        abs(float(row.get("advantage_min") or 0.0)) > 0
        or abs(float(row.get("advantage_max") or 0.0)) > 0
        for row in fresh_signals
    ):
        a5_failures.append("both fresh steps have zero or missing advantages")
    fresh_delta = any(digests.get(left) != digests.get(right) for left, right in ((0, 1), (1, 2)))
    if not fresh_delta:
        a5_failures.append("fresh versions 0→1→2 contain no sampled parameter delta")
    if phase == "final" and digests.get(2) == digests.get(3):
        a5_failures.append("resume version 2→3 contains no sampled parameter delta")

    audits = {
        "A1": {"status": "PASS" if a1_ok else "FAIL", "failures": a1_failures},
        "A2": {"status": "PASS" if not a2_failures else "FAIL", "failures": a2_failures},
        "A3": {"status": "PASS" if not a3_failures else "FAIL", "failures": a3_failures, "applicable": phase == "final"},
        "A4": {"status": "PASS" if a4_ok else "FAIL", "failures": a4_failures, "version_digests": digests},
        "A5": {
            "status": "PASS" if not a5_failures else "FAIL",
            "failures": a5_failures,
            "signals": signal_summary,
            "zero_grad_fraction": (
                sum(abs(float(row.get("grad_norm") or 0.0)) == 0 for row in signal_summary.values())
                / len(signal_summary)
                if signal_summary else None
            ),
        },
    }
    applicable = ["A1", "A2", "A4", "A5"] + (["A3"] if phase == "final" else [])
    p0_path = Path(paths["certificate_root"]) / "p0_preflight.json"
    p0_certificate = json.loads(p0_path.read_text()) if p0_path.is_file() else {}
    p0_commit = p0_certificate.get("evidence", {}).get("git_commit")
    resolved_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    resolved_manifest_sha256 = hashlib.sha256(resolved_payload).hexdigest()
    p0_resolved_manifest_sha256 = p0_certificate.get("evidence", {}).get("resolved_manifest_sha256")
    p0_resolved_manifest_path = p0_certificate.get("evidence", {}).get("resolved_manifest_path")
    resolved_manifest_file_matches = False
    if p0_resolved_manifest_path and Path(p0_resolved_manifest_path).is_file():
        frozen_resolved_manifest = json.loads(Path(p0_resolved_manifest_path).read_text(encoding="utf-8"))
        resolved_manifest_file_matches = frozen_resolved_manifest == manifest
    ledger_commits = {row.get("git_commit") for row in execution_records}
    current_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    p0_pass = (
        p0_certificate.get("status") == "PASS"
        and p0_commit == current_commit
        and p0_resolved_manifest_sha256 == resolved_manifest_sha256
        and resolved_manifest_file_matches
        and ledger_commits == {current_commit}
    )
    p1_pass = not step2_missing and all(step in signal_steps for step in (1, 2))
    p2_pass = phase != "final" or (
        not step3_missing and 3 in signal_steps and not a3_failures and digests.get(2) != digests.get(3)
    )
    gates = {
        "P0": "PASS" if p0_pass else "FAIL",
        "P1": "PASS" if p1_pass else "FAIL",
        "P2": "PASS" if p2_pass else ("NOT_APPLICABLE" if phase != "final" else "FAIL"),
    }
    required_gates = ["P0", "P1"] + (["P2"] if phase == "final" else [])
    passed = all(gates[name] == "PASS" for name in required_gates) and all(
        audits[name]["status"] == "PASS" for name in applicable
    )
    failed_order = [name for name in required_gates if gates[name] != "PASS"] + [
        name for name in applicable if audits[name]["status"] == "FAIL"
    ]
    report = {
        "phase": phase,
        "status": "PASS" if passed else "FAIL",
        "decision": ("GATE_A_PASS" if phase == "final" else "P1_AUDIT_PASS") if passed else f"GATE_A_NO_GO:{failed_order[0]}",
        "gates": gates,
        "audits": audits,
        "ledger_sha256": sha256_file(ledger_path) if ledger_path.is_file() else None,
        "step2_inventory": step2_inventory,
        "step3_inventory": step3_inventory,
    }
    inventory_records = [
        {"global_step": 2, "inventory": step2_inventory},
        *([{"global_step": 3, "inventory": step3_inventory}] if phase == "final" else []),
    ]
    return report, inventory_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("p1", "final"), default="final")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_frozen_manifest(args.manifest)
    except ValueError as error:
        print(json.dumps({
            "phase": args.phase,
            "status": "FAIL",
            "decision": "GATE_A_NO_GO:P0",
            "failures": [str(error)],
        }, indent=2, sort_keys=True))
        return 1
    report, inventory_records = build_report(manifest, args.phase)
    if args.write_report:
        name = "p1_audit_report.json" if args.phase == "p1" else "gate_a_final_report.json"
        target = args.output or Path(manifest["paths"]["certificate_root"]) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise SystemExit(f"refusing to overwrite append-only report: {target}")
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        from recurrent.research.gate_a_execution import append_gate_a_record

        for inventory_record in inventory_records:
            append_gate_a_record("checkpoint_inventory", **inventory_record)
        append_gate_a_record(
            "audit_result",
            phase=args.phase,
            status=report["status"],
            decision=report["decision"],
            report=str(target),
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
