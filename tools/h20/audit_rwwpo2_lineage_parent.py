#!/usr/bin/env python3
"""Authenticate the checkpoint and ledger prefix imported by a resumed attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.gate_a_execution import checkpoint_inventory, validate_jsonl_chain
from recurrent.research.rwwpo_ledger import tensor_shard_inventory
from tools.h20.audit_rwwpo_actual_loss import audit, canonical_sha


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execution_prefix_to_checkpoint(path: Path, *, checkpoint_round: int,
                                   expected_commit: str):
    """Read only the authenticated execution prefix ending at the checkpoint.

    Bytes after the checkpoint belong to a possibly failed suffix and are not
    parsed.  This is the operational meaning of the authenticated lineage DAG.
    """
    events = []
    raw_prefix = bytearray()
    with path.open("rb") as stream:
        for physical_line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception as error:
                raise ValueError(
                    f"execution prefix malformed before checkpoint:{physical_line_no}"
                ) from error
            events.append(row)
            raw_prefix.extend(raw)
            if row.get("record_type") == "checkpoint_inventory" \
                    and int(row.get("global_step", -1)) == checkpoint_round:
                break
    inventories = [row for row in events if row.get("record_type") ==
                   "checkpoint_inventory" and int(row.get("global_step", -1)) ==
                   checkpoint_round]
    if len(inventories) != 1 or validate_jsonl_chain(events) \
            or any(row.get("git_commit") != expected_commit for row in events):
        raise ValueError("execution checkpoint prefix chain/identity")
    return events, inventories[0], hashlib.sha256(raw_prefix).hexdigest()


def marker_prefix_audit(ledger_dir: Path, *, through_round: int,
                        record_limits: dict[str, int]) -> dict:
    selected = {}
    evidence = {}
    for rank in (0, 1):
        path = ledger_dir / f"transaction_rank{rank}.jsonl"
        previous = "0" * 64
        pending = None
        count = 0
        limit = int(record_limits[path.name])
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            if count >= limit:
                break
            row = json.loads(line)
            declared = row.get("record_sha256")
            if row.get("previous_record_sha256") != previous or canonical_sha(row) != declared:
                raise ValueError(f"transaction marker chain failure rank{rank}:{line_no}")
            previous = declared
            if int(row["global_step"]) > through_round:
                raise ValueError("checkpoint marker prefix crosses target round")
            if row.get("schema_version") != "rwwpo-transaction-v2":
                raise ValueError("parent prefix is not RWWPO-2")
            identity = tuple(row[key] for key in (
                "attempt_id", "rank", "global_step", "epoch", "minibatch"
            ))
            if row["phase"] == "intent":
                if pending is not None:
                    raise ValueError("nested transaction intent in parent prefix")
                pending = identity
                selected.setdefault(identity, {})["intent"] = row
            elif row["phase"] == "complete" and pending == identity:
                selected.setdefault(identity, {})["complete"] = row
                pending = None
            else:
                raise ValueError("orphan transaction completion in parent prefix")
            count += 1
        if count != limit:
            raise ValueError("transaction marker shorter than checkpoint prefix")
        if pending is not None:
            raise ValueError("checkpoint prefix ends inside a transaction")
        evidence[path.name] = {
            "forensic_full_file_sha256": sha256_file(path),
            "selected_record_count": count, "prefix_tail_sha256": previous,
        }
    if not selected or any(set(value) != {"intent", "complete"} for value in selected.values()):
        raise ValueError("parent transaction prefix lacks intent/complete closure")
    return {"transaction_count": len(selected), "ledgers": evidence}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-attempt-root", required=True)
    parser.add_argument("--parent-output-root", required=True)
    parser.add_argument("--checkpoint-round", type=int, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--experiment-seed", type=int, required=True)
    parser.add_argument("--resolved-contract", required=True)
    parser.add_argument("--resolved-contract-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--producer-commit",
        help=("Exact commit recorded by the parent attempt. Defaults to the "
              "auditor checkout; set only for an independently reviewed "
              "auditor-only compatibility correction."),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:checkout")
    producer_commit = args.producer_commit or head
    if re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:producer commit")
    if args.cell not in "ABCDE" or args.checkpoint_round <= 0 \
            or args.checkpoint_round % 10 != 0:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:identity")
    raw_attempt_root = Path(args.parent_attempt_root)
    raw_output_root = Path(args.parent_output_root)
    raw_resolved = Path(args.resolved_contract)
    if raw_attempt_root.is_symlink() or raw_output_root.is_symlink() \
            or raw_resolved.is_symlink():
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:source symlink")
    attempt_root = raw_attempt_root.resolve()
    output_root = raw_output_root.resolve()
    ledger_dir = attempt_root / "actual_loss"
    execution_path = attempt_root / "execution.jsonl"
    checkpoint = output_root / f"global_step_{args.checkpoint_round}"
    if any(path.is_symlink() for path in (attempt_root, output_root, ledger_dir,
                                          execution_path, checkpoint)):
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:symlink")
    if not (checkpoint.joinpath("actor").is_dir() and checkpoint.joinpath("data.pt").is_file()):
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:checkpoint incomplete")
    resolved_path = raw_resolved.resolve()
    if not resolved_path.is_file() \
            or sha256_file(resolved_path) != args.resolved_contract_sha256:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:resolved contract bytes")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    declared_resolved = resolved.pop("report_sha256", None)
    recomputed_resolved = hashlib.sha256(json.dumps(
        resolved, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared_resolved != recomputed_resolved \
            or resolved.get("status") != "PASS" \
            or resolved.get("decision") != "RWWPO2_RESOLVED_CONTRACT_PASS" \
            or resolved.get("git_commit") != producer_commit:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:resolved contract receipt")
    try:
        events, inventory_event, execution_prefix_sha256 = execution_prefix_to_checkpoint(
            execution_path, checkpoint_round=args.checkpoint_round,
            expected_commit=producer_commit,
        )
    except ValueError as error:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:" + str(error)) from error
    if inventory_event.get("inventory") != checkpoint_inventory(checkpoint):
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:checkpoint inventory")
    if inventory_event.get("rwwpo2_resolved_contract_file_sha256") != \
            args.resolved_contract_sha256 \
            or inventory_event.get("rwwpo2_resolved_contract_report_sha256") != \
            declared_resolved \
            or inventory_event.get("rwwpo2_source_manifest_sha256") != \
            resolved.get("source_manifest_sha256"):
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:checkpoint contract binding")
    anchors = inventory_event.get("rwwpo_ledger_anchors", {})
    expected_names = {
        f"{kind}_rank{rank}.jsonl" for kind in ("actual_loss", "transaction")
        for rank in (0, 1)
    }
    if set(anchors) != expected_names:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:ledger anchors")
    record_limits = {
        name: int(anchor.get("record_count", -1)) for name, anchor in anchors.items()
    }
    if any(value < 1 for value in record_limits.values()):
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:ledger prefix record count")
    ledgers = [ledger_dir / f"actual_loss_rank{rank}.jsonl" for rank in (0, 1)]
    # A recovery prefix may legitimately contain only rejected K2 proposals.
    # Activity is a program-level R50 estimand, not a prerequisite for proving
    # that a checkpoint/ledger prefix is safe to replay.
    actual = audit(
        ledgers, require_method=False, through_round=args.checkpoint_round,
        record_limits=record_limits,
    )
    if actual["schema_versions"] != ["rwwpo-actual-loss-v3"] \
            or actual["audited_through_round"] != args.checkpoint_round:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:actual-loss prefix")
    receipt_rows = []
    for path in ledgers:
        selected = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if selected >= record_limits[path.name]:
                break
            selected += 1
            row = json.loads(line)
            if int(row["global_step"]) > args.checkpoint_round:
                raise SystemExit("RWWPO2_LINEAGE_NO_GO:actual prefix crosses checkpoint")
            receipt_rows.append(row)
    if {str(row.get("host_variant")) for row in receipt_rows} != {args.cell} \
            or {int(row.get("experiment_seed", -1)) for row in receipt_rows} != {
                args.experiment_seed
            }:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:cell/seed drift")
    terminal_clocks = set()
    for rank in (0, 1):
        rank_rows = sorted(
            (row for row in receipt_rows if int(row["rank"]) == rank),
            key=lambda row: int(row["proposal_clock"]),
        )
        if not rank_rows:
            raise SystemExit("RWWPO2_LINEAGE_NO_GO:rank prefix empty")
        terminal_clocks.add(int(rank_rows[-1]["accepted_optimizer_clock_after"]))
    if len(terminal_clocks) != 1:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:terminal accepted clock rank drift")
    marker = marker_prefix_audit(
        ledger_dir, through_round=args.checkpoint_round,
        record_limits=record_limits,
    )
    expected_transactions = (
        args.checkpoint_round - int(actual["audited_start_round"]) + 1
    ) * 2 * 2
    if marker["transaction_count"] != expected_transactions:
        # One intent/complete pair for each round x inner x rank.
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:marker K2 closure")

    start_round = int(inventory_event.get("rwwpo_tensor_inventory", {}).get(
        "start_round", actual["audited_start_round"]
    ))
    tensor_inventory = tensor_shard_inventory(
        ledger_dir, start_round=start_round, through_round=args.checkpoint_round,
        record_limits=record_limits,
    )
    if inventory_event.get("rwwpo_tensor_inventory") != tensor_inventory:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:tensor inventory")
    prefix_evidence = {}
    for name, anchor in sorted(anchors.items()):
        path = ledger_dir / name
        lines = [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]
        count = int(anchor.get("record_count", -1))
        if count < 1 or len(lines) < count:
            raise SystemExit("RWWPO2_LINEAGE_NO_GO:ledger prefix count")
        prefix = b"".join(lines[:count])
        prefix_sha = hashlib.sha256(prefix).hexdigest()
        tail = json.loads(lines[count - 1])["record_sha256"]
        if prefix_sha != anchor.get("prefix_sha256") or tail != anchor.get("tail_sha256"):
            raise SystemExit("RWWPO2_LINEAGE_NO_GO:ledger prefix binding")
        prefix_evidence[name] = {
            "record_count": count, "prefix_sha256": prefix_sha, "tail_sha256": tail,
        }
    seed_path = output_root / "rollout_seed_audit.jsonl"
    seed_anchor = inventory_event.get("rwwpo_rollout_seed_anchor") or {}
    seed_count = int(seed_anchor.get("record_count", -1))
    if seed_count < 1 or seed_path.is_symlink() or not seed_path.is_file():
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:rollout seed checkpoint anchor")
    seed_lines = [line for line in seed_path.read_bytes().splitlines(keepends=True)
                  if line.strip()]
    if len(seed_lines) < seed_count:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:rollout seed prefix short")
    seed_prefix = b"".join(seed_lines[:seed_count])
    if hashlib.sha256(seed_prefix).hexdigest() != seed_anchor.get("prefix_sha256") \
            or hashlib.sha256(seed_lines[seed_count - 1].rstrip(
                b"\r\n")).hexdigest() != seed_anchor.get("terminal_record_sha256"):
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:rollout seed prefix binding")
    sync = [row for row in events if row.get("record_type") == "weight_sync_summary"
            and row.get("sync_kind") == "post_actor_update"
            and int(row.get("global_step", -1)) == args.checkpoint_round]
    if len(sync) != 1 or sync[0].get("worker_ranks") != [0, 1] \
            or re.fullmatch(r"[0-9a-f]{64}", str(sync[0].get("sampled_tensor_digest", ""))) is None:
        raise SystemExit("RWWPO2_LINEAGE_NO_GO:weight sync")

    report = {
        "schema_version": "rwwpo2-lineage-parent-v1",
        "status": "PASS", "decision": "RWWPO2_LINEAGE_PARENT_PASS",
        # Keep the established resume-facing `git_commit` as the producer
        # identity, while separately binding the independent auditor code.
        "git_commit": producer_commit,
        "producer_git_commit": producer_commit,
        "auditor_git_commit": head,
        "auditor_source_sha256": sha256_file(Path(__file__).resolve()),
        "cell": args.cell,
        "experiment_seed": args.experiment_seed,
        "checkpoint_round": args.checkpoint_round,
        "parent_attempt_root": str(attempt_root),
        "parent_output_root": str(output_root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_inventory": inventory_event["inventory"],
        "checkpoint_inventory_event_sha256": inventory_event["record_sha256"],
        "resolved_contract_file_sha256": args.resolved_contract_sha256,
        "resolved_contract_report_sha256": declared_resolved,
        "source_manifest_sha256": resolved["source_manifest_sha256"],
        "execution_ledger_forensic_full_sha256": sha256_file(execution_path),
        "execution_prefix_sha256": execution_prefix_sha256,
        "execution_prefix_tail_sha256": events[-1]["record_sha256"],
        "ledger_prefix_evidence": prefix_evidence,
        "rollout_seed_prefix_evidence": {
            "record_count": seed_count,
            "prefix_sha256": hashlib.sha256(seed_prefix).hexdigest(),
            "terminal_record_sha256": seed_anchor["terminal_record_sha256"],
            "forensic_full_file_sha256": sha256_file(seed_path),
        },
        "tensor_inventory": tensor_inventory,
        "actual_loss_summary": actual,
        "accepted_optimizer_clock_at_checkpoint": next(iter(terminal_clocks)),
        "marker_prefix_summary": marker,
        "failed_suffix_imported": False,
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "checkpoint_round": args.checkpoint_round,
        "report_sha256": report["report_sha256"],
        "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
