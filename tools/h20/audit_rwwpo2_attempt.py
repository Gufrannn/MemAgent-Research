#!/usr/bin/env python3
"""Independent attempt-level audit for an RWWPO-2 R50 or R400 endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.actor_batch import stable_identity_int64
from recurrent.research.gate_a_execution import checkpoint_inventory, validate_jsonl_chain
from recurrent.research.rwwpo_ledger import tensor_shard_inventory
from recurrent.research.rwwpo_transaction import (
    off_behavior_exposed, stateless_proposal_lr,
)
from tools.h20.audit_rwwpo_actual_loss import (
    audit, canonical_sha, hydrate_authenticated_v3_receipt,
    validate_rwwpo2_rng_phase_digests,
)


CELL_CONTRACT = {
    "D": ("original_tokenwise", "none"),
    "C": ("original_tokenwise", "feasible_backtracking"),
    "E": ("per_write_joint", "feasible_backtracking"),
    "B": ("whole_prefix", "feasible_backtracking"),
    "A": ("whole_prefix", "hard_rollback"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_resolved(path: Path, *, expected_sha: str, commit: str) -> dict:
    if sha256_file(path) != expected_sha:
        raise ValueError("resolved contract file SHA mismatch")
    row = json.loads(path.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != "RWWPO2_RESOLVED_CONTRACT_PASS" \
            or row.get("git_commit") != commit:
        raise ValueError("resolved contract receipt invalid")
    return {**row, "report_sha256": declared}


def verified_report(path: Path, *, decision: str, commit: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{decision} receipt path invalid")
    row = json.loads(path.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != decision or row.get("git_commit") != commit:
        raise ValueError(f"{decision} receipt invalid")
    return {**row, "report_sha256": declared}


def validate_transaction_failure_boundary(
        ledger_dir: Path, *, through_round: int) -> list[dict]:
    """Reject malformed failures and any failure inside an audited prefix.

    A later failed suffix does not invalidate a checkpoint-authenticated earlier
    prefix.  It remains append-only evidence in the failed attempt root and is
    excluded from the resumed canonical lineage.
    """
    failure_ledgers = sorted(ledger_dir.glob("failure_rank*.jsonl"))
    evidence = []
    for path in failure_ledgers:
        match = re.fullmatch(r"failure_rank([01])\.jsonl", path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise ValueError("malformed transaction failure artifact")
        lines = [line for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not lines:
            raise ValueError("empty transaction failure artifact")
        previous = "0" * 64
        for line in lines:
            try:
                row = json.loads(line)
            except (TypeError, ValueError) as error:
                raise ValueError("malformed transaction failure JSON") from error
            if row.get("schema_version") != "rwwpo2-transaction-failure-v1" \
                    or row.get("status") != "NO_GO" \
                    or row.get("decision") != \
                    "RWWPO2_TRANSACTION_FAILURE_PRESERVED" \
                    or int(row.get("rank", -1)) != int(match.group(1)) \
                    or row.get("previous_record_sha256") != previous \
                    or row.get("record_sha256") != canonical_sha(row):
                raise ValueError("transaction failure chain/identity")
            global_step = int(row.get("global_step", -1))
            inner_id = int(row.get("inner_id", -1))
            if global_step < 1 or inner_id not in (1, 2) \
                    or int(row.get("proposal_clock", -1)) != \
                    2 * (global_step - 1) + inner_id:
                raise ValueError("transaction failure coordinate")
            allowed_identity = {
                ("RWWPO_PREFIX_TRUST_REGION_VIOLATION", "precondition"),
                ("RWWPO2_POST_COMMIT_FORWARD_CLOSURE_FAILURE",
                 "post_commit_verify"),
            }
            failure_magnitude = float(row.get(
                "current_reference_max_abs", float("nan")))
            if (row.get("reason"), row.get("phase")) not in allowed_identity \
                    or not math.isfinite(failure_magnitude) \
                    or failure_magnitude < 0 \
                    or not row.get("prefix_rows") \
                    or not row.get("prefix_stats") \
                    or not all(isinstance(value, str) and re.fullmatch(
                        r"[0-9a-f]{64}", value) for value in (
                            row.get("behavior_batch_digest"),
                            row.get("transaction_entry_buffer_digest"))):
                raise ValueError("transaction failure semantic evidence")
            if global_step <= int(through_round):
                raise ValueError("transaction failure inside audited prefix")
            previous = row["record_sha256"]
            evidence.append(row)
    return evidence


def validate_post_commit_forward_binding(rows, *, tau_logprob: float) -> None:
    """Bind every committed-state certificate to the frozen numeric oracle."""
    tau_logprob = float(tau_logprob)
    for row in rows:
        diagnostics = row.get("mechanism_diagnostics", {})
        if diagnostics.get("post_commit_forward_verified") is not True \
                or not math.isclose(float(diagnostics.get(
                    "post_commit_forward_verification_tolerance", -1.0)),
                    tau_logprob, rel_tol=0, abs_tol=0):
            raise ValueError("post-commit forward binding")


def validate_recovery_prune_evidence(
        events: list[dict], *, expected_checkpoint_rounds: list[int],
        output_root: Path) -> dict:
    """Require an authenticated intent/delete/complete pair for every prune."""
    expected = [int(value) for value in expected_checkpoint_rounds]
    retained = set(expected[-2:])
    pruned = expected[:-2]
    checkpoint_events = {
        int(row["global_step"]): row for row in events
        if row.get("record_type") == "checkpoint_inventory"
        and int(row.get("global_step", -1)) in expected
    }
    anchor_events = {
        int(row["global_step"]): row for row in events
        if row.get("record_type") == "rwwpo2_actor_anchor_inventory"
        and int(row.get("global_step", -1)) in expected
    }

    def indexed(record_type: str) -> dict[int, dict]:
        selected = [row for row in events if row.get("record_type") == record_type]
        by_round = {}
        for row in selected:
            round_id = int(row.get("pruned_round", -1))
            if round_id not in pruned or round_id in by_round:
                raise ValueError("unexpected or duplicate recovery prune evidence")
            by_round[round_id] = row
        return by_round

    intents = indexed("rwwpo2_recovery_prune_intent")
    completes = indexed("rwwpo2_recovery_pruned")
    if set(intents) != set(pruned) or set(completes) != set(pruned):
        raise ValueError("recovery prune intent/complete closure")

    for round_id in expected:
        root = output_root / f"global_step_{round_id}"
        if round_id in retained:
            if not root.is_dir():
                raise ValueError("retained recovery roots")
            continue
        intent = intents[round_id]
        complete = completes[round_id]
        checkpoint = checkpoint_events.get(round_id)
        anchor = anchor_events.get(round_id)
        prune_at = expected[expected.index(round_id) + 2]
        resolved_root = str(root.resolve())
        common = (
            checkpoint is not None,
            anchor is not None,
            int(intent.get("global_step", -1)) == prune_at,
            int(complete.get("global_step", -1)) == prune_at,
            intent.get("pruned_root") == resolved_root,
            complete.get("pruned_root") == resolved_root,
            intent.get("checkpoint_inventory_record_sha256") == (
                checkpoint or {}).get("record_sha256"),
            complete.get("checkpoint_inventory_record_sha256") == (
                checkpoint or {}).get("record_sha256"),
            intent.get("scientific_anchor_inventory_record_sha256") == (
                anchor or {}).get("record_sha256"),
            complete.get("scientific_anchor_inventory_record_sha256") == (
                anchor or {}).get("record_sha256"),
            intent.get("scientific_anchor_preserved") is True,
            complete.get("scientific_anchor_preserved") is True,
            complete.get("pruned_root_absent") is True,
            complete.get("prune_intent_record_sha256") == intent.get(
                "record_sha256"),
            int(complete.get("record_index", -1)) == int(intent.get(
                "record_index", -2)) + 1,
        )
        if root.exists() or not all(common):
            raise ValueError("recovery prune semantic closure")
    return {
        "retained_rounds": sorted(retained),
        "pruned_rounds": pruned,
        "two_phase_evidence": True,
    }


def execution_prefix_through_round(path: Path, *, target_round: int,
                                   expected_commit: str):
    """Authenticate only the completed execution prefix for `target_round`.

    A later failed attempt suffix may be truncated or malformed.  Once the
    target checkpoint has been observed, bytes from a later logical round are
    outside this segment and are deliberately not parsed.
    """
    events = []
    raw_prefix = bytearray()
    checkpoint_seen = False
    with path.open("rb") as stream:
        for physical_line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception as error:
                if checkpoint_seen:
                    break
                raise ValueError(
                    f"execution prefix malformed before target:{physical_line_no}"
                ) from error
            step = int(row.get("global_step", -1))
            if checkpoint_seen and step > target_round:
                break
            if step > target_round:
                raise ValueError("execution advanced beyond target before checkpoint")
            events.append(row)
            raw_prefix.extend(raw)
            if row.get("record_type") == "checkpoint_inventory" and step == target_round:
                checkpoint_seen = True
    inventories = [row for row in events if row.get("record_type") ==
                   "checkpoint_inventory" and int(row.get("global_step", -1)) ==
                   target_round]
    if not checkpoint_seen or len(inventories) != 1 \
            or validate_jsonl_chain(events) \
            or any(row.get("git_commit") != expected_commit for row in events):
        raise ValueError("execution target-prefix chain/identity")
    return events, inventories[0], hashlib.sha256(raw_prefix).hexdigest()


def marker_audit(ledger_dir: Path, actual_identities: dict, *, through_round: int,
                 record_limits: dict[str, int]) -> dict:
    completions = {}
    intents = {}
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
                raise ValueError(f"marker chain rank{rank}:{line_no}")
            previous = declared
            if int(row["global_step"]) > int(through_round):
                raise ValueError("marker checkpoint prefix crosses target round")
            identity = tuple(row[key] for key in (
                "attempt_id", "rank", "global_step", "epoch", "minibatch"
            ))
            actual = actual_identities.get(identity)
            if actual is None or row.get("schema_version") != "rwwpo-transaction-v2" \
                    or int(row.get("inner_id", 0)) != int(actual["inner_id"]) \
                    or int(row.get("proposal_clock", 0)) != int(actual["proposal_clock"]):
                raise ValueError("marker/actual coordinate drift")
            if row["phase"] == "intent":
                if pending is not None or identity in intents:
                    raise ValueError("nested/duplicate marker intent")
                pending = identity
                intents[identity] = row["model_digest"]
            elif row["phase"] == "complete" and pending == identity:
                if identity in completions:
                    raise ValueError("duplicate marker completion")
                pending = None
                completions[identity] = row["model_digest"]
            else:
                raise ValueError("orphan marker completion")
            count += 1
        if count != limit:
            raise ValueError("marker ledger shorter than checkpoint prefix")
        if pending is not None:
            raise ValueError("attempt ends in interrupted transaction")
        evidence[path.name] = {
            "forensic_full_file_sha256": sha256_file(path),
            "record_count": count, "prefix_tail_sha256": previous,
        }
    if set(intents) != set(actual_identities) or set(completions) != set(actual_identities):
        raise ValueError("marker/actual identity bijection")
    for identity, row in actual_identities.items():
        if intents[identity] != row["pre_digests"]["model"] \
                or completions[identity] != row["commit_digests"]["model"]:
            raise ValueError("marker model digest binding")
    return evidence


def projection_separation(sketches: dict) -> float:
    fields = ("sum", "alternating_projection", "sawtooth_projection")
    left = [float(sketches["B"][field]) for field in fields]
    right = [float(sketches["E"][field]) for field in fields]
    control = [float(sketches["C"][field]) for field in fields]
    numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    denominator = max(math.sqrt(sum(value ** 2 for value in control)), 1e-30)
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resolved-contract", required=True)
    parser.add_argument("--resolved-contract-sha256", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--cell", choices=sorted(CELL_CONTRACT), required=True)
    parser.add_argument("--experiment-seed", type=int, required=True)
    parser.add_argument("--target-round", type=int, required=True)
    parser.add_argument("--segment-producer-commit")
    parser.add_argument("--cross-commit-compatibility")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:checkout")
    segment_producer_commit = args.segment_producer_commit or head
    compatibility = None
    if segment_producer_commit != head:
        if re.fullmatch(r"[0-9a-f]{40}", segment_producer_commit) is None \
                or not args.cross_commit_compatibility:
            raise SystemExit(
                "RWWPO2_ATTEMPT_AUDIT_NO_GO:cross-commit segment arguments"
            )
        try:
            compatibility = verified_report(
                Path(args.cross_commit_compatibility).resolve(),
                decision="RWWPO2_CROSS_COMMIT_RESUME_COMPATIBILITY_PASS",
                commit=head,
            )
        except ValueError as error:
            raise SystemExit(
                "RWWPO2_ATTEMPT_AUDIT_NO_GO:" + str(error)
            ) from error
        if compatibility.get("producer_git_commit") != segment_producer_commit \
                or compatibility.get("consumer_git_commit") != head \
                or compatibility.get("producer_resolved_contract_file_sha256") != \
                args.resolved_contract_sha256 \
                or compatibility.get("algorithmic_source_or_contract_change") is not False:
            raise SystemExit(
                "RWWPO2_ATTEMPT_AUDIT_NO_GO:cross-commit segment compatibility"
            )
    elif args.cross_commit_compatibility is not None:
        raise SystemExit(
            "RWWPO2_ATTEMPT_AUDIT_NO_GO:unexpected cross-commit segment receipt"
        )
    if args.target_round <= 0 or args.target_round > 400 \
            or args.target_round % 10 != 0:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:target recovery round")
    raw_attempt_root = Path(args.attempt_root)
    raw_output_root = Path(args.output_root)
    raw_resolved = Path(args.resolved_contract)
    raw_preflight = Path(args.preflight)
    if any(path.is_symlink() for path in (
            raw_attempt_root, raw_output_root, raw_resolved, raw_preflight)) \
            or (args.cross_commit_compatibility is not None and
                Path(args.cross_commit_compatibility).is_symlink()):
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:source symlink")
    attempt_root = raw_attempt_root.resolve()
    output_root = raw_output_root.resolve()
    ledger_dir = attempt_root / "actual_loss"
    execution_path = attempt_root / "execution.jsonl"
    checkpoint = output_root / f"global_step_{args.target_round}"
    if any(path.is_symlink() for path in (attempt_root, output_root, ledger_dir,
                                          execution_path, checkpoint)):
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:symlink")
    if not (checkpoint.joinpath("actor").is_dir() and checkpoint.joinpath("data.pt").is_file()):
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:target checkpoint")
    try:
        preflight = verified_report(
            raw_preflight.resolve(), decision="RWWPO2_PREFLIGHT_PASS",
            commit=segment_producer_commit,
        )
        resolved_commit = (
            segment_producer_commit if compatibility is not None else
            preflight.get("cross_commit_producer_git_commit") or head
        )
        resolved = verified_resolved(
            raw_resolved.resolve(),
            expected_sha=args.resolved_contract_sha256,
            commit=resolved_commit,
        )
    except ValueError as error:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:" + str(error)) from error
    if raw_preflight.resolve() != attempt_root / "certificates/p0_preflight.json" \
            or preflight.get("cell") != args.cell \
            or int(preflight.get("experiment_seed", -1)) != args.experiment_seed \
            or int(preflight.get("target_round", -1)) < args.target_round \
            or preflight.get("resolved_contract_file_sha256") != \
                args.resolved_contract_sha256 \
            or preflight.get("resolved_contract_report_sha256") != \
                resolved["report_sha256"] \
            or preflight.get("source_manifest_sha256") != \
                resolved["source_manifest_sha256"] \
            or preflight.get("s128_consumed_by_training") is not False:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:preflight identity binding")
    preflight_target_round = int(preflight["target_round"])
    if preflight_target_round == 400:
        if not all(isinstance(preflight.get(name), str) and re.fullmatch(
                r"[0-9a-f]{64}", preflight[name]) for name in (
                    "r50_program_gate_report_sha256",
                    "r50_program_gate_file_sha256",
                    "confirmation_seal_report_sha256",
                    "confirmation_seal_file_sha256",
                )):
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:R400 preflight gate binding")
    elif preflight_target_round == 50 and any(preflight.get(name) is not None for name in (
            "r50_program_gate_report_sha256", "r50_program_gate_file_sha256",
            "confirmation_seal_report_sha256", "confirmation_seal_file_sha256",
    )):
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:R50 preflight gate drift")
    manifest = resolved["manifest"]
    objective, controller = CELL_CONTRACT[args.cell]
    if manifest["method_cells"][args.cell]["objective_variant"] != objective \
            or manifest["method_cells"][args.cell]["controller_variant"] != controller:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:cell contract")

    try:
        events, target_inventory_event, execution_prefix_sha256 = \
            execution_prefix_through_round(
                execution_path, target_round=args.target_round,
                expected_commit=segment_producer_commit,
            )
    except ValueError as error:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:" + str(error)) from error
    if target_inventory_event.get("inventory") != checkpoint_inventory(checkpoint):
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:checkpoint inventory")
    if target_inventory_event.get("rwwpo2_resolved_contract_file_sha256") != \
            args.resolved_contract_sha256 \
            or target_inventory_event.get(
                "rwwpo2_resolved_contract_report_sha256") != resolved["report_sha256"] \
            or target_inventory_event.get("rwwpo2_source_manifest_sha256") != \
            resolved["source_manifest_sha256"]:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:checkpoint contract binding")
    ledger_anchors = target_inventory_event.get("rwwpo_ledger_anchors", {})
    expected_ledger_names = {
        f"{kind}_rank{rank}.jsonl" for kind in ("actual_loss", "transaction")
        for rank in (0, 1)
    }
    try:
        validate_transaction_failure_boundary(
            ledger_dir, through_round=args.target_round)
    except ValueError as error:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:" + str(error)) from error
    if set(ledger_anchors) != expected_ledger_names:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:checkpoint ledger anchors")
    record_limits = {
        name: int(anchor.get("record_count", -1))
        for name, anchor in ledger_anchors.items()
    }
    if any(value < 1 for value in record_limits.values()):
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:checkpoint ledger counts")

    ledgers = [ledger_dir / f"actual_loss_rank{rank}.jsonl" for rank in (0, 1)]
    # Segment evidence must remain auditable even when every proposal in that
    # segment rejected. Nontrivial activity/exposure is enforced across the
    # complete R50 assignment by audit_rwwpo2_r50_program.py.
    actual_summary = audit(
        ledgers, require_method=False, through_round=args.target_round,
        record_limits=record_limits,
    )
    if actual_summary["schema_versions"] != ["rwwpo-actual-loss-v3"] \
            or actual_summary["objective_variants"] != [objective] \
            or actual_summary["controller_variants"] != [controller] \
            or actual_summary["audited_through_round"] != args.target_round:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:actual-loss identity")
    rows = []
    for ledger in ledgers:
        selected = 0
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if selected >= record_limits[ledger.name]:
                break
            selected += 1
            receipt = json.loads(line)
            if int(receipt["global_step"]) > args.target_round:
                raise SystemExit(
                    "RWWPO2_ATTEMPT_AUDIT_NO_GO:actual prefix crosses target"
                )
            row = hydrate_authenticated_v3_receipt(receipt, ledger)
            # Defense in depth: the actual-loss audit above already validates
            # this closure, but the formal attempt entry must visibly bind the
            # four phase digests to pre/commit state as well.
            validate_rwwpo2_rng_phase_digests(row)
            rows.append(row)
    start_round = min(int(row["global_step"]) for row in rows)
    if int(preflight.get("lineage_start_round", -1)) != start_round:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:preflight lineage start")
    if start_round == 1:
        if preflight.get("phase") != "fresh" \
                or preflight.get("resume_round") is not None \
                or preflight.get("lineage_parent_report_sha256") is not None:
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:fresh preflight lineage")
    elif preflight.get("phase") != "resume" \
            or int(preflight.get("resume_round", -1)) != start_round - 1 \
            or not isinstance(preflight.get("lineage_parent_report_sha256"), str):
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:resume preflight lineage")
    expected_rounds = list(range(start_round, args.target_round + 1))
    if sorted({int(row["global_step"]) for row in rows}) != expected_rounds:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:round continuity")
    if {str(row["host_variant"]) for row in rows} != {args.cell} \
            or {int(row["experiment_seed"]) for row in rows} != {args.experiment_seed}:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:host/seed")
    identities = {
        (row["attempt_id"], int(row["rank"]), int(row["global_step"]),
         int(row["epoch"]), int(row["minibatch"])): row for row in rows
    }
    marker_evidence = marker_audit(
        ledger_dir, identities, through_round=args.target_round,
        record_limits=record_limits,
    )

    thresholds = resolved["numeric_thresholds"]
    schedule = resolved["proposal_schedule"]
    gradient_sketch_chunk_elements = int(
        resolved["gradient_sketch_chunk_elements"])
    fsdp_parameter_commit_primitive = str(
        resolved["fsdp_parameter_commit_primitive"])
    fsdp_writeback_max_wall_seconds = float(
        resolved["fsdp_parameter_writeback_max_wall_seconds"])
    max_trial_forward_wall_seconds = float(
        resolved["max_trial_forward_wall_seconds_per_transaction"])
    behavior_tolerance = float(resolved["behavior_coefficient_tolerance"])
    maximum_loo = float(resolved["maximum_root_loo_feasibility_flip_fraction"])
    try:
        validate_post_commit_forward_binding(
            rows, tau_logprob=float(thresholds["tau_logprob"]))
    except ValueError as error:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:" + str(error)) from error
    round_groups = {}
    for row in rows:
        round_groups.setdefault(int(row["global_step"]), []).append(row)
        expected_lr = stateless_proposal_lr(
            base_lr=schedule["base_lr"], warmup_proposals=schedule["warmup_proposals"],
            total_proposals=schedule["total_proposals"],
            proposal_id=int(row["proposal_clock"]), kind=schedule["kind"],
        )
        diagnostics = row.get("mechanism_diagnostics", {})
        if int(diagnostics.get("gradient_sketch_chunk_elements", -1)) != \
                gradient_sketch_chunk_elements:
            raise SystemExit(
                "RWWPO2_ATTEMPT_AUDIT_NO_GO:gradient sketch chunk binding")
        if diagnostics.get("fsdp_parameter_commit_primitive") != \
                fsdp_parameter_commit_primitive:
            raise SystemExit(
                "RWWPO2_ATTEMPT_AUDIT_NO_GO:FSDP commit primitive binding")
        writeback_times = diagnostics.get(
            "fsdp_parameter_writeback_wall_seconds")
        if float(diagnostics.get(
                "fsdp_parameter_writeback_max_wall_seconds", -1)) != \
                fsdp_writeback_max_wall_seconds \
                or not isinstance(writeback_times, list) \
                or not 1 <= len(writeback_times) <= 7 \
                or any(not math.isfinite(float(value))
                       or not 0.0 <= float(value) <=
                       fsdp_writeback_max_wall_seconds
                       for value in writeback_times) \
                or float(diagnostics.get(
                    "max_trial_forward_wall_seconds", -1)) != \
                    max_trial_forward_wall_seconds \
                or not 0.0 <= float(row.get(
                    "trial_forward_wall_seconds", -1)) <= \
                    max_trial_forward_wall_seconds:
            raise SystemExit(
                "RWWPO2_ATTEMPT_AUDIT_NO_GO:FSDP/trial wall-time binding")
        if not math.isclose(float(diagnostics.get("proposal_lr", -1)), expected_lr,
                            rel_tol=0, abs_tol=0):
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:stateless LR")
        supported_loo = [
            stat for stat in row["post_prefix_stats"]
            if bool(stat.get("root_loo_supported"))
        ]
        if not supported_loo:
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:no supported root LOO turn")
        if max(float(stat["root_loo_feasibility_flip_fraction"])
               for stat in supported_loo) > maximum_loo:
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:root LOO stability")

    eligible = []
    exposed = []
    geometry_activated = []
    behavior_logprob_max = 0.0
    behavior_coefficient_max = 0.0
    behavior_parameter_projection_max = 0.0
    salvage_count = 0
    reject_count = 0
    round_diagnostics = []
    for round_id in expected_rounds:
        group = round_groups[round_id]
        if {(int(row["rank"]), int(row["inner_id"])) for row in group} != {
                (0, 1), (1, 1), (0, 2), (1, 2)
        }:
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:K2 round closure")
        for inner_id in (1, 2):
            paired = [row for row in group
                      if int(row["inner_id"]) == inner_id]
            writeback_evidence = {
                json.dumps(row["mechanism_diagnostics"][
                    "fsdp_parameter_writeback_wall_seconds"],
                    separators=(",", ":"))
                for row in paired
            }
            trial_wall_evidence = {
                float(row["trial_forward_wall_seconds"])
                for row in paired
            }
            if len(writeback_evidence) != 1 or len(trial_wall_evidence) != 1:
                raise SystemExit(
                    "RWWPO2_ATTEMPT_AUDIT_NO_GO:distributed wall-time drift")
        inner1 = [row for row in group if int(row["inner_id"]) == 1]
        inner2 = [row for row in group if int(row["inner_id"]) == 2]
        for row in inner1:
            max_delta = max(
                abs(float(current) - float(old))
                for current_row, old_row, mask_row in zip(
                    row["current_log_prob"], row["old_log_prob"], row["response_mask"]
                )
                for current, old, active in zip(current_row, old_row, mask_row) if active
            )
            behavior_logprob_max = max(behavior_logprob_max, max_delta)
            if max_delta > float(thresholds["tau_logprob"]):
                raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:inner1 behavior identity")
            shadow = row["mechanism_diagnostics"]["shadow_coefficients"]
            current_max = max(float(shadow[name]["max_abs"])
                              for name in ("C_vs_B", "C_vs_E", "B_vs_E"))
            behavior_coefficient_max = max(behavior_coefficient_max, current_max)
            if current_max > behavior_tolerance:
                raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:behavior shadow mismatch")
            sketches = row["mechanism_diagnostics"].get("shadow_parameter_gradient_sketches", {})
            if round_id <= 50 and not sketches:
                raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:missing R50 behavior shadow")
            if sketches:
                separation = projection_separation(sketches)
                behavior_parameter_projection_max = max(
                    behavior_parameter_projection_max, separation
                )
                if separation > float(thresholds["tau_gradient"]):
                    raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:behavior parameter shadow")
        exposure_records = {
            json.dumps(row["mechanism_diagnostics"].get("inner1_exposure"), sort_keys=True)
            for row in group
        }
        if len(exposure_records) != 1:
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:distributed exposure drift")
        exposure = json.loads(next(iter(exposure_records)))
        recomputed_exposure = off_behavior_exposed(
            relative_parameter_displacement=float(exposure["relative_parameter_displacement"]),
            writer_logprob_rms_value=float(exposure["writer_logprob_rms"]),
            tau_theta=float(thresholds["tau_theta"]),
            tau_logprob=float(thresholds["tau_logprob"]),
        )
        if bool(exposure["off_behavior_exposed"]) != recomputed_exposure:
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:exposure decision")
        inner2_sketches = {
            json.dumps(row["mechanism_diagnostics"].get(
                "shadow_parameter_gradient_sketches", {}), sort_keys=True)
            for row in inner2
        }
        if round_id <= 50 and (len(inner2_sketches) != 1 or inner2_sketches == {"{}"}):
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:missing R50 inner2 shadow")
        separation = None
        if inner2_sketches and inner2_sketches != {"{}"}:
            separation = projection_separation(json.loads(next(iter(inner2_sketches))))
        eligible.append(round_id)
        activated = False
        if recomputed_exposure:
            exposed.append(round_id)
            activated = bool(
                separation is not None and separation > float(thresholds["tau_gradient"])
            )
            geometry_activated.append(activated)
        transaction = inner2[0]
        alpha = float(transaction["alpha_committed"])
        salvage_count += int(0 < alpha < 1)
        reject_count += int(alpha == 0)
        round_diagnostics.append({
            "round_id": round_id,
            "cluster_unit": "rollout_round",
            "eligible": True,
            "exposed": bool(recomputed_exposure),
            "geometry_activated_given_exposed": bool(activated),
            "projection_separation": separation,
            "relative_parameter_displacement": float(
                exposure["relative_parameter_displacement"]),
            "writer_logprob_rms": float(exposure["writer_logprob_rms"]),
            "maximum_root_loo_feasibility_flip_fraction": max(
                float(stat["root_loo_feasibility_flip_fraction"])
                for row in group for stat in row["post_prefix_stats"]
                if bool(stat.get("root_loo_supported"))
            ),
            "root_loo_supported_turn_count": sum(
                bool(stat.get("root_loo_supported"))
                for row in group for stat in row["post_prefix_stats"]
            ),
            "root_loo_unsupported_turn_count": sum(
                not bool(stat.get("root_loo_supported"))
                for row in group for stat in row["post_prefix_stats"]
            ),
            "inner2_alpha_committed": alpha,
        })

    seed_path = output_root / "rollout_seed_audit.jsonl"
    if not seed_path.is_file():
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:rollout seed ledger")
    seed_anchor = target_inventory_event.get("rwwpo_rollout_seed_anchor") or {}
    seed_limit = int(seed_anchor.get("record_count", -1))
    if seed_limit < 1:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:rollout seed checkpoint anchor")
    seed_raw_lines = [line for line in seed_path.read_bytes().splitlines(keepends=True)
                      if line.strip()]
    if len(seed_raw_lines) < seed_limit:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:rollout seed prefix short")
    seed_prefix = b"".join(seed_raw_lines[:seed_limit])
    if hashlib.sha256(seed_prefix).hexdigest() != seed_anchor.get("prefix_sha256") \
            or hashlib.sha256(seed_raw_lines[seed_limit - 1].rstrip(
                b"\r\n")).hexdigest() != seed_anchor.get("terminal_record_sha256"):
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:rollout seed prefix binding")
    seed_rows = [json.loads(line) for line in seed_raw_lines[:seed_limit]]
    turn_identity = {}
    for row in seed_rows:
        if row.get("record_type") != "trajectory_turn_seed":
            continue
        key = (int(row["global_step"]), int(row["sample_index"]), int(row["turn"]))
        example_id = str(row["stable_example_id"])
        trajectory_id = str(row["trajectory_id"])
        expected_example = f"frozen_train_row:{int(row['dataset_index'])}"
        expected_trajectory = f"{expected_example}:seed:{int(row['trajectory_seed'])}"
        if example_id != expected_example or trajectory_id != expected_trajectory \
                or key in turn_identity:
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:stable identity")
        turn_identity[key] = (
            stable_identity_int64(example_id), stable_identity_int64(trajectory_id)
        )
    for row in rows:
        for sample_index, turn, example_hash, trajectory_hash in zip(
                row["sample_index"], row["trajectory_turn"],
                row["example_identity_hash"], row["trajectory_identity_hash"]):
            if turn_identity.get((int(row["global_step"]), int(sample_index), int(turn))) != (
                    int(example_hash), int(trajectory_hash)):
                raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:actual/rollout identity join")
    segment_seed_rows = [
        row for row in seed_rows
        if row.get("record_type") == "trajectory_turn_seed"
        and start_round <= int(row["global_step"]) <= args.target_round
    ]
    rollout_groups = {
        (int(row["global_step"]), str(row["stable_example_id"]))
        for row in segment_seed_rows
    }
    trajectories = {
        (int(row["global_step"]), str(row["trajectory_id"]))
        for row in segment_seed_rows
    }
    if len(rollout_groups) != len(expected_rounds) * 4 \
            or len(trajectories) != len(expected_rounds) * 8:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:rollout budget closure")
    behavior_rows = [row for row in rows if int(row["inner_id"]) == 1]
    writer_turns = set()
    writer_tokens = final_tokens = active_tokens = 0
    for row in behavior_rows:
        for trajectory, turn, writer_mask, answer_mask, response_mask in zip(
                row["trajectory_identity_hash"], row["trajectory_turn"],
                row["writer_mask"], row["answer_mask"], row["response_mask"]):
            writer_count = sum(bool(value) for value in writer_mask)
            writer_tokens += writer_count
            final_tokens += sum(bool(value) for value in answer_mask)
            active_tokens += sum(bool(value) for value in response_mask)
            if writer_count:
                writer_turns.add((int(row["global_step"]), int(trajectory), int(turn)))

    serialized_events = json.dumps(events, sort_keys=True).lower()
    if "hotpotqa_dev.parquet" in serialized_events or "s128_" in serialized_events:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:S128 reached training execution")
    sync = [row for row in events if row.get("record_type") == "weight_sync_summary"
            and row.get("sync_kind") == "post_actor_update"
            and start_round <= int(row.get("global_step", -1)) <= args.target_round]
    if [int(row["global_step"]) for row in sync] != expected_rounds \
            or any(row.get("worker_ranks") != [0, 1] for row in sync):
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:weight sync sequence")
    final_clock = {int(row["accepted_optimizer_clock_after"]) for row in rows
                   if int(row["global_step"]) == args.target_round and int(row["inner_id"]) == 2}
    if len(final_clock) != 1 \
            or sync[-1].get("rwwpo2_accepted_optimizer_clocks") != [next(iter(final_clock))]:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:accepted clock/weight sync")
    signals = [int(row["global_step"]) for row in events
               if row.get("record_type") == "execution_signal"
               and start_round <= int(row.get("global_step", -1)) <= args.target_round]
    if signals != expected_rounds:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:execution signal sequence")
    tensor_inventory = tensor_shard_inventory(
        ledger_dir, start_round=start_round, through_round=args.target_round,
        record_limits=record_limits,
    )
    if target_inventory_event.get("rwwpo_tensor_inventory") != tensor_inventory:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:tensor inventory")
    segment_checkpoint_events = [
        row for row in events if row.get("record_type") == "checkpoint_inventory"
        and start_round <= int(row.get("global_step", -1)) <= args.target_round
    ]
    expected_checkpoint_rounds = [
        value for value in expected_rounds if value % 10 == 0
    ]
    if [int(row["global_step"]) for row in segment_checkpoint_events] != \
            expected_checkpoint_rounds:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:recovery checkpoint sequence")
    if compatibility is not None:
        # The producer prefix predates the consumer's two-phase prune
        # evidence contract.  Its target checkpoint and all scientific actor
        # anchors are authenticated independently below; deletion of an older
        # recovery-only root is not imported into the resumed lineage.
        retained = expected_checkpoint_rounds[-2:]
        if any(not (output_root / f"global_step_{value}").is_dir()
               for value in retained) \
                or any(row.get("record_type") ==
                       "rwwpo2_recovery_prune_intent" for row in events):
            raise SystemExit(
                "RWWPO2_ATTEMPT_AUDIT_NO_GO:producer recovery boundary"
            )
        recovery_prune_summary = {
            "mode": "producer_prefix_before_consumer_two_phase_contract",
            "retained_rounds": retained,
            "older_recovery_roots_not_imported":
                expected_checkpoint_rounds[:-2],
            "two_phase_evidence": False,
            "cross_commit_compatibility_report_sha256":
                compatibility["report_sha256"],
        }
    else:
        try:
            recovery_prune_summary = validate_recovery_prune_evidence(
                events, expected_checkpoint_rounds=expected_checkpoint_rounds,
                output_root=output_root,
            )
        except ValueError as error:
            raise SystemExit(
                "RWWPO2_ATTEMPT_AUDIT_NO_GO:" + str(error)
            ) from error
    anchor_events = [row for row in events if row.get("record_type") ==
                     "rwwpo2_actor_anchor_inventory" and start_round <= int(
                         row.get("global_step", -1)) <= args.target_round]
    expected_anchor_rounds = sorted(
        int(value) for value in manifest["checkpointing"]["scientific_actor_anchors"]
        if start_round <= int(value) <= args.target_round
    )
    if sorted(int(row["global_step"]) for row in anchor_events) != expected_anchor_rounds:
        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:scientific anchor sequence")
    anchor_record_sha256 = {}
    checkpoint_event_by_round = {
        int(row["global_step"]): row for row in events
        if row.get("record_type") == "checkpoint_inventory"
    }
    for anchor_event in anchor_events:
        anchor_round = int(anchor_event["global_step"])
        anchor_root = output_root / "scientific_anchors" / f"round_{anchor_round}"
        if anchor_event.get("inventory") != checkpoint_inventory(anchor_root):
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:scientific anchor inventory")
        if anchor_round % 10 == 0:
            source_event = checkpoint_event_by_round.get(anchor_round)
            if source_event is None or anchor_event.get("storage_kind") != \
                    "hardlink_from_recovery_model_shards" \
                    or anchor_event.get(
                        "source_recovery_checkpoint_inventory_record_sha256"
                    ) != source_event["record_sha256"]:
                raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:anchor/recovery binding")
            # Only the newest two recovery roots are retained. If this anchor's
            # source still exists, independently verify shared inodes; otherwise
            # its earlier authenticated source event and current anchor bytes
            # remain the immutable binding.
            source_root = output_root / f"global_step_{anchor_round}" / "actor"
            if source_root.exists():
                for rank in (0, 1):
                    name = f"model_world_size_2_rank_{rank}.pt"
                    source = source_root / name
                    anchor = anchor_root / "actor" / name
                    if source.is_symlink() or anchor.is_symlink() \
                            or not source.is_file() or not anchor.is_file() \
                            or source.stat().st_ino != anchor.stat().st_ino:
                        raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:anchor hardlink")
        elif anchor_event.get("storage_kind") != "direct_actor_only_checkpoint" \
                or sorted(int(row["rank"]) for row in anchor_event.get(
                    "worker_acknowledgements", [])) != [0, 1]:
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:direct anchor rank closure")
        anchor_record_sha256[str(anchor_round)] = anchor_event["record_sha256"]
    resume_lineage = None
    if start_round > 1:
        resume = [row for row in events if row.get("record_type") == "resume_load"]
        if len(resume) != 1 or not resume[0].get("rwwpo2_lineage_parent") \
                or resume[0]["rwwpo2_lineage_parent"].get("failed_suffix_imported") is not False:
            raise SystemExit("RWWPO2_ATTEMPT_AUDIT_NO_GO:resume lineage execution binding")
        resume_lineage = resume[0]["rwwpo2_lineage_parent"]
        compatibility_sha = preflight.get(
            "cross_commit_compatibility_report_sha256"
        )
        execution_compatibility = resume_lineage.get(
            "cross_commit_compatibility"
        )
        if compatibility_sha is None:
            if execution_compatibility is not None \
                    or resume_lineage.get("producer_git_commit") != head \
                    or resume_lineage.get("consumer_git_commit") != head:
                raise SystemExit(
                    "RWWPO2_ATTEMPT_AUDIT_NO_GO:same-commit resume binding"
                )
        elif not isinstance(execution_compatibility, dict) \
                or execution_compatibility.get("report_sha256") != compatibility_sha \
                or execution_compatibility.get("producer_git_commit") != \
                preflight.get("cross_commit_producer_git_commit") \
                or execution_compatibility.get("consumer_git_commit") != head \
                or resume_lineage.get("producer_git_commit") != \
                preflight.get("lineage_parent_git_commit") \
                or resume_lineage.get("consumer_git_commit") != head:
            raise SystemExit(
                "RWWPO2_ATTEMPT_AUDIT_NO_GO:cross-commit resume binding"
            )

    report = {
        "schema_version": "rwwpo2-attempt-audit-v1",
        "status": "PASS",
        "decision": f"RWWPO2_R{args.target_round}_ATTEMPT_AUDIT_PASS",
        "git_commit": head, "program_version": "rwwpo2-k2",
        "segment_producer_git_commit": segment_producer_commit,
        "segment_cross_commit_compatibility_report_sha256": (
            None if compatibility is None else compatibility["report_sha256"]
        ),
        "cell": args.cell, "objective_variant": objective,
        "controller_variant": controller, "experiment_seed": args.experiment_seed,
        "attempt_root": str(attempt_root), "output_root": str(output_root),
        "resolved_contract_path": str(raw_resolved.resolve()),
        "resolved_contract_file_sha256": args.resolved_contract_sha256,
        "resolved_contract_report_sha256": resolved["report_sha256"],
        "source_manifest_sha256": resolved["source_manifest_sha256"],
        "preflight_path": str(raw_preflight.resolve()),
        "preflight_file_sha256": sha256_file(raw_preflight.resolve()),
        "preflight_report_sha256": preflight["report_sha256"],
        "preflight_phase": preflight["phase"],
        "preflight_target_round": preflight_target_round,
        "preflight_lineage_parent_report_sha256": preflight[
            "lineage_parent_report_sha256"
        ],
        "preflight_lineage_parent_git_commit": preflight.get(
            "lineage_parent_git_commit"
        ),
        "cross_commit_compatibility_report_sha256": preflight.get(
            "cross_commit_compatibility_report_sha256"
        ),
        "cross_commit_producer_git_commit": preflight.get(
            "cross_commit_producer_git_commit"
        ),
        "r50_program_gate_report_sha256": preflight[
            "r50_program_gate_report_sha256"
        ],
        "r50_program_gate_file_sha256": preflight[
            "r50_program_gate_file_sha256"
        ],
        "confirmation_seal_report_sha256": preflight[
            "confirmation_seal_report_sha256"
        ],
        "confirmation_seal_file_sha256": preflight[
            "confirmation_seal_file_sha256"
        ],
        "start_round": start_round, "target_round": args.target_round,
        "lineage_parent": resume_lineage,
        "behavior_batch_count": len(eligible),
        "actor_transaction_count": len(eligible) * 2,
        "distributed_transaction_record_count": len(rows),
        "budget_ledger": {
            "rollout_groups": len(rollout_groups),
            "trajectories": len(trajectories),
            "writer_turns": len(writer_turns),
            "writer_tokens_once_per_behavior_batch": writer_tokens,
            "final_answer_tokens_once_per_behavior_batch": final_tokens,
            "active_response_tokens_once_per_behavior_batch": active_tokens,
            "ppo_epoch_indices": sorted({int(row["epoch"]) for row in rows}),
            "optimizer_minibatch_indices": sorted({int(row["minibatch"]) for row in rows}),
            "actor_optimizer_proposals": len(eligible) * 2,
            "critic_optimizer_updates": 0,
            "auxiliary_fit_updates": 0,
        },
        "actual_loss_summary": actual_summary,
        "mechanism": {
            "eligible_round_count": len(eligible),
            "exposed_round_count": len(exposed),
            "exposure_rate": len(exposed) / len(eligible),
            "geometry_activation_count_given_exposed": sum(geometry_activated),
            "geometry_activation_rate_given_exposed": (
                sum(geometry_activated) / len(geometry_activated)
                if geometry_activated else None
            ),
            "behavior_old_current_logprob_max_abs_delta": behavior_logprob_max,
            "behavior_coefficient_max_abs_delta": behavior_coefficient_max,
            "behavior_parameter_projection_max_relative_delta":
                behavior_parameter_projection_max,
            "inner2_salvaged_partial_commit_count": salvage_count,
            "inner2_reject_count": reject_count,
            "round_diagnostics": round_diagnostics,
            "uncertainty_unit": "rollout_round; prompt-root LOO within round",
        },
        "thresholds": thresholds,
        "maximum_root_loo_feasibility_flip_fraction": maximum_loo,
        "marker_evidence": marker_evidence,
        "recovery_prune_summary": recovery_prune_summary,
        "tensor_inventory": tensor_inventory,
        "execution_ledger_forensic_full_sha256": sha256_file(execution_path),
        "execution_prefix_sha256": execution_prefix_sha256,
        "execution_prefix_tail_sha256": events[-1]["record_sha256"],
        "rollout_seed_audit_forensic_full_sha256": sha256_file(seed_path),
        "rollout_seed_prefix_sha256": hashlib.sha256(seed_prefix).hexdigest(),
        "checkpoint_inventory_record_sha256": target_inventory_event["record_sha256"],
        "target_checkpoint_inventory": target_inventory_event["inventory"],
        "target_checkpoint_inventory_sha256": hashlib.sha256(json.dumps(
            target_inventory_event["inventory"], sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
        "scientific_anchor_inventory_record_sha256": anchor_record_sha256,
        "s128_consumed": False,
        "performance_evaluated": False,
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "cell": args.cell, "seed": args.experiment_seed,
        "mechanism": report["mechanism"], "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
