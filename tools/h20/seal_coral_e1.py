#!/usr/bin/env python3
"""Seal only trainer-produced CORAL E1 receipts against real checkpoints."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.cosi import canonical_sha256, sha256_file
from recurrent.research.gate_a_execution import checkpoint_inventory, validate_jsonl_chain
from tools.h20.audit_coral_e1 import PREREGISTRATION, PROPOSAL_STEPS, validate_proposal


def inventory_digest(path):
    inventory = checkpoint_inventory(path)
    if not inventory:
        raise ValueError(f"CORAL_E1_NO_GO: empty checkpoint inventory {path}")
    return canonical_sha256(inventory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--sketch-oracle", required=True)
    parser.add_argument("--gate-a-ledger", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) is None:
        raise ValueError("CORAL_E1_NO_GO: exact expected commit")
    capture = Path(args.capture_root).resolve()
    training = Path(args.training_output).resolve()
    base = Path(args.base_model).resolve()
    ledger_path = Path(args.gate_a_ledger).resolve()
    oracle_path = Path(args.sketch_oracle).resolve()
    if any(not path.is_dir() for path in (capture, training, base)) \
            or not ledger_path.is_file() or not oracle_path.is_file():
        raise ValueError("CORAL_E1_NO_GO: producer/checkpoint/ledger paths")
    oracle = json.loads(oracle_path.read_text())
    from recurrent.research.coral_e1 import validate_fsdp_sketch_oracle_report
    validate_fsdp_sketch_oracle_report(oracle)
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
    failures = validate_jsonl_chain(ledger)
    if failures:
        raise ValueError(f"CORAL_E1_NO_GO: Gate A ledger chain {failures}")
    sync = {
        (int(row["global_step"]), str(row["sync_kind"])): row
        for row in ledger if row.get("record_type") == "weight_sync_summary"
    }
    proposals = []
    bindings = []
    base_digest = inventory_digest(base)
    for step in PROPOSAL_STEPS:
        path = capture / f"proposal_step_{step:02d}.json"
        if not path.is_file():
            raise ValueError("CORAL_E1_NO_GO: exact eight trainer receipts required")
        proposal = json.loads(path.read_text())
        validate_proposal(proposal, step, args.expected_commit)
        source_sync = sync.get((step - 1, "fresh_initial" if step == 1 else "post_actor_update"))
        proposal_sync = sync.get((step, "post_actor_update"))
        if source_sync is None or proposal_sync is None \
                or source_sync.get("sampled_tensor_digest") != proposal["source_weight_sample_digest"] \
                or proposal_sync.get("sampled_tensor_digest") != proposal["proposal_weight_sample_digest"]:
            raise ValueError("CORAL_E1_NO_GO: receipt/Gate A weight binding")
        source_digest = (base_digest if step == 1 else
                         inventory_digest(training / f"global_step_{step - 1}" / "actor"))
        proposal_digest = inventory_digest(training / f"global_step_{step}" / "actor")
        proposals.append(proposal)
        bindings.append({
            "global_step": step,
            "source_checkpoint_inventory_sha256": source_digest,
            "proposal_checkpoint_inventory_sha256": proposal_digest,
            "proposal_sha256": proposal["proposal_sha256"],
        })
    if len(list(capture.glob("proposal_step_*.json"))) != len(PROPOSAL_STEPS):
        raise ValueError("CORAL_E1_NO_GO: unexpected/adaptive proposal count")
    evidence = {
        "schema": "memagent.coral.e1.v3",
        "git_commit": args.expected_commit,
        "preregistration": PREREGISTRATION,
        "gate_a_ledger_sha256": sha256_file(ledger_path),
        "fsdp_sketch_oracle_report_sha256": oracle["report_sha256"],
        "fsdp_sketch_oracle_report": oracle,
        "proposal_bindings": bindings,
        "proposals": proposals,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "SEALED", "evidence_sha256": evidence["evidence_sha256"]}))


if __name__ == "__main__":
    main()
