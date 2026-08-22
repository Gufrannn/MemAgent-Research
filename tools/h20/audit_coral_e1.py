#!/usr/bin/env python3
"""Audit trainer-produced, single-update CORAL occupancy-response evidence."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.coral_e1 import (
    SKETCH_BASIS_SHA256, validate_dataproto_clone_oracle_report,
    validate_fsdp_sketch_oracle_report,
)
from recurrent.research.cosi import canonical_sha256, require_sha256

PROPOSAL_STEPS = tuple(range(1, 16, 2))
PREREGISTRATION = {
    "proposal_steps": list(PROPOSAL_STEPS),
    "roots_per_proposal": 4,
    "total_root_clusters": 32,
    "writer_replicas": 2,
    "terminal_action_policy": "both_branches_freshly_sampled_at_fixed_proposal_weights",
    "terminal_request_seed_coupling": "common_cached_vs_refreshed",
    "branch_reward_and_advantage": "independently_recomputed",
    "gradient_sketch_basis_sha256": SKETCH_BASIS_SHA256,
    "symmetric_relative_response_range": [0.0, 2.0],
    "min_global_mean_response": 0.15,
    "inference_unit": "writer_proposal_mean_over_four_never_reused_roots",
    "proposal_clusters": 8,
    "min_proposal_cluster_lcb": 0.05,
    "proposal_cluster_lcb_critical_value": 2.365,
    "min_length_matched_roots": 8,
    "max_length_delta_tokens": 2,
    "min_length_matched_mean_response": 0.12,
    "min_nonconflicting_large_response_fraction": 0.25,
    "min_passing_proposals": 4,
    "max_duplicate_relative_response": 1e-6,
    "diagnostic_only_no_method_warmstart": True,
}

PROPOSAL_FIELDS = {
    "schema", "producer", "git_commit", "global_step",
    "source_weight_sample_digest", "proposal_weight_sample_digest",
    "gradient_sketch_basis_sha256", "root_inventory_sha256", "records",
    "proposal_sha256",
}
RECORD_FIELDS = {
    "root_id", "dataset_index", "writer_replicas", "common_trajectory_seeds",
    "common_terminal_request_seeds",
    "cached_memory_token_ids_sha256", "refreshed_memory_token_ids_sha256",
    "cached_prompt_token_ids_sha256", "refreshed_prompt_token_ids_sha256",
    "cached_terminal_answer_token_ids_sha256",
    "refreshed_terminal_answer_token_ids_sha256",
    "cached_reward_sha256", "refreshed_reward_sha256",
    "cached_advantage_sha256", "refreshed_advantage_sha256",
    "terminal_action_policy",
    "cached_memory_token_count", "refreshed_memory_token_count",
    "cached_gradient_sha256", "refreshed_gradient_sha256",
    "cached_gradient_norm", "refreshed_gradient_norm",
    "symmetric_relative_response", "duplicate_control_response_norm",
    "same_batch_writer_answer_cosine", "tensor_source",
}


def finite(value, field, low=-math.inf, high=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"CORAL_E1_NO_GO: {field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f"CORAL_E1_NO_GO: {field} outside range")
    return value


def validate_proposal(value, expected_step, expected_commit):
    if not isinstance(value, dict) or set(value) != PROPOSAL_FIELDS \
            or value.get("schema") != "memagent.coral.e1-proposal.v3" \
            or value.get("producer") != "ray_ppo_trainer_actual_loss_backward":
        raise ValueError("CORAL_E1_NO_GO: proposal producer/schema drift")
    unsigned = {key: item for key, item in value.items() if key != "proposal_sha256"}
    if value["proposal_sha256"] != canonical_sha256(unsigned):
        raise ValueError("CORAL_E1_NO_GO: proposal authentication")
    if value["git_commit"] != expected_commit or value["global_step"] != expected_step:
        raise ValueError("CORAL_E1_NO_GO: proposal commit/step binding")
    if value["gradient_sketch_basis_sha256"] != SKETCH_BASIS_SHA256:
        raise ValueError("CORAL_E1_NO_GO: post-selected gradient basis")
    for field in (
        "source_weight_sample_digest", "proposal_weight_sample_digest",
        "root_inventory_sha256",
    ):
        require_sha256(value[field], field)
    if value["source_weight_sample_digest"] == value["proposal_weight_sample_digest"]:
        raise ValueError("CORAL_E1_NO_GO: writer proposal inactive")
    records = value["records"]
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("CORAL_E1_NO_GO: exact b4 root records required")
    roots = []
    normalized = []
    for row in records:
        if not isinstance(row, dict) or set(row) != RECORD_FIELDS:
            raise ValueError("CORAL_E1_NO_GO: actual-loss record fields drifted")
        root = row["root_id"]
        if not isinstance(root, str) or not root or root in roots \
                or row["writer_replicas"] != 2:
            raise ValueError("CORAL_E1_NO_GO: root/replica identity")
        if isinstance(row["dataset_index"], bool) \
                or not isinstance(row["dataset_index"], int) \
                or row["dataset_index"] < 0:
            raise ValueError("CORAL_E1_NO_GO: dataset root identity")
        seeds = row["common_trajectory_seeds"]
        if not isinstance(seeds, list) or len(seeds) != 2 \
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds) \
                or seeds != sorted(set(seeds)):
            raise ValueError("CORAL_E1_NO_GO: common seed coupling")
        terminal_seeds = row["common_terminal_request_seeds"]
        if not isinstance(terminal_seeds, list) or len(terminal_seeds) != 2 \
                or any(isinstance(seed, bool) or not isinstance(seed, int)
                       for seed in terminal_seeds) \
                or terminal_seeds != sorted(set(terminal_seeds)):
            raise ValueError("CORAL_E1_NO_GO: common terminal request seeds")
        for field in (
            "cached_memory_token_ids_sha256", "refreshed_memory_token_ids_sha256",
            "cached_prompt_token_ids_sha256", "refreshed_prompt_token_ids_sha256",
            "cached_terminal_answer_token_ids_sha256",
            "refreshed_terminal_answer_token_ids_sha256",
            "cached_reward_sha256", "refreshed_reward_sha256",
            "cached_advantage_sha256", "refreshed_advantage_sha256",
            "cached_gradient_sha256", "refreshed_gradient_sha256",
        ):
            require_sha256(row[field], field)
        if row["terminal_action_policy"] \
                != "both_branches_freshly_sampled_at_fixed_proposal_weights":
            raise ValueError("CORAL_E1_NO_GO: terminal action sampling policy")
        if row["tensor_source"] != "actual_terminal_answer_loss_backward":
            raise ValueError("CORAL_E1_NO_GO: synthetic/self-reported tensor source")
        cached = finite(row["cached_gradient_norm"], "cached_gradient_norm", 1e-12)
        refreshed = finite(row["refreshed_gradient_norm"], "refreshed_gradient_norm", 1e-12)
        response = finite(row["symmetric_relative_response"],
                          "symmetric_relative_response", 0, 2)
        duplicate = finite(row["duplicate_control_response_norm"],
                           "duplicate_control_response_norm", 0)
        duplicate_relative = 2 * duplicate / (cached + refreshed)
        if duplicate_relative > PREREGISTRATION["max_duplicate_relative_response"]:
            raise ValueError("CORAL_E1_NO_GO: duplicate numeric noise exceeds aperture")
        cosine = finite(row["same_batch_writer_answer_cosine"],
                        "same_batch_writer_answer_cosine", -1, 1)
        token_counts = (row["cached_memory_token_count"], row["refreshed_memory_token_count"])
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1
               for item in token_counts):
            raise ValueError("CORAL_E1_NO_GO: materialized memory token counts")
        roots.append(root)
        normalized.append({
            "root_id": root,
            "dataset_index": row["dataset_index"],
            "response": response,
            "length_delta": abs(token_counts[1] - token_counts[0]),
            "same_batch_cosine": cosine,
            "memory_changed": row["cached_memory_token_ids_sha256"]
                              != row["refreshed_memory_token_ids_sha256"],
        })
    if value["root_inventory_sha256"] != canonical_sha256(sorted(roots)):
        raise ValueError("CORAL_E1_NO_GO: root inventory authentication")
    proposal_mean = math.fsum(row["response"] for row in normalized) / len(normalized)
    return (
        normalized,
        proposal_mean,
        proposal_mean >= PREREGISTRATION["min_global_mean_response"],
    )


def audit_evidence(evidence):
    fields = {
        "schema", "git_commit", "preregistration", "gate_a_ledger_sha256",
        "dataproto_clone_oracle_report_sha256",
        "dataproto_clone_oracle_report",
        "fsdp_sketch_oracle_report_sha256",
        "fsdp_sketch_oracle_report",
        "proposal_bindings", "proposals", "evidence_sha256",
    }
    if not isinstance(evidence, dict) or set(evidence) != fields \
            or evidence.get("schema") != "memagent.coral.e1.v4":
        raise ValueError("CORAL_E1_NO_GO: evidence schema")
    unsigned = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    if evidence["evidence_sha256"] != canonical_sha256(unsigned):
        raise ValueError("CORAL_E1_NO_GO: evidence authentication")
    commit = evidence["git_commit"]
    if re.fullmatch(r"[0-9a-f]{40}", str(commit)) is None \
            or evidence["preregistration"] != PREREGISTRATION:
        raise ValueError("CORAL_E1_NO_GO: commit/preregistration drift")
    require_sha256(evidence["gate_a_ledger_sha256"], "gate_a_ledger_sha256")
    require_sha256(
        evidence["dataproto_clone_oracle_report_sha256"],
        "dataproto_clone_oracle_report_sha256",
    )
    validate_dataproto_clone_oracle_report(
        evidence["dataproto_clone_oracle_report"]
    )
    if evidence["dataproto_clone_oracle_report_sha256"] \
            != evidence["dataproto_clone_oracle_report"]["report_sha256"]:
        raise ValueError("CORAL_E1_NO_GO: embedded DataProto clone oracle binding")
    require_sha256(
        evidence["fsdp_sketch_oracle_report_sha256"],
        "fsdp_sketch_oracle_report_sha256",
    )
    validate_fsdp_sketch_oracle_report(evidence["fsdp_sketch_oracle_report"])
    if evidence["fsdp_sketch_oracle_report_sha256"] \
            != evidence["fsdp_sketch_oracle_report"]["report_sha256"]:
        raise ValueError("CORAL_E1_NO_GO: embedded FSDP oracle binding")
    proposals = evidence["proposals"]
    bindings = evidence["proposal_bindings"]
    if not isinstance(proposals, list) or len(proposals) != 8 \
            or not isinstance(bindings, list) or len(bindings) != 8:
        raise ValueError("CORAL_E1_NO_GO: exact eight single-update proposals")
    all_rows = []
    proposal_means = []
    proposal_passes = 0
    seen_roots = set()
    seen_dataset_indices = set()
    for step, proposal, binding in zip(PROPOSAL_STEPS, proposals, bindings):
        rows, proposal_mean, proposal_pass = validate_proposal(proposal, step, commit)
        expected_binding_fields = {
            "global_step", "source_checkpoint_inventory_sha256",
            "proposal_checkpoint_inventory_sha256", "proposal_sha256",
        }
        if not isinstance(binding, dict) or set(binding) != expected_binding_fields \
                or binding["global_step"] != step \
                or binding["proposal_sha256"] != proposal["proposal_sha256"]:
            raise ValueError("CORAL_E1_NO_GO: checkpoint/proposal binding")
        require_sha256(binding["source_checkpoint_inventory_sha256"], "source inventory")
        require_sha256(binding["proposal_checkpoint_inventory_sha256"], "proposal inventory")
        if binding["source_checkpoint_inventory_sha256"] \
                == binding["proposal_checkpoint_inventory_sha256"]:
            raise ValueError("CORAL_E1_NO_GO: source/proposal checkpoint alias")
        roots = {row["root_id"] for row in rows}
        dataset_indices = {row["dataset_index"] for row in rows}
        if roots & seen_roots or dataset_indices & seen_dataset_indices \
                or len(dataset_indices) != 4:
            raise ValueError("CORAL_E1_NO_GO: adaptive root reuse across proposals")
        seen_roots.update(roots)
        seen_dataset_indices.update(dataset_indices)
        all_rows.extend(rows)
        proposal_means.append(proposal_mean)
        proposal_passes += int(proposal_pass)
    if len(all_rows) != 32:
        raise ValueError("CORAL_E1_NO_GO: exact 32 root clusters")
    responses = [row["response"] for row in all_rows]
    global_mean = math.fsum(responses) / len(responses)
    # Four roots share one writer proposal and therefore one proposal-level
    # shock.  The eight proposal means, not the 32 nested roots, are the
    # independent top-level units for this preregistered descriptive LCB.
    proposal_cluster_mean = math.fsum(proposal_means) / len(proposal_means)
    standard_error = statistics.stdev(proposal_means) / math.sqrt(len(proposal_means))
    lcb = (
        proposal_cluster_mean
        - PREREGISTRATION["proposal_cluster_lcb_critical_value"] * standard_error
    )
    matched = [row for row in all_rows if row["length_delta"] <= 2]
    matched_mean = (math.fsum(row["response"] for row in matched) / len(matched)
                    if matched else None)
    beyond = math.fsum(
        row["response"] >= PREREGISTRATION["min_global_mean_response"]
        and row["same_batch_cosine"] >= 0 for row in all_rows
    ) / len(all_rows)
    changed = math.fsum(row["memory_changed"] for row in all_rows) / len(all_rows)
    passed = (
        global_mean >= PREREGISTRATION["min_global_mean_response"]
        and lcb >= PREREGISTRATION["min_proposal_cluster_lcb"]
        and len(matched) >= PREREGISTRATION["min_length_matched_roots"]
        and matched_mean is not None
        and matched_mean >= PREREGISTRATION["min_length_matched_mean_response"]
        and beyond >= PREREGISTRATION["min_nonconflicting_large_response_fraction"]
        and proposal_passes >= PREREGISTRATION["min_passing_proposals"]
        and changed > 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "decision": "CORAL_E1_PASS" if passed else "CORAL_E1_NO_GO",
        "global_mean_symmetric_relative_response": global_mean,
        "proposal_cluster_mean_symmetric_relative_response": proposal_cluster_mean,
        "proposal_cluster_lcb": lcb,
        "proposal_cluster_standard_error": standard_error,
        "proposal_means": proposal_means,
        "length_matched_roots": len(matched),
        "length_matched_mean_response": matched_mean,
        "nonconflicting_large_response_fraction": beyond,
        "memory_changed_fraction": changed,
        "passing_proposals": proposal_passes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    summary = audit_evidence(evidence)
    report = {
        "schema": "memagent.coral.e1-report.v4",
        **summary,
        "estimand": "fixed-proposal terminal-answer gradient response to same-root materialized-memory refresh",
        "evidence_sha256": evidence["evidence_sha256"],
    }
    report["report_sha256"] = canonical_sha256(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
