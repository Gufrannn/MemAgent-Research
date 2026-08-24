#!/usr/bin/env python3
"""Independent read-only audit of the two-rank RWWPO-2 numeric oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MULTIPLIER = 16.0
GRADIENT_SKETCH_CHUNK_ELEMENTS = 8_388_608
FSDP_PARAMETER_COMMIT_PRIMITIVE = \
    "fsdp_unitwise_allgather_summon_writeback_v1"
STREAMED_ORACLE_MICROBATCHES = 7
STREAMED_ORACLE_SEQUENCE_LENGTH = 8191
TRANSACTION_CLOSURE_SEQUENCE_LENGTH = 8191
TRANSACTION_CLOSURE_ACTIVE_TOKENS = 1024
TRANSACTION_WRITEBACK_MAX_WALL_SECONDS = 120.0
STREAMED_REPLAY_CALIBRATION = {
    "microbatches": STREAMED_ORACLE_MICROBATCHES,
    "sequence_length": STREAMED_ORACLE_SEQUENCE_LENGTH,
    "active_response_tokens": 1024,
    "synthetic_label_free": True,
    "gradient_checkpointing": True,
    "gradient_checkpointing_use_reentrant": False,
    "remove_padding_flash_attention_patch": True,
    "fsdp_auto_wrap_policy": "default_transformer_no_split_modules",
    "fsdp_sharding_strategy": "FULL_SHARD",
    "fsdp_use_orig_params": False,
    "fsdp_sync_module_states": True,
    "fsdp_forward_prefetch": False,
    "model_load_dtype": "float32",
    "fsdp_sharded_parameter_dtype": "float32",
    "fsdp_param_dtype": "bfloat16",
    "fsdp_reduce_dtype": "float32",
    "fsdp_buffer_dtype": "float32",
    "cuda_autocast_dtype": "bfloat16",
    "selective_logprob_kernel":
        "verl.utils.torch_functional.logprobs_from_logits",
    "transaction_closure_probe": "unitwise_fp32_shard_to_bf16_forward_v1",
    "transaction_optimizer_probe": "adamw_fp32_shard_step_v1",
    "transaction_optimizer_lr": 1e-6,
    "transaction_optimizer_betas": [0.9, 0.999],
    "transaction_optimizer_weight_decay": 0.01,
    "transaction_optimizer_grad_clip": 1.0,
    "transaction_closure_sequence_length":
        TRANSACTION_CLOSURE_SEQUENCE_LENGTH,
    "transaction_closure_active_tokens": TRANSACTION_CLOSURE_ACTIVE_TOKENS,
    "transaction_writeback_max_wall_seconds":
        TRANSACTION_WRITEBACK_MAX_WALL_SECONDS,
    "fsdp_parameter_commit_primitive": FSDP_PARAMETER_COMMIT_PRIMITIVE,
}
FLOORS = {
    "tau_theta": 1e-12,
    "tau_logprob": 1e-6,
    "tau_gradient": 1e-8,
    "tau_coefficient": 1e-10,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fsdp_transaction_closure(closure, *, tau_logprob: float):
    """Independently validate the complete two-rank transaction probe."""
    if not isinstance(closure, list) \
            or sorted(int(item.get("rank", -1)) for item in closure) != [0, 1]:
        raise ValueError("transaction closure identity")
    phase_names = [
        "T0_behavior", "T1_after_backward", "T2_after_real_optimizer_step",
        "T3_legacy_raw_restore_diagnostic",
        "T4_safe_behavior_writeback", "T5_safe_candidate_recommit",
        "T6_safe_restore_fresh",
    ]
    safe_error_names = {
        "after_backward_max_abs", "safe_noop_writeback_max_abs",
        "safe_candidate_recommit_max_abs", "safe_restore_max_abs",
        "safe_second_forward_max_abs",
    }
    for item in closure:
        safe_errors = item.get("safe_errors")
        raw = item.get("legacy_raw_copy_diagnostic")
        optimizer = item.get("optimizer_probe")
        phases = item.get("phases")
        writeback = item.get("writeback_wall_seconds")
        if item.get("status") != "PASS" \
                or item.get("decision") != \
                    "RWWPO2_FSDP_TRANSACTION_CLOSURE_PASS" \
                or item.get("primitive") != FSDP_PARAMETER_COMMIT_PRIMITIVE \
                or int(item.get("sequence_length", 0)) != \
                    TRANSACTION_CLOSURE_SEQUENCE_LENGTH \
                or int(item.get("active_tokens", 0)) != \
                    TRANSACTION_CLOSURE_ACTIVE_TOKENS \
                or float(item.get("writeback_max_wall_seconds", -1)) != \
                    TRANSACTION_WRITEBACK_MAX_WALL_SECONDS \
                or not re.fullmatch(r"[0-9a-f]{64}", str(
                    item.get("behavior_logprob_digest", ""))) \
                or float(item.get("tau_logprob", -1)) != float(tau_logprob) \
                or not isinstance(safe_errors, dict) \
                or set(safe_errors) != safe_error_names \
                or any(not math.isfinite(float(value)) or float(value) < 0
                       or float(value) > float(tau_logprob)
                       for value in safe_errors.values()) \
                or not math.isfinite(float(item.get(
                    "safe_candidate_activation_max_abs", -1))) \
                or float(item["safe_candidate_activation_max_abs"]) <= \
                    float(tau_logprob) \
                or not isinstance(raw, dict) \
                or set(raw) != {"candidate_activation_max_abs", "restore_max_abs"} \
                or any(not math.isfinite(float(value)) or float(value) < 0
                       for value in raw.values()) \
                or not isinstance(optimizer, dict) \
                or set(optimizer) != {
                    "kind", "lr", "betas", "weight_decay", "grad_clip",
                    "step_calls", "grad_norm", "proposal_max_abs",
                    "state_entry_counts",
                } \
                or optimizer.get("kind") != "AdamW" \
                or float(optimizer.get("lr", -1)) != 1e-6 \
                or optimizer.get("betas") != [0.9, 0.999] \
                or float(optimizer.get("weight_decay", -1)) != 0.01 \
                or float(optimizer.get("grad_clip", -1)) != 1.0 \
                or int(optimizer.get("step_calls", -1)) != 1 \
                or not math.isfinite(float(optimizer.get(
                    "grad_norm", float("nan")))) \
                or float(optimizer.get("grad_norm", -1)) <= 0 \
                or not math.isfinite(float(optimizer.get(
                    "proposal_max_abs", float("nan")))) \
                or float(optimizer.get("proposal_max_abs", -1)) <= 0 \
                or not isinstance(optimizer.get("state_entry_counts"), list) \
                or len(optimizer["state_entry_counts"]) != 2 \
                or any(not isinstance(value, int) or value <= 0
                       for value in optimizer["state_entry_counts"]) \
                or len(set(optimizer["state_entry_counts"])) != 1 \
                or not isinstance(writeback, dict) \
                or set(writeback) != {
                    "safe_behavior", "safe_candidate",
                    "safe_candidate_recommit", "safe_restore"} \
                or any(not math.isfinite(float(value)) or float(value) < 0
                       or float(value) >
                       TRANSACTION_WRITEBACK_MAX_WALL_SECONDS
                       for value in writeback.values()) \
                or not isinstance(phases, list) \
                or [phase.get("phase") for phase in phases] != phase_names:
            raise ValueError("transaction closure semantics")
        for phase in phases:
            inventory = phase.get("execution_inventory")
            if not isinstance(inventory, dict) \
                    or int(inventory.get("unit_count", 0)) < 1 \
                    or int(inventory.get("managed_unit_count", 0)) != \
                        int(inventory.get("unit_count", -1)) \
                    or inventory.get("training_states") != {
                        "TrainingState.IDLE": int(inventory["unit_count"])} \
                    or not isinstance(inventory.get("storage"), dict) \
                    or not any(
                        key.startswith("flat_param_data:torch.float32:cuda")
                        and int(value.get("tensor_count", -1)) ==
                            int(inventory["managed_unit_count"])
                        and int(value.get("numel", 0)) > 0
                        and int(value.get("allocated_bytes", 0)) > 0
                        and int(value.get("nonzero_data_ptr_count", -1)) ==
                            int(inventory["managed_unit_count"])
                        for key, value in inventory["storage"].items()
                        if isinstance(value, dict)):
                raise ValueError("transaction closure execution inventory")
        phase_by_name = {phase["phase"]: phase for phase in phases}
        if phase_by_name["T0_behavior"].get("logprob_digest") != \
                item["behavior_logprob_digest"] \
                or float(phase_by_name["T1_after_backward"].get(
                    "max_abs", -1)) != float(
                        safe_errors["after_backward_max_abs"]) \
                or float(phase_by_name["T2_after_real_optimizer_step"].get(
                    "optimizer_proposal_max_abs", -1)) != float(
                        optimizer["proposal_max_abs"]) \
                or float(phase_by_name[
                    "T3_legacy_raw_restore_diagnostic"].get(
                        "candidate_activation_max_abs", -1)) != float(
                            raw["candidate_activation_max_abs"]) \
                or float(phase_by_name[
                    "T3_legacy_raw_restore_diagnostic"].get(
                        "restore_max_abs", -1)) != float(
                            raw["restore_max_abs"]) \
                or float(phase_by_name["T4_safe_behavior_writeback"].get(
                    "max_abs", -1)) != float(
                        safe_errors["safe_noop_writeback_max_abs"]) \
                or float(phase_by_name["T5_safe_candidate_recommit"].get(
                    "candidate_activation_max_abs", -1)) != float(
                        item["safe_candidate_activation_max_abs"]) \
                or float(phase_by_name["T5_safe_candidate_recommit"].get(
                    "recommit_max_abs", -1)) != float(
                        safe_errors["safe_candidate_recommit_max_abs"]) \
                or float(phase_by_name["T6_safe_restore_fresh"].get(
                    "max_abs", -1)) != float(
                        safe_errors["safe_restore_max_abs"]) \
                or float(phase_by_name["T6_safe_restore_fresh"].get(
                    "second_forward_max_abs", -1)) != float(
                        safe_errors["safe_second_forward_max_abs"]) \
                or optimizer["state_entry_counts"][int(item["rank"])] != int(
                    phase_by_name["T2_after_real_optimizer_step"]
                    ["execution_inventory"]["managed_unit_count"]):
            raise ValueError("transaction closure phase binding")
    distributed_fields = (
        "sequence_length", "active_tokens", "tau_logprob",
        "writeback_max_wall_seconds", "writeback_wall_seconds",
        "safe_errors", "safe_candidate_activation_max_abs",
        "legacy_raw_copy_diagnostic", "optimizer_probe",
    )
    if any(closure[0].get(field) != closure[1].get(field)
           for field in distributed_fields):
        raise ValueError("transaction closure distributed binding")
    return closure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", required=True)
    parser.add_argument("--oracle-report-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:checkout")
    raw_root = Path(args.oracle_root)
    if raw_root.is_symlink():
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:root symlink")
    root = raw_root.resolve()
    report_path = root / "numeric_oracle.json"
    if report_path.is_symlink() or not report_path.is_file() \
            or sha256_file(report_path) != args.oracle_report_sha256:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:report identity")
    row = json.loads(report_path.read_text(encoding="utf-8"))
    declared = row.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared != actual or row.get("status") != "PASS" \
            or row.get("decision") != "RWWPO2_NUMERIC_ORACLE_PASS" \
            or row.get("git_commit") != head or int(row.get("world_size", 0)) != 2:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:receipt")
    if float(row.get("threshold_multiplier", -1)) != MULTIPLIER \
            or int(row.get("gradient_sketch_chunk_elements", -1)) != \
                GRADIENT_SKETCH_CHUNK_ELEMENTS \
            or row.get("streamed_replay_calibration") != \
                STREAMED_REPLAY_CALIBRATION \
            or row.get("threshold_floors") != FLOORS:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:threshold rule")
    observed = row.get("observed", {})
    if set(observed) != {
        "repeated_logprob_max_abs", "repeated_gradient_projection_relative_l2",
        "streamed_replay_gradient_projection_relative_l2",
        "save_load_parameter_relative_l2", "save_load_logprob_max_abs",
        "save_load_gradient_projection_relative_l2",
        "behavior_actual_loss_logprob_max_abs",
        "behavior_actual_loss_coefficient_max_abs",
        "behavior_actual_loss_gradient_projection_relative_l2",
        "allreduce_max_abs",
    } or any(not math.isfinite(float(value)) or float(value) < 0
             for value in observed.values()):
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:observed numerics")
    expected_thresholds = {
        "tau_theta": max(FLOORS["tau_theta"], MULTIPLIER * float(
            observed["save_load_parameter_relative_l2"])),
        "tau_logprob": max(FLOORS["tau_logprob"], MULTIPLIER * max(
            float(observed["repeated_logprob_max_abs"]),
            float(observed["save_load_logprob_max_abs"]),
            float(observed["behavior_actual_loss_logprob_max_abs"]))),
        "tau_gradient": max(FLOORS["tau_gradient"], MULTIPLIER * max(
            float(observed["repeated_gradient_projection_relative_l2"]),
            float(observed["streamed_replay_gradient_projection_relative_l2"]),
            float(observed["save_load_gradient_projection_relative_l2"]),
            float(observed["behavior_actual_loss_gradient_projection_relative_l2"]))),
        "tau_coefficient": max(FLOORS["tau_coefficient"], MULTIPLIER * float(
            observed["behavior_actual_loss_coefficient_max_abs"])),
    }
    if row.get("thresholds") != expected_thresholds \
            or float(observed["allreduce_max_abs"]) != 0.0:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:threshold reconstruction")
    if row.get("fsdp_parameter_commit_primitive") != \
            FSDP_PARAMETER_COMMIT_PRIMITIVE:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:transaction closure identity")
    try:
        closure = validate_fsdp_transaction_closure(
            row.get("fsdp_transaction_closure"),
            tau_logprob=float(expected_thresholds["tau_logprob"]))
    except (TypeError, ValueError) as error:
        raise SystemExit(
            "RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:" + str(error)) from error
    gpu_pair = row.get("gpu_pair")
    binding = row.get("gpu_binding")
    if not isinstance(gpu_pair, list) or len(gpu_pair) != 2 \
            or gpu_pair != sorted(set(int(value) for value in gpu_pair)) \
            or not isinstance(binding, list) or len(binding) != 2 \
            or any("NVIDIA H20" not in value or "GPU-" not in value for value in binding):
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:GPU binding")
    state_inventory = []
    evidence = row.get("rank_state_evidence", [])
    if sorted(int(item.get("rank", -1)) for item in evidence) != [0, 1]:
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:rank closure")
    for item in evidence:
        relative = item.get("relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute() \
                or re.fullmatch(r"state/rank_[01]\.pt", relative) is None:
            raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:state path")
        path = root / relative
        if path.is_symlink() or not path.is_file() or root not in path.resolve().parents \
                or path.stat().st_size != int(item.get("state_size", -1)) \
                or sha256_file(path) != item.get("state_sha256"):
            raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:state evidence")
        state_inventory.append({
            "rank": int(item["rank"]), "relative_path": relative,
            "size": path.stat().st_size, "sha256": sha256_file(path),
        })
    model_path = Path(row.get("model_path", ""))
    if not model_path.is_absolute() or not model_path.joinpath("config.json").is_file() \
            or sha256_file(model_path / "config.json") != row.get("model_config_sha256"):
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_AUDIT_NO_GO:model identity")
    audit = {
        "schema_version": "rwwpo2-numeric-oracle-audit-v1",
        "status": "PASS", "decision": "RWWPO2_NUMERIC_ORACLE_AUDIT_PASS",
        "git_commit": head, "oracle_root": str(root),
        "oracle_report_file_sha256": args.oracle_report_sha256,
        "oracle_report_sha256": declared, "thresholds": expected_thresholds,
        "gradient_sketch_chunk_elements": GRADIENT_SKETCH_CHUNK_ELEMENTS,
        "streamed_replay_calibration": row["streamed_replay_calibration"],
        "fsdp_parameter_commit_primitive": FSDP_PARAMETER_COMMIT_PRIMITIVE,
        "fsdp_transaction_closure": closure,
        "gpu_pair": gpu_pair, "gpu_binding": binding,
        "rank_state_inventory": state_inventory,
    }
    raw = json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False)
    audit["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": audit["decision"],
        "output": str(output.resolve()), "thresholds": expected_thresholds,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
