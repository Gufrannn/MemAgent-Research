#!/usr/bin/env python3
"""Static fail-closed firewall for the RWWPO-2 scientific path."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SCIENTIFIC=(
    ROOT/"recurrent/research/rwwpo_transaction.py",
    ROOT/"recurrent/research/rwwpo_ledger.py",
    ROOT/"verl/trainer/ppo/core_algos.py",
    ROOT/"verl/workers/actor/dp_actor.py",
)
ENTRY=(
    ROOT/"gate_a_execution_ledger.schema.json",
    ROOT/"recurrent/research/gate_a_execution.py",
    ROOT/"experiments/7b_gate_a/run_gate_a.sh",
    ROOT/"scripts/h20/rwwpo2_common.sh",
    ROOT/"scripts/h20/run_qwen25_7b_rwwpo2.sh",
    ROOT/"scripts/h20/run_rwwpo2_numeric_oracle.sh",
    ROOT/"tools/h20/run_rwwpo2_release_tests.py",
    ROOT/"tools/h20/verify_rwwpo2_release_tests.py",
    ROOT/"tools/h20/rwwpo2_pytest_evidence_plugin.py",
    ROOT/"tools/h20/preflight_rwwpo2.py",
    ROOT/"tools/h20/audit_rwwpo_actual_loss.py",
    ROOT/"tools/h20/run_rwwpo2_e0.py",
    ROOT/"tools/h20/materialize_rwwpo2_resolved_contract.py",
    ROOT/"tools/h20/audit_rwwpo2_attempt.py",
    ROOT/"tools/h20/audit_rwwpo2_lineage_parent.py",
    ROOT/"tools/h20/audit_rwwpo2_cross_commit_resume.py",
    ROOT/"tools/h20/audit_rwwpo2_r50_program.py",
    ROOT/"tools/h20/materialize_rwwpo_diagnostic_eval_manifest.py",
    ROOT/"tools/h20/compare_rwwpo2_hotpot_t20_bde.py",
    ROOT/"scripts/h20/run_rwwpo2_hotpot_t20_bde_diagnostic.sh",
    ROOT/"tools/h20/audit_rwwpo2_base_protocol.py",
    ROOT/"tools/h20/audit_rwwpo2_data_boundary.py",
    ROOT/"tools/h20/calibrate_rwwpo2_numeric_oracle.py",
    ROOT/"tools/h20/audit_rwwpo2_numeric_oracle.py",
    ROOT/"tools/h20/seal_rwwpo2_confirmation_set.py",
    ROOT/"tools/h20/preflight_rwwpo2_confirmation.py",
    ROOT/"tools/h20/materialize_rwwpo2_confirmation_eval.py",
    ROOT/"tools/h20/audit_rwwpo2_confirmation_eval.py",
    ROOT/"tools/h20/finalize_rwwpo2_confirmation.py",
    ROOT/"scripts/h20/run_rwwpo2_confirmation_eval.sh",
    ROOT/"recurrent/research/rwwpo2_confirmation.py",
    ROOT/"recurrent/research/rwwpo2_babilong.py",
    ROOT/"recurrent/research/hotpotqa_dense_reward.py",
    ROOT/"verl/trainer/ppo/ray_trainer.py",
    ROOT/"manifests/h20/rwwpo2_babilong_pilot_v1.json",
    ROOT/"tools/h20/audit_rwwpo2_babilong_fixtures.py",
    ROOT/"tools/h20/audit_rwwpo2_babilong_data_boundary.py",
    ROOT/"tools/h20/fetch_rwwpo2_babilong_source.py",
    ROOT/"tools/h20/materialize_rwwpo2_babilong.py",
    ROOT/"tools/h20/audit_rwwpo2_babilong_bundle.py",
    ROOT/"tools/h20/preflight_rwwpo2_babilong.py",
    ROOT/"tools/h20/materialize_rwwpo2_babilong_eval.py",
    ROOT/"tools/h20/audit_rwwpo2_babilong_eval.py",
    ROOT/"tools/h20/compare_rwwpo2_babilong.py",
    ROOT/"scripts/h20/run_rwwpo2_babilong_eval.sh",
    ROOT/"scripts/h20/run_rwwpo2_babilong_bd_t20.sh",
    ROOT/"scripts/h20/run_rwwpo2_babilong_cell.sh",
    ROOT/"scripts/h20/run_rwwpo2_babilong_prepare_and_bd_t20.sh",
    ROOT/"tests/h20/test_rwwpo2_babilong.py",
    ROOT/"rwwpo2_actual_loss_receipt.schema.json",
    ROOT/"rwwpo2_experiment_manifest.schema.json",
)
FORBIDDEN=("paired_effect","ccod","bopr","ncr","gold_answer","future_chunk",
           "hotpotqa_dev.parquet","s128_original","babilong")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    if head!=args.expected_commit or dirty:
        raise SystemExit("RWWPO2_SOURCE_FIREWALL_NO_GO:checkout")
    raw_manifest=Path(args.manifest)
    if raw_manifest.is_symlink() or not raw_manifest.is_file():
        raise SystemExit("RWWPO2_SOURCE_FIREWALL_NO_GO:manifest symlink/missing")
    manifest_path=raw_manifest.resolve()
    manifest=json.loads(manifest_path.read_text())
    violations=[]
    for path in SCIENTIFIC:
        lowered=path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN:
            if token in lowered:
                violations.append(f"{path.relative_to(ROOT)}:{token}")
    method=manifest.get("method",{})
    training=manifest.get("training",{})
    if manifest.get("program")!="RWWPO-2": violations.append("program")
    if method.get("inner_transactions_per_round")!=2: violations.append("K2")
    if method.get("optimizer_minibatches_per_inner_transaction")!=1:
        violations.append("K2 minibatch mapping")
    if method.get("optimizer_steps_per_inner_transaction")!=1:
        violations.append("K2 optimizer-step mapping")
    if training.get("ppo_epochs")!=2: violations.append("ppo_epochs")
    if training.get("critic_optimizer_updates")!=0: violations.append("critic")
    if training.get("auxiliary_fit_updates")!=0: violations.append("auxiliary")
    if method.get("proposal_clock")!="2*(round_id-1)+inner_id": violations.append("clock")
    if manifest.get("performance",{}).get("s128_role")!=(
            "adaptive_development_only_forbidden_during_r50_and_r400_training"):
        violations.append("S128 role")
    if method.get("attempt_id_in_algorithmic_seed", False):
        violations.append("attempt identity enters algorithm")
    if training.get("ppo_mini_batch_size") != training.get("train_batch_size"):
        violations.append("not one full optimizer minibatch")
    actor_source=(ROOT/"verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
    for token in (
        "inner_id = int(epoch + 1)",
        "RWWPO transactions require one full optimizer minibatch per inner update",
        "RWWPO2_BEHAVIOR_BATCH_MUTATED_BETWEEN_INNER_UPDATES",
        "logical_proposal_id = proposal_clock(round_id, inner_id)",
        "decision = largest_tested_feasible(",
        "transaction_optimizer_step_calls += 1",
        "RWWPO2_OPTIMIZER_STEP_COUNT_DRIFT",
        '"behavior_coefficient_tolerance", 1e-9',
        '"behavior_gradient_tolerance",1e-7',
        '"shared_kl_loss": float(shared_additive_loss.detach().item())',
        'ref_log_prob=(joined("ref_log_prob") if rwwpo2_enabled else None)',
        '"ref_log_prob": joined("ref_log_prob")',
        "transaction_entry_rng = rng_snapshot()",
        "named_buffer_snapshot(self.actor_module)",
        "if rwwpo2_enabled else None",
        "restore_named_buffers(",
        "behavior_forward_rng_digests = ordered_rng_state_digests(",
        "def replay_behavior_log_probs():",
        "replay_with_rng_snapshots(",
        '"replay_rng_bound": True',
        '"behavior_current_logprob_integrity_verified"',
        '"fsdp_parameter_commit_primitive"',
        '"fsdp_parameter_writeback_wall_seconds"',
        "RWWPO2_FSDP_WRITEBACK_BUDGET_EXCEEDED",
        "_timed_set_interpolated_parameters(",
        "post_commit_forward_verified = True",
        'rwwpo_controller == "none"',
        "post_constraint_valid",
        '"transaction_entry_buffer_digest"',
        '"terminal_buffer_digest"',
        "RWWPO2_POST_COMMIT_FORWARD_CLOSURE_FAILURE",
        "append_transaction_failure_record(",
        "restore_rng(transaction_entry_rng)",
        '"transaction_entry_rng_digest": pre_digests["rng"]',
    ):
        if token not in actor_source:
            violations.append("missing executable K2 mapping:"+token)
    ledger_source=(ROOT/"recurrent/research/rwwpo_ledger.py").read_text(
        encoding="utf-8")
    actual_auditor=(ROOT/"tools/h20/audit_rwwpo_actual_loss.py").read_text(
        encoding="utf-8")
    for token in ("ROOT = Path(__file__).resolve().parents[2]",
                  "sys.path.insert(0, str(ROOT))"):
        if token not in actual_auditor:
            violations.append("actual-loss direct entry:"+token)
    if all(token in actual_auditor for token in (
            "sys.path.insert(0, str(ROOT))",
            "from recurrent.research.rwwpo_transaction")) \
            and actual_auditor.index("sys.path.insert(0, str(ROOT))") > \
                actual_auditor.index("from recurrent.research.rwwpo_transaction"):
        violations.append("actual-loss repo bootstrap order")
    for token in ("ref_log_prob", "append_transaction_failure_record",
                  "rwwpo2-transaction-failure-v1"):
        if token not in ledger_source:
            violations.append("actual-loss tensor ledger:"+token)
    for token in ("independently_recompute_actual_loss", "actual_loss_contract",
                  "shared_kl_loss", "active_logprob_gradient_l2",
                  "post_commit_forward_verified",
                  "post_commit_forward_verification_max_abs",
                  "transaction_entry_buffer_digest",
                  "terminal_buffer_digest",
                  "invalid canonical backtracking evidence",
                  "validate_rwwpo2_rng_phase_digests",
                  "RWWPO-2 RNG phase digest closure",
                  "RWWPO-2 rejected transaction RNG rollback",
                  "RWWPO-2 behavior replay RNG binding",
                  "reconstruct_authenticated_prefix_rows",
                  "writer_row_log_ratio = ((log_prob_tensor - old) * writer_mask).sum(dim=-1)",
                  "torch.cumsum(writer_row_log_ratio[indices], dim=0)",
                  "immutable behavior logprob digest mismatch",
                  "RWWPO-2 FSDP/behavior-reference closure",
                  "RWWPO-2 FSDP/trial wall-time closure",
                  "distributed FSDP/trial wall-time drift"):
        if token not in actual_auditor:
            violations.append("actual-loss independent audit:"+token)
    if '"actual_loss_contract"' not in actor_source:
        violations.append("actual-loss producer:actual_loss_contract")
    receipt_schema=(ROOT/"rwwpo2_actual_loss_receipt.schema.json").read_text(
        encoding="utf-8")
    for token in ('"behavior_forward_rng_digests"',
                  '"behavior_forward_rng_aggregate_digest"',
                  '"replay_microbatch_count"', '"replay_rng_bound"',
                  '"behavior_current_logprob_digest"',
                  '"behavior_current_logprob_integrity_verified"',
                  '"fsdp_parameter_commit_primitive"',
                  '"fsdp_parameter_writeback_max_wall_seconds"',
                  '"fsdp_parameter_writeback_wall_seconds"',
                  '"max_trial_forward_wall_seconds"'):
        if token not in receipt_schema:
            violations.append("actual-loss replay RNG schema:"+token)
    transaction_source=(ROOT/"recurrent/research/rwwpo_transaction.py").read_text(
        encoding="utf-8")
    for token in ("np.random.seed", "np.random.get_state", "np.random.set_state",
                  '"torch_cuda"', "def named_buffer_snapshot(",
                  "def restore_named_buffers(", "def module_state_digest(",
                  "def ordered_rng_state_digests(",
                  "def replay_with_rng_snapshots(",
                  "def _set_fsdp_interpolated_parameters(",
                  "torch.distributed.all_gather_into_tensor(",
                  "torch.distributed.all_gather_object(",
                  "FSDP.summon_full_params(",
                  "writeback=True", "RWWPO2_FSDP_WRITEBACK_LOCAL_SHARD_MISMATCH",
                  "RWWPO2_FSDP_DISTRIBUTED_INVENTORY_DRIFT",
                  "RWWPO2_FSDP_WRITEBACK_MAX_WALL_SECONDS = 120.0",
                  "finally:\n        restore_rng(terminal)"):
        if token not in transaction_source:
            violations.append("complete transaction RNG:"+token)
    seed_signature=transaction_source.split("def logical_transaction_seed",1)
    if len(seed_signature)!=2 or "attempt" in seed_signature[1].split(")",1)[0]:
        violations.append("attempt in logical seed signature")
    numeric_oracle=(ROOT/"tools/h20/calibrate_rwwpo2_numeric_oracle.py").read_text(
        encoding="utf-8")
    for token in ("ROOT = Path(__file__).resolve().parents[2]",
                  "sys.path.insert(0, str(ROOT))",
                  "from verl.trainer.ppo.core_algos import",
                  "GRADIENT_SKETCH_CHUNK_ELEMENTS=RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS",
                  "local_gradient_sketch_sufficient_statistics",
                  "STREAMED_ORACLE_MICROBATCHES=7",
                  "STREAMED_ORACLE_SEQUENCE_LENGTH=8191",
                  "TRANSACTION_CLOSURE_SEQUENCE_LENGTH=8191",
                  "TRANSACTION_CLOSURE_ACTIVE_TOKENS=1024",
                  "TRANSACTION_WRITEBACK_MAX_WALL_SECONDS=120.0",
                  "streamed_replay_gradient_projection_relative_l2",
                  '"synthetic_label_free":True',
                  "gradient_checkpointing_enable(",
                  'gradient_checkpointing_kwargs={"use_reentrant":False}',
                  "get_fsdp_wrap_policy(",
                  "auto_wrap_policy=auto_wrap_policy",
                  "mixed_precision=mixed_precision",
                  "sharding_strategy=ShardingStrategy.FULL_SHARD",
                  "sync_module_states=True,use_orig_params=False",
                  "forward_prefetch=False",
                  "torch_dtype=torch.float32",
                  '"fsdp_sharded_parameter_dtype":"float32"',
                  "transaction_closure_probe(",
                  "transaction_backward_probe(",
                  "torch.optim.AdamW(",
                  "model.clip_grad_norm_(max_norm=1.0)",
                  "optimizer.step()",
                  '"optimizer_probe"',
                  '"step_calls": 1',
                  '"transaction_optimizer_probe":"adamw_fp32_shard_step_v1"',
                  "timed_safe_writeback(",
                  "RWWPO2_FSDP_TRANSACTION_CLOSURE_NO_GO",
                  "RWWPO2_FSDP_WRITEBACK_BUDGET_NO_GO",
                  "set_interpolated_parameters(",
                  'torch.autocast(device_type="cuda",dtype=torch.bfloat16)',
                  "apply_monkey_patch(model=model,ulysses_sp_size=1)",
                  "logprobs_from_logits(",
                  "inplace_backward=True"):
        if token not in numeric_oracle:
            violations.append("numeric oracle direct entry:"+token)
    for token in ("torch.log_softmax(logits[:,:-1].float()",
                  "torch.log_softmax(logits[:, :-1].float()"):
        if token in numeric_oracle.split("def streamed_replay_gradient",1)[1].split(
                "def behavior_actual_loss_gradient",1)[0]:
            violations.append("numeric oracle unbounded streamed logprob:"+token)
    behavior_old_anchor=(
        'with torch.no_grad():\n'
        '        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):\n'
        '            behavior_logits=model(input_ids=tokens,use_cache=False).logits\n'
        '            behavior_old_logp=logprobs_from_logits('
    )
    if behavior_old_anchor not in numeric_oracle:
        violations.append("numeric oracle behavior-old autocast binding")
    for token in ("RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS = 8_388_608",
                  "def local_gradient_sketch_sufficient_statistics(",
                  "raw_gradient.is_contiguous()",
                  "flattened = raw_gradient.view(-1)",
                  "RWWPO2_GRADIENT_SKETCH_NONCONTIGUOUS_GRADIENT_NO_GO",
                  "for chunk_start in range(0, flattened.numel(), chunk_elements)"):
        if token not in transaction_source:
            violations.append("registered gradient sketch:"+token)
    for token in ("local_gradient_sketch_sufficient_statistics(",
                  "RWWPO2_GRADIENT_SKETCH_CHUNK_CONTRACT_DRIFT",
                  "[RWWPO2_BEHAVIOR_GRADIENT_DIAG]",
                  '"gradient_sketch_chunk_elements":'):
        if token not in actor_source:
            violations.append("actor registered gradient sketch:"+token)
    for token in ("parameter.grad.detach().double().flatten()",
                  "parameter.grad.detach().flatten()"):
        if token in numeric_oracle or token in actor_source:
            violations.append("gradient sketch full-shard materialization:"+token)
    if all(token in numeric_oracle for token in (
            "sys.path.insert(0, str(ROOT))",
            "from verl.trainer.ppo.core_algos import")) \
            and numeric_oracle.index("sys.path.insert(0, str(ROOT))") > \
                numeric_oracle.index("from verl.trainer.ppo.core_algos import"):
        violations.append("numeric oracle repo bootstrap order")
    launcher=(ROOT/"scripts/h20/run_qwen25_7b_rwwpo2.sh").read_text(encoding="utf-8")
    for token in (
        'VAL="$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet"',
        "PPO_EPOCHS=2", "SAVE_FREQ=10", "MAX_ACTOR_CKPT_TO_KEEP=2",
        "RWWPO_R50_PROGRAM_GATE", "RWWPO_CONFIRMATION_SEAL",
        "RWWPO_BEHAVIOR_COEFFICIENT_TOLERANCE",
        "RWWPO_BEHAVIOR_GRADIENT_TOLERANCE",
        "RWWPO_GRADIENT_SKETCH_CHUNK_ELEMENTS",
        "RWWPO_FSDP_PARAMETER_COMMIT_PRIMITIVE",
        "RWWPO_FSDP_WRITEBACK_MAX_WALL_SECONDS",
        "RWWPO_MAX_TRIAL_FORWARD_SECONDS",
        "RWWPO_RELEASE_TEST_RECEIPT",
        "RWWPO_CROSS_COMMIT_COMPATIBILITY_RECEIPT",
        "--cross-commit-compatibility",
    ):
        if token not in launcher:
            violations.append("launcher contract:"+token)
    if any(token in launcher for token in ("kill -9", "pkill", "killall")):
        violations.append("destructive process management")
    gate=(ROOT/"experiments/7b_gate_a/run_gate_a.sh").read_text(encoding="utf-8")
    for token in (
        "trainer.val_before_train=False", "trainer.test_freq=-1",
        "reward_model.enable=False",
        'custom_reward_function.path=$CODE/recurrent/research/hotpotqa_dense_reward.py',
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.trajectory_seed_mode=independent",
    ):
        if token not in gate:
            violations.append("gate protocol:"+token)
    trainer=(ROOT/"verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    attempt=(ROOT/"tools/h20/audit_rwwpo2_attempt.py").read_text(encoding="utf-8")
    lineage=(ROOT/"tools/h20/audit_rwwpo2_lineage_parent.py").read_text(encoding="utf-8")
    for token in ('parser.add_argument(\n        "--producer-commit"',
                  '"producer_git_commit": producer_commit',
                  '"auditor_git_commit": head',
                  '"auditor_source_sha256": sha256_file(Path(__file__).resolve())'):
        if token not in lineage:
            violations.append("lineage compatibility auditor binding:"+token)
    compatibility=(ROOT/"tools/h20/audit_rwwpo2_cross_commit_resume.py").read_text(
        encoding="utf-8")
    preflight=(ROOT/"tools/h20/preflight_rwwpo2.py").read_text(encoding="utf-8")
    for token in (
        "PROTECTED_EXACT_SOURCES",
        "TRAINER_COMPATIBILITY_EXCLUSIONS",
        "trainer_projection(",
        "producer_resolved_contract_reused",
        "consumer_numeric_contract_substitution_forbidden",
        "algorithmic_source_or_contract_change",
        "launcher_projection(",
        "RWWPO2_CROSS_COMMIT_COMPATIBILITY_WIRING",
        "RWWPO2_CROSS_COMMIT_RESUME_COMPATIBILITY_PASS",
    ):
        if token not in compatibility:
            violations.append("cross-commit compatibility producer:"+token)
    for token in (
        "--cross-commit-compatibility",
        "cross_commit_compatibility_report_sha256",
        "cross_commit_producer_git_commit",
        "producer_resolved_contract_file_sha256",
    ):
        if token not in preflight:
            violations.append("cross-commit preflight binding:"+token)
    for token in (
        "RWWPO2_RESUME_CROSS_COMMIT_COMPATIBILITY_DRIFT",
        '"cross_commit_compatibility"',
        '"producer_git_commit"',
        '"consumer_git_commit"',
    ):
        if token not in trainer:
            violations.append("cross-commit runtime binding:"+token)
    for token in (
        "cross_commit_compatibility_report_sha256",
        "cross-commit resume binding",
        "cross_commit_producer_git_commit",
        "--segment-producer-commit",
        "--cross-commit-compatibility",
    ):
        if token not in attempt:
            violations.append("cross-commit attempt audit:"+token)
    r50=(ROOT/"tools/h20/audit_rwwpo2_r50_program.py").read_text(encoding="utf-8")
    for token in (
        "--cross-commit-compatibility",
        "segment compatibility binding",
        "resolved_contract_producer_git_commit",
    ):
        if token not in r50:
            violations.append("cross-commit R50 aggregation:"+token)
    t20_runner=(ROOT/"scripts/h20/run_rwwpo2_hotpot_t20_bde_diagnostic.sh").read_text(
        encoding="utf-8")
    t20_compare=(ROOT/"tools/h20/compare_rwwpo2_hotpot_t20_bde.py").read_text(
        encoding="utf-8")
    for token in (
        "RWWPO2_TRAINING_COMMIT", "RWWPO2_S128_RESOLVED_SHA256",
        "RWWPO2_S128_MANIFEST_HASH", "for cell in B D E",
        "--diagnostic-only", "wait_for_idle",
    ):
        if token not in t20_runner:
            violations.append("T20 B/D/E diagnostic runner:"+token)
    for token in (
        "development_diagnostic_not_blind_final",
        "single_seed_fixed_S128_descriptive_only",
        '"B", "D"', '"E", "D"', '"B", "E"',
    ):
        if token not in t20_compare:
            violations.append("T20 B/D/E diagnostic comparison:"+token)
    for token in ("rwwpo_ledger_anchors", "rwwpo_tensor_inventory",
                  "rwwpo_rollout_seed_anchor",
                  "rwwpo2_resolved_contract_file_sha256",
                  "rwwpo2_resolved_contract_report_sha256"):
        if token not in trainer:
            violations.append("checkpoint prefix anchor:"+token)
    for token in ("record_limits=record_limits", "execution_prefix_sha256"):
        if token not in attempt or token not in lineage:
            violations.append("recovery prefix audit:"+token)
    for token in (
        "rwwpo2_recovery_prune_intent",
        "rwwpo2_recovery_pruned",
        "prune_intent_record_sha256",
        "scientific_anchor_inventory_record_sha256",
        "RWWPO2_RECOVERY_PRUNE_INTENT_NOT_RECORDED",
        "RWWPO2_RECOVERY_PRUNE_COMPLETE_NOT_RECORDED",
    ):
        if token not in trainer:
            violations.append("two-phase recovery prune producer:"+token)
    for token in (
        "validate_recovery_prune_evidence(",
        "recovery prune intent/complete closure",
        "recovery prune semantic closure",
        '"two_phase_evidence": True',
    ):
        if token not in attempt:
            violations.append("two-phase recovery prune audit:"+token)
    for token in (
        'parser.add_argument("--preflight", required=True)',
        '"preflight_report_sha256"', "R400 preflight gate binding",
        "preflight lineage start", "validate_rwwpo2_rng_phase_digests(row)",
        "validate_transaction_failure_boundary(",
        "transaction failure inside audited prefix",
        "validate_post_commit_forward_binding(",
        "post-commit forward binding",
        "FSDP/trial wall-time binding",
        "distributed wall-time drift",
    ):
        if token not in attempt:
            violations.append("attempt/preflight binding:"+token)
    preflight=(ROOT/"tools/h20/preflight_rwwpo2.py").read_text(encoding="utf-8")
    numeric_auditor=(ROOT/"tools/h20/audit_rwwpo2_numeric_oracle.py").read_text(
        encoding="utf-8")
    resolver=(ROOT/"tools/h20/materialize_rwwpo2_resolved_contract.py").read_text(
        encoding="utf-8")
    for token in ("streamed_replay_calibration",
                  "streamed_replay_gradient_projection_relative_l2",
                  "fsdp_transaction_closure",
                  "validate_fsdp_transaction_closure",
                  "FSDP_PARAMETER_COMMIT_PRIMITIVE",
                  "transaction_optimizer_probe",
                  '"TrainingState.IDLE"',
                  '"transaction closure phase binding"',
                  '"transaction closure distributed binding"'):
        if token not in numeric_auditor:
            violations.append("numeric auditor streaming calibration:"+token)
    for token in ("streamed_replay_calibration",
                  "STREAMED_REPLAY_CALIBRATION",
                  "fsdp_transaction_closure",
                  "validate_fsdp_transaction_closure",
                  "FSDP_PARAMETER_COMMIT_PRIMITIVE",
                  "transaction_optimizer_probe"):
        if token not in resolver:
            violations.append("resolved streaming calibration:"+token)
    for token in ("gradient sketch chunk binding",
                  "streamed replay calibration binding",
                  "FSDP transaction closure binding",
                  "FSDP transaction closure semantics",
                  "fsdp_parameter_writeback_max_wall_seconds",
                  "max_trial_forward_wall_seconds_per_transaction",
                  "validate_fsdp_transaction_closure"):
        if token not in preflight:
            violations.append("preflight registered gradient sketch:"+token)
    numeric_launcher=(ROOT/"scripts/h20/run_rwwpo2_numeric_oracle.sh").read_text(
        encoding="utf-8")
    release_producer=(ROOT/"tools/h20/run_rwwpo2_release_tests.py").read_text(
        encoding="utf-8")
    release_verifier=(ROOT/"tools/h20/verify_rwwpo2_release_tests.py").read_text(
        encoding="utf-8")
    for token in ("--release-test-receipt", "verify_release_test_receipt"):
        if token not in preflight:
            violations.append("training preflight release-test gate:"+token)
    for token in ("ROOT = Path(__file__).resolve().parents[2]",
                  "sys.path.insert(0, str(ROOT))"):
        if token not in preflight:
            violations.append("training preflight direct entry:"+token)
    if all(token in preflight for token in (
            "sys.path.insert(0, str(ROOT))",
            "from tools.h20.verify_rwwpo2_release_tests import")) \
            and preflight.index("sys.path.insert(0, str(ROOT))") > \
                preflight.index("from tools.h20.verify_rwwpo2_release_tests import"):
        violations.append("training preflight repo bootstrap order")
    for token in ("RWWPO_RELEASE_TEST_RECEIPT",
                  "verify_rwwpo2_release_tests.py"):
        if token not in numeric_launcher:
            violations.append("numeric oracle release-test gate:"+token)
    for token in ("TEST_INVENTORY", "RWWPO2_RELEASE_TESTS_PASS",
                  "checkout_postcondition", "pytest_command(mode=\"collect\"",
                  "runtime_environment"):
        if token not in release_producer:
            violations.append("release-test producer:"+token)
    for token in ("--junitxml", "junit_summary", "test_source_sha256",
                  "release-test Python environment drift",
                  "python_executable_sha256", "installed_distributions_sha256",
                  "collect_current_node_ids", "non-PASS/skip/xfail"):
        if token not in release_verifier:
            violations.append("release-test verifier:"+token)
    if '"tests/h20/test_rwwpo2_babilong.py"' not in release_verifier:
        violations.append("release-test BABILong inventory")
    babilong_contract_path = ROOT/"manifests/h20/rwwpo2_babilong_pilot_v1.json"
    babilong_contract = json.loads(babilong_contract_path.read_text(encoding="utf-8"))
    expected_babilong_contract = {
        "schema_version": "rwwpo2-babilong-adapter-v1",
        "dataset_id": "RMT-team/babilong",
        "dataset_revision": "e3a924b6686759422257925a695cbbb4b2684936",
        "source_rows_per_cell": 100,
        "lengths": ["32k", "128k"],
        "task_depth": {"qa1": 1, "qa2": 2, "qa3": 3},
        "chunk_size": 5000,
        "max_chunks": {"32k": 8, "128k": 32},
        "generation_seed": 602214076,
        "partition_sizes_per_cell": {"development": 8, "confirmation": 92},
        "primary_metric": "official_case_insensitive_target_substring_accuracy",
        "key_secondary": "strict_normalized_exact_match",
    }
    for field, value in expected_babilong_contract.items():
        if babilong_contract.get(field) != value:
            violations.append("BABILong frozen contract:"+field)
    babilong_adapter = (ROOT/"recurrent/research/rwwpo2_babilong.py").read_text(
        encoding="utf-8")
    for token in (
        "SOURCE_REVISION = \"e3a924b6686759422257925a695cbbb4b2684936\"",
        "DEVELOPMENT_SOURCE_INDICES = {",
        "truncation is forbidden",
        "def official_substring_accuracy(",
        "def paired_descriptive_difference(",
    ):
        if token not in babilong_adapter:
            violations.append("BABILong executable adapter:"+token)
    babilong_bundle_auditor = (
        ROOT/"tools/h20/audit_rwwpo2_babilong_bundle.py").read_text(encoding="utf-8")
    for token in (
        "validate_frozen_contract(manifest)",
        'partition_indices(length, task, materialized["partition"])',
        "RWWPO2_BABILONG_BUNDLE_AUDIT_PASS",
    ):
        if token not in babilong_bundle_auditor:
            violations.append("BABILong bundle auditor:"+token)
    babilong_eval_auditor = (
        ROOT/"tools/h20/audit_rwwpo2_babilong_eval.py").read_text(encoding="utf-8")
    for token in (
        "score_babilong_output(",
        "summarize_babilong_metrics(",
        "RWWPO2_BABILONG_CONFIRMATION_EVAL_PASS",
        "validate_attempt_identity_rows(",
    ):
        if token not in babilong_eval_auditor:
            violations.append("BABILong eval auditor:"+token)
    babilong_eval_launcher = (
        ROOT/"scripts/h20/run_rwwpo2_babilong_eval.sh").read_text(encoding="utf-8")
    for token in (
        "verify_rwwpo2_release_tests.py", "flock -n 8", "flock -n 9",
        "GPU occupied pass", "preflight_rwwpo2_babilong.py",
    ):
        if token not in babilong_eval_launcher:
            violations.append("BABILong eval launcher:"+token)
    babilong_boundary = (
        ROOT/"tools/h20/audit_rwwpo2_babilong_data_boundary.py").read_text(
            encoding="utf-8")
    for token in (
        "_actor_consumed_rows(", "train_intersect_babilong_root",
        "train_intersect_babilong_content", "raw_examples_emitted",
        "RWWPO2_BABILONG_DATA_BOUNDARY_AUDIT_PASS",
    ):
        if token not in babilong_boundary:
            violations.append("BABILong data boundary:"+token)
    confirmation=(ROOT/"tools/h20/finalize_rwwpo2_confirmation.py").read_text(
        encoding="utf-8")
    confirmation_protocol=(ROOT/"recurrent/research/rwwpo2_confirmation.py").read_text(
        encoding="utf-8")
    for token in (
        "one_sided_exact_paired_sign_flip", "holm_two_test_decisions",
        "score_terminal_output", "metric row reconstruction",
        "margin_centered_differences", "range(2026, 2034)",
    ):
        if token not in confirmation:
            violations.append("confirmation finalizer:"+token)
    for token in (
        "repository_relative_path", "path_sha256",
        "confirmation_data_sha256", "repeat=len(values)",
    ):
        if token not in confirmation_protocol:
            violations.append("confirmation protocol:"+token)
    if violations:
        raise SystemExit("RWWPO2_SOURCE_FIREWALL_NO_GO:"+",".join(violations))
    report={"status":"PASS","decision":"RWWPO2_SOURCE_FIREWALL_PASS",
            "git_commit":head,"manifest_sha256":hashlib.sha256(
                manifest_path.read_bytes()).hexdigest(),
            "scientific_source_sha256":{
                str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in SCIENTIFIC},
            "entry_source_sha256":{
                str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in ENTRY}}
    raw=json.dumps(report,sort_keys=True,separators=(",",":"))
    report["report_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(report,sort_keys=True,indent=2)+"\n")


if __name__=="__main__": main()
