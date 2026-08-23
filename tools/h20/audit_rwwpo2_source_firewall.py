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
    ROOT/"experiments/7b_gate_a/run_gate_a.sh",
    ROOT/"scripts/h20/rwwpo2_common.sh",
    ROOT/"scripts/h20/run_qwen25_7b_rwwpo2.sh",
    ROOT/"scripts/h20/run_rwwpo2_numeric_oracle.sh",
    ROOT/"tools/h20/preflight_rwwpo2.py",
    ROOT/"tools/h20/audit_rwwpo_actual_loss.py",
    ROOT/"tools/h20/run_rwwpo2_e0.py",
    ROOT/"tools/h20/materialize_rwwpo2_resolved_contract.py",
    ROOT/"tools/h20/audit_rwwpo2_attempt.py",
    ROOT/"tools/h20/audit_rwwpo2_lineage_parent.py",
    ROOT/"tools/h20/audit_rwwpo2_r50_program.py",
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
    ROOT/"recurrent/research/hotpotqa_dense_reward.py",
    ROOT/"rwwpo2_actual_loss_receipt.schema.json",
    ROOT/"rwwpo2_experiment_manifest.schema.json",
)
FORBIDDEN=("paired_effect","ccod","bopr","ncr","gold_answer","future_chunk",
           "hotpotqa_dev.parquet","s128_original")


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
        "transaction_optimizer_step_calls += 1",
        "RWWPO2_OPTIMIZER_STEP_COUNT_DRIFT",
        '"behavior_coefficient_tolerance", 1e-9',
        '"behavior_gradient_tolerance",1e-7',
        '"shared_kl_loss": float(shared_additive_loss.detach().item())',
        'ref_log_prob=(joined("ref_log_prob") if rwwpo2_enabled else None)',
        '"ref_log_prob": joined("ref_log_prob")',
    ):
        if token not in actor_source:
            violations.append("missing executable K2 mapping:"+token)
    ledger_source=(ROOT/"recurrent/research/rwwpo_ledger.py").read_text(
        encoding="utf-8")
    actual_auditor=(ROOT/"tools/h20/audit_rwwpo_actual_loss.py").read_text(
        encoding="utf-8")
    for token in ("ref_log_prob",):
        if token not in ledger_source:
            violations.append("actual-loss tensor ledger:"+token)
    for token in ("independently_recompute_actual_loss", "actual_loss_contract",
                  "shared_kl_loss", "active_logprob_gradient_l2"):
        if token not in actual_auditor:
            violations.append("actual-loss independent audit:"+token)
    if '"actual_loss_contract"' not in actor_source:
        violations.append("actual-loss producer:actual_loss_contract")
    transaction_source=(ROOT/"recurrent/research/rwwpo_transaction.py").read_text(
        encoding="utf-8")
    seed_signature=transaction_source.split("def logical_transaction_seed",1)
    if len(seed_signature)!=2 or "attempt" in seed_signature[1].split(")",1)[0]:
        violations.append("attempt in logical seed signature")
    launcher=(ROOT/"scripts/h20/run_qwen25_7b_rwwpo2.sh").read_text(encoding="utf-8")
    for token in (
        'VAL="$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet"',
        "PPO_EPOCHS=2", "SAVE_FREQ=10", "MAX_ACTOR_CKPT_TO_KEEP=2",
        "RWWPO_R50_PROGRAM_GATE", "RWWPO_CONFIRMATION_SEAL",
        "RWWPO_BEHAVIOR_COEFFICIENT_TOLERANCE",
        "RWWPO_BEHAVIOR_GRADIENT_TOLERANCE",
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
        'parser.add_argument("--preflight", required=True)',
        '"preflight_report_sha256"', "R400 preflight gate binding",
        "preflight lineage start",
    ):
        if token not in attempt:
            violations.append("attempt/preflight binding:"+token)
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
