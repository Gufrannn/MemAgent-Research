#!/usr/bin/env python3
"""Fail closed if TF-RWWPO imports forbidden evidence or changes frozen constants."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SCIENTIFIC=[
    ROOT/"recurrent/research/rwwpo_transaction.py",
    ROOT/"verl/workers/actor/dp_actor.py",
    ROOT/"verl/trainer/ppo/core_algos.py",
]
FORBIDDEN=("paired_effect", "ccod", "bopr", "ncr", "gold_answer", "future_chunk")

def main():
    violations=[]
    for path in SCIENTIFIC:
        lowered=path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN:
            if token in lowered: violations.append(f"{path.relative_to(ROOT)}:{token}")
    manifest=json.loads((ROOT/"manifests/h20/qwen25_7b_tf_rwwpo_seed2026.json").read_text())
    method=manifest["method"]
    if method["q_min"]!=0.5: violations.append("q_min drift")
    if method["alpha_grid"]!=[1.0,0.5,0.25,0.125,0.0625,0.03125]: violations.append("alpha grid drift")
    if manifest["training"]["target_steps"]!=[5,10,15,20,25]: violations.append("anchor drift")
    if manifest["training"].get("ppo_epochs")!=1: violations.append("PPO epoch drift")
    if manifest["training"].get("critic_optimizer_updates")!=0: violations.append("critic update drift")
    if manifest["training"].get("auxiliary_fit_updates")!=0: violations.append("auxiliary update drift")
    if manifest["training"].get("runtime_effective_prompt_filter_length")!=40000:
        violations.append("effective prompt-filter drift")
    launcher=(ROOT/"scripts/h20/run_qwen25_7b_rwwpo.sh").read_text(encoding="utf-8")
    gate=(ROOT/"experiments/7b_gate_a/run_gate_a.sh").read_text(encoding="utf-8")
    if "PPO_EPOCHS=1" not in launcher or '"actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS"' not in gate:
        violations.append("PPO epochs are not explicitly frozen")
    if "reward_model.enable=False" not in gate:
        violations.append("reward model fit is not explicitly disabled")
    trainer=(ROOT/"verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    try:
        example_hash=trainer.split(
            'batch.batch["rwwpo_example_identity_hash"] = torch.tensor(',1
        )[1].split('batch.batch["rwwpo_trajectory_identity_hash"]',1)[0]
    except IndexError:
        violations.append("missing RWWPO identity hash control flow")
    else:
        if "stable_identity_int64(value) for value in example_ids" not in example_hash or "for value in uids" in example_hash:
            violations.append("ephemeral UUID used as RWWPO audit identity")
    if 'gen_batch_output.non_tensor_batch.update(identity_columns)' not in trainer:
        violations.append("recurrent turn identity columns are not attached")
    if violations: raise SystemExit("TF_RWWPO_SOURCE_FIREWALL_NO_GO:"+",".join(violations))
    print(json.dumps({"status":"PASS","decision":"TF_RWWPO_SOURCE_FIREWALL_PASS"},sort_keys=True))

if __name__=="__main__": main()
