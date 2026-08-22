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
    if violations: raise SystemExit("TF_RWWPO_SOURCE_FIREWALL_NO_GO:"+",".join(violations))
    print(json.dumps({"status":"PASS","decision":"TF_RWWPO_SOURCE_FIREWALL_PASS"},sort_keys=True))

if __name__=="__main__": main()
