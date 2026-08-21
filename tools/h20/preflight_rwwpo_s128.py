#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from recurrent.research.stable_eval_identity import stable_eval_runtime_config_sha256,validate_resolved_manifest
from tools.h20.preflight_qwen25_7b_stable_i4x2 import compose_resolved_trainer_config,sha256_file

def overrides(a, expected):
    common=json.loads((ROOT/"manifests/h20/qwen25_7b_s128_it_commands.json").read_text())["common_trainer_overrides"]
    replacements={"${VALIDATION_PATH}":a.validation,"${MODEL_PATH}":a.model,"${REPO_DIR}":str(ROOT),"${EXPERIMENT_NAME}":f"qwen25_7b_rwwpo_s128_t{a.step}_{a.run_id}","${INTERFACE_ID}":f"RWWPO{a.step}","${ATTEMPT_ID}":f"rwwpo_t{a.step}_{a.run_id}","${INTERFACE_ROOT}":a.output,"${TERMINAL_DIR}":str(Path(a.output)/"terminal"),"${RESOLVED_MANIFEST_PATH}":a.resolved_manifest,"${EVAL_MANIFEST_HASH}":a.eval_hash,"${TURN_LEDGER_PATH}":str(Path(a.output)/"trajectory_turns.jsonl"),"${EXECUTION_SUMMARY_PATH}":str(Path(a.output)/"execution_summary.json"),"${EXPECTED_RUNTIME_CONFIG_SHA256}":expected,"${T25_CHECKPOINT}":a.checkpoint}
    result=[]
    for item in common:
        for key,value in replacements.items(): item=item.replace(key,value)
        result.append(item)
    result += ["trainer.resume_mode=actor_only_eval",f"trainer.resume_from_path={a.checkpoint}","+trainer.eval_identity.weight_source=actor_checkpoint",f"+trainer.eval_identity.expected_global_step={a.step}"]
    return result
def main():
    p=argparse.ArgumentParser()
    for name in ("checkpoint","validation","model","resolved-manifest","resolved-manifest-sha256","eval-hash","run-id","output"): p.add_argument("--"+name,required=True)
    p.add_argument("--step",type=int,choices=[5,10,15,20,25],required=True); a=p.parse_args()
    if sha256_file(Path(a.resolved_manifest))!=a.resolved_manifest_sha256: raise SystemExit("RWWPO_S128_NO_GO:resolved manifest SHA")
    resolved=validate_resolved_manifest(json.loads(Path(a.resolved_manifest).read_text()))
    if resolved["eval_manifest_hash"]!=a.eval_hash: raise SystemExit("RWWPO_S128_NO_GO:eval hash")
    if Path(a.checkpoint).name!=f"global_step_{a.step}": raise SystemExit("RWWPO_S128_NO_GO:checkpoint step")
    provisional=overrides(a,"0"*64); digest=stable_eval_runtime_config_sha256(compose_resolved_trainer_config(ROOT,provisional))
    final=overrides(a,digest)
    if stable_eval_runtime_config_sha256(compose_resolved_trainer_config(ROOT,final))!=digest: raise SystemExit("RWWPO_S128_NO_GO:unstable config hash")
    print("\n".join(final))
if __name__=="__main__": main()
