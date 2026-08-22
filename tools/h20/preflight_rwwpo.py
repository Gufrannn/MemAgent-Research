#!/usr/bin/env python3
import argparse, hashlib, json, os, re, subprocess
from pathlib import Path

def receipt(path, decision, commit):
    row=json.loads(Path(path).read_text()); declared=row.pop("report_sha256",None)
    actual=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return declared==actual and row.get("status")=="PASS" and row.get("decision")==decision and row.get("git_commit")==commit
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--gpu-pair",required=True); p.add_argument("--e0",required=True); p.add_argument("--e1"); p.add_argument("--baseline-import",required=True); p.add_argument("--original-resolved-manifest",required=True); p.add_argument("--original-resolved-sha256",required=True); p.add_argument("--phase",choices=["full","t5","continue"],required=True); p.add_argument("--resume-step",type=int); a=p.parse_args()
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(); branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip()
    pair=[int(x) for x in a.gpu_pair.split(",")]
    if len(pair)!=2 or pair!=sorted(set(pair)): raise SystemExit("RWWPO_NO_GO:GPU_PAIR must be two distinct canonical ascending IDs")
    if head!=a.expected_commit or not re.fullmatch(r"[0-9a-f]{40}",head): raise SystemExit("RWWPO_NO_GO:wrong_commit")
    if branch!="h20/qwen25-7b-rwwpo-t25-frozen-20260822": raise SystemExit("RWWPO_NO_GO:wrong_branch")
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip(): raise SystemExit("RWWPO_NO_GO:dirty_tree")
    if not receipt(a.e0,"RWWPO_E0_PASS",head): raise SystemExit("RWWPO_NO_GO:E0")
    if not receipt(a.baseline_import,"ORIGINAL_BASELINE_IMPORT_PASS",head): raise SystemExit("RWWPO_NO_GO:baseline_import")
    original_path=Path(a.original_resolved_manifest)
    if not original_path.is_file() or hashlib.sha256(original_path.read_bytes()).hexdigest()!=a.original_resolved_sha256:
        raise SystemExit("RWWPO_NO_GO:accepted Original resolved manifest SHA mismatch")
    method=json.loads(Path(a.manifest).read_text()); original=json.loads(original_path.read_text())
    expected=method["training"]
    source=original.get("training", original)
    aliases={"seed":"seed","train_batch_size":"train_batch_size","rollout_n":"rollout_n",
             "ppo_mini_batch_size":"ppo_mini_batch_size","chunk_size":"chunk_size","max_chunks":"max_chunks",
             "max_prompt_length":"max_prompt_length","max_response_length":"max_response_length",
             "learning_rate":"actor_learning_rate","kl_loss_coefficient":"kl_loss_coefficient"}
    drift={key:(expected[key],source.get(source_key)) for key,source_key in aliases.items() if source.get(source_key)!=expected[key]}
    if drift: raise SystemExit("RWWPO_NO_GO:Original protocol drift:"+json.dumps(drift,sort_keys=True))
    if a.phase=="continue" and a.resume_step not in (5,10,15,20): raise SystemExit("RWWPO_NO_GO:resume_step")
    diagnostic_e1=bool(a.e1 and Path(a.e1).is_file() and receipt(a.e1,"RWWPO_E1_PASS",head))
    print(json.dumps({"status":"PASS","decision":"RWWPO_PREFLIGHT_PASS","git_commit":head,"gpu_pair":pair,"phase":a.phase,"original_resolved_manifest_sha256":a.original_resolved_sha256,"optional_original_actual_loss_e1_pass":diagnostic_e1},sort_keys=True))
if __name__=="__main__": main()
