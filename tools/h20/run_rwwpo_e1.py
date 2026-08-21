#!/usr/bin/env python3
"""Frozen Original actual-loss feasibility; never synthesizes missing evidence."""
import argparse, hashlib, json, math, subprocess, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.h20.audit_rwwpo_actual_loss import audit


def main():
    p=argparse.ArgumentParser(); p.add_argument("--original-ledger", action="append", required=True); p.add_argument("--output", required=True); p.add_argument("--expected-commit", required=True); a=p.parse_args()
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if head!=a.expected_commit: raise SystemExit("RWWPO_E1_NO_GO: commit mismatch")
    for path in a.original_ledger:
        if not Path(path).is_file(): raise SystemExit("PENDING_ACTUAL_LOSS_LEDGER: missing frozen Original actual-loss evidence")
    base=audit(a.original_ledger, require_method=True)
    if base["modes"] != ["original_collection"]:
        raise SystemExit("RWWPO_E1_NO_GO: evidence is not frozen Original collection mode")
    rows=[]
    for source in a.original_ledger:
        rows.extend(json.loads(line) for line in Path(source).read_text().splitlines() if line.strip())
    grouped={}
    for row in rows:
        grouped.setdefault((row["attempt_id"],row["global_step"],row["epoch"],row["minibatch"]),[]).append(row)
    observations=[]
    for identity,group in grouped.items():
        prefix=[item for row in group for item in row["prefix_rows"]]
        token_by_turn={}; write=[]
        for row in group:
            for old,cur,mask,turn in zip(row["old_log_prob"],row["current_log_prob"],row["writer_mask"],row["trajectory_turn"]):
                values=[c-o for o,c,m in zip(old,cur,mask) if bool(m)]
                if values: token_by_turn.setdefault(int(turn),[]).extend(values); write.append((int(turn),sum(values)))
        for stat in group[0]["prefix_stats"]:
            turn=stat["turn"]; selected=[item for item in prefix if item["turn"]==turn]
            write_values=[value for item_turn,value in write if item_turn==turn]
            token=token_by_turn.get(turn,[])
            if not selected or not write_values or not token: continue
            peak=max(write_values); weights=[math.exp(v-peak) for v in write_values]; total=sum(weights); weights=[v/total for v in weights]
            write_ess=1/(len(weights)*sum(v*v for v in weights))
            observations.append({"identity":identity,"turn":turn,"mean_prefix_length":sum(x["prefix_token_count"] for x in selected)/len(selected),
                "prefix_ess":stat["ess_fraction"],"write_ess":write_ess,
                "token_clipfrac":sum(abs(v)>math.log(1.2) for v in token)/len(token),
                "token_approx_kl":sum(-v for v in token)/len(token),
                "max_abs_token_log_ratio":max(map(abs,token)),
                "max_abs_prefix_log_ratio":max(abs(x["log_ratio"]) for x in selected)})
    collapse=base["min_prefix_ess"] < 0.95
    lengths={round(x["mean_prefix_length"],6) for x in observations}
    local_not_sufficient=any(x["prefix_ess"]<.95 and x["token_clipfrac"]==0 and abs(x["token_approx_kl"])<.02 for x in observations)
    per_write_not_sufficient=any(x["prefix_ess"]<.95 and x["write_ess"]>=.95 for x in observations)
    aperture=any(.01 < x["max_abs_prefix_log_ratio"] < 4.0 for x in observations)
    same_length_counterexample=any(a["mean_prefix_length"]==b["mean_prefix_length"] and abs(a["prefix_ess"]-b["prefix_ess"])>.02 for i,a in enumerate(observations) for b in observations[i+1:])
    loo_rmse=float("inf")
    if len(observations)>=8:
        errors=[]
        for held in range(len(observations)):
            train=[x for i,x in enumerate(observations) if i!=held]
            design=np.asarray([[1,x["mean_prefix_length"],x["token_approx_kl"],x["token_clipfrac"],x["write_ess"]] for x in train],dtype=float)
            target=np.asarray([x["prefix_ess"] for x in train],dtype=float)
            beta=np.linalg.lstsq(design,target,rcond=None)[0]
            x=observations[held]; prediction=np.asarray([1,x["mean_prefix_length"],x["token_approx_kl"],x["token_clipfrac"],x["write_ess"]])@beta
            errors.append((prediction-x["prefix_ess"])**2)
        loo_rmse=math.sqrt(sum(errors)/len(errors))
    length_not_proxy=same_length_counterexample and loo_rmse>.01
    status="PASS" if len(observations)>=8 and collapse and length_not_proxy and local_not_sufficient and per_write_not_sufficient and aperture else "FAIL"
    report={"status":status,"decision":"RWWPO_E1_PASS" if status=="PASS" else "RWWPO_E1_NO_GO",
            "git_commit":head,"source_ledgers":[{"path":str(Path(x).resolve()),"sha256":hashlib.sha256(Path(x).read_bytes()).hexdigest()} for x in a.original_ledger],
            "record_count":base["record_count"],"min_prefix_ess":base["min_prefix_ess"],"prefix_collapse_observed":collapse,
            "optimizer_turn_observations":len(observations),"distinct_prefix_lengths":len(lengths),"length_not_pure_proxy":length_not_proxy,
            "same_length_prefix_ess_counterexample":same_length_counterexample,"heldout_diagnostics_rmse":loo_rmse,"token_kl_computed":True,
            "token_clip_does_not_exclude_prefix_collapse":local_not_sufficient,
            "per_write_ess_does_not_exclude_prefix_collapse":per_write_not_sufficient,
            "nonzero_feasible_aperture":aperture}
    raw=json.dumps(report,sort_keys=True,separators=(",",":")); report["report_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    raise SystemExit(0 if status=="PASS" else 1)
if __name__=="__main__": main()
