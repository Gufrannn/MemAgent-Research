#!/usr/bin/env python3
"""Read-only, preregistered COSI E1 residual audit."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from recurrent.research.cosi import canonical_sha256, root_contrasts, validate_four_cell_bundle

TRANSITIONS=("5-10","10-15","15-20","20-25")
def wilson_lower(k,n,z=1.6448536269514722):
    p=k/n; d=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d
def effect(values):
    mean=math.fsum(values)/len(values); var=math.fsum((x-mean)**2 for x in values)/max(1,len(values)-1)
    return abs(mean)/(math.sqrt(var) if var>0 else float("inf"))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--evidence",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    evidence=json.loads(Path(a.evidence).read_text())
    if set(evidence)!={"schema","preregistration","transitions","noise_replays","conditional_residual","evidence_sha256"}: raise ValueError("COSI_E1_NO_GO: fields")
    unsigned={k:v for k,v in evidence.items() if k!="evidence_sha256"}
    if evidence["schema"]!="memagent.cosi.e1-evidence.v1" or evidence["evidence_sha256"]!=canonical_sha256(unsigned): raise ValueError("COSI_E1_NO_GO: authentication")
    pre=evidence["preregistration"]
    expected={"roots_per_transition":32,"writer_replicas":2,"boundary":"middle_eligible_writer","noise_roots":8,"min_reversal_lcb":.10,"min_standardized_effect":.30,"noise_multiplier":5.0,"min_cv_mse_improvement":.10,"transitions":list(TRANSITIONS)}
    if pre!=expected or set(evidence["transitions"])!=set(TRANSITIONS): raise ValueError("COSI_E1_NO_GO: preregistration drift")
    noise=evidence["noise_replays"]
    if len(noise)!=8 or any(set(x)!={"root_id","max_abs_score_difference","all_hashes_equal"} for x in noise): raise ValueError("COSI_E1_NO_GO: noise inventory")
    if not all(x["all_hashes_equal"] for x in noise): raise ValueError("COSI_E1_NO_GO: replay hash nondeterminism")
    ceiling=max(float(x["max_abs_score_difference"]) for x in noise)
    summaries={}; passing_reversals=0; effect_pass=False
    for name in TRANSITIONS:
        bundle=validate_four_cell_bundle(evidence["transitions"][name]); rows=root_contrasts(bundle)
        if len(rows)!=32: raise ValueError("COSI_E1_NO_GO: exact roots required")
        rev=sum(float(r["writer_old"])>0 and float(r["closed"])<0 for r in rows); lcb=wilson_lower(rev,32)
        passing_reversals += lcb>.10
        cont=[float(r["continuation_old"]) for r in rows]; inter=[float(r["interaction"]) for r in rows]
        cont_mean=abs(math.fsum(cont)/32); inter_mean=abs(math.fsum(inter)/32)
        local_effect=max(effect(cont),effect(inter)); this_effect=local_effect>=.30 and max(cont_mean,inter_mean)>5*ceiling
        effect_pass |= this_effect
        summaries[name]={"reversal_count":rev,"reversal_wilson_lcb":lcb,"max_standardized_effect":local_effect,"effect_over_noise_pass":this_effect}
    residual=evidence["conditional_residual"]
    if set(residual)!={"protocol","target","features","folds","standardized_residual_effect","multiplicity_adjusted_p"} or residual["protocol"]!="leave_one_transition_out_fixed_ridge_residual_v1" or residual["target"]!="continuation_or_interaction_term" or residual["features"]!=["token_kl","sequence_kl","candidate_length","turn"] or residual["folds"]!=4: raise ValueError("COSI_E1_NO_GO: conditional residual protocol")
    # The target is C or I itself after conditioning on KL/length, never the
    # algebraic sum W+C+I. This prevents label-component leakage.
    conditional_pass=float(residual["standardized_residual_effect"])>=.30 and float(residual["multiplicity_adjusted_p"])<=.05
    passed=passing_reversals>=2 and effect_pass and conditional_pass
    report={"schema":"memagent.cosi.e1-report.v1","status":"PASS" if passed else "FAIL","decision":"COSI_E1_PASS" if passed else "COSI_E1_NO_GO_MERGE","noise_ceiling":ceiling,"transition_summaries":summaries,"transitions_with_reversal_lcb_pass":passing_reversals,"effect_pass":effect_pass,"conditional_residual_pass":conditional_pass}
    report["report_sha256"]=canonical_sha256(report)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f: json.dump(report,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps(report,sort_keys=True)); return 0 if passed else 2
if __name__=="__main__": raise SystemExit(main())
