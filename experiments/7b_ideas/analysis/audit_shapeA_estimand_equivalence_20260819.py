#!/usr/bin/env python3
"""Fail-closed, cluster-weighted multi-write Shape A audit."""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

LEDGER_REQUIRED=("stable_example_id","checkpoint_hash","write_id","prebranch_eligibility_manifest_hash",
 "target_write_count","measurement_available","measurement_exclusion_reason","eligible_write_count",
 "eligible","attempted","factual_commit_complete","noop_retain_complete","pair_qualified","failure_reason",
 "shapeA_stage","experiment_name","turn_type","required_component_pattern","required_components_complete",
 "joint_null_stratum_yield_sufficient")
CALIBRATION_REQUIRED=("calibration_stratum_key","dstar_calibration_manifest_hash","dstar_raw",
 "joint_null_median","joint_null_mad","d_star")
OUTCOME_REQUIRED=("y_commit","y_retain")
STAGES={"pilot4","audit32","B128"}

def _wls(x,y,w):
    root=np.sqrt(w);return np.linalg.lstsq(x*root[:,None],y*root,rcond=None)[0]

def _prepare(rows):
    missing=[(i,key) for i,row in enumerate(rows) for key in LEDGER_REQUIRED if key not in row]
    if missing:raise ValueError(f"missing required eligibility/attrition ledger fields: {missing[:5]}")
    keys=[(str(row["stable_example_id"]),str(row["checkpoint_hash"]),str(row["write_id"])) for row in rows]
    if any(not a or not checkpoint or not write for a,checkpoint,write in keys) or len(keys)!=len(set(keys)):
        raise ValueError("(stable_example_id, checkpoint_hash, write_id) composite key must be present and unique")
    hashes={row["prebranch_eligibility_manifest_hash"] for row in rows};experiments={row["experiment_name"] for row in rows}
    stages={row["shapeA_stage"] for row in rows}
    if len(hashes)!=1 or any(not value for value in hashes):raise ValueError("one frozen prebranch eligibility manifest hash is required")
    if len(experiments)!=1 or any(not value for value in experiments):raise ValueError("one unique experiment name is required per rerun")
    if len(stages)!=1 or not stages<=STAGES:raise ValueError(f"one valid Shape A stage required, got {stages}")
    target_counts=Counter(a for a,_,_ in keys);target_n=len(target_counts)
    eligible_counts=Counter(str(row["stable_example_id"]) for row in rows if row["measurement_available"] is True)
    paired_examples={stable for stable,count in eligible_counts.items() if count>0};n=len(paired_examples)
    if not 2<=n<=128:raise ValueError(f"measurable independent stable_example_id n must be in [2,128], got {n}")
    strata={}
    for row,(stable,_,_) in zip(rows,keys):
        if row["target_write_count"]!=target_counts[stable] or not isinstance(row["target_write_count"],int):
            raise ValueError("target_write_count must equal outcome-blind m_i^target for every row")
        if row["eligible_write_count"]!=eligible_counts[stable] or not isinstance(row["eligible_write_count"],int):
            raise ValueError("eligible_write_count must equal frozen measurable m_i^elig, never qualified count")
        available=row["measurement_available"] is True
        if bool(row["eligible"])!=available:raise ValueError("eligible must equal prebranch R=1 measurement availability")
        if available and (row["required_components_complete"] is not True or row["joint_null_stratum_yield_sufficient"] is not True):
            raise ValueError("R=1 eligible write requires complete components and sufficient own joint-null stratum")
        if row.get("row_level_hc3") is True or row.get("independent_unit") not in (None,"stable_example_id"):
            raise ValueError("row-level independence/HC3 forbidden; cluster by stable_example_id")
        if row.get("complete_case_primary") or row.get("ipw_primary") or row.get("imputation_primary"):
            raise ValueError("complete-case/IPW/imputation primary is forbidden")
        if not available:
            if row["attempted"] or row["factual_commit_complete"] or row["noop_retain_complete"] or row["pair_qualified"]:
                raise ValueError("R=0 measurement exclusion is prebranch and cannot be recorded as arm attempt/failure")
            if row["failure_reason"] not in (None,"") or not row["measurement_exclusion_reason"]:
                raise ValueError("R=0 needs measurement_exclusion_reason and no arm failure_reason")
            continue
        if row["measurement_exclusion_reason"] not in (None,""):
            raise ValueError("R=1 row cannot carry measurement_exclusion_reason")
        missing_calibration=[key for key in CALIBRATION_REQUIRED if key not in row]
        if missing_calibration:raise ValueError(f"R=1 row missing D_star calibration fields: {missing_calibration}")
        expected_stratum=f"{row['turn_type']}|{row['required_component_pattern']}"
        if row["calibration_stratum_key"]!=expected_stratum:
            raise ValueError("D_star calibration must use its own turn_type x required-component-pattern stratum")
        if not row["dstar_calibration_manifest_hash"]:raise ValueError("frozen D_star calibration manifest hash required")
        median=float(row["joint_null_median"]);mad=float(row["joint_null_mad"]);raw=float(row["dstar_raw"])
        if not np.isfinite([median,mad,raw]).all() or mad<=0:raise ValueError("joint-null median/MAD invalid")
        expected_d=float(np.clip((raw-median)/mad,-5,5))
        if not np.isclose(float(row["d_star"]),expected_d):
            raise ValueError("D_star must be continuous joint-null median/MAD calibration clipped to [-5,5]")
        signature=(median,mad,row["dstar_calibration_manifest_hash"])
        if expected_stratum in strata and strata[expected_stratum]!=signature:
            raise ValueError("joint-null scale changed within frozen stratum")
        strata[expected_stratum]=signature
    weights=np.asarray([0.0 if row["measurement_available"] is not True else 1.0/row["eligible_write_count"] for row in rows])
    target_weights=np.asarray([1.0/row["target_write_count"] for row in rows])
    if any("analysis_weight" in row and not np.isclose(float(row["analysis_weight"]),weight) for row,weight in zip(rows,weights)):
        raise ValueError("analysis_weight must remain fixed at 1/m_i^elig")
    qualified=[];ledger=[];exclusions=[];coverage=defaultdict(float);arm_mass={"factual_only":0.0,"noop_only":0.0,"neither":0.0}
    for index,(row,(stable,checkpoint,write),weight) in enumerate(zip(rows,keys,weights)):
        if row["measurement_available"] is not True:
            exclusions.append({"stable_example_id":stable,"checkpoint_hash":checkpoint,"write_id":write,
              "target_weight":target_weights[index],"measurement_available":False,
              "measurement_exclusion_reason":row["measurement_exclusion_reason"],"arm_failure":False})
            ledger.append({"stable_example_id":stable,"checkpoint_hash":checkpoint,"write_id":write,
              "target_write_count":row["target_write_count"],"eligible_write_count":row["eligible_write_count"],
              "target_weight":target_weights[index],"fixed_paired_weight":0.0,"measurement_available":False,
              "eligible":False,"attempted":False,"factual_commit_complete":False,"noop_retain_complete":False,
              "pair_qualified":False,"failure_reason":None})
            continue
        f=row["factual_commit_complete"] is True;n_complete=row["noop_retain_complete"] is True
        components=row["required_components_complete"] is True;yield_ok=row["joint_null_stratum_yield_sufficient"] is True
        pair=bool(row["pair_qualified"] and f and n_complete)
        if pair and row["attempted"] is not True:raise ValueError("pair-qualified write was not attempted")
        if pair and row["failure_reason"] not in (None,""):raise ValueError("complete pair cannot carry failure_reason")
        if not pair and not row["failure_reason"]:raise ValueError("incomplete/unqualified eligible write requires failure_reason")
        if pair:
            missing_outcome=[key for key in OUTCOME_REQUIRED if key not in row]
            if missing_outcome:raise ValueError(f"pair-qualified row={index} missing outcomes={missing_outcome}")
            qualified.append(index);coverage[stable]+=weight
        elif f and not n_complete:arm_mass["factual_only"]+=weight
        elif n_complete and not f:arm_mass["noop_only"]+=weight
        else:arm_mass["neither"]+=weight
        ledger.append({"stable_example_id":stable,"checkpoint_hash":checkpoint,"write_id":write,
          "target_write_count":row["target_write_count"],"eligible_write_count":row["eligible_write_count"],
          "target_weight":target_weights[index],"fixed_paired_weight":weight,"measurement_available":True,
          "eligible":True,"attempted":bool(row["attempted"]),
          "factual_commit_complete":f,"noop_retain_complete":n_complete,"pair_qualified":pair,
          "required_components_complete":components,"joint_null_stratum_yield_sufficient":yield_ok,
          "failure_reason":row["failure_reason"]})
    for stable in paired_examples:coverage.setdefault(stable,0.0)
    total_coverage=sum(coverage.values())/n;primary_ready=all(np.isclose(value,1.0) for value in coverage.values())
    measurement_by_example=defaultdict(float)
    for row,(stable,_,_),weight in zip(rows,keys,target_weights):
        if row["measurement_available"] is True:measurement_by_example[stable]+=weight
    measurement_rate=sum(measurement_by_example.values())/target_n
    return {"keys":keys,"target_counts":target_counts,"eligible_counts":eligible_counts,"weights":weights,
      "target_weights":target_weights,"n":n,"target_n":target_n,"qualified":qualified,"ledger":ledger,"exclusions":exclusions,
      "coverage":dict(coverage),"total_coverage":total_coverage,"primary_ready":primary_ready,
      "measurement_rate":measurement_rate,"measurement_by_example":dict(measurement_by_example),
      "full_closure_target_weight":float(n),"observed_missing_weight":float(n-sum(coverage.values())),
      "arm_asymmetry":arm_mass,"stage":next(iter(stages)),"manifest_hash":next(iter(hashes)),
      "experiment_name":next(iter(experiments))}

def _diagnostic(prepared):
    return {"status":"CONSTRUCTION_DIAGNOSTIC_ONLY","primary_authorized":False,
      "reason":"POST_BRANCH_MISSING_OR_UNQUALIFIED_WRITE","eligible_weight_coverage":prepared["total_coverage"],
      "measurement_availability_rate_R1_over_target":prepared["measurement_rate"],
      "measurement_availability_by_stable_example":prepared["measurement_by_example"],
      "measurement_R0_exclusion_ledger":prepared["exclusions"],
      "paired_closure_coverage_R1":prepared["total_coverage"],
      "full_closure_target_weight":prepared["full_closure_target_weight"],
      "observed_postbranch_missing_weight":prepared["observed_missing_weight"],
      "coverage_by_stable_example":prepared["coverage"],"arm_asymmetry_weight_mass":prepared["arm_asymmetry"],
      "eligibility_attrition_ledger":prepared["ledger"],"missing_weight_reallocated":False,
      "complete_case_primary":False,"ipw_primary":False,"imputation_primary":False,
      "rerun_requirement":"same_frozen_eligibility_manifest_new_unique_experiment_name_retain_old_ledger",
      "prebranch_eligibility_manifest_hash":prepared["manifest_hash"],"shapeA_stage":prepared["stage"]}

def aggregate_oof_loss(rows):
    prepared=_prepare(rows)
    if not prepared["primary_ready"]:return _diagnostic(prepared)
    indices=[i for i,row in enumerate(rows) if row["measurement_available"] is True]
    if any("oof_loss" not in rows[i] for i in indices):raise ValueError("oof_loss required for every R=1 write")
    totals=defaultdict(float)
    for i in indices:
        stable=prepared["keys"][i][0];totals[stable]+=prepared["weights"][i]*float(rows[i]["oof_loss"])
    return {"independent_n":prepared["n"],"example_mean_oof_loss":sum(totals.values())/prepared["n"],
      "aggregation":"within_example_1_over_m_elig_then_across_examples"}

def audit(rows:list[dict],*,require_primary=False)->dict:
    prepared=_prepare(rows)
    if not prepared["primary_ready"]:
        result=_diagnostic(prepared)
        if require_primary:raise ValueError(f"SHAPEA_PRIMARY_COVERAGE_FAIL: coverage={prepared['total_coverage']}")
        return result
    indices=[i for i,row in enumerate(rows) if row["measurement_available"] is True]
    selected=[rows[i] for i in indices];weights=prepared["weights"][indices];q=len(selected);n=prepared["n"]
    d=np.asarray([row["d_star"] for row in selected],dtype=float);yc=np.asarray([row["y_commit"] for row in selected],dtype=float)
    yr=np.asarray([row["y_retain"] for row in selected],dtype=float)
    if not np.isfinite(np.r_[d,yc,yr]).all():raise ValueError("non-finite audit value")
    paired_x=np.column_stack([np.ones(q),d])
    if np.linalg.matrix_rank(paired_x)!=2:raise ValueError("paired design matrix is rank deficient")
    paired=_wls(paired_x,yr-yc,weights)[1]
    y=np.r_[yc,yr];arm=np.r_[np.zeros(q),np.ones(q)];stacked_w=np.r_[weights,weights]
    pair_fe=np.vstack([np.eye(q),np.eye(q)])[:,1:]
    stacked_x=np.column_stack([np.ones(2*q),pair_fe,arm,arm*np.r_[d,d]])
    if np.linalg.matrix_rank(stacked_x)!=stacked_x.shape[1]:raise ValueError("write-pair FE stacked design matrix is rank deficient")
    stacked=_wls(stacked_x,y,stacked_w)[-1]
    if not np.isclose(paired,stacked,rtol=1e-9,atol=1e-10):raise ValueError(f"estimand implementation mismatch: {paired} vs {stacked}")
    return {"status":"SHAPEA_PRIMARY_COVERAGE_QUALIFIED","primary_authorized":False,"audit_only":True,
      "claim_authorized":False,"p_values_emitted":False,"independent_n":n,"write_rows":q,
      "row_key":["stable_example_id","checkpoint_hash","write_id"],"outcome":"H_H=y_retain-y_commit",
      "primary_higher_means":"more_harmful","dstar_higher_means":"worse",
      "tau_H_secondary_alias":"y_commit-y_retain=-H_H","tau_H_used_as_primary":False,
      "weight":"1/m_i^elig","eligible_weight_coverage":1.0,"missing_weight_reallocated":False,
      "measurement_availability_rate_R1_over_target":prepared["measurement_rate"],
      "measurement_availability_by_stable_example":prepared["measurement_by_example"],
      "measurement_R0_exclusion_ledger":prepared["exclusions"],"paired_closure_coverage_R1":1.0,
      "full_closure_target_weight":prepared["full_closure_target_weight"],"observed_postbranch_missing_weight":0.0,
      "paired_harm_slope":float(paired),"stacked_retain_arm_x_d":float(stacked),
      "stacked_is_second_evidence":False,"algebraically_equivalent":True,
      "outer_fold_cluster":"stable_example_id","bootstrap_cluster":"stable_example_id",
      "interval_cluster":"stable_example_id","row_level_hc3":False,
      "dstar_calibration":"frozen_turn_type_x_required_component_pattern_joint_null_median_mad_clip_-5_5",
      "dstar_truncate_at_zero":False,"q95_role":"secondary_anomaly_flag_only",
      "legacy_valid_n_96_rescue":False,
      "arm_asymmetry_weight_mass":prepared["arm_asymmetry"],"eligibility_attrition_ledger":prepared["ledger"],
      "prebranch_eligibility_manifest_hash":prepared["manifest_hash"],"shapeA_stage":prepared["stage"]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--require-primary",action="store_true");args=parser.parse_args()
    rows=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    print(json.dumps(audit(rows,require_primary=args.require_primary),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
