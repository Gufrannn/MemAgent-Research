#!/usr/bin/env python3
"""Cluster-weighted multi-write Shape A algebra audit; no p-values/claim authority."""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

REQUIRED=("stable_example_id","write_id","y_commit","y_retain","d_star")

def _wls(x,y,w):
    root=np.sqrt(w);return np.linalg.lstsq(x*root[:,None],y*root,rcond=None)[0]

def _prepare(rows):
    missing=[(i,key) for i,row in enumerate(rows) for key in REQUIRED if key not in row]
    if missing:raise ValueError(f"missing required fields including mandatory write_id: {missing[:5]}")
    keys=[(str(row["stable_example_id"]),str(row["write_id"])) for row in rows]
    if any(not a or not b for a,b in keys) or len(keys)!=len(set(keys)):
        raise ValueError("(stable_example_id, write_id) composite key must be present and unique")
    counts=Counter(a for a,_ in keys);n=len(counts)
    if not 2<=n<=128:raise ValueError(f"independent stable_example_id n must be in [2,128], got {n}")
    if any(row.get("row_level_hc3") is True or row.get("independent_unit") not in (None,"stable_example_id") for row in rows):
        raise ValueError("row-level independence/HC3 forbidden; cluster by stable_example_id")
    weights=np.asarray([1.0/counts[a] for a,_ in keys])
    if any("analysis_weight" in row and not np.isclose(float(row["analysis_weight"]),weight) for row,weight in zip(rows,weights)):
        raise ValueError("analysis_weight must equal 1/m_i so each stable example has total weight one")
    return keys,counts,weights,n

def aggregate_oof_loss(rows):
    keys,counts,weights,n=_prepare(rows)
    if any("oof_loss" not in row for row in rows):raise ValueError("oof_loss required")
    totals=defaultdict(float)
    for row,(stable,_),weight in zip(rows,keys,weights):totals[stable]+=weight*float(row["oof_loss"])
    return {"independent_n":n,"example_mean_oof_loss":sum(totals.values())/n,
      "aggregation":"within_example_1_over_m_then_across_examples"}

def audit(rows:list[dict])->dict:
    keys,counts,weights,n=_prepare(rows);q=len(rows)
    d=np.asarray([row["d_star"] for row in rows],dtype=float)
    yc=np.asarray([row["y_commit"] for row in rows],dtype=float)
    yr=np.asarray([row["y_retain"] for row in rows],dtype=float)
    if not np.isfinite(np.r_[d,yc,yr]).all():raise ValueError("non-finite audit value")
    paired_x=np.column_stack([np.ones(q),d])
    if np.linalg.matrix_rank(paired_x)!=2:raise ValueError("paired design matrix is rank deficient")
    paired=_wls(paired_x,yr-yc,weights)[1]
    # Stacked arm coding is retain=1, commit=0 with one fixed effect per write pair.
    y=np.r_[yc,yr];arm=np.r_[np.zeros(q),np.ones(q)];stacked_w=np.r_[weights,weights]
    pair_fe=np.vstack([np.eye(q),np.eye(q)])[:,1:]
    stacked_x=np.column_stack([np.ones(2*q),pair_fe,arm,arm*np.r_[d,d]])
    if np.linalg.matrix_rank(stacked_x)!=stacked_x.shape[1]:raise ValueError("write-pair FE stacked design matrix is rank deficient")
    stacked=_wls(stacked_x,y,stacked_w)[-1]
    if not np.isclose(paired,stacked,rtol=1e-9,atol=1e-10):
        raise ValueError(f"estimand implementation mismatch: paired={paired}, stacked={stacked}")
    totals=defaultdict(float)
    for (stable,_),weight in zip(keys,weights):totals[stable]+=weight
    if any(not np.isclose(value,1.0) for value in totals.values()):raise AssertionError("example weights do not sum to one")
    return {"audit_only":True,"claim_authorized":False,"p_values_emitted":False,
      "independent_n":n,"write_rows":q,"independent_unit":"stable_example_id",
      "row_key":["stable_example_id","write_id"],"outcome":"H_H=y_retain-y_commit",
      "weight":"1/m_i","example_total_weight":1.0,"paired_harm_slope":float(paired),
      "stacked_retain_arm_x_d":float(stacked),"stacked_is_second_evidence":False,
      "algebraically_equivalent":True,"outer_fold_cluster":"stable_example_id",
      "bootstrap_cluster":"stable_example_id","interval_cluster":"stable_example_id","row_level_hc3":False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,required=True);args=parser.parse_args()
    rows=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    print(json.dumps(audit(rows),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
