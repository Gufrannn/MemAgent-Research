"""Outcome-free D* semantic-replica reliability gate."""
from __future__ import annotations
import numpy as np

def _ranks(x):
    order=np.argsort(x, kind="stable"); ranks=np.empty(len(x), float); ranks[order]=np.arange(len(x)); return ranks
def _spearman(a,b): return float(np.corrcoef(_ranks(a), _ranks(b))[0,1])
def _icc(a,b):
    x=np.column_stack([a,b]); n,k=x.shape; gm=x.mean(); row=x.mean(1); col=x.mean(0)
    msr=k*np.sum((row-gm)**2)/(n-1); msc=n*np.sum((col-gm)**2)/(k-1)
    mse=(np.sum((x-row[:,None]-col[None,:]+gm)**2))/((n-1)*(k-1))
    return float((msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n))
def _lower(a,b,fn,seed=2026,reps=400):
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(reps):
        idx=rng.integers(0,len(a),len(a)); value=fn(a[idx],b[idx])
        if np.isfinite(value): vals.append(value)
    return float(np.quantile(vals,.025)) if vals else float("nan")

def audit(rows:list[dict], *, independent_semantic_replicas:bool, isomorphic_replica_contract:bool,
          deterministic_rerun:bool=False)->dict:
    base={"theory_no_go_authorized":False,"training_authorized":False,"deployment_authorized":False,
          "stacked_arm_x_d_second_primary":False,"still_requires":["P2_marginal","semantic_pairing_null"]}
    if deterministic_rerun or not independent_semantic_replicas or not isomorphic_replica_contract:
        return {**base,"status":"MEASUREMENT_NOT_IDENTIFIED"}
    ids=[str(r["stable_example_id"]) for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError("reader repeats/duplicate stable IDs cannot increase independent n")
    valid=np.array([bool(r["valid_a"]) and bool(r["valid_b"]) for r in rows]); selected=[r for r,v in zip(rows,valid) if v]
    if len(selected)<24: return {**base,"status":"MEASUREMENT_INCOMPLETE","valid_independent_n":len(selected)}
    a=np.array([r["D_a"] for r in selected],float); b=np.array([r["D_b"] for r in selected],float)
    sp=_spearman(a,b); sp_lo=_lower(a,b,_spearman); icc=_icc(a,b); icc_lo=_lower(a,b,_icc)
    agree=float(np.mean([bool(r["valid_a"])==bool(r["valid_b"]) for r in rows])); strata={}
    for key in dict.fromkeys((str(r["role"]),str(r["checkpoint"])) for r in rows):
        group=[r for r in rows if (str(r["role"]),str(r["checkpoint"]))==key]
        strata["|".join(key)]=float(np.mean([bool(r["valid_a"])==bool(r["valid_b"]) for r in group]))
    qa=np.quantile(a,.75); qb=np.quantile(b,.75); sa=set(np.where(a>=qa)[0]); sb=set(np.where(b>=qb)[0]); jac=len(sa&sb)/len(sa|sb)
    passed=len(selected)>=24 and sp>=.60 and sp_lo>=.30 and icc>=.50 and icc_lo>0 and agree>=.90 and min(strata.values())>=.80 and jac>=.40
    return {**base,"status":"MEASUREMENT_RELIABLE" if passed else "MEASUREMENT_INCOMPLETE","valid_independent_n":len(selected),
      "spearman":sp,"spearman_bootstrap95_lower":sp_lo,"icc_A1":icc,"icc_bootstrap95_lower":icc_lo,
      "validity_agreement":agree,"role_checkpoint_agreement":strata,"top_quartile_jaccard":jac}
