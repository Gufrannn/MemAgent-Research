"""P/H/E estimand firewall and exact same-candidate commit/retain replay records.

The E builder is deliberately a small state machine: a proposal is materialized
once, each execution arm is emitted once, and neither arm may invoke the writer.
This is the record-layer contract used by the manifest preflight.
"""
from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

E_PAIR_KEYS=("stable_id","turn","upstream_state_hash","candidate_memory_token_hash","suffix_hash","writer_seed","reader_seed_or_coupling_id","endpoint_version")
E_ARM_KEYS=("arm","loaded_memory_token_hash","output_token_ids_hash","endpoint_value")
E_ARMS={"commit","retain"}
P_LABEL="P_SAME_ANCHOR_POLICY_RELATIVE_CREDIT"
H_LABEL="H_PRIVILEGED_PROXY"
E_LABEL="E_SAME_CANDIDATE_EXECUTION_EFFECT"

def token_hash(token_ids:list[int])->str:
    return hashlib.sha256(json.dumps([int(x) for x in token_ids],separators=(",",":")).encode()).hexdigest()
def classify_estimand(value:dict[str,Any])->str:
    fixed_candidate=bool(value.get("same_materialized_candidate")); real_endpoint=bool(value.get("shared_suffix_real_endpoint"))
    flags=(bool(value.get("same_anchor_policy_rerollout")),
           bool(value.get("teacher_forced_updated_vs_old_target_answerability")),
           bool(fixed_candidate and real_endpoint and value.get("commit_retain_only_intervention")))
    if sum(flags)!=1:
        raise ValueError("ESTIMAND_CLASSIFICATION_FAIL: P/H/E definitions are mutually exclusive")
    if flags[0] and not fixed_candidate: return P_LABEL
    if flags[1]: return H_LABEL
    if flags[2]: return E_LABEL
    raise ValueError("ESTIMAND_CLASSIFICATION_FAIL: unknown or mixed P/H/E object")

def materialize_proposal(*,stable_id:str,turn:int,upstream_state_hash:str,candidate_token_ids:list[int],suffix_hash:str,
 writer_seed:int,reader_seed_or_coupling_id:str,endpoint_version:str,old_memory_token_hash:str)->dict[str,Any]:
    candidate_hash=token_hash(candidate_token_ids)
    if candidate_hash==old_memory_token_hash:
        raise ValueError("E_QUALIFICATION_FAIL: materialized candidate equals retained old memory")
    return {"stable_id":stable_id,"turn":int(turn),"upstream_state_hash":upstream_state_hash,
      "candidate_memory_token_hash":candidate_hash,"suffix_hash":suffix_hash,"writer_seed":int(writer_seed),
      "reader_seed_or_coupling_id":reader_seed_or_coupling_id,"endpoint_version":endpoint_version,
      "old_memory_token_hash":old_memory_token_hash,"candidate_materialization_count":1,"writer_runs_after_materialization":0}
def build_arm_record(proposal:dict[str,Any],*,arm:str,output_token_ids:list[int],endpoint_value:float)->dict[str,Any]:
    if arm not in E_ARMS: raise ValueError(f"E_QUALIFICATION_FAIL: invalid arm {arm}")
    if proposal.get("candidate_materialization_count")!=1 or proposal.get("writer_runs_after_materialization")!=0:
        raise ValueError("E_QUALIFICATION_FAIL: candidate must materialize once and writer cannot rerun in either arm")
    record={key:proposal[key] for key in E_PAIR_KEYS}; record.update({"arm":arm,
      "loaded_memory_token_hash":proposal["candidate_memory_token_hash"] if arm=="commit" else proposal["old_memory_token_hash"],
      "output_token_ids_hash":token_hash(output_token_ids),"endpoint_value":float(endpoint_value),
      "candidate_materialization_count":1,"writer_runs_after_materialization":0,
      "estimand":E_LABEL})
    return record

@dataclass
class ExactNoopV2ReplayBuilder:
    """One-proposal/two-arm constructor; duplicate arms are never overwritten."""
    proposal: dict[str,Any]
    _records: dict[str,dict[str,Any]]=field(default_factory=dict,init=False)

    def record_endpoint(self,*,arm:str,output_token_ids:list[int],endpoint_value:float,writer_ran:bool=False)->dict[str,Any]:
        if writer_ran:
            raise ValueError("E_QUALIFICATION_FAIL: writer ran after candidate materialization")
        if arm in self._records:
            raise ValueError(f"E_QUALIFICATION_FAIL: duplicate arm={arm}; no last-write-wins")
        record=build_arm_record(self.proposal,arm=arm,output_token_ids=output_token_ids,endpoint_value=endpoint_value)
        self._records[arm]=record
        return dict(record)

    def complete_pair(self)->list[dict[str,Any]]:
        if set(self._records)!=E_ARMS:
            raise ValueError(f"E_QUALIFICATION_FAIL: incomplete arms={sorted(self._records)}")
        return [dict(self._records[arm]) for arm in ("commit","retain")]

def append_record(path:str|Path,record:dict[str,Any])->None:
    """Append exactly one immutable replay record; validation rejects duplicates."""
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(target,os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o644)
    try:os.write(fd,(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n").encode())
    finally:os.close(fd)
