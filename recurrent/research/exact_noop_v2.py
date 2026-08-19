"""P/H/E firewall plus horizon-explicit same-candidate replay records."""
from __future__ import annotations
import hashlib,json,os
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any

BASE_PAIR_KEYS=("stable_id","turn","upstream_state_hash","candidate_memory_token_hash",
                "writer_seed","reader_seed_or_coupling_id","endpoint_version")
MODE_PAIR_KEYS={"E0":("immediate_probe_contract_hash",),
                "EH":("exogenous_suffix_contract_hash","future_policy_hash","horizon"),
                "EF":("forced_realized_suffix_hash","horizon")}
E_PAIR_KEYS=BASE_PAIR_KEYS  # compatibility export; validators append mode-specific keys.
E_ARM_KEYS=("arm","loaded_memory_token_hash","output_token_ids_hash","endpoint_value")
E_ARMS={"commit","retain"};ESTIMAND_MODES={"E0","EH","EF","VG"}
P_LABEL="P_SAME_ANCHOR_POLICY_RELATIVE_CREDIT";H_LABEL="H_PRIVILEGED_PROXY";E_LABEL="E_SAME_CANDIDATE_EXECUTION_EFFECT"

def token_hash(token_ids:list[int])->str:
    return hashlib.sha256(json.dumps([int(x) for x in token_ids],separators=(",",":")).encode()).hexdigest()
def classify_estimand(value:dict[str,Any])->str:
    fixed=bool(value.get("same_materialized_candidate"));endpoint=bool(value.get("shared_suffix_real_endpoint"))
    flags=(bool(value.get("same_anchor_policy_rerollout")),bool(value.get("teacher_forced_updated_vs_old_target_answerability")),
           bool(fixed and endpoint and value.get("commit_retain_only_intervention")))
    if sum(flags)!=1:raise ValueError("ESTIMAND_CLASSIFICATION_FAIL: P/H/E definitions are mutually exclusive")
    if flags[0] and not fixed:return P_LABEL
    if flags[1]:return H_LABEL
    if flags[2]:return E_LABEL
    raise ValueError("ESTIMAND_CLASSIFICATION_FAIL: unknown or mixed P/H/E object")

def materialize_proposal(*,stable_id:str,turn:int,upstream_state_hash:str,candidate_token_ids:list[int],
 writer_seed:int,reader_seed_or_coupling_id:str,endpoint_version:str,old_memory_token_hash:str,
 estimand_mode:str,mode_contract:dict[str,Any])->dict[str,Any]:
    if estimand_mode not in {"E0","EH","EF"}:raise ValueError("E_QUALIFICATION_FAIL: proposal builder supports E0/EH/EF only")
    if "suffix_hash" in mode_contract:raise ValueError("E_QUALIFICATION_FAIL: LEGACY_SUFFIX_SEMANTICS_AMBIGUOUS")
    required=MODE_PAIR_KEYS[estimand_mode];missing=[key for key in required if key not in mode_contract]
    if missing:raise ValueError(f"E_QUALIFICATION_FAIL: missing horizon contract {missing}")
    candidate_hash=token_hash(candidate_token_ids)
    if candidate_hash==old_memory_token_hash:raise ValueError("E_QUALIFICATION_FAIL: materialized candidate equals retained old memory")
    proposal={"stable_id":stable_id,"turn":int(turn),"upstream_state_hash":upstream_state_hash,
      "candidate_memory_token_hash":candidate_hash,"writer_seed":int(writer_seed),
      "reader_seed_or_coupling_id":reader_seed_or_coupling_id,"endpoint_version":endpoint_version,
      "old_memory_token_hash":old_memory_token_hash,"candidate_materialization_count":1,
      "writer_runs_after_materialization":0,"estimand_mode":estimand_mode}
    proposal.update({key:mode_contract[key] for key in required});return proposal

def build_arm_record(proposal:dict[str,Any],*,arm:str,output_token_ids:list[int],endpoint_value:float,
                     realized_trajectory_hash:str|None=None,future_hash_chain:list[str]|None=None)->dict[str,Any]:
    mode=proposal.get("estimand_mode")
    if arm not in E_ARMS:raise ValueError(f"E_QUALIFICATION_FAIL: invalid arm {arm}")
    if mode not in MODE_PAIR_KEYS:raise ValueError("E_QUALIFICATION_FAIL: missing/invalid estimand_mode")
    if proposal.get("candidate_materialization_count")!=1 or proposal.get("writer_runs_after_materialization")!=0:
        raise ValueError("E_QUALIFICATION_FAIL: candidate must materialize once and writer cannot rerun")
    keys=BASE_PAIR_KEYS+MODE_PAIR_KEYS[mode]
    record={key:proposal[key] for key in keys};record.update({"estimand_mode":mode,"arm":arm,
      "loaded_memory_token_hash":proposal["candidate_memory_token_hash"] if arm=="commit" else proposal["old_memory_token_hash"],
      "output_token_ids_hash":token_hash(output_token_ids),"endpoint_value":float(endpoint_value),
      "candidate_materialization_count":1,"writer_runs_after_materialization":0,"estimand":E_LABEL})
    if mode=="EH":
        if not realized_trajectory_hash or not future_hash_chain:raise ValueError("E_QUALIFICATION_FAIL: EH requires per-arm realized trajectory/hash chain")
        record.update({"realized_trajectory_hash":realized_trajectory_hash,"future_hash_chain":list(future_hash_chain),
          "endogenous_future_fields":["prompts","candidates","memory_states","decisions"],"allow_arm_trajectory_divergence":True})
    return record

def build_vg_record(*,stable_id:str,common_initial_state_hash:str,fixed_policy_hash:str,horizon:int,
                    endpoint_version:str,endpoint_value:float,realized_trajectory_hash:str,future_hash_chain:list[str])->dict[str,Any]:
    return {"estimand_mode":"VG","stable_id":stable_id,"common_initial_state_hash":common_initial_state_hash,
      "fixed_policy_hash":fixed_policy_hash,"horizon":int(horizon),"endpoint_version":endpoint_version,
      "endpoint_value":float(endpoint_value),"realized_trajectory_hash":realized_trajectory_hash,
      "future_hash_chain":list(future_hash_chain),"closed_loop_fixed_policy":True,
      "policy_generated_all_actions":True,"training_authorized":False}

@dataclass
class ExactNoopV2ReplayBuilder:
    proposal:dict[str,Any];_records:dict[str,dict[str,Any]]=field(default_factory=dict,init=False)
    def record_endpoint(self,*,arm:str,output_token_ids:list[int],endpoint_value:float,writer_ran:bool=False,
                        realized_trajectory_hash:str|None=None,future_hash_chain:list[str]|None=None)->dict[str,Any]:
        if writer_ran:raise ValueError("E_QUALIFICATION_FAIL: writer ran after candidate materialization")
        if arm in self._records:raise ValueError(f"E_QUALIFICATION_FAIL: duplicate arm={arm}; no last-write-wins")
        record=build_arm_record(self.proposal,arm=arm,output_token_ids=output_token_ids,endpoint_value=endpoint_value,
          realized_trajectory_hash=realized_trajectory_hash,future_hash_chain=future_hash_chain)
        self._records[arm]=record;return dict(record)
    def complete_pair(self)->list[dict[str,Any]]:
        if set(self._records)!=E_ARMS:raise ValueError(f"E_QUALIFICATION_FAIL: incomplete arms={sorted(self._records)}")
        return [dict(self._records[arm]) for arm in ("commit","retain")]

def append_record(path:str|Path,record:dict[str,Any])->None:
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(target,os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o644)
    try:os.write(fd,(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n").encode())
    finally:os.close(fd)
