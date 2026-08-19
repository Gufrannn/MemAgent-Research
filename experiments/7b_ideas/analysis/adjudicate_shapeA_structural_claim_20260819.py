"""Schema-v8 mechanical Shape A claim downgrader; never authorizes training."""
from __future__ import annotations
CORE=("provenance_and_data_processing","semantic_measurement_reliability","estimand_and_inference_integrity","semantic_pairing_specificity")
VALID={"pass","fail","incomplete","not_run"}
DOWN={"provenance_and_data_processing":"ACCESS_MISMATCH_OR_UNKNOWN_PROVENANCE_NO_GO",
 "semantic_measurement_reliability":"MEASUREMENT_INCOMPLETE",
 "estimand_and_inference_integrity":"INVALID_INFERENCE_NO_GO",
 "semantic_pairing_specificity":"GENERIC_RELATIONAL_COMPRESSION_ONLY"}
def adjudicate(payload):
    if payload.get("schema_version")!="shapeA-structural-v8": raise ValueError("unsupported_schema_version")
    statuses={}
    for name in CORE:
        gate=payload.get("gates",{}).get(name,{}); status=gate.get("status")
        if status not in VALID: raise ValueError(f"invalid_or_missing_gate:{name}")
        if status in {"pass","fail"} and (not gate.get("evidence_hash") or gate.get("thresholds_frozen_before_outcome") is not True):
            raise ValueError(f"unfrozen_or_unhashed_gate:{name}")
        statuses[name]=status
    for name in CORE:
        if statuses[name]=="fail": return _result(DOWN[name],name,statuses)
        if statuses[name]!="pass": return _result("STRUCTURAL_CLAIM_INCOMPLETE",name,statuses)
    temporal=payload.get("gates",{}).get("temporal_specificity",{}).get("status","not_run")
    return {**_result("OBLIGATION_SEMANTIC_RELATIONAL_COMPRESSION", "four_core_gates_pass",statuses),
      "temporal_claim_scope":"same_role_update_local" if temporal=="pass" else "not_update_local_or_unresolved",
      "update_local_wording_authorized":temporal=="pass"}
def _result(claim,reason,statuses): return {"admissible_claim":claim,"reason":reason,"gate_statuses":statuses,
 "method_training_authorized":False,"online_deployment_claim_authorized":False}
def self_test():
    base={"schema_version":"shapeA-structural-v8","gates":{n:{"status":"pass","evidence_hash":"a"*64,"thresholds_frozen_before_outcome":True} for n in CORE}}
    assert adjudicate(base)["admissible_claim"]=="OBLIGATION_SEMANTIC_RELATIONAL_COMPRESSION"
    for name,claim in DOWN.items():
        x={"schema_version":base["schema_version"],"gates":{k:dict(v) for k,v in base["gates"].items()}}; x["gates"][name]["status"]="fail"
        assert adjudicate(x)["admissible_claim"]==claim
