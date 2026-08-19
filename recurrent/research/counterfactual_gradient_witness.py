"""CPU-safe W4 capture hooks; this module never calls backward or optimizer.step."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

FORBIDDEN_EVIDENCE_BASES={
    "gradient_difference_norm", "gradient_norm_only", "scalar_advantage_sign",
    "single_parameter_delta",
}


def vector_hash(values:list[float])->str:
    payload=json.dumps([float(x) for x in values],separators=(",",":"),allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def capture_w4_event(*,stable_id:str,group_id:str,candidate_hash:str,pair_key_hash:str,
                     checkpoint_hash:str,subspace_hash:str,y_commit:float,y_retain:float,
                     reader_coupling_id:str,
                     writer_token_score_gradients:list[list[float]],writer_token_mask:list[bool],
                     policy_controlled_token_kinds:list[str],
                     credit_writer_gradient:list[float],task_writer_gradient:list[float],
                     regularizer_writer_gradient:list[float],total_writer_gradient:list[float],
                     loss_graph_hash:str,actual_candidate_hash:str,
                     actual_group_id:str,actual_checkpoint_hash:str,actual_subspace_hash:str)->dict[str,Any]:
    """Create an immutable witness row from already-captured gradients."""
    if (len(writer_token_score_gradients)!=len(writer_token_mask) or
            len(writer_token_mask)!=len(policy_controlled_token_kinds) or not writer_token_score_gradients):
        raise ValueError("W4_NO_GO: writer token gradient/mask length mismatch")
    if "eos_or_stop" not in policy_controlled_token_kinds or not all(writer_token_mask):
        raise ValueError("W4_NO_GO: writer score mask must cover the complete sequence including EOS/stop")
    dimensions={len(row) for row in writer_token_score_gradients}
    components=(credit_writer_gradient,task_writer_gradient,regularizer_writer_gradient,total_writer_gradient)
    if len(dimensions)!=1 or not dimensions or any(len(vector)!=next(iter(dimensions)) for vector in components):
        raise ValueError("W4_NO_GO: gradient subspace dimensions mismatch")
    flattened=[float(x) for row in writer_token_score_gradients for x in row]
    values=flattened+[float(x) for vector in components for x in vector]+[float(y_commit),float(y_retain)]
    if not all(math.isfinite(x) for x in values):raise ValueError("W4_NO_GO: non-finite capture")
    if any(abs(float(total)-float(task)-float(reg))>1e-8 for total,task,reg in
           zip(total_writer_gradient,task_writer_gradient,regularizer_writer_gradient)):
        raise ValueError("W4_NO_GO: G_total != G_task + G_reg in capture subspace")
    return {
      "stable_id":stable_id,"group_id":group_id,"candidate_hash":candidate_hash,
      "exact_noop_pair_key_hash":pair_key_hash,"checkpoint_hash":checkpoint_hash,
      "subspace_hash":subspace_hash,"reader_coupling_id":reader_coupling_id,
      "y_commit":float(y_commit),"y_retain":float(y_retain),
      "writer_token_score_gradients":[[float(x) for x in row] for row in writer_token_score_gradients],
      "writer_token_mask":[bool(x) for x in writer_token_mask],
      "policy_controlled_token_kinds":[str(x) for x in policy_controlled_token_kinds],
      "writer_score_mask_includes_eos_or_stop":True,"writer_score_mask_complete":True,
      "credit_writer_gradient":[float(x) for x in credit_writer_gradient],
      "task_writer_gradient":[float(x) for x in task_writer_gradient],
      "regularizer_writer_gradient":[float(x) for x in regularizer_writer_gradient],
      "total_writer_gradient":[float(x) for x in total_writer_gradient],
      "loss_graph_hash":loss_graph_hash,
      "actual_candidate_hash":actual_candidate_hash,"actual_group_id":actual_group_id,
      "actual_checkpoint_hash":actual_checkpoint_hash,"actual_subspace_hash":actual_subspace_hash,
      "score_gradient_hash":vector_hash(flattened),
      "credit_gradient_hash":vector_hash(credit_writer_gradient),
      "task_gradient_hash":vector_hash(task_writer_gradient),
      "regularizer_gradient_hash":vector_hash(regularizer_writer_gradient),
      "total_gradient_hash":vector_hash(total_writer_gradient),
    }


def writer_score_gradient_sum(event:dict[str,Any])->list[float]:
    token_grads=event["writer_token_score_gradients"];mask=event["writer_token_mask"]
    dimension=len(token_grads[0]);summed=[0.0]*dimension
    for active,row in zip(mask,token_grads):
        if active:
            for index,value in enumerate(row):summed[index]+=float(value)
    return summed


def witness_vectors(event:dict[str,Any])->tuple[float,list[float],list[float],list[float],list[float],list[float]]:
    tau=float(event["y_commit"])-float(event["y_retain"])
    summed=writer_score_gradient_sum(event)
    return (tau,[tau*value for value in summed],
      [float(x) for x in event["credit_writer_gradient"]],
      [float(x) for x in event["task_writer_gradient"]],
      [float(x) for x in event["regularizer_writer_gradient"]],
      [float(x) for x in event["total_writer_gradient"]])
