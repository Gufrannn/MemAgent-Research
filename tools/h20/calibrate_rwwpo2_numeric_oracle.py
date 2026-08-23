#!/usr/bin/env python3
"""Two-rank BF16/FSDP no-op oracle for preregistered RWWPO-2 tolerances.

Run only via torchrun with exactly two already-locked H20s.  The threshold rule
and floors are code-frozen; the script does not read R50 or performance data.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedStateDictConfig,StateDictType
from transformers import AutoModelForCausalLM

from verl.trainer.ppo.core_algos import (
    agg_loss, compute_policy_loss, compute_rwwpo_policy_loss, kl_penalty,
)
from recurrent.research.rwwpo_transaction import (
    RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS,
    local_gradient_sketch_sufficient_statistics,
)


MULTIPLIER=16.0
FLOORS={"tau_theta":1e-12,"tau_logprob":1e-6,"tau_gradient":1e-8,
        "tau_coefficient":1e-10}
GRADIENT_SKETCH_CHUNK_ELEMENTS=RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS
STREAMED_ORACLE_MICROBATCHES=7
STREAMED_ORACLE_SEQUENCE_LENGTH=8191


def global_max(value,device):
    tensor=torch.tensor(float(value),dtype=torch.float64,device=device)
    dist.all_reduce(tensor,op=dist.ReduceOp.MAX)
    return float(tensor.item())


def gradient_sketch(model):
    values=local_gradient_sketch_sufficient_statistics(model.parameters())
    dist.all_reduce(values,op=dist.ReduceOp.SUM)
    values[0]=values[0].sqrt()
    return values


def projection_relative(left, right):
    """Match the registered B-vs-E projected-gradient separation statistic."""
    numerator=(left[1:]-right[1:]).norm()
    denominator=left[1:].norm().clamp_min(1e-30)
    return float((numerator/denominator).item())


def projection_relative_to_control(left, right, control):
    """Use the frozen C-host denominator for every C/E/B pair."""
    numerator=(left[1:]-right[1:]).norm()
    denominator=control[1:].norm().clamp_min(1e-30)
    return float((numerator/denominator).item())


def forward_and_backward(model,input_ids):
    model.zero_grad(set_to_none=True)
    logits=model(input_ids=input_ids,use_cache=False).logits
    labels=input_ids[:,1:]
    selected=torch.log_softmax(logits[:,:-1].float(),dim=-1).gather(
        -1,labels.unsqueeze(-1)).squeeze(-1)
    (-selected.mean()).backward()
    return selected.detach(),gradient_sketch(model)


def streamed_replay_gradient(model, *, vocab, rank):
    """Match the actor's maximum seven-section long-context replay shape.

    Inputs and coefficients are synthetic and label-free.  The probe measures
    only repeated BF16/FlashAttention/FSDP backward noise under the registered
    streaming accumulation pattern; it cannot observe R50 or task outcomes.
    """
    model.zero_grad(set_to_none=True)
    for microbatch_id in range(STREAMED_ORACLE_MICROBATCHES):
        input_ids=(torch.arange(
            STREAMED_ORACLE_SEQUENCE_LENGTH,dtype=torch.long,
            device=torch.cuda.current_device()
        ) + 1009*microbatch_id + 65537*rank).remainder_(vocab).unsqueeze(0)
        logits=model(input_ids=input_ids,use_cache=False).logits
        selected=torch.log_softmax(logits[:,:-1].float(),dim=-1).gather(
            -1,input_ids[:,1:].unsqueeze(-1)).squeeze(-1)
        active=min(1024,selected.shape[1])
        coefficient=torch.zeros_like(selected)
        coordinate=torch.arange(active,device=selected.device)
        coefficient[:,-active:]=torch.where(
            torch.bitwise_and(coordinate,1)==0,1.0,-1.0
        ).to(coefficient.dtype).div_(active)
        (selected*coefficient).sum().backward()
        del input_ids,logits,selected,coefficient,coordinate
    return gradient_sketch(model)


def behavior_actual_loss_gradient(model, input_ids, old_logp, objective):
    """Run complete C/E/B actual losses at one common behavior point."""
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=input_ids, use_cache=False).logits
    labels = input_ids[:, 1:]
    selected = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(
        -1, labels.unsqueeze(-1)).squeeze(-1)
    response_mask = torch.zeros_like(selected, dtype=torch.bool)
    response_mask[:, :2] = True
    final_mask = torch.tensor([False, False, False, True], device=selected.device)
    writer_mask = response_mask & (~final_mask).unsqueeze(-1)
    sample_index = torch.tensor([0, 0, 1, 1], device=selected.device)
    trajectory_turn = torch.tensor([0, 1, 0, 1], device=selected.device)
    advantages = torch.tensor(
        [[1.0], [1.0], [-1.0], [-1.0]], device=selected.device
    ).expand_as(selected)
    if objective == "C":
        loss, _, _, _ = compute_policy_loss(
            old_log_prob=old_logp, log_prob=selected,
            advantages=advantages, response_mask=response_mask,
            cliprange=0.2, cliprange_low=0.2, cliprange_high=0.2,
            clip_ratio_c=3.0, loss_agg_mode="token-mean",
        )
    else:
        loss, _ = compute_rwwpo_policy_loss(
            old_log_prob=old_logp, log_prob=selected,
            advantages=advantages, response_mask=response_mask,
            writer_mask=writer_mask, final_mask=final_mask,
            sample_index=sample_index, trajectory_turn=trajectory_turn,
            cliprange=0.2, cliprange_low=0.2, cliprange_high=0.2,
            clip_ratio_c=3.0, writer_log_ratio_cap=4.0,
            writer_objective="per_write_joint" if objective == "E" else "whole_prefix",
        )
    # Calibrate the complete actor loss.  The reference term is common to
    # C/E/B but has a non-zero derivative, so it must be present in the
    # coefficient and parameter-gradient noise measurement.
    reference = old_logp - 0.25
    shared_kl = agg_loss(
        kl_penalty(selected, reference, "low_var_kl"), response_mask, "token-mean"
    ) * 0.001
    loss = loss + shared_kl
    coefficient, = torch.autograd.grad(loss, selected, retain_graph=True)
    loss.backward()
    return selected.detach(), coefficient.detach(), gradient_sketch(model)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--output-root",required=True)
    args=parser.parse_args()
    if not os.path.isabs(args.model) or not os.path.isabs(args.output_root):
        raise SystemExit("RWWPO2_NUMERIC_ORACLE_NO_GO:absolute paths required")
    dist.init_process_group("nccl")
    rank=dist.get_rank(); world=dist.get_world_size()
    if world!=2: raise RuntimeError("RWWPO2_NUMERIC_ORACLE_REQUIRES_WORLD_SIZE_2")
    local_rank=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
    device=torch.device("cuda",local_rank)
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain"],text=True).strip()
    if head!=args.expected_commit or dirty:
        raise RuntimeError("RWWPO2_NUMERIC_ORACLE_CHECKOUT_DRIFT")
    root=Path(args.output_root)
    if rank==0:
        root.mkdir(parents=True,exist_ok=False)
        (root/"RUN_ID_CONSUMED").write_text(head+"\n")
        (root/"state").mkdir()
    dist.barrier()
    model=AutoModelForCausalLM.from_pretrained(
        args.model,local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to(device)
    model.eval()
    model=FSDP(model,device_id=device,sync_module_states=True)
    vocab=int(model.module.config.vocab_size)
    tokens=torch.tensor([
        [value%vocab for value in row]
        for row in (
            (1,17,23,91,7,41,3,11),
            (2,19,29,89,5,43,13,31),
            (3,11,37,83,17,47,23,53),
            (4,13,31,79,19,59,29,61),
        )
    ],dtype=torch.long,device=device)
    first_logp,first_gradient=forward_and_backward(model,tokens)
    second_logp,second_gradient=forward_and_backward(model,tokens)
    repeated_logp_error=global_max((first_logp-second_logp).abs().max().item(),device)
    repeated_gradient_error=global_max(
        projection_relative(first_gradient,second_gradient),device)
    torch.manual_seed(17011+rank); torch.cuda.manual_seed_all(17011+rank)
    first_streamed_gradient=streamed_replay_gradient(model,vocab=vocab,rank=rank)
    torch.manual_seed(17011+rank); torch.cuda.manual_seed_all(17011+rank)
    second_streamed_gradient=streamed_replay_gradient(model,vocab=vocab,rank=rank)
    streamed_replay_gradient_error=global_max(
        projection_relative(first_streamed_gradient,second_streamed_gradient),device)
    before=[parameter.detach().cpu().double().clone() for parameter in model.parameters()]
    config=ShardedStateDictConfig(offload_to_cpu=True)
    state_path=root/"state"/f"rank_{rank}.pt"
    with FSDP.state_dict_type(model,StateDictType.SHARDED_STATE_DICT,config):
        state=model.state_dict(); torch.save(state,state_path)
        state_sha=hashlib.sha256(state_path.read_bytes()).hexdigest()
        loaded=torch.load(state_path,map_location="cpu",weights_only=False)
        model.load_state_dict(loaded)
    after=[parameter.detach().cpu().double().clone() for parameter in model.parameters()]
    numerator=sum(float((right-left).square().sum()) for left,right in zip(before,after))
    denominator=sum(float(left.square().sum()) for left in before)
    values=torch.tensor([numerator,denominator],dtype=torch.float64,device=device)
    dist.all_reduce(values,op=dist.ReduceOp.SUM)
    save_load_parameter_error=float(values[0].sqrt().item()/max(values[1].sqrt().item(),1e-30))
    loaded_logp,loaded_gradient=forward_and_backward(model,tokens)
    save_load_logp_error=global_max((first_logp-loaded_logp).abs().max().item(),device)
    save_load_gradient_error=global_max(
        projection_relative(first_gradient,loaded_gradient),device)
    allreduce_probe=torch.arange(1,257,dtype=torch.float64,device=device)
    expected=allreduce_probe*world; dist.all_reduce(allreduce_probe)
    allreduce_error=global_max((allreduce_probe-expected).abs().max().item(),device)
    with torch.no_grad():
        behavior_logits=model(input_ids=tokens,use_cache=False).logits
        behavior_old_logp=torch.log_softmax(
            behavior_logits[:,:-1].float(),dim=-1).gather(
                -1,tokens[:,1:].unsqueeze(-1)).squeeze(-1).detach()
    actual_losses={}
    for objective in ("C","E","B"):
        actual_losses[objective]=behavior_actual_loss_gradient(
            model,tokens,behavior_old_logp,objective)
    behavior_logprob_error=global_max(max(
        float((value[0]-behavior_old_logp).abs().max().item())
        for value in actual_losses.values()),device)
    behavior_coefficient_error=global_max(max(
        float((actual_losses[left][1]-actual_losses[right][1]).abs().max().item())
        for left,right in (("C","E"),("C","B"),("B","E"))),device)
    behavior_gradient_error=global_max(max(
        projection_relative_to_control(
            actual_losses[left][2],actual_losses[right][2],actual_losses["C"][2])
        for left,right in (("C","E"),("C","B"),("B","E"))),device)
    local={"rank":rank,"relative_path":state_path.relative_to(root).as_posix(),
           "state_size":state_path.stat().st_size,"state_sha256":state_sha}
    gathered=[None]*world; dist.all_gather_object(gathered,local)
    if rank==0:
        gpu_pair=os.environ.get("GPU_PAIR","")
        if not gpu_pair or gpu_pair.split(",") != sorted(set(gpu_pair.split(",")),key=int):
            raise RuntimeError("RWWPO2_NUMERIC_ORACLE_GPU_PAIR_DRIFT")
        gpu_binding=subprocess.check_output([
            "nvidia-smi","-i",gpu_pair,"--query-gpu=index,uuid,name",
            "--format=csv,noheader"],text=True).strip().splitlines()
        observed={"repeated_logprob_max_abs":repeated_logp_error,
                  "repeated_gradient_projection_relative_l2":repeated_gradient_error,
                  "streamed_replay_gradient_projection_relative_l2":
                      streamed_replay_gradient_error,
                  "save_load_parameter_relative_l2":save_load_parameter_error,
                  "save_load_logprob_max_abs":save_load_logp_error,
                  "save_load_gradient_projection_relative_l2":save_load_gradient_error,
                  "behavior_actual_loss_logprob_max_abs":behavior_logprob_error,
                  "behavior_actual_loss_coefficient_max_abs":behavior_coefficient_error,
                  "behavior_actual_loss_gradient_projection_relative_l2":behavior_gradient_error,
                  "allreduce_max_abs":allreduce_error}
        thresholds={
            "tau_theta":max(FLOORS["tau_theta"],MULTIPLIER*save_load_parameter_error),
            "tau_logprob":max(FLOORS["tau_logprob"],MULTIPLIER*max(
                repeated_logp_error,save_load_logp_error,behavior_logprob_error)),
            "tau_gradient":max(FLOORS["tau_gradient"],MULTIPLIER*max(
                repeated_gradient_error,streamed_replay_gradient_error,
                save_load_gradient_error,behavior_gradient_error)),
            "tau_coefficient":max(FLOORS["tau_coefficient"],
                                  MULTIPLIER*behavior_coefficient_error),
        }
        report={"status":"PASS","decision":"RWWPO2_NUMERIC_ORACLE_PASS",
                "git_commit":head,"world_size":world,"model_path":os.path.realpath(args.model),
                "model_config_sha256":hashlib.sha256(
                    (Path(args.model)/"config.json").read_bytes()).hexdigest(),
                "gpu_pair":[int(value) for value in gpu_pair.split(",")],
                "gpu_binding":gpu_binding,
                "gradient_sketch_chunk_elements":GRADIENT_SKETCH_CHUNK_ELEMENTS,
                "streamed_replay_calibration":{
                    "microbatches":STREAMED_ORACLE_MICROBATCHES,
                    "sequence_length":STREAMED_ORACLE_SEQUENCE_LENGTH,
                    "active_response_tokens":1024,
                    "synthetic_label_free":True,
                },
                "threshold_multiplier":MULTIPLIER,"threshold_floors":FLOORS,
                "observed":observed,"thresholds":thresholds,"rank_state_evidence":gathered}
        raw=json.dumps(report,sort_keys=True,separators=(",",":"),allow_nan=False)
        report["report_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
        (root/"numeric_oracle.json").write_text(
            json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    dist.barrier()
    # Preserve the rank state artifacts. Their authenticated bytes are needed
    # for an independent audit; successful evidence must not self-delete.
    dist.destroy_process_group()


if __name__=="__main__": main()
