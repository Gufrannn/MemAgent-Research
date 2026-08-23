#!/usr/bin/env python3
"""Commit-bound CPU E0 for RWWPO-2 K1 degeneracy and K2 separation."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import torch

from recurrent.research.rwwpo_transaction import (
    logical_transaction_seed,
    proposal_clock,
    set_stateless_proposal_lr,
    stateless_proposal_lr,
)
from verl.trainer.ppo.core_algos import (
    agg_loss, compute_policy_loss, compute_rwwpo_policy_loss, kl_penalty,
)


def signed(report):
    raw=json.dumps(report,sort_keys=True,separators=(",",":"))
    return {**report,"report_sha256":hashlib.sha256(raw.encode()).hexdigest()}


def gradients(old,current,adv,response,writer,final,sample,turn):
    outputs={}
    for label,objective in (("E","per_write_joint"),("B","whole_prefix")):
        value=current.clone().requires_grad_(True)
        loss,_=compute_rwwpo_policy_loss(old,value,adv,response,writer,final,sample,turn,
            .2,.2,.2,writer_objective=objective)
        shared=agg_loss(kl_penalty(value,old-.2,"low_var_kl"),response,"token-mean")*.001
        outputs[label]=torch.autograd.grad(loss+shared,value)[0]
    value=current.clone().requires_grad_(True)
    loss,*_=compute_policy_loss(old,value,adv,response,.2,.2,.2,loss_agg_mode="token-mean")
    shared=agg_loss(kl_penalty(value,old-.2,"low_var_kl"),response,"token-mean")*.001
    outputs["C"]=torch.autograd.grad(loss+shared,value)[0]
    return outputs


def parameter_gradients(old, current, adv, response, writer, final, sample, turn):
    """Evaluate C/E/B on one common host and return full parameter gradients."""
    features = torch.tensor([
        [1.0, -0.5, 0.25, 0.0], [0.5, 1.0, -0.25, 0.75],
        [-0.5, 0.25, 1.0, -0.75], [0.75, -1.0, 0.5, 0.25],
        [0.25, 0.5, -0.75, 1.0],
    ], dtype=torch.float64)
    target = current.detach()
    outputs = {}
    for label, objective in (("C", "original_tokenwise"),
                             ("E", "per_write_joint"), ("B", "whole_prefix")):
        weight = torch.nn.Parameter(torch.zeros((4, 3), dtype=torch.float64))
        bias = torch.nn.Parameter(torch.zeros((3,), dtype=torch.float64))
        logits = old + features @ weight + bias
        # Move the common host to the requested log-probability point without
        # changing the Jacobian with respect to the toy parameters.
        logits = logits + (target - logits.detach())
        if label == "C":
            loss, *_ = compute_policy_loss(
                old, logits, adv, response, .2, .2, .2,
                loss_agg_mode="token-mean",
            )
        else:
            loss, _ = compute_rwwpo_policy_loss(
                old, logits, adv, response, writer, final, sample, turn,
                .2, .2, .2, writer_objective=objective,
            )
        loss = loss + agg_loss(
            kl_penalty(logits, old - .2, "low_var_kl"), response, "token-mean"
        ) * .001
        grads = torch.autograd.grad(loss, (weight, bias))
        outputs[label] = torch.cat([value.reshape(-1) for value in grads])
    return outputs


def transition_kernel_closure(schedule):
    """Exact K1 induction given one exact full gradient and complete state."""
    parameters = {
        label: torch.nn.Parameter(torch.tensor([0.25, -0.5, 0.75], dtype=torch.float64))
        for label in ("C", "E", "B")
    }
    optimizers = {
        label: torch.optim.AdamW([parameter], lr=0.0, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0, foreach=False)
        for label, parameter in parameters.items()
    }
    for optimizer in optimizers.values():
        optimizer.param_groups[0]["rwwpo2_accepted_optimizer_clock"] = 0
    exact_round_closure = []
    for round_id in range(1, 6):
        common_gradient = torch.tensor(
            [round_id / 10.0, -round_id / 20.0, round_id / 40.0],
            dtype=torch.float64,
        )
        proposal_id = proposal_clock(round_id, 1)
        for label in ("C", "E", "B"):
            optimizer = optimizers[label]
            optimizer.zero_grad(set_to_none=True)
            parameters[label].grad = common_gradient.clone()
            set_stateless_proposal_lr(
                optimizer, base_lr=schedule["base_lr"],
                warmup_proposals=schedule["warmup_proposals"],
                total_proposals=schedule["total_proposals"],
                proposal_id=proposal_id, kind=schedule["kind"],
            )
            optimizer.step()
            optimizer.param_groups[0]["rwwpo2_accepted_optimizer_clock"] += 1
        reference_parameter = parameters["C"].detach()
        parameter_equal = all(torch.equal(reference_parameter, parameters[label].detach())
                              for label in ("E", "B"))
        reference_state = optimizers["C"].state[parameters["C"]]
        state_equal = all(
            set(reference_state) == set(optimizers[label].state[parameters[label]])
            and all(
                torch.equal(reference_state[key], optimizers[label].state[parameters[label]][key])
                if torch.is_tensor(reference_state[key]) else
                reference_state[key] == optimizers[label].state[parameters[label]][key]
                for key in reference_state
            )
            and optimizers[label].param_groups[0]["lr"] == optimizers["C"].param_groups[0]["lr"]
            and optimizers[label].param_groups[0]["rwwpo2_accepted_optimizer_clock"] ==
                optimizers["C"].param_groups[0]["rwwpo2_accepted_optimizer_clock"]
            for label in ("E", "B")
        )
        exact_round_closure.append(parameter_equal and state_equal)
    return exact_round_closure


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    if head!=args.expected_commit or dirty:
        raise SystemExit("RWWPO2_E0_NO_GO:checkout")
    manifest_path=Path(args.manifest)
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("program")!="RWWPO-2" or manifest["training"].get("ppo_epochs")!=2:
        raise SystemExit("RWWPO2_E0_NO_GO:manifest K2")
    old=torch.zeros((5,3),dtype=torch.float64)
    response=torch.tensor([[1,1,0],[1,0,0],[1,1,1],[1,1,0],[1,1,1]],dtype=torch.bool)
    final=torch.tensor([0,0,0,1,1],dtype=torch.bool)
    writer=response & (~final).unsqueeze(-1)
    sample=torch.tensor([0,1,0,0,1]); turn=torch.tensor([0,0,1,2,1])
    scalar=torch.tensor([.7,-.4,.7,.7,-.4],dtype=torch.float64)
    advantage=scalar.unsqueeze(-1).expand_as(old)*response
    behavior=gradients(old,old,advantage,response,writer,final,sample,turn)
    behavior_max=max(float((behavior[left]-behavior[right]).abs().max())
                     for left,right in (("C","E"),("C","B"),("E","B")))
    displaced=old+writer*torch.tensor([[.1],[-.2],[.3],[0.],[0.]],dtype=old.dtype)
    off=gradients(old,displaced,advantage,response,writer,final,sample,turn)
    off_be=max(float((off["B"]-off["E"]).abs().max()),
               float((off["B"]-off["C"]).abs().max()))
    clocks=[proposal_clock(1,1),proposal_clock(1,2),proposal_clock(2,1),proposal_clock(400,2)]
    schedule=manifest["method"]["proposal_schedule"]
    lrs=[stateless_proposal_lr(base_lr=schedule["base_lr"],
        warmup_proposals=schedule["warmup_proposals"],total_proposals=schedule["total_proposals"],
        proposal_id=value,kind=schedule["kind"]) for value in clocks]
    seed=logical_transaction_seed(experiment_seed=2026,round_id=27,inner_id=2,
                                  rank=1,stream="actor_transaction")
    behavior_parameter=parameter_gradients(
        old,old,advantage,response,writer,final,sample,turn)
    behavior_parameter_max=max(float((behavior_parameter[left]-behavior_parameter[right]).abs().max())
        for left,right in (("C","E"),("C","B"),("E","B")))
    off_parameter=parameter_gradients(
        old,displaced,advantage,response,writer,final,sample,turn)
    off_parameter_max=max(float((off_parameter["B"]-off_parameter["E"]).abs().max()),
                            float((off_parameter["B"]-off_parameter["C"]).abs().max()))
    k1_round_closure=transition_kernel_closure(schedule)
    seed_replay=logical_transaction_seed(experiment_seed=2026,round_id=27,inner_id=2,
                                         rank=1,stream="actor_transaction")
    seed_other_stream=logical_transaction_seed(experiment_seed=2026,round_id=27,inner_id=2,
                                                rank=1,stream="trial_forward")
    status=(behavior_max<=1e-12 and behavior_parameter_max<=1e-12
            and off_be>1e-8 and off_parameter_max>1e-8
            and clocks==[1,2,3,800] and lrs==[5e-7,1e-6,1e-6,1e-6]
            and all(k1_round_closure) and seed==seed_replay and seed!=seed_other_stream)
    report=signed({"status":"PASS" if status else "FAIL",
        "decision":"RWWPO2_E0_PASS" if status else "RWWPO2_E0_NO_GO",
        "git_commit":head,"manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "behavior_point_max_full_logprob_gradient_delta":behavior_max,
        "behavior_point_max_full_parameter_gradient_delta":behavior_parameter_max,
        "off_behavior_max_gradient_separation":off_be,
        "off_behavior_max_parameter_gradient_separation":off_parameter_max,
        "proposal_clocks":clocks,"proposal_lrs":lrs,"logical_seed_probe":seed,
        "logical_seed_replay_equal":seed==seed_replay,
        "logical_seed_stream_separated":seed!=seed_other_stream,
        "k1_transition_kernel_round_closure":k1_round_closure,
        "k1_scope":"common-host full parameter gradients plus exact complete optimizer-state induction",
        "k2_scope":"off-behavior objective geometry"})
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(report,sort_keys=True,indent=2)+"\n")
    raise SystemExit(0 if status else 1)


if __name__=="__main__": main()
