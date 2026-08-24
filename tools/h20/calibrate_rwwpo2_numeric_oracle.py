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
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,ShardedStateDictConfig,ShardingStrategy,StateDictType,
)
from transformers import AutoModelForCausalLM

from verl.trainer.ppo.core_algos import (
    agg_loss, compute_policy_loss, compute_rwwpo_policy_loss, kl_penalty,
)
from recurrent.research.rwwpo_transaction import (
    RWWPO2_FSDP_PARAMETER_COMMIT_PRIMITIVE,
    RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS,
    local_gradient_sketch_sufficient_statistics,
    parameter_snapshot, set_interpolated_parameters, tensor_content_digest,
)
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.utils.torch_functional import logprobs_from_logits


MULTIPLIER=16.0
FLOORS={"tau_theta":1e-12,"tau_logprob":1e-6,"tau_gradient":1e-8,
        "tau_coefficient":1e-10}
GRADIENT_SKETCH_CHUNK_ELEMENTS=RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS
STREAMED_ORACLE_MICROBATCHES=7
STREAMED_ORACLE_SEQUENCE_LENGTH=8191
TRANSACTION_CLOSURE_SEQUENCE_LENGTH=8191
TRANSACTION_CLOSURE_ACTIVE_TOKENS=1024
TRANSACTION_WRITEBACK_MAX_WALL_SECONDS=120.0
STREAMED_REPLAY_CALIBRATION={
    "microbatches":STREAMED_ORACLE_MICROBATCHES,
    "sequence_length":STREAMED_ORACLE_SEQUENCE_LENGTH,
    "active_response_tokens":1024,
    "synthetic_label_free":True,
    "gradient_checkpointing":True,
    "gradient_checkpointing_use_reentrant":False,
    "remove_padding_flash_attention_patch":True,
    "fsdp_auto_wrap_policy":"default_transformer_no_split_modules",
    "fsdp_sharding_strategy":"FULL_SHARD",
    "fsdp_use_orig_params":False,
    "fsdp_sync_module_states":True,
    "fsdp_forward_prefetch":False,
    "model_load_dtype":"float32",
    "fsdp_sharded_parameter_dtype":"float32",
    "fsdp_param_dtype":"bfloat16",
    "fsdp_reduce_dtype":"float32",
    "fsdp_buffer_dtype":"float32",
    "cuda_autocast_dtype":"bfloat16",
    "selective_logprob_kernel":"verl.utils.torch_functional.logprobs_from_logits",
    "transaction_closure_probe":"unitwise_fp32_shard_to_bf16_forward_v1",
    "transaction_optimizer_probe":"adamw_fp32_shard_step_v1",
    "transaction_optimizer_lr":1e-6,
    "transaction_optimizer_betas":[0.9,0.999],
    "transaction_optimizer_weight_decay":0.01,
    "transaction_optimizer_grad_clip":1.0,
    "transaction_closure_sequence_length":TRANSACTION_CLOSURE_SEQUENCE_LENGTH,
    "transaction_closure_active_tokens":TRANSACTION_CLOSURE_ACTIVE_TOKENS,
    "transaction_writeback_max_wall_seconds":
        TRANSACTION_WRITEBACK_MAX_WALL_SECONDS,
    "fsdp_parameter_commit_primitive":RWWPO2_FSDP_PARAMETER_COMMIT_PRIMITIVE,
}


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
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        logits=model(input_ids=input_ids,use_cache=False).logits
        labels=input_ids[:,1:]
        selected=logprobs_from_logits(
            logits=logits[:,:-1],labels=labels,inplace_backward=True)
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
        position_ids=torch.arange(
            STREAMED_ORACLE_SEQUENCE_LENGTH,dtype=torch.long,
            device=input_ids.device).unsqueeze(0)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            logits=model(
                input_ids=input_ids,attention_mask=None,position_ids=position_ids,
                use_cache=False).logits.squeeze(0)
            labels=torch.roll(input_ids,shifts=-1,dims=1).squeeze(0)
            selected=logprobs_from_logits(
                logits=logits,labels=labels,inplace_backward=True).unsqueeze(0)
            active=min(1024,selected.shape[1]-1)
            coefficient=torch.zeros_like(selected)
            coordinate=torch.arange(active,device=selected.device)
            coefficient[:,-active-1:-1]=torch.where(
                torch.bitwise_and(coordinate,1)==0,1.0,-1.0
            ).to(coefficient.dtype).div_(active)
            streamed_loss=(selected*coefficient).sum()
        streamed_loss.backward()
        del input_ids,position_ids,labels,logits,selected,coefficient,coordinate,streamed_loss
    return gradient_sketch(model)


def behavior_actual_loss_gradient(model, input_ids, old_logp, objective):
    """Run complete C/E/B actual losses at one common behavior point."""
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, use_cache=False).logits
        labels = input_ids[:, 1:]
        selected = logprobs_from_logits(
            logits=logits[:, :-1],labels=labels,inplace_backward=True)
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


def transaction_forward_logprob(model, input_ids):
    """Label-free forward used by the pre-R50 transaction closure probe."""
    with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, use_cache=False).logits
        selected = logprobs_from_logits(
            logits=logits[:, :-1], labels=input_ids[:, 1:],
            inplace_backward=True)
    if selected.shape[1] < TRANSACTION_CLOSURE_ACTIVE_TOKENS:
        raise RuntimeError("RWWPO2_TRANSACTION_CLOSURE_SEQUENCE_TOO_SHORT")
    return selected[:, -TRANSACTION_CLOSURE_ACTIVE_TOKENS:].detach()


def transaction_backward_probe(model, input_ids):
    """Exercise one real long-context BF16/FSDP backward without updating."""
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, use_cache=False).logits
        selected = logprobs_from_logits(
            logits=logits[:, :-1], labels=input_ids[:, 1:],
            inplace_backward=True
        )[:, -TRANSACTION_CLOSURE_ACTIVE_TOKENS:]
        coordinate = torch.arange(
            TRANSACTION_CLOSURE_ACTIVE_TOKENS, device=selected.device)
        coefficient = torch.where(
            torch.bitwise_and(coordinate, 1) == 0, 1.0, -1.0
        ).to(selected.dtype).div_(TRANSACTION_CLOSURE_ACTIVE_TOKENS)
        loss = (selected * coefficient.unsqueeze(0)).sum()
    loss.backward()
    del logits, selected, coordinate, coefficient, loss


def fsdp_execution_inventory(model):
    """Aggregate-only diagnostics for FSDP's dynamic execution storage."""
    units = FSDP.fsdp_modules(model)
    inventory = {
        "unit_count": len(units), "managed_unit_count": 0,
        "training_states": {}, "storage": {},
    }
    for unit in units:
        state = str(getattr(unit, "training_state", "missing"))
        inventory["training_states"][state] = \
            inventory["training_states"].get(state, 0) + 1
        flat = getattr(unit, "_flat_param", None)
        if flat is None:
            continue
        inventory["managed_unit_count"] += 1
        for name, tensor in (
                ("flat_param_data", flat.data),
                ("local_shard", getattr(flat, "_local_shard", None)),
                ("mp_shard", getattr(flat, "_mp_shard", None)),
                ("full_param_padded", getattr(
                    flat, "_full_param_padded", None))):
            if not torch.is_tensor(tensor):
                continue
            key = f"{name}:{tensor.dtype}:{tensor.device.type}"
            row = inventory["storage"].setdefault(key, {
                "tensor_count": 0, "numel": 0, "allocated_bytes": 0,
                "nonzero_data_ptr_count": 0,
            })
            row["tensor_count"] += 1
            row["numel"] += int(tensor.numel())
            row["allocated_bytes"] += int(
                tensor.untyped_storage().nbytes())
            row["nonzero_data_ptr_count"] += int(tensor.data_ptr() != 0)
    return inventory


def transaction_closure_probe(model, input_ids, *, tau_logprob, device):
    """Prove the exact live FP32-shard -> BF16-forward commit primitive.

    The safe closure errors are compared with a threshold calibrated only from
    independent repeat/save-load/behavior probes.  The legacy raw-copy path is
    retained solely as a diagnostic and never contributes to the threshold.
    """
    model.zero_grad(set_to_none=True)
    if input_ids.shape != (1, TRANSACTION_CLOSURE_SEQUENCE_LENGTH):
        raise RuntimeError("RWWPO2_TRANSACTION_CLOSURE_SHAPE_DRIFT")

    def timed_safe_writeback(before, proposed, alpha):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        set_interpolated_parameters(
            model, before, proposed, alpha, synchronize_fsdp=True)
        torch.cuda.synchronize(device)
        elapsed = global_max(time.perf_counter() - started, device)
        if elapsed > TRANSACTION_WRITEBACK_MAX_WALL_SECONDS:
            raise RuntimeError("RWWPO2_FSDP_WRITEBACK_BUDGET_NO_GO")
        return elapsed

    behavior = transaction_forward_logprob(model, input_ids)
    behavior_cpu = behavior.cpu().clone()
    behavior_digest = tensor_content_digest(behavior_cpu)
    old = parameter_snapshot(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999),
        weight_decay=0.01)
    phases = [{
        "phase": "T0_behavior", "logprob_digest": behavior_digest,
        "execution_inventory": fsdp_execution_inventory(model),
    }]

    # T1 follows a real 8191-token backward without an optimizer step.  It
    # isolates FSDP execution-lifecycle drift from parameter-update rollback.
    transaction_backward_probe(model, input_ids)
    after_backward = transaction_forward_logprob(model, input_ids)
    after_backward_error = global_max(
        (after_backward - behavior).abs().max().item(), device)
    phases.append({
        "phase": "T1_after_backward", "max_abs": after_backward_error,
        "execution_inventory": fsdp_execution_inventory(model),
    })
    grad_norm = model.clip_grad_norm_(max_norm=1.0)
    if not torch.isfinite(grad_norm) or float(grad_norm.detach().item()) <= 0.0:
        raise RuntimeError("RWWPO2_TRANSACTION_OPTIMIZER_NONFINITE_NO_GO")
    optimizer.step()
    optimizer_proposed = parameter_snapshot(model)
    optimizer_proposal_max_abs = 0.0
    for before, after in zip(old, optimizer_proposed):
        left = before.view(-1)
        right = after.view(-1)
        for start in range(0, left.numel(), 1_048_576):
            stop = min(left.numel(), start + 1_048_576)
            optimizer_proposal_max_abs = max(
                optimizer_proposal_max_abs,
                float((right[start:stop] - left[start:stop]).abs().max().item()))
    optimizer_proposal_max_abs = global_max(
        optimizer_proposal_max_abs, device)
    optimizer_state_entry_count = len(optimizer.state)
    optimizer_state_entry_counts = [None] * dist.get_world_size()
    dist.all_gather_object(
        optimizer_state_entry_counts, optimizer_state_entry_count)
    optimizer_phase_inventory = fsdp_execution_inventory(model)
    if optimizer_proposal_max_abs <= 0.0 \
            or len(set(int(value) for value in
                       optimizer_state_entry_counts)) != 1 \
            or any(int(value) != int(
                optimizer_phase_inventory["managed_unit_count"])
                   for value in optimizer_state_entry_counts):
        raise RuntimeError("RWWPO2_TRANSACTION_OPTIMIZER_INACTIVE_NO_GO")
    phases.append({
        "phase": "T2_after_real_optimizer_step",
        "optimizer_proposal_max_abs": optimizer_proposal_max_abs,
        "execution_inventory": optimizer_phase_inventory,
    })

    # A deterministic synthetic proposal.  It is label/reward independent and
    # touches only a bounded prefix of each rank-local flat shard.
    proposed = []
    for parameter_index, optimizer_candidate in enumerate(
            optimizer_proposed):
        candidate = optimizer_candidate.clone()
        flat = candidate.view(-1)
        active = min(4096, flat.numel())
        if active:
            coordinate = torch.arange(active, dtype=torch.int64)
            direction = torch.where(
                torch.bitwise_and(coordinate + parameter_index, 1) == 0,
                1.0, -1.0).to(dtype=flat.dtype)
            flat[:active].add_(direction, alpha=2.0 ** -10)
        proposed.append(candidate)

    # Record, but never calibrate from, the legacy raw-copy failure mode.
    with torch.no_grad():
        for target, candidate in zip(model.parameters(), proposed):
            target.copy_(candidate.to(
                device=target.device, dtype=target.dtype))
    raw_candidate = transaction_forward_logprob(model, input_ids)
    with torch.no_grad():
        for target, before in zip(model.parameters(), old):
            target.copy_(before.to(device=target.device, dtype=target.dtype))
    raw_restored = transaction_forward_logprob(model, input_ids)
    raw_restore_error = global_max(
        (raw_restored - behavior).abs().max().item(), device)
    raw_candidate_activation = global_max(
        (raw_candidate - behavior).abs().max().item(), device)
    phases.append({
        "phase": "T3_legacy_raw_restore_diagnostic",
        "candidate_activation_max_abs": raw_candidate_activation,
        "restore_max_abs": raw_restore_error,
        "execution_inventory": fsdp_execution_inventory(model),
    })

    # Reconstruct the behavior point through the registered safe primitive.
    writeback_wall_seconds = {}
    writeback_wall_seconds["safe_behavior"] = timed_safe_writeback(
        old, old, 0.0)
    safe_behavior = transaction_forward_logprob(model, input_ids)
    safe_noop_error = global_max(
        (safe_behavior - behavior).abs().max().item(), device)
    phases.append({
        "phase": "T4_safe_behavior_writeback",
        "max_abs": safe_noop_error,
        "execution_inventory": fsdp_execution_inventory(model),
    })

    writeback_wall_seconds["safe_candidate"] = timed_safe_writeback(
        old, proposed, 1.0)
    safe_candidate = transaction_forward_logprob(model, input_ids)
    # Recommit the identical candidate so the certificate covers accepted as
    # well as rejected/alpha-zero terminal states.
    writeback_wall_seconds["safe_candidate_recommit"] = \
        timed_safe_writeback(old, proposed, 1.0)
    safe_candidate_repeat = transaction_forward_logprob(model, input_ids)
    safe_candidate_activation = global_max(
        (safe_candidate - safe_behavior).abs().max().item(), device)
    safe_candidate_repeat_error = global_max(
        (safe_candidate_repeat - safe_candidate).abs().max().item(), device)
    phases.append({
        "phase": "T5_safe_candidate_recommit",
        "candidate_activation_max_abs": safe_candidate_activation,
        "recommit_max_abs": safe_candidate_repeat_error,
        "execution_inventory": fsdp_execution_inventory(model),
    })

    writeback_wall_seconds["safe_restore"] = timed_safe_writeback(
        old, old, 0.0)
    safe_restored_first = transaction_forward_logprob(model, input_ids)
    safe_restored_second = transaction_forward_logprob(model, input_ids)
    safe_restore_error = global_max(
        (safe_restored_first - safe_behavior).abs().max().item(), device)
    safe_second_forward_error = global_max(
        (safe_restored_second - safe_restored_first).abs().max().item(),
        device)
    phases.append({
        "phase": "T6_safe_restore_fresh", "max_abs": safe_restore_error,
        "second_forward_max_abs": safe_second_forward_error,
        "execution_inventory": fsdp_execution_inventory(model),
    })

    safe_errors = {
        "after_backward_max_abs": after_backward_error,
        "safe_noop_writeback_max_abs": safe_noop_error,
        "safe_candidate_recommit_max_abs": safe_candidate_repeat_error,
        "safe_restore_max_abs": safe_restore_error,
        "safe_second_forward_max_abs": safe_second_forward_error,
    }
    if any(value > float(tau_logprob) for value in safe_errors.values()):
        raise RuntimeError("RWWPO2_FSDP_TRANSACTION_CLOSURE_NO_GO")
    if safe_candidate_activation <= float(tau_logprob):
        raise RuntimeError("RWWPO2_FSDP_TRANSACTION_PROPOSAL_INACTIVE_NO_GO")
    if tensor_content_digest(behavior_cpu) != behavior_digest \
            or tensor_content_digest(behavior) != behavior_digest:
        raise RuntimeError("RWWPO2_BEHAVIOR_REFERENCE_MUTATED_NO_GO")
    return {
        "rank": dist.get_rank(),
        "status": "PASS",
        "decision": "RWWPO2_FSDP_TRANSACTION_CLOSURE_PASS",
        "primitive": RWWPO2_FSDP_PARAMETER_COMMIT_PRIMITIVE,
        "behavior_logprob_digest": behavior_digest,
        "sequence_length": TRANSACTION_CLOSURE_SEQUENCE_LENGTH,
        "active_tokens": TRANSACTION_CLOSURE_ACTIVE_TOKENS,
        "tau_logprob": float(tau_logprob),
        "writeback_max_wall_seconds":
            TRANSACTION_WRITEBACK_MAX_WALL_SECONDS,
        "writeback_wall_seconds": writeback_wall_seconds,
        "safe_errors": safe_errors,
        "safe_candidate_activation_max_abs": safe_candidate_activation,
        "legacy_raw_copy_diagnostic": {
            "candidate_activation_max_abs": raw_candidate_activation,
            "restore_max_abs": raw_restore_error,
        },
        "optimizer_probe": {
            "kind": "AdamW", "lr": 1e-6,
            "betas": [0.9, 0.999], "weight_decay": 0.01,
            "grad_clip": 1.0, "step_calls": 1,
            "grad_norm": float(grad_norm.detach().item()),
            "proposal_max_abs": optimizer_proposal_max_abs,
            "state_entry_counts": [int(value)
                                   for value in optimizer_state_entry_counts],
        },
        "phases": phases,
    }


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
        args.model,local_files_only=True,torch_dtype=torch.float32,
        attn_implementation="flash_attention_2")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant":False})
    apply_monkey_patch(model=model,ulysses_sp_size=1)
    auto_wrap_policy=get_fsdp_wrap_policy(
        module=model,config={"min_num_params":0})
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,reduce_dtype=torch.float32,
        buffer_dtype=torch.float32)
    model=FSDP(
        model,device_id=device,sync_module_states=True,use_orig_params=False,
        auto_wrap_policy=auto_wrap_policy,mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,forward_prefetch=False)
    model.train()
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
    del before, after, state, loaded
    allreduce_probe=torch.arange(1,257,dtype=torch.float64,device=device)
    expected=allreduce_probe*world; dist.all_reduce(allreduce_probe)
    allreduce_error=global_max((allreduce_probe-expected).abs().max().item(),device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            behavior_logits=model(input_ids=tokens,use_cache=False).logits
            behavior_old_logp=logprobs_from_logits(
                logits=behavior_logits[:,:-1],labels=tokens[:,1:],
                inplace_backward=True).detach()
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
    transaction_tokens=(torch.arange(
        TRANSACTION_CLOSURE_SEQUENCE_LENGTH,dtype=torch.long,device=device
    ) + 104729*rank + 17).remainder_(vocab).unsqueeze(0)
    transaction_probe=transaction_closure_probe(
        model,transaction_tokens,tau_logprob=thresholds["tau_logprob"],
        device=device)
    transaction_probes=[None]*world
    dist.all_gather_object(transaction_probes,transaction_probe)
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
        report={"status":"PASS","decision":"RWWPO2_NUMERIC_ORACLE_PASS",
                "git_commit":head,"world_size":world,"model_path":os.path.realpath(args.model),
                "model_config_sha256":hashlib.sha256(
                    (Path(args.model)/"config.json").read_bytes()).hexdigest(),
                "gpu_pair":[int(value) for value in gpu_pair.split(",")],
                "gpu_binding":gpu_binding,
                "gradient_sketch_chunk_elements":GRADIENT_SKETCH_CHUNK_ELEMENTS,
                "streamed_replay_calibration":STREAMED_REPLAY_CALIBRATION,
                "fsdp_parameter_commit_primitive":
                    RWWPO2_FSDP_PARAMETER_COMMIT_PRIMITIVE,
                "fsdp_transaction_closure":transaction_probes,
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
