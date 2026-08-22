#!/usr/bin/env python3
"""Two-rank CUDA/FSDP oracle for CORAL's distributed gradient sketch."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.coral_e1 import SKETCH_BASIS_SHA256
from recurrent.research.cosi import canonical_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from recurrent.research.coral_e1 import (
        SKETCH_SPEC, fixed_count_sketch, sketch_bucket_and_sign,
    )

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2 or not torch.cuda.is_available():
        raise RuntimeError("CORAL_E1_NO_GO: oracle requires exact two-rank CUDA")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(20260822)
    weight = torch.linspace(-0.2, 0.3, 13 * 17, dtype=torch.float32).reshape(13, 17)
    module = torch.nn.Linear(17, 13, bias=False, device=device)
    with torch.no_grad():
        module.weight.copy_(weight.to(device))
    fsdp = FSDP(module, device_id=device, use_orig_params=False)
    local_input = (
        torch.arange(4 * 17, dtype=torch.float32, device=device).reshape(4, 17)
        / 100 + rank / 10
    )
    loss = fsdp(local_input).square().mean()
    loss.backward()
    actual_sketch, _ = fixed_count_sketch(fsdp.named_parameters(), rank, world)
    dist.all_reduce(actual_sketch, op=dist.ReduceOp.SUM)
    local_gradient = next(fsdp.parameters()).grad.detach().reshape(-1).float()
    shard_size = torch.tensor([local_gradient.numel()], device=device, dtype=torch.int64)
    gathered_sizes = [torch.zeros_like(shard_size) for _ in range(world)]
    dist.all_gather(gathered_sizes, shard_size)
    if len({int(item.item()) for item in gathered_sizes}) != 1:
        raise RuntimeError("CORAL_E1_NO_GO: unexpected uneven tiny-FSDP shards")
    gathered = [torch.empty_like(local_gradient) for _ in range(world)]
    dist.all_gather(gathered, local_gradient)

    report = None
    denominator_model = torch.nn.Linear(3, 1, bias=False, device=device)
    with torch.no_grad():
        denominator_model.weight.copy_(
            torch.tensor([[0.1, -0.2, 0.3]], device=device)
        )
    denominator_fsdp = FSDP(
        denominator_model, device_id=device, use_orig_params=False,
    )
    features = (
        torch.arange(2 * 4 * 3, device=device, dtype=torch.float32)
        .reshape(2, 4, 3) / 17 + rank / 13
    )
    advantages = torch.tensor(
        [[1.0, -0.5, 0.25, 0.75], [-0.25, 0.5, 1.0, -1.0]],
        device=device,
    )
    response_mask = (
        torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], device=device)
        if rank == 0 else
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], device=device)
    ).float()
    final_mask = torch.tensor([False, True], device=device)
    full_denominator = response_mask.sum()

    def denominator_gradient(role):
        denominator_fsdp.zero_grad(set_to_none=True)
        logits = denominator_fsdp(features).squeeze(-1)
        if role == "full":
            mask = response_mask
        elif role == "writer":
            mask = response_mask * (~final_mask).float().unsqueeze(-1)
        elif role == "answer":
            mask = response_mask * final_mask.float().unsqueeze(-1)
        else:
            raise AssertionError(role)
        # This is the actor's token-mean numerator with the frozen pre-mask
        # local full-response denominator. Splitting active rows into dynamic
        # microbatches is algebraically the same sum.
        loss_value = -(logits * advantages * mask).sum() / full_denominator
        loss_value.backward()
        shard = next(denominator_fsdp.parameters()).grad.detach().reshape(-1)
        shards = [torch.empty_like(shard) for _ in range(world)]
        dist.all_gather(shards, shard)
        return torch.cat(shards)

    full_denominator_gradient = denominator_gradient("full")
    writer_denominator_gradient = denominator_gradient("writer")
    answer_denominator_gradient = denominator_gradient("answer")
    denominator_closure_error = float((
        full_denominator_gradient
        - writer_denominator_gradient
        - answer_denominator_gradient
    ).abs().max().item())

    # Collision-heavy projection calibration.  The earlier tiny FSDP oracle
    # has only 221 real gradient elements for 256 buckets per basis and mainly
    # validates shard assembly.  This deterministic vector has far more
    # coordinates than buckets and therefore exercises the actual CountSketch
    # collision regime used for a large model.
    dense_elements = 1_000_003
    dense_start = (dense_elements * rank) // world
    dense_end = (dense_elements * (rank + 1)) // world
    dense_indices = torch.arange(
        dense_start, dense_end, device=device, dtype=torch.float64,
    )
    dense_values = (
        torch.sin(dense_indices * 0.017)
        + torch.cos(dense_indices * 0.013)
        + 0.1 * torch.sin(dense_indices * 0.0007)
    )

    class DenseHolder:
        grad = None

    dense_holder = DenseHolder()
    dense_holder.grad = dense_values
    dense_sketch, dense_squared_norm = fixed_count_sketch(
        [("dense_collision_calibration", dense_holder)], rank, world,
    )
    dist.all_reduce(dense_sketch, op=dist.ReduceOp.SUM)
    dist.all_reduce(dense_squared_norm, op=dist.ReduceOp.SUM)
    dense_projected_norm = float(dense_sketch.norm().item())
    dense_exact_norm = float(dense_squared_norm.sqrt().item())
    dense_projection_relative_norm_error = abs(
        dense_projected_norm - dense_exact_norm
    ) / max(dense_exact_norm, 1e-30)

    # Exercise real-model parameter ordinals.  The tiny FSDP module above has
    # only one flat parameter and therefore cannot detect Python-int products
    # that overflow when converted to a signed-int64 tensor offset.
    ordinal_parameters = 64
    ordinal_named_gradients = []
    ordinal_values = []
    for ordinal in range(ordinal_parameters):
        holder = DenseHolder()
        value = float(1 + ordinal + rank * ordinal_parameters)
        holder.grad = torch.tensor([value], device=device, dtype=torch.float64)
        ordinal_named_gradients.append((f"ordinal_probe_{ordinal:03d}", holder))
        ordinal_values.append(value)
    ordinal_sketch, _ = fixed_count_sketch(
        ordinal_named_gradients, rank, world,
    )
    ordinal_reference = torch.zeros_like(ordinal_sketch)
    for ordinal, value in enumerate(ordinal_values):
        for basis in range(SKETCH_SPEC["independent_bases"]):
            bucket, sign = sketch_bucket_and_sign(0, ordinal, rank, basis)
            ordinal_reference[
                basis * SKETCH_SPEC["buckets_per_basis"] + bucket
            ] += value * sign
    ordinal_reference.div_(math.sqrt(SKETCH_SPEC["independent_bases"]))
    dist.all_reduce(ordinal_sketch, op=dist.ReduceOp.SUM)
    dist.all_reduce(ordinal_reference, op=dist.ReduceOp.SUM)
    ordinal_calibration_max_abs_error = float(
        (ordinal_sketch - ordinal_reference).abs().max().item()
    )

    if rank == 0:
        reference = torch.nn.Linear(17, 13, bias=False, device=device)
        with torch.no_grad():
            reference.weight.copy_(weight.to(device))
        combined_input = torch.cat([
            torch.arange(4 * 17, dtype=torch.float32, device=device).reshape(4, 17)
            / 100 + other / 10 for other in range(world)
        ])
        reference(combined_input).square().mean().backward()
        full_gradient = reference.weight.grad.detach().reshape(-1).float()
        padded = torch.zeros(sum(item.numel() for item in gathered), device=device)
        padded[:full_gradient.numel()] = full_gradient
        gathered_gradient = torch.cat(gathered)
        full_gradient_max_abs_error = float(
            (gathered_gradient - padded).abs().max().item()
        )

        class Holder:
            grad = None

        reference_sketch = torch.zeros_like(actual_sketch)
        shard_length = gathered[0].numel()
        for other in range(world):
            holder = Holder()
            holder.grad = padded[other * shard_length:(other + 1) * shard_length]
            piece, _ = fixed_count_sketch([("_flat_param", holder)], other, world)
            reference_sketch.add_(piece)
        sketch_max_abs_error = float(
            (actual_sketch - reference_sketch).abs().max().item()
        )
        projected_norm = float(actual_sketch.norm().item())
        exact_norm = float(full_gradient.double().norm().item())
        projection_relative_norm_error = abs(projected_norm - exact_norm) / max(exact_norm, 1e-30)
        passed = (
            full_gradient_max_abs_error <= 2e-5
            and sketch_max_abs_error <= 1e-7
            and projection_relative_norm_error <= 0.10
            and dense_projection_relative_norm_error <= 0.10
            and denominator_closure_error <= 1e-6
            and ordinal_calibration_max_abs_error <= 1e-12
        )
        report = {
            "schema": "memagent.coral.e1-fsdp-sketch-oracle.v4",
            "status": "PASS" if passed else "FAIL",
            "decision": "CORAL_E1_SKETCH_ORACLE_PASS" if passed
                        else "CORAL_E1_SKETCH_ORACLE_NO_GO",
            "world_size": world,
            "backend": "nccl",
            "basis_sha256": SKETCH_BASIS_SHA256,
            "full_gradient_elements": int(full_gradient.numel()),
            "padded_shard_elements": int(padded.numel()),
            "full_gradient_max_abs_error": full_gradient_max_abs_error,
            "sketch_max_abs_error": sketch_max_abs_error,
            "sketch_assembly_aperture": 1e-7,
            "projection_relative_norm_error": projection_relative_norm_error,
            "projection_error_aperture": 0.10,
            "collision_calibration_elements": dense_elements,
            "collision_calibration_buckets_per_basis": 256,
            "collision_calibration_exact_norm": dense_exact_norm,
            "collision_calibration_projected_norm": dense_projected_norm,
            "collision_calibration_relative_norm_error":
                dense_projection_relative_norm_error,
            "collision_calibration_error_aperture": 0.10,
            "two_rank_local_denominators": [5, 7],
            "denominator_gradient_closure_max_abs_error": denominator_closure_error,
            "denominator_gradient_closure_aperture": 1e-6,
            "ordinal_calibration_parameters": ordinal_parameters,
            "ordinal_calibration_max_abs_error": ordinal_calibration_max_abs_error,
            "ordinal_calibration_error_aperture": 1e-12,
        }
        report["report_sha256"] = canonical_sha256(report)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    dist.barrier()
    if rank == 0 and report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
