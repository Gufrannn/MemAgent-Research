#!/usr/bin/env python3
"""Fail-closed MIC-v2 CPU gates.

Only the ``e0`` command is released in this phase.  No command in this module
allocates a GPU or launches actor training.  Later E1/oracle commands must be
added in a separate reviewed commit after the E0 certificate is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from recurrent.research.mic_v2 import (
    CONTRACT_SHA256,
    GROUP_SIZE,
    SCHEMA,
    T_MAX,
    action_slot_receipts,
    bind_full_branch_arm_states,
    canonical_content_root,
    canonical_json,
    expand_role_credits,
    enumerate_oracle_toy_mdp,
    fixed_slot_actor_loss,
    full_branch_accounting,
    full_branch_block_schedule,
    full_branch_future_counter_key,
    full_branch_matched_slot_count,
    group_centered_broadcast,
    logo_port_actor_loss,
    logo_port_sequence_loss,
    mechanism_cell_credits,
    sampled_token_masks,
    seal_credit_bundle,
    sha256_file,
    sha256_json,
    sibling_reconstruction,
    sparse_branch_accounting,
    sparse_future_counter_key,
    sparse_branch_schedule,
    stable_fold_assignments,
    standardized_group_credit,
    validate_boundary_pair,
    validate_boundary_state,
    validate_content_root,
    verify_sealed_credit_bundle,
    write_json_new,
)


E0_IDS = tuple(f"E0-{index}" for index in range(1, 14))
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _state(root: str, trajectory: str, turn: int, chunks: list[str], memories: list[str], phase: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "phase": phase,
        "content_root_id": root,
        "stable_example_id": f"{root}:example",
        "trajectory_id": trajectory,
        "turn_index": turn,
        "question": "Which materialized fact supports the answer?",
        "arrived_chunks": chunks,
        "materialized_memory_history": memories,
        "current_memory": memories[-1] if memories else "",
        "public_metadata": {
            "chunk_schedule_id": "oracle-v1",
            "arrived_context_token_count": 2 * len(chunks),
            "prior_active_turn_count": max(0, turn - 1),
            "exogenous_termination": False,
            "policy_termination": False,
            "forced_truncation": False,
        },
    }


def _expect_rejection(function: Callable[[], Any], contains: str | None = None) -> str:
    try:
        function()
    except (ValueError, RuntimeError) as exc:
        if contains is not None and contains not in str(exc):
            raise AssertionError(f"wrong rejection: {exc}") from exc
        return str(exc)
    raise AssertionError("negative control was accepted")


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(f"MIC_V2_NO_GO: {message}")


def _require_json_finite(value: Any, path: str = "evidence") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_json_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_json_finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"MIC_V2_NO_GO: non-finite certificate field {path}")


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised by the H20 entry
        raise RuntimeError("MIC_V2_NO_GO: torch is required for E0 autograd tests") from exc
    return torch


def _loss_inputs(torch, lengths: list[int], roles: list[str], credits: list[float], *, width: int = 6):
    sampled_np, writer_np, answer_np = sampled_token_masks(
        sampled_lengths=lengths, roles=roles, token_width=width,
    )
    sampled = torch.tensor(sampled_np)
    writer = torch.tensor(writer_np)
    answer = torch.tensor(answer_np)
    scalar = torch.tensor(credits, dtype=torch.float64).unsqueeze(-1)
    credit_tensor = (scalar * sampled).detach()
    return sampled, writer, answer, credit_tensor


def e0_1() -> dict[str, Any]:
    oracle = enumerate_oracle_toy_mdp()
    result = oracle["decomposition"]
    _require(oracle["joint_row_count"] == 32, "E0-1 joint enumeration coverage")
    _require(abs(oracle["joint_probability"] - 1.0) <= 1e-15,
             "E0-1 joint probability closure")
    _require(all(abs(value) > 1e-6 for value in result["chunk_credits"]),
             "E0-1 chunk transition disappeared")
    _require(all(abs(value) > 1e-6 for value in result["writer_credits"]),
             "E0-1 writer transition disappeared")
    _require(abs(result["closure_error"]) <= 1e-15, "E0-1 closure")
    return oracle


def e0_2() -> dict[str, Any]:
    oracle = enumerate_oracle_toy_mdp()
    p = oracle["action1_probability_given_chunk1"]
    probabilities = np.asarray([1.0 - p, p], dtype=np.float64)
    action_scores = np.asarray([-p, 1.0 - p], dtype=np.float64)
    action_values = np.asarray(oracle["action1_q_values_given_chunk1"], dtype=np.float64)
    pre_value = float(probabilities @ action_values)
    mic_gradient = float(probabilities @ (action_scores * (action_values - pre_value)))
    terminal_gradient = float(oracle["action1_direct_terminal_score_gradient"])
    analytic = p * (1.0 - p) * (action_values[1] - action_values[0])
    shuffled = float(probabilities @ (action_scores * (action_values[::-1] - pre_value)))
    _require(abs(mic_gradient - terminal_gradient) <= 1e-15, "E0-2 gradient identity")
    _require(abs(mic_gradient - analytic) <= 1e-15, "E0-2 analytic gradient")
    _require(abs(shuffled - terminal_gradient) > 0.1, "E0-2 shuffled negative control")
    return {"oracle_gradient": mic_gradient, "terminal_gradient": terminal_gradient,
            "shuffled_gradient": shuffled,
            "joint_table_sha256": oracle["joint_table_sha256"],
            "action_q_values": action_values.tolist()}


def e0_3() -> dict[str, Any]:
    pre = _state("root-a", "traj-a", 2, ["c1", "c2"], ["m1"], "pre_write")
    post = _state("root-a", "traj-a", 2, ["c1", "c2"], ["m1", "m2"], "post_write")
    before, after = validate_boundary_pair(pre, post)
    overwrite = dict(post)
    overwrite["materialized_memory_history"] = ["m2"]
    cache_taint = dict(post)
    cache_taint["public_metadata"] = {"writer_cache": "secret"}
    synonym_taint = dict(post)
    synonym_taint["public_metadata"] = {"terminal_reward": 1.0}
    raw_taint = dict(post)
    raw_taint["raw_writer_completion"] = "unmaterialized bytes"
    rejections = [
        _expect_rejection(lambda: validate_boundary_state(overwrite), "phase-nested"),
        _expect_rejection(lambda: validate_boundary_state(cache_taint), "forbidden"),
        _expect_rejection(lambda: validate_boundary_state(synonym_taint), "unknown public"),
        _expect_rejection(lambda: validate_boundary_state(raw_taint), "forbidden"),
    ]
    return {"pre_sha256": before["state_sha256"], "post_sha256": after["state_sha256"],
            "negative_controls": rejections}


def _constant_oof_predictions(root_outcomes: dict[str, float], namespace: str) -> dict[str, float]:
    assignments = stable_fold_assignments(list(root_outcomes), namespace, 4)
    predictions = {}
    for root, held_fold in assignments.items():
        fit = [value for other, value in root_outcomes.items() if assignments[other] != held_fold]
        if not fit:
            raise AssertionError("empty E0 fit fold")
        predictions[root] = float(np.mean(fit))
    return predictions


def e0_4() -> dict[str, Any]:
    contents = [(f"question-{index}", [f"chunk-{index}-a", f"chunk-{index}-b"])
                for index in range(32)]
    roots = [canonical_content_root(question, chunks) for question, chunks in contents]
    base = stable_fold_assignments(roots, "e1-selection", 4)
    expanded = stable_fold_assignments(["aaa-earlier", *roots, "zzz-later"], "e1-selection", 4)
    _require(all(base[root] == expanded[root] for root in roots), "E0-4 set-invariant folds")
    outcomes = {root: float(index % 2) for index, root in enumerate(roots)}
    before = _constant_oof_predictions(outcomes, "e1-selection")
    target = roots[7]
    mutated = dict(outcomes)
    mutated[target] = 999.0
    after = _constant_oof_predictions(mutated, "e1-selection")
    _require(before[target] == after[target], "E0-4 held outcome entered fit")
    alias_rows = [(canonical_content_root(question, chunks), f"alias-{alias}")
                  for question, chunks in contents for alias in range(3)]
    _require(all(base[root] == stable_fold_assignments(roots, "e1-selection", 4)[root]
                 for root, _ in alias_rows), "E0-4 aliases crossed folds")
    _expect_rejection(lambda: validate_content_root(
        "0" * 64, contents[0][0], contents[0][1]), "does not bind")
    return {"fold_assignment_sha256": sha256_json(base), "held_root": target,
            "held_prediction": before[target], "alias_count": len(alias_rows)}


def e0_5() -> dict[str, Any]:
    torch = _torch()
    theta = torch.tensor([0.21, -0.33, 0.11], dtype=torch.float64, requires_grad=True)
    old = theta.detach().clone()
    credit = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float64)
    ratio = torch.exp(theta - old)
    ppo = torch.minimum(ratio * credit, torch.clamp(ratio, 0.8, 1.2) * credit)
    minimized = -ppo.sum()
    gradient = torch.autograd.grad(minimized, theta)[0]
    _require(torch.allclose(gradient, -credit, atol=1e-12, rtol=0), "E0-5 score sign")
    sampled, writer, answer, sealed = _loss_inputs(
        torch, [2, 1], ["writer", "answer"], [0.7, -0.2], width=3,
    )
    base = {
        "old_log_prob": torch.zeros((2, 3), dtype=torch.float64),
        "log_prob": torch.zeros((2, 3), dtype=torch.float64, requires_grad=True),
        "credits": sealed,
        "sampled_mask": sampled, "writer_mask": writer, "answer_mask": answer,
        "global_scheduled_slots": 18, "local_scheduled_slots": 18,
        "reference_length": 3.0, "clip_low": 0.2, "clip_high": 0.2,
    }
    varying = sealed.clone()
    varying[0, 1] = varying[0, 0] + 0.1
    old_with_grad = base["old_log_prob"].clone().requires_grad_()
    rejections = [
        _expect_rejection(lambda: fixed_slot_actor_loss(
            **{**base, "credits": varying}), "varies inside"),
        _expect_rejection(lambda: fixed_slot_actor_loss(
            **{**base, "old_log_prob": old_with_grad}), "old log probabilities"),
        _expect_rejection(lambda: fixed_slot_actor_loss(
            **{**base, "sampled_mask": sampled.to(torch.int64)}), "masks must be boolean"),
    ]
    return {"minimized_loss_gradient": gradient.tolist(), "expected": (-credit).tolist(),
            "immutability_rejections": rejections}


def e0_6() -> dict[str, Any]:
    sampled, writer, answer, receipts = action_slot_receipts(
        slots=[
            {"role": "writer", "sampled_token_ids": [11, 12, 2], "termination": "sampled_eos"},
            {"role": "answer", "sampled_token_ids": [21, 22, 23, 24, 25],
             "termination": "forced_truncation"},
            {"role": "inactive", "sampled_token_ids": [], "termination": "policy_termination"},
            {"role": "inactive", "sampled_token_ids": [], "termination": "exogenous_termination"},
        ], token_width=5, eos_token_id=2,
    )
    _require(sampled.sum() == 8 and writer.sum() == 3 and answer.sum() == 5, "E0-6 mask counts")
    _require(not sampled[2].any(), "E0-6 inactive slot")
    _expect_rejection(lambda: sampled_token_masks(
        sampled_lengths=[1], roles=["inactive"], token_width=3), "fictitious")
    _expect_rejection(lambda: action_slot_receipts(
        slots=[{"role": "writer", "sampled_token_ids": [9, 2],
                "termination": "forced_truncation"}], token_width=2, eos_token_id=2,
    ), "fictitious EOS")
    return {"sampled_tokens": int(sampled.sum()), "writer_tokens": int(writer.sum()),
            "answer_tokens": int(answer.sum()),
            "inactive_tokens": int(sampled[2].sum() + sampled[3].sum()),
            "slot_receipts": receipts}


def e0_7() -> dict[str, Any]:
    torch = _torch()

    def writer_gradient(answer_length: int, padding_logprob: float) -> float:
        parameter = torch.tensor(0.13, dtype=torch.float64, requires_grad=True)
        old = torch.zeros((2, 6), dtype=torch.float64)
        logp = torch.zeros((2, 6), dtype=torch.float64)
        logp[0] = parameter
        logp[1, answer_length:] = padding_logprob
        sampled, writer, answer, credits = _loss_inputs(
            torch, [2, answer_length], ["writer", "answer"], [0.8, -0.2],
        )
        kl = torch.full((2, 6), 0.1, dtype=torch.float64)
        entropy = torch.full((2, 6), 0.2, dtype=torch.float64)
        kl[1, answer_length:] = float("inf")
        entropy[1, answer_length:] = float("nan")
        loss, _ = fixed_slot_actor_loss(
            old_log_prob=old, log_prob=logp, credits=credits,
            sampled_mask=sampled, writer_mask=writer, answer_mask=answer,
            global_scheduled_slots=36, local_scheduled_slots=36,
            reference_length=4.0, clip_low=0.2, clip_high=0.2,
            kl_token=kl, entropy_token=entropy, beta_kl=0.001, beta_entropy=0.0,
        )
        return float(torch.autograd.grad(loss, parameter)[0].item())

    short = writer_gradient(1, 1000.0)
    long = writer_gradient(6, -1000.0)
    _require(math.isfinite(short) and math.isfinite(long), "E0-7 masked padding overflow")
    _require(abs(short - long) <= 1e-15, "E0-7 cross-slot isolation")
    return {"writer_gradient_short_answer_extreme_padding": short,
            "writer_gradient_long_answer": long}


def e0_8() -> dict[str, Any]:
    torch = _torch()

    def gradient(duplicate: int) -> float:
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        old = torch.zeros((2 * duplicate, 4), dtype=torch.float64)
        logp = parameter.expand_as(old)
        lengths = [2, 3] * duplicate
        roles = ["writer", "answer"] * duplicate
        scalars = [0.5, -0.25] * duplicate
        sampled, writer, answer, credits = _loss_inputs(
            torch, lengths, roles, scalars, width=4,
        )
        loss, _ = fixed_slot_actor_loss(
            old_log_prob=old, log_prob=logp, credits=credits,
            sampled_mask=sampled, writer_mask=writer, answer_mask=answer,
            global_scheduled_slots=36 * duplicate,
            local_scheduled_slots=36 * duplicate,
            reference_length=3.0, clip_low=0.2, clip_high=0.2,
        )
        return float(torch.autograd.grad(loss, parameter)[0].item())

    original, duplicated = gradient(1), gradient(2)
    _require(abs(original - duplicated) <= 1e-15, "E0-8 duplication invariance")
    return {"original_gradient": original, "duplicated_gradient": duplicated}


def _e0_9_distributed_worker(rank: int, init_method: str, output_dir: str) -> None:
    torch = _torch()
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo", init_method=init_method, rank=rank, world_size=2,
    )
    try:
        local_lengths = ([1, 5], [2, 0])[rank]
        local_roles = (["writer", "answer"], ["writer", "inactive"])[rank]
        local_scalars = ([0.8, -0.1], [0.4, 0.0])[rank]
        old = torch.zeros((2, 6), dtype=torch.float64)
        sampled, writer, answer, credits = _loss_inputs(
            torch, list(local_lengths), list(local_roles), list(local_scalars),
        )
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        loss, receipt = fixed_slot_actor_loss(
            old_log_prob=old, log_prob=parameter.expand_as(old), credits=credits,
            sampled_mask=sampled, writer_mask=writer, answer_mask=answer,
            global_scheduled_slots=72, local_scheduled_slots=36,
            reference_length=4.0, clip_low=0.2, clip_high=0.2, world_size=2,
        )
        local_gradient = torch.autograd.grad(loss, parameter)[0]
        distributed_gradient = local_gradient.detach().clone()
        dist.all_reduce(distributed_gradient, op=dist.ReduceOp.SUM)
        distributed_gradient /= 2.0
        record = {
            "rank": rank, "local_gradient": float(local_gradient),
            "distributed_mean_gradient": float(distributed_gradient),
            "receipt": receipt,
        }
        target = Path(output_dir) / f"rank_{rank}.json"
        with target.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def e0_9() -> dict[str, Any]:
    torch = _torch()
    old = torch.zeros((4, 6), dtype=torch.float64)
    lengths = [1, 5, 2, 0]
    roles = ["writer", "answer", "writer", "inactive"]
    scalars = [0.8, -0.1, 0.4, 0.0]
    sampled, writer, answer, credits = _loss_inputs(torch, lengths, roles, scalars)

    full_parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    full_loss, full_receipt = fixed_slot_actor_loss(
        old_log_prob=old, log_prob=full_parameter.expand_as(old), credits=credits,
        sampled_mask=sampled, writer_mask=writer, answer_mask=answer,
        global_scheduled_slots=72, local_scheduled_slots=72,
        reference_length=4.0, clip_low=0.2, clip_high=0.2,
    )
    full_grad = torch.autograd.grad(full_loss, full_parameter)[0]
    with tempfile.TemporaryDirectory(prefix="mic-v2-e0-ddp-") as directory:
        init_file = Path(directory) / "process_group_init"
        init_method = f"file://{init_file}"
        torch.multiprocessing.spawn(
            _e0_9_distributed_worker,
            args=(init_method, directory),
            nprocs=2,
            join=True,
        )
        records = [json.loads((Path(directory) / f"rank_{rank}.json").read_text())
                   for rank in range(2)]
    ddp_mean = torch.tensor(records[0]["distributed_mean_gradient"], dtype=torch.float64)
    _require(all(abs(row["distributed_mean_gradient"] - float(ddp_mean)) <= 1e-15
                 for row in records), "E0-9 ranks disagree after all-reduce")
    _require(torch.allclose(ddp_mean, full_grad, atol=1e-12, rtol=0),
             "E0-9 distributed gradient")
    receipts = [row["receipt"] for row in records]
    active_slots = [
        row["receipt"]["active_writer_slots"] + row["receipt"]["active_answer_slots"]
        for row in records
    ]
    _require(active_slots == [2, 1], "E0-9 unequal active-slot fixture drifted")
    reconstructed_loss = sum(item["local_loss"] for item in receipts) / 2.0
    _require(abs(reconstructed_loss - full_receipt["local_loss"]) <= 1e-12,
             "E0-9 distributed loss reconstruction")
    return {"single_process_gradient": float(full_grad), "two_rank_gradient": float(ddp_mean),
            "rank_receipt_sha256": sha256_json(receipts),
            "rank_local_gradients": [row["local_gradient"] for row in records],
            "rank_active_slots": active_slots, "backend": "gloo", "process_count": 2}


def e0_10() -> dict[str, Any]:
    torch = _torch()
    actor = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    critic = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
    representation = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
    critic_value = critic * representation
    detached_credit = critic_value.detach().reshape(1, 1)
    bundle = seal_credit_bundle(
        block_id="oracle-block", behavior_checkpoint_sha256="a" * 64,
        fold_receipts=[{"fold": 0, "held_root_sha256": "b" * 64}],
        rows=[{"content_root_id": "root", "trajectory_id": "trajectory",
               "turn_index": 1, "writer_credit": float(detached_credit), "answer_credit": 0.0}],
    )
    before = canonical_json(bundle)
    optimizer = torch.optim.SGD([actor], lr=0.1)
    _require(all(id(parameter) not in {id(critic), id(representation)}
                 for group in optimizer.param_groups for parameter in group["params"]),
             "E0-10 critic parameter entered actor optimizer")
    loss = -(torch.exp(actor) * detached_credit).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _require(critic.grad is None and representation.grad is None, "E0-10 critic gradient path")
    _require(canonical_json(bundle) == before, "E0-10 sealed bundle mutation")
    verify_sealed_credit_bundle(bundle, bundle["bundle_sha256"])
    return {"critic_grad": None, "representation_grad": None,
            "sealed_bundle_sha256": bundle["bundle_sha256"]}


def e0_11() -> dict[str, Any]:
    returns = [0.0, 0.25, 0.5, 1.0]
    broadcast = group_centered_broadcast(returns)
    reconstructed = sibling_reconstruction(returns)
    _require(np.allclose(broadcast, reconstructed, atol=1e-15), "E0-11 c_G reconstruction")
    mic_raw = [0.1, -0.2, 0.3, -0.4]
    postpost_raw = [0.2, -0.1, 0.5, -0.2]
    cells = mechanism_cell_credits(
        returns=returns, mic_write_deltas=mic_raw,
        postpost_deltas=postpost_raw, gate=[0, 1, 0, 1],
    )
    _require(all(np.array_equal(row["answer"], broadcast) for row in cells.values()),
             "E0-11 answer routing drifted")
    expected_gated = np.asarray([broadcast[0], 0.75 * mic_raw[1],
                                 broadcast[2], 0.75 * mic_raw[3]])
    _require(np.allclose(cells["MIC-Gated"]["writer"], expected_gated, atol=1e-15),
             "E0-11 binary gate routing")
    logo = standardized_group_credit(returns)
    _require(not np.allclose(logo, broadcast), "E0-11 LoGo standardization isolation")
    _require(np.array_equal(standardized_group_credit([1.0] * 4), np.zeros(4)),
             "E0-11 zero variance group")
    serializable_cells = {name: {role: values.tolist() for role, values in row.items()}
                          for name, row in cells.items()}
    torch = _torch()
    _, writer_mask_np, _ = sampled_token_masks(
        sampled_lengths=[1, 2, 3, 1, 0, 0, 0, 0],
        roles=["writer"] * 4 + ["inactive"] * 4, token_width=3,
    )
    _, _, answer_mask_np = sampled_token_masks(
        sampled_lengths=[0, 0, 0, 0, 2, 1, 3, 2],
        roles=["inactive"] * 4 + ["answer"] * 4, token_width=3,
    )
    writer_mask = torch.tensor(writer_mask_np)
    answer_mask = torch.tensor(answer_mask_np)
    routed_hashes = {}
    for name, row in cells.items():
        writer_sequence = torch.tensor([*row["writer"], 0, 0, 0, 0], dtype=torch.float64)
        answer_sequence = torch.tensor([0, 0, 0, 0, *row["answer"]], dtype=torch.float64)
        routed = expand_role_credits(
            writer_sequence_credits=writer_sequence.detach(),
            answer_sequence_credits=answer_sequence.detach(),
            writer_mask=writer_mask, answer_mask=answer_mask,
        )
        for replica in range(4):
            _require(torch.all(routed[replica].masked_select(writer_mask[replica])
                               == float(row["writer"][replica])),
                     f"E0-11 {name} writer token credit")
            _require(torch.all(routed[4 + replica].masked_select(answer_mask[4 + replica])
                               == float(broadcast[replica])),
                     f"E0-11 {name} answer token credit")
        routed_hashes[name] = hashlib.sha256(routed.numpy().tobytes()).hexdigest()
    return {"c_g": 0.75, "broadcast": broadcast.tolist(),
            "cells_sha256": sha256_json(serializable_cells),
            "routed_token_sha256": routed_hashes, "logo_standardized": logo.tolist()}


def e0_12() -> dict[str, Any]:
    schedule = sparse_branch_schedule("root-sparse", 2026)
    _require(schedule == sparse_branch_schedule("root-sparse", 2026), "E0-12 fixed schedule")
    _require(schedule["writer_arm_keys"][0] != schedule["writer_arm_keys"][1],
             "E0-12 action arms share a counter")
    future_coordinates = [
        (future_turn, "writer")
        for future_turn in range(schedule["turn"] + 1, T_MAX + 1)
    ] + [(T_MAX + 1, "answer")]
    futures = [sparse_future_counter_key(
        experiment_seed=2026, content_root_id="root-sparse", replica=schedule["replica"],
        turn=schedule["turn"], future_turn=future_turn, role=role, future_seed_index=0,
    ) for future_turn, role in future_coordinates]
    _require(len(set(futures)) == len(futures), "E0-12 future counters collide")
    _expect_rejection(lambda: sparse_future_counter_key(
        experiment_seed=2026, content_root_id="root-sparse", replica=schedule["replica"],
        turn=schedule["turn"], future_turn=schedule["turn"], role="writer",
        future_seed_index=0), "role/turn")
    receipt = sparse_branch_accounting(
        trunk_tokens=10, arm_writer_tokens=[3, 5], continuation_tokens=[7, 9],
        leaf_returns=[0.8, 0.2], other_replica_returns=[0.1, 0.4, 0.5],
        model_forward_tokens=34, model_backward_tokens=34,
        h20_seconds=4.0, wall_seconds=2.0, active=True,
    )
    _require(receipt["physical_model_tokens"] == 34, "E0-12 physical tokens")
    _require(receipt["actor_weighted_tokens"] == 22, "E0-12 weighted tokens")
    _require(np.allclose(receipt["pair_credits"], [0.45, -0.45]), "E0-12 pair credit")
    scores = [0.7, -0.3]
    delivered = 0.5 * sum(score * credit for score, credit in zip(scores, receipt["pair_credits"]))
    expected_delivered = 0.75 / 2.0 * (
        scores[0] * (0.8 - 0.2) + scores[1] * (0.2 - 0.8)
    )
    _require(abs(delivered - expected_delivered) <= 1e-15,
             "E0-12 half-weight behavior-point score")
    trunk_score = 0.2
    downstream_scores = [[0.1, -0.2], [0.3, 0.4]]
    complete_delivered = (
        trunk_score * receipt["trunk_credit"]
        + delivered
        + 0.5 * sum(
            score * receipt["downstream_leaf_credits"][leaf]
            for leaf in range(2) for score in downstream_scores[leaf]
        )
    )
    other_mean = np.mean([0.1, 0.4, 0.5])
    branch_advantages = [0.75 * (0.8 - other_mean), 0.75 * (0.2 - other_mean)]
    expected_complete = (
        trunk_score * np.mean(branch_advantages)
        + expected_delivered
        + 0.5 * sum(
            score * branch_advantages[leaf]
            for leaf in range(2) for score in downstream_scores[leaf]
        )
    )
    _require(abs(complete_delivered - expected_complete) <= 1e-15,
             "E0-12 complete trunk/arm/future credit delivery")
    inactive = sparse_branch_accounting(
        trunk_tokens=10, arm_writer_tokens=[0, 0], continuation_tokens=[0, 0],
        leaf_returns=[], other_replica_returns=[],
        model_forward_tokens=10, model_backward_tokens=10,
        h20_seconds=1.0, wall_seconds=1.0, active=False,
    )
    _require(inactive["physical_model_tokens"] == 10, "E0-12 inactive trunk accounting")
    _require(inactive["pair_credits"] == [0.0, 0.0]
             and not inactive["outcome_based_fallback"], "E0-12 inactive fallback")
    _expect_rejection(lambda: sparse_branch_accounting(
        trunk_tokens=10, arm_writer_tokens=[3, 5], continuation_tokens=[7, 9],
        leaf_returns=[0.8, 0.2], other_replica_returns=[0.1, 0.4, 0.5],
        model_forward_tokens=1, model_backward_tokens=0,
        h20_seconds=4.0, wall_seconds=2.0, active=True,
    ), "does not reconstruct")
    _expect_rejection(lambda: sparse_branch_accounting(
        trunk_tokens=10, arm_writer_tokens=[3, 5], continuation_tokens=[7, 9],
        leaf_returns=[0.8, 0.2], other_replica_returns=[0.1, 0.4],
        model_forward_tokens=34, model_backward_tokens=34,
        h20_seconds=4.0, wall_seconds=2.0, active=True,
    ), "return groups")
    return {"schedule": schedule, "future_counter_sha256": sha256_json(futures),
            "accounting": receipt, "inactive_accounting": inactive}


def e0_13() -> dict[str, Any]:
    torch = _torch()
    schedule = full_branch_block_schedule(
        block_id="oracle-block", content_root_ids=[f"root-{index}" for index in range(64)],
        experiment_seed=2026,
    )
    selected = schedule["selected_count"]
    _require(schedule["candidate_count"] == 64 * T_MAX, "E0-13 candidate schedule")
    _require(selected == 64 * T_MAX // 4, "E0-13 exact 25 percent schedule")
    first = schedule["records"][0]
    _require(len(set(first["writer_arm_keys"])) == 4, "E0-13 action counter independence")
    future_keys = [full_branch_future_counter_key(
        experiment_seed=2026, block_id="oracle-block",
        content_root_id=first["content_root_id"], turn=first["turn_index"],
        arm=arm, future_turn=future_turn,
        role="answer" if future_turn == T_MAX + 1 else "writer",
    ) for arm in range(4) for future_turn in range(first["turn_index"] + 1, T_MAX + 2)]
    _require(len(set(future_keys)) == len(future_keys), "E0-13 future counter independence")
    matched_cell_future_keys = [full_branch_future_counter_key(
        experiment_seed=2026, block_id="oracle-block",
        content_root_id=first["content_root_id"], turn=first["turn_index"],
        arm=arm, future_turn=future_turn,
        role="answer" if future_turn == T_MAX + 1 else "writer",
    ) for arm in range(4) for future_turn in range(first["turn_index"] + 1, T_MAX + 2)]
    _require(future_keys == matched_cell_future_keys, "E0-13 paired cells lost common randomness")
    _expect_rejection(lambda: full_branch_future_counter_key(
        experiment_seed=2026, block_id="oracle-block",
        content_root_id=first["content_root_id"], turn=first["turn_index"],
        arm=0, future_turn=T_MAX + 1, role="writer"), "role/turn")
    other_seed = full_branch_block_schedule(
        block_id="oracle-block", content_root_ids=[f"root-{index}" for index in range(64)],
        experiment_seed=2027,
    )
    _require([(row["content_root_id"], row["turn_index"]) for row in schedule["records"]] == [
        (row["content_root_id"], row["turn_index"]) for row in other_seed["records"]
    ], "E0-13 structural schedule changed across seeds")
    _require(other_seed["records"][0]["writer_arm_keys"] != first["writer_arm_keys"],
             "E0-13 action streams reused across seeds")
    global_trajectory_ids = [
        f"{first['content_root_id']}:global-replica-{replica}"
        for replica in range(GROUP_SIZE)
    ]
    anchor_trajectory_id = global_trajectory_ids[first["anchor_replica"]]
    pre_template = _state(
        first["content_root_id"], anchor_trajectory_id, first["turn_index"],
        [f"chunk-{index}" for index in range(1, first["turn_index"] + 1)],
        [f"memory-{index}" for index in range(1, first["turn_index"])], "pre_write",
    )
    arm_states = [json.loads(json.dumps(pre_template)) for _ in range(4)]
    arm_binding = bind_full_branch_arm_states(
        pre_states=arm_states, schedule_record=first,
        global_trajectory_ids_by_replica=global_trajectory_ids,
    )
    tampered_arms = [json.loads(json.dumps(pre_template)) for _ in range(4)]
    tampered_arms[3]["question"] = "different pre-state"
    _expect_rejection(lambda: bind_full_branch_arm_states(
        pre_states=tampered_arms, schedule_record=first,
        global_trajectory_ids_by_replica=global_trajectory_ids), "do not share exact")
    wrong_anchor_arms = [json.loads(json.dumps(pre_template)) for _ in range(4)]
    wrong_anchor_arms[0]["trajectory_id"] = "wrong-global-replica"
    _expect_rejection(lambda: bind_full_branch_arm_states(
        pre_states=wrong_anchor_arms, schedule_record=first,
        global_trajectory_ids_by_replica=global_trajectory_ids), "anchor replica")
    forged_mapping = [f"forged-{replica}" for replica in range(GROUP_SIZE)]
    _expect_rejection(lambda: bind_full_branch_arm_states(
        pre_states=arm_states, schedule_record=first,
        global_trajectory_ids_by_replica=forged_mapping), "anchor replica")
    active_selected = selected - 1
    matched_slots = full_branch_matched_slot_count(
        root_count=64, selected_scheduled_boundaries=selected,
    )
    _require(matched_slots == 64 * 4 * (T_MAX + 1) + 4 * selected,
             "E0-13 Matched denominator")
    accounting = full_branch_accounting(
        root_count=64, selected_scheduled_boundaries=selected,
        selected_active_boundaries=active_selected,
        global_actor_tokens=12000, local_writer_actor_tokens=2000,
        reward_continuation_tokens=7000, terminal_continuations=4 * active_selected,
        model_forward_tokens=21000, model_backward_tokens=14000,
        h20_seconds=800.0, wall_seconds=400.0,
    )
    _require(accounting["fixed_actor_slots"] == matched_slots, "E0-13 accounting slots")
    _require(accounting["selected_inactive_boundaries"] == 1,
             "E0-13 inactive selected boundary lost fixed slots")
    _require(accounting["actor_regularizer_tokens"] == 14000,
             "E0-13 continuation entered regularizers")
    _require(accounting["physical_model_tokens"] == 21000, "E0-13 physical tokens")
    _expect_rejection(lambda: full_branch_accounting(
        root_count=64, selected_scheduled_boundaries=selected,
        selected_active_boundaries=active_selected,
        global_actor_tokens=12000, local_writer_actor_tokens=2000,
        reward_continuation_tokens=7000, terminal_continuations=4 * active_selected,
        model_forward_tokens=0, model_backward_tokens=0,
        h20_seconds=800.0, wall_seconds=400.0,
    ), "does not reconstruct")

    global_returns = [0.0, 0.25, 0.5, 1.0]
    local_returns = [0.1, 0.4, 0.7, 0.9]
    global_standardized = standardized_group_credit(global_returns)
    local_standardized = standardized_group_credit(local_returns)
    standardized = np.concatenate([
        global_standardized,
        global_standardized,
        local_standardized,
    ])
    port_roles = ["global_writer"] * 4 + ["global_answer"] * 4 + ["local_writer"] * 4
    lengths = [1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 3]
    mask_np, _, _ = sampled_token_masks(
        sampled_lengths=lengths, roles=["writer"] * 12, token_width=3,
    )
    mask = torch.tensor(mask_np)
    old = torch.zeros((12, 3), dtype=torch.float64)
    sequence_mean_log_ratios = np.asarray([
        0.0, 0.1, -0.2, 0.3,
        0.05, -0.1, 0.2, -0.05,
        0.15, -0.15, 0.25, -0.25,
    ])
    logp_values = np.zeros((12, 3), dtype=np.float64)
    for row_index, length in enumerate(lengths):
        logp_values[row_index, :length] = sequence_mean_log_ratios[row_index]
        logp_values[row_index, length:] = np.inf
    logp = torch.tensor(logp_values, dtype=torch.float64, requires_grad=True)
    credits = torch.tensor(standardized, dtype=torch.float64).detach()
    pg_only = logo_port_sequence_loss(
        old_log_prob=old, log_prob=logp, credits=credits.detach(), sampled_mask=mask,
    )
    kl = torch.full((12, 3), 0.2, dtype=torch.float64)
    entropy = torch.full((12, 3), 0.7, dtype=torch.float64)
    kl[~mask] = float("inf")
    entropy[~mask] = float("nan")
    port_loss, port_receipt = logo_port_actor_loss(
        old_log_prob=old, log_prob=logp, credits=credits.detach(), sampled_mask=mask,
        kl_token=kl, entropy_token=entropy, beta_kl=0.001, beta_entropy=0.0,
    )
    _require(torch.isfinite(port_loss), "E0-13 Port loss is non-finite")
    ratios = np.exp(sequence_mean_log_ratios)
    expected_terms = []
    for ratio, credit in zip(ratios, standardized):
        clipped_ratio = min(max(float(ratio), 0.8), 1.2)
        ordinary = max(-float(ratio) * float(credit), -clipped_ratio * float(credit))
        expected_terms.append(min(-3.0 * float(credit), ordinary) if credit < 0 else ordinary)
    expected_pg = float(np.mean(expected_terms))
    _require(abs(float(pg_only.detach()) - expected_pg) <= 1e-12,
             "E0-13 geometric ratio or dual clipping")
    expected_kl = 0.2
    _require(abs(float(port_loss.detach()) - (expected_pg + 0.001 * expected_kl)) <= 1e-12,
             "E0-13 complete Port objective")
    port_loss.backward()
    _require(torch.isfinite(logp.grad).all(), "E0-13 Port gradient")
    zero_credits = torch.tensor(standardized_group_credit([0.5] * 4), dtype=torch.float64)
    zero_old = torch.zeros((4, 1), dtype=torch.float64)
    zero_logp = torch.zeros((4, 1), dtype=torch.float64, requires_grad=True)
    zero_mask = torch.ones((4, 1), dtype=torch.bool)
    zero_loss = logo_port_sequence_loss(
        old_log_prob=zero_old, log_prob=zero_logp,
        credits=zero_credits.detach(), sampled_mask=zero_mask,
    )
    _require(float(zero_loss.detach()) == 0.0, "E0-13 zero-variance Port group")
    _expect_rejection(lambda: logo_port_sequence_loss(
        old_log_prob=old.clone().requires_grad_(), log_prob=logp,
        credits=credits, sampled_mask=mask), "old log probabilities")
    _expect_rejection(lambda: logo_port_sequence_loss(
        old_log_prob=old, log_prob=logp, credits=credits,
        sampled_mask=mask.to(torch.int64)), "must be boolean")
    return {"selected_boundaries": selected, "matched_global_plus_local_slots": matched_slots,
            "port_loss": float(port_loss.detach()), "schedule_sha256": schedule["schedule_sha256"],
            "future_counter_sha256": sha256_json(future_keys),
            "arm_state_binding": arm_binding,
            "port_role_credit_receipt": [
                {"role": role, "credit": float(credit)}
                for role, credit in zip(port_roles, standardized)
            ],
            "global_standardized_credits": global_standardized.tolist(),
            "local_standardized_credits": local_standardized.tolist(),
            "port_receipt": port_receipt, "accounting": accounting}


E0_TESTS: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
    ("E0-1", e0_1), ("E0-2", e0_2), ("E0-3", e0_3), ("E0-4", e0_4),
    ("E0-5", e0_5), ("E0-6", e0_6), ("E0-7", e0_7), ("E0-8", e0_8),
    ("E0-9", e0_9), ("E0-10", e0_10), ("E0-11", e0_11), ("E0-12", e0_12),
    ("E0-13", e0_13),
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def run_e0(repo: Path, expected_commit: str, output: Path, run_id: str) -> dict[str, Any]:
    if sys.flags.optimize != 0:
        raise RuntimeError("MIC_V2_NO_GO: optimized Python disables oracle checks")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RuntimeError("MIC_V2_NO_GO: run ID is not a safe stable identifier")
    if not repo.is_absolute() or not repo.is_dir():
        raise RuntimeError("MIC_V2_NO_GO: repository path must be an existing absolute directory")
    actual_commit = _git(repo, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError("MIC_V2_NO_GO: exact Git commit mismatch")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("MIC_V2_NO_GO: worktree is dirty")
    contract = repo / "docs/papers/mic_v2_scientific_contract_20260825.md"
    if sha256_file(contract) != CONTRACT_SHA256:
        raise RuntimeError("MIC_V2_NO_GO: scientific contract digest mismatch")
    manifest_path = repo / "manifests/h20/qwen25_7b_mic_v2_preregistration.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "memagent.mic.v2.preregistration" \
            or manifest.get("status") != "FROZEN_SCIENCE_IMPLEMENTATION_E0_ONLY" \
            or manifest.get("contract", {}).get("sha256") != CONTRACT_SHA256 \
            or manifest.get("e0", {}).get("required_tests") != list(E0_IDS) \
            or manifest.get("e0", {}).get("gpu_required") is not False:
        raise RuntimeError("MIC_V2_NO_GO: preregistration manifest drifted")
    manifest_sha256 = sha256_file(manifest_path)
    results = []
    for test_id, function in E0_TESTS:
        try:
            evidence = function()
            _require_json_finite(evidence)
        except Exception:
            failure = {
                "schema": SCHEMA, "kind": "e0_certificate", "status": "FAIL",
                "decision": "MIC_V2_E0_NO_GO", "git_commit": actual_commit,
                "run_id": run_id, "output_path": str(output),
                "contract_sha256": CONTRACT_SHA256, "failed_test": test_id,
                "preregistration_manifest_sha256": manifest_sha256,
                "traceback": traceback.format_exc(), "completed": results,
            }
            failure["certificate_sha256"] = sha256_json(failure)
            write_json_new(output, failure)
            return failure
        results.append({"id": test_id, "status": "PASS", "evidence": evidence,
                        "evidence_sha256": sha256_json(evidence)})
    _require(tuple(row["id"] for row in results) == E0_IDS, "E0 suite coverage drifted")
    report = {
        "schema": SCHEMA, "kind": "e0_certificate", "status": "PASS",
        "decision": "MIC_V2_E0_PASS", "git_commit": actual_commit,
        "run_id": run_id, "output_path": str(output),
        "contract_sha256": CONTRACT_SHA256, "python": sys.version,
        "preregistration_manifest_sha256": manifest_sha256,
        "numpy_version": np.__version__, "tests": results,
    }
    try:
        import torch
        report["torch_version"] = torch.__version__
    except Exception:
        report["torch_version"] = "unavailable"
    report["certificate_sha256"] = sha256_json(report)
    write_json_new(output, report)
    return report


def _verify_e0_certificate_payload(
    payload: Mapping[str, Any], *, run_id: str, expected_commit: str,
    output: Path, manifest_sha256: str,
) -> dict[str, Any]:
    unsigned = dict(payload)
    digest = unsigned.pop("certificate_sha256", None)
    if digest != sha256_json(unsigned):
        raise RuntimeError("MIC_V2_NO_GO: E0 certificate digest mismatch")
    if payload.get("schema") != SCHEMA or payload.get("kind") != "e0_certificate":
        raise RuntimeError("MIC_V2_NO_GO: E0 certificate schema mismatch")
    if payload.get("status") != "PASS" or payload.get("decision") != "MIC_V2_E0_PASS":
        raise RuntimeError("MIC_V2_NO_GO: E0 certificate is not PASS")
    if payload.get("run_id") != run_id or payload.get("output_path") != str(output):
        raise RuntimeError("MIC_V2_NO_GO: E0 certificate run/output identity mismatch")
    if payload.get("git_commit") != expected_commit:
        raise RuntimeError("MIC_V2_NO_GO: E0 certificate commit mismatch")
    if payload.get("contract_sha256") != CONTRACT_SHA256 \
            or payload.get("preregistration_manifest_sha256") != manifest_sha256:
        raise RuntimeError("MIC_V2_NO_GO: E0 certificate authority mismatch")
    tests = payload.get("tests", [])
    if [row.get("id") for row in tests] != list(E0_IDS) \
            or any(row.get("status") != "PASS" for row in tests) \
            or any(row.get("evidence_sha256") != sha256_json(row.get("evidence")) for row in tests):
        raise RuntimeError("MIC_V2_NO_GO: E0 certificate coverage mismatch")
    _require_json_finite(payload)
    return {
        "status": "PASS", "decision": "MIC_V2_E0_PASS",
        "git_commit": expected_commit, "run_id": run_id,
        "certificate_sha256": digest, "certificate": str(output),
    }


def verify_e0_certificate(
    repo: Path, expected_commit: str, output: Path, run_id: str,
) -> dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch(run_id) or not repo.is_absolute() or not output.is_absolute():
        raise RuntimeError("MIC_V2_NO_GO: E0 verification identity is invalid")
    if _git(repo, "rev-parse", "HEAD") != expected_commit or _git(repo, "status", "--porcelain"):
        raise RuntimeError("MIC_V2_NO_GO: E0 verification checkout drifted")
    contract = repo / "docs/papers/mic_v2_scientific_contract_20260825.md"
    manifest = repo / "manifests/h20/qwen25_7b_mic_v2_preregistration.json"
    if sha256_file(contract) != CONTRACT_SHA256:
        raise RuntimeError("MIC_V2_NO_GO: E0 verification contract drifted")
    payload = json.loads(output.read_text(encoding="utf-8"))
    return _verify_e0_certificate_payload(
        payload, run_id=run_id, expected_commit=expected_commit,
        output=output, manifest_sha256=sha256_file(manifest),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    e0 = subparsers.add_parser("e0")
    e0.add_argument("--repo", required=True, type=Path)
    e0.add_argument("--expected-commit", required=True)
    e0.add_argument("--output", required=True, type=Path)
    e0.add_argument("--run-id", required=True)
    verify = subparsers.add_parser("verify-e0")
    verify.add_argument("--repo", required=True, type=Path)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument("--output", required=True, type=Path)
    verify.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "e0":
        report = run_e0(
            args.repo.resolve(), args.expected_commit, args.output.resolve(), args.run_id,
        )
        print(canonical_json(report))
        return 0 if report["status"] == "PASS" else 2
    if args.command == "verify-e0":
        summary = verify_e0_certificate(
            args.repo.resolve(), args.expected_commit, args.output.resolve(), args.run_id,
        )
        print(canonical_json(summary))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
