"""Append-only actual-loss evidence for RWWPO and RWWPO-2."""
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path

import torch


LEDGER_VERSION = "rwwpo-actual-loss-v2"
TENSOR_LEDGER_VERSION = "rwwpo-actual-loss-v3"


def _tolist(tensor):
    return tensor.detach().to("cpu").tolist()


def _cpu_tensor(tensor):
    if not torch.is_tensor(tensor):
        raise TypeError("RWWPO tensor ledger accepts tensors only")
    return tensor.detach().to("cpu").contiguous().clone()


def _append_jsonl(path, record):
    # JSON's historical NaN/Infinity extension is not admissible evidence.
    # Reject it before a receipt can enter the append-only chain.
    canonical = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    record["record_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    line = (json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(line)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while appending RWWPO ledger")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _previous_tail(path):
    if not path.exists():
        return "0" * 64
    if path.is_symlink():
        raise ValueError("RWWPO ledger cannot be a symlink")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1])["record_sha256"] if lines else "0" * 64


def _write_tensor_shard_exclusive(root, *, rank, global_step, inner_id, payload):
    shard_dir = root / "tensor_shards" / f"rank{int(rank)}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if shard_dir.is_symlink() or root not in shard_dir.resolve().parents:
        raise ValueError("RWWPO tensor shard path escaped ledger root")
    shard = shard_dir / f"round_{int(global_step):04d}_inner_{int(inner_id)}.pt"
    if shard.exists() or shard.is_symlink():
        raise FileExistsError(f"RWWPO tensor shard already exists: {shard}")
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(shard, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while creating RWWPO tensor shard")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    relative = shard.relative_to(root).as_posix()
    inventory = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(payload.items())
    }
    return relative, hashlib.sha256(raw).hexdigest(), len(raw), inventory


def tensor_shard_inventory(ledger_dir, *, start_round=1, through_round,
                           record_limits=None):
    """Authenticate and inventory all v3 shards through a checkpoint round."""
    root = Path(ledger_dir).resolve()
    items = []
    for ledger in sorted(root.glob("actual_loss_rank*.jsonl")):
        if ledger.is_symlink():
            raise ValueError("RWWPO ledger cannot be a symlink")
        limit = None if record_limits is None else record_limits.get(ledger.name)
        selected_record_count = 0
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if limit is not None and selected_record_count >= int(limit):
                break
            selected_record_count += 1
            row = json.loads(line)
            if row.get("schema_version") != TENSOR_LEDGER_VERSION:
                continue
            if (int(row["global_step"]) < int(start_round)
                    or int(row["global_step"]) > int(through_round)):
                continue
            evidence = row.get("tensor_shard", {})
            relative = evidence.get("relative_path")
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError("RWWPO tensor shard path is not relative")
            shard = root / relative
            resolved = shard.resolve()
            if root not in resolved.parents or shard.is_symlink() or not shard.is_file():
                raise ValueError("RWWPO tensor shard path escape/symlink/missing")
            raw = shard.read_bytes()
            if len(raw) != int(evidence.get("size", -1)) or hashlib.sha256(raw).hexdigest() != evidence.get("sha256"):
                raise ValueError("RWWPO tensor shard size/hash mismatch")
            items.append({
                "rank": int(row["rank"]), "round": int(row["global_step"]),
                "inner_id": int(row["inner_id"]), "proposal_clock": int(row["proposal_clock"]),
                "relative_path": relative, "sha256": evidence["sha256"],
                "size": int(evidence["size"]),
            })
        if limit is not None and selected_record_count != int(limit):
            raise ValueError(
                f"RWWPO ledger shorter than checkpoint prefix: {ledger.name}"
            )
    items.sort(key=lambda row:(row["proposal_clock"],row["rank"]))
    expected={(2*(round_id-1)+inner,rank)
              for round_id in range(int(start_round),int(through_round)+1)
              for inner in (1,2) for rank in (0,1)}
    actual={(row["proposal_clock"],row["rank"]) for row in items}
    if actual != expected or len(items) != len(expected):
        raise ValueError("RWWPO tensor shard inventory lacks exact K2 x rank2 closure")
    canonical=json.dumps(items,sort_keys=True,separators=(",",":"))
    return {"start_round":int(start_round),"through_round":int(through_round),
            "shard_count":len(items),"items":items,
            "inventory_sha256":hashlib.sha256(canonical.encode()).hexdigest()}


def append_transaction_marker(*, ledger_dir, attempt_id, rank, global_step, epoch, minibatch,
                              phase, model_digest, inner_id=None, proposal_clock=None):
    if phase not in ("intent", "complete"):
        raise ValueError("bad transaction marker phase")
    path = Path(ledger_dir).resolve() / f"transaction_rank{int(rank)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "rwwpo-transaction-v2" if inner_id is not None else "rwwpo-transaction-v1",
        "attempt_id": str(attempt_id), "rank": int(rank),
        "global_step": int(global_step), "epoch": int(epoch),
        "minibatch": int(minibatch), "phase": phase,
        "model_digest": str(model_digest),
        "inner_id": None if inner_id is None else int(inner_id),
        "proposal_clock": None if proposal_clock is None else int(proposal_clock),
        "previous_record_sha256": _previous_tail(path),
    }
    _append_jsonl(path, record)
    return str(path)


def append_transaction_failure_record(
        *, ledger_dir, attempt_id, rank, global_step, inner_id,
        proposal_clock, reason, phase, prefix_rows, prefix_stats,
        current_reference_max_abs, behavior_batch_digest,
        transaction_entry_buffer_digest, diagnostics=None):
    """Append an authenticated diagnostic for a transaction that cannot commit.

    Failure receipts are never accepted as training evidence.  They preserve the
    actual prefix certificate that caused a fail-closed exit, including failures
    before the normal intent marker/tensor receipt can be written.
    """
    allowed = {
        "RWWPO_PREFIX_TRUST_REGION_VIOLATION",
        "RWWPO2_POST_COMMIT_FORWARD_CLOSURE_FAILURE",
    }
    global_step = int(global_step)
    inner_id = int(inner_id)
    proposal_clock = int(proposal_clock)
    current_reference_max_abs = float(current_reference_max_abs)
    if reason not in allowed or phase not in ("precondition", "post_commit_verify"):
        raise ValueError("invalid RWWPO-2 transaction failure identity")
    if global_step < 1 or inner_id not in (1, 2) \
            or proposal_clock != 2 * (global_step - 1) + inner_id:
        raise ValueError("invalid RWWPO-2 transaction failure coordinate")
    if not math.isfinite(current_reference_max_abs) \
            or current_reference_max_abs < 0:
        raise ValueError("invalid RWWPO-2 transaction failure magnitude")
    if not prefix_rows or not prefix_stats:
        raise ValueError("RWWPO-2 transaction failure requires prefix evidence")
    if not all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in (
            behavior_batch_digest, transaction_entry_buffer_digest)):
        raise ValueError("invalid RWWPO-2 transaction failure digest")
    lexical_root = Path(ledger_dir)
    if lexical_root.is_symlink():
        raise ValueError("RWWPO failure ledger root cannot be a symlink")
    lexical_root.mkdir(parents=True, exist_ok=True)
    root = lexical_root.resolve()
    path = root / f"failure_rank{int(rank)}.jsonl"
    record = {
        "schema_version": "rwwpo2-transaction-failure-v1",
        "status": "NO_GO",
        "decision": "RWWPO2_TRANSACTION_FAILURE_PRESERVED",
        "attempt_id": str(attempt_id), "rank": int(rank),
        "global_step": global_step, "inner_id": inner_id,
        "proposal_clock": proposal_clock,
        "reason": str(reason), "phase": str(phase),
        "prefix_rows": list(prefix_rows),
        "prefix_stats": list(prefix_stats),
        "current_reference_max_abs": current_reference_max_abs,
        "behavior_batch_digest": str(behavior_batch_digest),
        "transaction_entry_buffer_digest": str(
            transaction_entry_buffer_digest),
        "diagnostics": dict(diagnostics or {}),
        "previous_record_sha256": _previous_tail(path),
    }
    _append_jsonl(path, record)
    return str(path)


def append_actual_loss_record(*, ledger_dir, attempt_id, mode, rank, global_step, epoch, minibatch,
                              old_log_prob, current_log_prob, response_mask,
                              proposed_post_log_prob, committed_log_prob,
                              writer_mask, answer_mask, trajectory_turn,
                              sample_index, example_identity_hash, trajectory_identity_hash,
                              advantages, denominator, prefix_stats,
                              prefix_rows, post_prefix_rows, post_prefix_stats, q_min,
                              writer_log_ratio_cap, constraint_pass, accepted,
                              root_q_min=None,
                              objective_variant="whole_prefix", controller_variant="hard_rollback",
                              alpha_grid=None, alpha_test_order=None, alpha_committed=1.0,
                              accepted_nonzero=True, proposal_zero=False, trial_evidence=None,
                              full_parameter_displacement_norm=0.0,
                              committed_parameter_displacement_norm=0.0,
                              pre_digests=None, commit_digests=None,
                              trial_forward_wall_seconds=0.0,
                              mechanism_diagnostics=None, gradient_norm=0.0,
                              program_version="legacy", inner_id=None, proposal_clock=None,
                              accepted_optimizer_clock_before=None,
                              accepted_optimizer_clock_after=None, logical_seed=None,
                              experiment_seed=None, host_variant="legacy",
                              behavior_batch_digest=None, ref_log_prob=None):
    if not ledger_dir:
        raise ValueError("RWWPO enabled without required ledger_dir")
    lexical_root = Path(ledger_dir)
    if lexical_root.is_symlink():
        raise ValueError("RWWPO ledger root cannot be a symlink")
    lexical_root.mkdir(parents=True, exist_ok=True)
    root = lexical_root.resolve()
    path = root / f"actual_loss_rank{int(rank)}.jsonl"
    previous = _previous_tail(path)
    trial_evidence = list(trial_evidence or [])
    alpha_grid = list(alpha_grid or [1.0])
    alpha_test_order = list(alpha_test_order or [1.0])
    tensor_v3 = str(program_version) == "rwwpo2-k2"
    if tensor_v3 and (inner_id not in (1, 2) or proposal_clock is None or logical_seed is None):
        raise ValueError("RWWPO-2 tensor ledger requires inner/proposal/seed coordinates")
    if tensor_v3 and ref_log_prob is None:
        raise ValueError("RWWPO-2 tensor ledger requires reference log probabilities")

    common = {
        "attempt_id": str(attempt_id), "mode": str(mode),
        "global_step": int(global_step), "rank": int(rank), "epoch": int(epoch),
        "minibatch": int(minibatch), "denominator": int(denominator),
        "prefix_rows": prefix_rows, "prefix_stats": prefix_stats,
        "post_prefix_rows": post_prefix_rows, "post_prefix_stats": post_prefix_stats,
        "q_min": float(q_min),
        "root_q_min": float(q_min if root_q_min is None else root_q_min),
        "writer_log_ratio_cap": float(writer_log_ratio_cap),
        "constraint_pass": bool(constraint_pass), "accepted": bool(accepted),
        "objective_variant": str(objective_variant),
        "controller_variant": str(controller_variant),
        "alpha_grid": alpha_grid, "alpha_test_order": alpha_test_order,
        "alpha_committed": float(alpha_committed),
        "accepted_nonzero": bool(accepted_nonzero),
        "proposal_zero": bool(proposal_zero),
        "full_parameter_displacement_norm": float(full_parameter_displacement_norm),
        "committed_parameter_displacement_norm": float(committed_parameter_displacement_norm),
        "pre_digests": dict(pre_digests or {}),
        "commit_digests": dict(commit_digests or {}),
        "trial_forward_wall_seconds": float(trial_forward_wall_seconds),
        "mechanism_diagnostics": mechanism_diagnostics or {},
        "gradient_norm": float(gradient_norm),
        "previous_record_sha256": previous,
    }
    writer_active = writer_mask.bool()
    if not writer_active.any():
        raise ValueError("RWWPO actual-loss ledger requires writer tokens")
    common["full_writer_logprob_movement"] = float(
        (proposed_post_log_prob - current_log_prob)[writer_active].detach().abs().max().item())
    common["committed_writer_logprob_movement"] = float(
        (committed_log_prob - current_log_prob)[writer_active].detach().abs().max().item())

    if tensor_v3:
        payload = {
            "old_log_prob": _cpu_tensor(old_log_prob),
            "current_log_prob": _cpu_tensor(current_log_prob),
            "ref_log_prob": _cpu_tensor(ref_log_prob),
            "proposed_post_log_prob": _cpu_tensor(proposed_post_log_prob),
            "committed_log_prob": _cpu_tensor(committed_log_prob),
            "response_mask": _cpu_tensor(response_mask.bool()),
            "writer_mask": _cpu_tensor(writer_mask.bool()),
            "answer_mask": _cpu_tensor(answer_mask.bool()),
            "trajectory_turn": _cpu_tensor(trajectory_turn),
            "sample_index": _cpu_tensor(sample_index),
            "example_identity_hash": _cpu_tensor(example_identity_hash),
            "trajectory_identity_hash": _cpu_tensor(trajectory_identity_hash),
            "advantages": _cpu_tensor(advantages),
        }
        small_trials = []
        for index, trial in enumerate(trial_evidence):
            trial = dict(trial)
            tensor = trial.pop("log_prob", None)
            if tensor is None:
                raise ValueError("RWWPO-2 trial is missing logprob tensor")
            key = f"trial_log_prob_{index:02d}"
            payload[key] = _cpu_tensor(tensor)
            trial["tensor_key"] = key
            small_trials.append(trial)
        relative, sha256, size, inventory = _write_tensor_shard_exclusive(
            root, rank=rank, global_step=global_step, inner_id=inner_id, payload=payload)
        record = {
            **common,
            "schema_version": TENSOR_LEDGER_VERSION,
            "program_version": "rwwpo2-k2",
            "inner_id": int(inner_id), "proposal_clock": int(proposal_clock),
            "accepted_optimizer_clock_before": int(accepted_optimizer_clock_before or 0),
            "accepted_optimizer_clock_after": int(accepted_optimizer_clock_after or 0),
            "logical_seed": int(logical_seed), "experiment_seed": int(experiment_seed),
            "host_variant": str(host_variant),
            "behavior_batch_digest": str(behavior_batch_digest),
            "trial_evidence": small_trials,
            "tensor_shard": {"relative_path": relative, "sha256": sha256,
                             "size": size, "inventory": inventory},
        }
    else:
        json_trials = []
        for trial in trial_evidence:
            trial = dict(trial)
            if torch.is_tensor(trial.get("log_prob")):
                trial["log_prob"] = _tolist(trial["log_prob"])
            json_trials.append(trial)
        record = {
            **common, "schema_version": LEDGER_VERSION,
            "old_log_prob": _tolist(old_log_prob),
            "current_log_prob": _tolist(current_log_prob),
            "proposed_post_log_prob": _tolist(proposed_post_log_prob),
            "committed_log_prob": _tolist(committed_log_prob),
            "response_mask": _tolist(response_mask.to(dtype=old_log_prob.dtype)),
            "writer_mask": _tolist(writer_mask.to(dtype=old_log_prob.dtype)),
            "answer_mask": _tolist(answer_mask.to(dtype=old_log_prob.dtype)),
            "trajectory_turn": _tolist(trajectory_turn), "sample_index": _tolist(sample_index),
            "example_identity_hash": _tolist(example_identity_hash),
            "trajectory_identity_hash": _tolist(trajectory_identity_hash),
            "advantages": _tolist(advantages), "trial_evidence": json_trials,
        }
    _append_jsonl(path, record)
    return str(path)
