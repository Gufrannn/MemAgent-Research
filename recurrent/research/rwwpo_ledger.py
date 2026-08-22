"""Append-only actual-loss evidence for Paper I RWWPO."""
import hashlib
import json
import os
from pathlib import Path


LEDGER_VERSION = "rwwpo-actual-loss-v2"


def _tolist(tensor):
    return tensor.detach().to("cpu").tolist()


def append_transaction_marker(*, ledger_dir, attempt_id, rank, global_step, epoch, minibatch,
                              phase, model_digest):
    if phase not in ("intent", "complete"):
        raise ValueError("bad transaction marker phase")
    path=Path(ledger_dir).resolve()/f"transaction_rank{int(rank)}.jsonl"
    path.parent.mkdir(parents=True,exist_ok=True)
    previous="0"*64
    if path.exists():
        lines=[line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines: previous=json.loads(lines[-1])["record_sha256"]
    record={"schema_version":"rwwpo-transaction-v1","attempt_id":str(attempt_id),
            "rank":int(rank),"global_step":int(global_step),"epoch":int(epoch),
            "minibatch":int(minibatch),"phase":phase,"model_digest":str(model_digest),
            "previous_record_sha256":previous}
    canonical=json.dumps(record,sort_keys=True,separators=(",",":"))
    record["record_sha256"]=hashlib.sha256(canonical.encode()).hexdigest()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    try:
        os.write(fd,(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n").encode()); os.fsync(fd)
    finally: os.close(fd)


def append_actual_loss_record(*, ledger_dir, attempt_id, mode, rank, global_step, epoch, minibatch,
                              old_log_prob, current_log_prob, response_mask,
                              proposed_post_log_prob, committed_log_prob,
                              writer_mask, answer_mask, trajectory_turn,
                              sample_index, example_identity_hash, trajectory_identity_hash,
                              advantages, denominator, prefix_stats,
                              prefix_rows, post_prefix_rows, post_prefix_stats, q_min,
                              writer_log_ratio_cap, constraint_pass, accepted,
                              objective_variant="whole_prefix", controller_variant="hard_rollback",
                              alpha_grid=None, alpha_test_order=None, alpha_committed=1.0,
                              accepted_nonzero=True, proposal_zero=False, trial_evidence=None,
                              full_parameter_displacement_norm=0.0,
                              committed_parameter_displacement_norm=0.0,
                              pre_digests=None, commit_digests=None,
                              trial_forward_wall_seconds=0.0,
                              mechanism_diagnostics=None, gradient_norm=0.0):
    if not ledger_dir:
        raise ValueError("RWWPO enabled without required ledger_dir")
    root = Path(ledger_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"actual_loss_rank{int(rank)}.jsonl"
    previous = "0" * 64
    if path.exists():
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            previous = json.loads(lines[-1])["record_sha256"]
    record = {
        "schema_version": LEDGER_VERSION,
        "attempt_id": str(attempt_id), "mode": str(mode), "global_step": int(global_step), "rank": int(rank), "epoch": int(epoch),
        "minibatch": int(minibatch), "old_log_prob": _tolist(old_log_prob),
        "current_log_prob": _tolist(current_log_prob),
        "proposed_post_log_prob": _tolist(proposed_post_log_prob),
        "committed_log_prob": _tolist(committed_log_prob),
        "response_mask": _tolist(response_mask.to(dtype=old_log_prob.dtype)),
        "writer_mask": _tolist(writer_mask.to(dtype=old_log_prob.dtype)),
        "answer_mask": _tolist(answer_mask.to(dtype=old_log_prob.dtype)),
        "trajectory_turn": _tolist(trajectory_turn), "sample_index": _tolist(sample_index),
        "example_identity_hash": _tolist(example_identity_hash),
        "trajectory_identity_hash": _tolist(trajectory_identity_hash),
        "advantages": _tolist(advantages), "denominator": int(denominator),
        "prefix_rows": prefix_rows, "prefix_stats": prefix_stats,
        "post_prefix_rows": post_prefix_rows, "post_prefix_stats": post_prefix_stats, "q_min": float(q_min),
        "writer_log_ratio_cap": float(writer_log_ratio_cap),
        "constraint_pass": bool(constraint_pass), "accepted": bool(accepted),
        "objective_variant": str(objective_variant),
        "controller_variant": str(controller_variant),
        "alpha_grid": list(alpha_grid or [1.0]),
        "alpha_test_order": list(alpha_test_order or [1.0]),
        "alpha_committed": float(alpha_committed),
        "accepted_nonzero": bool(accepted_nonzero),
        "proposal_zero": bool(proposal_zero),
        "trial_evidence": trial_evidence or [],
        "full_parameter_displacement_norm": float(full_parameter_displacement_norm),
        "committed_parameter_displacement_norm": float(committed_parameter_displacement_norm),
        "full_writer_logprob_movement": max(
            abs(float(p)-float(c)) for post,cur,mask in zip(_tolist(proposed_post_log_prob),
            _tolist(current_log_prob),_tolist(writer_mask)) for p,c,m in zip(post,cur,mask) if bool(m)),
        "committed_writer_logprob_movement": max(
            abs(float(p)-float(c)) for post,cur,mask in zip(_tolist(committed_log_prob),
            _tolist(current_log_prob),_tolist(writer_mask)) for p,c,m in zip(post,cur,mask) if bool(m)),
        "pre_digests": dict(pre_digests or {}),
        "commit_digests": dict(commit_digests or {}),
        "trial_forward_wall_seconds": float(trial_forward_wall_seconds),
        "mechanism_diagnostics": mechanism_diagnostics or {},
        "gradient_norm": float(gradient_norm),
        "previous_record_sha256": previous,
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["record_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)
    return str(path)
