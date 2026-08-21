"""Append-only actual-loss evidence for Paper I RWWPO."""
import hashlib
import json
import os
from pathlib import Path


LEDGER_VERSION = "rwwpo-actual-loss-v1"


def _tolist(tensor):
    return tensor.detach().to("cpu").tolist()


def append_actual_loss_record(*, ledger_dir, attempt_id, mode, rank, global_step, epoch, minibatch,
                              old_log_prob, current_log_prob, response_mask,
                              proposed_post_log_prob,
                              writer_mask, answer_mask, trajectory_turn,
                              sample_index, advantages, denominator, prefix_stats,
                              prefix_rows, post_prefix_rows, post_prefix_stats, q_min,
                              writer_log_ratio_cap, constraint_pass, accepted):
    if not ledger_dir:
        raise ValueError("RWWPO enabled without required ledger_dir")
    root = Path(ledger_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"actual_loss_rank{int(rank)}.jsonl"
    record = {
        "schema_version": LEDGER_VERSION,
        "attempt_id": str(attempt_id), "mode": str(mode), "global_step": int(global_step), "rank": int(rank), "epoch": int(epoch),
        "minibatch": int(minibatch), "old_log_prob": _tolist(old_log_prob),
        "current_log_prob": _tolist(current_log_prob),
        "proposed_post_log_prob": _tolist(proposed_post_log_prob),
        "response_mask": _tolist(response_mask.to(dtype=old_log_prob.dtype)),
        "writer_mask": _tolist(writer_mask.to(dtype=old_log_prob.dtype)),
        "answer_mask": _tolist(answer_mask.to(dtype=old_log_prob.dtype)),
        "trajectory_turn": _tolist(trajectory_turn), "sample_index": _tolist(sample_index),
        "advantages": _tolist(advantages), "denominator": int(denominator),
        "prefix_rows": prefix_rows, "prefix_stats": prefix_stats,
        "post_prefix_rows": post_prefix_rows, "post_prefix_stats": post_prefix_stats, "q_min": float(q_min),
        "writer_log_ratio_cap": float(writer_log_ratio_cap),
        "constraint_pass": bool(constraint_pass), "accepted": bool(accepted),
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
