#!/usr/bin/env python3
"""Read-only reconstruction of RWWPO actual-loss evidence."""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

# Direct-file invocations put tools/h20, rather than the repository root, on
# sys.path.  Bootstrap the authenticated repository before importing project
# modules so the formal user-facing auditor does not depend on ambient
# PYTHONPATH or an editable install.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from recurrent.research.rwwpo_transaction import (
    RWWPO2_FSDP_PARAMETER_COMMIT_PRIMITIVE, digest,
    RWWPO2_FSDP_WRITEBACK_MAX_WALL_SECONDS,
    largest_tested_feasible, logical_transaction_seed,
    prefix_distribution_stats, proposal_clock,
    tensor_content_digest,
)


MODERN_SCHEMAS = {"rwwpo-actual-loss-v2", "rwwpo-actual-loss-v3"}
_V3_VALIDATOR = None


def _masked_mean(values, mask):
    denominator = mask.sum()
    if int(denominator.item()) <= 0:
        return values.sum() * 0.0
    return (values * mask).sum() / denominator


def _independent_tokenwise_ppo(old, current, advantages, mask, contract):
    ratio = torch.exp(current - old)
    clipped = torch.clamp(
        ratio, 1.0 - float(contract["cliprange_low"]),
        1.0 + float(contract["cliprange_high"]),
    )
    loss = torch.maximum(-advantages * ratio, -advantages * clipped)
    loss = torch.where(
        advantages < 0,
        torch.minimum(-advantages * float(contract["clip_ratio_c"]), loss),
        loss,
    )
    return _masked_mean(loss, mask)


def _independent_writer_surrogate(row, old, current, advantages, response, writer, answer,
                                  contract):
    objective = row["objective_variant"]
    if objective == "original_tokenwise":
        return _independent_tokenwise_ppo(
            old, current, advantages, response, contract)

    denominator = response.sum()
    answer_loss = _independent_tokenwise_ppo(
        old, current, advantages, answer, contract)
    answer_numerator = answer_loss * answer.sum()
    sample_index = [int(value) for value in row["sample_index"]]
    turns = [int(value) for value in row["trajectory_turn"]]
    writer_rows = [index for index in range(len(sample_index)) if bool(writer[index].any())]
    trajectory_losses = []
    cap = float(contract["writer_log_ratio_cap"])
    for sid in sorted(set(sample_index[index] for index in writer_rows)):
        indices = sorted(
            (index for index in writer_rows if sample_index[index] == sid),
            key=lambda index: turns[index],
        )
        final_advantage = advantages[indices[-1]][writer[indices[-1]]].mean()
        if objective == "whole_prefix":
            log_ratio = sum((current[index] - old[index])[writer[index]].sum()
                            for index in indices)
            trajectory_losses.append(-torch.exp(torch.clamp(log_ratio, -cap, cap))
                                     * final_advantage)
        elif objective == "per_write_joint":
            for index in indices:
                log_ratio = (current[index] - old[index])[writer[index]].sum()
                trajectory_losses.append(-torch.exp(torch.clamp(log_ratio, -cap, cap))
                                         * final_advantage)
        else:
            raise ValueError("RWWPO-2 independent audit saw unknown objective")
    if not trajectory_losses:
        raise ValueError("RWWPO-2 independent audit found no writer trajectory")
    return (torch.stack(trajectory_losses).sum() + answer_numerator) / denominator


def independently_recompute_actual_loss(row):
    """Rebuild the complete actor loss without calling the producer loss helpers."""
    diagnostics = row["mechanism_diagnostics"]
    contract = diagnostics["actual_loss_contract"]
    old = torch.tensor(row["old_log_prob"], dtype=torch.float64)
    current = torch.tensor(
        row["current_log_prob"], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor(row["ref_log_prob"], dtype=torch.float64)
    advantages = torch.tensor(row["advantages"], dtype=torch.float64)
    response = torch.tensor(row["response_mask"], dtype=torch.bool)
    writer = torch.tensor(row["writer_mask"], dtype=torch.bool)
    answer = torch.tensor(row["answer_mask"], dtype=torch.bool)
    surrogate = _independent_writer_surrogate(
        row, old, current, advantages, response, writer, answer, contract)
    if contract != {
            "loss_agg_mode": "token-mean", "cliprange": 0.2,
            "cliprange_low": 0.2, "cliprange_high": 0.2,
            "clip_ratio_c": 3.0, "writer_log_ratio_cap": 4.0,
            "use_kl_loss": True, "kl_loss_type": "low_var_kl",
            "kl_loss_coefficient": 0.001, "entropy_coefficient": 0.0}:
        raise ValueError("RWWPO-2 actual-loss contract drift")
    kl = reference - current
    low_variance_kl = torch.clamp(torch.exp(kl) - kl - 1.0, min=-10.0, max=10.0)
    shared_kl = _masked_mean(low_variance_kl, response) * 0.001
    total = surrogate + shared_kl
    gradient, = torch.autograd.grad(total, current)
    return {
        "surrogate_loss": float(surrogate.detach().item()),
        "shared_kl_loss": float(shared_kl.detach().item()),
        "policy_loss": float(total.detach().item()),
        "active_logprob_gradient_l2": float(
            gradient[response].detach().norm().item()),
    }


def validate_v3_schema(receipt):
    global _V3_VALIDATOR
    if _V3_VALIDATOR is None:
        import jsonschema
        schema_path = ROOT / "rwwpo2_actual_loss_receipt.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        _V3_VALIDATOR = jsonschema.Draft202012Validator(schema)
    errors = sorted(_V3_VALIDATOR.iter_errors(receipt), key=lambda item: list(item.path))
    if errors:
        raise ValueError("RWWPO-2 receipt schema failure: " + errors[0].message)


def validate_rwwpo2_rng_phase_digests(row):
    """Bind diagnostic RNG phases to the authenticated transaction state."""
    diagnostics = row.get("mechanism_diagnostics", {})
    rng_fields = (
        "transaction_entry_rng_digest",
        "logical_seeded_rng_digest",
        "proposal_gradient_rng_digest",
        "terminal_rng_digest",
    )
    if any(not isinstance(diagnostics.get(field), str)
           or re.fullmatch(r"[0-9a-f]{64}", diagnostics[field]) is None
           for field in rng_fields) \
            or diagnostics["transaction_entry_rng_digest"] != \
                row.get("pre_digests", {}).get("rng") \
            or diagnostics["terminal_rng_digest"] != \
                row.get("commit_digests", {}).get("rng"):
        raise ValueError("RWWPO-2 RNG phase digest closure")
    if not bool(row.get("accepted_nonzero")) \
            and diagnostics["terminal_rng_digest"] != \
                diagnostics["transaction_entry_rng_digest"]:
        raise ValueError("RWWPO-2 rejected transaction RNG rollback")
    replay_digests = diagnostics.get("behavior_forward_rng_digests")
    replay_aggregate = diagnostics.get(
        "behavior_forward_rng_aggregate_digest")
    if diagnostics.get("replay_rng_bound") is not True \
            or not isinstance(replay_digests, list) or not replay_digests \
            or any(not isinstance(value, str) or re.fullmatch(
                r"[0-9a-f]{64}", value) is None for value in replay_digests) \
            or diagnostics.get("replay_microbatch_count") != len(replay_digests) \
            or not isinstance(replay_aggregate, str) \
            or re.fullmatch(r"[0-9a-f]{64}", replay_aggregate) is None \
            or replay_aggregate != digest(replay_digests):
        raise ValueError("RWWPO-2 behavior replay RNG binding")


def hydrate_authenticated_v3_receipt(receipt, ledger_path):
    ledger_root = Path(ledger_path).resolve().parent
    evidence = receipt.get("tensor_shard", {})
    relative = evidence.get("relative_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("tensor shard path is not relative")
    shard = ledger_root / relative
    resolved = shard.resolve()
    if ledger_root not in resolved.parents or shard.is_symlink() or not shard.is_file():
        raise ValueError("tensor shard path escape/symlink/missing")
    raw = shard.read_bytes()
    if len(raw) != int(evidence.get("size", -1)):
        raise ValueError("tensor shard size mismatch")
    if hashlib.sha256(raw).hexdigest() != evidence.get("sha256"):
        raise ValueError("tensor shard hash mismatch")
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or any(not torch.is_tensor(value) for value in payload.values()):
        raise ValueError("tensor shard payload is not tensor-only")
    actual_inventory = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(payload.items())
    }
    if actual_inventory != evidence.get("inventory"):
        raise ValueError("tensor shard inventory mismatch")
    required = {"old_log_prob", "current_log_prob", "ref_log_prob",
                "proposed_post_log_prob",
                "committed_log_prob", "response_mask", "writer_mask", "answer_mask",
                "trajectory_turn", "sample_index", "example_identity_hash",
                "trajectory_identity_hash", "advantages"}
    trial_keys = {str(item.get("tensor_key")) for item in receipt.get("trial_evidence", [])}
    if set(payload) != required | trial_keys or any(not key.startswith("trial_log_prob_") for key in trial_keys):
        raise ValueError("tensor shard required/trial key closure failure")
    diagnostics = receipt.get("mechanism_diagnostics", {})
    if tensor_content_digest(payload["current_log_prob"]) != diagnostics.get(
            "behavior_current_logprob_digest"):
        raise ValueError("RWWPO-2 immutable behavior logprob digest mismatch")
    row = dict(receipt)
    for key in required:
        row[key] = payload[key].tolist()
    # Keep authenticated tensors by reference for evidence reconstruction that
    # must preserve the producer's dtype and Torch reduction semantics.  The
    # JSON-compatible lists above remain the public/audit representation.
    row["_authenticated_tensors"] = payload
    trials = []
    for item in receipt.get("trial_evidence", []):
        item = dict(item)
        key = item.pop("tensor_key")
        item["log_prob"] = payload[key].tolist()
        item["_authenticated_log_prob_tensor"] = payload[key]
        trials.append(item)
    row["trial_evidence"] = trials
    return row


def reconstruct_authenticated_prefix_rows(row, log_prob_tensor):
    """Rebuild producer prefix rows with the authenticated tensor dtype.

    The producer computes each writer-row reduction and the trajectory prefix
    with Torch in the log-probability dtype.  Converting shard values to Python
    floats first silently changes float32 accumulation to float64 and can reject
    an otherwise exact checkpoint prefix by one float32 ULP.
    """
    payload = row.get("_authenticated_tensors")
    if not isinstance(payload, dict) or not torch.is_tensor(log_prob_tensor):
        raise ValueError("authenticated tensor payload missing for prefix reconstruction")
    old = payload["old_log_prob"]
    writer_mask = payload["writer_mask"].bool()
    sample_index = payload["sample_index"]
    trajectory_turn = payload["trajectory_turn"]
    if log_prob_tensor.shape != old.shape or writer_mask.shape != old.shape:
        raise ValueError("authenticated prefix tensor shape mismatch")
    if log_prob_tensor.dtype != old.dtype:
        raise ValueError("authenticated prefix logprob dtype mismatch")
    writer_rows = torch.nonzero(writer_mask.any(dim=-1), as_tuple=False).flatten()
    writer_row_log_ratio = ((log_prob_tensor - old) * writer_mask).sum(dim=-1)
    reconstructed = []
    for sid in torch.unique(sample_index[writer_rows], sorted=True):
        indices = writer_rows[sample_index[writer_rows] == sid]
        order = torch.argsort(trajectory_turn[indices], stable=True)
        indices = indices[order]
        prefix = torch.cumsum(writer_row_log_ratio[indices], dim=0)
        token_count = torch.cumsum(writer_mask[indices].sum(dim=-1), dim=0)
        for index, value, tokens in zip(indices, prefix, token_count):
            reconstructed.append({
                "turn": int(trajectory_turn[index].item()),
                "sample_index": int(sid.item()),
                "log_ratio": float(value.item()),
                "prefix_token_count": int(tokens.item()),
            })
    return reconstructed


def prefix_rows_match(declared, reconstructed, *, dtype=None):
    declared = sorted(declared, key=lambda item: (item["sample_index"], item["turn"]))
    reconstructed = sorted(
        reconstructed, key=lambda item: (item["sample_index"], item["turn"]))
    if len(declared) != len(reconstructed):
        return False
    for expected, actual in zip(declared, reconstructed):
        if any(expected[field] != actual[field] for field in (
                "turn", "sample_index", "prefix_token_count")):
            return False
        if dtype is None:
            if not math.isclose(float(expected["log_ratio"]),
                                float(actual["log_ratio"]),
                                rel_tol=1e-9, abs_tol=1e-10):
                return False
        else:
            # JSON stores the producer's scalar exactly as a Python number.
            # Round both sides back into the authenticated tensor dtype and
            # demand bit-identical values; this is not a tolerance relaxation.
            expected_value = torch.tensor(float(expected["log_ratio"]), dtype=dtype)
            actual_value = torch.tensor(float(actual["log_ratio"]), dtype=dtype)
            if not torch.equal(expected_value, actual_value):
                return False
    return True


def canonical_sha(record):
    payload = dict(record)
    payload.pop("record_sha256", None)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def audit(paths, require_method=True, *, start_round=None, through_round=None,
          record_limits=None):
    if start_round is not None and int(start_round) < 1:
        raise ValueError("invalid actual-loss audit start round")
    if through_round is not None and int(through_round) < 1:
        raise ValueError("invalid actual-loss audit terminal round")
    if start_round is not None and through_round is not None \
            and int(start_round) > int(through_round):
        raise ValueError("actual-loss audit round range is empty")
    rows, seen = [], set()
    for path in paths:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"actual-loss ledger is missing or a symlink: {path}")
        limit = None if record_limits is None else record_limits.get(path.name)
        if limit is not None and int(limit) < 1:
            raise ValueError(f"invalid actual-loss prefix limit for {path.name}")
        previous = "0" * 64
        selected_record_count = 0
        with path.open(encoding="utf-8") as handle:
            for physical_line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                if limit is not None and selected_record_count >= int(limit):
                    break
                selected_record_count += 1
                line_no = physical_line_no
                receipt = json.loads(line)
                if receipt.get("schema_version") not in ("rwwpo-actual-loss-v1", *MODERN_SCHEMAS):
                    raise ValueError(f"bad schema at {path}:{line_no}")
                if receipt.get("schema_version") == "rwwpo-actual-loss-v3":
                    validate_v3_schema(receipt)
                if receipt.get("mode") not in ("rwwpo_method", "original_collection"):
                    raise ValueError(f"bad mode at {path}:{line_no}")
                if receipt["schema_version"] in MODERN_SCHEMAS:
                    expected_rank=int(Path(path).stem.rsplit("rank",1)[1])
                    if int(receipt.get("rank",-1))!=expected_rank:
                        raise ValueError(f"rank/file identity mismatch at {path}:{line_no}")
                if canonical_sha(receipt) != receipt.get("record_sha256"):
                    raise ValueError(f"record hash mismatch at {path}:{line_no}")
                if receipt.get("schema_version") in MODERN_SCHEMAS:
                    if receipt.get("previous_record_sha256") != previous:
                        raise ValueError(f"append-only hash chain mismatch at {path}:{line_no}")
                    previous=receipt["record_sha256"]
                receipt_round = int(receipt["global_step"])
                if start_round is not None and receipt_round < int(start_round):
                    continue
                if through_round is not None and receipt_round > int(through_round):
                    continue
                row = (hydrate_authenticated_v3_receipt(receipt, path)
                       if receipt["schema_version"] == "rwwpo-actual-loss-v3" else receipt)
                identity = (row["attempt_id"], row["global_step"], row["rank"], row["epoch"], row["minibatch"])
                if identity in seen:
                    raise ValueError(f"duplicate optimizer identity: {identity}")
                seen.add(identity)
                shape_fields = [
                    "old_log_prob", "current_log_prob", "proposed_post_log_prob",
                    "response_mask", "writer_mask", "answer_mask", "advantages",
                ]
                if row["schema_version"] == "rwwpo-actual-loss-v3":
                    shape_fields.append("ref_log_prob")
                shapes = [len(row[key]) for key in shape_fields]
                if len(set(shapes)) != 1 or shapes[0] != len(row["sample_index"]):
                    raise ValueError("row/tensor alignment failure")
                if row["schema_version"] in MODERN_SCHEMAS:
                    if len(row.get("example_identity_hash",[]))!=shapes[0] or len(row.get("trajectory_identity_hash",[]))!=shapes[0]:
                        raise ValueError("stable identity alignment failure")
                    for sid in set(row["sample_index"]):
                        idx=[i for i,value in enumerate(row["sample_index"]) if value==sid]
                        if len({row["example_identity_hash"][i] for i in idx})!=1 or len({row["trajectory_identity_hash"][i] for i in idx})!=1:
                            raise ValueError("stable trajectory identity drift")
                denominator = 0
                for response, writer, answer in zip(row["response_mask"], row["writer_mask"], row["answer_mask"]):
                    if not (len(response) == len(writer) == len(answer)):
                        raise ValueError("token shape failure")
                    for r, w, a in zip(response, writer, answer):
                        if bool(r) != (bool(w) ^ bool(a)):
                            raise ValueError("role mask closure failure")
                        denominator += int(bool(r))
                if denominator != row["denominator"]:
                    raise ValueError("denominator mismatch")
                numeric_matrices = (
                    "old_log_prob", "current_log_prob", "proposed_post_log_prob",
                    "committed_log_prob", "advantages",
                ) if row["schema_version"] in MODERN_SCHEMAS else (
                    "old_log_prob", "current_log_prob", "proposed_post_log_prob",
                    "advantages",
                )
                if row["schema_version"] == "rwwpo-actual-loss-v3":
                    numeric_matrices = (*numeric_matrices, "ref_log_prob")
                if any(not math.isfinite(float(value))
                       for field in numeric_matrices
                       for tensor_row, active_row in zip(row[field], row["response_mask"])
                       for value, active in zip(tensor_row, active_row) if bool(active)):
                    raise ValueError("non-finite active-token actual-loss tensor")
                if row["schema_version"] in MODERN_SCHEMAS:
                    allowed_objectives = (("whole_prefix", "per_write_joint", "original_tokenwise")
                                          if row["schema_version"] == "rwwpo-actual-loss-v3"
                                          else ("whole_prefix", "original_tokenwise"))
                    allowed_controllers = (("none", "hard_rollback", "feasible_backtracking")
                                           if row["schema_version"] == "rwwpo-actual-loss-v3"
                                           else ("hard_rollback", "feasible_backtracking"))
                    if row.get("objective_variant") not in allowed_objectives:
                        raise ValueError("objective variant identity drift")
                    if row.get("controller_variant") not in allowed_controllers:
                        raise ValueError("controller variant identity drift")
                    scalar_fields = (
                        "alpha_committed", "full_parameter_displacement_norm",
                        "committed_parameter_displacement_norm",
                        "full_writer_logprob_movement",
                        "committed_writer_logprob_movement",
                        "trial_forward_wall_seconds", "gradient_norm",
                    )
                    if any(not math.isfinite(float(row.get(field, float("nan"))))
                           or float(row.get(field, -1)) < 0 for field in scalar_fields):
                        raise ValueError("non-finite/negative transaction scalar")
                    frozen=[1.0,0.5,0.25,0.125,0.0625,0.03125]
                    if row.get("controller_variant")=="feasible_backtracking" and row.get("alpha_grid")!=frozen:
                        raise ValueError("alpha grid drift")
                    alpha=float(row.get("alpha_committed",-1))
                    if alpha not in (0.0,*frozen): raise ValueError("invalid committed alpha")
                    if bool(row.get("accepted_nonzero")) != (alpha>0 and not bool(row.get("proposal_zero"))):
                        raise ValueError("zero proposal or accepted decision mismatch")
                    if bool(row.get("accepted")) != bool(row.get("accepted_nonzero")):
                        raise ValueError("legacy/new accepted decision mismatch")
                    required={"model","optimizer","scheduler","scaler","rng"}
                    if set(row.get("pre_digests",{}))!=required or set(row.get("commit_digests",{}))!=required:
                        raise ValueError("transaction digest closure failure")
                    if row["schema_version"] == "rwwpo-actual-loss-v3":
                        validate_rwwpo2_rng_phase_digests(row)
                    if alpha==0:
                        for key in required:
                            if row["pre_digests"][key] != row["commit_digests"][key]:
                                raise ValueError("rollback digest mismatch")
                    if row["schema_version"] == "rwwpo-actual-loss-v3":
                        if row.get("program_version") != "rwwpo2-k2":
                            raise ValueError("RWWPO-2 program identity drift")
                        diagnostics = row.get("mechanism_diagnostics", {})
                        buffer_digests = (
                            diagnostics.get("transaction_entry_buffer_digest"),
                            diagnostics.get("terminal_buffer_digest"),
                        )
                        if diagnostics.get("optimizer_step_calls") != 1 \
                                or diagnostics.get(
                                    "post_commit_forward_verified") is not True \
                                or any(not math.isfinite(float(diagnostics.get(field, float("nan"))))
                                       for field in ("policy_loss", "surrogate_loss",
                                                     "shared_kl_loss",
                                                     "active_logprob_gradient_l2",
                                                     "post_commit_forward_verification_max_abs",
                                                     "post_commit_forward_verification_tolerance")) \
                                or float(diagnostics.get("active_logprob_gradient_l2", -1)) < 0 \
                                or float(diagnostics.get(
                                    "post_commit_forward_verification_max_abs", -1)) < 0 \
                                or float(diagnostics.get(
                                    "post_commit_forward_verification_max_abs", float("inf"))) > \
                                    float(diagnostics.get(
                                        "post_commit_forward_verification_tolerance", -1)) \
                                or any(not isinstance(value, str) or not re.fullmatch(
                                    r"[0-9a-f]{64}", value) for value in buffer_digests) \
                                or buffer_digests[0] != buffer_digests[1]:
                            raise ValueError("RWWPO-2 loss/optimizer-step evidence")
                        if diagnostics.get(
                                "fsdp_parameter_commit_primitive") != \
                                RWWPO2_FSDP_PARAMETER_COMMIT_PRIMITIVE \
                                or diagnostics.get(
                                    "behavior_current_logprob_integrity_verified") is not True \
                                or float(diagnostics.get(
                                    "behavior_current_logprob_integrity_max_abs", -1)) != 0.0:
                            raise ValueError(
                                "RWWPO-2 FSDP/behavior-reference closure")
                        writeback_times = diagnostics.get(
                            "fsdp_parameter_writeback_wall_seconds")
                        if float(diagnostics.get(
                                "fsdp_parameter_writeback_max_wall_seconds", -1)) != \
                                RWWPO2_FSDP_WRITEBACK_MAX_WALL_SECONDS \
                                or not isinstance(writeback_times, list) \
                                or not 1 <= len(writeback_times) <= 7 \
                                or any(not math.isfinite(float(value))
                                       or not 0.0 <= float(value) <=
                                       RWWPO2_FSDP_WRITEBACK_MAX_WALL_SECONDS
                                       for value in writeback_times) \
                                or float(diagnostics.get(
                                    "max_trial_forward_wall_seconds", -1)) != 600.0 \
                                or not math.isfinite(float(receipt.get(
                                    "trial_forward_wall_seconds", float("nan")))) \
                                or not 0.0 <= float(receipt.get(
                                    "trial_forward_wall_seconds", -1)) <= 600.0:
                            raise ValueError(
                                "RWWPO-2 FSDP/trial wall-time closure")
                        recomputed_loss = independently_recompute_actual_loss(row)
                        for field, actual_value in recomputed_loss.items():
                            if not math.isclose(
                                    float(diagnostics[field]), float(actual_value),
                                    rel_tol=2e-5, abs_tol=2e-6):
                                raise ValueError(
                                    f"RWWPO-2 independently recomputed actual loss mismatch: {field}")
                        inner = int(row.get("inner_id", 0))
                        if inner != int(row["epoch"]) + 1:
                            raise ValueError("RWWPO-2 epoch/inner transaction mismatch")
                        expected_proposal = proposal_clock(int(row["global_step"]), inner)
                        if int(row.get("proposal_clock", 0)) != expected_proposal:
                            raise ValueError("RWWPO-2 proposal clock mismatch")
                        expected_seed = logical_transaction_seed(
                            experiment_seed=int(row["experiment_seed"]),
                            round_id=int(row["global_step"]), inner_id=inner,
                            rank=int(row["rank"]), stream="actor_transaction")
                        if int(row.get("logical_seed", -1)) != expected_seed:
                            raise ValueError("RWWPO-2 logical seed mismatch")
                        before = int(row.get("accepted_optimizer_clock_before", -1))
                        after = int(row.get("accepted_optimizer_clock_after", -1))
                        if after != before + int(bool(row["accepted"])):
                            raise ValueError("RWWPO-2 accepted optimizer clock mismatch")
                        if row.get("host_variant") not in ("A", "B", "C", "D", "E"):
                            raise ValueError("RWWPO-2 host variant missing")
                        if not isinstance(row.get("behavior_batch_digest"), str) or len(row["behavior_batch_digest"]) != 64:
                            raise ValueError("RWWPO-2 behavior batch digest malformed")
                for index in range(len(row["sample_index"])):
                    active=[j for j,value in enumerate(row["writer_mask"][index]) if bool(value)]
                    if not active:
                        continue
                    advantages={round(float(row["advantages"][index][j]),12) for j in active}
                    if len(advantages)!=1:
                        raise ValueError("writer advantage is not scalar within a write")
                if row["schema_version"] == "rwwpo-actual-loss-v3":
                    payload = row["_authenticated_tensors"]
                    recomputed = reconstruct_authenticated_prefix_rows(
                        row, payload["current_log_prob"])
                    prefix_dtype = payload["current_log_prob"].dtype
                else:
                    recomputed=[]
                    for sid in sorted(set(row["sample_index"])):
                        indices=[i for i,value in enumerate(row["sample_index"]) if value==sid and any(bool(x) for x in row["writer_mask"][i])]
                        indices.sort(key=lambda i:row["trajectory_turn"][i]); running=0.0; tokens=0
                        for index in indices:
                            active=[j for j,value in enumerate(row["writer_mask"][index]) if bool(value)]
                            running += sum(float(row["current_log_prob"][index][j])-float(row["old_log_prob"][index][j]) for j in active)
                            tokens += len(active)
                            recomputed.append({"turn":int(row["trajectory_turn"][index]),"sample_index":int(sid),"log_ratio":running,"prefix_token_count":tokens})
                    prefix_dtype = None
                if not prefix_rows_match(row["prefix_rows"], recomputed,
                                         dtype=prefix_dtype):
                    raise ValueError("prefix rows do not reconstruct from actual-loss tensors")
                if row["schema_version"]=="rwwpo-actual-loss-v3":
                    root_by_sample={}
                    for sid,root in zip(row["sample_index"],row["example_identity_hash"]):
                        root_by_sample.setdefault(int(sid),str(root))
                        if root_by_sample[int(sid)]!=str(root):
                            raise ValueError("root identity drift inside trajectory")
                    if any(str(item.get("root_identity_hash"))!=root_by_sample.get(
                            int(item["sample_index"])) for item in row["prefix_rows"]):
                        raise ValueError("prefix/root tensor identity mismatch")
                post_logprob_field = ("committed_log_prob"
                                      if row["schema_version"] in MODERN_SCHEMAS
                                      else "proposed_post_log_prob")
                if row["schema_version"] == "rwwpo-actual-loss-v3":
                    post_tensor = row["_authenticated_tensors"][post_logprob_field]
                    post_actual = reconstruct_authenticated_prefix_rows(row, post_tensor)
                    post_dtype = post_tensor.dtype
                else:
                    post_actual=[]
                    for sid in sorted(set(row["sample_index"])):
                        indices=[i for i,value in enumerate(row["sample_index"]) if value==sid and any(bool(x) for x in row["writer_mask"][i])]
                        indices.sort(key=lambda i:row["trajectory_turn"][i]); running=0.0; tokens=0
                        for index in indices:
                            active=[j for j,value in enumerate(row["writer_mask"][index]) if bool(value)]
                            running += sum(float(row[post_logprob_field][index][j])-float(row["old_log_prob"][index][j]) for j in active); tokens += len(active)
                            post_actual.append({"turn":int(row["trajectory_turn"][index]),"sample_index":int(sid),"log_ratio":running,"prefix_token_count":tokens})
                    post_dtype = None
                if not prefix_rows_match(row["post_prefix_rows"], post_actual,
                                         dtype=post_dtype):
                    raise ValueError("post-step prefix rows do not reconstruct")
                if row["schema_version"] in MODERN_SCHEMAS:
                    order=row["alpha_test_order"]
                    evidence=row["trial_evidence"]
                    if order != [item.get("alpha") for item in evidence]:
                        raise ValueError("trial test order/evidence mismatch")
                    if not order or order[0] != 1.0 or any(a not in row["alpha_grid"] for a in order):
                        raise ValueError("trial alpha evidence malformed")
                    declared_alpha=float(row["alpha_committed"])
                    if row["controller_variant"] == "feasible_backtracking":
                        try:
                            decision = largest_tested_feasible(
                                [(item["alpha"], item.get("feasible"))
                                 for item in evidence],
                                proposal_zero=bool(row["proposal_zero"]),
                                alpha_grid=row["alpha_grid"])
                        except ValueError as exc:
                            raise ValueError(
                                f"invalid canonical backtracking evidence: {exc}") from exc
                        expected_alpha = decision.alpha
                    elif row["controller_variant"] == "none":
                        if order != [1.0]:
                            raise ValueError("no-controller trial evidence must contain alpha=1 only")
                        expected_alpha = 0.0 if row["proposal_zero"] else 1.0
                    else:
                        if order != [1.0]:
                            raise ValueError("hard-rollback trial evidence must contain alpha=1 only")
                        expected_alpha = (0.0 if row["proposal_zero"] or
                                          not bool(evidence[0].get("feasible")) else 1.0)
                    if declared_alpha != expected_alpha:
                        raise ValueError("committed alpha is not canonical controller decision")
                    for trial in evidence:
                        if len(trial.get("log_prob",[])) != len(row["old_log_prob"]):
                            raise ValueError("trial logprob row alignment failure")
                        if row["schema_version"] == "rwwpo-actual-loss-v3":
                            trial_tensor = trial["_authenticated_log_prob_tensor"]
                            trial_actual = reconstruct_authenticated_prefix_rows(
                                row, trial_tensor)
                            trial_dtype = trial_tensor.dtype
                        else:
                            trial_actual=[]
                            for sid in sorted(set(row["sample_index"])):
                                indices=[i for i,value in enumerate(row["sample_index"])
                                         if value==sid and any(bool(x) for x in row["writer_mask"][i])]
                                indices.sort(key=lambda i:row["trajectory_turn"][i]); running=0.0; tokens=0
                                for index in indices:
                                    active=[j for j,value in enumerate(row["writer_mask"][index]) if bool(value)]
                                    running += sum(float(trial["log_prob"][index][j])-float(row["old_log_prob"][index][j]) for j in active)
                                    tokens += len(active)
                                    trial_actual.append({"turn":int(row["trajectory_turn"][index]),
                                        "sample_index":int(sid),"log_ratio":running,"prefix_token_count":tokens})
                            trial_dtype = None
                        if not prefix_rows_match(trial["prefix_rows"], trial_actual,
                                                 dtype=trial_dtype):
                            raise ValueError("trial prefix rows do not reconstruct from actual logprobs")
                        if row["schema_version"]=="rwwpo-actual-loss-v3" and any(
                                str(item.get("root_identity_hash"))!=root_by_sample.get(
                                    int(item["sample_index"])) for item in trial["prefix_rows"]):
                            raise ValueError("trial prefix/root tensor identity mismatch")
                    if evidence[0]["log_prob"] != row["proposed_post_log_prob"]:
                        raise ValueError("full proposal logprob does not bind alpha=1 trial")
                    verification_reference = None
                    if declared_alpha>0:
                        selected=[trial for trial in evidence if float(trial["alpha"])==declared_alpha]
                        if len(selected)!=1:
                            raise ValueError("committed trial is not unique")
                        verification_reference = selected[0]["log_prob"]
                        if row["schema_version"] != "rwwpo-actual-loss-v3":
                            if row["committed_log_prob"]!=verification_reference:
                                raise ValueError("committed logprob is not selected trial logprob")
                            if row["post_prefix_rows"]!=selected[0]["prefix_rows"] or row["post_prefix_stats"]!=selected[0]["prefix_stats"]:
                                raise ValueError("committed post certificate is not selected trial certificate")
                    else:
                        verification_reference = row["current_log_prob"]
                        if row["schema_version"] != "rwwpo-actual-loss-v3":
                            if row["committed_log_prob"]!=verification_reference:
                                raise ValueError("zero-alpha commit did not restore behavior logprob")
                            if row["post_prefix_rows"]!=row["prefix_rows"] or row["post_prefix_stats"]!=row["prefix_stats"]:
                                raise ValueError("zero-alpha post certificate did not restore pre-step certificate")
                    if row["schema_version"] == "rwwpo-actual-loss-v3":
                        active_deltas = [
                            abs(float(committed)-float(reference))
                            for committed_row, reference_row, mask_row in zip(
                                row["committed_log_prob"], verification_reference,
                                row["response_mask"])
                            for committed, reference, active in zip(
                                committed_row, reference_row, mask_row) if bool(active)
                        ]
                        actual_verification_max = max(active_deltas, default=0.0)
                        declared_verification_max = float(row[
                            "mechanism_diagnostics"
                        ]["post_commit_forward_verification_max_abs"])
                        if actual_verification_max > declared_verification_max \
                                and not math.isclose(
                                    actual_verification_max,
                                    declared_verification_max,
                                    rel_tol=1e-9, abs_tol=1e-12):
                            raise ValueError(
                                "local post-commit verification exceeds declared global max")
                        row["_audited_post_verification_local_max"] = \
                            actual_verification_max
                for stat in row["prefix_stats"]:
                    expected = 1.0 / (1.0 + stat["chi2"])
                    if not math.isclose(stat["ess_fraction"], expected, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError("ESS/chi-square identity failure")
                rows.append(row)
        if limit is not None and selected_record_count != int(limit):
            raise ValueError(
                f"actual-loss ledger shorter than checkpoint prefix: {path.name}"
            )
    if not rows:
        raise ValueError("missing actual-loss rows")
    if any(row["schema_version"] in MODERN_SCHEMAS for row in rows):
        expected_names={"actual_loss_rank0.jsonl","actual_loss_rank1.jsonl"}
        if {Path(path).name for path in paths} != expected_names:
            raise ValueError("actual-loss v2 must cover exactly rank0 and rank1")
    groups={}
    for row in rows:
        key=(row["attempt_id"],row["global_step"],row["epoch"],row["minibatch"])
        groups.setdefault(key,[]).append(row)
    v3_rows=[row for row in rows if row["schema_version"]=="rwwpo-actual-loss-v3"]
    if v3_rows:
        round_groups={}
        for row in v3_rows:
            round_groups.setdefault((row["attempt_id"],row["global_step"],row["minibatch"]),[]).append(row)
        for round_key,round_rows in round_groups.items():
            coordinates={(int(row["rank"]),int(row["inner_id"])) for row in round_rows}
            if coordinates != {(0,1),(1,1),(0,2),(1,2)}:
                raise ValueError(f"RWWPO-2 round lacks exact K2 x rank2 closure: {round_key}")
            for rank in (0,1):
                rank_round=[row for row in round_rows if int(row["rank"])==rank]
                if len({row["behavior_batch_digest"] for row in rank_round}) != 1:
                    raise ValueError(f"RWWPO-2 frozen behavior batch drift rank{rank}: {round_key}")
                if len({row["denominator"] for row in rank_round}) != 1:
                    raise ValueError(f"RWWPO-2 frozen denominator drift rank{rank}: {round_key}")
                if len(rank_round) != 2:
                    raise ValueError(f"RWWPO-2 rank does not contain K2 rows: {round_key}")
                first, second = sorted(rank_round, key=lambda item: int(item["inner_id"]))
                frozen_fields = (
                    "old_log_prob", "ref_log_prob", "advantages", "response_mask",
                    "writer_mask", "answer_mask", "sample_index", "trajectory_turn",
                    "example_identity_hash", "trajectory_identity_hash",
                )
                if any(first[field] != second[field] for field in frozen_fields):
                    raise ValueError(
                        f"RWWPO-2 independently reconstructed frozen-batch drift "
                        f"rank{rank}: {round_key}"
                    )
            for field in ("host_variant","experiment_seed"):
                if len({row[field] for row in round_rows}) != 1:
                    raise ValueError(f"RWWPO-2 round field drift {field}: {round_key}")
        for rank in (0,1):
            rank_rows=sorted((row for row in v3_rows if int(row["rank"])==rank),
                             key=lambda row:int(row["proposal_clock"]))
            proposals=[int(row["proposal_clock"]) for row in rank_rows]
            if len(proposals)!=len(set(proposals)) or any(
                right != left+1 for left,right in zip(proposals,proposals[1:])):
                raise ValueError(f"RWWPO-2 proposal clock discontinuity rank{rank}")
            for left,right in zip(rank_rows,rank_rows[1:]):
                if int(right["accepted_optimizer_clock_before"]) != int(
                        left["accepted_optimizer_clock_after"]):
                    raise ValueError(f"RWWPO-2 accepted optimizer clock discontinuity rank{rank}")
    for key,group in groups.items():
        if group[0]["schema_version"] in MODERN_SCHEMAS and sorted(int(row["rank"]) for row in group)!=[0,1]:
            raise ValueError(f"optimizer transaction lacks exact rank0/rank1 coverage for {key}")
        if group[0]["schema_version"] == "rwwpo-actual-loss-v3":
            declared_post_max = {
                float(row["mechanism_diagnostics"][
                    "post_commit_forward_verification_max_abs"])
                for row in group
            }
            actual_post_max = max(float(row.pop(
                "_audited_post_verification_local_max")) for row in group)
            if len(declared_post_max) != 1 or not math.isclose(
                    actual_post_max, next(iter(declared_post_max)),
                    rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"distributed post-commit verification max drift for {key}")
            writeback_evidence = {
                json.dumps(row["mechanism_diagnostics"][
                    "fsdp_parameter_writeback_wall_seconds"],
                    separators=(",", ":"))
                for row in group
            }
            trial_wall_evidence = {
                float(row["trial_forward_wall_seconds"])
                for row in group
            }
            if len(writeback_evidence) != 1 or len(trial_wall_evidence) != 1:
                raise ValueError(
                    f"distributed FSDP/trial wall-time drift for {key}")
        q_values={float(row["q_min"]) for row in group}
        root_q_values={float(row.get("root_q_min",row["q_min"])) for row in group}
        caps={float(row["writer_log_ratio_cap"]) for row in group}
        decisions={bool(row["accepted"]) for row in group}
        if len(q_values)!=1 or len(root_q_values)!=1 or len(caps)!=1 or len(decisions)!=1:
            raise ValueError(f"distributed decision/threshold disagreement for {key}")
        def reconstruct(field):
            combined=[item for row in group for item in row[field]]
            if group[0]["schema_version"] == "rwwpo-actual-loss-v3":
                return prefix_distribution_stats(
                    combined, q_min=next(iter(q_values)),
                    root_q_min=next(iter(root_q_values)),
                    log_ratio_cap=next(iter(caps)))
            result=[]
            for turn in sorted({item["turn"] for item in combined}):
                values=[item["log_ratio"] for item in combined if item["turn"]==turn]
                peak=max(values); raw=[math.exp(value-peak) for value in values]; total=sum(raw)
                weights=[value/total for value in raw]; chi2=len(values)*sum(value*value for value in weights)-1
                result.append({"turn":turn,"ess_fraction":1/(1+chi2),"chi2":chi2,
                               "max_abs_log_ratio":max(abs(value) for value in values),
                               "feasible":1/(1+chi2)>=next(iter(q_values)) and
                                          max(abs(value) for value in values)<=next(iter(caps))})
            return result
        for rows_field, stats_field in (("prefix_rows","prefix_stats"),("post_prefix_rows","post_prefix_stats")):
            expected=reconstruct(rows_field)
            for row in group:
                declared=row[stats_field]
                fields=("ess_fraction","chi2","max_abs_log_ratio")
                if group[0]["schema_version"]=="rwwpo-actual-loss-v3":
                    fields += ("root_ess_fraction","root_chi2",
                               "root_loo_complete_fraction",
                               "root_loo_feasibility_flip_fraction")
                if len(declared)!=len(expected) or any(
                    item["turn"]!=value["turn"] or any(
                        not math.isclose(float(item[field]),float(value[field]),
                                         rel_tol=1e-9,abs_tol=1e-10)
                        for field in fields)
                    for item,value in zip(declared,expected)):
                    raise ValueError(f"global {stats_field} do not reconstruct for {key}")
                if group[0]["schema_version"]=="rwwpo-actual-loss-v3" and any(
                        bool(item.get("root_loo_supported")) !=
                        bool(value["root_loo_supported"])
                        for item,value in zip(declared,expected)):
                    raise ValueError(
                        f"global {stats_field} LOO support does not reconstruct for {key}"
                    )
        if group[0]["schema_version"] in MODERN_SCHEMAS:
            orders={tuple(float(x) for x in row["alpha_test_order"]) for row in group}
            if len(orders)!=1: raise ValueError(f"distributed trial order disagreement for {key}")
            for trial_alpha in next(iter(orders)):
                combined=[]
                declared_feasible=set()
                for row in group:
                    trial=[item for item in row["trial_evidence"] if float(item["alpha"])==trial_alpha]
                    if len(trial)!=1: raise ValueError(f"trial evidence missing for {key}")
                    combined.extend(trial[0]["prefix_rows"]); declared_feasible.add(bool(trial[0]["feasible"]))
                if len(declared_feasible)!=1: raise ValueError(f"distributed trial feasibility drift for {key}")
                if group[0]["schema_version"]=="rwwpo-actual-loss-v3":
                    actual_feasible=all(row["feasible"] for row in prefix_distribution_stats(
                        combined,q_min=next(iter(q_values)),root_q_min=next(iter(root_q_values)),
                        log_ratio_cap=next(iter(caps))))
                else:
                    actual_feasible=True
                    for turn in sorted({item["turn"] for item in combined}):
                        values=[item["log_ratio"] for item in combined if item["turn"]==turn]
                        peak=max(values); weights=[math.exp(v-peak) for v in values]; total=sum(weights)
                        weights=[v/total for v in weights]; chi2=len(values)*sum(v*v for v in weights)-1
                        actual_feasible &= 1/(1+chi2)>=next(iter(q_values)) and max(abs(v) for v in values)<=next(iter(caps))
                if next(iter(declared_feasible)) != actual_feasible:
                    raise ValueError(f"forged trial feasibility for {key} alpha={trial_alpha}")
            alpha_values={float(row["alpha_committed"]) for row in group}
            if len(alpha_values)!=1: raise ValueError(f"distributed alpha disagreement for {key}")
            alpha=next(iter(alpha_values))
            expected_accept=alpha>0 and not any(bool(row["proposal_zero"]) for row in group)
            if next(iter(decisions)) != expected_accept:
                raise ValueError(f"accepted decision is not certified by committed alpha for {key}")
            if alpha>0:
                selected=[]
                for row in group:
                    match=[trial for trial in row["trial_evidence"] if float(trial["alpha"])==alpha]
                    if len(match)!=1: raise ValueError(f"committed trial missing for {key}")
                    selected.extend(match[0]["prefix_rows"])
                if group[0]["schema_version"]=="rwwpo-actual-loss-v3":
                    feasible=all(row["feasible"] for row in prefix_distribution_stats(
                        selected,q_min=next(iter(q_values)),root_q_min=next(iter(root_q_values)),
                        log_ratio_cap=next(iter(caps))))
                else:
                    values_by_turn={turn:[item["log_ratio"] for item in selected if item["turn"]==turn]
                                    for turn in sorted({item["turn"] for item in selected})}
                    feasible=True
                    for values in values_by_turn.values():
                        peak=max(values); weights=[math.exp(v-peak) for v in values]; total=sum(weights)
                        weights=[v/total for v in weights]; chi2=len(values)*sum(v*v for v in weights)-1
                        feasible &= 1/(1+chi2)>=next(iter(q_values)) and max(abs(v) for v in values)<=next(iter(caps))
                if not feasible and group[0]["controller_variant"] != "none":
                    raise ValueError(f"committed alpha is globally infeasible for {key}")
        else:
            expected_accept=all(s["feasible"] for s in reconstruct("post_prefix_rows"))
            if next(iter(decisions)) != expected_accept: raise ValueError(f"accepted decision is not certified by post statistics for {key}")
        post_pass=all(s["feasible"] for s in reconstruct("post_prefix_rows"))
        if group[0].get("controller_variant") != "none" and not post_pass:
            raise ValueError(f"fresh committed post certificate is infeasible for {key}")
        pre_pass=all(s["feasible"] for s in reconstruct("prefix_rows"))
        if group[0].get("controller_variant") != "none":
            if any(bool(row["constraint_pass"])!=pre_pass for row in group):
                raise ValueError(f"pre-step constraint decision mismatch for {key}")
    # On-policy behavior-point closure requires current == old before the
    # optimizer step; using that equality's violation as an activity signal
    # incorrectly rejects a healthy on-policy run.  Method activity is instead
    # certified by a non-zero proposed post-step movement that survived the
    # distributed trust-region acceptance decision.
    behavior_point_max_delta = max(abs(c-o) for row in rows
        for old,cur,mask in zip(row["old_log_prob"],row["current_log_prob"],
                                row["response_mask"])
        for o,c,active_token in zip(old,cur,mask) if bool(active_token))
    active = any(bool(group[0]["accepted"]) and
        any(abs(p-c) > 1e-10 for row in group
            for post,cur,mask in zip(row["proposed_post_log_prob"],
                                     row["current_log_prob"],row["response_mask"])
            for p,c,active_token in zip(post,cur,mask) if bool(active_token))
        for group in groups.values())
    if require_method and not active:
        raise ValueError("RWWPO_METHOD_INACTIVE")
    step_summaries={}
    for step in sorted({int(r["global_step"]) for r in rows}):
        selected=[r for r in rows if int(r["global_step"])==step]
        step_groups=[g for key,g in groups.items() if int(key[1])==step]
        step_summaries[str(step)]={"accepted_fraction":sum(bool(g[0]["accepted"]) for g in step_groups)/len(step_groups),
            "alpha_committed":[float(g[0].get("alpha_committed",1.0 if g[0]["accepted"] else 0.0)) for g in step_groups],
            "accepted_nonzero_count":sum(bool(g[0].get("accepted_nonzero",g[0]["accepted"])) for g in step_groups),
            "max_proposed_update":max(abs(p-c) for r in selected
                for post,cur,mask in zip(r["proposed_post_log_prob"],
                                         r["current_log_prob"],r["response_mask"])
                for p,c,active_token in zip(post,cur,mask) if bool(active_token))}
    v2_groups=[group for group in groups.values() if group[0]["schema_version"] in MODERN_SCHEMAS]
    nonzero_alphas=[float(group[0]["alpha_committed"]) for group in v2_groups
                    if bool(group[0]["accepted_nonzero"])]
    return {"status": "PASS", "decision": "RWWPO_ACTUAL_LOSS_LEDGER_PASS",
            "record_count": len(rows), "method_active": active,
            "behavior_point_max_delta": behavior_point_max_delta,
            "modes": sorted({row["mode"] for row in rows}),
            "schema_versions": sorted({row["schema_version"] for row in rows}),
            "objective_variants": sorted({row.get("objective_variant","legacy") for row in rows}),
            "controller_variants": sorted({row.get("controller_variant","legacy") for row in rows}),
            "nonzero_commit_count": len(nonzero_alphas),
            "committed_nonzero_alphas": nonzero_alphas,
            "min_prefix_ess": min(s["ess_fraction"] for r in rows for s in r["prefix_stats"]),
            "min_post_prefix_ess": min(s["ess_fraction"] for r in rows for s in r["post_prefix_stats"]),
            "min_prefix_root_ess": min(s.get("root_ess_fraction",1.0)
                                       for r in rows for s in r["prefix_stats"]),
            "min_post_prefix_root_ess": min(s.get("root_ess_fraction",1.0)
                                            for r in rows for s in r["post_prefix_stats"]),
            "max_root_loo_feasibility_flip_fraction": max((
                s.get("root_loo_feasibility_flip_fraction", 0.0)
                for r in rows for s in r["post_prefix_stats"]
                if s.get("root_loo_supported")
            ), default=0.0),
            "root_loo_supported_turn_count": sum(
                bool(s.get("root_loo_supported"))
                for r in rows for s in r["post_prefix_stats"]),
            "root_loo_unsupported_turn_count": sum(
                not bool(s.get("root_loo_supported"))
                for r in rows for s in r["post_prefix_stats"]),
            "accepted_fraction": sum(bool(next(iter(g))["accepted"]) for g in groups.values())/len(groups),
            "max_proposed_update": max(abs(p-c) for r in rows
                for post,cur,mask in zip(r["proposed_post_log_prob"],
                                         r["current_log_prob"],r["response_mask"])
                for p,c,active_token in zip(post,cur,mask) if bool(active_token)),
            "audited_start_round": min(int(row["global_step"]) for row in rows),
            "audited_through_round": max(int(row["global_step"]) for row in rows),
            "steps":step_summaries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledgers", nargs="+")
    parser.add_argument("--allow-behavior-only", action="store_true")
    parser.add_argument("--start-round", type=int)
    parser.add_argument("--through-round", type=int)
    args = parser.parse_args()
    print(json.dumps(audit(
        args.ledgers, not args.allow_behavior_only,
        start_round=args.start_round, through_round=args.through_round,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
