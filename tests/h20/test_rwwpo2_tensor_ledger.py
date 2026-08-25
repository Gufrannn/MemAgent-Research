import json

import pytest
import torch
import jsonschema
from pathlib import Path

from recurrent.research.rwwpo_ledger import (
    append_actual_loss_record, append_transaction_failure_record,
    tensor_shard_inventory,
)
from recurrent.research.rwwpo_transaction import (
    digest, logical_transaction_seed, prefix_distribution_stats, proposal_clock,
    tensor_content_digest,
)
from tools.h20.audit_rwwpo_actual_loss import (
    audit, canonical_sha, hydrate_authenticated_v3_receipt,
    independently_recompute_actual_loss, prefix_rows_match,
    reconstruct_authenticated_prefix_rows, validate_rwwpo2_rng_phase_digests,
)


def _digest_state(prefix):
    return {"model":prefix+"0"*63,"optimizer":prefix+"1"*63,
            "scheduler":prefix+"2"*63,"scaler":"not_applicable_bfloat16",
            "rng":prefix+"3"*63}


def _append_round(root, rank, inner, *, active=True, policy_loss=None,
                  verified_delta=0.0):
    old=torch.zeros((1,2),dtype=torch.float64)
    ref=torch.full((1,2),-0.2,dtype=torch.float64)
    current=torch.full((1,2),0.1*(inner-1) if active else 0.,dtype=torch.float64)
    proposed=torch.full((1,2),0.1*inner if active else 0.,dtype=torch.float64)
    committed=proposed+float(verified_delta)
    mask=torch.tensor([[True,True]])
    sample=torch.tensor([rank])
    root_name=str(100+rank)
    pre_local=[{"turn":0,"sample_index":rank,"root_identity_hash":root_name,
                "log_ratio":0.2*(inner-1) if active else 0.,"prefix_token_count":2}]
    trial_ratio=0.2*inner if active else 0.
    trial_local=[{"turn":0,"sample_index":rank,"root_identity_hash":root_name,
                  "log_ratio":trial_ratio,"prefix_token_count":2}]
    post_ratio=trial_ratio+2.0*float(verified_delta)
    post_local=[{"turn":0,"sample_index":rank,"root_identity_hash":root_name,
                 "log_ratio":post_ratio,"prefix_token_count":2}]
    pre_global=prefix_distribution_stats([
        {"turn":0,"sample_index":0,"root_identity_hash":"100",
         "log_ratio":0.2*(inner-1) if active else 0.},
        {"turn":0,"sample_index":1,"root_identity_hash":"101",
         "log_ratio":0.2*(inner-1) if active else 0.}],q_min=.5,root_q_min=.5,log_ratio_cap=4.)
    trial_global=prefix_distribution_stats([
        {"turn":0,"sample_index":0,"root_identity_hash":"100","log_ratio":trial_ratio},
        {"turn":0,"sample_index":1,"root_identity_hash":"101","log_ratio":trial_ratio}],
        q_min=.5,root_q_min=.5,log_ratio_cap=4.)
    post_global=prefix_distribution_stats([
        {"turn":0,"sample_index":0,"root_identity_hash":"100","log_ratio":post_ratio},
        {"turn":0,"sample_index":1,"root_identity_hash":"101","log_ratio":post_ratio}],
        q_min=.5,root_q_min=.5,log_ratio_cap=4.)
    before=inner-1 if active else 0
    current_value=float(current[0,0].item())
    surrogate_loss=-torch.exp(torch.tensor(2.0*current_value,dtype=torch.float64))/2.0
    kl_value=-0.2-current_value
    shared_kl_loss=(torch.exp(torch.tensor(kl_value,dtype=torch.float64))-kl_value-1.0)*.001
    expected_policy_loss=surrogate_loss+shared_kl_loss
    gradient_coefficient=(-torch.exp(torch.tensor(2.0*current_value,dtype=torch.float64))/2.0
                          + .001*(1.0-torch.exp(torch.tensor(kl_value,dtype=torch.float64)))/2.0)
    active_gradient_l2=float((gradient_coefficient.abs()*torch.sqrt(
        torch.tensor(2.0,dtype=torch.float64))).item())
    if policy_loss is None:
        policy_loss=float(expected_policy_loss.item())
    pre_digests=_digest_state("a")
    commit_digests=_digest_state("b") if active else dict(pre_digests)
    behavior_forward_rng_digests = ["8"*64, "9"*64]
    append_actual_loss_record(
        ledger_dir=root,attempt_id="rwwpo2_test",mode="rwwpo_method",rank=rank,
        global_step=1,epoch=inner-1,minibatch=0,old_log_prob=old,
        ref_log_prob=ref,
        current_log_prob=current,proposed_post_log_prob=proposed,
        committed_log_prob=committed,response_mask=mask,writer_mask=mask,
        answer_mask=torch.zeros_like(mask),trajectory_turn=torch.tensor([0]),
        sample_index=sample,example_identity_hash=torch.tensor([100+rank]),
        trajectory_identity_hash=torch.tensor([200+rank]),
        advantages=torch.ones_like(old),denominator=2,prefix_rows=pre_local,
        prefix_stats=pre_global,post_prefix_rows=post_local,
        post_prefix_stats=post_global,q_min=.5,root_q_min=.5,
        writer_log_ratio_cap=4.,constraint_pass=True,accepted=active,
        objective_variant="whole_prefix",controller_variant="feasible_backtracking",
        alpha_grid=[1.,.5,.25,.125,.0625,.03125],alpha_test_order=[1.],
        alpha_committed=1. if active else 0.,accepted_nonzero=active,
        proposal_zero=not active,
        trial_evidence=[{"alpha":1.,"feasible":True,"log_prob":proposed,
                         "prefix_rows":trial_local,"prefix_stats":trial_global}],
        full_parameter_displacement_norm=.1 if active else 0.,
        committed_parameter_displacement_norm=.1 if active else 0.,
        pre_digests=pre_digests,commit_digests=commit_digests,
        mechanism_diagnostics={"policy_loss":policy_loss,
                               "surrogate_loss":float(surrogate_loss.item()),
                               "shared_kl_loss":float(shared_kl_loss.item()),
                               "actual_loss_contract":{
                                   "loss_agg_mode":"token-mean","cliprange":.2,
                                   "cliprange_low":.2,"cliprange_high":.2,
                                   "clip_ratio_c":3.0,"writer_log_ratio_cap":4.0,
                                   "use_kl_loss":True,"kl_loss_type":"low_var_kl",
                                   "kl_loss_coefficient":.001,"entropy_coefficient":0.0},
                               "active_logprob_gradient_l2":active_gradient_l2,
                               "post_commit_forward_verified":True,
                               "post_commit_forward_verification_max_abs":abs(
                                   float(verified_delta)),
                               "post_commit_forward_verification_tolerance":1e-6,
                               "behavior_current_logprob_digest":
                                   tensor_content_digest(current),
                               "behavior_current_logprob_integrity_max_abs":0.0,
                               "behavior_current_logprob_integrity_verified":True,
                               "fsdp_parameter_commit_primitive":
                                   "fsdp_unitwise_allgather_summon_writeback_v1",
                               "fsdp_parameter_writeback_max_wall_seconds":120.0,
                               "fsdp_parameter_writeback_wall_seconds":[1.25],
                               "max_trial_forward_wall_seconds":600.0,
                               "transaction_entry_buffer_digest":"7"*64,
                               "terminal_buffer_digest":"7"*64,
                               "optimizer_step_calls":1,
                               "proposal_lr":1e-6,
                               "gradient_sketch_chunk_elements":8388608,
                               "transaction_entry_rng_digest":pre_digests["rng"],
                               "logical_seeded_rng_digest":"c"*64,
                               "proposal_gradient_rng_digest":"d"*64,
                               "behavior_forward_rng_digests":
                                   behavior_forward_rng_digests,
                               "behavior_forward_rng_aggregate_digest":digest(
                                   behavior_forward_rng_digests),
                               "replay_microbatch_count":len(
                                   behavior_forward_rng_digests),
                               "replay_rng_bound":True,
                               "terminal_rng_digest":commit_digests["rng"],
                               "shadow_coefficients":{},
                               "inner1_exposure":{
                                   "relative_parameter_displacement":.1,
                                   "writer_logprob_rms":.1,
                                   "off_behavior_exposed":True}},
        gradient_norm=1.,program_version="rwwpo2-k2",
        inner_id=inner,proposal_clock=proposal_clock(1,inner),
        accepted_optimizer_clock_before=before,
        accepted_optimizer_clock_after=before+int(active),
        logical_seed=logical_transaction_seed(experiment_seed=2026,round_id=1,
            inner_id=inner,rank=rank,stream="actor_transaction"),
        experiment_seed=2026,host_variant="B",behavior_batch_digest="f"*64)


def test_v3_nonfinite_json_evidence_is_rejected_before_receipt_append(tmp_path):
    with pytest.raises(ValueError, match="Out of range float values"):
        _append_round(tmp_path, 0, 1, policy_loss=float("nan"))
    assert not (tmp_path / "actual_loss_rank0.jsonl").exists()


def test_v3_tensor_shards_roundtrip_and_jsonl_stays_small(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    paths=[tmp_path/f"actual_loss_rank{rank}.jsonl" for rank in (0,1)]
    result=audit(paths,require_method=True)
    assert result["status"]=="PASS"
    assert result["schema_versions"]==["rwwpo-actual-loss-v3"]
    schema=json.loads((Path(__file__).resolve().parents[2]/
                       "rwwpo2_actual_loss_receipt.schema.json").read_text())
    for path in paths:
        for line in path.read_text().splitlines():
            receipt=json.loads(line)
            jsonschema.Draft202012Validator(schema).validate(receipt)
            assert "old_log_prob" not in receipt
        assert path.stat().st_size < 100_000


def test_v3_fresh_post_forward_may_differ_from_trial_within_frozen_tolerance(
        tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner,verified_delta=5e-7)
    result=audit([tmp_path/"actual_loss_rank0.jsonl",
                  tmp_path/"actual_loss_rank1.jsonl"],require_method=True)
    assert result["status"]=="PASS"
    receipt=json.loads((tmp_path/"actual_loss_rank0.jsonl").read_text().splitlines()[0])
    hydrated=hydrate_authenticated_v3_receipt(
        receipt,tmp_path/"actual_loss_rank0.jsonl")
    assert hydrated["trial_evidence"][0]["prefix_rows"] != \
        hydrated["post_prefix_rows"]


def test_v3_prefix_reconstruction_preserves_float32_torch_accumulation():
    old = torch.zeros((1, 3), dtype=torch.float32)
    current = torch.tensor(
        [[100_000_000.0, 1.0, -100_000_000.0]], dtype=torch.float32)
    writer = torch.ones_like(current, dtype=torch.bool)
    row = {"_authenticated_tensors": {
        "old_log_prob": old,
        "writer_mask": writer,
        "sample_index": torch.tensor([0]),
        "trajectory_turn": torch.tensor([4]),
    }}
    actual = reconstruct_authenticated_prefix_rows(row, current)
    producer_value = float(((current - old) * writer).sum(dim=-1)[0].item())
    python_value = sum(float(value) for value in current[0].tolist())
    assert producer_value != python_value
    declared = [{
        "turn": 4, "sample_index": 0,
        "log_ratio": producer_value, "prefix_token_count": 3,
    }]
    assert prefix_rows_match(declared, actual, dtype=torch.float32)
    forged = [{**declared[0], "log_ratio": python_value}]
    assert not prefix_rows_match(forged, actual, dtype=torch.float32)


def test_v3_tensor_shard_byte_tamper_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    receipt=json.loads((tmp_path/"actual_loss_rank0.jsonl").read_text().splitlines()[0])
    shard=tmp_path/receipt["tensor_shard"]["relative_path"]
    raw=bytearray(shard.read_bytes()); raw[-1]^=1; shard.write_bytes(raw)
    with pytest.raises(ValueError,match="hash mismatch"):
        audit([tmp_path/"actual_loss_rank0.jsonl",tmp_path/"actual_loss_rank1.jsonl"])


def test_actual_loss_ledger_symlink_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    linked=tmp_path/"linked_rank0.jsonl"
    linked.symlink_to(tmp_path/"actual_loss_rank0.jsonl")
    with pytest.raises(ValueError,match="symlink"):
        audit([linked,tmp_path/"actual_loss_rank1.jsonl"])


def test_checkpoint_record_limit_excludes_malformed_failed_suffix(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    paths=[tmp_path/f"actual_loss_rank{rank}.jsonl" for rank in (0,1)]
    limits={path.name:2 for path in paths}
    for path in paths:
        with path.open("a",encoding="utf-8") as stream:
            stream.write('{"malformed_failed_suffix":')
    result=audit(paths,require_method=True,through_round=1,record_limits=limits)
    assert result["status"]=="PASS"
    inventory=tensor_shard_inventory(
        tmp_path,start_round=1,through_round=1,record_limits=limits)
    assert inventory["shard_count"]==4


def test_checkpoint_record_limit_rejects_short_prefix(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    paths=[tmp_path/f"actual_loss_rank{rank}.jsonl" for rank in (0,1)]
    limits={path.name:3 for path in paths}
    with pytest.raises(ValueError,match="shorter than checkpoint prefix"):
        audit(paths,require_method=False,through_round=1,record_limits=limits)


def _tamper_last_receipt(path, mutate):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    mutate(rows[-1])
    rows[-1]["record_sha256"]=canonical_sha(rows[-1])
    path.write_text("".join(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n"
                            for row in rows))


def test_v3_self_reported_extra_optimizer_step_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        optimizer_step_calls=2))
    # The schema is intentionally the first line of defense.  A future schema
    # relaxation must still reach the independent semantic step-count check.
    with pytest.raises(ValueError,match=(
            "receipt schema failure|loss/optimizer-step evidence")):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_missing_post_commit_forward_verification_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        post_commit_forward_verified=False))
    with pytest.raises(ValueError,match=(
            "receipt schema failure|loss/optimizer-step evidence")):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_post_commit_forward_delta_above_tolerance_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        post_commit_forward_verification_max_abs=2e-6))
    with pytest.raises(ValueError,match="loss/optimizer-step evidence"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_post_commit_buffer_digest_drift_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        terminal_buffer_digest="8"*64))
    with pytest.raises(ValueError,match="loss/optimizer-step evidence"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_behavior_reference_digest_tamper_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        behavior_current_logprob_digest="e"*64))
    with pytest.raises(ValueError,match="immutable behavior logprob digest"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_fsdp_commit_primitive_drift_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        fsdp_parameter_commit_primitive="legacy_raw_shard_copy"))
    with pytest.raises(ValueError,match=(
            "receipt schema failure|FSDP/behavior-reference closure")):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_fsdp_writeback_over_budget_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        fsdp_parameter_writeback_wall_seconds=[120.0001]))
    with pytest.raises(ValueError,match=(
            "receipt schema failure|FSDP/trial wall-time closure")):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_trial_wall_time_over_budget_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row.__setitem__(
        "trial_forward_wall_seconds",600.0001))
    with pytest.raises(ValueError,match=(
            "receipt schema failure|FSDP/trial wall-time closure")):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_cross_rank_writeback_time_drift_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        fsdp_parameter_writeback_wall_seconds=[2.0]))
    with pytest.raises(ValueError,match="distributed FSDP/trial wall-time drift"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_transaction_failure_diagnostic_is_append_only_and_chained(tmp_path):
    kwargs=dict(
        ledger_dir=tmp_path,attempt_id="failed_attempt",rank=0,
        global_step=2,inner_id=2,proposal_clock=4,
        reason="RWWPO_PREFIX_TRUST_REGION_VIOLATION",
        phase="precondition",
        prefix_rows=[{"turn":0,"sample_index":0,
                      "root_identity_hash":"root","log_ratio":4.1}],
        prefix_stats=[{"turn":0,"feasible":False}],
        current_reference_max_abs=.01,
        behavior_batch_digest="a"*64,
        transaction_entry_buffer_digest="b"*64,
        diagnostics={"q_min":.5})
    append_transaction_failure_record(**kwargs)
    append_transaction_failure_record(**kwargs)
    path=tmp_path/"failure_rank0.jsonl"
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==2
    assert rows[0]["status"]=="NO_GO"
    assert rows[0]["previous_record_sha256"]=="0"*64
    assert rows[1]["previous_record_sha256"]==rows[0]["record_sha256"]


@pytest.mark.parametrize("field,value,error", [
    ("proposal_clock", 3, "coordinate"),
    ("current_reference_max_abs", float("inf"), "magnitude"),
    ("behavior_batch_digest", "not-a-sha", "digest"),
    ("prefix_rows", [], "prefix evidence"),
])
def test_transaction_failure_diagnostic_rejects_malformed_evidence(
        tmp_path, field, value, error):
    kwargs=dict(
        ledger_dir=tmp_path,attempt_id="failed_attempt",rank=0,
        global_step=2,inner_id=2,proposal_clock=4,
        reason="RWWPO_PREFIX_TRUST_REGION_VIOLATION",
        phase="precondition",
        prefix_rows=[{"turn":0,"sample_index":0,
                      "root_identity_hash":"root","log_ratio":4.1}],
        prefix_stats=[{"turn":0,"feasible":False}],
        current_reference_max_abs=.01,
        behavior_batch_digest="a"*64,
        transaction_entry_buffer_digest="b"*64,
        diagnostics={"q_min":.5})
    kwargs[field] = value
    with pytest.raises(ValueError, match=error):
        append_transaction_failure_record(**kwargs)


def test_transaction_failure_diagnostic_rejects_symlink_root(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        append_transaction_failure_record(
            ledger_dir=linked,attempt_id="failed_attempt",rank=0,
            global_step=2,inner_id=2,proposal_clock=4,
            reason="RWWPO_PREFIX_TRUST_REGION_VIOLATION",
            phase="precondition",
            prefix_rows=[{"turn":0,"sample_index":0,
                          "root_identity_hash":"root","log_ratio":4.1}],
            prefix_stats=[{"turn":0,"feasible":False}],
            current_reference_max_abs=.01,
            behavior_batch_digest="a"*64,
            transaction_entry_buffer_digest="b"*64)


def test_v3_self_reported_actual_loss_is_independently_recomputed(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        policy_loss=float(row["mechanism_diagnostics"]["policy_loss"])+.25))
    with pytest.raises(ValueError,match="independently recomputed actual loss"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_rehashed_extra_trial_after_first_feasible_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    def append_impossible_tail(row):
        extra=dict(row["trial_evidence"][0])
        extra.update(alpha=.5,feasible=False)
        row["alpha_test_order"].append(.5)
        row["trial_evidence"].append(extra)
    _tamper_last_receipt(path,append_impossible_tail)
    with pytest.raises(ValueError,match="terminate tested prefix"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_rehashed_truncated_all_infeasible_grid_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["trial_evidence"][0].update(
        feasible=False))
    with pytest.raises(ValueError,match="did not exhaust alpha grid"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_missing_rng_phase_digest_is_rejected_after_receipt_rehash(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].pop(
        "transaction_entry_rng_digest"))
    with pytest.raises(ValueError,match="receipt schema failure"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_missing_behavior_replay_rng_binding_is_rejected_after_rehash(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].pop(
        "behavior_forward_rng_digests"))
    with pytest.raises(ValueError,match="receipt schema failure"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_reordered_behavior_replay_rng_schedule_is_rejected_after_rehash(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"]
        ["behavior_forward_rng_digests"].reverse())
    with pytest.raises(ValueError,match="behavior replay RNG binding"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_behavior_replay_rng_count_is_rejected_after_rehash(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        replay_microbatch_count=1))
    with pytest.raises(ValueError,match="behavior replay RNG binding"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_forged_entry_rng_digest_is_rejected_after_receipt_rehash(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        transaction_entry_rng_digest="e"*64))
    with pytest.raises(ValueError,match="RNG phase digest closure"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_forged_terminal_rng_digest_is_rejected_after_receipt_rehash(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["mechanism_diagnostics"].update(
        terminal_rng_digest="e"*64))
    with pytest.raises(ValueError,match="RNG phase digest closure"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_rejected_terminal_rng_must_equal_entry_after_rehash(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner,active=False)
    path=tmp_path/"actual_loss_rank1.jsonl"
    def forge_rejected_terminal(row):
        row["mechanism_diagnostics"]["terminal_rng_digest"]="e"*64
        row["commit_digests"]["rng"]="e"*64
    _tamper_last_receipt(path,forge_rejected_terminal)
    with pytest.raises(ValueError,match="rejected transaction RNG rollback"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path],require_method=False)


def test_formal_rng_phase_semantics_accept_authentic_hydrated_row(tmp_path):
    _append_round(tmp_path,0,1)
    ledger=tmp_path/"actual_loss_rank0.jsonl"
    receipt=json.loads(ledger.read_text().splitlines()[0])
    row=hydrate_authenticated_v3_receipt(receipt,ledger)
    validate_rwwpo2_rng_phase_digests(row)


def test_v3_frozen_reference_input_drift_between_inner_transactions_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    receipt=json.loads((tmp_path/"actual_loss_rank0.jsonl").read_text().splitlines()[1])
    shard=tmp_path/receipt["tensor_shard"]["relative_path"]
    payload=torch.load(shard,map_location="cpu",weights_only=True)
    payload["ref_log_prob"][0,0] += .05
    torch.save(payload,shard)
    raw=shard.read_bytes()
    receipt["tensor_shard"]["size"]=len(raw)
    import hashlib
    receipt["tensor_shard"]["sha256"]=hashlib.sha256(raw).hexdigest()
    hydrated=hydrate_authenticated_v3_receipt(
        receipt,tmp_path/"actual_loss_rank0.jsonl")
    receipt["mechanism_diagnostics"].update(
        independently_recompute_actual_loss(hydrated))
    receipt["record_sha256"]=canonical_sha(receipt)
    lines=(tmp_path/"actual_loss_rank0.jsonl").read_text().splitlines()
    lines[1]=json.dumps(receipt,sort_keys=True,separators=(",",":"))
    (tmp_path/"actual_loss_rank0.jsonl").write_text("\n".join(lines)+"\n")
    with pytest.raises(ValueError,match="frozen-batch drift"):
        audit([tmp_path/"actual_loss_rank0.jsonl",tmp_path/"actual_loss_rank1.jsonl"])


def test_v3_attempt_dependent_or_forged_logical_seed_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row.__setitem__(
        "logical_seed",int(row["logical_seed"])+1))
    with pytest.raises(ValueError,match="logical seed mismatch"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_forged_post_root_certificate_is_rejected(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner)
    path=tmp_path/"actual_loss_rank1.jsonl"
    _tamper_last_receipt(path,lambda row: row["post_prefix_stats"][0].__setitem__(
        "root_ess_fraction",.99))
    with pytest.raises(ValueError,match="post certificate|global post_prefix_stats"):
        audit([tmp_path/"actual_loss_rank0.jsonl",path])


def test_v3_method_inactive_is_rejected_when_required(tmp_path):
    for inner in (1,2):
        for rank in (0,1):
            _append_round(tmp_path,rank,inner,active=False)
    paths=[tmp_path/"actual_loss_rank0.jsonl",tmp_path/"actual_loss_rank1.jsonl"]
    assert audit(paths,require_method=False)["method_active"] is False
    with pytest.raises(ValueError,match="RWWPO_METHOD_INACTIVE"):
        audit(paths,require_method=True)
