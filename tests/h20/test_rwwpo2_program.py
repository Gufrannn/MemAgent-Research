import copy
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import torch

from tools.h20.audit_rwwpo2_r50_program import wilson_interval
from tools.h20.audit_rwwpo2_attempt import (
    execution_prefix_through_round, validate_transaction_failure_boundary,
    validate_post_commit_forward_binding, validate_recovery_prune_evidence,
)
from tools.h20.audit_rwwpo2_lineage_parent import execution_prefix_to_checkpoint
from tools.h20.audit_rwwpo2_numeric_oracle import (
    validate_fsdp_transaction_closure,
)
from tools.h20.preflight_rwwpo2 import receipt
from tools.h20.verify_rwwpo2_release_tests import (
    TEST_INVENTORY, canonical_sha as release_test_canonical_sha,
    collect_current_node_ids, junit_summary as release_test_junit_summary,
    node_evidence, runtime_environment, sha256_file, verify_release_test_receipt,
)
from recurrent.research.gate_a_execution import (
    append_jsonl, checkpoint_inventory, validate_jsonl_chain,
)
from recurrent.research.rwwpo2_confirmation import (
    generation_protocol_projection, holm_two_test_decisions,
    one_sided_exact_paired_sign_flip, signed_report,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "manifests/h20/qwen25_7b_rwwpo2_r400_k2_seed2026.json"
SCHEMA_PATH = ROOT / "rwwpo2_experiment_manifest.schema.json"


def _signed(row):
    import hashlib
    raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**row, "report_sha256": hashlib.sha256(raw.encode()).hexdigest()}


def test_manifest_is_schema_valid_and_k2_is_unambiguous():
    manifest = json.loads(MANIFEST_PATH.read_text())
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert manifest["method"]["inner_transactions_per_round"] == 2
    assert manifest["method"]["optimizer_minibatches_per_inner_transaction"] == 1
    assert manifest["method"]["optimizer_steps_per_inner_transaction"] == 1
    assert manifest["training"]["ppo_epochs"] == 2
    assert manifest["training"]["maximum_actor_proposals_per_seed_cell"] == 800
    assert manifest["training"]["critic_optimizer_updates"] == 0
    assert manifest["training"]["auxiliary_fit_updates"] == 0
    assert {key: manifest["training"][key] for key in (
        "loss_agg_mode", "clip_ratio", "clip_ratio_low", "clip_ratio_high",
        "clip_ratio_c", "use_kl_loss", "kl_loss_type",
        "kl_loss_coefficient", "entropy_coefficient",
    )} == {
        "loss_agg_mode": "token-mean", "clip_ratio": .2,
        "clip_ratio_low": .2, "clip_ratio_high": .2, "clip_ratio_c": 3.,
        "use_kl_loss": True, "kl_loss_type": "low_var_kl",
        "kl_loss_coefficient": .001, "entropy_coefficient": 0.,
    }


@pytest.mark.parametrize(
    "path,value",
    [
        (("method", "optimizer_steps_per_inner_transaction"), 2),
        (("training", "ppo_epochs"), 1),
        (("training", "kl_loss_type"), "mse"),
        (("training", "loss_agg_mode"), "seq-mean-token-sum"),
        (("training", "confirmatory_seed_values"), [2026, 2027, 2028]),
        (("performance", "s128_role"), "blind_final"),
        (("checkpointing", "full_recovery_keep"), 5),
        (("backend", "hf_fallback"), True),
    ],
)
def test_manifest_schema_rejects_frozen_contract_drift(path, value):
    manifest = json.loads(MANIFEST_PATH.read_text())
    schema = json.loads(SCHEMA_PATH.read_text())
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(manifest)


def test_cells_and_confirmatory_inference_are_frozen_not_three_seed_pseudoreplication():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["method_cells"] == {
        "D": {"objective_variant": "original_tokenwise", "controller_variant": "none"},
        "C": {"objective_variant": "original_tokenwise", "controller_variant": "feasible_backtracking"},
        "E": {"objective_variant": "per_write_joint", "controller_variant": "feasible_backtracking"},
        "B": {"objective_variant": "whole_prefix", "controller_variant": "feasible_backtracking"},
        "A": {"objective_variant": "whole_prefix", "controller_variant": "hard_rollback", "maximum_round": 50},
    }
    assert manifest["training"]["mechanism_seeds"] == [2026, 2027, 2028]
    assert manifest["training"]["confirmatory_seed_values"] == list(range(2026, 2034))
    assert manifest["performance"]["co_primary_contrasts"] == ["B-D", "B-E"]
    assert manifest["performance"]["minimum_effects"] == {"B-D": .02, "B-E": .01}


def test_actor_source_has_one_counted_step_per_inner_and_frozen_batch():
    source = (ROOT / "verl/workers/actor/dp_actor.py").read_text()
    assert "if int(self.config.ppo_epochs) != 2" in source
    assert "if rwwpo_enabled and len(dataloader) != 1" in source
    assert "RWWPO transactions require one full optimizer minibatch per inner update" in source
    assert "transaction_optimizer_step_calls += 1" in source
    assert "transaction_optimizer_step_calls != 1" in source
    assert '"optimizer_step_calls": transaction_optimizer_step_calls' in source
    assert '"shared_kl_loss": float(shared_additive_loss.detach().item())' in source
    assert 'ref_log_prob=(joined("ref_log_prob") if rwwpo2_enabled else None)' in source
    assert '"ref_log_prob": joined("ref_log_prob")' in source
    assert "RWWPO2_BEHAVIOR_BATCH_MUTATED_BETWEEN_INNER_UPDATES" in source
    assert "named_buffer_snapshot(self.actor_module)" in source
    assert "if rwwpo2_enabled else None" in source
    assert "restore_named_buffers(" in source
    assert "behavior_forward_rng_digests = ordered_rng_state_digests(" in source
    assert "def replay_behavior_log_probs():" in source
    assert "replay_with_rng_snapshots(" in source
    assert '"replay_rng_bound": True' in source
    assert "post_commit_forward_verified = True" in source
    assert 'rwwpo_controller == "none"' in source
    assert "post_constraint_valid" in source
    assert '"transaction_entry_buffer_digest"' in source
    assert '"terminal_buffer_digest"' in source
    assert "RWWPO2_POST_COMMIT_FORWARD_CLOSURE_FAILURE" in source
    assert "append_transaction_failure_record(" in source
    legacy_branch = source.split("elif accepted:", 1)[1].split(
        "commit_params = parameter_snapshot", 1)[0]
    assert "restore_named_buffers" not in legacy_branch
    # R50 must include both behavior and off-behavior same-host shadows every round.
    assert "r50_host_shadow = round_id <=" in source
    assert "inner_id == 2 and round_id <=" not in source


def test_launcher_is_dynamic_locked_append_only_and_s128_free():
    common = (ROOT / "scripts/h20/rwwpo2_common.sh").read_text()
    launcher = (ROOT / "scripts/h20/run_qwen25_7b_rwwpo2.sh").read_text()
    assert "GPU_PAIR" in common and "RWWPO_GPU_A < RWWPO_GPU_B" in common
    assert "memagent_h20_gpu_${RWWPO_GPU_A}.lock" in common
    assert "memagent_h20_gpu_${RWWPO_GPU_B}.lock" in common
    assert "flock -n 8" in common and "flock -n 9" in common
    assert "run ID already consumed" in common
    assert not any(token in common + launcher for token in ("kill -9", "pkill", "killall"))
    assert "SAVE_FREQ=10" in launcher and "MAX_ACTOR_CKPT_TO_KEEP=2" in launcher
    assert "hotpotqa_dev.parquet" not in launcher
    assert "RWWPO_R50_PROGRAM_GATE" in launcher
    assert "RWWPO_CONFIRMATION_SEAL" in launcher


def test_attempt_id_cannot_enter_logical_seed_or_algorithmic_randomness():
    source = (ROOT / "recurrent/research/rwwpo_transaction.py").read_text()
    signature = source.split("def logical_transaction_seed", 1)[1].split(")", 1)[0]
    body = source.split("def logical_transaction_seed", 1)[1].split(
        "def seed_transaction_rng", 1)[0]
    assert "attempt" not in signature
    assert "attempt" not in body
    launcher = (ROOT / "scripts/h20/rwwpo2_common.sh").read_text()
    assert "RWWPO_ATTEMPT_ID=$RWWPO_RUN_ID" in launcher
    assert "RUN_SEED" not in launcher.split("rwwpo2_export_runtime", 1)[1]


def test_trial_and_fresh_certificates_use_behavior_microbatch_rng_replay():
    actor = (ROOT / "verl/workers/actor/dp_actor.py").read_text()
    transaction = (ROOT / "recurrent/research/rwwpo_transaction.py").read_text()
    schema = (ROOT / "rwwpo2_actual_loss_receipt.schema.json").read_text()
    helper = actor.split("def replay_behavior_log_probs():", 1)[1].split(
        "current_log_prob =", 1)[0]
    trial = actor.split("for alpha in candidates:", 1)[1].split(
        "trial_prefix_rows =", 1)[0]
    trial_rwwpo2_replay = (
        "if rwwpo2_enabled:\n"
        "                                with torch.no_grad():\n"
        "                                    trial_log_prob = "
        "replay_behavior_log_probs()"
    )
    trial_legacy_replay = (
        "else:\n"
        "                                restore_rng(proposal_gradient_rng)"
    )
    trial_rwwpo2_start = trial.index(trial_rwwpo2_replay)
    trial_rwwpo2 = trial[trial_rwwpo2_start:trial.index(
        "\n                            else:", trial_rwwpo2_start)]
    fresh = actor.split(
        "# A trial tensor is not a commit certificate.", 1)[1].split(
            "post_prefix_rows =", 1)[0]
    assert "restore_named_buffers(" in helper
    assert "replay_with_rng_snapshots(" in helper
    assert "finally:" in helper
    assert trial_rwwpo2_replay in trial_rwwpo2
    assert "restore_rng(proposal_gradient_rng)" not in trial_rwwpo2
    assert trial_legacy_replay in trial
    assert "replay_behavior_log_probs()" in fresh
    assert "restore_rng(proposal_gradient_rng)" not in fresh
    assert "def ordered_rng_state_digests(" in transaction
    assert "finally:\n        restore_rng(terminal)" in transaction
    for field in ("behavior_forward_rng_digests",
                  "behavior_forward_rng_aggregate_digest",
                  "replay_microbatch_count", "replay_rng_bound"):
        assert field in schema


def test_recovery_contract_authenticates_prefix_and_excludes_failed_suffix():
    lineage = (ROOT / "tools/h20/audit_rwwpo2_lineage_parent.py").read_text()
    trainer = (ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
    attempt = (ROOT / "tools/h20/audit_rwwpo2_attempt.py").read_text()
    firewall = (ROOT / "tools/h20/audit_rwwpo2_source_firewall.py").read_text()
    for token in ("prefix_sha256", "tail_sha256", "tensor_inventory",
                  "checkpoint_inventory_event_sha256", "failed_suffix_imported",
                  "record_limits", "rwwpo_rollout_seed_anchor",
                  'parser.add_argument(\n        "--producer-commit"',
                  '"producer_git_commit": producer_commit',
                  '"auditor_git_commit": head',
                  '"auditor_source_sha256": sha256_file(Path(__file__).resolve())'):
        assert token in lineage
    assert "execution_prefix_to_checkpoint" in lineage
    assert "execution_prefix_through_round" in attempt
    assert "record_limits=record_limits" in attempt
    assert "RWWPO2_RESUME_ACCEPTED_OPTIMIZER_CLOCK_DRIFT" in trainer
    assert "rwwpo_rollout_seed_anchor" in trainer
    assert "rwwpo2_recovery_prune_intent" in trainer
    assert "rwwpo2_recovery_pruned" in trainer
    assert "prune_intent_record_sha256" in trainer
    prune_method = trainer[
        trainer.index("    def _prune_rwwpo2_recovery_roots(self):"):
        trainer.index("    def _save_rwwpo2_actor_anchor(self):")
    ]
    assert "from recurrent.research.gate_a_execution import (" in prune_method
    assert "append_gate_a_record," in prune_method
    assert prune_method.index('"rwwpo2_recovery_prune_intent"') < \
        prune_method.index("shutil.rmtree(path)") < \
        prune_method.index('"rwwpo2_recovery_pruned"')
    assert "scientific_anchor_preserved=True" in trainer
    assert "anchor hardlink" in attempt
    assert '"resolved_contract_file_sha256"' in attempt
    assert '"resolved_contract_report_sha256"' in attempt
    assert 'parser.add_argument("--preflight", required=True)' in attempt
    assert '"preflight_report_sha256"' in attempt
    assert "R400 preflight gate binding" in attempt
    assert "preflight lineage start" in attempt
    assert "validate_rwwpo2_rng_phase_digests(row)" in attempt
    assert "validate_transaction_failure_boundary(" in attempt
    assert "validate_post_commit_forward_binding(" in attempt
    assert "validate_recovery_prune_evidence(" in attempt
    for bound_source in (
            'ROOT/"gate_a_execution_ledger.schema.json"',
            'ROOT/"recurrent/research/gate_a_execution.py"',
            'ROOT/"verl/trainer/ppo/ray_trainer.py"'):
        assert bound_source in firewall
    gate = (ROOT / "tools/h20/audit_rwwpo2_r50_program.py").read_text()
    assert "segment contract binding" in gate


def _prune_events(output_root: Path):
    def digest(value: int) -> str:
        return format(value % 16, "x") * 64

    events = []
    record_index = 0
    checkpoint_sha = {}
    anchor_sha = {}
    for round_id in (10, 20, 30, 40):
        checkpoint_sha[round_id] = digest(round_id // 10)
        anchor_sha[round_id] = digest(round_id // 10 + 4)
        events.extend((
            {"record_type": "checkpoint_inventory", "global_step": round_id,
             "record_index": record_index,
             "record_sha256": checkpoint_sha[round_id]},
            {"record_type": "rwwpo2_actor_anchor_inventory",
             "global_step": round_id, "record_index": record_index + 1,
             "record_sha256": anchor_sha[round_id]},
        ))
        record_index += 2
    for pruned_round, prune_at in ((10, 30), (20, 40)):
        root = str((output_root / f"global_step_{pruned_round}").resolve())
        intent_sha = digest(pruned_round // 10 + 8)
        common = {
            "global_step": prune_at, "pruned_round": pruned_round,
            "pruned_root": root,
            "checkpoint_inventory_record_sha256": checkpoint_sha[pruned_round],
            "scientific_anchor_inventory_record_sha256": anchor_sha[pruned_round],
            "scientific_anchor_preserved": True,
        }
        events.extend((
            {**common, "record_type": "rwwpo2_recovery_prune_intent",
             "record_index": record_index, "record_sha256": intent_sha},
            {**common, "record_type": "rwwpo2_recovery_pruned",
             "record_index": record_index + 1,
             "record_sha256": digest(pruned_round // 10 + 10),
             "prune_intent_record_sha256": intent_sha,
             "pruned_root_absent": True},
        ))
        record_index += 2
    for retained in (30, 40):
        (output_root / f"global_step_{retained}").mkdir()
    return events


def test_recovery_prune_requires_two_phase_authenticated_closure(tmp_path):
    events = _prune_events(tmp_path)
    summary = validate_recovery_prune_evidence(
        events, expected_checkpoint_rounds=[10, 20, 30, 40],
        output_root=tmp_path,
    )
    assert summary == {
        "retained_rounds": [30, 40],
        "pruned_rounds": [10, 20],
        "two_phase_evidence": True,
    }

    schema = json.loads((ROOT / "gate_a_execution_ledger.schema.json").read_text())
    for row in events:
        if row["record_type"] not in {
                "rwwpo2_recovery_prune_intent", "rwwpo2_recovery_pruned"}:
            continue
        schema_row = {
            **row,
            "experiment_name": "rwwpo2",
            "git_commit": "a" * 40,
            "run_id": "b" * 32,
            "recorded_at": "2026-08-26T00:00:00+00:00",
            "previous_record_sha256": "0" * 64,
        }
        jsonschema.Draft202012Validator(schema).validate(schema_row)

    missing_complete = [row for row in events if not (
        row["record_type"] == "rwwpo2_recovery_pruned"
        and row.get("pruned_round") == 10
    )]
    with pytest.raises(ValueError, match="intent/complete"):
        validate_recovery_prune_evidence(
            missing_complete, expected_checkpoint_rounds=[10, 20, 30, 40],
            output_root=tmp_path,
        )

    forged = copy.deepcopy(events)
    next(row for row in forged if row["record_type"] ==
         "rwwpo2_recovery_pruned")["prune_intent_record_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="semantic closure"):
        validate_recovery_prune_evidence(
            forged, expected_checkpoint_rounds=[10, 20, 30, 40],
            output_root=tmp_path,
        )


def test_recovery_prune_rejects_delete_before_complete(tmp_path):
    events = _prune_events(tmp_path)
    incomplete = [row for row in events if not (
        row["record_type"] == "rwwpo2_recovery_pruned"
        and row.get("pruned_round") == 20
    )]
    with pytest.raises(ValueError, match="intent/complete"):
        validate_recovery_prune_evidence(
            incomplete, expected_checkpoint_rounds=[10, 20, 30, 40],
            output_root=tmp_path,
        )


def test_recovery_prune_rejects_forged_post_delete_absence(tmp_path):
    events = _prune_events(tmp_path)
    (tmp_path / "global_step_10").mkdir()
    with pytest.raises(ValueError, match="semantic closure"):
        validate_recovery_prune_evidence(
            events, expected_checkpoint_rounds=[10, 20, 30, 40],
            output_root=tmp_path,
        )


def test_recovery_prune_runtime_executes_r30_two_phase_closure(
        tmp_path, monkeypatch):
    """Execute the real method body so a missing method-local import is fatal."""
    trainer_source = (ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
    trainer_tree = ast.parse(trainer_source)
    trainer_class = next(
        node for node in trainer_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer"
    )
    prune_method = copy.deepcopy(next(
        node for node in trainer_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prune_rwwpo2_recovery_roots"
    ))
    fixture_class = ast.ClassDef(
        name="PruneFixture", bases=[], keywords=[],
        body=[prune_method], decorator_list=[],
    )
    namespace = {"json": json, "os": os, "re": re, "shutil": shutil}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[fixture_class], type_ignores=[])), "<prune-fixture>", "exec"),
         namespace)

    output_root = tmp_path / "output"
    output_root.mkdir()
    for round_id in (10, 20, 30):
        checkpoint = output_root / f"global_step_{round_id}"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "actor" / "model.pt").write_bytes(
            f"model-{round_id}".encode())
        (checkpoint / "data.pt").write_bytes(f"data-{round_id}".encode())
    anchor_root = output_root / "scientific_anchors/round_10"
    (anchor_root / "actor").mkdir(parents=True)
    (anchor_root / "actor/model.pt").write_bytes(b"anchor-10")

    ledger = tmp_path / "execution.jsonl"
    common = {
        "experiment_name": "rwwpo2", "git_commit": "a" * 40,
        "run_id": "b" * 32, "recorded_at": "2026-08-26T00:00:00+00:00",
    }
    checkpoint_records = {}
    for round_id in (10, 20, 30):
        checkpoint_records[round_id] = append_jsonl(ledger, {
            **common, "record_type": "checkpoint_inventory",
            "global_step": round_id,
            "inventory": checkpoint_inventory(
                output_root / f"global_step_{round_id}"),
        })
    anchor_record = append_jsonl(ledger, {
        **common, "record_type": "rwwpo2_actor_anchor_inventory",
        "global_step": 10, "inventory": checkpoint_inventory(anchor_root),
    })
    monkeypatch.setenv("GATE_A_FROZEN_AUDIT", "1")
    monkeypatch.setenv("GATE_A_EXECUTION_LEDGER", str(ledger))
    monkeypatch.setenv("GATE_A_EXPERIMENT_NAME", "rwwpo2")
    monkeypatch.setenv("GATE_A_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("GATE_A_RUN_ID", "b" * 32)

    actor = {"rwwpo": {"enable": True, "program_version": "rwwpo2-k2"}}
    fixture = namespace["PruneFixture"]()
    fixture.global_steps = 30
    fixture.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(actor=actor),
        trainer=SimpleNamespace(default_local_dir=str(output_root)),
    )
    fixture._prune_rwwpo2_recovery_roots()

    assert not (output_root / "global_step_10").exists()
    assert (output_root / "global_step_20").is_dir()
    assert (output_root / "global_step_30").is_dir()
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert validate_jsonl_chain(rows) == []
    intent, complete = rows[-2:]
    assert intent["record_type"] == "rwwpo2_recovery_prune_intent"
    assert complete["record_type"] == "rwwpo2_recovery_pruned"
    assert complete["prune_intent_record_sha256"] == intent["record_sha256"]
    assert intent["checkpoint_inventory_record_sha256"] == \
        checkpoint_records[10]["record_sha256"]
    assert intent["scientific_anchor_inventory_record_sha256"] == \
        anchor_record["record_sha256"]
    assert complete["pruned_root_absent"] is True


def test_attempt_failure_evidence_is_prefix_aware_and_fail_closed(tmp_path):
    assert validate_transaction_failure_boundary(
        tmp_path, through_round=10) == []
    failure = tmp_path / "failure_rank0.jsonl"
    failure.write_text("")
    with pytest.raises(ValueError, match="empty transaction failure"):
        validate_transaction_failure_boundary(tmp_path, through_round=10)
    failure.unlink()
    failure.mkdir()
    with pytest.raises(ValueError, match="malformed transaction failure"):
        validate_transaction_failure_boundary(tmp_path, through_round=10)
    failure.rmdir()
    from recurrent.research.rwwpo_ledger import append_transaction_failure_record
    append_transaction_failure_record(
        ledger_dir=tmp_path,attempt_id="failed",rank=0,global_step=12,
        inner_id=1,proposal_clock=23,
        reason="RWWPO_PREFIX_TRUST_REGION_VIOLATION",phase="precondition",
        prefix_rows=[{"turn":0,"sample_index":0,
                      "root_identity_hash":"root","log_ratio":4.1}],
        prefix_stats=[{"turn":0,"feasible":False}],
        current_reference_max_abs=.1,
        behavior_batch_digest="a"*64,
        transaction_entry_buffer_digest="b"*64)
    assert len(validate_transaction_failure_boundary(
        tmp_path, through_round=10)) == 1
    with pytest.raises(ValueError, match="inside audited prefix"):
        validate_transaction_failure_boundary(tmp_path, through_round=12)


def test_attempt_binds_post_commit_verification_to_numeric_oracle():
    row = {"mechanism_diagnostics": {
        "post_commit_forward_verified": True,
        "post_commit_forward_verification_tolerance": 1e-6,
    }}
    validate_post_commit_forward_binding([row], tau_logprob=1e-6)
    drifted = copy.deepcopy(row)
    drifted["mechanism_diagnostics"][
        "post_commit_forward_verification_tolerance"] = 2e-6
    with pytest.raises(ValueError, match="post-commit forward binding"):
        validate_post_commit_forward_binding([drifted], tau_logprob=1e-6)


def test_execution_checkpoint_prefix_ignores_malformed_failed_suffix(tmp_path):
    path=tmp_path/"execution.jsonl"
    commit="a"*40
    append_jsonl(path,{"record_type":"weight_sync_summary",
                       "global_step":10,"git_commit":commit})
    append_jsonl(path,{"record_type":"checkpoint_inventory",
                       "global_step":10,"git_commit":commit,
                       "inventory":{}})
    with path.open("ab") as stream:
        stream.write(b'{"malformed_failed_suffix":')
    events,checkpoint,prefix_sha=execution_prefix_to_checkpoint(
        path,checkpoint_round=10,expected_commit=commit)
    assert len(events)==2 and checkpoint["global_step"]==10
    assert len(prefix_sha)==64


def test_completed_attempt_prefix_keeps_same_round_anchor_then_stops_at_future(tmp_path):
    path=tmp_path/"execution.jsonl"
    commit="b"*40
    for row in (
        {"record_type":"checkpoint_inventory","global_step":10,
         "git_commit":commit,"inventory":{}},
        {"record_type":"rwwpo2_actor_anchor_inventory","global_step":10,
         "git_commit":commit,"inventory":{}},
        {"record_type":"execution_signal","global_step":11,
         "git_commit":commit},
    ):
        append_jsonl(path,row)
    events,checkpoint,_=execution_prefix_through_round(
        path,target_round=10,expected_commit=commit)
    assert [row["record_type"] for row in events]==[
        "checkpoint_inventory","rwwpo2_actor_anchor_inventory"]
    assert checkpoint["global_step"]==10


def test_numeric_oracle_calibrates_the_registered_projection_statistic():
    producer = (ROOT / "tools/h20/calibrate_rwwpo2_numeric_oracle.py").read_text()
    auditor = (ROOT / "tools/h20/audit_rwwpo2_numeric_oracle.py").read_text()
    resolver = (ROOT / "tools/h20/materialize_rwwpo2_resolved_contract.py").read_text()
    assert "ROOT = Path(__file__).resolve().parents[2]" in producer
    assert "sys.path.insert(0, str(ROOT))" in producer
    assert producer.index("sys.path.insert(0, str(ROOT))") < producer.index(
        "from verl.trainer.ppo.core_algos import"
    )
    for token in ("repeated_gradient_projection_relative_l2",
                  "streamed_replay_gradient_projection_relative_l2",
                  "save_load_gradient_projection_relative_l2",
                  "behavior_actual_loss_logprob_max_abs",
                  "behavior_actual_loss_coefficient_max_abs",
                  "behavior_actual_loss_gradient_projection_relative_l2",
                  "tau_coefficient"):
        assert token in producer and token in auditor
    assert all("streamed_replay_calibration" in source
               for source in (producer,auditor,resolver))
    assert "gradient_sketch_relative_l2" not in producer
    assert "threshold_multiplier" in producer and "threshold reconstruction" in auditor
    assert "GRADIENT_SKETCH_CHUNK_ELEMENTS=RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS" in producer
    assert "parameter.grad.detach().double().flatten()" not in producer
    assert all("gradient_sketch_chunk_elements" in source
               for source in (producer,auditor,resolver))
    streamed = producer.split("def streamed_replay_gradient",1)[1].split(
        "def behavior_actual_loss_gradient",1)[0]
    for token in ("gradient_checkpointing_enable(",
                  'gradient_checkpointing_kwargs={"use_reentrant":False}',
                  "get_fsdp_wrap_policy(",
                  "auto_wrap_policy=auto_wrap_policy",
                  "mixed_precision=mixed_precision",
                  "sharding_strategy=ShardingStrategy.FULL_SHARD",
                  "sync_module_states=True,use_orig_params=False",
                  "forward_prefetch=False",
                  'torch.autocast(device_type="cuda",dtype=torch.bfloat16)',
                  "apply_monkey_patch(model=model,ulysses_sp_size=1)",
                  "logprobs_from_logits(", "inplace_backward=True"):
        assert token in producer
    assert "torch.log_softmax" not in streamed
    behavior_old_anchor = (
        'with torch.no_grad():\n'
        '        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):\n'
        '            behavior_logits=model(input_ids=tokens,use_cache=False).logits\n'
        '            behavior_old_logp=logprobs_from_logits('
    )
    assert behavior_old_anchor in producer
    for token in ("gradient_checkpointing", "fsdp_auto_wrap_policy",
                  "cuda_autocast_dtype",
                  "selective_logprob_kernel", "model_load_dtype",
                  "fsdp_sharded_parameter_dtype",
                  "transaction_closure_probe",
                  "transaction_optimizer_probe",
                  "transaction_optimizer_lr",
                  "transaction_optimizer_betas",
            "transaction_optimizer_weight_decay",
            "transaction_optimizer_grad_clip",
                  "transaction_closure_sequence_length",
                  "transaction_closure_active_tokens",
                  "transaction_writeback_max_wall_seconds",
                  "fsdp_parameter_commit_primitive"):
        assert all(token in source for source in (producer,auditor,resolver))
    assert "torch_dtype=torch.float32" in producer
    assert "torch_dtype=torch.bfloat16" not in producer.split(
        "AutoModelForCausalLM.from_pretrained", 1)[1].split(")", 1)[0]
    for token in (
            "RWWPO2_FSDP_TRANSACTION_CLOSURE_NO_GO",
            "safe_candidate_recommit_max_abs", "safe_restore_max_abs",
            "transaction_backward_probe(model, input_ids)",
            '"step_calls": 1',
            "legacy_raw_copy_diagnostic", "FSDP.summon_full_params(",
            "torch.distributed.all_gather_into_tensor(",
            "RWWPO2_FSDP_DISTRIBUTED_INVENTORY_DRIFT"):
        assert token in producer + (ROOT / "recurrent/research/rwwpo_transaction.py").read_text()


def test_live_rwwpo2_binds_fsdp_safe_writeback_and_behavior_reference():
    actor = (ROOT / "verl/workers/actor/dp_actor.py").read_text()
    auditor = (ROOT / "tools/h20/audit_rwwpo_actual_loss.py").read_text()
    schema = (ROOT / "rwwpo2_actual_loss_receipt.schema.json").read_text()
    for token in (
            "synchronize_fsdp=rwwpo2_enabled",
            "behavior_current_logprob_digest",
            "behavior_current_logprob_integrity_verified",
            "RWWPO2_FSDP_PARAMETER_COMMIT_PRIMITIVE",
            "RWWPO2_FSDP_WRITEBACK_MAX_WALL_SECONDS",
            "_timed_set_interpolated_parameters",
            "RWWPO2_FSDP_WRITEBACK_BUDGET_EXCEEDED",
            "already the exact committed parameter state"):
        assert token in actor
    for token in (
            "immutable behavior logprob digest mismatch",
            "RWWPO-2 FSDP/behavior-reference closure"):
        assert token in auditor
    assert "fsdp_unitwise_allgather_summon_writeback_v1" in schema
    assert '"fsdp_parameter_writeback_max_wall_seconds": {"const": 120.0}' in schema
    assert '"max_trial_forward_wall_seconds": {"const": 600.0}' in schema
    frozen_batch = actor.split("frozen_digest = digest({", 1)[1].split("})", 1)[0]
    assert "current_log_prob" not in frozen_batch


def _numeric_transaction_closure_fixture():
    inventory = {
        "unit_count": 2, "managed_unit_count": 2,
        "training_states": {"TrainingState.IDLE": 2},
        "storage": {"flat_param_data:torch.float32:cuda": {
            "tensor_count": 2, "numel": 8, "allocated_bytes": 32,
            "nonzero_data_ptr_count": 2,
        }},
    }
    return [{
        "rank": rank, "status": "PASS",
        "decision": "RWWPO2_FSDP_TRANSACTION_CLOSURE_PASS",
        "primitive": "fsdp_unitwise_allgather_summon_writeback_v1",
        "behavior_logprob_digest": str(rank) * 64,
        "sequence_length": 8191, "active_tokens": 1024,
        "tau_logprob": 1e-6,
        "writeback_max_wall_seconds": 120.0,
        "writeback_wall_seconds": {
            "safe_behavior": 1.0, "safe_candidate": 1.1,
            "safe_candidate_recommit": 1.2, "safe_restore": 1.3,
        },
        "safe_errors": {
            "after_backward_max_abs": 0.0,
            "safe_noop_writeback_max_abs": 0.0,
            "safe_candidate_recommit_max_abs": 0.0,
            "safe_restore_max_abs": 0.0,
            "safe_second_forward_max_abs": 0.0,
        },
        "safe_candidate_activation_max_abs": 0.01,
        "legacy_raw_copy_diagnostic": {
            "candidate_activation_max_abs": 0.01, "restore_max_abs": 0.8,
        },
        "optimizer_probe": {
            "kind": "AdamW", "lr": 1e-6,
            "betas": [0.9, 0.999], "weight_decay": 0.01,
            "grad_clip": 1.0, "step_calls": 1, "grad_norm": 1.5,
            "proposal_max_abs": 1e-6, "state_entry_counts": [2, 2],
        },
        "phases": [
            {"phase": "T0_behavior", "logprob_digest": str(rank) * 64,
             "execution_inventory": copy.deepcopy(inventory)},
            {"phase": "T1_after_backward", "max_abs": 0.0,
             "execution_inventory": copy.deepcopy(inventory)},
            {"phase": "T2_after_real_optimizer_step",
             "optimizer_proposal_max_abs": 1e-6,
             "execution_inventory": copy.deepcopy(inventory)},
            {"phase": "T3_legacy_raw_restore_diagnostic",
             "candidate_activation_max_abs": 0.01,
             "restore_max_abs": 0.8,
             "execution_inventory": copy.deepcopy(inventory)},
            {"phase": "T4_safe_behavior_writeback", "max_abs": 0.0,
             "execution_inventory": copy.deepcopy(inventory)},
            {"phase": "T5_safe_candidate_recommit",
             "candidate_activation_max_abs": 0.01,
             "recommit_max_abs": 0.0,
             "execution_inventory": copy.deepcopy(inventory)},
            {"phase": "T6_safe_restore_fresh", "max_abs": 0.0,
             "second_forward_max_abs": 0.0,
             "execution_inventory": copy.deepcopy(inventory)},
        ],
    } for rank in (0, 1)]


def test_numeric_transaction_closure_semantics_are_independently_enforced():
    authentic = _numeric_transaction_closure_fixture()
    assert validate_fsdp_transaction_closure(
        authentic, tau_logprob=1e-6) == authentic
    attacks = []
    drift = copy.deepcopy(authentic)
    drift[1]["sequence_length"] = 8
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[0]["safe_errors"]["safe_restore_max_abs"] = 0.8
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[0]["writeback_wall_seconds"]["safe_restore"] = 121.0
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[0]["safe_candidate_activation_max_abs"] = 1e-6
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[0]["optimizer_probe"]["proposal_max_abs"] = 0.0
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[1]["optimizer_probe"]["state_entry_counts"] = [2, 0]
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[0]["optimizer_probe"]["step_calls"] = 0
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[0]["optimizer_probe"]["grad_norm"] = 0.0
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[0]["optimizer_probe"]["state_entry_counts"] = [2, 3]
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[0]["phases"][2]["execution_inventory"]["training_states"] = {
        "TrainingState.FORWARD_BACKWARD": 2}
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[0]["phases"][5]["recommit_max_abs"] = 1e-7
    attacks.append(drift)
    drift = copy.deepcopy(authentic)
    drift[0]["phases"].pop()
    attacks.append(drift)
    for attack in attacks:
        with pytest.raises(ValueError, match="transaction closure"):
            validate_fsdp_transaction_closure(attack, tau_logprob=1e-6)


def test_manifest_freezes_fsdp_transaction_cost_and_commit_contract():
    manifest = json.loads((ROOT /
        "manifests/h20/qwen25_7b_rwwpo2_r400_k2_seed2026.json").read_text())
    method = manifest["method"]
    assert method["fsdp_parameter_commit_primitive"] == \
        "fsdp_unitwise_allgather_summon_writeback_v1"
    assert method["fsdp_parameter_writeback_max_wall_seconds"] == 120.0
    assert method["max_trial_forward_wall_seconds_per_transaction"] == 600.0


def test_chunked_gradient_sketch_matches_registered_full_vector_projection():
    from recurrent.research.rwwpo_transaction import (
        local_gradient_sketch_sufficient_statistics,
    )
    parameters=[]
    for shape,offset in (((3,5),-0.7),((2,4),0.3)):
        parameter=torch.nn.Parameter(torch.zeros(shape,dtype=torch.float32))
        parameter.grad=torch.arange(parameter.numel(),dtype=torch.float32).reshape(
            shape).div_(7).add_(offset)
        parameters.append(parameter)
    actual=local_gradient_sketch_sufficient_statistics(
        parameters,chunk_elements=3)
    expected=torch.zeros(4,dtype=torch.float64)
    for parameter_index,parameter in enumerate(parameters):
        gradient=parameter.grad.detach().double().flatten()
        coordinate=torch.arange(gradient.numel(),dtype=torch.int64)
        alternating=((coordinate+parameter_index)&1).double().mul_(2).sub_(1)
        saw=(((coordinate+17*parameter_index)%257).double()-128.)/128.
        expected[0]+=gradient.square().sum(); expected[1]+=gradient.sum()
        expected[2]+=(gradient*alternating).sum(); expected[3]+=(gradient*saw).sum()
    torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)


def test_chunked_gradient_sketch_rejects_noncontiguous_full_shard_copy():
    from recurrent.research.rwwpo_transaction import (
        local_gradient_sketch_sufficient_statistics,
    )
    parameter=torch.nn.Parameter(torch.zeros((3,4),dtype=torch.float32))
    parameter.grad=torch.arange(12,dtype=torch.float32).reshape(4,3).t()
    assert not parameter.grad.is_contiguous()
    with pytest.raises(RuntimeError,match="NONCONTIGUOUS_GRADIENT_NO_GO"):
        local_gradient_sketch_sufficient_statistics(
            [parameter],chunk_elements=3)


def test_live_actor_uses_the_registered_bounded_gradient_sketch():
    actor=(ROOT/"verl/workers/actor/dp_actor.py").read_text()
    transaction=(ROOT/"recurrent/research/rwwpo_transaction.py").read_text()
    assert "local_gradient_sketch_sufficient_statistics(" in actor
    assert "RWWPO2_GRADIENT_SKETCH_CHUNK_CONTRACT_DRIFT" in actor
    assert '"gradient_sketch_chunk_elements":' in actor
    assert "def local_gradient_sketch_sufficient_statistics(" in transaction
    assert "parameter.grad.detach().double().flatten()" not in actor
    assert "parameter.grad.detach().flatten()" not in actor


def test_numeric_oracle_direct_file_entry_imports_repo_from_foreign_cwd(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/h20/calibrate_rwwpo2_numeric_oracle.py"),
         "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--expected-commit" in result.stdout


def test_preflight_direct_file_entry_imports_repo_from_foreign_cwd(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/h20/preflight_rwwpo2.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--release-test-receipt" in result.stdout


def test_release_test_receipt_is_machine_bound_to_both_gpu_entries(tmp_path):
    producer = (ROOT / "tools/h20/run_rwwpo2_release_tests.py").read_text()
    verifier = (ROOT / "tools/h20/verify_rwwpo2_release_tests.py").read_text()
    numeric = (ROOT / "scripts/h20/run_rwwpo2_numeric_oracle.sh").read_text()
    launcher = (ROOT / "scripts/h20/run_qwen25_7b_rwwpo2.sh").read_text()
    preflight = (ROOT / "tools/h20/preflight_rwwpo2.py").read_text()
    for token in ("TEST_INVENTORY", "RWWPO2_RELEASE_TESTS_PASS",
                  "checkout_postcondition", "pytest_command(mode=\"collect\""):
        assert token in producer
    for token in ("--junitxml", "junit_summary", "test_source_sha256",
                  "python_executable_sha256", "installed_distributions_sha256",
                  "collect_current_node_ids", "non-PASS/skip/xfail"):
        assert token in verifier
    assert "verify_rwwpo2_release_tests.py" in numeric
    assert "RWWPO_RELEASE_TEST_RECEIPT" in numeric + launcher
    assert "verify_release_test_receipt" in preflight


def test_release_test_receipt_reopens_nodes_environment_log_and_sources(
        tmp_path, monkeypatch):
    work = tmp_path / "work"
    root = work / "logs/rwwpo2_release_tests/release_test_fixture"
    root.mkdir(parents=True)
    tombstone = root / "RUN_ID_CONSUMED"
    collect_log = root / "pytest_collect.log"
    collection_json = root / "collection.json"
    log = root / "pytest.log"
    execution_json = root / "execution.json"
    junit = root / "pytest.xml"
    manifest_sha = sha256_file(MANIFEST_PATH)
    commit = "a" * 40
    tombstone.write_text(f"{commit}:{manifest_sha}\n")
    collect_log.write_text("fixture collection pass\n")
    log.write_text("fixture pass\n")
    nodeids = collect_current_node_ids()
    collection_json.write_text(json.dumps({
        "schema_version": "rwwpo2-pytest-node-evidence-v1",
        "mode": "collect", "pytest_exitstatus": 0,
        "collected_node_ids": nodeids,
        "phase_reports": {nodeid: [] for nodeid in nodeids},
    }, sort_keys=True, indent=2) + "\n")
    execution_json.write_text(json.dumps({
        "schema_version": "rwwpo2-pytest-node-evidence-v1",
        "mode": "execute", "pytest_exitstatus": 0,
        "collected_node_ids": nodeids,
        "phase_reports": {nodeid: [
            {"when": "setup", "outcome": "passed", "wasxfail": False},
            {"when": "call", "outcome": "passed", "wasxfail": False},
            {"when": "teardown", "outcome": "passed", "wasxfail": False},
        ] for nodeid in nodeids},
    }, sort_keys=True, indent=2) + "\n")
    junit.write_text('<testsuites><testsuite>' + "".join(
        f'<testcase classname="fixture" name="pass_{index}"/>'
        for index in range(len(nodeids))
    ) + '</testsuite></testsuites>\n')
    def evidence(path):
        return {"relative_path": path.name, "size": path.stat().st_size,
                "sha256": sha256_file(path)}
    row = {
        "schema_version": "rwwpo2-release-tests-v1",
        "status": "PASS", "decision": "RWWPO2_RELEASE_TESTS_PASS",
        "git_commit": commit, "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": manifest_sha, "run_id": root.name,
        "runtime_environment": runtime_environment(),
        "pytest_collection_returncode": 0, "pytest_returncode": 0,
        "checkout_postcondition": True,
        "test_inventory": list(TEST_INVENTORY),
        "test_source_sha256": {
            relative: sha256_file(ROOT / relative) for relative in TEST_INVENTORY
        },
        "collected_node_ids": nodeids,
        "run_id_tombstone": evidence(tombstone),
        "pytest_collect_log": evidence(collect_log),
        "collection_evidence": evidence(collection_json),
        "pytest_log": evidence(log),
        "execution_evidence": evidence(execution_json),
        "junit_xml": evidence(junit),
        "junit_summary": release_test_junit_summary(junit),
    }
    row["report_sha256"] = release_test_canonical_sha(row)
    report = root / "release_tests.json"
    report.write_text(json.dumps(row, sort_keys=True, indent=2) + "\n")
    report_sha = sha256_file(report)
    verify_release_test_receipt(
        report, receipt_sha256=report_sha, expected_commit=commit,
        manifest_path=MANIFEST_PATH, manifest_sha256=manifest_sha,
        work_root=work,
    )
    log.write_text("forged pass\n")
    with pytest.raises(ValueError, match="evidence byte drift"):
        verify_release_test_receipt(
            report, receipt_sha256=report_sha, expected_commit=commit,
            manifest_path=MANIFEST_PATH, manifest_sha256=manifest_sha,
            work_root=work,
        )
    log.write_text("fixture pass\n")
    drifted_environment = copy.deepcopy(row["runtime_environment"])
    drifted_environment["installed_distributions_sha256"] = "f" * 64
    monkeypatch.setattr(
        "tools.h20.verify_rwwpo2_release_tests.runtime_environment",
        lambda: drifted_environment,
    )
    with pytest.raises(ValueError, match="Python environment drift"):
        verify_release_test_receipt(
            report, receipt_sha256=report_sha, expected_commit=commit,
            manifest_path=MANIFEST_PATH, manifest_sha256=manifest_sha,
            work_root=work,
        )


def test_release_test_node_evidence_rejects_skip_or_deselection(tmp_path):
    nodeids = [f"{relative}::test_fixture" for relative in TEST_INVENTORY]
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps({
        "schema_version": "rwwpo2-pytest-node-evidence-v1",
        "mode": "execute", "pytest_exitstatus": 0,
        "collected_node_ids": nodeids,
        "phase_reports": {nodeid: [
            {"when": "setup", "outcome": "passed", "wasxfail": False},
            {"when": "call", "outcome": "passed", "wasxfail": False},
            {"when": "teardown", "outcome": "passed", "wasxfail": False},
        ] for nodeid in nodeids},
    }))
    row = json.loads(execution.read_text())
    row["phase_reports"][nodeids[0]][1]["outcome"] = "skipped"
    execution.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="skip/xfail"):
        node_evidence(execution, mode="execute")
    row["phase_reports"].pop(nodeids[-1])
    execution.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="execution outcomes"):
        node_evidence(execution, mode="execute")


def test_runtime_uses_distinct_behavior_coefficient_and_parameter_tolerances():
    launcher = (ROOT / "scripts/h20/run_qwen25_7b_rwwpo2.sh").read_text()
    gate = (ROOT / "experiments/7b_gate_a/run_gate_a.sh").read_text()
    actor = (ROOT / "verl/workers/actor/dp_actor.py").read_text()
    attempt = (ROOT / "tools/h20/audit_rwwpo2_attempt.py").read_text()
    for token in ("RWWPO_BEHAVIOR_COEFFICIENT_TOLERANCE",
                  "RWWPO_BEHAVIOR_GRADIENT_TOLERANCE"):
        assert token in launcher and token in gate
    assert '"behavior_coefficient_tolerance", 1e-9' in actor
    assert 'resolved["behavior_coefficient_tolerance"]' in attempt


def test_signed_receipt_rejects_commit_and_byte_tamper(tmp_path):
    commit = "a" * 40
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_signed({
        "status": "PASS", "decision": "X_PASS", "git_commit": commit,
        "value": 1,
    })))
    assert receipt(str(path), decision="X_PASS", commit=commit)["value"] == 1
    with pytest.raises(ValueError, match="invalid X_PASS receipt"):
        receipt(str(path), decision="X_PASS", commit="b" * 40)
    tampered = json.loads(path.read_text()); tampered["value"] = 2
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="invalid X_PASS receipt"):
        receipt(str(path), decision="X_PASS", commit=commit)


def test_round_cluster_interval_does_not_count_two_inner_transactions_as_independent():
    assert wilson_interval(10, 50) != wilson_interval(20, 100)
    low, high = wilson_interval(10, 50)
    assert 0 <= low < .2 < high <= 1


def test_contract_language_forbids_convergence_and_s128_relabeling():
    prereg = (ROOT / "docs/papers/rwwpo2_r400_k2_preregistration_20260823.md").read_text()
    self_check = (ROOT / "docs/papers/rwwpo2_evidence_self_check_20260823.md").read_text()
    assert "not a convergence claim" in prereg
    assert "adaptive development benchmark" in prereg
    assert "R400 may not be called converged" in prereg
    assert "early-budget" in self_check
    assert "optional, non-committing common-host probe" not in prereg
    assert "margin-centered" in prereg


def test_confirmation_exact_test_is_margin_centered_and_retains_zeros():
    assert one_sided_exact_paired_sign_flip([0.01] * 8) == 1 / 256
    assert one_sided_exact_paired_sign_flip([0.0] * 8) == 1.0
    with pytest.raises(ValueError, match="exactly eight"):
        one_sided_exact_paired_sign_flip([0.01] * 7)
    decisions = holm_two_test_decisions({"B-D": 1 / 256, "B-E": 2 / 256})
    assert all(item["reject"] for item in decisions.values())


def test_confirmation_entry_is_raw_artifact_backed_and_protocol_normalized():
    finalizer = (ROOT / "tools/h20/finalize_rwwpo2_confirmation.py").read_text()
    protocol = (ROOT / "recurrent/research/rwwpo2_confirmation.py").read_text()
    auditor = (ROOT / "tools/h20/audit_rwwpo2_confirmation_eval.py").read_text()
    runner = (ROOT / "scripts/h20/run_rwwpo2_confirmation_eval.sh").read_text()
    for token in ("terminal/400.jsonl", "score_terminal_output",
                  "metric row reconstruction", "expected_assignments"):
        assert token in finalizer
    for token in ("repository_relative_path", "path_sha256",
                  "confirmation_data_sha256",
                  "hydra_pre_dataset_max_prompt_length",
                  "memory_dataset_effective_max_prompt_length"):
        assert token in protocol
    for token in ("validate_actor_only_checkpoint_acknowledgements",
                  "optimizer_step_calls", "turn schedule"):
        assert token in auditor
    assert "memagent_h20_gpu_${GPU_A}.lock" in runner
    assert "flock -n 8" in runner and "flock -n 9" in runner
    assert not any(token in runner for token in ("kill -9", "pkill", "killall"))


def test_confirmation_protocol_binds_pre_and_post_dataset_prompt_limits(tmp_path):
    reward = ROOT / "recurrent/research/hotpotqa_dense_reward.py"
    config = {
        "recurrent": {"memory": {"config": {
            "max_chunks": 8, "chunk_size": 5000,
        }}},
        "data": {
            "shuffle": False, "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 1, "dataloader_num_workers": 0,
            "include_source_order_index": True, "truncation": "center",
            "context_key": "context", "val_max_samples": 512,
            "max_prompt_length": 8192, "max_response_length": 1024,
        },
        "actor_rollout_ref": {
            "model": {"use_remove_padding": True},
            "rollout": {key: value for key, value in {
                "name": "vllm", "mode": "sync", "n": 1,
                "tensor_model_parallel_size": 1,
                "dtype": "bfloat16", "load_format": "dummy_dtensor",
                "ignore_eos": False, "enforce_eager": False,
                "free_cache_engine": False, "gpu_memory_utilization": .55,
                "use_fire_sampling": False,
                "max_num_batched_tokens": 16384, "max_num_seqs": 16,
                "val_kwargs": {"do_sample": False},
            }.items()},
        },
        "reward_model": {"reward_manager": "naive"},
        "custom_reward_function": {
            "path": str(reward), "name": "compute_score",
            "reward_kwargs": {},
        },
    }
    protocol = generation_protocol_projection(
        config, repo=ROOT, confirmation_data_sha256="a" * 64,
        model={"id": "Qwen/Qwen2.5-7B-Instruct", "revision": "b" * 40},
    )
    assert protocol["data"]["hydra_pre_dataset_max_prompt_length"] == 8192
    assert protocol["data"]["memory_dataset_effective_max_prompt_length"] == 40000


def test_signed_confirmation_report_rejects_symlink(tmp_path):
    commit = "a" * 40
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_signed({
        "status": "PASS", "decision": "CONFIRM_PASS", "git_commit": commit,
    })))
    link = tmp_path / "linked.json"
    link.symlink_to(report)
    with pytest.raises(ValueError, match="symlink"):
        signed_report(link, decision="CONFIRM_PASS", commit=commit)
