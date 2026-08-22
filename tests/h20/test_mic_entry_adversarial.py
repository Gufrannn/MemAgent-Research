import importlib.util
import hashlib
import json
import os
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

from recurrent.research.mic import (
    CriticCheckpoint, append_jsonl_new, calibration_report, cross_fitted_values,
    innovation_ledger, sha256_file,
)
from recurrent.research.gate_a_execution import append_jsonl

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("mic_pipeline", REPO / "tools/h20/mic_pipeline.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


@pytest.mark.parametrize("value", ["0,0", "2,1", "2", "x,3", "-1,2", "1,2,3"])
def test_gpu_pair_drift_rejected(value):
    with pytest.raises(ValueError, match="GPU pair"):
        PIPELINE.parse_gpu_pair(value)


def test_gpu_pair_drift_real_shell_entry_rejection(tmp_path):
    env = os.environ.copy()
    env.update({
        "MEMAGENT_MIC_WORK_ROOT": str(tmp_path), "MEMAGENT_MIC_REPO_DIR": str(REPO),
        "MEMAGENT_MIC_EXPECTED_COMMIT": "a" * 40, "MEMAGENT_MIC_GPU_PAIR": "3,1",
        "MEMAGENT_MIC_RUN_ID": "pair-test",
    })
    result = subprocess.run(
        ["bash", "-c", f"source {REPO / 'scripts/h20/mic_common.sh'}"],
        env=env, text=True, capture_output=True,
    )
    assert result.returncode == 70
    assert "canonical ascending" in result.stderr


@pytest.mark.parametrize("extra,code,message", [
    ({"MEMAGENT_MIC_ENABLE": "0", "RUN_SEED": "2026"}, 60, "inactive method"),
    ({"MEMAGENT_MIC_ENABLE": "1", "RUN_SEED": "7"}, 61, "trajectory seed"),
])
def test_method_inactive_and_seed_tamper_real_training_entry(tmp_path, extra, code, message):
    env = os.environ.copy()
    env.update({"MEMAGENT_MIC_REQUIRED": "1", "WORK_ROOT": str(tmp_path), **extra})
    result = subprocess.run(
        ["bash", str(REPO / "experiments/7b_gate_a/run_gate_a.sh")],
        env=env, text=True, capture_output=True,
    )
    assert result.returncode == code
    assert message in result.stderr


def test_full_manifest_parity_rejects_unwhitelisted_drift(tmp_path):
    original = {"data": {"seed": 2026}, "algorithm": {"mic": {"enabled": False}}}
    method = {"data": {"seed": 7}, "algorithm": {"mic": {"enabled": True}}}
    original_path, method_path = tmp_path / "original.json", tmp_path / "method.json"
    original_path.write_text(json.dumps(original)); method_path.write_text(json.dumps(method))
    inventory = {"resolved_config_parity": {
        "original": str(original_path), "method": str(method_path),
        "original_sha256": sha256_file(original_path), "method_sha256": sha256_file(method_path),
    }}
    manifest = {"only_allowed_scientific_differences": ["algorithm.mic"]}
    with pytest.raises(ValueError, match="illegal resolved config diff"):
        PIPELINE.validate_full_resolved_parity(manifest, inventory)


def test_method_inactive_rejected_by_full_parity(tmp_path):
    payload = {"data": {"seed": 2026}, "algorithm": {"mic": {"enabled": False}}}
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    one.write_text(json.dumps(payload)); two.write_text(json.dumps(payload))
    inventory = {"resolved_config_parity": {
        "original": str(one), "method": str(two),
        "original_sha256": sha256_file(one), "method_sha256": sha256_file(two),
    }}
    with pytest.raises(ValueError, match="illegal resolved config diff"):
        PIPELINE.validate_full_resolved_parity(
            {"only_allowed_scientific_differences": ["algorithm.mic"]}, inventory
        )


def test_checkpoint_resume_and_ledger_tamper_rejected(tmp_path):
    checkpoint = tmp_path / "critic.json"
    CriticCheckpoint("a" * 40, "b" * 64, "c" * 64, {
        "oof": {}, "history_states": [], "history_outcomes": {},
        "parent_checkpoint_sha256": None,
    }).write_new(checkpoint)
    payload = json.loads(checkpoint.read_text())
    payload["critic_payload"]["parent_checkpoint_sha256"] = "f" * 64
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        CriticCheckpoint.read(checkpoint, expected_actor_commit="a" * 40)


def test_resume_corruption_real_training_entry_rejection(tmp_path):
    env = os.environ.copy()
    env.update({
        "WORK_ROOT": str(tmp_path), "PHASE": "resume", "RESUME_SOURCE_STEP": "5",
        "RESUME_TOTAL_STEPS": "10", "RESUME_FROM": str(tmp_path / "broken_step"),
        "MEMAGENT_MIC_REQUIRED": "1", "MEMAGENT_MIC_ENABLE": "1", "RUN_SEED": "2026",
    })
    result = subprocess.run(
        ["bash", str(REPO / "experiments/7b_gate_a/run_gate_a.sh")],
        env=env, text=True, capture_output=True,
    )
    assert result.returncode == 45
    assert "Missing complete step-5 checkpoint" in result.stderr


def _audit_fixture(tmp_path):
    gate_specs = {
        "p0": ("MIC_P0_PASS", {"git_commit": "a" * 40, "run_id": "audit-run"}),
        "e0": ("MIC_E0_PASS", {}), "e1": ("MIC_E1_PASS", {}),
        "paper_review": ("MIC_PAPER_REVIEW_GO", {}),
        "baseline": ("MIC_BASELINE_IMPORT_PASS", {}),
    }
    paths = {}
    for name, (decision, extra) in gate_specs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"status": "PASS", "decision": decision, **extra}))
        paths[name] = str(path)
    return SimpleNamespace(**paths, ledger=str(tmp_path / "ledger.jsonl"),
                           weight_ledger=str(tmp_path / "weights.jsonl"),
                           target_step=5, output=None)


def _valid_advantage_row(checkpoint, digest):
    return {
        "record_type": "mic_advantage_delivery", "global_step": 5,
        "critic_checkpoint": str(checkpoint), "critic_checkpoint_sha256": digest,
        "oof_bundle_sha256": "b" * 64, "parent_critic_checkpoint_sha256": None,
        "maximum_closure_error": 0.0,
        "calibration": {
            "mse": 0.2, "mae": 0.3, "calibration_slope": 0.8,
            "calibration_intercept": 0.1, "writer_innovation_mean": 0.01,
            "writer_innovation_variance": 0.4, "answer_residual_variance": 0.6,
        },
        "delivery": {"writer_active_tokens": 20, "answer_active_tokens": 5},
    }


def _real_checkpoint_and_advantage(tmp_path, *, degenerate=False):
    states, outcomes = [], {}
    for index in range(8):
        trajectory, root = f"audit-t{index}", f"audit-r{index}"
        outcome = 0.0 if degenerate else (1.0 if index % 2 else -1.0)
        outcomes[trajectory] = outcome
        common = {"stable_example_id": f"audit-e{index}", "stable_root_id": root,
                  "trajectory_id": trajectory, "question": "q"}
        states.extend([
            {**common, "turn_index": 0, "visible_chunks": [],
             "materialized_memory": "", "materialized_memory_history": [],
             "is_prewrite": True},
            {**common, "turn_index": 1, "visible_chunks": ["good" if outcome > 0 else "bad"],
             "materialized_memory": "good" if outcome > 0 else "bad",
             "materialized_memory_history": ["good" if outcome > 0 else "bad"],
             "is_prewrite": False},
        ])
    oof = cross_fitted_values(states, outcomes, fold_count=4, dimension=8)
    cumulative = innovation_ledger(oof, outcomes)
    checkpoint = tmp_path / "critic.json"
    digest = CriticCheckpoint("a" * 40, oof["bundle_sha256"], "c" * 64, {
        "oof": oof, "history_states": states, "history_outcomes": outcomes,
        "parent_checkpoint_sha256": None,
    }).write_new(checkpoint)
    advantage = _valid_advantage_row(checkpoint, digest)
    advantage.update({
        "oof_bundle_sha256": oof["bundle_sha256"],
        "current_trajectory_ids": list(outcomes),
        "innovation_ledger_sha256": cumulative["ledger_sha256"],
        "cumulative_innovation_ledger_sha256": cumulative["ledger_sha256"],
        "maximum_closure_error": cumulative["maximum_closure_error"],
        "calibration": calibration_report(cumulative),
    })
    return checkpoint, digest, advantage


def _valid_gradient_row():
    return {"record_type": "mic_actual_gradient_delivery", "global_step": 5,
            "role_metrics": {
                "mic_gradient/writer_pg_loss": 0.2,
                "mic_gradient/answer_pg_loss": 0.1,
                "mic_gradient/writer_active_tokens": 20,
                "mic_gradient/answer_active_tokens": 5,
                "mic_gradient/writer_logprob_grad_l2": 0.4,
                "mic_gradient/answer_logprob_grad_l2": 0.2,
                "mic_gradient/writer_logprob_grad_abs_max": 0.1,
                "mic_gradient/answer_logprob_grad_abs_max": 0.05,
            }}


def _append_valid_weight_sync(path, *, summary_digest="d" * 64):
    parameters = ["model.layers.0.self_attn.o_proj.weight"]
    for rank in (0, 1):
        append_jsonl(path, {
            "record_type": "weight_sync_ack", "sync_kind": "post_actor_update",
            "git_commit": "a" * 40, "run_id": "audit-run",
            "global_step": 5, "actor_version": 5, "vllm_ack_version": 5,
            "vllm_worker_rank": rank,
            "actor_master_sampled_tensor_digest": "e" * 64,
            "actor_rollout_sampled_tensor_digest": summary_digest,
            "actor_sampled_tensor_digest": summary_digest,
            "vllm_sampled_tensor_digest": summary_digest,
            "weight_transfer_format": "dtensor",
            "loaded_parameter_count": 199, "model_parameter_count": 199,
            "loaded_parameter_names_sha256": "f" * 64,
            "model_parameter_names_sha256": "f" * 64,
            "audited_loaded_parameters": parameters,
            "sampled_parameter_dtypes": {parameters[0]: "torch.bfloat16"},
        })
    append_jsonl(path, {
        "record_type": "weight_sync_summary", "sync_kind": "post_actor_update",
        "git_commit": "a" * 40, "run_id": "audit-run",
        "global_step": 5, "actor_version": 5, "worker_ranks": [0, 1],
        "sampled_tensor_digest": summary_digest,
        "actor_master_sampled_tensor_digest": "e" * 64,
    })


def test_checkpoint_tamper_rejected_through_real_audit_entry(tmp_path):
    args = _audit_fixture(tmp_path)
    checkpoint = tmp_path / "critic.json"
    digest = CriticCheckpoint("a" * 40, "b" * 64, "c" * 64, {
        "oof": {}, "history_states": [], "history_outcomes": {},
        "parent_checkpoint_sha256": None,
    }).write_new(checkpoint)
    append_jsonl_new(args.ledger, {
        "record_type": "mic_advantage_delivery", "global_step": 5,
        "critic_checkpoint": str(checkpoint), "critic_checkpoint_sha256": digest,
        "oof_bundle_sha256": "b" * 64, "parent_critic_checkpoint_sha256": None,
        "maximum_closure_error": 0.0,
    })
    append_jsonl_new(args.ledger, {"record_type": "mic_actual_gradient_delivery", "global_step": 5})
    payload = json.loads(checkpoint.read_text()); payload["critic_payload"]["history_states"] = ["x"]
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        PIPELINE.audit(args)


def test_ledger_tamper_rejected_through_real_audit_entry(tmp_path):
    args = _audit_fixture(tmp_path)
    append_jsonl_new(args.ledger, {"record_type": "mic_actual_gradient_delivery", "global_step": 5})
    row = json.loads(Path(args.ledger).read_text()); row["global_step"] = 6
    Path(args.ledger).write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="execution ledger chain corrupted"):
        PIPELINE.audit(args)


def test_weight_sync_tamper_rejected_through_real_audit_entry(tmp_path):
    args = _audit_fixture(tmp_path)
    _, _, advantage = _real_checkpoint_and_advantage(tmp_path)
    append_jsonl_new(args.ledger, advantage)
    append_jsonl_new(args.ledger, _valid_gradient_row())
    Path(args.weight_ledger).write_text(json.dumps({
        "record_index": 0, "previous_record_sha256": "0" * 64,
        "record_sha256": "f" * 64, "sync_kind": "post_actor_update", "global_step": 5,
    }) + "\n")
    with pytest.raises(ValueError, match="weight-sync ledger chain corrupted"):
        PIPELINE.audit(args)


def test_weight_sync_ack_summary_mismatch_rejected_through_real_audit_entry(tmp_path):
    args = _audit_fixture(tmp_path)
    _, _, advantage = _real_checkpoint_and_advantage(tmp_path)
    append_jsonl_new(args.ledger, advantage)
    append_jsonl_new(args.ledger, _valid_gradient_row())
    _append_valid_weight_sync(args.weight_ledger, summary_digest="d" * 64)
    rows = [json.loads(line) for line in Path(args.weight_ledger).read_text().splitlines()]
    rows[-1]["sampled_tensor_digest"] = "c" * 64
    # Re-chain the deliberately self-consistent tampered ledger so the semantic
    # summary-vs-ack guard, rather than the hash-chain guard, must reject it.
    Path(args.weight_ledger).unlink()
    for row in rows:
        for key in ("record_index", "previous_record_sha256", "record_sha256"):
            row.pop(key, None)
        append_jsonl(args.weight_ledger, row)
    with pytest.raises(ValueError, match="weight-sync summary mismatch"):
        PIPELINE.audit(args)


@pytest.mark.parametrize("field,value,message", [
    ("actor_version", 4, "summary version mismatch"),
    ("run_id", "other-run", "run binding mismatch"),
    ("git_commit", "b" * 40, "run binding mismatch"),
])
def test_weight_sync_summary_identity_tamper_rejected_after_rechain(
        tmp_path, field, value, message):
    args = _audit_fixture(tmp_path)
    _, _, advantage = _real_checkpoint_and_advantage(tmp_path)
    append_jsonl_new(args.ledger, advantage)
    append_jsonl_new(args.ledger, _valid_gradient_row())
    _append_valid_weight_sync(args.weight_ledger)
    rows = [json.loads(line) for line in Path(args.weight_ledger).read_text().splitlines()]
    rows[-1][field] = value
    Path(args.weight_ledger).unlink()
    for row in rows:
        for key in ("record_index", "previous_record_sha256", "record_sha256"):
            row.pop(key, None)
        append_jsonl(args.weight_ledger, row)
    with pytest.raises(ValueError, match=message):
        PIPELINE.audit(args)


@pytest.mark.parametrize("field,value,message", [
    ("mic_gradient/writer_logprob_grad_l2", 0.0, "writer gradient is zero"),
    ("mic_gradient/writer_pg_loss", float("nan"), "gradient metric is non-finite"),
    ("mic_gradient/writer_active_tokens", 0, "gradient tokens inactive"),
])
def test_unhealthy_actual_gradient_rejected_through_audit(tmp_path, field, value, message):
    args = _audit_fixture(tmp_path)
    _, _, advantage = _real_checkpoint_and_advantage(tmp_path)
    append_jsonl_new(args.ledger, advantage)
    gradient = _valid_gradient_row(); gradient["role_metrics"][field] = value
    append_jsonl_new(args.ledger, gradient)
    with pytest.raises(ValueError, match=message):
        PIPELINE.audit(args)


def test_degenerate_on_policy_signal_rejected_through_audit(tmp_path):
    args = _audit_fixture(tmp_path)
    _, _, advantage = _real_checkpoint_and_advantage(tmp_path, degenerate=True)
    append_jsonl_new(args.ledger, advantage)
    append_jsonl_new(args.ledger, _valid_gradient_row())
    with pytest.raises(ValueError, match="writer innovation is degenerate"):
        PIPELINE.audit(args)


def test_real_calibration_producer_flows_through_real_audit(tmp_path):
    args = _audit_fixture(tmp_path)
    _, _, advantage = _real_checkpoint_and_advantage(tmp_path)
    calibration = advantage["calibration"]
    assert calibration["mae"] >= 0
    append_jsonl_new(args.ledger, advantage)
    append_jsonl_new(args.ledger, _valid_gradient_row())
    _append_valid_weight_sync(args.weight_ledger)
    report = PIPELINE.audit(args)
    assert report["decision"] == "MIC_T5_AUDIT_PASS"
    assert report["on_policy_health"][5]["calibration"]["mae"] == pytest.approx(
        calibration["mae"]
    )


def test_source_firewall_real_entry_passes():
    result = subprocess.run(
        ["python3", str(REPO / "tools/h20/audit_mic_source_firewall.py")],
        cwd=REPO, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MIC_SOURCE_FIREWALL_PASS" in result.stdout


def _p0_fixture(tmp_path, monkeypatch, *, authority="0" * 64):
    original = tmp_path / "original.json"
    curve = tmp_path / "curve.json"
    training_report = tmp_path / "training_report.json"
    manifest = tmp_path / "manifest.json"
    original.write_text("{}")
    curve.write_text(json.dumps({"status": "PASS", "decision": "ORIGINAL_S128_CURVE_PASS"}))
    training_report.write_text(json.dumps({"status": "PASS"}))
    required = [
        "MEMAGENT_MIC_WORK_ROOT", "MEMAGENT_MIC_REPO_DIR",
        "MEMAGENT_MIC_EXPECTED_COMMIT", "MEMAGENT_MIC_GPU_PAIR",
        "MEMAGENT_MIC_RUN_ID", "MEMAGENT_MIC_ORIGINAL_RESOLVED_MANIFEST",
        "MEMAGENT_MIC_ORIGINAL_CURVE_REPORT",
    ]
    manifest.write_text(json.dumps({
        "runtime": {"required_environment": required},
        "original_manifest_equal_paths": [], "only_allowed_scientific_differences": ["algorithm.mic"],
        "certified_read_only_sources": {
            "original_s128_curve": {"final_report": str(curve)},
            "original_t25_training": {
                "resolved": str(original), "final_report": str(training_report),
                "final_report_sha256": sha256_file(training_report),
            },
        },
    }))
    values = {
        "MEMAGENT_MIC_WORK_ROOT": str(tmp_path), "MEMAGENT_MIC_REPO_DIR": str(REPO),
        "MEMAGENT_MIC_EXPECTED_COMMIT": "a" * 40, "MEMAGENT_MIC_GPU_PAIR": "1,3",
        "MEMAGENT_MIC_RUN_ID": "mic-entry-test", "MEMAGENT_MIC_ORIGINAL_RESOLVED_MANIFEST": str(original),
        "MEMAGENT_MIC_ORIGINAL_CURVE_REPORT": str(curve),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return SimpleNamespace(manifest=str(manifest), output=str(tmp_path / "p0.json"), check_runtime=False)


def test_wrong_commit_real_p0_rejection(tmp_path, monkeypatch):
    args = _p0_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(PIPELINE, "git", lambda *parts: "b" * 40 if parts == ("rev-parse", "HEAD") else "")
    with pytest.raises(ValueError, match="exact Git commit mismatch"):
        PIPELINE.p0(args)


def test_dirty_tree_real_p0_rejection(tmp_path, monkeypatch):
    args = _p0_fixture(tmp_path, monkeypatch)
    def fake_git(*parts):
        if parts == ("rev-parse", "HEAD"): return "a" * 40
        if parts == ("branch", "--show-current"): return PIPELINE.BRANCH
        if parts == ("status", "--porcelain"): return " M tampered.py"
        return ""
    monkeypatch.setattr(PIPELINE, "git", fake_git)
    with pytest.raises(ValueError, match="dirty worktree"):
        PIPELINE.p0(args)


def test_fake_baseline_authority_rejected_at_real_evaluation_import(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"files": [], "prediction_files": []}))
    monkeypatch.setenv("MEMAGENT_MIC_BASELINE_INVENTORY", str(inventory))
    monkeypatch.setenv("MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="evaluation inventory authority"):
        PIPELINE.import_baseline(SimpleNamespace(manifest=str(tmp_path / "manifest.json"),
                                                 output=str(tmp_path / "baseline.json")))


def test_gpu_lock_conflict_real_shell_entry_rejection(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    flock = fake_bin / "flock"
    flock.write_text("#!/usr/bin/env bash\nexit 1\n")
    flock.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "MEMAGENT_MIC_WORK_ROOT": str(tmp_path / "work"),
        "MEMAGENT_MIC_REPO_DIR": str(REPO),
        "MEMAGENT_MIC_EXPECTED_COMMIT": "a" * 40,
        "MEMAGENT_MIC_GPU_PAIR": "1,3",
        "MEMAGENT_MIC_RUN_ID": "lock-test",
    })
    result = subprocess.run(
        ["bash", "-c", f"source {REPO / 'scripts/h20/mic_common.sh'}; mic_acquire_gpu_locks"],
        env=env, text=True, capture_output=True,
    )
    assert result.returncode == 75
    assert "MIC_NO_GO: lock conflict" in result.stderr


def test_wrong_commit_and_dirty_tree_guards_are_executable_contracts():
    common = (REPO / "scripts/h20/mic_common.sh").read_text()
    assert 'git rev-parse HEAD' in common and 'MEMAGENT_MIC_EXPECTED_COMMIT' in common
    assert 'git status --porcelain' in common
    assert 'flock -n 8' in common and 'flock -n 9' in common
    assert 'nvidia-smi -i "$MEMAGENT_MIC_GPU_PAIR"' in common


def test_seed_weight_sync_and_resume_tamper_guards_are_connected():
    trainer = (REPO / "verl/trainer/ppo/ray_trainer.py").read_text()
    continuation = (REPO / "scripts/h20/continue_qwen25_7b_mic.sh").read_text()
    assert "trajectory_seed_mode" in trainer and "trajectory_base_seeds" in trainer
    assert "_audit_gate_a_weight_sync" in trainer
    assert "critic resume checkpoint is missing" in trainer
    assert "critic history trajectory collision" in trainer
    assert "5:10|10:15|15:20|20:25" in continuation


def _identity_rows():
    sources, identities, generations = [], [], []
    for index in range(128):
        question, context = f"q-{index}", f"c-{index}"
        identity = {
            "stable_key": f"key-{index}", "source_order_index": index,
            "source_question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "source_context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        }
        sources.append({"question": question, "context": context, "ground_truth": f"a-{index}"})
        identities.append(identity)
        generations.append({**identity, "output": f"prediction-{index}"})
    return sources, identities, generations


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_permuted_generation_real_prepare_eval_exact_joins(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    sources, identities, generations = _identity_rows()
    generations.reverse()
    identity_path, generation_path = tmp_path / "identity.jsonl", tmp_path / "generation.jsonl"
    _write_jsonl(identity_path, identities); _write_jsonl(generation_path, generations)
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame(sources))
    report = PIPELINE.prepare_eval(SimpleNamespace(
        generations=str(generation_path), identity_source=str(identity_path),
        validation=str(tmp_path / "validation.parquet"), output=str(tmp_path / "predictions.jsonl"),
    ))
    assert report["decision"] == "MIC_S128_PREPARE_PASS"
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert rows[0]["stable_key"] == "key-0" and rows[0]["output"] == "prediction-0"


def test_generation_identity_tamper_real_prepare_eval_rejection(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    sources, identities, generations = _identity_rows()
    generations[0]["source_question_sha256"] = "f" * 64
    identity_path, generation_path = tmp_path / "identity.jsonl", tmp_path / "generation.jsonl"
    _write_jsonl(identity_path, identities); _write_jsonl(generation_path, generations)
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame(sources))
    with pytest.raises(ValueError, match="generation identity binding mismatch"):
        PIPELINE.prepare_eval(SimpleNamespace(
            generations=str(generation_path), identity_source=str(identity_path),
            validation=str(tmp_path / "validation.parquet"), output=str(tmp_path / "predictions.jsonl"),
        ))
