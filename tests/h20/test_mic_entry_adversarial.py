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
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_fixed_s128
from recurrent.research.stable_eval_identity import canonical_sha256, stable_key

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


def test_mic_ray_task_environment_is_explicit_and_fail_closed(monkeypatch):
    from verl.trainer.main_ppo import _task_runner_runtime_env

    required = {
        "MEMAGENT_MIC_WORK_ROOT": "/frozen/work",
        "MEMAGENT_MIC_REPO_DIR": "/frozen/repo",
        "MEMAGENT_MIC_EXPECTED_COMMIT": "a" * 40,
        "MEMAGENT_MIC_RUN_ID": "mic-eval",
    }
    monkeypatch.setenv("MEMAGENT_MIC_REQUIRED", "1")
    for key, value in required.items():
        monkeypatch.setenv(key, value)
    runtime_env = _task_runner_runtime_env()["env_vars"]
    assert runtime_env["MEMAGENT_MIC_REQUIRED"] == "1"
    for key, value in required.items():
        assert runtime_env[key] == value

    monkeypatch.delenv("MEMAGENT_MIC_REPO_DIR")
    with pytest.raises(RuntimeError, match="Ray task environment missing.*REPO_DIR"):
        _task_runner_runtime_env()


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


def test_real_eval_launcher_emits_actor_only_overrides_without_training_ledgers(tmp_path):
    model, train, validation = tmp_path / "model", tmp_path / "train.parquet", tmp_path / "val.parquet"
    model.mkdir(); (model / "config.json").write_text("{}")
    train.write_bytes(b"train"); validation.write_bytes(b"validation")
    experiment = "mic-eval-entry"
    checkpoint = tmp_path / "logs/memory_agent" / experiment / "global_step_5"
    (checkpoint / "actor").mkdir(parents=True); (checkpoint / "data.pt").write_bytes(b"data")
    identity, summary, generation = (tmp_path / name for name in (
        "identity.jsonl", "summary.json", "generation.jsonl"
    ))
    identity.write_text("{}\n")
    env = os.environ.copy()
    env.update({
        "WORK_ROOT": str(tmp_path), "CODE": str(REPO), "PYTHON": os.sys.executable,
        "MODEL": str(model), "TRAIN": str(train), "VAL": str(validation),
        "PHASE": "resume", "EXP": experiment, "RESUME_SOURCE_STEP": "5",
        "RESUME_TOTAL_STEPS": "6", "N_GPUS": "2", "FSDP_SIZE": "2",
        "CUDA_VISIBLE_DEVICES": "6,7", "MEMAGENT_MIC_REQUIRED": "1",
        "MEMAGENT_MIC_ENABLE": "1", "RUN_SEED": "2026", "EMIT_TRAINER_OVERRIDES": "1",
        "MEMAGENT_MIC_EVAL_STEP": "5", "MEMAGENT_MIC_EVAL_DIR": str(tmp_path / "raw"),
        "MEMAGENT_MIC_EVAL_IDENTITY_PATH": str(identity),
        "MEMAGENT_MIC_EVAL_IDENTITY_SHA256": hashlib.sha256(identity.read_bytes()).hexdigest(),
        "MEMAGENT_MIC_EVAL_SUMMARY_PATH": str(summary),
            "MEMAGENT_MIC_EVAL_GENERATION_PATH": str(generation),
            "MEMAGENT_MIC_EVAL_TRAINING_AUDIT_SHA256": "a" * 64,
            "MEMAGENT_MIC_EVAL_ORIGINAL_PROTOCOL_SHA256": "b" * 64,
            "MEMAGENT_MIC_EVAL_ORIGINAL_REWARD_CODE_SHA256": "c" * 64,
    })
    result = subprocess.run(
        ["bash", str(REPO / "experiments/7b_gate_a/run_gate_a.sh")], env=env,
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    overrides = json.loads(result.stdout.splitlines()[-1])
    assert "algorithm.mic.enabled=false" in overrides
    assert "trainer.resume_mode=mic_actor_only_eval" in overrides
    assert not any("critic_checkpoint_root" in value or "ledger_path" in value
                   for value in overrides)


def test_eval_all_parent_exports_baseline_authority_to_final_entry(tmp_path):
    script_dir = tmp_path / "scripts" / "h20"
    script_dir.mkdir(parents=True)
    eval_all = script_dir / "eval_all_qwen25_7b_mic.sh"
    eval_all.write_text((REPO / "scripts/h20/eval_all_qwen25_7b_mic.sh").read_text())
    eval_all.chmod(0o755)
    (script_dir / "eval_audit_qwen25_7b_mic.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )
    p0 = tmp_path / "p0.json"
    p0.write_text(json.dumps({"original_curve_report_sha256": "a" * 64}))
    capture = tmp_path / "captured.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${1:-} == -c ]]; then\n"
        "  if [[ $2 == *inventory_path* ]]; then echo "
        + str(tmp_path / "baseline_inventory.json") + "; else echo "
        + "a" * 64 + "; fi\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n%s\\n' \"${MEMAGENT_MIC_BASELINE_INVENTORY:-}\" "
        "\"${MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256:-}\" >\"$CAPTURE\"\n"
    )
    fake_python.chmod(0o755)
    (script_dir / "mic_common.sh").write_text("\n".join([
        f"MIC_PYTHON={fake_python}", f"MIC_P0={p0}",
        f"MIC_BASELINE={tmp_path / 'baseline.json'}", f"MIC_CERT={tmp_path / 'cert'}",
        f"MIC_ROOT={tmp_path / 'root'}", f"MIC_OUTPUT={tmp_path / 'output'}",
        f"MIC_CHECKPOINT_AUTHORITY={tmp_path / 'authority.json'}",
        f"MIC_CHECKPOINT_AUTHORITY_CERT={tmp_path / 'authority_cert.json'}",
        f"MIC_WEIGHT_LEDGER={tmp_path / 'weights.jsonl'}",
        f"MIC_LEDGER={tmp_path / 'mic.jsonl'}", f"MIC_E0={tmp_path / 'e0.json'}",
        f"MIC_PAPER_REVIEW={tmp_path / 'paper.json'}",
        "mic_require_checkout() { :; }", "mic_require_training_gates() { :; }", "",
    ]))
    result = subprocess.run(
        ["bash", str(eval_all)], env={**os.environ, "CAPTURE": str(capture),
                                      "MEMAGENT_MIC_REPO_DIR": str(tmp_path)},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == [
        str(tmp_path / "baseline_inventory.json"), "a" * 64,
    ]


def test_partial_anchor_attempts_are_preserved_and_retry_allocates_new_directory(tmp_path):
    env = {**os.environ,
           "MEMAGENT_MIC_WORK_ROOT": str(tmp_path / "work"),
           "MEMAGENT_MIC_REPO_DIR": str(REPO),
           "MEMAGENT_MIC_EXPECTED_COMMIT": "a" * 40,
           "MEMAGENT_MIC_GPU_PAIR": "1,3", "MEMAGENT_MIC_RUN_ID": "retry-test"}
    command = f'''source "{REPO / 'scripts/h20/mic_common.sh'}"
mkdir -p "$MIC_ROOT/eval_t5_attempts/attempt_0001"
mkdir -p "$MIC_ROOT/eval_t10_attempts/attempt_0001"
first=$(mic_next_eval_attempt 5)
second=$(mic_next_eval_attempt 10)
test -d "$MIC_ROOT/eval_t5_attempts/attempt_0001"
test -d "$MIC_ROOT/eval_t10_attempts/attempt_0001"
printf '%s\n%s\n' "$first" "$second"
'''
    result = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(tmp_path / "work/logs/mic_frozen_20260822/retry-test/"
            "eval_t5_attempts/attempt_0002"),
        str(tmp_path / "work/logs/mic_frozen_20260822/retry-test/"
            "eval_t10_attempts/attempt_0002"),
    ]


def test_partial_baseline_materialization_is_preserved_and_retry_allocates_new_directory(
        tmp_path):
    env = {**os.environ,
           "MEMAGENT_MIC_WORK_ROOT": str(tmp_path / "work"),
           "MEMAGENT_MIC_REPO_DIR": str(REPO),
           "MEMAGENT_MIC_EXPECTED_COMMIT": "a" * 40,
           "MEMAGENT_MIC_GPU_PAIR": "1,3", "MEMAGENT_MIC_RUN_ID": "baseline-retry"}
    command = f'''source "{REPO / 'scripts/h20/mic_common.sh'}"
mkdir -p "$MIC_BASELINE_ATTEMPTS/attempt_0001/rows"
retry=$(mic_next_baseline_attempt)
test -d "$MIC_BASELINE_ATTEMPTS/attempt_0001/rows"
printf '%s\n' "$retry"
'''
    result = subprocess.run(["bash", "-c", command], env=env, text=True,
                            capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(
        tmp_path / "work/logs/mic_frozen_20260822/baseline-retry/"
        "baseline_materialization_attempts/attempt_0002"
    )


def test_completed_anchor_is_reauthenticated_before_skip(tmp_path):
    script_dir = tmp_path / "scripts" / "h20"
    script_dir.mkdir(parents=True)
    entry = script_dir / "eval_audit_qwen25_7b_mic.sh"
    entry.write_text((REPO / "scripts/h20/eval_audit_qwen25_7b_mic.sh").read_text())
    entry.chmod(0o755)
    cert = tmp_path / "cert"; cert.mkdir()
    root = tmp_path / "run"; attempt = root / "eval_t5_attempts/attempt_0001"
    attempt.mkdir(parents=True)
    (cert / "t5_eval.json").write_text(json.dumps({"evaluation_root": str(attempt)}))
    (cert / "t5_audit.json").write_text("{}")
    output = tmp_path / "output"; (output / "global_step_5/actor").mkdir(parents=True)
    baseline_inventory = tmp_path / "baseline_inventory.json"; baseline_inventory.write_text("{}")
    baseline = tmp_path / "baseline.json"; baseline.write_text("{}")
    checkpoint_cert = tmp_path / "checkpoint_cert.json"; checkpoint_cert.write_text("{}")
    p0 = tmp_path / "p0.json"; p0.write_text(json.dumps({
        "original_curve_report_sha256": "a" * 64,
    }))
    identity = tmp_path / "identity.jsonl"; identity.write_text("{}\n")
    capture = tmp_path / "capture.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(f'''#!/usr/bin/env bash
if [[ ${{1:-}} == -c ]]; then
  case "$2" in
    *original_curve_report_sha256*) echo {"a" * 64} ;;
    *'name="Original"'*) echo {identity} ;;
    *evaluation_root*) echo {attempt} ;;
  esac
  exit 0
fi
if [[ "$*" == *prepare-eval* ]]; then
  echo prepare-verify-called >"$CAPTURE"
  exit 42
fi
exit 0
''')
    fake_python.chmod(0o755)
    common = script_dir / "mic_common.sh"
    common.write_text("\n".join([
        f"MIC_PYTHON={fake_python}", f"MIC_P0={p0}", f"MIC_CERT={cert}",
        f"MIC_ROOT={root}", f"MIC_BASELINE_INVENTORY={baseline_inventory}",
        f"MIC_BASELINE={baseline}", f"MIC_MANIFEST={tmp_path / 'manifest.json'}",
        f"MIC_OUTPUT={output}", f"MIC_CHECKPOINT_AUTHORITY_CERT={checkpoint_cert}",
        f"MIC_CHECKPOINT_AUTHORITY={tmp_path / 'authority.json'}",
        f"MIC_LEDGER={tmp_path / 'mic.jsonl'}", f"MIC_WEIGHT_LEDGER={tmp_path / 'weight.jsonl'}",
        f"MIC_CURVE_RESOLVED={tmp_path / 'resolved.json'}",
        f"MIC_CURVE_AUTHORITY={tmp_path / 'curve_authority.json'}",
        "mic_require_checkout() { :; }", "mic_require_training_gates() { :; }",
        "mic_require_gate() { :; }",
        "mic_acquire_gpu_locks() { echo unexpected-gpu-lock >&2; exit 99; }",
        "mic_require_idle() { :; }", "mic_next_eval_attempt() { exit 98; }",
        "mic_export_evaluation() { :; }", "",
    ]))
    env = {**os.environ, "CAPTURE": str(capture),
           "MEMAGENT_MIC_WORK_ROOT": str(tmp_path),
           "MEMAGENT_MIC_REPO_DIR": str(tmp_path),
           "MEMAGENT_MIC_ORIGINAL_CURVE_REPORT": str(tmp_path / "curve.json")}
    result = subprocess.run(["bash", str(entry), "5"], env=env, text=True,
                            capture_output=True)
    assert result.returncode == 42
    assert capture.read_text().strip() == "prepare-verify-called"
    assert "unexpected-gpu-lock" not in result.stderr


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


def test_fresh_release_requires_training_time_full_checkpoint_inventory(tmp_path):
    args = _audit_fixture(tmp_path)
    p0 = json.loads(Path(args.p0).read_text())
    p0["requires_training_checkpoint_inventory"] = True
    Path(args.p0).write_text(json.dumps(p0))
    _, _, advantage = _real_checkpoint_and_advantage(tmp_path)
    append_jsonl_new(args.ledger, advantage)
    append_jsonl_new(args.ledger, _valid_gradient_row())
    _append_valid_weight_sync(args.weight_ledger)
    with pytest.raises(ValueError, match="required training-time actor checkpoint inventory absent"):
        PIPELINE.audit(args)


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
    inventory.write_text(json.dumps({"source_curve_report": str(tmp_path / "curve.json"),
                                     "source_curve_report_sha256": "e" * 64,
                                     "files": [], "prediction_files": []}))
    p0 = tmp_path / "p0.json"
    p0.write_text(json.dumps({"status": "PASS", "decision": "MIC_P0_PASS",
                              "original_curve_report": str(tmp_path / "curve.json"),
                              "original_curve_report_sha256": "e" * 64}))
    monkeypatch.setenv("MEMAGENT_MIC_BASELINE_INVENTORY", str(inventory))
    monkeypatch.setenv("MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="certified curve trust root"):
        PIPELINE.import_baseline(SimpleNamespace(manifest=str(tmp_path / "manifest.json"),
                                                 p0=str(p0), output=str(tmp_path / "baseline.json")))


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
            "raw_row_position": index,
            "source_question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "source_context_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "ground_truth": f"a-{index}",
        }
        sources.append({"question": question, "context": context,
                        "reward_model": {"ground_truth": f"a-{index}"}})
        identities.append(identity)
        generations.append({**identity, "output": f"prediction-{index}"})
    return sources, identities, generations


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _prepare_eval_args(tmp_path, identity_path, generation_path):
    validation = tmp_path / "validation.parquet"
    validation.write_bytes(b"frozen-validation")
    p0 = tmp_path / "p0.json"
    p0.write_text(json.dumps({
        "status": "PASS", "decision": "MIC_P0_PASS",
        "git_commit": "a" * 40, "run_id": "audit-run",
    }))
    reconstructed_reward = tmp_path / "original/recurrent/research/hotpotqa_dense_reward.py"
    reconstructed_reward.parent.mkdir(parents=True)
    reconstructed_reward.write_bytes(b"certified-reward-code")
    checkpoint = tmp_path / "global_step_5"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    inventory = []
    for rank in (0, 1):
        shard = actor / f"model_world_size_2_rank_{rank}.pt"
        shard.write_bytes(f"rank-{rank}".encode())
        inventory.append({"path": f"actor/{shard.name}", "size": shard.stat().st_size,
                          "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()})
    training_audit = tmp_path / "t5_audit.json"
    training_audit.write_text(json.dumps({
        "status": "PASS", "decision": "MIC_T5_AUDIT_PASS", "mic_steps": [5],
        "gate_sha256": {"p0": hashlib.sha256(p0.read_bytes()).hexdigest()},
    }))
    digest = "d" * 64
    worker_evidence = [{"rank": rank} for rank in (0, 1)]
    before = {
        "actor_master_sampled_tensor_digest": "e" * 64,
        "actor_rollout_sampled_tensor_digest": digest,
        "vllm_sampled_tensor_digest": digest, "vllm_pre_sync_sampled_tensor_digest": None,
        "worker_ranks": [0, 1], "worker_evidence": worker_evidence,
    }
    after = {**before, "vllm_pre_sync_sampled_tensor_digest": digest}
    summary = tmp_path / "execution_summary.json"
    summary.write_text(json.dumps({
        "schema": "memagent.mic.eval.v1",
        "record_type": "mic_read_only_execution_summary", "global_step": 5,
        "run_id": "audit-run",
        "evaluation_git_commit": "a" * 40,
        "training_audit_sha256": hashlib.sha256(training_audit.read_bytes()).hexdigest(),
        "identity_path": str(identity_path.resolve()),
        "identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
        "generation_path": str(generation_path.resolve()),
        "generation_sha256": hashlib.sha256(generation_path.read_bytes()).hexdigest(),
        "generation_protocol_evidence": {
            "method_generation_protocol_sha256": "b" * 64,
            "original_generation_protocol_sha256": "a" * 64,
            "original_protocol_reconstruction_path": str(reconstructed_reward.resolve()),
            "reward_code_sha256": hashlib.sha256(reconstructed_reward.read_bytes()).hexdigest(),
        },
        "checkpoint_source": str(checkpoint.resolve()), "checkpoint_inventory": inventory,
        "actor_checkpoint_load_acks": [
            {"rank": rank, "optimizer_loaded": False, "lr_scheduler_loaded": False,
             "rng_loaded": False, "dataloader_loaded": False} for rank in (0, 1)
        ],
        "actor_update_calls": 0, "validation_only": True,
        "weight_snapshot_before": before, "weight_snapshot_after": after,
    }))
    return SimpleNamespace(
        generations=str(generation_path), identity_source=str(identity_path),
        validation=str(validation), baseline=str(tmp_path / "baseline.json"),
        p0=str(p0),
        execution_summary=str(summary), training_audit=str(training_audit),
        checkpoint=str(checkpoint), step=5, output=str(tmp_path / "predictions.jsonl"),
        report=str(tmp_path / "prepare.json"), verify_existing=False,
        checkpoint_authority=str(tmp_path / "checkpoint_authority.json"),
        checkpoint_authority_certificate=str(tmp_path / "checkpoint_authority_cert.json"),
        output_root=str(tmp_path), weight_ledger=str(tmp_path / "weights.jsonl"),
        mic_ledger=str(tmp_path / "mic.jsonl"),
    )


def _stub_prepare_baseline_authority(monkeypatch, args, identity_path):
    monkeypatch.setattr(PIPELINE, "import_baseline", lambda _: {
        "interfaces": {"Original5": {"path": str(identity_path.resolve())}},
        "validation_path": str(Path(args.validation).resolve()),
        "validation_sha256": hashlib.sha256(Path(args.validation).read_bytes()).hexdigest(),
        "shared_generation_protocol_sha256": "a" * 64,
        "original_reward_code_sha256": hashlib.sha256(
            (Path(args.execution_summary).parent /
             "original/recurrent/research/hotpotqa_dense_reward.py").read_bytes()
        ).hexdigest(),
    })
    monkeypatch.setattr(PIPELINE, "verify_checkpoint_authority", lambda *unused: {
        "checkpoint_authority_sha256": "c" * 64,
    })
    monkeypatch.setattr(PIPELINE, "_verify_training_weight_prefix", lambda *unused: "w" * 64)


def test_permuted_generation_real_prepare_eval_exact_joins(tmp_path, monkeypatch):
    sources, identities, generations = _identity_rows()
    generations.reverse()
    identity_path, generation_path = tmp_path / "identity.jsonl", tmp_path / "generation.jsonl"
    _write_jsonl(identity_path, identities); _write_jsonl(generation_path, generations)
    monkeypatch.setattr(PIPELINE, "_load_parquet_rows", lambda _: sources)
    monkeypatch.setattr(PIPELINE, "git", lambda *args: "a" * 40)
    monkeypatch.setenv("MEMAGENT_MIC_RUN_ID", "audit-run")
    args = _prepare_eval_args(tmp_path, identity_path, generation_path)
    _stub_prepare_baseline_authority(monkeypatch, args, identity_path)
    report = PIPELINE.prepare_eval(args)
    assert report["decision"] == "MIC_S128_PREPARE_PASS"
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert rows[0]["stable_key"] == "key-0" and rows[0]["output"] == "prediction-0"


def test_parquet_ground_truth_normalizes_numpy_arrays_to_certified_json_lists():
    np = pytest.importorskip("numpy")
    source = {"reward_model": {"ground_truth": np.asarray(["alpha", "beta"])}}
    ground_truth = PIPELINE._parquet_ground_truth(source, row=0)
    assert ground_truth == ["alpha", "beta"]
    assert canonical_sha256(ground_truth) == canonical_sha256(["alpha", "beta"])


def test_generation_identity_tamper_real_prepare_eval_rejection(tmp_path, monkeypatch):
    sources, identities, generations = _identity_rows()
    generations[0]["source_question_sha256"] = "f" * 64
    identity_path, generation_path = tmp_path / "identity.jsonl", tmp_path / "generation.jsonl"
    _write_jsonl(identity_path, identities); _write_jsonl(generation_path, generations)
    monkeypatch.setattr(PIPELINE, "_load_parquet_rows", lambda _: sources)
    monkeypatch.setattr(PIPELINE, "git", lambda *args: "a" * 40)
    monkeypatch.setenv("MEMAGENT_MIC_RUN_ID", "audit-run")
    args = _prepare_eval_args(tmp_path, identity_path, generation_path)
    _stub_prepare_baseline_authority(monkeypatch, args, identity_path)
    with pytest.raises(ValueError, match="generation identity binding mismatch"):
        PIPELINE.prepare_eval(args)


def test_validation_ground_truth_tamper_real_prepare_eval_rejection(tmp_path, monkeypatch):
    sources, identities, generations = _identity_rows()
    sources[0]["reward_model"]["ground_truth"] = "tampered"
    identity_path, generation_path = tmp_path / "identity.jsonl", tmp_path / "generation.jsonl"
    _write_jsonl(identity_path, identities); _write_jsonl(generation_path, generations)
    monkeypatch.setattr(PIPELINE, "_load_parquet_rows", lambda _: sources)
    monkeypatch.setattr(PIPELINE, "git", lambda *args: "a" * 40)
    monkeypatch.setenv("MEMAGENT_MIC_RUN_ID", "audit-run")
    args = _prepare_eval_args(tmp_path, identity_path, generation_path)
    _stub_prepare_baseline_authority(monkeypatch, args, identity_path)
    with pytest.raises(ValueError, match="frozen ground truth mismatch"):
        PIPELINE.prepare_eval(args)


def test_original_method_generation_protocol_mismatch_is_rejected(tmp_path, monkeypatch):
    sources, identities, generations = _identity_rows()
    identity_path, generation_path = tmp_path / "identity.jsonl", tmp_path / "generation.jsonl"
    _write_jsonl(identity_path, identities); _write_jsonl(generation_path, generations)
    monkeypatch.setattr(PIPELINE, "_load_parquet_rows", lambda _: sources)
    monkeypatch.setattr(PIPELINE, "git", lambda *args: "a" * 40)
    monkeypatch.setenv("MEMAGENT_MIC_RUN_ID", "audit-run")
    args = _prepare_eval_args(tmp_path, identity_path, generation_path)
    _stub_prepare_baseline_authority(monkeypatch, args, identity_path)
    summary = json.loads(Path(args.execution_summary).read_text())
    summary["generation_protocol_evidence"]["original_generation_protocol_sha256"] = "f" * 64
    Path(args.execution_summary).write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="read-only evaluation summary binding mismatch"):
        PIPELINE.prepare_eval(args)


def test_checkpoint_drift_real_prepare_eval_rejection(tmp_path, monkeypatch):
    sources, identities, generations = _identity_rows()
    identity_path, generation_path = tmp_path / "identity.jsonl", tmp_path / "generation.jsonl"
    _write_jsonl(identity_path, identities); _write_jsonl(generation_path, generations)
    monkeypatch.setattr(PIPELINE, "_load_parquet_rows", lambda _: sources)
    monkeypatch.setattr(PIPELINE, "git", lambda *args: "a" * 40)
    monkeypatch.setenv("MEMAGENT_MIC_RUN_ID", "audit-run")
    args = _prepare_eval_args(tmp_path, identity_path, generation_path)
    _stub_prepare_baseline_authority(monkeypatch, args, identity_path)
    (Path(args.checkpoint) / "actor/model_world_size_2_rank_0.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint inventory changed"):
        PIPELINE.prepare_eval(args)


def test_release_pinned_checkpoint_authority_rejects_pre_eval_shard_replacement(tmp_path):
    output_root = tmp_path / "run"
    for step in (5, 10, 15, 20, 25):
        actor = output_root / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        for rank in (0, 1):
            (actor / f"model_world_size_2_rank_{rank}.pt").write_bytes(
                f"step={step},rank={rank}".encode()
            )
    records = PIPELINE._actor_model_inventory(output_root, (5, 10, 15, 20, 25))
    authority = tmp_path / "authority.json"
    p0 = tmp_path / "checkpoint_p0.json"
    p0.write_text(json.dumps({"git_commit": "a" * 40, "run_id": "audit-run"}))
    authority.write_text(json.dumps({
        "schema": "memagent.mic.v1",
        "authority_kind": "out_of_band_operator_sha256_pinned",
        "training_git_commit": "a" * 40, "run_id": "audit-run",
        "checkpoint_authority_sha256": PIPELINE.sha256_json(records),
    }))
    certificate = tmp_path / "certificate.json"
    PIPELINE.materialize_checkpoint_authority(SimpleNamespace(
        authority=str(authority), p0=str(p0), output_root=str(output_root),
        ledger=str(tmp_path / "unused-ledger.jsonl"), output=str(certificate),
    ))
    PIPELINE.verify_checkpoint_authority(
        authority, certificate, output_root, 5, p0, tmp_path / "unused-ledger.jsonl"
    )
    (output_root / "global_step_5/actor/model_world_size_2_rank_0.pt").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="differs from release authority"):
        PIPELINE.verify_checkpoint_authority(
            authority, certificate, output_root, 5, p0,
            tmp_path / "unused-ledger.jsonl",
        )


def test_fresh_checkpoint_authority_replays_training_time_inventory(tmp_path):
    output_root = tmp_path / "fresh-run"
    ledger = tmp_path / "mic.jsonl"
    for step in (5, 10, 15, 20, 25):
        actor = output_root / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        shards = []
        for rank in (0, 1):
            shard = actor / f"model_world_size_2_rank_{rank}.pt"
            shard.write_bytes(f"fresh-step={step},rank={rank}".encode())
            shards.append({
                "path": f"actor/{shard.name}", "size": shard.stat().st_size,
                "sha256": sha256_file(shard),
            })
        append_jsonl_new(ledger, {
            "record_type": "mic_actor_checkpoint_inventory", "global_step": step,
            "checkpoint_path": str((output_root / f"global_step_{step}").resolve()),
            "model_shards": shards, "model_shards_sha256": PIPELINE.sha256_json(shards),
            "git_commit": "a" * 40, "run_id": "fresh-audit-run",
        })
    p0 = tmp_path / "p0.json"
    p0.write_text(json.dumps({
        "git_commit": "a" * 40, "run_id": "fresh-audit-run",
        "requires_training_checkpoint_inventory": True,
    }))
    certificate = tmp_path / "certificate.json"
    args = SimpleNamespace(
        authority=str(tmp_path / "not-used.json"), p0=str(p0),
        output_root=str(output_root), ledger=str(ledger), output=str(certificate),
    )
    report = PIPELINE.materialize_checkpoint_authority(args)
    assert report["authority_kind"] == "training_ledger_checkpoint_inventory"
    PIPELINE.verify_checkpoint_authority(
        args.authority, certificate, output_root, 25, p0, ledger,
    )
    (output_root / "global_step_25/actor/model_world_size_2_rank_1.pt").write_bytes(
        b"replaced-after-training"
    )
    with pytest.raises(ValueError, match="differ from training-time inventory"):
        PIPELINE.verify_checkpoint_authority(
            args.authority, certificate, output_root, 25, p0, ledger,
        )


def test_training_weight_prefix_rejects_cross_run_replay(tmp_path):
    ledger = tmp_path / "weights.jsonl"
    _append_valid_weight_sync(ledger)
    audit_report = {"weight_sync_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest()}
    snapshot = {
        "actor_master_sampled_tensor_digest": "e" * 64,
        "actor_rollout_sampled_tensor_digest": "d" * 64,
        "vllm_sampled_tensor_digest": "d" * 64,
    }
    with pytest.raises(ValueError, match="cross-run replay"):
        PIPELINE._verify_training_weight_prefix(
            ledger, audit_report, 5, snapshot, "a" * 40, "different-run",
        )


def test_evaluation_entry_is_read_only_and_attested():
    common = (REPO / "scripts/h20/mic_common.sh").read_text()
    entry = (REPO / "scripts/h20/eval_audit_qwen25_7b_mic.sh").read_text()
    trainer = (REPO / "verl/trainer/ppo/ray_trainer.py").read_text()
    assert "mic_export_evaluation" in entry and "mic_export_training" not in entry
    assert "GATE_A_FROZEN_AUDIT=0" in common
    assert "GATE_A_EXECUTION_LEDGER" in common
    assert "unset MEMAGENT_MIC_LEDGER_PATH MEMAGENT_MIC_CRITIC_ROOT" in common
    assert "mic_read_only_execution_summary" in trainer
    assert "actor checkpoint changed during evaluation" in trainer
    assert 'sync_kind="stable_eval_before"' in trainer
    assert 'sync_kind="stable_eval_after"' in trainer


def _baseline_materialization_fixture(tmp_path, monkeypatch):
    validation = tmp_path / "validation.parquet"
    validation.write_bytes(b"frozen-s128")
    sources, frozen_rows = [], []
    for index in range(128):
        ground_truth = f"answer-{index}"
        sources.append({"reward_model": {"ground_truth": ground_truth}})
        frozen_rows.append({
            "example_id": str(index), "semantic_dataset_index": index,
            "source_order_index": index, "raw_row_position": index,
            "production_effective_position": index, "context_token_count": 10,
            "source_question_hash": hashlib.sha256(f"q-{index}".encode()).hexdigest(),
            "source_context_hash": hashlib.sha256(f"c-{index}".encode()).hexdigest(),
            "ground_truth_hash": canonical_sha256(ground_truth),
        })
    monkeypatch.setattr(PIPELINE, "_load_parquet_rows", lambda _: sources)
    identity_payload = {
        "source_dataset": {"parquet_sha256": hashlib.sha256(validation.read_bytes()).hexdigest()},
        "rows": frozen_rows,
    }
    resolved = tmp_path / "curve_resolved.json"
    resolved.write_text(json.dumps({
        "identity_payload": identity_payload,
        "eval_manifest_hash": canonical_sha256(identity_payload),
        "execution_binding": {"trainer_configuration": {
            "shared_generation_protocol_sha256": "a" * 64,
        }, "execution_code_sha256": {
            "recurrent/research/hotpotqa_dense_reward.py": "b" * 64,
        }},
    }))
    interfaces = ("I", "Original5", "Original10", "Original15", "Original20", "Original25")
    evidence = {}
    search_root = tmp_path / "logs"
    for interface_index, interface in enumerate(interfaces):
        terminal_rows, metric_rows = [], []
        for index, frozen in enumerate(frozen_rows):
            terminal = {
                **frozen, "eval_manifest_hash": canonical_sha256(identity_payload),
                "replica_id": 0, "trajectory_seed": index + 100,
                "trajectory_id": f"trajectory-{index}",
                "output": f"answer-{index}" if interface_index % 2 == 0 else "wrong",
            }
            terminal_rows.append(terminal)
            scored = score_terminal_output(terminal["output"], f"answer-{index}")
            metric_rows.append({
                "stable_key": json.dumps(stable_key(terminal), separators=(",", ":")),
                "source_order_index": index, "eval_manifest_hash": terminal["eval_manifest_hash"],
                "example_id": terminal["example_id"], "replica_id": 0,
                "trajectory_seed": terminal["trajectory_seed"],
                "trajectory_id": terminal["trajectory_id"], **scored,
            })
        terminal_path = search_root / interface / "terminal" / f"{interface_index}.jsonl"
        terminal_path.parent.mkdir(parents=True)
        _write_jsonl(terminal_path, terminal_rows)
        evidence[interface] = {
            "artifacts": {f"terminal/{interface_index}.jsonl": {
                "sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest()
            }},
            "independent_metric_rows_sha256": canonical_sha256(metric_rows),
            "metrics": summarize_fixed_s128(metric_rows),
        }
    curve = tmp_path / "curve.json"
    curve.write_text(json.dumps({
        "status": "PASS", "decision": "ORIGINAL_S128_CURVE_PASS",
        "evidence": {
            "interfaces": evidence,
            "resolved_manifest_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        },
    }))
    p0 = tmp_path / "p0.json"
    p0.write_text(json.dumps({
        "status": "PASS", "decision": "MIC_P0_PASS",
        "original_curve_report": str(curve.resolve()),
        "original_curve_report_sha256": hashlib.sha256(curve.read_bytes()).hexdigest(),
    }))
    curve_authority = tmp_path / "curve_authority.json"
    curve_authority.write_text(json.dumps({
        "schema": "memagent.mic.v1",
        "authority_kind": "out_of_band_original_curve_sha256_pinned",
        "curve_report_path": str(curve.resolve()),
        "curve_report_sha256": hashlib.sha256(curve.read_bytes()).hexdigest(),
        "curve_resolved_path": str(resolved.resolve()),
        "curve_resolved_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }))
    return SimpleNamespace(
        p0=str(p0), curve_report=str(curve), curve_resolved=str(resolved),
        curve_authority=str(curve_authority),
        search_root=str(search_root), validation=str(validation),
        output_root=str(tmp_path / "materialized" / "rows"),
        output=str(tmp_path / "materialized" / "inventory.json"),
    )


def test_certified_baseline_materializes_and_recomputes_metric_rows(tmp_path, monkeypatch):
    args = _baseline_materialization_fixture(tmp_path, monkeypatch)
    report = PIPELINE.materialize_baseline(args)
    assert report["decision"] == "MIC_BASELINE_MATERIALIZE_PASS"
    inventory = json.loads(Path(args.output).read_text())
    assert {row["interface"] for row in inventory["prediction_files"]} == {
        "I", "Original5", "Original10", "Original15", "Original20", "Original25"
    }
    assert all(Path(row["path"]).is_file() for row in inventory["prediction_files"])
    p0 = json.loads(Path(args.p0).read_text())
    monkeypatch.setenv("MEMAGENT_MIC_BASELINE_INVENTORY", args.output)
    monkeypatch.setenv(
        "MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256", p0["original_curve_report_sha256"]
    )
    imported = PIPELINE.import_baseline(SimpleNamespace(
        p0=args.p0, manifest=str(tmp_path / "unused.json"),
        output=str(tmp_path / "baseline_import.json"),
        curve_authority=args.curve_authority,
    ))
    assert imported["decision"] == "MIC_BASELINE_IMPORT_PASS"


def test_existing_self_consistent_fake_baseline_is_reauthenticated_against_curve(tmp_path, monkeypatch):
    args = _baseline_materialization_fixture(tmp_path, monkeypatch)
    PIPELINE.materialize_baseline(args)
    p0 = json.loads(Path(args.p0).read_text())
    monkeypatch.setenv("MEMAGENT_MIC_BASELINE_INVENTORY", args.output)
    monkeypatch.setenv(
        "MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256", p0["original_curve_report_sha256"]
    )
    baseline_path = tmp_path / "baseline_import.json"
    PIPELINE.import_baseline(SimpleNamespace(
        p0=args.p0, manifest=str(tmp_path / "unused.json"),
        output=str(baseline_path), verify_existing=False,
        curve_authority=args.curve_authority,
    ))
    inventory = json.loads(Path(args.output).read_text())
    item = next(row for row in inventory["prediction_files"] if row["interface"] == "Original25")
    normalized = Path(item["path"])
    rows = [json.loads(line) for line in normalized.read_text().splitlines()]
    rows[0]["output"] = "forged-perfect-answer"
    _write_jsonl(normalized, rows)
    item["sha256"] = hashlib.sha256(normalized.read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(inventory))
    forged = json.loads(baseline_path.read_text())
    forged["inventory_sha256"] = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    forged["interfaces"]["Original25"]["sha256"] = item["sha256"]
    baseline_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="normalized row 0"):
        PIPELINE.import_baseline(SimpleNamespace(
            p0=args.p0, manifest=str(tmp_path / "unused.json"),
            output=str(baseline_path), verify_existing=True,
            curve_authority=args.curve_authority,
        ))


def test_certified_baseline_rejects_metric_digest_drift(tmp_path, monkeypatch):
    args = _baseline_materialization_fixture(tmp_path, monkeypatch)
    curve = json.loads(Path(args.curve_report).read_text())
    curve["evidence"]["interfaces"]["Original25"]["independent_metric_rows_sha256"] = "f" * 64
    Path(args.curve_report).write_text(json.dumps(curve))
    p0 = json.loads(Path(args.p0).read_text())
    p0["original_curve_report_sha256"] = hashlib.sha256(
        Path(args.curve_report).read_bytes()
    ).hexdigest()
    Path(args.p0).write_text(json.dumps(p0))
    authority = json.loads(Path(args.curve_authority).read_text())
    authority["curve_report_sha256"] = p0["original_curve_report_sha256"]
    Path(args.curve_authority).write_text(json.dumps(authority))
    with pytest.raises(ValueError, match="Original25 metric-row digest"):
        PIPELINE.materialize_baseline(args)


def test_replaced_curve_resolved_execution_binding_is_rejected(tmp_path, monkeypatch):
    args = _baseline_materialization_fixture(tmp_path, monkeypatch)
    resolved = json.loads(Path(args.curve_resolved).read_text())
    resolved["execution_binding"]["trainer_configuration"][
        "shared_generation_protocol_sha256"
    ] = "f" * 64
    Path(args.curve_resolved).write_text(json.dumps(resolved))
    with pytest.raises(ValueError, match="release-pinned curve authority"):
        PIPELINE.materialize_baseline(args)


def test_final_audit_rejects_self_consistent_eval_report_rewrite(tmp_path, monkeypatch):
    keys = [f"stable-{index}" for index in range(128)]
    original_rows = [
        {"stable_key": key, "output": "wrong", "ground_truth": f"answer-{index}"}
        for index, key in enumerate(keys)
    ]
    original_path = tmp_path / "original.jsonl"
    _write_jsonl(original_path, original_rows)
    original_scored = [
        {"stable_key": row["stable_key"],
         **score_terminal_output(row["output"], row["ground_truth"])}
        for row in original_rows
    ]
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text("{}")
    monkeypatch.setenv("MEMAGENT_MIC_BASELINE_INVENTORY", str(inventory_path))
    interface = {
        "path": str(original_path),
        "sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "aggregate": summarize_fixed_s128(original_scored),
        "stable_key_inventory_sha256": PIPELINE.sha256_json(keys),
    }
    certified = {
        "status": "PASS", "decision": "MIC_BASELINE_IMPORT_PASS",
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "validation_path": str(tmp_path / "validation.parquet"),
        "interfaces": {f"Original{step}": interface for step in (5, 10, 15, 20, 25)},
    }
    monkeypatch.setattr(PIPELINE, "import_baseline", lambda _: certified)
    monkeypatch.setattr(PIPELINE, "prepare_eval", lambda _: {
        "status": "PASS", "checkpoint_authority_sha256": "c" * 64,
    })
    monkeypatch.setattr(PIPELINE, "audit", lambda _: {
        "status": "PASS", "decision": "MIC_T25_AUDIT_PASS",
    })
    cert_root, eval_root = tmp_path / "certificates", tmp_path / "evaluations"
    cert_root.mkdir(); eval_root.mkdir()
    for step in (5, 10, 15, 20, 25):
        anchor = eval_root / f"eval_t{step}_attempts" / "attempt_0001"
        anchor.mkdir(parents=True)
        (anchor / "prepare.json").write_text("{}")
        predictions = [
            {"stable_key": key, "output": f"answer-{index}",
             "ground_truth": f"answer-{index}"}
            for index, key in enumerate(keys)
        ]
        _write_jsonl(anchor / "predictions.jsonl", predictions)
        (cert_root / f"t{step}_audit.json").write_text(json.dumps({
            "status": "PASS", "decision": f"MIC_T{step}_AUDIT_PASS",
        }))
        PIPELINE.evaluate(SimpleNamespace(
            predictions=str(anchor / "predictions.jsonl"), baseline=str(tmp_path / "baseline.json"),
            p0=str(tmp_path / "p0.json"), step=step,
            output=str(cert_root / f"t{step}_eval.json"), verify_existing=False,
        ))
    forged_path = cert_root / "t25_eval.json"
    forged = json.loads(forged_path.read_text())
    forged["aggregate"]["token_f1"] = 0.0
    forged["token_f1_delta_pp"] = 0.0
    forged["delta_percentage_points"]["token_f1"] = 0.0
    forged_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="evaluation certificate differs from recomputation"):
        PIPELINE.final_eval_audit(SimpleNamespace(
            baseline=str(tmp_path / "baseline.json"), p0=str(tmp_path / "p0.json"),
            health_root=str(cert_root), eval_root=str(eval_root), output_root=str(tmp_path / "run"),
            checkpoint_authority=str(tmp_path / "authority.json"),
            checkpoint_authority_certificate=str(tmp_path / "authority_cert.json"),
            weight_ledger=str(tmp_path / "weights.jsonl"),
            mic_ledger=str(tmp_path / "mic.jsonl"), e0=str(tmp_path / "e0.json"),
            paper_review=str(tmp_path / "paper.json"),
            output=str(cert_root / "final.json"),
        ))
