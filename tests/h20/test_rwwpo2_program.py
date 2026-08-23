import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import torch

from tools.h20.audit_rwwpo2_r50_program import wilson_interval
from tools.h20.audit_rwwpo2_attempt import execution_prefix_through_round
from tools.h20.audit_rwwpo2_lineage_parent import execution_prefix_to_checkpoint
from tools.h20.preflight_rwwpo2 import receipt
from tools.h20.verify_rwwpo2_release_tests import (
    TEST_INVENTORY, canonical_sha as release_test_canonical_sha,
    collect_current_node_ids, junit_summary as release_test_junit_summary,
    node_evidence, runtime_environment, sha256_file, verify_release_test_receipt,
)
from recurrent.research.gate_a_execution import append_jsonl
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


def test_recovery_contract_authenticates_prefix_and_excludes_failed_suffix():
    lineage = (ROOT / "tools/h20/audit_rwwpo2_lineage_parent.py").read_text()
    trainer = (ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
    attempt = (ROOT / "tools/h20/audit_rwwpo2_attempt.py").read_text()
    for token in ("prefix_sha256", "tail_sha256", "tensor_inventory",
                  "checkpoint_inventory_event_sha256", "failed_suffix_imported",
                  "record_limits", "rwwpo_rollout_seed_anchor"):
        assert token in lineage
    assert "execution_prefix_to_checkpoint" in lineage
    assert "execution_prefix_through_round" in attempt
    assert "record_limits=record_limits" in attempt
    assert "RWWPO2_RESUME_ACCEPTED_OPTIMIZER_CLOCK_DRIFT" in trainer
    assert "rwwpo_rollout_seed_anchor" in trainer
    assert "rwwpo2_recovery_pruned" in trainer
    assert "scientific_anchor_preserved=True" in trainer
    assert "anchor hardlink" in attempt
    assert '"resolved_contract_file_sha256"' in attempt
    assert '"resolved_contract_report_sha256"' in attempt
    assert 'parser.add_argument("--preflight", required=True)' in attempt
    assert '"preflight_report_sha256"' in attempt
    assert "R400 preflight gate binding" in attempt
    assert "preflight lineage start" in attempt
    assert "validate_rwwpo2_rng_phase_digests(row)" in attempt
    gate = (ROOT / "tools/h20/audit_rwwpo2_r50_program.py").read_text()
    assert "segment contract binding" in gate


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
