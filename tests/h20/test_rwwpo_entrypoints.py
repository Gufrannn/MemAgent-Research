import hashlib, json, os, subprocess
from pathlib import Path
import pytest
from tools.h20.materialize_rwwpo_baseline_bundle import authenticated_root,safe_file
from tools.h20.audit_rwwpo_baseline_bundle import safe_file as audit_safe_file

ROOT=Path(__file__).resolve().parents[2]

def run(script,*args,env=None):
    return subprocess.run([str(script),*map(str,args)],cwd=ROOT,text=True,capture_output=True,env=env)

def test_preflight_rejects_noncanonical_gpu_pair(tmp_path):
    for name,decision in (("e0","RWWPO_E0_PASS"),("e1","RWWPO_E1_PASS")):
        (tmp_path/f"{name}.json").write_text(json.dumps({"status":"PASS","decision":decision}))
    (tmp_path/"baseline.json").write_text(json.dumps({"status":"PASS","decision":"ORIGINAL_BASELINE_IMPORT_PASS"}))
    original=ROOT/"manifests/h20/qwen25_7b_original_t25_seed2026.json"
    original_sha=hashlib.sha256(original.read_bytes()).hexdigest()
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    result=run(ROOT/"tools/h20/preflight_rwwpo.py","--manifest",ROOT/"manifests/h20/qwen25_7b_rwwpo_seed2026.json","--expected-commit",head,"--gpu-pair","7,3","--e0",tmp_path/"e0.json","--e1",tmp_path/"e1.json","--baseline-import",tmp_path/"baseline.json","--original-resolved-manifest",original,"--original-resolved-sha256",original_sha,"--phase","t5","--objective-variant","whole_prefix","--controller-variant","hard_rollback")
    assert result.returncode != 0 and "canonical" in result.stderr

def test_actual_ledger_rejects_hash_tamper(tmp_path):
    row={"schema_version":"rwwpo-actual-loss-v1","attempt_id":"t5_primary","mode":"rwwpo_method","global_step":1,"rank":0,"epoch":0,"minibatch":0,"old_log_prob":[[0.]],"current_log_prob":[[.1]],"response_mask":[[1.]],"writer_mask":[[1.]],"answer_mask":[[0.]],"trajectory_turn":[0],"sample_index":[0],"advantages":[[1.]],"denominator":1,"prefix_stats":[{"turn":0,"batch_size":1,"ess_fraction":1.,"chi2":0.}],"q_min":.5,"constraint_pass":True,"record_sha256":"0"*64}
    path=tmp_path/"bad.jsonl"; path.write_text(json.dumps(row)+"\n")
    result=run(ROOT/"tools/h20/audit_rwwpo_actual_loss.py",path)
    assert result.returncode != 0 and "hash mismatch" in result.stderr

def test_launcher_requires_semantic_identity_and_explicit_pair():
    env=os.environ.copy(); env.update(RWWPO_WORK_ROOT="/tmp/rwwpo-test",RWWPO_REPO_DIR=str(ROOT),RWWPO_EXPECTED_COMMIT="0"*40,RWWPO_PHASE="t5")
    result=run(ROOT/"scripts/h20/run_qwen25_7b_rwwpo.sh",env=env)
    assert result.returncode != 0 and "GPU_PAIR" in result.stderr

def _signed(record):
    record=dict(record)
    record["record_sha256"]=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return record

def test_actual_ledger_rejects_forged_post_acceptance(tmp_path):
    stat={"turn":0,"batch_size":1,"ess_fraction":1.0,"chi2":0.0,"max_abs_log_ratio":0.1}
    row={"schema_version":"rwwpo-actual-loss-v1","attempt_id":"t5_primary","mode":"rwwpo_method","global_step":1,"rank":0,"epoch":0,"minibatch":0,
         "old_log_prob":[[0.0]],"current_log_prob":[[0.1]],"proposed_post_log_prob":[[0.1]],"response_mask":[[1.0]],"writer_mask":[[1.0]],"answer_mask":[[0.0]],
         "trajectory_turn":[0],"sample_index":[0],"advantages":[[1.0]],"denominator":1,
         "prefix_rows":[{"turn":0,"sample_index":0,"log_ratio":0.1,"prefix_token_count":1}],"prefix_stats":[stat],
         "post_prefix_rows":[{"turn":0,"sample_index":0,"log_ratio":0.1,"prefix_token_count":1}],"post_prefix_stats":[stat],
         "q_min":0.5,"writer_log_ratio_cap":4.0,"constraint_pass":True,"accepted":False}
    path=tmp_path/"forged.jsonl"; path.write_text(json.dumps(_signed(row))+"\n")
    result=run(ROOT/"tools/h20/audit_rwwpo_actual_loss.py",path)
    assert result.returncode != 0 and "accepted decision" in result.stderr

def test_compare_rejects_forged_receipt(tmp_path):
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    method={"status":"PASS","decision":"RWWPO_T5_S128_PASS","git_commit":head,"metrics":{"token_f1":1.0},"report_sha256":"0"*64}
    base={"status":"PASS","decision":"ORIGINAL_BASELINE_IMPORT_PASS","git_commit":head,"aggregates":{"Original5":{"token_f1":1.0}},"report_sha256":"0"*64}
    (tmp_path/"method.json").write_text(json.dumps(method)); (tmp_path/"base.json").write_text(json.dumps(base))
    result=run(ROOT/"tools/h20/compare_rwwpo_anchor.py","--method",tmp_path/"method.json","--baseline-import",tmp_path/"base.json","--step",5,"--expected-commit",head,"--output",tmp_path/"out.json")
    assert result.returncode != 0 and "receipt" in result.stderr

def test_baseline_materializer_rejects_path_escape_and_symlink(tmp_path):
    root=tmp_path/"source"; root.mkdir(); outside=tmp_path/"outside.json"; outside.write_text("{}")
    with pytest.raises(ValueError,match="escape"):
        safe_file(outside,root)
    link=root/"terminal.jsonl"; link.symlink_to(outside)
    with pytest.raises(ValueError,match="symlink"):
        safe_file(link,root)

def test_baseline_materializer_rejects_unauthorized_curve_root(tmp_path):
    authority=tmp_path/"authority"; authority.mkdir()
    outside=tmp_path/"outside"; outside.mkdir()
    with pytest.raises(ValueError,match="unauthorized"):
        authenticated_root(outside,[authority])

def test_baseline_audit_rejects_dotdot_terminal_escape(tmp_path):
    root=tmp_path/"authority"; (root/"terminal").mkdir(parents=True)
    outside=tmp_path/"outside.jsonl"; outside.write_text("{}\n")
    with pytest.raises(SystemExit,match="escape"):
        audit_safe_file(root/"terminal"/".."/".."/"outside.jsonl",root)

def test_full_launcher_is_single_fresh_t25_run():
    text=(ROOT/"scripts/h20/run_qwen25_7b_rwwpo.sh").read_text()
    assert "RWWPO_PHASE == full" in text
    assert "FRESH_TOTAL_STEPS=25" in text
    assert "--e1" not in text.split("PREFLIGHT=",1)[1].split("if [[",1)[0]

def test_importer_requires_independent_bundle_audit():
    text=(ROOT/"tools/h20/import_rwwpo_original_baseline.py").read_text()
    assert 'add_argument("--bundle-audit",required=True)' in text
    assert "RWWPO_BASELINE_BUNDLE_AUDIT_PASS" in text
    assert "authority_chain" in text

def test_materializer_and_auditor_pin_repository_authority():
    for name in ("materialize_rwwpo_baseline_bundle.py","audit_rwwpo_baseline_bundle.py"):
        text=(ROOT/"tools/h20"/name).read_text()
        assert "CANONICAL_AUTHORITY" in text
        assert "noncanonical authority" in text

def test_original5_authority_digest_matches_h20_final_report_readback():
    authority=json.loads((ROOT/"manifests/h20/rwwpo_original_evidence_authority_20260822.json").read_text())
    assert authority["original_s128_curve"]["canonical_metric_row_digests"]["Original5"] == "58b01ad5e523ee8853c05af691a65948a0d905d22f2c6ffb0590484c5a38a30d"

def test_authority_pins_all_h20_interface_roots():
    authority=json.loads((ROOT/"manifests/h20/rwwpo_original_evidence_authority_20260822.json").read_text())
    roots=authority["original_s128_curve"]["authenticated_interface_roots"]
    assert set(roots)=={"I","Original5","Original10","Original15","Original20","Original25"}
    assert roots["I"].endswith("/s128_it_original_t25_frozen_20260821/interface_i_base")
    assert roots["Original25"].endswith("/s128_it_original_t25_frozen_20260821/interface_t25_original")

def test_method_runtime_has_fail_closed_numeric_health_checks():
    text=(ROOT/"verl/workers/actor/dp_actor.py").read_text()
    assert "non-finite policy loss" in text
    assert "non-finite active-token log probability" in text
    assert "non-finite gradient norm" in text

def test_rwwpo_runtime_does_not_append_gpu_minibatch_as_metrics():
    text=(ROOT/"verl/workers/actor/dp_actor.py").read_text()
    marker='append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item(),'
    tail=text.split(marker,1)[1].split("continue",1)[0]
    assert "data = {}" in tail

def test_activity_audit_uses_accepted_post_step_movement_not_behavior_drift():
    text=(ROOT/"tools/h20/audit_rwwpo_actual_loss.py").read_text()
    active=text.split("active =",1)[1].split("if require_method",1)[0]
    assert 'group[0]["accepted"]' in active
    assert 'row["proposed_post_log_prob"]' in active
    assert 'row["current_log_prob"]' in active
    assert 'row["old_log_prob"]' not in active

def test_target_anchor_acceptance_gate_remains_fail_closed():
    text=(ROOT/"tools/h20/audit_rwwpo_run.py").read_text()
    assert 'target_actual.get("accepted_fraction",0) <= 0' in text
    assert 'target_actual.get("max_proposed_update",0) <= 1e-10' in text

def test_checkpoint_producer_records_inventory_after_data_state_save():
    text=(ROOT/"verl/trainer/ppo/ray_trainer.py").read_text()
    save=text.split("def _save_checkpoint(self):",1)[1].split(
        "def _audit_gate_a_weight_sync",1)[0]
    assert save.index("torch.save(dataloader_state_dict") < save.index(
        'append_gate_a_record(\n                "checkpoint_inventory"')
    assert "inventory=checkpoint_inventory(local_global_step_folder)" in save

def test_post_hoc_s128_requires_explicit_diagnostic_label_and_separate_root():
    launcher=(ROOT/"scripts/h20/run_rwwpo_s128_anchor.sh").read_text()
    preflight=(ROOT/"tools/h20/preflight_rwwpo_s128.py").read_text()
    audit=(ROOT/"tools/h20/audit_rwwpo_s128.py").read_text()
    assert "RWWPO_DIAGNOSTIC_ONLY" in launcher
    assert "RWWPO_EVAL_ATTEMPT_SUFFIX" in launcher
    assert "--diagnostic-only" in preflight
    assert 'raw.get("diagnostic_only") is not True' in preflight
    assert '"DIAGNOSTIC_ONLY" if a.diagnostic_only' in audit
    assert 'a.diagnostic_only != (raw_resolved.get("diagnostic_only") is True)' in audit

def test_formal_compare_cannot_accept_diagnostic_status():
    text=(ROOT/"tools/h20/compare_rwwpo_anchor.py").read_text()
    assert 'row.get("status")!="PASS"' in text

def test_tf_launcher_pins_new_branch_and_controller():
    text=(ROOT/"scripts/h20/run_qwen25_7b_tf_rwwpo.sh").read_text()
    assert "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822" in text
    assert "RWWPO_CONTROLLER_VARIANT=feasible_backtracking" in text

def test_tf_manifest_freezes_grid_constraint_and_t25_chain():
    row=json.loads((ROOT/"manifests/h20/qwen25_7b_tf_rwwpo_seed2026.json").read_text())
    assert row["method"]["q_min"]==0.5
    assert row["method"]["alpha_grid"]==[1.0,.5,.25,.125,.0625,.03125]
    assert row["training"]["target_steps"]==[5,10,15,20,25]

def test_transactional_scheduler_is_not_advanced_by_worker():
    actor=(ROOT/"verl/workers/actor/dp_actor.py").read_text()
    worker=(ROOT/"verl/workers/fsdp_workers.py").read_text()
    assert "self.actor_lr_scheduler.step()" in actor
    assert 'if not self.config.actor.get("rwwpo", {}).get("enable", False)' in worker

def test_transaction_audit_rejects_interrupted_trial():
    text=(ROOT/"tools/h20/audit_rwwpo_run.py").read_text()
    assert "interrupted trial transaction" in text
    assert "transaction marker chain" in text

def test_old_diagnostic_is_never_promoted_to_formal():
    row=json.loads((ROOT/"manifests/h20/rwwpo_hard_rollback_diagnostic_inventory_20260822.json").read_text())
    assert row["status"]=="DIAGNOSTIC_ONLY"
    assert row["formal_training_health"]=="NO_GO"
