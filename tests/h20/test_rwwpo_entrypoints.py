import hashlib, json, os, subprocess
from pathlib import Path

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
    result=run(ROOT/"tools/h20/preflight_rwwpo.py","--manifest",ROOT/"manifests/h20/qwen25_7b_rwwpo_seed2026.json","--expected-commit",head,"--gpu-pair","7,3","--e0",tmp_path/"e0.json","--e1",tmp_path/"e1.json","--baseline-import",tmp_path/"baseline.json","--original-resolved-manifest",original,"--original-resolved-sha256",original_sha,"--phase","t5")
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
