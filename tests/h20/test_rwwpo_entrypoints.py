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
