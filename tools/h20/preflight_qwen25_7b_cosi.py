#!/usr/bin/env python3
"""Fail-closed COSI preflight. It never allocates or modifies a GPU."""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from recurrent.research.cosi import canonical_sha256, sha256_file, validate_ledger

BRANCH="h20/qwen25-7b-cosi-t25-frozen-20260822"
def git(*args): return subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()
def authenticated(path, decision):
    value=json.loads(path.read_text()); digest=value.get("report_sha256")
    unsigned={k:v for k,v in value.items() if k!="report_sha256"}
    if value.get("status")!="PASS" or value.get("decision")!=decision or digest!=canonical_sha256(unsigned):
        raise ValueError(f"COSI_NO_GO: invalid gate {path}")
    return value
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--stage",choices=["research","t5","continue"],required=True); p.add_argument("--write-certificate",action="store_true"); a=p.parse_args()
    env=os.environ; required=["MEMAGENT_COSI_WORK_ROOT","MEMAGENT_COSI_REPO_DIR","MEMAGENT_COSI_EXPECTED_COMMIT","MEMAGENT_COSI_GPU_PAIR","MEMAGENT_COSI_BASELINE_INDEX","MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST","MEMAGENT_COSI_S128_RESOLVED_MANIFEST"]
    missing=[x for x in required if not env.get(x)];
    if missing: raise ValueError(f"COSI_NO_GO: missing explicit bindings {missing}")
    repo=Path(env["MEMAGENT_COSI_REPO_DIR"]).resolve(); work=Path(env["MEMAGENT_COSI_WORK_ROOT"]).resolve()
    if repo!=ROOT.resolve() or not work.is_absolute(): raise ValueError("COSI_NO_GO: repository/work-root binding")
    expected=env["MEMAGENT_COSI_EXPECTED_COMMIT"]
    if not re.fullmatch(r"[0-9a-f]{40}",expected) or git("rev-parse","HEAD")!=expected: raise ValueError("COSI_NO_GO: exact commit mismatch")
    if git("branch","--show-current")!=BRANCH or git("status","--porcelain"): raise ValueError("COSI_NO_GO: wrong branch or dirty worktree")
    pair=env["MEMAGENT_COSI_GPU_PAIR"]
    if not re.fullmatch(r"[0-9]+,[0-9]+",pair): raise ValueError("COSI_NO_GO: GPU pair syntax")
    gpu=[int(v) for v in pair.split(",")]
    if gpu!=sorted(set(gpu)) or len(gpu)!=2: raise ValueError("COSI_NO_GO: GPU pair must be distinct canonical ascending")
    manifest_path=Path(a.manifest).resolve(); manifest=json.loads(manifest_path.read_text())
    if manifest.get("schema")!="memagent.cosi.run-manifest.v1" or not manifest.get("fresh_base_only") or manifest.get("method_active_from_update")!=1: raise ValueError("COSI_NO_GO: manifest method/fresh-base drift")
    if manifest["training"]!={"seed":2026,"trajectory_seed_mode":"independent","train_batch_size":4,"rollout_n":2,"ppo_mini_batch_size":4,"chunk_size":5000,"max_chunks":8,"actor_learning_rate":0.000001,"kl_loss_coefficient":0.001,"anchors":[5,10,15,20,25]}: raise ValueError("COSI_NO_GO: frozen training fields drifted")
    from recurrent.research.coral import validate_config
    validate_config(manifest.get("coral", {}))
    original_path=Path(env["MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST"]).resolve()
    if not original_path.is_file(): raise ValueError("COSI_NO_GO: accepted Original resolved manifest missing")
    original=json.loads(original_path.read_text())
    original_model=original.get("model",{})
    original_config_sha=original_model.get("config_sha256")
    if original_config_sha is None:
        original_config_sha=next((item.get("sha256") for item in original_model.get("files",[]) if item.get("path")=="config.json"),None)
    comparisons={
        "model.id": (manifest["model"]["id"], original_model.get("id")),
        "model.revision": (manifest["model"]["revision"], original_model.get("revision")),
        "model.config_sha256": (manifest["model"]["config_sha256"], original_config_sha),
        "data.train_sha256": (manifest["data"]["train_sha256"], original["data"]["train_sha256"]),
        "data.validation_sha256": (manifest["data"]["validation_sha256"], original["data"]["validation_sha256"]),
        "data.shuffle": (manifest["data"]["shuffle"], original["data"]["shuffle"]),
        "training.seed": (manifest["training"]["seed"], original["training"]["seed"]),
        "training.trajectory_seed_mode": (manifest["training"]["trajectory_seed_mode"], original["training"]["trajectory_seed_mode"]),
        "training.train_batch_size": (manifest["training"]["train_batch_size"], original["training"]["train_batch_size"]),
        "training.rollout_n": (manifest["training"]["rollout_n"], original["training"]["rollout_n"]),
        "training.ppo_mini_batch_size": (manifest["training"]["ppo_mini_batch_size"], original["training"]["ppo_mini_batch_size"]),
        "training.chunk_size": (manifest["training"]["chunk_size"], original["training"]["chunk_size"]),
        "training.max_chunks": (manifest["training"]["max_chunks"], original["training"]["max_chunks"]),
        "training.actor_learning_rate": (manifest["training"]["actor_learning_rate"], original["training"]["actor_learning_rate"]),
        "training.kl_loss_coefficient": (manifest["training"]["kl_loss_coefficient"], original["training"]["kl_loss_coefficient"]),
        "backend.rollout": (manifest["backend"]["rollout"], original["backend"]["rollout"]),
        "backend.reward_manager": ("naive", original["backend"]["reward_manager"]),
    }
    drift={key:{"method":left,"original":right} for key,(left,right) in comparisons.items() if left!=right}
    if drift: raise ValueError(f"COSI_NO_GO: non-whitelisted Original protocol drift {drift}")
    cert=work/"logs"/"cosi_preflight"/"certificates"; gates={}
    if a.stage in ("t5","continue"):
        # The repository never self-authorizes.  T5 becomes runnable only
        # after all externally issued, content-addressed gates below pass.
        if manifest.get("training_authorized") is not False \
                or manifest.get("authorization_mode") \
                != "external_authenticated_gates_only":
            raise ValueError("COSI_NO_GO: repository attempted self-authorization")
        expected_gate_hashes={
            "paper":"MEMAGENT_COSI_PAPER_REVIEW_SHA256",
            "e0":"MEMAGENT_COSI_E0_REPORT_SHA256",
            "e1":"MEMAGENT_COSI_E1_REPORT_SHA256",
            "baseline":"MEMAGENT_COSI_BASELINE_REPORT_SHA256",
        }
        for gate,variable in expected_gate_hashes.items():
            value=env.get(variable,"")
            if not re.fullmatch(r"[0-9a-f]{64}",value):
                raise ValueError(f"COSI_NO_GO: missing external expected hash {variable}")
        gates["paper"]=authenticated(cert/"paper_framing_review.json","CORAL_PAPER_FRAMING_GO")
        gates["e0"]=authenticated(cert/"coral_e0.json","CORAL_E0_PASS")
        gates["e1"]=authenticated(cert/"coral_e1_final_report.json","CORAL_E1_PASS")
        gates["baseline"]=authenticated(cert/"baseline_import.json","COSI_BASELINE_IMPORT_PASS")
        gate_paths={
            "paper":cert/"paper_framing_review.json",
            "e0":cert/"coral_e0.json",
            "e1":cert/"coral_e1_final_report.json",
            "baseline":cert/"baseline_import.json",
        }
        for gate,path in gate_paths.items():
            if sha256_file(path)!=env[expected_gate_hashes[gate]]:
                raise ValueError(f"COSI_NO_GO: external expected hash mismatch for {gate}")
    if a.stage=="continue": gates["t5"]=authenticated(cert/"t5_health.json","COSI_T5_HEALTH_PASS")
    report={"schema":"memagent.cosi.preflight.v1","status":"PASS","decision":f"COSI_{a.stage.upper()}_P0_PASS","stage":a.stage,"git_commit":expected,"manifest_sha256":sha256_file(manifest_path),"original_resolved_manifest_sha256":sha256_file(original_path),"gpu_pair":gpu,"gate_hashes":{k:v["report_sha256"] for k,v in gates.items()}}
    report["report_sha256"]=canonical_sha256(report)
    if a.write_certificate:
        cert.mkdir(parents=True,exist_ok=True); out=cert/f"p0_{a.stage}.json"
        with out.open("x") as f: json.dump(report,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
