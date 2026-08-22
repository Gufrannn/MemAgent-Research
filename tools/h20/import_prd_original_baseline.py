#!/usr/bin/env python3
"""Read-only, per-file-SHA import of the certified Original S128 curve."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_fixed_s128
from recurrent.research.stable_eval_identity import canonical_sha256, stable_key
from tools.h20.preflight_qwen25_7b_stable_i4x2 import _load_parquet_rows

ANCHORS={"I":0,"Original5":5,"Original10":10,"Original15":15,"Original20":20,"Original25":25}
ROW_DIGESTS={
    "I":"fd4c6763c8d8a6caa0389082f1fa838dc510d872b99e6283c1483c4427336c64",
    "Original5":"58b01ad5e523ee8853c05af691a659480d0d905d22f2c6ffb0590484c5a38a30d",
    "Original10":"bc5c29e7e6f163828758cb68dca1237f9d970af24217f60e272ba2945017b4a4",
    "Original15":"3e8ae48f4a092ec136c568397037b9f270532bb0ab92a6b976c4de66c2c02b2f",
    "Original20":"8a831d5d96c4f963f53a6a8d2c01a6a1414724a8a189a05f5a89c68d56494cd8",
    "Original25":"4db791e409edeb269b56b1633b07c272ef04abf8b15da5c479a1e7822a93b2d6",
}
STABLE_RESOLVED_SHA="6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411"
TRAINING_REPORT_SHA="33cab1eb09eefd89b7f764d0f2c6851eac5e58dc7c0a3d147c30ce05522c9040"
TRAINING_REPORT_PATH=Path("/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/original_t25_final_report.json")
TRAINING_RESOLVED_PATH=Path("/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json")

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--inventory"); p.add_argument("--curve-report"); p.add_argument("--stable-resolved"); p.add_argument("--validation-parquet"); p.add_argument("--original-training-final-report"); p.add_argument("--original-training-resolved"); p.add_argument("--output",required=True); a=p.parse_args()
    if a.curve_report:
        required=(a.stable_resolved,a.validation_parquet,a.original_training_final_report,a.original_training_resolved)
        if not all(required): p.error("curve-report mode requires stable-resolved, validation-parquet, original-training-final-report, and original-training-resolved")
        report_path=Path(a.curve_report).resolve(); stable_path=Path(a.stable_resolved).resolve()
        training_report_path=Path(a.original_training_final_report).resolve(); training_resolved=Path(a.original_training_resolved).resolve(); failures=[]; results={}; imported=[]
        report=json.loads(report_path.read_text()); stable=json.loads(stable_path.read_text())
        if report.get("status")!="PASS" or report.get("decision")!="ORIGINAL_S128_CURVE_PASS": failures.append("curve report is not certified PASS")
        if sha(stable_path)!=STABLE_RESOLVED_SHA: failures.append("stable S128 resolved SHA mismatch")
        if stable.get("eval_manifest_hash")!="351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a": failures.append("stable S128 eval identity mismatch")
        training_report=json.loads(training_report_path.read_text())
        if training_report_path!=TRAINING_REPORT_PATH or sha(training_report_path)!=TRAINING_REPORT_SHA: failures.append("Original training final report path/SHA mismatch")
        if training_report.get("status")!="PASS" or training_report.get("decision")!="ORIGINAL_T25_PASS": failures.append("Original training report is not certified PASS")
        if training_resolved!=TRAINING_RESOLVED_PATH: failures.append("Original training resolved manifest path mismatch")
        training_p0=training_report_path.parent/"p0_preflight.json"
        if not training_p0.is_file() or sha(training_p0)!=training_report.get("p0_certificate_sha256"): failures.append("Original training P0 certificate authentication failed")
        else:
            p0=json.loads(training_p0.read_text()); evidence=p0.get("evidence",{})
            if p0.get("status")!="PASS" or p0.get("decision")!="T25_P0_PASS": failures.append("Original training P0 is not PASS")
            if Path(str(evidence.get("resolved_manifest_path",""))).resolve()!=training_resolved: failures.append("Original training P0 resolved path mismatch")
            if canonical_sha256(json.loads(training_resolved.read_text()))!=evidence.get("resolved_manifest_sha256"): failures.append("Original training resolved canonical digest mismatch")
        raw=_load_parquet_rows(Path(a.validation_parquet).resolve()); truth={}
        for row in stable.get("identity_payload",{}).get("rows",[]):
            reward=raw[int(row["raw_row_position"])].get("reward_model"); reward=json.loads(reward) if isinstance(reward,str) else reward
            gt=reward["ground_truth"]
            if canonical_sha256(gt)!=row["ground_truth_hash"]: failures.append(f"ground truth hash mismatch at {row['source_order_index']}")
            truth[int(row["source_order_index"])]=gt
        interfaces=report.get("evidence",{}).get("interfaces",{})
        for name,anchor in ANCHORS.items():
            ev=interfaces.get(name,{}) ; root=Path(str(ev.get("root",""))).resolve(); artifacts=ev.get("artifacts",{})
            for rel,spec in artifacts.items():
                path=(root/rel).resolve()
                if not path.is_file() or path.is_symlink() or sha(path)!=spec.get("sha256") or path.stat().st_size!=spec.get("size"): failures.append(f"artifact inventory mismatch: {name}/{rel}")
                else: imported.append({"interface":name,"path":str(path),"sha256":spec["sha256"],"size":spec["size"]})
            terminal=root/f"terminal/{anchor}.jsonl"; rows=[]
            if terminal.is_file():
                for source in (json.loads(line) for line in terminal.read_text().splitlines() if line.strip()):
                    order=int(source["source_order_index"]); scored=score_terminal_output(source["output"],truth[order])
                    rows.append({"stable_key":json.dumps(stable_key(source),separators=(",",":")),"source_order_index":order,"eval_manifest_hash":source["eval_manifest_hash"],"example_id":source["example_id"],"replica_id":source["replica_id"],"trajectory_seed":source["trajectory_seed"],"trajectory_id":source["trajectory_id"],**scored})
            if len(rows)!=128 or canonical_sha256(rows)!=ROW_DIGESTS[name]: failures.append(f"canonical metric rows mismatch: {name}")
            else: results[str(anchor)]=summarize_fixed_s128(rows)
        payload={"schema_version":2,"timestamp":datetime.now(timezone.utc).isoformat(),"status":"PASS" if not failures else "FAIL","decision":"PRD_ORIGINAL_BASELINE_IMPORT_PASS" if not failures else "ORIGINAL_BASELINE_PROTOCOL_MISMATCH","source_curve_report":str(report_path),"source_curve_report_sha256":sha(report_path),"stable_resolved":str(stable_path),"stable_resolved_sha256":sha(stable_path),"original_training_final_report":str(training_report_path),"original_training_final_report_sha256":sha(training_report_path),"original_training_p0_sha256":sha(training_p0) if training_p0.is_file() else None,"original_training_resolved":str(training_resolved),"original_training_resolved_sha256":sha(training_resolved),"canonical_metric_row_digests":ROW_DIGESTS,"imported_files":imported,"recomputed":results,"actual_loss_status":"PENDING_ACTUAL_LOSS_LEDGER","failures":failures}
        out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("x") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n")
        return 0 if not failures else 5
    if not a.inventory: p.error("one of --inventory or --curve-report is required")
    inv_path=Path(a.inventory).resolve(); inv=json.loads(inv_path.read_text()); failures=[]; results={}; imported=[]
    if inv.get("readonly") is not True or inv.get("anchors") != [0,5,10,15,20,25]: failures.append("inventory contract mismatch")
    seen=None
    for item in inv.get("files",[]):
        path=Path(item["path"]).resolve()
        if not path.is_file() or sha(path)!=item.get("sha256"): failures.append(f"SHA mismatch: {path}"); continue
        rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if len(rows)!=128: failures.append(f"not fixed S128: {path}"); continue
        keys=[str(x.get("stable_key","")) for x in rows]
        if len(set(keys))!=128 or (seen is not None and set(keys)!=seen): failures.append(f"stable-ID mismatch: {path}"); continue
        seen=set(keys)
        scored=[score_terminal_output(x["terminal_output"],x["ground_truth"]) for x in rows]
        aggregate=summarize_fixed_s128(scored)
        expected=item.get("expected_aggregate")
        if expected and any(abs(float(aggregate[k])-float(expected[k]))>1e-12 for k in ("normalized_exact_match","token_f1","format_success")): failures.append(f"aggregate mismatch: {path}"); continue
        results[str(item["anchor"])]=aggregate; imported.append({"path":str(path),"sha256":item["sha256"]})
    if set(results)!={"0","5","10","15","20","25"}: failures.append("incomplete anchor curve")
    payload={"schema_version":1,"timestamp":datetime.now(timezone.utc).isoformat(),"status":"PASS" if not failures else "FAIL","decision":"PRD_ORIGINAL_BASELINE_IMPORT_PASS" if not failures else "ORIGINAL_BASELINE_PROTOCOL_MISMATCH","source_inventory":str(inv_path),"source_inventory_sha256":sha(inv_path),"imported_files":imported,"recomputed":results,"failures":failures}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n")
    return 0 if not failures else 5
if __name__=="__main__": raise SystemExit(main())
