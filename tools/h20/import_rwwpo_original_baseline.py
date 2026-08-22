#!/usr/bin/env python3
import argparse, hashlib, json, time, sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_fixed_s128

def main():
    p=argparse.ArgumentParser(); p.add_argument("--bundle",required=True); p.add_argument("--bundle-sha256",required=True); p.add_argument("--bundle-audit",required=True); p.add_argument("--output",required=True); p.add_argument("--expected-commit",required=True); a=p.parse_args()
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if head!=a.expected_commit: raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:git_commit")
    bundle_path=Path(a.bundle)
    if hashlib.sha256(bundle_path.read_bytes()).hexdigest()!=a.bundle_sha256: raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:bundle_sha")
    audit=json.loads(Path(a.bundle_audit).read_text()); declared=audit.pop("report_sha256",None); actual=hashlib.sha256(json.dumps(audit,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if declared!=actual or audit.get("status")!="PASS" or audit.get("decision")!="RWWPO_BASELINE_BUNDLE_AUDIT_PASS" or audit.get("git_commit")!=head or audit.get("bundle_sha256")!=a.bundle_sha256: raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:bundle_audit")
    spec=json.loads(bundle_path.read_text()); imported=[]
    if audit.get("authority_sha256")!=spec.get("authority_sha256"): raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:authority_chain")
    if set(spec.get("interfaces",{}))!={"I","Original5","Original10","Original15","Original20","Original25"}: raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:interface_inventory")
    for item in spec.get("files",[]):
        path=Path(item["path"]).resolve()
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
            raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:file_sha")
        imported.append({"path":str(path),"sha256":item["sha256"]})
    aggregates={}; inventories=None
    for interface,item in spec.get("interfaces",{}).items():
        source=Path(item["rows_path"]); path=source.resolve()
        if source.is_symlink() or path.parent!=bundle_path.resolve().parent: raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:rows_path_escape")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]: raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:rows_sha")
        raw=json.loads(path.read_text()); rows=raw if isinstance(raw,list) else raw["rows"]
        keys=[str(row["stable_key"]) for row in rows]
        if len(keys)!=128 or len(set(keys))!=128: raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:stable_ids")
        if inventories is None: inventories=set(keys)
        elif inventories!=set(keys): raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:stable_id_join")
        scored=[score_terminal_output(row["output"],row["ground_truth"]) for row in rows]
        aggregates[interface]=summarize_fixed_s128(scored)
        expected=item["expected_aggregate"]
        if any(abs(aggregates[interface][k]-expected[k])>1e-12 for k in expected): raise SystemExit("ORIGINAL_BASELINE_PROTOCOL_MISMATCH:aggregate")
        imported.append({"path":str(path),"sha256":item["sha256"]})
    out={"status":"PASS","decision":"ORIGINAL_BASELINE_IMPORT_PASS","git_commit":head,"authority_sha256":audit["authority_sha256"],"source_bundle_sha256":a.bundle_sha256,"imported_at_unix":int(time.time()),"files":imported,"aggregates":aggregates,"source_bundle":str(bundle_path.resolve())}
    raw=json.dumps(out,sort_keys=True,separators=(",",":")); out["report_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
if __name__=="__main__": main()
