#!/usr/bin/env python3
"""Frozen Original actual-loss feasibility; never synthesizes missing evidence."""
import argparse, hashlib, json, math, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.h20.audit_rwwpo_actual_loss import audit


def main():
    p=argparse.ArgumentParser(); p.add_argument("--original-ledger", action="append", required=True); p.add_argument("--output", required=True); p.add_argument("--expected-commit", required=True); a=p.parse_args()
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if head!=a.expected_commit: raise SystemExit("RWWPO_E1_NO_GO: commit mismatch")
    for path in a.original_ledger:
        if not Path(path).is_file(): raise SystemExit("PENDING_ACTUAL_LOSS_LEDGER: missing frozen Original actual-loss evidence")
    base=audit(a.original_ledger, require_method=True)
    if base["modes"] != ["original_collection"]:
        raise SystemExit("RWWPO_E1_NO_GO: evidence is not frozen Original collection mode")
    rows=[]
    for source in a.original_ledger:
        rows.extend(json.loads(line) for line in Path(source).read_text().splitlines() if line.strip())
    observations=[]
    for row in rows:
        writer_tokens=sum(int(bool(v)) for token_row in row["writer_mask"] for v in token_row)
        local=[]
        for old,cur,mask in zip(row["old_log_prob"],row["current_log_prob"],row["writer_mask"]):
            local.extend(c-o for o,c,m in zip(old,cur,mask) if bool(m))
        for stat in row["prefix_stats"]:
            observations.append((writer_tokens, stat["ess_fraction"], max(map(abs,local))))
    collapse=base["min_prefix_ess"] < 0.95
    lengths={x[0] for x in observations}
    same_length_variation=any(max(v[1] for v in observations if v[0]==length)-min(v[1] for v in observations if v[0]==length)>1e-6 for length in lengths)
    local_not_sufficient=any(ess < .95 and max_abs <= math.log(1.2) for _,ess,max_abs in observations)
    status="PASS" if collapse and len(lengths)>1 and same_length_variation and local_not_sufficient else "FAIL"
    report={"status":status,"decision":"RWWPO_E1_PASS" if status=="PASS" else "RWWPO_E1_NO_GO",
            "git_commit":head,"source_ledgers":[{"path":str(Path(x).resolve()),"sha256":hashlib.sha256(Path(x).read_bytes()).hexdigest()} for x in a.original_ledger],
            "record_count":base["record_count"],"min_prefix_ess":base["min_prefix_ess"],"prefix_collapse_observed":collapse,
            "distinct_writer_lengths":len(lengths),"same_length_ess_variation":same_length_variation,
            "token_clip_does_not_exclude_prefix_collapse":local_not_sufficient}
    raw=json.dumps(report,sort_keys=True,separators=(",",":")); report["report_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    raise SystemExit(0 if status=="PASS" else 1)
if __name__=="__main__": main()
