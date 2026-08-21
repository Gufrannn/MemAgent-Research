#!/usr/bin/env python3
"""Read-only, per-file-SHA import of the certified Original S128 curve."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_fixed_s128

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--inventory",required=True); p.add_argument("--output",required=True); a=p.parse_args()
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
