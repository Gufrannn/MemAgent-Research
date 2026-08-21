#!/usr/bin/env python3
"""Authenticate a read-only Original evidence bundle and recompute S128 metrics."""
from __future__ import annotations
import argparse, json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.cosi import canonical_sha256, sha256_file
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_fixed_s128

INTERFACES = ("I", "Original5", "Original10", "Original15", "Original20", "Original25")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--bundle-index", required=True); p.add_argument("--output", required=True)
    a=p.parse_args(); index_path=Path(a.bundle_index).resolve(); index=json.loads(index_path.read_text())
    if set(index) != {"schema","source_commit","eval_manifest_hash","files","interfaces","index_sha256"}:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: index fields drifted")
    unsigned={k:v for k,v in index.items() if k!="index_sha256"}
    if index["schema"]!="memagent.original-s128.readonly-bundle.v1" or index["index_sha256"]!=canonical_sha256(unsigned):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: index authentication failed")
    if re.fullmatch(r"[0-9a-f]{40}",str(index["source_commit"])) is None:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: source commit is not exact")
    if set(index["interfaces"]) != set(INTERFACES):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: six interfaces required")
    root=index_path.parent
    if os.access(root,os.W_OK):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: evidence bundle is not read-only")
    inventory=[]
    for item in index["files"]:
        path=(root/item["path"]).resolve()
        if root not in path.parents or not path.is_file() or sha256_file(path)!=item["sha256"] or path.stat().st_size!=item["size"]:
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: source file mismatch")
        inventory.append(dict(item))
    aggregates={}; stable_inventory=None
    declared_paths={item["path"] for item in index["files"]}
    for name in INTERFACES:
        spec=index["interfaces"][name]; path=(root/spec["predictions_path"]).resolve()
        if spec["predictions_path"] not in declared_paths:
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: prediction absent from file inventory")
        if sha256_file(path)!=spec["predictions_sha256"]: raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: prediction hash")
        raw=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(raw)!=128 or any(set(row)!={"stable_id","terminal_output","ground_truth"} for row in raw):
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: prediction rows")
        ids=[str(row["stable_id"]) for row in raw]
        if len(set(ids))!=128 or (stable_inventory is not None and ids!=stable_inventory):
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: stable-ID join/order")
        stable_inventory=ids
        summary=summarize_fixed_s128([score_terminal_output(row["terminal_output"],row["ground_truth"]) for row in raw])
        if canonical_sha256(summary)!=spec["expected_aggregate_sha256"]:
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: independently recomputed aggregate")
        aggregates[name]=summary
    report={"schema":"memagent.cosi.baseline-import.v1","status":"PASS","decision":"COSI_BASELINE_IMPORT_PASS",
            "source_root":str(root),"source_read_only":True,"source_commit":index["source_commit"],
            "eval_manifest_hash":index["eval_manifest_hash"],"files":inventory,"aggregates":aggregates,
            "stable_inventory_sha256":canonical_sha256(stable_inventory),"imported_at":datetime.now(timezone.utc).isoformat()}
    report["report_sha256"]=canonical_sha256(report)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f: json.dump(report,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps(report,sort_keys=True))

if __name__=="__main__": main()
