#!/usr/bin/env python3
"""Bind strict validation generations to one authenticated PRD checkpoint."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from recurrent.research.stable_eval_identity import canonical_sha256
from tools.h20.preflight_qwen25_7b_stable_i4x2 import _load_parquet_rows

STABLE_SHA="6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411"
DATA_SHA="54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6"
def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->int:
    p=argparse.ArgumentParser()
    for name in ("input","output","stable-resolved","validation-parquet","checkpoint-metadata","weight-sync-receipt","run-id","git-commit","frontier-id","global-step"): p.add_argument("--"+name,required=True)
    a=p.parse_args(); source=Path(a.input); output=Path(a.output); stable_path=Path(a.stable_resolved); data=Path(a.validation_parquet); metadata=Path(a.checkpoint_metadata)
    if sha(stable_path)!=STABLE_SHA or sha(data)!=DATA_SHA: raise SystemExit("PRD_NO_GO: frozen S128 identity/data SHA mismatch")
    stable=json.loads(stable_path.read_text()); rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(rows)!=128: raise SystemExit("PRD_NO_GO: strict producer did not generate exactly 128 rows")
    raw=_load_parquet_rows(data); identities=stable["identity_payload"]["rows"]
    if [int(row["source_order_index"]) for row in identities]!=list(range(128)): raise SystemExit("PRD_NO_GO: stable order is not 0..127")
    receipt=Path(a.weight_sync_receipt); receipt_payload=json.loads(receipt.read_text())
    if receipt_payload.get("status")!="PASS" or receipt_payload.get("decision")!="PRD_S128_WEIGHT_SYNC_PASS" or int(receipt_payload.get("global_step",-1))!=int(a.global_step): raise SystemExit("PRD_NO_GO: invalid S128 weight-sync receipt")
    bound=[]; step=int(a.global_step); metadata_sha=sha(metadata); receipt_sha=sha(receipt)
    for order,(generation,identity) in enumerate(zip(rows,identities)):
        reward=raw[int(identity["raw_row_position"])]["reward_model"]; reward=json.loads(reward) if isinstance(reward,str) else reward
        if canonical_sha256(reward["ground_truth"])!=identity["ground_truth_hash"]: raise SystemExit(f"PRD_NO_GO: ground truth drift row {order}")
        if int(generation.get("step",-1))!=step: raise SystemExit(f"PRD_NO_GO: generation checkpoint step drift row {order}")
        bound.append({"stable_key":json.dumps((stable["eval_manifest_hash"],identity["example_id"],0),separators=(",",":")),
            "terminal_output":generation["output"],"ground_truth":reward["ground_truth"],"source_order_index":order,
            "run_id":a.run_id,"git_commit":a.git_commit,"frontier_id":a.frontier_id,"global_step":step,
            "checkpoint_metadata_sha256":metadata_sha})
        bound[-1]["weight_sync_receipt_sha256"]=receipt_sha
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        for row in bound: stream.write(json.dumps(row,sort_keys=True)+"\n")
    return 0
if __name__=="__main__": raise SystemExit(main())
