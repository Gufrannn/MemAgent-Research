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
from recurrent.research.stable_eval_identity import validate_resolved_manifest
from tools.h20.preflight_qwen25_7b_stable_i4x2 import _load_parquet_rows

INTERFACES = ("I", "Original5", "Original10", "Original15", "Original20", "Original25")
SOURCE_COMMIT = "fbb9bad4a4facad6a5bfc73d74186eb58cb5fe0e"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--bundle-index", required=True); p.add_argument("--output", required=True)
    p.add_argument("--expected-bundle-index-sha256", required=True)
    p.add_argument("--expected-eval-manifest-sha256", required=True)
    p.add_argument("--s128-resolved-manifest", required=True)
    p.add_argument("--expected-s128-resolved-manifest-sha256", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--expected-validation-sha256", required=True)
    a=p.parse_args(); index_path=Path(a.bundle_index).resolve(); index=json.loads(index_path.read_text())
    if re.fullmatch(r"[0-9a-f]{64}", a.expected_bundle_index_sha256) is None \
            or sha256_file(index_path) != a.expected_bundle_index_sha256:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: external bundle-index SHA")
    if set(index) != {"schema","source_commit","eval_manifest_hash","files","interfaces","index_sha256"}:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: index fields drifted")
    unsigned={k:v for k,v in index.items() if k!="index_sha256"}
    if index["schema"]!="memagent.original-s128.readonly-bundle.v1" or index["index_sha256"]!=canonical_sha256(unsigned):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: index authentication failed")
    if index["source_commit"] != SOURCE_COMMIT:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: source commit is not the audited Original commit")
    if re.fullmatch(r"[0-9a-f]{64}", a.expected_eval_manifest_sha256) is None \
            or index["eval_manifest_hash"] != a.expected_eval_manifest_sha256:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: frozen eval-manifest hash")
    if set(index["interfaces"]) != set(INTERFACES):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: six interfaces required")
    root=index_path.parent
    if os.access(root,os.W_OK) or os.access(index_path,os.W_OK):
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: evidence bundle is not read-only")
    resolved_path=Path(a.s128_resolved_manifest).resolve()
    validation_path=Path(a.validation).resolve()
    if re.fullmatch(r"[0-9a-f]{64}", a.expected_s128_resolved_manifest_sha256) is None \
            or sha256_file(resolved_path) != a.expected_s128_resolved_manifest_sha256 \
            or re.fullmatch(r"[0-9a-f]{64}", a.expected_validation_sha256) is None \
            or sha256_file(validation_path) != a.expected_validation_sha256:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: authenticated S128 source binding")
    resolved=validate_resolved_manifest(json.loads(resolved_path.read_text()))
    if resolved["eval_manifest_hash"] != a.expected_eval_manifest_sha256 \
            or len(resolved["identity_payload"]["rows"]) != 128:
        raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: authenticated S128 identity")
    expected_ids=[str(row["example_id"]) for row in resolved["identity_payload"]["rows"]]
    parquet_ground_truth={}
    for row in _load_parquet_rows(validation_path):
        extra=row.get("extra_info"); reward=row.get("reward_model")
        if isinstance(extra,str): extra=json.loads(extra)
        if isinstance(reward,str): reward=json.loads(reward)
        if not isinstance(extra,dict) or not isinstance(reward,dict) or "index" not in extra \
                or "ground_truth" not in reward:
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: validation ground-truth schema")
        identity=str(int(extra["index"]))
        if identity in parquet_ground_truth:
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: duplicate validation identity")
        parquet_ground_truth[identity]=reward["ground_truth"]
    for row in resolved["identity_payload"]["rows"]:
        stable_id=str(row["example_id"])
        if stable_id not in parquet_ground_truth \
                or canonical_sha256(parquet_ground_truth[stable_id]) != row["ground_truth_hash"]:
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: ground-truth identity/hash join")
    inventory=[]
    for item in index["files"]:
        path=(root/item["path"]).resolve()
        if root not in path.parents or not path.is_file() or os.access(path,os.W_OK) \
                or sha256_file(path)!=item["sha256"] or path.stat().st_size!=item["size"]:
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
        if len(set(ids))!=128 or ids != expected_ids \
                or (stable_inventory is not None and ids!=stable_inventory):
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: stable-ID join/order")
        stable_inventory=ids
        if any(row["ground_truth"] != parquet_ground_truth[str(row["stable_id"])] for row in raw):
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: embedded ground truth drift")
        summary=summarize_fixed_s128([
            score_terminal_output(row["terminal_output"],parquet_ground_truth[str(row["stable_id"])])
            for row in raw
        ])
        if canonical_sha256(summary)!=spec["expected_aggregate_sha256"]:
            raise ValueError("ORIGINAL_BASELINE_PROTOCOL_MISMATCH: independently recomputed aggregate")
        aggregates[name]=summary
    report={"schema":"memagent.cosi.baseline-import.v1","status":"PASS","decision":"COSI_BASELINE_IMPORT_PASS",
            "source_root":str(root),"source_read_only":True,"source_commit":index["source_commit"],
            "eval_manifest_hash":index["eval_manifest_hash"],"files":inventory,"aggregates":aggregates,
            "bundle_index_sha256":sha256_file(index_path),
            "s128_resolved_manifest_sha256":sha256_file(resolved_path),
            "validation_sha256":sha256_file(validation_path),
            "ground_truth_source":"authenticated validation parquet joined by stable example_id",
            "stable_inventory_sha256":canonical_sha256(stable_inventory),"imported_at":datetime.now(timezone.utc).isoformat()}
    report["report_sha256"]=canonical_sha256(report)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f: json.dump(report,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps(report,sort_keys=True))

if __name__=="__main__": main()
