#!/usr/bin/env python3
"""Read-only content/root overlap audit for PRD training and fixed S128."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
from recurrent.research.stable_eval_identity import canonical_sha256, sha256_text
from tools.h20.preflight_qwen25_7b_stable_i4x2 import _load_parquet_rows, _mapping, _question_text

VAL_SHA="54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6"
STABLE_SHA="6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411"
def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(row:dict)->tuple[str,str,str]:
    reward=_mapping(row.get("reward_model"),"reward_model")
    return sha256_text(_question_text(row.get("prompt"))),sha256_text(str(row.get("context"))),canonical_sha256(reward.get("ground_truth"))
def root_id(row:dict)->str:
    value=_mapping(row.get("extra_info"),"extra_info").get("index")
    if isinstance(value,bool) or not isinstance(value,int): raise ValueError(f"extra_info.index must be an integer, got {value!r}")
    return str(value)
def validated_root_ids(rows:list)->set[str]:
    values=[root_id(dict(row)) for row in rows]
    if len(set(values))!=len(values): raise ValueError("training root IDs are not unique")
    return set(values)
def compute_intersections(train_rows:list, frozen:list)->dict[str,int]:
    train_roots=validated_root_ids(train_rows)
    train_ids={identity(dict(row)) for row in train_rows}
    s128_ids={(row["source_question_hash"],row["source_context_hash"],row["ground_truth_hash"]) for row in frozen}
    s128_roots={str(row["example_id"]) for row in frozen}
    return {"train_pool_and_s128_content":len(train_ids&s128_ids),
            "train_pool_and_s128_root":len(train_roots&s128_roots),
            "critic_fit_and_s128":0,"selection_and_s128":128}
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--train",type=Path,required=True); p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--stable-resolved",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); failures=[]
    if a.train.resolve()!=Path("/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_train_32k.parquet"): failures.append("train path mismatch")
    if sha(a.validation)!=VAL_SHA: failures.append("validation SHA mismatch")
    if sha(a.stable_resolved)!=STABLE_SHA: failures.append("stable resolved SHA mismatch")
    train_rows=_load_parquet_rows(a.train); stable=json.loads(a.stable_resolved.read_text()); frozen=stable["identity_payload"]["rows"]
    try: intersections=compute_intersections(train_rows,frozen)
    except ValueError as exc: failures.append(str(exc)); intersections={"train_pool_and_s128_content":-1,"train_pool_and_s128_root":-1,"critic_fit_and_s128":0,"selection_and_s128":128}
    if intersections["train_pool_and_s128_content"] or intersections["train_pool_and_s128_root"]: failures.append("training pool intersects fixed S128")
    report={"schema_version":1,"status":"PASS" if not failures else "FAIL","decision":"PRD_DATA_OVERLAP_PASS" if not failures else "PRD_DATA_OVERLAP_NO_GO",
        "sources":{"train":{"path":str(a.train.resolve()),"sha256":sha(a.train),"rows":len(train_rows)},
        "fixed_s128_validation":{"path":str(a.validation.resolve()),"sha256":sha(a.validation),"resolved":str(a.stable_resolved.resolve()),"resolved_sha256":sha(a.stable_resolved),"rows":len(frozen)}},
        "identity":"tuple(sha256(question),sha256(context),canonical_sha256(ground_truth)); root=extra_info.index/example_id",
        "intersections":intersections,
        "fit_semantics":{"actor":"first 25 no-shuffle groups from train after production filtering; exact consumed identities require execution ledger",
        "prior":"same actor training trajectories; no S128 unless train overlap fails","critic":"none (GRPO, grpo_use_adv=False)",
        "selection":"all 128 fixed rows have already been used for certified Original curve and route development; conservatively counted as adaptive"},
        "adaptive_status":"FIXED_S128_IS_DEVELOPMENT_BENCHMARK_NOT_BLIND_FINAL_TEST",
        "producer":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__).resolve())},
        "evidence":{"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},"failures":failures}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as stream: json.dump(report,stream,indent=2,sort_keys=True); stream.write("\n")
    return 0 if not failures else 4
if __name__=="__main__": raise SystemExit(main())
