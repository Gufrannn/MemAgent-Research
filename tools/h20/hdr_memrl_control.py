#!/usr/bin/env python3
"""Single fail-closed control surface for HDR preflight, gates, and audits."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.hdr_memrl import (HDRContractError, HorizonReceipt,
    aggregate_predictions, build_horizon_receipt, evaluate_horizons, prediction_metrics,
    stable_root_id, validate_evidence_equated, write_json)

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

def load(p): return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def append(path, rec):
    from recurrent.research.gate_a_execution import append_jsonl
    append_jsonl(path,rec)
def git(*a): return subprocess.check_output(["git","-C",str(ROOT),*a],text=True).strip()

def manifest_checked(path):
    m=load(path)
    if m.get("branch") != "h20/qwen25-7b-hdr-memrl-t25-frozen-20260822": raise HDRContractError("wrong frozen branch")
    if not HEX40.fullmatch(os.environ.get("MEMAGENT_HDR_EXPECTED_COMMIT","")): raise HDRContractError("exact expected commit required")
    if git("rev-parse","HEAD") != os.environ["MEMAGENT_HDR_EXPECTED_COMMIT"]: raise HDRContractError("HEAD/expected commit mismatch")
    if git("branch","--show-current") != m["branch"]: raise HDRContractError("wrong checkout branch")
    if git("status","--porcelain"): raise HDRContractError("dirty checkout")
    pair=os.environ.get("GPU_PAIR","")
    if not re.fullmatch(r"\d+,\d+",pair): raise HDRContractError("GPU_PAIR must be explicit a,b")
    g=list(map(int,pair.split(",")))
    if g != sorted(set(g)) or len(g)!=2: raise HDRContractError("GPU_PAIR must be two distinct canonical ascending IDs")
    return m,g

def preflight(a):
    m,g=manifest_checked(a.manifest)
    accepted=load(a.accepted_manifest)
    if digest(a.accepted_manifest)!=m["accepted_original_resolved_manifest_sha256"]: raise HDRContractError("accepted Original manifest SHA drift")
    training=accepted["training"]
    exact={"train_batch_size":4,"rollout_n":2,"ppo_mini_batch_size":4,"seed":2026,"trajectory_seed_mode":"independent","chunk_size":5000,"max_chunks":8,"actor_learning_rate":0.000001,"kl_loss_coefficient":0.001}
    for key,value in exact.items():
        if training.get(key)!=value: raise HDRContractError(f"accepted Original field drift: training.{key}")
    if int(training["train_batch_size"])*int(training["rollout_n"]) != m["budget"]["trajectories_per_update"]: raise HDRContractError("trajectory budget mismatch")
    if accepted["model"]["revision"] != m["model_revision"] or accepted["data"]["train_sha256"] != m["train_sha256"]: raise HDRContractError("base/data drift")
    if m["training"]["source"] != "fresh_base" or m["training"]["first_method_update"] != 1: raise HDRContractError("warm-start/method activation violation")
    if accepted["backend"]!={"rollout":"vllm","evaluation":"vllm","allow_hf_fallback":False,"reward_manager":"naive"}: raise HDRContractError("backend/reward drift")
    resolved=copy.deepcopy(accepted)
    resolved["hdr_method"]={"enabled_from_update":1,"horizons":m["horizons"],"dro":m["dro"],"scheduler_seed":2026,"stable_root_identity":True}
    resolved["training"]["source_step"]=0; resolved["training"]["first_update_step"]=1; resolved["training"]["target_step"]=25; resolved["training"]["resume_mode"]="disable_then_same_run_anchor_resume"
    variant="uniform" if m.get("variant")=="uniform_multi_horizon_erm" else "dro"
    resolved["experiment_name"]=f"qwen25_7b_hdr_memrl_{variant}_seed2026_{os.environ.get('HDR_RUN_ID','UNBOUND')}"
    resolved_path=str(Path(a.output).with_name("p0_resolved_method_manifest.json")); write_json(resolved_path,resolved)
    cert={"status":"PASS","decision":"HDR_P0_PASS","git_commit":git("rev-parse","HEAD"),"gpu_pair":g,
          "manifest_sha256":digest(a.manifest),"accepted_manifest_sha256":digest(a.accepted_manifest),"resolved_method_manifest":resolved_path,"resolved_method_manifest_sha256":digest(resolved_path),"whitelist_differences":["hdr_method","training.source_step","training.first_update_step","training.target_step","training.resume_mode","experiment_name"]}
    write_json(a.output,cert); append(a.ledger,{"record_type":"p0",**cert}); print(json.dumps(cert,sort_keys=True))

def e0(a):
    m=load(a.manifest); rows=load(a.receipts); hs=m["horizons"]["train"]
    rs=[HorizonReceipt(str(x["root_id"]),int(x["horizon"]),str(x["terminal_query_sha256"]),
        str(x["evidence_sha256"]),int(x["evidence_token_count"]),tuple(map(tuple,x["chunk_bounds"])),tuple(x["chunk_sha256"]),tuple(tuple(map(int,c)) for c in x["chunks"])) for x in rows]
    report=validate_evidence_equated(rs,hs)
    # Recompute every receipt from the frozen source parquet and tokenizer;
    # internal hash consistency alone is not evidence authority.
    authority=load(a.accepted_manifest)
    if digest(a.source_parquet)!=authority["data"]["validation_sha256"]: raise HDRContractError("E0 source parquet authority mismatch")
    model_root=Path(a.tokenizer_root)
    for item in authority["model"]["files"]:
        if item["path"] in {"tokenizer.json","tokenizer_config.json","vocab.json","merges.txt"}:
            p=model_root/item["path"]
            if not p.is_file() or digest(p)!=item["sha256"]: raise HDRContractError(f"E0 tokenizer inventory drift: {p}")
    import pandas as pd
    from transformers import AutoTokenizer
    source=pd.read_parquet(a.source_parquet); suite=pd.read_parquet(a.suite_parquet)
    tok=AutoTokenizer.from_pretrained(a.tokenizer_root,local_files_only=True)
    source_by_index={int(row.get("extra_info",{}).get("index",pos)):row for pos,row in source.iterrows()}
    recomputed=[]
    for pos,row in suite.iterrows():
        prompt=row["prompt"]; query=prompt[0]["content"] if isinstance(prompt,(list,tuple)) else prompt.tolist()[0]["content"]
        idx=int(row.get("extra_info",{}).get("index",pos)); original=source_by_index.get(idx)
        if original is None or str(original["context"])!=str(row["context"]): raise HDRContractError("suite row not bound to source context/index")
        op=original["prompt"]; oq=op[0]["content"] if isinstance(op,(list,tuple)) else op.tolist()[0]["content"]
        if oq!=query: raise HDRContractError("suite terminal query differs from source")
        tokens=tok.encode(str(row["context"]),add_special_tokens=False)
        if len(tokens)>40000: raise HDRContractError("E0 refuses source truncation")
        rid=stable_root_id(dataset_sha256=authority["data"]["validation_sha256"],source_index=idx,query=query)
        if str(row.get("stable_root_id_receipt"))!=rid: raise HDRContractError("suite stable root mismatch")
        recomputed.append(build_horizon_receipt(rid,query,tokens,int(row["horizon_id"])))
    authoritative=validate_evidence_equated(recomputed,hs)
    if authoritative["suite_sha256"]!=report["suite_sha256"]: raise HDRContractError("E0 receipt/source recomputation mismatch")
    train=set(load(a.train_roots)); ev=set(load(a.eval_roots))
    if digest(a.train_source_parquet)!=authority["data"]["train_sha256"]: raise HDRContractError("E0 train source authority mismatch")
    train_df=pd.read_parquet(a.train_source_parquet); authoritative_train=set()
    for pos,row in train_df.iterrows():
        prompt=row["prompt"]; query=prompt[0]["content"] if isinstance(prompt,(list,tuple)) else prompt.tolist()[0]["content"]
        idx=int(row.get("extra_info",{}).get("index",pos)); authoritative_train.add(stable_root_id(dataset_sha256=authority["data"]["train_sha256"],source_index=idx,query=query))
    if train!=authoritative_train: raise HDRContractError("train root authority set mismatch")
    if ev!={r.root_id for r in recomputed}: raise HDRContractError("eval root authority set mismatch")
    overlap=train&ev
    if overlap: raise HDRContractError(f"root leakage: {len(overlap)}")
    report.update(decision="HDR_E0_PASS",root_split_overlap=0)
    write_json(a.output,report); append(a.ledger,{"record_type":"e0",**report}); print(json.dumps(report,sort_keys=True))

def e1(a):
    manifest=load(a.manifest); hs=set(manifest["horizons"]["train"]); roots={}
    receipt_rows=load(a.receipts)
    receipts=[HorizonReceipt(str(x["root_id"]),int(x["horizon"]),str(x["terminal_query_sha256"]),str(x["evidence_sha256"]),int(x["evidence_token_count"]),tuple(map(tuple,x["chunk_bounds"])),tuple(x["chunk_sha256"]),tuple(tuple(map(int,c)) for c in x["chunks"])) for x in receipt_rows]
    closure=validate_evidence_equated(receipts,sorted(hs)); receipt_map={(r.root_id,r.horizon):r.as_dict() for r in receipts}
    if not HEX40.fullmatch(a.git_commit) or a.git_commit!=git("rev-parse","HEAD"): raise HDRContractError("E1 git commit binding mismatch")
    rows=load(a.rows); seen=set()
    for r in rows:
        key=(str(r["root_id"]),int(r["horizon"]))
        if key in seen or key not in receipt_map: raise HDRContractError("duplicate or foreign E1 root×horizon")
        seen.add(key)
        if int(r["horizon"]) not in hs or r.get("cap_hit") or r.get("truncated") or not r.get("evidence_equated"): raise HDRContractError("invalid E1 row")
        if str(Path(r.get("model_path","")).resolve()) != str(Path(a.model_path).resolve()) or int(r.get("seed",-1))!=a.seed: raise HDRContractError("E1 model/seed binding mismatch")
        if r.get("receipt") != receipt_map[key]: raise HDRContractError("E1 receipt binding mismatch")
        calc=prediction_metrics(str(r["prediction"]),str(r["gold"]))
        for metric in calc:
            if abs(float(r.get(metric,float("nan")))-calc[metric])>1e-12: raise HDRContractError("E1 self-reported metric mismatch")
        roots.setdefault(str(r["root_id"]),[]).append(r)
    if seen != set(receipt_map) or not roots or any({int(x["horizon"]) for x in rs}!=hs for rs in roots.values()): raise HDRContractError("E1 paired closure failure")
    ranges=[max(float(x["token_f1"]) for x in rs)-min(float(x["token_f1"]) for x in rs) for rs in roots.values()]
    # Token control: compare horizon coefficient after demeaning within root and regressing tokens.
    xs=[]; ys=[]
    for rs in roots.values():
        mf=sum(float(x["token_f1"]) for x in rs)/len(rs); mt=sum(float(x["total_input_tokens"]) for x in rs)/len(rs)
        for x in rs: xs.append(float(x["total_input_tokens"])-mt); ys.append(float(x["token_f1"])-mf)
    den=sum(x*x for x in xs); beta=sum(x*y for x,y in zip(xs,ys))/den if den else 0.0
    residual=[]
    for rs in roots.values():
        vals=[float(x["token_f1"])-beta*float(x["total_input_tokens"]) for x in rs]
        residual.append(max(vals)-min(vals))
    # Root-clustered randomization test: permute horizon labels within roots,
    # preserving token/score pairs and using a pre-registered deterministic seed.
    observed=sum(residual)/len(residual); rng=random.Random(20260822); null=[]
    root_rows=[list(rs) for rs in roots.values()]
    for _ in range(2000):
        by_h={h:[] for h in hs}
        for rs in root_rows:
            labels=[int(x["horizon"]) for x in rs]; rng.shuffle(labels)
            for x,h in zip(rs,labels): by_h[h].append(float(x["token_f1"])-beta*float(x["total_input_tokens"]))
        null.append(max(sum(v)/len(v) for v in by_h.values())-min(sum(v)/len(v) for v in by_h.values()))
    horizon_effect=max(sum(float(x["token_f1"])-beta*float(x["total_input_tokens"]) for x in rows if int(x["horizon"])==h)/sum(1 for x in rows if int(x["horizon"])==h) for h in hs)-min(sum(float(x["token_f1"])-beta*float(x["total_input_tokens"]) for x in rows if int(x["horizon"])==h)/sum(1 for x in rows if int(x["horizon"])==h) for h in hs)
    p_value=(1+sum(x>=horizon_effect for x in null))/(len(null)+1)
    cfg=manifest["e1_gate"]
    mean_range=sum(ranges)/len(ranges); mean_res=sum(residual)/len(residual)
    go=mean_range>=float(cfg["min_mean_f1_range"]) and mean_res>=float(cfg["min_token_controlled_mean_f1_range"]) and p_value<=float(cfg.get("max_permutation_p",.05))
    report={"status":"PASS" if go else "NO-GO","decision":"HDR_E1_PASS" if go else "NO-GO_E1",
            "root_count":len(roots),"mean_f1_range":mean_range,"token_slope":beta,
            "token_controlled_mean_f1_range":mean_res,"horizon_effect":horizon_effect,"permutation_p":p_value,
            "suite_sha256":closure["suite_sha256"],"model_path":str(Path(a.model_path).resolve()),"seed":a.seed,"git_commit":a.git_commit}
    write_json(a.output,report); append(a.ledger,{"record_type":"e1",**report}); print(json.dumps(report,sort_keys=True))
    if not go: raise SystemExit(42)

def baseline(a):
    """Read-only import from the certified curve; no standalone bundle exists."""
    from recurrent.research.gate_a_execution import validate_jsonl_chain
    from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_fixed_s128
    from recurrent.research.stable_eval_identity import canonical_sha256, stable_key
    expected_digests={"I":"fd4c6763c8d8a6caa0389082f1fa838dc510d872b99e6283c1483c4427336c64","Original5":"58b01ad5e523ee8853c05af691a659480d0d905d22f2c6ffb0590484c5a38a30d","Original10":"bc5c29e7e6f163828758cb68dca1237f9d970af24217f60e272ba2945017b4a4","Original15":"3e8ae48f4a092ec136c568397037b9f270532bb0ab92a6b976c4de66c2c02b2f","Original20":"8a831d5d96c4f963f53a6a8d2c01a6a1414724a8a189a05f5a89c68d56494cd8","Original25":"4db791e409edeb269b56b1633b07c272ef04abf8b15da5c479a1e7822a93b2d6"}
    curve=load(a.final_report); ledger_path=Path(a.curve_ledger); records=[json.loads(x) for x in ledger_path.read_text().splitlines() if x.strip()]
    failures=validate_jsonl_chain(records)
    if failures: raise HDRContractError(f"Original curve ledger invalid: {failures}")
    tail=records[-1] if records else {}
    report_path=Path(a.final_report).resolve()
    authenticated_path=tail.get("artifact") or tail.get("report")
    authenticated_sha=tail.get("artifact_sha256") or tail.get("report_sha256")
    if tail.get("record_type")!="audit_result" or tail.get("status")!="PASS" or Path(str(authenticated_path)).resolve()!=report_path or authenticated_sha!=digest(report_path): raise HDRContractError("curve ledger does not authenticate final report")
    if curve.get("status")!="PASS" or curve.get("decision")!="ORIGINAL_S128_CURVE_PASS": raise HDRContractError("Original curve final report is not PASS")
    evidence=curve.get("evidence",{}); interfaces=evidence.get("interfaces",{})
    if set(interfaces)!=set(expected_digests) or evidence.get("metric_rows_sha256")!=expected_digests: raise HDRContractError("Original curve canonical-row digest closure mismatch")
    if digest(a.stable_resolved)!="6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411": raise HDRContractError("stable-S128 resolved authority mismatch")
    stable=load(a.stable_resolved); identity=stable.get("identity_payload",{}).get("rows",[])
    if len(identity)!=128: raise HDRContractError("stable-S128 identity denominator mismatch")
    import pandas as pd
    raw=pd.read_parquet(a.validation_parquet); gt={}
    for row in identity:
        order=int(row["source_order_index"]); reward=raw.iloc[int(row["raw_row_position"])]["reward_model"]
        if isinstance(reward,str): reward=json.loads(reward)
        truth=reward["ground_truth"]
        if canonical_sha256(truth)!=row["ground_truth_hash"]: raise HDRContractError("stable-S128 ground truth drift")
        gt[order]=truth
    imported=[]; recomputed={}; metric_rows_out={}
    for anchor in expected_digests:
        info=interfaces[anchor]; root=Path(info["root"]); artifacts=info.get("artifacts",{})
        terminals=[rel for rel in artifacts if re.fullmatch(r"terminal/\d+\.jsonl",rel)]
        if len(terminals)!=1: raise HDRContractError(f"{anchor} certified terminal artifact missing")
        rel=terminals[0]; item=artifacts[rel]; p=(root/rel).resolve()
        if not p.is_file() or digest(p)!=item["sha256"] or p.stat().st_size!=item["size"]: raise HDRContractError(f"{anchor} terminal inventory drift")
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if len(rows)!=128: raise HDRContractError(f"{anchor} terminal denominator mismatch")
        metric=[]
        for row in rows:
            order=int(row["source_order_index"]); scored=score_terminal_output(row["output"],gt[order])
            metric.append({"stable_key":json.dumps(stable_key(row),separators=(",",":")),"source_order_index":order,"eval_manifest_hash":row["eval_manifest_hash"],"example_id":row["example_id"],"replica_id":row["replica_id"],"trajectory_seed":row["trajectory_seed"],"trajectory_id":row["trajectory_id"],**scored})
        if canonical_sha256(metric)!=expected_digests[anchor]: raise HDRContractError(f"{anchor} independently recomputed rows digest mismatch")
        summary=summarize_fixed_s128(metric)
        reported=info.get("metrics")
        if summary!=reported: raise HDRContractError(f"{anchor} independently recomputed aggregate mismatch")
        recomputed[anchor]={"count":128,"em":summary["normalized_exact_match"],"token_f1":summary["token_f1"],"format":summary["format_success"]}
        metric_rows_out[anchor]=metric
        imported.append({"path":str(p),"sha256":item["sha256"],"size":item["size"],"kind":"predictions","anchor":anchor})
    materialized=Path(a.materialized_rows); materialized.parent.mkdir(parents=True,exist_ok=True)
    if materialized.exists(): raise HDRContractError("refusing to overwrite materialized Original rows")
    write_json(materialized,metric_rows_out)
    report={"status":"PASS","decision":"ORIGINAL_BASELINE_IMPORT_PASS","files":imported,
            "recomputed_aggregates":recomputed,"canonical_metric_rows_sha256":expected_digests,
            "materialized_metric_rows":str(materialized.resolve()),"materialized_metric_rows_sha256":digest(materialized),
            "curve_final_report":str(report_path),"curve_final_report_sha256":digest(report_path),"curve_ledger_sha256":digest(ledger_path)}
    write_json(a.output,report); append(a.ledger,{"record_type":"baseline_import",**report}); print(json.dumps(report,sort_keys=True))

def ev(a):
    rows=load(a.rows)
    for row in rows:
        if "prediction" in row:
            row.update(prediction_metrics(str(row["prediction"]),str(row["gold"])))
    report=evaluate_horizons(rows,a.nominal,a.unseen)
    report["decision"]="HDR_HORIZON_EVAL_PASS"; write_json(a.output,report); print(json.dumps(report,sort_keys=True))

def train_health(a):
    """Cheap checkpoint/numerics/sync gate; never performs S128 generation."""
    from recurrent.research.gate_a_execution import validate_jsonl_chain
    step=Path(a.output_root)/f"global_step_{a.anchor}"
    binding=step/"hdr_checkpoint_binding.json"; state=step/"hdr_dro_state.json"
    if not step.is_dir() or not binding.is_file() or not state.is_file():
        raise HDRContractError(f"incomplete training checkpoint at T{a.anchor}")
    b=load(binding)
    if b.get("global_step")!=a.anchor or b.get("git_commit")!=git("rev-parse","HEAD"):
        raise HDRContractError("training checkpoint identity drift")
    for item in b.get("inventory",[]):
        p=step/item["path"]
        if not p.is_file() or digest(p)!=item["sha256"] or p.stat().st_size!=item["size"]:
            raise HDRContractError(f"checkpoint inventory mismatch: {item['path']}")
    s=load(state); weights=[float(x) for x in s.get("weights",[])]
    if not weights or any(not __import__("math").isfinite(x) or x<0 for x in weights) or abs(sum(weights)-1)>1e-9:
        raise HDRContractError("non-finite or invalid DRO state")
    sync=[json.loads(x) for x in Path(a.weight_sync_ledger).read_text().splitlines() if x.strip()]
    failures=validate_jsonl_chain(sync)
    if failures: raise HDRContractError(f"weight-sync ledger invalid: {failures}")
    relevant=[r for r in sync if int(r.get("global_step",-1))<=a.anchor]
    post={int(r["global_step"]) for r in relevant if r.get("record_type")=="weight_sync_summary" and r.get("sync_kind")=="post_actor_update"}
    if post!=set(range(1,a.anchor+1)) or any(r.get("status") not in (None,"PASS") for r in relevant):
        raise HDRContractError("missing or failed weight-sync evidence")
    report={"status":"PASS","decision":f"HDR_T{a.anchor}_TRAIN_HEALTH_PASS",
            "anchor":a.anchor,"git_commit":git("rev-parse","HEAD"),
            "checkpoint_binding_sha256":digest(binding),"dro_state_sha256":digest(state),
            "weight_sync_ledger_sha256":digest(a.weight_sync_ledger),
            "s128_evaluation_performed":False}
    write_json(a.output,report); append(a.ledger,{"record_type":"train_health",**report})
    print(json.dumps(report,sort_keys=True))

def final_audit(a):
    from recurrent.research.gate_a_execution import validate_jsonl_chain
    root=Path(a.run_root); output=Path(a.output_root); failures=[]
    records=[json.loads(x) for x in (output/"hdr_execution_ledger.jsonl").read_text().splitlines() if x.strip()]
    failures += validate_jsonl_chain(records)
    dro=[r for r in records if r.get("record_type")=="dro_update"]
    steps=[int(r["global_step"]) for r in dro]
    if steps!=list(range(1,26)): failures.append(f"DRO steps not exact 1..25: {steps}")
    for r in dro:
        if r.get("git_commit")!=git("rev-parse","HEAD") or r.get("seed")!=2026 or r.get("trajectory_budget")!=8: failures.append(f"invalid DRO identity/budget at {r.get('global_step')}")
        if set(map(int,r.get("counts",{})))!={8,12,16,32} or sum(map(int,r.get("counts",{}).values()))!=8: failures.append(f"invalid horizon counts at {r.get('global_step')}")
    for step in (5,10,15,20,25):
        p=output/f"global_step_{step}"/"hdr_dro_state.json"
        if not p.is_file(): failures.append(f"missing dual checkpoint {step}"); continue
        st=load(p)
        if abs(sum(map(float,st.get("weights",[])))-1)>1e-9: failures.append(f"invalid dual simplex {step}")
        bp=output/f"global_step_{step}"/"hdr_checkpoint_binding.json"
        if not bp.is_file(): failures.append(f"missing checkpoint binding {step}")
        else:
            b=load(bp)
            if b.get("global_step")!=step or b.get("git_commit")!=git("rev-parse","HEAD"): failures.append(f"checkpoint binding drift {step}")
            for item in b.get("inventory",[]):
                fp=output/f"global_step_{step}"/item["path"]
                if not fp.is_file() or digest(fp)!=item["sha256"] or fp.stat().st_size!=item["size"]: failures.append(f"checkpoint inventory tamper {step}/{item['path']}")
            checkpoint_records=[r for r in records if r.get("record_type")=="checkpoint" and r.get("global_step")==step]
            if len(checkpoint_records)!=1 or checkpoint_records[0].get("checkpoint_binding_sha256")!=digest(bp): failures.append(f"checkpoint ledger binding mismatch {step}")
    root_ledger=root/"hdr_execution_ledger.jsonl"
    if not root_ledger.is_file(): failures.append("missing root launch/gate ledger")
    else:
        rr=[json.loads(x) for x in root_ledger.read_text().splitlines() if x.strip()]; failures+=validate_jsonl_chain(rr)
        launches=[r for r in rr if r.get("record_type")=="launch"]
        if [r.get("target_step") for r in launches] != [25] or launches[0].get("source_step")!=0: failures.append("continuous fresh-T25 launch identity mismatch")
        if len({(r.get("git_commit"),r.get("gpu_pair"),r.get("run_id")) for r in launches})>1: failures.append("launch commit/GPU/run drift")
    sync_path=root/"hdr_weight_sync_ledger.jsonl"
    if not sync_path.is_file(): failures.append("missing distributed weight-sync ledger")
    else:
        sync=[json.loads(x) for x in sync_path.read_text().splitlines() if x.strip()]
        failures += validate_jsonl_chain(sync)
        got={int(r["global_step"]) for r in sync if r.get("record_type")=="weight_sync_summary" and r.get("sync_kind")=="post_actor_update"}
        if got!=set(range(1,26)): failures.append(f"weight-sync steps mismatch: {sorted(got)}")
    for name,decision in (("p0.json","HDR_P0_PASS"),("e0.json","HDR_E0_PASS"),("e1.json","HDR_E1_PASS"),("baseline_import.json","ORIGINAL_BASELINE_IMPORT_PASS"),("paper_review.json","PAPER_FRAMING_GO"),("t5_train_health.json","HDR_T5_TRAIN_HEALTH_PASS")):
        p=root/"certificates"/name
        if not p.is_file() or load(p).get("decision")!=decision: failures.append(f"invalid gate {name}")
    # Every locally produced certificate must be the unique payload committed
    # to the append-only root ledger; a free-standing PASS JSON is insufficient.
    baseline_path=root/"certificates/baseline_import.json"
    if root_ledger.is_file():
        type_for={"p0.json":"p0","e0.json":"e0","e1.json":"e1","baseline_import.json":"baseline_import","t5_train_health.json":"train_health"}
        for name,record_type in type_for.items():
            p=root/"certificates"/name
            if not p.is_file(): continue
            cert=load(p); hits=[r for r in rr if r.get("record_type")==record_type and all(r.get(k)==v for k,v in cert.items())]
            if len(hits)!=1: failures.append(f"certificate/root-ledger binding mismatch: {name}")
        review=root/"certificates/paper_review.json"; expected_review=os.environ.get("MEMAGENT_HDR_REVIEW_SHA256","")
        if not HEX64.fullmatch(expected_review) or (review.is_file() and digest(review)!=expected_review): failures.append("paper review external SHA authority mismatch")
        if baseline_path.is_file():
            baseline=load(baseline_path); materialized=Path(baseline.get("materialized_metric_rows",""))
            if set(baseline.get("canonical_metric_rows_sha256",{}))!={"I","Original5","Original10","Original15","Original20","Original25"}: failures.append("baseline canonical anchor closure mismatch")
            if not materialized.is_file() or digest(materialized)!=baseline.get("materialized_metric_rows_sha256"): failures.append("baseline materialized rows SHA mismatch")
    for step in (5,10,15,20,25):
        p=root/"certificates"/f"t{step}_health.json"
        if not p.is_file() or load(p).get("decision")!=f"HDR_T{step}_HEALTH_PASS": failures.append(f"missing anchor health {step}"); continue
        cert=load(p)
        if root_ledger.is_file():
            hits=[r for r in rr if r.get("record_type")=="audit" and r.get("anchor")==step and all(r.get(k)==v for k,v in cert.items())]
            if len(hits)!=1: failures.append(f"anchor health/root-ledger binding mismatch {step}")
        for field in ("checkpoint_binding_sha256","merge_receipt_sha256","method_s128_sha256","method_horizons_sha256","method_suite_sha256","method_receipts_sha256","stable_resolved_sha256","validation_parquet_sha256","baseline_import_sha256"):
            if not HEX64.fullmatch(str(cert.get(field,""))): failures.append(f"anchor health missing input SHA {step}/{field}")
        if cert.get("stable_resolved_sha256")!="6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411" or cert.get("validation_parquet_sha256")!="54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6": failures.append(f"anchor frozen S128 authority mismatch {step}")
        if baseline_path.is_file() and digest(baseline_path)!=cert.get("baseline_import_sha256"): failures.append(f"anchor baseline import SHA mismatch {step}")
        bp=output/f"global_step_{step}/hdr_checkpoint_binding.json"
        if bp.is_file() and digest(bp)!=cert.get("checkpoint_binding_sha256"): failures.append(f"anchor checkpoint/health SHA mismatch {step}")
        eval_root=root/"eval"
        for path,field in ((eval_root/f"t{step}_merge_receipt.json","merge_receipt_sha256"),(eval_root/f"t{step}_s128_nominal.json","method_s128_sha256"),(eval_root/f"t{step}_horizons.json","method_horizons_sha256")):
            if not path.is_file() or digest(path)!=cert.get(field): failures.append(f"anchor health input tamper {step}/{path.name}")
        suite_root=root/"eval"
        for path,field in ((suite_root/"fixed_s128_nominal_h8.parquet","method_suite_sha256"),(suite_root/"fixed_s128_nominal_receipts.json","method_receipts_sha256")):
            if not path.is_file() or digest(path)!=cert.get(field): failures.append(f"anchor frozen suite tamper {step}/{path.name}")
    report={"status":"FAIL" if failures else "PASS","decision":"HDR_FINAL_AUDIT_FAIL" if failures else "HDR_FINAL_AUDIT_PASS","failures":failures,"git_commit":git("rev-parse","HEAD")}
    write_json(a.report,report); print(json.dumps(report,sort_keys=True))
    if failures: raise SystemExit(3)

def health_gate(a):
    baseline=load(a.baseline_import)
    if baseline.get("decision")!="ORIGINAL_BASELINE_IMPORT_PASS": raise HDRContractError("uncertified baseline import")
    anchor=f"Original{a.anchor}"
    binding=load(a.checkpoint_binding)
    if binding.get("global_step")!=a.anchor or binding.get("git_commit")!=git("rev-parse","HEAD"): raise HDRContractError("health checkpoint binding mismatch")
    checkpoint_root=Path(a.checkpoint_binding).parent
    for item in binding.get("inventory",[]):
        p=checkpoint_root/item["path"]
        if not p.is_file() or digest(p)!=item["sha256"] or p.stat().st_size!=item["size"]: raise HDRContractError("health checkpoint inventory tamper")
    merge=load(a.merge_receipt)
    if merge.get("decision")!="HDR_MERGED_MODEL_BOUND" or merge.get("checkpoint_binding_sha256")!=digest(a.checkpoint_binding) or str(Path(merge.get("merged_model","")).resolve())!=str(Path(a.model_path).resolve()): raise HDRContractError("merged model/checkpoint binding mismatch")
    for item in merge.get("merged_inventory",[]):
        p=Path(a.model_path)/item["path"]
        if not p.is_file() or digest(p)!=item["sha256"] or p.stat().st_size!=item["size"]: raise HDRContractError("merged model inventory tamper")
    if anchor not in baseline.get("recomputed_aggregates",{}): raise HDRContractError("baseline anchor absent")
    method_rows=load(a.method_s128)
    if digest(a.stable_resolved)!="6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411": raise HDRContractError("health stable-S128 authority drift")
    stable=load(a.stable_resolved); identity=stable.get("identity_payload",{}).get("rows",[])
    if len(identity)!=128: raise HDRContractError("health stable-S128 denominator drift")
    import pandas as pd
    from recurrent.research.s128_hotpot_metrics import score_terminal_output
    from recurrent.research.stable_eval_identity import canonical_sha256
    raw=pd.read_parquet(a.validation_parquet); by_order={int(x["source_order_index"]):x for x in identity}
    suite=pd.read_parquet(a.method_suite); suite_sha=digest(a.method_suite)
    suite_by_order={int(r["source_order_index"]):r for _,r in suite.iterrows()}
    receipt_rows=load(a.method_receipts); receipt_by_key={(str(x["root_id"]),int(x["horizon"])):x for x in receipt_rows}
    if len(suite_by_order)!=128 or len(receipt_rows)!=128: raise HDRContractError("nominal suite/receipt fixed-S128 closure mismatch")
    for row in method_rows:
        order=int(row.get("source_order_index",-1)); ident=by_order.get(order); sr=suite_by_order.get(order)
        if ident is None or sr is None or int(row.get("raw_row_position",-1))!=int(ident["raw_row_position"]) or int(sr["raw_row_position"])!=int(ident["raw_row_position"]): raise HDRContractError("method Stable-S128 row identity mismatch")
        if row.get("identity_resolved_sha256")!=digest(a.stable_resolved) or row.get("suite_sha256")!=suite_sha or sr.get("identity_resolved_sha256")!=digest(a.stable_resolved): raise HDRContractError("method suite/identity SHA mismatch")
        reward=raw.iloc[int(ident["raw_row_position"])]["reward_model"]
        if isinstance(reward,str): reward=json.loads(reward)
        truth=reward["ground_truth"]
        if canonical_sha256(truth)!=ident["ground_truth_hash"] or row.get("gold")!=truth: raise HDRContractError("method fixed-S128 ground truth mismatch")
        key=(str(row["root_id"]),int(row["horizon"])); expected_receipt=receipt_by_key.get(key)
        if expected_receipt is None or row.get("receipt")!=expected_receipt: raise HDRContractError("method nominal receipt authority mismatch")
        scored=score_terminal_output(row["prediction"],truth)
        if any(abs(float(row[k])-float(scored[v]))>1e-12 for k,v in (("em","exact_match"),("token_f1","token_f1"),("format","format_success"))): raise HDRContractError("method metric recomputation mismatch")
    method=aggregate_predictions(method_rows)
    if method["count"]!=128: raise HDRContractError("fixed-S128 method evaluation must contain exactly 128 unique rows")
    materialized=Path(baseline.get("materialized_metric_rows",""))
    if not materialized.is_file() or digest(materialized)!=baseline.get("materialized_metric_rows_sha256"): raise HDRContractError("materialized Original rows authority mismatch")
    authority_rows=load(materialized).get(anchor,[])
    if len(authority_rows)!=128 or {int(x["source_order_index"]) for x in method_rows}!={int(x["source_order_index"]) for x in authority_rows}: raise HDRContractError("method/Original fixed-S128 identity join mismatch")
    if any(str(Path(x.get("model_path","")).resolve())!=str(Path(a.model_path).resolve()) or int(x.get("seed",-1))!=a.seed for x in method_rows): raise HDRContractError("method S128 model/seed binding mismatch")
    sreceipts=[]
    for x in method_rows:
        r=x.get("receipt",{}); sreceipts.append(HorizonReceipt(str(r["root_id"]),int(r["horizon"]),str(r["terminal_query_sha256"]),str(r["evidence_sha256"]),int(r["evidence_token_count"]),tuple(map(tuple,r["chunk_bounds"])),tuple(r["chunk_sha256"]),tuple(tuple(map(int,c)) for c in r["chunks"])))
    validate_evidence_equated(sreceipts,[a.nominal])
    original=baseline["recomputed_aggregates"][anchor]
    hrows=load(a.method_horizons)
    if any(str(Path(x.get("model_path","")).resolve())!=str(Path(a.model_path).resolve()) or int(x.get("seed",-1))!=a.seed for x in hrows): raise HDRContractError("method horizon model/seed binding mismatch")
    hrs=[]
    for x in hrows:
        r=x.get("receipt",{})
        hrs.append(HorizonReceipt(str(r["root_id"]),int(r["horizon"]),str(r["terminal_query_sha256"]),str(r["evidence_sha256"]),int(r["evidence_token_count"]),tuple(map(tuple,r["chunk_bounds"])),tuple(r["chunk_sha256"]),tuple(tuple(map(int,c)) for c in r["chunks"])))
    validate_evidence_equated(hrs,sorted({a.nominal,*a.unseen,8,12,16,32}))
    for row in hrows:
        if "prediction" in row: row.update(prediction_metrics(str(row["prediction"]),str(row["gold"])))
    heval=evaluate_horizons(hrows,a.nominal,a.unseen)
    method_by_order={int(x["source_order_index"]):x for x in method_rows}; original_by_order={int(x["source_order_index"]):x for x in authority_rows}
    paired={}
    for method_key,original_key,label in (("em","exact_match","em"),("token_f1","token_f1","token_f1"),("format","format_success","format")):
        deltas=[float(method_by_order[i][method_key])-float(original_by_order[i][original_key]) for i in range(128)]
        paired[label]={"mean_difference":sum(deltas)/128,"improved":sum(x>0 for x in deltas),"unchanged":sum(x==0 for x in deltas),"worsened":sum(x<0 for x in deltas)}
    failures=[]
    if float(method["token_f1"]) < float(original["token_f1"])-.02: failures.append("nominal_s128_noninferiority")
    uniform=None
    if a.variant=="dro" and a.anchor==25:
        if not a.uniform_horizons: raise HDRContractError("T25 requires uniform-ERM horizon baseline")
        uniform=load(a.uniform_horizons)
        if uniform.get("status")!="PASS" or uniform.get("decision")!="UNIFORM_T25_HEALTH_PASS" or uniform.get("git_commit")!=git("rev-parse","HEAD"): raise HDRContractError("uniform T25 health authority mismatch")
        uniform_eval=uniform.get("method_horizons",{})
        if float(heval["worst"]["token_f1"]) < float(uniform_eval["worst"]["token_f1"])+.02: failures.append("uniform_erm_worst_gain")
        if float(method["token_f1"]) < float(original["token_f1"])-.01: failures.append("t25_nominal_one_point_floor")
    prefix="HDR" if a.variant=="dro" else "UNIFORM"
    report={"status":"FAIL" if failures else "PASS","decision":f"{prefix}_T{a.anchor}_HEALTH_FAIL" if failures else f"{prefix}_T{a.anchor}_HEALTH_PASS","variant":a.variant,"anchor":a.anchor,"method_s128":method,"original_s128":original,"paired_descriptive_method_minus_original":paired,"method_horizons":heval,"uniform_horizons":uniform,"failures":failures,"git_commit":git("rev-parse","HEAD"),"checkpoint_binding_sha256":digest(a.checkpoint_binding),"merge_receipt_sha256":digest(a.merge_receipt),"method_s128_sha256":digest(a.method_s128),"method_horizons_sha256":digest(a.method_horizons),"method_suite_sha256":digest(a.method_suite),"method_receipts_sha256":digest(a.method_receipts),"stable_resolved_sha256":digest(a.stable_resolved),"validation_parquet_sha256":digest(a.validation_parquet),"baseline_import_sha256":digest(a.baseline_import),"model_path":str(Path(a.model_path).resolve()),"seed":a.seed}
    write_json(a.output,report); append(a.ledger,{"record_type":"audit",**report})
    if failures: raise SystemExit(4)

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True)
    q=s.add_parser("preflight"); q.add_argument("--manifest",required=True); q.add_argument("--accepted-manifest",required=True); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=preflight)
    q=s.add_parser("e0"); q.add_argument("--manifest",required=True); q.add_argument("--accepted-manifest",required=True); q.add_argument("--source-parquet",required=True); q.add_argument("--train-source-parquet",required=True); q.add_argument("--suite-parquet",required=True); q.add_argument("--tokenizer-root",required=True); q.add_argument("--receipts",required=True); q.add_argument("--train-roots",required=True); q.add_argument("--eval-roots",required=True); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=e0)
    q=s.add_parser("e1"); q.add_argument("--manifest",required=True); q.add_argument("--rows",required=True); q.add_argument("--receipts",required=True); q.add_argument("--model-path",required=True); q.add_argument("--seed",type=int,required=True); q.add_argument("--git-commit",required=True); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=e1)
    q=s.add_parser("baseline-import"); q.add_argument("--final-report",required=True); q.add_argument("--curve-ledger",required=True); q.add_argument("--stable-resolved",required=True); q.add_argument("--validation-parquet",required=True); q.add_argument("--materialized-rows",required=True); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=baseline)
    q=s.add_parser("train-health"); q.add_argument("--anchor",type=int,choices=[5,10,15,20,25],required=True); q.add_argument("--output-root",required=True); q.add_argument("--weight-sync-ledger",required=True); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=train_health)
    q=s.add_parser("evaluate"); q.add_argument("--rows",required=True); q.add_argument("--nominal",type=int,required=True); q.add_argument("--unseen",type=int,nargs="*",default=[]); q.add_argument("--output",required=True); q.set_defaults(fn=ev)
    q=s.add_parser("final-audit"); q.add_argument("--run-root",required=True); q.add_argument("--output-root",required=True); q.add_argument("--report",required=True); q.set_defaults(fn=final_audit)
    q=s.add_parser("health-gate"); q.add_argument("--variant",choices=["dro","uniform"],default="dro"); q.add_argument("--anchor",type=int,choices=[5,10,15,20,25],required=True); q.add_argument("--checkpoint-binding",required=True); q.add_argument("--merge-receipt",required=True); q.add_argument("--baseline-import",required=True); q.add_argument("--method-s128",required=True); q.add_argument("--method-horizons",required=True); q.add_argument("--method-suite",required=True); q.add_argument("--method-receipts",required=True); q.add_argument("--stable-resolved",required=True); q.add_argument("--validation-parquet",required=True); q.add_argument("--uniform-horizons"); q.add_argument("--model-path",required=True); q.add_argument("--seed",type=int,required=True); q.add_argument("--nominal",type=int,required=True); q.add_argument("--unseen",type=int,nargs="*",default=[]); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=health_gate)
    a=p.parse_args()
    try: a.fn(a)
    except HDRContractError as e: print(f"HDR_NO_GO:{e}",file=sys.stderr); raise SystemExit(2)
if __name__=="__main__": main()
