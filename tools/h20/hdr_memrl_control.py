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
    if not HEX64.fullmatch(a.expected_bundle_sha256) or digest(a.bundle)!=a.expected_bundle_sha256: raise HDRContractError("baseline bundle authority SHA mismatch")
    if digest(a.authority_manifest)!="c27b45dfdb08aaa9c30a11cc353a97a902ac170754f6511f07b5d232be2ebd28": raise HDRContractError("Original curve authority manifest drift")
    authority=load(a.authority_manifest)
    if authority.get("scope",{}).get("interfaces") != ["I","Original5","Original10","Original15","Original20","Original25"]: raise HDRContractError("Original authority anchor closure mismatch")
    bundle=load(a.bundle); imported=[]
    recomputed={}
    for item in bundle["files"]:
        p=Path(item["path"])
        if not p.is_file() or digest(p)!=item["sha256"]: raise HDRContractError(f"baseline evidence SHA mismatch: {p}")
        imported.append({"path":str(p.resolve()),"sha256":digest(p),"size":p.stat().st_size,"kind":item.get("kind"),"anchor":item.get("anchor")})
        if item.get("kind") == "predictions":
            recomputed[str(item["anchor"])]=aggregate_predictions(load(p))
    expected=bundle["expected_aggregates"]
    required={"I","Original5","Original10","Original15","Original20","Original25"}
    if set(expected)!=required or set(recomputed)!=required: raise HDRContractError("baseline bundle must contain exact I/T5/T10/T15/T20/T25 anchors")
    for anchor, values in recomputed.items():
        if anchor not in expected: raise HDRContractError(f"baseline aggregate missing anchor {anchor}")
        for key in ("em","token_f1","format"):
            if abs(float(values[key])-float(expected[anchor][key]))>float(bundle.get("metric_tolerance",1e-12)):
                raise HDRContractError(f"baseline aggregate mismatch {anchor}/{key}")
    report={"status":"PASS","decision":"ORIGINAL_BASELINE_IMPORT_PASS","files":imported,
            "expected_aggregates":expected,"recomputed_aggregates":recomputed,"bundle_sha256":a.expected_bundle_sha256,"authority_manifest_sha256":digest(a.authority_manifest)}
    write_json(a.output,report); append(a.ledger,{"record_type":"baseline_import",**report}); print(json.dumps(report,sort_keys=True))

def ev(a):
    rows=load(a.rows)
    for row in rows:
        if "prediction" in row:
            row.update(prediction_metrics(str(row["prediction"]),str(row["gold"])))
    report=evaluate_horizons(rows,a.nominal,a.unseen)
    report["decision"]="HDR_HORIZON_EVAL_PASS"; write_json(a.output,report); print(json.dumps(report,sort_keys=True))

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
        if [r.get("target_step") for r in launches] != [5,10,15,20,25]: failures.append("launch anchor chain mismatch")
        if len({(r.get("git_commit"),r.get("gpu_pair"),r.get("run_id")) for r in launches})>1: failures.append("launch commit/GPU/run drift")
    sync_path=root/"hdr_weight_sync_ledger.jsonl"
    if not sync_path.is_file(): failures.append("missing distributed weight-sync ledger")
    else:
        sync=[json.loads(x) for x in sync_path.read_text().splitlines() if x.strip()]
        failures += validate_jsonl_chain(sync)
        got={int(r["global_step"]) for r in sync if r.get("record_type")=="weight_sync_summary" and r.get("sync_kind")=="post_actor_update"}
        if got!=set(range(1,26)): failures.append(f"weight-sync steps mismatch: {sorted(got)}")
    for name,decision in (("p0.json","HDR_P0_PASS"),("e0.json","HDR_E0_PASS"),("e1.json","HDR_E1_PASS"),("baseline_import.json","ORIGINAL_BASELINE_IMPORT_PASS"),("paper_review.json","PAPER_FRAMING_GO")):
        p=root/"certificates"/name
        if not p.is_file() or load(p).get("decision")!=decision: failures.append(f"invalid gate {name}")
    for step in (5,10,15,20,25):
        p=root/"certificates"/f"t{step}_health.json"
        if not p.is_file() or load(p).get("decision")!=f"HDR_T{step}_HEALTH_PASS": failures.append(f"missing anchor health {step}")
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
    method_rows=load(a.method_s128); method=aggregate_predictions(method_rows)
    if method["count"]!=128: raise HDRContractError("fixed-S128 method evaluation must contain exactly 128 unique rows")
    anchor_files=[x for x in baseline.get("files",[]) if x.get("kind")=="predictions" and str(x.get("anchor"))==anchor]
    if len(anchor_files)!=1: raise HDRContractError("baseline stable-ID authority file missing")
    authority_rows=load(anchor_files[0]["path"])
    if {str(x["stable_id"]) for x in method_rows}!={str(x["stable_id"]) for x in authority_rows}: raise HDRContractError("method S128 stable-ID join mismatch")
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
    oeval=load(a.original_horizons)
    imported_by_path={str(Path(x["path"]).resolve()):x["sha256"] for x in baseline.get("files",[])}
    original_path=str(Path(a.original_horizons).resolve())
    if imported_by_path.get(original_path)!=digest(a.original_horizons): raise HDRContractError("Original horizon evaluation is not baseline-authority imported")
    if oeval.get("status")!="PASS": raise HDRContractError("Original horizon authority absent")
    failures=[]
    if float(method["token_f1"]) < float(original["token_f1"])-.02: failures.append("nominal_s128_noninferiority")
    if a.variant=="dro" and a.anchor==5 and float(heval["worst"]["token_f1"]) < float(oeval["worst"]["token_f1"])+.02: failures.append("worst_horizon_gain")
    uniform=None
    if a.variant=="dro" and a.anchor==25:
        if not a.uniform_horizons: raise HDRContractError("T25 requires uniform-ERM horizon baseline")
        uniform=load(a.uniform_horizons)
        if uniform.get("status")!="PASS" or uniform.get("decision")!="UNIFORM_T25_HEALTH_PASS" or uniform.get("git_commit")!=git("rev-parse","HEAD"): raise HDRContractError("uniform T25 health authority mismatch")
        uniform_eval=uniform.get("method_horizons",{})
        if float(heval["worst"]["token_f1"]) < float(uniform_eval["worst"]["token_f1"])+.02: failures.append("uniform_erm_worst_gain")
        if float(method["token_f1"]) < float(original["token_f1"])-.01: failures.append("t25_nominal_one_point_floor")
    prefix="HDR" if a.variant=="dro" else "UNIFORM"
    report={"status":"FAIL" if failures else "PASS","decision":f"{prefix}_T{a.anchor}_HEALTH_FAIL" if failures else f"{prefix}_T{a.anchor}_HEALTH_PASS","variant":a.variant,"anchor":a.anchor,"method_s128":method,"original_s128":original,"method_horizons":heval,"original_horizons":oeval,"uniform_horizons":uniform,"failures":failures,"git_commit":git("rev-parse","HEAD"),"checkpoint_binding_sha256":digest(a.checkpoint_binding),"merge_receipt_sha256":digest(a.merge_receipt),"method_s128_sha256":digest(a.method_s128),"method_horizons_sha256":digest(a.method_horizons),"original_horizons_sha256":digest(a.original_horizons),"model_path":str(Path(a.model_path).resolve()),"seed":a.seed}
    write_json(a.output,report); append(a.ledger,{"record_type":"audit",**report})
    if failures: raise SystemExit(4)

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True)
    q=s.add_parser("preflight"); q.add_argument("--manifest",required=True); q.add_argument("--accepted-manifest",required=True); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=preflight)
    q=s.add_parser("e0"); q.add_argument("--manifest",required=True); q.add_argument("--accepted-manifest",required=True); q.add_argument("--source-parquet",required=True); q.add_argument("--train-source-parquet",required=True); q.add_argument("--suite-parquet",required=True); q.add_argument("--tokenizer-root",required=True); q.add_argument("--receipts",required=True); q.add_argument("--train-roots",required=True); q.add_argument("--eval-roots",required=True); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=e0)
    q=s.add_parser("e1"); q.add_argument("--manifest",required=True); q.add_argument("--rows",required=True); q.add_argument("--receipts",required=True); q.add_argument("--model-path",required=True); q.add_argument("--seed",type=int,required=True); q.add_argument("--git-commit",required=True); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=e1)
    q=s.add_parser("baseline-import"); q.add_argument("--bundle",required=True); q.add_argument("--expected-bundle-sha256",required=True); q.add_argument("--authority-manifest",required=True); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=baseline)
    q=s.add_parser("evaluate"); q.add_argument("--rows",required=True); q.add_argument("--nominal",type=int,required=True); q.add_argument("--unseen",type=int,nargs="*",default=[]); q.add_argument("--output",required=True); q.set_defaults(fn=ev)
    q=s.add_parser("final-audit"); q.add_argument("--run-root",required=True); q.add_argument("--output-root",required=True); q.add_argument("--report",required=True); q.set_defaults(fn=final_audit)
    q=s.add_parser("health-gate"); q.add_argument("--variant",choices=["dro","uniform"],default="dro"); q.add_argument("--anchor",type=int,choices=[5,10,15,20,25],required=True); q.add_argument("--checkpoint-binding",required=True); q.add_argument("--merge-receipt",required=True); q.add_argument("--baseline-import",required=True); q.add_argument("--method-s128",required=True); q.add_argument("--method-horizons",required=True); q.add_argument("--original-horizons",required=True); q.add_argument("--uniform-horizons"); q.add_argument("--model-path",required=True); q.add_argument("--seed",type=int,required=True); q.add_argument("--nominal",type=int,required=True); q.add_argument("--unseen",type=int,nargs="*",default=[]); q.add_argument("--output",required=True); q.add_argument("--ledger",required=True); q.set_defaults(fn=health_gate)
    a=p.parse_args()
    try: a.fn(a)
    except HDRContractError as e: print(f"HDR_NO_GO:{e}",file=sys.stderr); raise SystemExit(2)
if __name__=="__main__": main()
