#!/usr/bin/env python3
"""Strict-vLLM evidence-equated evaluator for base or merged actor weights."""
import argparse, fcntl, hashlib, json, math, os, re, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))

def _json_safe(value):
 """Recursively convert pandas/numpy payloads to strict JSON primitives."""
 if isinstance(value,dict): return {str(k):_json_safe(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)): return [_json_safe(v) for v in value]
 if hasattr(value,"tolist"):
  converted=value.tolist()
  return _json_safe(converted)
 if hasattr(value,"item"):
  try: return _json_safe(value.item())
  except (TypeError,ValueError): pass
 if value is None or isinstance(value,(str,int,float,bool)): return value
 raise TypeError(f"HDR_NO_GO:unsupported_json_value:{type(value).__module__}.{type(value).__name__}")

def _write_snapshot(path,rows):
 tmp=path.with_name(path.name+".tmp")
 tmp.write_text(json.dumps(_json_safe(rows),indent=2,sort_keys=True,allow_nan=False)+"\n")
 os.replace(tmp,path)

def _load_partial(path):
 if not path.exists(): return []
 try: rows=json.loads(path.read_text())
 except (OSError,json.JSONDecodeError) as e: raise SystemExit(f"HDR_NO_GO:invalid_partial_snapshot:{e}")
 if not isinstance(rows,list) or not all(isinstance(row,dict) for row in rows):
  raise SystemExit("HDR_NO_GO:invalid_partial_snapshot_shape")
 return rows

def _digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def _append_progress(ledger,path,rows,suite_sha,model_path,seed):
 from recurrent.research.gate_a_execution import append_jsonl
 append_jsonl(ledger,{"record_type":"hdr_eval_progress","output_path":str(path),"snapshot_sha256":_digest(path),"completed_rows":len(rows),"suite_sha256":suite_sha,"model_path":model_path,"seed":seed})

def _verify_progress(ledger,path,rows,suite_sha,model_path,seed):
 from recurrent.research.gate_a_execution import validate_jsonl_chain
 try: records=[json.loads(x) for x in Path(ledger).read_text().splitlines() if x.strip()]
 except (OSError,json.JSONDecodeError) as e: raise SystemExit(f"HDR_NO_GO:invalid_progress_ledger:{e}")
 if validate_jsonl_chain(records): raise SystemExit("HDR_NO_GO:invalid_progress_ledger_chain")
 matches=[r for r in records if r.get("record_type")=="hdr_eval_progress" and r.get("output_path")==str(path)]
 expected={"snapshot_sha256":_digest(path),"completed_rows":len(rows),"suite_sha256":suite_sha,"model_path":model_path,"seed":seed}
 if not matches or any(matches[-1].get(k)!=v for k,v in expected.items()): raise SystemExit("HDR_NO_GO:partial_snapshot_ledger_mismatch")

def _verify_metrics(saved,calc,ordinal):
 for key,value in calc.items():
  got=saved.get(key)
  if key not in saved or isinstance(got,bool) or not isinstance(got,(int,float)) or not math.isfinite(float(got)) or not math.isfinite(float(value)) or abs(float(got)-float(value))>1e-12:
   raise SystemExit(f"HDR_NO_GO:partial_snapshot_metric_mismatch:{ordinal}")

def main():
 p=argparse.ArgumentParser(); p.add_argument("--suite",required=True); p.add_argument("--model",required=True); p.add_argument("--output",required=True); p.add_argument("--ledger",required=True); p.add_argument("--seed",type=int,default=2026); p.add_argument("--tensor-parallel-size",type=int,default=2); a=p.parse_args()
 pair=os.environ.get("CUDA_VISIBLE_DEVICES","")
 if not re.fullmatch(r"\d+,\d+",pair): raise SystemExit("HDR_NO_GO:explicit_GPU_PAIR_required")
 gpu=list(map(int,pair.split(",")))
 if gpu!=sorted(set(gpu)): raise SystemExit("HDR_NO_GO:canonical_GPU_PAIR_required")
 work=Path(os.environ.get("MEMAGENT_HDR_WORK_ROOT",""))
 if not work.is_absolute(): raise SystemExit("HDR_NO_GO:absolute_work_root_required")
 locks=[]
 for g in gpu:
  lp=work/"locks"/f"memagent_h20_gpu_{g}.lock"; lp.parent.mkdir(parents=True,exist_ok=True); f=lp.open("a+")
  try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError: raise SystemExit("HDR_NO_GO:gpu_lock_conflict")
  locks.append(f)
 apps=subprocess.check_output(["nvidia-smi","-i",pair,"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
 if apps.strip(): raise SystemExit("HDR_NO_GO:gpu_occupied_no_process_killed")
 out=Path(a.output)
 if out.exists(): raise SystemExit("HDR_NO_GO:evaluation_output_exists")
 out.parent.mkdir(parents=True,exist_ok=True)
 import pandas as pd
 from transformers import AutoTokenizer
 try:
  from vllm import LLM, SamplingParams
 except Exception as e: raise SystemExit(f"HDR_NO_GO:strict_vllm_unavailable:{e}")
 from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
 from recurrent.research.hdr_memrl import build_horizon_receipt, prediction_metrics, stable_root_id
 from recurrent.utils import chat_template
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); template=chat_template(tok)
 llm=LLM(model=a.model,tensor_parallel_size=a.tensor_parallel_size,trust_remote_code=False,enforce_eager=False)
 df=pd.read_parquet(a.suite); receipts=[]
 suite_sha=hashlib.sha256(Path(a.suite).read_bytes()).hexdigest()
 partial=out.with_name(out.name+".partial")
 rows=_load_partial(partial)
 if len(rows)>len(df): raise SystemExit("HDR_NO_GO:partial_snapshot_too_long")
 if rows: _verify_progress(a.ledger,partial,rows,suite_sha,str(Path(a.model).resolve()),a.seed)
 for ordinal,(pos,row) in enumerate(df.iterrows()):
  prompt=row["prompt"]; query=prompt[0]["content"] if isinstance(prompt,(list,tuple)) else prompt.tolist()[0]["content"]
  idx=int(row.get("extra_info",{}).get("index",pos)); tokens=tok.encode(str(row["context"]),add_special_tokens=False)
  if len(tokens)>40000: raise SystemExit("HDR_NO_GO:evaluator_refuses_truncated_evidence")
  horizon=int(row["horizon_id"]); rid=str(row.get("stable_root_id_receipt") or stable_root_id(dataset_sha256="0"*64,source_index=idx,query=query))
  receipt=build_horizon_receipt(rid,query,tokens,horizon); receipts.append(receipt); memory="No previous memory"; total=0; cap=False; trace=[]
  reward=row.get("reward_model",{})
  if isinstance(reward,str):
   try: reward=json.loads(reward)
   except json.JSONDecodeError: raise SystemExit("HDR_NO_GO:invalid_reward_model_json")
  gold=_json_safe(reward.get("ground_truth",reward.get("target",row.get("answer",""))) if isinstance(reward,dict) else row.get("answer",""))
  if ordinal<len(rows):
   saved=rows[ordinal]
   expected={"stable_id":f"{rid}:h{horizon}","root_id":rid,"source_order_index":int(row.get("source_order_index",idx)),"raw_row_position":int(row.get("raw_row_position",idx)),"identity_resolved_sha256":_json_safe(row.get("identity_resolved_sha256")),"ground_truth_hash":_json_safe(row.get("ground_truth_hash")),"suite_sha256":suite_sha,"horizon":horizon,"gold":gold,"truncated":False,"evidence_equated":True,"model_path":str(Path(a.model).resolve()),"seed":a.seed,"receipt":receipt.as_dict()}
   if any(_json_safe(saved.get(k))!=_json_safe(v) for k,v in expected.items()):
    raise SystemExit(f"HDR_NO_GO:partial_snapshot_identity_mismatch:{ordinal}")
   trace=saved.get("generation_trace")
   if not isinstance(trace,list) or len(trace)!=horizon+1: raise SystemExit(f"HDR_NO_GO:partial_snapshot_trace_shape:{ordinal}")
   for turn,(start,stop) in enumerate(receipt.chunk_bounds):
    chunk=tok.decode(tokens[start:stop],skip_special_tokens=False)
    msg=template.format(message=TEMPLATE.format(prompt=query,memory=memory,chunk=chunk)); count=len(tok.encode(msg,add_special_tokens=False)); item=trace[turn]
    if not isinstance(item,dict) or item.get("input_tokens")!=count or not isinstance(item.get("output_text"),str) or item.get("finish_reason") not in ("stop","length",None): raise SystemExit(f"HDR_NO_GO:partial_snapshot_trace_mismatch:{ordinal}:{turn}")
    total+=count; memory=item["output_text"]
   msg=template.format(message=TEMPLATE_FINAL_BOXED.format(prompt=query,memory=memory)); count=len(tok.encode(msg,add_special_tokens=False)); item=trace[-1]
   if not isinstance(item,dict) or item.get("input_tokens")!=count or not isinstance(item.get("output_text"),str) or item.get("finish_reason") not in ("stop","length",None): raise SystemExit(f"HDR_NO_GO:partial_snapshot_final_trace_mismatch:{ordinal}")
   total+=count; cap=any(x.get("finish_reason")=="length" for x in trace)
   if saved.get("prediction")!=item["output_text"] or saved.get("total_input_tokens")!=total or saved.get("cap_hit") is not cap: raise SystemExit(f"HDR_NO_GO:partial_snapshot_control_mismatch:{ordinal}")
   if row.get("identity_resolved_sha256"):
    from recurrent.research.s128_hotpot_metrics import score_terminal_output
    scored=score_terminal_output(saved["prediction"],gold); calc={"em":scored["exact_match"],"token_f1":scored["token_f1"],"format":scored["format_success"]}
   else: calc=prediction_metrics(str(saved["prediction"]),str(gold))
   _verify_metrics(saved,calc,ordinal)
   print(f"HDR_EVAL_RESUME {ordinal+1}/{len(df)}",flush=True)
   continue
  for turn,(start,stop) in enumerate(receipt.chunk_bounds):
   chunk=tok.decode(tokens[start:stop],skip_special_tokens=False)
   msg=template.format(message=TEMPLATE.format(prompt=query,memory=memory,chunk=chunk)); total+=len(tok.encode(msg,add_special_tokens=False))
   sp=SamplingParams(n=1,temperature=1.0,top_p=1.0,max_tokens=1024,seed=a.seed+idx*1009+horizon*37+turn)
   ans=llm.generate([msg],sp,use_tqdm=False)[0].outputs[0]; memory=ans.text; cap |= ans.finish_reason=="length"; trace.append({"input_tokens":len(tok.encode(msg,add_special_tokens=False)),"output_text":ans.text,"finish_reason":ans.finish_reason})
  msg=template.format(message=TEMPLATE_FINAL_BOXED.format(prompt=query,memory=memory)); total+=len(tok.encode(msg,add_special_tokens=False))
  sp=SamplingParams(n=1,temperature=0.0,top_p=1.0,max_tokens=1024,seed=a.seed+idx*1009+horizon*37+horizon)
  ans=llm.generate([msg],sp,use_tqdm=False)[0].outputs[0]; cap |= ans.finish_reason=="length"; trace.append({"input_tokens":len(tok.encode(msg,add_special_tokens=False)),"output_text":ans.text,"finish_reason":ans.finish_reason})
  rec={"stable_id":f"{rid}:h{horizon}","root_id":rid,"source_order_index":int(row.get("source_order_index",idx)),"raw_row_position":int(row.get("raw_row_position",idx)),"identity_resolved_sha256":row.get("identity_resolved_sha256"),"ground_truth_hash":row.get("ground_truth_hash"),"suite_sha256":suite_sha,"horizon":horizon,"prediction":ans.text,"gold":gold,"total_input_tokens":total,"cap_hit":bool(cap),"truncated":False,"evidence_equated":True,"model_path":str(Path(a.model).resolve()),"seed":a.seed,"receipt":receipt.as_dict(),"generation_trace":trace}
  if row.get("identity_resolved_sha256"):
   from recurrent.research.s128_hotpot_metrics import score_terminal_output
   scored=score_terminal_output(ans.text,gold); rec.update(em=scored["exact_match"],token_f1=scored["token_f1"],format=scored["format_success"])
  else: rec.update(prediction_metrics(ans.text,str(gold)))
  rows.append(_json_safe(rec))
  _write_snapshot(partial,rows)
  _append_progress(a.ledger,partial,rows,suite_sha,str(Path(a.model).resolve()),a.seed)
  print(f"HDR_EVAL_PROGRESS {len(rows)}/{len(df)}",flush=True)
 from recurrent.research.hdr_memrl import validate_evidence_equated
 validate_evidence_equated(receipts, sorted(set(int(x) for x in df["horizon_id"].tolist())))
 _write_snapshot(out,rows)
 if partial.exists(): partial.unlink()
if __name__=="__main__": main()
