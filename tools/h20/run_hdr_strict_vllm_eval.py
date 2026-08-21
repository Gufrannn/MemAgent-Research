#!/usr/bin/env python3
"""Strict-vLLM evidence-equated evaluator for base or merged actor weights."""
import argparse, fcntl, hashlib, json, os, re, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))

def main():
 p=argparse.ArgumentParser(); p.add_argument("--suite",required=True); p.add_argument("--model",required=True); p.add_argument("--output",required=True); p.add_argument("--seed",type=int,default=2026); p.add_argument("--tensor-parallel-size",type=int,default=2); a=p.parse_args()
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
 df=pd.read_parquet(a.suite); rows=[]; receipts=[]
 for pos,row in df.iterrows():
  prompt=row["prompt"]; query=prompt[0]["content"] if isinstance(prompt,(list,tuple)) else prompt.tolist()[0]["content"]
  idx=int(row.get("extra_info",{}).get("index",pos)); tokens=tok.encode(str(row["context"]),add_special_tokens=False)
  if len(tokens)>40000: raise SystemExit("HDR_NO_GO:evaluator_refuses_truncated_evidence")
  horizon=int(row["horizon_id"]); rid=str(row.get("stable_root_id_receipt") or stable_root_id(dataset_sha256="0"*64,source_index=idx,query=query))
  receipt=build_horizon_receipt(rid,query,tokens,horizon); receipts.append(receipt); memory="No previous memory"; total=0; cap=False
  for turn,(start,stop) in enumerate(receipt.chunk_bounds):
   chunk=tok.decode(tokens[start:stop],skip_special_tokens=False)
   msg=template.format(message=TEMPLATE.format(prompt=query,memory=memory,chunk=chunk)); total+=len(tok.encode(msg,add_special_tokens=False))
   sp=SamplingParams(n=1,temperature=1.0,top_p=1.0,max_tokens=1024,seed=a.seed+idx*1009+horizon*37+turn)
   ans=llm.generate([msg],sp,use_tqdm=False)[0].outputs[0]; memory=ans.text; cap |= ans.finish_reason=="length"
  msg=template.format(message=TEMPLATE_FINAL_BOXED.format(prompt=query,memory=memory)); total+=len(tok.encode(msg,add_special_tokens=False))
  sp=SamplingParams(n=1,temperature=0.0,top_p=1.0,max_tokens=1024,seed=a.seed+idx*1009+horizon*37+horizon)
  ans=llm.generate([msg],sp,use_tqdm=False)[0].outputs[0]; cap |= ans.finish_reason=="length"
  reward=row.get("reward_model",{}); gold=reward.get("ground_truth",reward.get("target",row.get("answer",""))) if isinstance(reward,dict) else row.get("answer","")
  rec={"stable_id":f"{rid}:h{horizon}","root_id":rid,"horizon":horizon,"prediction":ans.text,"gold":str(gold),"total_input_tokens":total,"cap_hit":bool(cap),"truncated":False,"evidence_equated":True,"model_path":str(Path(a.model).resolve()),"seed":a.seed,"receipt":receipt.as_dict()}; rec.update(prediction_metrics(ans.text,str(gold))); rows.append(rec)
 from recurrent.research.hdr_memrl import validate_evidence_equated
 validate_evidence_equated(receipts, sorted(set(int(x) for x in df["horizon_id"].tolist())))
 out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
