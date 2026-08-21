#!/usr/bin/env python3
"""Build an evidence-equated root×horizon parquet and machine receipts."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from recurrent.research.hdr_memrl import build_horizon_receipt, stable_root_id, validate_evidence_equated, write_json

def main():
 p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--input-sha256",required=True); p.add_argument("--model",required=True); p.add_argument("--horizons",type=int,nargs="+",required=True); p.add_argument("--count",type=int,default=128); p.add_argument("--output-parquet",required=True); p.add_argument("--receipts",required=True); p.add_argument("--roots",required=True); a=p.parse_args()
 import pandas as pd
 from transformers import AutoTokenizer
 h=hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
 if h!=a.input_sha256: raise SystemExit("HDR_NO_GO:dataset_sha_mismatch")
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 df=pd.read_parquet(a.input); out=[]; receipts=[]; roots=[]
 for source_pos,row in df.iterrows():
  if len(roots) >= a.count: break
  prompt=row["prompt"]; query=prompt[0]["content"] if isinstance(prompt,(list,tuple)) else prompt.tolist()[0]["content"]
  idx=int(row.get("extra_info",{}).get("index",source_pos)); rid=stable_root_id(dataset_sha256=h,source_index=idx,query=query)
  tokens=tok.encode(str(row["context"]),add_special_tokens=False)
  if len(tokens)>40000: continue
  if not tokens: raise SystemExit("HDR_NO_GO:empty_effective_evidence")
  roots.append(rid)
  for horizon in a.horizons:
   receipt=build_horizon_receipt(rid,query,tokens,horizon); receipts.append(receipt)
   item=row.to_dict(); item["horizon_id"]=horizon; item["stable_root_id_receipt"]=rid; out.append(item)
 validate_evidence_equated(receipts,a.horizons)
 if len(set(roots)) != a.count: raise SystemExit(f"HDR_NO_GO:only_{len(set(roots))}_nontruncated_roots")
 Path(a.output_parquet).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(out).to_parquet(a.output_parquet,index=False)
 write_json(a.receipts,[r.as_dict() for r in receipts]); write_json(a.roots,sorted(set(roots)))
if __name__=="__main__": main()
