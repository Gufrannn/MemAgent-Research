#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
from recurrent.research.hdr_memrl import stable_root_id
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--input-sha256",required=True); p.add_argument("--output",required=True); a=p.parse_args()
if hashlib.sha256(Path(a.input).read_bytes()).hexdigest()!=a.input_sha256: raise SystemExit("HDR_NO_GO:root_source_sha_mismatch")
import pandas as pd
df=pd.read_parquet(a.input); roots=[]
for pos,row in df.iterrows():
 prompt=row["prompt"]; query=prompt[0]["content"] if isinstance(prompt,(list,tuple)) else prompt.tolist()[0]["content"]
 idx=int(row.get("extra_info",{}).get("index",pos)); roots.append(stable_root_id(dataset_sha256=a.input_sha256,source_index=idx,query=query))
if len(roots)!=len(set(roots)): raise SystemExit("HDR_NO_GO:duplicate_stable_train_root")
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(sorted(roots),indent=2)+"\n")
