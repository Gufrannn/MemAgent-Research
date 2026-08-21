#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--checkpoint-binding",required=True); p.add_argument("--merged-model",required=True); p.add_argument("--output",required=True); a=p.parse_args()
binding_path=Path(a.checkpoint_binding); binding=json.loads(binding_path.read_text()); root=binding_path.parent
for item in binding.get("inventory",[]):
 f=root/item["path"]
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=item["sha256"] or f.stat().st_size!=item["size"]: raise SystemExit("HDR_NO_GO:source_checkpoint_tamper_before_merge_binding")
merged=Path(a.merged_model); files=[]
for f in sorted(x for x in merged.rglob("*") if x.is_file()): files.append({"path":str(f.relative_to(merged)),"size":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()})
if not files: raise SystemExit("HDR_NO_GO:empty_merged_model")
receipt={"status":"PASS","decision":"HDR_MERGED_MODEL_BOUND","checkpoint_binding":str(binding_path.resolve()),"checkpoint_binding_sha256":hashlib.sha256(binding_path.read_bytes()).hexdigest(),"global_step":binding["global_step"],"git_commit":binding["git_commit"],"merged_model":str(merged.resolve()),"merged_inventory":files}
out=Path(a.output)
if out.exists(): raise SystemExit("HDR_NO_GO:merge_receipt_exists")
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
