#!/usr/bin/env python3
"""Freeze current RWWPO actor shards for explicitly non-claim diagnostic S128."""
import argparse, hashlib, json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--source-sha256",required=True)
    p.add_argument("--checkpoint-root",required=True); p.add_argument("--training-commit",required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    source=Path(a.source).resolve(); root=Path(a.checkpoint_root).resolve(); out=Path(a.output)
    if out.exists(): raise SystemExit("RWWPO_DIAGNOSTIC_MANIFEST_NO_GO:output exists")
    if sha(source)!=a.source_sha256: raise SystemExit("RWWPO_DIAGNOSTIC_MANIFEST_NO_GO:source SHA")
    if subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()!=a.expected_commit: raise SystemExit("RWWPO_DIAGNOSTIC_MANIFEST_NO_GO:checkout")
    if not re.fullmatch(r"[0-9a-f]{40}",a.training_commit): raise SystemExit("RWWPO_DIAGNOSTIC_MANIFEST_NO_GO:training commit")
    manifest=json.loads(source.read_text()); artifacts=manifest.setdefault("execution_binding",{}).setdefault("model_artifacts",{})
    for step in (5,10,15,20,25):
        checkpoint=root/f"global_step_{step}"; shards=[]
        for rank in (0,1):
            path=checkpoint/"actor"/f"model_world_size_2_rank_{rank}.pt"
            if not path.is_file(): raise SystemExit(f"RWWPO_DIAGNOSTIC_MANIFEST_NO_GO:missing {path}")
            shards.append({"path":str(path.relative_to(checkpoint)),"size":path.stat().st_size,"sha256":sha(path)})
        artifacts[f"RWWPO{step}"]={"kind":"post_hoc_diagnostic_actor_checkpoint","path":str(checkpoint),"global_step":step,"training_git_commit":a.training_commit,"actor_model_shards":shards,"fsdp_world_size":2,"load_mode":"actor_only"}
    manifest.update({"diagnostic_only":True,"diagnostic_reason":"training-time checkpoint inventory event missing; current actor shards frozen post hoc for non-claim evaluation only","diagnostic_materializer_git_commit":a.expected_commit,"diagnostic_materialized_at":datetime.now(timezone.utc).isoformat(),"training_git_commit":a.training_commit})
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"DIAGNOSTIC_ONLY","decision":"RWWPO_POST_HOC_CHECKPOINT_MANIFEST_MATERIALIZED","output":str(out.resolve()),"sha256":sha(out)},sort_keys=True))
if __name__=="__main__": main()
