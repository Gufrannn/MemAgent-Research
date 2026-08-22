#!/usr/bin/env python3
"""Materialize and freeze the exact PRD prior without trusting a cache alias."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
REVISION="c89bee90d9f811437d9735454613c35b4a3c4dc8"

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--target",type=Path,required=True)
    parser.add_argument("--certificate",type=Path,required=True); parser.add_argument("--download",action="store_true")
    args=parser.parse_args(); target=args.target.resolve(); failures=[]
    if args.download:
        if target.exists() and any(target.iterdir()): raise SystemExit("PRD_NO_GO: prior target is nonempty")
        target.mkdir(parents=True,exist_ok=True)
        subprocess.run(["huggingface-cli","download",MODEL,"--revision",REVISION,"--local-dir",str(target)],check=True)
    try:
        config=json.loads((target/"config.json").read_text())
        if config.get("hidden_size")!=896 or config.get("num_hidden_layers")!=24: failures.append("wrong prior architecture")
    except Exception as exc: failures.append(f"invalid config: {exc}")
    inventory=[]
    for path in sorted(target.iterdir()) if target.is_dir() else []:
        if path.is_symlink(): failures.append(f"symlink forbidden: {path.name}")
        elif path.is_file() and path.suffix in {".json",".safetensors",".txt"}:
            inventory.append({"path":path.name,"size":path.stat().st_size,"sha256":sha(path)})
    if not any(row["path"].endswith(".safetensors") for row in inventory): failures.append("weights missing")
    payload={"schema_version":1,"status":"PASS" if not failures else "FAIL",
        "decision":"PRD_PRIOR_MATERIALIZATION_PASS" if not failures else "PRD_PRIOR_MATERIALIZATION_NO_GO",
        "model":MODEL,"revision":REVISION,"target":str(target),"inventory":inventory,"failures":failures}
    args.certificate.parent.mkdir(parents=True,exist_ok=True)
    with args.certificate.open("x") as stream: json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n")
    return 0 if not failures else 2
if __name__=="__main__": raise SystemExit(main())
