#!/usr/bin/env python3
from __future__ import annotations
import argparse, fcntl, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

def digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def verify(path: Path) -> tuple[int, str | None]:
    previous = None
    count = 0
    if not path.exists(): return count, previous
    if path.is_symlink(): raise ValueError("ledger must not be a symlink")
    run_id = git_commit = None
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["sequence"] != count or record["previous_record_sha256"] != previous:
            raise ValueError(f"ledger chain mismatch at sequence {count}")
        if run_id is None: run_id, git_commit = record["run_id"], record["git_commit"]
        if record["run_id"] != run_id or record["git_commit"] != git_commit:
            raise ValueError(f"ledger identity mismatch at sequence {count}")
        previous = digest(record); count += 1
    return count, previous

def append_record(path: Path, run_id: str, event: str, git_commit: str, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    lock_path=path.with_suffix(path.suffix+".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        count,previous=verify(path)
        record={"schema_version":1,"sequence":count,"timestamp":datetime.now(timezone.utc).isoformat(),"run_id":run_id,"event":event,"git_commit":git_commit,"payload":payload,"previous_record_sha256":previous}
        with path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n"); f.flush(); os.fsync(f.fileno())
        verify(path)

def main() -> int:
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True)
    a=s.add_parser("append")
    for key in ("ledger","run_id","event","git_commit","payload"): a.add_argument("--"+key.replace("_","-"),required=True)
    v=s.add_parser("verify"); v.add_argument("--ledger",required=True)
    x=p.parse_args(); path=Path(x.ledger)
    if x.cmd=="verify":
        count,previous=verify(path); print(json.dumps({"status":"PASS","records":count,"head_sha256":previous})); return 0
    payload_path=Path(x.payload)
    if payload_path.is_symlink() or not payload_path.is_file(): raise ValueError("payload must be a regular non-symlink file")
    payload=json.loads(payload_path.read_text())
    append_record(path,x.run_id,x.event,x.git_commit,payload)
    return 0
if __name__=="__main__": raise SystemExit(main())
