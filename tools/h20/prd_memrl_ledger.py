#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

def digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def verify(path: Path) -> tuple[int, str | None]:
    previous = None
    count = 0
    if not path.exists(): return count, previous
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["sequence"] != count or record["previous_record_sha256"] != previous:
            raise ValueError(f"ledger chain mismatch at sequence {count}")
        previous = digest(record); count += 1
    return count, previous

def main() -> int:
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True)
    a=s.add_parser("append")
    for key in ("ledger","run_id","event","git_commit","payload"): a.add_argument("--"+key.replace("_","-"),required=True)
    v=s.add_parser("verify"); v.add_argument("--ledger",required=True)
    x=p.parse_args(); path=Path(x.ledger); count,previous=verify(path)
    if x.cmd=="verify": print(json.dumps({"status":"PASS","records":count,"head_sha256":previous})); return 0
    payload=json.loads(Path(x.payload).read_text())
    record={"schema_version":1,"sequence":count,"timestamp":datetime.now(timezone.utc).isoformat(),"run_id":x.run_id,"event":x.event,"git_commit":x.git_commit,"payload":payload,"previous_record_sha256":previous}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n"); f.flush()
    verify(path); return 0
if __name__=="__main__": raise SystemExit(main())
