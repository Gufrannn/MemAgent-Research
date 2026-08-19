#!/usr/bin/env python3
"""Fail-closed validator for exact same-materialized-candidate E pairs."""
import argparse,hashlib,json,math,re,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from recurrent.research.exact_noop_v2 import E_PAIR_KEYS,E_ARM_KEYS,E_ARMS,E_LABEL

HASH_KEYS=("upstream_state_hash","candidate_memory_token_hash","suffix_hash",
           "loaded_memory_token_hash","output_token_ids_hash")
SHA256=re.compile(r"^[0-9a-f]{64}$")

def validate(rows):
    if not rows: raise ValueError("E_QUALIFICATION_FAIL: empty manifest")
    groups=defaultdict(list); seen=set()
    for index,row in enumerate(rows):
        missing=[key for key in E_PAIR_KEYS+E_ARM_KEYS if key not in row]
        if missing: raise ValueError(f"E_QUALIFICATION_FAIL: row={index} missing={missing}")
        if row["arm"] not in E_ARMS: raise ValueError(f"E_QUALIFICATION_FAIL: invalid arm={row['arm']}")
        if row.get("estimand")!=E_LABEL:
            raise ValueError("E_QUALIFICATION_FAIL: manifest is not preregistered E estimand")
        if not row["stable_id"] or not row["reader_seed_or_coupling_id"] or not row["endpoint_version"]:
            raise ValueError("E_QUALIFICATION_FAIL: empty identity/coupling/endpoint version")
        if not isinstance(row["turn"],int) or not isinstance(row["writer_seed"],int):
            raise ValueError("E_QUALIFICATION_FAIL: turn and writer_seed must be integers")
        invalid_hashes=[name for name in HASH_KEYS if not isinstance(row[name],str) or not SHA256.fullmatch(row[name])]
        if invalid_hashes: raise ValueError(f"E_QUALIFICATION_FAIL: invalid SHA-256 fields={invalid_hashes}")
        key=tuple(row[name] for name in E_PAIR_KEYS); unique=(key,row["arm"])
        if unique in seen: raise ValueError(f"E_QUALIFICATION_FAIL: duplicate pair arm; no last-write-wins: {unique}")
        seen.add(unique);groups[key].append(row)
        if row.get("candidate_materialization_count")!=1 or row.get("writer_runs_after_materialization")!=0:
            raise ValueError("E_QUALIFICATION_FAIL: proposal/writer execution contract violated")
        if not isinstance(row["endpoint_value"],(int,float)) or not math.isfinite(row["endpoint_value"]):
            raise ValueError("E_QUALIFICATION_FAIL: endpoint_value nonfinite")
    for key,pair in groups.items():
        if {row["arm"] for row in pair}!=E_ARMS or len(pair)!=2:
            raise ValueError(f"E_QUALIFICATION_FAIL: same-pair key lacks exact commit/retain arms: {key}")
        commit=next(row for row in pair if row["arm"]=="commit");retain=next(row for row in pair if row["arm"]=="retain")
        if commit["loaded_memory_token_hash"]!=commit["candidate_memory_token_hash"]:
            raise ValueError("E_QUALIFICATION_FAIL: commit did not load candidate memory")
        if retain["loaded_memory_token_hash"]==retain["candidate_memory_token_hash"]:
            raise ValueError("E_QUALIFICATION_FAIL: retain/NOOP must load old memory while referencing same candidate hash")
    return {"status":"E_QUALIFIED","pair_count":len(groups),"estimand":"same-candidate execution effect",
      "state_level_writer_policy_risk_identified":False,"training_authorized":False,"select_best_estimand":False}
def self_test():
    from recurrent.research.exact_noop_v2 import materialize_proposal,build_arm_record
    proposal=materialize_proposal(stable_id="e1",turn=1,upstream_state_hash="a"*64,candidate_token_ids=[1,2],suffix_hash="b"*64,
      writer_seed=7,reader_seed_or_coupling_id="r",endpoint_version="v1",old_memory_token_hash="c"*64)
    rows=[build_arm_record(proposal,arm="commit",output_token_ids=[3],endpoint_value=1),build_arm_record(proposal,arm="retain",output_token_ids=[4],endpoint_value=0)]
    assert validate(rows)["status"]=="E_QUALIFIED"
    for bad in (rows+[dict(rows[0])],[rows[0]], [{k:v for k,v in rows[0].items() if k!="suffix_hash"},rows[1]]):
        try:validate(bad)
        except ValueError as exc:assert "E_QUALIFICATION_FAIL" in str(exc)
        else:raise AssertionError("invalid E manifest accepted")
    print("exact_noop_v2_manifest_self_test=ok")
def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest");p.add_argument("--self-test",action="store_true");a=p.parse_args()
    if a.self_test:self_test();return
    if not a.manifest:p.error("--manifest required")
    raw=Path(a.manifest).read_bytes();rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    result=validate(rows);result["manifest_sha256"]=hashlib.sha256(raw).hexdigest()
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
