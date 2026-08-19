#!/usr/bin/env python3
"""Fail-closed horizon-explicit exact-NOOP replay validator."""
import argparse,hashlib,json,math,re,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from recurrent.research.exact_noop_v2 import BASE_PAIR_KEYS,MODE_PAIR_KEYS,E_ARM_KEYS,E_ARMS,E_LABEL

SHA256=re.compile(r"^[0-9a-f]{64}$")
HASH_FIELDS={"upstream_state_hash","candidate_memory_token_hash","loaded_memory_token_hash","output_token_ids_hash",
  "immediate_probe_contract_hash","exogenous_suffix_contract_hash","future_policy_hash","forced_realized_suffix_hash",
  "realized_trajectory_hash","common_initial_state_hash","fixed_policy_hash"}
def _fail(reason):raise ValueError(f"E_QUALIFICATION_FAIL: {reason}")
def _hash(value):return isinstance(value,str) and bool(SHA256.fullmatch(value))

def _validate_vg(rows):
    seen=set()
    for index,row in enumerate(rows):
        required=("stable_id","common_initial_state_hash","fixed_policy_hash","horizon","endpoint_version",
          "endpoint_value","realized_trajectory_hash","future_hash_chain","closed_loop_fixed_policy","policy_generated_all_actions")
        missing=[key for key in required if key not in row]
        if missing:_fail(f"VG row={index} missing={missing}")
        if row["stable_id"] in seen:_fail("VG stable_id must be unique; repeats do not increase n")
        seen.add(row["stable_id"])
        if not all(_hash(row[key]) for key in ("common_initial_state_hash","fixed_policy_hash","realized_trajectory_hash")):_fail("VG invalid hash")
        if row["closed_loop_fixed_policy"] is not True or row["policy_generated_all_actions"] is not True:_fail("VG requires real fixed-policy closed-loop execution")
        if not isinstance(row["horizon"],int) or row["horizon"]<1:_fail("VG invalid horizon")
        if not row["future_hash_chain"] or not all(_hash(x) for x in row["future_hash_chain"]):_fail("VG missing future hash chain")
    return {"status":"VG_QUALIFIED_CLOSED_LOOP_VALUE","estimand_mode":"VG","independent_n":len(rows),
      "shape_a_execution_effect_qualified":False,"closed_loop_policy_value":True,"training_authorized":False}

def validate(rows):
    if not rows:_fail("empty manifest")
    if any("suffix_hash" in row for row in rows):_fail("LEGACY_SUFFIX_SEMANTICS_AMBIGUOUS: migrate to explicit exogenous/forced contract field")
    modes={row.get("estimand_mode") for row in rows}
    if None in modes or len(modes)!=1:_fail(f"exactly one preregistered estimand_mode required, got {modes}")
    mode=next(iter(modes))
    if mode=="VG":return _validate_vg(rows)
    if mode not in MODE_PAIR_KEYS:_fail(f"invalid estimand_mode={mode}")
    pair_keys=BASE_PAIR_KEYS+MODE_PAIR_KEYS[mode];groups=defaultdict(list);seen=set()
    for index,row in enumerate(rows):
        required=pair_keys+E_ARM_KEYS
        missing=[key for key in required if key not in row]
        if missing:_fail(f"row={index} missing={missing}")
        if row["arm"] not in E_ARMS:_fail(f"invalid arm={row['arm']}")
        if row.get("estimand")!=E_LABEL:_fail("manifest is not preregistered E execution effect")
        if not row["stable_id"] or not row["reader_seed_or_coupling_id"] or not row["endpoint_version"]:_fail("empty identity/coupling/endpoint")
        if not isinstance(row["turn"],int) or not isinstance(row["writer_seed"],int):_fail("turn/writer_seed must be integers")
        if any(name in row and not _hash(row[name]) for name in HASH_FIELDS):_fail(f"row={index} invalid SHA-256 field")
        if mode in {"EH","EF"} and (not isinstance(row["horizon"],int) or row["horizon"]<1):_fail("invalid horizon")
        key=tuple(row[name] for name in pair_keys);unique=(key,row["arm"])
        if unique in seen:_fail(f"duplicate pair arm; no last-write-wins: {unique}")
        seen.add(unique);groups[key].append(row)
        if row.get("candidate_materialization_count")!=1 or row.get("writer_runs_after_materialization")!=0:_fail("proposal/writer execution contract violated")
        if not isinstance(row["endpoint_value"],(int,float)) or not math.isfinite(row["endpoint_value"]):_fail("endpoint_value nonfinite")
        if mode=="EH":
            if row.get("allow_arm_trajectory_divergence") is not True:_fail("EH must allow endogenous arm trajectory divergence")
            if row.get("endogenous_future_fields")!=["prompts","candidates","memory_states","decisions"]:_fail("EH endogenous future fields not explicit")
            if not _hash(row.get("realized_trajectory_hash")):_fail("EH missing per-arm realized trajectory hash")
            chain=row.get("future_hash_chain")
            if not isinstance(chain,list) or not chain or not all(_hash(x) for x in chain):_fail("EH invalid per-arm future hash chain")
    for key,pair in groups.items():
        if {row["arm"] for row in pair}!=E_ARMS or len(pair)!=2:_fail(f"same-pair key lacks exact commit/retain arms: {key}")
        commit=next(row for row in pair if row["arm"]=="commit");retain=next(row for row in pair if row["arm"]=="retain")
        if commit["loaded_memory_token_hash"]!=commit["candidate_memory_token_hash"]:_fail("commit did not load candidate memory")
        if retain["loaded_memory_token_hash"]==retain["candidate_memory_token_hash"]:_fail("retain must load old memory while referencing same candidate")
    labels={"E0":"immediate probe execution effect","EH":"regime-conditional total execution effect of one update",
            "EF":"forced-realized-suffix controlled execution effect"}
    return {"status":"E_QUALIFIED","estimand_mode":mode,"estimand":labels[mode],"pair_count":len(groups),
      "pair_keys":list(pair_keys),"realized_trajectory_is_pair_key":False,"shape_a_execution_effect_qualified":mode=="EH",
      "closed_loop_policy_value":False,"state_level_writer_policy_risk_identified":False,
      "training_authorized":False,"select_best_estimand":False}

def self_test():
    from recurrent.research.exact_noop_v2 import materialize_proposal,build_arm_record
    proposal=materialize_proposal(stable_id="e1",turn=1,upstream_state_hash="a"*64,candidate_token_ids=[1,2],
      writer_seed=7,reader_seed_or_coupling_id="r",endpoint_version="v1",old_memory_token_hash="c"*64,
      estimand_mode="EH",mode_contract={"exogenous_suffix_contract_hash":"b"*64,"future_policy_hash":"d"*64,"horizon":3})
    rows=[build_arm_record(proposal,arm="commit",output_token_ids=[3],endpoint_value=1,realized_trajectory_hash="e"*64,future_hash_chain=["f"*64]),
      build_arm_record(proposal,arm="retain",output_token_ids=[4],endpoint_value=0,realized_trajectory_hash="1"*64,future_hash_chain=["2"*64])]
    result=validate(rows);assert result["status"]=="E_QUALIFIED" and not result["realized_trajectory_is_pair_key"]
    for bad in (rows+[dict(rows[0])],[rows[0]],[dict(rows[0],suffix_hash="9"*64),rows[1]]):
        try:validate(bad)
        except ValueError as exc:assert "E_QUALIFICATION_FAIL" in str(exc)
        else:raise AssertionError("invalid horizon replay accepted")
    print("exact_noop_horizon_manifest_self_test=ok")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest");parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--require-shape-a-e",action="store_true");args=parser.parse_args()
    if args.self_test:self_test();return
    if not args.manifest:parser.error("--manifest required")
    raw=Path(args.manifest).read_bytes();rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    result=validate(rows)
    if args.require_shape_a_e and not result["shape_a_execution_effect_qualified"]:_fail("launcher requires EH Shape A execution effect")
    result["manifest_sha256"]=hashlib.sha256(raw).hexdigest();print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
