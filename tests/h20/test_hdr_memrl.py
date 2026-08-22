import json, os, subprocess, sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from recurrent.research.hdr_memrl import *
from tools.h20.hdr_memrl_control import assert_suite_row_identity

def suite(hs=(2,4), roots=3):
    out=[]
    for i in range(roots):
        rid=stable_root_id(dataset_sha256="a"*64,source_index=i,query=f"q{i}")
        for h in hs: out.append(build_horizon_receipt(rid,f"q{i}",list(range(16)),h))
    return out

def test_e0_closes_exactly():
    r=validate_evidence_equated(suite(),[2,4]); assert r["pair_count"]==6

@pytest.mark.parametrize("mutation",["query","evidence","bounds","duplicate","missing"])
def test_e0_adversarial_rejects(mutation):
    rs=suite(roots=1); d=rs[0].as_dict()
    if mutation=="query": d["terminal_query_sha256"]="0"*64
    elif mutation=="evidence": d["evidence_sha256"]="0"*64
    elif mutation=="bounds": d["chunk_bounds"][0][1]-=1
    elif mutation=="duplicate": rs.append(rs[0])
    elif mutation=="missing": rs.pop()
    if mutation in {"query","evidence","bounds"}:
        rs[0]=HorizonReceipt(d["root_id"],d["horizon"],d["terminal_query_sha256"],d["evidence_sha256"],d["evidence_token_count"],tuple(map(tuple,d["chunk_bounds"])),tuple(d["chunk_sha256"]),tuple(tuple(c) for c in d["chunks"]))
    with pytest.raises(HDRContractError): validate_evidence_equated(rs,[2,4])

def test_scheduler_budget_determinism_balance():
    roots=[r.root_id for r in suite(roots=4)[::2]]
    s=BalancedHorizonScheduler([2,4],4,2026)
    a=s.assign(roots,1); assert a==s.assign(roots,1); assert sorted(x["horizon"] for x in a)==[2,2,4,4]

def test_e0_rejects_forged_chunk_hash_and_payload():
    r=suite(roots=1); d=r[0].as_dict(); d["chunk_sha256"][0]="0"*64
    r[0]=HorizonReceipt(d["root_id"],d["horizon"],d["terminal_query_sha256"],d["evidence_sha256"],d["evidence_token_count"],tuple(map(tuple,d["chunk_bounds"])),tuple(d["chunk_sha256"]),tuple(tuple(c) for c in d["chunks"]))
    with pytest.raises(HDRContractError): validate_evidence_equated(r,[2,4])

def test_scheduler_rejects_duplicate_roots_and_budget_drift():
    s=BalancedHorizonScheduler([2,4],2,1)
    with pytest.raises(HDRContractError): s.assign(["x","x"],1)
    with pytest.raises(HDRContractError): s.assign(["x"],1)

def test_dro_upweights_hard_group_and_projects_kl():
    d=OnlineGroupDRO.create([2,4],1.0,.02); state=d.update({2:0.1,4:.9},{2:2,4:2})
    assert state["weights"][1]>state["weights"][0]
    kl=sum(w*__import__('math').log(w/.5) for w in state["weights"]); assert kl<=.0200000001

def test_dro_checkpoint_roundtrip_and_multipliers():
    d=OnlineGroupDRO.create([2,4],.1,.2); d.update({2:.2,4:.8},{2:1,4:1})
    e=OnlineGroupDRO.from_state_dict(d.state_dict()); assert e.weights==d.weights
    ms=e.sample_multipliers([2,2,4,4]); assert abs(sum(ms)-4)<1e-9

@pytest.mark.parametrize("losses,counts",[({2:.1},{2:1,4:0}),({2:.1,4:float('nan')},{2:1,4:1})])
def test_dro_rejects_missing_or_nonfinite(losses,counts):
    with pytest.raises(HDRContractError): OnlineGroupDRO.create([2,4],.1,.2).update(losses,counts)

def test_evaluator_nominal_worst_unseen():
    rows=[]
    for root in ["a","b"]:
        for h,f in [(2,.8),(3,.7),(4,.5)]: rows.append(dict(root_id=root,horizon=h,em=f,token_f1=f,format=1,evidence_equated=True,truncated=False))
    r=evaluate_horizons(rows,2,[3]); assert r["worst_horizon"]==4 and r["unseen"][3]["token_f1"]==.7

@pytest.mark.parametrize("field,value",[("truncated",True),("evidence_equated",False)])
def test_evaluator_fail_closed(field,value):
    row=dict(root_id="r",horizon=2,em=1,token_f1=1,format=1,evidence_equated=True,truncated=False); row[field]=value
    with pytest.raises(HDRContractError): evaluate_horizons([row],2,[])

def test_evaluator_rejects_cross_root_incomplete_horizons():
    rows=[dict(root_id="a",horizon=2,em=1,token_f1=1,format=1,evidence_equated=True,truncated=False),dict(root_id="b",horizon=4,em=1,token_f1=1,format=1,evidence_equated=True,truncated=False)]
    with pytest.raises(HDRContractError): evaluate_horizons(rows,2,[])

def test_entry_contains_required_real_guards():
    common=(ROOT/"scripts/h20/hdr_memrl_common.sh").read_text(); run=(ROOT/"scripts/h20/run_qwen25_7b_hdr_memrl.sh").read_text()
    for needle in ["dirty_tree","wrong_commit","gpu_lock_conflict","gpu_occupied_no_process_killed","PAPER_FRAMING_GO"]: assert needle in common
    for needle in ["fresh_output_exists","hdr_dro_state.json","FRESH_TOTAL_STEPS=25","method_is_one_continuous_fresh_T25_run"]: assert needle in run

def test_no_original_warmstart_or_kill_in_launcher():
    text=(ROOT/"scripts/h20/run_qwen25_7b_hdr_memrl.sh").read_text()
    assert "step3" not in text.lower() and "kill " not in text and "pkill" not in text

def test_manifest_budget_and_fresh_activation():
    m=json.loads((ROOT/"manifests/h20/qwen25_7b_hdr_memrl_seed2026.json").read_text())
    assert m["training"]["source"]=="fresh_base" and m["training"]["first_method_update"]==1
    assert m["budget"]["trajectories_per_update"]==m["budget"]["train_batch_size"]*m["budget"]["rollout_n"]

def test_memory_agent_has_real_hdr_path():
    t=(ROOT/"recurrent/impls/memory.py").read_text(); tr=(ROOT/"verl/trainer/ppo/ray_trainer.py").read_text()
    assert "hdr_bounds" in t and "horizon_id" in t and "OnlineGroupDRO" in tr and "sample_multipliers" in tr

def test_training_gate_does_not_require_original_curve_bundle():
    common=(ROOT/"scripts/h20/hdr_memrl_common.sh").read_text()
    assert "MEMAGENT_HDR_BASELINE_BUNDLE_SHA256" not in common
    assert '"baseline_import.json"' not in common

def test_original_curve_import_is_inventory_bound_and_recomputed():
    control=(ROOT/"tools/h20/hdr_memrl_control.py").read_text()
    for needle in ["curve ledger does not authenticate final report","certified terminal artifact missing","independently recomputed rows digest mismatch","stable-S128 ground truth drift"]:
        assert needle in control
    assert "--bundle" not in control and "--final-report" in control

def test_fixed_s128_suite_requires_frozen_identity_positions():
    prep=(ROOT/"tools/h20/prepare_hdr_horizon_suite.py").read_text()
    evaluator=(ROOT/"tools/h20/run_hdr_strict_vllm_eval.py").read_text()
    runbook=(ROOT/"docs/h20/hdr_memrl_h20_runbook.md").read_text()
    assert "raw_row_position" in prep and "identity_resolved_sha_mismatch" in prep
    assert '"source_order_index"' in evaluator
    assert "--identity-resolved-sha256 6c17c818" in runbook

def test_runbook_trains_once_then_evaluates_all_anchors():
    text=(ROOT/"docs/h20/hdr_memrl_h20_runbook.md").read_text()
    assert text.count("HDR_TARGET_STEP=25 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh")==1
    assert "for HDR_ANCHOR in 5 10 15 20 25" in text
    assert text.index("HDR_TARGET_STEP=25") < text.index("baseline-import") < text.index("for HDR_ANCHOR")

def test_health_rejects_authorized_root_swap_between_source_orders():
    identity={"ground_truth_hash":"a"*64,"raw_row_position":7}
    suite_row={"stable_root_id_receipt":"1"*64,"horizon_id":8,"ground_truth_hash":"a"*64,"raw_row_position":7,"identity_resolved_sha256":"b"*64}
    honest={"root_id":"1"*64,"horizon":8,"stable_id":f"{'1'*64}:h8"}
    assert_suite_row_identity(honest,suite_row,identity,"b"*64)
    swapped={"root_id":"2"*64,"horizon":8,"stable_id":f"{'2'*64}:h8"}
    with pytest.raises(HDRContractError,match="root/order permutation"):
        assert_suite_row_identity(swapped,suite_row,identity,"b"*64)

def test_health_rejects_all_horizon_suite_raw_position_swap():
    identity={"ground_truth_hash":"a"*64,"raw_row_position":7}
    suite_row={"stable_root_id_receipt":"1"*64,"horizon_id":8,"ground_truth_hash":"a"*64,"raw_row_position":9,"identity_resolved_sha256":"b"*64}
    row={"root_id":"1"*64,"horizon":8,"stable_id":f"{'1'*64}:h8"}
    with pytest.raises(HDRContractError,match="raw-position permutation"):
        assert_suite_row_identity(row,suite_row,identity,"b"*64)

def test_all_horizon_metrics_use_frozen_ground_truth_not_row_gold():
    text=(ROOT/"tools/h20/hdr_memrl_control.py").read_text()
    assert 'row.get("gold")!=truth' in text
    assert 'score_terminal_output(row["prediction"],truth)' in text
