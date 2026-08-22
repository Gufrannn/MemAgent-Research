import json, os, subprocess, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]


def test_strict_vllm_evaluator_serializes_numpy_and_writes_atomic_snapshot(tmp_path):
    import importlib.util
    import numpy as np

    script = ROOT / "tools/h20/run_hdr_strict_vllm_eval.py"
    spec = importlib.util.spec_from_file_location("hdr_strict_eval", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = tmp_path / "rows.json.partial"
    module._write_snapshot(
        output,
        [{"gold": np.asarray(["answer"]), "score": np.float64(0.5), "index": np.int64(7)}],
    )
    assert json.loads(output.read_text()) == [
        {"gold": ["answer"], "index": 7, "score": 0.5}
    ]
    assert not output.with_name(output.name + ".tmp").exists()
    with pytest.raises(SystemExit, match="partial_snapshot_metric_mismatch:0"):
        module._verify_metrics({}, {"token_f1": 0.5}, 0)
    with pytest.raises(SystemExit, match="partial_snapshot_metric_mismatch:0"):
        module._verify_metrics({"token_f1": "nan"}, {"token_f1": 0.5}, 0)


def test_hdr_eval_progress_record_conforms_to_ledger_schema(tmp_path):
    import importlib.util
    import jsonschema

    script = ROOT / "tools/h20/run_hdr_strict_vllm_eval.py"
    spec = importlib.util.spec_from_file_location("hdr_strict_eval_schema", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    snapshot = tmp_path / "rows.json.partial"
    module._write_snapshot(snapshot, [{"stable_id": "a" * 64 + ":h2"}])
    ledger = tmp_path / "ledger.jsonl"
    module._append_progress(ledger, snapshot, [{}], "b" * 64, str(tmp_path / "model"), 2026)
    record = json.loads(ledger.read_text())
    schema = json.loads((ROOT / "hdr_execution_ledger.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(record)

sys.path.insert(0,str(ROOT))
from recurrent.research.hdr_memrl import *
from tools.h20.hdr_memrl_control import assert_suite_row_identity


def test_strict_vllm_real_main_smoke_serializes_and_resumes(monkeypatch, tmp_path):
    import importlib.util
    import shutil
    import types
    import numpy as np
    import pandas as pd
    import transformers

    script = ROOT / "tools/h20/run_hdr_strict_vllm_eval.py"
    spec = importlib.util.spec_from_file_location("hdr_strict_eval_main", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs): return cls()
        def encode(self, text, add_special_tokens=False):
            return [i + 1 for i, _ in enumerate(str(text))] or [1]
        def decode(self, tokens, skip_special_tokens=False): return "chunk"
        def apply_chat_template(self, messages, **_kwargs): return messages[-1]["content"]

    class FakeSamplingParams:
        def __init__(self, **kwargs): self.kwargs = kwargs

    class FakeLLM:
        generate_calls = 0
        def __init__(self, **_kwargs): pass
        def generate(self, *_args, **_kwargs):
            type(self).generate_calls += 1
            answer = types.SimpleNamespace(text="\\boxed{answer}", finish_reason="stop")
            return [types.SimpleNamespace(outputs=[answer])]

    rid = "a" * 64
    base_row = {
        "prompt": np.asarray([{"content": "question"}], dtype=object),
        "context": "abcdefgh",
        "extra_info": {"index": 0},
        "stable_root_id_receipt": rid,
        "source_order_index": 0,
        "raw_row_position": 0,
        "identity_resolved_sha256": None,
        "ground_truth_hash": None,
        "reward_model": {"ground_truth": np.asarray(["answer"])},
    }
    frame = pd.DataFrame([{**base_row, "horizon_id": 2}, {**base_row, "horizon_id": 4}])
    suite_path = tmp_path / "suite.parquet"
    suite_path.write_bytes(b"entry-smoke-suite")
    model_path = tmp_path / "model"
    model_path.mkdir()
    output = tmp_path / "result.json"
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams))
    monkeypatch.setitem(sys.modules, "recurrent.impls.memory", types.SimpleNamespace(
        TEMPLATE="Q={prompt} M={memory} C={chunk}",
        TEMPLATE_FINAL_BOXED="Q={prompt} M={memory}",
    ))
    monkeypatch.setitem(sys.modules, "recurrent.utils", types.SimpleNamespace(
        chat_template=lambda _tokenizer: "{message}",
    ))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_a, **_k: "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    monkeypatch.setenv("MEMAGENT_HDR_WORK_ROOT", str(tmp_path))
    argv = [str(script), "--suite", str(suite_path), "--model", str(model_path), "--output", str(output), "--ledger", str(ledger)]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    saved = json.loads(output.read_text())
    assert saved[0]["gold"] == ["answer"]
    assert saved[0]["stable_id"] == f"{rid}:h2"
    assert len(saved) == 2

    partial = output.with_name(output.name + ".partial")
    module._write_snapshot(partial, saved[:1])
    first_progress = ledger.read_text().splitlines()[0]
    ledger.write_text(first_progress + "\n")
    output.unlink()
    FakeLLM.generate_calls = 0
    module.main()
    assert json.loads(output.read_text()) == saved
    assert FakeLLM.generate_calls == 5
    assert not partial.exists()

    for field, value, error in [
        ("stable_id", "forged:h2", "partial_snapshot_identity_mismatch:0"),
        ("gold", ["forged"], "partial_snapshot_identity_mismatch:0"),
        ("total_input_tokens", 1, "partial_snapshot_control_mismatch:0"),
        ("token_f1", 0.123, "partial_snapshot_metric_mismatch:0"),
        ("token_f1", "nan", "partial_snapshot_metric_mismatch:0"),
    ]:
        tampered = [dict(saved[0])]
        tampered[0][field] = value
        module._write_snapshot(partial, tampered)
        ledger.write_text(first_progress + "\n")
        if output.exists(): output.unlink()
        with pytest.raises(SystemExit, match="partial_snapshot_ledger_mismatch"): module.main()

    missing_metric = [dict(saved[0])]
    del missing_metric[0]["token_f1"]
    module._write_snapshot(partial, missing_metric)
    module._append_progress(ledger, partial, missing_metric, saved[0]["suite_sha256"], saved[0]["model_path"], saved[0]["seed"])
    with pytest.raises(SystemExit, match="partial_snapshot_metric_mismatch:0"): module.main()

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
