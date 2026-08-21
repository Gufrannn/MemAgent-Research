#!/usr/bin/env python3
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from recurrent.research.s128_hotpot_metrics import score_terminal_output,summarize_fixed_s128
from recurrent.research.stable_eval_identity import validate_attempt_identity_rows,validate_resolved_manifest
from tools.h20.audit_qwen25_7b_s128_it import _ground_truth_by_source_order

def main():
    p=argparse.ArgumentParser(); p.add_argument("--eval-root",required=True); p.add_argument("--step",type=int,choices=[5,10,15,20,25],required=True); p.add_argument("--validation",required=True); p.add_argument("--resolved-manifest",required=True); p.add_argument("--expected-manifest-sha256",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    resolved_path=Path(a.resolved_manifest)
    if hashlib.sha256(resolved_path.read_bytes()).hexdigest()!=a.expected_manifest_sha256: raise SystemExit("RWWPO_S128_AUDIT_NO_GO:identity manifest SHA")
    resolved=validate_resolved_manifest(json.loads(resolved_path.read_text()))
    terminal_path=Path(a.eval_root)/"terminal"/f"{a.step}.jsonl"; summary_path=Path(a.eval_root)/"execution_summary.json"
    if not terminal_path.is_file() or not summary_path.is_file(): raise SystemExit("RWWPO_S128_AUDIT_NO_GO:missing terminal/summary")
    rows=[json.loads(x) for x in terminal_path.read_text().splitlines() if x.strip()]
    validate_attempt_identity_rows(rows,examples=128,replicas=1)
    if [int(r["source_order_index"]) for r in rows]!=list(range(128)): raise SystemExit("RWWPO_S128_AUDIT_NO_GO:source order")
    if any(r["eval_manifest_hash"]!=resolved["eval_manifest_hash"] or int(r["step"])!=a.step for r in rows): raise SystemExit("RWWPO_S128_AUDIT_NO_GO:identity/step")
    summary=json.loads(summary_path.read_text())
    if any(int(summary.get(key,-1))!=0 for key in ("actor_update_calls","optimizer_step_calls","checkpoint_save_calls")): raise SystemExit("RWWPO_S128_AUDIT_NO_GO:mutation")
    truth=_ground_truth_by_source_order({"data":{"validation":a.validation}},resolved)
    metrics=[score_terminal_output(row["output"],truth[int(row["source_order_index"])]) for row in rows]
    aggregate=summarize_fixed_s128(metrics)
    report={"status":"PASS","decision":f"RWWPO_T{a.step}_S128_PASS","step":a.step,"metrics":aggregate,"terminal_sha256":hashlib.sha256(terminal_path.read_bytes()).hexdigest(),"resolved_manifest_sha256":a.expected_manifest_sha256}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
if __name__=="__main__": main()
