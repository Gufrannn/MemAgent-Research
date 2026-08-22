from __future__ import annotations
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("orch", REPO / "tools/h20/prd_memrl_orchestrator.py")
orch = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(orch)


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.commit = "a" * 40; self.run_id = "prd-test-0001"
        self.base = self.root / "baseline.json"; self.p0 = self.root / "p0.json"
        dump(self.base, {"status":"PASS", "decision":"PRD_ORIGINAL_BASELINE_IMPORT_PASS",
             "stable_resolved_sha256":"6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411",
             "original_training_final_report":"/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/original_t25_final_report.json",
             "original_training_final_report_sha256":"33cab1eb09eefd89b7f764d0f2c6851eac5e58dc7c0a3d147c30ce05522c9040",
             "original_training_p0_sha256":"c"*64,
             "original_training_resolved_sha256":"b"*64,
             "actual_loss_status":"PENDING_ACTUAL_LOSS_LEDGER",
             "recomputed":{str(anchor):{"token_f1":.5} for anchor in (0,5,10,15,20,25)}})
        self.prior = self.root / "prior"
        self.prior.mkdir()
        prior_config = self.prior / "config.json"
        prior_config.write_text('{"hidden_size":896,"num_hidden_layers":24}')
        dump(self.p0, {
            "status":"PASS", "decision":"PRD_P0_PASS",
            "evidence":{
                "git_commit":self.commit,
                "prior_model":{
                    "id":"Qwen/Qwen2.5-0.5B-Instruct",
                    "revision":"c89bee90d9f811437d9735454613c35b4a3c4dc8",
                    "path":str(self.prior.resolve()),
                    "files":[{"path":"config.json","sha256":hashlib.sha256(prior_config.read_bytes()).hexdigest()}],
                },
            },
        })
        args = type("A", (), dict(run_root=str(self.root/"run"), run_id=self.run_id,
            commit=self.commit, gpu_pair="2,7", baseline=str(self.base), p0=str(self.p0)))
        orch.command_bind(args)

    def tearDown(self): self.tmp.cleanup()

    def assertNoGo(self, fn, text):
        with self.assertRaises(SystemExit) as ctx: fn()
        self.assertIn(text, str(ctx.exception))

    def test_three_capacity_outputs_are_isolated(self):
        for cap in orch.CAPACITIES:
            self.assertTrue((self.root/"run"/"frontier"/f"c{int(cap)}").is_dir())

    def test_full_run_rejects_warmstart_and_capacity_drift(self):
        common = dict(run_root=str(self.root/"run"), run_id=self.run_id, commit=self.commit, stage="full")
        self.assertNoGo(lambda: orch.command_stage(type("A",(),dict(**common,capacity="64",resume=None))), "capacity")
        self.assertNoGo(lambda: orch.command_stage(type("A",(),dict(**common,capacity="128",resume="/tmp/step3"))), "warm-start")

    def test_continuation_rejects_foreign_resume(self):
        args=type("A",(),dict(run_root=str(self.root/"run"),run_id=self.run_id,commit=self.commit,
             stage="continue",capacity="128",resume=str(self.root/"foreign")))
        self.assertNoGo(lambda: orch.command_stage(args), "t5_health.json")

    def test_checkpoint_rejects_method_identity_and_sync_tamper(self):
        cp=self.root/"cp"; components={"actor","actor_optimizer","actor_scheduler","prior","prior_optimizer",
            "prior_scheduler","dual","rng","global_step","frontier_id","weight_sync"}
        meta={"components":sorted(components),"global_step":5,"frontier_id":"c128","git_commit":self.commit,
              "run_id":self.run_id,"method_active":True,"weight_sync":{"verified":False}}
        dump(cp/"prd_checkpoint.json",meta)
        self.assertNoGo(lambda:orch.validate_checkpoint(cp,5,128.,self.commit,self.run_id),"weight sync")
        meta["weight_sync"]["verified"]=True; meta["run_id"]="random-uuid"; dump(cp/"prd_checkpoint.json",meta)
        self.assertNoGo(lambda:orch.validate_checkpoint(cp,5,128.,self.commit,self.run_id),"identity")

    def test_fixed_s128_rejects_duplicate_identity(self):
        path=self.root/"rows.jsonl"
        row={"stable_key":"same","terminal_output":"<answer>x</answer>","ground_truth":"x"}
        path.write_text("".join(json.dumps(row)+"\n" for _ in range(128)))
        self.assertNoGo(lambda:orch.metric_rows(path),"stable identity")

    def test_fixed_s128_recomputes_raw_outputs(self):
        path=self.root/"valid_rows.jsonl"
        rows=({"stable_key":f"k{i}","terminal_output":"Answer: x","ground_truth":"x"} for i in range(128))
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        metrics,keys=orch.metric_rows(path)
        self.assertEqual(len(keys),128); self.assertEqual(metrics["normalized_exact_match"],1.0)

    def test_bound_baseline_tamper_is_detectable(self):
        resolved=json.loads((self.root/"run"/"resolved_run.json").read_text())
        self.base.write_text("{}\n")
        self.assertNotEqual(orch.digest(self.base),resolved["baseline_sha256"])

    def test_bind_rejects_incomplete_or_proxy_baseline(self):
        bad=self.root/"bad-baseline.json"; payload={"status":"PASS","decision":"PRD_ORIGINAL_BASELINE_IMPORT_PASS",
            "stable_resolved_sha256":"6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411",
            "original_training_final_report":"/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/original_t25_final_report.json",
            "original_training_final_report_sha256":"33cab1eb09eefd89b7f764d0f2c6851eac5e58dc7c0a3d147c30ce05522c9040",
            "original_training_p0_sha256":"c"*64,
            "original_training_resolved_sha256":"b"*64,"actual_loss_status":"INFERRED_FROM_AGGREGATE",
            "recomputed":{str(anchor):{} for anchor in (0,5,10,15,20)}}
        dump(bad,payload)
        args=type("A",(),dict(run_root=str(self.root/"bad-run"),run_id="bad",commit=self.commit,
            gpu_pair="2,7",baseline=str(bad),p0=str(self.p0)))
        self.assertNoGo(lambda:orch.command_bind(args),"six-anchor")

    def test_bind_rejects_untrusted_original_training_report(self):
        bad=self.root/"bad-training-report.json"; payload=json.loads(self.base.read_text())
        payload["original_training_final_report_sha256"]="0"*64; dump(bad,payload)
        args=type("A",(),dict(run_root=str(self.root/"bad-training-run"),run_id="bad-training",
            commit=self.commit,gpu_pair="2,7",baseline=str(bad),p0=str(self.p0)))
        self.assertNoGo(lambda:orch.command_bind(args),"authentication")

    def test_ledger_rejects_identity_drift(self):
        ledger=self.root/"ledger.jsonl"; payload=self.root/"payload.json"; dump(payload,{"ok":True})
        tool=REPO/"tools/h20/prd_memrl_ledger.py"
        common=[sys.executable,str(tool),"append","--ledger",str(ledger),"--event","E0","--payload",str(payload)]
        subprocess.run(common+["--run-id",self.run_id,"--git-commit",self.commit],check=True)
        result=subprocess.run(common+["--run-id","different-run","--git-commit",self.commit],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("identity mismatch",result.stderr)

    def test_entry_keeps_dynamic_pair_locks_and_production_runner(self):
        common=(REPO/"scripts/h20/prd_memrl_common.sh").read_text()
        entry=(REPO/"scripts/h20/run_qwen25_7b_prd_memrl.sh").read_text()
        self.assertIn("GPU_PAIR must be N,M",common)
        self.assertIn("first < second",common)
        self.assertIn("memagent_h20_gpu_${first}.lock",common)
        self.assertIn("flock -n 8",common)
        self.assertIn("selected GPU pair is occupied; no process was changed",common)
        self.assertIn("PRD_PRIOR_MODEL must be explicit", entry)
        self.assertIn("FRESH_TOTAL_STEPS=25", entry)
        self.assertIn("train-t25|recover-from-t5", entry)
        self.assertNotIn("t5-gate)", entry)
        self.assertIn("experiments/7b_gate_a/run_gate_a.sh", entry)
        self.assertLess(entry.index("prd_acquire_gpu_locks"), entry.index("experiments/7b_gate_a/run_gate_a.sh"))

    def test_entry_rejects_dirty_wrong_branch_and_wrong_commit(self):
        common=(REPO/"scripts/h20/prd_memrl_common.sh").read_text()
        for guard in ("wrong branch", "wrong exact commit", "dirty worktree"):
            self.assertIn(guard,common)

    def test_t5_health_is_inline_and_s128_free(self):
        trainer=(REPO/"verl/trainer/ppo/ray_trainer.py").read_text()
        self.assertIn('self.global_steps == 5',trainer)
        self.assertIn('PRD_T5_HEALTH_PASS',trainer)
        block=trainer[trainer.index('PRD T5 numerical health NO-GO'):trainer.index('PRD_T5_HEALTH_PASS')]
        self.assertNotIn('_validate()',block)
        self.assertNotIn('fixed_s128',block)


if __name__ == "__main__": unittest.main()
