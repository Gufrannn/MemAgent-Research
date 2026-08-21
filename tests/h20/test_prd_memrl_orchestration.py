from __future__ import annotations
import importlib.util
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
             "recomputed":{"5":{"token_f1":.5}}})
        dump(self.p0, {"status":"PASS", "decision":"PRD_P0_PASS", "evidence":{"git_commit":self.commit}})
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

    def test_t5_rejects_warmstart_and_capacity_drift(self):
        common = dict(run_root=str(self.root/"run"), run_id=self.run_id, commit=self.commit, stage="t5")
        self.assertNoGo(lambda: orch.command_stage(type("A",(),dict(**common,capacity="64",resume=None))), "capacity")
        self.assertNoGo(lambda: orch.command_stage(type("A",(),dict(**common,capacity="128",resume="/tmp/step3"))), "warm-start")

    def test_continuation_rejects_foreign_resume(self):
        args=type("A",(),dict(run_root=str(self.root/"run"),run_id=self.run_id,commit=self.commit,
             stage="continue",capacity="128",resume=str(self.root/"foreign")))
        self.assertNoGo(lambda: orch.command_stage(args), "t5_gate.json")

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
        self.assertIn("experiments/7b_gate_a/run_gate_a.sh", entry)
        self.assertLess(entry.index("prd_acquire_gpu_locks"), entry.index("experiments/7b_gate_a/run_gate_a.sh"))

    def test_entry_rejects_dirty_wrong_branch_and_wrong_commit(self):
        common=(REPO/"scripts/h20/prd_memrl_common.sh").read_text()
        for guard in ("wrong branch", "wrong exact commit", "dirty worktree"):
            self.assertIn(guard,common)


if __name__ == "__main__": unittest.main()
