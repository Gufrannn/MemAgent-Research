import json
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class CoralFrozenContractTests(unittest.TestCase):
    def test_e0_exact_shared_policy_counterexample(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"e0.json"
            completed=subprocess.run(
                ["python3",str(ROOT/"tools/h20/coral_e0.py"),"--output",str(output)],
                text=True,capture_output=True,
            )
            self.assertEqual(completed.returncode,0,completed.stderr)
            report=json.loads(output.read_text())
            self.assertEqual(report["decision"],"CORAL_E0_PASS")
            self.assertEqual(report["loss_aggregation"],
                             "token-mean over the same two-token Original denominator")
            self.assertTrue(report["shared_parameter"])
            self.assertTrue(report["occupancy_response_sign_flip"])
            self.assertLess(float(report["simultaneous_masked_grpo"]["return"]),
                            float(report["old"]["return"]))
            self.assertGreater(float(report["coral"]["return"]),
                               float(report["old"]["return"]))

    def test_manifest_enables_update_one_and_preserves_original_batch(self):
        manifest=json.loads((ROOT/"manifests/h20/qwen25_7b_cosi_seed2026.json").read_text())
        self.assertEqual(manifest["branch"],"h20/qwen25-7b-cosi-t25-frozen-20260822")
        self.assertTrue(manifest["fresh_base_only"]);self.assertEqual(manifest["method_active_from_update"],1)
        self.assertEqual((manifest["training"]["train_batch_size"],manifest["training"]["rollout_n"],manifest["training"]["ppo_mini_batch_size"]),(4,2,4))
        self.assertEqual(manifest["training"]["anchors"],[5,10,15,20,25])
        self.assertTrue(manifest["coral"]["enabled"])
        self.assertFalse(manifest["training_authorized"])
        self.assertEqual(
            manifest["authorization_mode"], "external_authenticated_gates_only",
        )
        from recurrent.research.coral_e1 import SKETCH_BASIS_SHA256
        diagnostic = manifest["occupancy_response_diagnostic"]
        self.assertEqual(
            diagnostic["gradient_sketch_basis_sha256"],
            SKETCH_BASIS_SHA256,
        )
        self.assertNotIn("candidate_sampling", diagnostic)
        self.assertEqual(
            diagnostic["terminal_action_policy"],
            "both_branches_freshly_sampled_at_fixed_proposal_weights",
        )
        self.assertEqual(
            diagnostic["inference_unit"],
            "writer_proposal_mean_over_four_never_reused_roots",
        )

    def test_launchers_use_dynamic_per_gpu_locks_and_never_kill(self):
        common=(ROOT/"scripts/h20/cosi_common.sh").read_text()
        self.assertIn("memagent_h20_gpu_${COSI_GPU_A}.lock",common)
        self.assertIn("memagent_h20_gpu_${COSI_GPU_B}.lock",common)
        self.assertIn("nvidia-smi -i \"$MEMAGENT_COSI_GPU_PAIR\"",common)
        for forbidden in ("kill -9","pkill","killall","CUDA_VISIBLE_DEVICES=6,7"):
            self.assertNotIn(forbidden,common)

    def test_t5_is_fresh_and_continuation_is_exact_t5(self):
        t5=(ROOT/"scripts/h20/run_qwen25_7b_cosi_t5.sh").read_text()
        cont=(ROOT/"scripts/h20/resume_qwen25_7b_cosi_t5_to_t25.sh").read_text()
        self.assertIn("PHASE=fresh",t5);self.assertNotIn("Original3",t5)
        self.assertIn("RESUME_FROM=$OUTPUT/global_step_5",cont)
        self.assertIn("RESUME_TOTAL_STEPS=25",cont)
        self.assertIn("--stage continue",cont)

    def test_e1_real_entry_owns_measurements_and_checkpoint_binding(self):
        entry=(ROOT/"scripts/h20/run_qwen25_7b_coral_e1_producer.sh").read_text()
        trainer=(ROOT/"verl/trainer/ppo/ray_trainer.py").read_text()
        worker=(ROOT/"verl/workers/fsdp_workers.py").read_text()
        sealer=(ROOT/"tools/h20/seal_coral_e1.py").read_text()
        self.assertIn("CORAL_E1_CAPTURE_DIR",entry)
        self.assertIn("FRESH_TOTAL_STEPS=15",entry)
        self.assertIn("_coral_e1_regenerate",trainer)
        self.assertIn("_coral_e1_measure_root",trainer)
        self.assertIn("measure_coral_role_gradient",worker)
        self.assertIn("measurement mutated optimizer state",worker)
        self.assertIn("checkpoint_inventory",sealer)
        self.assertIn("coral_e1_fsdp_sketch_oracle.py",entry)
        self.assertIn("torch.distributed.run",entry)
        self.assertIn("_coral_e1_resample_terminal",trainer)
        self.assertIn("both terminal branches must be proposal-resampled",trainer)
        self.assertNotIn("MEASUREMENT_ROOT",entry)

    def test_trainer_wires_role_phase_without_fictitious_anchor(self):
        trainer=(ROOT/"verl/trainer/ppo/ray_trainer.py").read_text()
        actor=(ROOT/"verl/workers/actor/dp_actor.py").read_text()
        self.assertIn('batch.meta_info["coral_phase"]',trainer)
        self.assertIn('CORAL_EXECUTION_LEDGER',trainer)
        self.assertIn('batch["response_mask"] = active_mask',actor)
        self.assertIn('coral_full_token_total',actor)
        self.assertIn('denominator_scale = response_mask.sum() / coral_full_token_total',actor)
        self.assertIn('role_covered_order',trainer)
        self.assertIn('actor_vllm_sampled_tensor_digest',trainer)
        self.assertNotIn('coral_inactive_kl_coef',actor)
        self.assertNotIn('coral/inactive_kl_loss',trainer)

    def test_t5_preflight_requires_external_gates_without_self_authorization(self):
        preflight=(ROOT/"tools/h20/preflight_qwen25_7b_cosi.py").read_text()
        runbook=(ROOT/"docs/h20/cosi_research_closure_20260822.md").read_text()
        self.assertIn('manifest.get("training_authorized") is not False', preflight)
        self.assertIn('external_authenticated_gates_only', preflight)
        for variable in (
            "MEMAGENT_COSI_PAPER_REVIEW_SHA256",
            "MEMAGENT_COSI_E0_REPORT_SHA256",
            "MEMAGENT_COSI_E1_REPORT_SHA256",
            "MEMAGENT_COSI_BASELINE_REPORT_SHA256",
        ):
            self.assertIn(variable, preflight)
            self.assertIn(f"export {variable}=<EXTERNALLY_ISSUED_", runbook)
            self.assertNotIn(f"export {variable}=$(shasum", runbook)
        self.assertIn("must arrive through a trusted channel independent", runbook)

    def test_source_firewall(self):
        paths=[ROOT/"recurrent/research/coral.py",ROOT/"docs/papers/coral_paper_draft.md"]
        text="\n".join(path.read_text() for path in paths)
        for forbidden in ("CCOD","BOPR","NCR","gold answer as input","future chunk as input"):
            self.assertNotIn(forbidden,text)

    def test_real_shell_entry_rejects_pair_commit_and_dirty_tree(self):
        common=ROOT/"scripts/h20/cosi_common.sh"
        base={**os.environ,
              "MEMAGENT_COSI_WORK_ROOT":str(ROOT.parent),
              "MEMAGENT_COSI_REPO_DIR":str(ROOT),
              "MEMAGENT_COSI_EXPECTED_COMMIT":subprocess.check_output(
                  ["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
              "MEMAGENT_COSI_GPU_PAIR":"2,1"}
        reversed_pair=subprocess.run(
            ["bash","-c",f'source "{common}"'],env=base,text=True,capture_output=True
        )
        self.assertNotEqual(reversed_pair.returncode,0)
        self.assertIn("gpu_pair_not_canonical",reversed_pair.stderr)

        base["MEMAGENT_COSI_GPU_PAIR"]="0,1"
        base["MEMAGENT_COSI_EXPECTED_COMMIT"]="0"*40
        wrong_commit=subprocess.run(
            ["bash","-c",f'source "{common}"; cosi_checkout_guard'],
            env=base,text=True,capture_output=True,
        )
        self.assertNotEqual(wrong_commit.returncode,0)
        self.assertIn("wrong_commit",wrong_commit.stderr)

        base["MEMAGENT_COSI_EXPECTED_COMMIT"]=subprocess.check_output(
            ["git","-C",str(ROOT),"rev-parse","HEAD"],text=True
        ).strip()
        dirty=subprocess.run(
            ["bash","-c",f'source "{common}"; cosi_checkout_guard'],
            env=base,text=True,capture_output=True,
        )
        self.assertNotEqual(dirty.returncode,0)
        self.assertIn("dirty_tree",dirty.stderr)

if __name__=="__main__":unittest.main()
