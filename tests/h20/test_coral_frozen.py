import copy
import json
import os
import subprocess
import sys
import tempfile
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
        self.assertEqual(diagnostic["schema"], "memagent.coral.e1.v4")
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

    def test_original_protocol_leaf_copy_and_drift_rejection(self):
        from tools.h20.preflight_qwen25_7b_cosi import validate_original_protocol
        method=json.loads((ROOT/"manifests/h20/qwen25_7b_cosi_seed2026.json").read_text())
        original=json.loads((ROOT/"manifests/h20/qwen25_7b_original_t25_seed2026.json").read_text())
        receipt=validate_original_protocol(method, original)
        self.assertGreaterEqual(len(receipt["compared_leaves"]), 25)
        self.assertIn("tokenizer.json", receipt["original_model_file_inventory"])
        mutations=[]
        value=json.loads(json.dumps(original))
        value["training"]["max_num_seqs"] += 1
        mutations.append(value)
        value=json.loads(json.dumps(original))
        value["backend"]["reward_manager"]="forged"
        mutations.append(value)
        value=json.loads(json.dumps(original))
        value["model"]["files"]=[
            row for row in value["model"]["files"] if row["path"] != "tokenizer.json"
        ]
        mutations.append(value)
        for value in mutations:
            with self.subTest():
                with self.assertRaisesRegex(ValueError,"COSI_NO_GO"):
                    validate_original_protocol(method,value)

    def test_real_runner_emits_frozen_coral_protocol(self):
        runner=ROOT/"experiments/7b_gate_a/run_gate_a.sh"
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            model=work/"model"; model.mkdir(); (model/"config.json").write_text("{}")
            train=work/"train.parquet"; train.write_bytes(b"train")
            validation=work/"validation.parquet"; validation.write_bytes(b"validation")
            env={**os.environ,"WORK_ROOT":str(work),"CODE":str(ROOT),
                 "MODEL":str(model),"TRAIN":str(train),"VAL":str(validation),
                 "PYTHON":sys.executable,"EMIT_TRAINER_OVERRIDES":"1",
                 "CUDA_VISIBLE_DEVICES":"2,7","TRAIN_BATCH_SIZE":"4",
                 "ROLLOUT_N":"2","PPO_MINI_BATCH_SIZE":"4","N_GPUS":"2",
                 "FSDP_SIZE":"2","RUN_SEED":"2026"}
            coral_overrides = [
                "+algorithm.coral.enabled=true",
                "+algorithm.coral.active_from_update=1",
                "+algorithm.coral.schedule=odd_writer_even_terminal_answer_v2",
                "+algorithm.coral.role_partition=nonfinal_memory_writer_vs_final_answer",
                "+algorithm.coral.require_recurrent=true",
                "+algorithm.coral.require_grpo=true",
                "+algorithm.coral.require_gate_a_sync=true",
                "trainer.project_name=memagent_coral",
            ]
            reference=subprocess.run(
                ["bash",str(runner)], env=env,text=True,capture_output=True
            )
            self.assertEqual(reference.returncode,0,reference.stderr)
            reference_overrides=json.loads(reference.stdout.splitlines()[-1])
            completed=subprocess.run(
                ["bash",str(runner),*coral_overrides],env=env,text=True,capture_output=True
            )
            self.assertEqual(completed.returncode,0,completed.stderr)
            overrides=json.loads(completed.stdout.splitlines()[-1])
            # The real Method entry must be the accepted Original argv plus
            # exactly the pre-registered CORAL whitelist, in the same order.
            self.assertEqual(overrides[:len(reference_overrides)],reference_overrides)
            self.assertEqual(overrides[len(reference_overrides):],coral_overrides)
            required={
                "algorithm.adv_estimator=grpo","algorithm.grpo_use_adv=False",
                "actor_rollout_ref.rollout.n=2","+actor_rollout_ref.rollout.seed=2026",
                "+actor_rollout_ref.rollout.trajectory_seed_mode=independent",
                "data.train_batch_size=4","actor_rollout_ref.actor.ppo_mini_batch_size=4",
                "actor_rollout_ref.actor.optim.lr=1e-6",
                "actor_rollout_ref.actor.kl_loss_coef=0.001",
                "reward_model.reward_manager=naive",
                "+custom_reward_function.reward_kwargs.f1_weight=0.95",
                "+custom_reward_function.reward_kwargs.grounded_box_bonus=0.05",
                "+algorithm.coral.enabled=true","trainer.project_name=memagent_coral",
            }
            self.assertTrue(required.issubset(set(overrides)),required-set(overrides))
            config=(ROOT/"verl/trainer/config/ppo_trainer.yaml").read_text()
            self.assertIn("ppo_epochs: 1",config)
            self.assertIn('loss_agg_mode: "token-mean"',config)

    def test_launchers_use_dynamic_per_gpu_locks_and_never_kill(self):
        common=(ROOT/"scripts/h20/cosi_common.sh").read_text()
        self.assertIn("memagent_h20_gpu_${COSI_GPU_A}.lock",common)
        self.assertIn("memagent_h20_gpu_${COSI_GPU_B}.lock",common)
        self.assertIn("nvidia-smi -i \"$MEMAGENT_COSI_GPU_PAIR\"",common)
        for forbidden in ("kill -9","pkill","killall","CUDA_VISIBLE_DEVICES=6,7"):
            self.assertNotIn(forbidden,common)

    def test_t5_is_fresh_and_continuation_is_exact_t5(self):
        t5=(ROOT/"scripts/h20/run_qwen25_7b_cosi_t25.sh").read_text()
        cont=(ROOT/"scripts/h20/resume_qwen25_7b_cosi_t5_to_t25.sh").read_text()
        self.assertIn("PHASE=fresh",t5);self.assertIn("FRESH_TOTAL_STEPS=25",t5)
        self.assertIn("for step in 5 10 15 20 25",t5);self.assertNotIn("Original3",t5)
        self.assertIn("RESUME_FROM=$OUTPUT/global_step_5",cont)
        self.assertIn("RESUME_TOTAL_STEPS=25",cont)
        self.assertIn("--stage continue",cont)
        self.assertIn("--exact-boundary",cont)
        old_t5=(ROOT/"scripts/h20/run_qwen25_7b_cosi_t5.sh").read_text()
        old_audit=(ROOT/"scripts/h20/audit_qwen25_7b_coral_t5.sh").read_text()
        self.assertIn("superseded",old_t5)
        self.assertIn("superseded",old_audit)

    def test_e1_real_entry_owns_measurements_and_checkpoint_binding(self):
        entry=(ROOT/"scripts/h20/run_qwen25_7b_coral_e1_producer.sh").read_text()
        trainer=(ROOT/"verl/trainer/ppo/ray_trainer.py").read_text()
        protocol=(ROOT/"verl/protocol.py").read_text()
        worker=(ROOT/"verl/workers/fsdp_workers.py").read_text()
        sealer=(ROOT/"tools/h20/seal_coral_e1.py").read_text()
        self.assertIn("CORAL_E1_CAPTURE_DIR",entry)
        self.assertIn("FRESH_TOTAL_STEPS=15",entry)
        self.assertIn("_coral_e1_regenerate",trainer)
        self.assertIn("_coral_e1_measure_root",trainer)
        self.assertIn("measure_coral_role_gradient",worker)
        self.assertIn("measurement mutated optimizer state",worker)
        self.assertIn("checkpoint_inventory",sealer)
        self.assertIn('parser.add_argument("--dataproto-clone-oracle", required=True)',sealer)
        self.assertIn('"dataproto_clone_oracle_report": clone_oracle',sealer)
        self.assertIn("validate_dataproto_clone_oracle_report",sealer)
        self.assertIn("coral_e1_fsdp_sketch_oracle.py",entry)
        self.assertIn("torch.distributed.run",entry)
        self.assertIn("coral_dataproto_clone_oracle.py",entry)
        self.assertIn("--dataproto-clone-oracle",entry)
        self.assertIn("coral_e1_seed2026_v5",entry)
        self.assertIn("coral_e1_seed2026_v3|coral_e1_seed2026_v4",entry)
        self.assertIn("_coral_e1_resample_terminal",trainer)
        self.assertIn("both terminal branches must be proposal-resampled",trainer)
        self.assertIn("def clone(self) -> \"DataProto\"",protocol)
        self.assertIn('"original_batch": original_batch.clone()',trainer)
        self.assertNotIn('"original_batch": deepcopy(original_batch)',trainer)
        self.assertNotIn("deepcopy(original_batch)",trainer)
        self.assertNotIn("MEASUREMENT_ROOT",entry)

        runbook=(ROOT/"docs/h20/cosi_research_closure_20260822.md").read_text()
        self.assertIn("MEMAGENT_COSI_E1_RUN_ID=coral_e1_seed2026_v5",runbook)
        self.assertNotIn("MEMAGENT_COSI_E1_RUN_ID=coral_e1_seed2026_v3",runbook)

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
        self.assertIn("MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST_SHA256", preflight)
        self.assertIn("<READ_ONLY_FROZEN_ORIGINAL_T25_RESOLVED_64HEX>", runbook)
        self.assertIn(
            "MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256="
            "6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411",
            runbook,
        )
        self.assertIn("must arrive through a trusted channel independent", runbook)

    def test_exhaustive_resolved_original_diff_allows_only_explicit_whitelist(self):
        from tools.h20.preflight_qwen25_7b_cosi import exhaustive_resolved_config_diff
        original = {
            "algorithm": {"adv_estimator": "grpo"},
            "custom_reward_function": {"path": "/original/reward.py"},
            "actor_rollout_ref": {"actor": {"optim": {"lr": 1e-6}}},
            "trainer": {
                "project_name": "original", "experiment_name": "original-run",
                "default_local_dir": "/original", "total_training_steps": 25,
                "save_freq": 5, "max_actor_ckpt_to_keep": 5,
                "resume_mode": "resume_path", "resume_from_path": "/step3",
            },
        }
        method = copy.deepcopy(original)
        method["algorithm"]["coral"] = {
            "enabled": True, "active_from_update": 1,
        }
        method["custom_reward_function"]["path"] = "/coral/reward.py"
        method["trainer"].update({
            "project_name": "coral", "experiment_name": "coral-run",
            "default_local_dir": "/coral", "total_training_steps": 5,
            "save_freq": 1, "max_actor_ckpt_to_keep": 30,
            "resume_mode": "disable", "resume_from_path": None,
        })
        observed = exhaustive_resolved_config_diff(original, method)
        self.assertIn("algorithm.coral", observed)
        drift = copy.deepcopy(method)
        drift["actor_rollout_ref"]["actor"]["optim"]["lr"] = 2e-6
        with self.assertRaisesRegex(ValueError, "outside explicit whitelist"):
            exhaustive_resolved_config_diff(original, drift)

    def test_continuation_rejects_gpu_commit_and_manifest_drift(self):
        from tools.h20.preflight_qwen25_7b_cosi import validate_continuation_binding
        value = {
            "stage": "t5",
            "git_commit": "a" * 40,
            "manifest_sha256": "b" * 64,
            "original_resolved_manifest_sha256": "c" * 64,
            "original_p0_certificate_sha256": "6" * 64,
            "s128_resolved_manifest_sha256": "d" * 64,
            "fresh_base_model_tokenizer_inventory_sha256": "e" * 64,
            "original_protocol_comparison_sha256": "f" * 64,
            "method_nonwhitelist_config_sha256": "2" * 64,
            "evidence_authority_sha256": "1" * 64,
            "gpu_pair": [2, 7],
        }
        arguments = {
            "expected_commit": "a" * 40,
            "manifest_sha256": "b" * 64,
            "original_manifest_sha256": "c" * 64,
            "original_p0_certificate_sha256": "6" * 64,
            "s128_manifest_sha256": "d" * 64,
            "model_inventory_sha256": "e" * 64,
            "protocol_comparison_sha256": "f" * 64,
            "method_nonwhitelist_config_sha256": "2" * 64,
            "evidence_authority_sha256": "1" * 64,
            "gpu_pair": [2, 7],
        }
        validate_continuation_binding(value, **arguments)
        for field, replacement in (
            ("git_commit", "d" * 40),
            ("manifest_sha256", "e" * 64),
            ("original_resolved_manifest_sha256", "f" * 64),
            ("original_p0_certificate_sha256", "7" * 64),
            ("s128_resolved_manifest_sha256", "1" * 64),
            ("fresh_base_model_tokenizer_inventory_sha256", "2" * 64),
            ("original_protocol_comparison_sha256", "3" * 64),
            ("method_nonwhitelist_config_sha256", "5" * 64),
            ("evidence_authority_sha256", "4" * 64),
            ("gpu_pair", [3, 7]),
        ):
            tampered = dict(value)
            tampered[field] = replacement
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "COSI_NO_GO"):
                    validate_continuation_binding(tampered, **arguments)

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
