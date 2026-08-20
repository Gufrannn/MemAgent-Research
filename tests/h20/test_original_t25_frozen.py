from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from recurrent.research.trajectory_seeding import (
    build_trajectory_seed_records,
    derive_turn_request_seeds,
)
from tools.h20 import audit_qwen25_7b_original_t25 as t25_audit
from tools.h20.audit_qwen25_7b_original_t25 import (
    _audit_exact_turn_schedule,
    _checkpoint_anchor_evidence,
    _complete_checkpoint_steps,
    _failures_for_persisted_anchor_record,
    _failures_for_resume_state_acks,
)
from tools.h20.preflight_qwen25_7b_original_t25 import (
    EXPECTED_BASE_COMMIT,
    assert_only_t25_config_differences,
    canonical_sha256,
    ledger_prefix_sha256,
    validate_contract,
    validate_resolved_t25_config,
)


MANIFEST_PATH = REPO / "manifests/h20/qwen25_7b_original_t25_seed2026.json"
COMMAND_PATH = REPO / "manifests/h20/qwen25_7b_original_t25_commands.json"


class OriginalT25FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.commands = json.loads(COMMAND_PATH.read_text(encoding="utf-8"))

    def test_scope_is_corrected_original_style_pilot_not_paper_reproduction(self) -> None:
        self.assertEqual(
            self.manifest["study"]["label"], "corrected Original-style 2-GPU pilot"
        )
        self.assertIs(self.manifest["study"]["not_original_paper_7b_reproduction"], True)
        self.assertIn("32-GPU", self.manifest["study"]["paper_scale_difference"])
        self.assertEqual(self.manifest["base_commit"], EXPECTED_BASE_COMMIT)
        self.assertEqual(len(self.manifest["base_commit"]), 40)

    def test_stable_i_and_gate_a_are_hard_prerequisites(self) -> None:
        stable = self.manifest["stable_identity_prerequisite"]
        self.assertEqual(stable["commit"], EXPECTED_BASE_COMMIT)
        self.assertEqual(stable["required_status"], "PASS")
        self.assertEqual(
            stable["required_decision"], "I_RECURRENT_IDENTITY_CANARY_PASS"
        )
        self.assertEqual(
            stable["eval_manifest_hash"],
            "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a",
        )
        self.assertEqual(
            stable["execution_ledger_prefix_sha256"],
            "cace28d198f0dfa040c9c00973885ee77b24cb18c079162e77b4b84e6330b136",
        )
        self.assertEqual(stable["execution_ledger_prefix_record_count"], 5)
        self.assertEqual(stable["execution_ledger_total_record_count"], 6)
        self.assertIn("stable_i4x2_frozen_20260821r2", stable["final_report"])
        prefix = [
            json.dumps({"record_index": index}, sort_keys=True, separators=(",", ":"))
            for index in range(5)
        ]
        expected_prefix_sha = hashlib.sha256(
            ("\n".join(prefix) + "\n").encode("utf-8")
        ).hexdigest()
        self.assertEqual(ledger_prefix_sha256([*prefix, '{"audit":true}'], 5), expected_prefix_sha)
        with self.assertRaisesRegex(ValueError, "requires 5 records"):
            ledger_prefix_sha256(prefix[:4], 5)
        source = self.manifest["source_gate_a"]
        self.assertEqual(source["global_step"], 3)
        self.assertEqual(
            source["final_report_sha256"],
            "5f8b67b496bd672cb6e89c9ec481c1de97adbf0a73c3459edd02aef79830dca4",
        )

    def test_exact_step_and_cursor_budget(self) -> None:
        training = self.manifest["training"]
        self.assertEqual(training["update_steps"], list(range(4, 26)))
        self.assertEqual(training["update_count"], 22)
        self.assertEqual(training["primary_scientific_endpoint"], 25)
        self.assertEqual(
            training["secondary_learning_curve_anchor_steps"], [5, 10, 15, 20]
        )
        self.assertEqual(training["technical_checkpoint_steps"], [5, 10, 15, 20, 25])
        self.assertEqual(
            training["expected_retained_complete_actor_checkpoints"],
            [5, 10, 15, 20, 25],
        )
        data = self.manifest["data"]
        self.assertEqual(data["source_consumed_prompt_count"], 12)
        self.assertEqual(data["continuation_source_order_start"], 12)
        self.assertEqual(data["continuation_source_order_stop_exclusive"], 100)
        self.assertEqual(data["continuation_prompt_count"], 88)
        self.assertEqual(88 * training["rollout_n"], 176)
        self.assertEqual(self.manifest["storage"]["retained_checkpoint_multiplier"], 5)
        self.assertEqual(self.manifest["storage"]["safety_margin_bytes"], 20 * 1024**3)

    def test_actor_retention_preserves_all_five_preregistered_anchors(self) -> None:
        manager_source = (
            REPO / "verl/utils/checkpoint/fsdp_checkpoint_manager.py"
        ).read_text(encoding="utf-8")
        trainer_source = (REPO / "verl/trainer/ppo/ray_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])",
            manager_source,
        )
        self.assertIn(
            'actor_local_path = os.path.join(local_global_step_folder, "actor")',
            trainer_source,
        )
        self.assertIn(
            "self.actor_rollout_wg.save_checkpoint(actor_local_path",
            trainer_source,
        )
        self.assertIn("shutil.rmtree(abs_path, ignore_errors=True)", (
            REPO / "verl/utils/checkpoint/checkpoint_manager.py"
        ).read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous_actor_paths: list[str] = []
            max_keep = 5
            for step in (5, 10, 15, 20, 25):
                step_root = root / f"global_step_{step}"
                actor = step_root / "actor"
                actor.mkdir(parents=True)
                (actor / "model.pt").write_bytes(b"actor")
                (step_root / "data.pt").write_bytes(b"cursor")
                if len(previous_actor_paths) >= max_keep:
                    keep_start = len(previous_actor_paths) - max_keep + 1
                    for actor_path in previous_actor_paths[:keep_start]:
                        shutil.rmtree(actor_path, ignore_errors=True)
                    previous_actor_paths = previous_actor_paths[keep_start:]
                previous_actor_paths.append(str(actor))

            complete = [
                step
                for step in (5, 10, 15, 20, 25)
                if (root / f"global_step_{step}" / "actor").is_dir()
            ]
            data_only = [
                step
                for step in (5, 10, 15, 20, 25)
                if not (root / f"global_step_{step}" / "actor").exists()
                and (root / f"global_step_{step}" / "data.pt").is_file()
            ]
            self.assertEqual(complete, [5, 10, 15, 20, 25])
            self.assertEqual(data_only, [])

    def test_gate_a_learning_configuration_is_frozen(self) -> None:
        self.assertEqual(validate_contract(self.manifest), [])
        training = self.manifest["training"]
        self.assertEqual(training["train_batch_size"], 4)
        self.assertEqual(training["rollout_n"], 2)
        self.assertEqual(training["ppo_mini_batch_size"], 4)
        self.assertEqual(training["actor_lr_warmup_steps"], 2)
        self.assertEqual(training["actor_learning_rate"], 1e-6)
        self.assertEqual(self.manifest["backend"]["reward_manager"], "naive")
        self.assertEqual(self.manifest["backend"]["rollout"], "vllm")
        self.assertIs(self.manifest["backend"]["allow_hf_fallback"], False)

    def test_only_allowlisted_hydra_differences_are_accepted(self) -> None:
        reference = {
            "trainer": {
                "experiment_name": "gate-a",
                "default_local_dir": "/gate",
                "total_training_steps": 3,
                "save_freq": 1,
                "max_actor_ckpt_to_keep": 3,
                "resume_from_path": "/step2",
            },
            "algorithm": {"adv_estimator": "grpo"},
            "reward_model": {"reward_manager": "naive"},
        }
        candidate = json.loads(json.dumps(reference))
        candidate["trainer"].update(
            experiment_name="t25",
            default_local_dir="/t25",
            total_training_steps=25,
            save_freq=5,
            max_actor_ckpt_to_keep=5,
            resume_from_path="/step3",
        )
        assert_only_t25_config_differences(reference, candidate)
        candidate["reward_model"]["reward_manager"] = "thread"
        with self.assertRaisesRegex(ValueError, "differs from Gate A outside"):
            assert_only_t25_config_differences(reference, candidate)

    def test_resolved_config_projection_fails_closed_on_missing_science_fields(self) -> None:
        failures = validate_resolved_t25_config(self.manifest, {})
        self.assertTrue(any("algorithm.adv_estimator" in item for item in failures))
        self.assertTrue(any("reward_model.reward_manager" in item for item in failures))
        self.assertTrue(any("actor_rollout_ref.rollout.name" in item for item in failures))
        self.assertTrue(any("actor.optim.lr_warmup_steps" in item for item in failures))

    def test_canonical_checkpoint_inventory_hash_contract(self) -> None:
        inventory = [
            {"path": "actor/model_world_size_2_rank_0.pt", "size": 11, "sha256": "a" * 64},
            {"path": "data.pt", "size": 3, "sha256": "b" * 64},
        ]
        expected = hashlib.sha256(
            json.dumps(
                inventory,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(canonical_sha256(inventory), expected)

    def test_secondary_checkpoint_anchor_tamper_is_detected(self) -> None:
        inventories = {
            step: [
                {
                    "path": "actor/model_world_size_2_rank_0.pt",
                    "size": step,
                    "sha256": f"{step:064x}",
                }
            ]
            for step in (5, 10, 15, 20, 25)
        }
        anchors = _checkpoint_anchor_evidence(
            Path("/frozen/original"), inventories, [5, 10, 15, 20, 25]
        )
        record = {
            "checkpoint_anchors": anchors,
            "checkpoint_anchors_sha256": canonical_sha256(anchors),
        }
        self.assertEqual(_failures_for_persisted_anchor_record(record, anchors), [])
        tampered = json.loads(json.dumps(anchors))
        tampered[1]["inventory"][0]["sha256"] = "f" * 64
        failures = _failures_for_persisted_anchor_record(record, tampered)
        self.assertTrue(any("anchor" in failure for failure in failures))
        missing = anchors[:-1]
        missing_record = {
            "checkpoint_anchors": missing,
            "checkpoint_anchors_sha256": canonical_sha256(missing),
        }
        failures = _failures_for_persisted_anchor_record(missing_record, anchors)
        self.assertTrue(any("anchor" in failure for failure in failures))

    def test_complete_checkpoint_steps_are_sorted_numerically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for step in (5, 10, 15, 20, 25):
                actor = output / f"global_step_{step}" / "actor"
                actor.mkdir(parents=True)
                (actor.parent / "data.pt").write_bytes(b"cursor")
                for rank in (0, 1):
                    for prefix in ("model", "optim", "extra_state"):
                        (actor / f"{prefix}_world_size_2_rank_{rank}.pt").write_bytes(
                            f"{step}:{prefix}:{rank}".encode("ascii")
                        )

            complete_steps, inventories = _complete_checkpoint_steps(output, 2)

        self.assertEqual(complete_steps, [5, 10, 15, 20, 25])
        self.assertEqual(sorted(inventories), [5, 10, 15, 20, 25])

    def test_rollout_seed_writer_and_t25_auditor_share_exact_filename(self) -> None:
        trainer = (REPO / "verl/trainer/ppo/ray_trainer.py").read_text(
            encoding="utf-8"
        )
        auditor = (
            REPO / "tools/h20/audit_qwen25_7b_original_t25.py"
        ).read_text(encoding="utf-8")
        expected = 'rollout_seed_audit.jsonl'
        self.assertIn(f'path = os.path.join(output_dir, "{expected}")', trainer)
        self.assertIn(
            f'seed_path = Path(paths["output"]) / "{expected}"', auditor
        )
        self.assertNotIn("rollout_trajectory_seeds.jsonl", auditor)

    def test_failed_audit_is_not_persisted_or_masked(self) -> None:
        underlying = {
            "status": "FAIL",
            "decision": "ORIGINAL_T25_NO_GO:AUDIT",
            "failures": ["REAL_UNDERLYING_FAILURE"],
        }
        output = io.StringIO()
        with (
            patch.object(
                t25_audit,
                "run_audit",
                return_value=(underlying, {}),
            ),
            patch.object(
                t25_audit,
                "persist_report",
                side_effect=AssertionError("persist must not be called for FAIL"),
            ) as persist,
            patch.object(
                sys,
                "argv",
                ["audit_qwen25_7b_original_t25.py", "--manifest", "unused", "--write-report"],
            ),
            contextlib.redirect_stdout(output),
        ):
            return_code = t25_audit.main()

        visible = json.loads(output.getvalue())
        self.assertEqual(return_code, 1)
        self.assertEqual(visible, underlying)
        persist.assert_not_called()

    def test_persisted_suffix_distinguishes_training_and_audit_code_commits(self) -> None:
        training_commit = "1" * 40
        audit_code_commit = "2" * 40
        run_id = "3" * 32
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p0 = root / "p0.json"
            report_path = root / "final.json"
            ledger_path = root / "ledger.jsonl"
            p0.write_text(
                json.dumps({"evidence": {"run_id": run_id}}), encoding="utf-8"
            )
            report = {
                "status": "PASS",
                "git_commit": training_commit,
                "audit_code_commit": audit_code_commit,
                "step25_checkpoint": {"inventory": [], "inventory_sha256": "4" * 64},
                "checkpoint_anchors": [],
            }
            manifest = {
                "experiment_name": "t25-audit-provenance-test",
                "paths": {
                    "final_report": str(report_path),
                    "execution_ledger": str(ledger_path),
                    "p0_certificate": str(p0),
                },
            }

            t25_audit.persist_report(report, manifest)
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            suffix = t25_audit.read_jsonl(ledger_path)

        self.assertEqual(persisted["git_commit"], training_commit)
        self.assertEqual(persisted["audit_code_commit"], audit_code_commit)
        self.assertEqual(len(suffix), 2)
        self.assertTrue(all(row["git_commit"] == training_commit for row in suffix))
        self.assertTrue(
            all(row["audit_code_commit"] == audit_code_commit for row in suffix)
        )

    def test_every_gpu67_launcher_uses_the_shared_lock(self) -> None:
        for relative in (
            "scripts/h20/gatea_frozen_common.sh",
            "scripts/h20/stable_i4x2_common.sh",
            "scripts/h20/original_t25_common.sh",
        ):
            text = (REPO / relative).read_text(encoding="utf-8")
            self.assertIn("locks/memagent_gate_a_gpu_6_7.lock", text, relative)
        preflight = (
            REPO / "tools/h20/preflight_qwen25_7b_original_t25.py"
        ).read_text(encoding="utf-8")
        self.assertIn('any("H20" not in line.upper()', preflight)

    def test_runner_uses_one_override_array_for_emit_and_execution(self) -> None:
        runner = (REPO / "experiments/7b_gate_a/run_gate_a.sh").read_text(
            encoding="utf-8"
        )
        self.assertEqual(runner.count("TRAINER_OVERRIDES=("), 1)
        self.assertIn("EMIT_TRAINER_OVERRIDES", runner)
        self.assertIn('verl.trainer.main_ppo "${TRAINER_OVERRIDES[@]}"', runner)
        self.assertIn("RESUME_TOTAL_STEPS=${RESUME_TOTAL_STEPS:-3}", runner)
        self.assertIn("SAVE_FREQ=${SAVE_FREQ:-1}", runner)
        self.assertIn("MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}", runner)

    def test_runtime_config_attestation_is_before_training(self) -> None:
        trainer = (REPO / "verl/trainer/ppo/ray_trainer.py").read_text(
            encoding="utf-8"
        )
        config_check = trainer.index("ORIGINAL_T25_EXPECTED_RUNTIME_CONFIG_SHA256")
        fit = trainer.index("def fit(self):")
        self.assertLess(config_check, fit)
        self.assertIn('"runtime_config"', trainer)
        schema = json.loads(
            (REPO / "gate_a_execution_ledger.schema.json").read_text(encoding="utf-8")
        )
        self.assertIn("runtime_config", schema["properties"]["record_type"]["enum"])

    def test_resume_audit_requires_scheduler_and_rng_restoration(self) -> None:
        preflight = (
            REPO / "tools/h20/preflight_qwen25_7b_original_t25.py"
        ).read_text(encoding="utf-8")
        audit = (
            REPO / "tools/h20/audit_qwen25_7b_original_t25.py"
        ).read_text(encoding="utf-8")
        worker = (REPO / "verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
        manager = (
            REPO / "verl/utils/checkpoint/fsdp_checkpoint_manager.py"
        ).read_text(encoding="utf-8")
        for key in ("cpu", "cuda", "numpy", "random"):
            self.assertIn(f'"{key}"', preflight)
        self.assertIn('ack.get("rng_restored") is not True', audit)
        self.assertIn('"rng_state_keys"', worker)
        self.assertIn('"lr_scheduler_loaded"', worker)
        self.assertIn('"rng_restored": rng_restored', manager)

        valid_ack = {
            "model_loaded": True,
            "optimizer_loaded": True,
            "extra_loaded": True,
            "rng_restored": True,
            "rng_state_keys": ["cpu", "cuda", "numpy", "random"],
            "lr_scheduler_loaded": True,
            "optimizer_step_max": 3,
            "lr_scheduler_last_epoch": 3,
        }
        self.assertEqual(
            _failures_for_resume_state_acks(
                [{**valid_ack, "rank": 0}, {**valid_ack, "rank": 1}]
            ),
            [],
        )
        tampered = [{**valid_ack, "rank": 0}, {**valid_ack, "rank": 1}]
        tampered[1]["rng_restored"] = False
        failures = _failures_for_resume_state_acks(tampered)
        self.assertTrue(any("scheduler/RNG" in failure for failure in failures))

    def test_wrapper_clears_inherited_cross_experiment_environment(self) -> None:
        common = (REPO / "scripts/h20/original_t25_common.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("compgen -v GATE_A_", common)
        for name in (
            "MEMAGENT_STABLE_I_WORK_ROOT",
            "MEMAGENT_STABLE_I_REPO_DIR",
            "MEMAGENT_STABLE_I_EXPECTED_COMMIT",
            "MEMAGENT_S128_IT_WORK_ROOT",
            "MEMAGENT_S128_IT_REPO_DIR",
            "MEMAGENT_S128_IT_EXPECTED_COMMIT",
        ):
            self.assertIn(name, common)
        runner = (
            REPO / "scripts/h20/resume_qwen25_7b_original_step3_to25.sh"
        ).read_text(encoding="utf-8")
        for binding in (
            'MODEL="$T25_WORK_ROOT/models/Qwen2.5-7B-Instruct"',
            'TRAIN="$T25_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet"',
            'VAL="$T25_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet"',
            "EMIT_TRAINER_OVERRIDES=0",
        ):
            self.assertIn(binding, runner)

    def test_write_certificate_requires_runtime_checks(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "tools/h20/preflight_qwen25_7b_original_t25.py"),
                "--manifest",
                str(MANIFEST_PATH),
                "--write-certificate",
            ],
            cwd=REPO,
            text=True,
            capture_output=True,
            env={**os.environ},
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("writing P0 requires --check-runtime", result.stdout)

    def test_context_derived_turn_schedule_catches_inactive_misalignment(self) -> None:
        active_counts = [1 + (index % 2) for index in range(100)]
        records: list[dict] = []
        for step in range(4, 26):
            base_rows = build_trajectory_seed_records(
                base_seed=2026,
                global_step=step,
                batch_size=8,
                rollout_n=2,
                mode="independent",
            )
            final_turn = max(active_counts[(step - 1) * 4 : step * 4])
            for base in base_rows:
                row = int(base["row"])
                group, replica = divmod(row, 2)
                enriched = {
                    **base,
                    "record_type": "trajectory_seed",
                    "uid": f"step-{step}-group-{group}",
                    "dataset_index": 1000 + (step - 1) * 4 + group,
                }
                records.append(enriched)
                for turn in range(active_counts[(step - 1) * 4 + group]):
                    records.append(
                        {
                            "record_type": "trajectory_turn_seed",
                            "global_step": step,
                            "sample_index": row,
                            "turn": turn,
                            "is_final": False,
                            "request_seed": derive_turn_request_seeds(
                                [int(base["trajectory_seed"])], [0], turn
                            )[0],
                            "trajectory_seed": int(base["trajectory_seed"]),
                            "uid": enriched["uid"],
                            "dataset_index": enriched["dataset_index"],
                            "group": group,
                            "replica": replica,
                        }
                    )
                records.append(
                    {
                        "record_type": "trajectory_turn_seed",
                        "global_step": step,
                        "sample_index": row,
                        "turn": final_turn,
                        "is_final": True,
                        "request_seed": derive_turn_request_seeds(
                            [int(base["trajectory_seed"])], [0], final_turn
                        )[0],
                        "trajectory_seed": int(base["trajectory_seed"]),
                        "uid": enriched["uid"],
                        "dataset_index": enriched["dataset_index"],
                        "group": group,
                        "replica": replica,
                    }
                )
        self.assertEqual(_audit_exact_turn_schedule(records, active_turn_counts=active_counts, rollout_n=2), [])
        first_turn = next(row for row in records if row.get("record_type") == "trajectory_turn_seed")
        first_turn["dataset_index"] = -1
        failures = _audit_exact_turn_schedule(records, active_turn_counts=active_counts, rollout_n=2)
        self.assertTrue(any("misaligned" in failure for failure in failures))

    def test_command_and_report_contract_paths(self) -> None:
        self.assertIs(self.commands["gpu_execution_authorized_by_this_manifest"], False)
        self.assertEqual(
            self.commands["required_sequence"],
            ["p0", "train_step4_to25", "readonly_audit"],
        )
        paths = self.manifest["paths"]
        self.assertTrue(paths["step25"].endswith("/global_step_25"))
        self.assertTrue(paths["final_report"].endswith("/original_t25_final_report.json"))
        audit_source = (
            REPO / "tools/h20/audit_qwen25_7b_original_t25.py"
        ).read_text(encoding="utf-8")
        for key in (
            '"status"',
            '"decision"',
            '"git_commit"',
            '"step25_checkpoint"',
            '"checkpoint_anchors"',
            '"inventory_sha256"',
            '"source_gate_a_commit"',
            '"source_gate_a_report_sha"',
        ):
            self.assertIn(key, audit_source)


if __name__ == "__main__":
    unittest.main()
