from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from recurrent.research.stable_eval_identity import (
    MANIFEST_ROW_FIELDS,
    OUTPUT_IDENTITY_FIELDS,
    TURN_LEDGER_NON_TENSOR_FIELDS,
    build_stable_eval_identities,
    canonical_sha256,
    stable_eval_runtime_config_sha256,
    trajectory_turn_record_from_columns,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds
from tools.h20.audit_qwen25_7b_stable_i4x2 import (
    audit_attempt,
    compare_deterministic_attempt_evidence,
)
from tools.h20.preflight_qwen25_7b_stable_i4x2 import (
    build_execution_binding,
    build_identity_payload,
    freeze_existing_s128_rows,
    load_manifest,
    model_loading_relevant_paths,
    render_trainer_overrides,
    validate_s128_freeze,
)


MANIFEST_PATH = REPO / "manifests/h20/qwen25_7b_stable_i4x2_seed2026.json"
FAKE_ENV = {
    "MEMAGENT_STABLE_I_WORK_ROOT": "/data/cw/memagent_work",
    "MEMAGENT_STABLE_I_REPO_DIR": str(REPO),
    "MEMAGENT_STABLE_I_EXPECTED_COMMIT": "a" * 40,
}


def fake_source_row(index: int, *, prompt_tokens: int = 10) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"question-{index}"}],
        "context": f"context-{index}",
        "reward_model": {"ground_truth": [f"answer-{index}"]},
        "extra_info": {"index": index * 7 + 3, "prompt_tokens": prompt_tokens},
    }


def resolved_manifest(rows: list[dict]) -> dict:
    payload = {
        "schema_version": 1,
        "interface_id": "I",
        "canary_source_order_indices": [0, 1, 2, 3],
        "rows": rows,
    }
    return {
        "identity_payload": payload,
        "eval_manifest_hash": canonical_sha256(payload),
        "execution_binding": {
            "canary_turn_schedule": {
                "active_turn_count_by_source_order": {
                    str(index): 1 for index in range(4)
                },
                "shared_final_turn": 1,
            },
            "trainer_configuration": {
                "attempts": {
                    "repeat_a": {"resolved_config_sha256": "6" * 64},
                    "repeat_b": {"resolved_config_sha256": "7" * 64},
                }
            },
        },
    }


def snapshot(manifest: dict, sync_kind: str) -> dict:
    parameter_names = manifest["weight_snapshot"]["parameter_names"]
    worker = {
        "optimizer_step_min": None,
        "optimizer_step_max": None,
        "optimizer_state_entry_count": 0,
        "optimizer_step_entry_count": 0,
        "optimizer_step_histogram": {},
        "lr_scheduler_last_epoch": 0,
        "weight_transfer_format": "dtensor",
        "loaded_parameter_count": 199,
        "model_parameter_count": 199,
        "loaded_parameter_names_sha256": "1" * 64,
        "model_parameter_names_sha256": "1" * 64,
        "audited_loaded_parameters": sorted(parameter_names),
        "sampled_parameter_dtypes": {name: "torch.bfloat16" for name in parameter_names},
    }
    return {
        "sync_kind": sync_kind,
        "worker_ranks": [0, 1],
        "actor_master_sampled_tensor_digest": "2" * 64,
        "actor_rollout_sampled_tensor_digest": "3" * 64,
        "vllm_sampled_tensor_digest": "3" * 64,
        "vllm_pre_sync_sampled_tensor_digest": (
            "3" * 64 if sync_kind == "stable_eval_after" else None
        ),
        "worker_evidence": [copy.deepcopy(worker), copy.deepcopy(worker)],
    }


def write_attempt(root: Path, attempt_id: str, manifest: dict, resolved: dict) -> None:
    semantic = []
    positions = []
    for row in resolved["identity_payload"]["rows"][:4]:
        semantic.extend([row["semantic_dataset_index"], row["semantic_dataset_index"]])
        positions.extend([row["source_order_index"], row["source_order_index"]])
    identities = build_stable_eval_identities(
        semantic_indices=semantic,
        source_order_indices=positions,
        replicas=2,
        base_seed=2026,
        interface_id="I",
        attempt_id=attempt_id,
        resolved_manifest=resolved,
        runtime_uuid_factory=iter(
            [f"{attempt_id}-runtime-{index}" for index in range(8)]
        ).__next__,
    )
    terminal = root / "terminal"
    terminal.mkdir(parents=True)
    (terminal / "0.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    **row,
                    "terminal_response_token_sha256": "4" * 64,
                    "output": f"answer-{row['source_repeated_row']}",
                    "score": 0.5,
                    "step": 0,
                },
                sort_keys=True,
            )
            + "\n"
            for row in identities
        ),
        encoding="utf-8",
    )
    turns = []
    for identity in identities:
        for recurrent_turn, is_final in ((0, False), (1, True)):
            request_seed = derive_turn_request_seeds(
                [identity["trajectory_seed"]], [0], recurrent_turn
            )[0]
            turns.append(
                {
                    "record_type": "trajectory_turn",
                    **identity,
                    "active_sample_index": identity["source_repeated_row"],
                    "request_seed": request_seed,
                    "configured_request_seed": request_seed,
                    "rollout_request_seed": request_seed,
                    "request_prompt_token_sha256": "5" * 64,
                    "returned_prompt_token_sha256": "5" * 64,
                    "rollout_worker_rank": identity["source_repeated_row"] % 2,
                    "is_final": is_final,
                    "trajectory_turn": recurrent_turn,
                    "response_token_sha256": "4" * 64,
                }
            )
    (root / "trajectory_turns.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in turns),
        encoding="utf-8",
    )
    summary = {
        "record_type": "execution_summary",
        "interface_id": "I",
        "attempt_id": attempt_id,
        "eval_manifest_hash": resolved["eval_manifest_hash"],
        "resolved_runtime_config_sha256": resolved["execution_binding"][
            "trainer_configuration"
        ]["attempts"][attempt_id]["resolved_config_sha256"],
        "global_step": 0,
        "actor_update_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_save_calls": 0,
        "resume_mode": "disable",
        "validation_only": True,
        "weight_snapshot_before": snapshot(manifest, "stable_eval_before"),
        "weight_snapshot_after": snapshot(manifest, "stable_eval_after"),
    }
    (root / "execution_summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "run.log").write_text("synthetic CPU audit fixture\n", encoding="utf-8")


class FrozenContractTests(unittest.TestCase):
    def test_turn_ledger_writer_uses_complete_identity_contract(self):
        trainer_source = (REPO / "verl/trainer/ppo/ray_trainer.py").read_text(
            encoding="utf-8"
        )
        writer_start = trainer_source.index("def _append_stable_eval_turn_ledger")
        writer_end = trainer_source.index("\n@contextmanager", writer_start)
        writer_source = trainer_source[writer_start:writer_end]
        self.assertIn("TURN_LEDGER_NON_TENSOR_FIELDS", writer_source)
        self.assertIn("trajectory_turn_record_from_columns", writer_source)
        for field in MANIFEST_ROW_FIELDS:
            self.assertIn(field, OUTPUT_IDENTITY_FIELDS)
            self.assertIn(field, TURN_LEDGER_NON_TENSOR_FIELDS)

    def test_turn_record_serializes_every_identity_and_request_field(self):
        frozen_rows, _ = freeze_existing_s128_rows(
            [fake_source_row(0)],
            prompt_token_length=lambda _prompt: 10,
            context_token_length=lambda _context: 5000,
            max_prompt_length=40000,
            max_context_length=40000,
        )
        identity = build_stable_eval_identities(
            semantic_indices=[3, 3],
            source_order_indices=[0, 0],
            replicas=2,
            base_seed=2026,
            interface_id="I",
            attempt_id="repeat_a",
            resolved_manifest=resolved_manifest(frozen_rows),
        )[0]
        columns = {field: [identity[field]] for field in OUTPUT_IDENTITY_FIELDS}
        columns.update(
            {
                "active_sample_index": [0],
                "request_seed": [11],
                "configured_request_seed": [11],
                "rollout_request_seed": [11],
                "request_prompt_token_sha256": ["a" * 64],
                "returned_prompt_token_sha256": ["a" * 64],
                "rollout_worker_rank": [0],
                "is_final": [False],
            }
        )
        record = trajectory_turn_record_from_columns(
            columns,
            row=0,
            trajectory_turn=2,
            response_token_sha256="b" * 64,
        )
        self.assertEqual(
            set(record),
            {
                "record_type",
                *TURN_LEDGER_NON_TENSOR_FIELDS,
                "trajectory_turn",
                "response_token_sha256",
            },
        )
        for field in TURN_LEDGER_NON_TENSOR_FIELDS:
            broken = dict(columns)
            broken.pop(field)
            with self.assertRaisesRegex(ValueError, "missing row fields"):
                trajectory_turn_record_from_columns(
                    broken,
                    row=0,
                    trajectory_turn=2,
                    response_token_sha256="b" * 64,
                )

    def test_manifest_has_no_implicit_shared_account_path_binding(self):
        with self.assertRaisesRegex(ValueError, "missing explicit stable-I runtime bindings"):
            load_manifest(MANIFEST_PATH, {})
        with self.assertRaisesRegex(ValueError, "must be an absolute path"):
            load_manifest(
                MANIFEST_PATH,
                {
                    **FAKE_ENV,
                    "MEMAGENT_STABLE_I_WORK_ROOT": "relative/shared/path",
                },
            )

    def test_manifest_is_existing_s128_recurrent_i_canary_only(self):
        manifest = load_manifest(MANIFEST_PATH, FAKE_ENV)
        self.assertEqual(manifest["branch"], "h20/qwen25-7b-stable-eval-i4x2-frozen-20260821")
        self.assertEqual(manifest["model"]["id"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(
            manifest["model"]["revision"],
            "a09a35458c702b33eeacc393d103063234e8bc28",
        )
        self.assertEqual(manifest["data"]["dataset_role"], "existing_project_fixed_s128")
        self.assertEqual(manifest["data"]["expected_raw_rows"], 128)
        self.assertEqual(manifest["data"]["expected_effective_rows"], 128)
        self.assertEqual(manifest["data"]["canary_source_order_indices"], [0, 1, 2, 3])
        self.assertFalse(manifest["data"]["shuffle"])
        self.assertEqual(manifest["data"]["dataloader_num_workers"], 0)
        self.assertTrue(manifest["data"]["include_source_order_index"])
        self.assertEqual(manifest["evaluation"]["attempts"], ["repeat_a", "repeat_b"])
        self.assertEqual(manifest["evaluation"]["replicas"], 2)
        self.assertTrue(manifest["evaluation"]["validation_only"])
        self.assertEqual(manifest["evaluation"]["actor_update_calls"], 0)
        self.assertEqual(manifest["backend"]["rollout"], "vllm")
        self.assertFalse(manifest["backend"]["allow_hf_fallback"])
        self.assertTrue(manifest["scope"]["not_a_paper_performance_evaluation"])
        self.assertTrue(manifest["scope"]["not_a_five_interface_gate"])
        self.assertTrue(manifest["scope"]["eval_manifest_hash_is_interface_neutral"])

    def test_command_manifest_cannot_self_authorize_gpu_or_scientific_claim(self):
        commands = json.loads(
            (REPO / "manifests/h20/qwen25_7b_stable_i4x2_commands.json").read_text()
        )
        self.assertFalse(commands["gpu_execution_authorized_by_this_manifest"])
        self.assertFalse(commands["scientific_claim_authorized_by_this_manifest"])
        self.assertEqual(
            commands["required_environment"],
            [
                "MEMAGENT_STABLE_I_WORK_ROOT",
                "MEMAGENT_STABLE_I_REPO_DIR",
                "MEMAGENT_STABLE_I_EXPECTED_COMMIT",
            ],
        )
        self.assertEqual(commands["required_sequence"], ["p0", "repeat_a", "repeat_b", "audit"])
        schema = json.loads((REPO / "stable_identity_execution_ledger.schema.json").read_text())
        self.assertIn("execution_binding_sha256", schema["required"])
        self.assertEqual(
            schema["properties"]["execution_binding_sha256"]["pattern"],
            "^[0-9a-f]{64}$",
        )

    def test_wrappers_use_scoped_bindings_strict_vllm_and_zero_training(self):
        common = (REPO / "scripts/h20/stable_i4x2_common.sh").read_text()
        runner = (REPO / "scripts/h20/run_qwen25_7b_stable_i4x2.sh").read_text()
        commands = json.loads(
            (REPO / "manifests/h20/qwen25_7b_stable_i4x2_commands.json").read_text()
        )
        overrides = commands["trainer_overrides"]
        self.assertIn("MEMAGENT_STABLE_I_WORK_ROOT", common)
        self.assertNotIn("readonly WORK_ROOT", common)
        self.assertIn("run_attempt repeat_a", runner)
        self.assertIn("stable_i_wait_for_attempt_cleanup", runner)
        self.assertIn("run_attempt repeat_b", runner)
        self.assertIn("--emit-trainer-overrides", runner)
        self.assertIn('"${trainer_overrides[@]}"', runner)
        self.assertIn("+data.include_source_order_index=True", overrides)
        self.assertIn("actor_rollout_ref.rollout.name=vllm", overrides)
        self.assertIn("actor_rollout_ref.rollout.mode=sync", overrides)
        self.assertIn("+trainer.val_only=True", overrides)
        self.assertIn("trainer.save_freq=-1", overrides)
        self.assertIn("trainer.resume_mode=disable", overrides)
        self.assertTrue(any("expected_runtime_config_sha256=" in item for item in overrides))
        self.assertIn("unset GATE_A_FROZEN_AUDIT", runner)
        self.assertIn("GATE_A_WEIGHT_DIGEST_SAMPLES=256", runner)

    def test_frozen_trainer_argv_is_single_source_and_config_hash_is_not_recursive(self):
        manifest = load_manifest(MANIFEST_PATH, FAKE_ENV)
        overrides = render_trainer_overrides(
            manifest,
            repo=REPO,
            attempt_id="repeat_a",
            eval_manifest_hash="5" * 64,
            expected_runtime_config_sha256="6" * 64,
        )
        self.assertTrue(overrides)
        self.assertFalse(any("${" in item for item in overrides))
        self.assertIn(
            "+trainer.eval_identity.expected_runtime_config_sha256=" + "6" * 64,
            overrides,
        )
        config = {
            "trainer": {
                "eval_identity": {
                    "enabled": True,
                    "expected_runtime_config_sha256": "0" * 64,
                }
            },
            "data": {"val_max_samples": 4},
        }
        reference = stable_eval_runtime_config_sha256(config)
        changed_self_hash = copy.deepcopy(config)
        changed_self_hash["trainer"]["eval_identity"][
            "expected_runtime_config_sha256"
        ] = "f" * 64
        self.assertEqual(reference, stable_eval_runtime_config_sha256(changed_self_hash))
        changed_budget = copy.deepcopy(config)
        changed_budget["data"]["val_max_samples"] = 8
        self.assertNotEqual(reference, stable_eval_runtime_config_sha256(changed_budget))


class S128PreflightTests(unittest.TestCase):
    def test_model_loading_inventory_detects_extra_weights_adapters_and_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [
                "config.json",
                "model-00001-of-00001.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
            ]
            for relative in expected:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            (root / "README.md").write_text("non-loading documentation")
            self.assertEqual(model_loading_relevant_paths(root), sorted(expected))
            for relative in (
                "unexpected/model.safetensors",
                "adapter_config.json",
                "chat_template.jinja",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"unexpected")
            actual = model_loading_relevant_paths(root)
            self.assertIn("unexpected/model.safetensors", actual)
            self.assertIn("adapter_config.json", actual)
            self.assertIn("chat_template.jinja", actual)

    def test_complete_existing_s128_survives_production_filter_and_is_frozen(self):
        rows = [fake_source_row(index) for index in range(128)]
        frozen, evidence = freeze_existing_s128_rows(
            rows,
            prompt_token_length=lambda _prompt: 10,
            context_token_length=lambda _context: 5000,
            max_prompt_length=40000,
            max_context_length=40000,
        )
        manifest = load_manifest(MANIFEST_PATH, FAKE_ENV)
        self.assertEqual(validate_s128_freeze(frozen, evidence, manifest["data"]), [])
        self.assertEqual(len(frozen), 128)
        self.assertEqual([row["source_order_index"] for row in frozen[:4]], [0, 1, 2, 3])
        self.assertEqual([row["raw_row_position"] for row in frozen], list(range(128)))
        self.assertEqual(frozen[0]["example_id"], str(rows[0]["extra_info"]["index"]))

        payload = build_identity_payload(manifest, rows=frozen)
        self.assertEqual(len(payload["rows"]), 128)
        self.assertEqual(payload["source_dataset"]["role"], "existing_project_fixed_s128")
        self.assertEqual(
            payload["base_model_protocol"]["revision"], manifest["model"]["revision"]
        )
        self.assertEqual(payload["decode"], {
            "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
        })
        self.assertEqual(payload["backend"]["rollout"], "vllm")
        for forbidden in (
            "interface_id",
            "model_artifact",
            "checkpoint_inventory_sha256",
            "git_commit",
            "recurrent",
            "canary_source_order_indices",
        ):
            self.assertNotIn(forbidden, payload)

        # Changing I to T, recurrent implementation details, or a checkpoint/
        # weight inventory must not split the stable source-row join domain.
        t_manifest = copy.deepcopy(manifest)
        t_manifest["evaluation"]["interface_id"] = "T"
        t_manifest["recurrent"]["chunk_size"] = 4096
        t_manifest["model"]["files"][3]["sha256"] = "f" * 64
        self.assertEqual(
            canonical_sha256(payload),
            canonical_sha256(build_identity_payload(t_manifest, rows=frozen)),
        )
        execution = build_execution_binding(
            manifest, repo=REPO, git_commit="a" * 40, rows=frozen
        )
        self.assertEqual(execution["interface_id"], "I")
        self.assertIn("file_manifest_sha256", execution["model_artifact"])
        self.assertNotEqual(
            canonical_sha256(execution),
            canonical_sha256(
                build_execution_binding(
                    t_manifest, repo=REPO, git_commit="b" * 40, rows=frozen
                )
            ),
        )
        self.assertEqual(
            build_execution_binding(
                t_manifest, repo=REPO, git_commit="b" * 40, rows=frozen
            )[
                "interface_id"
            ],
            "T",
        )

    def test_row_loss_or_filter_rejection_is_p0_failure_not_a_smaller_dataset(self):
        rows = [fake_source_row(index) for index in range(128)]
        frozen, evidence = freeze_existing_s128_rows(
            rows,
            prompt_token_length=lambda prompt: 50000 if prompt == rows[17]["prompt"] else 10,
            context_token_length=lambda _context: 5000,
            max_prompt_length=40000,
            max_context_length=40000,
        )
        manifest = load_manifest(MANIFEST_PATH, FAKE_ENV)
        failures = validate_s128_freeze(frozen, evidence, manifest["data"])
        self.assertTrue(any("did not retain the complete fixed S128" in item for item in failures))
        self.assertEqual(evidence["rejected_raw_positions"], [17])

        short, short_evidence = freeze_existing_s128_rows(
            rows[:-1],
            prompt_token_length=lambda _prompt: 10,
            context_token_length=lambda _context: 5000,
            max_prompt_length=40000,
            max_context_length=40000,
        )
        failures = validate_s128_freeze(short, short_evidence, manifest["data"])
        self.assertTrue(any("raw row count" in item for item in failures))


class ArtifactAuditTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(MANIFEST_PATH, FAKE_ENV)
        source = [fake_source_row(index) for index in range(128)]
        self.rows, _ = freeze_existing_s128_rows(
            source,
            prompt_token_length=lambda _prompt: 10,
            context_token_length=lambda _context: 5000,
            max_prompt_length=40000,
            max_context_length=40000,
        )
        self.resolved = resolved_manifest(self.rows)

    def test_attempt_requires_eight_rows_all_turn_seed_echoes_and_unchanged_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repeat_a"
            root.mkdir()
            write_attempt(root, "repeat_a", self.manifest, self.resolved)
            rows, failures, evidence = audit_attempt(
                attempt_id="repeat_a",
                attempt_root=root,
                resolved_manifest=self.resolved,
                manifest=self.manifest,
            )
            self.assertEqual(failures, [])
            self.assertEqual(len(rows), 8)
            self.assertEqual(evidence["terminal_row_count"], 8)
            self.assertEqual(evidence["turn_row_count"], 16)
            self.assertEqual(evidence["forbidden_checkpoint_files"], [])

    def test_seed_echo_weight_mutation_and_checkpoint_each_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repeat_a"
            root.mkdir()
            write_attempt(root, "repeat_a", self.manifest, self.resolved)
            turns_path = root / "trajectory_turns.jsonl"
            turns = [json.loads(line) for line in turns_path.read_text().splitlines()]
            turns[0]["configured_request_seed"] += 1
            turns_path.write_text(
                "".join(json.dumps(row) + "\n" for row in turns), encoding="utf-8"
            )
            summary_path = root / "execution_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["weight_snapshot_after"]["actor_master_sampled_tensor_digest"] = "9" * 64
            summary["weight_snapshot_after"]["vllm_pre_sync_sampled_tensor_digest"] = "8" * 64
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            (root / "model_world_size_2_rank_0.pt").write_bytes(b"forbidden")
            _, failures, _ = audit_attempt(
                attempt_id="repeat_a",
                attempt_root=root,
                resolved_manifest=self.resolved,
                manifest=self.manifest,
            )
            self.assertTrue(any("configured request seed mismatch" in item for item in failures))
            self.assertTrue(any("weight/optimizer evidence changed" in item for item in failures))
            self.assertTrue(any("vLLM drift was hidden" in item for item in failures))
            self.assertTrue(any("forbidden checkpoint" in item for item in failures))

    def test_terminal_result_and_rollout_seed_alias_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repeat_a"
            root.mkdir()
            write_attempt(root, "repeat_a", self.manifest, self.resolved)

            turns_path = root / "trajectory_turns.jsonl"
            turns = [json.loads(line) for line in turns_path.read_text().splitlines()]
            turns[0]["rollout_request_seed"] += 1
            turns_path.write_text(
                "".join(json.dumps(row) + "\n" for row in turns), encoding="utf-8"
            )

            terminal_path = root / "terminal/0.jsonl"
            terminal = [json.loads(line) for line in terminal_path.read_text().splitlines()]
            terminal[0].pop("output")
            terminal[1]["score"] = "not-a-number"
            terminal[2]["step"] = 1
            terminal_path.write_text(
                "".join(json.dumps(row) + "\n" for row in terminal), encoding="utf-8"
            )

            _, failures, _ = audit_attempt(
                attempt_id="repeat_a",
                attempt_root=root,
                resolved_manifest=self.resolved,
                manifest=self.manifest,
            )
            self.assertTrue(any("configured request seed mismatch" in item for item in failures))
            self.assertTrue(any("missing result fields" in item for item in failures))
            self.assertTrue(any("output is not text" in item for item in failures))
            self.assertTrue(any("score is not finite" in item for item in failures))
            self.assertTrue(any("step is not frozen at zero" in item for item in failures))

    def test_equal_numeric_strings_do_not_pass_as_integer_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repeat_a"
            root.mkdir()
            write_attempt(root, "repeat_a", self.manifest, self.resolved)
            turns_path = root / "trajectory_turns.jsonl"
            turns = [json.loads(line) for line in turns_path.read_text().splitlines()]
            for field in (
                "request_seed",
                "configured_request_seed",
                "rollout_request_seed",
            ):
                turns[0][field] = str(turns[0][field])
            turns_path.write_text(
                "".join(json.dumps(row) + "\n" for row in turns), encoding="utf-8"
            )
            _, failures, _ = audit_attempt(
                attempt_id="repeat_a",
                attempt_root=root,
                resolved_manifest=self.resolved,
                manifest=self.manifest,
            )
            self.assertTrue(any("has non-integer fields" in item for item in failures))

    def test_boolean_summary_counts_do_not_pass_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repeat_a"
            root.mkdir()
            write_attempt(root, "repeat_a", self.manifest, self.resolved)
            summary_path = root / "execution_summary.json"
            summary = json.loads(summary_path.read_text())
            for field in (
                "global_step",
                "actor_update_calls",
                "optimizer_step_calls",
                "checkpoint_save_calls",
            ):
                summary[field] = False
            summary["validation_only"] = 1
            summary["weight_snapshot_before"]["worker_evidence"][0][
                "loaded_parameter_names_sha256"
            ] = None
            summary["weight_snapshot_before"]["worker_evidence"][0][
                "model_parameter_names_sha256"
            ] = None
            summary["weight_snapshot_before"]["worker_evidence"][1][
                "sampled_parameter_dtypes"
            ] = {
                name: "torch.float16"
                for name in self.manifest["weight_snapshot"]["parameter_names"]
            }
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            _, failures, _ = audit_attempt(
                attempt_id="repeat_a",
                attempt_root=root,
                resolved_manifest=self.resolved,
                manifest=self.manifest,
            )
            self.assertTrue(any("summary field global_step is not an integer" in item for item in failures))
            self.assertTrue(any("validation_only is not boolean" in item for item in failures))
            self.assertTrue(any("parameter_names_sha256" in item for item in failures))
            self.assertTrue(any("sampled dtype values drifted" in item for item in failures))

    def test_turn_gap_terminal_reorder_and_response_misbinding_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repeat_a"
            root.mkdir()
            write_attempt(root, "repeat_a", self.manifest, self.resolved)

            turns_path = root / "trajectory_turns.jsonl"
            turns = [json.loads(line) for line in turns_path.read_text().splitlines()]
            first_key = turns[0]["trajectory_id"]
            for row in turns:
                if row["trajectory_id"] == first_key and not row["is_final"]:
                    row["trajectory_turn"] = 3
                if row["trajectory_id"] == first_key and row["is_final"]:
                    row["trajectory_turn"] = 2
            turns_path.write_text(
                "".join(json.dumps(row) + "\n" for row in turns), encoding="utf-8"
            )

            terminal_path = root / "terminal/0.jsonl"
            terminal = [json.loads(line) for line in terminal_path.read_text().splitlines()]
            terminal[0], terminal[1] = terminal[1], terminal[0]
            terminal[0]["terminal_response_token_sha256"] = "9" * 64
            terminal_path.write_text(
                "".join(json.dumps(row) + "\n" for row in terminal), encoding="utf-8"
            )

            _, failures, _ = audit_attempt(
                attempt_id="repeat_a",
                attempt_root=root,
                resolved_manifest=self.resolved,
                manifest=self.manifest,
            )
            self.assertTrue(any("terminal output order" in item for item in failures))
            self.assertTrue(any("frozen context schedule" in item for item in failures))
            self.assertTrue(any("frozen shared final turn" in item for item in failures))
            self.assertTrue(any("final response digest is not bound" in item for item in failures))

    def test_cross_attempt_response_or_turn_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            first_root = Path(directory) / "repeat_a"
            second_root = Path(directory) / "repeat_b"
            first_root.mkdir()
            second_root.mkdir()
            write_attempt(first_root, "repeat_a", self.manifest, self.resolved)
            write_attempt(second_root, "repeat_b", self.manifest, self.resolved)
            _, first_failures, first = audit_attempt(
                attempt_id="repeat_a",
                attempt_root=first_root,
                resolved_manifest=self.resolved,
                manifest=self.manifest,
            )
            _, second_failures, second = audit_attempt(
                attempt_id="repeat_b",
                attempt_root=second_root,
                resolved_manifest=self.resolved,
                manifest=self.manifest,
            )
            self.assertEqual(first_failures, [])
            self.assertEqual(second_failures, [])
            self.assertEqual(compare_deterministic_attempt_evidence(first, second), [])

            changed = copy.deepcopy(second)
            first_key = next(iter(changed["turn_path_by_stable_key"]))
            changed["turn_path_by_stable_key"][first_key][
                "response_token_sha256"
            ] = "9" * 64
            failures = compare_deterministic_attempt_evidence(first, changed)
            self.assertTrue(any("turn_path_by_stable_key" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
