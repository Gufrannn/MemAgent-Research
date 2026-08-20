from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from recurrent.research.gate_a_execution import (
    append_jsonl,
    checkpoint_inventory,
    load_frozen_manifest,
    partition_numeric_metrics,
    validate_jsonl_chain,
)
from tools.h20.audit_qwen25_7b_gatea import (
    audit_seeds,
    audit_sync,
    component_inventory,
    jsonl_records_sha256,
    validate_ledger_schema,
    verify_resume_source,
)
from recurrent.research.trajectory_seeding import build_trajectory_seed_records, derive_turn_request_seeds

digest_spec = importlib.util.spec_from_file_location(
    "gate_a_weight_sync", REPO / "verl/utils/gate_a_weight_sync.py"
)
digest_module = importlib.util.module_from_spec(digest_spec)
assert digest_spec.loader is not None
digest_spec.loader.exec_module(digest_module)
digest_sample_records = digest_module.digest_sample_records
evenly_spaced_indices = digest_module.evenly_spaced_indices


class FrozenManifestTests(unittest.TestCase):
    def test_manifest_freezes_required_h20_shape(self):
        path = REPO / "manifests/h20/qwen25_7b_gatea_seed2026.yaml"
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["branch"], "h20/qwen25-7b-gatea-2gpu-frozen-20260820")
        self.assertEqual(manifest["derived_from_commit"], "4304f81d896df59604ccbd66adee1009e030376a")
        self.assertEqual(manifest["gpu"]["declared_whitelist"], [6, 7])
        self.assertEqual(manifest["gpu"]["world_size"], 2)
        self.assertEqual(manifest["gpu"]["fsdp_size"], 2)
        self.assertEqual(manifest["backend"]["rollout"], "vllm")
        self.assertEqual(manifest["backend"]["evaluation"], "vllm")
        self.assertFalse(manifest["backend"]["allow_hf_fallback"])
        self.assertEqual(manifest["backend"]["reward_manager"], "naive")
        self.assertEqual(
            manifest["weight_sync"]["comparison_semantics"],
            "actor_projected_to_vllm_parameter_dtype",
        )
        self.assertEqual(manifest["weight_sync"]["transfer_format"], "dtensor")
        self.assertTrue(
            any(".self_attn.o_proj.weight" in name or ".mlp.down_proj.weight" in name
                for name in manifest["weight_sync"]["parameter_names"])
        )
        self.assertEqual(manifest["weight_sync"]["expected_loaded_parameter_count"], 199)
        self.assertEqual(len(manifest["model"]["files"]), 11)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["model"]["files"]))
        self.assertEqual(len(manifest["data"]["train_sha256"]), 64)
        self.assertEqual(len(manifest["data"]["validation_sha256"]), 64)

    def test_gate_a_run_freezes_dtensor_transfer(self):
        script = (REPO / "experiments/7b_gate_a/run_gate_a.sh").read_text()
        self.assertIn("actor_rollout_ref.rollout.load_format=dummy_dtensor", script)

    def test_data_manifest_hash_is_canonical(self):
        manifest = json.loads((REPO / "manifests/h20/qwen25_7b_gatea_seed2026.yaml").read_text())
        data = dict(manifest["data"])
        expected = data.pop("manifest_sha256")
        payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)

    def test_command_manifest_is_non_authorizing_and_schema_parses(self):
        commands = json.loads((REPO / "manifests/h20/qwen25_7b_gatea_commands.json").read_text())
        schema = json.loads((REPO / "gate_a_execution_ledger.schema.json").read_text())
        self.assertFalse(commands["gpu_execution_authorized_by_this_manifest"])
        self.assertEqual(commands["contract"], {
            "kind": "formal_gate_a", "physical_gpus": [6, 7], "world_size": 2,
            "execution_revision": "20260821r4",
        })
        self.assertEqual(commands["ledger_schema"], "gate_a_execution_ledger.schema.json")
        self.assertEqual(commands["required_environment"], [
            "MEMAGENT_GATEA_WORK_ROOT", "MEMAGENT_GATEA_REPO_DIR",
            "MEMAGENT_GATEA_EXPECTED_COMMIT",
        ])
        self.assertEqual(commands["working_directory"], "${MEMAGENT_GATEA_REPO_DIR}")
        self.assertEqual(commands["required_sequence"], ["p0", "p1", "p2", "audit"])
        self.assertIn("--write-certificate", commands["commands"]["p0"])
        self.assertNotIn("--write-report", commands["commands"]["audit"])
        self.assertIn("weight_sync_ack", schema["properties"]["record_type"]["enum"])
        self.assertIn("rollout_start", schema["properties"]["record_type"]["enum"])

    def test_wrappers_spawn_child_environment_for_readonly_bindings(self):
        for name in ("run_qwen25_7b_gatea_fresh2.sh", "resume_qwen25_7b_gatea_step2_to3.sh"):
            script = (REPO / "scripts/h20" / name).read_text()
            self.assertIn(
                'env WORK_ROOT="$GATEA_WORK_ROOT" CODE="$GATEA_CODE" PYTHON="$GATEA_PYTHON"',
                script,
            )
            self.assertNotIn("\nWORK_ROOT=$GATEA_WORK_ROOT", script)

    def test_public_runtime_bindings_are_task_scoped(self):
        common = (REPO / "scripts/h20/gatea_frozen_common.sh").read_text()
        self.assertIn("MEMAGENT_GATEA_WORK_ROOT", common)
        self.assertIn("MEMAGENT_GATEA_REPO_DIR", common)
        self.assertNotIn("readonly WORK_ROOT", common)
        self.assertNotIn("readonly REPO_DIR", common)
        self.assertNotIn("readonly PYTHON", common)
        self.assertIn("MEMAGENT_GATEA_EXPECTED_COMMIT", common)
        self.assertIn("flock -n 9", common)

    def test_fresh_wrapper_reuses_p0_without_overwrite(self):
        script = (REPO / "scripts/h20/run_qwen25_7b_gatea_fresh2.sh").read_text()
        self.assertIn("gatea_require_p0_commit", script)
        self.assertIn("--phase fresh --check-runtime", script)
        self.assertNotIn("--write-certificate", script)

    def test_runtime_paths_require_explicit_binding_without_selecting_repo(self):
        path = REPO / "manifests/h20/qwen25_7b_gatea_seed2026.yaml"
        first = load_frozen_manifest(path, {
            "MEMAGENT_GATEA_WORK_ROOT": "/data/cw/memagent_work",
            "MEMAGENT_GATEA_REPO_DIR": "/data/cw/memagent_work/code/MemAgent-Research",
            "MEMAGENT_GATEA_EXPECTED_COMMIT": "a" * 40,
        })
        second = load_frozen_manifest(path, {
            "MEMAGENT_GATEA_WORK_ROOT": "/data/cw/memagent_work",
            "MEMAGENT_GATEA_REPO_DIR": "/data/cw/memagent_work/MemAgent-Research",
            "MEMAGENT_GATEA_EXPECTED_COMMIT": "a" * 40,
        })
        self.assertEqual(first["repository"], "/data/cw/memagent_work/code/MemAgent-Research")
        self.assertEqual(second["repository"], "/data/cw/memagent_work/MemAgent-Research")
        with self.assertRaisesRegex(ValueError, "MEMAGENT_GATEA_EXPECTED_COMMIT"):
            load_frozen_manifest(path, {})
        with self.assertRaisesRegex(ValueError, "missing explicit runtime path bindings"):
            load_frozen_manifest(path, {
                "WORK_ROOT": "/somebody/elses/work",
                "REPO_DIR": "/somebody/elses/repository",
                "MEMAGENT_GATEA_EXPECTED_COMMIT": "a" * 40,
            })


class DigestTests(unittest.TestCase):
    def test_evenly_spaced_indices_include_endpoints(self):
        self.assertEqual(evenly_spaced_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(evenly_spaced_indices(2, 99), [0, 1])

    def test_digest_is_ordered_and_sensitive(self):
        first = [("a", (2,), (0, 1), b"12345678")]
        self.assertEqual(digest_sample_records(first), digest_sample_records(first))
        self.assertNotEqual(digest_sample_records(first), digest_sample_records([("a", (2,), (0, 1), b"12345679")]))


class LedgerAndAuditTests(unittest.TestCase):
    def test_append_jsonl_is_valid_and_checkpoint_inventory_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            append_jsonl(root / "ledger.jsonl", {"b": 2, "a": 1})
            record = json.loads((root / "ledger.jsonl").read_text())
            self.assertEqual((record["a"], record["b"], record["record_index"]), (1, 2, 0))
            self.assertEqual(validate_jsonl_chain([record]), [])
            (root / "step/actor").mkdir(parents=True)
            (root / "step/actor/model_world_size_2_rank_0.pt").write_bytes(b"model")
            inventory = checkpoint_inventory(root / "step")
            self.assertEqual(inventory[0]["size"], 5)
            self.assertEqual(inventory[0]["sha256"], hashlib.sha256(b"model").hexdigest())

    def test_hash_chain_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            append_jsonl(path, {"value": 1})
            append_jsonl(path, {"value": 2})
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(validate_jsonl_chain(records), [])
            records[0]["value"] = 99
            failures = validate_jsonl_chain(records)
            self.assertTrue(any("record hash mismatch" in failure for failure in failures))

    def test_ledger_schema_is_applied_to_records(self):
        schema = json.loads((REPO / "gate_a_execution_ledger.schema.json").read_text())
        failures = validate_ledger_schema([{
            "record_type": "execution_signal",
            "experiment_name": "fresh",
            "git_commit": "a" * 40,
            "run_id": "b" * 32,
            "recorded_at": "2026-08-21T00:00:00+00:00",
            "record_index": 0,
            "previous_record_sha256": "0" * 64,
            "record_sha256": "c" * 64,
            "global_step": 1,
            "actor_version": 1,
            "metrics": {},
        }], schema)
        self.assertTrue(any("nonfinite_metric_names" in failure for failure in failures))
        failures = validate_ledger_schema([{
            "record_type": "p0_preflight",
            "experiment_name": "p0",
            "git_commit": "a" * 40,
            "run_id": "b" * 32,
            "recorded_at": "2026-08-21T00:00:00+00:00",
            "record_index": 0,
            "previous_record_sha256": "0" * 64,
            "record_sha256": "c" * 64,
            "optimizer_step_histogram": {"0": 0},
        }], schema)
        self.assertTrue(any("optimizer_step_histogram.0" in failure for failure in failures))

    def test_checkpoint_inventory_requires_both_two_gpu_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            step = Path(directory) / "global_step_2"
            actor = step / "actor"
            actor.mkdir(parents=True)
            for rank in range(2):
                for prefix in ("model", "optim", "extra_state"):
                    (actor / f"{prefix}_world_size_2_rank_{rank}.pt").write_bytes(b"evidence")
            (step / "data.pt").write_bytes(b"cursor")
            _, missing = component_inventory(step, 2)
            self.assertEqual(missing, [])
            (actor / "optim_world_size_2_rank_1.pt").unlink()
            _, missing = component_inventory(step, 2)
            self.assertTrue(any(item.startswith("optim_ranks_") for item in missing))

    def test_resume_preflight_rejects_checkpoint_changed_after_p1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step = root / "fresh/global_step_2"
            actor = step / "actor"
            actor.mkdir(parents=True)
            for rank in range(2):
                for prefix in ("model", "optim", "extra_state"):
                    (actor / f"{prefix}_world_size_2_rank_{rank}.pt").write_bytes(
                        f"{prefix}-{rank}".encode()
                    )
            (step / "data.pt").write_bytes(b"cursor")
            inventory, missing = component_inventory(step, 2)
            self.assertEqual(missing, [])

            run_id = "b" * 32
            commit = "a" * 40
            ledger_path = root / "ledger.jsonl"
            append_jsonl(ledger_path, {
                "record_type": "p0_preflight", "experiment_name": "p0",
                "git_commit": commit, "run_id": run_id,
                "recorded_at": "2026-08-21T00:00:00+00:00",
            })
            prefix = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            certificates = root / "certificates"
            certificates.mkdir()
            (certificates / "p0_preflight.json").write_text(json.dumps({
                "status": "PASS", "evidence": {"run_id": run_id, "git_commit": commit},
            }))
            p1 = {
                "status": "PASS", "decision": "P1_AUDIT_PASS",
                "step2_inventory": inventory,
                "ledger_record_count": len(prefix),
                "ledger_sha256": jsonl_records_sha256(prefix),
                "ledger_tail_record_sha256": prefix[-1]["record_sha256"],
            }
            (certificates / "p1_audit_report.json").write_text(json.dumps(p1))
            for record in (
                {
                    "record_type": "checkpoint_inventory", "global_step": 2,
                    "inventory": inventory,
                },
                {"record_type": "audit_result", "phase": "p1", "status": "PASS"},
            ):
                append_jsonl(ledger_path, {
                    **record, "experiment_name": "fresh", "git_commit": commit,
                    "run_id": run_id, "recorded_at": "2026-08-21T00:00:01+00:00",
                })
            manifest = {
                "paths": {
                    "certificate_root": str(certificates),
                    "resume_source": str(step),
                    "execution_ledger": str(ledger_path),
                },
                "gpu": {"world_size": 2},
                "experiments": {"fresh": "fresh"},
                "ledger_schema": "gate_a_execution_ledger.schema.json",
            }
            self.assertEqual(verify_resume_source(manifest)["status"], "PASS")
            (actor / "model_world_size_2_rank_0.pt").write_bytes(b"changed")
            result = verify_resume_source(manifest)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("inventory differs" in failure for failure in result["failures"]))
            changed_inventory, _ = component_inventory(step, 2)
            p1["step2_inventory"] = changed_inventory
            (certificates / "p1_audit_report.json").write_text(json.dumps(p1))
            result = verify_resume_source(manifest)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any(
                "hash-chained P1 inventory" in failure for failure in result["failures"]
            ))

    def test_seed_schedule_reconstruction(self):
        rows = build_trajectory_seed_records(
            base_seed=2026, global_step=2, batch_size=8, rollout_n=2, mode="independent"
        )
        for row in rows:
            row["record_type"] = "trajectory_seed"
            row["uid"] = f"uid-{row['group']}"
            row["dataset_index"] = 4 + row["group"]
        turn_rows = []
        for row in rows:
            turn_rows.append({
                "record_type": "trajectory_turn_seed",
                "global_step": 2,
                "row": row["row"],
                "sample_index": row["row"],
                "group": row["group"],
                "replica": row["replica"],
                "turn": 0,
                "uid": row["uid"],
                "trajectory_seed": row["trajectory_seed"],
                "dataset_index": row["dataset_index"],
                "request_seed": derive_turn_request_seeds([row["trajectory_seed"]], [0], 0)[0],
                "is_final": True,
                "mode": "independent",
            })
        records = rows + turn_rows
        ok, failures = audit_seeds(
            records, 2026, 2, expected_steps=[2], expected_batch_size=8
        )
        self.assertTrue(ok, failures)
        rows[1]["trajectory_seed"] = rows[0]["trajectory_seed"]
        ok, failures = audit_seeds(
            records, 2026, 2, expected_steps=[2], expected_batch_size=8
        )
        self.assertFalse(ok)
        self.assertTrue(any("collision" in failure for failure in failures))

    def test_seed_audit_allows_inactive_gap_before_shared_final_turn(self):
        rows = build_trajectory_seed_records(
            base_seed=2026, global_step=1, batch_size=2, rollout_n=2, mode="independent"
        )
        records = []
        for row in rows:
            row = dict(row)
            row.update(record_type="trajectory_seed", uid="uid-0", dataset_index=0)
            records.append(row)
        active_turn_counts = {0: 5, 1: 6}
        for row in rows:
            source_row = int(row["row"])
            for turn in range(active_turn_counts[source_row]):
                records.append({
                    "record_type": "trajectory_turn_seed",
                    "global_step": 1,
                    "row": source_row,
                    "sample_index": source_row,
                    "group": int(row["group"]),
                    "replica": int(row["replica"]),
                    "turn": turn,
                    "uid": "uid-0",
                    "trajectory_seed": int(row["trajectory_seed"]),
                    "dataset_index": 0,
                    "request_seed": derive_turn_request_seeds(
                        [int(row["trajectory_seed"])], [0], turn
                    )[0],
                    "is_final": False,
                    "mode": "independent",
                })
            records.append({
                "record_type": "trajectory_turn_seed",
                "global_step": 1,
                "row": source_row,
                "sample_index": source_row,
                "group": int(row["group"]),
                "replica": int(row["replica"]),
                "turn": 6,
                "uid": "uid-0",
                "trajectory_seed": int(row["trajectory_seed"]),
                "dataset_index": 0,
                "request_seed": derive_turn_request_seeds(
                    [int(row["trajectory_seed"])], [0], 6
                )[0],
                "is_final": True,
                "mode": "independent",
            })
        ok, failures = audit_seeds(
            records, 2026, 2, expected_steps=[1], expected_batch_size=2
        )
        self.assertTrue(ok, failures)

        records = [
            record for record in records
            if not (
                record.get("record_type") == "trajectory_turn_seed"
                and record.get("sample_index") == 1
                and record.get("turn") == 3
            )
        ]
        ok, failures = audit_seeds(
            records, 2026, 2, expected_steps=[1], expected_batch_size=2
        )
        self.assertFalse(ok)
        self.assertTrue(any("active trajectory turns" in failure for failure in failures))

    def test_seed_audit_rejects_missing_step_group_and_cursor_reset(self):
        def records_for_step(step):
            rows = build_trajectory_seed_records(
                base_seed=2026, global_step=step, batch_size=8, rollout_n=2,
                mode="independent",
            )
            result = []
            for row in rows:
                uid = f"step-{step}-group-{row['group']}"
                dataset_index = (step - 1) * 4 + row["group"]
                result.append({
                    **row, "record_type": "trajectory_seed", "uid": uid,
                    "dataset_index": dataset_index,
                })
                result.append({
                    "record_type": "trajectory_turn_seed",
                    "global_step": step,
                    "row": row["row"],
                    "sample_index": row["row"],
                    "group": row["group"],
                    "replica": row["replica"],
                    "turn": 0,
                    "uid": uid,
                    "trajectory_seed": row["trajectory_seed"],
                    "dataset_index": dataset_index,
                    "request_seed": derive_turn_request_seeds(
                        [row["trajectory_seed"]], [0], 0
                    )[0],
                    "is_final": True,
                    "mode": "independent",
                })
            return result

        records = records_for_step(1) + records_for_step(2)
        ok, failures = audit_seeds(
            records, 2026, 2, expected_steps=[1, 2], expected_batch_size=8
        )
        self.assertTrue(ok, failures)

        only_step_one = [row for row in records if row["global_step"] == 1]
        ok, failures = audit_seeds(
            only_step_one, 2026, 2, expected_steps=[1, 2], expected_batch_size=8
        )
        self.assertFalse(ok)
        self.assertTrue(any("step coverage" in failure for failure in failures))

        missing_group = [
            row for row in records
            if not (row["global_step"] == 2 and row["group"] == 3)
        ]
        ok, failures = audit_seeds(
            missing_group, 2026, 2, expected_steps=[1, 2], expected_batch_size=8
        )
        self.assertFalse(ok)
        self.assertTrue(any("trajectory count" in failure for failure in failures))

        reset_cursor = [dict(row) for row in records]
        for row in reset_cursor:
            if row["global_step"] == 2:
                row["dataset_index"] -= 4
        ok, failures = audit_seeds(
            reset_cursor, 2026, 2, expected_steps=[1, 2], expected_batch_size=8
        )
        self.assertFalse(ok)
        self.assertTrue(any("dataset cursor" in failure for failure in failures))

    def test_nonfinite_metrics_are_preserved_as_failure_evidence(self):
        finite, nonfinite = partition_numeric_metrics({
            "actor/pg_loss": 0.1,
            "critic/rewards/mean": float("nan"),
            "critic/rewards/max": float("inf"),
            "timing/gen": 1.0,
        })
        self.assertEqual(finite, {"actor/pg_loss": 0.1})
        self.assertEqual(
            nonfinite, ["critic/rewards/max", "critic/rewards/mean"]
        )

    def test_two_worker_sync_ack(self):
        parameters = ["model.layers.0.input_layernorm.weight"]
        records = []
        for rank in range(2):
            records.append({
                "record_type": "weight_sync_ack",
                "actor_version": 2,
                "vllm_worker_rank": rank,
                "vllm_ack_version": 2,
                "actor_master_sampled_tensor_digest": "b" * 64,
                "actor_rollout_sampled_tensor_digest": "a" * 64,
                "actor_sampled_tensor_digest": "a" * 64,
                "vllm_sampled_tensor_digest": "a" * 64,
                "weight_transfer_format": "dtensor",
                "loaded_parameter_count": 10,
                "model_parameter_count": 10,
                "loaded_parameter_names_sha256": "c" * 64,
                "model_parameter_names_sha256": "c" * 64,
                "audited_loaded_parameters": parameters,
                "sampled_parameter_dtypes": {parameters[0]: "torch.bfloat16"},
            })
        records.append({
            "record_type": "weight_sync_summary",
            "actor_version": 2,
            "worker_ranks": [0, 1],
            "sampled_tensor_digest": "a" * 64,
            "actor_master_sampled_tensor_digest": "b" * 64,
        })
        ok, failures, digests = audit_sync(
            records, [2], [0, 1], required_parameters=parameters,
            required_transfer_format="dtensor", expected_loaded_parameter_count=10,
        )
        self.assertTrue(ok, failures)
        self.assertEqual(digests, {2: "a" * 64})
        ok, failures, _ = audit_sync(
            records, [2], [0, 1], required_parameters=parameters,
            required_transfer_format="dtensor", expected_loaded_parameter_count=199,
        )
        self.assertFalse(ok)
        self.assertTrue(any("full load count mismatch" in failure for failure in failures))
        records = [
            row for row in records
            if not (
                row.get("record_type") == "weight_sync_ack"
                and row.get("vllm_worker_rank") == 1
            )
        ]
        ok, failures, _ = audit_sync(
            records, [2], [0, 1], required_parameters=parameters,
            required_transfer_format="dtensor", expected_loaded_parameter_count=10,
        )
        self.assertFalse(ok)
        self.assertTrue(any("ack ranks" in failure for failure in failures))

    def test_sync_audit_rejects_missing_load_coverage(self):
        parameters = ["model.layers.0.input_layernorm.weight"]
        records = [{
            "record_type": "weight_sync_ack",
            "actor_version": 2,
            "vllm_worker_rank": rank,
            "vllm_ack_version": 2,
            "actor_master_sampled_tensor_digest": "b" * 64,
            "actor_rollout_sampled_tensor_digest": "a" * 64,
            "actor_sampled_tensor_digest": "a" * 64,
            "vllm_sampled_tensor_digest": "a" * 64,
            "weight_transfer_format": "dtensor",
            "loaded_parameter_count": 10,
            "model_parameter_count": 10,
            "loaded_parameter_names_sha256": "c" * 64,
            "model_parameter_names_sha256": "c" * 64,
            "audited_loaded_parameters": [],
            "sampled_parameter_dtypes": {parameters[0]: "torch.bfloat16"},
        } for rank in range(2)]
        records.append({
            "record_type": "weight_sync_summary",
            "actor_version": 2,
            "worker_ranks": [0, 1],
            "sampled_tensor_digest": "a" * 64,
            "actor_master_sampled_tensor_digest": "b" * 64,
        })
        ok, failures, _ = audit_sync(
            records, [2], [0, 1], required_parameters=parameters,
            required_transfer_format="dtensor", expected_loaded_parameter_count=10,
        )
        self.assertFalse(ok)
        self.assertTrue(any("load coverage mismatch" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
