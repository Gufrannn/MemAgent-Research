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

from recurrent.research.gate_a_execution import append_jsonl, checkpoint_inventory
from recurrent.research.gate_a_execution import load_frozen_manifest
from tools.h20.audit_qwen25_7b_gatea import audit_seeds, audit_sync, component_inventory
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
        })
        self.assertEqual(commands["ledger_schema"], "gate_a_execution_ledger.schema.json")
        self.assertEqual(commands["required_environment"], ["WORK_ROOT", "REPO_DIR"])
        self.assertEqual(commands["working_directory"], "${REPO_DIR}")
        self.assertEqual(commands["required_sequence"], ["p0", "p1", "p2", "audit"])
        self.assertIn("weight_sync_ack", schema["properties"]["record_type"]["enum"])

    def test_runtime_paths_require_explicit_binding_without_selecting_repo(self):
        path = REPO / "manifests/h20/qwen25_7b_gatea_seed2026.yaml"
        first = load_frozen_manifest(path, {
            "WORK_ROOT": "/data/cw/memagent_work",
            "REPO_DIR": "/data/cw/memagent_work/code/MemAgent-Research",
        })
        second = load_frozen_manifest(path, {
            "WORK_ROOT": "/data/cw/memagent_work",
            "REPO_DIR": "/data/cw/memagent_work/MemAgent-Research",
        })
        self.assertEqual(first["repository"], "/data/cw/memagent_work/code/MemAgent-Research")
        self.assertEqual(second["repository"], "/data/cw/memagent_work/MemAgent-Research")
        with self.assertRaisesRegex(ValueError, "missing explicit runtime path bindings"):
            load_frozen_manifest(path, {})


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
            self.assertEqual(json.loads((root / "ledger.jsonl").read_text()), {"a": 1, "b": 2})
            (root / "step/actor").mkdir(parents=True)
            (root / "step/actor/model_world_size_2_rank_0.pt").write_bytes(b"model")
            inventory = checkpoint_inventory(root / "step")
            self.assertEqual(inventory[0]["size"], 5)
            self.assertEqual(inventory[0]["sha256"], hashlib.sha256(b"model").hexdigest())

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

    def test_seed_schedule_reconstruction(self):
        rows = build_trajectory_seed_records(
            base_seed=2026, global_step=2, batch_size=8, rollout_n=2, mode="independent"
        )
        for row in rows:
            row["record_type"] = "trajectory_seed"
            row["uid"] = f"uid-{row['group']}"
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
                "request_seed": derive_turn_request_seeds([row["trajectory_seed"]], [0], 0)[0],
                "mode": "independent",
            })
        records = rows + turn_rows
        ok, failures = audit_seeds(records, 2026, 2)
        self.assertTrue(ok, failures)
        rows[1]["trajectory_seed"] = rows[0]["trajectory_seed"]
        ok, failures = audit_seeds(records, 2026, 2)
        self.assertFalse(ok)
        self.assertTrue(any("collision" in failure for failure in failures))

    def test_two_worker_sync_ack(self):
        records = []
        for rank in range(2):
            records.append({
                "record_type": "weight_sync_ack",
                "actor_version": 2,
                "vllm_worker_rank": rank,
                "vllm_ack_version": 2,
                "actor_sampled_tensor_digest": "a" * 64,
                "vllm_sampled_tensor_digest": "a" * 64,
            })
        ok, failures, digests = audit_sync(records, [2], [0, 1])
        self.assertTrue(ok, failures)
        self.assertEqual(digests, {2: "a" * 64})
        records.pop()
        ok, failures, _ = audit_sync(records, [2], [0, 1])
        self.assertFalse(ok)
        self.assertTrue(any("ack ranks" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
