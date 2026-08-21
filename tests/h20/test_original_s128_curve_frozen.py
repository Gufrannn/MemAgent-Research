from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from recurrent.research.gate_a_execution import append_jsonl, checkpoint_inventory
from recurrent.research.stable_eval_identity import canonical_sha256
from tools.h20.audit_qwen25_7b_original_s128_curve import (
    _audit_actor_only_load,
    _audit_ledger,
    build_curve_summary,
)
from tools.h20.preflight_qwen25_7b_original_s128_curve import (
    ANCHOR_STEPS,
    INTERFACES,
    SOURCE_COMMIT,
    _complete_checkpoint_inventory,
    _artifact_inventory,
    _training_prefix_order_contract,
    _training_sync_contract,
    build_interface_plan,
    validate_prior_import,
    validate_training_source,
)


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "manifests/h20/qwen25_7b_original_s128_curve_seed2026.json"
COMMANDS = REPO / "manifests/h20/qwen25_7b_original_s128_curve_commands.json"
RUNNER = REPO / "scripts/h20/run_qwen25_7b_original_s128_curve.sh"
COMMON = REPO / "scripts/h20/original_s128_curve_common.sh"
EXPERIMENT = (
    "qwen25_7b_h20_corrected_original_style_2gpu_pilot_resume3to25_"
    "strictvllm_naive_indseed_seed2026_20260821"
)


def _write_checkpoint(root: Path, step: int) -> list[dict]:
    step_root = root / f"global_step_{step}"
    actor = step_root / "actor"
    actor.mkdir(parents=True)
    for component in ("model", "optim", "extra_state"):
        for rank in (0, 1):
            (actor / f"{component}_world_size_2_rank_{rank}.pt").write_bytes(
                f"{step}:{component}:{rank}".encode()
            )
    (step_root / "data.pt").write_bytes(f"cursor:{step}".encode())
    return checkpoint_inventory(step_root)


def _sync_rows(
    step: int, digest: str, master: str, *, sync_kind: str = "post_actor_update"
) -> list[dict]:
    rows = []
    audited = [f"parameter.{index}" for index in range(8)]
    for rank in (0, 1):
        rows.append({
            "record_type": "weight_sync_ack",
            "global_step": step,
            "actor_version": step,
            "sync_kind": sync_kind,
            "vllm_worker_rank": rank,
            "vllm_ack_version": step,
            "actor_master_sampled_tensor_digest": master,
            "actor_rollout_sampled_tensor_digest": digest,
            "actor_sampled_tensor_digest": digest,
            "vllm_sampled_tensor_digest": digest,
            "weight_transfer_format": "dtensor",
            "loaded_parameter_count": 199,
            "model_parameter_count": 199,
            "loaded_parameter_names_sha256": "c" * 64,
            "model_parameter_names_sha256": "c" * 64,
            "optimizer_step_max": step,
            "lr_scheduler_last_epoch": step,
            "optimizer_state_entry_count": 4,
            "optimizer_step_entry_count": 4,
            "optimizer_step_histogram": {str(step): 4},
            "audited_loaded_parameters": audited,
            "sampled_parameter_dtypes": {
                name: "torch.bfloat16" for name in audited
            },
        })
    rows.append({
        "record_type": "weight_sync_summary",
        "global_step": step,
        "actor_version": step,
        "sync_kind": sync_kind,
        "worker_ranks": [0, 1],
        "sampled_tensor_digest": digest,
        "actor_master_sampled_tensor_digest": master,
    })
    return rows


def _write_training_attestation(root: Path, stable: dict) -> tuple[dict, Path]:
    checkpoint_root = root / "checkpoints"
    anchors = []
    digests = {str(step): f"{step:064x}" for step in range(3, 26)}
    for step in ANCHOR_STEPS:
        inventory = _write_checkpoint(checkpoint_root, step)
        anchors.append({
            "path": str(checkpoint_root / f"global_step_{step}"),
            "global_step": step,
            "inventory": inventory,
            "inventory_sha256": canonical_sha256(inventory),
        })

    ledger = root / "training.jsonl"
    common = {
        "git_commit": SOURCE_COMMIT,
        "experiment_name": EXPERIMENT,
        "run_id": "1" * 32,
    }
    append_jsonl(ledger, {**common, "record_type": "p0_preflight"})
    append_jsonl(ledger, {**common, "record_type": "runtime_config"})
    append_jsonl(ledger, {**common, "record_type": "resume_load"})
    for step in range(3, 26):
        if step > 3:
            append_jsonl(ledger, {
                **common,
                "record_type": "rollout_start",
                "global_step": step,
                "actor_version": step - 1,
                "sampled_tensor_digest": digests[str(step - 1)],
            })
        digest = digests[str(step)]
        master = f"{step + 100:064x}"
        sync_kind = "resume_loaded" if step == 3 else "post_actor_update"
        for row in _sync_rows(step, digest, master, sync_kind=sync_kind):
            append_jsonl(ledger, {**common, **row})
        if step > 3:
            append_jsonl(ledger, {
                **common,
                "record_type": "execution_signal",
                "global_step": step,
                "actor_version": step,
                "metrics": {"actor/grad_norm": 1.0},
                "nonfinite_metric_names": [],
            })
    prefix = [json.loads(line) for line in ledger.read_text().splitlines()]
    report = root / "training_report.json"
    report_payload = {
        "status": "PASS",
        "decision": "ORIGINAL_T25_PASS",
        "failures": [],
        "git_commit": SOURCE_COMMIT,
        "experiment_name": EXPERIMENT,
        "not_original_paper_7b_reproduction": True,
        "stable_identity_prerequisite": {
            "report_sha256": stable["sha256"],
            "eval_manifest_hash": stable["eval_manifest_hash"],
            "commit": stable["git_commit"],
        },
        "weight_sync": {"version_digests": digests},
        "checkpoint_anchors": anchors,
        "checkpoint_anchors_sha256": canonical_sha256(anchors),
        "step25_checkpoint": dict(anchors[-1]),
        "execution_ledger": {
            "training_prefix_record_count": len(prefix),
            "training_prefix_sha256": canonical_sha256(prefix),
            "training_prefix_tail_sha256": prefix[-1]["record_sha256"],
        },
    }
    report.write_text(json.dumps(report_payload))
    append_jsonl(ledger, {
        **common,
        "record_type": "checkpoint_inventory",
        "global_step": 25,
        "inventory": anchors[-1]["inventory"],
        "inventory_sha256": anchors[-1]["inventory_sha256"],
        "checkpoint_anchors": anchors,
        "checkpoint_anchors_sha256": canonical_sha256(anchors),
    })
    append_jsonl(ledger, {
        **common,
        "record_type": "audit_result",
        "status": "PASS",
        "decision": "ORIGINAL_T25_PASS",
        "report": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    })
    manifest = {
        "training_source": {
            "final_report": str(report),
            "execution_ledger": str(ledger),
            "checkpoint_root": str(checkpoint_root),
            "required_status": "PASS",
            "required_decision": "ORIGINAL_T25_PASS",
            "required_git_commit": SOURCE_COMMIT,
            "experiment_name": EXPERIMENT,
        }
    }
    return manifest, report


def _resolved_payload(source_commit: str) -> tuple[dict, str]:
    rows = [{
        "example_id": str(index + 7),
        "semantic_dataset_index": index + 7,
        "source_order_index": index,
        "raw_row_position": index,
        "production_effective_position": index,
        "context_token_count": 1,
        "source_question_hash": "1" * 64,
        "source_context_hash": "2" * 64,
        "ground_truth_hash": "3" * 64,
    } for index in range(128)]
    payload = {"schema_version": 1, "rows": rows}
    digest = canonical_sha256(payload)
    return {
        "identity_payload": payload,
        "eval_manifest_hash": digest,
        "cohort": {
            "source_order_indices": list(range(128)),
            "examples": 128,
            "replicas": 1,
            "interfaces": ["I", "T25"],
        },
        "execution_binding": {
            "git_commit": source_commit,
            "interfaces": ["I", "T25"],
            "base_seed": 2026,
            "replicas": 1,
            "execution_code_sha256": {
                "recurrent/research/s128_hotpot_metrics.py": "addc282e3d48dc5e7b6ccf30205fde58a0c0515cb6a2341fd293a5b5b83da286"
            },
            "trainer_configuration": {
                "shared_generation_protocol_sha256": "4" * 64,
                "interfaces": {
                    "I": {"resolved_config_sha256": "5" * 64},
                    "T25": {"resolved_config_sha256": "6" * 64},
                },
            },
        },
    }, digest


def _write_artifacts(root: Path, step: int) -> None:
    (root / "terminal").mkdir(parents=True)
    (root / "terminal" / f"{step}.jsonl").write_text("{}\n")
    (root / "trajectory_turns.jsonl").write_text("{}\n")
    (root / "execution_summary.json").write_text("{}\n")
    (root / "run.log").write_text("done\n")


def _write_prior_import(root: Path) -> tuple[dict, str]:
    source_commit = SOURCE_COMMIT
    resolved, eval_hash = _resolved_payload(source_commit)
    resolved_path = root / "resolved.json"
    resolved_path.write_text(json.dumps(resolved))
    p0_path = root / "p0.json"
    p0_payload = {
        "status": "PASS",
        "decision": "S128_IT_P0_PASS",
        "evidence": {
            "git_commit": source_commit,
            "branch": "h20/qwen25-7b-original-t25-s128-frozen-20260821",
            "resolved_manifest_sha256": hashlib.sha256(resolved_path.read_bytes()).hexdigest(),
            "runtime_binding_sha256": "7" * 64,
            "run_id": "8" * 32,
        },
    }
    p0_path.write_text(json.dumps(p0_payload))
    i_root, t_root = root / "i", root / "t25"
    _write_artifacts(i_root, 0)
    _write_artifacts(t_root, 25)
    ledger = root / "ledger.jsonl"
    execution_sha = canonical_sha256(resolved["execution_binding"])
    common = {
        "git_commit": source_commit,
        "run_id": "8" * 32,
        "eval_manifest_hash": eval_hash,
        "execution_binding_sha256": execution_sha,
        "runtime_binding_sha256": "7" * 64,
        "status": "PASS",
    }
    experiments = (
        "qwen25_7b_s128_it_p0_seed2026_20260821",
        "qwen25_7b_s128_i_base_seed2026_20260821",
        "qwen25_7b_s128_i_base_seed2026_20260821",
        "qwen25_7b_s128_t25_corrected_original_style_seed2026_20260821",
        "qwen25_7b_s128_t25_corrected_original_style_seed2026_20260821",
    )
    append_jsonl(ledger, {
        **common, "record_type": "s0_preflight", "interface_id": None,
        "experiment_name": experiments[0], "row_count": 128,
        "artifact": str(p0_path),
        "artifact_sha256": hashlib.sha256(p0_path.read_bytes()).hexdigest(),
    })
    for offset, (interface, interface_root, step) in enumerate(
        (("I", i_root, 0), ("T25", t_root, 25))
    ):
        append_jsonl(ledger, {
            **common, "record_type": "interface_start", "interface_id": interface,
            "experiment_name": experiments[1 + offset * 2], "artifacts": {},
        })
        from tools.h20.preflight_qwen25_7b_original_s128_curve import _artifact_inventory
        append_jsonl(ledger, {
            **common, "record_type": "interface_finish", "interface_id": interface,
            "experiment_name": experiments[2 + offset * 2],
            "artifacts": _artifact_inventory(interface_root, step=step),
        })
    prefix_sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    report_path = root / "report.json"
    report = {
        "status": "PASS", "decision": "S128_IT_PERFORMANCE_PASS", "failures": [],
        "evidence": {
            "eval_manifest_hash": eval_hash,
            "execution_ledger_records": 5,
            "execution_ledger_sha256": prefix_sha,
        },
    }
    report_path.write_text(json.dumps(report))
    append_jsonl(ledger, {
        **common, "record_type": "audit_result", "interface_id": None,
        "experiment_name": "qwen25_7b_s128_it_audit_seed2026_20260821",
        "decision": "S128_IT_PERFORMANCE_PASS", "artifact": str(report_path),
        "artifact_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    })
    spec = {
        "prior_s128_it_import": {
            "source_commit": source_commit,
            "p0_certificate": str(p0_path),
            "resolved_manifest": str(resolved_path),
            "final_report": str(report_path),
            "execution_ledger": str(ledger),
            "interfaces": {
                "I": {"source_interface": "I", "source_attempt": "base_i", "global_step": 0, "root": str(i_root)},
                "Original25": {"source_interface": "T25", "source_attempt": "original_t25", "global_step": 25, "root": str(t_root)},
            },
        }
    }
    return spec, eval_hash


class OriginalS128CurveFrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.commands = json.loads(COMMANDS.read_text())

    def test_curve_contract_is_exact_existing_s128_no_training(self) -> None:
        self.assertEqual(tuple(self.manifest["evaluation"]["interfaces"]), INTERFACES)
        self.assertEqual(tuple(self.manifest["training_source"]["anchor_steps"]), ANCHOR_STEPS)
        self.assertEqual(self.manifest["inherited_s128_contract"]["required_eval_manifest_hash"], "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a")
        self.assertIs(self.manifest["scope"]["existing_s128_only"], True)
        self.assertIs(self.manifest["scope"]["no_resampling"], True)
        self.assertIs(self.manifest["scope"]["no_training"], True)
        self.assertEqual(self.manifest["evaluation"]["replicas"], 1)
        self.assertIs(self.manifest["evaluation"]["do_sample"], False)
        self.assertEqual(self.manifest["evaluation"]["primary_metrics"], ["normalized_exact_match", "token_f1"])
        self.assertIs(self.manifest["evaluation"]["training_dense_reward_excluded_from_evaluation_claims"], True)

    def test_complete_checkpoint_and_each_training_sync_are_authenticated(self) -> None:
        stable = {
            "sha256": "a" * 64,
            "eval_manifest_hash": "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a",
            "git_commit": "c" * 40,
        }
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = _write_training_attestation(Path(temporary), stable)
            result = validate_training_source(manifest, stable)
            self.assertEqual(tuple(result["anchors"]), INTERFACES[1:])
            for step in ANCHOR_STEPS:
                anchor = result["anchors"][f"Original{step}"]
                self.assertEqual(len(anchor["inventory"]), 7)
                self.assertEqual(anchor["training_effective_actor_vllm_digest"], f"{step:064x}")
                self.assertEqual(anchor["training_actor_master_sampled_tensor_digest"], f"{step + 100:064x}")
            shard = Path(result["anchors"]["Original10"]["path"]) / "actor/model_world_size_2_rank_1.pt"
            shard.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                validate_training_source(manifest, stable)

    def test_sync_cross_anchor_or_missing_rank_fails_closed(self) -> None:
        rows = _sync_rows(10, "a" * 64, "b" * 64)
        self.assertEqual(_training_sync_contract(rows, step=10, report_digest="a" * 64)["actor_master_digest"], "b" * 64)
        with self.assertRaisesRegex(ValueError, "do not close"):
            _training_sync_contract(rows, step=10, report_digest="d" * 64)
        with self.assertRaisesRegex(ValueError, "worker-rank"):
            _training_sync_contract(rows[1:], step=10, report_digest="a" * 64)

    def test_sync_protocol_rejects_bool_and_float_integer_coercions(self) -> None:
        rows = _sync_rows(10, "a" * 64, "b" * 64)
        bool_ranks = json.loads(json.dumps(rows))
        bool_ranks[0]["vllm_worker_rank"] = False
        bool_ranks[1]["vllm_worker_rank"] = True
        with self.assertRaisesRegex(ValueError, "JSON integer"):
            _training_sync_contract(
                bool_ranks, step=10, report_digest="a" * 64
            )

        float_steps = json.loads(json.dumps(rows))
        for row in float_steps:
            row["global_step"] = 10.0
            row["actor_version"] = 10.0
        with self.assertRaises(ValueError):
            _training_sync_contract(
                float_steps, step=10, report_digest="a" * 64
            )

        float_count = json.loads(json.dumps(rows))
        float_count[0]["loaded_parameter_count"] = 199.0
        with self.assertRaises(ValueError):
            _training_sync_contract(
                float_count, step=10, report_digest="a" * 64
            )

    def test_training_prefix_replays_all_versions_in_semantic_order(self) -> None:
        stable = {
            "sha256": "a" * 64,
            "eval_manifest_hash": (
                "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a"
            ),
            "git_commit": "c" * 40,
        }
        with tempfile.TemporaryDirectory() as temporary:
            manifest, report_path = _write_training_attestation(
                Path(temporary), stable
            )
            ledger_path = Path(manifest["training_source"]["execution_ledger"])
            records = [
                json.loads(line) for line in ledger_path.read_text().splitlines()
            ][:-2]
            version_digests = json.loads(report_path.read_text())["weight_sync"][
                "version_digests"
            ]
            _training_prefix_order_contract(
                records, version_digests=version_digests
            )

            reordered = json.loads(json.dumps(records))
            reordered[8], reordered[9] = reordered[9], reordered[8]
            with self.assertRaisesRegex(ValueError, "sync group order"):
                _training_prefix_order_contract(
                    reordered, version_digests=version_digests
                )

            wrong_signal = json.loads(json.dumps(records))
            first_signal = next(
                row for row in wrong_signal
                if row.get("record_type") == "execution_signal"
            )
            first_signal["actor_version"] = 3
            with self.assertRaisesRegex(ValueError, "execution signal"):
                _training_prefix_order_contract(
                    wrong_signal, version_digests=version_digests
                )

    def test_checkpoint_inventory_rejects_missing_or_extra_state(self) -> None:
        inventory = [{"path": "data.pt", "size": 1, "sha256": "a" * 64}]
        with self.assertRaisesRegex(ValueError, "7-file"):
            _complete_checkpoint_inventory(inventory, step=5)

    def test_prior_i_t25_import_is_hash_authenticated_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, eval_hash = _write_prior_import(root)
            prior = validate_prior_import(manifest, expected_eval_manifest_hash=eval_hash)
            self.assertIs(prior["available"], True)
            plan_manifest = {"evaluation": {"attempt_ids": {name: name for name in INTERFACES}}, "paths": {name: str(root / name) for name in INTERFACES}}
            plan = build_interface_plan(plan_manifest, prior)
            self.assertEqual(plan["I"]["mode"], "import")
            self.assertEqual(plan["Original25"]["mode"], "import")
            self.assertTrue(all(plan[name]["mode"] == "run" for name in INTERFACES[1:-1]))
            source_terminal = Path(prior["interfaces"]["I"]["root"]) / "terminal/0.jsonl"
            source_terminal.write_text("tamper\n")
            with self.assertRaisesRegex(ValueError, "artifact hashes"):
                validate_prior_import(manifest, expected_eval_manifest_hash=eval_hash)

    def test_partial_prior_import_never_silently_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "only-p0.json"
            path.write_text("{}")
            manifest = {"prior_s128_it_import": {
                "source_commit": SOURCE_COMMIT,
                "p0_certificate": str(path),
                "resolved_manifest": str(path.parent / "missing-resolved"),
                "final_report": str(path.parent / "missing-report"),
                "execution_ledger": str(path.parent / "missing-ledger"),
                "interfaces": {
                    "I": {"root": str(path.parent / "missing-i")},
                    "Original25": {"root": str(path.parent / "missing-t")},
                },
            }}
            with self.assertRaisesRegex(ValueError, "partially present"):
                validate_prior_import(manifest, expected_eval_manifest_hash="a" * 64)

    def test_missing_prior_import_never_expands_to_six_gpu_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"prior_s128_it_import": {
                "source_commit": SOURCE_COMMIT,
                "p0_certificate": str(root / "p0"),
                "resolved_manifest": str(root / "resolved"),
                "final_report": str(root / "report"),
                "execution_ledger": str(root / "ledger"),
                "interfaces": {
                    "I": {"root": str(root / "i")},
                    "Original25": {"root": str(root / "t25")},
                },
            }}
            with self.assertRaisesRegex(ValueError, "required for read-only import"):
                validate_prior_import(
                    manifest, expected_eval_manifest_hash="a" * 64
                )
            with self.assertRaisesRegex(ValueError, "mandatory"):
                build_interface_plan(
                    {"evaluation": {"attempt_ids": {}}, "paths": {}},
                    {"available": False, "interfaces": {}},
                )

    def test_curve_summary_keeps_fixed_order_and_allows_nonmonotonic_results(self) -> None:
        metrics = {}
        successes = {"I": 12, "Original5": 20, "Original10": 8, "Original15": 31, "Original20": 30, "Original25": 40}
        for name in INTERFACES:
            metrics[name] = [{
                "stable_key": str(index),
                "exact_match": float(index < successes[name]),
                "token_f1": float(index < successes[name]),
                "format_success": 1.0,
                "sub_exact_match": float(index < successes[name]),
            } for index in range(128)]
        curve = build_curve_summary(metrics)
        self.assertEqual(curve["order"], list(INTERFACES))
        self.assertIs(curve["monotonicity_required"], False)
        self.assertLess(curve["points"]["Original10"]["token_f1"], curve["points"]["Original5"]["token_f1"])
        self.assertEqual(set(curve["paired_descriptive_vs_I"]), set(INTERFACES[1:]))

    def test_actor_only_ack_cannot_be_reused_for_another_anchor(self) -> None:
        checkpoint = {
            "path": "/frozen/global_step_5",
            "actor_model_shards": [
                {"path": f"actor/model_world_size_2_rank_{rank}.pt", "size": 10 + rank, "sha256": f"{rank + 1:064x}"}
                for rank in (0, 1)
            ],
        }
        acks = [{
            "rank": rank, "world_size": 2, "model_loaded": True,
            "model_shard_path": f"/frozen/global_step_5/actor/model_world_size_2_rank_{rank}.pt",
            "model_shard_size": 10 + rank, "model_shard_sha256": f"{rank + 1:064x}",
            "optimizer_loaded": False, "lr_scheduler_loaded": False,
            "rng_loaded": False, "dataloader_loaded": False,
            "optimizer_state_entry_count_before": 0,
            "optimizer_state_entry_count_after": 0,
            "lr_scheduler_last_epoch_before": 0,
            "lr_scheduler_last_epoch_after": 0,
        } for rank in (0, 1)]
        self.assertEqual(_audit_actor_only_load({"actor_checkpoint_load_acks": acks}, checkpoint), [])
        other = json.loads(json.dumps(checkpoint))
        other["actor_model_shards"][1]["sha256"] = "f" * 64
        self.assertTrue(_audit_actor_only_load({"actor_checkpoint_load_acks": acks}, other))

    def test_runner_imports_without_copy_and_clears_shared_writer_environment(self) -> None:
        runner = RUNNER.read_text()
        common = COMMON.read_text()
        self.assertIn("locks/memagent_gate_a_gpu_6_7.lock", common)
        self.assertIn("compgen -v GATE_A_", runner)
        self.assertIn("compgen -v ORIGINAL_T25_", runner)
        self.assertIn("if [[ $mode == import ]]", runner)
        self.assertNotIn("cp ", runner)
        self.assertNotIn("rsync", runner)
        self.assertLess(runner.index('if [[ $mode == import ]]'), runner.index('mkdir -p "$root"'))
        self.assertNotIn("main_ppo", json.dumps(self.commands["commands"]["p0"]))

    def test_curve_ledger_distinguishes_imports_from_new_gpu_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p0_path = root / "p0.json"
            p0_path.write_text("{}")
            plans = {}
            for interface_id in INTERFACES:
                interface_root = root / interface_id
                step = 0 if interface_id == "I" else int(interface_id.removeprefix("Original"))
                _write_artifacts(interface_root, step)
                plans[interface_id] = {
                    "mode": "import" if interface_id in ("I", "Original25") else "run",
                    "source_interface": "T25" if interface_id == "Original25" else interface_id,
                    "source_attempt": "source",
                    "global_step": step,
                    "root": str(interface_root),
                    "artifacts": _artifact_inventory(interface_root, step=step),
                }
            prior = {
                "p0_certificate_sha256": "1" * 64,
                "resolved_manifest_sha256": "2" * 64,
                "final_report_sha256": "3" * 64,
                "execution_ledger_sha256": "4" * 64,
                "execution_ledger_tail_sha256": "5" * 64,
            }
            execution = {"interface_plan": plans, "prior_s128_it_import": prior}
            resolved = {"execution_binding": execution}
            p0 = {"evidence": {
                "git_commit": SOURCE_COMMIT,
                "run_id": "6" * 32,
                "runtime_binding_sha256": "7" * 64,
            }}
            common = {
                "git_commit": SOURCE_COMMIT,
                "run_id": "6" * 32,
                "recorded_at": "2026-08-21T00:00:00+00:00",
                "eval_manifest_hash": "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a",
                "execution_binding_sha256": canonical_sha256(execution),
                "runtime_binding_sha256": "7" * 64,
                "status": "PASS",
            }
            ledger = root / "ledger.jsonl"
            append_jsonl(ledger, {
                **common, "record_type": "p0_preflight",
                "experiment_name": "qwen25_7b_original_s128_curve_p0_seed2026_20260821",
                "interface_id": None, "mode": None, "row_count": 128,
                "artifact": str(p0_path),
                "artifact_sha256": hashlib.sha256(p0_path.read_bytes()).hexdigest(),
            })
            for interface_id in INTERFACES:
                plan = plans[interface_id]
                if plan["mode"] == "import":
                    append_jsonl(ledger, {
                        **common, "record_type": "source_import",
                        "experiment_name": (
                            "qwen25_7b_s128_curve_i_base_seed2026_20260821"
                            if interface_id == "I" else
                            "qwen25_7b_s128_curve_original_step25_seed2026_20260821"
                        ),
                        "interface_id": interface_id, "mode": "import",
                        "artifacts": plan["artifacts"],
                        "source_interface": plan["source_interface"],
                        "source_attempt": "source", "source_root": plan["root"],
                        **{f"source_{key}": value for key, value in prior.items()},
                    })
                    continue
                experiment = f"qwen25_7b_s128_curve_original_step{plan['global_step']}_seed2026_20260821"
                append_jsonl(ledger, {
                    **common, "record_type": "interface_start",
                    "experiment_name": experiment, "interface_id": interface_id,
                    "mode": "run", "artifacts": {},
                })
                append_jsonl(ledger, {
                    **common, "record_type": "interface_finish",
                    "experiment_name": experiment, "interface_id": interface_id,
                    "mode": "run", "artifacts": plan["artifacts"],
                })
            records = [json.loads(line) for line in ledger.read_text().splitlines()]
            manifest = {
                "repository": str(REPO),
                "ledger_schema": "original_s128_curve_execution_ledger.schema.json",
                "paths": {"p0_certificate": str(p0_path), "final_report": str(root / "final.json")},
            }
            self.assertEqual(_audit_ledger(records, manifest=manifest, resolved=resolved, p0=p0), [])
            records[-1]["source_final_report_sha256"] = "f" * 64
            failures = _audit_ledger(records, manifest=manifest, resolved=resolved, p0=p0)
            self.assertTrue(any("source-import" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
