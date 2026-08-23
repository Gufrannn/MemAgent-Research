import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recurrent.research.coral_scope_audit import (
    content_identity_from_hashes,
    identity_inventory,
    overlap,
    parquet_row_identity,
    stable_row_identity,
    static_budget,
    validate_scope_report,
)
from recurrent.research.cosi import append_ledger, canonical_sha256
from recurrent.research.gate_a_execution import append_jsonl
from recurrent.research.trajectory_seeding import (
    build_trajectory_seed_records, derive_turn_request_seeds, stable_training_group_id,
)


ROOT = Path(__file__).resolve().parents[2]


class CoralScopeAuditTests(unittest.TestCase):
    def _row(self, question, context, index):
        return {
            "prompt": [{"role": "user", "content": question}],
            "context": context,
            "extra_info": {"index": index},
        }

    def test_scope_entry_reads_no_outcome_columns(self):
        source = (ROOT / "tools/h20/audit_coral_scientific_scope.py").read_text()
        self.assertIn('columns=["prompt", "context", "extra_info"]', source)
        self.assertNotIn('"reward_model"', source)
        self.assertNotIn('"ground_truth"', source)
        self.assertNotIn('parser.add_argument("--run-root")', source)
        self.assertNotIn('parser.add_argument("--p0-t5")', source)

    def test_content_identity_ignores_split_local_index(self):
        left = parquet_row_identity(self._row("q", "c", 3))
        right = parquet_row_identity(self._row("q", "c", 999))
        self.assertEqual(left["content_identity_sha256"], right["content_identity_sha256"])
        self.assertNotEqual(
            left["split_local_semantic_index"], right["split_local_semantic_index"]
        )

    def test_stable_and_parquet_identities_close(self):
        row = parquet_row_identity(self._row("question", "context", 7))
        stable = stable_row_identity({
            "example_id": "7",
            "source_question_hash": row["source_question_sha256"],
            "source_context_hash": row["source_context_sha256"],
        })
        self.assertEqual(row["content_identity_sha256"], stable["content_identity_sha256"])

    def test_overlap_reports_content_and_partial_matches(self):
        a = parquet_row_identity(self._row("same", "context-a", 1))
        b = parquet_row_identity(self._row("same", "context-b", 2))
        result = overlap(identity_inventory([a]), identity_inventory([b]))
        self.assertEqual(result["canonical_content_pair_count"], 0)
        self.assertEqual(result["question_hash_count"], 1)
        self.assertEqual(result["context_hash_count"], 0)

    def test_duplicate_content_fails_closed(self):
        row = parquet_row_identity(self._row("q", "c", 1))
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            identity_inventory([row, dict(row)], require_unique_content=True)
        inventory = identity_inventory([row, dict(row)])
        self.assertEqual(inventory["duplicate_content_row_count"], 1)

    def test_hash_validation(self):
        with self.assertRaisesRegex(ValueError, "invalid question"):
            content_identity_from_hashes("x", "0" * 64)

    def test_static_budget_is_early_pilot(self):
        manifest = {
            "training": {
                "train_batch_size": 4, "rollout_n": 2,
                "ppo_mini_batch_size": 4, "ppo_epochs": 1,
            },
            "role_exposure": {
                "primary_updates": 25, "memory_writer_updates": 13,
                "terminal_answer_updates": 12,
            },
            "protocol": {"advantage_estimator": "grpo"},
        }
        budget = static_budget(manifest)
        self.assertEqual(budget["prompt_groups_planned"], 100)
        self.assertEqual(budget["sampled_training_trajectories_planned"], 200)
        self.assertEqual(budget["actor_optimizer_updates_planned"], 25)
        self.assertEqual(budget["critic_fit_optimizer_updates"], 0)
        self.assertIn("not_convergence", budget["classification"])

    def _write_scope_fixture(self, root: Path, *, leak: bool = False):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:  # pragma: no cover - H20 venv has pyarrow
            self.skipTest(str(error))
        train_rows = [
            {**self._row("train-q1", "train-c1", 1), "reward_model": {"ground_truth": "a"}},
            {**self._row("train-q2", "train-c2", 2), "reward_model": {"ground_truth": "b"}},
        ]
        s128_rows = [
            {**self._row(f"eval-q{index}", f"eval-c{index}", 100 + index),
             "reward_model": {"ground_truth": f"answer-{index}"}}
            for index in range(128)
        ]
        if leak:
            train_rows[0] = {**self._row("eval-q0", "eval-c0", 999),
                             "reward_model": {"ground_truth": "different"}}
        data = root / "datasets/hotpotqa"
        data.mkdir(parents=True)
        train_path = data / "hotpotqa_train_32k.parquet"
        s128_path = data / "hotpotqa_dev.parquet"
        pq.write_table(pa.Table.from_pylist(train_rows), train_path)
        pq.write_table(pa.Table.from_pylist(s128_rows), s128_path)
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

        frozen = []
        for order, source in enumerate(s128_rows):
            identity = parquet_row_identity(source)
            frozen.append({
                "example_id": str(source["extra_info"]["index"]),
                "semantic_dataset_index": source["extra_info"]["index"],
                "source_order_index": order,
                "raw_row_position": order,
                "production_effective_position": order,
                "context_token_count": 1,
                "source_question_hash": identity["source_question_sha256"],
                "source_context_hash": identity["source_context_sha256"],
                "ground_truth_hash": canonical_sha256(source["reward_model"]["ground_truth"]),
            })
        payload = {"rows": frozen}
        stable = {"identity_payload": payload, "eval_manifest_hash": canonical_sha256(payload)}
        stable_path = root / "stable.json"
        stable_path.write_text(json.dumps(stable))
        manifest = {
            "branch": "h20/qwen25-7b-cosi-t25-frozen-20260822",
            "data": {"train_sha256": sha(train_path), "validation_sha256": sha(s128_path)},
            "evaluation": {"eval_manifest_hash": stable["eval_manifest_hash"]},
            "training": {"train_batch_size": 4, "rollout_n": 2,
                         "ppo_mini_batch_size": 4, "ppo_epochs": 1},
            "role_exposure": {"primary_updates": 25, "memory_writer_updates": 13,
                              "terminal_answer_updates": 12},
            "protocol": {"advantage_estimator": "grpo"},
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        source = root / "source"
        source.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=source, check=True)
        subprocess.run(["git", "config", "user.email", "audit@example.invalid"],
                       cwd=source, check=True)
        subprocess.run(["git", "config", "user.name", "Audit Test"], cwd=source, check=True)
        (source / "identity.txt").write_text("fixture\n")
        subprocess.run(["git", "add", "identity.txt"], cwd=source, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                                text=True, capture_output=True, check=True).stdout.strip()
        return manifest_path, stable_path, sha(stable_path), source, commit

    def _validation_kwargs(self, root, manifest_path, stable_path, source, commit):
        manifest = json.loads(manifest_path.read_text())
        return {
            "expected_commit": commit,
            "expected_manifest_path": str(manifest_path.resolve()),
            "expected_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "expected_repo": str(source.resolve()),
            "expected_work_root": str(root.resolve()),
            "expected_train_sha256": manifest["data"]["train_sha256"],
            "expected_s128_parquet_sha256": manifest["data"]["validation_sha256"],
            "expected_s128_resolved_path": str(stable_path.resolve()),
            "expected_s128_resolved_sha256": hashlib.sha256(
                stable_path.read_bytes()
            ).hexdigest(),
            "expected_eval_manifest_hash": manifest["evaluation"]["eval_manifest_hash"],
        }

    def test_real_entry_reports_clear_direct_overlap_and_adaptive_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, stable, stable_sha, source, commit = self._write_scope_fixture(root)
            output = root / "report.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "tools/h20/audit_coral_scientific_scope.py"),
                "--manifest", str(manifest), "--stable-resolved", str(stable),
                "--stable-resolved-sha256", stable_sha, "--work-root", str(root),
                "--repo-dir", str(source), "--expected-commit", commit,
                "--output", str(output),
            ], text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["set_intersections"]
                             ["actor_train_intersection_s128"]
                             ["canonical_content_pair_count"], 0)
            self.assertEqual(report["set_intersections"]
                             ["critic_fit_intersection_s128"]
                             ["canonical_content_pair_count"], 0)
            self.assertEqual(report["set_intersections"]
                             ["selection_intersection_s128"]
                             ["canonical_content_pair_count"], 128)
            self.assertEqual(report["actual_training_budget"]["status"],
                             "PENDING_ACTUAL_T25_LEDGER")
            validation = self._validation_kwargs(
                root, manifest, stable, source, commit
            )
            validate_scope_report(report, **validation)
            forged = json.loads(json.dumps(report))
            forged["set_intersections"]["actor_train_intersection_s128"] \
                ["canonical_content_pair_count"] = 1
            unsigned = {key: value for key, value in forged.items()
                        if key != "report_sha256"}
            forged["report_sha256"] = canonical_sha256(unsigned)
            with self.assertRaisesRegex(ValueError, "invalid intersection"):
                validate_scope_report(forged, **validation)
            forged = json.loads(json.dumps(report))
            forged["set_intersections"]["actor_train_intersection_s128"] \
                ["question_hash_count"] = 1
            unsigned = {key: value for key, value in forged.items()
                        if key != "report_sha256"}
            forged["report_sha256"] = canonical_sha256(unsigned)
            with self.assertRaisesRegex(ValueError, "partial actor"):
                validate_scope_report(forged, **validation)
            forged = json.loads(json.dumps(report))
            forged["data_roles"]["actor_training"][
                "content_identity_inventory_sha256"
            ] = "f" * 64
            unsigned = {key: value for key, value in forged.items()
                        if key != "report_sha256"}
            forged["report_sha256"] = canonical_sha256(unsigned)
            with self.assertRaisesRegex(ValueError, "recomputed inventory"):
                validate_scope_report(forged, **validation)
            forged = json.loads(json.dumps(report))
            forged["static_training_budget"]["memory_writer_active_updates"] = 25
            unsigned = {key: value for key, value in forged.items()
                        if key != "report_sha256"}
            forged["report_sha256"] = canonical_sha256(unsigned)
            with self.assertRaisesRegex(ValueError, "training-budget"):
                validate_scope_report(forged, **validation)

    def test_self_signed_minimal_scope_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, stable, _, source, commit = self._write_scope_fixture(root)
            forged = {
                "schema": "memagent.coral.scientific-scope-audit.v1",
                "status": "PASS",
                "decision": "CORAL_SCOPE_DIRECT_LEAKAGE_CLEAR_S128_ADAPTIVE_DEV_ONLY",
                "git_commit": commit,
            }
            forged["report_sha256"] = canonical_sha256(forged)
            with self.assertRaisesRegex(ValueError, "fields drifted"):
                validate_scope_report(
                    forged,
                    **self._validation_kwargs(root, manifest, stable, source, commit),
                )

    def test_real_entry_rejects_direct_content_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, stable, stable_sha, source, commit = self._write_scope_fixture(
                root, leak=True
            )
            output = root / "report.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "tools/h20/audit_coral_scientific_scope.py"),
                "--manifest", str(manifest), "--stable-resolved", str(stable),
                "--stable-resolved-sha256", stable_sha, "--work-root", str(root),
                "--repo-dir", str(source), "--expected-commit", commit,
                "--output", str(output),
            ], text=True, capture_output=True)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["set_intersections"]
                             ["actor_train_intersection_s128"]
                             ["canonical_content_pair_count"], 1)

    def test_incomplete_actual_ledger_stays_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            training = root / "training"
            training.mkdir()
            for step in range(1, 6):
                append_ledger(run / "coral_execution_ledger.jsonl", {
                    "event": "coral_role_update", "global_step": step,
                })
            from recurrent.research.coral_scope_audit import actual_budget
            result = actual_budget(run, training, expected_commit="a" * 40)
            self.assertEqual(result["status"], "PENDING_INCOMPLETE_T25_LEDGER")

    def test_complete_budget_requires_rank_optimizer_progress_and_anchors(self):
        from recurrent.research.coral_scope_audit import actual_budget
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            training = root / "training"
            commit = "a" * 40
            sync_contract = json.loads(
                (ROOT / "manifests/h20/qwen25_7b_original_t25_seed2026.json")
                .read_text()
            )["weight_sync"]
            sampled_parameters = sync_contract["parameter_names"]
            names_digest = "e" * 64
            for step in range(1, 26):
                phase = "memory_writer" if step % 2 else "terminal_answer"
                append_ledger(run / "coral_execution_ledger.jsonl", {
                    "event": "coral_role_update", "global_step": step,
                    "phase": phase, "actor_update_calls": step,
                    "active_tokens": 11, "inactive_tokens": 13,
                })
                digest = f"{step:064x}"
                master_digest = f"{step + 100:064x}"
                for rank in (0, 1):
                    append_jsonl(run / "gate_a_execution_ledger.jsonl", {
                        "record_type": "weight_sync_ack", "global_step": step,
                        "actor_version": step, "sync_kind": "post_actor_update",
                        "vllm_worker_rank": rank, "git_commit": commit,
                        "optimizer_state_entry_count": 2,
                        "optimizer_step_entry_count": 2,
                        "optimizer_step_min": step, "optimizer_step_max": step,
                        "optimizer_step_histogram": {str(step): 2},
                        "lr_scheduler_last_epoch": step,
                        "actor_sampled_tensor_digest": digest,
                        "actor_rollout_sampled_tensor_digest": digest,
                        "vllm_sampled_tensor_digest": digest,
                        "actor_master_sampled_tensor_digest": master_digest,
                        "vllm_ack_version": step,
                        "audited_loaded_parameters": sampled_parameters,
                        "sampled_parameter_dtypes": {
                            name: "torch.bfloat16" for name in sampled_parameters
                        },
                        "loaded_parameter_count": 199,
                        "model_parameter_count": 199,
                        "loaded_parameter_names_sha256": names_digest,
                        "model_parameter_names_sha256": names_digest,
                        "weight_transfer_format": "dtensor",
                    })
                append_jsonl(run / "gate_a_execution_ledger.jsonl", {
                    "record_type": "weight_sync_summary", "global_step": step,
                    "actor_version": step, "sync_kind": "post_actor_update",
                    "worker_ranks": [0, 1], "git_commit": commit,
                    "sampled_tensor_digest": digest,
                    "actor_master_sampled_tensor_digest": master_digest,
                })
            training.mkdir()
            seed_rows = []
            for step in range(1, 26):
                bases = build_trajectory_seed_records(
                    base_seed=2026, global_step=step, batch_size=8,
                    rollout_n=2, mode="independent",
                )
                for base in bases:
                    row = int(base["row"])
                    dataset_index = (step - 1) * 4 + int(base["group"])
                    uid = stable_training_group_id(
                        base_seed=2026, global_step=step,
                        dataset_index=dataset_index,
                    )
                    base.update({
                        "record_type": "trajectory_seed", "uid": uid,
                        "dataset_index": dataset_index,
                    })
                    seed_rows.extend([
                        base,
                        {"record_type": "trajectory_turn_seed", "global_step": step,
                         "row": row, "sample_index": row,
                         "group": base["group"], "replica": base["replica"],
                         "uid": uid, "trajectory_seed": base["trajectory_seed"],
                         "dataset_index": dataset_index, "mode": "independent",
                         "request_seed": derive_turn_request_seeds(
                             [int(base["trajectory_seed"])], [0], 0
                         )[0], "turn": 0, "is_final": False},
                        {"record_type": "trajectory_turn_seed", "global_step": step,
                         "row": row, "sample_index": row,
                         "group": base["group"], "replica": base["replica"],
                         "uid": uid, "trajectory_seed": base["trajectory_seed"],
                         "dataset_index": dataset_index, "mode": "independent",
                         "request_seed": derive_turn_request_seeds(
                             [int(base["trajectory_seed"])], [0], 1
                         )[0], "turn": 1, "is_final": True},
                    ])
            (training / "rollout_seed_audit.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in seed_rows)
            )
            for step in (5, 10, 15, 20, 25):
                checkpoint = training / f"global_step_{step}"
                (checkpoint / "actor").mkdir(parents=True)
                for component in ("model", "optim", "extra_state"):
                    for rank in (0, 1):
                        (checkpoint / "actor" /
                         f"{component}_world_size_2_rank_{rank}.pt").write_bytes(
                            f"{component}:{rank}".encode()
                        )
                (checkpoint / "data.pt").write_bytes(b"data")
            resolved = {"method_nonwhitelist_config_sha256": "7" * 64}
            p0 = {
                "schema": "memagent.cosi.preflight.v4", "status": "PASS",
                "decision": "COSI_T5_P0_PASS", "stage": "t5",
                "git_commit": commit, "manifest_sha256": "b" * 64,
                "original_resolved_manifest_sha256": "c" * 64,
                "original_p0_certificate_sha256": "d" * 64,
                "original_training_final_sha256": "e" * 64,
                "original_training_ledger_sha256": "f" * 64,
                "s128_resolved_manifest_sha256": "1" * 64,
                "s128_final_sha256": "2" * 64,
                "s128_ledger_sha256": "3" * 64,
                "evidence_authority_sha256": "4" * 64,
                "fresh_base_model_tokenizer_inventory_sha256": "5" * 64,
                "original_protocol_comparison_sha256": "6" * 64,
                "original_protocol_compared_leaves": ["data.train_files"],
                "resolved_config_comparison_sha256": canonical_sha256(resolved),
                "method_nonwhitelist_config_sha256": "7" * 64,
                "resolved_config_comparison": resolved,
                "gpu_pair": [2, 7],
                "gate_hashes": {
                    "paper": "8" * 64, "e0": "9" * 64, "e1": "a" * 64,
                    "baseline": "b" * 64, "scope": "c" * 64,
                },
            }
            p0["report_sha256"] = canonical_sha256(p0)
            p0_path = root / "p0_t5.json"
            p0_path.write_text(json.dumps(p0))
            p0_sha = hashlib.sha256(p0_path.read_bytes()).hexdigest()
            budget_kwargs = {
                "expected_p0_t5_file_sha256": p0_sha,
                "expected_dataset_cursor": list(range(100)),
                "expected_gpu_pair": [2, 7],
                "expected_gate_hashes": p0["gate_hashes"],
                "expected_original_resolved_sha256": "c" * 64,
                "expected_s128_resolved_sha256": "1" * 64,
                "expected_weight_sync_parameters": sampled_parameters,
                "expected_weight_transfer_format": "dtensor",
                "expected_loaded_parameter_count": 199,
            }
            result = actual_budget(
                run, training, expected_commit=commit,
                expected_manifest_sha256="b" * 64, p0_t5_path=p0_path,
                **budget_kwargs,
            )
            self.assertEqual(result["actor_optimizer_updates"], 25)
            self.assertEqual(result["optimizer_rank_acknowledgements"], 50)
            self.assertEqual(sorted(result["anchor_checkpoint_sha256"]),
                             ["10", "15", "20", "25", "5"])

            minimal_p0 = {
                "status": "PASS", "decision": "COSI_T5_P0_PASS",
                "git_commit": commit, "manifest_sha256": "b" * 64,
            }
            minimal_p0["report_sha256"] = canonical_sha256(minimal_p0)
            p0_path.write_text(json.dumps(minimal_p0))
            with self.assertRaisesRegex(ValueError, "complete T5 P0"):
                actual_budget(
                    run, training, expected_commit=commit,
                    expected_manifest_sha256="b" * 64, p0_t5_path=p0_path,
                    **{
                        **budget_kwargs,
                        "expected_p0_t5_file_sha256": hashlib.sha256(
                            p0_path.read_bytes()
                        ).hexdigest(),
                    },
                )
            p0_path.write_text(json.dumps(p0))

            with self.assertRaisesRegex(ValueError, "dataset/group identity"):
                actual_budget(
                    run, training, expected_commit=commit,
                    expected_manifest_sha256="b" * 64, p0_t5_path=p0_path,
                    **{
                        **budget_kwargs,
                        "expected_dataset_cursor": list(reversed(range(100))),
                    },
                )
            with self.assertRaisesRegex(ValueError, "external T5 P0 file SHA"):
                actual_budget(
                    run, training, expected_commit=commit,
                    expected_manifest_sha256="b" * 64, p0_t5_path=p0_path,
                    **{
                        **budget_kwargs,
                        "expected_p0_t5_file_sha256": "0" * 64,
                    },
                )
            with self.assertRaisesRegex(ValueError, "gate projection drift"):
                actual_budget(
                    run, training, expected_commit=commit,
                    expected_manifest_sha256="b" * 64, p0_t5_path=p0_path,
                    **{
                        **budget_kwargs,
                        "expected_gate_hashes": {
                            **p0["gate_hashes"], "scope": "d" * 64,
                        },
                    },
                )

            rows = [json.loads(line) for line in
                    (run / "gate_a_execution_ledger.jsonl").read_text().splitlines()]
            coverage_tampered = json.loads(json.dumps(rows))
            coverage_tampered[0].pop("audited_loaded_parameters")
            gate = run / "gate_a_execution_ledger.jsonl"
            gate.unlink()
            for row in coverage_tampered:
                for field in ("record_index", "previous_record_sha256", "record_sha256"):
                    row.pop(field, None)
                append_jsonl(gate, row)
            with self.assertRaisesRegex(ValueError, "complete Gate-A sync audit"):
                actual_budget(
                    run, training, expected_commit=commit,
                    expected_manifest_sha256="b" * 64, p0_t5_path=p0_path,
                    **budget_kwargs,
                )
            gate.unlink()
            for row in rows:
                for field in ("record_index", "previous_record_sha256", "record_sha256"):
                    row.pop(field, None)
                append_jsonl(gate, row)
            rows = [json.loads(line) for line in gate.read_text().splitlines()]
            rows[0]["optimizer_step_histogram"] = {"0": 2}
            # Rebuild a valid chain to prove the semantic validator, not the hash chain,
            # rejects a self-consistent false optimizer-progress claim.
            gate.unlink()
            for row in rows:
                for field in ("record_index", "previous_record_sha256", "record_sha256"):
                    row.pop(field, None)
                append_jsonl(gate, row)
            with self.assertRaisesRegex(ValueError, "optimizer did not advance"):
                actual_budget(
                    run, training, expected_commit=commit,
                    expected_manifest_sha256="b" * 64, p0_t5_path=p0_path,
                    **budget_kwargs,
                )

    def test_final_budget_inputs_reject_self_signed_semantic_p0_projection(self):
        from tools.h20.audit_qwen25_7b_cosi import trusted_actual_budget_inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate_root = root / "certificates"
            gate_root.mkdir()
            original = root / "original.json"
            stable = root / "stable.json"
            original.write_text("{}")
            stable.write_text("{}")
            resolved = {"method_nonwhitelist_config_sha256": "7" * 64}
            p0 = {
                "stage": "t5", "git_commit": "a" * 40,
                "manifest_sha256": "b" * 64,
                "original_resolved_manifest_sha256": "c" * 64,
                "original_p0_certificate_sha256": "d" * 64,
                "original_training_final_sha256": "e" * 64,
                "original_training_ledger_sha256": "f" * 64,
                "s128_resolved_manifest_sha256": "1" * 64,
                "s128_final_sha256": "2" * 64,
                "s128_ledger_sha256": "3" * 64,
                "fresh_base_model_tokenizer_inventory_sha256": "4" * 64,
                "original_protocol_comparison_sha256": canonical_sha256({
                    "data.train_sha256": {"method": "same", "original": "same"}
                }),
                "method_nonwhitelist_config_sha256": "7" * 64,
                "evidence_authority_sha256": canonical_sha256({"bound": True}),
                "gpu_pair": [2, 7],
                "original_protocol_compared_leaves": ["data.train_sha256"],
                "resolved_config_comparison": resolved,
                "resolved_config_comparison_sha256": canonical_sha256(resolved),
                "gate_hashes": {
                    "paper": "8" * 64, "e0": "9" * 64, "e1": "a" * 64,
                    "baseline": "b" * 64, "scope": "c" * 64,
                },
            }
            p0_path = gate_root / "p0_t5.json"
            p0_path.write_text(json.dumps(p0))
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}")
            original_authority = {
                "p0": {"evidence": {
                    "train_cursor_semantic_indices_0_to_99": list(range(100)),
                }},
                "resolved_sha256": "c" * 64, "p0_sha256": "d" * 64,
                "final_sha256": "e" * 64, "ledger_sha256": "f" * 64,
            }
            stable_authority = {
                "resolved_sha256": "1" * 64, "final_sha256": "2" * 64,
                "ledger_sha256": "3" * 64,
            }
            protocol = {
                "compared_leaves": {
                    "data.train_sha256": ("same", "same"),
                },
                "original_weight_sync_contract": {
                    "parameter_names": ["sample"],
                    "transfer_format": "dtensor",
                    "expected_loaded_parameter_count": 199,
                },
            }
            gate_reports = {
                "paper_framing_review": {"report_sha256": "8" * 64},
                "coral_e0": {"report_sha256": "9" * 64},
                "coral_e1_final_report": {"report_sha256": "a" * 64},
                "baseline_import": {"report_sha256": "b" * 64},
                "scope": {"report_sha256": "c" * 64},
            }
            environment = {
                "MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST": str(original),
                "MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST_SHA256": "c" * 64,
                "MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256": "1" * 64,
                "MEMAGENT_COSI_GPU_PAIR": "2,7",
                "MEMAGENT_COSI_T5_P0_SHA256": hashlib.sha256(
                    p0_path.read_bytes()
                ).hexdigest(),
            }
            patches = (
                mock.patch(
                    "tools.h20.audit_qwen25_7b_cosi.validate_original_training_authority",
                    return_value=original_authority,
                ),
                mock.patch(
                    "tools.h20.audit_qwen25_7b_cosi.validate_stable_s128_authority",
                    return_value=stable_authority,
                ),
                mock.patch(
                    "tools.h20.audit_qwen25_7b_cosi.validate_original_protocol",
                    return_value=protocol,
                ),
                mock.patch(
                    "tools.h20.audit_qwen25_7b_cosi.validate_local_artifacts",
                    return_value="4" * 64,
                ),
                mock.patch(
                    "tools.h20.audit_qwen25_7b_cosi.emit_method_overrides",
                    return_value=["method"],
                ),
                mock.patch(
                    "tools.h20.audit_qwen25_7b_cosi.validate_resolved_original_copy",
                    return_value=resolved,
                ),
                mock.patch(
                    "tools.h20.audit_qwen25_7b_cosi.validate_continuation_binding",
                ),
            )
            with mock.patch.dict(os.environ, environment, clear=False):
                with patches[0], patches[1], patches[2], patches[3], \
                        patches[4], patches[5], patches[6]:
                    result = trusted_actual_budget_inputs(
                        work=root, repo=root, manifest={
                            "evidence_authority": {"bound": True}
                        }, manifest_path=manifest_path, stable_path=stable,
                        expected_commit="a" * 40, gate_reports=gate_reports,
                        gate_root=gate_root,
                    )
                    self.assertEqual(result["expected_dataset_cursor"], list(range(100)))
                    p0["original_training_final_sha256"] = "0" * 64
                    p0_path.write_text(json.dumps(p0))
                    os.environ["MEMAGENT_COSI_T5_P0_SHA256"] = hashlib.sha256(
                        p0_path.read_bytes()
                    ).hexdigest()
                    with self.assertRaisesRegex(ValueError, "trusted semantic"):
                        trusted_actual_budget_inputs(
                            work=root, repo=root, manifest={
                                "evidence_authority": {"bound": True}
                            }, manifest_path=manifest_path, stable_path=stable,
                            expected_commit="a" * 40, gate_reports=gate_reports,
                            gate_root=gate_root,
                        )


if __name__ == "__main__":
    unittest.main()
