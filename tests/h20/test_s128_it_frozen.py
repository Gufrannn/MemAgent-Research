from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recurrent.research.gate_a_execution import append_jsonl, checkpoint_inventory
from recurrent.research.s128_hotpot_metrics import (
    extract_terminal_answer,
    paired_descriptive_summary,
    score_terminal_output,
    summarize_fixed_s128,
)
from recurrent.research.stable_eval_identity import (
    canonical_sha256,
    validate_actor_only_checkpoint_acknowledgements,
)
from tools.h20.preflight_qwen25_7b_s128_it import (
    _checkpoint_contract,
    _current_certificate,
    _stable_canary_contract,
    generation_protocol_projection,
)
from tools.h20.preflight_qwen25_7b_stable_i4x2 import build_identity_payload
from tools.h20 import audit_qwen25_7b_s128_it


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "manifests/h20/qwen25_7b_s128_it_seed2026.json"
COMMANDS = REPO / "manifests/h20/qwen25_7b_s128_it_commands.json"
COMMON_SCRIPT = REPO / "scripts/h20/s128_it_common.sh"
EXPECTED_COMMIT = "a" * 40


def _write_stable_canary(root: Path, *, commit: str = "b" * 40):
    payload = {
        "schema_version": 1,
        "rows": [{
            "example_id": "7", "semantic_dataset_index": 7,
            "source_order_index": 0, "raw_row_position": 0,
            "production_effective_position": 0, "context_token_count": 10,
            "source_question_hash": "1" * 64,
            "source_context_hash": "2" * 64,
            "ground_truth_hash": "3" * 64,
        }],
    }
    eval_hash = canonical_sha256(payload)
    resolved_path = root / "stable_resolved.json"
    execution_binding = {"contract": "stable-test"}
    execution_sha = canonical_sha256(execution_binding)
    resolved_path.write_text(json.dumps({
        "identity_payload": payload, "eval_manifest_hash": eval_hash,
        "execution_binding": execution_binding,
    }))
    ledger_path = root / "stable_ledger.jsonl"
    common = {
        "git_commit": commit, "eval_manifest_hash": eval_hash,
        "execution_binding_sha256": execution_sha,
    }
    for record_type, attempt_id in (
        ("s0_preflight", None),
        ("run_start", "repeat_a"), ("run_finish", "repeat_a"),
        ("run_start", "repeat_b"), ("run_finish", "repeat_b"),
    ):
        append_jsonl(ledger_path, {
            **common, "record_type": record_type, "attempt_id": attempt_id,
        })
    prefix_sha = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    report_path = root / "stable_report.json"
    report_path.write_text(json.dumps({
        "status": "PASS", "decision": "I_RECURRENT_IDENTITY_CANARY_PASS",
        "failures": [],
        "evidence": {
            "git_commit": commit, "eval_manifest_hash": eval_hash,
            "execution_ledger_records": 5,
            "execution_ledger_sha256": prefix_sha,
        },
    }))
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    append_jsonl(ledger_path, {
        **common, "record_type": "audit_result", "attempt_id": None,
        "status": "PASS",
        "decision": "I_RECURRENT_IDENTITY_CANARY_PASS", "git_commit": commit,
        "eval_manifest_hash": eval_hash, "artifact": str(report_path),
        "artifact_sha256": report_sha,
    })
    return ({
        "final_report": str(report_path),
        "resolved_manifest": str(resolved_path),
        "execution_ledger": str(ledger_path),
        "required_status": "PASS",
        "required_decision": "I_RECURRENT_IDENTITY_CANARY_PASS",
        "required_git_commit": commit,
        "required_eval_manifest_hash": eval_hash,
    }, eval_hash, report_sha)


def _write_t25_attestation(
    root: Path,
    *,
    report_path: Path,
    report_payload: dict,
    experiment: str,
    commit: str,
    step25_inventory: list[dict],
) -> Path:
    ledger_path = root / "t25_ledger.jsonl"
    run_id = "1" * 32
    common = {
        "experiment_name": experiment,
        "git_commit": commit,
        "run_id": run_id,
    }
    training_record_types = (
        ["p0_preflight", "runtime_config", "resume_load"]
        + ["rollout_start"] * 22
        + ["execution_signal"] * 22
        + ["weight_sync_ack"] * 46
        + ["weight_sync_summary"] * 23
    )
    for record_type in training_record_types:
        append_jsonl(ledger_path, {**common, "record_type": record_type})
    training_records = [
        json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()
    ]
    anchors = [
        {
            "path": str(root / f"global_step_{step}"),
            "global_step": step,
            "inventory": step25_inventory,
            "inventory_sha256": canonical_sha256(step25_inventory),
        }
        for step in (5, 10, 15, 20, 25)
    ]
    report_payload["checkpoint_anchors"] = anchors
    report_payload["checkpoint_anchors_sha256"] = canonical_sha256(anchors)
    report_payload["execution_ledger"] = {
        "path": str(ledger_path),
        "training_prefix_record_count": len(training_records),
        "training_prefix_sha256": canonical_sha256(training_records),
        "training_prefix_tail_sha256": training_records[-1]["record_sha256"],
    }
    report_path.write_text(json.dumps(report_payload))
    append_jsonl(
        ledger_path,
        {
            **common,
            "record_type": "checkpoint_inventory",
            "global_step": 25,
            "inventory": step25_inventory,
            "inventory_sha256": canonical_sha256(step25_inventory),
            "checkpoint_anchors": anchors,
            "checkpoint_anchors_sha256": canonical_sha256(anchors),
        },
    )
    append_jsonl(
        ledger_path,
        {
            **common,
            "record_type": "audit_result",
            "status": "PASS",
            "decision": "ORIGINAL_T25_PASS",
            "report": str(report_path),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
    )
    return ledger_path


class S128MetricTests(unittest.TestCase):
    def test_corrected_project_extraction_route_is_frozen(self):
        self.assertEqual(extract_terminal_answer(r"work \boxed{New {York}}"), ("New {York}", "boxed", True))
        # A balanced empty box is an empty boxed prediction.  It must not fall
        # back to an earlier explicit answer and accidentally score it.
        self.assertEqual(
            extract_terminal_answer("answer is gold\n" + r"\boxed{}"),
            ("", "boxed", False),
        )
        empty_box = score_terminal_output("answer is gold\n" + r"\boxed{}", "gold")
        self.assertEqual(empty_box["exact_match"], 0.0)
        self.assertEqual(empty_box["token_f1"], 0.0)
        self.assertEqual(extract_terminal_answer("Final answer: Paris\nextra"), ("Paris", "explicit", False))
        self.assertEqual(extract_terminal_answer("reasoning\nTokyo"), ("Tokyo", "last_line", False))
        self.assertEqual(extract_terminal_answer(""), ("", "last_line", False))

    def test_alias_max_and_normalized_em_f1(self):
        scored = score_terminal_output(r"\boxed{The New, York}", ["London", "new york"])
        self.assertEqual(scored["exact_match"], 1.0)
        self.assertEqual(scored["token_f1"], 1.0)
        self.assertEqual(scored["format_success"], 1.0)

    def test_missing_and_empty_are_zero_not_substring_true(self):
        scored = score_terminal_output("", "answer")
        self.assertEqual(scored["exact_match"], 0.0)
        self.assertEqual(scored["token_f1"], 0.0)
        self.assertEqual(scored["sub_exact_match"], 0.0)
        self.assertEqual(scored["format_success"], 0.0)

    def test_f1_matches_corrected_dense_component_without_yes_no_special_case(self):
        scored = score_terminal_output(r"\boxed{yes maybe}", "yes")
        self.assertAlmostEqual(scored["token_f1"], 2 / 3)
        self.assertEqual(scored["exact_match"], 0.0)

    def test_denominator_is_exactly_128(self):
        row = {"exact_match": 1.0, "token_f1": 0.5, "format_success": 1.0, "sub_exact_match": 1.0}
        summary = summarize_fixed_s128([row] * 128)
        self.assertEqual(summary["denominator"], 128)
        self.assertEqual(summary["normalized_exact_match"], 1.0)
        with self.assertRaisesRegex(ValueError, "exactly 128"):
            summarize_fixed_s128([row] * 127)

    def test_paired_summary_uses_same_128_keys_and_is_descriptive(self):
        i_rows = [
            {"stable_key": f"k{i}", "exact_match": 0.0, "token_f1": 0.0, "format_success": 1.0, "sub_exact_match": 0.0}
            for i in range(128)
        ]
        t_rows = [dict(row, token_f1=1.0) for row in i_rows]
        paired = paired_descriptive_summary(i_rows, t_rows)
        self.assertEqual(paired["token_f1"]["mean_difference"], 1.0)
        self.assertFalse(paired["causal"])
        self.assertFalse(paired["population_inference"])
        with self.assertRaisesRegex(ValueError, "same 128"):
            paired_descriptive_summary(i_rows, [dict(row, stable_key="bad") for row in t_rows])


class ActorOnlyCheckpointTests(unittest.TestCase):
    def _frozen(self, root: Path):
        step = root / "experiment" / "global_step_25"
        actor = step / "actor"
        actor.mkdir(parents=True)
        shards = []
        for rank, payload in ((0, b"rank-zero"), (1, b"rank-one")):
            path = actor / f"model_world_size_2_rank_{rank}.pt"
            path.write_bytes(payload)
            shards.append({
                "path": f"actor/{path.name}", "size": len(payload),
                "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            })
        return step, shards

    def test_acknowledgements_bind_each_rank_and_forbid_training_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            step, shards = self._frozen(Path(temporary))
            acknowledgements = []
            for rank, shard in enumerate(shards):
                acknowledgements.append({
                    "rank": rank, "world_size": 2, "model_loaded": True,
                    "model_shard_path": str(step / shard["path"]),
                    "model_shard_size": shard["size"],
                    "model_shard_sha256": shard["sha256"],
                    "optimizer_loaded": False, "lr_scheduler_loaded": False,
                    "rng_loaded": False, "dataloader_loaded": False,
                    "optimizer_state_entry_count_before": 0,
                    "optimizer_state_entry_count_after": 0,
                    "lr_scheduler_last_epoch_before": 0,
                    "lr_scheduler_last_epoch_after": 0,
                })
            checked = validate_actor_only_checkpoint_acknowledgements(
                acknowledgements, shards, global_step_folder=step, world_size=2
            )
            self.assertEqual([row["rank"] for row in checked], [0, 1])
            bad = [dict(row) for row in acknowledgements]
            bad[1]["optimizer_loaded"] = True
            with self.assertRaisesRegex(ValueError, "optimizer_loaded"):
                validate_actor_only_checkpoint_acknowledgements(
                    bad, shards, global_step_folder=step, world_size=2
                )
            bad = [dict(row) for row in acknowledgements]
            bad[1]["model_shard_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "digest"):
                validate_actor_only_checkpoint_acknowledgements(
                    bad, shards, global_step_folder=step, world_size=2
                )

    def test_checkpoint_contract_requires_exact_two_model_shards_and_report_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = "experiment"
            step, _ = self._frozen(root)
            stable_spec, stable_eval_hash, stable_report_sha = _write_stable_canary(root)
            (step / "actor/optim_world_size_2_rank_0.pt").write_bytes(b"optimizer")
            (step / "data.pt").write_bytes(b"data")
            inventory = checkpoint_inventory(step)
            report = root / "report.json"
            stable_ledger_sha = hashlib.sha256(
                Path(stable_spec["execution_ledger"]).read_bytes()
            ).hexdigest()
            report_payload = {
                "status": "PASS", "decision": "ORIGINAL_T25_PASS",
                "experiment_name": experiment,
                "git_commit": EXPECTED_COMMIT,
                "not_original_paper_7b_reproduction": True,
                "stable_identity_prerequisite": {
                    "status": "PASS", "decision": "I_RECURRENT_IDENTITY_CANARY_PASS",
                    "commit": stable_spec["required_git_commit"],
                    "report_sha256": stable_report_sha,
                    "eval_manifest_hash": stable_eval_hash,
                    "execution_ledger_sha256": stable_ledger_sha,
                },
                "weight_sync": {"version_digests": {"25": "d" * 64}},
                "step25_checkpoint": {
                    "path": str(step), "global_step": 25,
                    "inventory": inventory,
                    "inventory_sha256": canonical_sha256(inventory),
                },
            }
            t25_ledger = _write_t25_attestation(
                root,
                report_path=report,
                report_payload=report_payload,
                experiment=experiment,
                commit=EXPECTED_COMMIT,
                step25_inventory=inventory,
            )
            manifest = {"stable_identity_canary": stable_spec, "training_anchor": {
                "experiment_name": experiment, "checkpoint": str(step),
                "final_report": str(report), "global_step": 25,
                "execution_ledger": str(t25_ledger),
                "required_actor_shards": [
                    "actor/model_world_size_2_rank_0.pt",
                    "actor/model_world_size_2_rank_1.pt",
                ],
            }}
            frozen = _checkpoint_contract(manifest, expected_git_commit=EXPECTED_COMMIT)
            self.assertEqual(frozen["fsdp_world_size"], 2)
            self.assertEqual(frozen["training_effective_actor_vllm_digest"], "d" * 64)
            self.assertEqual(
                frozen["training_execution_ledger_sha256"],
                hashlib.sha256(t25_ledger.read_bytes()).hexdigest(),
            )
            original_t25_ledger = t25_ledger.read_text()
            tampered_lines = original_t25_ledger.splitlines()
            tampered_record = json.loads(tampered_lines[0])
            tampered_record["git_commit"] = "0" * 40
            tampered_lines[0] = json.dumps(tampered_record)
            t25_ledger.write_text("\n".join(tampered_lines) + "\n")
            with self.assertRaisesRegex(ValueError, "hash chain failed"):
                _checkpoint_contract(manifest, expected_git_commit=EXPECTED_COMMIT)
            t25_ledger.write_text(original_t25_ledger)
            with self.assertRaisesRegex(ValueError, "same frozen commit"):
                _checkpoint_contract(manifest, expected_git_commit="b" * 40)
            payload = json.loads(report.read_text())
            payload["stable_identity_prerequisite"]["report_sha256"] = "0" * 64
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "authenticate the current stable-I"):
                _checkpoint_contract(manifest, expected_git_commit=EXPECTED_COMMIT)
            payload["stable_identity_prerequisite"]["report_sha256"] = stable_report_sha
            payload["stable_identity_prerequisite"]["execution_ledger_sha256"] = "0" * 64
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "authenticate the current stable-I"):
                _checkpoint_contract(manifest, expected_git_commit=EXPECTED_COMMIT)
            payload["stable_identity_prerequisite"][
                "execution_ledger_sha256"
            ] = stable_ledger_sha
            # Rebuilding the ledger is the only legitimate way to change the
            # hash-bound report.  A direct report rewrite must be rejected.
            payload["tamper_note"] = "changed after T25 audit"
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "audit-result ledger tail"):
                _checkpoint_contract(manifest, expected_git_commit=EXPECTED_COMMIT)
            payload.pop("tamper_note")
            t25_ledger.unlink()
            _write_t25_attestation(
                root,
                report_path=report,
                report_payload=payload,
                experiment=experiment,
                commit=EXPECTED_COMMIT,
                step25_inventory=inventory,
            )
            extra = step / "actor/model_world_size_4_rank_2.pt"
            extra.write_bytes(b"bad")
            inventory = checkpoint_inventory(step)
            payload = json.loads(report.read_text())
            payload["step25_checkpoint"]["inventory"] = inventory
            payload["step25_checkpoint"]["inventory_sha256"] = canonical_sha256(inventory)
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "audit-result ledger tail"):
                _checkpoint_contract(manifest, expected_git_commit=EXPECTED_COMMIT)
            t25_ledger.unlink()
            _write_t25_attestation(
                root,
                report_path=report,
                report_payload=payload,
                experiment=experiment,
                commit=EXPECTED_COMMIT,
                step25_inventory=inventory,
            )
            with self.assertRaisesRegex(ValueError, "world-size 2"):
                _checkpoint_contract(manifest, expected_git_commit=EXPECTED_COMMIT)


class FrozenContractTests(unittest.TestCase):
    def test_audit_module_imports_without_gpu_runtime(self):
        self.assertTrue(callable(audit_qwen25_7b_s128_it.run_audit))

    def test_p0_prefix_is_fully_authenticated_before_gpu_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stable_spec, _, _ = _write_stable_canary(root)
            resolved_path = root / "p0_resolved.json"
            resolved_path.write_bytes(Path(stable_spec["resolved_manifest"]).read_bytes())
            resolved = json.loads(resolved_path.read_text())
            p0_path = root / "p0.json"
            ledger_path = root / "s128_p0_ledger.jsonl"
            run_id = "2" * 32
            runtime_binding = {"runtime": "frozen"}
            p0 = {
                "status": "PASS",
                "decision": "S128_IT_P0_PASS",
                "evidence": {
                    "git_commit": EXPECTED_COMMIT,
                    "expected_git_commit": EXPECTED_COMMIT,
                    "run_id": run_id,
                    "eval_manifest_hash": resolved["eval_manifest_hash"],
                    "execution_binding_sha256": canonical_sha256(
                        resolved["execution_binding"]
                    ),
                    "runtime_binding": runtime_binding,
                    "runtime_binding_sha256": canonical_sha256(runtime_binding),
                    "resolved_manifest_sha256": hashlib.sha256(
                        resolved_path.read_bytes()
                    ).hexdigest(),
                },
            }
            p0_path.write_text(json.dumps(p0))
            append_jsonl(
                ledger_path,
                {
                    "record_type": "s0_preflight",
                    "experiment_name": "qwen25_7b_s128_it_p0_seed2026_20260821",
                    "git_commit": EXPECTED_COMMIT,
                    "run_id": run_id,
                    "eval_manifest_hash": resolved["eval_manifest_hash"],
                    "execution_binding_sha256": canonical_sha256(
                        resolved["execution_binding"]
                    ),
                    "runtime_binding_sha256": canonical_sha256(runtime_binding),
                    "interface_id": None,
                    "status": "PASS",
                    "artifact": str(p0_path),
                    "artifact_sha256": hashlib.sha256(p0_path.read_bytes()).hexdigest(),
                    "row_count": 128,
                },
            )
            manifest = {
                "paths": {
                    "p0_certificate": str(p0_path),
                    "resolved_manifest": str(resolved_path),
                    "execution_ledger": str(ledger_path),
                }
            }
            with patch.dict(
                "os.environ",
                {"MEMAGENT_S128_IT_EXPECTED_COMMIT": EXPECTED_COMMIT},
            ):
                self.assertEqual(_current_certificate(manifest)[0]["status"], "PASS")
                tampered_p0 = json.loads(p0_path.read_text())
                tampered_p0["decision"] = "WRONG"
                p0_path.write_text(json.dumps(tampered_p0))
                with self.assertRaisesRegex(ValueError, "exact S128_IT_P0_PASS"):
                    _current_certificate(manifest)
                p0_path.write_text(json.dumps(p0))
                original_ledger = ledger_path.read_text()
                tampered_lines = original_ledger.splitlines()
                tampered_record = json.loads(tampered_lines[0])
                tampered_record["status"] = "FAIL"
                tampered_lines[0] = json.dumps(tampered_record)
                ledger_path.write_text("\n".join(tampered_lines) + "\n")
                with self.assertRaisesRegex(ValueError, "hash chain"):
                    _current_certificate(manifest)

    def test_manifest_is_existing_full_s128_n1_greedy_without_dense_claim(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["data"]["expected_raw_rows"], 128)
        self.assertEqual(manifest["data"]["expected_effective_rows"], 128)
        self.assertIn("question-only", manifest["data"]["sampling_semantics"])
        self.assertEqual(manifest["evaluation"]["replicas"], 1)
        self.assertFalse(manifest["evaluation"]["do_sample"])
        self.assertEqual(manifest["evaluation"]["primary_metrics"], ["normalized_exact_match", "token_f1"])
        self.assertNotIn("dense_reward", manifest["evaluation"]["primary_metrics"])
        self.assertTrue(manifest["evaluation"]["training_dense_reward_excluded_from_evaluation_claims"])
        self.assertTrue(manifest["scope"]["raw_context_r_not_rerun"])
        self.assertTrue(manifest["scope"]["not_original_paper_7b_reproduction"])
        canary = manifest["stable_identity_canary"]
        self.assertIn("stable_i4x2_frozen_20260821r2", canary["final_report"])
        self.assertIn("stable_i4x2_frozen_20260821r2", canary["resolved_manifest"])
        self.assertIn("stable_i4x2_frozen_20260821r2", canary["execution_ledger"])

    def test_command_protocol_is_shared_and_interface_diff_is_provenance_only(self):
        commands = json.loads(COMMANDS.read_text())
        common = commands["common_trainer_overrides"]
        self.assertIn("actor_rollout_ref.rollout.n=1", common)
        self.assertIn("actor_rollout_ref.rollout.val_kwargs.n=1", common)
        self.assertIn("actor_rollout_ref.rollout.val_kwargs.do_sample=False", common)
        allowed_prefixes = (
            "trainer.resume_mode=", "trainer.resume_from_path=",
            "+trainer.eval_identity.weight_source=",
            "+trainer.eval_identity.expected_global_step=",
        )
        for interface in ("I", "T25"):
            self.assertTrue(all(item.startswith(allowed_prefixes) for item in commands["interface_overrides"][interface]))
        self.assertFalse(commands["causal_claim_authorized_by_this_manifest"])
        self.assertIn("--validate-p0-prefix", COMMON_SCRIPT.read_text())

    def test_interface_neutral_identity_payload_matches_stable_canary_contract(self):
        stable = json.loads(
            (REPO / "manifests/h20/qwen25_7b_stable_i4x2_seed2026.json").read_text()
        )
        current = json.loads(MANIFEST.read_text())
        row = {
            "example_id": "7", "semantic_dataset_index": 7,
            "source_order_index": 0, "raw_row_position": 0,
            "production_effective_position": 0, "context_token_count": 10,
            "source_question_hash": "1" * 64, "source_context_hash": "2" * 64,
            "ground_truth_hash": "3" * 64,
        }
        self.assertEqual(
            build_identity_payload(current, rows=[row]),
            build_identity_payload(stable, rows=[row]),
        )

    def test_generation_protocol_projection_ignores_only_checkpoint_provenance(self):
        base = {
            "recurrent": {"enable": "memory"},
            "data": {
                "val_files": ["s128"], "shuffle": False,
                "filter_overlong_prompts": True, "filter_overlong_prompts_workers": 1,
                "dataloader_num_workers": 0, "include_source_order_index": True,
                "truncation": "center", "context_key": "context", "val_max_samples": 128,
                "max_prompt_length": 8192, "max_response_length": 1024,
            },
            "actor_rollout_ref": {
                "model": {"path": "base", "use_remove_padding": True},
                "rollout": {
                    "name": "vllm", "mode": "sync", "n": 1,
                    "tensor_model_parallel_size": 1, "max_num_batched_tokens": 16384,
                    "max_num_seqs": 16,
                    "val_kwargs": {"n": 1, "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1},
                },
            },
            "reward_model": {"reward_manager": "naive"},
            "custom_reward_function": {"path": "dense.py", "name": "compute_score"},
            "trainer": {"resume_mode": "disable", "resume_from_path": None},
        }
        checkpoint = json.loads(json.dumps(base))
        checkpoint["trainer"] = {"resume_mode": "actor_only_eval", "resume_from_path": "global_step_25"}
        self.assertEqual(generation_protocol_projection(base), generation_protocol_projection(checkpoint))

    def test_stable_canary_report_is_mandatory_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, eval_hash, _ = _write_stable_canary(
                root, commit="bd8b804c2cbf333f0f0650b729fd03a143d445b2"
            )
            manifest = {"stable_identity_canary": spec}
            self.assertEqual(
                _stable_canary_contract(
                    manifest, expected_eval_manifest_hash=eval_hash
                )["status"],
                "PASS",
            )
            with self.assertRaisesRegex(ValueError, "same manifest hash"):
                _stable_canary_contract(manifest, expected_eval_manifest_hash="d" * 64)
            resolved_path = Path(spec["resolved_manifest"])
            original_resolved = resolved_path.read_text()
            resolved = json.loads(original_resolved)
            resolved["eval_manifest_hash"] = "0" * 64
            resolved_path.write_text(json.dumps(resolved))
            with self.assertRaisesRegex(ValueError, "resolved evaluation manifest hash mismatch"):
                _stable_canary_contract(manifest)
            resolved_path.write_text(original_resolved)
            ledger_path = Path(spec["execution_ledger"])
            ledger_lines = ledger_path.read_text().splitlines()
            ledger = json.loads(ledger_lines[0])
            ledger["decision"] = "TAMPERED"
            ledger_lines[0] = json.dumps(ledger)
            ledger_path.write_text("\n".join(ledger_lines) + "\n")
            with self.assertRaisesRegex(ValueError, "hash chain failed"):
                _stable_canary_contract(manifest)

    def test_strict_terminal_and_turn_types_reject_json_coercions(self):
        identity = {
            "semantic_dataset_index": 7, "source_order_index": 0,
            "raw_row_position": 0, "production_effective_position": 0,
            "context_token_count": 10, "replica_id": 0,
            "source_repeated_row": 0, "trajectory_seed": 123,
        }
        self.assertEqual(
            audit_qwen25_7b_s128_it._strict_integer_failures(
                {**identity, "step": 25},
                (*audit_qwen25_7b_s128_it.STRICT_IDENTITY_INTEGER_FIELDS, "step"),
                label="terminal",
            ),
            [],
        )
        for bad in ("0", True, 0.0):
            failures = audit_qwen25_7b_s128_it._strict_integer_failures(
                {**identity, "source_order_index": bad, "step": 25},
                (*audit_qwen25_7b_s128_it.STRICT_IDENTITY_INTEGER_FIELDS, "step"),
                label="terminal",
            )
            self.assertTrue(any("source_order_index" in failure for failure in failures))
        valid_turn = {
            **identity, "record_type": "trajectory_turn", "active_sample_index": 0,
            "request_seed": 1, "configured_request_seed": 1,
            "rollout_request_seed": 1, "rollout_worker_rank": 0,
            "trajectory_turn": 0, "is_final": False,
        }
        self.assertEqual(
            audit_qwen25_7b_s128_it._strict_turn_type_failures(valid_turn, label="turn"),
            [],
        )
        for field, bad in (("is_final", 0), ("record_type", "bad"), ("request_seed", "1")):
            failures = audit_qwen25_7b_s128_it._strict_turn_type_failures(
                {**valid_turn, field: bad}, label="turn"
            )
            self.assertTrue(any(field in failure for failure in failures))

    def test_duplicate_turn_and_training_digest_tampering_fail_closed(self):
        seen = set()
        key = ("e" * 64, "7", 0)
        self.assertEqual(
            audit_qwen25_7b_s128_it._register_unique_turn(
                seen, key, 0, label="turn"
            ),
            [],
        )
        duplicate = audit_qwen25_7b_s128_it._register_unique_turn(
            seen, key, 0, label="turn"
        )
        self.assertTrue(any("duplicate trajectory turn" in item for item in duplicate))
        checkpoint = {"training_effective_actor_vllm_digest": "d" * 64}
        snapshot = {
            "actor_rollout_sampled_tensor_digest": "d" * 64,
            "vllm_sampled_tensor_digest": "d" * 64,
        }
        self.assertEqual(
            audit_qwen25_7b_s128_it._audit_t25_training_digest(snapshot, checkpoint),
            [],
        )
        snapshot["vllm_sampled_tensor_digest"] = "0" * 64
        self.assertTrue(
            audit_qwen25_7b_s128_it._audit_t25_training_digest(snapshot, checkpoint)
        )

    def test_ledger_contract_authenticates_phase_specific_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p0_path = root / "p0.json"
            p0_path.write_text('{"status":"PASS"}\n')
            interface_roots = {"I": root / "i", "T25": root / "t"}
            artifact_maps = {}
            for interface, step in (("I", 0), ("T25", 25)):
                interface_root = interface_roots[interface]
                paths = (
                    interface_root / f"terminal/{step}.jsonl",
                    interface_root / "trajectory_turns.jsonl",
                    interface_root / "execution_summary.json",
                    interface_root / "run.log",
                )
                for path in paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"{interface}:{path.name}\n")
                artifact_maps[interface] = {
                    str(path.relative_to(interface_root)): {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size": path.stat().st_size,
                    }
                    for path in paths
                }
            manifest = {
                "repository": str(REPO),
                "ledger_schema": "s128_it_execution_ledger.schema.json",
                "paths": {
                    "p0_certificate": str(p0_path),
                    "final_report": str(root / "final.json"),
                    "I": str(interface_roots["I"]),
                    "T25": str(interface_roots["T25"]),
                },
            }
            resolved = {
                "eval_manifest_hash": "e" * 64,
                "execution_binding": {"contract": "test"},
            }
            p0 = {"evidence": {
                "git_commit": EXPECTED_COMMIT,
                "run_id": "f" * 32,
                "runtime_binding_sha256": "a" * 64,
            }}
            common = {
                "git_commit": EXPECTED_COMMIT,
                "run_id": "f" * 32,
                "recorded_at": "2026-08-21T00:00:00+00:00",
                "eval_manifest_hash": "e" * 64,
                "execution_binding_sha256": canonical_sha256(
                    resolved["execution_binding"]
                ),
                "runtime_binding_sha256": "a" * 64,
                "status": "PASS",
            }
            ledger = root / "ledger.jsonl"
            append_jsonl(ledger, {
                **common, "record_type": "s0_preflight", "interface_id": None,
                "experiment_name": "qwen25_7b_s128_it_p0_seed2026_20260821",
                "artifact": str(p0_path),
                "artifact_sha256": hashlib.sha256(p0_path.read_bytes()).hexdigest(),
                "row_count": 128,
            })
            experiments = {
                "I": "qwen25_7b_s128_i_base_seed2026_20260821",
                "T25": "qwen25_7b_s128_t25_corrected_original_style_seed2026_20260821",
            }
            for interface in ("I", "T25"):
                append_jsonl(ledger, {
                    **common, "record_type": "interface_start",
                    "interface_id": interface,
                    "experiment_name": experiments[interface], "artifacts": {},
                })
                append_jsonl(ledger, {
                    **common, "record_type": "interface_finish",
                    "interface_id": interface,
                    "experiment_name": experiments[interface],
                    "artifacts": artifact_maps[interface],
                })
            records = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual(
                audit_qwen25_7b_s128_it._audit_ledger(
                    records, manifest=manifest, resolved=resolved, p0=p0
                ),
                [],
            )
            tampered = [dict(row) for row in records]
            tampered[1]["status"] = "FAIL"
            failures = audit_qwen25_7b_s128_it._audit_ledger(
                tampered, manifest=manifest, resolved=resolved, p0=p0
            )
            self.assertTrue(any("status is not PASS" in item for item in failures))
            tampered = [dict(row) for row in records]
            tampered[3]["artifacts"] = {"unexpected": {}}
            failures = audit_qwen25_7b_s128_it._audit_ledger(
                tampered, manifest=manifest, resolved=resolved, p0=p0
            )
            self.assertTrue(any("start record 3 artifacts" in item for item in failures))

    def test_shared_account_writer_environment_is_cleared(self):
        runner = (REPO / "scripts/h20/run_qwen25_7b_s128_it.sh").read_text()
        self.assertIn("compgen -v GATE_A_", runner)
        self.assertIn("unset ORIGINAL_T25_EXPECTED_RUNTIME_CONFIG_SHA256", runner)
        self.assertIn("unset ORIGINAL_T25_TRAINER_OVERRIDE_ARGV_SHA256", runner)
        self.assertIn("GATE_A_WEIGHT_DIGEST_PARAMETERS", runner)

    def test_actor_only_loader_never_calls_full_checkpoint_loader(self):
        manager = (REPO / "verl/utils/checkpoint/fsdp_checkpoint_manager.py").read_text()
        body = manager.split("def load_model_checkpoint_only", 1)[1].split("def save_checkpoint", 1)[0]
        self.assertIn("self.model.load_state_dict", body)
        self.assertNotIn("self.optimizer.load_state_dict", body)
        self.assertNotIn("load_rng_state", body)
        self.assertNotIn("lr_scheduler.load_state_dict", body)
        trainer = (REPO / "verl/trainer/ppo/ray_trainer.py").read_text()
        self.assertIn('resume_mode == "actor_only_eval"', trainer)
        self.assertIn("validate_actor_only_checkpoint_acknowledgements", trainer)
        self.assertIn("TURN_LEDGER_NON_TENSOR_FIELDS", trainer)
        self.assertIn("trajectory_turn_record_from_columns", trainer)


if __name__ == "__main__":
    unittest.main()
