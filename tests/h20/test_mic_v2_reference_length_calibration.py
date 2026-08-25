import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recurrent.research.mic_v2 import (
    MATERIALIZATION_PARSER_VERSION,
    canonical_json,
    materialized_memory_receipt,
    sampled_policy_mask_receipt,
    sha256_file,
    sha256_json,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds
from tools.h20.mic_v2_reference_length_calibration import (
    FORBIDDEN_GPU_INPUT_FIELDS,
    _label_blind_inputs,
    _read_ledger,
    _seed_authority,
    _source_firewall,
    _verify_model,
    _validate_ledger,
    finalize,
)
from tools.h20.run_qwen25_7b_mic_v2_reference_length_calibration import (
    _completion_receipt,
    _source_rows,
    _stable_seed,
)


REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "scripts/h20/run_qwen25_7b_mic_v2_reference_length_calibration.sh"


def frozen_row(index):
    return {
        "source_position": index,
        "semantic_dataset_index": 1000 + index,
        "question_sha256": f"{index + 1:064x}",
        "context_sha256": f"{index + 101:064x}",
        "content_root_id": f"{index + 201:064x}",
        "ground_truth_sha256": f"{index + 301:064x}",
        "full_example_id": f"{index + 401:064x}",
    }


def ledger_record(index, root, source_position, replica=0):
    trajectory_seed = _stable_seed(20260825, root, replica)
    terminators = [151645, 151643]
    chunk_hashes = [sha256_json([100]), sha256_json([101])]
    memory_hashes = []
    slots = []
    for slot in range(8):
        active = slot < 2
        value = {
            "slot_index": slot,
            "role": "writer",
            "active": active,
            "sampled_policy_tokens": 3 if active else 0,
            "sampled_token_ids": [10, 11, 151645] if active else [],
            "completion_sha256": f"{slot + 1:064x}" if active else None,
            "termination": "sampled_eos" if active else "exogenous_termination",
            "sampled_eos_counted": active,
            "sampled_terminal_token_id": 151645 if active else None,
            "request_seed": (
                derive_turn_request_seeds([trajectory_seed], [0], slot)[0]
                if active else None
            ),
        }
        if active:
            value.update(sampled_policy_mask_receipt(
                token_ids=value["sampled_token_ids"], termination="sampled_eos",
                termination_token_ids=terminators, token_width=1024,
            ))
            parsed, afterstate = materialized_memory_receipt(
                token_ids=value["sampled_token_ids"],
                termination_token_ids=terminators,
                content_root_id=root, trajectory_seed=trajectory_seed,
                turn_index=slot, arrived_chunk_token_sha256=chunk_hashes[:slot + 1],
                prior_memory_token_sha256=memory_hashes,
            )
            memory_hashes.append(afterstate["parsed_memory_sha256"])
            value.update(afterstate)
        else:
            value.update({
                "sampled_mask_width": 1024,
                "sampled_mask_true_count": 0,
                "sampled_mask_sha256": sha256_json([False] * 1024),
                "parsed_memory_token_ids": None,
                "parsed_memory_sha256": None,
                "parser_version": None,
                "afterstate_sha256": None,
            })
        slots.append(value)
    answer = {
        "slot_index": 8,
        "role": "answer",
        "active": True,
        "sampled_policy_tokens": 5,
        "sampled_token_ids": [20, 21, 22, 23, 151643],
        "completion_sha256": "f" * 64,
        "termination": "sampled_eos",
        "sampled_eos_counted": True,
        "sampled_terminal_token_id": 151643,
        "request_seed": derive_turn_request_seeds([trajectory_seed], [0], 8)[0],
        "parsed_memory_token_ids": None,
        "parsed_memory_sha256": None,
        "parser_version": None,
        "afterstate_sha256": None,
    }
    answer.update(sampled_policy_mask_receipt(
        token_ids=answer["sampled_token_ids"], termination="sampled_eos",
        termination_token_ids=terminators, token_width=1024,
    ))
    slots.append(answer)
    value = {
        "content_root_id": root,
        "schema": "memagent.mic.v2.reference-length-trajectory",
        "source_position": source_position,
        "replica": replica,
        "trajectory_seed": trajectory_seed,
        "active_writer_slots": 2,
        "git_commit": "a" * 40,
        "run_id": "calibration-test",
        "p0_sha256": "b" * 64,
        "slots": slots,
        "sampled_policy_tokens": 11,
        "record_index": index,
        "previous_record_sha256": "0" * 64,
    }
    for slot in slots:
        if slot["active"]:
            slot["completion_sha256"] = sha256_json(slot["sampled_token_ids"])
    return value


def tokenization_authority(roots):
    return {
        "tokenizer_vocab_size": 200000,
        "termination_token_ids": [151645, 151643],
        "receipts": [{
            "content_root_id": root,
            "active_writer_slots": 2,
            "chunk_token_ids_sha256": [sha256_json([100]), sha256_json([101])],
        } for root in roots],
    }


class MicV2ReferenceLengthCalibrationTest(unittest.TestCase):
    def _entry_fixture(self, temporary):
        root = Path(temporary)
        checkout = root / "checkout"
        subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", str(REPO), str(checkout)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(
            ["git", "-C", str(checkout), "switch", "--detach", head], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        work = root / "work"
        python = work / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/usr/bin/env bash\nexit 0\n")
        python.chmod(0o755)
        environment = dict(os.environ)
        environment.update({
            "MEMAGENT_MIC_V2_WORK_ROOT": str(work),
            "MEMAGENT_MIC_V2_REPO_DIR": str(checkout),
            "MEMAGENT_MIC_V2_EXPECTED_COMMIT": head,
            "MEMAGENT_MIC_V2_CALIBRATION_RUN_ID": "entry-test",
            "MEMAGENT_MIC_V2_GPU_PAIR": "4,5",
        })
        if shutil.which("flock") is None:
            fake_bin = root / "portable-bin"
            fake_bin.mkdir()
            fake_flock = fake_bin / "flock"
            fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n")
            fake_flock.chmod(0o755)
            environment["PATH"] = str(fake_bin) + os.pathsep + environment["PATH"]
        return work, checkout, head, environment

    def test_fresh_base_file_authority_matches_existing_certified_manifest(self):
        calibration = json.loads((
            REPO / "manifests/h20/qwen25_7b_mic_v2_reference_length_calibration.json"
        ).read_text())
        certified = json.loads((
            REPO / "manifests/h20/qwen25_7b_commit_retain_capture_seed2026.json"
        ).read_text())
        self.assertEqual(calibration["model"]["files"], certified["model"]["files"])
        self.assertEqual(
            calibration["model"]["path"],
            certified["model"]["path"].replace(
                "${MEMAGENT_COMMIT_RETAIN_WORK_ROOT}", "/data/cw/memagent_work"
            ),
        )
        self.assertEqual(calibration["recurrent"]["max_writer_slots"], 8)
        self.assertEqual(calibration["sampling"]["replicas"], 4)

    def test_projection_removes_every_outcome_field(self):
        rows = [frozen_row(index) for index in range(64)]
        manifest = {
            "data_freeze_authority": {
                "split": "reference_length_calibration", "content_roots": 64,
            },
            "source": {
                "path": "/frozen/train.parquet",
                "sha256": "a" * 64,
                "gpu_visible_columns": ["prompt", "context", "extra_info"],
            },
        }
        resolved = {"splits": {"reference_length_calibration": {"rows": rows}}}
        projected = _label_blind_inputs(manifest, resolved)
        encoded = canonical_json(projected).lower()
        for field in FORBIDDEN_GPU_INPUT_FIELDS:
            self.assertNotIn(f'"{field}"', encoded)
        self.assertEqual(len(projected["rows"]), 64)
        self.assertEqual(
            set(projected["rows"][0]),
            {"source_position", "semantic_dataset_index", "question_sha256",
             "context_sha256", "content_root_id"},
        )

    def test_replica_seeds_are_stable_and_independent(self):
        root = "a" * 64
        seeds = [_stable_seed(20260825, root, replica) for replica in range(4)]
        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual(seeds, [
            _stable_seed(20260825, root, replica) for replica in range(4)
        ])
        self.assertNotEqual(seeds[0], _stable_seed(20260825, "b" * 64, 0))

    def test_sampled_eos_is_counted_and_truncation_cannot_fake_eos(self):
        eos = _completion_receipt(
            [10, 2], "stop", termination_token_ids=[2, 3], maximum=4,
        )
        self.assertEqual(eos["sampled_policy_tokens"], 2)
        self.assertTrue(eos["sampled_eos_counted"])
        alternate = _completion_receipt(
            [10, 3], "stop", termination_token_ids=[2, 3], maximum=4,
        )
        self.assertEqual(alternate["sampled_terminal_token_id"], 3)
        with self.assertRaisesRegex(RuntimeError, "sampled terminal"):
            _completion_receipt(
                [2, 10, 3], "stop", termination_token_ids=[2, 3], maximum=4,
            )
        truncated = _completion_receipt(
            [10, 11, 12, 13], "length", termination_token_ids=[2, 3], maximum=4,
        )
        self.assertEqual(truncated["termination"], "forced_truncation")
        with self.assertRaisesRegex(RuntimeError, "forced truncation"):
            _completion_receipt(
                [10, 11, 12, 2], "length",
                termination_token_ids=[2, 3], maximum=4,
            )

    def test_gpu_source_artifact_rejects_any_extra_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "source.jsonl"
            frozen = {
                "source_position": 0,
                "semantic_dataset_index": 1,
                "question_sha256": hashlib.sha256(b"q").hexdigest(),
                "context_sha256": hashlib.sha256(b"c").hexdigest(),
                "content_root_id": "a" * 64,
            }
            record = {**frozen, "question": "q", "context": "c", "tainted": "x"}
            artifact.write_text(canonical_json(record) + "\n")
            manifest = {"source": {"gpu_input_fields": [
                *frozen.keys(), "question", "context",
            ]}}
            with self.assertRaisesRegex(RuntimeError, "schema differs"):
                _source_rows(
                    manifest,
                    {"source_artifact_path": str(artifact), "rows": [frozen]},
                )

    def test_fresh_base_recursive_inventory_rejects_extra_loading_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}")
            manifest = {"model": {"path": str(root), "files": [{
                "path": "config.json", "size": 2,
                "sha256": sha256_file(root / "config.json"),
            }]}}
            self.assertEqual(len(_verify_model(manifest)), 1)
            (root / "adapter_config.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "recursive file inventory"):
                _verify_model(manifest)

    def test_hash_chain_and_fixed_slot_statistic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            roots = ["a" * 64, "b" * 64]
            first = ledger_record(0, roots[0], 10)
            first["record_sha256"] = sha256_json(first)
            second = ledger_record(1, roots[1], 11)
            second["previous_record_sha256"] = first["record_sha256"]
            second["record_sha256"] = sha256_json(second)
            path.write_text(canonical_json(first) + "\n" + canonical_json(second) + "\n")
            manifest = {
                "sampling": {
                    "replicas": 1, "base_seed": 20260825,
                    "writer": {"max_tokens": 1024},
                    "answer": {"max_tokens": 1024},
                },
                "backend": {"termination_token_ids": [151645, 151643]},
            }
            inputs = {"rows": [
                {"content_root_id": roots[0], "source_position": 10},
                {"content_root_id": roots[1], "source_position": 11},
            ]}
            summary = _validate_ledger(
                manifest, inputs, path,
                expected_commit="a" * 40,
                run_id="calibration-test",
                p0_sha256="b" * 64,
                tokenization_authority=tokenization_authority(roots),
            )
            self.assertEqual(summary["trajectory_count"], 2)
            self.assertEqual(summary["scheduled_slot_count"], 18)
            self.assertEqual(summary["active_writer_slot_count"], 4)
            self.assertEqual(summary["total_policy_tokens"], 22)
            self.assertEqual(len(_read_ledger(path)), 2)

            tampered = copy.deepcopy(second)
            tampered["slots"][3]["sampled_policy_tokens"] = 1
            tampered.pop("record_sha256")
            tampered["record_sha256"] = sha256_json(tampered)
            path.write_text(canonical_json(first) + "\n" + canonical_json(tampered) + "\n")
            with self.assertRaisesRegex(RuntimeError, "slot token count"):
                _validate_ledger(
                    manifest, inputs, path,
                    expected_commit="a" * 40,
                    run_id="calibration-test",
                    p0_sha256="b" * 64,
                    tokenization_authority=tokenization_authority(roots),
                )

    def test_validator_rejects_role_swap_and_early_terminator(self):
        manifest = {
            "sampling": {
                "replicas": 1, "base_seed": 20260825,
                "writer": {"max_tokens": 1024}, "answer": {"max_tokens": 1024},
            },
            "backend": {"termination_token_ids": [151645, 151643]},
        }
        root = "a" * 64
        inputs = {"rows": [{"content_root_id": root, "source_position": 10}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            row = ledger_record(0, root, 10)
            row["slots"][0]["role"] = "answer"
            row["record_sha256"] = sha256_json(row)
            path.write_text(canonical_json(row) + "\n")
            with self.assertRaisesRegex(RuntimeError, "roles differ"):
                _validate_ledger(manifest, inputs, path)

            row = ledger_record(0, root, 10)
            row["slots"][0]["sampled_token_ids"] = [151645, 99, 151645]
            row["slots"][0]["completion_sha256"] = sha256_json(
                row["slots"][0]["sampled_token_ids"]
            )
            row["record_sha256"] = sha256_json(row)
            path.write_text(canonical_json(row) + "\n")
            with self.assertRaisesRegex(ValueError, "sampled terminator"):
                _validate_ledger(manifest, inputs, path)

    def test_shared_mask_and_memory_parser_match_training_semantics(self):
        raw = [10, 11, 151643]
        padded = raw + [151643] * 5
        terminators = {151645, 151643}
        # This is exactly `(isin.cumsum - isin).eq(0)` from the frozen VERL
        # training helper, expressed without importing the H20-only torch stack.
        training_mask = [
            not any(token in terminators for token in padded[:index])
            for index in range(len(padded))
        ]
        mask_source = (REPO / "verl/utils/torch_functional.py").read_text()
        worker_source = (REPO / "verl/workers/fsdp_workers.py").read_text()
        memory_source = (REPO / "recurrent/utils.py").read_text()
        self.assertIn("eos_mask.cumsum(dim=1) - eos_mask", mask_source)
        self.assertIn("self.generation_config.eos_token_id", worker_source)
        self.assertIn("attention_mask = ~(tensor == tokenizer.pad_token_id)", memory_source)
        self.assertIn("attention_mask &= ~(tensor == tokenizer.eos_token_id)", memory_source)
        receipt = sampled_policy_mask_receipt(
            token_ids=raw, termination="sampled_eos",
            termination_token_ids=[151645, 151643], token_width=8,
        )
        self.assertEqual(receipt["sampled_mask_true_count"], sum(training_mask))
        self.assertEqual(receipt["sampled_mask_sha256"], sha256_json(training_mask))
        parsed, evidence = materialized_memory_receipt(
            token_ids=raw, termination_token_ids=[151645, 151643],
            content_root_id="a" * 64, trajectory_seed=7, turn_index=0,
            arrived_chunk_token_sha256=["b" * 64],
            prior_memory_token_sha256=[],
        )
        training_materialized = [token for token in padded if token not in terminators]
        self.assertEqual(parsed, training_materialized)
        self.assertEqual(evidence["parser_version"], MATERIALIZATION_PARSER_VERSION)

    def test_validator_rejects_materialized_afterstate_tampering(self):
        manifest = {
            "sampling": {
                "replicas": 1, "base_seed": 20260825,
                "writer": {"max_tokens": 1024}, "answer": {"max_tokens": 1024},
            },
            "backend": {"termination_token_ids": [151645, 151643]},
        }
        root = "a" * 64
        inputs = {"rows": [{"content_root_id": root, "source_position": 10}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.jsonl"
            row = ledger_record(0, root, 10)
            row["slots"][0]["afterstate_sha256"] = "f" * 64
            row["record_sha256"] = sha256_json(row)
            path.write_text(canonical_json(row) + "\n")
            with self.assertRaisesRegex(RuntimeError, "materialization/afterstate"):
                _validate_ledger(
                    manifest, inputs, path,
                    tokenization_authority=tokenization_authority([root]),
                )

    def test_gpu_runner_source_firewall(self):
        evidence = _source_firewall(REPO)
        self.assertEqual(len(evidence), 5)
        self.assertTrue(all(len(value) == 64 for value in evidence.values()))
        entry_source = ENTRY.read_text()
        self.assertIn("--mode produce", entry_source)
        self.assertIn("--mode replay", entry_source)
        self.assertNotIn(
            'if [[ ! -e "$OUTPUT_ROOT/certificates/gpu_replay.json" ]]',
            entry_source,
        )

    def test_finalize_reconstructs_fixed_slot_mean_and_binds_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            (output / "certificates").mkdir()
            (output / "authorities").mkdir()
            (output / "trajectories").mkdir()
            roots = ["a" * 64, "b" * 64]
            inputs = {
                "schema": "memagent.mic.v2.reference-length-label-blind-inputs",
                "rows": [
                    {"content_root_id": roots[0], "source_position": 10},
                    {"content_root_id": roots[1], "source_position": 11},
                ],
            }
            inputs["inputs_sha256"] = sha256_json(inputs)
            p0 = {
                "git_commit": "a" * 40,
                "run_id": "calibration-test",
                "manifest_sha256": "c" * 64,
                "gpu_pair": [4, 5],
                "expected_trajectories": 2,
                "expected_scheduled_slots": 18,
                "tokenization_authority": tokenization_authority(roots),
                "materialization_authority": {
                    "parser_version": MATERIALIZATION_PARSER_VERSION,
                },
                "seed_authority": {
                    "trajectory_count": 2,
                    "active_request_count": 6,
                    "all_trajectory_seeds_unique": True,
                    "all_active_request_seeds_unique": True,
                    "trajectory_request_namespaces_disjoint": True,
                },
            }
            p0["p0_sha256"] = sha256_json(p0)
            (output / "authorities/label_blind_inputs.json").write_text(
                json.dumps(inputs, sort_keys=True) + "\n"
            )
            (output / "authorities/label_blind_source.jsonl").write_text("{}\n")
            (output / "certificates/p0.json").write_text(
                json.dumps(p0, sort_keys=True) + "\n"
            )
            ledger = output / "trajectories/length_receipts.jsonl"
            first = ledger_record(0, roots[0], 10)
            first["p0_sha256"] = p0["p0_sha256"]
            first["record_sha256"] = sha256_json(first)
            second = ledger_record(1, roots[1], 11)
            second["p0_sha256"] = p0["p0_sha256"]
            second["previous_record_sha256"] = first["record_sha256"]
            second["record_sha256"] = sha256_json(second)
            ledger.write_text(canonical_json(first) + "\n" + canonical_json(second) + "\n")
            execution = {
                "schema": "memagent.mic.v2.reference-length-execution",
                "status": "PASS",
                "git_commit": "a" * 40,
                "run_id": "calibration-test",
                "p0_sha256": p0["p0_sha256"],
                "gpu_pair": [4, 5],
                "physical_gpu_identity": [
                    {"index": 4, "uuid": "GPU-uuid-4", "name": "NVIDIA H20"},
                    {"index": 5, "uuid": "GPU-uuid-5", "name": "NVIDIA H20"},
                ],
                "vllm_version": "0.8.2",
                "strict_vllm": True,
                "tensor_parallel_size": 2,
                "prefix_cache_enabled": False,
                "termination_token_ids": [151645, 151643],
                "trainer_attached": False,
                "actor_updates": 0,
                "new_generate_calls_this_session": 6,
                "represented_generate_calls": 6,
                "trajectory_count": 2,
                "ledger_file_sha256": sha256_file(ledger),
            }
            execution["execution_sha256"] = sha256_json(execution)
            (output / "certificates/execution.json").write_text(
                json.dumps(execution, sort_keys=True) + "\n"
            )
            replay = {
                "schema": "memagent.mic.v2.reference-length-gpu-replay",
                "status": "PASS",
                "decision": "MIC_V2_REFERENCE_LENGTH_GPU_REPLAY_PASS",
                "git_commit": "a" * 40,
                "run_id": "calibration-test",
                "p0_sha256": p0["p0_sha256"],
                "execution_sha256": execution["execution_sha256"],
                "gpu_pair": [4, 5],
                "physical_gpu_identity": execution["physical_gpu_identity"],
                "vllm_version": "0.8.2",
                "termination_token_ids": [151645, 151643],
                "trajectory_count": 2,
                "regenerated_generate_calls": 6,
                "exact_token_match_count": 6,
                "ledger_file_sha256": sha256_file(ledger),
            }
            replay["gpu_replay_sha256"] = sha256_json(replay)
            (output / "certificates/gpu_replay.json").write_text(
                json.dumps(replay, sort_keys=True) + "\n"
            )
            manifest = {
                "model": {"required_vllm_version": "0.8.2"},
                "sampling": {
                    "replicas": 1, "base_seed": 20260825,
                    "writer": {"max_tokens": 1024},
                    "answer": {"max_tokens": 1024},
                },
                "backend": {"termination_token_ids": [151645, 151643]},
            }
            with mock.patch(
                "tools.h20.mic_v2_reference_length_calibration._runtime"
            ), mock.patch(
                "tools.h20.mic_v2_reference_length_calibration._manifest",
                return_value=(manifest, Path("manifest.json"), "c" * 64),
            ), mock.patch(
                "tools.h20.mic_v2_reference_length_calibration.preflight",
                return_value=p0,
            ):
                report = finalize(
                    REPO, "a" * 40, output, "calibration-test",
                )
                bad_replay = dict(replay)
                bad_replay.pop("gpu_replay_sha256")
                bad_replay["exact_token_match_count"] = 5
                bad_replay["gpu_replay_sha256"] = sha256_json(bad_replay)
                (output / "certificates/gpu_replay.json").write_text(
                    json.dumps(bad_replay, sort_keys=True) + "\n"
                )
                with self.assertRaisesRegex(RuntimeError, "GPU replay receipt"):
                    finalize(REPO, "a" * 40, output, "calibration-test")
                (output / "certificates/gpu_replay.json").write_text(
                    json.dumps(replay, sort_keys=True) + "\n"
                )
                tampered = dict(execution)
                tampered.pop("execution_sha256")
                tampered["reward"] = 1.0
                tampered["execution_sha256"] = sha256_json(tampered)
                (output / "certificates/execution.json").write_text(
                    json.dumps(tampered, sort_keys=True) + "\n"
                )
                with self.assertRaisesRegex(RuntimeError, "execution receipt"):
                    finalize(REPO, "a" * 40, output, "calibration-test")
            self.assertAlmostEqual(report["lbar_ref"], 22 / 18)
            self.assertEqual(report["scheduled_slot_count"], 18)
            self.assertEqual(report["decision"], "MIC_V2_REFERENCE_LENGTH_CALIBRATION_PASS")

    def test_seed_schedule_is_globally_unique_and_frozen(self):
        roots = ["a" * 64, "b" * 64]
        manifest = {
            "sampling": {"replicas": 4, "base_seed": 20260825},
            "recurrent": {"max_writer_slots": 8},
        }
        inputs = {"rows": [{"content_root_id": root} for root in roots]}
        tokenization = tokenization_authority(roots)
        authority = _seed_authority(manifest, inputs, tokenization)
        self.assertEqual(authority["trajectory_count"], 8)
        self.assertEqual(authority["active_request_count"], 24)
        self.assertTrue(authority["all_trajectory_seeds_unique"])
        self.assertTrue(authority["all_active_request_seeds_unique"])

    def test_real_shell_rejects_missing_environment_and_bad_pair(self):
        missing = subprocess.run(
            ["bash", str(ENTRY)], env={"PATH": os.environ.get("PATH", "")},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.assertEqual(missing.returncode, 40, missing.stdout)
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment.update({
                "MEMAGENT_MIC_V2_WORK_ROOT": temporary,
                "MEMAGENT_MIC_V2_REPO_DIR": str(REPO),
                "MEMAGENT_MIC_V2_EXPECTED_COMMIT": "0" * 40,
                "MEMAGENT_MIC_V2_CALIBRATION_RUN_ID": "entry-test",
                "MEMAGENT_MIC_V2_GPU_PAIR": "5,4",
            })
            result = subprocess.run(
                ["bash", str(ENTRY)], env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.assertEqual(result.returncode, 44, result.stdout)
            self.assertIn("canonical ascending", result.stdout)

    def test_real_shell_rejects_wrong_commit_dirty_and_existing_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            work, checkout, head, environment = self._entry_fixture(temporary)
            wrong = dict(environment)
            wrong["MEMAGENT_MIC_V2_EXPECTED_COMMIT"] = "0" * 40
            result = subprocess.run(
                ["bash", str(ENTRY)], env=wrong,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.assertEqual(result.returncode, 46, result.stdout)
            (checkout / "dirty-evidence.txt").write_text("dirty\n")
            result = subprocess.run(
                ["bash", str(ENTRY)], env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.assertEqual(result.returncode, 47, result.stdout)
            (checkout / "dirty-evidence.txt").unlink()
            output = work / "logs/mic_v2_reference_length/entry-test"
            output.mkdir(parents=True)
            result = subprocess.run(
                ["bash", str(ENTRY)], env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.assertEqual(result.returncode, 49, result.stdout)

    def test_real_shell_rejects_run_and_gpu_lock_conflicts(self):
        if shutil.which("flock") is None:
            self.skipTest("host lacks flock; H20 entry uses util-linux flock")
        for lock_name, expected_code in (
            ("memagent_mic_v2_entry-test.lock", 48),
            ("memagent_h20_gpu_4.lock", 51),
        ):
            with self.subTest(lock_name=lock_name), tempfile.TemporaryDirectory() as temporary:
                work, _checkout, _head, environment = self._entry_fixture(temporary)
                lock_root = work / "locks"
                lock_root.mkdir(parents=True)
                lock_path = lock_root / lock_name
                with lock_path.open("w") as stream:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    result = subprocess.run(
                        ["bash", str(ENTRY)], env=environment,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    )
                self.assertEqual(result.returncode, expected_code, result.stdout)

    def test_real_shell_rejects_occupied_gpu_without_killing(self):
        with tempfile.TemporaryDirectory() as temporary:
            work, _checkout, _head, environment = self._entry_fixture(temporary)
            fake_bin = Path(temporary) / "bin"
            fake_bin.mkdir()
            nvidia_smi = fake_bin / "nvidia-smi"
            nvidia_smi.write_text("#!/usr/bin/env bash\nprintf '424242\\n'\n")
            nvidia_smi.chmod(0o755)
            environment["PATH"] = str(fake_bin) + os.pathsep + environment["PATH"]
            result = subprocess.run(
                ["bash", str(ENTRY)], env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.assertEqual(result.returncode, 53, result.stdout)
            self.assertIn("no process was killed", result.stdout)

    def test_real_shell_never_skips_replay_for_preexisting_fake_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            work, _checkout, _head, environment = self._entry_fixture(temporary)
            environment["MEMAGENT_MIC_V2_CALIBRATION_RESUME"] = "1"
            python = work / ".venv/bin/python"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "case \"$*\" in\n"
                "  *\"--mode replay\"*) echo REPLAY_WAS_EXECUTED; exit 77;;\n"
                "  *) exit 0;;\n"
                "esac\n"
            )
            python.chmod(0o755)
            output = work / "logs/mic_v2_reference_length/entry-test"
            (output / "certificates").mkdir(parents=True)
            (output / "authorities").mkdir()
            (output / "trajectories").mkdir()
            (output / "certificates/execution.json").write_text("{}\n")
            (output / "certificates/gpu_replay.json").write_text("{}\n")
            fake_bin = Path(temporary) / "empty-gpu-bin"
            fake_bin.mkdir()
            nvidia_smi = fake_bin / "nvidia-smi"
            nvidia_smi.write_text("#!/usr/bin/env bash\nexit 0\n")
            nvidia_smi.chmod(0o755)
            environment["PATH"] = str(fake_bin) + os.pathsep + environment["PATH"]
            result = subprocess.run(
                ["bash", str(ENTRY)], env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.assertEqual(result.returncode, 77, result.stdout)
            self.assertIn("REPLAY_WAS_EXECUTED", result.stdout)
            self.assertFalse((output / "certificates/reference_length.json").exists())


if __name__ == "__main__":
    unittest.main()
