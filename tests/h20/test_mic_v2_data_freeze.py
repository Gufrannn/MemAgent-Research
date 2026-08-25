import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recurrent.research.mic_v2 import sha256_file, sha256_json
from tools.h20.mic_v2_data_freeze import (
    EXPECTED_SPLITS,
    _build_certificate,
    _construct_resolved,
    _deduplicate,
    _partition,
    _identity,
    _read_identities,
    _verify_self_digest,
    verify,
)


def manifest():
    return {
        "partition": {
            "order_namespace": "memagent-mic-v2-label-blind-split-order-v1",
            "seed": 20260825,
            "splits": [
                {"name": name, "content_roots": size}
                for name, size in EXPECTED_SPLITS
            ],
        }
    }


def row(index, *, root=None, ground_truth=None):
    root = root or f"root-{index:05d}"
    ground_truth = ground_truth or f"gt-{index:05d}"
    return {
        "source_position": index,
        "semantic_dataset_index": index,
        "question_sha256": f"q-{index:05d}",
        "context_sha256": f"c-{index:05d}",
        "ground_truth_sha256": ground_truth,
        "content_root_id": root,
        "full_example_id": f"full-{root}-{ground_truth}",
    }


class MicV2DataFreezeTest(unittest.TestCase):
    def test_policy_prompt_is_question_authority(self):
        base = {
            "prompt": [{"role": "user", "content": "policy question"}],
            "context": "context",
            "reward_model": {"ground_truth": ["answer"]},
            "extra_info": {"index": 1},
        }
        identity = _identity(0, base, "namespace")
        self.assertEqual(identity["question_sha256"], hashlib.sha256(
            b"policy question"
        ).hexdigest())
        mismatched = copy.deepcopy(base)
        mismatched["extra_info"]["question"] = "different"
        with self.assertRaisesRegex(RuntimeError, "differs from policy prompt"):
            _identity(0, mismatched, "namespace")

    def test_synthetic_parquet_materialization_and_source_drift(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        namespace = "memagent-mic-v2-content-split-v1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.parquet"
            s128_path = root / "s128.parquet"

            def parquet_rows(prefix, count):
                return [
                    {
                        "prompt": [{"role": "user", "content": f"{prefix}-question-{index}"}],
                        "context": f"{prefix}-context-{index}",
                        "reward_model": {"ground_truth": [f"{prefix}-answer-{index}"], "style": "rule"},
                        "extra_info": {
                            "index": index,
                            "num_docs": 2,
                            "question": f"{prefix}-question-{index}",
                        },
                    }
                    for index in range(count)
                ]

            pq.write_table(pa.Table.from_pylist(parquet_rows("train", 1500)), train_path)
            pq.write_table(pa.Table.from_pylist(parquet_rows("s128", 128)), s128_path)
            s128_identities = _read_identities(s128_path, 128, namespace)
            authority = {
                "eval_manifest_hash": "c" * 64,
                "identity_payload": {
                    "source_dataset": {"parquet_sha256": sha256_file(s128_path)},
                    "rows": [
                        {
                            "raw_row_position": index,
                            "source_order_index": index,
                            "production_effective_position": index,
                            "example_id": str(item["semantic_dataset_index"]),
                            "semantic_dataset_index": item["semantic_dataset_index"],
                            "source_question_hash": item["question_sha256"],
                            "source_context_hash": item["context_sha256"],
                            "ground_truth_hash": item["ground_truth_sha256"],
                        }
                        for index, item in enumerate(s128_identities)
                    ],
                },
            }
            authority["authority_sha256"] = sha256_json(authority)
            authority_path = root / "s128_authority.json"
            authority_path.write_text(json.dumps(authority, sort_keys=True) + "\n")
            value = manifest()
            value.update({
                "identity": {"namespace": namespace},
                "sources": {
                    "training_pool": {
                        "path": str(train_path), "sha256": sha256_file(train_path), "rows": 1500,
                    },
                    "exposed_s128": {
                        "path": str(s128_path), "sha256": sha256_file(s128_path), "rows": 128,
                        "authority_path": authority_path.name,
                        "authority_file_sha256": sha256_file(authority_path),
                        "eval_manifest_hash": "c" * 64,
                    },
                },
            })
            with mock.patch(
                "tools.h20.mic_v2_data_freeze._verify_e0",
                return_value={"file_sha256": "e0-file", "canonical_sha256": "e0-canon"},
            ):
                resolved = _construct_resolved(
                    root, "a" * 40, root / "output", "data-run", value,
                    root / "manifest.json", "b" * 64,
                )
                self.assertEqual(resolved["overlap_audit"]["selected_content_roots"], 1408)
                train_path.write_bytes(train_path.read_bytes() + b"tamper")
                with self.assertRaisesRegex(RuntimeError, "training parquet SHA differs"):
                    _construct_resolved(
                        root, "a" * 40, root / "output", "data-run", value,
                        root / "manifest.json", "b" * 64,
                    )

    def test_partition_is_label_blind_disjoint_and_excludes_s128(self):
        training = [row(index) for index in range(1500)]
        s128 = [row(10000 + index) for index in range(127)]
        s128.append(row(0))
        splits, audit = _partition(manifest(), copy.deepcopy(training), s128)
        self.assertEqual(tuple(
            (name, split["content_root_count"]) for name, split in splits.items()
        ), EXPECTED_SPLITS)
        self.assertEqual(audit["train_intersection_s128_content_roots"], 1)
        self.assertEqual(audit["all_selected_intersection_s128_content_roots"], 0)
        self.assertEqual(audit["all_selected_intersection_s128_full_examples"], 0)
        self.assertFalse(any(audit["split_pair_content_intersections"].values()))
        self.assertFalse(any(audit["split_pair_full_example_intersections"].values()))
        self.assertEqual(
            [block["block_id"] for block in splits["actor_training_b1_b8"]["blocks"]],
            [f"B{index}" for index in range(1, 9)],
        )

        changed_labels = copy.deepcopy(training)
        for item in changed_labels:
            if item["content_root_id"] == "root-00000":
                continue
            item["ground_truth_sha256"] = "changed-" + item["ground_truth_sha256"]
            item["full_example_id"] = "changed-" + item["full_example_id"]
        changed, _ = _partition(manifest(), changed_labels, s128)
        for name, _ in EXPECTED_SPLITS:
            self.assertEqual(
                [item["content_root_id"] for item in splits[name]["rows"]],
                [item["content_root_id"] for item in changed[name]["rows"]],
            )

    def test_aliases_stay_together_and_conflicting_labels_fail(self):
        aliases = [row(0, root="same", ground_truth="gt"),
                   row(1, root="same", ground_truth="gt")]
        grouped = _deduplicate(aliases)
        self.assertEqual(grouped["same"]["alias_source_positions"], [0, 1])
        self.assertEqual(grouped["same"]["alias_count"], 2)
        with self.assertRaisesRegex(RuntimeError, "conflicting ground truths"):
            _deduplicate([
                row(0, root="same", ground_truth="left"),
                row(1, root="same", ground_truth="right"),
            ])

    def test_s128_internal_and_cross_source_label_conflicts_fail(self):
        training = [row(index) for index in range(1500)]
        s128_internal = [row(10000 + index) for index in range(126)] + [
            row(20000, root="s128-same", ground_truth="left"),
            row(20001, root="s128-same", ground_truth="right"),
        ]
        with self.assertRaisesRegex(RuntimeError, "conflicting ground truths"):
            _partition(manifest(), copy.deepcopy(training), s128_internal)

        s128_cross = [row(10000 + index) for index in range(127)] + [
            row(20000, root="root-00000", ground_truth="different"),
        ]
        with self.assertRaisesRegex(RuntimeError, "cross-source.*conflicting"):
            _partition(manifest(), copy.deepcopy(training), s128_cross)

    def test_self_digest_rejects_certificate_tampering(self):
        payload = {"status": "PASS", "value": 1}
        payload["certificate_sha256"] = sha256_json(payload)
        self.assertEqual(
            _verify_self_digest(payload, "certificate_sha256", "digest differs"),
            payload["certificate_sha256"],
        )
        payload["value"] = 2
        with self.assertRaisesRegex(RuntimeError, "digest differs"):
            _verify_self_digest(payload, "certificate_sha256", "digest differs")

    def test_verify_rejects_self_consistent_resign_and_source_replay_failure(self):
        commit = "a" * 40
        run_id = "data-run"
        manifest_value = {"blocked_after_success": ["e1_collection"]}
        manifest_sha = "b" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            certificates = output / "certificates"
            certificates.mkdir(parents=True)
            resolved_path = output / "resolved_split_manifest.json"
            certificate_path = certificates / "data_freeze.json"
            splits = {}
            for name, size in EXPECTED_SPLITS:
                splits[name] = {
                    "content_root_count": size,
                    "content_root_ids_sha256": f"content-{name}",
                    "full_example_ids_sha256": f"full-{name}",
                    "rows": [],
                }
            splits["actor_training_b1_b8"]["blocks"] = [
                {"block_id": f"B{index}"} for index in range(1, 9)
            ]
            pairs = {
                f"{left[0]}__{right[0]}": 0
                for index, left in enumerate(EXPECTED_SPLITS)
                for right in EXPECTED_SPLITS[index + 1:]
            }
            resolved = {
                "git_commit": commit,
                "run_id": run_id,
                "output_root": str(output),
                "overlap_audit": {
                    "all_selected_intersection_s128_content_roots": 0,
                    "all_selected_intersection_s128_full_examples": 0,
                    "split_pair_content_intersections": pairs,
                    "split_pair_full_example_intersections": pairs,
                },
                "e0_authority": {"file_sha256": "e0-file", "canonical_sha256": "e0-canon"},
                "splits": splits,
            }
            resolved["resolved_manifest_sha256"] = sha256_json(resolved)

            def write_outputs(value):
                resolved_path.write_text(json.dumps(value, sort_keys=True) + "\n")
                certificate = _build_certificate(
                    actual_commit=commit, run_id=run_id, output_root=output,
                    manifest=manifest_value, manifest_sha=manifest_sha,
                    resolved_path=resolved_path, resolved=value,
                )
                certificate_path.write_text(json.dumps(certificate, sort_keys=True) + "\n")

            def fake_git(_repo, *args):
                return commit if args == ("rev-parse", "HEAD") else ""

            write_outputs(resolved)
            patches = (
                mock.patch("tools.h20.mic_v2_data_freeze._git", side_effect=fake_git),
                mock.patch(
                    "tools.h20.mic_v2_data_freeze._load_and_validate_manifest",
                    return_value=(manifest_value, root / "manifest.json", manifest_sha),
                ),
                mock.patch(
                    "tools.h20.mic_v2_data_freeze._construct_resolved",
                    return_value=copy.deepcopy(resolved),
                ),
            )
            with patches[0], patches[1], patches[2]:
                verify(root, commit, output, run_id)

            tampered = copy.deepcopy(resolved)
            tampered["splits"]["e1_dev"]["content_root_ids_sha256"] = "attacker"
            tampered.pop("resolved_manifest_sha256")
            tampered["resolved_manifest_sha256"] = sha256_json(tampered)
            write_outputs(tampered)
            with patches[0], patches[1], patches[2]:
                with self.assertRaisesRegex(RuntimeError, "does not replay"):
                    verify(root, commit, output, run_id)

            with mock.patch("tools.h20.mic_v2_data_freeze._git", side_effect=fake_git), \
                    mock.patch(
                        "tools.h20.mic_v2_data_freeze._load_and_validate_manifest",
                        return_value=(manifest_value, root / "manifest.json", manifest_sha),
                    ), mock.patch(
                        "tools.h20.mic_v2_data_freeze._construct_resolved",
                        side_effect=RuntimeError("MIC_V2_NO_GO: training parquet SHA differs"),
                    ):
                with self.assertRaisesRegex(RuntimeError, "training parquet SHA differs"):
                    verify(root, commit, output, run_id)


if __name__ == "__main__":
    unittest.main()
