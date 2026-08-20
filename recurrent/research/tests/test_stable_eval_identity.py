import itertools
import unittest

from recurrent.research.stable_eval_identity import (
    OUTPUT_IDENTITY_FIELDS,
    build_stable_eval_identities,
    canonical_sha256,
    detach_audit_meta_for_metrics,
    detach_identity_columns_for_metrics,
    identity_rows_to_columns,
    rows_from_columns_at_indices,
    sha256_text,
    stable_key,
    validate_attempt_identity_rows,
    validate_configured_request_binding,
    validate_repeated_attempts,
    validate_request_seed_echo,
    validate_resolved_manifest,
)


SPARSE_IDS = [101, 7, 400, 9]


def resolved_manifest():
    rows = []
    for source_order_index, semantic_id in enumerate(SPARSE_IDS):
        rows.append(
            {
                "example_id": str(semantic_id),
                "semantic_dataset_index": semantic_id,
                "source_order_index": source_order_index,
                "raw_row_position": source_order_index + 10,
                "production_effective_position": source_order_index,
                "context_token_count": 5000,
                "source_question_hash": sha256_text(f"question-{semantic_id}"),
                "source_context_hash": sha256_text(f"context-{semantic_id}"),
                "ground_truth_hash": sha256_text(f"answer-{semantic_id}"),
            }
        )
    payload = {
        "schema_version": 1,
        "source_parquet_sha256": "a" * 64,
        "interface_id": "I",
        "decode": {"do_sample": False, "temperature": 0, "top_p": 1, "top_k": -1},
        "rows": rows,
    }
    return {"identity_payload": payload, "eval_manifest_hash": canonical_sha256(payload)}


def repeated_columns(ids=SPARSE_IDS):
    semantic = []
    positions = []
    for position, semantic_id in enumerate(ids):
        semantic.extend([semantic_id, semantic_id])
        positions.extend([position, position])
    return semantic, positions


def build_rows(attempt="repeat_a", ids=SPARSE_IDS, uuid_prefix=None):
    semantic, positions = repeated_columns(ids)
    counter = itertools.count()
    prefix = uuid_prefix or attempt
    return build_stable_eval_identities(
        semantic_indices=semantic,
        source_order_indices=positions,
        replicas=2,
        base_seed=2026,
        interface_id="I",
        attempt_id=attempt,
        resolved_manifest=resolved_manifest(),
        runtime_uuid_factory=lambda: f"{prefix}-uuid-{next(counter)}",
    )


class StableEvaluationIdentityTest(unittest.TestCase):
    def test_sparse_semantic_ids_are_not_source_positions(self):
        rows = build_rows()
        self.assertEqual([row["example_id"] for row in rows], ["101", "101", "7", "7", "400", "400", "9", "9"])
        self.assertEqual([row["source_order_index"] for row in rows], [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual([row["source_repeated_row"] for row in rows], list(range(8)))

    def test_four_by_two_is_a_complete_bijection(self):
        rows = build_rows()
        validate_attempt_identity_rows(rows, examples=4, replicas=2)
        self.assertEqual(len({stable_key(row) for row in rows}), 8)
        self.assertEqual(len({row["trajectory_seed"] for row in rows}), 8)
        self.assertEqual(len({row["trajectory_id"] for row in rows}), 8)
        for group in range(4):
            group_rows = rows[group * 2 : group * 2 + 2]
            self.assertEqual({row["replica_id"] for row in group_rows}, {0, 1})
            self.assertEqual(len({row["example_id"] for row in group_rows}), 1)

    def test_seed_schedule_is_independent_of_dataloader_batching(self):
        whole = build_rows()
        split_rows = []
        manifest = resolved_manifest()
        for start, stop in ((0, 1), (1, 3), (3, 4)):
            ids = SPARSE_IDS[start:stop]
            semantic = [value for semantic_id in ids for value in (semantic_id, semantic_id)]
            positions = [value for position in range(start, stop) for value in (position, position)]
            counter = itertools.count(start * 2)
            split_rows.extend(
                build_stable_eval_identities(
                    semantic_indices=semantic,
                    source_order_indices=positions,
                    replicas=2,
                    base_seed=2026,
                    interface_id="I",
                    attempt_id="repeat_a",
                    resolved_manifest=manifest,
                    runtime_uuid_factory=lambda c=counter: f"split-{next(c)}",
                )
            )
        self.assertEqual(
            [(stable_key(row), row["trajectory_seed"], row["trajectory_id"]) for row in whole],
            [(stable_key(row), row["trajectory_seed"], row["trajectory_id"]) for row in split_rows],
        )

    def test_recurrent_final_reorder_indexes_every_identity_field_together(self):
        rows = build_rows(ids=SPARSE_IDS[:2])
        columns = identity_rows_to_columns(rows)
        reordered = rows_from_columns_at_indices(columns, [2, 0, 3, 1])
        self.assertEqual([row["source_repeated_row"] for row in reordered], [2, 0, 3, 1])
        self.assertEqual(
            [row["trajectory_id"] for row in reordered],
            [rows[index]["trajectory_id"] for index in [2, 0, 3, 1]],
        )
        self.assertEqual(
            [row["runtime_sample_uuid"] for row in reordered],
            [rows[index]["runtime_sample_uuid"] for index in [2, 0, 3, 1]],
        )

    def test_two_preregistered_attempts_keep_stable_fields_and_change_runtime_uuid(self):
        first = build_rows("repeat_a")
        second = build_rows("repeat_b")
        validate_repeated_attempts([first, second])
        self.assertEqual({stable_key(row) for row in first}, {stable_key(row) for row in second})
        self.assertFalse(
            {row["runtime_sample_uuid"] for row in first}
            & {row["runtime_sample_uuid"] for row in second}
        )

    def test_manifest_hash_tampering_fails_closed(self):
        manifest = resolved_manifest()
        manifest["identity_payload"]["rows"][0]["example_id"] = "tampered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_resolved_manifest(manifest)

    def test_manifest_rejects_position_and_semantic_alias_drift(self):
        manifest = resolved_manifest()
        payload = manifest["identity_payload"]
        payload["rows"][0]["example_id"] = "different"
        manifest["eval_manifest_hash"] = canonical_sha256(payload)
        with self.assertRaisesRegex(ValueError, "string form"):
            validate_resolved_manifest(manifest)

        manifest = resolved_manifest()
        payload = manifest["identity_payload"]
        payload["rows"][1]["production_effective_position"] = 99
        manifest["eval_manifest_hash"] = canonical_sha256(payload)
        with self.assertRaisesRegex(ValueError, "production_effective_position"):
            validate_resolved_manifest(manifest)

    def test_missing_source_position_never_falls_back_to_semantic_id(self):
        semantic, positions = repeated_columns()
        positions[0] = SPARSE_IDS[0]
        with self.assertRaisesRegex(ValueError, "not present in the frozen"):
            build_stable_eval_identities(
                semantic_indices=semantic,
                source_order_indices=positions,
                replicas=2,
                base_seed=2026,
                interface_id="I",
                attempt_id="repeat_a",
                resolved_manifest=resolved_manifest(),
            )

    def test_duplicate_runtime_uuid_fails(self):
        semantic, positions = repeated_columns()
        with self.assertRaisesRegex(ValueError, "runtime_sample_uuid"):
            build_stable_eval_identities(
                semantic_indices=semantic,
                source_order_indices=positions,
                replicas=2,
                base_seed=2026,
                interface_id="I",
                attempt_id="repeat_a",
                resolved_manifest=resolved_manifest(),
                runtime_uuid_factory=lambda: "duplicate",
            )

    def test_request_seed_echo_must_be_exact_and_row_aligned(self):
        validate_request_seed_echo([11, 22], [11, 22], [0, 1])
        with self.assertRaisesRegex(ValueError, "echo mismatch"):
            validate_request_seed_echo([11, 22], [11, 23], [0, 1])
        with self.assertRaisesRegex(ValueError, "not row-aligned"):
            validate_request_seed_echo([11, 22], [11], [0])

    def test_configured_seed_is_bound_to_returned_vllm_prompt(self):
        validate_configured_request_binding(
            [11, 22], [11, 22], ["a" * 64, "b" * 64],
            ["a" * 64, "b" * 64], [0, 1]
        )
        with self.assertRaisesRegex(ValueError, "returned prompt-token binding"):
            validate_configured_request_binding(
                [11], [11], ["a" * 64], ["b" * 64], [0]
            )

    def test_identity_schema_is_separate_from_reward_metrics(self):
        self.assertNotIn("reward", OUTPUT_IDENTITY_FIELDS)
        self.assertNotIn("score", OUTPUT_IDENTITY_FIELDS)
        self.assertNotIn("response", OUTPUT_IDENTITY_FIELDS)

        columns = identity_rows_to_columns(build_rows())
        columns.update(
            active_sample_index=list(range(8)),
            request_seed=list(range(8)),
            rollout_request_seed=list(range(8)),
            rollout_worker_rank=[0, 0, 0, 0, 1, 1, 1, 1],
            is_final=[True] * 8,
        )
        retained_reward_input = {"reward_model": ["unchanged"] * 8, **columns}
        retained_tensor_input = {
            "responses": "unchanged",
            "trajectory_turn": "audit-only",
        }
        detached = detach_identity_columns_for_metrics(
            retained_reward_input, retained_tensor_input
        )
        self.assertEqual(set(detached), set(OUTPUT_IDENTITY_FIELDS))
        self.assertEqual(retained_reward_input, {"reward_model": ["unchanged"] * 8})
        self.assertEqual(retained_tensor_input, {"responses": "unchanged"})

        first_meta = {
            "strict_eval_identity": True,
            "stable_eval_identity": {"audit": "only"},
            "unrelated": "unchanged",
        }
        second_meta = {"trajectory_base_seeds": [1, 2], "validate": True}
        detach_audit_meta_for_metrics(first_meta, second_meta)
        self.assertEqual(first_meta, {"unrelated": "unchanged"})
        self.assertEqual(second_meta, {"validate": True})

    def test_terminal_validator_rejects_replica_and_repeated_row_drift(self):
        rows = build_rows()
        rows[1]["replica_id"] = 7
        with self.assertRaisesRegex(ValueError, "replica IDs"):
            validate_attempt_identity_rows(rows, examples=4, replicas=2)

        rows = build_rows()
        rows[1]["source_repeated_row"] = 99
        with self.assertRaisesRegex(ValueError, "source_repeated_row"):
            validate_attempt_identity_rows(rows, examples=4, replicas=2)


if __name__ == "__main__":
    unittest.main()
