import copy
import tempfile
import unittest
from pathlib import Path

from recurrent.research.gate_a_execution import append_jsonl
from tools.h20.audit_coral_t5_health import validate_role_mechanism, validate_t5_evaluation


def updates():
    return [
        {
            "global_step": step,
            "phase": "memory_writer" if step % 2 else "terminal_answer",
            "active_tokens": 10,
            "inactive_tokens": 9,
            "active_grad_norm": 0.25,
            "actor_vllm_sampled_tensor_digest": f"{step:x}" * 64,
        }
        for step in range(1, 6)
    ]


def gate_rows(source):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gate.jsonl"
        for row in source:
            append_jsonl(path, {
                "record_type": "weight_sync_summary",
                "sync_kind": "post_actor_update",
                "global_step": row["global_step"],
                "sampled_tensor_digest": row["actor_vllm_sampled_tensor_digest"],
            })
        return [__import__("json").loads(line) for line in path.read_text().splitlines()]


class CoralT5HealthTests(unittest.TestCase):
    def test_t5_evaluation_protocol_and_finite_metrics(self):
        metric = {
            "denominator": 128, "normalized_exact_match": 0.2,
            "token_f1": 0.4, "format_success": 0.8,
            "historical_sub_exact_match_diagnostic": 0.5,
        }
        baseline = {
            "eval_manifest_hash": "a" * 64,
            "stable_inventory_sha256": "b" * 64,
            "aggregates": {"Original5": copy.deepcopy(metric)},
        }
        evaluation = {
            "step": 5, "checkpoint_inventory_sha256": "c" * 64,
            "eval_manifest_hash": "a" * 64,
            "stable_inventory_sha256": "b" * 64,
            "metrics": copy.deepcopy(metric),
        }
        validate_t5_evaluation(evaluation, baseline, "c" * 64)
        for field, replacement in (
            ("eval_manifest_hash", "d" * 64),
            ("stable_inventory_sha256", "e" * 64),
            ("checkpoint_inventory_sha256", "f" * 64),
        ):
            value = copy.deepcopy(evaluation); value[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_t5_evaluation(value, baseline, "c" * 64)
        value = copy.deepcopy(evaluation); value["metrics"]["token_f1"] = float("inf")
        with self.assertRaises(ValueError):
            validate_t5_evaluation(value, baseline, "c" * 64)

    def test_active_roles_and_weight_sync_pass(self):
        value = updates()
        norms, sync = validate_role_mechanism(value, gate_rows(value))
        self.assertEqual(set(norms), {"memory_writer", "terminal_answer"})
        self.assertEqual(len(sync), 5)

    def test_method_inactive_schedule_seed_and_sync_tampering_fail(self):
        original = updates()
        cases = []
        value = copy.deepcopy(original); value[0]["active_tokens"] = 0; cases.append((value, gate_rows(original)))
        value = copy.deepcopy(original); value[1]["active_grad_norm"] = 0.0; cases.append((value, gate_rows(original)))
        value = copy.deepcopy(original); value[2]["phase"] = "terminal_answer"; cases.append((value, gate_rows(original)))
        value = copy.deepcopy(original); value[3]["actor_vllm_sampled_tensor_digest"] = "f" * 64; cases.append((value, gate_rows(original)))
        value = copy.deepcopy(original); rows = gate_rows(value); rows[0]["sampled_tensor_digest"] = "e" * 64; cases.append((value, rows))
        for value, rows in cases:
            with self.subTest(), self.assertRaisesRegex(ValueError, "CORAL_T5_NO_GO"):
                validate_role_mechanism(value, rows)


if __name__ == "__main__":
    unittest.main()
