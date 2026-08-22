import tempfile
import unittest
from pathlib import Path

from recurrent.research.gate_a_execution import append_jsonl
from tools.h20.audit_coral_t5_health import (
    select_t5_updates, validate_exact_gate_boundary, validate_role_mechanism,
)


def updates():
    return [
        {
            "global_step": step,
            "phase": "memory_writer" if step % 2 else "terminal_answer",
            "active_tokens": 10,
            "inactive_tokens": 9,
            "active_grad_norm": 0.25,
            "active_pg_loss": 0.125,
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
    def test_active_roles_and_weight_sync_pass(self):
        value = updates()
        norms, sync = validate_role_mechanism(value, gate_rows(value))
        self.assertEqual(set(norms), {"memory_writer", "terminal_answer"})
        self.assertEqual(len(sync), 5)

    def test_recovery_accepts_only_exact_step5_boundary(self):
        original = updates()
        self.assertEqual(len(select_t5_updates(original, exact_boundary=True)), 5)
        with self.assertRaisesRegex(ValueError, "exact step5"):
            select_t5_updates(original + [{"global_step": 6}], exact_boundary=True)
        with self.assertRaisesRegex(ValueError, "exact step5"):
            select_t5_updates(original[:-1], exact_boundary=True)
        self.assertEqual(len(select_t5_updates(original + [{"global_step": 6}],
                                               exact_boundary=False)), 5)
        validate_exact_gate_boundary(gate_rows(original))
        asymmetric = gate_rows(original + [{
            **original[-1], "global_step": 6,
            "actor_vllm_sampled_tensor_digest": "6" * 64,
        }])
        with self.assertRaisesRegex(ValueError, "Gate-A ledger advanced"):
            validate_exact_gate_boundary(asymmetric)

    def test_method_inactive_schedule_seed_and_sync_tampering_fail(self):
        original = updates()
        cases = []
        import copy
        value = copy.deepcopy(original); value[0]["active_tokens"] = 0; cases.append((value, gate_rows(original)))
        value = copy.deepcopy(original); value[1]["active_grad_norm"] = 0.0; cases.append((value, gate_rows(original)))
        value = copy.deepcopy(original); value[1]["active_pg_loss"] = float("nan"); cases.append((value, gate_rows(original)))
        value = copy.deepcopy(original); value[2]["phase"] = "terminal_answer"; cases.append((value, gate_rows(original)))
        value = copy.deepcopy(original); value[3]["actor_vllm_sampled_tensor_digest"] = "f" * 64; cases.append((value, gate_rows(original)))
        value = copy.deepcopy(original); rows = gate_rows(value); rows[0]["sampled_tensor_digest"] = "e" * 64; cases.append((value, rows))
        for value, rows in cases:
            with self.subTest(), self.assertRaisesRegex(ValueError, "CORAL_T5_NO_GO"):
                validate_role_mechanism(value, rows)


if __name__ == "__main__":
    unittest.main()
