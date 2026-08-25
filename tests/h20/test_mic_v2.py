import unittest
import copy
from pathlib import Path

import numpy as np

from recurrent.research.mic_v2 import (
    CONTRACT_SHA256,
    SCHEMA,
    full_branch_block_schedule,
    full_branch_matched_slot_count,
    group_centered_broadcast,
    oracle_boundary_decomposition,
    sampled_token_masks,
    seal_credit_bundle,
    sha256_json,
    sibling_reconstruction,
    sparse_branch_accounting,
    stable_fold_assignments,
    validate_boundary_pair,
    validate_boundary_state,
    verify_sealed_credit_bundle,
)
from tools.h20.mic_v2_pipeline import E0_IDS, _verify_e0_certificate_payload


def boundary(phase="pre_write", *, turn=1, memories=None):
    memories = [] if memories is None else memories
    return {
        "schema": SCHEMA,
        "phase": phase,
        "content_root_id": "root",
        "stable_example_id": "example",
        "trajectory_id": "trajectory",
        "turn_index": turn,
        "question": "question",
        "arrived_chunks": [f"chunk-{index}" for index in range(turn)],
        "materialized_memory_history": memories,
        "current_memory": memories[-1] if memories else "",
        "public_metadata": {
            "chunk_schedule_id": "unit",
            "arrived_context_token_count": turn,
            "prior_active_turn_count": turn - 1,
            "exogenous_termination": False,
            "policy_termination": False,
            "forced_truncation": False,
        },
    }


class MicV2CoreTest(unittest.TestCase):
    def test_boundary_pair_extends_nested_history(self):
        pre = boundary("pre_write", turn=2, memories=["m1"])
        post = boundary("post_write", turn=2, memories=["m1", "m2"])
        checked_pre, checked_post = validate_boundary_pair(pre, post)
        self.assertNotEqual(checked_pre["state_sha256"], checked_post["state_sha256"])

    def test_recursive_firewall_and_overwrite_rejection(self):
        tainted = boundary()
        tainted["public_metadata"] = {"nested": {"future_chunk": "leak"}}
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_boundary_state(tainted)
        synonym = boundary()
        synonym["public_metadata"] = {"terminal_reward": 1.0}
        with self.assertRaisesRegex(ValueError, "unknown public"):
            validate_boundary_state(synonym)
        overwritten = boundary("post_write", turn=2, memories=["m2"])
        with self.assertRaisesRegex(ValueError, "phase-nested"):
            validate_boundary_state(overwritten)

    def test_global_hash_fold_is_set_invariant(self):
        roots = [f"r-{index}" for index in range(16)]
        one = stable_fold_assignments(roots, "e1-selection", 4)
        two = stable_fold_assignments(["aaa", *reversed(roots), "zzz"], "e1-selection", 4)
        self.assertTrue(all(one[root] == two[root] for root in roots))

    def test_oracle_interleaved_closure(self):
        report = oracle_boundary_decomposition(
            initial_post_value=0.2,
            pre_values=[0.3, 0.45],
            post_values=[0.55, 0.7],
            outcome=0.9,
        )
        self.assertLessEqual(abs(report["closure_error"]), 1e-15)
        self.assertTrue(np.allclose(report["chunk_credits"], [0.1, -0.1]))
        self.assertTrue(np.allclose(report["writer_credits"], [0.25, 0.25]))

    def test_broadcast_scale_reconstructs_sibling_form(self):
        returns = [0.0, 0.25, 0.5, 1.0]
        self.assertTrue(np.allclose(
            group_centered_broadcast(returns), sibling_reconstruction(returns), atol=1e-15,
        ))

    def test_inactive_slot_never_creates_fake_action(self):
        sampled, writer, answer = sampled_token_masks(
            sampled_lengths=[2, 0, 3], roles=["writer", "inactive", "answer"], token_width=4,
        )
        self.assertEqual(int(sampled.sum()), 5)
        self.assertFalse(sampled[1].any())
        self.assertFalse(np.any(writer & answer))
        with self.assertRaisesRegex(ValueError, "fictitious"):
            sampled_token_masks(sampled_lengths=[1], roles=["inactive"], token_width=2)

    def test_credit_bundle_detects_mutation(self):
        bundle = seal_credit_bundle(
            block_id="block", behavior_checkpoint_sha256="a" * 64,
            fold_receipts=[{"fold": 0}],
            rows=[{"content_root_id": "root", "trajectory_id": "trajectory", "turn_index": 1,
                   "writer_credit": 0.1, "answer_credit": -0.2}],
        )
        verify_sealed_credit_bundle(bundle, bundle["bundle_sha256"])
        bundle["rows"][0]["writer_credit"] = 9.0
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_sealed_credit_bundle(bundle, bundle["bundle_sha256"])

    def test_branch_accounting_uses_frozen_slots(self):
        sparse = sparse_branch_accounting(
            trunk_tokens=10, arm_writer_tokens=[3, 5], continuation_tokens=[7, 9],
            leaf_returns=[0.8, 0.2], other_replica_returns=[0.1, 0.4, 0.5],
            model_forward_tokens=34, model_backward_tokens=34,
            h20_seconds=4.0, wall_seconds=2.0, active=True,
        )
        self.assertEqual(sparse["physical_model_tokens"], 34)
        self.assertEqual(sparse["actor_weighted_tokens"], 22)
        self.assertEqual(full_branch_matched_slot_count(
            root_count=64, selected_scheduled_boundaries=11,
        ), 64 * 4 * 9 + 44)
        schedule = full_branch_block_schedule(
            block_id="b0", content_root_ids=[f"root-{index}" for index in range(64)],
            experiment_seed=2026,
        )
        self.assertEqual(schedule["candidate_count"], 512)
        self.assertEqual(schedule["selected_count"], 128)

    def test_e0_certificate_is_bound_to_run_and_rejects_tampering(self):
        output = Path("/absolute/mic-v2/run-a/certificates/e0.json")
        tests = []
        for test_id in E0_IDS:
            evidence = {"oracle": test_id, "value": 1.0}
            tests.append({
                "id": test_id,
                "status": "PASS",
                "evidence": evidence,
                "evidence_sha256": sha256_json(evidence),
            })
        payload = {
            "schema": SCHEMA,
            "kind": "e0_certificate",
            "status": "PASS",
            "decision": "MIC_V2_E0_PASS",
            "git_commit": "a" * 40,
            "run_id": "run-a",
            "output_path": str(output),
            "contract_sha256": CONTRACT_SHA256,
            "preregistration_manifest_sha256": "b" * 64,
            "python": "test",
            "numpy_version": "test",
            "torch_version": "test",
            "tests": tests,
        }
        payload["certificate_sha256"] = sha256_json(payload)
        verified = _verify_e0_certificate_payload(
            payload,
            run_id="run-a",
            expected_commit="a" * 40,
            output=output,
            manifest_sha256="b" * 64,
        )
        self.assertEqual(verified["run_id"], "run-a")

        tampered = copy.deepcopy(payload)
        tampered["tests"][0]["evidence"]["value"] = 2.0
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            _verify_e0_certificate_payload(
                tampered,
                run_id="run-a",
                expected_commit="a" * 40,
                output=output,
                manifest_sha256="b" * 64,
            )

        rebound = copy.deepcopy(payload)
        rebound["run_id"] = "run-b"
        unsigned = dict(rebound)
        unsigned.pop("certificate_sha256")
        rebound["certificate_sha256"] = sha256_json(unsigned)
        with self.assertRaisesRegex(RuntimeError, "run/output identity mismatch"):
            _verify_e0_certificate_payload(
                rebound,
                run_id="run-a",
                expected_commit="a" * 40,
                output=output,
                manifest_sha256="b" * 64,
            )


if __name__ == "__main__":
    unittest.main()
