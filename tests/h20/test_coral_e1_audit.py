import copy
import unittest

from recurrent.research.coral_e1 import (
    SKETCH_BASIS_SHA256, validate_dataproto_clone_oracle_report,
    validate_fsdp_sketch_oracle_report,
)
from recurrent.research.cosi import canonical_sha256
from tools.h20.audit_coral_e1 import (
    PREREGISTRATION, PROPOSAL_STEPS, audit_evidence, validate_proposal,
)

COMMIT = "1" * 40


def clone_oracle_report():
    value = {
        "schema": "memagent.coral.dataproto-clone-oracle.v1",
        "status": "PASS",
        "decision": "CORAL_DATAPROTO_CLONE_ORACLE_PASS",
        "batch_size": 8,
        "zero_leaf_keys": 0,
        "tensor_clone_independent": True,
        "non_tensor_clone_independent": True,
        "meta_clone_independent": True,
        "python_version": "3.10.14",
        "torch_version": "2.4.0+cu121",
        "tensordict_version": "0.6.2",
    }
    value["report_sha256"] = canonical_sha256(value)
    return value


def oracle_report():
    value = {
        "schema": "memagent.coral.e1-fsdp-sketch-oracle.v4",
        "status": "PASS",
        "decision": "CORAL_E1_SKETCH_ORACLE_PASS",
        "world_size": 2,
        "backend": "nccl",
        "basis_sha256": SKETCH_BASIS_SHA256,
        "full_gradient_elements": 221,
        "padded_shard_elements": 222,
        "full_gradient_max_abs_error": 1e-6,
        "sketch_max_abs_error": 5e-8,
        "sketch_assembly_aperture": 1e-7,
        "projection_relative_norm_error": 0.01,
        "projection_error_aperture": 0.10,
        "collision_calibration_elements": 1_000_003,
        "collision_calibration_buckets_per_basis": 256,
        "collision_calibration_exact_norm": 100.0,
        "collision_calibration_projected_norm": 101.0,
        "collision_calibration_relative_norm_error": 0.01,
        "collision_calibration_error_aperture": 0.10,
        "two_rank_local_denominators": [5, 7],
        "denominator_gradient_closure_max_abs_error": 1e-8,
        "denominator_gradient_closure_aperture": 1e-6,
        "ordinal_calibration_parameters": 64,
        "ordinal_calibration_max_abs_error": 0.0,
        "ordinal_calibration_error_aperture": 1e-12,
    }
    value["report_sha256"] = canonical_sha256(value)
    return value


def proposal(step):
    roots = [f"step-{step:02d}-root-{index}" for index in range(4)]
    records = []
    for index, root in enumerate(roots):
        records.append({
            "root_id": root,
            "dataset_index": (step - 1) * 4 + index,
            "writer_replicas": 2,
            "common_trajectory_seeds": [step * 100 + index * 2,
                                          step * 100 + index * 2 + 1],
            "common_terminal_request_seeds": [100000 + step * 100 + index * 2,
                                                100001 + step * 100 + index * 2],
            "cached_memory_token_ids_sha256": "a" * 64,
            "refreshed_memory_token_ids_sha256": "b" * 64,
            "cached_prompt_token_ids_sha256": "c" * 64,
            "refreshed_prompt_token_ids_sha256": "d" * 64,
            "cached_terminal_answer_token_ids_sha256": "1" * 64,
            "refreshed_terminal_answer_token_ids_sha256": "2" * 64,
            "cached_reward_sha256": "3" * 64,
            "refreshed_reward_sha256": "4" * 64,
            "cached_advantage_sha256": "5" * 64,
            "refreshed_advantage_sha256": "6" * 64,
            "terminal_action_policy": "both_branches_freshly_sampled_at_fixed_proposal_weights",
            "cached_memory_token_count": 100,
            "refreshed_memory_token_count": 101,
            "cached_gradient_sha256": "e" * 64,
            "refreshed_gradient_sha256": "f" * 64,
            "cached_gradient_norm": 1.0,
            "refreshed_gradient_norm": 1.0,
            "symmetric_relative_response": 0.2,
            "duplicate_control_response_norm": 0.0,
            "same_batch_writer_answer_cosine": 0.4,
            "tensor_source": "actual_terminal_answer_loss_backward",
        })
    value = {
        "schema": "memagent.coral.e1-proposal.v3",
        "producer": "ray_ppo_trainer_actual_loss_backward",
        "git_commit": COMMIT,
        "global_step": step,
        "source_weight_sample_digest": "2" * 64,
        "proposal_weight_sample_digest": "3" * 64,
        "gradient_sketch_basis_sha256": SKETCH_BASIS_SHA256,
        "root_inventory_sha256": canonical_sha256(sorted(roots)),
        "records": records,
    }
    value["proposal_sha256"] = canonical_sha256(value)
    return value


def evidence():
    proposals = [proposal(step) for step in PROPOSAL_STEPS]
    embedded_clone_oracle = clone_oracle_report()
    embedded_oracle = oracle_report()
    bindings = [{
        "global_step": step,
        "source_checkpoint_inventory_sha256": "4" * 64,
        "proposal_checkpoint_inventory_sha256": "5" * 64,
        "proposal_sha256": value["proposal_sha256"],
    } for step, value in zip(PROPOSAL_STEPS, proposals)]
    value = {
        "schema": "memagent.coral.e1.v4",
        "git_commit": COMMIT,
        "preregistration": PREREGISTRATION,
        "gate_a_ledger_sha256": "6" * 64,
        "dataproto_clone_oracle_report_sha256":
            embedded_clone_oracle["report_sha256"],
        "dataproto_clone_oracle_report": embedded_clone_oracle,
        "fsdp_sketch_oracle_report_sha256": embedded_oracle["report_sha256"],
        "fsdp_sketch_oracle_report": embedded_oracle,
        "proposal_bindings": bindings,
        "proposals": proposals,
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


class CoralE1AuditTests(unittest.TestCase):
    def test_clone_oracle_complete_contract_and_tamper_rejection(self):
        validate_dataproto_clone_oracle_report(clone_oracle_report())
        mutations = []
        value = clone_oracle_report()
        value["schema"] = "memagent.coral.dataproto-clone-oracle.v0"
        mutations.append(value)
        value = clone_oracle_report()
        value["zero_leaf_keys"] = 1
        mutations.append(value)
        value = clone_oracle_report()
        value["tensor_clone_independent"] = False
        mutations.append(value)
        value = clone_oracle_report()
        value["torch_version"] = ""
        mutations.append(value)
        for value in mutations:
            value["report_sha256"] = canonical_sha256({
                key: item for key, item in value.items()
                if key != "report_sha256"
            })
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "CORAL_E1_NO_GO"):
                    validate_dataproto_clone_oracle_report(value)

    def test_complete_v4_oracle_contract_and_forged_old_pass_rejection(self):
        validate_fsdp_sketch_oracle_report(oracle_report())
        old_forged = {
            "status": "PASS",
            "decision": "CORAL_E1_SKETCH_ORACLE_PASS",
            "basis_sha256": SKETCH_BASIS_SHA256,
        }
        old_forged["report_sha256"] = canonical_sha256(old_forged)
        with self.assertRaisesRegex(ValueError, "CORAL_E1_NO_GO"):
            validate_fsdp_sketch_oracle_report(old_forged)
        bad_calibration = oracle_report()
        bad_calibration["collision_calibration_projected_norm"] = 130.0
        bad_calibration["collision_calibration_relative_norm_error"] = 0.30
        bad_calibration["report_sha256"] = canonical_sha256({
            key: item for key, item in bad_calibration.items()
            if key != "report_sha256"
        })
        with self.assertRaisesRegex(ValueError, "CORAL_E1_NO_GO"):
            validate_fsdp_sketch_oracle_report(bad_calibration)
        float_dimension = oracle_report()
        float_dimension["full_gradient_elements"] = 221.0
        float_dimension["report_sha256"] = canonical_sha256({
            key: item for key, item in float_dimension.items()
            if key != "report_sha256"
        })
        with self.assertRaisesRegex(ValueError, "CORAL_E1_NO_GO"):
            validate_fsdp_sketch_oracle_report(float_dimension)
        float_denominator = oracle_report()
        float_denominator["two_rank_local_denominators"] = [5.0, 7]
        float_denominator["report_sha256"] = canonical_sha256({
            key: item for key, item in float_denominator.items()
            if key != "report_sha256"
        })
        with self.assertRaisesRegex(ValueError, "CORAL_E1_NO_GO"):
            validate_fsdp_sketch_oracle_report(float_denominator)
        assembly_over_aperture = oracle_report()
        assembly_over_aperture["sketch_max_abs_error"] = 1.0000001e-7
        assembly_over_aperture["report_sha256"] = canonical_sha256({
            key: item for key, item in assembly_over_aperture.items()
            if key != "report_sha256"
        })
        with self.assertRaisesRegex(ValueError, "CORAL_E1_NO_GO"):
            validate_fsdp_sketch_oracle_report(assembly_over_aperture)
        ordinal_over_aperture = oracle_report()
        ordinal_over_aperture["ordinal_calibration_max_abs_error"] = 1.01e-12
        ordinal_over_aperture["report_sha256"] = canonical_sha256({
            key: item for key, item in ordinal_over_aperture.items()
            if key != "report_sha256"
        })
        with self.assertRaisesRegex(ValueError, "CORAL_E1_NO_GO"):
            validate_fsdp_sketch_oracle_report(ordinal_over_aperture)

    def test_trainer_produced_root_cluster_evidence_passes(self):
        rows, proposal_mean, passed = validate_proposal(proposal(1), 1, COMMIT)
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(proposal_mean, 0.2)
        self.assertTrue(passed)
        report = audit_evidence(evidence())
        self.assertEqual(report["decision"], "CORAL_E1_PASS")
        self.assertAlmostEqual(report["proposal_cluster_lcb"], 0.2)

    def test_proposal_cluster_lcb_does_not_pseudoreplicate_roots(self):
        value = evidence()
        # Within-proposal roots are identical, while proposal means vary.  A
        # root-level SE would be spuriously divided by sqrt(32); the audited
        # SE must equal stdev(eight proposal means)/sqrt(8).
        proposal_values = [0.02, 0.02, 0.02, 0.02, 0.30, 0.30, 0.30, 0.30]
        for proposal_value, response in zip(value["proposals"], proposal_values):
            for row in proposal_value["records"]:
                row["symmetric_relative_response"] = response
            proposal_value["proposal_sha256"] = canonical_sha256({
                key: item for key, item in proposal_value.items()
                if key != "proposal_sha256"
            })
        for binding, proposal_value in zip(value["proposal_bindings"], value["proposals"]):
            binding["proposal_sha256"] = proposal_value["proposal_sha256"]
        value["evidence_sha256"] = canonical_sha256({
            key: item for key, item in value.items() if key != "evidence_sha256"
        })
        report = audit_evidence(value)
        expected_se = __import__("statistics").stdev(proposal_values) / (8 ** 0.5)
        self.assertAlmostEqual(report["proposal_cluster_standard_error"], expected_se)
        self.assertLess(report["proposal_cluster_lcb"], 0.05)
        self.assertEqual(report["decision"], "CORAL_E1_NO_GO")

    def test_tamper_basis_noise_root_reuse_and_checkpoint_alias_fail(self):
        mutations = []
        value = evidence()
        value["proposals"][0]["gradient_sketch_basis_sha256"] = "7" * 64
        mutations.append(value)
        value = evidence()
        value["proposals"][0]["records"][0]["duplicate_control_response_norm"] = 1.0
        mutations.append(value)
        value = evidence()
        value["proposals"][1]["records"][0]["root_id"] = \
            value["proposals"][0]["records"][0]["root_id"]
        mutations.append(value)
        value = evidence()
        value["proposal_bindings"][0]["proposal_checkpoint_inventory_sha256"] = "4" * 64
        mutations.append(value)
        value = evidence()
        value["proposals"][0]["records"][0]["common_terminal_request_seeds"] = [1, 1]
        mutations.append(value)
        value = evidence()
        value["proposals"][0]["records"][0]["terminal_action_policy"] = "reuse_source_answer"
        mutations.append(value)
        value = evidence()
        value["dataproto_clone_oracle_report"]["status"] = "FAIL"
        mutations.append(value)
        value = evidence()
        value["dataproto_clone_oracle_report_sha256"] = "7" * 64
        mutations.append(value)
        for value in mutations:
            for proposal_value in value["proposals"]:
                proposal_value["root_inventory_sha256"] = canonical_sha256(sorted(
                    row["root_id"] for row in proposal_value["records"]
                ))
                proposal_value["proposal_sha256"] = canonical_sha256({
                    key: item for key, item in proposal_value.items()
                    if key != "proposal_sha256"
                })
            for binding, proposal_value in zip(
                    value["proposal_bindings"], value["proposals"]):
                binding["proposal_sha256"] = proposal_value["proposal_sha256"]
            value["evidence_sha256"] = canonical_sha256({
                key: item for key, item in value.items() if key != "evidence_sha256"
            })
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "CORAL_E1_NO_GO"):
                    audit_evidence(value)


if __name__ == "__main__":
    unittest.main()
