import copy
import tempfile
import unittest
from pathlib import Path
from recurrent.research.cosi import sha256_file

from tools.h20.audit_qwen25_7b_cosi import (
    build_anchor_comparison, summarize_anchor_curve, validate_resume_record,
)


def metrics(em=0.25, f1=0.40, format_success=0.80):
    return {
        "denominator": 128,
        "normalized_exact_match": em,
        "token_f1": f1,
        "format_success": format_success,
        "historical_sub_exact_match_diagnostic": 0.50,
    }


def baseline():
    return {
        "eval_manifest_hash": "a" * 64,
        "stable_inventory_sha256": "b" * 64,
        "aggregates": {
            f"Original{step}": metrics(f1=0.30 + step / 1000)
            for step in (5, 10, 15, 20, 25)
        },
    }


def evaluation(step, f1=0.50):
    return {
        "step": step,
        "eval_manifest_hash": "a" * 64,
        "stable_inventory_sha256": "b" * 64,
        "metrics": metrics(f1=f1),
    }


class CoralFinalAnchorAuditTests(unittest.TestCase):
    def resume_record(self, source):
        acknowledgements = []
        for rank in (0, 1):
            acknowledgements.append({
                "rank": rank,
                "model_loaded": True,
                "optimizer_loaded": True,
                "extra_loaded": True,
                "rng_state_keys": ["cpu", "cuda", "numpy", "random"],
                "rng_restored": True,
                "lr_scheduler_loaded": True,
                "optimizer_state_entry_count": 100,
                "optimizer_step_entry_count": 100,
                "optimizer_step_min": 5,
                "optimizer_step_max": 5,
                "lr_scheduler_last_epoch": 5,
            })
        return {
            "global_step": 5,
            "resume_source": str(Path(source).resolve()),
            "actor_model_optimizer_extra_loaded": True,
            "actor_load_worker_acks": acknowledgements,
            "data_loaded": True,
            "data_sha256": sha256_file(Path(source) / "data.pt"),
        }

    def test_five_authenticated_anchor_comparisons_and_curve_summary(self):
        source = baseline()
        comparisons = [
            build_anchor_comparison(step, evaluation(step, 0.40 + step / 1000), source)
            for step in (5, 10, 15, 20, 25)
        ]
        self.assertEqual([row["step"] for row in comparisons], [5, 10, 15, 20, 25])
        self.assertTrue(all(
            abs(row["method_minus_original"]["token_f1"] - 0.10) < 1e-12
            for row in comparisons
        ))
        summary = summarize_anchor_curve(comparisons)
        self.assertAlmostEqual(summary["token_f1"]["mean_delta"], 0.10)
        self.assertAlmostEqual(summary["token_f1"]["worst_anchor_delta"], 0.10)

    def test_wrong_anchor_identity_protocol_and_metric_tampering_fail(self):
        mutations = []
        value = evaluation(5)
        value["step"] = 10
        mutations.append(value)
        value = evaluation(5)
        value["stable_inventory_sha256"] = "c" * 64
        mutations.append(value)
        value = evaluation(5)
        value["eval_manifest_hash"] = "d" * 64
        mutations.append(value)
        value = evaluation(5)
        value["metrics"]["denominator"] = 127
        mutations.append(value)
        value = evaluation(5)
        value["metrics"]["token_f1"] = float("nan")
        mutations.append(value)
        value = evaluation(5)
        value["metrics"]["unexpected"] = 0
        mutations.append(value)
        for value in mutations:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "CORAL_AUDIT_NO_GO"):
                    build_anchor_comparison(5, value, baseline())

    def test_missing_original_anchor_and_partial_final_curve_fail(self):
        source = baseline()
        del source["aggregates"]["Original5"]
        with self.assertRaisesRegex(ValueError, "CORAL_AUDIT_NO_GO"):
            build_anchor_comparison(5, evaluation(5), source)
        with self.assertRaisesRegex(ValueError, "CORAL_AUDIT_NO_GO"):
            summarize_anchor_curve([
                build_anchor_comparison(step, evaluation(step), baseline())
                for step in (5, 10, 15)
            ])

    def test_resume_requires_both_ranks_and_all_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "global_step_5"
            source.mkdir(); (source / "data.pt").write_bytes(b"rank-complete-data-state")
            validate_resume_record(self.resume_record(source), source)
            mutations = []
            value = self.resume_record(source)
            value["actor_load_worker_acks"].pop()
            mutations.append(value)
            value = self.resume_record(source)
            value["actor_load_worker_acks"][1]["rng_restored"] = False
            mutations.append(value)
            value = self.resume_record(source)
            value["actor_load_worker_acks"][0]["lr_scheduler_loaded"] = False
            mutations.append(value)
            value = self.resume_record(source)
            value["actor_load_worker_acks"][0]["optimizer_loaded"] = False
            mutations.append(value)
            value = self.resume_record(source)
            value["data_loaded"] = False
            mutations.append(value)
            value = self.resume_record(source)
            value["resume_source"] = "/tmp/wrong/global_step_5"
            mutations.append(value)
            value = self.resume_record(source)
            value["data_sha256"] = "d" * 64
            mutations.append(value)
            for value in mutations:
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "CORAL_AUDIT_NO_GO"):
                        validate_resume_record(value, source)


if __name__ == "__main__":
    unittest.main()
