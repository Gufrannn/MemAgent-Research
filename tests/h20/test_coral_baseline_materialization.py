import json
import tempfile
import unittest
from pathlib import Path

from recurrent.research.cosi import sha256_file
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_fixed_s128
from recurrent.research.stable_eval_identity import (
    MANIFEST_ROW_FIELDS,
    canonical_sha256,
    evaluation_trajectory_seed,
    stable_key,
    stable_trajectory_id,
)
from tools.h20.materialize_cosi_original_baseline import score_interface


class CoralBaselineMaterializationTests(unittest.TestCase):
    def fixture(self, parent: Path):
        root = parent / "authenticated-final-root"
        (root / "terminal").mkdir(parents=True)
        eval_hash = "a" * 64
        frozen_rows, terminal_rows, metric_rows = [], [], []
        for index in range(128):
            frozen = {
                "example_id": str(index), "semantic_dataset_index": index,
                "source_order_index": index, "raw_row_position": index,
                "production_effective_position": index, "context_token_count": 1,
                "source_question_hash": canonical_sha256(["q", index]),
                "source_context_hash": canonical_sha256(["c", index]),
                "ground_truth_hash": canonical_sha256("Paris"),
            }
            seed = evaluation_trajectory_seed(
                base_seed=2026, eval_manifest_hash=eval_hash,
                example_id=str(index), source_order_index=index, replica_id=0,
            )
            row = {
                **frozen, "source_repeated_row": index, "eval_manifest_hash": eval_hash,
                "replica_id": 0, "trajectory_seed": seed,
                "trajectory_id": stable_trajectory_id(
                    eval_manifest_hash=eval_hash, example_id=str(index),
                    replica_id=0, trajectory_seed=seed,
                ),
                "output": "\\boxed{Paris}",
            }
            scored = score_terminal_output(row["output"], "Paris")
            metric_rows.append({
                "stable_key": json.dumps(stable_key(row), separators=(",", ":")),
                "source_order_index": index, "eval_manifest_hash": eval_hash,
                "example_id": str(index), "replica_id": 0,
                "trajectory_seed": seed, "trajectory_id": row["trajectory_id"], **scored,
            })
            frozen_rows.append(frozen); terminal_rows.append(row)
        (root / "terminal/5.jsonl").write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in terminal_rows
        ))
        for name in ("trajectory_turns.jsonl", "execution_summary.json", "run.log"):
            (root / name).write_text("authenticated\n")
        artifacts = {}
        for relative in ("terminal/5.jsonl", "trajectory_turns.jsonl",
                         "execution_summary.json", "run.log"):
            path = root / relative
            artifacts[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}
        details = {
            "root": str(root), "artifacts": artifacts,
            "independent_metric_rows_sha256": canonical_sha256(metric_rows),
            "metrics": summarize_fixed_s128(metric_rows),
        }
        return ({"root": str(root), "global_step": 5}, details,
                {"identity_payload": {"rows": frozen_rows}, "eval_manifest_hash": eval_hash},
                {index: "Paris" for index in range(128)}, canonical_sha256(metric_rows))

    def test_recomputes_from_authenticated_terminal_text(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, details, stable, truth, digest = self.fixture(Path(directory))
            metrics, observed, inventory = score_interface(
                "Original5", plan=plan, details=details, stable=stable,
                ground_truth=truth, expected_digest=digest,
            )
            self.assertEqual(observed, digest)
            self.assertEqual(metrics["denominator"], 128)
            self.assertEqual(set(inventory), {
                "terminal/5.jsonl", "trajectory_turns.jsonl",
                "execution_summary.json", "run.log",
            })

    def test_rejects_root_digest_and_terminal_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, details, stable, truth, digest = self.fixture(Path(directory))
            forged = json.loads(json.dumps(details)); forged["root"] = str(Path(directory) / "other")
            with self.assertRaisesRegex(ValueError, "root binding"):
                score_interface("Original5", plan=plan, details=forged, stable=stable,
                                ground_truth=truth, expected_digest=digest)
            with self.assertRaisesRegex(ValueError, "metric rows"):
                score_interface("Original5", plan=plan, details=details, stable=stable,
                                ground_truth=truth, expected_digest="0" * 64)
            terminal = Path(plan["root"]) / "terminal/5.jsonl"
            terminal.write_text(terminal.read_text().replace("Paris", "Tokyo", 1))
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                score_interface("Original5", plan=plan, details=details, stable=stable,
                                ground_truth=truth, expected_digest=digest)

    def test_manifest_records_missing_original_actual_loss_without_blocking(self):
        manifest = json.loads((Path(__file__).resolve().parents[2] /
                               "manifests/h20/qwen25_7b_cosi_seed2026.json").read_text())
        self.assertEqual(manifest["evidence_authority"]["actual_loss"], {
            "status": "PENDING_ACTUAL_LOSS_LEDGER",
            "original_rank_ledgers_available": False,
            "forbid_metric_as_loss": True,
            "forbid_original_rerun": True,
        })


if __name__ == "__main__":
    unittest.main()
