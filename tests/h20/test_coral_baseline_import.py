import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from recurrent.research.cosi import canonical_sha256, sha256_file
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_fixed_s128


ROOT = Path(__file__).resolve().parents[2]
IMPORTER = ROOT / "tools/h20/import_cosi_original_baseline.py"
INTERFACES = ("I", "Original5", "Original10", "Original15", "Original20", "Original25")


class CoralBaselineImportEntryTests(unittest.TestCase):
    def _bundle(self, parent: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        bundle = parent / "readonly_original"
        bundle.mkdir()
        rows = [
            {
                "stable_id": str(index),
                "terminal_output": "\\boxed{Paris}" if index % 2 == 0 else "London",
                "ground_truth": "Paris",
            }
            for index in range(128)
        ]
        summary = summarize_fixed_s128(
            [score_terminal_output(row["terminal_output"], row["ground_truth"]) for row in rows]
        )
        validation = parent / "validation.parquet"
        pq.write_table(pa.Table.from_pylist([
            {
                "prompt": [{"role": "user", "content": f"question {index}"}],
                "context": f"context {index}",
                "extra_info": json.dumps({"index": index}),
                "reward_model": json.dumps({"ground_truth": "Paris"}),
            }
            for index in range(128)
        ]), validation)
        identity_rows = [{
            "example_id": str(index), "semantic_dataset_index": index,
            "source_order_index": index, "raw_row_position": index,
            "production_effective_position": index, "context_token_count": 1,
            "source_question_hash": canonical_sha256(f"question-{index}"),
            "source_context_hash": canonical_sha256(f"context-{index}"),
            "ground_truth_hash": canonical_sha256("Paris"),
        } for index in range(128)]
        identity_payload = {"rows": identity_rows}
        resolved = parent / "s128_resolved.json"
        resolved.write_text(json.dumps({
            "identity_payload": identity_payload,
            "eval_manifest_hash": canonical_sha256(identity_payload),
        }, sort_keys=True))
        files = []
        interfaces = {}
        for name in INTERFACES:
            path = bundle / f"{name}.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            files.append({"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size})
            interfaces[name] = {
                "predictions_path": path.name,
                "predictions_sha256": sha256_file(path),
                "expected_aggregate_sha256": canonical_sha256(summary),
            }
        unsigned = {
            "schema": "memagent.original-s128.readonly-bundle.v1",
            "source_commit": "fbb9bad4a4facad6a5bfc73d74186eb58cb5fe0e",
            "eval_manifest_hash": canonical_sha256(identity_payload),
            "files": files,
            "interfaces": interfaces,
        }
        index = {**unsigned, "index_sha256": canonical_sha256(unsigned)}
        (bundle / "index.json").write_text(json.dumps(index, sort_keys=True))
        for path in bundle.iterdir():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        bundle.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        return bundle, resolved, validation

    def _run(self, bundle: Path, resolved: Path, validation: Path, output: Path,
             expected_index_sha=None):
        eval_hash = json.loads(resolved.read_text())["eval_manifest_hash"]
        return subprocess.run(
            [sys.executable, str(IMPORTER), "--bundle-index", str(bundle / "index.json"),
             "--expected-bundle-index-sha256", expected_index_sha or sha256_file(bundle / "index.json"),
             "--expected-eval-manifest-sha256", eval_hash,
             "--s128-resolved-manifest", str(resolved),
             "--expected-s128-resolved-manifest-sha256", sha256_file(resolved),
             "--validation", str(validation),
             "--expected-validation-sha256", sha256_file(validation),
             "--output", str(output)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

    def _mutate_index(self, bundle: Path, mutation):
        bundle.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        path = bundle / "index.json"
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        index = json.loads(path.read_text())
        mutation(index)
        unsigned = {key: value for key, value in index.items() if key != "index_sha256"}
        index["index_sha256"] = canonical_sha256(unsigned)
        path.write_text(json.dumps(index, sort_keys=True))
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        bundle.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

    def test_real_import_recomputes_all_six_interfaces(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bundle, resolved, validation = self._bundle(parent)
            output = parent / "report.json"
            completed = self._run(bundle, resolved, validation, output)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["decision"], "COSI_BASELINE_IMPORT_PASS")
            self.assertEqual(set(report["aggregates"]), set(INTERFACES))
            self.assertTrue(report["source_read_only"])
            self.assertEqual(len(report["stable_inventory_sha256"]), 64)

    def test_entry_rejects_self_rehashed_protocol_and_metric_drift(self):
        mutations = (
            lambda index: index.__setitem__("eval_manifest_hash", "b" * 64),
            lambda index: index["interfaces"]["Original10"].__setitem__("expected_aggregate_sha256", "0" * 64),
            lambda index: index["interfaces"].pop("Original25"),
            lambda index: index.__setitem__("source_commit", "0" * 40),
        )
        for number, mutation in enumerate(mutations):
            with self.subTest(number=number), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                bundle, resolved, validation = self._bundle(parent)
                self._mutate_index(bundle, mutation)
                completed = self._run(bundle, resolved, validation, parent / "report.json")
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("ORIGINAL_BASELINE_PROTOCOL_MISMATCH", completed.stderr)

    def test_entry_rejects_writable_bundle_and_stable_identity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bundle, resolved, validation = self._bundle(parent)
            bundle.chmod(stat.S_IRWXU)
            completed = self._run(bundle, resolved, validation, parent / "writable.json")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("evidence bundle is not read-only", completed.stderr)

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bundle, resolved, validation = self._bundle(parent)
            source = bundle / "Original10.jsonl"
            source.chmod(stat.S_IRUSR | stat.S_IWUSR)
            completed = self._run(bundle, resolved, validation, parent / "writable_file.json")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("source file mismatch", completed.stderr)

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bundle, resolved, validation = self._bundle(parent)
            bundle.chmod(stat.S_IRWXU)
            path = bundle / "Original15.jsonl"
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            rows = path.read_text().splitlines()
            path.write_text("\n".join(reversed(rows)) + "\n")
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            self._mutate_index(
                bundle,
                lambda index: (
                    index["interfaces"]["Original15"].__setitem__("predictions_sha256", sha256_file(path)),
                    next(item for item in index["files"] if item["path"] == path.name).__setitem__("sha256", sha256_file(path)),
                ),
            )
            completed = self._run(bundle, resolved, validation, parent / "identity.json")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("stable-ID join/order", completed.stderr)

    def test_external_index_sha_rejects_consistently_rehashed_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bundle, resolved, validation = self._bundle(parent)
            trusted_index_sha = sha256_file(bundle / "index.json")
            bundle.chmod(stat.S_IRWXU)
            prediction = bundle / "Original5.jsonl"
            prediction.chmod(stat.S_IRUSR | stat.S_IWUSR)
            rows = [json.loads(line) for line in prediction.read_text().splitlines()]
            for row in rows:
                row["terminal_output"] = "Tokyo"
            prediction.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            prediction.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            replacement_summary = summarize_fixed_s128([
                score_terminal_output(row["terminal_output"], "Paris") for row in rows
            ])
            def mutation(index):
                item = next(value for value in index["files"] if value["path"] == prediction.name)
                item["sha256"] = sha256_file(prediction); item["size"] = prediction.stat().st_size
                spec = index["interfaces"]["Original5"]
                spec["predictions_sha256"] = sha256_file(prediction)
                spec["expected_aggregate_sha256"] = canonical_sha256(replacement_summary)
            self._mutate_index(bundle, mutation)
            completed = self._run(
                bundle, resolved, validation, parent / "replacement.json",
                expected_index_sha=trusted_index_sha,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("external bundle-index SHA", completed.stderr)

    def test_authenticated_bundle_cannot_override_parquet_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bundle, resolved, validation = self._bundle(parent)
            bundle.chmod(stat.S_IRWXU)
            prediction = bundle / "Original20.jsonl"
            prediction.chmod(stat.S_IRUSR | stat.S_IWUSR)
            rows = [json.loads(line) for line in prediction.read_text().splitlines()]
            rows[0]["ground_truth"] = "Rome"
            prediction.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            prediction.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            replacement_summary = summarize_fixed_s128([
                score_terminal_output(row["terminal_output"], row["ground_truth"]) for row in rows
            ])
            def mutation(index):
                item = next(value for value in index["files"] if value["path"] == prediction.name)
                item["sha256"] = sha256_file(prediction); item["size"] = prediction.stat().st_size
                spec = index["interfaces"]["Original20"]
                spec["predictions_sha256"] = sha256_file(prediction)
                spec["expected_aggregate_sha256"] = canonical_sha256(replacement_summary)
            self._mutate_index(bundle, mutation)
            completed = self._run(bundle, resolved, validation, parent / "ground_truth.json")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("embedded ground truth drift", completed.stderr)


if __name__ == "__main__":
    unittest.main()
