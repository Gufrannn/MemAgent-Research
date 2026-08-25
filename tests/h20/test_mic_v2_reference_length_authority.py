import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from tools.h20 import mic_v2_reference_length_authority as authority_module
from tools.h20.mic_v2_reference_length_authority import (
    ARTIFACT_SUFFIXES,
    AUTHORITY_COMMIT,
    AUTHORITY_FILE_SHA256,
    AUTHORITY_REL,
    CANONICAL_FIELDS,
    CODE_PATHS,
    EXECUTION_FIELDS,
    P0_FIELDS,
    PRODUCER_COMMIT,
    REFERENCE_FIELDS,
    REPLAY_FIELDS,
    _exact_result,
    _fixed_authority,
    _label_blind_inputs_frozen,
    _safe_repo_relative,
    _under,
    _verify_authority,
    sha256_file,
    sha256_json,
)


REPO = Path(__file__).resolve().parents[2]
AUTHORITY = REPO / "manifests/h20/qwen25_7b_mic_v2_reference_length_authority.json"
ENTRY = REPO / "scripts/h20/verify_qwen25_7b_mic_v2_reference_length_authority.sh"


class MicV2ReferenceLengthAuthorityTest(unittest.TestCase):
    def test_frozen_authority_self_digest_code_and_arithmetic(self):
        authority = json.loads(AUTHORITY.read_text())
        digest = authority.pop("authority_sha256")
        self.assertEqual(sha256_json(authority), digest)
        for receipt in authority["code_authority"].values():
            self.assertEqual(sha256_file(REPO / receipt["path"]), receipt["sha256"])
        result = authority["result"]
        self.assertEqual(result["writer_policy_tokens"] + result["answer_policy_tokens"],
                         result["total_policy_tokens"])
        self.assertEqual(result["trajectory_count"] * 9,
                         result["scheduled_slot_count"])
        self.assertEqual(result["total_policy_tokens"] / result["scheduled_slot_count"],
                         result["lbar_ref"])

    def _synthetic_bundle(self, root: Path):
        authority = json.loads(AUTHORITY.read_text())
        output_rel = "logs/synthetic/reference-run"
        output = root / output_rel
        output.mkdir(parents=True)
        authority["producer"]["output_root_relative_to_work_root"] = output_rel
        manifest = {
            "data_freeze_authority": {
                **authority["data_freeze_authority"],
                "root": str(root / "data-freeze"),
            },
            "source": {"path": "/frozen/train.parquet", "sha256": "1" * 64},
        }
        resolved = {"splits": {"reference_length_calibration": {"rows": [
            {
                "source_position": index,
                "semantic_dataset_index": index + 100,
                "question_sha256": hashlib.sha256(f"q{index}".encode()).hexdigest(),
                "context_sha256": hashlib.sha256(f"c{index}".encode()).hexdigest(),
                "content_root_id": hashlib.sha256(f"r{index}".encode()).hexdigest(),
            }
            for index in range(64)
        ]}}}
        inputs = _label_blind_inputs_frozen(manifest, resolved)
        result = authority["result"]
        seed = authority["seed_authority"]
        pair = authority["generation_authority"]["gpu_pair"]
        identity = authority["generation_authority"]["physical_gpu_identity"]
        loader = authority["generation_authority"]["config_loader_environment"]
        version = authority["generation_authority"]["vllm_version"]
        code_sha = {receipt["path"]: receipt["sha256"]
                    for name, receipt in authority["code_authority"].items()
                    if name != "scientific_contract"}
        p0 = {field: None for field in P0_FIELDS}
        p0.update({
            "schema": "memagent.mic.v2.reference-length-p0", "status": "PASS",
            "decision": "MIC_V2_REFERENCE_LENGTH_P0_PASS", "git_commit": PRODUCER_COMMIT,
            "run_id": authority["producer"]["run_id"], "output_root": str(output),
            "gpu_pair": pair,
            "scientific_contract_sha256": authority["code_authority"]["scientific_contract"]["sha256"],
            "manifest_path": str(root / authority["generation_authority"]["manifest_path"]),
            "manifest_sha256": authority["generation_authority"]["manifest_sha256"],
            "data_freeze_certificate_file_sha256": authority["data_freeze_authority"]["certificate_file_sha256"],
            "data_freeze_resolved_file_sha256": authority["data_freeze_authority"]["resolved_file_sha256"],
            "source_sha256": manifest["source"]["sha256"], "model_files": [],
            "tokenization_authority": {}, "seed_authority": seed,
            "materialization_authority": {}, "code_sha256": code_sha,
            "label_blind_inputs": str(output / ARTIFACT_SUFFIXES["label_blind_inputs"]),
            "label_blind_inputs_sha256": inputs["inputs_sha256"],
            "label_blind_source": {"file_sha256": "2" * 64},
            "expected_trajectories": result["trajectory_count"],
            "expected_scheduled_slots": result["scheduled_slot_count"],
        })
        p0["p0_sha256"] = sha256_json({k: v for k, v in p0.items() if k != "p0_sha256"})
        execution = {field: None for field in EXECUTION_FIELDS}
        execution.update({
            "schema": "memagent.mic.v2.reference-length-execution", "status": "PASS",
            "git_commit": PRODUCER_COMMIT, "run_id": p0["run_id"], "p0_sha256": p0["p0_sha256"],
            "gpu_pair": pair, "physical_gpu_identity": identity, "vllm_version": version,
            "config_loader_environment": loader, "strict_vllm": True,
            "tensor_parallel_size": 2, "prefix_cache_enabled": False,
            "termination_token_ids": [151645, 151643], "trainer_attached": False,
            "actor_updates": 0, "new_generate_calls_this_session": result["exact_token_match_count"],
            "represented_generate_calls": result["exact_token_match_count"],
            "trajectory_count": result["trajectory_count"], "ledger_file_sha256": "3" * 64,
        })
        execution["execution_sha256"] = sha256_json(
            {k: v for k, v in execution.items() if k != "execution_sha256"})
        replay = {field: None for field in REPLAY_FIELDS}
        replay.update({
            "schema": "memagent.mic.v2.reference-length-gpu-replay", "status": "PASS",
            "decision": "MIC_V2_REFERENCE_LENGTH_GPU_REPLAY_PASS", "git_commit": PRODUCER_COMMIT,
            "run_id": p0["run_id"], "p0_sha256": p0["p0_sha256"],
            "execution_sha256": execution["execution_sha256"], "gpu_pair": pair,
            "physical_gpu_identity": identity, "vllm_version": version,
            "config_loader_environment": loader, "termination_token_ids": [151645, 151643],
            "trajectory_count": result["trajectory_count"],
            "regenerated_generate_calls": result["exact_token_match_count"],
            "exact_token_match_count": result["exact_token_match_count"],
            "ledger_file_sha256": execution["ledger_file_sha256"],
        })
        replay["gpu_replay_sha256"] = sha256_json(
            {k: v for k, v in replay.items() if k != "gpu_replay_sha256"})
        reference = {field: None for field in REFERENCE_FIELDS}
        reference.update({
            "schema": "memagent.mic.v2.reference-length-certificate", "status": "PASS",
            "decision": "MIC_V2_REFERENCE_LENGTH_CALIBRATION_PASS", "git_commit": PRODUCER_COMMIT,
            "run_id": p0["run_id"], "output_root": str(output),
            "scientific_contract_sha256": p0["scientific_contract_sha256"],
            "manifest_sha256": p0["manifest_sha256"], "p0_sha256": p0["p0_sha256"],
            "label_blind_inputs_sha256": inputs["inputs_sha256"],
            "execution_sha256": execution["execution_sha256"],
            "gpu_replay_sha256": replay["gpu_replay_sha256"], "seed_authority": seed,
            "materialization_authority": {}, "statistic": result["statistic"],
            "lbar_ref": result["lbar_ref"],
            **{key: result[key] for key in (
                "trajectory_count", "scheduled_slot_count", "active_writer_slot_count",
                "writer_policy_tokens", "answer_policy_tokens", "total_policy_tokens")},
            "ledger_tail_sha256": "4" * 64, "ledger_file_sha256": execution["ledger_file_sha256"],
        })
        reference["certificate_sha256"] = sha256_json(
            {k: v for k, v in reference.items() if k != "certificate_sha256"})
        values = {"p0": p0, "execution": execution, "gpu_replay": replay,
                  "reference_length": reference, "label_blind_inputs": inputs}
        raw_values = {"label_blind_source": b"source\n", "trajectory_ledger": b"ledger\n",
                      "calibration_log": b"PASS\n"}
        for name, suffix in ARTIFACT_SUFFIXES.items():
            path = output / suffix
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = (json.dumps(values[name], sort_keys=True, separators=(",", ":")) + "\n").encode() \
                if name in values else raw_values[name]
            path.write_bytes(raw)
            authority["artifacts"][name]["path_relative_to_work_root"] = f"{output_rel}/{suffix}"
            authority["artifacts"][name]["file_sha256"] = hashlib.sha256(raw).hexdigest()
            if name in CANONICAL_FIELDS:
                authority["artifacts"][name]["canonical_field"] = CANONICAL_FIELDS[name]
                authority["artifacts"][name]["canonical_sha256"] = values[name][CANONICAL_FIELDS[name]]
        authority["artifacts"]["label_blind_source"]["file_sha256"] = p0["label_blind_source"]["file_sha256"]
        # Match the physical files to the cross-chain receipts.
        source_path = output / ARTIFACT_SUFFIXES["label_blind_source"]
        p0["label_blind_source"]["file_sha256"] = sha256_file(source_path)
        values["p0"]["p0_sha256"] = sha256_json({k: v for k, v in p0.items() if k != "p0_sha256"})
        # Rebuild dependent JSONs once after the source receipt changed.
        return authority, manifest, resolved, values, output

    def _write_signed_bundle(self, root: Path, authority, manifest, values, output):
        p0 = values["p0"]
        p0["p0_sha256"] = sha256_json({k: v for k, v in p0.items() if k != "p0_sha256"})
        values["execution"]["p0_sha256"] = p0["p0_sha256"]
        values["execution"]["execution_sha256"] = sha256_json(
            {k: v for k, v in values["execution"].items() if k != "execution_sha256"})
        values["gpu_replay"].update({"p0_sha256": p0["p0_sha256"],
                                      "execution_sha256": values["execution"]["execution_sha256"]})
        values["gpu_replay"]["gpu_replay_sha256"] = sha256_json(
            {k: v for k, v in values["gpu_replay"].items() if k != "gpu_replay_sha256"})
        values["reference_length"].update({
            "p0_sha256": p0["p0_sha256"],
            "execution_sha256": values["execution"]["execution_sha256"],
            "gpu_replay_sha256": values["gpu_replay"]["gpu_replay_sha256"],
        })
        values["reference_length"]["certificate_sha256"] = sha256_json(
            {k: v for k, v in values["reference_length"].items() if k != "certificate_sha256"})
        for name, value in values.items():
            path = output / ARTIFACT_SUFFIXES[name]
            path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            authority["artifacts"][name]["file_sha256"] = sha256_file(path)
            authority["artifacts"][name]["canonical_sha256"] = value[CANONICAL_FIELDS[name]]
        authority["artifacts"]["label_blind_source"]["file_sha256"] = sha256_file(
            output / ARTIFACT_SUFFIXES["label_blind_source"])
        authority["artifacts"]["trajectory_ledger"]["file_sha256"] = sha256_file(
            output / ARTIFACT_SUFFIXES["trajectory_ledger"])
        values["execution"]["ledger_file_sha256"] = authority["artifacts"]["trajectory_ledger"]["file_sha256"]
        values["gpu_replay"]["ledger_file_sha256"] = values["execution"]["ledger_file_sha256"]
        values["reference_length"]["ledger_file_sha256"] = values["execution"]["ledger_file_sha256"]
        # Re-sign once more after the ledger binding.
        return self._write_signed_bundle_final(authority, values, output)

    def _write_signed_bundle_final(self, authority, values, output):
        p0 = values["p0"]
        p0["p0_sha256"] = sha256_json({k: v for k, v in p0.items() if k != "p0_sha256"})
        execution = values["execution"]
        execution["p0_sha256"] = p0["p0_sha256"]
        execution["execution_sha256"] = sha256_json({k: v for k, v in execution.items() if k != "execution_sha256"})
        replay = values["gpu_replay"]
        replay.update({"p0_sha256": p0["p0_sha256"], "execution_sha256": execution["execution_sha256"]})
        replay["gpu_replay_sha256"] = sha256_json({k: v for k, v in replay.items() if k != "gpu_replay_sha256"})
        reference = values["reference_length"]
        reference.update({"p0_sha256": p0["p0_sha256"], "execution_sha256": execution["execution_sha256"],
                          "gpu_replay_sha256": replay["gpu_replay_sha256"]})
        reference["certificate_sha256"] = sha256_json({k: v for k, v in reference.items() if k != "certificate_sha256"})
        for name, value in values.items():
            path = output / ARTIFACT_SUFFIXES[name]
            path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            authority["artifacts"][name]["file_sha256"] = sha256_file(path)
            authority["artifacts"][name]["canonical_sha256"] = value[CANONICAL_FIELDS[name]]
        unsigned = dict(authority)
        unsigned.pop("authority_sha256", None)
        authority["authority_sha256"] = sha256_json(unsigned)
        return authority

    def test_full_synthetic_authority_passes_and_future_checkout_uses_historical_blobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            authority, manifest, resolved, values, output = self._synthetic_bundle(root)
            manifest_raw = json.dumps(manifest, sort_keys=True).encode()
            authority["generation_authority"]["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
            values["p0"]["manifest_sha256"] = authority["generation_authority"]["manifest_sha256"]
            values["reference_length"]["manifest_sha256"] = authority["generation_authority"]["manifest_sha256"]
            blobs = {authority["generation_authority"]["manifest_path"]: manifest_raw}
            for name, receipt in authority["code_authority"].items():
                blobs[receipt["path"]] = f"historical:{name}".encode()
                receipt["sha256"] = hashlib.sha256(blobs[receipt["path"]]).hexdigest()
                if name != "scientific_contract":
                    values["p0"]["code_sha256"][receipt["path"]] = receipt["sha256"]
                else:
                    values["p0"]["scientific_contract_sha256"] = receipt["sha256"]
                    values["reference_length"]["scientific_contract_sha256"] = receipt["sha256"]
            authority = self._write_signed_bundle(root, authority, manifest, values, output)
            def historical(_repo, _commit, path):
                return blobs[path]
            with mock.patch.object(authority_module, "_git_blob", side_effect=historical), \
                 mock.patch.object(authority_module, "_ancestor", return_value=True), \
                 mock.patch.object(authority_module, "_verify_data_freeze_frozen",
                                   return_value=({}, resolved)):
                report = _verify_authority(REPO, root, authority)
            self.assertEqual(report["status"], "PASS")

    def test_self_consistent_authority_resign_is_rejected_by_git_pin(self):
        forged = json.loads(AUTHORITY.read_text())
        forged["result"]["lbar_ref"] += 1.0
        forged["authority_sha256"] = sha256_json(
            {key: value for key, value in forged.items() if key != "authority_sha256"})
        raw = (json.dumps(forged, sort_keys=True) + "\n").encode()
        with mock.patch.object(authority_module, "_git_blob", return_value=raw), \
             mock.patch.object(authority_module, "_ancestor", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Git blob SHA"):
                _fixed_authority(REPO, AUTHORITY_COMMIT)

    def test_frozen_authority_loads_from_git_object_on_descendant_checkout(self):
        head = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
        ).strip()
        loaded = _fixed_authority(REPO, head)
        self.assertEqual(loaded["authority_sha256"],
                         json.loads(AUTHORITY.read_text())["authority_sha256"])
        self.assertEqual(sha256_file(AUTHORITY), AUTHORITY_FILE_SHA256)

    def test_cross_run_artifact_path_is_rejected_before_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            authority, _manifest, _resolved, _values, _output = self._synthetic_bundle(root)
            authority["artifacts"]["execution"]["path_relative_to_work_root"] = (
                "logs/synthetic/other-run/certificates/execution.json"
            )
            authority["authority_sha256"] = sha256_json(
                {key: value for key, value in authority.items() if key != "authority_sha256"})
            with mock.patch.object(authority_module, "_ancestor", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "receipt/path"):
                    _verify_authority(REPO, root, authority)

    def test_resolved_split_substitution_is_rejected_against_loaded_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            authority, manifest, resolved, values, output = self._synthetic_bundle(root)
            manifest_raw = json.dumps(manifest, sort_keys=True).encode()
            authority["generation_authority"]["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
            values["p0"]["manifest_sha256"] = authority["generation_authority"]["manifest_sha256"]
            values["reference_length"]["manifest_sha256"] = authority["generation_authority"]["manifest_sha256"]
            blobs = {authority["generation_authority"]["manifest_path"]: manifest_raw}
            for name, receipt in authority["code_authority"].items():
                blobs[receipt["path"]] = f"historical:{name}".encode()
                receipt["sha256"] = hashlib.sha256(blobs[receipt["path"]]).hexdigest()
                if name == "scientific_contract":
                    values["p0"]["scientific_contract_sha256"] = receipt["sha256"]
                    values["reference_length"]["scientific_contract_sha256"] = receipt["sha256"]
                else:
                    values["p0"]["code_sha256"][receipt["path"]] = receipt["sha256"]
            authority = self._write_signed_bundle(root, authority, manifest, values, output)
            substituted = deepcopy(resolved)
            substituted["splits"]["reference_length_calibration"]["rows"][0][
                "content_root_id"
            ] = "f" * 64
            with mock.patch.object(authority_module, "_git_blob",
                                   side_effect=lambda _r, _c, path: blobs[path]), \
                 mock.patch.object(authority_module, "_ancestor", return_value=True), \
                 mock.patch.object(authority_module, "_verify_data_freeze_frozen",
                                   return_value=({}, substituted)):
                with self.assertRaisesRegex(RuntimeError, "do not reconstruct"):
                    _verify_authority(REPO, root, authority)

    def test_repo_paths_and_canonical_mapping_are_fixed(self):
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            _safe_repo_relative("../outside.py")
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            _safe_repo_relative("/tmp/outside.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            authority, _manifest, _resolved, _values, _output = self._synthetic_bundle(root)
            authority["artifacts"]["p0"]["canonical_field"] = "execution_sha256"
            authority["authority_sha256"] = sha256_json(
                {key: value for key, value in authority.items() if key != "authority_sha256"})
            with self.assertRaisesRegex(RuntimeError, "receipt/path|canonical mapping"):
                _verify_authority(REPO, root, authority)

    def test_artifact_paths_cannot_escape_work_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(_under(root, "logs/evidence.json"),
                             root / "logs/evidence.json")
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                _under(root, "../outside.json")
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                _under(root, "/tmp/outside.json")

    def test_reference_result_rejects_changed_value(self):
        result = {
            "statistic": "arithmetic_mean_sampled_policy_tokens_over_scheduled_slots",
            "lbar_ref": 3.0,
            "trajectory_count": 1,
            "scheduled_slot_count": 9,
            "active_writer_slot_count": 1,
            "writer_policy_tokens": 18,
            "answer_policy_tokens": 9,
            "total_policy_tokens": 27,
            "exact_token_match_count": 2,
        }
        _exact_result(result, result)
        changed = dict(result)
        changed["lbar_ref"] = 4.0
        with self.assertRaisesRegex(RuntimeError, "reference result"):
            _exact_result(changed, result)

    def test_shell_entry_is_fail_closed(self):
        source = ENTRY.read_text()
        self.assertIn("MEMAGENT_MIC_V2_EXPECTED_COMMIT", source)
        self.assertIn("status --porcelain", source)
        self.assertIn("mic_v2_reference_length_authority.py", source)
        self.assertIn("set -euo pipefail", source)

    def test_verifier_is_self_contained_from_mutable_producer_helpers(self):
        source = (REPO / "tools/h20/mic_v2_reference_length_authority.py").read_text()
        self.assertNotIn("from recurrent.research.mic_v2 import", source)
        self.assertNotIn("from tools.h20.mic_v2_reference_length_calibration import", source)
        self.assertIn("def sha256_file(", source)
        self.assertIn("def sha256_json(", source)
        self.assertIn("def _verify_data_freeze_frozen(", source)
        self.assertIn("def _label_blind_inputs_frozen(", source)

    def test_real_shell_entry_rejects_missing_environment(self):
        result = subprocess.run(
            ["bash", str(ENTRY)], text=True, capture_output=True,
            env={"PATH": os.environ.get("PATH", "")},
        )
        self.assertEqual(result.returncode, 40)
        self.assertIn("missing required environment", result.stderr)

    def test_real_shell_entry_rejects_wrong_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary).resolve()
            python = work / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(Path(sys.executable).resolve())
            result = subprocess.run(
                ["bash", str(ENTRY)], text=True, capture_output=True,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "MEMAGENT_MIC_V2_WORK_ROOT": str(work),
                    "MEMAGENT_MIC_V2_REPO_DIR": str(REPO),
                    "MEMAGENT_MIC_V2_EXPECTED_COMMIT": "0" * 40,
                },
            )
        self.assertEqual(result.returncode, 43)
        self.assertIn("exact Git commit mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
