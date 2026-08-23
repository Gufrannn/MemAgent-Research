import json
import hashlib

import pytest

from tools.h20.audit_mic_scientific_evidence import (
    _intersection, _row_identity, _validate_chain, _validate_seed_producer_bindings,
)
from recurrent.research.mic import sha256_json


def parquet_row(index, question="q", context="c", answer=None):
    return {
        "prompt": [{"role": "user", "content": question}],
        "context": context,
        "reward_model": {"ground_truth": ["a"] if answer is None else answer},
        "extra_info": {"index": index},
    }


def test_content_intersection_ignores_filename_and_semantic_index():
    train = [_row_identity(parquet_row(11, question="same", context="same context"))]
    s128 = [_row_identity(parquet_row(9001, question="same", context="same context"))]
    root = _intersection(train, s128, "content_root_sha256")
    example = _intersection(train, s128, "content_example_sha256")
    assert root["intersection_count"] == 1
    assert example["intersection_count"] == 1


def test_root_overlap_distinguishes_ground_truth_drift():
    train = [_row_identity(parquet_row(1, answer=["one"]))]
    s128 = [_row_identity(parquet_row(2, answer=["two"]))]
    assert _intersection(train, s128, "content_root_sha256")["intersection_count"] == 1
    assert _intersection(train, s128, "content_example_sha256")["intersection_count"] == 0


def test_scientific_audit_rejects_tampered_training_ledger_chain():
    unsigned = {
        "sequence": 0, "previous_entry_sha256": "0" * 64,
        "record_type": "mic_advantage_delivery",
    }
    valid = {**unsigned, "entry_sha256": sha256_json(unsigned)}
    assert _validate_chain([valid])[0]["record_type"] == "mic_advantage_delivery"
    tampered = json.loads(json.dumps(valid))
    tampered["record_type"] = "tampered"
    with pytest.raises(ValueError, match="chain is corrupt"):
        _validate_chain([tampered])


def test_seed_dataset_index_substitution_breaks_producer_uid_binding():
    payload = {
        "namespace": "memagent-mic-prompt-group-v1",
        "global_step": 1,
        "dataset_index": 17,
    }
    uid = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    row = {"global_step": 1, "dataset_index": 17, "base_seed": 2026, "uid": uid}
    _validate_seed_producer_bindings([row])
    row["dataset_index"] = 18
    with pytest.raises(ValueError, match="producer contract"):
        _validate_seed_producer_bindings([row])


def test_adaptive_disclosure_is_explicit_and_disallows_blind_test():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    disclosure = json.loads((
        repo / "docs/papers/mic_adaptive_use_disclosure_20260823.json"
    ).read_text())
    assert disclosure["s128_rows_exposed"] == 128
    assert disclosure["classification"] == "ADAPTIVE_DEVELOPMENT_BENCHMARK"
    assert "blind held-out final test" in disclosure["forbidden_claims"]


def test_real_entry_is_read_only_and_content_hash_based():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    entry = (repo / "scripts/h20/audit_scientific_evidence_qwen25_7b_mic.sh").read_text()
    source = (repo / "tools/h20/audit_mic_scientific_evidence.py").read_text()
    assert "mic_require_checkout" in entry
    assert "nvidia-smi" not in entry
    assert "content_root_sha256" in source and "content_example_sha256" in source
    assert "audit_seeds(" in source and "replay_mic_training_audit" in source
    assert "refusing to overwrite" in source
