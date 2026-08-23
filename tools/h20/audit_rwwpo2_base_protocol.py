#!/usr/bin/env python3
"""One-time read-only base/model/Original-protocol audit for RWWPO-2."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.h20.preflight_qwen25_7b_original_t25 import model_loading_relevant_paths


EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"
EXPECTED_TRAIN_SHA = "798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-resolved", required=True)
    parser.add_argument("--original-resolved-sha256", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_BASE_PROTOCOL_NO_GO:checkout")
    raw_inputs = tuple(Path(value) for value in (
        args.original_resolved, args.model_root, args.train,
    ))
    if any(path.is_symlink() for path in raw_inputs):
        raise SystemExit("RWWPO2_BASE_PROTOCOL_NO_GO:symlink")
    original_path, model_root, train_path = (path.resolve() for path in raw_inputs)
    if sha256_file(original_path) != args.original_resolved_sha256:
        raise SystemExit("RWWPO2_BASE_PROTOCOL_NO_GO:Original resolved SHA")
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA:
        raise SystemExit("RWWPO2_BASE_PROTOCOL_NO_GO:train SHA")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    training = original.get("training")
    model = original.get("model")
    data = original.get("data")
    backend = original.get("backend")
    contract = original.get("contract")
    if not all(isinstance(value, dict) for value in (
            training, model, data, backend, contract)):
        raise SystemExit("RWWPO2_BASE_PROTOCOL_NO_GO:Original projection missing")
    expected_training = {
        "train_batch_size": 4, "rollout_n": 2, "ppo_mini_batch_size": 4,
        "chunk_size": 5000, "max_chunks": 8, "max_prompt_length": 8192,
        "max_response_length": 1024, "ppo_max_token_len_per_gpu": 16384,
        "log_prob_max_token_len_per_gpu": 32768, "max_num_batched_tokens": 16384,
        "max_num_seqs": 16, "gpu_memory_utilization": 0.55,
        "actor_learning_rate": 1e-6, "actor_lr_warmup_steps": 2,
        "clip_ratio_high": 0.2, "entropy_coefficient": 0.0,
        "kl_loss_coefficient": 0.001, "trajectory_seed_mode": "independent",
    }
    training_drift = {
        key: [training.get(key), expected]
        for key, expected in expected_training.items() if training.get(key) != expected
    }
    expected_data = {
        "train_sha256": EXPECTED_TRAIN_SHA, "shuffle": False,
        "dataloader_num_workers": 0, "filter_overlong_prompts": True,
        "production_effective_prompt_limit": 40000,
    }
    data_drift = {
        key: [data.get(key), expected]
        for key, expected in expected_data.items() if data.get(key) != expected
    }
    expected_backend = {
        "rollout": "vllm", "allow_hf_fallback": False, "reward_manager": "naive",
    }
    backend_drift = {
        key: [backend.get(key), expected]
        for key, expected in expected_backend.items() if backend.get(key) != expected
    }
    if training_drift or data_drift or backend_drift \
            or int(contract.get("world_size", -1)) != 2 \
            or model.get("id") != "Qwen/Qwen2.5-7B-Instruct" \
            or model.get("revision") != "a09a35458c702b33eeacc393d103063234e8bc28":
        raise SystemExit("RWWPO2_BASE_PROTOCOL_NO_GO:Original protocol drift:" +
                         json.dumps({"training": training_drift, "data": data_drift,
                                     "backend": backend_drift}, sort_keys=True))
    expected_files = sorted(str(item["path"]) for item in model.get("files", []))
    if model_loading_relevant_paths(model_root) != expected_files:
        raise SystemExit("RWWPO2_BASE_PROTOCOL_NO_GO:model loading inventory")
    actual_files = []
    for item in sorted(model["files"], key=lambda row: str(row["path"])):
        path = model_root / str(item["path"])
        if path.is_symlink() or not path.is_file() \
                or path.stat().st_size != int(item["size"]) \
                or sha256_file(path) != item["sha256"]:
            raise SystemExit("RWWPO2_BASE_PROTOCOL_NO_GO:model byte identity")
        actual_files.append({
            "path": str(item["path"]), "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    report = {
        "schema_version": "rwwpo2-base-protocol-audit-v1",
        "status": "PASS", "decision": "RWWPO2_BASE_PROTOCOL_AUDIT_PASS",
        "git_commit": head,
        "original_resolved_path": str(original_path),
        "original_resolved_sha256": args.original_resolved_sha256,
        "model_root": str(model_root), "model_id": model["id"],
        "model_revision": model["revision"], "model_files": actual_files,
        "train_path": str(train_path), "train_sha256": EXPECTED_TRAIN_SHA,
        "frozen_common_protocol": {
            "training": expected_training, "data": expected_data,
            "backend": expected_backend, "world_size": 2,
        },
        "preregistered_method_only_differences": [
            "fresh-base update-1 start instead of historical step3 continuation",
            "experiment seed assignment 2026..2033",
            "K2 actor transactions/ppo_epochs=2 and stateless proposal clock",
            "writer objective/controller/actual-loss evidence",
            "target round/checkpoint retention/experiment identity/output paths",
        ],
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "output": str(output.resolve()), "model_file_count": len(actual_files),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
