#!/usr/bin/env python3
"""No-gradient, time-safe actor-hidden feature extraction for MIC-v2 E1-dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.mic_v2 import canonical_json, sha256_file, sha256_json, write_json_new  # noqa: E402
from recurrent.research.mic_v2_e1 import (  # noqa: E402
    ACTOR_COMPONENT_DIMENSION,
    actor_hidden_interactions,
    rademacher_matrix,
    rademacher_projection,
    text_components_from_state,
)
from tools.h20.mic_v2_e1_pipeline import (  # noqa: E402
    MANIFEST_REL, _load, _source_firewall, _verify_inherited_lock_authority,
)
from tools.h20.mic_v2_reference_length_calibration import _verify_model  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MIC_V2_E1_NO_GO: {message}")


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"state artifact is missing: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _physical_pair(pair: list[int]) -> list[dict[str, Any]]:
    query = subprocess.run(
        ["nvidia-smi", "-i", ",".join(map(str, pair)),
         "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    _require(query.returncode == 0, "cannot authenticate feature GPU pair")
    result = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        _require(len(fields) == 3 and fields[0].isdigit()
                 and fields[1].startswith("GPU-") and fields[2] == "NVIDIA H20",
                 "feature GPU identity differs")
        result.append({"index": int(fields[0]), "uuid": fields[1], "name": fields[2]})
    _require([row["index"] for row in result] == pair
             and len({row["uuid"] for row in result}) == 2,
             "feature GPU pair/order differs")
    return result


def _encode_components(
    texts: list[str], tokenizer: Any, model: Any, *, probe_token_id: int,
    token_budget: int, maximum_batch: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    unique = sorted(set(texts))
    tokenized = {}
    for text in unique:
        ids = list(tokenizer.encode(text, add_special_tokens=False)) + [probe_token_id]
        _require(1 <= len(ids) <= token_budget, "actor-hidden component exceeds token budget")
        tokenized[text] = ids
    ordered = sorted(unique, key=lambda text: (len(tokenized[text]), text))
    result: dict[str, Any] = {}
    cursor = 0
    with torch.inference_mode():
        while cursor < len(ordered):
            batch, longest = [], 0
            while cursor < len(ordered) and len(batch) < maximum_batch:
                candidate = ordered[cursor]
                proposed_longest = max(longest, len(tokenized[candidate]))
                if batch and proposed_longest * (len(batch) + 1) > token_budget:
                    break
                batch.append(candidate)
                longest = proposed_longest
                cursor += 1
            _require(batch, "actor-hidden batching made no progress")
            padded, masks = [], []
            for text in batch:
                ids = tokenized[text]
                padding = longest - len(ids)
                padded.append(ids + [int(tokenizer.pad_token_id)] * padding)
                masks.append([1] * len(ids) + [0] * padding)
            input_ids = torch.tensor(padded, dtype=torch.long, device=model.device)
            attention = torch.tensor(masks, dtype=torch.long, device=model.device)
            output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
            hidden = output.last_hidden_state
            _require(hidden.ndim == 3 and hidden.shape[:2] == input_ids.shape,
                     "actor-hidden output shape differs")
            for row, text in enumerate(batch):
                vector = hidden[row, len(tokenized[text]) - 1].float().cpu().numpy()
                _require(vector.ndim == 1 and np.isfinite(vector).all(),
                         "actor-hidden component is non-finite")
                result[text] = vector
    _require(len(result) == len(unique), "actor-hidden component coverage differs")
    return result


def extract(
    repo: Path, expected_commit: str, output_root: Path, run_id: str, split: str,
    mode: str,
) -> dict[str, Any]:
    _require(split in ("e1_dev", "e1_holdout") and mode in ("produce", "replay")
             and sys.flags.optimize == 0 and repo.is_absolute() and output_root.is_absolute(),
             "feature runtime differs")
    _require(_git(repo, "rev-parse", "HEAD") == expected_commit
             and not _git(repo, "status", "--porcelain"),
             "feature Git authority differs")
    p0 = _load(output_root / "certificates/p0.json")
    unsigned_p0 = dict(p0)
    p0_sha = unsigned_p0.pop("p0_sha256", None)
    _require(p0_sha == sha256_json(unsigned_p0)
             and p0.get("git_commit") == expected_commit
             and p0.get("run_id") == run_id
             and p0.get("output_root") == str(output_root)
             and p0.get("split") == split, "feature P0 differs")
    manifest_path = repo / MANIFEST_REL
    manifest = _load(manifest_path)
    _require(sha256_file(manifest_path) == p0["manifest_sha256"], "feature manifest differs")
    relative_self = str(Path(__file__).resolve().relative_to(repo))
    _require(p0["code_sha256"].get(relative_self) == sha256_file(__file__),
             "feature extractor code differs")
    code_authority = _source_firewall(repo, manifest)
    model_authority = _verify_model(manifest)
    _require(code_authority == p0["code_sha256"]
             and model_authority == p0["model_files"],
             "feature code/model authority differs")
    states_receipt = _load(output_root / f"certificates/{split}_states.json")
    unsigned_states = dict(states_receipt)
    states_sha = unsigned_states.pop("states_sha256", None)
    state_path = Path(states_receipt.get("state_path", ""))
    _require(states_sha == sha256_json(unsigned_states)
             and states_receipt.get("schema") == "memagent.mic.v2.e1-time-safe-states"
             and states_receipt.get("status") == "PASS"
             and states_receipt.get("split") == split
             and states_receipt.get("git_commit") == expected_commit
             and states_receipt.get("run_id") == run_id
             and states_receipt.get("p0_sha256") == p0_sha
             and states_receipt.get("time_safe_field_firewall_pass") is True
             and state_path.is_file()
             and sha256_file(state_path) == states_receipt.get("state_file_sha256"),
             "time-safe state authority differs")
    states = _jsonl(state_path)
    _require(len(states) == states_receipt["state_row_count"], "time-safe state rows differ")

    pair = p0["gpu_pair"]
    _require(_verify_inherited_lock_authority(run_id) == pair,
             "feature extractor inherited GPU locks differ")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(map(str, pair)),
             "feature CUDA_VISIBLE_DEVICES differs")
    physical = _physical_pair(pair)
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    feature_spec = manifest["actor_hidden_features"]
    _require(transformers.__version__ == feature_spec["required_transformers_version"]
             and torch.__version__ == feature_spec["required_torch_version"]
             and not torch.cuda.is_initialized(), "feature runtime version/init differs")
    _require(feature_spec["component_projection_dimension"] == ACTOR_COMPONENT_DIMENSION
             and feature_spec["final_projection_dimensions"] == [128, 256],
             "feature projection preregistration differs")
    _require(feature_spec["history_aggregation"] == (
        "softmax cosine(question, each arrived chunk) with temperature 1; "
        "the turn-0 empty history is the all-zero hidden vector"
    ), "turn-0 history representation differs")
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["path"], trust_remote_code=True, local_files_only=True,
    )
    tokenizer.padding_side = "right"
    probe_token_id = int(tokenizer.eos_token_id)
    _require(probe_token_id == feature_spec["probe_token_id"], "feature probe token differs")
    model = AutoModel.from_pretrained(
        manifest["model"]["path"], trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).eval().requires_grad_(False).to("cuda:0")
    _require(not any(parameter.requires_grad for parameter in model.parameters()),
             "feature model parameters unexpectedly require gradients")

    components = []
    normalized_states = []
    for row in states:
        available_stages = row.get("available_stages")
        _require(available_stages in (["post"], ["pre", "post"])
                 and (row.get("turn") == 0) == (available_stages == ["post"]),
                 "actor-hidden stage availability differs")
        stages = {}
        for stage in available_stages:
            state = row[f"{stage}_state"]
            text = text_components_from_state(
                state, no_memory_text=manifest["recurrent"]["no_memory_text"],
            )
            components.extend([text["question"], text["current_memory"], *state["arrived_chunks"]])
            stages[stage] = (state, text)
        normalized_states.append((row, stages))
    hidden = _encode_components(
        components, tokenizer, model, probe_token_id=probe_token_id,
        token_budget=feature_spec["component_batch_token_budget"],
        maximum_batch=feature_spec["maximum_component_batch_size"],
    )
    del model
    torch.cuda.empty_cache()
    hidden_dimension = len(next(iter(hidden.values())))
    _require(hidden_dimension == feature_spec["hidden_dimension"],
             "feature hidden dimension differs")
    projection_hashes = {
        "component": hashlib.sha256(rademacher_matrix(
            hidden_dimension, ACTOR_COMPONENT_DIMENSION,
            "actor-hidden-component-rademacher",
        ).tobytes()).hexdigest(),
        "final_128": hashlib.sha256(rademacher_matrix(
            7 * ACTOR_COMPONENT_DIMENSION + 1, 128,
        ).tobytes()).hexdigest(),
        "final_256": hashlib.sha256(rademacher_matrix(
            7 * ACTOR_COMPONENT_DIMENSION + 1, 256,
        ).tobytes()).hexdigest(),
    }
    _require(projection_hashes == feature_spec["projection_matrix_sha256"],
             "feature projection matrix authority differs")

    feature_rows = []
    for row, stages in normalized_states:
        output = {
            "content_root_id": row["content_root_id"], "replica": row["replica"],
            "trajectory_id": row["trajectory_id"], "turn": row["turn"],
            "available_stages": row["available_stages"],
        }
        for stage in row["available_stages"]:
            state, text = stages[stage]
            interactions = actor_hidden_interactions(
                hidden[text["question"]], [hidden[chunk] for chunk in state["arrived_chunks"]],
                hidden[text["current_memory"]],
            )
            output[f"{stage}_actor_hidden_rademacher_128"] = rademacher_projection(
                interactions, 128, turn=row["turn"],
            ).astype(np.float32).tolist()
            output[f"{stage}_actor_hidden_rademacher_256"] = rademacher_projection(
                interactions, 256, turn=row["turn"],
            ).astype(np.float32).tolist()
        feature_rows.append(output)
    feature_path = output_root / f"features/{split}_actor_hidden.jsonl"
    if mode == "produce":
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        with feature_path.open("x", encoding="utf-8") as stream:
            for row in feature_rows:
                stream.write(canonical_json(row) + "\n")
    else:
        _require(_jsonl(feature_path) == feature_rows,
                 "independent actor-hidden feature replay differs")
    evidence_name = "actor-hidden-features" if mode == "produce" else "actor-hidden-features-replay"
    receipt = {
        "schema": f"memagent.mic.v2.e1-{evidence_name}", "status": "PASS",
        "decision": ("MIC_V2_E1_ACTOR_HIDDEN_FEATURES_PASS" if mode == "produce"
                     else "MIC_V2_E1_ACTOR_HIDDEN_FEATURES_REPLAY_PASS"),
        "split": split,
        "git_commit": expected_commit, "run_id": run_id, "p0_sha256": p0_sha,
        "states_sha256": states_sha, "state_row_count": len(states),
        "unique_component_count": len(hidden), "probe_token_id": probe_token_id,
        "hidden_dimension": hidden_dimension, "projection_matrix_sha256": projection_hashes,
        "physical_gpu_identity": physical, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "feature_path": str(feature_path), "feature_file_sha256": sha256_file(feature_path),
        "feature_canonical_sha256": sha256_json(feature_rows),
        "representation_gradients": "disabled", "generation_hidden_or_kv_reused": False,
        "independent_exact_replay": mode == "replay",
        "code_authority_sha256": sha256_json(code_authority),
        "model_authority_sha256": sha256_json(model_authority),
    }
    receipt["features_sha256"] = sha256_json(receipt)
    receipt_path = output_root / f"certificates/{split}_{evidence_name.replace('-', '_')}.json"
    if receipt_path.exists():
        _require(mode == "replay" and _load(receipt_path) == receipt,
                 "existing actor-hidden feature replay receipt differs")
    else:
        write_json_new(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("e1_dev", "e1_holdout"), required=True)
    parser.add_argument("--mode", choices=("produce", "replay"), required=True)
    args = parser.parse_args()
    print(canonical_json(extract(
        args.repo.resolve(), args.expected_commit, args.output_root.resolve(), args.run_id,
        args.split, args.mode,
    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
