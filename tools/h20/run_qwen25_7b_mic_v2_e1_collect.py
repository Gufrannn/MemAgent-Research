#!/usr/bin/env python3
"""Strict-vLLM, label-blind trajectory collector for MIC-v2 E1."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.mic_v2 import (  # noqa: E402
    MATERIALIZATION_PARSER_VERSION,
    canonical_json,
    materialized_memory_receipt,
    sha256_file,
    sha256_json,
    validate_boundary_pair,
)
from recurrent.research.serialization_credit_pilots import center_truncate_token_ids  # noqa: E402
from recurrent.research.trajectory_seeding import derive_turn_request_seeds  # noqa: E402
from tools.h20.mic_v2_e1_pipeline import (  # noqa: E402
    MANIFEST_REL, _source_firewall,
    _load,
    _verify_inherited_lock_authority,
    stable_e1_trajectory_seed,
)
from tools.h20.mic_v2_reference_length_calibration import _verify_model  # noqa: E402
from tools.h20.run_qwen25_7b_mic_v2_reference_length_calibration import (  # noqa: E402
    _append,
    _completion_receipt,
    _generate,
    _strict_vllm_environment,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MIC_V2_E1_NO_GO: {message}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True,
    ).strip()


def _ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows, previous = [], "0" * 64
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        digest = row.pop("record_sha256", None)
        _require(row.get("record_index") == index
                 and row.get("previous_record_sha256") == previous
                 and digest == sha256_json(row), "E1 trajectory hash chain differs")
        row["record_sha256"] = digest
        rows.append(row)
        previous = digest
    return rows


def _gpu_identity(pair: list[int]) -> list[dict[str, Any]]:
    selected = ",".join(str(item) for item in pair)
    query = subprocess.run(
        ["nvidia-smi", "-i", selected, "--query-gpu=index,uuid,name",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False,
    )
    _require(query.returncode == 0, "cannot authenticate E1 GPU pair")
    result = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        _require(len(fields) == 3 and fields[0].isdigit()
                 and fields[1].startswith("GPU-") and fields[2] == "NVIDIA H20",
                 "physical E1 GPU identity differs")
        result.append({"index": int(fields[0]), "uuid": fields[1], "name": fields[2]})
    _require([item["index"] for item in result] == pair
             and len({item["uuid"] for item in result}) == 2,
             "physical E1 GPU pair/order differs")
    return result


def _sources(p0: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(p0["gpu_source"]["path"])
    _require(path.is_file() and sha256_file(path) == p0["gpu_source"]["file_sha256"],
             "E1 GPU source artifact differs")
    rows = _read_jsonl(path)
    fields = {
        "source_position", "semantic_dataset_index", "question_sha256",
        "context_sha256", "content_root_id", "question", "context",
    }
    _require(len(rows) == 128 and all(set(row) == fields for row in rows),
             "E1 GPU source coverage/schema differs")
    for row in rows:
        _require(hashlib.sha256(row["question"].encode()).hexdigest() == row["question_sha256"]
                 and hashlib.sha256(row["context"].encode()).hexdigest() == row["context_sha256"],
                 "E1 GPU source text hash differs")
    return rows


def _boundary(
    *, phase: str, root: str, trajectory_id: str, turn: int, question: str,
    chunks: list[str], memories: list[str], arrived_tokens: int,
    chunk_schedule_id: str,
) -> dict[str, Any]:
    record = {
        "schema": "memagent.mic.v2", "phase": phase, "content_root_id": root,
        "stable_example_id": root, "trajectory_id": trajectory_id, "turn_index": turn,
        "question": question, "arrived_chunks": chunks,
        "materialized_memory_history": memories,
        "current_memory": memories[-1] if memories else "",
        "public_metadata": {
            "arrived_context_token_count": arrived_tokens,
            "chunk_schedule_id": chunk_schedule_id,
            "exogenous_termination": False, "forced_truncation": False,
            "policy_termination": False, "prior_active_turn_count": turn - 1,
        },
    }
    record["state_sha256"] = sha256_json(record)
    return record


def collect(repo: Path, expected_commit: str, output_root: Path, run_id: str, mode: str) -> dict[str, Any]:
    _require(sys.flags.optimize == 0 and mode in ("produce", "replay"),
             "E1 collector runtime mode differs")
    _require(repo.is_absolute() and output_root.is_absolute()
             and _git(repo, "rev-parse", "HEAD") == expected_commit
             and not _git(repo, "status", "--porcelain"),
             "E1 collector Git authority differs")
    p0 = _load(output_root / "certificates/p0.json")
    unsigned = dict(p0)
    p0_digest = unsigned.pop("p0_sha256", None)
    split = p0.get("split")
    _require(split in ("e1_dev", "e1_holdout")
             and p0_digest == sha256_json(unsigned)
             and p0.get("git_commit") == expected_commit and p0.get("run_id") == run_id
             and p0.get("output_root") == str(output_root),
             "E1 P0 identity differs")
    pair = p0["gpu_pair"]
    _require(_verify_inherited_lock_authority(run_id) == pair,
             "E1 collector inherited GPU locks differ")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(map(str, pair)),
             "E1 CUDA_VISIBLE_DEVICES differs")
    strict_environment = _strict_vllm_environment()
    physical = _gpu_identity(pair)
    manifest_path = repo / MANIFEST_REL
    manifest = _load(manifest_path)
    _require(sha256_file(manifest_path) == p0["manifest_sha256"], "E1 manifest differs")
    relative_self = str(Path(__file__).resolve().relative_to(repo))
    _require(p0["code_sha256"].get(relative_self) == sha256_file(__file__),
             "E1 GPU collector code differs")
    code_authority = _source_firewall(repo, manifest)
    model_authority = _verify_model(manifest)
    _require(code_authority == p0["code_sha256"]
             and model_authority == p0["model_files"],
             "E1 collector code/model authority differs")
    rows = _sources(p0)

    from transformers import AutoTokenizer
    import torch
    import vllm
    from vllm import LLM
    from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template

    _require(vllm.__version__ == manifest["model"]["required_vllm_version"]
             and not torch.cuda.is_initialized()
             and all(name in inspect.signature(LLM.generate).parameters
                     for name in ("prompt_token_ids", "sampling_params")),
             "E1 vLLM runtime differs")
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["path"], trust_remote_code=True, local_files_only=True,
    )
    generation = _load(Path(manifest["model"]["path"]) / "generation_config.json")
    terminators = generation["eos_token_id"]
    _require(terminators == manifest["backend"]["termination_token_ids"]
             and set(terminators) == {int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)},
             "E1 termination-token authority differs")
    _require(p0["tokenization_authority"]["parser_version"]
             == MATERIALIZATION_PARSER_VERSION
             == manifest["materialization"]["parser_version"],
             "E1 materialization parser authority differs")
    writer_template = TokenTemplate(chat_template(tokenizer).format(message=TEMPLATE), tokenizer)
    answer_template = TokenTemplate(
        chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED), tokenizer,
    )
    no_memory_ids = list(tokenizer.encode(
        manifest["recurrent"]["no_memory_text"], add_special_tokens=False,
    ))
    _require(p0["tokenization_authority"]["writer_template_sha256"]
             == hashlib.sha256(writer_template.template.encode("utf-8")).hexdigest()
             and p0["tokenization_authority"]["answer_template_sha256"]
             == hashlib.sha256(answer_template.template.encode("utf-8")).hexdigest()
             and p0["tokenization_authority"]["no_memory_token_ids_sha256"]
             == sha256_json(no_memory_ids),
             "E1 prompt-template authority differs")
    token_receipts = {row["content_root_id"]: row
                      for row in p0["tokenization_authority"]["receipts"]}
    prepared = []
    for row in rows:
        question_ids = list(tokenizer.encode(row["question"], add_special_tokens=False))
        context_ids = center_truncate_token_ids(
            list(tokenizer.encode(row["context"], add_special_tokens=False)),
            manifest["recurrent"]["max_context_tokens"],
        )
        size = manifest["recurrent"]["chunk_size"]
        chunks = [context_ids[offset:offset + size] for offset in range(0, len(context_ids), size)]
        receipt = token_receipts.get(row["content_root_id"])
        _require(receipt is not None and receipt["question_token_ids_sha256"] == sha256_json(question_ids)
                 and receipt["context_token_ids_sha256"] == sha256_json(context_ids)
                 and receipt["chunk_token_ids_sha256"] == [sha256_json(chunk) for chunk in chunks],
                 "E1 runtime tokenization differs")
        _require(receipt["question_token_count"] == len(question_ids)
                 and receipt["context_token_count"] == len(context_ids)
                 and receipt["active_writer_slots"] == len(chunks)
                 and 1 <= len(chunks) <= manifest["recurrent"]["max_writer_slots"],
                 "E1 runtime horizon/count receipt differs")
        prepared.append((row, question_ids, chunks, receipt))

    backend = manifest["backend"]
    llm = LLM(
        model=manifest["model"]["path"], tokenizer=manifest["model"]["path"],
        trust_remote_code=True, tensor_parallel_size=backend["tensor_parallel_size"],
        dtype=backend["dtype"], seed=backend["engine_seed"],
        gpu_memory_utilization=backend["gpu_memory_utilization"],
        swap_space=backend["swap_space_gib"], enforce_eager=backend["enforce_eager"],
        disable_custom_all_reduce=backend["disable_custom_all_reduce"],
        max_model_len=backend["max_model_len"],
        max_num_batched_tokens=backend["max_num_batched_tokens"],
        max_num_seqs=backend["max_num_seqs"],
        enable_prefix_caching=backend["enable_prefix_caching"],
    )
    ledger_path = output_root / f"trajectories/{split}.jsonl"
    existing = _ledger(ledger_path)
    expected_pairs = [(row["content_root_id"], replica) for row, *_rest in prepared
                      for replica in range(manifest["sampling"]["replicas"])]
    if mode == "produce":
        _require(not existing and not ledger_path.exists(),
                 "E1 production requires a fresh append-only attempt")
    else:
        _require(len(existing) == len(expected_pairs)
                 and [(row["content_root_id"], row["replica"]) for row in existing]
                     == expected_pairs,
                 "E1 replay requires the complete canonical ledger")
    generated, cursor = 0, 0
    for source, question_ids, chunk_ids, token_receipt in prepared:
        decoded_chunks = [tokenizer.decode(chunk, skip_special_tokens=False) for chunk in chunk_ids]
        schedule_id = sha256_json(token_receipt["chunk_token_ids_sha256"])
        for replica in range(manifest["sampling"]["replicas"]):
            trajectory_seed = stable_e1_trajectory_seed(
                manifest["sampling"]["base_seed"], source["content_root_id"], replica,
            )
            trajectory_id = sha256_json([
                "mic-v2-e1-trajectory", split, source["content_root_id"], replica,
            ])
            memory_ids, memory_texts, slots = no_memory_ids, [], []
            arrived_tokens = 0
            for turn_index, current_chunk in enumerate(chunk_ids, start=1):
                arrived_tokens += len(current_chunk)
                pre = _boundary(
                    phase="pre_write", root=source["content_root_id"], trajectory_id=trajectory_id,
                    turn=turn_index, question=source["question"],
                    chunks=decoded_chunks[:turn_index], memories=list(memory_texts),
                    arrived_tokens=arrived_tokens, chunk_schedule_id=schedule_id,
                )
                prompt = writer_template.format(
                    prompt=question_ids, memory=memory_ids, chunk=current_chunk,
                ).tolist()
                request_seed = derive_turn_request_seeds(
                    [trajectory_seed], [0], turn_index - 1,
                )[0]
                raw_ids, finish = _generate(llm, prompt, manifest["sampling"]["writer"], request_seed)
                generated += 1
                completion = _completion_receipt(
                    raw_ids, finish, termination_token_ids=terminators,
                    maximum=manifest["sampling"]["writer"]["max_tokens"],
                )
                memory_ids, materialization = materialized_memory_receipt(
                    token_ids=raw_ids, termination_token_ids=terminators,
                    content_root_id=source["content_root_id"], trajectory_seed=trajectory_seed,
                    turn_index=turn_index - 1,
                    arrived_chunk_token_sha256=token_receipt["chunk_token_ids_sha256"][:turn_index],
                    prior_memory_token_sha256=[slot["materialization"]["parsed_memory_sha256"]
                                               for slot in slots],
                )
                memory_texts.append(tokenizer.decode(memory_ids, skip_special_tokens=False))
                post = _boundary(
                    phase="post_write", root=source["content_root_id"], trajectory_id=trajectory_id,
                    turn=turn_index, question=source["question"],
                    chunks=decoded_chunks[:turn_index], memories=list(memory_texts),
                    arrived_tokens=arrived_tokens, chunk_schedule_id=schedule_id,
                )
                validate_boundary_pair(pre, post)
                slots.append({
                    "turn": turn_index, "request_seed": request_seed,
                    "pre_state": pre, "post_state": post,
                    "completion": completion, "materialization": materialization,
                })
            answer_seed = derive_turn_request_seeds(
                [trajectory_seed], [0], manifest["recurrent"]["max_writer_slots"],
            )[0]
            answer_prompt = answer_template.format(prompt=question_ids, memory=memory_ids).tolist()
            answer_ids, answer_finish = _generate(
                llm, answer_prompt, manifest["sampling"]["answer"], answer_seed,
            )
            generated += 1
            payload = {
                "schema": "memagent.mic.v2.e1-trajectory", "split": split,
                "git_commit": expected_commit,
                "run_id": run_id, "p0_sha256": p0_digest,
                "content_root_id": source["content_root_id"], "replica": replica,
                "trajectory_id": trajectory_id, "trajectory_seed": trajectory_seed,
                "active_writer_slots": len(slots), "writer_slots": slots,
                "answer_request_seed": answer_seed,
                "answer_completion": _completion_receipt(
                    answer_ids, answer_finish, termination_token_ids=terminators,
                    maximum=manifest["sampling"]["answer"]["max_tokens"],
                ),
                "terminal_text": tokenizer.decode(answer_ids, skip_special_tokens=False),
            }
            if mode == "produce":
                _append(ledger_path, payload)
            else:
                authenticated = dict(existing[cursor])
                for key in ("record_index", "previous_record_sha256", "record_sha256"):
                    authenticated.pop(key)
                _require(authenticated == payload, "E1 independent replay differs")
            cursor += 1
            print(canonical_json({"completed": cursor, "total": len(expected_pairs),
                                  "content_root_id": source["content_root_id"],
                                  "replica": replica}), flush=True)
    _require(generated == p0["seed_authority"]["active_request_count"]
             and cursor == len(expected_pairs),
             "E1 generated-call accounting differs")
    evidence_name = "replay" if mode == "replay" else "execution"
    receipt = {
        "schema": f"memagent.mic.v2.e1-{evidence_name}",
        "status": "PASS", "decision": f"MIC_V2_E1_{evidence_name.upper()}_PASS",
        "git_commit": expected_commit, "run_id": run_id, "split": split,
        "p0_sha256": p0_digest,
        "gpu_pair": pair, "physical_gpu_identity": physical,
        "vllm_version": vllm.__version__, "config_loader_environment": strict_environment,
        "trajectory_count": len(expected_pairs),
        "represented_generate_calls": p0["seed_authority"]["active_request_count"],
        "generate_calls_this_process": generated,
        "ledger_file_sha256": sha256_file(ledger_path),
        "code_authority_sha256": sha256_json(code_authority),
        "model_authority_sha256": sha256_json(model_authority),
    }
    digest_field = f"{evidence_name}_sha256"
    receipt[digest_field] = sha256_json(receipt)
    path = output_root / f"certificates/{split}_{evidence_name}.json"
    if path.exists():
        _require(_load(path) == receipt, "existing E1 execution receipt differs")
    else:
        from recurrent.research.mic_v2 import write_json_new
        write_json_new(path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("produce", "replay"), required=True)
    args = parser.parse_args()
    print(canonical_json(collect(args.repo.resolve(), args.expected_commit,
                                 args.output_root.resolve(), args.run_id, args.mode)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
