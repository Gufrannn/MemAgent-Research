#!/usr/bin/env python3
"""Strict-vLLM, label-blind producer for the MIC-v2 reference length."""

from __future__ import annotations

import argparse
import fcntl
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
    sampled_policy_mask_receipt,
    sha256_file,
    sha256_json,
    write_json_new,
)
from recurrent.research.serialization_credit_pilots import center_truncate_token_ids  # noqa: E402
from recurrent.research.trajectory_seeding import derive_turn_request_seeds  # noqa: E402
from tools.h20.mic_v2_reference_length_calibration import (  # noqa: E402
    MANIFEST_REL,
    _load,
    _read_ledger,
    _self_digest,
    _validate_ledger,
    stable_trajectory_seed,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MIC_V2_NO_GO: {message}")


def _stable_seed(base_seed: int, content_root_id: str, replica: int) -> int:
    return stable_trajectory_seed(base_seed, content_root_id, replica)


def _append(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        lines = [line for line in stream.read().splitlines() if line.strip()]
        previous = "0" * 64
        for index, line in enumerate(lines):
            row = json.loads(line)
            digest = row.pop("record_sha256", None)
            _require(row.get("record_index") == index
                     and row.get("previous_record_sha256") == previous
                     and digest == sha256_json(row), "existing append-only ledger differs")
            previous = digest
        record = {
            **dict(payload),
            "record_index": len(lines),
            "previous_record_sha256": previous,
        }
        record["record_sha256"] = sha256_json(record)
        stream.seek(0, os.SEEK_END)
        stream.write(canonical_json(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        return record


def _generate(
    llm: Any, prompt_ids: list[int], spec: Mapping[str, Any], seed: int,
) -> tuple[list[int], str]:
    from vllm import SamplingParams

    params = SamplingParams(**{**dict(spec), "seed": int(seed)})
    outputs = llm.generate(
        prompt_token_ids=[prompt_ids], sampling_params=[params], use_tqdm=False,
    )
    _require(len(outputs) == 1 and len(outputs[0].outputs) == 1,
             "strict single-request generation shape differs")
    _require(list(outputs[0].prompt_token_ids) == prompt_ids,
             "strict vLLM prompt token IDs differ")
    candidate = outputs[0].outputs[0]
    finish_reason = str(candidate.finish_reason)
    _require(finish_reason in ("stop", "length"),
             "strict vLLM finish reason differs")
    return list(candidate.token_ids), finish_reason


def _completion_receipt(
    raw_ids: list[int], finish_reason: str, *, termination_token_ids: list[int],
    maximum: int,
) -> dict[str, Any]:
    _require(raw_ids and all(type(item) is int and item >= 0 for item in raw_ids),
             "active generation emitted no valid policy tokens")
    if finish_reason == "stop":
        _require(raw_ids[-1] in termination_token_ids
                 and not any(token in termination_token_ids for token in raw_ids[:-1]),
                 "stop finish has no sampled terminal token")
        termination = "sampled_eos"
    else:
        _require(finish_reason == "length" and len(raw_ids) == maximum
                 and not any(token in termination_token_ids for token in raw_ids),
                 "length finish does not match forced truncation")
        termination = "forced_truncation"
    return {
        "sampled_token_ids": raw_ids,
        "sampled_policy_tokens": len(raw_ids),
        "completion_sha256": sha256_json(raw_ids),
        "termination": termination,
        "sampled_eos_counted": termination == "sampled_eos",
        "sampled_terminal_token_id": raw_ids[-1] if termination == "sampled_eos" else None,
        **sampled_policy_mask_receipt(
            token_ids=raw_ids,
            termination=termination,
            termination_token_ids=termination_token_ids,
            token_width=maximum,
        ),
    }


def _source_rows(manifest: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifact_path = Path(inputs["source_artifact_path"])
    source_rows = [
        json.loads(line)
        for line in artifact_path.read_text(encoding="utf-8").splitlines()
    ]
    _require(len(source_rows) == len(inputs["rows"]),
             "GPU source artifact coverage differs")
    expected_fields = set(manifest["source"]["gpu_input_fields"])
    rows = []
    for source, frozen in zip(source_rows, inputs["rows"]):
        _require(set(source) == expected_fields, "GPU source artifact schema differs")
        question = source["question"]
        context = source["context"]
        _require({key: source[key] for key in frozen} == frozen,
                 "GPU source identity differs")
        _require(hashlib.sha256(question.encode("utf-8")).hexdigest()
                 == frozen["question_sha256"], "question authority differs")
        _require(hashlib.sha256(context.encode("utf-8")).hexdigest()
                 == frozen["context_sha256"], "context authority differs")
        rows.append({"frozen": frozen, "question": question, "context": context})
    return rows


def _strict_vllm_environment() -> dict[str, str]:
    expected = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VLLM_USE_V1": "0",
        "VLLM_USE_MODELSCOPE": "False",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    _require(all(os.environ.get(key) == value for key, value in expected.items()),
             "strict vLLM environment differs")
    return expected


def run(
    repo: Path, expected_commit: str, output_root: Path, run_id: str, mode: str,
) -> dict[str, Any]:
    _require(sys.flags.optimize == 0, "optimized Python is forbidden")
    _require(mode in ("produce", "replay"), "unknown GPU execution mode")
    p0 = _load(output_root / "certificates/p0.json")
    inputs = _load(output_root / "authorities/label_blind_inputs.json")
    _self_digest(p0, "p0_sha256")
    _self_digest(inputs, "inputs_sha256")
    _require(p0.get("git_commit") == expected_commit and p0.get("run_id") == run_id
             and p0.get("output_root") == str(output_root), "P0 identity differs")
    pair = ",".join(str(item) for item in p0["gpu_pair"])
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == pair,
             "CUDA_VISIBLE_DEVICES differs from authenticated GPU pair")
    strict_environment = _strict_vllm_environment()
    source_artifact = p0.get("label_blind_source", {})
    source_artifact_path = output_root / "authorities/label_blind_source.jsonl"
    _require(source_artifact.get("path") == str(source_artifact_path)
             and source_artifact.get("file_sha256") == sha256_file(source_artifact_path)
             and source_artifact.get("rows") == 64,
             "label-blind GPU source artifact differs")
    inputs = dict(inputs)
    inputs["source_artifact_path"] = str(source_artifact_path)
    gpu_query = subprocess.run(
        ["nvidia-smi", "-i", pair, "--query-gpu=index,uuid,name",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    _require(gpu_query.returncode == 0, "cannot authenticate selected GPU pair")
    physical_gpu_identity = []
    for line in gpu_query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        _require(len(fields) == 3 and fields[0].isdigit()
                 and fields[1].startswith("GPU-") and len(fields[1]) > 4
                 and fields[2] == "NVIDIA H20", "physical GPU identity differs")
        physical_gpu_identity.append({
            "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
        })
    _require([item["index"] for item in physical_gpu_identity] == p0["gpu_pair"]
             and len({item["uuid"] for item in physical_gpu_identity}) == 2,
             "physical GPU identity/order differs")
    manifest = _load(repo / MANIFEST_REL)
    _require(sha256_file(repo / MANIFEST_REL) == p0["manifest_sha256"],
             "calibration manifest SHA differs")
    _require(manifest["model"].get("config_loader_environment")
             == {"VLLM_USE_MODELSCOPE": strict_environment["VLLM_USE_MODELSCOPE"]},
             "model config-loader authority differs")
    _require(sha256_file(repo / Path(__file__).resolve().relative_to(repo))
             == p0["code_sha256"][str(Path(__file__).resolve().relative_to(repo))],
             "GPU runner code SHA differs")
    execution_path = output_root / "certificates/execution.json"
    replay_path = output_root / "certificates/gpu_replay.json"
    if mode == "produce":
        _require(not execution_path.exists(), "completed calibration execution already exists")
    else:
        _require(execution_path.is_file(),
                 "GPU replay requires one completed execution")

    from transformers import AutoTokenizer
    import torch
    import vllm
    from vllm import LLM
    from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template

    _require(vllm.__version__ == manifest["model"]["required_vllm_version"],
             "vLLM version differs")
    _require(not torch.cuda.is_initialized(), "parent CUDA was initialized before vLLM spawn")
    _require(all(name in inspect.signature(LLM.generate).parameters
                 for name in ("prompt_token_ids", "sampling_params")),
             "vLLM generation API differs")
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["path"], trust_remote_code=True, local_files_only=True,
    )
    generation_config = _load(Path(manifest["model"]["path"]) / "generation_config.json")
    termination_token_ids = generation_config.get("eos_token_id")
    _require(termination_token_ids == manifest["backend"]["termination_token_ids"]
             and set(termination_token_ids)
                 == {int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)},
             "fresh-base termination token authority differs")
    writer_template_text = chat_template(tokenizer).format(message=TEMPLATE)
    answer_template_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
    tokenization_authority = p0["tokenization_authority"]
    _require(hashlib.sha256(writer_template_text.encode("utf-8")).hexdigest()
             == tokenization_authority["writer_template_sha256"]
             and hashlib.sha256(answer_template_text.encode("utf-8")).hexdigest()
                 == tokenization_authority["answer_template_sha256"],
             "recurrent prompt template authority differs")
    _require(p0["materialization_authority"]["parser_version"]
             == MATERIALIZATION_PARSER_VERSION
             == manifest["materialization"]["parser_version"],
             "materialization parser authority differs")
    _require(p0["seed_authority"]["trajectory_count"]
             == p0["expected_trajectories"]
             and p0["seed_authority"]["all_trajectory_seeds_unique"] is True
             and p0["seed_authority"]["all_active_request_seeds_unique"] is True
             and p0["seed_authority"]["trajectory_request_namespaces_disjoint"] is True,
             "seed schedule authority differs")
    writer_template = TokenTemplate(writer_template_text, tokenizer)
    answer_template = TokenTemplate(answer_template_text, tokenizer)
    no_memory_ids = list(tokenizer.encode(
        manifest["recurrent"]["no_memory_text"], add_special_tokens=False,
    ))
    sources = _source_rows(manifest, inputs)
    prepared = []
    token_receipts = {
        row["content_root_id"]: row for row in tokenization_authority["receipts"]
    }
    for source in sources:
        question_ids = list(tokenizer.encode(source["question"], add_special_tokens=False))
        _require(len(question_ids) <= manifest["recurrent"]["max_question_tokens"],
                 "question exceeds frozen policy prompt budget")
        context_ids = center_truncate_token_ids(
            list(tokenizer.encode(source["context"], add_special_tokens=False)),
            manifest["recurrent"]["max_context_tokens"],
        )
        chunk_size = manifest["recurrent"]["chunk_size"]
        chunks = [context_ids[offset:offset + chunk_size]
                  for offset in range(0, len(context_ids), chunk_size)]
        _require(1 <= len(chunks) <= manifest["recurrent"]["max_writer_slots"],
                 "writer horizon differs from frozen slot program")
        expected_tokens = token_receipts.get(source["frozen"]["content_root_id"])
        _require(expected_tokens is not None
                 and expected_tokens["question_token_count"] == len(question_ids)
                 and expected_tokens["question_token_ids_sha256"] == sha256_json(question_ids)
                 and expected_tokens["context_token_count"] == len(context_ids)
                 and expected_tokens["context_token_ids_sha256"] == sha256_json(context_ids)
                 and expected_tokens["active_writer_slots"] == len(chunks)
                 and expected_tokens["chunk_token_ids_sha256"]
                     == [sha256_json(chunk) for chunk in chunks],
                 "runtime tokenization differs from P0")
        prepared.append((source["frozen"], question_ids, chunks))

    backend = manifest["backend"]
    llm = LLM(
        model=manifest["model"]["path"],
        tokenizer=manifest["model"]["path"],
        trust_remote_code=True,
        tensor_parallel_size=backend["tensor_parallel_size"],
        dtype=backend["dtype"],
        seed=backend["engine_seed"],
        gpu_memory_utilization=backend["gpu_memory_utilization"],
        swap_space=backend["swap_space_gib"],
        enforce_eager=backend["enforce_eager"],
        disable_custom_all_reduce=backend["disable_custom_all_reduce"],
        max_model_len=backend["max_model_len"],
        max_num_batched_tokens=backend["max_num_batched_tokens"],
        max_num_seqs=backend["max_num_seqs"],
        enable_prefix_caching=backend["enable_prefix_caching"],
    )
    ledger = output_root / "trajectories/length_receipts.jsonl"
    existing = _read_ledger(ledger)
    if existing:
        _validate_ledger(
            manifest, inputs, ledger,
            expected_commit=expected_commit, run_id=run_id, p0_sha256=p0["p0_sha256"],
            tokenization_authority=tokenization_authority,
        )
    replicas = manifest["sampling"]["replicas"]
    expected_pairs = [(frozen["content_root_id"], replica)
                      for frozen, _question, _chunks in prepared
                      for replica in range(replicas)]
    _require([(row.get("content_root_id"), row.get("replica")) for row in existing]
             == expected_pairs[:len(existing)], "resume ledger is not an exact prefix")
    if mode == "replay":
        _require(len(existing) == len(expected_pairs),
                 "GPU replay requires a complete trajectory ledger")
    cursor = 0
    generate_calls = 0
    for frozen, question_ids, chunks in prepared:
        for replica in range(replicas):
            if mode == "produce" and cursor < len(existing):
                cursor += 1
                continue
            trajectory_seed = _stable_seed(
                manifest["sampling"]["base_seed"], frozen["content_root_id"], replica,
            )
            memory_ids = no_memory_ids
            memory_history_sha256 = []
            slots = []
            for turn in range(manifest["recurrent"]["max_writer_slots"]):
                if turn < len(chunks):
                    prompt_ids = writer_template.format(
                        prompt=question_ids, memory=memory_ids, chunk=chunks[turn],
                    ).tolist()
                    request_seed = derive_turn_request_seeds([trajectory_seed], [0], turn)[0]
                    raw_ids, finish_reason = _generate(
                        llm, prompt_ids, manifest["sampling"]["writer"], request_seed,
                    )
                    generate_calls += 1
                    receipt = _completion_receipt(
                        raw_ids, finish_reason,
                        termination_token_ids=termination_token_ids,
                        maximum=manifest["sampling"]["writer"]["max_tokens"],
                    )
                    memory_ids, afterstate = materialized_memory_receipt(
                        token_ids=raw_ids,
                        termination_token_ids=termination_token_ids,
                        content_root_id=frozen["content_root_id"],
                        trajectory_seed=trajectory_seed,
                        turn_index=turn,
                        arrived_chunk_token_sha256=expected_tokens[
                            "chunk_token_ids_sha256"
                        ][:turn + 1],
                        prior_memory_token_sha256=memory_history_sha256,
                    )
                    memory_history_sha256.append(afterstate["parsed_memory_sha256"])
                    slots.append({
                        "slot_index": turn,
                        "role": "writer",
                        "active": True,
                        "request_seed": request_seed,
                        **receipt,
                        **afterstate,
                    })
                else:
                    slots.append({
                        "slot_index": turn,
                        "role": "writer",
                        "active": False,
                        "request_seed": None,
                        "sampled_token_ids": [],
                        "sampled_policy_tokens": 0,
                        "completion_sha256": None,
                        "termination": "exogenous_termination",
                        "sampled_eos_counted": False,
                        "sampled_terminal_token_id": None,
                        "sampled_mask_width": manifest["sampling"]["writer"]["max_tokens"],
                        "sampled_mask_true_count": 0,
                        "sampled_mask_sha256": sha256_json(
                            [False] * manifest["sampling"]["writer"]["max_tokens"]
                        ),
                        "parsed_memory_token_ids": None,
                        "parsed_memory_sha256": None,
                        "parser_version": None,
                        "afterstate_sha256": None,
                    })
            answer_seed = derive_turn_request_seeds(
                [trajectory_seed], [0], manifest["recurrent"]["max_writer_slots"],
            )[0]
            answer_prompt = answer_template.format(
                prompt=question_ids, memory=memory_ids,
            ).tolist()
            answer_ids, answer_finish_reason = _generate(
                llm, answer_prompt, manifest["sampling"]["answer"], answer_seed,
            )
            generate_calls += 1
            slots.append({
                "slot_index": manifest["recurrent"]["max_writer_slots"],
                "role": "answer",
                "active": True,
                "request_seed": answer_seed,
                **_completion_receipt(
                    answer_ids, answer_finish_reason,
                    termination_token_ids=termination_token_ids,
                    maximum=manifest["sampling"]["answer"]["max_tokens"],
                ),
                "parsed_memory_token_ids": None,
                "parsed_memory_sha256": None,
                "parser_version": None,
                "afterstate_sha256": None,
            })
            trajectory_payload = {
                "schema": "memagent.mic.v2.reference-length-trajectory",
                "git_commit": expected_commit,
                "run_id": run_id,
                "p0_sha256": p0["p0_sha256"],
                "content_root_id": frozen["content_root_id"],
                "source_position": frozen["source_position"],
                "replica": replica,
                "trajectory_seed": trajectory_seed,
                "active_writer_slots": len(chunks),
                "sampled_policy_tokens": sum(slot["sampled_policy_tokens"] for slot in slots),
                "slots": slots,
            }
            if mode == "produce":
                _append(ledger, trajectory_payload)
            else:
                authenticated = dict(existing[cursor])
                for field in (
                    "record_index", "previous_record_sha256", "record_sha256",
                ):
                    authenticated.pop(field)
                _require(authenticated == trajectory_payload,
                         "independent GPU replay differs from trajectory ledger")
            cursor += 1
            print(canonical_json({
                "completed": cursor,
                "total": len(expected_pairs),
                "content_root_id": frozen["content_root_id"],
                "replica": replica,
            }), flush=True)
    if mode == "replay":
        execution = _load(execution_path)
        _self_digest(execution, "execution_sha256")
        _require(generate_calls == p0["seed_authority"]["active_request_count"],
                 "GPU replay did not regenerate every active action")
        replay_receipt = {
            "schema": "memagent.mic.v2.reference-length-gpu-replay",
            "status": "PASS",
            "decision": "MIC_V2_REFERENCE_LENGTH_GPU_REPLAY_PASS",
            "git_commit": expected_commit,
            "run_id": run_id,
            "p0_sha256": p0["p0_sha256"],
            "execution_sha256": execution["execution_sha256"],
            "gpu_pair": p0["gpu_pair"],
            "physical_gpu_identity": physical_gpu_identity,
            "vllm_version": vllm.__version__,
            "config_loader_environment": manifest["model"]["config_loader_environment"],
            "termination_token_ids": termination_token_ids,
            "trajectory_count": len(expected_pairs),
            "regenerated_generate_calls": generate_calls,
            "exact_token_match_count": generate_calls,
            "ledger_file_sha256": sha256_file(ledger),
        }
        replay_receipt["gpu_replay_sha256"] = sha256_json(replay_receipt)
        if replay_path.exists():
            _require(_load(replay_path) == replay_receipt,
                     "existing GPU replay differs from fresh regeneration")
        else:
            write_json_new(replay_path, replay_receipt)
        return replay_receipt

    receipt = {
        "schema": "memagent.mic.v2.reference-length-execution",
        "status": "PASS",
        "git_commit": expected_commit,
        "run_id": run_id,
        "p0_sha256": p0["p0_sha256"],
        "gpu_pair": p0["gpu_pair"],
        "physical_gpu_identity": physical_gpu_identity,
        "vllm_version": vllm.__version__,
        "config_loader_environment": manifest["model"]["config_loader_environment"],
        "strict_vllm": True,
        "tensor_parallel_size": backend["tensor_parallel_size"],
        "prefix_cache_enabled": False,
        "termination_token_ids": termination_token_ids,
        "trainer_attached": False,
        "actor_updates": 0,
        "new_generate_calls_this_session": generate_calls,
        "represented_generate_calls": p0["seed_authority"]["active_request_count"],
        "trajectory_count": len(expected_pairs),
        "ledger_file_sha256": sha256_file(ledger),
    }
    receipt["execution_sha256"] = sha256_json(receipt)
    write_json_new(execution_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("produce", "replay"), required=True)
    args = parser.parse_args()
    result = run(args.repo.resolve(), args.expected_commit,
                 args.output_root.resolve(), args.run_id, args.mode)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
