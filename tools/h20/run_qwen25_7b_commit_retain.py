#!/usr/bin/env python3
"""Strict-vLLM single-engine executor for four COMMIT/RETAIN capture pairs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.commit_retain_capture import (  # noqa: E402
    ARMS,
    build_capture_envelope,
    build_pair_record,
    build_state_blob,
    canonical_sha256,
    stable_turn_id,
    validate_capture_ledger,
)
from recurrent.research.gate_a_execution import append_jsonl, sha256_file  # noqa: E402
from recurrent.research.serialization_credit_pilots import (  # noqa: E402
    center_truncate_token_ids,
    read_jsonl,
    write_json_exclusive,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds  # noqa: E402
from tools.h20.preflight_qwen25_7b_commit_retain import (  # noqa: E402
    MANIFEST_REL,
    _gpu_profile,
    _current_binding,
    _expected_run_receipt,
    expected_git_commit,
    expected_pair_binding,
    experiment_name,
    load_manifest,
    validate_capture_credential,
    validate_p0,
)


PROCESS_INSTANCE_UUID = str(uuid.uuid4())
PROCESS_PID = os.getpid()
_ENGINE_CONSTRUCTION_COUNT = 0
_GENERATE_CALL_COUNT = 0


def _runtime(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, list[str]]:
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    profile = _gpu_profile(manifest)
    visible_devices = profile["visible_devices"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible_devices:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES must be exactly physical GPU{visible_devices}"
        )
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if os.environ.get("VLLM_USE_V1") != "0":
        raise ValueError("VLLM_USE_V1 must be 0")
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        raise ValueError("VLLM_WORKER_MULTIPROC_METHOD must be spawn")
    current_sha = _current_binding(manifest, resolved, full_model_sha=True)
    completed = subprocess.run(
        [
            "nvidia-smi", "-i", visible_devices, "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise ValueError(
            f"cannot authenticate GPU{visible_devices}: {completed.stderr.strip()}"
        )
    identities = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    try:
        observed_indices = [int(line.split(",", 1)[0].strip()) for line in identities]
    except (ValueError, IndexError):
        observed_indices = []
    if observed_indices != profile["physical_whitelist"]:
        raise ValueError(
            f"physical GPU indices {observed_indices} != {profile['physical_whitelist']}"
        )
    if identities != resolved["runtime_binding"]["physical_gpu_identity"]:
        raise ValueError("physical GPU UUID/name binding differs from P0")
    return manifest, resolved, current_sha, identities


def _tokenizer(manifest: Mapping[str, Any]):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        manifest["model"]["path"], trust_remote_code=True, local_files_only=True
    )


def _engine(manifest: Mapping[str, Any]):
    global _ENGINE_CONSTRUCTION_COUNT
    _ENGINE_CONSTRUCTION_COUNT += 1
    if _ENGINE_CONSTRUCTION_COUNT != 1:
        raise RuntimeError("capture process may construct exactly one vLLM engine")
    from vllm import LLM

    signature = inspect.signature(LLM.generate)
    for parameter in ("prompt_token_ids", "sampling_params"):
        if parameter not in signature.parameters:
            raise RuntimeError(f"vLLM LLM.generate lacks {parameter}")
    backend = manifest["backend"]
    return LLM(
        model=manifest["model"]["path"],
        tokenizer=manifest["model"]["path"],
        trust_remote_code=True,
        tensor_parallel_size=int(manifest["gpu"]["tensor_parallel_size"]),
        dtype=backend["dtype"],
        seed=int(backend["engine_seed"]),
        gpu_memory_utilization=float(backend["gpu_memory_utilization"]),
        swap_space=int(backend["swap_space_gib"]),
        enforce_eager=True,
        disable_custom_all_reduce=True,
        max_model_len=int(backend["max_model_len"]),
        max_num_batched_tokens=int(backend["max_num_batched_tokens"]),
        max_num_seqs=1,
        enable_prefix_caching=False,
    )


def _worker_multiprocessing_evidence(
    manifest: Mapping[str, Any], *, parent_cuda_initialized: bool
) -> dict[str, Any]:
    """Fail closed unless vLLM will spawn its tensor-parallel workers."""
    from vllm import envs
    from vllm.utils import get_mp_context

    if type(parent_cuda_initialized) is not bool:
        raise ValueError("parent CUDA initialization observation must be boolean")
    configured = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    observed = envs.VLLM_WORKER_MULTIPROC_METHOD
    context_method = get_mp_context().get_start_method()
    expected = manifest["backend"]["VLLM_WORKER_MULTIPROC_METHOD"]
    if expected != "spawn":
        raise ValueError("frozen worker multiprocessing method is not spawn")
    if configured != expected or observed != expected or context_method != expected:
        raise RuntimeError(
            "vLLM worker multiprocessing must be spawn before engine construction: "
            f"configured={configured}, observed={observed}, context={context_method}, "
            f"parent_cuda_initialized={parent_cuda_initialized}"
        )
    return {
        "worker_multiproc_method": expected,
        "vllm_observed_worker_multiproc_method": observed,
        "multiprocessing_context_method": context_method,
        "parent_cuda_initialized_before_engine": parent_cuda_initialized,
        "parent_cuda_initialization_policy": manifest["backend"][
            "parent_cuda_initialization_policy"
        ],
    }


def _generate_one(
    llm: Any, prompt_ids: list[int], sampling: Mapping[str, Any], *, seed: int
) -> tuple[list[int], int]:
    global _GENERATE_CALL_COUNT
    from vllm import SamplingParams

    _GENERATE_CALL_COUNT += 1
    params = {**dict(sampling), "seed": int(seed)}
    outputs = llm.generate(
        prompt_token_ids=[prompt_ids],
        sampling_params=[SamplingParams(**params)],
        use_tqdm=False,
    )
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("strict single-request generation returned an unexpected batch")
    if list(outputs[0].prompt_token_ids) != prompt_ids:
        raise RuntimeError("vLLM returned different prompt token IDs")
    return list(outputs[0].outputs[0].token_ids), _GENERATE_CALL_COUNT


def _loaded_state(source_role: str, source_turn_id: str | None, state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_role": source_role,
        "source_turn_id": source_turn_id,
        "state": dict(state),
    }


def _prompt(
    *,
    tokenizer: Any,
    prompt_ids: list[int],
    template_sha256: str,
    checkpoint_sha256: str,
    loaded_state: Mapping[str, Any],
    chunk_ids: list[int] | None,
) -> dict[str, Any]:
    result = {
        "text": tokenizer.decode(prompt_ids, skip_special_tokens=False),
        "token_ids": prompt_ids,
        "template_sha256": template_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "loaded_state_receipt": dict(loaded_state),
    }
    if chunk_ids is not None:
        result["chunk_token_ids"] = chunk_ids
    return result


def _writer_call(
    *,
    llm: Any,
    tokenizer: Any,
    writer_template: Any,
    question_ids: list[int],
    memory_state: Mapping[str, Any],
    loaded_receipt: Mapping[str, Any],
    chunk_ids: list[int],
    sampling: Mapping[str, Any],
    request_seed: int,
    writer_template_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    memory_ids = list(memory_state["token_ids"])
    prompt_ids = writer_template.format(
        prompt=question_ids, memory=memory_ids, chunk=chunk_ids
    ).tolist()
    raw_ids, call_index = _generate_one(llm, prompt_ids, sampling, seed=request_seed)
    eos_id = int(tokenizer.eos_token_id)
    removed_positions = [index for index, token_id in enumerate(raw_ids) if token_id == eos_id]
    state_ids = [token_id for token_id in raw_ids if token_id != eos_id]
    return {
        "prompt": _prompt(
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            template_sha256=writer_template_sha256,
            checkpoint_sha256=checkpoint_sha256,
            loaded_state=loaded_receipt,
            chunk_ids=chunk_ids,
        ),
        "sampling_params": dict(sampling),
        "request_seed": request_seed,
        "configured_request_seed": request_seed,
        "actual_request_seed": request_seed,
        "generate_call_index": call_index,
        "raw_completion_token_ids": raw_ids,
        "eos_token_id": eos_id,
        "eos_token_positions_removed": removed_positions,
        "eos_removal_semantics": "remove_all_eos_matching_native_unpad",
        "state_after": build_state_blob(state_ids),
        "output_text": tokenizer.decode(state_ids, skip_special_tokens=False),
    }


def _reader_call(
    *,
    llm: Any,
    tokenizer: Any,
    reader_template: Any,
    question_ids: list[int],
    memory_state: Mapping[str, Any],
    loaded_receipt: Mapping[str, Any],
    sampling: Mapping[str, Any],
    request_seed: int,
    reader_template_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    prompt_ids = reader_template.format(
        prompt=question_ids, memory=list(memory_state["token_ids"])
    ).tolist()
    output_ids, call_index = _generate_one(llm, prompt_ids, sampling, seed=request_seed)
    return {
        "prompt": _prompt(
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            template_sha256=reader_template_sha256,
            checkpoint_sha256=checkpoint_sha256,
            loaded_state=loaded_receipt,
            chunk_ids=None,
        ),
        "sampling_params": dict(sampling),
        "request_seed": request_seed,
        "configured_request_seed": request_seed,
        "actual_request_seed": request_seed,
        "generate_call_index": call_index,
        "output_token_ids": output_ids,
        "output_text": tokenizer.decode(output_ids, skip_special_tokens=False),
    }


def _source_inputs(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any], tokenizer: Any
) -> list[tuple[dict[str, Any], Mapping[str, Any], list[int], list[list[int]]]]:
    import pyarrow.parquet as parquet

    rows = parquet.read_table(
        manifest["data"]["validation"],
        columns=["prompt", "context", "reward_model", "extra_info"],
    ).to_pylist()
    chunk_size = int(manifest["recurrent"]["chunk_size"])
    result = []
    for frozen in resolved["frozen_pairs"]:
        source = rows[int(frozen["raw_row_position"])]
        question = str(source["prompt"][0]["content"])
        context = str(source["context"])
        ground_truth = [str(item) for item in source["reward_model"]["ground_truth"]]
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != frozen["source_question_hash"]:
            raise ValueError("source question differs from stable P0")
        if hashlib.sha256(context.encode("utf-8")).hexdigest() != frozen["source_context_hash"]:
            raise ValueError("source context differs from stable P0")
        if canonical_sha256(ground_truth) != frozen["ground_truth_hash"]:
            raise ValueError("source ground truth differs from stable P0")
        question_ids = list(tokenizer.encode(question, add_special_tokens=False))
        context_ids = center_truncate_token_ids(
            list(tokenizer.encode(context, add_special_tokens=False)),
            int(manifest["recurrent"]["max_context_tokens"]),
        )
        chunks = [context_ids[offset : offset + chunk_size] for offset in range(0, len(context_ids), chunk_size)]
        if canonical_sha256(question_ids) != frozen["question_token_ids_sha256"]:
            raise ValueError("question token IDs differ from P0")
        if canonical_sha256(context_ids) != frozen["context_token_ids_sha256"]:
            raise ValueError("context token IDs differ from P0")
        if [canonical_sha256(chunk) for chunk in chunks] != frozen["chunk_token_ids_sha256"]:
            raise ValueError("context chunk token IDs differ from P0")
        if len(chunks) != frozen["total_writer_turns"]:
            raise ValueError("writer horizon differs from P0")
        result.append((frozen, source, question_ids, chunks))
    return result


def capture(manifest_path: Path, *, credential_path: Path) -> dict[str, Any]:
    global _ENGINE_CONSTRUCTION_COUNT, _GENERATE_CALL_COUNT
    if _ENGINE_CONSTRUCTION_COUNT or _GENERATE_CALL_COUNT:
        raise RuntimeError("capture runner globals were already used")
    manifest, resolved, current_sha, physical_gpu_identity = _runtime(manifest_path)
    credential_evidence = validate_capture_credential(
        credential_path,
        manifest=manifest,
        resolved=resolved,
        current_binding_sha256=current_sha,
        require_live_parent=True,
    )
    capture_path = Path(manifest["paths"]["capture_ledger"])
    run_receipt_path = Path(manifest["paths"]["capture_run_receipt"])
    if capture_path.exists() or run_receipt_path.exists():
        raise FileExistsError("refuse to overwrite append-only capture artifacts")
    tokenizer = _tokenizer(manifest)
    from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template
    import torch
    import vllm

    if vllm.__version__ != "0.8.2":
        raise ValueError("runtime is not frozen vLLM 0.8.2")
    writer_template_text = chat_template(tokenizer).format(message=TEMPLATE)
    reader_template_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
    execution_binding = resolved["execution_binding"]
    profile = _gpu_profile(manifest)
    writer_template_sha = hashlib.sha256(writer_template_text.encode("utf-8")).hexdigest()
    reader_template_sha = hashlib.sha256(reader_template_text.encode("utf-8")).hexdigest()
    if writer_template_sha != execution_binding["writer_prompt_template_sha256"]:
        raise ValueError("writer prompt template differs from P0")
    if reader_template_sha != execution_binding["reader_prompt_template_sha256"]:
        raise ValueError("reader prompt template differs from P0")
    worker_multiprocessing = _worker_multiprocessing_evidence(
        manifest, parent_cuda_initialized=bool(torch.cuda.is_initialized())
    )
    writer_template = TokenTemplate(writer_template_text, tokenizer)
    reader_template = TokenTemplate(reader_template_text, tokenizer)
    no_memory_ids = list(tokenizer.encode(
        manifest["recurrent"]["no_memory_text"], add_special_tokens=False
    ))
    no_memory_state = build_state_blob(no_memory_ids)
    sources = _source_inputs(manifest, resolved, tokenizer)
    expected_global_calls = int(execution_binding["expected_global_generate_call_count"])
    if sum(int(item[0]["expected_pair_generate_calls"]) for item in sources) != expected_global_calls:
        raise ValueError("precomputed global call schedule drifted")
    llm = _engine(manifest)
    engine_id = f"commit-retain-engine-{uuid.uuid4()}"
    cache_namespace = f"no-prefix-cache-{uuid.uuid4()}"
    execution = {
        "backend": "vllm",
        "vllm_version": vllm.__version__,
        "strict_vllm": True,
        "tensor_parallel_size": int(manifest["gpu"]["tensor_parallel_size"]),
        "gpu_pair_slug": profile["pair_slug"],
        "physical_gpu_whitelist": profile["physical_whitelist"],
        "physical_gpu_identity": physical_gpu_identity,
        "visible_devices": profile["visible_devices"],
        "cuda_device_order": "PCI_BUS_ID",
        "prefix_cache_enabled": False,
        "max_num_seqs": 1,
        "one_prompt_per_generate_call": True,
        "engine_construction_count": 1,
        "full_model_sha_verified_at_capture_start": True,
        "trainer_attached": False,
        "actor_training_calls": 0,
        "engine_id": engine_id,
        "cache_namespace": cache_namespace,
        "process_instance_uuid": PROCESS_INSTANCE_UUID,
        "process_pid": PROCESS_PID,
        "global_generate_call_count": expected_global_calls,
        "engine_config_sha256": execution_binding["engine_config_sha256"],
        **worker_multiprocessing,
        **credential_evidence,
    }
    checkpoint_sha = execution_binding["model_manifest_sha256"]
    writer_decode = dict(manifest["intervention"]["writer_decode"])
    reader_decode = dict(manifest["intervention"]["reader_decode"])

    for frozen, source, question_ids, chunks in sources:
        trajectory_seed = int(frozen["trajectory_seed"])
        intervention = int(frozen["intervention_writer_turn"])
        total = int(frozen["total_writer_turns"])
        stable_write = frozen["stable_write_id"]
        prefix: list[dict[str, Any]] = []
        memory_state = no_memory_state
        source_role = "no_memory"
        source_turn_id: str | None = None
        for turn in range(intervention):
            request_seed = derive_turn_request_seeds([trajectory_seed], [0], turn)[0]
            generation = _writer_call(
                llm=llm,
                tokenizer=tokenizer,
                writer_template=writer_template,
                question_ids=question_ids,
                memory_state=memory_state,
                loaded_receipt=_loaded_state(source_role, source_turn_id, memory_state),
                chunk_ids=chunks[turn],
                sampling=writer_decode,
                request_seed=request_seed,
                writer_template_sha256=writer_template_sha,
                checkpoint_sha256=checkpoint_sha,
            )
            source_turn_id = stable_turn_id(
                stable_write_id=stable_write,
                phase="prefix_writer",
                arm="SHARED",
                writer_turn=turn,
            )
            memory_state = generation["state_after"]
            source_role = "previous_prefix_output"
            prefix.append(generation)
        old_state = memory_state
        old_turn_id = source_turn_id
        candidate_seed = derive_turn_request_seeds([trajectory_seed], [0], intervention)[0]
        candidate = _writer_call(
            llm=llm,
            tokenizer=tokenizer,
            writer_template=writer_template,
            question_ids=question_ids,
            memory_state=old_state,
            loaded_receipt=_loaded_state("old_state", old_turn_id, old_state),
            chunk_ids=chunks[intervention],
            sampling=writer_decode,
            request_seed=candidate_seed,
            writer_template_sha256=writer_template_sha,
            checkpoint_sha256=checkpoint_sha,
        )
        candidate_state = candidate["state_after"]
        candidate_turn_id = stable_turn_id(
            stable_write_id=stable_write,
            phase="candidate_writer",
            arm="SHARED",
            writer_turn=intervention,
        )
        if candidate_state["bytes_sha256"] == old_state["bytes_sha256"]:
            raise RuntimeError(
                f"candidate is byte-identical to old state for stable write {stable_write}"
            )

        arms: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            if arm == "COMMIT":
                arm_state = candidate_state
                initial_role = "candidate"
                initial_turn_id = candidate_turn_id
            else:
                # Exact RETAIN: keep and load the already-materialized old-state blob.
                # There is no decode/encode, tensor cast, or writer regeneration here.
                arm_state = old_state
                initial_role = "old_state"
                initial_turn_id = old_turn_id
            initial_receipt = _loaded_state(initial_role, initial_turn_id, arm_state)
            future: list[dict[str, Any]] = []
            loaded_role = initial_role
            loaded_turn_id = initial_turn_id
            for turn in range(intervention + 1, total):
                request_seed = derive_turn_request_seeds([trajectory_seed], [0], turn)[0]
                generation = _writer_call(
                    llm=llm,
                    tokenizer=tokenizer,
                    writer_template=writer_template,
                    question_ids=question_ids,
                    memory_state=arm_state,
                    loaded_receipt=_loaded_state(loaded_role, loaded_turn_id, arm_state),
                    chunk_ids=chunks[turn],
                    sampling=writer_decode,
                    request_seed=request_seed,
                    writer_template_sha256=writer_template_sha,
                    checkpoint_sha256=checkpoint_sha,
                )
                loaded_turn_id = stable_turn_id(
                    stable_write_id=stable_write,
                    phase="future_writer",
                    arm=arm,
                    writer_turn=turn,
                )
                loaded_role = "previous_future_output"
                arm_state = generation["state_after"]
                future.append(generation)
            reader_seed = derive_turn_request_seeds([trajectory_seed], [0], total)[0]
            final = _reader_call(
                llm=llm,
                tokenizer=tokenizer,
                reader_template=reader_template,
                question_ids=question_ids,
                memory_state=arm_state,
                loaded_receipt=_loaded_state("previous_future_output", loaded_turn_id, arm_state),
                sampling=reader_decode,
                request_seed=reader_seed,
                reader_template_sha256=reader_template_sha,
                checkpoint_sha256=checkpoint_sha,
            )
            arms[arm] = {
                "initial_loaded_state_receipt": initial_receipt,
                "future_turns": future,
                "final_reader": final,
            }

        future_turns = list(range(intervention + 1, total))
        shared_contract = {
            "intervention_writer_turn": intervention,
            "total_writer_turns": total,
            "trajectory_seed": trajectory_seed,
            "future_chunks": [
                {"writer_turn": turn, "token_ids": chunks[turn]} for turn in future_turns
            ],
            "horizon": {
                "future_writer_turns": future_turns,
                "future_writer_calls_per_arm": len(future_turns),
                "final_reader_calls_per_arm": 1,
                "terminal_writer_turn": total - 1,
            },
            "writer_checkpoint_sha256": checkpoint_sha,
            "reader_checkpoint_sha256": checkpoint_sha,
            "writer_prompt_template_sha256": writer_template_sha,
            "reader_prompt_template_sha256": reader_template_sha,
            "writer_decode": writer_decode,
            "reader_decode": reader_decode,
            "cache_contract": {
                "enable_prefix_caching": False,
                "max_num_seqs": 1,
                "one_prompt_per_generate_call": True,
                "kv_state_reuse_across_generate_calls": False,
                "same_engine_for_both_arms": True,
            },
            "cost_contract": {
                "shared_candidate_generation_calls": 1,
                "per_arm_writer_generation_calls": len(future_turns),
                "per_arm_reader_generation_calls": 1,
                "per_arm_total_generation_calls": len(future_turns) + 1,
                "per_arm_writer_max_tokens": writer_decode["max_tokens"],
                "per_arm_reader_max_tokens": reader_decode["max_tokens"],
                "budgets_identical_by_design": True,
                "realized_token_counts_are_measured_not_forced_equal": True,
            },
        }
        pair = build_pair_record(
            {
                **{field: frozen[field] for field in (
                    "example_id", "semantic_dataset_index", "source_order_index",
                    "raw_row_position", "production_effective_position", "eval_manifest_hash",
                    "source_question_hash", "source_context_hash", "ground_truth_hash",
                )},
                "trajectory_seed": trajectory_seed,
                "intervention_writer_turn": intervention,
                "total_writer_turns": total,
                "question_token_ids": question_ids,
                "ground_truth": [str(item) for item in source["reward_model"]["ground_truth"]],
                "no_memory_state": no_memory_state,
                "prefix_turns": prefix,
                "old_state": old_state,
                "candidate": candidate,
                "shared_contract": shared_contract,
                "arms": arms,
                "execution": execution,
            }
        )
        if pair["stable_write_id"] != stable_write:
            raise RuntimeError("runtime stable write differs from P0")
        envelope = build_capture_envelope(
            pair,
            experiment_name=experiment_name(manifest),
            git_commit=expected_git_commit(),
            run_id=manifest["run_id"],
            execution_binding_sha256=resolved["execution_binding_sha256"],
            runtime_binding_sha256=resolved["runtime_binding_sha256"],
            current_binding_sha256=current_sha,
        )
        append_jsonl(capture_path, envelope)

    if _ENGINE_CONSTRUCTION_COUNT != 1 or _GENERATE_CALL_COUNT != expected_global_calls:
        raise RuntimeError(
            "capture engine/generate counts differ from P0 schedule: "
            f"engine={_ENGINE_CONSTRUCTION_COUNT}, calls={_GENERATE_CALL_COUNT}, "
            f"expected={expected_global_calls}"
        )
    report = validate_capture_ledger(
        read_jsonl(capture_path),
        frozen_pairs=resolved["frozen_pairs"],
        experiment_name=experiment_name(manifest),
        git_commit=expected_git_commit(),
        run_id=manifest["run_id"],
        execution_binding_sha256=resolved["execution_binding_sha256"],
        runtime_binding_sha256=resolved["runtime_binding_sha256"],
        current_binding_sha256=current_sha,
        decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        writer_prompt_builder=lambda question, memory, chunk: writer_template.format(
            prompt=question, memory=memory, chunk=chunk
        ).tolist(),
        reader_prompt_builder=lambda question, memory: reader_template.format(
            prompt=question, memory=memory
        ).tolist(),
        expected_pair_binding=expected_pair_binding(manifest, resolved, tokenizer),
    )
    receipt = _expected_run_receipt(
        manifest=manifest,
        resolved=resolved,
        current_binding_sha256=current_sha,
        capture_report=report,
        capture_path=capture_path,
    )
    write_json_exclusive(run_receipt_path, receipt)
    return {
        **report,
        "capture_ledger": str(capture_path.resolve()),
        "capture_ledger_sha256": sha256_file(capture_path),
        "capture_run_receipt": str(run_receipt_path.resolve()),
        "capture_run_receipt_sha256": sha256_file(run_receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_REL)
    parser.add_argument("capture", nargs="?")
    parser.add_argument("--credential", type=Path, required=True)
    args = parser.parse_args()
    if args.capture not in (None, "capture"):
        parser.error("only the capture command is supported")
    report = capture(args.manifest, credential_path=args.credential)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
