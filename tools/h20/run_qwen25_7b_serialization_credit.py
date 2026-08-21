#!/usr/bin/env python3
"""Strict-vLLM, single-request executors for frozen 7B SMSB4/Tetrad4."""

from __future__ import annotations

import argparse
import hashlib
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

from recurrent.research.s128_hotpot_metrics import score_terminal_output  # noqa: E402
from recurrent.research.serialization_credit_pilots import (  # noqa: E402
    adjudicate_tetrad_pilot,
    best_length_derangement,
    build_capture_record,
    build_replay_request,
    build_tetrad_requests,
    canonical_sha256,
    center_truncate_token_ids,
    content_words,
    read_jsonl,
    split_documents,
    summarize_smsb_pilot,
    validate_replay,
    validate_capture_record,
    validate_single_request_token_budget,
    validate_tetrad_manifest,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds  # noqa: E402
from tools.h20.preflight_qwen25_7b_serialization_credit import (  # noqa: E402
    MANIFEST_REL,
    load_manifest,
    load_child_credential_claim,
    load_parent_authority_secret,
    validate_p0,
    verify_current_binding,
)


PROCESS_INSTANCE_UUID = str(uuid.uuid4())
PROCESS_PID = os.getpid()
_ENGINE_CONSTRUCTION_COUNT = 0
_GENERATE_CALL_COUNT = 0


def _runtime(
    manifest_path: Path, *, full_model_sha: bool
) -> tuple[dict[str, Any], dict[str, Any], str, list[str]]:
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != manifest["gpu"]["visible_devices"]:
        raise ValueError("CUDA_VISIBLE_DEVICES must be exactly the frozen physical GPU list 2,3")
    if os.environ.get("CUDA_DEVICE_ORDER") != manifest["gpu"]["cuda_device_order"]:
        raise ValueError("CUDA_DEVICE_ORDER must be exactly the frozen PCI_BUS_ID setting")
    if os.environ.get("VLLM_USE_V1") != manifest["backend"]["VLLM_USE_V1"]:
        raise ValueError("VLLM_USE_V1 differs from the frozen strict-vLLM setting")
    if int(manifest["gpu"]["tensor_parallel_size"]) != 2:
        raise ValueError("frozen runner requires tensor_parallel_size=2")
    current_binding_sha = verify_current_binding(
        manifest, resolved, full_model_sha=full_model_sha
    )
    completed = subprocess.run(
        [
            "nvidia-smi", "-i", manifest["gpu"]["visible_devices"],
            "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise ValueError(f"cannot authenticate physical GPU2-3 identity: {completed.stderr.strip()}")
    physical_gpu_identity = [
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    ]
    if physical_gpu_identity != resolved["runtime_binding"]["physical_gpu_identity"]:
        raise ValueError("physical GPU UUID/name binding differs from P0")
    return manifest, resolved, current_binding_sha, physical_gpu_identity


def _tokenizer(manifest: Mapping[str, Any]):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        manifest["model"]["path"], trust_remote_code=True, local_files_only=True
    )


def _engine(manifest: Mapping[str, Any]):
    global _ENGINE_CONSTRUCTION_COUNT
    _ENGINE_CONSTRUCTION_COUNT += 1
    if _ENGINE_CONSTRUCTION_COUNT != 1:
        raise RuntimeError("one Python child may construct exactly one vLLM engine")
    import inspect
    from vllm import LLM

    signature = inspect.signature(LLM.generate)
    for required_parameter in ("prompt_token_ids", "sampling_params"):
        if required_parameter not in signature.parameters:
            raise RuntimeError(
                f"vLLM LLM.generate lacks required {required_parameter} parameter"
            )

    backend = manifest["backend"]
    return LLM(
        model=manifest["model"]["path"],
        tokenizer=manifest["model"]["path"],
        trust_remote_code=True,
        tensor_parallel_size=2,
        dtype=backend["dtype"],
        seed=int(backend["engine_seed"]),
        gpu_memory_utilization=float(backend["gpu_memory_utilization"]),
        swap_space=int(backend["swap_space_gib"]),
        enforce_eager=bool(backend["enforce_eager"]),
        disable_custom_all_reduce=bool(backend["disable_custom_all_reduce"]),
        max_model_len=int(backend["max_model_len"]),
        max_num_batched_tokens=int(backend["max_num_batched_tokens"]),
        max_num_seqs=1,
        enable_prefix_caching=False,
    )


def _generate_one(
    llm: Any,
    prompt_ids: list[int],
    sampling: dict[str, Any],
    *,
    manifest: Mapping[str, Any],
):
    global _GENERATE_CALL_COUNT
    from vllm import SamplingParams

    params = dict(sampling)
    params.pop("do_sample", None)
    validate_single_request_token_budget(
        prompt_ids,
        params.get("max_tokens"),
        max_model_len=manifest["backend"]["max_model_len"],
        max_num_batched_tokens=manifest["backend"]["max_num_batched_tokens"],
    )
    _GENERATE_CALL_COUNT += 1
    outputs = llm.generate(
        prompt_token_ids=[prompt_ids],
        sampling_params=[SamplingParams(**params)],
        use_tqdm=False,
    )
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("strict single-request execution returned an unexpected batch")
    if list(outputs[0].prompt_token_ids) != prompt_ids:
        raise RuntimeError("vLLM returned a different prompt-token sequence")
    return outputs[0].outputs[0]


def _parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return parquet.read_table(
        path, columns=["prompt", "context", "reward_model", "extra_info"]
    ).to_pylist()


def _question(row: Mapping[str, Any]) -> str:
    prompt = row["prompt"]
    if len(prompt) != 1 or prompt[0]["role"] != "user":
        raise ValueError("fixed S128 prompt contract drifted")
    return str(prompt[0]["content"])


def _engine_evidence(
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    *,
    expected_generate_call_count: int,
    physical_gpu_identity: list[str],
) -> dict[str, Any]:
    return {
        "strict_vllm": True,
        "tensor_parallel_size": 2,
        "physical_gpu_whitelist": [2, 3],
        "physical_gpu_identity": physical_gpu_identity,
        "cuda_device_order": "PCI_BUS_ID",
        "visible_devices": "2,3",
        "prefix_cache_enabled": False,
        "single_request_only": True,
        "max_num_seqs": 1,
        "one_prompt_per_generate_call": True,
        "engine_config_sha256": resolved["execution_binding"]["engine_config_sha256"],
        "model_manifest_sha256": resolved["execution_binding"][
            "model_manifest_sha256"
        ],
        "required_vllm_version": manifest["backend"]["required_version"],
        "process_instance_uuid": PROCESS_INSTANCE_UUID,
        "process_pid": PROCESS_PID,
        "engine_construction_count": 1,
        "generate_call_count": expected_generate_call_count,
        "full_model_sha_verified_at_capture_start": True,
        "full_model_sha_verified_at_child_start": True,
    }


def capture_smsb(
    manifest_path: Path, output: Path, *, credential_path: Path
) -> dict[str, Any]:
    manifest, resolved, current_binding_sha, physical_gpu_identity = _runtime(
        manifest_path, full_model_sha=True
    )
    credential_evidence = load_child_credential_claim(
        credential_path,
        manifest=manifest,
        resolved=resolved,
        current_binding_sha=current_binding_sha,
        child_kind="smsb_capture",
        child_identity="capture4",
    )
    tokenizer = _tokenizer(manifest)
    from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template
    import vllm

    if vllm.__version__ != manifest["backend"]["required_version"]:
        raise ValueError("runtime vLLM version drifted")
    writer_template_text = chat_template(tokenizer).format(message=TEMPLATE)
    final_template_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
    if hashlib.sha256(final_template_text.encode("utf-8")).hexdigest() != resolved[
        "execution_binding"
    ]["final_prompt_template_sha256"]:
        raise ValueError("final prompt template hash drifted after P0")
    writer_template = TokenTemplate(writer_template_text, tokenizer)
    final_template = TokenTemplate(final_template_text, tokenizer)
    no_memory = tokenizer.encode(manifest["recurrent"]["no_memory_text"], add_special_tokens=False)
    rows = _parquet_rows(manifest["data"]["validation"])
    recurrent = manifest["recurrent"]
    source_inputs: list[tuple[dict[str, Any], Mapping[str, Any], list[int], list[int]]] = []
    expected_generate_calls = 0
    for pilot in resolved["pilot_rows"]:
        source = rows[int(pilot["raw_row_position"])]
        question_ids = list(tokenizer.encode(_question(source), add_special_tokens=False))
        context_ids = center_truncate_token_ids(
            list(tokenizer.encode(str(source["context"]), add_special_tokens=False)),
            recurrent["max_context_tokens"],
        )
        writer_calls = (len(context_ids) + int(recurrent["chunk_size"]) - 1) // int(
            recurrent["chunk_size"]
        )
        if writer_calls < 1 or writer_calls > int(recurrent["max_chunks"]):
            raise RuntimeError("selected context requires an invalid number of recurrent writer calls")
        expected_generate_calls += writer_calls + 2
        source_inputs.append((pilot, source, question_ids, context_ids))
    llm = _engine(manifest)
    engine_id = f"smsb-capture-{uuid.uuid4()}"
    cache_namespace = f"smsb-capture-cache-{uuid.uuid4()}"
    engine_config_sha = resolved["execution_binding"]["engine_config_sha256"]
    runtime_binding_sha = resolved["runtime_binding_sha256"]
    hashes = {
        "model": resolved["execution_binding"]["model_manifest_sha256"],
        "tokenizer": resolved["execution_binding"]["tokenizer_manifest_sha256"],
        "config": canonical_sha256(
            {
                "engine": resolved["execution_binding"]["engine_config"],
                "recurrent": recurrent,
            }
        ),
        "code": resolved["execution_binding"]["execution_code_combined_sha256"],
    }
    evidence = _engine_evidence(
        manifest,
        resolved,
        expected_generate_call_count=expected_generate_calls,
        physical_gpu_identity=physical_gpu_identity,
    )
    evidence.update(
        credential_evidence,
        runtime_binding_sha256=resolved["runtime_binding_sha256"],
        execution_binding_sha256=resolved["execution_binding_sha256"],
        current_binding_sha256=current_binding_sha,
        observed_parent_pid=os.getppid(),
    )
    records: list[dict[str, Any]] = []
    for pilot, source, question_ids, context_ids in source_inputs:
        question = _question(source)
        context = str(source["context"])
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != pilot["source_question_hash"]:
            raise ValueError("capture question hash differs from P0")
        if hashlib.sha256(context.encode("utf-8")).hexdigest() != pilot["source_context_hash"]:
            raise ValueError("capture context hash differs from P0")
        ground_truth = list(source["reward_model"]["ground_truth"])
        if canonical_sha256(ground_truth) != pilot["ground_truth_hash"]:
            raise ValueError("capture ground truth hash differs from P0")
        memory_ids: list[int] | None = None
        memory_ledger: list[dict[str, Any]] = []
        trajectory_seed = int(pilot["trajectory_seed"])
        chunk_size = int(recurrent["chunk_size"])
        for turn, offset in enumerate(range(0, len(context_ids), chunk_size)):
            if turn >= int(recurrent["max_chunks"]):
                raise RuntimeError("context exceeded the frozen maximum chunk count")
            input_memory_ids = (
                list(memory_ids) if memory_ids is not None else list(no_memory)
            )
            chunk_ids = list(context_ids[offset : offset + chunk_size])
            prompt_ids = writer_template.format(
                prompt=question_ids,
                memory=input_memory_ids,
                chunk=chunk_ids,
            ).tolist()
            if turn == 0 and canonical_sha256(prompt_ids) != pilot["writer_turn0_prompt_token_sha256"]:
                raise RuntimeError("writer turn0 prompt differs from P0")
            request_seed = derive_turn_request_seeds([trajectory_seed], [0], turn)[0]
            completion = _generate_one(
                llm,
                prompt_ids,
                {
                    **manifest["smsb"]["capture_writer_decode"],
                    "seed": request_seed,
                    "max_tokens": int(recurrent["max_memory_tokens"]),
                },
                manifest=manifest,
            )
            memory_ids = list(completion.token_ids)
            generate_call_index = _GENERATE_CALL_COUNT
            if memory_ids and memory_ids[-1] == tokenizer.eos_token_id:
                memory_ids.pop()
            memory_ledger.append(
                {
                    "turn": turn,
                    "writer_prompt_token_ids": prompt_ids,
                    "writer_prompt_token_ids_sha256": canonical_sha256(prompt_ids),
                    "writer_prompt_token_length": len(prompt_ids),
                    "chunk_start": offset,
                    "chunk_end": offset + len(chunk_ids),
                    "chunk_token_ids": chunk_ids,
                    "chunk_token_ids_sha256": canonical_sha256(chunk_ids),
                    "chunk_token_length": len(chunk_ids),
                    "input_memory_token_ids": input_memory_ids,
                    "input_memory_token_ids_sha256": canonical_sha256(
                        input_memory_ids
                    ),
                    "input_memory_token_length": len(input_memory_ids),
                    "text": tokenizer.decode(memory_ids, skip_special_tokens=False),
                    "token_ids": memory_ids,
                    "request_seed": request_seed,
                    "configured_request_seed": request_seed,
                    "actual_request_seed": request_seed,
                    "generate_call_index": generate_call_index,
                }
            )
        if memory_ids is None:
            raise RuntimeError("capture produced no writer turn")
        final_prompt_ids = final_template.format(
            prompt=question_ids, memory=memory_ids
        ).tolist()
        final_turn = len(memory_ledger)
        final_seed = derive_turn_request_seeds([trajectory_seed], [0], final_turn)[0]
        stochastic = _generate_one(
            llm,
            final_prompt_ids,
            {
                **manifest["smsb"]["capture_final_stochastic_decode"],
                "seed": final_seed,
                "max_tokens": int(recurrent["max_final_tokens"]),
            },
            manifest=manifest,
        )
        stochastic_call_index = _GENERATE_CALL_COUNT
        deterministic = _generate_one(
            llm,
            final_prompt_ids,
            {
                **manifest["smsb"]["capture_final_deterministic_control"],
                "seed": final_seed,
                "max_tokens": int(recurrent["max_final_tokens"]),
            },
            manifest=manifest,
        )
        deterministic_call_index = _GENERATE_CALL_COUNT
        records.append(
            build_capture_record(
                {
                    **{field: pilot[field] for field in (
                        "example_id", "semantic_dataset_index", "source_order_index",
                        "raw_row_position", "production_effective_position",
                        "eval_manifest_hash", "source_question_hash", "source_context_hash",
                        "ground_truth_hash",
                    )},
                    "experiment_id": f"{manifest['run_id']}:smsb4",
                    "engine_id": engine_id,
                    "cache_namespace": cache_namespace,
                    "memory_ledger": memory_ledger,
                    "question_token_ids": question_ids,
                    "final_memory_token_ids": memory_ids,
                    "final_prompt_token_ids": final_prompt_ids,
                    "answer_token_ids": list(stochastic.token_ids),
                    "temperature_zero_control_answer_token_ids": list(deterministic.token_ids),
                    "sampling_params": {
                        **manifest["smsb"]["capture_final_stochastic_decode"],
                        "max_tokens": int(recurrent["max_final_tokens"]),
                        "do_sample": True,
                    },
                    "trajectory_seed": trajectory_seed,
                    "request_seed": final_seed,
                    "final_stochastic_request_seed": final_seed,
                    "final_control_request_seed": final_seed,
                    "final_stochastic_configured_request_seed": final_seed,
                    "final_stochastic_actual_request_seed": final_seed,
                    "final_control_configured_request_seed": final_seed,
                    "final_control_actual_request_seed": final_seed,
                    "final_stochastic_generate_call_index": stochastic_call_index,
                    "final_control_generate_call_index": deterministic_call_index,
                    "hashes": hashes,
                    "vllm_version": vllm.__version__,
                    "updater_calls": len(memory_ledger),
                    "prompt_template_sha256": resolved["execution_binding"][
                        "final_prompt_template_sha256"
                    ],
                    "runtime_binding_sha256": runtime_binding_sha,
                    "engine_config_sha256": engine_config_sha,
                    "current_binding_sha256": current_binding_sha,
                    "execution": evidence,
                    "ground_truth": ground_truth,
                }
            )
        )
    if _ENGINE_CONSTRUCTION_COUNT != 1 or _GENERATE_CALL_COUNT != expected_generate_calls:
        raise RuntimeError(
            "capture engine/generate counts differ from the precomputed frozen schedule: "
            f"engine={_ENGINE_CONSTRUCTION_COUNT}, generate={_GENERATE_CALL_COUNT}, "
            f"expected_generate={expected_generate_calls}"
        )
    write_jsonl_exclusive(output, records)
    return {
        "status": "PASS",
        "decision": "SMSB_CAPTURE_COMPLETE",
        "capture_count": len(records),
        "engine_id": engine_id,
        "artifact": str(output.resolve()),
    }


def replay_smsb(
    manifest_path: Path,
    captures_path: Path,
    *,
    example_id: str,
    regime: str,
    output: Path,
    credential_path: Path,
) -> dict[str, Any]:
    manifest, resolved, current_binding_sha, physical_gpu_identity = _runtime(
        manifest_path, full_model_sha=True
    )
    captures = read_jsonl(captures_path)
    selected = [row for row in captures if str(row["example_id"]) == str(example_id)]
    if len(selected) != 1:
        raise ValueError("example_id must identify exactly one SMSB capture")
    capture = selected[0]
    validate_capture_record(capture)
    credential_evidence = load_child_credential_claim(
        credential_path,
        manifest=manifest,
        resolved=resolved,
        current_binding_sha=current_binding_sha,
        child_kind="smsb_replay",
        child_identity=f"{example_id}::{regime}",
    )
    tokenizer = _tokenizer(manifest)
    from recurrent.impls.memory import TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template
    import vllm

    if vllm.__version__ != manifest["backend"]["required_version"]:
        raise ValueError("runtime vLLM version drifted")
    if capture.get("current_binding_sha256") != current_binding_sha:
        raise ValueError("SMSB capture current binding differs from this replay child")

    engine_id = f"smsb-replay-{regime}-{uuid.uuid4()}"
    cache_namespace = f"smsb-replay-cache-{regime}-{uuid.uuid4()}"
    independent_seed = (
        int(capture["request_seed"]) + 20000033 if regime == "independent_seed" else None
    )
    request = build_replay_request(
        capture,
        regime,
        replay_engine_id=engine_id,
        replay_cache_namespace=cache_namespace,
        independent_seed=independent_seed,
    )
    template_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
    if hashlib.sha256(template_text.encode("utf-8")).hexdigest() != request[
        "prompt_template_sha256"
    ]:
        raise RuntimeError("fresh replay prompt template hash mismatch")
    prompt_ids = TokenTemplate(template_text, tokenizer).format(
        prompt=request["question_token_ids"], memory=request["final_memory_token_ids"]
    ).tolist()
    if prompt_ids != request["expected_prompt_token_ids"]:
        raise RuntimeError("fresh replay L0 reconstruction failed before generation")
    params = dict(request["sampling_params"])
    params["seed"] = request["request_seed"]
    llm = _engine(manifest)
    completion = _generate_one(llm, prompt_ids, params, manifest=manifest)
    if _ENGINE_CONSTRUCTION_COUNT != 1 or _GENERATE_CALL_COUNT != 1:
        raise RuntimeError("SMSB replay must construct one engine and call generate once")
    answer_ids = list(completion.token_ids)
    result = {
        "request_id": request["request_id"],
        "engine_id": engine_id,
        "cache_namespace": cache_namespace,
        "fresh_engine_verified": True,
        "cache_isolation_verified": True,
        "single_request_execution_verified": True,
        "max_num_seqs": 1,
        "prefix_cache_enabled": False,
        "observed_updater_calls": 0,
        "context_or_chunks_visible": False,
        "prompt_reconstructed_from_serialized_state": True,
        "hashes": capture["hashes"],
        "vllm_version": vllm.__version__,
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "engine_config_sha256": resolved["execution_binding"]["engine_config_sha256"],
        "model_manifest_sha256": resolved["execution_binding"][
            "model_manifest_sha256"
        ],
        "prompt_token_ids": prompt_ids,
        "answer_token_ids": list(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "tensor_parallel_size": 2,
        "physical_gpu_whitelist": [2, 3],
        "physical_gpu_identity": physical_gpu_identity,
        "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
        "process_instance_uuid": PROCESS_INSTANCE_UUID,
        "process_pid": PROCESS_PID,
        "engine_construction_count": _ENGINE_CONSTRUCTION_COUNT,
        "generate_call_count": _GENERATE_CALL_COUNT,
        "configured_request_seed": request["request_seed"],
        "actual_request_seed": request["request_seed"],
        "current_binding_sha256": current_binding_sha,
        "prompt_token_ids_sha256": canonical_sha256(prompt_ids),
        "answer_token_ids_sha256": canonical_sha256(answer_ids),
        "full_model_sha_verified_at_child_start": True,
        **credential_evidence,
    }
    validation = validate_replay(capture, request, result)
    payload = {
        "capture_id": capture["capture_id"],
        "request": request,
        "result": result,
        "validation": validation,
    }
    write_json_exclusive(output, payload)
    return {
        "status": "PASS" if validation["execution_valid"] else "FAIL",
        "decision": "SMSB_REPLAY_VALID" if validation["execution_valid"] else "SMSB_REPLAY_INVALID",
        "example_id": example_id,
        "regime": regime,
        "L0": validation["L0_prompt_identity"],
        "L1": validation["L1_deterministic_answer_identity"],
        "L2": validation["L2_matched_seed_answer_identity"],
    }


def adjudicate_smsb(
    manifest_path: Path,
    captures_path: Path,
    replays_dir: Path,
    receipts_dir: Path,
    output: Path,
) -> dict[str, Any]:
    manifest, resolved, current_binding_sha, _ = _runtime(
        manifest_path, full_model_sha=False
    )
    captures = read_jsonl(captures_path)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(replays_dir.glob("*.json"))
    ]
    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(receipts_dir.glob("*.json"))
    ]
    capture_receipts = [
        receipt for receipt in receipts if receipt.get("child_kind") == "smsb_capture"
    ]
    replay_receipts = [
        receipt for receipt in receipts if receipt.get("child_kind") == "smsb_replay"
    ]
    report = summarize_smsb_pilot(
        captures,
        payloads,
        expected_examples=4,
        capture_receipt=(capture_receipts[0] if len(capture_receipts) == 1 else None),
        replay_receipts=replay_receipts,
        authority_secret=load_parent_authority_secret(manifest, resolved),
    )
    report["capture_sha256"] = hashlib.sha256(captures_path.read_bytes()).hexdigest()
    report["replay_artifact_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(replays_dir.glob("*.json"))
    }
    report["parent_receipt_artifact_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(receipts_dir.glob("*.json"))
    }
    report["training_authorized"] = False
    report["method_selection_status"] = "PENDING_EVIDENCE_NO_SELECTION"
    report["current_binding_sha256"] = current_binding_sha
    report["runtime_binding_sha256"] = resolved["runtime_binding_sha256"]
    report["execution_binding_sha256"] = resolved["execution_binding_sha256"]
    write_json_exclusive(output, report)
    return report


def prepare_tetrad(
    manifest_path: Path,
    captures_path: Path,
    smsb_report_path: Path,
    authoring_output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    from tools.h20.audit_qwen25_7b_serialization_credit import (
        authenticate_smsb_gate,
    )

    authenticated_smsb = authenticate_smsb_gate(manifest_path)
    if (
        authenticated_smsb.get("status") != "PASS"
        or authenticated_smsb.get("decision") != "SMSB_AUTHENTICATED_GATE_PASS"
    ):
        raise ValueError(
            "Tetrad is blocked because the full SMSB ledger/report authentication failed"
        )
    manifest, resolved, current_binding_sha, _ = _runtime(
        manifest_path, full_model_sha=True
    )
    smsb_report = json.loads(smsb_report_path.read_text(encoding="utf-8"))
    if (
        smsb_report.get("status") != "PASS"
        or smsb_report.get("decision") != manifest["tetrad"]["requires_smsb_decision"]
        or smsb_report.get("E_det_pass") is not True
    ):
        raise ValueError("Tetrad is blocked until the exact SMSB E_det gate passes")
    if smsb_report.get("current_binding_sha256") != current_binding_sha:
        raise ValueError("SMSB report current binding differs at Tetrad start")
    captures = read_jsonl(captures_path)
    if len(captures) != 4 or len({row["example_id"] for row in captures}) != 4:
        raise ValueError("Tetrad pilot requires four unique SMSB captures")
    for capture in captures:
        validate_capture_record(capture)
    tokenizer = _tokenizer(manifest)
    from recurrent.impls.memory import TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template

    template_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
    template_hash = hashlib.sha256(template_text.encode("utf-8")).hexdigest()
    template = TokenTemplate(template_text, tokenizer)
    rows = _parquet_rows(manifest["data"]["validation"])
    lengths = {str(row["example_id"]): len(row["final_memory_token_ids"]) for row in captures}
    matching = best_length_derangement(
        lengths,
        maximum_caliper=int(manifest["tetrad"]["maximum_shuffle_memory_token_caliper"]),
    )
    capture_by_id = {str(row["example_id"]): row for row in captures}
    authoring: list[dict[str, Any]] = []
    for example_id in sorted(capture_by_id):
        capture = capture_by_id[example_id]
        source = rows[int(capture["raw_row_position"])]
        question = _question(source)
        context = source.get("context")
        raw_answers = source.get("reward_model", {}).get("ground_truth")
        if not isinstance(context, str):
            raise ValueError(f"source context is not text for {example_id}")
        if (
            not isinstance(raw_answers, list)
            or not raw_answers
            or any(not isinstance(value, str) or not value.strip() for value in raw_answers)
        ):
            raise ValueError(f"source ground truth is not strict text for {example_id}")
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != capture[
            "source_question_hash"
        ]:
            raise ValueError(f"source question hash mismatch for {example_id}")
        if hashlib.sha256(context.encode("utf-8")).hexdigest() != capture[
            "source_context_hash"
        ]:
            raise ValueError(f"source context hash mismatch for {example_id}")
        if canonical_sha256(raw_answers) != capture["ground_truth_hash"]:
            raise ValueError(f"source ground-truth hash mismatch for {example_id}")
        question_ids = list(capture["question_token_ids"])
        if tokenizer.encode(question, add_special_tokens=False) != question_ids:
            raise ValueError(f"question-token identity mismatch for {example_id}")
        answers = list(raw_answers)
        answer = answers[0].strip().strip('"')
        answer_norms = [value.strip().strip('"').lower() for value in answers]
        if not answer or any(not value for value in answer_norms):
            raise ValueError(f"ground truth contains an empty answer for {example_id}")
        answer_norm = answer_norms[0]
        question_words = content_words(question)
        documents = split_documents(context)
        for document in documents:
            haystack = f"{document['title']} {document['text']}".lower()
            document["answer_hit"] = any(
                candidate in haystack for candidate in answer_norms
            )
            document["question_overlap"] = len(question_words & content_words(haystack))
        answer_documents = sorted(
            (document for document in documents if document["answer_hit"]),
            key=lambda document: (-document["question_overlap"], document["number"]),
        )
        anchor_documents = sorted(
            (document for document in documents if not document["answer_hit"]),
            key=lambda document: (-document["question_overlap"], document["number"]),
        )
        selected_gold: list[dict[str, Any]] = []
        if answer_documents:
            selected_gold.append(answer_documents[0])
        if anchor_documents and anchor_documents[0]["question_overlap"] > 0:
            selected_gold.append(anchor_documents[0])
        evidence = "\n\n".join(
            f"{document['title']}: {document['text']}" for document in selected_gold
        )
        gold_text = f"Canonical answer: {answer}.\nCanonical evidence:\n{evidence}"
        gold_ids = tokenizer.encode(gold_text, add_special_tokens=False)[: lengths[example_id]]
        if not gold_ids:
            raise ValueError(f"canonical gold state is empty for {example_id}")
        gold_rendered = tokenizer.decode(gold_ids, skip_special_tokens=False).lower()
        if answer_norm not in gold_rendered:
            raise ValueError(
                f"canonical answer was removed by gold-state token truncation for {example_id}"
            )
        distractors = [
            document
            for document in documents
            if not document["answer_hit"]
            and document["question_overlap"] == 0
            and document["number"] not in {item["number"] for item in selected_gold}
        ]
        irrelevant_parts: list[str] = []
        irrelevant_titles: list[str] = []
        irrelevant_ids: list[int] = []
        for document in sorted(distractors, key=lambda item: item["number"]):
            irrelevant_titles.append(document["title"])
            irrelevant_parts.append(f"{document['title']}: {document['text']}")
            irrelevant_ids = tokenizer.encode(
                "\n\n".join(irrelevant_parts), add_special_tokens=False
            )
            if len(irrelevant_ids) >= lengths[example_id]:
                break
        if len(irrelevant_ids) < lengths[example_id]:
            raise ValueError(f"not enough legal irrelevant tokens for {example_id}")
        irrelevant_ids = irrelevant_ids[: lengths[example_id]]
        irrelevant_text = tokenizer.decode(irrelevant_ids, skip_special_tokens=False).lower()
        if any(
            candidate in irrelevant_text for candidate in answer_norms
        ) or question_words & content_words(irrelevant_text):
            raise ValueError(f"irrelevant leakage audit failed for {example_id}")
        common = {
            **{field: capture[field] for field in (
                "example_id", "semantic_dataset_index", "source_order_index",
                "raw_row_position", "production_effective_position", "eval_manifest_hash",
                "source_question_hash", "source_context_hash", "ground_truth_hash",
            )},
            "question": question,
            "question_token_ids": question_ids,
            "ground_truth": answers,
            "question_type": "hotpot_multihop",
            "answer_type": "hotpot_short_span",
            "checkpoint_hash": capture["hashes"]["model"],
            "model_hash": capture["hashes"]["model"],
            "tokenizer_hash": capture["hashes"]["tokenizer"],
            "hashes": dict(capture["hashes"]),
            "vllm_version": capture["vllm_version"],
            "runtime_binding_sha256": resolved["runtime_binding_sha256"],
            "engine_config_sha256": resolved["execution_binding"]["engine_config_sha256"],
            "current_binding_sha256": current_binding_sha,
            "full_model_sha_verified_at_tetrad_start": True,
            "prompt_protocol_hash": template_hash,
            "prompt_outside_memory_span_hash": canonical_sha256(
                {"template": template_hash, "question_token_ids": question_ids}
            ),
            "physical_gpu_identity": list(capture["execution"]["physical_gpu_identity"]),
            "cuda_device_order": capture["execution"]["cuda_device_order"],
            "generated": {
                "state_id": f"{example_id}:generated",
                "memory_token_ids": list(capture["final_memory_token_ids"]),
                "validity_status": "pass",
                "smsb_status": "pass",
            },
            "empty": {
                "state_id": f"{example_id}:empty", "memory_token_ids": [],
                "validity_status": "pass",
            },
            "irrelevant": {
                "state_id": f"{example_id}:within-example-distractor",
                "memory_token_ids": irrelevant_ids,
                "validity_status": "pass",
                "support_answer_bridge_leakage_audit": "pass",
                "length_match_audit": "pass",
                "selected_document_titles": irrelevant_titles,
                "audit_definition": "no_ground_truth_substring_and_zero_frozen_question_keyword_overlap",
            },
            "gold": {
                "state_id": f"{example_id}:canonical-positive-control",
                "memory_token_ids": gold_ids,
                "validity_status": "pass",
                "canonical_authoring_audit": "pass",
                "source_mode": (
                    "answer_document_plus_question_anchor"
                    if answer_documents
                    else "answer_control_fallback"
                ),
                "selected_document_titles": [item["title"] for item in selected_gold],
                "contains_ground_truth_by_design": True,
            },
            "shuffle_approved_donor_ids": [matching[example_id]],
            "shuffle_memory_token_delta": abs(lengths[example_id] - lengths[matching[example_id]]),
            "generated_memory_token_length": lengths[example_id],
            "gold_memory_token_length": len(gold_ids),
            "irrelevant_memory_token_length": len(irrelevant_ids),
        }
        authoring.append(common)

    def prompt_builder(question_ids: list[int], memory_ids: list[int]) -> list[int]:
        return template.format(prompt=question_ids, memory=memory_ids).tolist()

    requests = build_tetrad_requests(
        authoring,
        matching=matching,
        base_seed=int(manifest["backend"]["engine_seed"]),
        prompt_builder=prompt_builder,
        prompt_template_sha256=template_hash,
        capture_prompt_ids={
            example_id: capture["final_prompt_token_ids"]
            for example_id, capture in capture_by_id.items()
        },
    )
    validation = validate_tetrad_manifest(requests)
    write_jsonl_exclusive(authoring_output, authoring)
    write_jsonl_exclusive(manifest_output, requests)
    return {
        "status": "PASS",
        "decision": validation["decision"],
        "authoring": str(authoring_output.resolve()),
        "manifest": str(manifest_output.resolve()),
        "request_count": len(requests),
        "matching": matching,
        "effects_reportable": False,
        "training_authorized": False,
        "method_selection_status": "PENDING_EVIDENCE_NO_SELECTION",
        "current_binding_sha256": current_binding_sha,
    }


def run_tetrad_request(
    manifest_path: Path,
    tetrad_manifest_path: Path,
    *,
    request_id: str,
    output: Path,
    credential_path: Path,
) -> dict[str, Any]:
    manifest, resolved, current_binding_sha, physical_gpu_identity = _runtime(
        manifest_path, full_model_sha=True
    )
    rows = read_jsonl(tetrad_manifest_path)
    validate_tetrad_manifest(rows)
    selected = [row for row in rows if row["request_id"] == request_id]
    if len(selected) != 1:
        raise ValueError("request_id must identify exactly one Tetrad request")
    request = selected[0]
    credential_evidence = load_child_credential_claim(
        credential_path,
        manifest=manifest,
        resolved=resolved,
        current_binding_sha=current_binding_sha,
        child_kind="tetrad_replay",
        child_identity=request_id,
    )
    tokenizer = _tokenizer(manifest)
    from recurrent.impls.memory import TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template
    import vllm

    if vllm.__version__ != manifest["backend"]["required_version"]:
        raise ValueError("Tetrad runtime vLLM version drifted")
    if request.get("vllm_version") != vllm.__version__:
        raise ValueError("Tetrad request vLLM binding differs from runtime")
    if request.get("current_binding_sha256") != current_binding_sha:
        raise ValueError("Tetrad request current binding differs from runtime")

    template_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
    if hashlib.sha256(template_text.encode("utf-8")).hexdigest() != request[
        "prompt_template_sha256"
    ]:
        raise RuntimeError("Tetrad prompt template hash mismatch")
    prompt_ids = TokenTemplate(template_text, tokenizer).format(
        prompt=request["question_token_ids"], memory=request["memory_token_ids"]
    ).tolist()
    if prompt_ids != request["expected_prompt_token_ids"]:
        raise RuntimeError("Tetrad prompt reconstruction identity failed")
    engine_id = f"tetrad-{request['state_role']}-{uuid.uuid4()}"
    llm = _engine(manifest)
    completion = _generate_one(
        llm,
        prompt_ids,
        {
            "n": 1, "best_of": 1, "temperature": 0.0, "top_p": 1.0,
            "top_k": -1, "min_p": 0.0, "seed": int(request["request_seed"]),
            "max_tokens": int(manifest["recurrent"]["max_final_tokens"]),
        },
        manifest=manifest,
    )
    if _ENGINE_CONSTRUCTION_COUNT != 1 or _GENERATE_CALL_COUNT != 1:
        raise RuntimeError("Tetrad child must construct one engine and call generate once")
    text = tokenizer.decode(completion.token_ids, skip_special_tokens=False)
    metrics = score_terminal_output(text, request["ground_truth"])
    answer_ids = list(completion.token_ids)
    result = {
        "schema": "memagent.tetrad.pilot4.single-request.result.v2",
        "request_id": request["request_id"],
        "example_id": request["example_id"],
        "state_role": request["state_role"],
        "engine_id": engine_id,
        "cache_namespace": request["cache_namespace"],
        "fresh_engine_verified": True,
        "single_request_execution_verified": True,
        "tensor_parallel_size": 2,
        "physical_gpu_whitelist": [2, 3],
        "physical_gpu_identity": physical_gpu_identity,
        "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
        "max_num_seqs": 1,
        "prefix_cache_enabled": False,
        "observed_updater_calls": 0,
        "context_or_chunks_visible": False,
        "prompt_reconstructed_from_question_and_serialized_memory": True,
        "prompt_token_ids": prompt_ids,
        "prompt_token_sha256": canonical_sha256(prompt_ids),
        "prompt_token_ids_sha256": canonical_sha256(prompt_ids),
        "memory_token_length": len(request["memory_token_ids"]),
        "answer_token_ids": answer_ids,
        "answer_token_ids_sha256": canonical_sha256(answer_ids),
        "answer_text": text,
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "score": float(metrics["token_f1"]),
        "exact_match": float(metrics["exact_match"]),
        "format_success": float(metrics["format_success"]),
        "extraction_route": metrics["extraction_route"],
        "construction_only_pilot": True,
        "effects_reportable": False,
        "process_instance_uuid": PROCESS_INSTANCE_UUID,
        "process_pid": PROCESS_PID,
        "engine_construction_count": _ENGINE_CONSTRUCTION_COUNT,
        "generate_call_count": _GENERATE_CALL_COUNT,
        "configured_request_seed": request["request_seed"],
        "actual_request_seed": request["request_seed"],
        "vllm_version": vllm.__version__,
        "hashes": dict(request["hashes"]),
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "engine_config_sha256": resolved["execution_binding"]["engine_config_sha256"],
        "model_manifest_sha256": resolved["execution_binding"][
            "model_manifest_sha256"
        ],
        "current_binding_sha256": current_binding_sha,
        "full_model_sha_verified_at_child_start": True,
        **credential_evidence,
    }
    write_json_exclusive(output, result)
    return {
        "status": "PASS",
        "decision": "TETRAD_REQUEST_EXECUTION_VALID",
        "request_id": request_id,
        "score": result["score"],
        "exact_match": result["exact_match"],
    }


def list_tetrad_requests(manifest_path: Path, tetrad_manifest_path: Path) -> None:
    """Emit a strict three-column index for the shell's one-process-per-request loop."""
    _runtime(manifest_path, full_model_sha=False)
    rows = read_jsonl(tetrad_manifest_path)
    validate_tetrad_manifest(rows)
    for row in rows:
        fields = [row.get("request_id"), row.get("example_id"), row.get("state_role")]
        if any(
            not isinstance(value, str)
            or not value
            or any(character in value for character in ("\t", "\r", "\n"))
            for value in fields
        ):
            raise ValueError("Tetrad request index contains an unsafe/non-string field")
        print("\t".join(fields))


def adjudicate_tetrad(
    manifest_path: Path,
    tetrad_manifest_path: Path,
    authoring_path: Path,
    results_dir: Path,
    receipts_dir: Path,
    output: Path,
) -> dict[str, Any]:
    manifest, resolved, current_binding_sha, _ = _runtime(
        manifest_path, full_model_sha=False
    )
    manifest_rows = read_jsonl(tetrad_manifest_path)
    authoring_rows = read_jsonl(authoring_path)
    result_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results_dir.glob("*.json"))
    ]
    parent_receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(receipts_dir.glob("*.json"))
    ]
    tokenizer = _tokenizer(manifest)
    report = adjudicate_tetrad_pilot(
        manifest_rows,
        authoring_rows,
        result_rows,
        answer_decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        parent_receipts=parent_receipts,
        authority_secret=load_parent_authority_secret(manifest, resolved),
        competence_score_threshold=float(
            manifest["tetrad"]["canonical_competence_score_threshold_f1"]
        ),
        competence_rate_floor=float(
            manifest["tetrad"]["canonical_competence_rate_floor"]
        ),
    )
    report["manifest_sha256"] = hashlib.sha256(tetrad_manifest_path.read_bytes()).hexdigest()
    report["authoring_sha256"] = hashlib.sha256(authoring_path.read_bytes()).hexdigest()
    report["result_artifact_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(results_dir.glob("*.json"))
    }
    report["parent_receipt_artifact_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(receipts_dir.glob("*.json"))
    }
    report["current_binding_sha256"] = current_binding_sha
    report["runtime_binding_sha256"] = resolved["runtime_binding_sha256"]
    report["execution_binding_sha256"] = resolved["execution_binding_sha256"]
    write_json_exclusive(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_REL)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture-smsb")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--credential", type=Path, required=True)
    replay = subparsers.add_parser("replay-smsb")
    replay.add_argument("--captures", type=Path, required=True)
    replay.add_argument("--example-id", required=True)
    replay.add_argument("--regime", choices=("temperature_zero", "matched_seed", "independent_seed"), required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--credential", type=Path, required=True)
    adjudicate_s = subparsers.add_parser("adjudicate-smsb")
    adjudicate_s.add_argument("--captures", type=Path, required=True)
    adjudicate_s.add_argument("--replays-dir", type=Path, required=True)
    adjudicate_s.add_argument("--receipts-dir", type=Path, required=True)
    adjudicate_s.add_argument("--output", type=Path, required=True)
    prepare_t = subparsers.add_parser("prepare-tetrad")
    prepare_t.add_argument("--captures", type=Path, required=True)
    prepare_t.add_argument("--smsb-report", type=Path, required=True)
    prepare_t.add_argument("--authoring-output", type=Path, required=True)
    prepare_t.add_argument("--manifest-output", type=Path, required=True)
    run_t = subparsers.add_parser("run-tetrad-request")
    run_t.add_argument("--tetrad-manifest", type=Path, required=True)
    run_t.add_argument("--request-id", required=True)
    run_t.add_argument("--output", type=Path, required=True)
    run_t.add_argument("--credential", type=Path, required=True)
    list_t = subparsers.add_parser("list-tetrad-requests")
    list_t.add_argument("--tetrad-manifest", type=Path, required=True)
    adjudicate_t = subparsers.add_parser("adjudicate-tetrad")
    adjudicate_t.add_argument("--tetrad-manifest", type=Path, required=True)
    adjudicate_t.add_argument("--authoring", type=Path, required=True)
    adjudicate_t.add_argument("--results-dir", type=Path, required=True)
    adjudicate_t.add_argument("--receipts-dir", type=Path, required=True)
    adjudicate_t.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture-smsb":
        report = capture_smsb(
            args.manifest, args.output, credential_path=args.credential
        )
    elif args.command == "replay-smsb":
        report = replay_smsb(
            args.manifest, args.captures,
            example_id=args.example_id,
            regime=args.regime,
            output=args.output,
            credential_path=args.credential,
        )
    elif args.command == "adjudicate-smsb":
        report = adjudicate_smsb(
            args.manifest,
            args.captures,
            args.replays_dir,
            args.receipts_dir,
            args.output,
        )
    elif args.command == "prepare-tetrad":
        report = prepare_tetrad(
            args.manifest, args.captures, args.smsb_report,
            args.authoring_output, args.manifest_output,
        )
    elif args.command == "run-tetrad-request":
        report = run_tetrad_request(
            args.manifest, args.tetrad_manifest,
            request_id=args.request_id,
            output=args.output,
            credential_path=args.credential,
        )
    elif args.command == "list-tetrad-requests":
        list_tetrad_requests(args.manifest, args.tetrad_manifest)
        return 0
    else:
        report = adjudicate_tetrad(
            args.manifest, args.tetrad_manifest, args.authoring,
            args.results_dir, args.receipts_dir, args.output,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
