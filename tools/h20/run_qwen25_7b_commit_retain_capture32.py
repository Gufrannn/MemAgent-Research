#!/usr/bin/env python3
"""One-process, one-engine strict-vLLM producer for exact capture32."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from recurrent.research.serialization_credit_pilots import read_jsonl, write_json_exclusive  # noqa: E402
from recurrent.research.trajectory_seeding import derive_turn_request_seeds  # noqa: E402
import tools.h20.run_qwen25_7b_commit_retain as base  # noqa: E402
from tools.h20.preflight_qwen25_7b_commit_retain_capture32 import (  # noqa: E402
    EXPERIMENT_NAME,
    MANIFEST_REL,
    _current_binding,
    _expected_run_receipt,
    _gpu_identity,
    consume_credential_and_record_start,
    execution_frozen_pairs,
    expected_git_commit,
    expected_pair_binding,
    load_manifest,
    project_frozen_pair_eval_identity,
    validate_p0,
)


PROCESS_INSTANCE_UUID = str(uuid.uuid4())
PROCESS_PID = os.getpid()


_PAIR_IDENTITY_FIELDS = (
    "example_id", "semantic_dataset_index", "source_order_index",
    "raw_row_position", "production_effective_position", "eval_manifest_hash",
    "source_question_hash", "source_context_hash", "ground_truth_hash",
)


def _pair_identity_from_frozen(
    frozen_row: Mapping[str, Any], *, eval_manifest_hash: str
) -> dict[str, Any]:
    """Project the authenticated manifest-level hash into one execution row."""
    if not isinstance(eval_manifest_hash, str) or len(eval_manifest_hash) != 64 \
            or any(character not in "0123456789abcdef" for character in eval_manifest_hash):
        raise ValueError("capture32 eval_manifest_hash is not a canonical SHA-256")
    row_copy = frozen_row.get("eval_manifest_hash")
    if row_copy is not None and row_copy != eval_manifest_hash:
        raise ValueError("capture32 row-level eval_manifest_hash conflicts with P0")
    return {
        field: eval_manifest_hash if field == "eval_manifest_hash" else frozen_row[field]
        for field in _PAIR_IDENTITY_FIELDS
    }


def _runtime(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]:
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    visible = manifest["gpu"]["visible_devices"]
    expected_env = {
        "CUDA_VISIBLE_DEVICES": visible,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VLLM_USE_V1": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    for field, expected in expected_env.items():
        if os.environ.get(field) != expected:
            raise ValueError(f"capture32 {field} must be exactly {expected}")
    current_sha = _current_binding(manifest, resolved, full_model_sha=True)
    identities = _gpu_identity(manifest["gpu"]["physical_whitelist"])
    if identities != resolved["runtime_binding"]["physical_gpu_identity"]:
        raise ValueError("capture32 physical GPU UUID/PCI/name binding differs from P0")
    return manifest, resolved, current_sha, identities


def capture(manifest_path: Path, *, credential_path: Path) -> dict[str, Any]:
    if base._ENGINE_CONSTRUCTION_COUNT or base._GENERATE_CALL_COUNT:
        raise RuntimeError("capture32 runner generation globals were already used")
    manifest, resolved, current_sha, physical_gpu_identity = _runtime(manifest_path)
    # Atomic credential consumption and capture_started ledger append both
    # happen before torch/vLLM are imported or the engine is constructed.
    credential = consume_credential_and_record_start(
        manifest_path, credential_path=credential_path
    )
    capture_path = Path(manifest["paths"]["capture_ledger"])
    receipt_path = Path(manifest["paths"]["capture_run_receipt"])
    if capture_path.exists() or capture_path.is_symlink() \
            or receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("refuse to resume/overwrite a capture32 ledger")

    tokenizer = base._tokenizer(manifest)
    from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
    from recurrent.utils import TokenTemplate, chat_template
    import torch
    import vllm

    if vllm.__version__ != "0.8.2":
        raise ValueError("capture32 runtime must be vLLM 0.8.2")
    writer_text = chat_template(tokenizer).format(message=TEMPLATE)
    reader_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
    if hashlib.sha256(writer_text.encode()).hexdigest() != resolved["expected_pair_binding"]["writer_prompt_template_sha256"] \
            or hashlib.sha256(reader_text.encode()).hexdigest() != resolved["expected_pair_binding"]["reader_prompt_template_sha256"]:
        raise ValueError("capture32 runtime prompt templates differ from P0")
    worker = base._worker_multiprocessing_evidence(
        manifest, parent_cuda_initialized=bool(torch.cuda.is_initialized())
    )
    writer_template = TokenTemplate(writer_text, tokenizer)
    reader_template = TokenTemplate(reader_text, tokenizer)
    no_memory_ids = list(tokenizer.encode(
        manifest["recurrent"]["no_memory_text"], add_special_tokens=False
    ))
    no_memory_state = build_state_blob(no_memory_ids)
    frozen = execution_frozen_pairs(manifest, tokenizer)
    sources = base._source_inputs(manifest, {"frozen_pairs": frozen}, tokenizer)
    expected_calls = int(resolved["expected_global_generate_call_count"])
    if len(sources) != 32 or expected_calls != 353 \
            or sum(item[0]["expected_pair_generate_calls"] for item in sources) != 353:
        raise ValueError("capture32 exact 32/353 generation schedule drifted")

    llm = base._engine(manifest)
    engine_id = f"commit-retain-capture32-engine-{uuid.uuid4()}"
    cache_namespace = f"capture32-no-prefix-cache-{uuid.uuid4()}"
    execution = {
        "backend": "vllm", "vllm_version": vllm.__version__, "strict_vllm": True,
        "tensor_parallel_size": 2,
        "physical_gpu_whitelist": manifest["gpu"]["physical_whitelist"],
        "physical_gpu_identity": physical_gpu_identity,
        "visible_devices": manifest["gpu"]["visible_devices"],
        "cuda_device_order": "PCI_BUS_ID", "prefix_cache_enabled": False,
        "max_num_seqs": 1, "one_prompt_per_generate_call": True,
        "engine_construction_count": 1, "full_model_sha_verified_at_capture_start": True,
        "trainer_attached": False, "actor_training_calls": 0,
        "engine_id": engine_id, "cache_namespace": cache_namespace,
        "process_instance_uuid": PROCESS_INSTANCE_UUID, "process_pid": PROCESS_PID,
        "global_generate_call_count": expected_calls,
        "engine_config_sha256": resolved["expected_pair_binding"]["engine_config_sha256"],
        **worker, **credential,
    }
    checkpoint_sha = resolved["model_file_manifest_sha256"]
    writer_decode = dict(manifest["intervention"]["writer_decode"])
    reader_decode = dict(manifest["intervention"]["reader_decode"])

    for frozen_row, source, question_ids, chunks in sources:
        trajectory_seed = int(frozen_row["trajectory_seed"])
        intervention = int(frozen_row["intervention_writer_turn"])
        total = int(frozen_row["total_writer_turns"])
        stable_write = frozen_row["stable_write_id"]
        prefix: list[dict[str, Any]] = []
        memory_state = no_memory_state
        source_role, source_turn_id = "no_memory", None
        for turn in range(intervention):
            seed = derive_turn_request_seeds([trajectory_seed], [0], turn)[0]
            generation = base._writer_call(
                llm=llm, tokenizer=tokenizer, writer_template=writer_template,
                question_ids=question_ids, memory_state=memory_state,
                loaded_receipt=base._loaded_state(source_role, source_turn_id, memory_state),
                chunk_ids=chunks[turn], sampling=writer_decode, request_seed=seed,
                writer_template_sha256=resolved["expected_pair_binding"]["writer_prompt_template_sha256"],
                checkpoint_sha256=checkpoint_sha,
            )
            source_turn_id = stable_turn_id(
                stable_write_id=stable_write, phase="prefix_writer", arm="SHARED", writer_turn=turn
            )
            memory_state, source_role = generation["state_after"], "previous_prefix_output"
            prefix.append(generation)
        old_state, old_turn_id = memory_state, source_turn_id
        candidate_seed = derive_turn_request_seeds([trajectory_seed], [0], intervention)[0]
        candidate = base._writer_call(
            llm=llm, tokenizer=tokenizer, writer_template=writer_template,
            question_ids=question_ids, memory_state=old_state,
            loaded_receipt=base._loaded_state("old_state", old_turn_id, old_state),
            chunk_ids=chunks[intervention], sampling=writer_decode,
            request_seed=candidate_seed,
            writer_template_sha256=resolved["expected_pair_binding"]["writer_prompt_template_sha256"],
            checkpoint_sha256=checkpoint_sha,
        )
        candidate_state = candidate["state_after"]
        candidate_turn_id = stable_turn_id(
            stable_write_id=stable_write, phase="candidate_writer", arm="SHARED",
            writer_turn=intervention,
        )
        if candidate_state["bytes_sha256"] == old_state["bytes_sha256"]:
            raise RuntimeError(f"capture32 candidate equals old state: {stable_write}")

        arms: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            arm_state = candidate_state if arm == "COMMIT" else old_state
            initial_role = "candidate" if arm == "COMMIT" else "old_state"
            initial_turn_id = candidate_turn_id if arm == "COMMIT" else old_turn_id
            initial = base._loaded_state(initial_role, initial_turn_id, arm_state)
            future: list[dict[str, Any]] = []
            loaded_role, loaded_turn = initial_role, initial_turn_id
            for turn in range(intervention + 1, total):
                seed = derive_turn_request_seeds([trajectory_seed], [0], turn)[0]
                generation = base._writer_call(
                    llm=llm, tokenizer=tokenizer, writer_template=writer_template,
                    question_ids=question_ids, memory_state=arm_state,
                    loaded_receipt=base._loaded_state(loaded_role, loaded_turn, arm_state),
                    chunk_ids=chunks[turn], sampling=writer_decode, request_seed=seed,
                    writer_template_sha256=resolved["expected_pair_binding"]["writer_prompt_template_sha256"],
                    checkpoint_sha256=checkpoint_sha,
                )
                loaded_turn = stable_turn_id(
                    stable_write_id=stable_write, phase="future_writer", arm=arm,
                    writer_turn=turn,
                )
                loaded_role, arm_state = "previous_future_output", generation["state_after"]
                future.append(generation)
            reader_seed = derive_turn_request_seeds([trajectory_seed], [0], total)[0]
            final = base._reader_call(
                llm=llm, tokenizer=tokenizer, reader_template=reader_template,
                question_ids=question_ids, memory_state=arm_state,
                loaded_receipt=base._loaded_state("previous_future_output", loaded_turn, arm_state),
                sampling=reader_decode, request_seed=reader_seed,
                reader_template_sha256=resolved["expected_pair_binding"]["reader_prompt_template_sha256"],
                checkpoint_sha256=checkpoint_sha,
            )
            arms[arm] = {"initial_loaded_state_receipt": initial,
                         "future_turns": future, "final_reader": final}

        future_turns = list(range(intervention + 1, total))
        shared_contract = {
            "intervention_writer_turn": intervention, "total_writer_turns": total,
            "trajectory_seed": trajectory_seed,
            "future_chunks": [{"writer_turn": turn, "token_ids": chunks[turn]}
                              for turn in future_turns],
            "horizon": {"future_writer_turns": future_turns,
                        "future_writer_calls_per_arm": len(future_turns),
                        "final_reader_calls_per_arm": 1, "terminal_writer_turn": total - 1},
            "writer_checkpoint_sha256": checkpoint_sha,
            "reader_checkpoint_sha256": checkpoint_sha,
            "writer_prompt_template_sha256": resolved["expected_pair_binding"]["writer_prompt_template_sha256"],
            "reader_prompt_template_sha256": resolved["expected_pair_binding"]["reader_prompt_template_sha256"],
            "writer_decode": writer_decode, "reader_decode": reader_decode,
            "cache_contract": {"enable_prefix_caching": False, "max_num_seqs": 1,
                               "one_prompt_per_generate_call": True,
                               "kv_state_reuse_across_generate_calls": False,
                               "same_engine_for_both_arms": True},
            "cost_contract": {"shared_candidate_generation_calls": 1,
                              "per_arm_writer_generation_calls": len(future_turns),
                              "per_arm_reader_generation_calls": 1,
                              "per_arm_total_generation_calls": len(future_turns) + 1,
                              "per_arm_writer_max_tokens": writer_decode["max_tokens"],
                              "per_arm_reader_max_tokens": reader_decode["max_tokens"],
                              "budgets_identical_by_design": True,
                              "realized_token_counts_are_measured_not_forced_equal": True},
        }
        pair = build_pair_record({
            **_pair_identity_from_frozen(
                frozen_row, eval_manifest_hash=resolved["eval_manifest_hash"]
            ),
            "trajectory_seed": trajectory_seed, "intervention_writer_turn": intervention,
            "total_writer_turns": total, "question_token_ids": question_ids,
            "ground_truth": [str(item) for item in source["reward_model"]["ground_truth"]],
            "no_memory_state": no_memory_state, "prefix_turns": prefix,
            "old_state": old_state, "candidate": candidate,
            "shared_contract": shared_contract, "arms": arms, "execution": execution,
        })
        if pair["stable_write_id"] != stable_write:
            raise RuntimeError("capture32 runtime stable write differs from P0")
        append_jsonl(capture_path, build_capture_envelope(
            pair, experiment_name=EXPERIMENT_NAME, git_commit=expected_git_commit(),
            run_id=manifest["run_id"],
            execution_binding_sha256=resolved["execution_binding_sha256"],
            runtime_binding_sha256=resolved["runtime_binding_sha256"],
            current_binding_sha256=current_sha,
        ))

    if base._ENGINE_CONSTRUCTION_COUNT != 1 or base._GENERATE_CALL_COUNT != 353:
        raise RuntimeError("capture32 engine/generate count differs from exact 1/353")
    runtime_expected = {**expected_pair_binding(manifest, resolved, tokenizer), **{
        key: credential[key] for key in (
            "gpu_lock_binding_sha256", "lock_holder_receipt_sha256",
            "credential_consumption_sha256", "credential_consumption_file_sha256",
            "credential_consumption_path",
        )
    }}
    validation_frozen = project_frozen_pair_eval_identity(
        frozen, resolved["eval_manifest_hash"]
    )
    report = validate_capture_ledger(
        read_jsonl(capture_path), frozen_pairs=validation_frozen,
        experiment_name=EXPERIMENT_NAME,
        git_commit=expected_git_commit(), run_id=manifest["run_id"],
        execution_binding_sha256=resolved["execution_binding_sha256"],
        runtime_binding_sha256=resolved["runtime_binding_sha256"],
        current_binding_sha256=current_sha,
        decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
        writer_prompt_builder=lambda q, m, c: writer_template.format(prompt=q, memory=m, chunk=c).tolist(),
        reader_prompt_builder=lambda q, m: reader_template.format(prompt=q, memory=m).tolist(),
        expected_pair_binding=runtime_expected, expected_pair_count=32,
    )
    receipt = _expected_run_receipt(
        manifest=manifest, resolved=resolved, current_binding_sha256=current_sha,
        capture_report=report, capture_path=capture_path,
    )
    write_json_exclusive(receipt_path, receipt)
    return {**report, "capture_ledger": str(capture_path.resolve()),
            "capture_ledger_sha256": sha256_file(capture_path),
            "capture_run_receipt": str(receipt_path.resolve()),
            "capture_run_receipt_sha256": sha256_file(receipt_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_REL)
    parser.add_argument("capture", nargs="?")
    parser.add_argument("--credential", type=Path, required=True)
    args = parser.parse_args()
    if args.capture not in (None, "capture"):
        parser.error("only exact full capture32 is supported; no subset/resume/input-ledger")
    report = capture(args.manifest, credential_path=args.credential)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
