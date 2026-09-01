#!/usr/bin/env python3
"""Prepare a P33.2 UMA credit-trace instrumented worktree.

This script makes a copy of an official UMA checkout and applies a deliberately
small, default-off instrumentation patch.  It does not change prompts, memory
operators, datasets, rewards, metrics, generation parameters, reward formulas,
or optimizer math.  When ``UMA_CREDIT_TRACE`` is unset, the inserted code is a
no-op.  When ``UMA_CREDIT_TRACE_EXIT_AFTER_ADVANTAGE`` is explicitly enabled
together with tracing, the trainer returns immediately after true GRPO
advantages have been computed and traced, before critic/actor optimizer steps.

The script is intentionally exact-anchor based: if the upstream code around a
patch site changes, preparation fails closed instead of silently applying a
nearby edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


TRACE_MODULE = r'''"""Default-off JSONL tracing for UMA credit-flow anatomy.

This module is intentionally dependency-light and side-effect-free when
``UMA_CREDIT_TRACE`` is unset.  It is for P33.2 instrumentation only.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def credit_trace_enabled() -> bool:
    return str(os.getenv("UMA_CREDIT_TRACE", "")).strip().lower() in TRUE_VALUES


def credit_trace_path() -> Path:
    base = Path(os.getenv("UMA_CREDIT_TRACE_PATH", "uma_credit_trace.jsonl")).expanduser()
    host = socket.gethostname().replace("/", "_")
    pid = os.getpid()
    if base.exists() and base.is_dir():
        return base / f"uma_credit_trace.{host}.{pid}.jsonl"
    suffix = base.suffix or ".jsonl"
    return base.with_name(f"{base.stem}.{host}.{pid}{suffix}")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return [_json_safe(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
    except Exception:
        pass
    return str(value)


def stable_hash_payload(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def trace_event(event: str, payload: dict[str, Any]) -> None:
    if not credit_trace_enabled():
        return
    path = credit_trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": time.time(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "event": event,
        "payload": _json_safe(payload),
    }
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
'''


def run(cmd: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Patch anchor {label!r} expected once, found {count}")
    return text.replace(old, new, 1)


def patch_agent_loop(repo: Path) -> list[dict[str, str]]:
    path = repo / "external/verl/verl/experimental/agent_loop/agent_loop.py"
    before = path.read_text(encoding="utf-8")
    text = before
    text = replace_once(
        text,
        "from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class\n",
        "from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class\n"
        "from verl.utils.uma_credit_trace import credit_trace_enabled as uma_credit_trace_enabled\n"
        "from verl.utils.uma_credit_trace import stable_hash_payload as uma_credit_stable_hash_payload\n"
        "from verl.utils.uma_credit_trace import trace_event as uma_credit_trace_event\n",
        "agent_loop_import_trace",
    )
    text = replace_once(
        text,
        "        for i in range(len(batch)):\n"
        "            for o in outputs[i]:\n"
        "                o.data_source = batch.non_tensor_batch['data_source'][i]\n"
        "                o.uid = batch.non_tensor_batch['uid'][i]\n",
        "        for i in range(len(batch)):\n"
        "            for o in outputs[i]:\n"
        "                o.data_source = batch.non_tensor_batch['data_source'][i]\n"
        "                o.uid = batch.non_tensor_batch['uid'][i]\n"
        "                if uma_credit_trace_enabled():\n"
        "                    p33_2_info = trajectory_info[i]\n"
        "                    p33_2_trajectory_key = f\"step={p33_2_info.get('step')}|sample={p33_2_info.get('sample_index')}|rollout={p33_2_info.get('rollout_n')}|validate={p33_2_info.get('validate')}\"\n"
        "                    o.extra_fields['p33_2_trajectory_key'] = p33_2_trajectory_key\n"
        "                    o.extra_fields['p33_2_trajectory_step'] = p33_2_info.get('step')\n"
        "                    o.extra_fields['p33_2_sample_index'] = p33_2_info.get('sample_index')\n"
        "                    o.extra_fields['p33_2_rollout_n'] = p33_2_info.get('rollout_n')\n"
        "                    o.extra_fields['p33_2_validate'] = p33_2_info.get('validate')\n"
        "                    o.extra_fields['p33_2_batch_row_index'] = int(i)\n"
        "                    if 'index' in batch.non_tensor_batch:\n"
        "                        o.extra_fields['p33_2_original_index'] = str(batch.non_tensor_batch['index'][i])\n"
        "                    if 'agent_name' in batch.non_tensor_batch:\n"
        "                        o.extra_fields['p33_2_agent_name'] = str(batch.non_tensor_batch['agent_name'][i])\n",
        "agent_loop_attach_input_identity",
    )
    text = replace_once(
        text,
        "                    output.reward_score = reward_score\n"
        "                    output.extra_fields[\"reward_extra_info\"] = result[\"reward_extra_info\"]\n",
        "                    output.reward_score = reward_score\n"
        "                    if uma_credit_trace_enabled():\n"
        "                        reward_extra_info = result.get(\"reward_extra_info\", {})\n"
        "                        output_extra = output.extra_fields or {}\n"
        "                        uma_credit_trace_event(\"p33_2_reward_postprocess\", {\n"
        "                            \"uid\": output.uid,\n"
        "                            \"data_source\": output.data_source,\n"
        "                            \"trajectory_id\": output_extra.get(\"trajectory_id\"),\n"
        "                            \"trajectory_key\": output_extra.get(\"p33_2_trajectory_key\"),\n"
        "                            \"trajectory_step\": output_extra.get(\"p33_2_trajectory_step\"),\n"
        "                            \"sample_index\": output_extra.get(\"p33_2_sample_index\"),\n"
        "                            \"rollout_n\": output_extra.get(\"p33_2_rollout_n\"),\n"
        "                            \"validate\": output_extra.get(\"p33_2_validate\"),\n"
        "                            \"conversation_index\": output_extra.get(\"p33_2_conversation_index\"),\n"
        "                            \"memory_step_index\": output_extra.get(\"p33_2_memory_step_index\"),\n"
        "                            \"final_query_index\": output_extra.get(\"p33_2_final_query_index\"),\n"
        "                            \"is_final\": output_extra.get(\"is_final\"),\n"
        "                            \"num_finals\": num_finals,\n"
        "                            \"reward_score\": reward_score,\n"
        "                            \"outcome_reward_accumulator\": outcome_reward,\n"
        "                            \"qa_outcome_component_assigned\": reward_extra_info.get(\"outcome_reward\"),\n"
        "                            \"tool_reward\": reward_extra_info.get(\"tool_reward\"),\n"
        "                            \"format_reward\": reward_extra_info.get(\"format_reward\"),\n"
        "                            \"num_tools\": output_extra.get(\"num_tools\"),\n"
        "                            \"tool_rewards\": output_extra.get(\"tool_rewards\"),\n"
        "                            \"tool_counts\": output_extra.get(\"tool_counts\"),\n"
        "                            \"prompt_sha1\": uma_credit_stable_hash_payload(output.prompt_ids),\n"
        "                            \"response_sha1\": uma_credit_stable_hash_payload(output.response_ids),\n"
        "                            \"response_mask_sum\": int(output.response_mask.sum().item()) if hasattr(output.response_mask, \"sum\") else None,\n"
        "                        })\n"
        "                    output.extra_fields[\"reward_extra_info\"] = result[\"reward_extra_info\"]\n",
        "agent_loop_reward_trace",
    )
    text = replace_once(
        text,
        "        tasks = [\n"
        "            worker.generate_sequences.remote(chunk, verbose=(verbose and idx==0))\n"
        "            for idx, (worker, chunk) in enumerate(zip(self.agent_loop_workers, chunkes, strict=True))\n"
        "        ]\n",
        "        tasks = [\n"
        "            worker.generate_sequences.remote(chunk, verbose=(verbose and idx==0))\n"
        "            for idx, (worker, chunk) in enumerate(zip(self.agent_loop_workers, chunkes, strict=True))\n"
        "        ]\n"
        "        if uma_credit_trace_enabled():\n"
        "            uma_credit_task_order = {task: idx for idx, task in enumerate(tasks)}\n"
        "            uma_credit_completion_submission_indices = []\n"
        "            uma_credit_trace_event(\"p33_2_ray_tasks_submitted\", {\n"
        "                \"num_tasks\": len(tasks),\n"
        "                \"task_refs\": [str(task) for task in tasks],\n"
        "            })\n"
        "        else:\n"
        "            uma_credit_task_order = {}\n"
        "            uma_credit_completion_submission_indices = []\n",
        "agent_loop_ray_task_order_map",
    )
    text = replace_once(
        text,
        "            for task in ready:\n"
        "                result = ray.get(task)\n"
        "                outputs.append(result)\n"
        "                completed += 1\n",
        "            for task in ready:\n"
        "                result = ray.get(task)\n"
        "                outputs.append(result)\n"
        "                if uma_credit_trace_enabled():\n"
        "                    uma_credit_completion_submission_indices.append(uma_credit_task_order.get(task))\n"
        "                    uma_credit_trace_event(\"p33_2_ray_task_completed\", {\n"
        "                        \"completion_position\": completed,\n"
        "                        \"submission_index\": uma_credit_task_order.get(task),\n"
        "                        \"task_ref\": str(task),\n"
        "                        \"result_type\": type(result).__name__,\n"
        "                        \"result_len\": len(result) if hasattr(result, \"__len__\") else None,\n"
        "                    })\n"
        "                completed += 1\n",
        "agent_loop_ray_completion_trace",
    )
    text = replace_once(
        text,
        "        # 按原始顺序排序\n"
        "        task_to_result = {task: result for task, result in zip(tasks, outputs)}\n"
        "        outputs = [task_to_result[task] for task in tasks]\n",
        "        # 按原始顺序排序\n"
        "        if uma_credit_trace_enabled():\n"
        "            uma_credit_trace_event(\"p33_2_ray_reorder_precheck\", {\n"
        "                \"task_refs\": [str(task) for task in tasks],\n"
        "                \"completion_submission_indices\": uma_credit_completion_submission_indices,\n"
        "                \"completion_matches_submission_order\": uma_credit_completion_submission_indices == list(range(len(uma_credit_completion_submission_indices))),\n"
        "                \"num_outputs\": len(outputs),\n"
        "            })\n"
        "        task_to_result = {task: result for task, result in zip(tasks, outputs)}\n"
        "        outputs = [task_to_result[task] for task in tasks]\n",
        "agent_loop_ray_reorder_precheck",
    )
    path.write_text(text, encoding="utf-8")
    return [{"path": str(path.relative_to(repo)), "before_sha256": sha256_file(path), "note": "patched"}]


def patch_tool_mem_agent_loop(repo: Path) -> None:
    path = repo / "external/verl/verl/experimental/agent_loop/tool_mem_agent_loop.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register\n",
        "from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register\n"
        "from verl.utils.uma_credit_trace import credit_trace_enabled as uma_credit_trace_enabled\n",
        "tool_mem_agent_loop_import_trace",
    )
    text = replace_once(
        text,
        "            output = AgentLoopOutput(\n"
        "                prompt_ids=conversation.prompt_ids,\n"
        "                response_ids=conversation.response_ids[: self.response_length],\n"
        "                response_mask=conversation.response_mask[: self.response_length],\n"
        "                multi_modal_data={},\n"
        "                response_logprobs=conversation.response_logprobs[: self.response_length]\n"
        "                if conversation.response_logprobs\n"
        "                else None,\n"
        "                num_turns=conversation.user_turns + conversation.assistant_turns + 1,\n"
        "                metrics=agent_data.metrics,\n"
        "                extra_fields={\n"
        "                    'trajectory_id': trajectory_id,\n"
        "                    'is_final': conversation.is_final,\n"
        "                    'tool_rewards': conversation.tool_rewards,\n"
        "                    'num_tools': conversation.num_tools,\n"
        "                    'tool_counts': conversation.tool_counts,  # Add tool counts to extra_fields\n"
        "                },\n"
        "            )\n",
        "            extra_fields = {\n"
        "                'trajectory_id': trajectory_id,\n"
        "                'is_final': conversation.is_final,\n"
        "                'tool_rewards': conversation.tool_rewards,\n"
        "                'num_tools': conversation.num_tools,\n"
        "                'tool_counts': conversation.tool_counts,  # Add tool counts to extra_fields\n"
        "            }\n"
        "            if uma_credit_trace_enabled():\n"
        "                extra_fields['p33_2_conversation_index'] = i\n"
        "                if conversation.is_final:\n"
        "                    extra_fields['p33_2_memory_step_index'] = None\n"
        "                    extra_fields['p33_2_final_query_index'] = conversation.is_final\n"
        "                else:\n"
        "                    extra_fields['p33_2_memory_step_index'] = i\n"
        "                    extra_fields['p33_2_final_query_index'] = None\n"
        "            output = AgentLoopOutput(\n"
        "                prompt_ids=conversation.prompt_ids,\n"
        "                response_ids=conversation.response_ids[: self.response_length],\n"
        "                response_mask=conversation.response_mask[: self.response_length],\n"
        "                multi_modal_data={},\n"
        "                response_logprobs=conversation.response_logprobs[: self.response_length]\n"
        "                if conversation.response_logprobs\n"
        "                else None,\n"
        "                num_turns=conversation.user_turns + conversation.assistant_turns + 1,\n"
        "                metrics=agent_data.metrics,\n"
        "                extra_fields=extra_fields,\n"
        "            )\n",
        "tool_mem_agent_loop_conversation_identity",
    )
    path.write_text(text, encoding="utf-8")


def patch_ray_trainer(repo: Path) -> None:
    path = repo / "external/verl/verl/trainer/ppo/ray_trainer.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from verl.utils.tracking import ValidationGenerationsLogger\n",
        "from verl.utils.tracking import ValidationGenerationsLogger\n"
        "from verl.utils.uma_credit_trace import credit_trace_enabled as uma_credit_trace_enabled\n"
        "from verl.utils.uma_credit_trace import trace_event as uma_credit_trace_event\n",
        "ray_trainer_import_trace",
    )
    text = replace_once(
        text,
        "        advantages, returns = core_algos.compute_grpo_outcome_advantage(\n"
        "            token_level_rewards=data.batch[\"token_level_rewards\"],\n"
        "            response_mask=grpo_calculation_mask,\n"
        "            # index=data.non_tensor_batch[\"uid\"],\n"
        "            index=index_list,\n"
        "            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,\n"
        "        )\n"
        "        data.batch[\"advantages\"] = advantages\n"
        "        data.batch[\"returns\"] = returns\n",
        "        if uma_credit_trace_enabled():\n"
        "            raw_scores = data.batch[\"token_level_rewards\"].sum(dim=-1).detach().cpu()\n"
        "            group_values = defaultdict(list)\n"
        "            for row_idx, group_id in enumerate(index_list):\n"
        "                group_values[group_id].append(raw_scores[row_idx])\n"
        "            group_mean_std = {}\n"
        "            for group_id, vals in group_values.items():\n"
        "                vals_tensor = torch.stack(vals)\n"
        "                group_mean_std[group_id] = {\n"
        "                    \"mean\": vals_tensor.mean().item() if len(vals) > 1 else 0.0,\n"
        "                    \"std\": vals_tensor.std().item() if len(vals) > 1 else 1.0,\n"
        "                    \"n\": len(vals),\n"
        "                }\n"
        "            uma_credit_trace_event(\"p33_2_grpo_grouping_summary\", {\n"
        "                \"n_rows\": len(index_list),\n"
        "                \"n_groups\": len(group_values),\n"
        "                \"groups\": group_mean_std,\n"
        "            })\n"
        "        advantages, returns = core_algos.compute_grpo_outcome_advantage(\n"
        "            token_level_rewards=data.batch[\"token_level_rewards\"],\n"
        "            response_mask=grpo_calculation_mask,\n"
        "            # index=data.non_tensor_batch[\"uid\"],\n"
        "            index=index_list,\n"
        "            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,\n"
        "        )\n"
        "        if uma_credit_trace_enabled():\n"
        "            raw_scores = data.batch[\"token_level_rewards\"].sum(dim=-1).detach().cpu()\n"
        "            adv_cpu = advantages.detach().cpu()\n"
        "            mask_cpu = grpo_calculation_mask.detach().cpu()\n"
        "            for row_idx, group_id in enumerate(index_list):\n"
        "                mask = mask_cpu[row_idx].bool()\n"
        "                valid_adv = adv_cpu[row_idx][mask]\n"
        "                unique_count = int(torch.unique(valid_adv).numel()) if valid_adv.numel() else 0\n"
        "                payload = {\n"
        "                    \"row_index\": row_idx,\n"
        "                    \"uid\": data.non_tensor_batch.get(\"uid\", [None] * len(index_list))[row_idx],\n"
        "                    \"is_final\": data.non_tensor_batch.get(\"is_final\", [None] * len(index_list))[row_idx],\n"
        "                    \"trajectory_id\": data.non_tensor_batch.get(\"trajectory_id\", [None] * len(index_list))[row_idx],\n"
        "                    \"trajectory_key\": data.non_tensor_batch.get(\"p33_2_trajectory_key\", [None] * len(index_list))[row_idx],\n"
        "                    \"trajectory_step\": data.non_tensor_batch.get(\"p33_2_trajectory_step\", [None] * len(index_list))[row_idx],\n"
        "                    \"sample_index\": data.non_tensor_batch.get(\"p33_2_sample_index\", [None] * len(index_list))[row_idx],\n"
        "                    \"rollout_n\": data.non_tensor_batch.get(\"p33_2_rollout_n\", [None] * len(index_list))[row_idx],\n"
        "                    \"validate\": data.non_tensor_batch.get(\"p33_2_validate\", [None] * len(index_list))[row_idx],\n"
        "                    \"conversation_index\": data.non_tensor_batch.get(\"p33_2_conversation_index\", [None] * len(index_list))[row_idx],\n"
        "                    \"memory_step_index\": data.non_tensor_batch.get(\"p33_2_memory_step_index\", [None] * len(index_list))[row_idx],\n"
        "                    \"final_query_index\": data.non_tensor_batch.get(\"p33_2_final_query_index\", [None] * len(index_list))[row_idx],\n"
        "                    \"data_source\": data.non_tensor_batch.get(\"data_source\", [None] * len(index_list))[row_idx],\n"
        "                    \"original_index\": data.non_tensor_batch.get(\"p33_2_original_index\", [None] * len(index_list))[row_idx],\n"
        "                    \"batch_row_index\": data.non_tensor_batch.get(\"p33_2_batch_row_index\", [None] * len(index_list))[row_idx],\n"
        "                    \"agent_name\": data.non_tensor_batch.get(\"p33_2_agent_name\", [None] * len(index_list))[row_idx],\n"
        "                    \"grpo_group\": group_id,\n"
        "                    \"raw_score\": raw_scores[row_idx],\n"
        "                    \"grpo_group_mean\": group_mean_std[group_id][\"mean\"],\n"
        "                    \"grpo_group_std\": group_mean_std[group_id][\"std\"],\n"
        "                    \"grpo_group_n\": group_mean_std[group_id][\"n\"],\n"
        "                    \"advantage_scalar\": valid_adv[0] if valid_adv.numel() else None,\n"
        "                    \"advantage_unique_values_on_generated_tokens\": unique_count,\n"
        "                    \"response_mask_tokens\": int(mask.sum().item()),\n"
        "                    \"reward_score\": data.non_tensor_batch.get(\"score\", [None] * len(index_list))[row_idx],\n"
        "                    \"qa_outcome_component\": data.non_tensor_batch.get(\"outcome_reward\", [None] * len(index_list))[row_idx],\n"
        "                    \"tool_reward\": data.non_tensor_batch.get(\"tool_reward\", [None] * len(index_list))[row_idx],\n"
        "                    \"num_tools\": data.non_tensor_batch.get(\"num_tools\", [None] * len(index_list))[row_idx],\n"
        "                    \"tool_counts\": data.non_tensor_batch.get(\"tool_counts\", [None] * len(index_list))[row_idx],\n"
        "                }\n"
        "                uma_credit_trace_event(\"p33_2_grpo_advantage_row\", payload)\n"
        "        data.batch[\"advantages\"] = advantages\n"
        "        data.batch[\"returns\"] = returns\n",
        "ray_trainer_grpo_trace",
    )
    text = replace_once(
        text,
        "                        batch = compute_advantage(\n"
        "                            batch,\n"
        "                            adv_estimator=self.config.algorithm.adv_estimator,\n"
        "                            gamma=self.config.algorithm.gamma,\n"
        "                            lam=self.config.algorithm.lam,\n"
        "                            num_repeat=self.config.actor_rollout_ref.rollout.n,\n"
        "                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,\n"
        "                            config=self.config.algorithm,\n"
        "                        )\n"
        "\n"
        "                    # update critic\n",
        "                        batch = compute_advantage(\n"
        "                            batch,\n"
        "                            adv_estimator=self.config.algorithm.adv_estimator,\n"
        "                            gamma=self.config.algorithm.gamma,\n"
        "                            lam=self.config.algorithm.lam,\n"
        "                            num_repeat=self.config.actor_rollout_ref.rollout.n,\n"
        "                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,\n"
        "                            config=self.config.algorithm,\n"
        "                        )\n"
        "                        if uma_credit_trace_enabled() and str(os.getenv(\"UMA_CREDIT_TRACE_EXIT_AFTER_ADVANTAGE\", \"\")).strip().lower() in {\"1\", \"true\", \"yes\", \"on\", \"y\"}:\n"
        "                            uma_credit_trace_event(\"p33_2_exit_after_advantage_before_optimizer\", {\n"
        "                                \"global_steps\": self.global_steps,\n"
        "                                \"n_rows\": len(batch),\n"
        "                                \"adv_estimator\": self.config.algorithm.adv_estimator,\n"
        "                                \"norm_adv_by_std_in_grpo\": norm_adv_by_std_in_grpo,\n"
        "                                \"reason\": \"UMA_CREDIT_TRACE_EXIT_AFTER_ADVANTAGE requested; returning before critic/actor optimizer step\",\n"
        "                            })\n"
        "                            print(\"P33_2_EXIT_AFTER_ADVANTAGE_BEFORE_OPTIMIZER\")\n"
        "                            return\n"
        "\n"
        "                    # update critic\n",
        "ray_trainer_exit_after_advantage",
    )
    path.write_text(text, encoding="utf-8")


def patch_repo(src: Path, dst: Path, expected_commit: str) -> dict:
    if not src.exists():
        raise FileNotFoundError(src)
    commit = run(["git", "rev-parse", "HEAD"], cwd=src)
    if commit != expected_commit:
        raise RuntimeError(f"Source commit mismatch: expected {expected_commit}, got {commit}")
    if dst.exists():
        raise FileExistsError(f"Destination exists; refusing to overwrite: {dst}")
    shutil.copytree(src, dst)
    (dst / "external/verl/verl/utils/uma_credit_trace.py").write_text(TRACE_MODULE, encoding="utf-8")
    patch_agent_loop(dst)
    patch_tool_mem_agent_loop(dst)
    patch_ray_trainer(dst)
    run(["git", "add", "-N", "external/verl/verl/utils/uma_credit_trace.py"], cwd=dst)
    diff_paths = [
        "external/verl/verl/experimental/agent_loop/agent_loop.py",
        "external/verl/verl/experimental/agent_loop/tool_mem_agent_loop.py",
        "external/verl/verl/trainer/ppo/ray_trainer.py",
        "external/verl/verl/utils/uma_credit_trace.py",
    ]
    diff = run(["git", "diff", "--", *diff_paths], cwd=dst)
    diff_path = dst / "p33_2_uma_credit_instrumentation.diff"
    diff_path.write_text(diff + "\n", encoding="utf-8")
    manifest = {
        "status": "P33_2_INSTRUMENTED_WORKTREE_PREPARED",
        "scope": "default-off logging only",
        "source_repo": str(src),
        "destination_repo": str(dst),
        "source_commit": commit,
        "expected_commit": expected_commit,
        "forbidden_changes": [
            "prompt",
            "operator",
            "split",
            "metric",
            "generation_protocol",
            "reward_formula",
            "trainer_objective",
        ],
        "activation": {
            "enabled_env": "UMA_CREDIT_TRACE=1",
            "path_env": "UMA_CREDIT_TRACE_PATH=/path/to/uma_credit_trace.jsonl",
            "safe_exit_env": "UMA_CREDIT_TRACE_EXIT_AFTER_ADVANTAGE=1 returns after traced advantages and before optimizer steps",
            "default": "off",
        },
        "files_modified_or_added": {
            "added": ["external/verl/verl/utils/uma_credit_trace.py"],
            "modified": [
                "external/verl/verl/experimental/agent_loop/agent_loop.py",
                "external/verl/verl/experimental/agent_loop/tool_mem_agent_loop.py",
                "external/verl/verl/trainer/ppo/ray_trainer.py",
            ],
            "diff": "p33_2_uma_credit_instrumentation.diff",
        },
        "sha256": {
            "agent_loop.py": sha256_file(dst / "external/verl/verl/experimental/agent_loop/agent_loop.py"),
            "tool_mem_agent_loop.py": sha256_file(dst / "external/verl/verl/experimental/agent_loop/tool_mem_agent_loop.py"),
            "ray_trainer.py": sha256_file(dst / "external/verl/verl/trainer/ppo/ray_trainer.py"),
            "uma_credit_trace.py": sha256_file(dst / "external/verl/verl/utils/uma_credit_trace.py"),
            "diff": sha256_file(diff_path),
        },
    }
    (dst / "p33_2_uma_credit_instrumentation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-repo", required=True)
    parser.add_argument("--dst-repo", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    manifest = patch_repo(Path(args.src_repo), Path(args.dst_repo), args.expected_commit)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
