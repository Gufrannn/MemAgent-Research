# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (agg_loss, compute_policy_loss,
                                        compute_rwwpo_policy_loss, kl_penalty, grad_acc_mode)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _snapshot_local_optimizer_step(self):
        """CPU snapshot of local FSDP shards and optimizer for reject/rollback."""
        def cpu_clone(value):
            if torch.is_tensor(value):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key: cpu_clone(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cpu_clone(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cpu_clone(item) for item in value)
            return value
        params = [param.detach().cpu().clone() for param in self.actor_module.parameters()]
        return params, cpu_clone(self.actor_optimizer.state_dict())

    def _restore_local_optimizer_step(self, snapshot):
        params, optimizer = snapshot
        with torch.no_grad():
            for target, source in zip(self.actor_module.parameters(), params):
                target.copy_(source.to(device=target.device, dtype=target.dtype))
        self.actor_optimizer.load_state_dict(optimizer)

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        ######
        # ADD: loss mask
        ######
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask"]
        rwwpo_config = self.config.get("rwwpo", {})
        rwwpo_enabled = bool(rwwpo_config.get("enable", False))
        rwwpo_collect_original = bool(rwwpo_config.get("collect_original_only", False))
        if rwwpo_enabled and rwwpo_collect_original:
            raise ValueError("RWWPO method and Original collection modes are mutually exclusive")
        rwwpo_capture = rwwpo_enabled or rwwpo_collect_original
        if rwwpo_capture:
            select_keys.extend(["final_mask", "sample_index", "trajectory_turn", "rwwpo_global_step"])
            if self.config.loss_agg_mode != "token-mean":
                raise ValueError("RWWPO exact first-order contract requires Original token-mean aggregation")
            if not rwwpo_config.get("ledger_dir"):
                raise ValueError("RWWPO requires an explicit append-only actual-loss ledger_dir")
            if not rwwpo_config.get("attempt_id"):
                raise ValueError("RWWPO requires an explicit semantic attempt_id")
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')

        ######
        # ADD: check multirun padding mask
        ######
        padded = 'no_padding_mask' in data.batch
        if padded:
            from recurrent.utils import indexing_proto
            # batch is a TensorDict here, we need a DataProto for code reusing.
            proto = data.select(batch_keys=select_keys)
            # we need to drop empty samples, since they will implact sequence-level averaging loss
            batch = indexing_proto(proto, data.batch['no_padding_mask']).batch
        else:
            batch = data.select(batch_keys=select_keys).batch
        from recurrent.research.actor_batch import DIAG_PREFIX
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        response_valid_tokens = int(batch["response_mask"].sum().item())
        print(
            f"{DIAG_PREFIX} actor rank={rank}: per_rank_local_batch_size={len(batch)}, "
            f"response_valid_tokens={response_valid_tokens}, "
            f"micro_batch_size={self.config.ppo_micro_batch_size_per_gpu}, "
            f"computed_num_mini_batches/sections="
            f"{self.config.train_batch_size // self.config.ppo_mini_batch_size}, "
            f"tensor_shapes={{{', '.join(f'{key!r}: {tuple(value.shape)!r}' for key, value in batch.items())}}}"
        )
        if len(batch) < 1 or response_valid_tokens < 1:
            raise ValueError(
                f"{DIAG_PREFIX} actor rank={rank} has no trainable samples: "
                f"local_batch_size={len(batch)}, response_valid_tokens={response_valid_tokens}"
            )
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
            raise NotImplementedError("need to be fixed for multi-turn code")
        ######
        # ADD: Splits `proto` into `self.config.update_steps_per_batch` chunks.
        #     proto_split is similar to `np.array_split`/`torch.tensor_split`, support inequally-sized chunks.
        #     note that self.config.train_batch_size has been injected in verl/workers/fsdp_workers.py
        ######
        if rwwpo_capture:
            # A recurrent writer prefix is indivisible: never split its turns
            # across optimizer minibatches.
            groups = [torch.nonzero(batch["sample_index"] == sid, as_tuple=False).flatten()
                      for sid in torch.unique(batch["sample_index"], sorted=True)]
            target_sections = max(1, self.config.train_batch_size // self.config.ppo_mini_batch_size)
            bins, loads = [[] for _ in range(target_sections)], [0] * target_sections
            for group in sorted(groups, key=len, reverse=True):
                slot = min(range(target_sections), key=lambda idx: loads[idx])
                bins[slot].append(group); loads[slot] += len(group)
            dataloader = [batch[torch.cat(parts)] for parts in bins if parts]
        elif padded:
            from recurrent.utils import td_split
            dataloader = td_split(batch, self.config.train_batch_size // self.config.ppo_mini_batch_size)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                mini_batch_response_tokens = int(mini_batch["response_mask"].sum().item())
                if mini_batch_response_tokens < 1:
                    raise ValueError(
                        f"{DIAG_PREFIX} actor rank={rank} mini-batch has no trainable response tokens: "
                        f"mini_batch_size={len(mini_batch)}, response_valid_tokens=0"
                    )
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                    ###### NOTE: rearrange_micro_batches will generate max(num_micro for num_micro in all_dp_workers) and torch.distributed.all_reduce is called
                    ###### When debugging, set a breakpoint after here, or code will be stuck here.
                else:
                    ######
                    # ADD: I will not disable dynamic_bsz, just in case, use proto_split to get num_micro_batches
                    ######
                    if padded:
                        from recurrent.utils import td_split
                        num_micro_batches = -(-len(mini_batch) // self.config.ppo_micro_batch_size_per_gpu)
                        micro_batches = td_split(mini_batch, num_micro_batches)
                    else:   
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        # split batch into micro_batches
                        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                print(
                    f"{DIAG_PREFIX} actor rank={rank}: mini_batch_size={len(mini_batch)}, "
                    f"micro_batch_size={self.config.ppo_micro_batch_size_per_gpu}, "
                    f"computed_num_micro_batches/sections={len(micro_batches)}"
                )

                self.actor_optimizer.zero_grad()

                #######
                # ADD: For unbias grad_gcc, see MODIFY in below for more info.
                #######
                if not self.config.use_dynamic_bsz:
                    from warnings import warn
                    warn("Using dynamic bsz is highly recomended for multiturn since there will be padding samples")
                mini_batch_token_nums = data['response_mask'].sum()

                if rwwpo_capture:
                    forwarded = []
                    for micro_data in micro_batches:
                        if isinstance(micro_data, DataProto):
                            micro_data = {**micro_data.batch.to(torch.cuda.current_device()), **micro_data.non_tensor_batch}
                        else:
                            micro_data = micro_data.to(torch.cuda.current_device())
                        entropy, current = self._forward_micro_batch(
                            micro_batch=micro_data, temperature=temperature,
                            calculate_entropy=self.config.entropy_coeff != 0)
                        forwarded.append((micro_data, entropy, current))
                    def joined(key):
                        return torch.cat([item[0][key] for item in forwarded], dim=0)
                    current_log_prob = torch.cat([item[2] for item in forwarded], dim=0)
                    old_log_prob = joined("old_log_probs")
                    response_mask = joined("response_mask").bool()
                    final_mask = joined("final_mask").bool()
                    writer_mask = response_mask & (~final_mask).unsqueeze(-1)
                    policy_loss, rwwpo_metrics = compute_rwwpo_policy_loss(
                        old_log_prob=old_log_prob, log_prob=current_log_prob,
                        advantages=joined("advantages"), response_mask=response_mask,
                        writer_mask=writer_mask, final_mask=final_mask,
                        sample_index=joined("sample_index"), trajectory_turn=joined("trajectory_turn"),
                        cliprange=self.config.clip_ratio,
                        cliprange_low=self.config.clip_ratio_low,
                        cliprange_high=self.config.clip_ratio_high,
                        clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                        writer_log_ratio_cap=float(rwwpo_config.get("writer_log_ratio_cap", 4.0)))
                    if rwwpo_collect_original:
                        policy_loss, original_clipfrac, original_kl, original_lower = compute_policy_loss(
                            old_log_prob=old_log_prob, log_prob=current_log_prob,
                            advantages=joined("advantages"), response_mask=response_mask,
                            cliprange=self.config.clip_ratio, cliprange_low=self.config.clip_ratio_low,
                            cliprange_high=self.config.clip_ratio_high,
                            clip_ratio_c=self.config.get("clip_ratio_c", 3.0), loss_agg_mode="token-mean")
                        rwwpo_metrics["answer_clipfrac"] = original_clipfrac
                        rwwpo_metrics["answer_ppo_kl"] = original_kl
                        rwwpo_metrics["answer_clipfrac_lower"] = original_lower
                    if self.config.entropy_coeff != 0:
                        entropy = torch.cat([item[1] for item in forwarded], dim=0)
                        policy_loss = policy_loss - agg_loss(entropy, response_mask, "token-mean") * self.config.entropy_coeff
                    if self.config.use_kl_loss:
                        ref_log_prob = joined("ref_log_prob")
                        kld = kl_penalty(logprob=current_log_prob, ref_logprob=ref_log_prob,
                                         kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(kld, response_mask, "token-mean")
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    def build_global_stats(prefix_rows):
                        if torch.distributed.is_initialized():
                            gathered = [None] * torch.distributed.get_world_size()
                            torch.distributed.all_gather_object(gathered, prefix_rows)
                            prefix_rows = [row for rank_rows in gathered for row in rank_rows]
                        result = []
                        for turn in sorted({row["turn"] for row in prefix_rows}):
                            turn_rows = [row for row in prefix_rows if row["turn"] == turn]
                            values = torch.tensor([row["log_ratio"] for row in turn_rows], dtype=torch.float64)
                            weights = torch.softmax(values, dim=0); count = len(values)
                            chi2 = float((count * weights.square().sum() - 1).item())
                            result.append({"turn": turn, "batch_size": count,
                                "ess_fraction": 1.0 / (1.0 + chi2), "chi2": chi2,
                                "max_abs_log_ratio": float(values.abs().max().item()),
                                "mean_log_ratio": float(values.mean().item())})
                        return result
                    global_prefix_stats = build_global_stats(rwwpo_metrics["prefix_log_ratios"])
                    q_min = float(rwwpo_config.get("q_min", 0.5))
                    ratio_cap = float(rwwpo_config.get("writer_log_ratio_cap", 4.0))
                    constraint_pass = rwwpo_collect_original or all(
                        row["ess_fraction"] >= q_min and row["max_abs_log_ratio"] <= ratio_cap
                        for row in global_prefix_stats)
                    if rwwpo_enabled and not constraint_pass:
                        self.actor_optimizer.zero_grad()
                        raise RuntimeError("RWWPO_PREFIX_TRUST_REGION_VIOLATION: update refused before optimizer step")
                    policy_loss.backward()
                    snapshot = self._snapshot_local_optimizer_step() if rwwpo_enabled else None
                    grad_norm = self._optimizer_step()
                    post_log_prob = current_log_prob.detach()
                    post_prefix_stats = global_prefix_stats
                    post_prefix_rows = rwwpo_metrics["prefix_log_ratios"]
                    accepted = True
                    if rwwpo_enabled:
                        with torch.no_grad():
                            post_log_prob = torch.cat([
                                self._forward_micro_batch(item[0], temperature=temperature,
                                                          calculate_entropy=False)[1]
                                for item in forwarded], dim=0)
                            _, post_metrics = compute_rwwpo_policy_loss(
                                old_log_prob, post_log_prob, joined("advantages"), response_mask,
                                writer_mask, final_mask, joined("sample_index"), joined("trajectory_turn"),
                                self.config.clip_ratio, self.config.clip_ratio_low,
                                self.config.clip_ratio_high, self.config.get("clip_ratio_c", 3.0),
                                writer_log_ratio_cap=ratio_cap)
                        post_prefix_stats = build_global_stats(post_metrics["prefix_log_ratios"])
                        post_prefix_rows = post_metrics["prefix_log_ratios"]
                        accepted = all(row["ess_fraction"] >= q_min and
                                       row["max_abs_log_ratio"] <= ratio_cap
                                       for row in post_prefix_stats)
                        if not accepted:
                            self._restore_local_optimizer_step(snapshot)
                    from recurrent.research.rwwpo_ledger import append_actual_loss_record
                    append_actual_loss_record(
                        ledger_dir=rwwpo_config.get("ledger_dir"), attempt_id=rwwpo_config.get("attempt_id"),
                        mode="original_collection" if rwwpo_collect_original else "rwwpo_method", rank=rank,
                        global_step=int(joined("rwwpo_global_step")[0].item()), epoch=epoch,
                        minibatch=batch_idx, old_log_prob=old_log_prob, current_log_prob=current_log_prob,
                        proposed_post_log_prob=post_log_prob, response_mask=response_mask,
                        writer_mask=writer_mask, answer_mask=rwwpo_metrics["answer_mask"],
                        trajectory_turn=joined("trajectory_turn"), sample_index=joined("sample_index"),
                        advantages=joined("advantages"), denominator=rwwpo_metrics["denominator"].item(),
                        prefix_rows=rwwpo_metrics["prefix_log_ratios"], prefix_stats=global_prefix_stats,
                        post_prefix_rows=post_prefix_rows, post_prefix_stats=post_prefix_stats, q_min=q_min,
                        writer_log_ratio_cap=ratio_cap,
                        constraint_pass=constraint_pass, accepted=accepted)
                    append_to_dict(metrics, {
                        "actor/pg_loss": policy_loss.detach().item(),
                        "actor/pg_clipfrac": rwwpo_metrics["answer_clipfrac"].detach().item(),
                        "actor/ppo_kl": rwwpo_metrics["answer_ppo_kl"].detach().item(),
                        "actor/pg_clipfrac_lower": rwwpo_metrics["answer_clipfrac_lower"].detach().item(),
                        "rwwpo/min_prefix_ess": min(row["ess_fraction"] for row in global_prefix_stats),
                    })
                    append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item(),
                                             "rwwpo/update_accepted": float(accepted)})
                    continue

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload

                    #######
                    # MODIFIED: use loss_mask directly
                    #######
                    response_mask = data['response_mask']
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    ######
                    # MODIFY: we have to fix grad_acc computation: weighted averaging by token num in stead of len(data)
                    #         See 
                    #         If we use Dr. GRPO algorithm（unbias_length_enable）, then this fix is no 
                    #           more needed since policy averaging there is sequence-level.
                    #         Since we have a variant of batchsize, we also remove self.gradient_accumulation
                    ######
                    acc_grad_mode = grad_acc_mode(loss_agg_mode)
                    if acc_grad_mode == "seq":
                        loss = policy_loss * (len(data) / len(mini_batch)) # self.gradient_accumulation
                    elif acc_grad_mode == "token":
                        # weights by token nums, note that we want to apply a simple scalar, or the compute-graph will be extremely large.
                        loss = policy_loss * (response_mask.sum().item() / mini_batch_token_nums.item())
                    else:
                        raise NotImplementedError(f"Unsupported acc_grad_mode: {acc_grad_mode}")


                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
