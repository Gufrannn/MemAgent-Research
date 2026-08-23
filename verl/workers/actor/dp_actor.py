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
import json
import logging
import os
import time
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
from recurrent.research.rwwpo_transaction import (
    ALPHA_GRID, RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS, digest,
    displacement_norm, largest_tested_feasible,
    local_gradient_sketch_sufficient_statistics,
    logical_transaction_seed, off_behavior_exposed, parameter_snapshot,
    prefix_distribution_stats, proposal_clock, relative_displacement_norm, restore_rng, rng_snapshot,
    seed_transaction_rng, set_interpolated_parameters,
    set_stateless_proposal_lr, writer_logprob_rms_sufficient_statistics,
)

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None,
                 actor_lr_scheduler=None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.actor_lr_scheduler = actor_lr_scheduler
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _snapshot_local_optimizer_step(self, *, include_scheduler=True):
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
        scheduler = (cpu_clone(self.actor_lr_scheduler.state_dict())
                     if include_scheduler and self.actor_lr_scheduler is not None else None)
        return params, cpu_clone(self.actor_optimizer.state_dict()), scheduler

    def _transaction_digests(self, snapshot, rng_state, scheduler_evidence=None):
        return {
            "model": digest(snapshot[0]),
            "optimizer": digest(snapshot[1]),
            "scheduler": digest(snapshot[2] if scheduler_evidence is None else scheduler_evidence),
            "scaler": "not_applicable_bfloat16",
            "rng": digest(rng_state),
        }

    def _restore_local_optimizer_step(self, snapshot):
        params, optimizer, scheduler = snapshot
        with torch.no_grad():
            for target, source in zip(self.actor_module.parameters(), params):
                target.copy_(source.to(device=target.device, dtype=target.dtype))
        self.actor_optimizer.load_state_dict(optimizer)
        if scheduler is not None:
            self.actor_lr_scheduler.load_state_dict(scheduler)

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
        rwwpo_objective = str(rwwpo_config.get("objective_variant", "whole_prefix"))
        rwwpo_controller = str(rwwpo_config.get("controller_variant", "hard_rollback"))
        if rwwpo_objective not in ("whole_prefix", "per_write_joint", "original_tokenwise"):
            raise ValueError("RWWPO unknown objective_variant")
        if rwwpo_controller not in ("none", "hard_rollback", "feasible_backtracking"):
            raise ValueError("RWWPO unknown controller_variant")
        if rwwpo_enabled and rwwpo_collect_original:
            raise ValueError("RWWPO method and Original collection modes are mutually exclusive")
        rwwpo_capture = rwwpo_enabled or rwwpo_collect_original
        rwwpo2_enabled = bool(rwwpo_enabled and
                              str(rwwpo_config.get("program_version", "")) == "rwwpo2-k2")
        if rwwpo_capture:
            select_keys.extend(["final_mask", "sample_index", "trajectory_turn", "rwwpo_global_step",
                                "rwwpo_example_identity_hash", "rwwpo_trajectory_identity_hash"])
            if self.config.loss_agg_mode != "token-mean":
                raise ValueError("RWWPO exact first-order contract requires Original token-mean aggregation")
            if not rwwpo_config.get("ledger_dir"):
                raise ValueError("RWWPO requires an explicit append-only actual-loss ledger_dir")
            if not rwwpo_config.get("attempt_id"):
                raise ValueError("RWWPO requires an explicit semantic attempt_id")
            if rwwpo2_enabled:
                if int(self.config.ppo_epochs) != 2:
                    raise ValueError("RWWPO-2 requires exactly two inner actor transactions")
                if int(rwwpo_config.get("inner_transactions_per_round", 0)) != 2:
                    raise ValueError("RWWPO-2 K2 manifest/runtime mismatch")
                if (not bool(self.config.use_kl_loss)
                        or str(self.config.kl_loss_type) != "low_var_kl"
                        or float(self.config.kl_loss_coef) != 0.001
                        or float(self.config.entropy_coeff) != 0.0
                        or float(self.config.clip_ratio) != 0.2
                        or float(self.config.clip_ratio_low) != 0.2
                        or float(self.config.clip_ratio_high) != 0.2
                        or float(self.config.get("clip_ratio_c", 3.0)) != 3.0):
                    raise ValueError("RWWPO-2 complete actual-loss contract drift")
                required_schedule = {"kind", "base_lr", "warmup_proposals", "total_proposals"}
                if set(rwwpo_config.get("proposal_schedule", {})) != required_schedule:
                    raise ValueError("RWWPO-2 requires a complete stateless proposal schedule")
                for threshold in ("tau_theta", "tau_logprob", "tau_gradient"):
                    value = rwwpo_config.get(threshold)
                    if value is None or float(value) < 0:
                        raise ValueError(f"RWWPO-2 requires calibrated {threshold}")
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
            if rwwpo_enabled and len(dataloader) != 1:
                raise ValueError("RWWPO transactions require one full optimizer minibatch per inner update")
        elif padded:
            from recurrent.utils import td_split
            dataloader = td_split(batch, self.config.train_batch_size // self.config.ppo_mini_batch_size)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        rwwpo2_frozen_batch_digest = None
        rwwpo2_inner1_exposure = None
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
                    transaction_optimizer_step_calls = 0
                    round_id = int(data["rwwpo_global_step"][0].item())
                    inner_id = int(epoch + 1)
                    logical_proposal_id = None
                    proposal_lr = None
                    logical_seed = None
                    scheduler_evidence = None
                    accepted_optimizer_clock_before = None
                    # A reject is a true transaction rollback.  Freeze the
                    # complete process RNG state before even the logical reseed;
                    # the reseed and all materialization/shadow/trial draws are
                    # effects of this transaction and must disappear on alpha=0.
                    transaction_entry_rng = rng_snapshot()
                    logical_seeded_rng = transaction_entry_rng
                    if rwwpo2_enabled:
                        logical_proposal_id = proposal_clock(round_id, inner_id)
                        schedule = dict(rwwpo_config["proposal_schedule"])
                        proposal_lr = set_stateless_proposal_lr(
                            self.actor_optimizer,
                            base_lr=float(schedule["base_lr"]),
                            warmup_proposals=int(schedule["warmup_proposals"]),
                            total_proposals=int(schedule["total_proposals"]),
                            proposal_id=logical_proposal_id,
                            kind=str(schedule["kind"]),
                        )
                        clocks = {int(group.get("rwwpo2_accepted_optimizer_clock", 0))
                                  for group in self.actor_optimizer.param_groups}
                        if len(clocks) != 1:
                            raise RuntimeError("RWWPO2_ACCEPTED_OPTIMIZER_CLOCK_DRIFT")
                        accepted_optimizer_clock_before = next(iter(clocks))
                        logical_seed = logical_transaction_seed(
                            experiment_seed=int(rwwpo_config.get("experiment_seed", 2026)),
                            round_id=round_id, inner_id=inner_id, rank=rank,
                            stream="actor_transaction",
                        )
                        seed_transaction_rng(logical_seed)
                        logical_seeded_rng = rng_snapshot()
                        scheduler_evidence = {
                            "kind": "stateless_proposal_clock_v1",
                            "proposal_clock": logical_proposal_id,
                            "lr": proposal_lr,
                            "schedule": schedule,
                        }
                    if self.config.entropy_coeff != 0:
                        raise ValueError("RWWPO streaming exact-gradient path requires frozen entropy_coeff=0")
                    forwarded = []
                    for micro_data in micro_batches:
                        if isinstance(micro_data, DataProto):
                            micro_data = {**micro_data.batch.to(torch.cuda.current_device()), **micro_data.non_tensor_batch}
                        else:
                            micro_data = micro_data.to(torch.cuda.current_device())
                        # Pass one materializes full-minibatch log probabilities without
                        # retaining every microbatch graph.  The exact loss derivative
                        # with respect to these logits is injected in a streaming second
                        # pass below, bounding activation memory independently of the
                        # number of recurrent turns.
                        # Every later replay restores the exact RNG state that
                        # preceded this materialization.  Qwen currently has
                        # zero dropout, but the K1 transition-kernel theorem
                        # must not silently depend on that architecture detail.
                        forward_rng = rng_snapshot()
                        with torch.no_grad():
                            _, current = self._forward_micro_batch(
                                micro_batch=micro_data, temperature=temperature,
                                calculate_entropy=False)
                        forwarded.append((micro_data, current.detach(), forward_rng))
                    def joined(key):
                        return torch.cat([item[0][key] for item in forwarded], dim=0)
                    current_log_prob = torch.cat([item[1] for item in forwarded], dim=0).requires_grad_(True)
                    old_log_prob = joined("old_log_probs")
                    response_mask = joined("response_mask").bool()
                    final_mask = joined("final_mask").bool()
                    writer_mask = response_mask & (~final_mask).unsqueeze(-1)
                    frozen_digest = digest({
                        "old_log_prob": old_log_prob,
                        "ref_log_prob": joined("ref_log_prob") if self.config.use_kl_loss else None,
                        "advantages": joined("advantages"),
                        "response_mask": response_mask,
                        "writer_mask": writer_mask,
                        "final_mask": final_mask,
                        "sample_index": joined("sample_index"),
                        "trajectory_turn": joined("trajectory_turn"),
                    })
                    if rwwpo2_enabled:
                        if rwwpo2_frozen_batch_digest is None:
                            rwwpo2_frozen_batch_digest = frozen_digest
                        elif frozen_digest != rwwpo2_frozen_batch_digest:
                            raise RuntimeError("RWWPO2_BEHAVIOR_BATCH_MUTATED_BETWEEN_INNER_UPDATES")
                    whole_candidate_loss, whole_metrics = compute_rwwpo_policy_loss(
                        old_log_prob=old_log_prob, log_prob=current_log_prob,
                        advantages=joined("advantages"), response_mask=response_mask,
                        writer_mask=writer_mask, final_mask=final_mask,
                        sample_index=joined("sample_index"), trajectory_turn=joined("trajectory_turn"),
                        cliprange=self.config.clip_ratio,
                        cliprange_low=self.config.clip_ratio_low,
                        cliprange_high=self.config.clip_ratio_high,
                        clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                        writer_log_ratio_cap=float(rwwpo_config.get("writer_log_ratio_cap", 4.0)),
                        writer_objective="whole_prefix")
                    original_candidate_loss, original_clipfrac, original_kl, original_lower = compute_policy_loss(
                            old_log_prob=old_log_prob, log_prob=current_log_prob,
                            advantages=joined("advantages"), response_mask=response_mask,
                            cliprange=self.config.clip_ratio, cliprange_low=self.config.clip_ratio_low,
                            cliprange_high=self.config.clip_ratio_high,
                            clip_ratio_c=self.config.get("clip_ratio_c", 3.0), loss_agg_mode="token-mean")
                    per_write_candidate_loss, per_write_metrics = compute_rwwpo_policy_loss(
                        old_log_prob=old_log_prob, log_prob=current_log_prob,
                        advantages=joined("advantages"), response_mask=response_mask,
                        writer_mask=writer_mask, final_mask=final_mask,
                        sample_index=joined("sample_index"), trajectory_turn=joined("trajectory_turn"),
                        cliprange=self.config.clip_ratio,
                        cliprange_low=self.config.clip_ratio_low,
                        cliprange_high=self.config.clip_ratio_high,
                        clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                        writer_log_ratio_cap=float(rwwpo_config.get("writer_log_ratio_cap", 4.0)),
                        writer_objective="per_write_joint")
                    shared_additive_loss = current_log_prob.sum() * 0.0
                    if self.config.use_kl_loss:
                        ref_log_prob = joined("ref_log_prob")
                        kld = kl_penalty(logprob=current_log_prob, ref_logprob=ref_log_prob,
                                         kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(kld, response_mask, "token-mean")
                        shared_additive_loss = kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    whole_coefficient, = torch.autograd.grad(
                        whole_candidate_loss + shared_additive_loss,
                        current_log_prob, retain_graph=True)
                    original_coefficient, = torch.autograd.grad(
                        original_candidate_loss + shared_additive_loss,
                        current_log_prob, retain_graph=True)
                    per_write_coefficient, = torch.autograd.grad(
                        per_write_candidate_loss + shared_additive_loss,
                        current_log_prob, retain_graph=True)
                    if rwwpo_objective == "whole_prefix":
                        policy_loss, rwwpo_metrics = whole_candidate_loss, whole_metrics
                    elif rwwpo_objective == "per_write_joint":
                        policy_loss, rwwpo_metrics = per_write_candidate_loss, per_write_metrics
                    else:
                        policy_loss, rwwpo_metrics = original_candidate_loss, whole_metrics
                    policy_loss = policy_loss + shared_additive_loss
                    active = response_mask
                    coefficient_cosine = torch.nn.functional.cosine_similarity(
                        whole_coefficient[active].reshape(1,-1),
                        original_coefficient[active].reshape(1,-1)).item()
                    def coefficient_pair(left, right):
                        l = left[active].detach().double().flatten()
                        r = right[active].detach().double().flatten()
                        return {
                            "max_abs": float((l-r).abs().max().item()),
                            "relative_l2": float((l-r).norm().item() / max(l.norm().item(), 1e-30)),
                            "cosine": float(torch.nn.functional.cosine_similarity(
                                l.reshape(1,-1), r.reshape(1,-1)).item()),
                        }
                    shadow_coefficient_diagnostics = {
                        "C_vs_B": coefficient_pair(original_coefficient, whole_coefficient),
                        "C_vs_E": coefficient_pair(original_coefficient, per_write_coefficient),
                        "B_vs_E": coefficient_pair(whole_coefficient, per_write_coefficient),
                        "host_variant": str(rwwpo_config.get("cell", "legacy")),
                        "host_point": ("behavior" if inner_id == 1 else (
                            "off_behavior" if bool((rwwpo2_inner1_exposure or {}).get(
                                "off_behavior_exposed",False)) else "behavior_unexposed")),
                    }
                    if rwwpo2_enabled and inner_id == 1:
                        tolerance = float(rwwpo_config.get(
                            "behavior_coefficient_tolerance", 1e-9
                        ))
                        if max(shadow_coefficient_diagnostics[name]["max_abs"]
                               for name in ("C_vs_B", "C_vs_E", "B_vs_E")) > tolerance:
                            raise RuntimeError("RWWPO2_BEHAVIOR_POINT_SHADOW_GRADIENT_MISMATCH")
                    if rwwpo_collect_original or rwwpo_objective == "original_tokenwise":
                        rwwpo_metrics["answer_clipfrac"] = original_clipfrac
                        rwwpo_metrics["answer_ppo_kl"] = original_kl
                        rwwpo_metrics["answer_clipfrac_lower"] = original_lower
                    if not torch.isfinite(policy_loss):
                        raise RuntimeError("RWWPO_NUMERIC_HEALTH_FAILURE: non-finite policy loss")
                    if not torch.isfinite(current_log_prob[response_mask]).all():
                        raise RuntimeError("RWWPO_NUMERIC_HEALTH_FAILURE: non-finite active-token log probability")
                    def with_root_identities(prefix_rows):
                        identities = joined("rwwpo_example_identity_hash").detach().cpu().tolist()
                        sample_ids = joined("sample_index").detach().cpu().tolist()
                        root_by_sample = {}
                        for sid, root in zip(sample_ids, identities):
                            if sid in root_by_sample and root_by_sample[sid] != root:
                                raise RuntimeError("RWWPO2_ROOT_IDENTITY_DRIFT_WITHIN_TRAJECTORY")
                            root_by_sample[sid] = root
                        annotated = []
                        for row in prefix_rows:
                            if row["sample_index"] not in root_by_sample:
                                raise RuntimeError("RWWPO2_PREFIX_ROOT_IDENTITY_MISSING")
                            annotated.append({**row, "root_identity_hash": str(
                                root_by_sample[row["sample_index"]])})
                        return annotated
                    q_min = float(rwwpo_config.get("q_min", 0.5))
                    root_q_min = float(rwwpo_config.get("root_q_min", q_min))
                    ratio_cap = float(rwwpo_config.get("writer_log_ratio_cap", 4.0))
                    def build_global_stats(prefix_rows):
                        prefix_rows = with_root_identities(prefix_rows)
                        if torch.distributed.is_initialized():
                            gathered = [None] * torch.distributed.get_world_size()
                            torch.distributed.all_gather_object(gathered, prefix_rows)
                            prefix_rows = [row for rank_rows in gathered for row in rank_rows]
                        return prefix_distribution_stats(
                            prefix_rows, q_min=q_min, root_q_min=root_q_min,
                            log_ratio_cap=ratio_cap)
                    local_prefix_rows = with_root_identities(rwwpo_metrics["prefix_log_ratios"])
                    global_prefix_stats = build_global_stats(local_prefix_rows)
                    constraint_pass = rwwpo_collect_original or all(
                        row["feasible"]
                        for row in global_prefix_stats)
                    if (rwwpo_enabled and rwwpo_controller != "none"
                            and not constraint_pass):
                        self.actor_optimizer.zero_grad()
                        restore_rng(transaction_entry_rng)
                        raise RuntimeError("RWWPO_PREFIX_TRUST_REGION_VIOLATION: update refused before optimizer step")
                    def stream_parameter_gradient(coefficient):
                        gradient_cursor = 0
                        for micro_data, _, forward_rng in forwarded:
                            restore_rng(forward_rng)
                            _, live_log_prob = self._forward_micro_batch(
                                micro_batch=micro_data, temperature=temperature,
                                calculate_entropy=False)
                            next_cursor = gradient_cursor + live_log_prob.shape[0]
                            (live_log_prob * coefficient[
                                gradient_cursor:next_cursor]).sum().backward()
                            gradient_cursor = next_cursor
                        if gradient_cursor != current_log_prob.shape[0]:
                            raise RuntimeError("RWWPO_STREAMING_GRADIENT_ALIGNMENT_FAILURE")

                    def current_parameter_gradient_sketch():
                        chunk_elements = int(rwwpo_config.get(
                            "gradient_sketch_chunk_elements",
                            RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS,
                        ))
                        if rwwpo2_enabled and chunk_elements != \
                                RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS:
                            raise RuntimeError(
                                "RWWPO2_GRADIENT_SKETCH_CHUNK_CONTRACT_DRIFT")
                        values = local_gradient_sketch_sufficient_statistics(
                            self.actor_module.parameters(),
                            chunk_elements=chunk_elements,
                        )
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(values,op=torch.distributed.ReduceOp.SUM)
                        return {"l2":float(values[0].sqrt().item()),
                                "sum":float(values[1].item()),
                                "alternating_projection":float(values[2].item()),
                                "sawtooth_projection":float(values[3].item())}

                    shadow_parameter_gradient_sketches={}
                    shadow_parameter_gradient_pairwise_relative={}
                    shadow_anchor = round_id in {
                        int(value) for value in rwwpo_config.get("shadow_host_rounds", [])
                    }
                    r50_host_shadow = round_id <= int(
                        rwwpo_config.get("r50_shadow_every_round_through", 0)
                    )
                    if rwwpo2_enabled and (shadow_anchor or r50_host_shadow):
                        shadow_rng=rng_snapshot()
                        for label,coefficient in (
                                ("C",original_coefficient),("E",per_write_coefficient),
                                ("B",whole_coefficient)):
                            self.actor_optimizer.zero_grad()
                            restore_rng(shadow_rng)
                            stream_parameter_gradient(coefficient)
                            shadow_parameter_gradient_sketches[label]=current_parameter_gradient_sketch()
                        self.actor_optimizer.zero_grad()
                        restore_rng(shadow_rng)
                        if inner_id==1:
                            tolerance=float(rwwpo_config.get("behavior_gradient_tolerance",1e-7))
                            fields=("sum","alternating_projection","sawtooth_projection")
                            control=shadow_parameter_gradient_sketches["C"]
                            control_norm=max(sum(float(control[field])**2
                                                 for field in fields)**0.5,1e-30)
                            def projected_relative(left,right):
                                return (sum((float(left[field])-float(right[field]))**2
                                            for field in fields)**0.5/control_norm)
                            shadow_parameter_gradient_pairwise_relative = {
                                f"{left}_vs_{right}": projected_relative(
                                    shadow_parameter_gradient_sketches[left],
                                    shadow_parameter_gradient_sketches[right])
                                for left,right in (("C","E"),("C","B"),("B","E"))
                            }
                            maximum_relative = max(
                                shadow_parameter_gradient_pairwise_relative.values())
                            if maximum_relative > tolerance:
                                print("[RWWPO2_BEHAVIOR_GRADIENT_DIAG] " + json.dumps({
                                    "rank": rank,
                                    "round_id": round_id,
                                    "inner_id": inner_id,
                                    "tolerance": tolerance,
                                    "maximum_relative": maximum_relative,
                                    "pairwise_relative":
                                        shadow_parameter_gradient_pairwise_relative,
                                    "coefficient_diagnostics":
                                        shadow_coefficient_diagnostics,
                                    "gradient_sketches":
                                        shadow_parameter_gradient_sketches,
                                    "gradient_sketch_chunk_elements":
                                        RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS,
                                }, sort_keys=True), flush=True)
                                raise RuntimeError("RWWPO2_BEHAVIOR_PARAMETER_GRADIENT_SKETCH_MISMATCH")

                    logprob_gradient, = torch.autograd.grad(policy_loss, current_log_prob)
                    if not torch.isfinite(logprob_gradient[response_mask]).all():
                        raise RuntimeError("RWWPO_NUMERIC_HEALTH_FAILURE: non-finite logprob gradient")
                    stream_parameter_gradient(logprob_gradient)
                    snapshot = (self._snapshot_local_optimizer_step(include_scheduler=not rwwpo2_enabled)
                                if rwwpo_capture else None)
                    proposal_gradient_rng = rng_snapshot() if rwwpo_capture else None
                    pre_digests = (self._transaction_digests(
                        snapshot, transaction_entry_rng,
                        scheduler_evidence=scheduler_evidence)
                        if rwwpo_capture else {})
                    if rwwpo_enabled:
                        from recurrent.research.rwwpo_ledger import append_transaction_marker
                        append_transaction_marker(
                            ledger_dir=rwwpo_config.get("ledger_dir"),attempt_id=rwwpo_config.get("attempt_id"),
                            rank=rank,global_step=int(joined("rwwpo_global_step")[0].item()),epoch=epoch,
                            minibatch=batch_idx,phase="intent",model_digest=pre_digests["model"],
                            inner_id=inner_id, proposal_clock=logical_proposal_id)
                    transaction_optimizer_step_calls += 1
                    grad_norm = self._optimizer_step()
                    if rwwpo2_enabled and transaction_optimizer_step_calls != 1:
                        raise RuntimeError("RWWPO2_OPTIMIZER_STEP_COUNT_DRIFT")
                    if not torch.isfinite(grad_norm):
                        raise RuntimeError("RWWPO_NUMERIC_HEALTH_FAILURE: non-finite gradient norm")
                    post_log_prob = current_log_prob.detach()
                    post_prefix_stats = global_prefix_stats
                    post_prefix_rows = local_prefix_rows
                    accepted = True
                    alpha_committed = 1.0
                    trial_rows = []
                    full_params = parameter_snapshot(self.actor_module) if rwwpo_capture else None
                    full_displacement = displacement_norm(snapshot[0], full_params) if rwwpo_capture else 0.0
                    if rwwpo_capture and torch.distributed.is_initialized():
                        squared=torch.tensor(full_displacement**2,dtype=torch.float64,
                                             device=torch.cuda.current_device())
                        torch.distributed.all_reduce(squared,op=torch.distributed.ReduceOp.SUM)
                        full_displacement=float(squared.sqrt().item())
                    proposal_zero = full_displacement <= 1e-15
                    if rwwpo_enabled:
                        alpha_grid = tuple(float(value) for value in rwwpo_config.get("alpha_grid", ALPHA_GRID))
                        if alpha_grid != ALPHA_GRID:
                            raise RuntimeError("RWWPO_ALPHA_GRID_DRIFT")
                        tested = {}
                        chosen_payload = None
                        full_log_prob = None
                        candidates = alpha_grid if rwwpo_controller == "feasible_backtracking" else (1.0,)
                        trial_started = time.perf_counter()
                        for alpha in candidates:
                            set_interpolated_parameters(self.actor_module, snapshot[0], full_params, alpha)
                            restore_rng(proposal_gradient_rng)
                            with torch.no_grad():
                                trial_log_prob = torch.cat([
                                    self._forward_micro_batch(item[0], temperature=temperature,
                                                              calculate_entropy=False)[1]
                                    for item in forwarded], dim=0)
                                _, trial_metrics = compute_rwwpo_policy_loss(
                                    old_log_prob, trial_log_prob, joined("advantages"), response_mask,
                                    writer_mask, final_mask, joined("sample_index"), joined("trajectory_turn"),
                                    self.config.clip_ratio, self.config.clip_ratio_low,
                                    self.config.clip_ratio_high, self.config.get("clip_ratio_c", 3.0),
                                    writer_log_ratio_cap=ratio_cap,
                                    writer_objective=(rwwpo_objective
                                                      if rwwpo_objective != "original_tokenwise"
                                                      else "whole_prefix"))
                            trial_prefix_rows = with_root_identities(
                                trial_metrics["prefix_log_ratios"])
                            stats = build_global_stats(trial_prefix_rows)
                            feasible = all(row["feasible"] for row in stats)
                            tested[alpha] = feasible
                            trial_rows.append({"alpha": alpha, "feasible": feasible,
                                "log_prob": trial_log_prob.detach().cpu(),
                                "prefix_rows": trial_prefix_rows, "prefix_stats": stats})
                            if alpha == 1.0:
                                full_log_prob = trial_log_prob.detach()
                            if (feasible or rwwpo_controller == "none") and chosen_payload is None:
                                chosen_payload = (trial_log_prob.detach(), trial_prefix_rows, stats)
                            if rwwpo_controller == "feasible_backtracking" and feasible:
                                # Descending fixed grid: this is the largest tested feasible point.
                                break
                        trial_forward_wall_seconds = time.perf_counter() - trial_started
                        max_trial_seconds = float(rwwpo_config.get("max_trial_forward_wall_seconds", 600.0))
                        if trial_forward_wall_seconds > max_trial_seconds:
                            self._restore_local_optimizer_step(snapshot)
                            restore_rng(transaction_entry_rng)
                            raise RuntimeError("RWWPO_TRIAL_FORWARD_BUDGET_EXCEEDED")
                        if rwwpo_controller == "feasible_backtracking":
                            decision = largest_tested_feasible(
                                [(row["alpha"], row["feasible"]) for row in trial_rows],
                                proposal_zero=proposal_zero, alpha_grid=candidates)
                            alpha_committed = decision.alpha
                            accepted = decision.accepted_nonzero
                        elif rwwpo_controller == "hard_rollback":
                            accepted = bool(tested[1.0]) and not proposal_zero
                            alpha_committed = 1.0 if accepted else 0.0
                        else:
                            accepted = not proposal_zero
                            alpha_committed = 1.0 if accepted else 0.0
                        if accepted:
                            set_interpolated_parameters(self.actor_module, snapshot[0], full_params, alpha_committed)
                            post_log_prob, post_prefix_rows, post_prefix_stats = chosen_payload
                        else:
                            self._restore_local_optimizer_step(snapshot)
                            restore_rng(transaction_entry_rng)
                            post_log_prob = current_log_prob.detach()
                            post_prefix_rows = local_prefix_rows
                            post_prefix_stats = global_prefix_stats
                        # All ranks must make the same transactional decision.
                        if torch.distributed.is_initialized():
                            decisions = [None] * torch.distributed.get_world_size()
                            torch.distributed.all_gather_object(decisions, alpha_committed)
                            if len(set(decisions)) != 1:
                                self._restore_local_optimizer_step(snapshot)
                                restore_rng(transaction_entry_rng)
                                raise RuntimeError("RWWPO_RANK_DECISION_DRIFT")
                        commit_params = parameter_snapshot(self.actor_module)
                        if accepted and rwwpo2_enabled:
                            for group in self.actor_optimizer.param_groups:
                                group["rwwpo2_accepted_optimizer_clock"] = (
                                    accepted_optimizer_clock_before + 1)
                        if accepted and self.actor_lr_scheduler is not None and not rwwpo2_enabled:
                            self.actor_lr_scheduler.step()
                        commit_scheduler_evidence = scheduler_evidence
                        commit_digests = {"model": digest(commit_params),
                                          "optimizer": digest(self.actor_optimizer.state_dict()),
                                          "scheduler": digest(commit_scheduler_evidence)
                                          if rwwpo2_enabled else (
                                              digest(self.actor_lr_scheduler.state_dict())
                                              if self.actor_lr_scheduler is not None else digest(None)),
                                          "scaler": "not_applicable_bfloat16",
                                          "rng": digest(rng_snapshot())}
                        if not accepted and commit_digests != pre_digests:
                            raise RuntimeError("RWWPO_TRANSACTION_ROLLBACK_DIGEST_MISMATCH")
                    else:
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
                                writer_log_ratio_cap=ratio_cap,
                                writer_objective=(rwwpo_objective
                                                  if rwwpo_objective != "original_tokenwise"
                                                  else "whole_prefix"))
                        post_prefix_rows = with_root_identities(post_metrics["prefix_log_ratios"])
                        post_prefix_stats = build_global_stats(post_prefix_rows)
                        full_log_prob = post_log_prob
                        trial_rows = [{"alpha": 1.0, "feasible": True,
                                       "log_prob": post_log_prob.detach().cpu(),
                                       "prefix_rows": post_prefix_rows, "prefix_stats": post_prefix_stats}]
                        accepted = not proposal_zero
                        alpha_committed = 1.0 if accepted else 0.0
                        commit_params = parameter_snapshot(self.actor_module)
                        commit_digests = {"model": digest(commit_params),
                                          "optimizer": digest(self.actor_optimizer.state_dict()),
                                          "scheduler": digest(self.actor_lr_scheduler.state_dict())
                                          if self.actor_lr_scheduler is not None else digest(None),
                                          "scaler": "not_applicable_bfloat16", "rng": digest(rng_snapshot())}
                        trial_forward_wall_seconds = 0.0
                    committed_displacement=displacement_norm(snapshot[0],commit_params)
                    relative_committed_displacement=relative_displacement_norm(snapshot[0],commit_params)
                    if torch.distributed.is_initialized():
                        displacement_values=torch.tensor([
                            committed_displacement**2,
                            sum(float(before.double().square().sum()) for before in snapshot[0]),
                        ],dtype=torch.float64,device=torch.cuda.current_device())
                        torch.distributed.all_reduce(displacement_values,op=torch.distributed.ReduceOp.SUM)
                        committed_displacement=float(displacement_values[0].sqrt().item())
                        relative_committed_displacement=float(
                            displacement_values[0].sqrt().item() /
                            max(displacement_values[1].sqrt().item(),1e-30))
                    local_writer_mse_sum, local_writer_trajectory_count = (
                        writer_logprob_rms_sufficient_statistics(
                            post_log_prob, current_log_prob, writer_mask,
                            joined("sample_index")))
                    writer_sums=torch.tensor([
                        local_writer_mse_sum, float(local_writer_trajectory_count),
                    ],dtype=torch.float64,device=torch.cuda.current_device())
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(writer_sums,op=torch.distributed.ReduceOp.SUM)
                    committed_writer_rms=float((writer_sums[0]/writer_sums[1]).sqrt().item())
                    if rwwpo2_enabled and inner_id==1:
                        rwwpo2_inner1_exposure={
                            "relative_parameter_displacement": relative_committed_displacement,
                            "writer_logprob_rms": committed_writer_rms,
                            "off_behavior_exposed": off_behavior_exposed(
                                relative_parameter_displacement=relative_committed_displacement,
                                writer_logprob_rms_value=committed_writer_rms,
                                tau_theta=float(rwwpo_config["tau_theta"]),
                                tau_logprob=float(rwwpo_config["tau_logprob"]),
                            ),
                        }
                    from recurrent.research.rwwpo_ledger import append_actual_loss_record
                    behavior_point_max_delta=float((current_log_prob-old_log_prob)[response_mask].abs().max().item())
                    if torch.distributed.is_initialized():
                        worst=torch.tensor(behavior_point_max_delta,dtype=torch.float64,
                                           device=torch.cuda.current_device())
                        torch.distributed.all_reduce(worst,op=torch.distributed.ReduceOp.MAX)
                        behavior_point_max_delta=float(worst.item())
                    append_actual_loss_record(
                        ledger_dir=rwwpo_config.get("ledger_dir"), attempt_id=rwwpo_config.get("attempt_id"),
                        mode="original_collection" if rwwpo_collect_original else "rwwpo_method", rank=rank,
                        global_step=int(joined("rwwpo_global_step")[0].item()), epoch=epoch,
                        minibatch=batch_idx, old_log_prob=old_log_prob, current_log_prob=current_log_prob,
                        ref_log_prob=(joined("ref_log_prob") if rwwpo2_enabled else None),
                        proposed_post_log_prob=full_log_prob if rwwpo_enabled else post_log_prob,
                        committed_log_prob=post_log_prob, response_mask=response_mask,
                        writer_mask=writer_mask, answer_mask=rwwpo_metrics["answer_mask"],
                        trajectory_turn=joined("trajectory_turn"), sample_index=joined("sample_index"),
                        example_identity_hash=joined("rwwpo_example_identity_hash"),
                        trajectory_identity_hash=joined("rwwpo_trajectory_identity_hash"),
                        advantages=joined("advantages"), denominator=rwwpo_metrics["denominator"].item(),
                        prefix_rows=local_prefix_rows, prefix_stats=global_prefix_stats,
                        post_prefix_rows=post_prefix_rows, post_prefix_stats=post_prefix_stats, q_min=q_min,
                        root_q_min=root_q_min,
                        writer_log_ratio_cap=ratio_cap,
                        constraint_pass=constraint_pass, accepted=accepted,
                        objective_variant=rwwpo_objective, controller_variant=rwwpo_controller,
                        alpha_grid=list(ALPHA_GRID), alpha_test_order=[row["alpha"] for row in trial_rows],
                        alpha_committed=alpha_committed, accepted_nonzero=accepted,
                        proposal_zero=proposal_zero, trial_evidence=trial_rows,
                        full_parameter_displacement_norm=full_displacement,
                        committed_parameter_displacement_norm=committed_displacement,
                        pre_digests=pre_digests, commit_digests=commit_digests,
                        trial_forward_wall_seconds=trial_forward_wall_seconds if rwwpo_enabled else 0.0,
                        mechanism_diagnostics={
                            "policy_loss": float(policy_loss.detach().item()),
                            "surrogate_loss": float(
                                (policy_loss - shared_additive_loss).detach().item()),
                            "shared_kl_loss": float(shared_additive_loss.detach().item()),
                            "actual_loss_contract": {
                                "loss_agg_mode": str(self.config.loss_agg_mode),
                                "cliprange": float(self.config.clip_ratio),
                                "cliprange_low": float(
                                    self.config.clip_ratio if self.config.clip_ratio_low is None
                                    else self.config.clip_ratio_low),
                                "cliprange_high": float(
                                    self.config.clip_ratio if self.config.clip_ratio_high is None
                                    else self.config.clip_ratio_high),
                                "clip_ratio_c": float(self.config.get("clip_ratio_c", 3.0)),
                                "writer_log_ratio_cap": float(ratio_cap),
                                "use_kl_loss": bool(self.config.use_kl_loss),
                                "kl_loss_type": str(self.config.kl_loss_type),
                                "kl_loss_coefficient": float(self.config.kl_loss_coef),
                                "entropy_coefficient": float(self.config.entropy_coeff),
                            },
                            "active_logprob_gradient_l2": float(
                                logprob_gradient[response_mask].detach().double().norm().item()),
                            "optimizer_step_calls": transaction_optimizer_step_calls,
                            "token_approx_kl": rwwpo_metrics["token_approx_kl"],
                            "token_clipfrac": rwwpo_metrics["token_clipfrac"],
                            "per_write_rows": rwwpo_metrics["per_write_rows"],
                            "per_write_stats": rwwpo_metrics["per_write_stats"],
                            "covariance_diagnostics": rwwpo_metrics["covariance_diagnostics"],
                            "whole_prefix_tokenwise_gradient_cosine": coefficient_cosine,
                            "shadow_coefficients": shadow_coefficient_diagnostics,
                            "shadow_parameter_gradient_sketches": shadow_parameter_gradient_sketches,
                            "shadow_parameter_gradient_pairwise_relative":
                                shadow_parameter_gradient_pairwise_relative,
                            "gradient_sketch_chunk_elements":
                                RWWPO2_GRADIENT_SKETCH_CHUNK_ELEMENTS,
                            "host_variant": str(rwwpo_config.get("cell", "legacy")),
                            "behavior_batch_digest": frozen_digest,
                            "round_id": round_id,
                            "inner_id": inner_id,
                            "proposal_clock": logical_proposal_id,
                            "proposal_lr": proposal_lr,
                            "logical_seed": logical_seed,
                            "accepted_optimizer_clock_before": accepted_optimizer_clock_before,
                            "accepted_optimizer_clock_after": (
                                accepted_optimizer_clock_before + int(bool(accepted))
                                if accepted_optimizer_clock_before is not None else None),
                            "transaction_entry_rng_digest": pre_digests["rng"],
                            "logical_seeded_rng_digest": digest(logical_seeded_rng),
                            "proposal_gradient_rng_digest": digest(
                                proposal_gradient_rng),
                            "terminal_rng_digest": commit_digests["rng"],
                            "inner1_exposure": rwwpo2_inner1_exposure,
                        }, gradient_norm=float(grad_norm.detach().item()),
                        program_version=("rwwpo2-k2" if rwwpo2_enabled else "legacy"),
                        inner_id=inner_id, proposal_clock=logical_proposal_id,
                        accepted_optimizer_clock_before=accepted_optimizer_clock_before,
                        accepted_optimizer_clock_after=(
                            accepted_optimizer_clock_before + int(bool(accepted))
                            if accepted_optimizer_clock_before is not None else None),
                        logical_seed=logical_seed,
                        experiment_seed=int(rwwpo_config.get("experiment_seed", 2026)),
                        host_variant=str(rwwpo_config.get("cell", "legacy")),
                        behavior_batch_digest=frozen_digest)
                    if rwwpo_enabled:
                        append_transaction_marker(
                            ledger_dir=rwwpo_config.get("ledger_dir"),attempt_id=rwwpo_config.get("attempt_id"),
                            rank=rank,global_step=int(joined("rwwpo_global_step")[0].item()),epoch=epoch,
                            minibatch=batch_idx,phase="complete",model_digest=commit_digests["model"],
                            inner_id=inner_id, proposal_clock=logical_proposal_id)
                    append_to_dict(metrics, {
                        "actor/pg_loss": policy_loss.detach().item(),
                        "actor/pg_clipfrac": rwwpo_metrics["answer_clipfrac"].detach().item(),
                        "actor/ppo_kl": rwwpo_metrics["answer_ppo_kl"].detach().item(),
                        "actor/pg_clipfrac_lower": rwwpo_metrics["answer_clipfrac_lower"].detach().item(),
                        "rwwpo/min_prefix_ess": min(row["ess_fraction"] for row in global_prefix_stats),
                        "rwwpo/min_prefix_root_ess": min(
                            row["root_ess_fraction"] for row in global_prefix_stats),
                        "rwwpo/max_root_loo_flip_fraction": max(
                            row["root_loo_feasibility_flip_fraction"]
                            for row in global_prefix_stats),
                    })
                    append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item(),
                                             "rwwpo/update_accepted": float(accepted),
                                             "rwwpo/alpha_committed": float(alpha_committed),
                                             "rwwpo/post_min_prefix_ess": min(row["ess_fraction"] for row in post_prefix_stats),
                                             "rwwpo/post_min_prefix_root_ess": min(
                                                 row["root_ess_fraction"] for row in post_prefix_stats),
                                             "rwwpo/post_max_abs_prefix_log_ratio": max(row["max_abs_log_ratio"] for row in post_prefix_stats),
                                             "rwwpo/behavior_point_max_delta": behavior_point_max_delta,
                                             "rwwpo/scheduler_managed_transactionally": float(not rwwpo2_enabled),
                                             "rwwpo/proposal_clock": float(logical_proposal_id or 0),
                                             "rwwpo/proposal_lr": float(proposal_lr or 0),
                                             "rwwpo/off_behavior_exposed": float(
                                                 bool((rwwpo2_inner1_exposure or {}).get(
                                                     "off_behavior_exposed",False)))})
                    # The legacy non-RWWPO path appends its final scalar metric
                    # dictionary once more after the minibatch loop.  Do not let
                    # that epilogue see this branch's GPU minibatch container:
                    # doing so leaks CUDA tensors into reduce_metrics/NumPy after
                    # an otherwise successful optimizer update.
                    data = {}
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
