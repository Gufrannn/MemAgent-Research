# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import logging
import hashlib
from contextlib import contextmanager
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from codetiming import Timer

from verl import DataProto

from .interface import RAgent, RConfig
from .research.stable_eval_identity import validate_configured_request_binding
from .research.trajectory_seeding import derive_turn_request_seeds
from .utils import (chat_template, create_attention_mask, create_position_ids,
                    graceful_padding, indexing_proto,
                    pad_tensor_list_to_length)

logger = logging.getLogger(__file__)
logger.setLevel('INFO')



@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timing_raw.get(name, 0.) + timer.last




class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: RConfig,
        agent_cls: Type[RAgent]
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.world_size = actor_rollout_wg.world_size
        self.agent = agent_cls(tokenizer, config)
        self.chat_template = chat_template(tokenizer)
        self.PADDING_WORD_TOKENS = tokenizer.encode(self.chat_template.format(message="Hello."), add_special_tokens=False)


    from functools import lru_cache
    @lru_cache(maxsize=3)
    def get_paddings(self, shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return padding_token_ids, padding_attention_masks, padding_position_ids
        """
        pad_shape = shape[1:]
        padding_word_ids = self.PADDING_WORD_TOKENS
        padding_token_ids = torch.full(pad_shape, fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        padding_attention_masks = torch.zeros(pad_shape, dtype=torch.long)
        padding_position_ids = torch.zeros(pad_shape, dtype=torch.long)
        # token_ids <pad> <pad> <pad> <tok> <tok> <tok>
        # attn_mask 0     0     0     1     1     1
        # posit_ids 0     0     0     0     1     2
        padding_token_ids[-len(padding_word_ids):] = torch.tensor(padding_word_ids, dtype=torch.long)
        padding_attention_masks[-len(padding_word_ids):] = 1
        padding_position_ids[-len(padding_word_ids):] = torch.arange(0, len(padding_word_ids))
        return padding_token_ids, padding_attention_masks, padding_position_ids
    
    def generate_with_graceful_padding(self, input_ids: torch.Tensor,
                                    attention_masks: torch.Tensor,
                                    position_ids: torch.Tensor,
                                    meta_info: dict):

        """
        batch may not be divisible by wordsize.
        Use "Hello" as padding, insert padding data into batch so that data 
        """
        bsz = input_ids.shape[0]
        meta_info = dict(meta_info)
        request_seeds = meta_info.pop("request_seeds", None)
        if request_seeds is not None and len(request_seeds) != bsz:
            raise ValueError(f"request_seeds must align with active requests: {len(request_seeds)} != {bsz}")

        group_nums = self.world_size
        remainder = bsz % group_nums
        if remainder:
            # Example pattern for bsz=7, group_nums=3:
            # no_padding_mask: [1, 1, 1, 0, 1, 1, 0, 1, 1]
            # padding_index:   [0, 1, 2, -1, 3, 4, -1, 5, 6]
            padding_index, no_padding_mask = graceful_padding(bsz, group_nums)
            padding_token_ids, padding_attention_masks, padding_position_ids = self.get_paddings(input_ids.shape)
            def padding_by_index(tensor, padding, padding_index):
                if not len(padding.shape) == 2:
                    padding = padding.unsqueeze(0)
                # 2. prepare data for padding, concat padding to the end of batch
                tensor_for_indexing = torch.cat([tensor, padding], dim=0)
                # 3. index, -1 will select padding, else select the corresponding original data 
                return tensor_for_indexing[padding_index]
            
            input_ids = padding_by_index(input_ids, padding_token_ids, padding_index)
            attention_masks = padding_by_index(attention_masks, padding_attention_masks, padding_index)
            position_ids = padding_by_index(position_ids, padding_position_ids, padding_index)
            if request_seeds is not None:
                request_seeds = [
                    int(request_seeds[int(index)]) if int(index) >= 0 else 0
                    for index in padding_index
                ]

        non_tensors = None
        if request_seeds is not None:
            non_tensors = {"request_seeds": np.asarray(request_seeds, dtype=np.uint64)}
        batch = DataProto.from_dict(tensors={
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_masks
        }, non_tensors=non_tensors, meta_info=meta_info)
        output_batch = self.actor_rollout_wg.generate_sequences(batch)
        if remainder:
            # 4. remove padding
            output_batch = indexing_proto(output_batch, no_padding_mask)
        return output_batch

    def run_llm_loop(self, gen_batch, timing_raw) -> Tuple[DataProto, torch.BoolTensor, torch.LongTensor]:
        """Run main LLM generation loop.
        genbatch: 'context_ids','context_length','prompt_ids'
        timing_raw: timing dict used in ray_trainer, note that we will accumulate the time cost in this loop, instead of override each time as in ray_trainer.
        see `_timer` implementation at the top of this file for more details.
        """
        active_num_list = [] # trace the active number of sample in each turn
        gen_output_list = [] # store I/O batch in each turn, used for policy optimization
        meta_info = gen_batch.meta_info #  do_sample, is_validate, eos/pad are stored in here.
        pad_token_id = self.tokenizer.pad_token_id
        self.agent.start(gen_batch, timing_raw)
        trajectory_base_seeds = meta_info.get("trajectory_base_seeds")
        strict_eval_identity = bool(meta_info.get("strict_eval_identity", False))
        stable_eval_identity = meta_info.get("stable_eval_identity")
        if strict_eval_identity:
            if trajectory_base_seeds is None:
                raise ValueError("strict stable evaluation identity requires trajectory_base_seeds")
            if not isinstance(stable_eval_identity, dict) or not stable_eval_identity:
                raise ValueError("strict stable evaluation identity requires row-aligned identity columns")
            identity_lengths = {key: len(values) for key, values in stable_eval_identity.items()}
            if set(identity_lengths.values()) != {len(gen_batch)}:
                raise ValueError(
                    "stable evaluation identity columns must align with recurrent input rows: "
                    f"batch={len(gen_batch)}, columns={identity_lengths}"
                )
        recurrent_turn = 0
        # Main generation loop, agent should indicate whether to stop
        while not self.agent.done():
            with _timer('mt_prepare', timing_raw):
                messages, meta_info_gen = self.agent.action()
                meta_info_gen.update(meta_info)
                if trajectory_base_seeds is not None:
                    active_sample_indices = self.agent.sample_index_list[-1].tolist()
                    meta_info_gen["request_seeds"] = derive_turn_request_seeds(
                        trajectory_base_seeds, active_sample_indices, recurrent_turn
                    )
                    meta_info_gen["trajectory_seed_turn"] = recurrent_turn
                # [len(x) for x in messages] == [len(x[x!=pad_token_id]) for x in input_ids]
                # torch.all(attention_masks.sum(-1) == torch.tensor([len(x[x!=pad_token_id]) for x in input_ids]))
                input_ids = pad_tensor_list_to_length(messages, 
                                                pad_token_id=pad_token_id,
                                                max_length=meta_info_gen['input_pad_to'], 
                                                left_pad=True)
                attention_masks = create_attention_mask(input_ids, pad_token_id=pad_token_id)
                position_ids = create_position_ids(attention_masks)
                active_num_list.append(len(messages))
                logger.info(f'padding done')
            with _timer('mt_gen', timing_raw):
                gen_output = self.generate_with_graceful_padding(input_ids, attention_masks, position_ids, meta_info_gen)
                logger.info('generation done')
            with _timer('mt_update', timing_raw):
                gen_output = self.agent.update(gen_output)
                if bool(meta_info.get("mic_capture_post_write", False)):
                    current_final = self.agent.final_mask_list[-1]
                    if len(current_final) != len(gen_output):
                        raise ValueError("MIC_NO_GO: final mask is not row-aligned for state capture")
                    responses = gen_output.batch["responses"].detach().cpu()
                    memories = []
                    memory_hashes = []
                    for response, is_final in zip(responses, current_final.tolist()):
                        if is_final:
                            memories.append(None)
                            memory_hashes.append(None)
                            continue
                        token_ids = response[response != pad_token_id].tolist()
                        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                        memories.append(text)
                        memory_hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
                    gen_output.non_tensor_batch["mic_materialized_memory"] = np.asarray(
                        memories, dtype=object
                    )
                    gen_output.non_tensor_batch["mic_materialized_memory_sha256"] = np.asarray(
                        memory_hashes, dtype=object
                    )
                    gen_output.batch["mic_turn_index"] = torch.full(
                        (len(gen_output),), recurrent_turn + 1, dtype=torch.long,
                        device=gen_output.batch["responses"].device,
                    )
                if trajectory_base_seeds is not None:
                    if len(gen_output) != len(active_sample_indices):
                        raise ValueError(
                            "trajectory turn identity is not row-aligned after recurrent agent update: "
                            f"output={len(gen_output)}, active={len(active_sample_indices)}"
                        )
                    gen_output.batch["trajectory_turn"] = torch.full(
                        (len(gen_output),), recurrent_turn, dtype=torch.long,
                        device=gen_output.batch["responses"].device,
                    )
                    gen_output.non_tensor_batch["request_seed"] = np.asarray(
                        meta_info_gen["request_seeds"], dtype=np.uint64
                    )
                    if strict_eval_identity:
                        configured = gen_output.non_tensor_batch.get(
                            "configured_request_seed"
                        )
                        request_prompt_hashes = gen_output.non_tensor_batch.get(
                            "request_prompt_token_sha256"
                        )
                        returned_prompt_hashes = gen_output.non_tensor_batch.get(
                            "returned_prompt_token_sha256"
                        )
                        worker_ranks = gen_output.non_tensor_batch.get("rollout_worker_rank")
                        expected = np.asarray(meta_info_gen["request_seeds"], dtype=np.uint64)
                        if any(
                            value is None
                            for value in (
                                configured,
                                request_prompt_hashes,
                                returned_prompt_hashes,
                                worker_ranks,
                            )
                        ):
                            raise ValueError(
                                "strict stable evaluation identity requires configured seed, "
                                "returned prompt-token, and worker-rank evidence"
                            )
                        validate_configured_request_binding(
                            expected,
                            configured,
                            request_prompt_hashes,
                            returned_prompt_hashes,
                            worker_ranks,
                        )
                        active_identity_indices = np.asarray(active_sample_indices, dtype=np.int64)
                        for key, values in stable_eval_identity.items():
                            gen_output.non_tensor_batch[key] = np.asarray(values, dtype=object)[
                                active_identity_indices
                            ]
                        gen_output.non_tensor_batch["active_sample_index"] = active_identity_indices
                        gen_output.non_tensor_batch["is_final"] = np.asarray(
                            self.agent.final_mask_list[-1].tolist(), dtype=bool
                        )
                gen_output_list.append(gen_output)
                logger.info('agent update done')
                recurrent_turn += 1
        final_mask, sample_index = self.agent.end()
        
        # OK, now we've got all we need in gen_output_list, and the final_mask indicates which one is final answer.
        assert len(sample_index) == sum(active_num_list)
        assert sum(final_mask) == len(gen_batch)
        logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")
        return DataProto.concat(gen_output_list), final_mask, sample_index # pyright: ignore
