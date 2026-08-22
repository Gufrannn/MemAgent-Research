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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import math
import os
import re
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    #######
    # MODIFIED: use loss_mask directly
    #######
    response_mask = compute_response_mask(data)

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    if "response_mask" in data.batch:
        return data.batch["response_mask"]
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, norm_adv_by_std_in_grpo=True):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data

def compute_1D_grpo_advantage(token_level_rewards: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   use_adv: bool = True):
    """
    adapted from `core_algos`, but return 1D advantage only.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs,)
        Returns: `(torch.Tensor)`
            shape: (bs,)
    """
    scores = token_level_rewards.sum(dim=-1)
    from collections import defaultdict
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                if use_adv:
                    id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                if use_adv:
                    id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if use_adv:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
    return scores


def _append_rollout_seed_audit(output_dir: str, records: list[dict[str, object]]) -> None:
    """Append exact per-trajectory seeds without touching training semantics."""
    if not records:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "rollout_seed_audit.jsonl")
    with open(path, "a", encoding="utf-8") as audit_file:
        for record in records:
            audit_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _to_python_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _build_validation_identities(
    test_batch,
    *,
    val_n: int,
    resolved_manifest,
    base_seed: int,
    interface_id: str,
    attempt_id: str,
):
    """Build frozen evaluation identities before recurrent pop/reorder."""
    from recurrent.research.stable_eval_identity import (
        build_stable_eval_identities,
        identity_rows_to_columns,
    )

    if "index" in test_batch.batch:
        semantic_indices = _to_python_list(test_batch.batch["index"])
    elif "index" in test_batch.non_tensor_batch:
        semantic_indices = _to_python_list(test_batch.non_tensor_batch["index"])
    else:
        raise ValueError("strict evaluation identity requires dataset index; UUID/text fallback is forbidden")
    if "source_order_index" in test_batch.batch:
        source_order_indices = _to_python_list(test_batch.batch["source_order_index"])
    elif "source_order_index" in test_batch.non_tensor_batch:
        source_order_indices = _to_python_list(test_batch.non_tensor_batch["source_order_index"])
    else:
        raise ValueError(
            "strict evaluation identity requires source_order_index; semantic index cannot be used as a fallback"
        )
    rows = build_stable_eval_identities(
        semantic_indices=semantic_indices,
        source_order_indices=source_order_indices,
        replicas=int(val_n),
        base_seed=int(base_seed),
        interface_id=str(interface_id),
        attempt_id=str(attempt_id),
        resolved_manifest=resolved_manifest,
    )
    return identity_rows_to_columns(rows)


def _response_token_sha256(response: torch.Tensor) -> str:
    values = (
        response.detach().cpu().to(dtype=torch.int64).contiguous().numpy().astype("<i8", copy=False)
    )
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _append_stable_eval_turn_ledger(path: str, output_batch: DataProto) -> None:
    """Append row-aligned recurrent request evidence after vLLM prompt checks."""
    from recurrent.research.stable_eval_identity import (
        TURN_LEDGER_NON_TENSOR_FIELDS,
        trajectory_turn_record_from_columns,
        validate_configured_request_binding,
    )

    required_non_tensor = TURN_LEDGER_NON_TENSOR_FIELDS
    missing = [key for key in required_non_tensor if key not in output_batch.non_tensor_batch]
    if missing:
        raise ValueError(f"stable evaluation turn ledger is missing row fields: {missing}")
    if "trajectory_turn" not in output_batch.batch or "responses" not in output_batch.batch:
        raise ValueError("stable evaluation turn ledger requires trajectory_turn and responses tensors")
    ledger_parent = os.path.dirname(path)
    if ledger_parent:
        os.makedirs(ledger_parent, exist_ok=True)
    validate_configured_request_binding(
        output_batch.non_tensor_batch["request_seed"],
        output_batch.non_tensor_batch["configured_request_seed"],
        output_batch.non_tensor_batch["request_prompt_token_sha256"],
        output_batch.non_tensor_batch["returned_prompt_token_sha256"],
        output_batch.non_tensor_batch["rollout_worker_rank"],
    )
    requested = [int(value) for value in output_batch.non_tensor_batch["request_seed"]]
    rollout_alias = [
        int(value) for value in output_batch.non_tensor_batch["rollout_request_seed"]
    ]
    if rollout_alias != requested:
        raise ValueError(
            "stable evaluation rollout seed alias is not row-aligned with configured requests: "
            f"{rollout_alias} != {requested}"
        )
    with open(path, "a", encoding="utf-8") as stream:
        for row in range(len(output_batch)):
            record = trajectory_turn_record_from_columns(
                output_batch.non_tensor_batch,
                row=row,
                trajectory_turn=int(output_batch.batch["trajectory_turn"][row].item()),
                response_token_sha256=_response_token_sha256(
                    output_batch.batch["responses"][row]
                ),
            )
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self._stable_eval_runtime_config_sha256 = None
        expected_t25_config_sha = os.environ.get(
            "ORIGINAL_T25_EXPECTED_RUNTIME_CONFIG_SHA256", ""
        )
        if expected_t25_config_sha:
            from recurrent.research.gate_a_execution import (
                append_gate_a_record,
                gate_a_enabled,
                runtime_config_sha256,
            )

            if not gate_a_enabled():
                raise ValueError(
                    "T25 runtime-config attestation requires the append-only evidence writer"
                )
            resolved_t25_config = OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            )
            actual_t25_config_sha = runtime_config_sha256(resolved_t25_config)
            if actual_t25_config_sha != expected_t25_config_sha:
                raise ValueError(
                    "T25 resolved Hydra config differs from P0: "
                    f"actual={actual_t25_config_sha}, expected={expected_t25_config_sha}"
                )
            override_argv_sha = os.environ.get(
                "ORIGINAL_T25_TRAINER_OVERRIDE_ARGV_SHA256", ""
            )
            if re.fullmatch(r"[0-9a-f]{64}", override_argv_sha) is None:
                raise ValueError("T25 trainer override argv SHA-256 is missing or invalid")
            append_gate_a_record(
                "runtime_config",
                resolved_config_sha256=actual_t25_config_sha,
                override_argv_sha256=override_argv_sha,
            )
        early_eval_identity = config.trainer.get("eval_identity", None)
        if early_eval_identity is not None and bool(
            early_eval_identity.get("enabled", False)
        ):
            from recurrent.research.stable_eval_identity import (
                stable_eval_runtime_config_sha256,
            )

            resolved_config = OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            )
            actual_config_sha = stable_eval_runtime_config_sha256(resolved_config)
            expected_config_sha = str(
                early_eval_identity.get("expected_runtime_config_sha256", "")
            )
            if actual_config_sha != expected_config_sha:
                raise ValueError(
                    "strict stable evaluation resolved Hydra config differs from P0: "
                    f"actual={actual_config_sha}, expected={expected_config_sha}"
                )
            self._stable_eval_runtime_config_sha256 = actual_config_sha
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self._actor_update_calls = 0
        self._stable_eval_actor_checkpoint_load_acks = None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        if self.reward_fn and self.val_reward_fn:
            self.reward_fn.trainer = self
            self.val_reward_fn.trainer = self
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        eval_identity_config = config.trainer.get("eval_identity", None)
        if eval_identity_config is not None and bool(eval_identity_config.get("enabled", False)):
            failures = []
            if not bool(config.trainer.get("val_before_train", False)):
                failures.append("trainer.val_before_train must be true")
            if not bool(config.trainer.get("val_only", False)):
                failures.append("trainer.val_only must be true")
            if str(config.actor_rollout_ref.rollout.name) != "vllm":
                failures.append("actor_rollout_ref.rollout.name must be vllm")
            if str(config.actor_rollout_ref.rollout.mode) != "sync":
                failures.append("only synchronous vLLM is admitted by this frozen canary")
            if str(config.recurrent.enable) != "memory":
                failures.append("recurrent.enable must be memory")
            if not bool(config.data.get("include_source_order_index", False)):
                failures.append("data.include_source_order_index must be true")
            if int(config.actor_rollout_ref.rollout.val_kwargs.n) != int(
                eval_identity_config.get("replicas", 0)
            ):
                failures.append("validation n must equal eval_identity.replicas")
            if bool(config.actor_rollout_ref.rollout.val_kwargs.do_sample):
                failures.append("validation must be deterministic (do_sample=false)")
            if float(config.actor_rollout_ref.rollout.val_kwargs.temperature) != 0.0:
                failures.append("validation temperature must be 0")
            if float(config.actor_rollout_ref.rollout.val_kwargs.top_p) != 1.0:
                failures.append("validation top_p must be 1")
            if int(config.actor_rollout_ref.rollout.val_kwargs.top_k) != -1:
                failures.append("validation top_k must be -1")
            if int(config.trainer.get("save_freq", 0)) != -1:
                failures.append("trainer.save_freq must be -1")
            weight_source = str(
                eval_identity_config.get("weight_source", "base_model")
            )
            expected_global_step = int(
                eval_identity_config.get("expected_global_step", 0)
            )
            if weight_source == "base_model":
                if str(config.trainer.get("resume_mode", "")) != "disable":
                    failures.append("base-model evaluation requires trainer.resume_mode=disable")
                if config.trainer.get("resume_from_path", None) not in (None, ""):
                    failures.append("base-model evaluation requires an empty resume_from_path")
                if expected_global_step != 0:
                    failures.append("base-model evaluation expected_global_step must be zero")
            elif weight_source == "actor_checkpoint":
                if str(config.trainer.get("resume_mode", "")) != "actor_only_eval":
                    failures.append(
                        "checkpoint evaluation requires trainer.resume_mode=actor_only_eval"
                    )
                checkpoint_path = config.trainer.get("resume_from_path", None)
                if not isinstance(checkpoint_path, str) or not checkpoint_path:
                    failures.append("checkpoint evaluation requires an explicit resume_from_path")
                elif os.path.basename(os.path.realpath(checkpoint_path)) != (
                    f"global_step_{expected_global_step}"
                ):
                    failures.append(
                        "checkpoint path basename must exactly equal "
                        "eval_identity.expected_global_step"
                    )
                if expected_global_step <= 0:
                    failures.append("checkpoint evaluation expected_global_step must be positive")
            else:
                failures.append(
                    f"trainer.eval_identity.weight_source is invalid: {weight_source}"
                )
            if not config.trainer.get("validation_data_dir", None):
                failures.append("trainer.validation_data_dir is required")
            for key in (
                "resolved_manifest_path",
                "expected_manifest_hash",
                "interface_id",
                "attempt_id",
                "turn_ledger_path",
                "execution_summary_path",
                "expected_runtime_config_sha256",
            ):
                if not eval_identity_config.get(key, None):
                    failures.append(f"trainer.eval_identity.{key} is required")
            if not os.environ.get("GATE_A_WEIGHT_DIGEST_PARAMETERS", ""):
                failures.append("sampled-weight parameter list is required for read-only evaluation audit")
            if not os.environ.get("GATE_A_WEIGHT_DIGEST_SAMPLES", ""):
                failures.append("sampled-weight sample count is required for read-only evaluation audit")
            if os.environ.get("GATE_A_FROZEN_AUDIT", "0") == "1":
                failures.append("Gate A evidence writer must be disabled for stable evaluation")
            if int(config.data.get("val_max_samples", 0)) != int(
                eval_identity_config.get("examples", 0)
            ):
                failures.append("data.val_max_samples must equal eval_identity.examples")
            if failures:
                raise ValueError("strict stable evaluation identity configuration is invalid: " + "; ".join(failures))

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"
            
        # TODO: check consistency with implementation in `init_worker`
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True

        if config.recurrent.enable:
            from recurrent.interface import RRegister
            enabled_conf = getattr(config.recurrent, config.recurrent.enable)
            if not self.async_rollout_mode:
                self.recurrent_register = RRegister.from_filename(enabled_conf.path, enabled_conf.name)
            else:
                assert config.actor_rollout_ref.rollout.chat_scheduler == "recurrent.async_utils.ChatCompletionProxy", \
                    "`chat_scheduler` must be set to `recurrent.async_utils.ChatCompletionProxy` when using async rollout in Recurrent RL"
                assert hasattr(enabled_conf, "async_path") and hasattr(enabled_conf, "async_name"), \
                    f"`async_path` and `async_name` must be set in recurrent.{config.recurrent.enable} when using async rollout"
                self.recurrent_register = RRegister.from_filename(enabled_conf.async_path, enabled_conf.async_name)

            conf = dict(enabled_conf.config) if enabled_conf.config is not None else {}
            self.recurrent_config = self.recurrent_register.config_cls(**conf)
        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        if self.config.recurrent.enable:
            self.train_dataset = self.recurrent_register.dataset_cls(
                recurrent_config=self.recurrent_config,
                data_config=self.config.data,
                data_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
        else:    
            from verl.utils.import_utils import load_extern_type
            if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
                dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
                if not issubclass(dataset_cls, Dataset):
                    raise TypeError(f"The custom dataset class '{self.config.data.custom_cls.name}' from '{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset")
            else:
                dataset_cls = RLHFDataset
            self.train_dataset = dataset_cls(
                data_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                config=self.config.data,
            )
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=self.config.data.get("dataloader_num_workers", 8),
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)
        if self.config.recurrent.enable:
            self.val_dataset = self.recurrent_register.dataset_cls(
                recurrent_config=self.recurrent_config,
                data_config=self.config.data,
                data_files=self.config.data.val_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
        else:
            dataset_cls: type[RLHFDataset] # important for static type checking
            self.val_dataset = dataset_cls(
                data_files=self.config.data.val_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                config=self.config.data,
            )
        val_max_samples = self.config.data.get("val_max_samples", None)
        if val_max_samples is not None:
            val_max_samples = int(val_max_samples)
            if val_max_samples < 1 or val_max_samples > len(self.val_dataset):
                raise ValueError(
                    f"data.val_max_samples must be in [1, {len(self.val_dataset)}], got {val_max_samples}"
                )
            if not hasattr(self.val_dataset, "dataframe") or not hasattr(self.val_dataset.dataframe, "select"):
                raise TypeError("val_max_samples requires a dataset with deterministic dataframe.select support")
            self.val_dataset.dataframe = self.val_dataset.dataframe.select(range(val_max_samples))
        # consider the design of single controller with a large val dataset in multi-modal scenarios
        # may lead to oom issues
        val_batch_size = self.config.data.val_batch_size or len(self.val_dataset)
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, identities=None):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        if identities is not None:
            for key, values in identities.items():
                if len(values) != n:
                    raise ValueError(f"identity column {key} has {len(values)} rows, expected {n}")
                base_data[key] = values

        # Stable-identity evidence is append-only across preregistered attempt
        # directories.  Refuse an accidental rerun/overwrite at the same path.
        open_mode = "x" if identities is not None else "w"
        with open(filename, open_mode, encoding="utf-8") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        eval_identity_config = self.config.trainer.get("eval_identity", None)
        strict_eval_identity = bool(
            eval_identity_config is not None and eval_identity_config.get("enabled", False)
        )
        resolved_eval_manifest = None
        turn_ledger_path = None
        if strict_eval_identity:
            from recurrent.research.stable_eval_identity import load_resolved_manifest

            resolved_eval_manifest = load_resolved_manifest(
                str(eval_identity_config.resolved_manifest_path),
                expected_hash=str(eval_identity_config.expected_manifest_hash),
            )
            turn_ledger_path = os.path.realpath(str(eval_identity_config.turn_ledger_path))
            turn_ledger_parent = os.path.dirname(turn_ledger_path)
            if turn_ledger_parent:
                os.makedirs(turn_ledger_parent, exist_ok=True)
            try:
                with open(turn_ledger_path, "x", encoding="utf-8"):
                    pass
            except FileExistsError as error:
                raise FileExistsError(
                    "stable evaluation turn ledger already exists; preregistered attempts may not overwrite evidence: "
                    f"{turn_ledger_path}"
                ) from error

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_identities: dict[str, list] = defaultdict(list)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            val_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
            validation_seed_records = None
            identity_batch = None
            if strict_eval_identity:
                identity_batch = _build_validation_identities(
                    test_batch,
                    val_n=val_n,
                    resolved_manifest=resolved_eval_manifest,
                    base_seed=int(eval_identity_config.base_seed),
                    interface_id=str(eval_identity_config.interface_id),
                    attempt_id=str(eval_identity_config.attempt_id),
                )
                validation_seeds = [int(value) for value in identity_batch["trajectory_seed"]]
            else:
                trajectory_seed_mode = self.config.actor_rollout_ref.rollout.get("trajectory_seed_mode", None)
                if trajectory_seed_mode not in (None, "", "legacy_shared"):
                    from recurrent.research.trajectory_seeding import build_trajectory_seed_records

                    validation_seed_records = build_trajectory_seed_records(
                        base_seed=int(self.config.actor_rollout_ref.rollout.get("seed", 0)),
                        global_step=int(self.global_steps),
                        batch_size=len(test_batch),
                        rollout_n=val_n,
                        mode=str(trajectory_seed_mode),
                    )
                validation_seeds = (
                    [int(record["trajectory_seed"]) for record in validation_seed_records]
                    if validation_seed_records is not None else [0] * len(test_batch)
                )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            if not self.config.recurrent.enable:
                input_ids = test_batch.batch['input_ids']
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)
            else:
                ###### ADD
                sample_inputs.extend(test_batch.non_tensor_batch['sample_uuid'])

            ###### ADD
            if self.config.recurrent.enable:
                batch_keys_to_pop, non_tensor_batch_keys_to_pop = self.val_dataset.get_bactch_keys()
            else:
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_inputs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            if validation_seed_records is not None or strict_eval_identity:
                test_gen_batch.meta_info["trajectory_base_seeds"] = validation_seeds
            if strict_eval_identity:
                test_gen_batch.meta_info["strict_eval_identity"] = True
                test_gen_batch.meta_info["stable_eval_identity"] = identity_batch
                if not self.config.recurrent.enable:
                    test_gen_batch.non_tensor_batch["request_seeds"] = np.asarray(
                        validation_seeds, dtype=np.uint64
                    )
                    for key, values in identity_batch.items():
                        test_gen_batch.non_tensor_batch[key] = np.asarray(values, dtype=object)

            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')
            ######
            # ADD: no need to padding. indexing is needed.
            ######
            if not self.config.recurrent.enable:
                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                if not self.async_rollout_mode:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    # TODO: Maybe no more padding needed?
                    self.async_rollout_manager.wake_up()
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                    self.async_rollout_manager.sleep()
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            else:
                from recurrent.utils import final_batch
                output_gen_batch, final_mask, sample_index = self.generation_manager.run_llm_loop(test_gen_batch, {})
                if strict_eval_identity:
                    _append_stable_eval_turn_ledger(turn_ledger_path, output_gen_batch)
                test_output_gen_batch = final_batch(output_gen_batch, final_mask, sample_index)

            if strict_eval_identity:
                from recurrent.research.stable_eval_identity import (
                    detach_audit_meta_for_metrics,
                    detach_identity_columns_for_metrics,
                )

                terminal_response_token_sha256 = [
                    _response_token_sha256(response)
                    for response in test_output_gen_batch.batch["responses"]
                ]
                identity_batch = detach_identity_columns_for_metrics(
                    test_output_gen_batch.non_tensor_batch,
                    test_output_gen_batch.batch,
                )
                identity_batch["terminal_response_token_sha256"] = (
                    terminal_response_token_sha256
                )
                detach_audit_meta_for_metrics(
                    test_batch.meta_info, test_output_gen_batch.meta_info
                )

                # Identity evidence is deliberately kept outside the reward
                # and metric DataProto.  This makes the instrumentation
                # observational: reward code receives the same fields it did
                # before stable identity was enabled.
                test_batch.non_tensor_batch.pop("source_order_index", None)
                if "source_order_index" in test_batch.batch:
                    test_batch.batch.pop("source_order_index")

            if identity_batch is not None:
                for key, values in identity_batch.items():
                    sample_identities[key].extend(values)

            print('validation generation end')
            
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        if strict_eval_identity:
            from recurrent.research.stable_eval_identity import (
                identity_columns_to_rows,
                validate_attempt_identity_rows,
            )

            validate_attempt_identity_rows(
                identity_columns_to_rows(sample_identities),
                examples=int(eval_identity_config.examples),
                replicas=int(eval_identity_config.replicas),
            )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                identities=sample_identities or None,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

        if self.config.recurrent.enable:
            if not self.async_rollout_mode:
                from recurrent.generation_manager import LLMGenerationManager
                self.generation_manager = LLMGenerationManager(
                    tokenizer=self.tokenizer,
                    actor_rollout_wg=self.actor_rollout_wg,
                    config=self.recurrent_config,
                    agent_cls=self.recurrent_register.agent_cls,
                )
            else:
                from recurrent.async_generation_manager import AsyncLLMGenerationManager
                self.generation_manager = AsyncLLMGenerationManager(
                    tokenizer=self.tokenizer,
                    async_server=self.async_rollout_manager,
                    config=self.recurrent_config,
                    rollout_config=self.config.actor_rollout_ref.rollout,
                    agent_cls=self.recurrent_register.agent_cls,
                )

    def _save_checkpoint(self):
        eval_identity_config = self.config.trainer.get("eval_identity", None)
        if eval_identity_config is not None and bool(eval_identity_config.get("enabled", False)):
            raise RuntimeError(
                "strict stable evaluation is validation-only; checkpoint creation is forbidden"
            )
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _audit_gate_a_weight_sync(self, *, global_step: int, actor_version: int, sync_kind: str) -> None:
        from recurrent.research.gate_a_execution import append_gate_a_record, gate_a_enabled

        if not gate_a_enabled():
            return
        acknowledgements = self.actor_rollout_wg.audit_actor_vllm_weight_sync(
            global_step, actor_version, sync_kind
        )
        expected_ranks = list(range(self.actor_rollout_wg.world_size))
        actual_ranks = sorted(int(ack["vllm_worker_rank"]) for ack in acknowledgements)
        actor_master_digests = {
            ack["actor_master_sampled_tensor_digest"] for ack in acknowledgements
        }
        actor_digests = {
            ack["actor_rollout_sampled_tensor_digest"] for ack in acknowledgements
        }
        vllm_digests = {ack["vllm_sampled_tensor_digest"] for ack in acknowledgements}
        if actual_ranks != expected_ranks:
            raise RuntimeError(
                f"Gate A vLLM acknowledgement ranks mismatch: expected={expected_ranks}, actual={actual_ranks}"
            )
        if len(actor_master_digests) != 1:
            raise RuntimeError(
                "Gate A actor master sampled-tensor digests differ across ranks: "
                f"actor_master={sorted(actor_master_digests)}"
            )
        if len(actor_digests) != 1 or len(vllm_digests) != 1 or actor_digests != vllm_digests:
            raise RuntimeError(
                "Gate A effective actor-rollout/vLLM sampled-tensor digests diverged: "
                f"actor_rollout={sorted(actor_digests)}, vllm={sorted(vllm_digests)}"
            )
        append_gate_a_record(
            "weight_sync_summary",
            global_step=int(global_step),
            actor_version=int(actor_version),
            sync_kind=str(sync_kind),
            worker_ranks=actual_ranks,
            sampled_tensor_digest=next(iter(actor_digests)),
            actor_master_sampled_tensor_digest=next(iter(actor_master_digests)),
        )
        self._gate_a_synced_actor_version = int(actor_version)
        self._gate_a_synced_actor_digest = next(iter(actor_digests))

    def _coral_e1_rebase_behavior(self, value: DataProto) -> DataProto:
        """Recompute PPO behavior/reference logits at the fixed proposal weights."""
        result = value.clone()
        result.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        padded, pad_size = pad_dataproto_to_divisor(
            result, self.actor_rollout_wg.world_size,
        )
        old = self.actor_rollout_wg.compute_log_prob(padded)
        old = unpad_dataproto(old, pad_size)
        result.batch["old_log_probs"] = old.batch["old_log_probs"]
        if self.use_reference_policy:
            ref = self.ref_policy_wg.compute_ref_log_prob(padded)
            ref = unpad_dataproto(ref, pad_size)
            result.batch["ref_log_prob"] = ref.batch["ref_log_prob"]
        return result

    def _coral_e1_regenerate(self, gen_batch: DataProto, original_batch: DataProto,
                             timing_raw: dict) -> DataProto:
        """Regenerate a full trajectory only to materialize proposal-memory prompts."""
        self._audit_coral_e1_rollout_start("refreshed_memory_materialization")
        refreshed, final_mask, sample_index = self.generation_manager.run_llm_loop(
            gen_batch.clone(), timing_raw,
        )
        if int(final_mask.sum().item()) != len(original_batch):
            raise RuntimeError("CORAL_E1_NO_GO: refreshed final/root cardinality drift")
        source_rows = sample_index.detach().cpu().numpy()
        source_uids = np.asarray(original_batch.non_tensor_batch["uid"], dtype=object)
        refreshed.non_tensor_batch["uid"] = source_uids[source_rows]
        refreshed.batch["sample_index"] = sample_index.to(refreshed.batch["responses"].device)
        refreshed.batch["final_mask"] = final_mask
        if "rollout_trajectory_seed" not in original_batch.non_tensor_batch:
            raise RuntimeError("CORAL_E1_NO_GO: common trajectory seeds missing")
        source_seeds = np.asarray(
            original_batch.non_tensor_batch["rollout_trajectory_seed"], dtype=np.uint64,
        )
        refreshed.non_tensor_batch["trajectory_seed"] = source_seeds[source_rows]
        refreshed.non_tensor_batch["trajectory_id"] = np.asarray([
            f"{source_uids[row]}:{int(source_seeds[row])}" for row in source_rows
        ], dtype=object)
        if "index" in original_batch.batch:
            source_dataset_indices = original_batch.batch["index"].detach().cpu().numpy()
        elif "index" in original_batch.non_tensor_batch:
            source_dataset_indices = np.asarray(original_batch.non_tensor_batch["index"])
        else:
            raise RuntimeError("CORAL_E1_NO_GO: dataset root identity missing")
        refreshed.non_tensor_batch["dataset_index"] = source_dataset_indices[source_rows]
        refreshed.batch["response_mask"] = compute_response_mask(refreshed)
        return refreshed

    def _audit_coral_e1_rollout_start(self, branch: str) -> None:
        """Bind every extra E1 rollout to the just-synchronized proposal actor."""
        from recurrent.research.gate_a_execution import append_gate_a_record, gate_a_enabled
        if not gate_a_enabled():
            raise RuntimeError("CORAL_E1_NO_GO: Gate A audit must own diagnostic rollout")
        if getattr(self, "_gate_a_synced_actor_version", None) != int(self.global_steps) \
                or not getattr(self, "_gate_a_synced_actor_digest", None):
            raise RuntimeError("CORAL_E1_NO_GO: diagnostic rollout is not proposal-bound")
        append_gate_a_record(
            "coral_e1_rollout_start",
            global_step=int(self.global_steps),
            actor_version=int(self._gate_a_synced_actor_version),
            sampled_tensor_digest=str(self._gate_a_synced_actor_digest),
            branch=str(branch),
        )

    def _coral_e1_resample_terminal(self, materialized: DataProto,
                                    original_batch: DataProto,
                                    branch: str) -> DataProto:
        """Sample terminal answers at θ_W from one branch's materialized prompt.

        Both cached-old-memory and regenerated-memory branches call this method
        after the same proposal sync.  Source-policy terminal actions, rewards,
        and advantages are discarded.
        """
        final_rows = torch.nonzero(
            materialized.batch["final_mask"], as_tuple=False,
        ).flatten()
        if len(final_rows) != len(original_batch):
            raise RuntimeError("CORAL_E1_NO_GO: terminal prompt/root cardinality")
        final_source_rows = materialized.batch["sample_index"][final_rows].long()
        if sorted(final_source_rows.detach().cpu().tolist()) != list(range(len(original_batch))):
            raise RuntimeError("CORAL_E1_NO_GO: terminal prompt source permutation")
        selected = materialized[final_rows]
        prompts = selected.batch["prompts"]
        prompt_length = prompts.size(-1)
        prompt_attention = selected.batch["attention_mask"][:, :prompt_length]
        prompt_positions = selected.batch["position_ids"][..., :prompt_length]
        trajectory_seeds = np.asarray(
            original_batch.non_tensor_batch["rollout_trajectory_seed"], dtype=np.uint64,
        )[final_source_rows.detach().cpu().numpy()]
        from recurrent.research.trajectory_seeding import (
            derive_coral_terminal_contrast_seeds,
        )
        terminal_request_seeds = derive_coral_terminal_contrast_seeds(
            trajectory_seeds.tolist()
        )
        self._audit_coral_e1_rollout_start(f"{branch}_terminal_answer")
        terminal = self.generation_manager.generate_with_graceful_padding(
            prompts, prompt_attention, prompt_positions,
            {
                "do_sample": True,
                "validate": False,
                "pad_to": int(selected.batch["responses"].size(-1)),
                "generation_kwargs": {
                    "max_tokens": int(selected.batch["responses"].size(-1)), "n": 1,
                },
                "request_seeds": terminal_request_seeds,
            },
        )
        terminal.batch["response_mask"] = compute_response_mask(terminal)
        terminal.batch["final_mask"] = torch.ones(len(terminal), dtype=torch.bool)
        terminal.batch["sample_index"] = final_source_rows.to(
            terminal.batch["responses"].device
        )
        source_indices = final_source_rows.detach().cpu().numpy()
        source_uids = np.asarray(original_batch.non_tensor_batch["uid"], dtype=object)
        if "index" in original_batch.batch:
            dataset_indices = original_batch.batch["index"].detach().cpu().numpy()
        else:
            dataset_indices = np.asarray(original_batch.non_tensor_batch["index"])
        terminal.non_tensor_batch = {
            "uid": source_uids[source_indices],
            "trajectory_seed": trajectory_seeds,
            "trajectory_id": np.asarray([
                f"{source_uids[index]}:{int(trajectory_seeds[row])}"
                for row, index in enumerate(source_indices)
            ], dtype=object),
            "dataset_index": dataset_indices[source_indices],
            "request_seed": np.asarray(terminal_request_seeds, dtype=np.uint64),
        }

        nonfinal_rows = torch.nonzero(
            ~materialized.batch["final_mask"], as_tuple=False,
        ).flatten()
        nonfinal_source = materialized[nonfinal_rows]
        batch_keys = [
            "prompts", "responses", "input_ids", "attention_mask", "position_ids",
            "response_mask", "final_mask", "sample_index",
        ]
        non_tensor_keys = [
            "uid", "trajectory_seed", "trajectory_id", "dataset_index", "request_seed",
        ]
        missing_identity = [
            key for key in non_tensor_keys
            if key not in materialized.non_tensor_batch
        ]
        if missing_identity:
            raise RuntimeError(
                "CORAL_E1_NO_GO: materialized branch is missing row identity "
                f"{missing_identity}"
            )
        nonfinal_source = nonfinal_source.select(
            batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys,
        )
        terminal = terminal.select(
            batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys,
        )
        combined = DataProto.concat([nonfinal_source, terminal])
        final_mask = combined.batch["final_mask"]
        sample_index = combined.batch["sample_index"]
        from recurrent.utils import final_batch
        reward_batch = final_batch(combined, final_mask, sample_index).union(
            original_batch.clone()
        )
        reward_tensor, _ = compute_reward(reward_batch, self.reward_fn)
        advantage_scalar = compute_1D_grpo_advantage(
            token_level_rewards=reward_tensor,
            index=reward_batch.non_tensor_batch["uid"],
            use_adv=self.config.algorithm.grpo_use_adv,
        )[sample_index]
        response_length = combined.batch["responses"].size(-1)
        advantages = advantage_scalar.unsqueeze(-1).tile([1, response_length]) \
            * combined.batch["response_mask"]
        combined.batch["advantages"] = advantages
        combined.batch["returns"] = advantages
        combined.batch["token_level_scores"] = reward_tensor[sample_index]
        combined.batch["token_level_rewards"] = reward_tensor[sample_index]
        combined.meta_info["coral_e1_branch"] = branch
        combined.meta_info["coral_e1_terminal_request_seeds"] = terminal_request_seeds
        return self._coral_e1_rebase_behavior(combined)

    def _coral_e1_measure_root(self, value: DataProto, root_id: str,
                               phase: str) -> dict:
        """Run one root-cluster actual-loss backward and verify rank agreement."""
        uids = np.asarray(value.non_tensor_batch["uid"], dtype=object)
        rows = np.flatnonzero(uids == root_id)
        if not len(rows):
            raise RuntimeError("CORAL_E1_NO_GO: root missing from materialized batch")
        root = value[rows]
        root, pad_size = pad_dataproto_to_divisor(
            root, self.actor_rollout_wg.world_size,
        )
        valid_rows = torch.ones(len(root), dtype=torch.bool)
        if pad_size:
            valid_rows[-pad_size:] = False
            root.batch["response_mask"][~valid_rows] = 0
            root.batch["final_mask"][~valid_rows] = False
        from recurrent.research.coral import role_covered_order
        root.reorder(role_covered_order(
            root.batch["final_mask"], self.actor_rollout_wg.world_size, valid_rows,
        ))
        root.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        root.meta_info["global_token_num"] = torch.sum(
            root.batch["attention_mask"], dim=-1,
        ).tolist()
        output = self.actor_rollout_wg.measure_coral_role_gradient(root, phase)
        sketches = output.batch["gradient_sketch"].detach().cpu().double()
        norms = output.batch["gradient_squared_norm"].detach().cpu().double().sqrt()
        if len(sketches) != self.actor_rollout_wg.world_size \
                or not torch.allclose(sketches, sketches[0].expand_as(sketches), rtol=0, atol=0) \
                or not torch.allclose(norms, norms[0].expand_as(norms), rtol=0, atol=0):
            raise RuntimeError("CORAL_E1_NO_GO: distributed gradient receipt disagreement")
        vector = sketches[0].tolist()
        from recurrent.research.cosi import canonical_sha256
        return {
            "sketch": vector,
            "sketch_sha256": canonical_sha256(vector),
            "gradient_norm": math.sqrt(math.fsum(item * item for item in vector)),
            "full_gradient_norm": float(norms[0].item()),
            "actual_loss": float(output.batch["actual_loss"][0].item()),
            "active_tokens": int(output.batch["active_tokens"][0].item()),
            "full_tokens": int(output.batch["full_tokens"][0].item()),
        }

    def _coral_e1_write_proposal(self, cached: DataProto, refreshed: DataProto,
                                 capture_dir: str, source_weight_digest: str) -> None:
        """Write an exclusive, trainer-produced cached/refreshed E1 proposal."""
        from recurrent.research.coral_e1 import SKETCH_BASIS_SHA256
        from recurrent.research.cosi import canonical_sha256

        if cached.meta_info.get("coral_e1_branch") != "cached_old_memory" \
                or refreshed.meta_info.get("coral_e1_branch") != "refreshed_memory":
            raise RuntimeError("CORAL_E1_NO_GO: both terminal branches must be proposal-resampled")
        cached_uids = sorted(set(map(str, cached.non_tensor_batch["uid"])))
        refreshed_uids = sorted(set(map(str, refreshed.non_tensor_batch["uid"])))
        if cached_uids != refreshed_uids or len(cached_uids) != 4:
            raise RuntimeError("CORAL_E1_NO_GO: exact b4 outcome-blind root inventory")

        def cosine(left, right):
            dot = math.fsum(a * b for a, b in zip(left, right))
            nl = math.sqrt(math.fsum(a * a for a in left))
            nr = math.sqrt(math.fsum(a * a for a in right))
            return dot / max(nl * nr, 1e-30)

        records = []
        for root_id in cached_uids:
            cached_answer = self._coral_e1_measure_root(
                cached, root_id, "terminal_answer",
            )
            cached_duplicate = self._coral_e1_measure_root(
                cached, root_id, "terminal_answer",
            )
            refreshed_answer = self._coral_e1_measure_root(
                refreshed, root_id, "terminal_answer",
            )
            cached_writer = self._coral_e1_measure_root(
                cached, root_id, "memory_writer",
            )
            response = [b - a for a, b in zip(
                cached_answer["sketch"], refreshed_answer["sketch"],
            )]
            duplicate = [b - a for a, b in zip(
                cached_answer["sketch"], cached_duplicate["sketch"],
            )]
            response_norm = math.sqrt(math.fsum(x * x for x in response))
            duplicate_norm = math.sqrt(math.fsum(x * x for x in duplicate))
            denominator = max(
                cached_answer["gradient_norm"] + refreshed_answer["gradient_norm"],
                1e-12,
            )
            cached_rows = np.flatnonzero(
                np.asarray(cached.non_tensor_batch["uid"], dtype=object) == root_id
            )
            refreshed_rows = np.flatnonzero(
                np.asarray(refreshed.non_tensor_batch["uid"], dtype=object) == root_id
            )

            def tensor_hash(value, rows, key, final):
                selected = value[rows]
                mask = selected.batch["final_mask"] == final
                tensor = selected.batch[key][mask].detach().cpu().tolist()
                return canonical_sha256(tensor), tensor

            cached_memory_hash, _ = tensor_hash(
                cached, cached_rows, "responses", False,
            )
            refreshed_memory_hash, _ = tensor_hash(
                refreshed, refreshed_rows, "responses", False,
            )
            cached_prompt_hash, _ = tensor_hash(
                cached, cached_rows, "input_ids", True,
            )
            refreshed_prompt_hash, _ = tensor_hash(
                refreshed, refreshed_rows, "input_ids", True,
            )
            cached_answer_hash, _ = tensor_hash(
                cached, cached_rows, "responses", True,
            )
            refreshed_answer_hash, _ = tensor_hash(
                refreshed, refreshed_rows, "responses", True,
            )
            cached_root_value = cached[cached_rows]
            refreshed_root_value = refreshed[refreshed_rows]
            cached_final_mask = cached_root_value.batch["final_mask"]
            refreshed_final_mask = refreshed_root_value.batch["final_mask"]
            cached_reward_hash = canonical_sha256(
                cached_root_value.batch["token_level_scores"][cached_final_mask]
                .detach().cpu().tolist()
            )
            refreshed_reward_hash = canonical_sha256(
                refreshed_root_value.batch["token_level_scores"][refreshed_final_mask]
                .detach().cpu().tolist()
            )
            cached_advantage_hash = canonical_sha256(
                cached_root_value.batch["advantages"][cached_final_mask]
                .detach().cpu().tolist()
            )
            refreshed_advantage_hash = canonical_sha256(
                refreshed_root_value.batch["advantages"][refreshed_final_mask]
                .detach().cpu().tolist()
            )
            cached_seed_values = sorted(set(map(
                int, np.asarray(cached.non_tensor_batch["trajectory_seed"])[cached_rows]
            )))
            refreshed_seed_values = sorted(set(map(
                int, np.asarray(refreshed.non_tensor_batch["trajectory_seed"])[refreshed_rows]
            )))
            if cached_seed_values != refreshed_seed_values or len(cached_seed_values) != 2:
                raise RuntimeError("CORAL_E1_NO_GO: common future seed coupling drift")
            cached_terminal_seeds = sorted(set(map(
                int,
                np.asarray(cached_root_value.non_tensor_batch["request_seed"])[
                    cached_root_value.batch["final_mask"].detach().cpu().numpy()
                ],
            )))
            refreshed_terminal_seeds = sorted(set(map(
                int,
                np.asarray(refreshed_root_value.non_tensor_batch["request_seed"])[
                    refreshed_root_value.batch["final_mask"].detach().cpu().numpy()
                ],
            )))
            if cached_terminal_seeds != refreshed_terminal_seeds \
                    or len(cached_terminal_seeds) != 2:
                raise RuntimeError("CORAL_E1_NO_GO: terminal request seed coupling drift")
            cached_dataset_indices = sorted(set(map(
                int, np.asarray(cached.non_tensor_batch["dataset_index"])[cached_rows]
            )))
            refreshed_dataset_indices = sorted(set(map(
                int, np.asarray(refreshed.non_tensor_batch["dataset_index"])[refreshed_rows]
            )))
            if cached_dataset_indices != refreshed_dataset_indices \
                    or len(cached_dataset_indices) != 1:
                raise RuntimeError("CORAL_E1_NO_GO: dataset root coupling drift")
            cached_memory_tokens = int(
                cached[cached_rows].batch["response_mask"]
                [~cached[cached_rows].batch["final_mask"]].sum().item()
            )
            refreshed_memory_tokens = int(
                refreshed[refreshed_rows].batch["response_mask"]
                [~refreshed[refreshed_rows].batch["final_mask"]].sum().item()
            )
            records.append({
                "root_id": root_id,
                "dataset_index": cached_dataset_indices[0],
                "writer_replicas": 2,
                "common_trajectory_seeds": cached_seed_values,
                "common_terminal_request_seeds": cached_terminal_seeds,
                "cached_memory_token_ids_sha256": cached_memory_hash,
                "refreshed_memory_token_ids_sha256": refreshed_memory_hash,
                "cached_prompt_token_ids_sha256": cached_prompt_hash,
                "refreshed_prompt_token_ids_sha256": refreshed_prompt_hash,
                "cached_terminal_answer_token_ids_sha256": cached_answer_hash,
                "refreshed_terminal_answer_token_ids_sha256": refreshed_answer_hash,
                "cached_reward_sha256": cached_reward_hash,
                "refreshed_reward_sha256": refreshed_reward_hash,
                "cached_advantage_sha256": cached_advantage_hash,
                "refreshed_advantage_sha256": refreshed_advantage_hash,
                "terminal_action_policy": "both_branches_freshly_sampled_at_fixed_proposal_weights",
                "cached_memory_token_count": cached_memory_tokens,
                "refreshed_memory_token_count": refreshed_memory_tokens,
                "cached_gradient_sha256": cached_answer["sketch_sha256"],
                "refreshed_gradient_sha256": refreshed_answer["sketch_sha256"],
                "cached_gradient_norm": cached_answer["gradient_norm"],
                "refreshed_gradient_norm": refreshed_answer["gradient_norm"],
                "symmetric_relative_response": 2 * response_norm / denominator,
                "duplicate_control_response_norm": duplicate_norm,
                "same_batch_writer_answer_cosine": cosine(
                    cached_writer["sketch"], cached_answer["sketch"],
                ),
                "tensor_source": "actual_terminal_answer_loss_backward",
            })
        proposal = {
            "schema": "memagent.coral.e1-proposal.v3",
            "producer": "ray_ppo_trainer_actual_loss_backward",
            "git_commit": str(os.environ.get("GATE_A_GIT_COMMIT", "")),
            "global_step": int(self.global_steps),
            "source_weight_sample_digest": str(source_weight_digest),
            "proposal_weight_sample_digest": str(self._gate_a_synced_actor_digest),
            "gradient_sketch_basis_sha256": SKETCH_BASIS_SHA256,
            "root_inventory_sha256": canonical_sha256(cached_uids),
            "records": records,
        }
        proposal["proposal_sha256"] = canonical_sha256(proposal)
        destination = os.path.join(
            capture_dir, f"proposal_step_{int(self.global_steps):02d}.json",
        )
        os.makedirs(capture_dir, exist_ok=True)
        with open(destination, "x", encoding="utf-8") as stream:
            stream.write(json.dumps(proposal, indent=2, sort_keys=True) + "\n")

    def _stable_eval_weight_snapshot(self, *, sync_kind: str) -> dict:
        """Synchronize and attest actor/vLLM weights without writing Gate A evidence."""
        from recurrent.research.gate_a_execution import gate_a_enabled

        if gate_a_enabled():
            raise RuntimeError(
                "stable evaluation refuses to run while the Gate A evidence writer is enabled"
            )
        acknowledgements = self.actor_rollout_wg.audit_actor_vllm_weight_sync(
            int(self.global_steps), int(self.global_steps), str(sync_kind)
        )
        expected_ranks = list(range(self.actor_rollout_wg.world_size))
        actual_ranks = sorted(int(ack["vllm_worker_rank"]) for ack in acknowledgements)
        if actual_ranks != expected_ranks:
            raise RuntimeError(
                "stable evaluation vLLM acknowledgement ranks mismatch: "
                f"expected={expected_ranks}, actual={actual_ranks}"
            )

        def require_sha256(ack: dict, field: str) -> str:
            value = ack.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError(
                    f"stable evaluation worker acknowledgement has invalid {field}: {value!r}"
                )
            try:
                int(value, 16)
            except ValueError as error:
                raise RuntimeError(
                    f"stable evaluation worker acknowledgement has invalid {field}: {value!r}"
                ) from error
            return value

        master_digests = {
            require_sha256(ack, "actor_master_sampled_tensor_digest")
            for ack in acknowledgements
        }
        actor_digests = {
            require_sha256(ack, "actor_rollout_sampled_tensor_digest")
            for ack in acknowledgements
        }
        vllm_digests = {
            require_sha256(ack, "vllm_sampled_tensor_digest")
            for ack in acknowledgements
        }
        vllm_pre_sync_digests = [
            require_sha256(ack, "vllm_pre_sync_sampled_tensor_digest")
            for ack in acknowledgements
        ]
        if len(master_digests) != 1:
            raise RuntimeError(
                f"stable evaluation actor master digests differ across ranks: {sorted(master_digests)}"
            )
        if len(actor_digests) != 1 or actor_digests != vllm_digests:
            raise RuntimeError(
                "stable evaluation actor/vLLM sampled digests diverged: "
                f"actor={sorted(actor_digests)}, vllm={sorted(vllm_digests)}"
            )
        # The first snapshot wakes dummy-loaded vLLM workers, whose pre-sync
        # values are intentionally outside the experiment contract.  The final
        # snapshot, however, must observe the same pre-sync value on every
        # worker so validation-time drift cannot be hidden by this sync call.
        pre_sync_digest = None
        if sync_kind == "stable_eval_after" and len(set(vllm_pre_sync_digests)) != 1:
            raise RuntimeError(
                "stable evaluation pre-sync vLLM digests differ across workers: "
                f"{sorted(set(vllm_pre_sync_digests))}"
            )
        if sync_kind == "stable_eval_after":
            pre_sync_digest = vllm_pre_sync_digests[0]
        if any(
            int(ack["loaded_parameter_count"]) != int(ack["model_parameter_count"])
            for ack in acknowledgements
        ):
            raise RuntimeError("stable evaluation vLLM sync did not cover every model parameter")

        invariant_ack_fields = (
            "optimizer_step_min",
            "optimizer_step_max",
            "optimizer_state_entry_count",
            "optimizer_step_entry_count",
            "optimizer_step_histogram",
            "lr_scheduler_last_epoch",
            "weight_transfer_format",
            "loaded_parameter_count",
            "model_parameter_count",
            "loaded_parameter_names_sha256",
            "model_parameter_names_sha256",
            "audited_loaded_parameters",
            "sampled_parameter_dtypes",
        )
        return {
            "sync_kind": str(sync_kind),
            "worker_ranks": actual_ranks,
            "actor_master_sampled_tensor_digest": next(iter(master_digests)),
            "actor_rollout_sampled_tensor_digest": next(iter(actor_digests)),
            "vllm_sampled_tensor_digest": next(iter(vllm_digests)),
            "vllm_pre_sync_sampled_tensor_digest": pre_sync_digest,
            "worker_evidence": [
                {field: ack.get(field) for field in invariant_ack_fields}
                for ack in sorted(acknowledgements, key=lambda item: int(item["vllm_worker_rank"]))
            ],
        }

    def _audit_gate_a_rollout_start(self, *, global_step: int) -> None:
        from recurrent.research.gate_a_execution import append_gate_a_record, gate_a_enabled

        if not gate_a_enabled():
            return
        expected_version = int(global_step) - 1
        actual_version = getattr(self, "_gate_a_synced_actor_version", None)
        digest = getattr(self, "_gate_a_synced_actor_digest", None)
        if actual_version != expected_version or not digest:
            raise RuntimeError(
                "Gate A rollout is not bound to the immediately preceding actor/vLLM sync: "
                f"global_step={global_step}, expected_actor_version={expected_version}, "
                f"synced_actor_version={actual_version}, digest={digest}"
            )
        append_gate_a_record(
            "rollout_start",
            global_step=int(global_step),
            actor_version=actual_version,
            sampled_tensor_digest=digest,
        )

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "actor_only_eval":
            eval_identity_config = self.config.trainer.get("eval_identity", None)
            if not (
                eval_identity_config is not None
                and bool(eval_identity_config.get("enabled", False))
                and str(eval_identity_config.get("weight_source", ""))
                == "actor_checkpoint"
            ):
                raise ValueError(
                    "trainer.resume_mode=actor_only_eval is reserved for strict "
                    "validation-only actor-checkpoint evaluation"
                )
            global_step_folder = str(self.config.trainer.resume_from_path)
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
            global_step_folder = os.path.realpath(global_step_folder)
            expected_global_step = int(eval_identity_config.expected_global_step)
            expected_basename = f"global_step_{expected_global_step}"
            if os.path.basename(global_step_folder) != expected_basename:
                raise ValueError(
                    "actor-only checkpoint must have the exact frozen global-step basename: "
                    f"actual={os.path.basename(global_step_folder)}, expected={expected_basename}"
                )
            parsed_global_step = int(global_step_folder.rsplit("global_step_", 1)[-1])
            if parsed_global_step != expected_global_step:
                raise ValueError(
                    "actor-only checkpoint global step differs from the frozen evaluation "
                    f"contract: parsed={parsed_global_step}, expected={expected_global_step}"
                )
            actor_path = os.path.join(global_step_folder, "actor")
            if not os.path.isdir(actor_path):
                raise FileNotFoundError(
                    f"actor-only evaluation checkpoint is missing actor directory: {actor_path}"
                )
            self.global_steps = expected_global_step
            acknowledgements = self.actor_rollout_wg.load_model_checkpoint_only(
                actor_path,
                del_local_after_load=False,
            )
            from recurrent.research.stable_eval_identity import (
                load_resolved_manifest,
                validate_actor_only_checkpoint_acknowledgements,
            )

            resolved_eval_manifest = load_resolved_manifest(
                str(eval_identity_config.resolved_manifest_path),
                expected_hash=str(eval_identity_config.expected_manifest_hash),
            )
            interface_id = str(eval_identity_config.interface_id)
            frozen_artifact = (
                resolved_eval_manifest.get("execution_binding", {})
                .get("model_artifacts", {})
                .get(interface_id)
            )
            frozen_shards = (
                frozen_artifact.get("actor_model_shards")
                if isinstance(frozen_artifact, dict)
                else None
            )
            if not isinstance(frozen_shards, list):
                raise RuntimeError(
                    "actor-only checkpoint evaluation lacks frozen actor shard inventory "
                    f"for interface {interface_id}"
                )
            self._stable_eval_actor_checkpoint_load_acks = (
                validate_actor_only_checkpoint_acknowledgements(
                    acknowledgements,
                    frozen_shards,
                    global_step_folder=global_step_folder,
                    world_size=self.actor_rollout_wg.world_size,
                )
            )
            return self.global_steps
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        actor_load_acks = self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            from recurrent.research.gate_a_execution import append_gate_a_record, gate_a_enabled, sha256_file

            if gate_a_enabled():
                append_gate_a_record(
                    "resume_load",
                    global_step=self.global_steps,
                    resume_source=os.path.realpath(global_step_folder),
                    actor_model_optimizer_extra_loaded=True,
                    actor_load_worker_acks=actor_load_acks,
                    data_loaded=True,
                    data_sha256=sha256_file(dataloader_local_path),
                )
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        coral_config = self.config.algorithm.get("coral", None)
        coral_enabled = bool(coral_config is not None and coral_config.get("enabled", False))
        coral_e1_capture_dir = os.getenv("CORAL_E1_CAPTURE_DIR")
        if coral_e1_capture_dir is not None and (
            not os.path.isabs(coral_e1_capture_dir) or not coral_enabled
        ):
            raise ValueError(
                "CORAL_E1_NO_GO: capture directory must be absolute and CORAL active"
            )
        if coral_enabled:
            from recurrent.research.coral import validate_config
            from recurrent.research.gate_a_execution import gate_a_enabled
            validate_config(coral_config)
            if not self.config.recurrent.enable \
                    or self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
                raise ValueError("CORAL_NO_GO: CORAL requires recurrent GRPO")
            if int(self.config.data.train_batch_size) != int(
                self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ) or str(self.config.actor_rollout_ref.actor.loss_agg_mode) != "token-mean" \
                    or str(self.config.actor_rollout_ref.rollout.get(
                        "trajectory_seed_mode", ""
                    )) != "independent":
                raise ValueError(
                    "CORAL_NO_GO: frozen full-batch denominator and independent seeds required"
                )
            if not gate_a_enabled():
                raise ValueError("CORAL_NO_GO: audited actor/vLLM weight sync is required")
        self._audit_gate_a_weight_sync(
            global_step=self.global_steps,
            actor_version=self.global_steps,
            sync_kind="fresh_initial" if self.global_steps == 0 else "resume_loaded",
        )

        eval_identity_config = self.config.trainer.get("eval_identity", None)
        strict_eval_identity = bool(
            eval_identity_config is not None and eval_identity_config.get("enabled", False)
        )
        stable_eval_weight_before = None
        if strict_eval_identity:
            stable_eval_weight_before = self._stable_eval_weight_snapshot(
                sync_kind="stable_eval_before"
            )

        if self.config.trainer.get("val_only", False):
            if not self.config.trainer.get("val_before_train", False):
                raise ValueError("trainer.val_only requires trainer.val_before_train=true")
            if self.val_reward_fn is None:
                raise ValueError("trainer.val_only requires a validation reward function; refusing to enter training")

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                if strict_eval_identity:
                    if self._actor_update_calls != 0:
                        raise RuntimeError(
                            "stable evaluation identity canary reached actor update unexpectedly: "
                            f"calls={self._actor_update_calls}"
                        )
                    stable_eval_weight_after = self._stable_eval_weight_snapshot(
                        sync_kind="stable_eval_after"
                    )
                    for field in (
                        "actor_master_sampled_tensor_digest",
                        "actor_rollout_sampled_tensor_digest",
                        "vllm_sampled_tensor_digest",
                        "worker_ranks",
                        "worker_evidence",
                    ):
                        if stable_eval_weight_before[field] != stable_eval_weight_after[field]:
                            raise RuntimeError(
                                "stable evaluation mutated actor/optimizer/vLLM state: "
                                f"field={field}, before={stable_eval_weight_before[field]}, "
                                f"after={stable_eval_weight_after[field]}"
                            )
                    if stable_eval_weight_after[
                        "vllm_pre_sync_sampled_tensor_digest"
                    ] != stable_eval_weight_before["vllm_sampled_tensor_digest"]:
                        raise RuntimeError(
                            "stable evaluation vLLM weights drifted before the final read-only sync: "
                            f"before_post={stable_eval_weight_before['vllm_sampled_tensor_digest']}, "
                            f"after_pre={stable_eval_weight_after['vllm_pre_sync_sampled_tensor_digest']}"
                        )
                    summary_path = os.path.realpath(str(eval_identity_config.execution_summary_path))
                    summary_parent = os.path.dirname(summary_path)
                    if summary_parent:
                        os.makedirs(summary_parent, exist_ok=True)
                    summary = {
                        "record_type": "execution_summary",
                        "interface_id": str(eval_identity_config.interface_id),
                        "attempt_id": str(eval_identity_config.attempt_id),
                        "eval_manifest_hash": str(eval_identity_config.expected_manifest_hash),
                        "resolved_runtime_config_sha256": self._stable_eval_runtime_config_sha256,
                        "global_step": int(self.global_steps),
                        "actor_update_calls": int(self._actor_update_calls),
                        "optimizer_step_calls": 0,
                        "checkpoint_save_calls": 0,
                        "resume_mode": str(self.config.trainer.resume_mode),
                        "weight_source": str(
                            eval_identity_config.get("weight_source", "base_model")
                        ),
                        "checkpoint_load_mode": (
                            "actor_only"
                            if str(eval_identity_config.get("weight_source", "base_model"))
                            == "actor_checkpoint"
                            else "none"
                        ),
                        "checkpoint_source": (
                            os.path.realpath(str(self.config.trainer.resume_from_path))
                            if self.config.trainer.resume_from_path
                            else None
                        ),
                        "actor_checkpoint_load_acks": (
                            self._stable_eval_actor_checkpoint_load_acks
                            if self._stable_eval_actor_checkpoint_load_acks is not None
                            else []
                        ),
                        "validation_only": True,
                        "weight_snapshot_before": stable_eval_weight_before,
                        "weight_snapshot_after": stable_eval_weight_after,
                    }
                    with open(summary_path, "x", encoding="utf-8") as stream:
                        stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                ###### ADD
                if self.config.recurrent.enable:
                    batch_keys_to_pop, non_tensor_batch_keys_to_pop = self.train_dataset.get_bactch_keys()
                else:
                    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                is_last_step = self.global_steps >= self.total_training_steps

                ####################
                # original code here

                coral_e1_context = None
                with _timer('step', timing_raw):
                    with _timer('gen', timing_raw):
                        if not self.config.recurrent.enable:
                            # generate a batch
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                self.async_rollout_manager.wake_up()
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                self.async_rollout_manager.sleep()

                            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                                with _timer("gen_max", timing_raw):
                                    gen_baseline_batch = deepcopy(gen_batch)
                                    gen_baseline_batch.meta_info["do_sample"] = False
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                    batch.batch['reward_baselines'] = reward_baseline_tensor
                                    # del gen_baseline_batch, gen_baseline_output

                            batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                    dtype=object)
                            # repeat to align with repeated responses in rollout
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                        else:
                            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                                raise NotImplementedError("REMAX is not implemented for recurrent.")
                            from recurrent.research.trajectory_seeding import stable_training_group_id
                            if "index" in batch.batch:
                                pre_repeat_indices = batch.batch["index"].detach().cpu().tolist()
                            elif "index" in batch.non_tensor_batch:
                                pre_repeat_indices = np.asarray(batch.non_tensor_batch["index"]).tolist()
                            else:
                                raise RuntimeError("recurrent training requires deterministic dataset index identity")
                            batch.non_tensor_batch['uid'] = np.array([
                                stable_training_group_id(
                                    base_seed=int(self.config.actor_rollout_ref.rollout.get("seed", 0)),
                                    global_step=int(self.global_steps),
                                    dataset_index=int(dataset_index),
                                )
                                for dataset_index in pre_repeat_indices
                            ], dtype=object)
                            # Note that we repeat outside the loop, since the generated responses are not aligned and we cannot
                            # simply union them.
                            # Also, just as what happened in validate, we will always set n=1 in generation_kwargs.
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            trajectory_seed_mode = self.config.actor_rollout_ref.rollout.get("trajectory_seed_mode", None)
                            if trajectory_seed_mode not in (None, "", "legacy_shared"):
                                from recurrent.research.trajectory_seeding import build_trajectory_seed_records

                                trajectory_seed_records = build_trajectory_seed_records(
                                    base_seed=int(self.config.actor_rollout_ref.rollout.get("seed", 0)),
                                    global_step=int(self.global_steps),
                                    batch_size=len(gen_batch),
                                    rollout_n=int(self.config.actor_rollout_ref.rollout.n),
                                    mode=str(trajectory_seed_mode),
                                )
                                trajectory_base_seeds = [
                                    int(record["trajectory_seed"]) for record in trajectory_seed_records
                                ]
                                if "index" in batch.batch:
                                    dataset_indices = batch.batch["index"].detach().cpu().tolist()
                                elif "index" in batch.non_tensor_batch:
                                    dataset_indices = np.asarray(batch.non_tensor_batch["index"]).tolist()
                                else:
                                    raise RuntimeError(
                                        "Gate A requires the frozen dataset index on every trajectory"
                                    )
                                gen_batch.meta_info["trajectory_base_seeds"] = trajectory_base_seeds
                                batch.non_tensor_batch["rollout_trajectory_seed"] = np.asarray(
                                    trajectory_base_seeds, dtype=np.uint64
                                )
                                for record in trajectory_seed_records:
                                    record["record_type"] = "trajectory_seed"
                                    source_row = int(record["row"])
                                    record["uid"] = str(batch.non_tensor_batch["uid"][source_row])
                                    record["dataset_index"] = int(dataset_indices[source_row])
                                _append_rollout_seed_audit(
                                    str(self.config.trainer.default_local_dir), trajectory_seed_records
                                )
                            self._audit_gate_a_rollout_start(global_step=self.global_steps)
                            gen_batch_output, final_mask, sample_index = self.generation_manager.run_llm_loop(gen_batch, timing_raw)

                            assert final_mask.sum().item() == len(batch.batch), \
                                "The number of final responses should be equal to the number of prompts." \
                                f"{len(batch.non_tensor_batch['uid'])} != {len(batch.batch)}"
                            
                            # This is a simplified diagram to show how sample_index works.
                            # DataProto and 2D tensors represented as a list of samples.

                            # ex. batch = [s1, s2, s3, s4]
                            #     gen_batch = [s1_turn1, s2_turn1, s3_turn1, s4_turn1, s1_turn2, s3_turn2, s3_turn3, s1_final, s2_final, s3_final, s4_final]
                            #     final_mask = [      F,        F,        F,        F,        F,        F,        F,        T,        T,        T,        T]
                            #     sample_index = [    0,        1,        2,        3,        0,        2,        2,        0,        1,        2,        3]
                            
                            # then, batch[sample_index] will be
                            #                 [      s1,       s2,       s3,       s4,       s1,       s3,       s3,       s1,       s2,       s3,       s4]
                            # We map info from original_sample to gen_batch_output now, e.x. in reward computation

                            # Preserve stable trajectory identity on every recurrent turn.  DataProto indexing,
                            # padding and dispatch then keep these columns aligned with the tensors they describe.
                            source_rows = sample_index.detach().cpu().numpy()
                            source_uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
                            gen_batch_output.non_tensor_batch["uid"] = source_uids[source_rows]
                            gen_batch_output.batch["sample_index"] = sample_index.to(
                                gen_batch_output.batch["responses"].device
                            )
                            if "rollout_trajectory_seed" in batch.non_tensor_batch:
                                source_seeds = np.asarray(
                                    batch.non_tensor_batch["rollout_trajectory_seed"], dtype=np.uint64
                                )
                                gen_batch_output.non_tensor_batch["trajectory_seed"] = source_seeds[source_rows]
                                source_trajectory_ids = np.asarray(
                                    [f"{uid}:{int(seed)}" for uid, seed in zip(source_uids, source_seeds)],
                                    dtype=object,
                                )
                            else:
                                source_trajectory_ids = np.asarray(
                                    [f"{uid}:{row}" for row, uid in enumerate(source_uids)], dtype=object
                                )
                            gen_batch_output.non_tensor_batch["trajectory_id"] = source_trajectory_ids[
                                source_rows
                            ]
                            gen_batch_output.non_tensor_batch["dataset_index"] = np.asarray(
                                dataset_indices, dtype=np.int64
                            )[source_rows]
                            if "trajectory_turn" in gen_batch_output.batch:
                                request_seeds = np.asarray(
                                    gen_batch_output.non_tensor_batch["request_seed"], dtype=np.uint64
                                )
                                turns = gen_batch_output.batch["trajectory_turn"].detach().cpu().tolist()
                                final_flags = final_mask.detach().cpu().tolist()
                                turn_records = []
                                rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                                for output_row, source_row in enumerate(source_rows.tolist()):
                                    group, replica = divmod(int(source_row), rollout_n)
                                    turn_records.append({
                                        "record_type": "trajectory_turn_seed",
                                        "global_step": int(self.global_steps),
                                        "row": int(source_row),
                                        "sample_index": int(source_row),
                                        "group": group,
                                        "replica": replica,
                                        "turn": int(turns[output_row]),
                                        "uid": str(source_uids[source_row]),
                                        "trajectory_seed": int(source_seeds[source_row]),
                                        "dataset_index": int(dataset_indices[source_row]),
                                        "request_seed": int(request_seeds[output_row]),
                                        "is_final": bool(final_flags[output_row]),
                                        "mode": str(trajectory_seed_mode),
                                    })
                                _append_rollout_seed_audit(
                                    str(self.config.trainer.default_local_dir), turn_records
                                )

                            # Will be used in advantage computation.
                            gen_batch_output.batch['final_mask'] = final_mask
                            gen_batch_output.check_consistency()

                            # padding for forwarding
                            original_batch = batch # keep original info for reward computation
                            batch = gen_batch_output

                            workflow_metrics = gen_batch_output.meta_info.pop("metrics", {})
                            metrics.update(workflow_metrics)
                    ####################
                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    #######
                    # MODIFY: we do not want to break the order in batch since we have to maintain `final_mask` and `sample_index`
                    #######
                    if self.config.trainer.balance_batch and not self.config.recurrent.enable:
                        self._balance_batch(batch, metrics=metrics)
                    #######

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            raise NotImplementedError("RM is not implemented for recurrent.")
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if not self.config.recurrent.enable:
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        else:
                            from recurrent.utils import final_batch
                            ##### make sure that samples in indexed proto are in same order as original_batch
                            reward_batch = final_batch(batch, final_mask, sample_index).union(original_batch)
                            reward_tensor, reward_extra_infos_dict = compute_reward(reward_batch, self.reward_fn)
                            # pad for log_prob
                            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                            
                    if self.config.recurrent.enable and self.config.algorithm.get("filter_groups", None):  
                        # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        reward_batch.non_tensor_batch["seq_reward"] = reward_tensor.sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(reward_batch.non_tensor_batch["uid"], reward_batch.non_tensor_batch["seq_reward"]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        kept_traj_idxs = np.array(kept_traj_idxs, dtype=int)

                        batch = batch[kept_traj_idxs]
                        final_mask = final_mask[kept_traj_idxs]
                        sample_index = sample_index[kept_traj_idxs]
                        metrics["train/kept_prompts"] = len(kept_prompt_uids)
                        metrics["train/kept_samples"] = len(kept_traj_idxs)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)


                    with _timer('adv', timing_raw):
                        ###############
                        # ORIGINAL
                        ###############
                        if not self.config.recurrent.enable:
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            print(f"{list(reward_extra_infos_dict.keys())=}")
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.grpo_use_adv  # GRPO adv normalization factor
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )
                            #############
                        else:
                            batch = unpad_dataproto(batch, pad_size)
                        
                            if self.use_rm:
                                raise NotImplementedError("RM is not implemented for recurrent.")

                            # Hack compute_advantage function, because it will return a 2D tensor
                            # I'd rather have a 1D adv, manually indexing and then apply it to every non-mask token.
                            # use reward_batch: have to unmask and remask
                            # use batch: incorrect adv
                            if not self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                                raise NotImplementedError("Only GRPO is implemented for recurrent.")
                            advantage_scalar = compute_1D_grpo_advantage(token_level_rewards=reward_tensor, 
                                                                         index=reward_batch.non_tensor_batch['uid'],
                                                                         use_adv=self.config.algorithm.grpo_use_adv)
                            advantage_scalar = advantage_scalar[sample_index]

                            # apply adv to non-mask tokens
                            response_length = batch.batch['responses'].size(-1)
                            eos_mask = batch.batch['response_mask']
                            advantages = advantage_scalar.unsqueeze(-1).tile([1, response_length]) * eos_mask
                            batch.batch['advantages'] = advantages
                            batch.batch['returns'] = advantages                             
                            # turns of a sample will have the same final reward, now we mapping turns to samples
                            batch.batch['token_level_scores'] = reward_tensor[sample_index]

                            if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                                raise NotImplementedError("KL penalty is not implemented for recurrent.")
                            
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                            if coral_e1_capture_dir:
                                from recurrent.research.coral import phase_for_step
                                if phase_for_step(int(self.global_steps)) == "memory_writer":
                                    coral_e1_context = {
                                        "cached": batch.clone(),
                                        "gen_batch": gen_batch.clone(),
                                        "original_batch": original_batch.clone(),
                                        "source_weight_digest": str(
                                            self._gate_a_synced_actor_digest
                                        ),
                                    }


                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        wsz = self.actor_rollout_wg.world_size
                        if self.config.recurrent.enable:
                            from recurrent.research.actor_batch import DIAG_PREFIX, validate_active_actor_batch

                            active_batch_size = len(batch)
                            response_valid_tokens = int(batch.batch["response_mask"].sum().item())
                            validate_active_actor_batch(
                                active_batch_size=active_batch_size,
                                world_size=wsz,
                                response_token_count=response_valid_tokens,
                            )
                            tensor_shapes = {
                                key: tuple(value.shape) for key, value in batch.batch.items()
                            }
                            print(
                                f"{DIAG_PREFIX} before actor dispatch: global_batch_size={len(batch)}, "
                                f"active_batch_size={active_batch_size}, world_size={wsz}, "
                                f"response_valid_tokens={response_valid_tokens}, tensor_shapes={tensor_shapes}"
                            )
                            ########
                            # ADD: paddding for actor updating.
                            ########
                            if len(batch) % wsz != 0:
                                from recurrent.utils import graceful_padding, indexing_proto
                                padding_index, no_padding_mask = graceful_padding(len(batch), wsz)
                                # use batch[-1] as padding, masked out
                                batch = indexing_proto(batch, padding_index)
                                batch.batch['attention_mask'][~no_padding_mask, :] = 0
                                batch.batch['response_mask'][~no_padding_mask, :] = 0
                                batch.batch['no_padding_mask'] = no_padding_mask
                                batch.meta_info['padded'] = True
                            else:
                                # still need this to activate recurrent-related code in `update_actor`
                                batch.batch['no_padding_mask'] = torch.ones(len(batch), dtype=torch.bool)

                            local_batch_size = len(batch) // wsz
                            per_rank_response_tokens = (
                                batch.batch["response_mask"]
                                .reshape(wsz, local_batch_size, -1)
                                .sum(dim=(1, 2))
                                .tolist()
                            )
                            if any(int(count) < 1 for count in per_rank_response_tokens):
                                raise ValueError(
                                    f"{DIAG_PREFIX} at least one actor rank has no trainable response tokens: "
                                    f"per_rank_local_batch_size={local_batch_size}, "
                                    f"per_rank_response_valid_tokens={per_rank_response_tokens}"
                                )
                            print(
                                f"{DIAG_PREFIX} after actor padding: global_batch_size={len(batch)}, "
                                f"active_batch_size={active_batch_size}, per_rank_local_batch_size={local_batch_size}, "
                                f"per_rank_response_valid_tokens={per_rank_response_tokens}, "
                                f"padding_samples={len(batch) - active_batch_size}"
                            )

                        # update actor
                        with _timer("update_actor", timing_raw):
                            eval_identity_config = self.config.trainer.get("eval_identity", None)
                            if eval_identity_config is not None and bool(
                                eval_identity_config.get("enabled", False)
                            ):
                                raise RuntimeError(
                                    "strict stable evaluation reached actor update; refusing before mutation"
                                )
                            if coral_enabled:
                                from recurrent.research.coral import (
                                    phase_for_step, role_covered_order, role_masks,
                                )
                                coral_phase = phase_for_step(int(self.global_steps))
                                valid_rows = None
                                if "no_padding_mask" in batch.batch:
                                    valid_rows = batch.batch["no_padding_mask"].to(dtype=torch.bool)
                                batch.reorder(role_covered_order(
                                    batch.batch["final_mask"],
                                    int(self.actor_rollout_wg.world_size),
                                    valid_rows,
                                ))
                                batch.meta_info["coral_phase"] = coral_phase
                                audit_batch = batch
                                if "no_padding_mask" in batch.batch:
                                    audit_batch = batch[batch.batch["no_padding_mask"].to(dtype=torch.bool)]
                                active_role_mask, inactive_role_mask = role_masks(
                                    audit_batch.batch["response_mask"],
                                    audit_batch.batch["final_mask"],
                                    coral_phase,
                                )
                                coral_active_tokens = int(active_role_mask.sum().item())
                                coral_inactive_tokens = int(inactive_role_mask.sum().item())
                                metrics[f"coral/phase_{coral_phase}"] = 1.0
                            self._actor_update_calls += 1
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        self._audit_gate_a_weight_sync(
                            global_step=self.global_steps,
                            actor_version=self.global_steps,
                            sync_kind="post_actor_update",
                        )
                        if coral_e1_context is not None:
                            refreshed_materialized_e1 = self._coral_e1_regenerate(
                                coral_e1_context["gen_batch"],
                                coral_e1_context["original_batch"],
                                timing_raw,
                            )
                            cached_e1 = self._coral_e1_resample_terminal(
                                coral_e1_context["cached"],
                                coral_e1_context["original_batch"],
                                "cached_old_memory",
                            )
                            refreshed_e1 = self._coral_e1_resample_terminal(
                                refreshed_materialized_e1,
                                coral_e1_context["original_batch"],
                                "refreshed_memory",
                            )
                            self._coral_e1_write_proposal(
                                cached_e1, refreshed_e1,
                                str(coral_e1_capture_dir),
                                coral_e1_context["source_weight_digest"],
                            )
                        if coral_enabled:
                            coral_ledger = os.getenv("CORAL_EXECUTION_LEDGER")
                            if not coral_ledger:
                                raise RuntimeError(
                                    "CORAL_NO_GO: CORAL_EXECUTION_LEDGER is required"
                                )
                            from recurrent.research.cosi import append_ledger
                            active_grad_norm = float(
                                actor_output_metrics.get("actor/grad_norm", float("nan"))
                            )
                            pg_loss = float(actor_output_metrics.get("actor/pg_loss", float("nan")))
                            if not math.isfinite(active_grad_norm) or active_grad_norm <= 0 \
                                    or not math.isfinite(pg_loss) \
                                    or coral_active_tokens < 1 or coral_inactive_tokens < 1:
                                raise RuntimeError(
                                    "CORAL_NO_GO: missing/non-finite role mechanism metrics"
                                )
                            append_ledger(coral_ledger, {
                                "event": "coral_role_update",
                                "global_step": int(self.global_steps),
                                "phase": str(coral_phase),
                                "actor_update_calls": int(self._actor_update_calls),
                                "weight_sync_required": True,
                                "actor_vllm_sampled_tensor_digest": str(
                                    self._gate_a_synced_actor_digest
                                ),
                                "active_grad_norm": active_grad_norm,
                                "active_pg_loss": pg_loss,
                                "active_tokens": coral_active_tokens,
                                "inactive_tokens": coral_inactive_tokens,
                            })

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                if batch.meta_info.get('padded', False):
                    from recurrent.utils import indexing_proto
                    batch = indexing_proto(batch, batch.batch['no_padding_mask'])
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                from recurrent.research.gate_a_execution import (
                    append_gate_a_record,
                    gate_a_enabled,
                    partition_numeric_metrics,
                )

                if gate_a_enabled():
                    finite_metrics, nonfinite_metric_names = partition_numeric_metrics(metrics)
                    append_gate_a_record(
                        "execution_signal",
                        global_step=self.global_steps,
                        actor_version=self.global_steps,
                        metrics=finite_metrics,
                        nonfinite_metric_names=nonfinite_metric_names,
                    )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
