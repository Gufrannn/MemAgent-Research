# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import ray

from verl import DataProto


@lru_cache(maxsize=None)
def _load_custom_reward_callable(file_path: str, function_name: str):
    module_digest = hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16]
    module_name = f"_verl_custom_reward_{module_digest}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for reward file '{file_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")
    return getattr(module, function_name)


@dataclass(frozen=True)
class FileRewardFunction:
    """Pickle-safe lazy loader for file-backed custom reward functions."""

    file_path: str
    function_name: str
    reward_kwargs: dict = field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        raw_fn = _load_custom_reward_callable(self.file_path, self.function_name)
        merged_kwargs = {**kwargs, **self.reward_kwargs}
        return raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config):
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    try:
        function_name = reward_fn_config.get("name")
        _load_custom_reward_callable(file_path, function_name)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    return FileRewardFunction(file_path, function_name, reward_kwargs)


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == "batch":
        from verl.workers.reward_manager import BatchRewardManager

        reward_manager_cls = BatchRewardManager
    elif reward_manager_name == "dapo":
        from verl.workers.reward_manager import DAPORewardManager

        reward_manager_cls = DAPORewardManager
    elif reward_manager_name == 'thread':
        from verl.workers.reward_manager import ThreadRewardManager

        reward_manager_cls = ThreadRewardManager
    else:
        raise NotImplementedError

    compute_score = get_custom_reward_fn(config)
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result["reward_extra_info"]
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
