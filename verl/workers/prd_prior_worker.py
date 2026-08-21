"""Trainable history-blind coding prior for PRD-MemRL.

This worker is deliberately separate from the actor.  Its only conditioning
input is the already constructed ``prd_prior_*`` sequence (M_t and t), followed
by the actor-written response.  Raw environment/history fields are rejected at
the worker boundary so an accidental trainer change fails closed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn
from verl.utils.torch_functional import logprobs_from_logits


_REQUIRED = {
    "prd_prior_input_ids",
    "prd_prior_attention_mask",
    "prd_prior_position_ids",
    "responses",
    "writer_mask",
}
_FORBIDDEN_FRAGMENTS = (
    "evidence",
    "history_chunk",
    "new_chunk",
    "future",
    "gold",
    "ground_truth",
    "reward",
    "outcome",
    "answer",
)


def validate_history_blind_batch(batch: Mapping[str, torch.Tensor]) -> None:
    missing = sorted(_REQUIRED.difference(batch))
    if missing:
        raise KeyError(f"PRD prior batch missing required fields: {missing}")
    forbidden = sorted(
        key for key in batch if any(fragment in key.lower() for fragment in _FORBIDDEN_FRAGMENTS)
    )
    if forbidden:
        raise RuntimeError(f"PRD_PRIOR_TAINT forbidden fields reached prior worker: {forbidden}")
    ids = batch["prd_prior_input_ids"]
    attention = batch["prd_prior_attention_mask"]
    positions = batch["prd_prior_position_ids"]
    responses = batch["responses"]
    writer_mask = batch["writer_mask"]
    if ids.ndim != 2 or attention.shape != ids.shape or positions.shape != ids.shape:
        raise ValueError("PRD prior input ids/attention/position tensors must be aligned rank-2 tensors")
    if responses.ndim != 2 or writer_mask.shape != responses.shape:
        raise ValueError("PRD responses and writer_mask must be aligned rank-2 tensors")
    if ids.shape[0] != responses.shape[0] or ids.shape[1] < responses.shape[1] + 1:
        raise ValueError("PRD prior sequence must contain a prefix and every response token")
    if not torch.all((writer_mask == 0) | (writer_mask == 1)):
        raise ValueError("writer_mask must be binary")
    if torch.any(writer_mask > attention[:, -responses.shape[1] :]):
        raise ValueError("writer_mask selects padding/non-visible prior tokens")


@dataclass
class ProjectedDual:
    value: float
    learning_rate: float
    maximum: float
    updates: int = 0

    def update(self, observed_rate: float, capacity_nats: float) -> float:
        if not (torch.isfinite(torch.tensor(observed_rate)) and torch.isfinite(torch.tensor(capacity_nats))):
            raise ValueError("dual update requires finite rate and capacity")
        self.value = min(self.maximum, max(0.0, self.value + self.learning_rate * (observed_rate - capacity_nats)))
        self.updates += 1
        return self.value

    def state_dict(self) -> dict:
        return {"value": self.value, "learning_rate": self.learning_rate, "maximum": self.maximum, "updates": self.updates}

    def load_state_dict(self, state: Mapping) -> None:
        expected = {"value", "learning_rate", "maximum", "updates"}
        if set(state) != expected:
            raise RuntimeError(f"invalid PRD dual checkpoint keys: {sorted(state)}")
        if float(state["learning_rate"]) != self.learning_rate or float(state["maximum"]) != self.maximum:
            raise RuntimeError("PRD dual hyperparameters differ from checkpoint")
        self.value = float(state["value"])
        self.updates = int(state["updates"])


class PRDPriorWorker(Worker):
    """Independent FSDP autoregressive prior q(M_{t+1}|M_t,t)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        if not dist.is_initialized():
            dist.init_process_group()
        self.config = config
        self.prd = config.prd_memrl
        self.device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),), mesh_dim_names=["fsdp"])
        self.dual = ProjectedDual(
            value=float(self.prd.dual.initial_value),
            learning_rate=float(self.prd.dual.learning_rate),
            maximum=float(self.prd.dual.maximum),
        )
        self.update_steps = 0

    @staticmethod
    def _validate_data(data: DataProto) -> None:
        """Inspect every DataProto namespace, not merely tensors used by forward."""
        validate_history_blind_batch(data.batch)
        names = set(data.batch.keys()) | set(data.non_tensor_batch.keys()) | set(data.meta_info.keys())
        forbidden = sorted(
            key for key in names if any(fragment in key.lower() for fragment in _FORBIDDEN_FRAGMENTS)
        )
        if forbidden:
            raise RuntimeError(f"PRD_PRIOR_TAINT forbidden fields reached prior worker: {forbidden}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from verl.utils.fsdp_utils import get_init_weight_context_manager
        from verl.utils.torch_dtypes import PrecisionType

        prior_cfg = self.prd.prior
        local_path = copy_to_local(prior_cfg.model_path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=prior_cfg.get("trust_remote_code", False))
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=prior_cfg.get("trust_remote_code", False))
        dtype = PrecisionType.to_dtype(prior_cfg.fsdp_config.get("model_dtype", "fp32"))
        context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh)
        with context():
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                config=model_config,
                torch_dtype=dtype,
                attn_implementation=prior_cfg.get("attn_implementation", "flash_attention_2"),
                trust_remote_code=prior_cfg.get("trust_remote_code", False),
            )
            if prior_cfg.get("gradient_checkpointing", True):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        dist.barrier()
        mixed = prior_cfg.fsdp_config.get("mixed_precision", {})
        mp = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(mixed.get("param_dtype", "bf16")),
            reduce_dtype=PrecisionType.to_dtype(mixed.get("reduce_dtype", "fp32")),
            buffer_dtype=PrecisionType.to_dtype(mixed.get("buffer_dtype", "fp32")),
        )
        self.prior_module = FSDP(
            model,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=get_fsdp_wrap_policy(model, prior_cfg.fsdp_config.get("wrap_policy")),
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            sync_module_states=True,
            device_mesh=self.device_mesh,
        )
        optim_cfg = prior_cfg.optim
        self.optimizer = torch.optim.AdamW(
            self.prior_module.parameters(), lr=float(optim_cfg.lr),
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
            weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.prior_module, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer, checkpoint_contents=["model", "optimizer", "extra"],
        )

    def _token_log_probs(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        validate_history_blind_batch(batch)
        responses = batch["responses"]
        output = self.prior_module(
            input_ids=batch["prd_prior_input_ids"],
            attention_mask=batch["prd_prior_attention_mask"],
            position_ids=batch["prd_prior_position_ids"],
            use_cache=False,
        )
        logits = output.logits[:, -responses.shape[1] - 1 : -1, :].float()
        temperature = float(self.prd.prior.get("temperature", 1.0))
        if temperature <= 0:
            raise ValueError("prior temperature must be positive")
        return logprobs_from_logits(logits / temperature, responses)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_prior_log_prob(self, data: DataProto):
        data = data.to(torch.cuda.current_device())
        self._validate_data(data)
        self.prior_module.eval()
        with torch.no_grad():
            size = int(self.prd.prior.micro_batch_size_per_gpu)
            log_probs = torch.cat([self._token_log_probs(chunk) for chunk in data.batch.split(size)], dim=0)
        return DataProto.from_dict(tensors={"prior_log_probs": log_probs.cpu()})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_prior(self, data: DataProto):
        data = data.to(torch.cuda.current_device())
        self._validate_data(data)
        self.prior_module.train()
        self.optimizer.zero_grad(set_to_none=True)
        size = int(self.prd.prior.micro_batch_size_per_gpu)
        token_count = data.batch["writer_mask"].sum()
        if token_count.item() <= 0:
            raise RuntimeError("PRD_METHOD_INACTIVE prior update has zero writer tokens")
        loss_total = torch.zeros((), device=torch.cuda.current_device())
        for chunk in data.batch.split(size):
            log_probs = self._token_log_probs(chunk)
            mask = chunk["writer_mask"].to(log_probs.dtype)
            chunk_loss = -(log_probs * mask).sum() / token_count
            chunk_loss.backward()
            loss_total += chunk_loss.detach()
        grad_norm = self.prior_module.clip_grad_norm_(float(self.prd.prior.optim.grad_clip))
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"non-finite PRD prior grad norm: {grad_norm}")
        self.optimizer.step()
        self.lr_scheduler.step()
        self.update_steps += 1
        # FSDP synchronizes gradients; min/max proves every rank advanced together.
        step_tensor = torch.tensor([self.update_steps, self.update_steps], device=torch.cuda.current_device(), dtype=torch.long)
        dist.all_reduce(step_tensor[:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(step_tensor[1:], op=dist.ReduceOp.MAX)
        if step_tensor[0] != step_tensor[1]:
            raise RuntimeError("PRD prior distributed optimizer-step drift")
        metrics = {"prd/prior_nll": float(loss_total), "prd/prior_grad_norm": float(grad_norm), "prd/prior_tokens": int(token_count), "prd/prior_update_steps": self.update_steps}
        return DataProto(meta_info={"metrics": metrics})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def audit_weight_sync(self):
        """Return a rank-stable checksum of the sharded parameter moments."""
        device = torch.cuda.current_device()
        moments = torch.zeros(3, device=device, dtype=torch.float64)
        with torch.no_grad():
            for parameter in self.prior_module.parameters():
                values = parameter.detach().double()
                moments[0] += values.sum()
                moments[1] += values.square().sum()
                moments[2] += values.numel()
        dist.all_reduce(moments, op=dist.ReduceOp.SUM)
        if not torch.isfinite(moments).all():
            raise RuntimeError("non-finite PRD prior parameters")
        canonical = ":".join(f"{value:.17g}" for value in moments.cpu().tolist())
        return {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "prior_update_steps": self.update_steps,
            "global_parameter_moments_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_dual(self, observed_rate: float, capacity_nats: float):
        # All ranks receive identical scalars; broadcast rank 0 result as a hard sync.
        value = self.dual.update(observed_rate, capacity_nats) if dist.get_rank() == 0 else 0.0
        state = torch.tensor([value, self.dual.updates if dist.get_rank() == 0 else 0], device=torch.cuda.current_device(), dtype=torch.float64)
        dist.broadcast(state, src=0)
        self.dual.value, self.dual.updates = float(state[0]), int(state[1])
        return self.dual.state_dict()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        self.checkpoint_manager.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        dist.barrier()
        if dist.get_rank() == 0:
            os.makedirs(local_path, exist_ok=True)
            state = {"schema_version": 1, "global_step": int(global_step), "prior_update_steps": self.update_steps, "dual": self.dual.state_dict()}
            payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
            temp_path = os.path.join(local_path, ".prd_state.json.tmp")
            final_path = os.path.join(local_path, "prd_state.json")
            with open(temp_path, "w", encoding="utf-8") as stream:
                stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, final_path)
        dist.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if hdfs_path is not None:
            raise NotImplementedError("PRD sidecar checkpoint does not support HDFS")
        evidence = self.checkpoint_manager.load_checkpoint(local_path, None, del_local_after_load)
        state_path = os.path.join(local_path, "prd_state.json")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"missing PRD prior/dual checkpoint sidecar: {state_path}")
        with open(state_path, encoding="utf-8") as stream:
            state = json.load(stream)
        if state.get("schema_version") != 1:
            raise RuntimeError("unsupported PRD checkpoint schema")
        self.update_steps = int(state["prior_update_steps"])
        self.dual.load_state_dict(state["dual"])
        canonical = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(canonical).digest()
        digest_tensor = torch.tensor(list(digest), device=torch.cuda.current_device(), dtype=torch.uint8)
        gathered = [torch.empty_like(digest_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, digest_tensor)
        if any(not torch.equal(gathered[0], item) for item in gathered[1:]):
            raise RuntimeError("PRD checkpoint sidecar differs across ranks")
        return {**(evidence or {}), "prd_state_sha256": hashlib.sha256(canonical).hexdigest(), "prior_update_steps": self.update_steps, "dual_updates": self.dual.updates, "dual_value": self.dual.value}
