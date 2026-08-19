import numpy as np
import pytest
import torch

from recurrent.research.actor_batch import build_actor_batch_plan, validate_active_actor_batch
from recurrent.utils import graceful_padding, indexing_proto, td_split
from verl import DataProto


def test_batch_smaller_than_world_size_fails_before_distributed_split():
    with pytest.raises(ValueError, match=r"active_batch_size=3, world_size=4"):
        validate_active_actor_batch(active_batch_size=3, world_size=4, response_token_count=7)


def test_uneven_active_trajectories_are_mask_padded_without_duplication():
    padding_index, loss_sample_mask = graceful_padding(5, 4)
    assert padding_index.tolist() == [0, 1, -1, 2, -1, 3, -1, 4]
    assert loss_sample_mask.tolist() == [True, True, False, True, False, True, False, True]
    assert padding_index[loss_sample_mask].tolist() == list(range(5))


def test_masked_padding_contributes_to_no_actor_objective_or_metrics():
    padding_index, loss_sample_mask = graceful_padding(5, 4)
    real_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    padded_values = torch.cat([real_values, torch.tensor([999.0])])[padding_index]
    for objective in ("policy_loss", "entropy", "kl", "metric"):
        masked_total = padded_values[loss_sample_mask].sum()
        assert masked_total.item() == real_values.sum().item(), objective


def test_all_zero_response_loss_mask_fails_clearly():
    response_mask = torch.zeros(8, 16, dtype=torch.bool)
    with pytest.raises(ValueError, match="response_valid_tokens=0"):
        validate_active_actor_batch(
            active_batch_size=len(response_mask),
            world_size=8,
            response_token_count=int(response_mask.sum()),
        )


def test_normal_divisible_batch_preserves_original_update_plan():
    plan = build_actor_batch_plan(
        train_batch_size=4, rollout_n=2, ppo_mini_batch_size=4, data_parallel_world_size=8
    )
    assert plan.update_steps_per_batch == 1
    assert plan.local_mini_batch_size == 1
    assert plan.local_train_batch_size == 1


def test_gate_a_regression_reports_invalid_prompt_mini_batch_before_split():
    with pytest.raises(ValueError, match=r"TRAIN_BATCH_SIZE=2.*PPO_MINI_BATCH_SIZE=4"):
        build_actor_batch_plan(
            train_batch_size=2, rollout_n=2, ppo_mini_batch_size=4, data_parallel_world_size=2
        )


def test_rollout_expansion_happens_before_per_rank_normalization():
    plan = build_actor_batch_plan(
        train_batch_size=2, rollout_n=2, ppo_mini_batch_size=2, data_parallel_world_size=2
    )
    assert plan.global_rollout_batch_size == 4
    assert plan.update_steps_per_batch == 1
    assert plan.global_rollout_mini_batch_size == 4
    assert plan.local_mini_batch_size == 2
    assert plan.local_train_batch_size == 2


def test_rollout_mini_batch_must_divide_data_parallel_world_size():
    with pytest.raises(ValueError, match=r"global_rollout_mini_batch_size=6.*data_parallel_world_size=4"):
        build_actor_batch_plan(
            train_batch_size=6, rollout_n=2, ppo_mini_batch_size=3, data_parallel_world_size=4
        )


def test_invalid_split_never_reaches_torch_sections_zero():
    proto = DataProto.from_dict(tensors={"x": torch.arange(3)})
    with pytest.raises(ValueError, match="computed_num_micro_batches/sections=0"):
        td_split(proto.batch, 0)


def test_trajectory_identity_fields_remain_aligned_after_reorder():
    proto = DataProto.from_dict(
        tensors={
            "sample_index": torch.tensor([0, 1, 2]),
            "responses": torch.tensor([[10], [20], [30]]),
        },
        non_tensors={
            "trajectory_seed": np.asarray([100, 101, 102], dtype=np.uint64),
            "trajectory_id": np.asarray(["a", "b", "c"], dtype=object),
        },
    )
    reordered = indexing_proto(proto, torch.tensor([2, 0, 1]))
    assert reordered.batch["sample_index"].tolist() == [2, 0, 1]
    assert reordered.batch["responses"].squeeze(-1).tolist() == [30, 10, 20]
    assert reordered.non_tensor_batch["trajectory_seed"].tolist() == [102, 100, 101]
    assert reordered.non_tensor_batch["trajectory_id"].tolist() == ["c", "a", "b"]
