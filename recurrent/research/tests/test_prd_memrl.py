import pytest
import torch

from recurrent.research.prd_memrl import (
    PriorTaintError,
    ProjectedDual,
    assert_rate_not_length,
    conditional_rate_nats,
    load_prd_checkpoint,
    save_prd_checkpoint,
    validate_capacity_frontier,
    validate_prior_record,
)


def test_synthetic_channel_kl_upper_bound_and_dual_converges():
    # Binary symmetric channel, uniform input. I(X;Y)=ln2-H_b(e).
    error = 0.2
    mutual_information = torch.log(torch.tensor(2.0)) + error * torch.log(torch.tensor(error)) + (1-error) * torch.log(torch.tensor(1-error))
    actor = torch.log(torch.tensor([[1-error], [1-error], [error], [error]]))
    prior = torch.full_like(actor, -torch.log(torch.tensor(2.0)))
    _, upper_bound = conditional_rate_nats(actor, prior, torch.ones_like(actor))
    assert upper_bound >= mutual_information - 1e-6

    dual = ProjectedDual(capacity_nats=0.1, learning_rate=0.5)
    for _ in range(10):
        dual.step(0.2)
    assert dual.value > 0
    for _ in range(20):
        dual.step(0.0)
    assert dual.value == 0


@pytest.mark.parametrize("field", ["new_evidence", "history_chunk", "gold", "future", "reward"])
def test_static_prior_taint_firewall_rejects_forbidden_fields(field):
    record = {"previous_memory": "m", "turn_index": 1, field: "leak"}
    with pytest.raises(PriorTaintError):
        validate_prior_record(record)


def test_prior_context_is_exact_and_stable():
    first = validate_prior_record({"previous_memory": "m", "turn_index": 1})
    second = validate_prior_record({"turn_index": 1, "previous_memory": "m"})
    assert first.context_sha256 == second.context_sha256
    with pytest.raises(PriorTaintError):
        validate_prior_record({"previous_memory": "m", "turn_index": 1, "unused": None})


def test_rate_is_log_density_ratio_not_token_length():
    actor = torch.tensor([[-0.1, -0.1], [-0.1, -0.1]])
    prior = torch.tensor([[-2.0, -2.0], [-0.2, -0.2]])
    per_row, mean = conditional_rate_nats(actor, prior, torch.ones_like(actor))
    assert per_row.tolist() == pytest.approx([3.8, 0.2])
    assert mean.item() == pytest.approx(2.0)
    assert assert_rate_not_length([3.8, 0.2, 1.0], [2, 5, 3]) > 0
    with pytest.raises(ValueError, match="RATE_IS_LENGTH"):
        assert_rate_not_length([1.0, 2.0, 3.0], [1, 2, 3])


def test_frontier_requires_multiple_canonical_capacities():
    assert validate_capacity_frontier([0.0, 0.5, 1.0]) == (0.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        validate_capacity_frontier([0.5])
    with pytest.raises(ValueError):
        validate_capacity_frontier([1.0, 0.5, 2.0])


def test_actor_prior_dual_checkpoint_is_complete(tmp_path):
    dual = ProjectedDual(1.0, 0.1, 0.3)
    dual.step(1.5)
    path = save_prd_checkpoint(tmp_path, actor_state={"w": torch.tensor([1])}, prior_state={"w": torch.tensor([2])}, dual=dual)
    actor, prior, restored = load_prd_checkpoint(path)
    assert actor["w"].item() == 1 and prior["w"].item() == 2
    assert restored.state_dict() == dual.state_dict()
