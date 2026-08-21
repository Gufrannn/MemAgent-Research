import torch

from verl.trainer.ppo.core_algos import compute_policy_loss, compute_rwwpo_policy_loss


def _batch(dtype=torch.float64):
    old = torch.zeros((5, 3), dtype=dtype)
    response = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    final = torch.tensor([0, 0, 0, 1, 1], dtype=torch.bool)
    writer = response & (~final).unsqueeze(-1)
    sample = torch.tensor([0, 1, 0, 0, 1])
    turn = torch.tensor([0, 0, 1, 2, 1])
    adv_scalar = torch.tensor([0.7, -0.4, 0.7, 0.7, -0.4], dtype=dtype)
    adv = adv_scalar.unsqueeze(-1).expand_as(old) * response
    return old, response, final, writer, sample, turn, adv


def test_behavior_point_writer_gradient_exactly_matches_original():
    old, response, final, writer, sample, turn, adv = _batch()
    current_original = old.clone().requires_grad_(True)
    original, *_ = compute_policy_loss(old, current_original, adv, response, 0.2, 0.2, 0.2,
                                       loss_agg_mode="token-mean")
    original_grad, = torch.autograd.grad(original, current_original)

    current_rwwpo = old.clone().requires_grad_(True)
    rwwpo, _ = compute_rwwpo_policy_loss(old, current_rwwpo, adv, response, writer, final,
                                         sample, turn, 0.2, 0.2, 0.2)
    rwwpo_grad, = torch.autograd.grad(rwwpo, current_rwwpo)
    torch.testing.assert_close(rwwpo_grad[writer], original_grad[writer], rtol=0, atol=1e-12)
    torch.testing.assert_close(rwwpo_grad[final.unsqueeze(-1) & response],
                               original_grad[final.unsqueeze(-1) & response], rtol=0, atol=1e-12)


def test_mask_closure_is_fail_closed():
    old, response, final, writer, sample, turn, adv = _batch()
    writer = writer.clone()
    writer[0, 0] = False
    try:
        compute_rwwpo_policy_loss(old, old, adv, response, writer, final, sample, turn, 0.2, 0.2, 0.2)
    except ValueError as exc:
        assert "closure" in str(exc)
    else:
        raise AssertionError("invalid masks were accepted")


def test_prefix_ess_detects_concentration():
    old, response, final, writer, sample, turn, adv = _batch()
    current = old.clone()
    current[0, :2] = 2.0
    _, metrics = compute_rwwpo_policy_loss(old, current, adv, response, writer, final,
                                           sample, turn, 0.2, 0.2, 0.2)
    assert metrics["prefix_stats"][0]["ess_fraction"] < 1.0
    assert metrics["prefix_stats"][0]["chi2"] > 0.0

