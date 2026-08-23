import copy
import random

import numpy as np
import pytest
import torch

from recurrent.research.rwwpo_transaction import (
    ALPHA_GRID, digest, displacement_norm, largest_tested_feasible,
    logical_transaction_seed, off_behavior_exposed, parameter_snapshot,
    prefix_distribution_stats, proposal_clock, set_interpolated_parameters,
    restore_rng, rng_snapshot, seed_transaction_rng, stateless_proposal_lr,
    writer_logprob_rms,
)


def test_alpha_one_feasible():
    decision = largest_tested_feasible({alpha: alpha == 1.0 for alpha in ALPHA_GRID})
    assert decision.alpha == 1.0 and decision.accepted_nonzero


def test_full_infeasible_half_feasible():
    decision = largest_tested_feasible({alpha: alpha <= 0.5 for alpha in ALPHA_GRID})
    assert decision.alpha == 0.5


def test_only_smallest_feasible():
    decision = largest_tested_feasible({alpha: alpha == ALPHA_GRID[-1] for alpha in ALPHA_GRID})
    assert decision.alpha == 1 / 32


def test_all_nonzero_infeasible():
    decision = largest_tested_feasible({alpha: False for alpha in ALPHA_GRID})
    assert decision.alpha == 0 and not decision.accepted_nonzero


def test_nonmonotone_feasibility_chooses_largest_tested():
    feasible = {alpha: alpha in (0.5, 0.125) for alpha in ALPHA_GRID}
    assert largest_tested_feasible(feasible).alpha == 0.5


def test_untested_candidate_is_fail_closed():
    with pytest.raises(ValueError, match="not actually tested"):
        largest_tested_feasible({1.0: False, 0.5: True})


def test_zero_proposal_cannot_be_accepted():
    decision = largest_tested_feasible({alpha: True for alpha in ALPHA_GRID}, proposal_zero=True)
    assert decision.alpha == 0 and decision.proposal_zero and not decision.accepted_nonzero


def test_scaled_parameter_displacement_is_real_not_ledger_only():
    model = torch.nn.Linear(2, 1, bias=False).double()
    old = parameter_snapshot(model)
    full = [value + 2.0 for value in old]
    set_interpolated_parameters(model, old, full, 0.25)
    committed = parameter_snapshot(model)
    torch.testing.assert_close(committed[0], old[0] + 0.5)
    assert displacement_norm(old, committed) == pytest.approx(0.25 * displacement_norm(old, full))


def test_optimizer_rollback_digest_detects_tamper():
    model = torch.nn.Linear(2, 1).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = copy.deepcopy(optimizer.state_dict())
    optimizer.param_groups[0]["lr"] = 7e-3
    assert digest(before) != digest(optimizer.state_dict())


def test_proposal_clock_and_stateless_warmup_are_logical_not_attempt_based():
    assert [proposal_clock(1, 1), proposal_clock(1, 2), proposal_clock(2, 1)] == [1, 2, 3]
    assert stateless_proposal_lr(base_lr=1e-6, warmup_proposals=2,
                                 total_proposals=800, proposal_id=1) == pytest.approx(5e-7)
    assert stateless_proposal_lr(base_lr=1e-6, warmup_proposals=2,
                                 total_proposals=800, proposal_id=2) == pytest.approx(1e-6)
    assert stateless_proposal_lr(base_lr=1e-6, warmup_proposals=2,
                                 total_proposals=800, proposal_id=799) == pytest.approx(1e-6)


def test_logical_seed_has_no_attempt_coordinate():
    kwargs=dict(experiment_seed=2026,round_id=27,inner_id=2,rank=1,stream="actor")
    assert logical_transaction_seed(**kwargs) == logical_transaction_seed(**kwargs)
    assert logical_transaction_seed(**kwargs) != logical_transaction_seed(**{**kwargs,"round_id":28})


def test_reject_restores_complete_pre_reseed_transaction_rng_state():
    random.seed(17)
    np.random.seed(19)
    torch.manual_seed(23)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(29)
    entry = rng_snapshot()

    def draw_all():
        values = [random.random(), float(np.random.random()), float(torch.rand(()))]
        if torch.cuda.is_available():
            values.extend(float(torch.rand((), device=index).cpu())
                          for index in range(torch.cuda.device_count()))
        return values

    restore_rng(entry)
    expected = draw_all()
    restore_rng(entry)
    seed_transaction_rng(logical_transaction_seed(
        experiment_seed=2026, round_id=27, inner_id=2, rank=0,
        stream="actor_transaction"))
    for _ in range(7):
        draw_all()
    restore_rng(entry)
    assert draw_all() == expected
    assert "numpy" in entry and "torch_cpu" in entry and "python" in entry


def test_exposure_requires_parameter_and_writer_movement():
    assert off_behavior_exposed(relative_parameter_displacement=2e-6,
                                writer_logprob_rms_value=3e-5,
                                tau_theta=1e-6,tau_logprob=1e-5)
    assert not off_behavior_exposed(relative_parameter_displacement=0.0,
                                    writer_logprob_rms_value=3e-5,
                                    tau_theta=1e-6,tau_logprob=1e-5)
    assert not off_behavior_exposed(relative_parameter_displacement=2e-6,
                                    writer_logprob_rms_value=0.0,
                                    tau_theta=1e-6,tau_logprob=1e-5)


def test_writer_logprob_rms_is_trajectory_balanced():
    behavior=torch.zeros((3,4),dtype=torch.float64)
    current=torch.tensor([[1.,1.,1.,1.],[3.,0.,0.,0.],[3.,0.,0.,0.]],dtype=torch.float64)
    mask=torch.tensor([[1,1,1,1],[1,0,0,0],[1,0,0,0]],dtype=torch.bool)
    sample=torch.tensor([0,1,1])
    # trajectory 0 MSE=1; trajectory 1 MSE=9, hence sqrt((1+9)/2)
    assert writer_logprob_rms(current,behavior,mask,sample)==pytest.approx(5**0.5)


def test_prefix_stats_separate_trajectory_and_prompt_root_support():
    rows=[
        {"turn":0,"sample_index":0,"root_identity_hash":"a","log_ratio":0.0},
        {"turn":0,"sample_index":1,"root_identity_hash":"a","log_ratio":0.0},
        {"turn":0,"sample_index":2,"root_identity_hash":"b","log_ratio":3.0},
        {"turn":0,"sample_index":3,"root_identity_hash":"b","log_ratio":3.0},
    ]
    stats=prefix_distribution_stats(rows,q_min=0.4,root_q_min=0.5,log_ratio_cap=4.0)[0]
    assert stats["batch_size"]==4 and stats["root_count"]==2
    assert stats["root_ess_fraction"] < 1.0
    assert len(stats["root_loo"])==2
    assert not stats["root_loo_supported"]
    assert stats["root_loo_feasibility_flip_fraction"] == 0.0


def test_prefix_stats_support_root_loo_only_with_three_roots():
    rows = [
        {"turn": 0, "sample_index": index,
         "root_identity_hash": root, "log_ratio": 0.0}
        for index, root in enumerate(("a", "b", "c"))
    ]
    stats = prefix_distribution_stats(
        rows, q_min=0.4, root_q_min=0.5, log_ratio_cap=4.0
    )[0]
    assert stats["root_loo_supported"]
    assert stats["root_loo_complete_fraction"] == 1.0
    assert stats["root_loo_feasibility_flip_fraction"] == 0.0


def test_prefix_stats_reject_missing_root_identity():
    with pytest.raises(ValueError,match="stable root"):
        prefix_distribution_stats(
            [{"turn":0,"sample_index":0,"log_ratio":0.0}],
            q_min=0.5,root_q_min=0.5,log_ratio_cap=4.0)
