import copy

import pytest
import torch

from recurrent.research.rwwpo_transaction import (
    ALPHA_GRID, digest, displacement_norm, largest_tested_feasible,
    parameter_snapshot, set_interpolated_parameters,
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
