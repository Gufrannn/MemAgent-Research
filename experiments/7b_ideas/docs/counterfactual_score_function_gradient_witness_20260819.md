# Counterfactual score-function audit (W4 v8)

W4 is zero-GPU capture and CPU audit infrastructure. It never authorizes a gradient pilot, optimizer step, rollout, training claim, or algorithm-novelty claim.

## Endpoint-target parity comes first

Every audit must carry a complete endpoint parity ledger for horizon, future policy, suffix contract, reader, answer cell and normalization, reward components/weights/scale, invalid/truncation/missing rules, row weights, and candidate/score-mask joinability. Each training and scientific definition is content hashed. Missing, unknown, or internally inconsistent entries produce `ENDPOINT_TARGET_AMBIGUOUS` and cap the result at W3.

`CV_same_endpoint` is legal only when every row is identical. A candidate-independent exact-NOOP with exogenous coupling is then only a control variate: it has the same expected writer gradient as raw commit return or a legal group estimator, and is neither a new nor a truer action-value target. An IID including-self all-mean group estimator has expectation scale `(n-1)/n`; for `n=4` this is `3/4`. Multiplication by `n/(n-1)` removes the scale, and the result is algebraically identical to LOO in every realized batch. The scientific comparison is the cross-independent-group paired mean of `G_credit_debiased-G_CF`, whose theoretical null is zero, plus its variance. MSE is omitted unless an independent many-action or replication reference exists. A stable nonzero mean first diagnoses endpoint parity, IID sampling, baseline independence, selection, score masking, or loss reconstruction—not recovered or lost credit.

When any endpoint definition differs, the only legal mode is `OM_distinct_endpoint`. It is named `surrogate_objective_gradient_mismatch`, never lost credit for the same reward. RED calibration must already pass. Independent candidate groups report a clustered interval for `G_eval-G_train`, an endpoint-label-shuffle null, and equal-scale component ablations. For `p=.5`, `R_train(A)=2` has gradient zero while `Y_eval(A)=2+A` has gradient `.25`; exact-NOOP subtraction cannot repair that target mismatch. The exact check lives in `analysis/enumerate_w4_endpoint_target_20260819.py`.

Single-batch alignment, captured ratios, silent/opposed mass, nonzero norms, nonzero gradient rate, scalar advantage sign, and a one-parameter delta are permanently forbidden as credit evidence. Equal rewards make the group-centered gradient zero; a constant external baseline can still produce a nonzero finite-batch `constant * sum(score)`, which is zero-mean score noise rather than credit evidence.

## Policy derivative scope

The policy-node ledger records role, checkpoint, parameter identity, `shares_theta`, target-policy versus frozen-environment semantics, token span including EOS/stop, mask hash, actual/reference inclusion, arm parity, and stop-gradient behavior. Actual and reference score-node sets must match.

`L_frozen_future_policy` freezes future writer, answer, and reader under `pi_bar` with stop-gradient. Only the complete current-candidate score, including policy-controlled EOS/stop, is differentiated; its sole legal name is `frozen_future_local_writer_gradient`. This is the recommended future W4 mode.

`T_tied_recurrent_policy` treats every future score node sharing theta as part of the terminal recurrent-policy gradient. If all such nodes are present in actual and reference graphs, the legal name is `full_terminal_recurrent_policy_gradient`. A current-writer-only graph may be audited only as `local_recurrent_semi_gradient`; it cannot support a full recurrent-policy claim. With independent Bernoulli actions and `R=A0*A1`, the current and future terms are `.125` each, so the full tied gradient is `.25`; the current-only mask omits half. `analysis/enumerate_w4_future_policy_20260819.py` checks this exactly.

## Gradient geometry is a separate frozen choice

Raw Euclidean, full, sketch, or Rademacher gradient cosine is not invariant to invertible reparameterization. It is always labeled `fixed_coordinate_secondary_diagnostic_only`; it cannot independently select a method or support a parameterization-invariant directional-conflict claim. If no direction adjudication is requested, the mandatory mode is `none_no_direction_adjudication`.

A future directional audit must choose exactly one geometry before endpoint unblinding. `Fisher_tested_subspace` records matching actual/reference parameter-block hash, projection hash, effective rank, condition number, relative damping, eigen cutoff, and sensitivity-manifest hash. Its only legal claim is `empirical_Fisher_tested_subspace_geometry`. `optimizer_delivery` reconstructs `DeltaTheta_actual` without state mutation from real Adam moments, learning rate, clipping, weight decay, accumulation, and scaling, then reports `g_endpoint^T DeltaTheta_actual`. Its only legal claim is optimizer-delivery evidence, not pure credit evidence. Different actual/reference parameter blocks, a post-outcome geometry choice, or an incomplete delivery reconstruction is `W4_NO_GO`.

Euclidean reversal without Fisher/delivery reversal is classified as `COORDINATE_SCALE_ARTIFACT`. Negative optimizer delivery is `DELIVERY_CONFLICT_NOT_AUTOMATIC_CREDIT_FAILURE`. In the exact 2D check, an invertible scale change moves Euclidean cosine from `.316` to `-.956`, while the empirical-Fisher bilinear remains `.5` and the covector-tangent delivery pairing remains `.1`. See `analysis/enumerate_w4_gradient_geometry_20260819.py`.

Four prefrozen independent candidate groups are plumbing only; at least twenty are needed before a scientific audit is even ready. Expected equivalence, variance/MSE reduction, an equal-cost win over LOO and state-value controls, and fresh endpoint safety may at most make a separate engineering application eligible to request. A new frozen authorization would still be required. The launcher unconditionally rejects W4 pilot and optimizer requests. Adaptive-stop v4 remains unchanged: T100 cannot control continuation, 200 is terminal, and 400 is never automatic.
