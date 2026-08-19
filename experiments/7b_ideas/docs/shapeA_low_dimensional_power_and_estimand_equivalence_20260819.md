# Shape A: frozen low-dimensional estimand contract

The independent unit is `stable_example_id` (at most 128 paired examples). Arms, turns, seeds, repeated folds, and tokens never increase independent sample size.

For factual/NOOP outcomes, `tau_i = Y_iF - Y_iN`. The paired regression `tau ~ D` and a fully balanced linear stacked `arm × D` regression estimate the same effect-modification estimand under identical weights and covariates. The paired form is the sole primary representation. The stacked form is only an algebra/implementation consistency audit: it is not a second positive result, is not tested separately, and the more significant representation may never be selected.

The sole primary incremental comparison is:

- `B_raw: tau ~ P2_raw_T0`
- `B_struct: tau ~ P2_raw_T0 + D_star`

`D_star` is exactly one preregistered aggregate scalar. Analysis uses four outer folds grouped by `stable_example_id` and a low-capacity linear/ridge model; preprocessing and any fixed small ridge grid are fitted within training folds. P2 T0 is candidate-free, direction-blind, and information/budget matched. T1 candidate gates cannot flow backward into T0.

Binary harm, direction accuracy, arm-specific slopes, and role/turn/reader-policy specificity are secondary scope gates. If confirmatory harm events are fewer than 20, multivariable logistic claims and AUROC are forbidden as primary outcomes. Report event counts and simple preregistered intervals/strata instead.

This contract is enforced by `evidence_ledger.schema.json` and `validate_shape_a_contract`; README prose alone cannot authorize training.
