# Shape A: frozen low-dimensional estimand contract

The independent unit is `stable_example_id` (at most 128 paired examples). Arms, turns, seeds, repeated folds, and tokens never increase independent sample size.

For each `(stable_example_id, write_id)`, harm is `H_H=Y_retain-Y_commit`. With `m_i` writes, every write receives `1/m_i`, so each stable example has total weight one. The paired harm regression and a stacked retain=1/commit=0 write-pair fixed-effect `arm×D` coefficient estimate the same effect-modification slope. Paired is the sole primary; stacked is algebra-only and never a second result. Folds, bootstrap, intervals, and OOF loss aggregation cluster by stable example; row-level HC3 is forbidden.

The sole primary incremental comparison is:

- `B_raw: tau ~ P2_raw_T0`
- `B_struct: tau ~ P2_raw_T0 + D_star`

`D_star` is exactly one preregistered aggregate scalar. Analysis uses four outer folds grouped by `stable_example_id` and a low-capacity linear/ridge model; preprocessing and any fixed small ridge grid are fitted within training folds. P2 T0 is candidate-free, direction-blind, and information/budget matched. T1 candidate gates cannot flow backward into T0.

Binary harm, direction accuracy, arm-specific slopes, and role/turn/reader-policy specificity are secondary scope gates. If confirmatory harm events are fewer than 20, multivariable logistic claims and AUROC are forbidden as primary outcomes. Report event counts and simple preregistered intervals/strata instead.

This contract is enforced by `evidence_ledger.schema.json` and `validate_shape_a_contract`; README prose alone cannot authorize training.
