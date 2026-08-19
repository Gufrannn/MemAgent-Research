# Shape A: frozen low-dimensional estimand contract

The independent unit is `stable_example_id` (at most 128 paired examples). Arms, turns, seeds, repeated folds, and tokens never increase independent sample size.

For each `(stable_example_id, checkpoint_hash, write_id)`, primary harm is `H_H=Y_retain-Y_commit`; `tau_H=-H_H` is a secondary alias only. The target denominator retains R=0 D-star exclusions. In the frozen R=1 measurable stratum, each eligible write receives `1/m_i^elig`; postbranch missing weight is never reallocated. The paired harm regression and stacked retain=1/commit=0 write-pair FE coefficient estimate the same slope. Paired is sole primary; stacked is algebra-only. Folds, bootstrap, intervals, and OOF aggregation cluster by stable example; row-level HC3 is forbidden.

The sole primary incremental comparison is:

- `B_raw: H_H ~ P2_raw_T0`
- `B_struct: H_H ~ P2_raw_T0 + D_star`

`D_star` is exactly one preregistered aggregate scalar. Analysis uses four outer folds grouped by `stable_example_id` and a low-capacity linear/ridge model; preprocessing and any fixed small ridge grid are fitted within training folds. P2 T0 is candidate-free, direction-blind, and information/budget matched. T1 candidate gates cannot flow backward into T0.

Binary harm, direction accuracy, arm-specific slopes, and role/turn/reader-policy specificity are secondary scope gates. If confirmatory harm events are fewer than 20, multivariable logistic claims and AUROC are forbidden as primary outcomes. Report event counts and simple preregistered intervals/strata instead.

This contract is enforced by `evidence_ledger.schema.json` and `validate_shape_a_contract`; README prose alone cannot authorize training.
