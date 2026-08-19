# Closed-loop oracle semantics and opportunity v2

`terminal_pairwise_IUT` alone decides fixed-policy closed-loop actionability. `oracle_semantics_and_opportunity` is an orthogonal auxiliary: failure appends `WITH_ORACLE_OPPORTUNITY_INVALID` and cannot revoke a passed IUT.

In deterministic D mode, the pointwise package oracle and its gap over the best executed fixed package are reportable. In stochastic S mode, raw `E[max_g Y_ig]` is a coupling-dependent hindsight-luck envelope, not an example-heterogeneity oracle. S must estimate conditional policy means with independent assignment/evaluation seed folds, acknowledge finite-replicate plugin-max winner's curse, and report within-example Monte Carlo variation. Audit16 K=4 supports only luck-envelope/coupling/MC feasibility and never stable per-example oracle labels. In F mode the maximum is only a seed-manifest hindsight envelope.

The exact null has identical Bernoulli(.5) policies: best fixed and conditional-mean oracle are both .5, so true selection opportunity is zero. Independent-draw pointwise maxima are .75 for two policies and .9375 for four, while common randomness gives .5. Thus raw random maxima depend on coupling and cannot establish heterogeneity.
