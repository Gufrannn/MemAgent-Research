# Certificate-gated commit/rollback v3 multi-write audit

This downstream analyzer is analysis-only and accepts exactly one checkpoint per invocation. Its unique decision key is `(stable_example_id, write_id, checkpoint_hash)`. The prefrozen `eligible_write_count` must equal the actual eligible rows for every example; every row has weight `1/m_i^elig`, each example has total weight one, and postbranch missing weight is never transferred.

Arm means, best constant, rowwise oracle, certificate value, gain, opportunity, regret, coverage, effect probabilities, and harm/benefit mass all use the same pre-candidate weight. One hundred percent pair-complete and pair-qualified branch closure is mandatory. Bootstrap resamples stable-example clusters with all their writes; writes are never IID observations and never increase independent n.

Potential outcomes may not be averaged within an example before taking the rowwise oracle. In the exact counterexample, every example has one good and one bad write with opposite effects. Opportunity is `.5`, and a correct frozen certificate captures gain `.5`; row-simple averaging erases this actionability and is rejected. The result is neither sequential nor closed-loop policy value and never unlocks training.

The eligible-stratum error probabilities are named only `eligible_target_harmful_commit_probability` and `eligible_target_beneficial_rejection_probability`; every report carries `raw_pool_probability_identified=false`. Eligible `1/m_i^elig` weights do not identify deployment/raw-pool risk. A complete target→R1→pair ledger can bound a binary raw event by `[M_obs/N_raw,(M_obs+M_miss)/N_raw]`, but raw policy value remains undefined without a prefrozen R=0 fallback. Missing rows are never silently mapped to commit or rollback.
