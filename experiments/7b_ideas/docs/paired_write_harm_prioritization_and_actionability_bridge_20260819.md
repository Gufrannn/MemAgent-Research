# Paired write-harm prioritization and actionability bridge

This is a secondary offline audit only after Shape A v8, exact factual/NOOP shared-suffix pairing, stable identity, measurement reliability, inference integrity, and semantic-pairing specificity have passed. It adds no rollout or training, is not a second primary, and does not authorize an online safety policy.

The v2 decision key is `(stable_example_id, write_id)`. The target population first samples an example uniformly and then one of its prefrozen eligible writes uniformly, so every row has fixed weight `1/eligible_write_count` and every example has total weight one. Postbranch missing weight is never reallocated. One hundred percent complete and pair-qualified eligible-write closure is mandatory.

Define commit harm `h=Y_NOOP-Y_factual`, with higher frozen `D_star` preregistered as more dangerous. The paired prioritized-harm curve uses cumulative eligible weight—not row count—and integrates adjacent weight intervals. Ties use an outcome-blind SHA-256 hash of the composite example/write key. Bootstrap resamples whole stable-example clusters and retains every write; writes never increase independent n.

For a frozen top-q NOOP rule, all arm means, best constant, rowwise oracle, opportunity, myopic value, gain, coverage, effect mass, and regret use the same pre-candidate weights. `Gain>Opportunity` is a hard join/direction/missingness failure. The only allowed name is “myopic eligible-write actionability under a fixed shared-suffix protocol”; it is not sequential or closed-loop value.

The exact multi-write counterexample has one example with two writes of harm `+1` and `-1`. Mean harm is zero, but rowwise opportunity is `.5`, and a frozen score can capture gain `.5`. Averaging writes before the oracle incorrectly erases the selection opportunity and is prohibited.

Eligible weights identify only the uniform-example→uniform-R=1-eligible-write population. Tables must use `eligible_target_harmful_commit_probability` and `eligible_target_beneficial_rejection_probability`, with `raw_pool_probability_identified=false`. The raw target population instead uses `1/m_i^target` and retains R=0, construction failures, and unpaired rows. A complete target→R1→pair attrition ledger may provide only the event bound `[M_obs/N_raw,(M_obs+M_miss)/N_raw]`. Neither `N_raw=192` nor 100% post-R1 branch closure identifies raw policy value. Without a frozen R=0 fallback, raw policy value is undefined; missing rows are never assigned commit or reject.

Thresholds must be frozen outside confirmation. Descriptive 10/25/50% curve points cannot be searched for the best observed gain. Ranking evidence, available binary action opportunity, and frozen-gate value are distinct layers: failure at a later layer cannot be upgraded manually. Privileged `D_pre^audit` supports at most a finite-protocol offline actionability ceiling; deployment requires an independently admissible online score.
