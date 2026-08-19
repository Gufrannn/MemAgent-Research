# P/H/E held-fixed estimand and exact-NOOP v2 contract

This contract prevents three scientifically different objects from being pooled or selected after outcomes are known.

- **P — same-anchor policy relative credit.** Memory-R2-style rerollout fixes only the pre-session state. Candidate, operation, and post-state are resampled. Its label is `P_SAME_ANCHOR_POLICY_RELATIVE_CREDIT`.
- **H — privileged proxy.** HiMPO-style teacher-forced updated-versus-old target answerability reads a materialized candidate and oracle target information. Its label is `H_PRIVILEGED_PROXY`.
- **E — same-candidate execution effect.** E fixes upstream state S and exactly one materialized candidate C, then changes only commit(C) versus retain(old memory), with a shared suffix and real endpoint. Its label is `E_SAME_CANDIDATE_EXECUTION_EFFECT`.

P is not E and H is not E. A single candidate/state E contrast does not identify state-level writer-policy risk. The preregistered estimand cannot be changed after inspecting outcomes, and P/H/E cannot be searched for the most favorable result.

## Exact-NOOP v2 join

The proposal phase materializes the candidate exactly once. After materialization, neither commit nor retain may run the writer. Retain loads old memory but still records the identical candidate hash. Both arms must share exactly these keys: `stable_id`, `turn`, `upstream_state_hash`, `candidate_memory_token_hash`, `suffix_hash`, `writer_seed`, `reader_seed_or_coupling_id`, and `endpoint_version`. Each arm separately records `arm`, `loaded_memory_token_hash`, `output_token_ids_hash`, and `endpoint_value`.

The validator requires exactly one commit and one retain row per complete eight-key tuple. Missing keys, duplicate arm rows, invalid hashes, non-finite endpoints, incomplete pairs, suffix/endpoint/coupling mismatch, repeated candidate materialization, or writer execution after materialization produce `E_QUALIFICATION_FAIL`. There is no last-write-wins behavior.

The record constructor is `recurrent/research/exact_noop_v2.py`; the machine schema is `schemas/exact_noop_v2.schema.json`; the preflight is `analysis/validate_exact_noop_v2_manifest_20260819.py`. Its success only qualifies the replay object. It does not independently authorize training or any claim. Existing replay that lacks this v2 record is not eligible for Shape A.
