# P/H/E held-fixed estimand and exact-NOOP v2 contract

This contract prevents three scientifically different objects from being pooled or selected after outcomes are known.

- **P — same-anchor policy relative credit.** Memory-R2-style rerollout fixes only the pre-session state. Candidate, operation, and post-state are resampled. Its label is `P_SAME_ANCHOR_POLICY_RELATIVE_CREDIT`.
- **H — privileged proxy.** HiMPO-style teacher-forced updated-versus-old target answerability reads a materialized candidate and oracle target information. Its label is `H_PRIVILEGED_PROXY`.
- **E — same-candidate execution effect.** E fixes upstream state S and one materialized candidate C, then changes commit(C) versus retain(old memory). Every record further declares E0, EH, or EF; VG is a separate closed-loop object.

P is not E and H is not E. A single candidate/state E contrast does not identify state-level writer-policy risk. The preregistered estimand cannot be changed after inspecting outcomes, and P/H/E cannot be searched for the most favorable result.

## Exact-NOOP v2 join

The proposal phase materializes the candidate exactly once. Neither arm reruns the writer. EH shares the base identity/state/candidate/writer/reader/endpoint keys plus `exogenous_suffix_contract_hash`, `future_policy_hash`, and `horizon`. That contract covers only exogenous future chunks/order/budget. Each arm separately records its loaded memory, output, endpoint, realized trajectory, and future hash chain; realized trajectory is not a pair key.

The validator requires exactly one commit and retain row per mode-specific key. Missing/mixed modes, ambiguous legacy `suffix_hash`, duplicate arms, invalid hashes, incomplete pairs, external-contract mismatch, repeated materialization, or writer execution produce `E_QUALIFICATION_FAIL`.

The record constructor is `recurrent/research/exact_noop_v2.py`; the machine schema is `schemas/exact_noop_v2.schema.json`; the preflight is `analysis/validate_exact_noop_v2_manifest_20260819.py`. Its success only qualifies the replay object. It does not independently authorize training or any claim. Existing replay that lacks this v2 record is not eligible for Shape A.
