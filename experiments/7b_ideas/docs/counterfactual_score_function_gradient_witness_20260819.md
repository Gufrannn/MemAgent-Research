# Counterfactual score-function gradient witness (W4 v2)

W4 is an analysis-only witness at one frozen checkpoint. Its legal reference requires an on-policy candidate C from that checkpoint, an exact same-candidate E commit/retain pair, a candidate-independent retain/NOOP baseline `B_N(S)`, an exact writer-token mask, and a frozen shared-suffix endpoint. Define `tau(C)=Y_commit-Y_retain` and `G_CF=tau(C) sum_writer grad log pi(C|S)`.

`G_CF` is compared with `G_E`, the writer-gradient subvector from the actual complete GRPO loss graph for the same candidate, group, checkpoint, and frozen parameter subspace. The actual group construction, routed bonus, log probabilities, clipping, and KL must all be reconstructable. The audit reports vector alignment, a signed captured coefficient, and effect-weighted silent/opposing mass. These are descriptive witness quantities, not automatic method or training authorization.

Off-policy candidates; candidate-, RNG-, or cache-dependent NOOP; an inexact writer mask; a mismatched suffix; or an unreconstructable group/bonus/logprob/clip/KL produces `W4_NO_GO` and caps the claim at W3. A nonzero `||G_E-G_CF||`, a gradient norm, scalar advantage sign, or one parameter delta is forbidden as W4 evidence.

The retain/NOOP reader seed and all coupling randomness must be fixed before candidate sampling and satisfy `U independent of C | S`. Deriving a reader seed from the candidate hash, advancing RNG as a function of candidate length, or allowing candidate-dependent cache state is `W4_NO_GO`. The writer score mask covers the complete policy-controlled random sequence, including EOS or the stop decision. Validity rules, truncation rules, and row selection are frozen before tau; selecting `tau>0` or conditioning on any endpoint is forbidden. P-A or P-B success cannot bypass any of these three identity gates.

`analysis/audit_counterfactual_score_function_identity_20260819.py` provides an outcome-free CPU control: a constant candidate-independent baseline must reproduce the direct commit-return gradient on full candidate support. Candidate-dependent baselines, tau-conditioned selection, and an omitted stop decision are required negative controls.

The default is `optimizer_steps=0` and `new_rollouts=false`. Four independent events are plumbing only. At least 20 independent events with a frozen nonzero material-effect threshold are required before a scientific audit can begin. Even a qualified capture does not itself authorize a W4 claim. A one-step update could occur only in a future uniquely selected method microtrial after W4 is otherwise established, evaluated on a fresh locked set; it cannot serve as W4 evidence.

The CPU capture helpers never call backward or an optimizer. The launcher rejects all W4 gradient pilot and W4 optimizer requests, after requiring both the evidence ledger and outcome-blind single-extension router decision.
