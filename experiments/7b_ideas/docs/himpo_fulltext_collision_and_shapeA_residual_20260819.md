# HiMPO full-text collision and the residual Shape A claim

Primary-source audit: Yan et al., *HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents*, arXiv:2606.16285v1. HiMPO fixes the same compressed pre-write state, substitutes updated versus previous memory, scores oracle-target answerability, applies a target-conditioned hindsight filter, and scatters memory-specific advantage only to memory tokens.

The following are permanently unavailable as novelty claims here: local counterfactual memory credit; updated/new versus previous/old memory target answerability; hindsight-filtered memory blame; and memory-token-only local advantage. Changing to HotpotQA, Qwen2.5, GRPO, or strict vLLM does not restore novelty. NCR/local credit that merely writes an updated-vs-previous target-answerability delta into writer reward is `NO_METHOD`.

HiMPO is candidate-aware T1 credit because the updated memory has already materialized and oracle target information is read. It cannot enter Shape A's candidate-free T0 comparison as a feature or baseline. T0 only compares equal-permission pre-action old-state, direction-blind raw marginal summaries, and P2 audit structure. The residual Shape A scope is limited to outcome-independent pre-candidate D; P2 structure beating those frozen pre-action marginal summaries; held-out exact materialized candidate factual/NOOP effect modification; arm/role/time and writer-reader scope; and a linked state→harm→actual-GRPO-estimator witness.

If a future unique T1 candidate passes every evidence gate, a HiMPO-like baseline must match writer mask, eligible rows, oracle information, forward budget, token budget, and scale. The current suite only supplies a fail-closed schema/validator: it does not authorize baseline implementation, training, step 400, or C256.

Primary source: [HiMPO arXiv:2606.16285v1](https://arxiv.org/abs/2606.16285).
