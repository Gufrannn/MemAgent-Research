# PRD-MemRL primary-source audit — 2026-08-22

This is an independent framing review, not an experimental result or an
implementation certificate.

## Search axes

- Recurrent/agent memory RL: MemAgent, Memory-R1, ReMemR1, and CMI-Mem.
- Sequence-level language-policy optimization: SPPO; PPO/InstructGPT.
- Credit assignment: Counterfactual Credit Assignment and Chunked-TD.
- Safe/constrained improvement: CPO and SPIBB.
- Information bottleneck/rate distortion: Deep VIB, Variational Predictive IB,
  InfoBot, Capacity-Limited Actor-Critic, and RepDIB.
- Robust RL/DRO: robust MDP foundations and the ICML 2025 average-reward DRRL
  reduction framework.

The draft claim matrix records what each primary source solves, the remaining
gap, and the comparison or naming restriction it imposes. Links point to
publisher pages, papers, or official repositories rather than surveys.

## Adversarial findings

1. The learned-prior variational bound, primal-dual constrained RL, and a
   rate–utility tradeoff are not individually novel.
2. A broad source `H_t` cannot support a “new evidence” claim. The actor
   interface must isolate `E_t`, `M_t`, and public turn metadata.
3. Unweighted behavior samples scored with a post-update actor do not generally
   estimate either the behavior or current channel KL.
4. A health gate is not safe policy improvement; a capacity sweep is not DRO;
   terminal accuracy is not proof of predictive sufficiency.
5. The strongest residual is the complete measurement-and-control object: a
   source-firewalled recurrent text channel with sequence code length,
   successive accounting, capacity intervention, and future-use diagnostics.

## Verdict

**REFRAME-GO for E0/E1, NO-GO for T5 until both gates pass.** The tightened
route can support a full paper only if rate is distinct from length and fixed-
reference KL and a non-degenerate multi-capacity frontier exists. Otherwise it
must terminate or merge rather than be cosmetically renamed.
