---
type: concept
tags: [dpo, preference-optimization, supervised, preference-pairs]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# DPO (Direct Preference Optimization)

Supervised method that directly optimizes a language model from preference pairs without explicit RL. Derived from the Bradley-Terry preference model and the optimal policy under a KL-constrained reward objective.

## Mechanism

**Objective:**

    L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

where `y_w` = chosen response, `y_l` = rejected response, β controls KL strength.

No reward model, no online sampling, no critic. Purely supervised on a preference dataset.

**Derivation (Rafailov 2023, recapped in [[arora2026-rlvr-foundations]]):** Starting from the Bradley-Terry preference model `P(o⁺ ≻ o⁻|s) = σ(r*(s,o⁺) − r*(s,o⁻))` and substituting the optimal-policy reparametrization `r*(s,a) = β log(π*(a|s)/π_ref(a|s)) + β log Z_β(s)`, the `log Z_β` terms cancel in the difference and the objective reduces to a binary cross-entropy loss over preference pairs. Minimizing it targets the **same** KL-regularized optimum `π*` that PPO and GRPO target.

**Sample complexity** (Prop 2.3, informal): `n_DPO = O(d_π/ε²)` assuming adequate offline coverage; counts preference *pairs*, so the true completion count is `2 d_π/ε²` — comparable to GRPO at G=2.

**GRPO↔DPO structural equivalence:** For binary rewards with G=2, the GRPO gradient is `∇ log(π(o⁺|s)/π(o⁻|s))` — same direction as the DPO gradient. When `π_θ ≈ π_ref` the two gradients are approximately proportional. DPO is offline; GRPO is inherently on-policy.

**Why DPO requires no online RL loop:** The loss is supervised over a fixed preference dataset. However, obtaining/refreshing the preference pairs themselves still requires generating completions. Iterative (on-policy) DPO variants close the gap with GRPO at the cost of rollouts.

> [!warning] DPO's advantage diminishes under verifiable rewards. When labels are individual scalar correct/incorrect judgments, preference pairs must be constructed synthetically (e.g., pair a correct with a randomly-sampled incorrect completion). The pairing procedure is an extra design choice, and the offline distributional mismatch (Xiong 2024, Kim 2026) becomes the dominant weakness — motivating exp 2.9.

## In This Project

Built on LLaMA-3 8B. Preference pairs generated synthetically from verifiable rewards (correct = chosen, incorrect = rejected). Compared against GRPO under full, sparse (10%), and noisy (10% flipped) label conditions in exp 2.9.

> [!note] Uncertain: Whether synthetic pair generation quality significantly limits DPO performance relative to human-labeled preferences. To be investigated in exp 2.9.

## Connections
- [[preference-pairs]] — required input format
- [[grpo]] — compared in exp 2.9
- [[verifiable-reward]] — constructs preference pairs
- [[kl-penalty]] — implicit in β term

## Key Sources
- [[arora2026-rlvr-foundations]] — derives the GRPO↔DPO gradient equivalence at G=2, gives `n_DPO = O(d_π/ε²)` offline-coverage sample complexity, and identifies the offline-mismatch weakness motivating exp 2.9
