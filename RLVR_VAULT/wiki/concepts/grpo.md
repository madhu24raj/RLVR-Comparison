---
type: concept
tags: [grpo, policy-gradient, rl, critic-free, advantage-estimation]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# GRPO (Group Relative Policy Optimization)

Critic-free policy optimization from DeepSeek-R1. Computes advantages by normalizing rewards within a group of responses to the same prompt, eliminating the need for a separate value network.

## Mechanism

For G responses `{o_1, ..., o_G}` sampled from the same prompt:

    A_i = (r_i - mean(r)) / std(r)

**Objective:**

    L_GRPO = E[min(r_t(θ) · A_i, clip(r_t(θ), 1-ε, 1+ε) · A_i)] - β · KL(π || π_ref)

No value network trained. Lower memory footprint than PPO (no critic parameters or optimizer states).

**Variance of the advantage estimator** ([[arora2026-rlvr-foundations]], Theorem 2.1):

    Var[r_i − r̄_G] = σ*²(s) · (1 − 1/G)   (exact, all G ≥ 2)

Larger G *increases* numerator variance toward the population reward variance σ*² — it does **not** reduce advantage variance. What larger G buys is a more accurate group-mean baseline: `r̄_G` concentrates on `μ*(s) = E_π[r(s,·)]` at rate `O(1/√G)`. In practice most of the estimation-accuracy benefit is captured by `G ∈ [4, 16]`.

**Bias of the advantage estimator** (mean of `Â_i`, not unbiasedness w.r.t. true A*):

- Population normalization (denominator `σ*(s)`): `E[Â_i] = 0` exactly.
- Empirical normalization (denominator `σ_{r,G}`): `E[Â_i] = O(1/G)` (heuristic, asymptotic).

**Sample complexity** (Prop 2.3, informal): `n_GRPO = O(C_π/ε²)` — **independent of G**, because the G-fold variance reduction exactly cancels the G-fold per-step rollout cost.

**GRPO ↔ DPO equivalence:** For G=2 with binary verifiable rewards, the group-normalized advantages become `Â⁺ = +1, Â⁻ = −1` and the policy gradient reduces to `∇ log(π(o⁺|s)/π(o⁻|s))` — the same direction as the DPO gradient (Mroueh 2025, cited in [[arora2026-rlvr-foundations]]). Key difference: GRPO is on-policy; DPO is offline.

## In This Project

Baseline "no critic" condition in exp 2.8. Head-to-head with PPO and DPO in exp 2.7. Compared against DPO under sparse and noisy labels in exp 2.9.

## Connections
- [[ppo]] — compared in exp 2.7 and 2.8
- [[advantage-estimation]] — group-normalized variant
- [[kl-penalty]] — explicit in GRPO objective
- [[verifiable-reward]] — reward signal for group ranking
- [[dpo]] — compared in exp 2.9; structurally equivalent at G=2 on binary rewards
- [[critic-network]] — GRPO's group baseline replaces the PPO critic (no function-approx bias)

## Key Sources
- [[arora2026-rlvr-foundations]] — derives Var[r_i − r̄_G] = σ*²(1 − 1/G), `O(1/√G)` baseline concentration, G-independent sample complexity, and GRPO↔DPO equivalence at G=2
