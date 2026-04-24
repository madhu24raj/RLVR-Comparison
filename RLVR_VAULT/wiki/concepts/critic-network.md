---
type: concept
tags: [critic-network, value-network, ppo, advantage-estimation, exp-2.8]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# Critic Network

Value network V(s) estimating expected cumulative reward from a given state. Required by PPO for advantage estimation via GAE. Not used by GRPO or DPO.

## Mechanism

**Architecture:** Separate model head on top of the policy backbone, or a smaller separate transformer.

**Training objective:** `L_VF = E[(V(s_t) - R_t)^2]` where R_t is the discounted Monte Carlo return.

**Sizes swept in exp 2.8:**
- None — GRPO baseline (no critic)
- Small — shallow MLP head on frozen policy (~10M params)
- Medium — full LM head fine-tuned (~800M params)
- Large — separate transformer (~8B params)

**Monte Carlo advantage error:**

    MC_error = mean(|V(s_t) - MC_return_t|) over rollouts

Key metric in exp 2.8 crossover plot: critic error vs accuracy.

## In This Project

Central variable in exp 2.8. Hypothesis: larger critics reduce advantage estimation error, improving sample efficiency — but compute cost may not justify the gain vs GRPO's zero-cost baseline.

**Theoretical framing** ([[arora2026-rlvr-foundations]]):
- Critic approximation error `ε_V = sup_s |V_φ(s) − V^π(s)|` determines an **irreducible** PPO advantage bias of `ε_V/(1−γ)`. This bias does not decrease with more samples — it is set by the critic function class.
- Statistical cost: `d_V` function-class complexity term in `n_PPO = O((d_V + C_π)/ε²)`.
- The **PPO↔GRPO crossover** (Theorem 2.5 prediction):

      ε*_V  ≈  (1 − γ) · √(σ*²(1 − 1/G) − σ*²_A)

  For critic error below `ε*_V`, PPO beats GRPO; above it, GRPO wins. Exp 2.8 tests this crossover directly across the {none, small, medium, large} × {G=4,8,16} grid.
- For long reasoning chains `(1−γ)^{-1} ≫ 1` with binary sparse rewards, the `ε_V/(1−γ)` bias term dominates and the crossover pushes toward GRPO regardless of critic size.

## Connections
- [[ppo]] — requires critic for GAE
- [[advantage-estimation]] — uses critic output
- [[grpo]] — critic-free baseline for comparison

## Key Sources
- [[arora2026-rlvr-foundations]] — derives the `ε_V/(1−γ)` bias floor and the crossover formula `ε*_V ≈ (1−γ)·√(σ*²(1 − 1/G) − σ*²_A)` tested in exp 2.8
