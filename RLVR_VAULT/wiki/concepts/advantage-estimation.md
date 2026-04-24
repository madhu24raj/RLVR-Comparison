---
type: concept
tags: [advantage-estimation, gae, critic, ppo, grpo]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# Advantage Estimation

`A(s,a) = Q(s,a) - V(s)` measures how much better action `a` is than the expected action at state `s`. Central to all policy gradient methods.

## Mechanism

**PPO (GAE):**

    δ_t = r_t + γV(s_{t+1}) - V(s_t)
    A_t^GAE = Σ_{k=0}^{T} (γλ)^k · δ_{t+k}

Requires trained critic V(s). λ controls bias-variance tradeoff.

**GRPO (group normalization):**

    A_i = (r_i - mean(r_group)) / std(r_group)

No critic. See [[grpo]] for the exact variance formula.

> [!warning] "Unbiased given sufficient group size" — clarification from [[arora2026-rlvr-foundations]]. "Bias" in the paper's Theorem 2.5 refers to `E[Â^GRPO_i]` (the mean of the baseline-subtracted estimator), not unbiasedness with respect to the true Q-function advantage `A*(s,a)`. Under population normalization, `E[Â^GRPO_i] = 0` exactly for all G ≥ 2; under empirical normalization it is heuristically `O(1/G)`. The GRPO advantage is **not** an unbiased estimator of the true advantage at any finite G — it is an unbiasedly-centered normalized reward.

**Monte Carlo advantage error (exp 2.8 metric):**

    error = mean(|V(s) - MC_return|) over rollouts

Tracks how accurately the critic approximates the true return. Swept across critic sizes in exp 2.8.

## In This Project

Key comparison axis in exp 2.8: does a larger critic reduce advantage estimation error enough to justify compute cost vs GRPO's zero-critic baseline?

## Connections
- [[ppo]] — GAE-based estimation
- [[grpo]] — group-normalization-based estimation
- [[critic-network]] — required for GAE
- [[rlvr]] — reward signal feeding estimates

## Key Sources
- [[arora2026-rlvr-foundations]] — formalizes the PPO bias floor `ε_V/(1−γ)` vs GRPO group-baseline variance `σ*²(1 − 1/G)`; specifies the MC advantage error metric used in exp 2.8
