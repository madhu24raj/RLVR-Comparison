---
type: concept
tags: [ppo, policy-gradient, rl, critic, advantage-estimation]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# PPO (Proximal Policy Optimization)

Policy gradient method using a clipped surrogate objective to prevent excessively large policy updates. Requires a learned critic (value network) to estimate advantages via GAE.

## Mechanism

**Clipped objective:**

    L_CLIP = E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]

where `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` is the probability ratio and ε is the clip range (typically 0.2).

**Advantage estimation (GAE):**

    A_t = Σ (γλ)^k · δ_{t+k}   where δ_t = r_t + γV(s_{t+1}) - V(s_t)

**Value loss:** `L_VF = E[(V(s_t) - R_t)^2]`

**Bias–variance profile** ([[arora2026-rlvr-foundations]], Theorem 2.5):

    |E[Â^PPO(s,a)] − A*(s,a)|  ≤  ε_V / (1 − γ)       (irreducible bias floor)
    Var[Â^PPO(s,a)]            =  O(σ*²_A) ≤ O(σ²_R)  (low variance)

where `ε_V = sup_s |V_φ(s) − V^π(s)|` is the critic approximation error. A good critic can reduce policy-gradient variance from `O(σ²_R)` (return variance) down to `O(σ*²_A)` (residual advantage variance) — often orders of magnitude. The cost is that a misspecified critic introduces an irreducible bias that grows with horizon `(1 − γ)^{-1}` and can never be eliminated by more samples.

**Sample complexity** (Prop 2.3, informal):

    n_PPO = O((d_V + C_π)/ε²) + O(ε_V / ((1 − γ)ε))     # additive bias floor

**Preferred regime:** dense rewards and a critic class that realizes V^π with small ε_V. On binary-sparse RLVR tasks with long reasoning chains, the `ε_V/(1−γ)` term dominates and GRPO becomes preferable. The theoretical PPO↔GRPO crossover critic error is `ε*_V ≈ (1 − γ)·√(σ*²(1 − 1/G) − σ*²_A)` (verified in exp 2.8).

## In This Project

Used in exp 2.7 (head-to-head) and exp 2.8 (critic size sweep). Built on LLaMA-3 8B with a modular critic. Advantage estimation error tracked to compare against GRPO's critic-free approach.

## Connections
- [[grpo]] — critic-free alternative; compared in exp 2.7 and 2.8
- [[advantage-estimation]] — core mechanism
- [[critic-network]] — required component
- [[kl-penalty]] — added as regularization
- [[rlvr]] — reward signal source

## Key Sources
- [[arora2026-rlvr-foundations]] — establishes the `ε_V/(1−γ)` bias floor, the variance reduction argument, and the PPO↔GRPO crossover formula tested in exp 2.8
