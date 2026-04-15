---
type: concept
tags: [ppo, policy-gradient, rl, critic, advantage-estimation]
sources: []
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

## In This Project

Used in exp 2.7 (head-to-head) and exp 2.8 (critic size sweep). Built on LLaMA-3 8B with a modular critic. Advantage estimation error tracked to compare against GRPO's critic-free approach.

## Connections
- [[grpo]] — critic-free alternative; compared in exp 2.7 and 2.8
- [[advantage-estimation]] — core mechanism
- [[critic-network]] — required component
- [[kl-penalty]] — added as regularization
- [[rlvr]] — reward signal source

## Key Sources
_(populated on paper ingest)_
