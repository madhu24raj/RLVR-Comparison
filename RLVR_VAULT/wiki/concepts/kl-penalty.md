---
type: concept
tags: [kl-penalty, regularization, policy-constraint, ppo, grpo, dpo]
sources: []
updated: 2026-04-15
---

# KL Penalty

KL divergence `KL(π_θ || π_ref)` between the current policy and a reference policy (typically the SFT/base model). Regularization to prevent policy drift from the pretrained distribution.

## Mechanism

**Soft KL penalty (added to reward):**

    r'(s,a) = r(s,a) - β · log(π_θ(a|s) / π_ref(a|s))

β is the KL coefficient — higher β keeps the policy closer to π_ref.

PPO's ratio clipping is a related proxy that limits per-step policy change without an explicit KL term.

## In This Project

Applied explicitly in GRPO objective. PPO uses clipping as a proxy. DPO has an implicit KL term controlled by β. Kept matched across all three methods in exp 2.7 for fair comparison.

## Connections
- [[ppo]], [[grpo]], [[dpo]] — all incorporate KL constraint
- [[rlvr]] — KL penalty balances reward maximization against reference drift

## Key Sources
_(populated on paper ingest)_
