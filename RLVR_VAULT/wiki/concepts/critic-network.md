---
type: concept
tags: [critic-network, value-network, ppo, advantage-estimation, exp-2.8]
sources: []
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

## Connections
- [[ppo]] — requires critic for GAE
- [[advantage-estimation]] — uses critic output
- [[grpo]] — critic-free baseline for comparison

## Key Sources
_(populated on paper ingest)_
