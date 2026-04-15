---
type: concept
tags: [grpo, policy-gradient, rl, critic-free, advantage-estimation]
sources: []
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

## In This Project

Baseline "no critic" condition in exp 2.8. Head-to-head with PPO and DPO in exp 2.7. Compared against DPO under sparse and noisy labels in exp 2.9.

## Connections
- [[ppo]] — compared in exp 2.7 and 2.8
- [[advantage-estimation]] — group-normalized variant
- [[kl-penalty]] — explicit in GRPO objective
- [[verifiable-reward]] — reward signal for group ranking
- [[dpo]] — compared in exp 2.9

## Key Sources
_(populated on paper ingest)_
