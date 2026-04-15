---
type: concept
tags: [dpo, preference-optimization, supervised, preference-pairs]
sources: []
updated: 2026-04-15
---

# DPO (Direct Preference Optimization)

Supervised method that directly optimizes a language model from preference pairs without explicit RL. Derived from the Bradley-Terry preference model and the optimal policy under a KL-constrained reward objective.

## Mechanism

**Objective:**

    L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

where `y_w` = chosen response, `y_l` = rejected response, β controls KL strength.

No reward model, no online sampling, no critic. Purely supervised on a preference dataset.

## In This Project

Built on LLaMA-3 8B. Preference pairs generated synthetically from verifiable rewards (correct = chosen, incorrect = rejected). Compared against GRPO under full, sparse (10%), and noisy (10% flipped) label conditions in exp 2.9.

> [!note] Uncertain: Whether synthetic pair generation quality significantly limits DPO performance relative to human-labeled preferences. To be investigated in exp 2.9.

## Connections
- [[preference-pairs]] — required input format
- [[grpo]] — compared in exp 2.9
- [[verifiable-reward]] — constructs preference pairs
- [[kl-penalty]] — implicit in β term

## Key Sources
_(populated on paper ingest)_
