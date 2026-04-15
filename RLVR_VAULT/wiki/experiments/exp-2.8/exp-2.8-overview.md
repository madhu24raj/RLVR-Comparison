---
type: experiment
tags: [exp-2.8, ppo, grpo, critic-sweep, advantage-estimation]
sources: []
updated: 2026-04-15
---

# Exp 2.8 — Critic Size Sweep: PPO Variants vs GRPO

## Goal
Determine whether larger critic networks reduce advantage estimation error enough to justify their compute cost, relative to GRPO's zero-critic baseline.

## Setup
- Methods: PPO with critic sizes {none (=GRPO baseline), small, medium, large}
- Dataset: [[gsm8k]] (primary)
- Key output: crossover plot of Monte Carlo advantage error vs accuracy
- Additional metrics: compute cost, convergence speed

## Critic Size Definitions

| Variant | Architecture | Approx. Params |
|---------|-------------|----------------|
| None (GRPO) | No critic | 0 |
| Small | MLP head on frozen policy | ~10M |
| Medium | Full LM head fine-tuned | ~800M |
| Large | Separate transformer | ~8B |

## Aggregate Results
_(populated after runs complete)_

| Critic Size | MC Advantage Error | GSM8K Accuracy | Compute Cost |
|-------------|-------------------|----------------|-------------|
| None (GRPO) | | | |
| Small | | | |
| Medium | | | |
| Large | | | |

## Per-Run Pages
_(created as runs complete)_

## Key Findings
_(populated after runs complete)_

## Open Questions
- At what critic size does accuracy improvement flatten (crossover point)?
- Is the crossover point compute-budget-dependent?
