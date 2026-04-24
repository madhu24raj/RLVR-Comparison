---
type: experiment
tags: [exp-2.8, ppo, grpo, critic-sweep, advantage-estimation]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# Exp 2.8 — Critic Size Sweep: PPO Variants vs GRPO

## Goal
Determine whether larger critic networks reduce advantage estimation error enough to justify their compute cost, relative to GRPO's zero-critic baseline.

## Setup
- Methods: PPO with critic sizes {none (=GRPO baseline), small, medium, large}
- Dataset: [[gsm8k]] (primary)
- Key output: crossover plot of Monte Carlo advantage error vs accuracy
- **GRPO group sizes:** `G ∈ {4, 8, 16}` (full {critic-size × G} grid)
- Additional metrics: compute cost, convergence speed, critic approximation error `ε_V` on a held-out value estimation task, PPO advantage bias vs MC ground truth

## Theoretical Prediction ([[arora2026-rlvr-foundations]], Theorem 2.5 + §3.2)

The PPO↔GRPO crossover in critic approximation error:

    ε*_V  ≈  (1 − γ) · √( σ*²(1 − 1/G) − σ*²_A )

- For critic error **below** `ε*_V`: PPO wins (low variance dominates).
- For critic error **above** `ε*_V`: GRPO wins (bias floor dominates).

The sweep should trace out an error-vs-accuracy curve with PPO improving as critic size grows until the crossover, then flattening against the GRPO line. Verify the measured `ε*_V` matches the predicted value for each G.

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
