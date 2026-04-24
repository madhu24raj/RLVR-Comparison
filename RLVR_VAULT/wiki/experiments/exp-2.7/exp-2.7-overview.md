---
type: experiment
tags: [exp-2.7, ppo, grpo, dpo, gsm8k, humaneval, head-to-head]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# Exp 2.7 — Head-to-Head: PPO vs GRPO vs DPO

## Goal
Compare all three alignment methods on GSM8K and HumanEval with matched compute across 3 random seeds.

## Setup
- Methods: [[ppo]], [[grpo]], [[dpo]] (all on [[llama-3-8b]])
- Datasets: [[gsm8k]], [[humaneval]]
- Seeds: 3 (matched across methods)
- Compute: matched GPU-hours per method
- Metrics: accuracy, training stability, convergence speed, advantage estimation error `|Â − A^MC|` (A^MC estimated from 1000 MC rollouts per state)
- **Rollout allocation (from [[arora2026-rlvr-foundations]] §3.1):** PPO = 1 rollout/prompt (+ critic forward); GRPO = G=8 rollouts/prompt; DPO = 2 rollouts/prompt (one positive, one negative). Total completion count matched across methods.
- **Synthetic pair construction for DPO:** each correct completion paired with a random incorrect completion from the same batch.

## Theoretical Predictions (from [[arora2026-rlvr-foundations]], Remark 2.4)
- **GRPO > PPO** on binary-reward RLVR: PPO's critic is hard to train on sparse binary signals, so the `ε_V/(1−γ)` bias floor dominates.
- **GRPO > offline DPO** due to on-policy distributional coverage (Xiong 2024, Kim 2026 separation).
- **Advantage-estimation-error plot:** PPO should show a non-zero irreducible floor; GRPO's group-mean baseline error should decay as `O(1/√G)`.

## Aggregate Results
_(populated after runs complete)_

| Method | GSM8K Acc (mean±std) | HumanEval Acc (mean±std) | Convergence Step | Stability |
|--------|----------------------|--------------------------|-----------------|-----------|
| PPO | | | | |
| GRPO | | | | |
| DPO | | | | |

## Per-Run Pages
_(created as runs complete)_

## Key Findings
_(populated after runs complete)_

## Open Questions
- Does GRPO's critic-free advantage estimation hurt accuracy on harder problems (HumanEval)?
- Does DPO's offline training limit adaptability compared to online methods?
- How does training stability compare across methods at matched compute?
