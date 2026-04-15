---
type: experiment
tags: [exp-2.7, ppo, grpo, dpo, gsm8k, humaneval, head-to-head]
sources: []
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
- Metrics: accuracy, training stability, convergence speed, advantage estimation error

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
