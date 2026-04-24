---
type: experiment
tags: [exp-2.9, grpo, dpo, label-regimes, sparse-labels, noisy-labels]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# Exp 2.9 — Label Regimes: GRPO vs DPO

## Goal
Compare GRPO and DPO robustness under three label availability conditions: full labels, sparse labels (10%), and noisy labels (10% flipped).

## Setup
- Methods: [[grpo]], [[dpo]] (both on [[llama-3-8b]])
- Dataset: [[gsm8k]] (primary)
- Label conditions:
  - **Full:** 100% of training prompts have verifiable reward
  - **Sparse:** Only 10% of prompts have any correct rollout (rest yield reward=0 for all samples)
  - **Noisy:** 10% of reward labels randomly flipped (0→1 or 1→0)
- Metrics: accuracy per condition, training stability, convergence speed, **calibration of DPO's implicit reward** `r̂(s,a) = β log(π_θ(a|s)/π_ref(a|s))` (correlation with true correctness), distributional coverage of training vs evaluation completions.

## Theoretical Prediction ([[arora2026-rlvr-foundations]] §3.3, Remark 2.4)
- **GRPO more robust to label noise:** group normalization is self-contained within each batch, so a few flipped labels perturb `r̄_G` by `O(1/G)` but don't bias downstream policy updates systematically.
- **GRPO more robust to sparse labels:** any correct/incorrect label directly provides a gradient signal via the group baseline.
- **DPO's implicit reward should degrade first:** noisy preference pairs erode the `log(π/π_ref)` alignment with the true reward, breaking the connection to `π*`. Offline coverage failure compounds this under sparse labels.

## Aggregate Results
_(populated after runs complete)_

| Method | Full Labels | Sparse (10%) | Noisy (10% flip) |
|--------|------------|--------------|-----------------|
| GRPO | | | |
| DPO | | | |

## Per-Run Pages
_(created as runs complete)_

## Key Findings
_(populated after runs complete)_

## Open Questions
- Does DPO's offline nature make it more or less robust to sparse labels than GRPO's online sampling?
- Under noisy labels, does GRPO's group normalization provide implicit noise robustness?
