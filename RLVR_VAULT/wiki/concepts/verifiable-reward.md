---
type: concept
tags: [verifiable-reward, reward-signal, rlvr, gsm8k, humaneval]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# Verifiable Reward

Scalar reward computed by checking model output correctness against ground truth, without a learned reward model.

## Mechanism

**GSM8K:** Extract final numeric answer from model response (regex/parser). Compare to ground truth. Reward = 1.0 if match, 0.0 otherwise.

**HumanEval:** Execute generated Python code against provided unit tests. Reward = fraction of tests passed (or binary pass@1).

No reward model parameters. No reward model training. Reward cannot be hacked via distributional shift of a reward model.

## In This Project

Primary reward signal for PPO and GRPO (online). Used offline to rank rollouts and construct preference pairs for DPO.

## Connections
- [[rlvr]] — the broader paradigm
- [[gsm8k]], [[humaneval]] — verifiable benchmark sources
- [[preference-pairs]] — downstream use for DPO
- [[grpo]] — uses group of verifiable rewards to compute advantages

## Key Sources
- [[arora2026-rlvr-foundations]] — uses the binary-reward `r ∈ {0,1}` setting as the canonical RLVR regime in Theorem 2.5 and the E2.9 label-regime experiment
