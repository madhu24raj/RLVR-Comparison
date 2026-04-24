---
type: concept
tags: [rlvr, reward-signal, verifiable-reward, rl]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# RLVR (Reinforcement Learning from Verifiable Rewards)

RL training paradigm where reward signals are computed by checking output correctness against ground truth — math answers, code execution results — rather than a learned reward model.

## Mechanism

1. Model generates a response to a prompt
2. Response is checked against a verifiable criterion (numeric answer match, code passes unit tests)
3. Binary or scalar reward returned: 1.0 (correct) or 0.0 (incorrect)
4. Reward used directly as training signal

No reward model training required. Reward hacking via reward model overoptimization is eliminated by design.

## In This Project

Unifying reward framework for all three methods. PPO and GRPO use verifiable rewards online (per rollout). DPO uses them offline to construct preference pairs.

**Why RLVR is the regime where GRPO shines** ([[arora2026-rlvr-foundations]], Remark 2.4): verifiable tasks have (i) binary sparse rewards, which make the PPO critic hard to train (large `ε_V`), amplifying the `ε_V/(1−γ)` bias floor; (ii) no natural paired-preference format, forcing DPO to synthesize pairs from scalar rewards and lose its offline-data-efficiency argument. GRPO's group-relative baseline handles both issues directly: no critic to train, no pair construction needed, and the `O(1/√G)` baseline concentration gives a clean accuracy knob.

## Connections
- [[verifiable-reward]] — the reward computation mechanism
- [[ppo]], [[grpo]], [[dpo]] — all use RLVR as reward signal
- [[gsm8k]], [[humaneval]] — the two verifiable benchmarks

## Key Sources
- [[arora2026-rlvr-foundations]] — frames all three methods as estimators of the same KL-regularized optimum and explains why RLVR's binary sparse rewards favor GRPO
