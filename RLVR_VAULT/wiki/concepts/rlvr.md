---
type: concept
tags: [rlvr, reward-signal, verifiable-reward, rl]
sources: []
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

## Connections
- [[verifiable-reward]] — the reward computation mechanism
- [[ppo]], [[grpo]], [[dpo]] — all use RLVR as reward signal
- [[gsm8k]], [[humaneval]] — the two verifiable benchmarks

## Key Sources
_(populated on paper ingest)_
