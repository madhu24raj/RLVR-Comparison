# RLVR Vault — Index

Content-oriented catalog of all wiki pages. Updated on every ingest and after any significant page change.

Parse log: `grep "^## \[" log.md | tail -10`
Search wiki: `./search.sh "query" [subdir]`

---

## Concepts

- [[ppo]] — Proximal Policy Optimization: clipped surrogate objective, requires learned critic, standard in RLHF
- [[grpo]] — Group Relative Policy Optimization: critic-free, normalizes advantages within a response group
- [[dpo]] — Direct Preference Optimization: supervised objective from preference pairs, no RL required
- [[rlvr]] — Reinforcement Learning from Verifiable Rewards: reward from answer correctness, no reward model
- [[advantage-estimation]] — A(s,a) = Q(s,a) - V(s); GAE in PPO vs group normalization in GRPO
- [[kl-penalty]] — KL(π||π_ref) regularization; prevents policy drift from base model
- [[verifiable-reward]] — Binary reward from checking output correctness (numeric match, code execution)
- [[preference-pairs]] — (prompt, chosen, rejected) triples for DPO; synthetically generated via verifiable rewards
- [[critic-network]] — Value network V(s) for PPO; swept across sizes (none/small/medium/large) in exp 2.8

---

## Entities

- [[llama-3-8b]] — Meta LLaMA-3 8B; base policy model for all three methods in this project
- [[gsm8k]] — Grade School Math benchmark; 8,500 problems with verifiable numeric answers
- [[humaneval]] — 164 Python coding problems; verified by unit test execution; pass@k metric

---

## Experiments

- [[exp-2.7-overview]] — Head-to-head: PPO vs GRPO vs DPO on GSM8K + HumanEval, 3 seeds, matched compute
- [[exp-2.8-overview]] — Critic size sweep: PPO (none/small/medium/large) vs GRPO; crossover plot of error vs accuracy
- [[exp-2.9-overview]] — Label regimes: GRPO vs DPO under full / sparse (10%) / noisy (10% flip) labels

---

## Sources

_(populated on first ingest)_

---

## Outputs

_(populated on first query that produces a reusable artifact)_
