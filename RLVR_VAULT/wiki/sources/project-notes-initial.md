---
type: source
tags: [project-notes, ppo, grpo, dpo, exp-2.7, exp-2.8, exp-2.9, gsm8k, humaneval]
sources: []
updated: 2026-04-15
---

# Project Notes — Initial Planning Notes

**Raw file:** `../../../raw/notes/project-notes.md`
**Date captured:** 2026-04-15

## Summary

Internal planning notes scoping the RLVR comparison project. Defines the three core experiments (2.7, 2.8, 2.9), three alignment methods (PPO, GRPO, DPO), two benchmarks (GSM8K, HumanEval), and task ownership per team member.

## Key Claims
- All three methods built on LLaMA-3 8B for fair comparison
- PPO requires a modular critic; GRPO and DPO do not
- DPO requires a synthetic preference pair pipeline using verifiable rewards
- Exp 2.7: head-to-head, matched compute, 3 seeds — logs accuracy, stability, convergence, advantage estimation error
- Exp 2.8: critic size sweep (none/small/medium/large); generates crossover plot of critic error vs accuracy
- Exp 2.9: GRPO vs DPO under full / sparse (10%) / noisy (10% flipped) label conditions

## Relevance to This Project

Primary project scoping document. Informs experiment page structure, metric tracking requirements, and the DPO preference pair pipeline design.

## Connections
- [[ppo]], [[grpo]], [[dpo]] — the three methods
- [[gsm8k]], [[humaneval]] — the two benchmarks
- [[exp-2.7-overview]], [[exp-2.8-overview]], [[exp-2.9-overview]] — experiment scoping
- [[preference-pairs]] — DPO pipeline requirement called out explicitly
- [[critic-network]] — PPO modular critic requirement
- [[advantage-estimation]] — tracked metric across all experiments
