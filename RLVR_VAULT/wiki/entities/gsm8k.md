---
type: entity
tags: [gsm8k, benchmark, math, verifiable-reward]
sources: []
updated: 2026-04-15
---

# GSM8K

Grade School Math benchmark. 8,500 grade-school math word problems with verifiable numeric answers.

## Key Properties
- Size: 8,500 problems (train: 7,473 / test: 1,319)
- Task: multi-step arithmetic word problems
- Answer format: final numeric value
- Verification: exact match after extracting final answer

## In This Project

Primary math benchmark. Used in exps 2.7, 2.8, and 2.9. Verifiable rewards computed by extracting and comparing the final numeric answer. Sparse label condition in exp 2.9 uses 10% of training prompts.

## Connections
- [[verifiable-reward]] — reward computation on this dataset
- [[rlvr]] — paradigm this dataset enables
- [[ppo]], [[grpo]], [[dpo]] — all trained and evaluated here

## Key Sources
_(populated on paper ingest)_
