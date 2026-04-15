---
type: entity
tags: [humaneval, benchmark, code, verifiable-reward]
sources: []
updated: 2026-04-15
---

# HumanEval

164 hand-crafted Python programming problems, each with a function signature, docstring, and unit tests.

## Key Properties
- Size: 164 problems
- Task: complete a Python function given its signature and docstring
- Verification: execute generated code against provided unit tests
- Standard metric: pass@k

## In This Project

Secondary code benchmark. Used in exp 2.7 (head-to-head across PPO/GRPO/DPO). Verifiable rewards from code execution (pass@1 used as RL training signal).

## Connections
- [[verifiable-reward]] — reward from code execution
- [[rlvr]] — paradigm this dataset enables
- [[ppo]], [[grpo]], [[dpo]] — evaluated here in exp 2.7

## Key Sources
_(populated on paper ingest)_
