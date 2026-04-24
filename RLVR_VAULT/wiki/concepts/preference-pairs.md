---
type: concept
tags: [preference-pairs, dpo, synthetic-data, rlvr]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# Preference Pairs

(prompt, chosen, rejected) triples required by DPO. In this project, generated synthetically from verifiable rewards rather than human annotation.

## Mechanism

**Synthetic generation pipeline:**
1. For each prompt, sample G rollouts from the current (or SFT) policy
2. Score each rollout with the verifiable reward function
3. Pair highest-reward response (chosen) with lowest-reward response (rejected)
4. Filter pairs where chosen == rejected reward (ambiguous signal)

G (group size) is a hyperparameter affecting pair quality and diversity.

## In This Project

Built as a standalone pipeline. Used to construct the DPO training dataset for exp 2.9. Quality of synthetic pairs is a potential confounder when comparing DPO to GRPO under sparse label conditions.

> [!note] Uncertain: Whether pairs generated under sparse labels (10% of prompts have any correct rollout) provide sufficient coverage for stable DPO training.

## Connections
- [[dpo]] — consumes preference pairs
- [[verifiable-reward]] — scoring mechanism
- [[grpo]] — alternative that doesn't require pairs
- [[rlvr]] — reward source

## Key Sources
- [[arora2026-rlvr-foundations]] — argues that on verifiable tasks, synthetic pair construction is an avoidable detour: GRPO consumes the scalar rewards directly without losing the DPO↔GRPO gradient equivalence
