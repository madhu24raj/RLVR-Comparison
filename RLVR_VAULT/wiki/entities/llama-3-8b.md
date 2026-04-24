---
type: entity
tags: [llama, meta, base-model, transformer]
sources: [arora2026-rlvr-foundations]
updated: 2026-04-15
---

# LLaMA-3 8B

Meta's LLaMA-3 8B parameter open-weight language model. Base policy model for all three alignment methods in this project.

## Key Properties
- Parameters: 8 billion
- Architecture: transformer decoder, grouped-query attention (GQA), 32 layers
- Context length: 8,192 tokens
- Training: pretrained on ~15T tokens

## In This Project

Base model for PPO, GRPO, and DPO fine-tuning. In PPO: also used as backbone for the critic network (with a value head). All experiments use the same base checkpoint for fair comparison.

## Connections
- [[ppo]], [[grpo]], [[dpo]] — fine-tuning targets
- [[critic-network]] — PPO adds a value head to this model

## Key Sources
- [[arora2026-rlvr-foundations]] — specifies LLaMA-3 8B as the common backbone for all methods in E2.7/2.8/2.9
