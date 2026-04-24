# RLVR Vault — Activity Log

Append-only. Each entry: `## [YYYY-MM-DD] {type} | {title}`
Types: `ingest`, `query`, `lint`

Parse recent entries: `grep "^## \[" log.md | tail -10`

---

## [2026-04-15] ingest | Wiki initialization — seed pages and schema

## [2026-04-15] ingest | Project Notes — Initial Planning Notes

## [2026-04-15] ingest | Arora 2026 — Foundations of RL with Verifiable Rewards (project brief)
- New: wiki/sources/arora2026-rlvr-foundations.md
- Updated concepts: ppo, grpo, dpo, advantage-estimation, critic-network, rlvr, kl-penalty, preference-pairs, verifiable-reward (all backlinked + theoretical framing integrated)
- Updated experiments: exp-2.7-overview, exp-2.8-overview, exp-2.9-overview (theoretical predictions + crossover formula added)
- Updated entities: llama-3-8b, gsm8k, humaneval (source backlink)
- Index: added arora2026-rlvr-foundations under Sources
- Key theorems captured: Var[r_i − r̄_G] = σ*²(1 − 1/G); PPO bias floor ε_V/(1−γ); ε*_V crossover formula; n_GRPO independent of G; GRPO↔DPO gradient equivalence at G=2
- Contradiction flagged on advantage-estimation.md: clarified that "unbiased given sufficient group size" refers to E[Â] not unbiasedness w.r.t. true A*

