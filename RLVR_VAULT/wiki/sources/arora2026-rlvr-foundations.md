---
type: source
tags: [ppo, grpo, dpo, rlvr, theory, bias-variance, sample-complexity, project-brief]
sources: []
updated: 2026-04-15
---

# Foundations of RL with Verifiable Rewards: Comparing GRPO with PPO and DPO (Arora, 2026)

**Citation:** Arora, R. (2026). Foundations of RL with Verifiable Rewards: Comparing Group Relative Policy Optimization with PPO and DPO. Project brief, Feb 27 2026.
**Raw file:** `../../raw/papers/Project-7-GRPO-vs-DPO-vs-PPO.pdf`

## Summary
Theoretical comparison of PPO, GRPO, and DPO as three estimators of the same KL-regularized reward-maximization optimum `π* ∝ π_ref · exp(r/β)`. Establishes the bias–variance profile of each method, derives comparative sample complexity bounds, and specifies the three experiments (E2.7, E2.8, E2.9) that structure this project. Functions as the canonical project brief.

## Key Claims
- **Unified objective:** All three methods optimize `J_β(π) = E[r] − β·KL(π ‖ π_ref)` with the same closed-form maximizer. They differ only in how they estimate the advantage/reward gradient. — eq. (1)–(2).
- **GRPO advantage numerator variance:** `Var[r_i − r̄_G] = σ*²(1 − 1/G)` exact for all `G ≥ 2`. Larger G *increases* numerator variance toward σ*², but *decreases* baseline estimation error at rate `O(1/√G)`. — Theorem 2.1.
- **GRPO bias:** 0 under population normalization (denominator `σ*(s)`); heuristic `O(1/G)` under empirical normalization (denominator `σ_{r,G}`). "Bias" here refers to `E[Â_i]` (baseline estimator), not unbiasedness w.r.t. the true Q-advantage. — eq. (12)–(13).
- **PPO advantage bias floor:** `|E[Â^PPO] − A*| ≤ ε_V/(1−γ)`, irreducible when the critic class cannot represent `V^π`. Variance `O(σ*²_A) ≤ O(σ²_R)`. — eq. (14)–(15).
- **Sample complexity (informal, Prop 2.3):** `n_PPO = O((d_V + C_π)/ε² + ε_V/((1−γ)ε))`; `n_GRPO = O(C_π/ε²)` **independent of G** (G-fold variance reduction cancels G-fold rollout cost); `n_DPO = O(d_π/ε²)` assuming adequate offline coverage (≈ GRPO with G=2 in true completion count).
- **PPO–GRPO crossover (Theorem 2.5 + E2.8 prediction):** `ε*_V ≈ (1−γ)·√(σ*²(1 − 1/G) − σ*²_A)`. Below this critic error PPO wins; above it GRPO wins.
- **GRPO–DPO equivalence (Mroueh 2025):** For `G=2` with binary verifiable rewards, GRPO normalized advantages reduce to `Â⁺ = +1, Â⁻ = −1` and the gradient becomes `∇ log(π(o⁺)/π(o⁻))` — the DPO gradient direction. Structural equivalence, but GRPO is on-policy while DPO is offline.
- **DPO offline limitation:** Requires paired comparisons; for verifiable RLVR tasks pairs must be synthesized from scalar rewards. Offline coverage failure is the main empirical weakness (Xiong et al. 2024; Kim et al. 2026).
- **Per-step compute ordering:** PPO = 2× (critic); GRPO = G× (rollouts); DPO = 2× (π_ref forward pass). GRPO cheaper per-step than DPO when `G ≤ 4`.

## Relevance to This Project
This *is* the project brief. It defines the theoretical framing and specifies all three experiments:
- **E2.7** — head-to-head at matched compute on GSM8K + HumanEval. Predicts GRPO > PPO on binary-RLVR (sparse reward, hard-to-train critic) and > offline DPO (on-policy coverage).
- **E2.8** — critic capacity sweep {none, small, medium, large} × G ∈ {4, 8, 16}. Verify the crossover formula `ε*_V`.
- **E2.9** — label regimes (full / sparse 10% / noisy 10%). Predicts GRPO more robust to label noise and sparsity than DPO.

Every concept and experiment page in this vault inherits its theoretical framing from this source.

## Connections
- [[ppo]] — critic-based; bias floor `ε_V/(1−γ)` derived here
- [[grpo]] — group-baseline; variance theorem derived here
- [[dpo]] — offline preference-pair method; equivalence to GRPO@G=2 shown
- [[rlvr]] — project's reward paradigm
- [[advantage-estimation]] — central axis of comparison
- [[critic-network]] — source of PPO's irreducible bias; swept in E2.8
- [[kl-penalty]] — shared regularizer across all three methods
- [[preference-pairs]] — synthetic construction required for DPO on verifiable tasks
- [[verifiable-reward]] — binary reward assumption in Theorem 2.5
- [[exp-2.7-overview]], [[exp-2.8-overview]], [[exp-2.9-overview]] — experiments defined in §3
