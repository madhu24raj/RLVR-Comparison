# ppo_specs — PPO for RLVR

This folder implements **Proximal Policy Optimization (PPO)** as the PPO component
of the RLVR comparison study (Experiments E2.7 and E2.8).  It is one of three
training methods in the project; the others (GRPO, DPO) live in their own modules.

---

## Overview

The project compares three RLVR training methods on GSM8K math reasoning:

| Method | Rollouts per prompt | Baseline type |
|--------|--------------------|-|
| **PPO** ← this folder | 1 (+ critic) | Learned value function |
| GRPO | G = 8 | Group mean of G rollouts |
| DPO | 2 (chosen + rejected) | Reference policy log-ratio |

PPO's key distinction is its **learned critic** (value function). E2.8 sweeps four
critic capacities to locate the crossover where critic approximation error tips the
balance toward GRPO.

---

## Architecture

```
ppo_specs/
│
├── config.py          ← PPOConfig dataclass + experiment presets
├── critic.py          ← Four critic architectures + build_critic() factory
├── advantage.py       ← Advantage computation, MC estimation, error metrics
├── ppo_trainer.py     ← PPOTrainer: rollouts → PPO-clip update → eval
├── utils.py           ← Shared helpers (cycle_batch, setup_mc_baselines)
│
├── run_e2_7.py        ← E2.7 head-to-head experiment (PPO portion)
├── run_e2_8.py        ← E2.8 critic quality sweep
│
└── specs/             ← Fix specifications from code review (see below)
    ├── readability.md
    ├── safety.md
    ├── logic.md
    └── performance.md
```

### Dependency graph

```
config.py
    │
    ▼
ppo_trainer.py ──► critic.py
    │          ──► advantage.py
    │          ──► src/data.py      (data loading)
    │          ──► src/rewards.py   (gsm8k_reward)
    │          ──► eval/metrics.py  (ExperimentLogger, accuracy)
    │
    ▼
run_e2_7.py ──► utils.py
run_e2_8.py ──► utils.py
```

---

## Quick Start

### Local smoke test (no GPU required, ~5 min on CPU)

```bash
# from RLVR-Comparison/
pip install -r requirements.txt

# E2.7 PPO head-to-head: 5 training steps, 4 prompts/step, skip MC estimation
python ppo_specs/run_e2_7.py --local-test --no-mc

# E2.8 critic sweep: 5 steps, 2 critic capacities (none + small)
python ppo_specs/run_e2_8.py --local-test
```

Expected output files:
```
results/ppo_local_test.json
results/ppo_e2_8_none_seed42.json
results/ppo_e2_8_small_seed42.json
```

### Full runs (cluster, Llama-3-8B)

Before running on the cluster, change `model_name` in `config.py`:
```python
model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
```

```bash
# E2.7 — three seeds
for seed in 0 1 2; do
    python ppo_specs/run_e2_7.py --seed $seed
done

# E2.8 — full critic sweep, one seed
python ppo_specs/run_e2_8.py --seed 0

# Single critic capacity
python ppo_specs/run_e2_8.py --capacity medium --seed 0
```

---

## File Reference

### `config.py`

Defines `PPOConfig` (dataclass) and three preset factory functions.

| Function | Purpose |
|----------|---------|
| `local_test_config()` | 5-step smoke test with 20 training samples |
| `e2_7_config(seed)` | E2.7 head-to-head (200 steps, 500 samples) |
| `e2_8_config(capacity, seed)` | One cell of the E2.8 sweep (150 steps) |
| `copy_config(cfg, **overrides)` | Immutable copy with selected fields changed |

Key config fields:

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | Swap to Llama-3-8B for cluster |
| `critic_capacity` | `"medium"` | `"none"` / `"small"` / `"medium"` / `"large"` |
| `clip_epsilon` | `0.2` | PPO surrogate clipping bound |
| `n_ppo_epochs` | `1` | Gradient updates per collected batch |
| `kl_coeff` | `0.0` | KL penalty weight (0 = disabled) |
| `batch_size` | `8` | Prompts per training step |

---

### `critic.py`

Four critic architectures for E2.8. All share the interface:
```python
value: Tensor[B] = critic(hidden_state: Tensor[B, hidden_size])
is_trainable: bool = critic.is_trainable()
```

| Capacity | Class | Architecture | ~Params (0.5B backbone) |
|----------|-------|--------------|------------------------|
| `"none"` | `REINFORCEBaseline` | Returns zeros; trainer uses batch-mean reward | 0 |
| `"small"` | `SmallCriticMLP` | Linear(H,256) → ReLU → Linear(256,1) | ~230 K |
| `"medium"` | `MediumCriticHead` | Linear(H,1) | ~897 |
| `"large"` | `LargeCriticMLP` | Linear(H,2H) → GELU → Linear(2H,2H) → GELU → Linear(2H,1) | ~5 M |

`H` = `model.config.hidden_size` (896 for Qwen2.5-0.5B; 4096 for Llama-3-8B).

---

### `advantage.py`

Three public functions:

| Function | Returns | Used in |
|----------|---------|---------|
| `compute_advantages(rewards, values, gamma, normalize)` | `Tensor[B]` advantages | `ppo_trainer.ppo_update` |
| `estimate_mc_advantages(policy, tokenizer, prompts, gts, reward_fn, n_samples, ...)` | `dict[prompt → mean_reward]` | `utils.setup_mc_baselines` |
| `advantage_estimation_error(est_baselines, mc_baselines)` | `float` MAE | `run_e2_7`, `run_e2_8` |
| `critic_approximation_error(critic_values, mc_baselines)` | `float` RMSE (εV) | `run_e2_8` |

---

### `ppo_trainer.py`

Main training class `PPOTrainer`.  Constructed via `load_ppo_trainer(config, device)`.

**Public methods:**

| Method | Description |
|--------|-------------|
| `generate_rollouts(prompts, gts)` | Generate completions; compute rewards, old log-probs, critic values |
| `ppo_update(batch)` | One PPO-clip + critic-MSE gradient step; returns metrics dict |
| `train_step(prompts, gts)` | `generate_rollouts` + `n_ppo_epochs × ppo_update` |
| `evaluate(prompts, gts, n_eval)` | Greedy decoding accuracy on first `n_eval` examples |
| `_eval_critic_on_prompts(prompts)` | Evaluate `V̂(s)` on reference prompts (needed for E2.7/E2.8 metrics — see `specs/logic.md`) |

Metrics returned by `train_step`:
```python
{
  "policy_loss":     float,
  "critic_loss":     float,
  "mean_reward":     float,
  "reward_variance": float,
  "mean_advantage":  float,
  "clip_fraction":   float,   # fraction of ratios that were clipped
  "accuracy":        float,   # fraction of batch with reward=1
  "total_rollouts":  int,     # cumulative rollout count (x-axis for E2.7 (iii))
}
```

---

### `utils.py`

| Function | Description |
|----------|-------------|
| `cycle_batch(items, step, batch_size)` | Circular slice through a list |
| `setup_mc_baselines(trainer, prompts, gts, n_steps, max_new_tokens, device)` | Estimate MC baselines on first 5 training prompts |

---

### `run_e2_7.py` / `run_e2_8.py`

Experiment entry points. Both accept `--local-test` and `--seed N`.

`run_e2_7.py` measures (per spec):
1. Final test accuracy at fixed rollout budget
2. Training stability (reward variance per iteration)
3. Convergence speed (accuracy vs total rollout count)
4. Advantage estimation error `|Â − A_MC|`

`run_e2_8.py` sweeps critics and measures per capacity:
1. Final accuracy
2. Critic approximation error εV = RMSE(V̂(s), V_MC(s))
3. Advantage bias
4. Sample-efficiency curves

---

## PPO Algorithm

### Single-step PPO for RLVR

Because GSM8K has a **terminal reward** (binary 0/1 at end of generation), each
episode is a single step:

```
state  s = prompt tokens
action a = full generated response
reward r = gsm8k_reward(completion, ground_truth) ∈ {0, 1}
```

GAE reduces to:
```
A_i = r_i − V̂(s_i)
```

For `capacity="none"`, `V̂(s_i)` is replaced by the batch mean `ē = mean(r)`.

### Training loop (one step)

```
1. Generate rollouts
   ├── Tokenise prompts
   ├── model.generate() → completions
   ├── gsm8k_reward()   → binary rewards
   ├── _sequence_log_prob() → old log-probs (stored before update)
   └── critic(last_hidden) → V̂(s)

2. Compute advantages
   └── A = r − V̂  (normalised to zero mean, unit variance)

3. PPO-clip update  (repeated n_ppo_epochs times)
   ├── Recompute log-probs under current policy: log π_θ(a|s)
   ├── ratio ρ = exp(log π_θ − log π_θ_old)
   ├── L_CLIP = −E[min(ρA, clip(ρ, 1−ε, 1+ε)A)]
   ├── L_V    = MSE(V̂(s), r)
   ├── total  = L_CLIP + 0.5 · L_V  [+ kl_coeff · KL]
   └── AdamW step on policy + critic

4. Log metrics
```

---

## Measured Quantities (Experiments)

### E2.7

| Measurement | Metric key | Where computed |
|-------------|------------|----------------|
| (i) Final test accuracy | `test_accuracy` | `PPOTrainer.evaluate()` |
| (ii) Training stability | `reward_variance` | `np.var(reward_history[-window:])` |
| (iii) Convergence speed | `total_rollouts` vs `test_accuracy` | Logged every `eval_every` steps |
| (iv) Advantage error | `advantage_error` | `advantage_estimation_error(V̂, V_MC)` |

### E2.8

| Measurement | Metric key | Where computed |
|-------------|------------|----------------|
| (i) Final accuracy | `final_accuracy` | `PPOTrainer.evaluate()` |
| (ii) Critic error εV | `mean_ev` | `critic_approximation_error(V̂_ref, V_MC_ref)` |
| (iii) Advantage bias | `mean_bias` | `advantage_estimation_error(V̂_ref, V_MC_ref)` |
| (iv) Sample efficiency | `accuracy_curve` | `[(rollouts, acc), …]` per capacity |

---

## Scaling to Cluster

When moving from local to cluster, apply these changes in order:

| Step | Change | File |
|------|--------|------|
| 1 | Change `model_name` to `meta-llama/Meta-Llama-3-8B-Instruct` | `config.py` |
| 2 | Enable bfloat16: auto-select dtype based on device | `ppo_trainer.py:load_ppo_trainer` |
| 3 | Enable gradient checkpointing | `ppo_trainer.py:load_ppo_trainer` |
| 4 | Batch rollout generation (P1 in `specs/performance.md`) | `ppo_trainer.py:generate_rollouts` |
| 5 | Integrate `accelerate` for multi-GPU (P7 in `specs/performance.md`) | `ppo_trainer.py`, run scripts |
| 6 | Raise `n_mc` in `utils.setup_mc_baselines` to 1000 | `utils.py` |

See `specs/performance.md` for the full priority list and code snippets.

---

## Recent Fixes (2026-04-07)

Bugs fixed during the performance and data pipeline review:

| File | Fix | Severity |
|------|-----|----------|
| `src/rewards.py` | L10: `extract_answer_from_completion` now takes the **last** `####` match (was first), consistent with `data.py:extract_answer` | Medium |
| `ppo_trainer.py` | L11: `_sequence_log_prob` returns `torch.zeros(1)` for empty responses (was scalar `0.0`, causing shape mismatch in `torch.stack`) | Medium |

Previously fixed (already in codebase):

| Fix | Summary |
|-----|---------|
| L1 | `n_ppo_epochs` loop implemented with precomputed advantages |
| L2 | `kl_coeff * kl` applied in total loss |
| L3/L4 | `_eval_critic_on_prompts()` used for trainable critics in E2.7/E2.8 |
| L6 | Advantages precomputed once before K-epoch PPO loop |
| L7 | Advantage normalization uses full z-score (mean + std) |
| S1 | Log-ratio clamped to [-20, 20] before `exp()` |
| S3 | Zero-std advantage normalization handled correctly |
| TD-1 | `critic_loss_coeff` configurable (was hardcoded 0.5) |

---

## Known Issues

The `specs/` folder documents issues found by code review.
Items that may affect experimental validity:

| Spec | ID | Summary | Severity |
|------|----|---------|----------|
| `logic.md` | L12 | `format_prompt` uses plain text, not chat template (confound for instruct models) | Medium |
| `logic.md` | L13 | Reward "last number" fallback can match intermediate calculations | Low |
| `logic.md` | L15 | `n_eval=20` during training makes convergence curves noisy | Low |
| `performance.md` | P1 | Per-sample generation: ~10-15x slower than batched | Critical |
| `performance.md` | P2 | Double forward pass per sample in PPO update | Critical |
| `performance.md` | P6 | No gradient checkpointing (OOM for 8B models) | Cluster Blocker |
| `performance.md` | P8 | float32 on GPU (2x slower than bfloat16) | Moderate |
| `performance.md` | P9 | Generation logits discarded and recomputed | Moderate |

Fix P1, P6, and P8 before running on the cluster.

---

## Extending the Code

### Add a new critic architecture

1. Add a class to `ppo_specs/critic.py` inheriting from `nn.Module` with
   `forward(hidden_state)` and `is_trainable()` methods.
2. Add a branch in `build_critic(capacity, hidden_size)`.
3. Add the name to `CRITIC_CAPACITIES` in `config.py`.

### Swap the reward function

`PPOTrainer` accepts any `reward_fn(completion: str, ground_truth: str) -> float`.
Pass a different function to `PPOTrainer(...)` or `load_ppo_trainer`.

### Add a new dataset

1. Add a loader in `src/data.py` returning `(prompts, ground_truths)` lists.
2. Pass a matching reward function when constructing `PPOTrainer`.
3. Update the experiment scripts' data-loading block.
