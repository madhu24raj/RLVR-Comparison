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
├── checkpoint.py      ← Checkpoint save/load, signal handling, resume
├── utils.py           ← Shared helpers (cycle_batch, setup_mc_baselines)
│
├── run_e2_7.py        ← E2.7 head-to-head experiment (PPO portion)
├── run_e2_8.py        ← E2.8 critic quality sweep
│
├── tests/             ← Test suite (see Tests section below)
│
└── specs/             ← Fix specifications from code review (see below)
    ├── readability.md
    ├── safety.md
    ├── logic.md
    ├── performance.md
    ├── batched_generation.md
    ├── memory_optimization.md
    ├── checkpointing.md
    ├── cluster_deployment.md
    └── distributed.md
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
            ──► checkpoint.py  (save/load/resume, signal handling)
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

### Tests

```
ppo_specs/tests/
├── test_trainer.py          ← Core PPO trainer tests
├── test_advantage.py        ← Advantage computation and critic tests
├── test_data_rewards.py     ← Data loading and reward function tests
├── test_batched_ops.py      ← Batched generation correctness tests
├── test_checkpoint.py       ← Checkpoint save/load/resume tests
├── test_cluster_features.py ← Dtype, gradient checkpointing tests
└── test_e2e_pipeline.py     ← End-to-end integration tests
```

Run all tests:
```bash
python -m pytest ppo_specs/tests/ -v -m "not slow"   # fast tests only
python -m pytest ppo_specs/tests/ -v                   # all tests (needs model download)
```

---

## Cluster Deployment

The codebase now supports running on GPU clusters out of the box. Key features:

### Dtype selection

Models load in bfloat16 on GPU by default (`torch_dtype: "auto"`). Override via config:
```python
cfg = e2_7_config(seed=0)
cfg.torch_dtype = "bfloat16"   # explicit bf16
cfg.torch_dtype = "float32"    # force fp32 (CPU or debugging)
```
Or leave as `"auto"` (default) for automatic detection: bfloat16 on CUDA, float32 on CPU.

### Gradient checkpointing

Required for 8B+ models to avoid OOM. Enable in config:
```python
cfg.gradient_checkpointing = True
```
Uses `use_reentrant=False` for compatibility with torch.compile and multi-GPU.

### Checkpoint and resume

Checkpoints are saved atomically (write to temp dir, then rename) and rotated
to keep the last K versions:
```python
cfg.checkpoint_every = 20      # save every 20 steps
cfg.keep_checkpoints = 3       # keep last 3
cfg.checkpoint_dir = "results/checkpoints"
```

To resume from a checkpoint:
```python
cfg.resume_from = "auto"       # find latest checkpoint
cfg.resume_from = "results/checkpoints/ppo_e2_7_seed0/checkpoint_step_000100"  # specific
```

Full state is restored: model weights, critic, optimizers, RNG states, logger, and
training counters. Config compatibility is verified via hash on resume.

### Signal handling for SLURM preemption

The training loop uses `GracefulExitHandler` to catch SIGTERM/SIGINT. When a
signal is received, the current step completes and a checkpoint is saved before
exit. This integrates with SLURM's `--signal=TERM@60` preemption notification.

### SLURM scripts

See `scripts/` for example SLURM submission scripts with checkpoint/resume support.

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
| `torch_dtype` | `"auto"` | `"auto"` / `"float32"` / `"bfloat16"` -- auto = bf16 on GPU |
| `gradient_checkpointing` | `False` | Enable gradient checkpointing (required for 8B+) |
| `checkpoint_every` | `20` | Save checkpoint every N steps (0 = disabled) |
| `keep_checkpoints` | `3` | Keep last K checkpoints (0 = keep all) |
| `checkpoint_dir` | `"results/checkpoints"` | Directory for checkpoint storage |
| `resume_from` | `""` | Checkpoint path, or `"auto"` for latest |

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
  "kl_divergence":   float,   # KL(pi_old || pi_new), step-level divergence
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

## Running on a Compute Cluster

### Step 1: Environment setup

On the cluster's login node, run the setup script to create the conda environment,
install PyTorch with CUDA, and pre-download model weights:

```bash
# Basic setup (Qwen2.5-0.5B for development)
bash scripts/setup_env.sh

# Include Llama-3-8B weights (requires HuggingFace access token)
bash scripts/setup_env.sh --large-model
```

If you don't have the setup script or prefer manual setup:

```bash
conda create -n rlvr python=3.11 -y
conda activate rlvr
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Pre-download model weights (avoids downloading during a GPU job)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
  AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# Pre-download GSM8K dataset
python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')"
```

### Step 2: Configure the model

For cluster runs with Llama-3-8B, edit `ppo_specs/config.py`:

```python
model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
```

Or override it in the preset functions. The dtype and gradient checkpointing
are handled automatically (bfloat16 on GPU, gradient checkpointing when enabled).

### Step 3: Submit jobs via SLURM

SLURM scripts are in `scripts/`. They handle environment activation, model caching,
NCCL tuning, preemption signal handling, and checkpoint/resume.

```bash
# ── E2.7: Head-to-Head ───────────────────────────────────────────────

# Single GPU smoke test
sbatch scripts/slurm_e2_7.sh

# Single GPU, full run, seed 0
sbatch --export=ALL,SEED=0 scripts/slurm_e2_7.sh

# 3-seed sweep as a SLURM job array (runs seeds 42, 123, 456 in parallel)
sbatch --export=ALL,SLURM_MODE=array --array=0-2 scripts/slurm_e2_7.sh

# Multi-GPU (4x A100), required for 8B models
sbatch --export=ALL,SLURM_MODE=multigpu,NUM_GPUS=4 --gres=gpu:4 scripts/slurm_e2_7.sh

# ── E2.8: Critic Sweep ──────────────────────────────────────────────

# All 4 critic capacities sequentially
sbatch scripts/slurm_e2_8.sh

# One capacity per SLURM task (parallel)
sbatch --export=ALL,SLURM_MODE=parallel --array=0-3 scripts/slurm_e2_8.sh

# Single capacity
sbatch --export=ALL,CAPACITY=medium,SEED=0 scripts/slurm_e2_8.sh
```

Override cluster-specific settings at submission time:

```bash
sbatch -p your_partition -A your_account \
  --export=ALL,SEED=0,MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
  scripts/slurm_e2_7.sh
```

### Step 4: Checkpoint and resume

Checkpoints are saved automatically every N steps (default 20). If a job is
preempted or crashes, resume from the latest checkpoint:

```bash
# Auto-resume: finds the latest checkpoint for this experiment
python ppo_specs/run_e2_7.py --seed 0 --resume-from auto

# Resume from a specific checkpoint
python ppo_specs/run_e2_7.py --seed 0 \
  --resume-from results/checkpoints/ppo_e2_7_seed0/checkpoint_step_000100
```

The SLURM scripts handle preemption automatically via SIGUSR1 trapping.
When SLURM sends the preemption signal, the current training step completes,
a checkpoint is saved, and the job requeues itself.

### Step 5: Monitor training

Logs are written to `logs/` and results to `results/`:

```bash
# Check job status
squeue -u $USER

# Tail the output log
tail -f logs/e2_7_<jobid>_0.out

# Check saved results
cat results/ppo_e2_7_seed0.json | python -m json.tool | head -20

# List checkpoints
ls results/checkpoints/ppo_e2_7_seed0/
```

### Running without SLURM

If your cluster uses a different scheduler or you're running interactively:

```bash
# Interactive GPU session
srun --gres=gpu:1 --mem=64G --time=4:00:00 --pty bash

# Then run directly
conda activate rlvr
cd /path/to/RLVR-Comparison
python ppo_specs/run_e2_7.py --seed 0 --checkpoint-every 20
```

For 8B models, enable gradient checkpointing in `config.py`:

```python
cfg = e2_7_config(seed=0)
cfg.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
cfg.gradient_checkpointing = True  # required to avoid OOM
# torch_dtype="auto" already selects bfloat16 on GPU
```

### GPU requirements

| Model | Min GPU | Recommended | Config changes needed |
|-------|---------|-------------|----------------------|
| Qwen2.5-0.5B | 1x RTX 3090 (24GB) | 1x A100 40GB | None (default) |
| Llama-3-8B | 1x A100 80GB | 2x A100 80GB | `gradient_checkpointing=True` |
| Llama-3-8B + LoRA | 1x A100 40GB | 1x A100 80GB | See `specs/memory_optimization.md` |

### Implementation status

| Feature | Status |
|---------|--------|
| Batched generation (left-padding) | Implemented |
| Auto bfloat16 on GPU | Implemented |
| Gradient checkpointing | Implemented |
| Checkpoint save/load/resume | Implemented |
| SIGTERM/SIGINT graceful exit | Implemented |
| SLURM job scripts | Provided |
| Multi-GPU via Accelerate/FSDP | Open (see `specs/distributed.md`) |
| Wandb integration | Spec only (see `specs/checkpointing.md`) |

---

## Recent Fixes

### 2026-04-08 -- Cluster readiness

Distributed computing and cluster deployment features:

| File | Change | Category |
|------|--------|----------|
| `ppo_trainer.py` | Batched generation with left-padding (fixes P1) | Performance |
| `ppo_trainer.py` | Batched `_policy_log_probs` and `_critic_forward` (partial P2 fix) | Performance |
| `ppo_trainer.py` | Batched `evaluate()` with configurable sub-batch size | Performance |
| `ppo_trainer.py` | Auto bfloat16 on GPU, fp32 on CPU (fixes P8) | Performance |
| `ppo_trainer.py` | `log_softmax` always in float32 for numerical stability (fixes S14) | Safety |
| `ppo_trainer.py` | Gradient checkpointing support (fixes P6) | Memory |
| `ppo_trainer.py` | Empty response shape fix (fixes S4) | Safety |
| `ppo_trainer.py` | Batched critic/log-prob eliminates GPU memory leaks (fixes S5, S6, S7) | Safety |
| `ppo_trainer.py` | Attention mask excludes padding from log-probs (fixes S13) | Safety |
| `advantage.py` | Batched MC rollouts with `repeat()` (fixes P4) | Performance |
| `run_e2_8.py` | Distinct if/else for critic eval (fixes R7) | Readability |
| `config.py` | New fields: `torch_dtype`, `gradient_checkpointing`, `checkpoint_every`, `keep_checkpoints`, `checkpoint_dir`, `resume_from` | Config |
| `checkpoint.py` | New module: atomic checkpoint save/load, rotation, signal handling | Reliability |
| `run_e2_7.py` | Checkpoint integration with `GracefulExitHandler` and resume | Reliability |
| `ppo_trainer.py` | `kl_divergence` added to metrics dict | Metrics |

### 2026-04-07 -- Data pipeline and bug fixes

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

| Spec | ID | Summary | Severity | Status |
|------|----|---------|----------|--------|
| `logic.md` | L12 | `format_prompt` uses plain text, not chat template | Medium | Open |
| `logic.md` | L13 | Reward "last number" fallback can match intermediate calculations | Low | Open |
| `logic.md` | L15 | `n_eval=20` during training makes convergence curves noisy | Low | Open |
| `performance.md` | P2 | Separate policy/critic forward passes (partially batched) | Moderate | Partial |
| `performance.md` | P7 | No multi-GPU / accelerate support | Cluster Blocker | Open |
| `performance.md` | P9 | Generation logits discarded and recomputed | Moderate | Open |

Previously critical cluster blockers P1, P6, and P8 are now fixed.

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
