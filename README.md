# RLVR-Comparison: GRPO vs. PPO vs. DPO on Verifiable Rewards

**Course:** Machine Learning: Learning Theory (JHU, Spring 2026)  
**Instructor:** Raman Arora  
**Authors:** Tomoya Furutani, Madhumitha Rajaprakash, Kiran Shay, Andrew Gilbert  

---

## Overview

This repository implements and empirically compares three post-training methods for language
models under verifiable reward settings:

| Method | Advantage source | Critic | On-policy |
|--------|-----------------|--------|-----------|
| **PPO** | Learned critic $V_\phi(s)$ | Yes (4 capacities) | Yes |
| **GRPO** | Group-mean baseline over $G$ rollouts | No | Yes |
| **DPO** | Implicit log-ratio from preference pairs | No | No (offline) |

All three methods optimize the same KL-regularized reward maximization objective but differ in
how they estimate the advantage signal. The project tests three theoretical predictions from
Arora (2026):

- **E2.7** – Head-to-head accuracy, stability, and convergence under matched compute budgets.
- **E2.8** – PPO critic capacity sweep to locate the bias–variance crossover with GRPO.
- **E2.9** – Robustness to label noise and sparsity (verifiable rewards vs. preference pairs).

Primary task: **GSM8K** grade-school math reasoning with binary verifiable rewards.  
Secondary task: **Verifiable JSON extraction** (DPO, Task 1).

---

## Repository Structure

```
RLVR-Comparison/
│
├── src/                        # Shared data, rewards, and utilities
│   ├── data.py                 # get_experiment_subset(), format_prompt_with_template()
│   ├── rewards.py              # gsm8k_reward(), trl_reward_fn(), batch_reward()
│   ├── dpo_pairs.py            # Synthetic preference-pair construction
│   └── critic.py               # Shared critic utilities
│
├── eval/
│   └── metrics.py              # ExperimentLogger, accuracy(), reward_variance(),
│                               #   compute_mc_advantage(), advantage_estimation_error()
│
├── ppo_specs/                  # PPO implementation (Experiments E2.7, E2.8)
│   ├── config.py               # PPOConfig dataclass + preset factories
│   ├── critic.py               # Four critic architectures + build_critic()
│   ├── advantage.py            # Advantage computation + MC estimation
│   ├── ppo_trainer.py          # PPOTrainer: rollouts → PPO-clip update → eval
│   ├── checkpoint.py           # Atomic checkpoint save/load + signal handling
│   ├── utils.py                # cycle_batch(), setup_mc_baselines()
│   ├── run_e2_7.py             # E2.7 head-to-head (PPO portion)
│   ├── run_e2_8.py             # E2.8 critic capacity sweep
│   ├── reward_model.py         # Optional learned reward model wrapper
│   ├── tests/                  # Full test suite (33+ tests)
│   ├── specs/                  # Design specs and code-review findings
│   └── results/                # Output JSON files from completed runs
│
├── grpo_specs/                 # GRPO implementation (Experiments E2.7, E2.8, E2.9)
│   └── grpo_trainer.py         # TRL GRPOTrainer wrapper + ComputeBudget,
│                               #   GRPOCallback, compute_epsilon_v()
│
├── dpo_specs/                  # DPO implementation (Experiments E2.7, E2.9, Task 1)
│   ├── dpo_trainer.py          # DPOTrainer (local / small-scale)
│   ├── run_e2_7.py             # E2.7 DPO head-to-head
│   ├── DPO.ipynb               # Jupyter notebook: Task 1 JSON extraction
│   ├── dpo_plot_results.py     # Result plotting
│   ├── dpo_cluster/
│   │   ├── dpo_full_experiment.py   # Full 3-seed E2.7 + E2.9 script (cluster)
│   │   ├── dpo_full_experiment.sh   # SLURM submission script
│   │   ├── dpo_train.py             # Single-run training script
│   │   └── dpo_train.sh             # SLURM single-run script
│   └── dpo_results/            # Saved checkpoints, JSON results, and plots
│       ├── dpo_all_results.json
│       └── plots/
│           ├── fig1_e27_convergence.png
│           ├── fig2_e29_label_regimes.png
│           └── fig3_e27_stability.png
│
├── shared/
│   └── per_token_loss.py       # Per-token PPO loss (TRL/InstructGPT convention)
│
├── configs/
│   ├── ppo.yaml
│   ├── accelerate_single_gpu.yaml
│   ├── accelerate_multi_gpu.yaml
│   └── accelerate_cpu.yaml
│
├── scripts/
│   ├── setup_env.sh            # Conda environment setup
│   ├── slurm_e2_7.sh           # SLURM: E2.7 PPO (Bernese cluster)
│   ├── slurm_e2_8.sh           # SLURM: E2.8 critic sweep
│   ├── run_full_gpu.sh         # Non-SLURM interactive GPU run
│   └── run_no_slurm.sh         # CPU smoke test
│
├── RLVR_VAULT/                 # Obsidian knowledge base
│   └── wiki/
│       ├── concepts/           # grpo.md, ppo.md, dpo.md, advantage-estimation.md, ...
│       ├── experiments/        # exp-2.7/, exp-2.8/, exp-2.9/ overviews
│       └── sources/            # Annotated paper summaries
│
├── notebooks/
│   └── ppo_self_judge_experiments.ipynb
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Environment Setup

### Requirements

- Python 3.11
- PyTorch ≥ 2.2 with CUDA 12.x (for GPU runs)
- HuggingFace `transformers`, `datasets`, `trl`, `peft`, `accelerate`
- `bitsandbytes` (for QLoRA / DPO cluster runs)

### Install

```bash
# Clone the repository
git clone <repo-url>
cd RLVR-Comparison

# Option A: automated setup (creates conda env 'rlvr', installs torch + deps)
bash scripts/setup_env.sh

# Option B: manual
conda create -n rlvr python=3.11 -y
conda activate rlvr
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Pre-download model weights to avoid downloading during a GPU job:

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
"

python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')"
```

### HuggingFace access (Llama-3-8B)

```bash
huggingface-cli login   # paste your HF token
python -c "
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
"
```

---

## Reproducing Experiments

### Smoke Tests (CPU, ~5 min, no GPU required)

```bash
conda activate rlvr

# PPO smoke test (5 steps, 4 prompts/step)
python ppo_specs/run_e2_7.py --local-test --no-mc

# PPO E2.8 smoke (5 steps, 2 critic capacities)
python ppo_specs/run_e2_8.py --local-test

# GRPO smoke test
python grpo_specs/grpo_trainer.py --seeds 0 --G 4 --completion_budget 200

# DPO smoke test
python dpo_specs/run_e2_7.py --smoke
```

Expected output files:
```
ppo_specs/results/ppo_local_test.json
ppo_specs/results/ppo_e2_8_none_seed42.json
ppo_specs/results/ppo_e2_8_small_seed42.json
```

---

### E2.7 — Head-to-Head Comparison

#### PPO (Qwen2.5-0.5B, Bernese cluster)

PPO experiments were run on the **Bernese cluster** using the provided SLURM scripts.

```bash
# 3-seed sweep (submit as array)
sbatch --export=ALL,SLURM_MODE=array --array=0-2 scripts/slurm_e2_7.sh

# Or sequentially
for seed in 0 1 2; do
    python ppo_specs/run_e2_7.py --seed $seed
done
```

Results: `ppo_specs/results/ppo_e2_7_seed{0,1,2}.json`

#### GRPO (Qwen2.5-0.5B via TRL, G=8)

```bash
python grpo_specs/grpo_trainer.py --seeds 0 1 2 --G 8 --completion_budget 8000
# Results → results/e2_7/grpo/grpo_full_summary.json
```

#### DPO on GSM8K (Llama-3-8B + QLoRA — requires A100)

*JHU Rockfish cluster:*
```bash
sbatch dpo_specs/dpo_cluster/dpo_full_experiment.sh
```

*Local (if GPU available):*
```bash
python dpo_specs/dpo_cluster/dpo_full_experiment.py
# Results → dpo_specs/dpo_results/dpo_all_results.json
```

---

### E2.8 — Critic Quality Sweep

Sweeps four critic capacities: `none` (REINFORCE), `small` (~230K params),
`medium` (~900 params), `large` (~5M params).

```bash
# All four capacities, seed 0 (sequential)
python ppo_specs/run_e2_8.py --seed 0

# One capacity at a time
python ppo_specs/run_e2_8.py --capacity medium --seed 0

# SLURM: parallel, one job per capacity (Bernese cluster)
sbatch --export=ALL,SLURM_MODE=parallel --array=0-3 scripts/slurm_e2_8.sh
```

Results: `ppo_specs/results/ppo_e2_8_{none,small,medium,large}_seed0.json`

---

### E2.9 — Label Regime Comparison

#### GRPO under full / sparse / noisy rewards

```bash
python grpo_specs/grpo_trainer.py --label_regime full   --seeds 0 1 2
python grpo_specs/grpo_trainer.py --label_regime sparse --seeds 0 1 2
python grpo_specs/grpo_trainer.py --label_regime noisy  --seeds 0 1 2
```

#### DPO under full / sparse / noisy preference pairs

Included automatically in the full DPO run above, or run for a single regime:

```bash
# E2.9a: full labels
python dpo_specs/dpo_cluster/dpo_full_experiment.py --mode full --seed 42

# E2.9b: sparse labels (10%)
python dpo_specs/dpo_cluster/dpo_full_experiment.py --mode sparse --seed 42

# E2.9c: noisy labels (10% flipped)
python dpo_specs/dpo_cluster/dpo_full_experiment.py --mode noisy --seed 42
```

---

### Task 1 — DPO Verifiable JSON Extraction

Interactive notebook (Qwen1.5-0.5B + LoRA, 50 steps, CPU/single GPU):

```bash
jupyter notebook dpo_specs/DPO.ipynb
```

Or run headless:

```bash
jupyter nbconvert --to notebook --execute dpo_specs/DPO.ipynb \
    --output dpo_specs/DPO_executed.ipynb
```

---

## Cluster Details

### Compute Environments

| Experiment | Cluster | Hardware | Script |
|------------|---------|----------|--------|
| PPO E2.7, E2.8 | **Bernese cluster** | 1× A100 40 GB | `scripts/slurm_e2_7.sh`, `slurm_e2_8.sh` |
| DPO E2.7, E2.9 | JHU Rockfish | A100 GPU partition | `dpo_specs/dpo_cluster/dpo_full_experiment.sh` |
| GRPO E2.7–E2.9 | JHU Rockfish / local | 1× GPU | `grpo_specs/grpo_trainer.py` |

### Checkpointing and Resume (PPO)

Checkpoints are saved atomically every 20 steps to
`results/checkpoints/ppo_e2_7_seed{N}/`. On SLURM preemption (`SIGTERM`), the current
step completes and a checkpoint is written before exit (via `GracefulExitHandler`).

```bash
# Resume from latest checkpoint
python ppo_specs/run_e2_7.py --seed 0 --resume-from auto

# Resume from specific checkpoint
python ppo_specs/run_e2_7.py --seed 0 \
    --resume-from results/checkpoints/ppo_e2_7_seed0/checkpoint_step_000100
```

### Switching to Llama-3-8B (PPO)

In `ppo_specs/config.py`:
```python
model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
```
And enable gradient checkpointing in the config:
```python
cfg.gradient_checkpointing = True   # required for 8B+ to avoid OOM
```

### GPU Requirements

| Model | Min GPU | Recommended |
|-------|---------|-------------|
| Qwen2.5-0.5B | 1× RTX 3090 (24 GB) | 1× A100 40 GB |
| Llama-3-8B | 1× A100 80 GB | 2× A100 80 GB |
| Llama-3-8B + LoRA (DPO) | 1× A100 40 GB | 1× A100 80 GB |

---

## Key Hyperparameters

### PPO (`ppo_specs/config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | Swap to Llama-3-8B for cluster |
| `critic_capacity` | `"medium"` | `"none"` / `"small"` / `"medium"` / `"large"` |
| `clip_epsilon` | 0.2 | PPO surrogate clipping bound |
| `n_ppo_epochs` | 4 | Gradient updates per rollout batch |
| `kl_coeff` | 0.01 | Reference-KL penalty weight |
| `batch_size` | 16 | Prompts per training step |
| `learning_rate` | 5e-6 | AdamW |
| `n_steps` | 200 | Total training steps |
| `torch_dtype` | `"auto"` | bf16 on GPU, fp32 on CPU |
| `gradient_checkpointing` | `False` | Set `True` for 8B+ models |

### GRPO (`grpo_specs/grpo_trainer.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `G` | 8 | Group size; sweep {4, 8, 16} for E2.8 |
| `learning_rate` | 1e-5 | |
| `beta` | 0.04 | KL penalty vs. frozen reference |
| `epsilon` | 0.2 | PPO-clip bound |
| `batch_size` | 4 | Prompts per step |
| `completion_budget` | 8000 | Total completions (E2.7 shared budget) |

### DPO (`dpo_specs/dpo_cluster/dpo_full_experiment.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MODEL_ID` | `meta-llama/Meta-Llama-3-8B` | |
| `MAX_STEPS` | 200 | |
| `beta` | 0.1 | DPO temperature |
| `learning_rate` | 5e-5 | cosine scheduler |
| LoRA `r` / `alpha` | 16 / 32 | QLoRA, 4-bit NF4 quantization |
| `TRAIN_SAMPLES` | 2000 | |
| `EVAL_SAMPLES` | 200 | |

---

## Results Summary

### E2.7 — PPO (Qwen2.5-0.5B, GSM8K, Bernese cluster)

| Metric | Value |
|--------|-------|
| Mean test accuracy | 0.42 |
| Best test accuracy | 0.49 (step 20) |
| Final test accuracy | 0.38 (step 180) |
| Mean format compliance | 0.85 |
| Final advantage error $|\hat{A} - A_{\mathrm{MC}}|$ | ≈ 1.04 |
| Mean clip fraction | 0.20 |
| Total rollouts | 3,136 |

### E2.7 — DPO (Llama-3-8B, GSM8K, 3 seeds, JHU Rockfish)

| Metric | Value |
|--------|-------|
| Mean final reward margin | 18.59 |
| Reward margin variance (mean ± std) | 28.40 ± 3.68 |
| Train loss (mean) | 0.0283 |

### E2.9 — DPO Label Regime (seed 42)

| Regime | Final Margin | Variance | Train Loss |
|--------|-------------|----------|------------|
| Full (2000 pairs) | 18.86 | 28.22 | 0.0282 |
| Sparse (10%) | 15.56 | 23.12 | 0.0272 |
| Noisy (10% flip) | **2.15** | **0.45** | **0.3133** |

### E2.8 — GRPO Advantage Error vs. Group Size

| G | ε_v |
|---|-----|
| 4 | ≈ 0.11 |
| 8 | ≈ 0.11 (lower variance) |
| 16 | ≈ 0.04 |

### Task 1 — DPO JSON Extraction (Qwen1.5-0.5B + LoRA)

| Metric | Value |
|--------|-------|
| JSON parse accuracy | 94.0% (47/50) |
| Final train / eval loss | 0.029 / 0.051 |
| Reward margin Δ | +1.7815 |

---

## Running Tests

```bash
# Fast tests only (no model download required)
python -m pytest ppo_specs/tests/ -v -m "not slow"

# Full test suite
python -m pytest ppo_specs/tests/ -v

# Individual test files
python -m pytest ppo_specs/tests/test_training_logic.py   # 33 PPO correctness tests
python -m pytest ppo_specs/tests/test_scaling.py          # 31 scaling/memory tests
python -m pytest ppo_specs/tests/test_advantage.py        # Advantage computation
python -m pytest ppo_specs/tests/test_checkpoint.py       # Checkpoint save/load/resume
```

---

## Known Issues

| ID | Description | Status |
|----|-------------|--------|
| KL-bug | Reference-KL estimator drifts to large negative values (impossible for KL ≥ 0); omitted from plots | Open |
| L12 | `format_prompt` uses plain text, not chat template | Open |
| L13 | Reward "last number" fallback can match intermediate calculations | Open |
| P7 | No multi-GPU / Accelerate FSDP support | Open |

See `ppo_specs/specs/` for the full set of code-review findings and fix specifications.

---

## Code Organization Notes

- **Shared infrastructure** (`src/`, `eval/`) is imported by all three trainers.
- **`grpo_specs/STALE/`** contains an earlier custom GRPO implementation; the active code is
  `grpo_specs/grpo_trainer.py`, which wraps TRL's `GRPOTrainer`.
- **`RLVR_VAULT/`** is an [Obsidian](https://obsidian.md) knowledge base with concept notes,
  experiment overviews, and annotated paper summaries. Open the vault folder in Obsidian, or
  read the markdown files directly.
- DPO result plots are pre-generated in `dpo_specs/dpo_results/plots/`.

---

## Citation

If you use this code, please cite the course project specification and the key upstream works:

```
Arora, R. (2026). Foundations of RL with Verifiable Rewards. JHU course notes.

Shao et al. (2024). DeepSeekMath. arXiv:2402.03300.
Schulman et al. (2017). Proximal Policy Optimization. arXiv:1707.06347.
Rafailov et al. (2023). Direct Preference Optimization. NeurIPS 2023.
Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.
```