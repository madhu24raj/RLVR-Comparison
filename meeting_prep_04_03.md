# RLVR-Comparison Meeting Prep — April 3, 2026

## Overview

The project foundation is in place. I built the shared infrastructure that all three experiments (E2.7, E2.8, E2.9) depend on: data loading, reward computation, DPO preference pair construction, and the evaluation/logging framework. Everything is tested and ready for the trainer implementations.

---

## What I Built

### 1. Data Loading (`src/data.py`)

**Purpose:** Load and preprocess GSM8K for all experiments.

**Functions:**

- `extract_answer(answer_text)` — Parses GSM8K's native `#### <number>` format to pull out the ground-truth answer string.

- `load_gsm8k(split, n_samples, seed)` — Loads the GSM8K dataset from HuggingFace, extracts ground-truth answers for each question, and optionally samples a random subset with a fixed seed for reproducibility.

- `format_prompt(question, system_prompt)` — Wraps a GSM8K question into a chat-style prompt compatible with Qwen and Llama tokenizers. Uses a system prompt that instructs the model to show work and end with `#### <number>`.

- `get_experiment_subset(n=100, seed=42)` — Returns the standard 100-prompt training subset and the full test set. This is the function all experiments should call to ensure they're using the exact same data. The seed is fixed at 42 for reproducibility across all runs.
- TF: is this function deleted?

**Design decisions:**
- 100-prompt train subset as specified in the project plan
- Seed=42 hardcoded to guarantee all team members get identical data splits
- Chat formatting is model-agnostic (works with both Qwen and Llama)

---

### 2. Reward Functions (`src/rewards.py`)

**Purpose:** Score model completions with binary rewards (correct = 1.0, incorrect = 0.0).

**Functions:**

- `extract_answer_from_completion(completion)` — Robust multi-format parser that handles:
  - `#### 42` (GSM8K native format)
  - `\boxed{42}` (LaTeX format some models use)
  - `"The answer is 42"` (natural language)
  - Falls back to extracting the last number in the completion
  - Returns `None` if no number is found

- `normalize_number(s)` — Safely converts a string to float, handles commas (e.g., "1,234" -> 1234.0). Returns `None` on failure.

- `gsm8k_reward(completion, ground_truth)` — The core reward function. Extracts the predicted answer from the completion, normalizes both predicted and ground truth, compares with 1e-6 tolerance for floating point issues. Returns 1.0 for correct, 0.0 for incorrect or any parsing failure.

- `batch_reward(completions, ground_truths)` — Vectorized version of `gsm8k_reward` for processing multiple completions at once.

- `trl_reward_fn(completions, ground_truth, **kwargs)` — Wrapper specifically designed to plug into TRL's GRPOTrainer. Matches the function signature TRL expects, making integration straightforward.

**Design decisions:**
- Binary rewards (not soft/partial credit) per the project spec
- Robust extraction handles multiple answer formats because different models format answers differently
- 1e-6 tolerance for float comparison prevents false negatives from rounding
- TRL wrapper means the GRPO trainer can use this directly without adaptation

---

### 3. DPO Preference Pair Construction (`src/dpo_pairs.py`)

**Purpose:** Convert binary reward signals into preference pairs (chosen/rejected) for DPO training, since DPO requires paired comparisons rather than scalar rewards.

**Key components:**

- `PreferencePair` dataclass — Holds `(prompt, chosen, rejected)` where `chosen` is a correct completion (reward=1) and `rejected` is an incorrect completion (reward=0).

- `construct_pairs_from_batch(prompts, completions, rewards, strategy, seed)` — Core pairing logic:
  1. Groups all completions by their source prompt
  2. Separates correct (reward=1.0) and incorrect (reward=0.0) completions for each prompt
  3. Applies pairing strategy:
     - `"random"`: Each correct completion paired with one randomly selected incorrect completion. Produces fewer pairs, less redundancy.
     - `"all"`: Every correct completion paired with every incorrect completion (combinatorial). Produces more training data but with redundancy.
  4. Skips any prompt that has only correct or only incorrect completions (can't form a pair)

- `pairs_to_dataset(pairs)` — Converts a list of `PreferencePair` objects into the dict format TRL's DPOTrainer expects: `{"prompt": [...], "chosen": [...], "rejected": [...]}`.

**Design decisions:**
- Two strategies gives us flexibility for E2.9 experiments
- Filtering out prompts with no contrast prevents degenerate training pairs
- Output format directly compatible with TRL DPOTrainer

**This module is critical for E2.9 (label regimes):** When we test sparse labels (10%) or noisy labels (10% flipped), the pair construction will naturally handle it since it operates on whatever rewards it receives.

---

### 4. Evaluation & Logging (`eval/metrics.py`)

**Purpose:** Track all metrics across experiments and produce comparison plots.

**Metric functions:**

- `accuracy(rewards)` — Fraction of completions with reward=1.0. Used for final test accuracy (primary metric).

- `reward_variance(rewards_per_step)` — Takes a list of reward lists (one per training step) and computes the variance at each step. Tracks training stability over time.

- `compute_mc_advantage(rewards_per_prompt)` — Computes the Monte Carlo baseline (mean reward per prompt) for advantage estimation. Used in E2.8 to compare PPO's critic-based advantages vs. GRPO's group-based advantages.

- `advantage_estimation_error(estimated, mc_advantages)` — Mean absolute error between estimated advantages (from critic or group) and Monte Carlo ground-truth advantages. Key metric for E2.8.

**ExperimentLogger class:**

- `__init__(experiment_name, method, config)` — Creates a logger for a specific run (e.g., "exp_2_7_ppo_seed0").
- `log_step(step, **metrics)` — Appends a metrics dict for a given training step. Call this every N steps during training.
- `save()` — Writes the full log to `results/{experiment_name}.json`.
- `load()` — Reads a previously saved log for analysis/plotting.

**Plotting functions:**

- `plot_convergence(logs, metric)` — Takes multiple ExperimentLogger outputs and plots convergence curves. PPO=blue, GRPO=green, DPO=orange (colors predefined). Supports any metric name.

- `plot_advantage_error(errors, group_sizes)` — Plots advantage estimation error vs. GRPO group size, with a theoretical O(1/sqrt(G)) reference line for comparison.

**Tested:** `results/test_run.json` contains a 10-step dummy run confirming the logger saves and loads correctly. Accuracy ramps from 0.3 to 0.75 in the test data.

---

### 5. Dependencies (`requirements.txt`)

```
trl>=0.12.0          # PPO, GRPO, DPO trainers
transformers>=4.45.0 # Model loading, tokenization
torch                # Core ML framework
accelerate           # Multi-GPU / mixed precision
datasets             # HuggingFace dataset loading
scipy                # Statistical computations
numpy                # Numerical operations
matplotlib           # Plotting
wandb                # Experiment tracking (optional)
```

---

## What's Not Built Yet

### Critical: Trainer Implementations

| Component | Status | Owner (TBD) | Notes |
|-----------|--------|-------------|-------|
| PPO trainer | Not started | — | Needs modular critic network (MLP with configurable hidden sizes). TRL's PPOTrainer as base. |
| GRPO trainer | Not started | — | TRL's GRPOTrainer + `trl_reward_fn` from rewards.py. Need to configure group size G. |
| DPO trainer | Not started | — | TRL's DPOTrainer + `pairs_to_dataset` from dpo_pairs.py. |
| PPO critic module | Not started | — | Custom `nn.Module` with variable hidden layers for E2.8 sweep. |

### Critical: Experiment Scripts

| Script | Status | Experiment |
|--------|--------|-----------|
| `scripts/train_exp_2_7.py` | Not started | Head-to-head: PPO vs GRPO vs DPO, 3 seeds each |
| `scripts/train_exp_2_8.py` | Not started | Critic sweep: PPO with varying critic sizes vs GRPO |
| `scripts/train_exp_2_9.py` | Not started | Label regimes: full, sparse (10%), noisy (10% flipped) |

### Other Missing Pieces

- **HumanEval data loader** — Only GSM8K is implemented. Need `src/data.py` extended with HumanEval loading + code execution reward function.
- **Config files** — `configs/` directory is empty. Need YAML/JSON configs for hyperparameters (learning rate, batch size, group size G, critic sizes, etc.).
- **Matched-compute enforcement** — Need logic to ensure PPO, GRPO, and DPO all use equivalent compute (same number of forward passes / FLOPs).
- **Multi-seed aggregation** — Need a script to load results from 3 seeds per method and compute mean/std for final tables and plots.

---

## Suggested Task Division

### E2.7 Lead (Head-to-Head)
- Build GRPO trainer using TRL GRPOTrainer + `trl_reward_fn`
- Set up central experiment logging with `ExperimentLogger`
- Write `scripts/train_exp_2_7.py` runner (3 methods x 3 seeds = 9 runs)
- Define hyperparameter configs

### E2.8 Lead (PPO + Critic Sweep)
- Build PPO trainer using TRL PPOTrainer
- Implement custom critic module (`nn.Module`) with configurable hidden sizes (e.g., [64], [128, 64], [256, 128, 64])
- Write `scripts/train_exp_2_8.py` runner
- Implement Monte Carlo advantage computation and comparison logic

### E2.9 Lead (DPO + Label Regimes)
- Build DPO trainer using TRL DPOTrainer + `pairs_to_dataset`
- Implement label perturbation:
  - Sparse: randomly keep only 10% of reward labels
  - Noisy: flip 10% of reward labels (1->0, 0->1)
- Write `scripts/train_exp_2_9.py` runner
- Compare GRPO vs DPO under degraded labels

---

## Questions for the Team

1. **Model choice** — Are we going with Qwen 2.5 7B as the base model, or Llama 3 8B? The data loader supports both but we need to lock this in.
2. **Compute resources** — Where are we running training? Colab Pro, university cluster, or someone's GPU? PPO with a critic will be the most memory-intensive.
3. **Timeline** — Can we get trainers implemented by next week? The infrastructure is ready to go.
4. **HumanEval priority** — Is HumanEval in scope for the first round, or should we focus on GSM8K only and add HumanEval later?
5. **Wandb** — Are we using wandb for logging, or just the JSON-based ExperimentLogger I built? Wandb would make it easier to share results across the team.
