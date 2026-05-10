"""
grpo_trl.py
───────────
GRPO trainer for E2.7 / E2.9, built on top of the shared project
infrastructure described in the April 3 meeting notes.

Depends on:
  src/data.py       — get_experiment_subset(), format_prompt_with_template()
  src/rewards.py    — trl_reward_fn, batch_reward
  eval/metrics.py   — ExperimentLogger, accuracy, reward_variance,
                      compute_mc_advantage, advantage_estimation_error

Key data.py facts that shape this file:
  - get_experiment_subset() returns HuggingFace Dataset objects (not lists).
    Columns: 'question', 'answer', 'ground_truth'.
  - 'ground_truth' is already the extracted number string (e.g. "7"),
    extracted by data.py's load_gsm8k() via .map(). Do NOT re-extract.
  - format_prompt_with_template(question, tokenizer) is the correct call
    for chat-template-aware formatting. format_prompt() is the plain fallback.
  - The system prompt requests \\boxed{} (not ####) because Qwen has a
    strong prior toward LaTeX. rewards.py accepts both formats.

Key rewards.py facts:
  - trl_reward_fn(completions, ground_truth, **kwargs) -> list[float]
    Passes straight to batch_reward -> gsm8k_reward.
  - gsm8k_reward expects ground_truth as a bare number string ("7"),
    NOT the full GSM8K answer ("She has #### 7"). Already handled by data.py.
  - Strict extraction: no "last number" fallback. Completions that don't
    use ####, \\boxed{}, or "the answer is" get reward 0 (see rewards.py L13).

TRL's GRPOTrainer handles:
  - G completions per prompt
  - Group-normalised advantage computation
  - PPO-clip surrogate + KL penalty against frozen ref model
  - Mixed precision, gradient clipping

We add:
  - Plugging into the shared reward / data / logging interfaces
  - Compute budget enforcement (matched completion count for E2.7)
  - All E2.7 metrics: stability, convergence speed, εV
  - Label regime support (full / sparse / noisy) for E2.9

Usage:
    python grpo_trl.py --seeds 0 1 2 --G 8 --completion_budget 8000
    python grpo_trl.py --label_regime noisy --seeds 0 1 2   # E2.9
    python grpo_trl.py --G 4 --output_dir results/e2_8/grpo_G4  # E2.8 sweep

    # 2.7: grpo_trl.py --G 8 --completion_budget 800 --output_dir results/e2_7/grpo
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ── Shared project infrastructure ─────────────────────────────────────────────
from src.data import get_experiment_subset, format_prompt_with_template
from src.rewards import trl_reward_fn, batch_reward
from eval.metrics import (
    ExperimentLogger,
    accuracy,
    reward_variance,
    compute_mc_advantage,
    advantage_estimation_error,
)

# ── Constants ─────────────────────────────────────────────────────────────────
# TODO: confirm model choice
MODEL_NAME    = "Qwen/Qwen2.5-0.5B-Instruct"
LABEL_REGIMES = ("full", "sparse", "noisy")

SPARSE_KEEP_FRACTION = 0.10   # E2.9: keep 10% of reward labels
NOISY_FLIP_PROB      = 0.10   # E2.9: flip 10% of reward labels


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Label-regime reward wrappers
#     Wrap trl_reward_fn from src/rewards.py. Regime logic lives here so
#     shared infrastructure stays clean.
# ══════════════════════════════════════════════════════════════════════════════

def make_sparse_reward_fn(seed: int = 42) -> Callable:
    """
    Sparse regime (E2.9): keep only SPARSE_KEEP_FRACTION of reward labels.
    Masked examples always receive 0.0, regardless of correctness.
    Mask is fixed per seed for reproducibility.
    """
    rng = random.Random(seed)

    def sparse_fn(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
        rewards = trl_reward_fn(completions, ground_truth, **kwargs)
        return [r if rng.random() < SPARSE_KEEP_FRACTION else 0.0 for r in rewards]

    return sparse_fn


def make_noisy_reward_fn(seed: int = 42) -> Callable:
    """
    Noisy regime (E2.9): flip NOISY_FLIP_PROB fraction of reward labels.
    1.0 -> 0.0 and 0.0 -> 1.0 with probability NOISY_FLIP_PROB.
    """
    rng = random.Random(seed)

    def noisy_fn(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
        rewards = trl_reward_fn(completions, ground_truth, **kwargs)
        return [(1.0 - r) if rng.random() < NOISY_FLIP_PROB else r for r in rewards]

    return noisy_fn


def get_reward_fn(label_regime: str, seed: int) -> Callable:
    """Return the appropriate TRL-compatible reward function for the regime."""
    if label_regime == "sparse":
        return make_sparse_reward_fn(seed=seed)
    elif label_regime == "noisy":
        return make_noisy_reward_fn(seed=seed)
    else:
        return trl_reward_fn   # full regime: use shared fn directly


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Dataset preparation
#
#     get_experiment_subset() returns HuggingFace Dataset objects with columns:
#       'question'      — raw question string
#       'answer'        — full GSM8K answer string (includes chain of thought)
#       'ground_truth'  — already-extracted number string, e.g. "7"
#
#     We use 'ground_truth' directly — do NOT call extract_answer() again.
#     TRL's GRPOTrainer passes dataset columns as **kwargs to the reward fn,
#     so the 'ground_truth' column flows through to trl_reward_fn automatically.
# ══════════════════════════════════════════════════════════════════════════════

def build_trl_dataset(hf_dataset: Dataset, tokenizer) -> Dataset:
    """
    Convert the shared HF Dataset into the format TRL's GRPOTrainer expects.

    TRL requires a 'prompt' column. We keep 'ground_truth' so TRL passes it
    to the reward function as a kwarg. The 'answer' column is dropped — we
    only need ground_truth (already extracted).

    Uses format_prompt_with_template() so the system prompt requests \\boxed{}
    format, which Qwen produces more reliably than #### (see data.py comments).
    """
    def _format(example):
        return {
            "prompt":       format_prompt_with_template(example["question"], tokenizer=tokenizer),
            "ground_truth": example["ground_truth"],  # already a bare number string
        }

    return hf_dataset.map(_format, remove_columns=hf_dataset.column_names)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Compute budget
#     Shared across methods in E2.7 — same instance passed to PPO, GRPO, DPO.
#     Completion counts per method per prompt:
#       PPO:  1  (single rollout; critic not counted per spec)
#       GRPO: G
#       DPO:  2  (one positive, one negative)
# ══════════════════════════════════════════════════════════════════════════════

class ComputeBudget:
    """
    Tracks total completions generated and enforces a shared budget.
    Pass the same instance to all three trainers for matched-compute E2.7.
    """

    def __init__(self, total_completions: int):
        self.total = total_completions
        self._used = 0
        self._log: List[Tuple[int, str]] = []

    def charge(self, n: int, method: str = "grpo"):
        self._used += n
        self._log.append((n, method))

    def exhausted(self) -> bool:
        return self._used >= self.total

    @property
    def used(self) -> int:
        return self._used

    def fraction_used(self) -> float:
        return self._used / self.total

    def summary(self) -> Dict:
        by_method: Dict[str, int] = {}
        for n, m in self._log:
            by_method[m] = by_method.get(m, 0) + n
        return {
            "total_used":   self._used,
            "total_budget": self.total,
            "by_method":    by_method,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Advantage estimation error  εV
#
#     Uses compute_mc_advantage() and advantage_estimation_error() from
#     eval/metrics.py for consistency with PPO's εV computation.
#
#     ground_truth passed to batch_reward() must be bare number strings —
#     which they are, since we pull from the 'ground_truth' column that
#     data.py already extracted.
# ══════════════════════════════════════════════════════════════════════════════

def compute_epsilon_v(
    model,
    tokenizer,
    eval_dataset: Dataset,         # HF Dataset with 'question' + 'ground_truth'
    G_grpo: int,
    G_oracle: int = 32,
    n_prompts: int = 30,
    device: str = "cuda",
) -> float:
    """
    εV = E[ MAE(A_grpo, A_oracle) ]

    A_oracle: G_oracle samples -> approximates true V(s) = E[r|s]
    A_grpo:   G_grpo   samples -> what GRPO actually uses as baseline

    Uses compute_mc_advantage() and advantage_estimation_error() from
    eval/metrics.py so the metric is computed identically across all methods.
    """
    model.eval()
    eps = 1e-6
    all_estimated: List[float] = []
    all_mc:        List[float] = []

    n = min(n_prompts, len(eval_dataset))

    with torch.no_grad():
        for i in range(n):
            example = eval_dataset[i]
            # format_prompt_with_template for consistent prompt format
            prompt = format_prompt_with_template(example["question"], tokenizer=tokenizer)
            gt     = example["ground_truth"]   # already a bare number string

            def _sample_rewards(G: int) -> List[float]:
                enc = tokenizer(
                    [prompt] * G,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(device)
                out = model.generate(
                    **enc,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
                pl = enc["attention_mask"].sum(dim=1)
                completions = [
                    tokenizer.decode(out[j][pl[j]:], skip_special_tokens=True)
                    for j in range(G)
                ]
                # batch_reward from src/rewards.py; gt is already extracted
                return batch_reward(completions, [gt] * G)

            # Oracle baseline
            r_oracle = _sample_rewards(G_oracle)
            mu_oracle = np.mean(r_oracle)
            mc_adv = [r - mu_oracle for r in r_oracle]   

            # GRPO group baseline
            r_grpo   = _sample_rewards(G_grpo)
            mu_grpo  = np.mean(r_grpo)
            grpo_adv = [r - mu_grpo for r in r_grpo]

            k = min(len(grpo_adv), len(mc_adv))
            all_estimated.extend(grpo_adv[:k])
            all_mc.extend(mc_adv[:k])

    # advantage_estimation_error() from eval/metrics.py
    return advantage_estimation_error(all_estimated, all_mc)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRL callback — budget enforcement + logging
# ══════════════════════════════════════════════════════════════════════════════

class GRPOCallback(TrainerCallback):
    """
    Hooks into TRL's training loop to:
      1. Charge the compute budget after each step (batch_size x G completions)
      2. Log metrics to ExperimentLogger (eval/metrics.py)
      3. Run periodic greedy-decode evaluation
      4. Stop training when the compute budget is exhausted
    """

    def __init__(
        self,
        budget:       ComputeBudget,
        logger:       ExperimentLogger,
        G:            int,
        batch_size:   int,
        eval_dataset: Dataset,       # HF Dataset with 'question' + 'ground_truth'
        tokenizer,
        device:       str,
        eval_every:   int = 50,
    ):
        self.budget       = budget
        self.logger       = logger
        self.G            = G
        self.batch_size   = batch_size
        self.eval_dataset = eval_dataset
        self.tokenizer    = tokenizer
        self.device       = device
        self.eval_every   = eval_every

        # Rolling history for stability metrics
        self._rewards: List[float] = []
        self._losses:  List[float] = []

    def on_step_end(self, args, state, control, **kwargs):
        # ── Charge budget: batch_size prompts x G completions ─────────────────
        self.budget.charge(self.batch_size * self.G, method="grpo")

        # ── Pull metrics from TRL's log history ───────────────────────────────
        logs   = state.log_history[-1] if state.log_history else {}
        reward = logs.get("reward",        logs.get("mean_reward", 0.0))
        loss   = logs.get("loss",          logs.get("policy_loss", 0.0))
        kl     = logs.get("kl_divergence", logs.get("kl",          0.0))

        self._rewards.append(reward)
        self._losses.append(loss)

        # ── Log to shared ExperimentLogger ────────────────────────────────────
        self.logger.log_step(
            step=state.global_step,
            reward=reward,
            loss=loss,
            kl=kl,
            clip_fraction=logs.get("clip_fraction", 0.0),
            budget_used=self.budget.used,
            budget_fraction=self.budget.fraction_used(),
        )

        # ── Periodic evaluation ───────────────────────────────────────────────
        if state.global_step % self.eval_every == 0:
            model    = kwargs.get("model")
            eval_acc = self._run_eval(model)
            self.logger.log_step(step=state.global_step, eval_accuracy=eval_acc)
            print(
                f"  step {state.global_step:4d} | "
                f"budget {self.budget.used}/{self.budget.total} "
                f"({self.budget.fraction_used():.1%}) | "
                f"eval_acc={eval_acc:.4f} | kl={kl:.4f}"
            )

        # ── Stop when budget exhausted ────────────────────────────────────────
        if self.budget.exhausted():
            print(f"\n[BUDGET] Exhausted at step {state.global_step}. Stopping.")
            control.should_training_stop = True

    def _run_eval(self, model, n_eval: int = 100, batch_size: int = 16) -> float:
        """
        Greedy-decode accuracy on the first n_eval examples of eval_dataset.

        Prompts are formatted with format_prompt_with_template() for
        consistency. ground_truth is pulled from the 'ground_truth' column
        (already a bare number string — no re-extraction needed).
        """
        if model is None:
            return 0.0
        model.eval()
        rewards_all: List[float] = []
        n = min(n_eval, len(self.eval_dataset))

        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = self.eval_dataset.select(range(start, min(start + batch_size, n)))
                prompts = [
                    format_prompt_with_template(ex["question"], tokenizer=self.tokenizer)
                    for ex in batch
                ]
                gts = [ex["ground_truth"] for ex in batch]  # already extracted

                enc = self.tokenizer(
                    prompts, return_tensors="pt",
                    truncation=True, max_length=512, padding=True,
                ).to(self.device)
                out = model.generate(
                    **enc,
                    max_new_tokens=256,
                    do_sample=False,   # greedy for deterministic eval
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                pl = enc["attention_mask"].sum(dim=1)
                completions = [
                    self.tokenizer.decode(out[i][pl[i]:], skip_special_tokens=True)
                    for i in range(len(prompts))
                ]
                rewards_all.extend(batch_reward(completions, gts))

        model.train()
        return accuracy(rewards_all)   # eval/metrics.py

    def stability_metrics(self, window: int = 50) -> Dict[str, float]:
        """
        Training stability over last `window` steps.
        """
        r = self._rewards[-window:]
        l = self._losses[-window:]
        # r_var = r_var = float(np.var(self._rewards[-window:])) # TODO: double-check reward variance (within time stamp or across time stamp)
        return {
            "reward_mean":          float(np.mean(r)) if r else 0.0,
            "reward_std":           float(np.std(r))  if r else 0.0,
            "loss_mean":            float(np.mean(l)) if l else 0.0,
            "loss_std":             float(np.std(l))  if l else 0.0,
            # "reward_variance_mean": float(np.mean(r_var)) if r_var else 0.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRL GRPOConfig builder
# ══════════════════════════════════════════════════════════════════════════════

def build_grpo_config(args, seed: int) -> GRPOConfig:
    return GRPOConfig(
        output_dir=os.path.join(args.output_dir, f"seed{seed}"),

        # GRPO group size — sweep {4, 8, 16} for E2.8
        num_generations=args.G,

        # Generation
        max_prompt_length=512,
        max_completion_length=512,
        temperature=0.7,
        top_p=0.9,

        # PPO-clip
        epsilon=0.2,

        # KL against frozen reference model
        beta=0.04,

        # Optimiser
        learning_rate=1e-5,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,

        # Set high — budget callback controls early stopping
        num_train_epochs=100,

        logging_steps=args.log_every,
        report_to="none",
        seed=seed,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,   # keep ground_truth column for reward fn
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Single-seed run
# ══════════════════════════════════════════════════════════════════════════════

def run_single_seed(args, seed: int, budget: ComputeBudget) -> Dict:
    """Full GRPO training run for one seed. Returns all E2.7 / E2.9 metrics."""

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*64}")
    print(f"  GRPO | seed={seed} | G={args.G} | regime={args.label_regime}")
    print(f"{'='*64}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Data ──────────────────────────────────────────────────────────────────
    # get_experiment_subset() returns HF Datasets with 'ground_truth' already
    # extracted (bare number strings). seed=42 is fixed inside; all methods
    # see identical splits.
    train_hf, test_hf = get_experiment_subset(n=100, seed=42)

    train_ds = build_trl_dataset(train_hf, tokenizer)
    eval_ds  = build_trl_dataset(test_hf,  tokenizer)

    # ── Reward function — regime-aware ────────────────────────────────────────
    reward_fn = get_reward_fn(args.label_regime, seed=seed)

    # ── ExperimentLogger ──────────────────────────────────────────────────────
    # Naming convention: exp_2_7_grpo_seed0 / exp_2_9_grpo_noisy_seed0
    exp_name = f"exp_2_7_grpo_seed{seed}"
    if args.label_regime != "full":
        exp_name = f"exp_2_9_grpo_{args.label_regime}_seed{seed}"

    logger = ExperimentLogger(
        experiment_name=exp_name,
        output_dir=args.output_dir,
    )

    # ── TRL GRPOTrainer ───────────────────────────────────────────────────────
    grpo_cfg = build_grpo_config(args, seed)
    trainer  = GRPOTrainer(
        model=args.model,
        config=grpo_cfg,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # ── Callback — budget enforcement + metric logging ────────────────────────
    callback = GRPOCallback(
        budget=budget,
        logger=logger,
        G=args.G,
        batch_size=args.batch_size,
        eval_dataset=test_hf,     # pass original HF Dataset (has 'question' col)
        tokenizer=tokenizer,
        device=device,
        eval_every=args.eval_every,
    )
    trainer.add_callback(callback)

    # ── Train ─────────────────────────────────────────────────────────────────
    t_start = time.time()
    trainer.train()
    wall_time = time.time() - t_start

    # ── Final accuracy ────────────────────────────────────────────────────────
    final_acc = callback._run_eval(trainer.model, n_eval=100)
    print(f"\n  Final accuracy: {final_acc:.4f}")

    # ── εV — advantage estimation error ──────────────────────────────────────
    print("  Computing εV (advantage estimation error)...")
    eps_v = compute_epsilon_v(
        model=trainer.model,
        tokenizer=tokenizer,
        eval_dataset=test_hf,    # original HF Dataset with 'question' column
        G_grpo=args.G,
        G_oracle=32,
        n_prompts=30,
        device=device,
    )
    print(f"  εV = {eps_v:.6f}")

    # ── Stability metrics ─────────────────────────────────────────────────────
    stability = callback.stability_metrics(window=50)

    # ── Save via ExperimentLogger ─────────────────────────────────────────────
    logger.log_step(
        step=-1,   # sentinel: final summary entry
        final_accuracy=final_acc,
        advantage_error_eps_v=eps_v,
        wall_time_s=wall_time,
        **stability,
        **budget.summary(),
    )
    logger.save()

    return {
        "seed":         seed,
        "label_regime": args.label_regime,
        "G":            args.G,
        "final": {
            "eval_accuracy":         final_acc,
            "advantage_error_eps_v": eps_v,
            "loss_std":              stability["loss_std"],
            "reward_std":            stability["reward_std"],
            "reward_mean":           stability["reward_mean"],
            "wall_time_s":           wall_time,
            "total_completions":     budget.used,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Multi-seed aggregation
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(results: List[Dict]) -> Dict:
    """Mean +/- std across seeds for all scalar final metrics."""
    keys    = list(results[0]["final"].keys())
    summary = {"n_seeds": len(results), "seeds": [r["seed"] for r in results]}
    for k in keys:
        vals = [r["final"][k] for r in results if r["final"].get(k) is not None]
        if vals and isinstance(vals[0], (int, float)):
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"]  = float(np.std(vals))
    return summary


def print_summary(summary: Dict, args):
    print(f"\n{'='*64}")
    print(f"  GRPO | G={args.G} | regime={args.label_regime} | {summary['n_seeds']} seeds")
    print(f"{'='*64}")
    rows = [
        ("eval_accuracy",         "Accuracy"),
        ("loss_std",              "Loss std (stability)"),
        ("reward_std",            "Reward std (stability)"),
        ("advantage_error_eps_v", "εV (advantage error)"),
    ]
    for key, label in rows:
        mu  = summary.get(f"{key}_mean")
        std = summary.get(f"{key}_std")
        if mu is not None:
            print(f"  {label:<30s}: {mu:.4f} +/- {std:.4f}")
    print(f"{'='*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="GRPO trainer for E2.7 / E2.8 / E2.9")
    p.add_argument("--model",      default=MODEL_NAME)
    p.add_argument("--seeds",      nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--G",          type=int,  default=8,
                   help="GRPO group size — sweep {4, 8, 16} for E2.8")
    p.add_argument("--completion_budget", type=int, default=8000,
                   help="Total completions budget shared with PPO / DPO (E2.7)")
    p.add_argument("--batch_size",  type=int, default=4,
                   help="Prompts per step; effective completions = batch_size x G")
    p.add_argument("--eval_every",  type=int, default=50)
    p.add_argument("--log_every",   type=int, default=10)
    p.add_argument("--label_regime", choices=LABEL_REGIMES, default="full",
                   help="Label regime for E2.9 (full / sparse / noisy)")
    p.add_argument("--output_dir",  default="results/e2_7/grpo")
    return p.parse_args()


def main():
    args    = parse_args()
    results = []
    for seed in args.seeds:
        # Each seed gets its own budget — seeds are independent runs
        budget = ComputeBudget(total_completions=args.completion_budget)
        result = run_single_seed(args, seed=seed, budget=budget)
        results.append(result)

    summary = aggregate(results)
    print_summary(summary, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"grpo_{args.label_regime}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()