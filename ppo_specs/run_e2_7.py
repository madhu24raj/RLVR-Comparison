"""
E2.7 (PPO portion): Head-to-Head Comparison Under Matched Compute.

Measurements collected per the assignment spec:
  (i)  final test accuracy at a fixed total rollout budget
  (ii) training stability  – reward variance per iteration
  (iii) convergence speed  – accuracy vs. rollout count
  (iv) advantage estimation error |Â - A_MC|

Results are saved to:
  results/<experiment_name>.json   (step-level log)

Usage
─────
Local smoke test (5 steps, tiny dataset):
    python ppo_specs/run_e2_7.py --local-test

Full cluster run (200 steps, seed 0):
    python ppo_specs/run_e2_7.py --seed 0

Skip slow MC estimation:
    python ppo_specs/run_e2_7.py --local-test --no-mc
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from src.data import load_gsm8k, format_prompt
from eval.metrics import ExperimentLogger
from ppo_specs.config import PPOConfig, local_test_config, e2_7_config
from ppo_specs.ppo_trainer import load_ppo_trainer
from ppo_specs.advantage import estimate_mc_advantages, advantage_estimation_error


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cycle_batch(items: list, step: int, batch_size: int) -> list:
    """Return a contiguous slice of length batch_size, wrapping around."""
    n = len(items)
    start = (step * batch_size) % n
    if start + batch_size <= n:
        return items[start : start + batch_size]
    # wrap
    return items[start:] + items[: (start + batch_size) - n]


# ── Main experiment ───────────────────────────────────────────────────────────

def run_e2_7(config: PPOConfig, compute_mc: bool = True) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[E2.7] Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("[E2.7] Loading GSM8K …")
    train_ds = load_gsm8k("train", n_samples=config.n_train_samples, seed=config.seed)
    test_ds  = load_gsm8k("test",  n_samples=200)

    train_prompts = [format_prompt(ex["question"]) for ex in train_ds]
    train_gts     = [ex["ground_truth"] for ex in train_ds]
    test_prompts  = [format_prompt(ex["question"]) for ex in test_ds]
    test_gts      = [ex["ground_truth"] for ex in test_ds]

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = load_ppo_trainer(config, device)

    # ── MC baseline (reference for advantage error) ───────────────────────────
    # Use the first 5 training prompts as a fixed reference set.
    # n_samples=10 locally; raise to 1000 on the cluster for paper-quality estimates.
    mc_baselines: dict = {}
    if compute_mc:
        n_mc = 10 if config.n_steps <= 10 else 50
        ref_p  = train_prompts[:5]
        ref_gt = train_gts[:5]
        print(f"[E2.7] Estimating MC baselines ({n_mc} samples × {len(ref_p)} prompts) …")
        mc_baselines = estimate_mc_advantages(
            trainer.model, trainer.tokenizer,
            ref_p, ref_gt, trainer.reward_fn,
            n_samples=n_mc,
            max_new_tokens=config.max_new_tokens,
            device=str(device),
        )
        print("[E2.7] MC baselines:", {k[-30:]: f"{v:.3f}" for k, v in mc_baselines.items()})

    # ── Training loop ─────────────────────────────────────────────────────────
    logger = ExperimentLogger(config.experiment_name, config.output_dir)
    reward_window: list[float] = []  # rolling window for variance (stability)

    for step in range(config.n_steps):
        batch_p  = _cycle_batch(train_prompts, step, config.batch_size)
        batch_gt = _cycle_batch(train_gts,     step, config.batch_size)

        metrics = trainer.train_step(batch_p, batch_gt)
        reward_window.append(metrics["mean_reward"])

        if step % config.log_every == 0:
            print(
                f"  step {step:3d} | reward={metrics['mean_reward']:.3f} "
                f"| acc={metrics['accuracy']:.3f} "
                f"| policy_loss={metrics['policy_loss']:.4f} "
                f"| critic_loss={metrics['critic_loss']:.4f}"
            )

        # ── Periodic evaluation ───────────────────────────────────────────────
        if step % config.eval_every == 0:
            test_acc = trainer.evaluate(test_prompts, test_gts, n_eval=20)

            # (ii) Training stability: variance over the last window
            window = reward_window[-config.eval_every:] if len(reward_window) >= config.eval_every \
                     else reward_window
            stability = float(np.var(window))

            # (iv) Advantage estimation error
            adv_error = None
            if mc_baselines:
                # Current critic baseline estimate: mean reward of this batch
                est = np.full(len(mc_baselines), metrics["mean_reward"])
                mc  = np.array(list(mc_baselines.values()))
                adv_error = advantage_estimation_error(est, mc)

            log_entry: dict = {
                "total_rollouts": metrics["total_rollouts"],   # (iii) x-axis
                "train_accuracy": metrics["accuracy"],
                "test_accuracy":  test_acc,                    # (i)
                "reward_variance": stability,                  # (ii)
                "policy_loss":    metrics["policy_loss"],
                "critic_loss":    metrics["critic_loss"],
                "clip_fraction":  metrics["clip_fraction"],
            }
            if adv_error is not None:
                log_entry["advantage_error"] = adv_error       # (iv)

            logger.log_step(step, **log_entry)
            print(
                f"    → test_acc={test_acc:.3f} "
                f"| stability(var)={stability:.4f}"
                + (f" | adv_error={adv_error:.4f}" if adv_error is not None else "")
            )

    logger.save()

    # ── Final evaluation ──────────────────────────────────────────────────────
    final_acc = trainer.evaluate(test_prompts, test_gts, n_eval=50)
    print(f"\n[E2.7] Final test accuracy (PPO, {config.critic_capacity} critic): {final_acc:.3f}")
    print(f"[E2.7] Log saved to {config.output_dir}/{config.experiment_name}.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2.7: PPO head-to-head on GSM8K")
    parser.add_argument(
        "--local-test", action="store_true",
        help="Run with tiny config to verify the pipeline locally",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-mc", action="store_true",
        help="Skip Monte Carlo advantage estimation (faster)",
    )
    args = parser.parse_args()

    cfg = local_test_config() if args.local_test else e2_7_config(seed=args.seed)
    cfg.seed = args.seed

    run_e2_7(cfg, compute_mc=not args.no_mc)
