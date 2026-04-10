"""
E2.7 (GRPO portion): Head-to-Head Comparison Under Matched Compute.

Measurements:
  (i)   final test accuracy at a fixed total rollout budget
  (ii)  training stability -- reward variance per iteration
  (iii) convergence speed -- accuracy vs. rollout count

Usage:
    python grpo_specs/run_e2_7.py --local-test
    python grpo_specs/run_e2_7.py --seed 0
"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import random

import torch
import numpy as np
from transformers import set_seed as transformers_set_seed

from src.data import load_gsm8k, format_prompt_with_template
from eval.metrics import ExperimentLogger
from grpo_specs.config import GRPOConfig, local_test_config, e2_7_config
from grpo_specs.grpo_trainer import load_grpo_trainer


def cycle_batch(items, step, batch_size):
    """Deterministic cycling through items by step."""
    n = len(items)
    start = (step * batch_size) % n
    end = start + batch_size
    if end <= n:
        return items[start:end]
    return items[start:] + items[:end - n]


def run_e2_7(config: GRPOConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    transformers_set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    print(f"[E2.7-GRPO] Device: {device}")

    # -- Data --
    print("[E2.7-GRPO] Loading GSM8K ...")
    train_ds = load_gsm8k("train", n_samples=config.n_train_samples, seed=config.seed)
    test_ds  = load_gsm8k("test",  n_samples=config.n_test_samples)

    train_gts = [ex["ground_truth"] for ex in train_ds]
    test_gts  = [ex["ground_truth"] for ex in test_ds]

    # -- Trainer --
    trainer = load_grpo_trainer(config, device)

    train_prompts = [
        format_prompt_with_template(ex["question"], trainer.tokenizer) for ex in train_ds
    ]
    test_prompts = [
        format_prompt_with_template(ex["question"], trainer.tokenizer) for ex in test_ds
    ]

    # -- Training loop --
    logger = ExperimentLogger(config.experiment_name, config.output_dir)
    reward_window: list[float] = []

    for step in range(config.n_steps):
        batch_p  = cycle_batch(train_prompts, step, config.batch_size)
        batch_gt = cycle_batch(train_gts,     step, config.batch_size)

        metrics = trainer.train_step(batch_p, batch_gt)
        reward_window.append(metrics["mean_reward"])

        if step % config.log_every == 0:
            print(
                f"  step {step:3d} | reward={metrics['mean_reward']:.3f} "
                f"| acc={metrics['accuracy']:.3f} "
                f"| policy_loss={metrics['policy_loss']:.4f} "
                f"| kl={metrics['kl_divergence']:.4f}"
            )

        if step % config.eval_every == 0:
            test_acc = trainer.evaluate(test_prompts, test_gts, n_eval=config.eval_size)

            window = reward_window[-config.eval_every:] if len(reward_window) >= config.eval_every \
                     else reward_window
            stability = float(np.var(window))

            logger.log_step(step, **{
                "total_rollouts": metrics["total_rollouts"],
                "train_accuracy": metrics["accuracy"],
                "test_accuracy":  test_acc,
                "reward_variance": stability,
                "policy_loss":    metrics["policy_loss"],
                "clip_fraction":  metrics["clip_fraction"],
                "kl_divergence":  metrics["kl_divergence"],
                "kl_ref_divergence": metrics["kl_ref_divergence"],
            })
            logger.save()
            print(
                f"    -> test_acc={test_acc:.3f} "
                f"| stability(var)={stability:.4f}"
            )

    logger.save()

    # -- Final evaluation --
    final_acc = trainer.evaluate(test_prompts, test_gts, n_eval=config.final_eval_size)
    print(f"\n[E2.7-GRPO] Final test accuracy: {final_acc:.3f}")
    print(f"[E2.7-GRPO] Log saved to {config.output_dir}/{config.experiment_name}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2.7: GRPO head-to-head on GSM8K")
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = local_test_config() if args.local_test else e2_7_config(seed=args.seed)
    cfg.seed = args.seed
    run_e2_7(cfg)
