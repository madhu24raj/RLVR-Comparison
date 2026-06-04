"""
GRPO E2.7: Head-to-Head Comparison Under Matched Compute.

From-scratch GRPO trainer (no TRL dependency). Uses the same shared
infrastructure as the PPO trainer for fair comparison.

Usage:
    # Local smoke test
    python grpo_specs/run_grpo.py --local-test

    # Full run with 1.5B on GPU
    python grpo_specs/run_grpo.py --model-name Qwen/Qwen2.5-1.5B-Instruct --seed 42

    # With memory optimizations for constrained GPUs
    python grpo_specs/run_grpo.py --model-name Qwen/Qwen2.5-1.5B-Instruct --seed 42 --gradient-checkpointing
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

from src.tasks import get_task
from eval.metrics import ExperimentLogger
from grpo_specs.STALE.config import GRPOConfig, local_test_config, e2_7_config
from grpo_specs.STALE.grpo_trainer import load_grpo_trainer
from grpo_specs.checkpoint import (
    save_grpo_checkpoint, load_grpo_checkpoint,
    find_latest_checkpoint, restore_rng_states, GracefulExitHandler,
)


def cycle_batch(items, step, batch_size):
    """Deterministic cycling through items by step."""
    n = len(items)
    start = (step * batch_size) % n
    end = start + batch_size
    if end <= n:
        return items[start:end]
    return items[start:] + items[:end - n]


def run_grpo(config: GRPOConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    transformers_set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    print(f"[GRPO] Device: {device}")

    # -- Data --
    # The task abstraction selects the dataset(s): "gsm8k" loads GSM8K for both
    # splits; "humaneval" trains on MBPP and evaluates on HumanEval. Both emit
    # the same (question, ground_truth) schema, so the access below is uniform.
    task = get_task(config.task)
    print(f"[GRPO] Task: {task.name} | Loading data ...")
    train_ds = task.load("train", n_samples=config.n_train_samples, seed=config.seed)
    test_ds  = task.load(task.eval_split, n_samples=config.n_test_samples, seed=config.seed)

    train_gts = [ex["ground_truth"] for ex in train_ds]
    test_gts  = [ex["ground_truth"] for ex in test_ds]

    # -- Resume? (resolve before building the trainer so we can load policy
    #    weights straight from the checkpoint dir) --
    ckpt_dir = f"{config.checkpoint_dir}/{config.experiment_name}"
    resume_state = None
    model_path_override = None
    resume_path = config.resume_from
    if resume_path:
        if resume_path == "auto":
            resume_path = find_latest_checkpoint(ckpt_dir)
        if resume_path:
            print(f"[GRPO] Resuming from {resume_path}")
            resume_state = load_grpo_checkpoint(resume_path, device)
            if resume_state.get("task") and resume_state["task"] != config.task:
                print(f"[GRPO] WARNING: checkpoint task={resume_state['task']!r} "
                      f"!= config task={config.task!r}")
            model_path_override = resume_state["model_path"]
        else:
            print(f"[GRPO] No checkpoint found in {ckpt_dir}; starting fresh.")

    # -- Trainer --
    trainer = load_grpo_trainer(config, device, model_path_override=model_path_override)

    train_prompts = [task.format_prompt(ex, trainer.tokenizer) for ex in train_ds]
    test_prompts  = [task.format_prompt(ex, trainer.tokenizer) for ex in test_ds]

    # -- Training loop --
    logger = ExperimentLogger(config.experiment_name, config.output_dir)
    reward_window: list[float] = []
    start_step = 0

    if resume_state is not None:
        trainer.optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        trainer.total_rollouts = resume_state["total_rollouts"]
        logger.log = resume_state["logger_log"]
        restore_rng_states(resume_state["rng_states"])
        start_step = resume_state["step"] + 1
        print(f"[GRPO] Resumed at step {start_step} "
              f"(total_rollouts={trainer.total_rollouts})")

    # Catches SIGTERM/SIGINT and Slurm's pre-preemption SIGUSR1 so we can save
    # before the runtime goes away.
    exit_handler = GracefulExitHandler()

    def _checkpoint(step, keep):
        # Checkpointing is opt-in: with --checkpoint-every 0 (default), runs
        # write no model (preserves the original from-scratch behavior).
        if config.checkpoint_every <= 0:
            return
        save_grpo_checkpoint(trainer, step, config, logger, ckpt_dir, keep)

    for step in range(start_step, config.n_steps):
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

        # -- Periodic checkpoint --
        if config.checkpoint_every > 0 and (step + 1) % config.checkpoint_every == 0:
            _checkpoint(step, keep=config.keep_checkpoints)

        # -- Graceful exit (disconnect / preemption) --
        if exit_handler.should_exit:
            print(f"[GRPO] Graceful exit requested at step {step}; saving checkpoint.")
            _checkpoint(step, keep=0)  # keep=0 -> don't rotate this emergency save
            logger.save()
            return

    logger.save()

    # -- Final checkpoint (so a completed run always leaves usable weights) --
    _checkpoint(config.n_steps - 1, keep=0)

    # -- Final evaluation --
    final_acc = trainer.evaluate(test_prompts, test_gts, n_eval=config.final_eval_size)
    print(f"\n[GRPO] Final test accuracy (GRPO, G={config.n_rollouts_per_prompt}): {final_acc:.3f}")
    print(f"[GRPO] Log saved to {config.output_dir}/{config.experiment_name}.json")
    print(f"[GRPO] Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO head-to-head on GSM8K")
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves memory)")
    parser.add_argument("--G", type=int, default=None,
                        help="Override group size (completions per prompt)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (prompts per step)")
    parser.add_argument("--task", type=str, default=None,
                        choices=["gsm8k", "humaneval"],
                        help="Task: gsm8k (default) or humaneval (MBPP train -> HumanEval eval)")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Override number of training steps (handy for smoke tests)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override generation length (lower = faster smoke test)")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save a checkpoint every N steps (0 = disabled). "
                             "Enables resume across disconnects / Slurm limits.")
    parser.add_argument("--keep-checkpoints", type=int, default=None,
                        help="Keep last K periodic checkpoints (0 = keep all)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint dir to resume from, or 'auto' for the latest")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write the metrics JSON (point at Drive on Colab). "
                             "Also sets checkpoint dir under it unless --checkpoint-dir is given.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Where to write checkpoints (defaults to <output-dir>/checkpoints)")
    args = parser.parse_args()

    cfg = local_test_config() if args.local_test else e2_7_config(seed=args.seed)
    cfg.seed = args.seed

    if args.task is not None:
        cfg.task = args.task
    if args.model_name:
        cfg.model_name = args.model_name
    if args.G is not None:
        cfg.n_rollouts_per_prompt = args.G
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.n_steps is not None:
        cfg.n_steps = args.n_steps
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens

    # Output + checkpoint locations (point at a mounted Drive on Colab).
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
        cfg.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.keep_checkpoints is not None:
        cfg.keep_checkpoints = args.keep_checkpoints
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # Store gradient_checkpointing on config for load_grpo_trainer
    cfg._gradient_checkpointing = args.gradient_checkpointing

    run_grpo(cfg)
