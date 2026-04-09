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

import random
import torch
import numpy as np
from transformers import set_seed as transformers_set_seed

from src.data import load_gsm8k, format_prompt_with_template
from eval.metrics import ExperimentLogger
from ppo_specs.config import PPOConfig, local_test_config, e2_7_config
from ppo_specs.ppo_trainer import load_ppo_trainer
from ppo_specs.advantage import estimate_mc_advantages, advantage_estimation_error
from ppo_specs.checkpoint import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    restore_rng_states, GracefulExitHandler,
)
from ppo_specs.utils import cycle_batch


# ── Main experiment ───────────────────────────────────────────────────────────

def run_e2_7(config: PPOConfig, compute_mc: bool = True) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Full reproducibility: seed all RNGs including transformers' internal state
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    transformers_set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    print(f"[E2.7] Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("[E2.7] Loading GSM8K …")
    train_ds = load_gsm8k("train", n_samples=config.n_train_samples, seed=config.seed)
    test_ds  = load_gsm8k("test",  n_samples=config.n_test_samples)

    train_gts     = [ex["ground_truth"] for ex in train_ds]
    test_gts      = [ex["ground_truth"] for ex in test_ds]

    # ── Trainer ───────────────────────────────────────────────────────────────
    # Loaded before prompt formatting so we can use the model's chat template (L12).
    trainer = load_ppo_trainer(config, device)

    train_prompts = [
        format_prompt_with_template(ex["question"], trainer.tokenizer) for ex in train_ds
    ]
    test_prompts  = [
        format_prompt_with_template(ex["question"], trainer.tokenizer) for ex in test_ds
    ]

    # ── Resume from checkpoint if requested ──────────────────────────────────
    start_step = 0
    resume_path = config.resume_from
    if resume_path:
        ckpt_dir = f"{config.checkpoint_dir}/{config.experiment_name}"
        if resume_path == "auto":
            resume_path = find_latest_checkpoint(ckpt_dir)
        if resume_path:
            print(f"[E2.7] Resuming from {resume_path}")
            state = load_checkpoint(resume_path, config, device)

            # Restore model from checkpoint (reload from saved path)
            from transformers import AutoModelForCausalLM
            trainer.model = AutoModelForCausalLM.from_pretrained(
                state["model_path"],
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            ).to(device)
            trainer.policy_optimizer = torch.optim.AdamW(
                trainer.model.parameters(), lr=config.learning_rate
            )
            trainer.policy_optimizer.load_state_dict(state["policy_optimizer_state_dict"])

            trainer.critic.load_state_dict(state["critic_state_dict"])
            if trainer.critic_optimizer and state["critic_optimizer_state_dict"]:
                trainer.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])

            trainer.step = state["trainer_step"]
            trainer.total_rollouts = state["total_rollouts"]
            start_step = state["step"] + 1  # resume from next step
            logger = ExperimentLogger(config.experiment_name, config.output_dir)
            logger.log = state["logger_log"]

            restore_rng_states(state["rng_states"])
            print(f"[E2.7] Resumed at step {start_step}")

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
            temperature=config.temperature,
            device=str(device),
        )
        print("[E2.7] MC baselines:", {k[-30:]: f"{v:.3f}" for k, v in mc_baselines.items()})

    # ── Training loop ─────────────────────────────────────────────────────────
    logger = ExperimentLogger(config.experiment_name, config.output_dir)
    reward_window: list[float] = []  # rolling window for variance (stability)

    # ── Set up graceful exit handler ─────────────────────────────────────
    exit_handler = GracefulExitHandler()
    ckpt_dir = f"{config.checkpoint_dir}/{config.experiment_name}"

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
                f"| critic_loss={metrics['critic_loss']:.4f}"
            )

        # ── Periodic evaluation ───────────────────────────────────────────────
        if step % config.eval_every == 0:
            test_acc = trainer.evaluate(test_prompts, test_gts, n_eval=config.eval_size)

            # (ii) Training stability: variance over the last window
            window = reward_window[-config.eval_every:] if len(reward_window) >= config.eval_every \
                     else reward_window
            stability = float(np.var(window))

            # (iv) Advantage estimation error
            # NOTE: MC baselines were estimated from the INITIAL policy.
            # As training progresses, V(s) under the current policy diverges
            # from V_MC(s) under the initial policy.  This metric therefore
            # measures critic tracking of the initial-policy value function.
            # See logic.md L6 for discussion of this limitation.
            adv_error = None
            if mc_baselines:
                mc_vals = np.array(list(mc_baselines.values()))
                ref_p_for_eval = list(mc_baselines.keys())

                if config.critic_capacity == "none":
                    # REINFORCE uses batch-mean reward as baseline.  To compare
                    # fairly against MC baselines on the SAME reference prompts,
                    # we generate rollouts on those prompts and take their mean.
                    # This avoids the apples-to-oranges error of comparing
                    # batch-mean on arbitrary current prompts vs MC on reference
                    # prompts.
                    # Note: we temporarily generate without incrementing
                    # total_rollouts so the rollout budget metric stays clean.
                    saved_rollouts = trainer.total_rollouts
                    ref_batch = trainer.generate_rollouts(ref_p_for_eval, train_gts[:5])
                    trainer.total_rollouts = saved_rollouts  # undo count
                    ref_mean = float(ref_batch.rewards().mean())
                    est = np.full(len(mc_vals), ref_mean)
                else:
                    est = trainer._eval_critic_on_prompts(ref_p_for_eval)

                adv_error = advantage_estimation_error(est, mc_vals)

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
            logger.save()  # flush to disk after each eval
            print(
                f"    → test_acc={test_acc:.3f} "
                f"| stability(var)={stability:.4f}"
                + (f" | adv_error={adv_error:.4f}" if adv_error is not None else "")
            )

        # ── Checkpoint save ──────────────────────────────────────────────
        if config.checkpoint_every > 0 and (step + 1) % config.checkpoint_every == 0:
            save_checkpoint(trainer, step, config, logger, ckpt_dir, config.keep_checkpoints)

        # ── Graceful exit check ──────────────────────────────────────────
        if exit_handler.should_exit:
            print(f"[E2.7] Graceful exit requested at step {step}")
            save_checkpoint(trainer, step, config, logger, ckpt_dir, keep_checkpoints=0)
            logger.save()
            return

    # Save final checkpoint
    if config.checkpoint_every > 0:
        save_checkpoint(trainer, config.n_steps - 1, config, logger, ckpt_dir, config.keep_checkpoints)

    logger.save()

    # ── Final evaluation ──────────────────────────────────────────────────────
    final_acc = trainer.evaluate(test_prompts, test_gts, n_eval=config.final_eval_size)
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
    parser.add_argument(
        "--checkpoint-every", type=int, default=None,
        help="Save checkpoint every N steps (0 = disabled)",
    )
    parser.add_argument(
        "--resume-from", type=str, default="",
        help="Path to checkpoint dir, or 'auto' for latest",
    )
    args = parser.parse_args()

    cfg = local_test_config() if args.local_test else e2_7_config(seed=args.seed)
    cfg.seed = args.seed
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.resume_from:
        cfg.resume_from = args.resume_from

    run_e2_7(cfg, compute_mc=not args.no_mc)
