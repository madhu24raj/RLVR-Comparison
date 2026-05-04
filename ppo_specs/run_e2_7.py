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
    # ── DDP detection ────────────────────────────────────────────────────────
    # When launched via `accelerate launch` or `torchrun`, LOCAL_RANK is set.
    # In that case, opt into Accelerator-driven DDP. Otherwise, behavior is
    # identical to the legacy single-process path.
    USE_DDP = "LOCAL_RANK" in os.environ
    if USE_DDP:
        from accelerate import Accelerator
        from accelerate.utils import set_seed as accelerate_set_seed
        accelerator = Accelerator()
        device = accelerator.device
        # One Accelerate seed call covers python/numpy/torch/torch.cuda streams.
        accelerate_set_seed(config.seed)
        # Critical pitfall: shard divisibility (§7.6 hazards).
        assert config.batch_size % accelerator.num_processes == 0, (
            f"config.batch_size={config.batch_size} not divisible by "
            f"accelerator.num_processes={accelerator.num_processes}; "
            f"adjust config."
        )
    else:
        accelerator = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Full reproducibility: seed all RNGs including transformers' internal state
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        transformers_set_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    # Helpers for rank-0 gating and conditional barriers.
    def _is_main():
        return accelerator is None or accelerator.is_main_process

    def _print(*args, **kwargs):
        if _is_main():
            print(*args, **kwargs)

    def _wait():
        if accelerator is not None:
            accelerator.wait_for_everyone()

    _print(f"[E2.7] Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    _print("[E2.7] Loading GSM8K …")
    train_ds = load_gsm8k("train", n_samples=config.n_train_samples, seed=config.seed)
    test_ds  = load_gsm8k("test",  n_samples=config.n_test_samples)

    train_gts     = [ex["ground_truth"] for ex in train_ds]
    test_gts      = [ex["ground_truth"] for ex in test_ds]

    # ── Trainer ───────────────────────────────────────────────────────────────
    # Loaded before prompt formatting so we can use the model's chat template (L12).
    trainer, diagnostic_fn = load_ppo_trainer(
        config, accelerator if accelerator is not None else device,
    )

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
            _print(f"[E2.7] Resuming from {resume_path}")
            state = load_checkpoint(resume_path, config, device)

            # Restore model from checkpoint (reload from saved path).
            # Under DDP we let Accelerate place the module via prepare(); under
            # the legacy path we explicitly .to(device) as before.
            from transformers import AutoModelForCausalLM
            from ppo_specs.ppo_trainer import _build_adamw
            reloaded = AutoModelForCausalLM.from_pretrained(
                state["model_path"],
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            )
            if accelerator is None:
                reloaded = reloaded.to(device)
            trainer.model = reloaded
            trainer.policy_optimizer = _build_adamw(
                trainer.model.parameters(),
                lr=config.learning_rate,
                use_8bit=config.optimizer_8bit,
                use_fused=config.optimizer_fused,
            )
            trainer.policy_optimizer.load_state_dict(state["policy_optimizer_state_dict"])

            # §7.6.3: after reassignment the DDP wrapping from PPOTrainer.__init__
            # is gone — re-prepare so subsequent backward calls all-reduce.
            if accelerator is not None:
                trainer.model, trainer.policy_optimizer = accelerator.prepare(
                    trainer.model, trainer.policy_optimizer,
                )

            trainer.critic.load_state_dict(state["critic_state_dict"])
            if trainer.critic_optimizer and state["critic_optimizer_state_dict"]:
                trainer.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])

            trainer.step = state["trainer_step"]
            trainer.total_rollouts = state["total_rollouts"]
            start_step = state["step"] + 1  # resume from next step
            logger = ExperimentLogger(config.experiment_name, config.output_dir)
            logger.log = state["logger_log"]

            restore_rng_states(state["rng_states"])
            _print(f"[E2.7] Resumed at step {start_step}")

    # ── MC baseline (reference for advantage error) ───────────────────────────
    # Use the first 5 training prompts as a fixed reference set.
    # n_samples=10 locally; raise to 1000 on the cluster for paper-quality estimates.
    mc_baselines: dict = {}
    if compute_mc:
        # §7.6.5: under DDP, every rank with identical seeds would duplicate
        # the MC loop. Compute on rank 0 only and broadcast.
        if _is_main():
            n_mc = 10 if config.n_steps <= 10 else 50
            ref_p  = train_prompts[:5]
            ref_gt = train_gts[:5]
            _print(f"[E2.7] Estimating MC baselines ({n_mc} samples × {len(ref_p)} prompts) …")
            mc_baselines = estimate_mc_advantages(
                trainer.model, trainer.tokenizer,
                ref_p, ref_gt, trainer.reward_fn,
                n_samples=n_mc,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                device=str(device),
            )
            _print("[E2.7] MC baselines:", {k[-30:]: f"{v:.3f}" for k, v in mc_baselines.items()})
        if accelerator is not None:
            from accelerate.utils import broadcast_object_list
            mc_baselines_list = [mc_baselines]
            broadcast_object_list(mc_baselines_list, from_process=0)
            mc_baselines = mc_baselines_list[0]

    # ── Training loop ─────────────────────────────────────────────────────────
    if not resume_path or not resume_path.strip():
        logger = ExperimentLogger(config.experiment_name, config.output_dir)
    reward_history: list[float] = []  # accumulated rewards for variance (stability)

    # ── Set up graceful exit handler ─────────────────────────────────────
    exit_handler = GracefulExitHandler()
    ckpt_dir = f"{config.checkpoint_dir}/{config.experiment_name}"

    for step in range(start_step, config.n_steps):
        batch_p  = cycle_batch(train_prompts, step, config.batch_size)
        batch_gt = cycle_batch(train_gts,     step, config.batch_size)

        # Set question context for self-judge reward (no-op for deterministic).
        # §7.6.2: under DDP, generate_rollouts shards batch_p internally per
        # rank, so the self-judge wrapper's _idx indexes into the LOCAL shard.
        # We must pass the local shard to set_questions so questions and
        # completions stay aligned.
        if hasattr(trainer.reward_fn, 'set_questions'):
            if accelerator is not None and accelerator.num_processes > 1:
                from ppo_specs.ppo_trainer import _shard_list
                local_p = _shard_list(
                    batch_p, accelerator.process_index, accelerator.num_processes,
                )
                trainer.reward_fn.set_questions(local_p)
            else:
                trainer.reward_fn.set_questions(batch_p)

        metrics = trainer.train_step(batch_p, batch_gt)
        reward_history.append(metrics["mean_reward"])

        if step % config.log_every == 0:
            _print(
                f"  step {step:3d} | reward={metrics['mean_reward']:.3f} "
                f"| acc={metrics['accuracy']:.3f} "
                f"| parse={metrics['parse_success_rate']:.2f} "
                f"| boxed={metrics['format_match_rate']:.2f} "
                f"| policy_loss={metrics['policy_loss']:.4f} "
                f"| critic_loss={metrics['critic_loss']:.4f} "
                f"| kl={metrics['kl_divergence']:.4f}"
            )

        # ── Periodic evaluation ───────────────────────────────────────────────
        if step % config.eval_every == 0:
            test_acc = trainer.evaluate(test_prompts, test_gts, n_eval=config.eval_size)

            # Dual-track: when using self_judge/combined, evaluate deterministic
            # accuracy on held-out set to detect reward hacking.
            diagnostic_test_acc = None
            if diagnostic_fn is not None:
                saved_reward_fn = trainer.reward_fn
                trainer.reward_fn = diagnostic_fn
                diagnostic_test_acc = trainer.evaluate(
                    test_prompts, test_gts, n_eval=config.eval_size,
                )
                trainer.reward_fn = saved_reward_fn

            # (ii) Training stability: variance over the last window
            window = reward_history[-config.eval_every:] if len(reward_history) >= config.eval_every \
                     else reward_history
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
                "kl_divergence":  metrics["kl_divergence"],
                "kl_ref_divergence": metrics["kl_ref_divergence"],
                # Phase-1 reward-starvation diagnostics
                "parse_success_rate":  metrics["parse_success_rate"],
                "format_match_rate":   metrics["format_match_rate"],
                "reward_nonzero_rate": metrics["reward_nonzero_rate"],
                "reward_mode": config.reward_mode,
                "diagnostic_test_accuracy": diagnostic_test_acc,
            }
            if adv_error is not None:
                log_entry["advantage_error"] = adv_error       # (iv)

            if _is_main():
                logger.log_step(step, **log_entry)
                logger.save()  # flush to disk after each eval
                _print(
                    f"    → test_acc={test_acc:.3f} "
                    f"| stability(var)={stability:.4f}"
                    + (f" | adv_error={adv_error:.4f}" if adv_error is not None else "")
                )

        # ── Checkpoint save ──────────────────────────────────────────────
        if config.checkpoint_every > 0 and (step + 1) % config.checkpoint_every == 0:
            _wait()
            if _is_main():
                save_checkpoint(
                    trainer, step, config, logger, ckpt_dir,
                    config.keep_checkpoints, accelerator=accelerator,
                )

        # ── Graceful exit check ──────────────────────────────────────────
        # §7.6.4: broadcast exit signal so ranks don't deadlock when only one
        # rank received SIGTERM.
        if accelerator is not None:
            exit_flag = torch.tensor(int(exit_handler.should_exit), device=device)
            exit_flag = accelerator.reduce(exit_flag, reduction="max")
            should_exit_all = bool(exit_flag.item())
        else:
            should_exit_all = exit_handler.should_exit

        if should_exit_all:
            _wait()
            if _is_main():
                _print(f"[E2.7] Graceful exit requested at step {step}")
                save_checkpoint(
                    trainer, step, config, logger, ckpt_dir,
                    keep_checkpoints=0, accelerator=accelerator,
                )
                logger.save()
            _wait()
            return

    # Save final checkpoint
    if config.checkpoint_every > 0:
        _wait()
        if _is_main():
            save_checkpoint(
                trainer, config.n_steps - 1, config, logger, ckpt_dir,
                config.keep_checkpoints, accelerator=accelerator,
            )

    if _is_main():
        logger.save()

    # ── Final evaluation ──────────────────────────────────────────────────────
    # trainer.evaluate uses gather_for_metrics under DDP and returns the same
    # value on every rank, so the print can run on rank 0 only.
    final_acc = trainer.evaluate(test_prompts, test_gts, n_eval=config.final_eval_size)
    _print(f"\n[E2.7] Final test accuracy (PPO, {config.critic_capacity} critic): {final_acc:.3f}")
    _print(f"[E2.7] Log saved to {config.output_dir}/{config.experiment_name}.json")


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
    parser.add_argument(
        "--reward-mode", type=str, default=None,
        choices=["deterministic", "self_judge", "combined"],
        help="Reward mode: deterministic (binary), self_judge (log-likelihood), combined",
    )
    parser.add_argument(
        "--self-judge-weight", type=float, default=None,
        help="Weight for self_judge in combined mode (0-1)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Override config.model_name (e.g. meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Enable gradient checkpointing (required for 8B+)",
    )
    parser.add_argument(
        "--optimizer-8bit", action="store_true",
        help="Enable bnb 8-bit AdamW (saves ~48 GB at 8B)",
    )
    parser.add_argument(
        "--optimizer-fused", action="store_true",
        help="Enable torch fused AdamW kernel (CUDA only)",
    )
    parser.add_argument(
        "--reference-quant", type=str, default=None,
        choices=["none", "int8", "nf4"],
        help="Quantize the frozen reference model",
    )
    parser.add_argument(
        "--length-bucketed-generation", action="store_true",
        help="Enable length-bucketed generation (saves ~3 s/step at 8B)",
    )
    parser.add_argument(
        "--reward-model-capacity", type=str, default=None,
        choices=["none", "small", "large"],
        help="Learned reward model tier (none = use deterministic gsm8k_reward only)",
    )
    parser.add_argument(
        "--reward-model-name", type=str, default=None,
        help="HF hub id or local path of the learned reward model checkpoint",
    )
    parser.add_argument(
        "--reward-blend-alpha", type=float, default=None,
        help="Blend factor for combining learned RM with gsm8k_reward (0=verifier only, 1=RM only)",
    )
    parser.add_argument(
        "--reward-model-reuse-reference", action="store_true",
        help="Reuse the KL-anchor reference model as the RM base (saves ~16GB at 8B)",
    )
    args = parser.parse_args()

    cfg = local_test_config() if args.local_test else e2_7_config(seed=args.seed)
    cfg.seed = args.seed
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if args.reward_mode:
        cfg.reward_mode = args.reward_mode
        if cfg.reward_mode != "deterministic" and cfg.reference_kl_coeff == 0:
            cfg.reference_kl_coeff = 0.01  # auto-enable reference model
    if args.self_judge_weight is not None:
        cfg.self_judge_weight = args.self_judge_weight
    if args.model_name:
        cfg.model_name = args.model_name
    if args.gradient_checkpointing:
        cfg.gradient_checkpointing = True
    if args.optimizer_8bit:
        cfg.optimizer_8bit = True
    if args.optimizer_fused:
        cfg.optimizer_fused = True
    if args.reference_quant is not None:
        cfg.reference_quant = args.reference_quant
    if args.length_bucketed_generation:
        cfg.length_bucketed_generation = True
    if args.reward_model_capacity is not None:
        cfg.reward_model_capacity = args.reward_model_capacity
    if args.reward_model_name is not None:
        cfg.reward_model_name = args.reward_model_name
    if args.reward_blend_alpha is not None:
        cfg.reward_blend_alpha = args.reward_blend_alpha
    if args.reward_model_reuse_reference:
        cfg.reward_model_reuse_reference = True

    run_e2_7(cfg, compute_mc=not args.no_mc)
