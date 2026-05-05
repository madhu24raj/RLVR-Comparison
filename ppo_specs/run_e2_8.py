"""
E2.8: Critic Quality and the PPO–GRPO Crossover (PPO portion).

Sweeps critic capacities ["none", "small", "medium", "large"] and records:
  (i)  final accuracy per capacity
  (ii) critic approximation error εV = RMSE(V̂(s), V_MC(s))
  (iii) advantage bias |estimated_baseline - MC_baseline|
  (iv) sample-efficiency curves (accuracy vs total rollouts)

Each capacity run starts from the same pretrained weights so results are
directly comparable.

Theoretical prediction (Theorem 2.5):
  For εV < ε*V (accurate critic) → PPO outperforms GRPO with small G
  For εV > ε*V                   → GRPO is preferable
  ε*V ≈ (1 - γ) √(σ*²(1 - 1/G) - σ*²_A)

Results
───────
  results/ppo_e2_8_<capacity>_seed<N>.json  – per-step log for each run
  results/e2_8_sweep_summary.json           – crossover table

Usage
─────
Local smoke test (2 capacities, tiny data):
    python ppo_specs/run_e2_8.py --local-test

Single capacity:
    python ppo_specs/run_e2_8.py --capacity medium --seed 0

Full sweep:
    python ppo_specs/run_e2_8.py --seed 0
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
from ppo_specs.config import PPOConfig, CRITIC_CAPACITIES, local_test_config, e2_8_config, copy_config
from ppo_specs.ppo_trainer import load_ppo_trainer
from ppo_specs.advantage import (
    estimate_mc_advantages,
    advantage_estimation_error,
    critic_approximation_error,
)
from ppo_specs.utils import cycle_batch


# ── DDP detection ─────────────────────────────────────────────────────────────
# Opt-in: if LOCAL_RANK is set (by `accelerate launch` or `torchrun`), route
# through Accelerator. Otherwise the default behavior is identical to the
# pre-DDP single-process code path.
USE_DDP = "LOCAL_RANK" in os.environ


# ── Single capacity run ───────────────────────────────────────────────────────

def run_one_capacity(
    capacity: str,
    base_config: PPOConfig,
    train_prompts: list,
    train_gts: list,
    test_prompts: list,
    test_gts: list,
    mc_baselines: dict,
    device: torch.device,
    accelerator=None,
) -> dict:
    """
    Train PPO with one critic capacity; return summary metrics.

    Starts from fresh pretrained weights every call so runs are independent.
    """
    def _is_main():
        return accelerator is None or accelerator.is_main_process

    def _print(*a, **kw):
        if _is_main():
            print(*a, **kw)

    def _wait():
        if accelerator is not None:
            accelerator.wait_for_everyone()

    _print(f"\n{'='*55}")
    _print(f"  Critic capacity: {capacity.upper()}")
    _print(f"{'='*55}")

    cfg = copy_config(
        base_config,
        critic_capacity=capacity,
        experiment_name=f"ppo_e2_8_{capacity}_seed{base_config.seed}",
    )

    trainer, _ = load_ppo_trainer(
        cfg, accelerator if accelerator is not None else device
    )
    logger  = ExperimentLogger(cfg.experiment_name, cfg.output_dir)

    accuracy_curve: list[tuple[int, float]] = []
    ev_samples:     list[float] = []   # critic error εV per eval step
    bias_samples:   list[float] = []   # advantage bias per eval step

    for step in range(cfg.n_steps):
        batch_p  = cycle_batch(train_prompts, step, cfg.batch_size)
        batch_gt = cycle_batch(train_gts,     step, cfg.batch_size)

        # Set question context for self-judge reward (no-op for deterministic).
        # Under DDP, generate_rollouts shards batch_p internally per rank, so
        # the self-judge wrapper indexes into the LOCAL shard. Pass the local
        # shard so questions and completions stay aligned. Without this call,
        # reward_mode='self_judge' crashes on the first reward in E2.8 sweeps.
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

        if step % cfg.eval_every == 0:
            test_acc = trainer.evaluate(test_prompts, test_gts, n_eval=cfg.eval_size)
            accuracy_curve.append((metrics["total_rollouts"], test_acc))

            # (ii) εV: compare critic baseline vs MC ground truth
            # NOTE: MC baselines are from the initial policy. As training
            # progresses this measures tracking of the initial value function.
            # See logic.md L6 for discussion.
            ev = float("nan")
            bias = float("nan")
            if mc_baselines:
                mc_vals = np.array(list(mc_baselines.values()))
                ref_prompts_for_eval = list(mc_baselines.keys())

                if capacity == "none":
                    # Generate rollouts on the SAME reference prompts used for
                    # MC estimation, and use their mean as the REINFORCE baseline.
                    # This avoids comparing batch-mean on arbitrary prompts vs
                    # MC on reference prompts (apples-to-oranges).
                    saved_rollouts = trainer.total_rollouts
                    ref_batch = trainer.generate_rollouts(
                        ref_prompts_for_eval, train_gts[:5]
                    )
                    trainer.total_rollouts = saved_rollouts  # don't corrupt budget
                    ref_mean = float(ref_batch.rewards().mean())
                    est_vals = np.full(len(mc_vals), ref_mean)
                else:
                    est_vals = trainer._eval_critic_on_prompts(ref_prompts_for_eval)

                ev   = critic_approximation_error(est_vals, mc_vals)
                bias = advantage_estimation_error(est_vals, mc_vals)
                ev_samples.append(ev)
                bias_samples.append(bias)

            logger.log_step(
                step,
                test_accuracy=test_acc,
                critic_error_ev=ev,
                advantage_bias=bias,
                total_rollouts=metrics["total_rollouts"],
                mean_reward=metrics["mean_reward"],
                policy_loss=metrics["policy_loss"],
                critic_loss=metrics["critic_loss"],
                kl_divergence=metrics["kl_divergence"],
                kl_ref_divergence=metrics["kl_ref_divergence"],
                # Phase-1 reward-starvation diagnostics
                parse_success_rate=metrics["parse_success_rate"],
                format_match_rate=metrics["format_match_rate"],
                reward_nonzero_rate=metrics["reward_nonzero_rate"],
            )
            _print(
                f"  step {step:3d} | test_acc={test_acc:.3f} "
                f"| εV={ev:.4f} | bias={bias:.4f} "
                f"| kl={metrics['kl_divergence']:.4f}"
            )

    if _is_main():
        logger.save()

    final_acc  = trainer.evaluate(test_prompts, test_gts, n_eval=cfg.final_eval_size)
    mean_ev    = float(np.nanmean(ev_samples))   if ev_samples   else float("nan")
    mean_bias  = float(np.nanmean(bias_samples)) if bias_samples else float("nan")

    _print(f"\n  {capacity}: final_acc={final_acc:.3f}  εV={mean_ev:.4f}  bias={mean_bias:.4f}")

    return {
        "capacity":        capacity,
        "final_accuracy":  final_acc,
        "mean_ev":         mean_ev,
        "mean_bias":       mean_bias,
        "accuracy_curve":  accuracy_curve,   # [(rollouts, acc), …]
    }


# ── Full sweep ────────────────────────────────────────────────────────────────

def run_e2_8(config: PPOConfig, capacities: list[str]) -> None:
    if USE_DDP:
        from accelerate import Accelerator, DistributedDataParallelKwargs
        from accelerate.utils import set_seed as accelerate_set_seed
        # find_unused_parameters=False: fail-fast if any param is dropped
        # from the grad pass (and ~5-10 ms/step faster than the default).
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        device = accelerator.device
        accelerate_set_seed(config.seed)
        assert config.batch_size % accelerator.num_processes == 0, (
            f"config.batch_size={config.batch_size} not divisible by "
            f"accelerator.num_processes={accelerator.num_processes}"
        )
        # Multi-node guard: see run_e2_7.py for rationale.
        if accelerator.num_machines > 1 and not os.environ.get("MASTER_ADDR"):
            raise RuntimeError(
                f"Accelerator num_machines={accelerator.num_machines} but "
                f"MASTER_ADDR is unset. Multi-node training requires "
                f"MASTER_ADDR/MASTER_PORT (out-of-scope for Phase 2)."
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

    def _is_main():
        return accelerator is None or accelerator.is_main_process

    def _print(*a, **kw):
        if _is_main():
            print(*a, **kw)

    def _wait():
        if accelerator is not None:
            accelerator.wait_for_everyone()

    _print(f"[E2.8] Device: {device}")

    # ── Shared data ───────────────────────────────────────────────────────────
    _print("[E2.8] Loading GSM8K …")
    train_ds = load_gsm8k("train", n_samples=config.n_train_samples, seed=config.seed)
    test_ds  = load_gsm8k("test",  n_samples=config.n_test_samples)

    train_gts     = [ex["ground_truth"] for ex in train_ds]
    test_gts      = [ex["ground_truth"] for ex in test_ds]

    # Load a temporary trainer first so we can use its tokenizer's chat template
    # for prompt formatting (L12). Reused below for MC baseline estimation.
    tmp_cfg     = copy_config(config, critic_capacity="none")
    tmp_trainer, _ = load_ppo_trainer(
        tmp_cfg, accelerator if accelerator is not None else device
    )

    train_prompts = [
        format_prompt_with_template(ex["question"], tmp_trainer.tokenizer) for ex in train_ds
    ]
    test_prompts  = [
        format_prompt_with_template(ex["question"], tmp_trainer.tokenizer) for ex in test_ds
    ]

    # ── MC baselines (shared reference, estimated once) ───────────────────────
    # §7.6.5: gate MC on rank 0 then broadcast — otherwise every rank duplicates
    # the (expensive) MC loop with identical seeds.
    n_mc = 10 if config.n_steps <= 10 else 50
    ref_p  = train_prompts[:5]
    ref_gt = train_gts[:5]
    _print(f"[E2.8] Estimating MC baselines ({n_mc} samples × {len(ref_p)} prompts) …")
    mc_baselines: dict = {}
    if _is_main():
        mc_baselines = estimate_mc_advantages(
            tmp_trainer.model, tmp_trainer.tokenizer,
            ref_p, ref_gt, tmp_trainer.reward_fn,
            n_samples=n_mc,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            device=str(device),
        )
    if accelerator is not None:
        from accelerate.utils import broadcast_object_list
        mc_baselines_list = [mc_baselines]
        broadcast_object_list(mc_baselines_list, from_process=0)
        mc_baselines = mc_baselines_list[0]

    # Free VRAM before sweep: move model to CPU first, then GC.
    # Barrier on entry so all ranks have finished MC broadcast before tear-down.
    _wait()
    import gc
    tmp_trainer.model.cpu()
    del tmp_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _wait()

    _print("[E2.8] MC baselines:", {k[-30:]: f"{v:.3f}" for k, v in mc_baselines.items()})

    # ── Capacity sweep ────────────────────────────────────────────────────────
    results = []
    for cap in capacities:
        # Reset RNG state before each capacity for fair comparison.
        if USE_DDP:
            from accelerate.utils import set_seed as accelerate_set_seed
            accelerate_set_seed(config.seed)
        else:
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            transformers_set_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        result = run_one_capacity(
            cap, config,
            train_prompts, train_gts,
            test_prompts,  test_gts,
            mc_baselines,  device,
            accelerator=accelerator,
        )
        results.append(result)
        # Barrier so all ranks have finished the capacity run before any cache
        # eviction; rank N could otherwise still be mid-step while rank 0 frees
        # the allocator. No-op in legacy single-process mode.
        _wait()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _wait()

    # ── Summary ───────────────────────────────────────────────────────────────
    _wait()
    out_path = Path(config.output_dir) / "e2_8_sweep_summary.json"
    if _is_main():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    _print(f"\n[E2.8] Sweep summary saved to {out_path}")
    _print("\n=== E2.8 Crossover Summary ===")
    _print(f"{'Capacity':<10} {'Final Acc':<12} {'εV (RMSE)':<12} {'Adv Bias':<10}")
    _print("-" * 46)
    for r in results:
        _print(
            f"{r['capacity']:<10} "
            f"{r['final_accuracy']:<12.3f} "
            f"{r['mean_ev']:<12.4f} "
            f"{r['mean_bias']:<10.4f}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2.8: PPO critic quality sweep")
    parser.add_argument(
        "--local-test", action="store_true",
        help="Tiny config; runs only 'none' and 'small' critics",
    )
    parser.add_argument(
        "--capacity", type=str, default=None,
        help="Run a single capacity. Default: all four.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
        "--checkpoint-every", type=int, default=None,
        help="Save checkpoint every N steps (0 = disabled). NOTE: E2.8 "
             "currently does not implement resume; this flag is accepted "
             "for SLURM compatibility but only the periodic save will engage.",
    )
    parser.add_argument(
        "--resume-from", type=str, default="",
        help="Path to checkpoint dir, or 'auto' for latest. NOTE: E2.8 does "
             "not yet support resume; this flag is accepted for SLURM "
             "compatibility but resumption will be skipped with a warning.",
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

    if args.local_test:
        cfg = local_test_config()
        cfg.seed = args.seed
        caps = [args.capacity] if args.capacity else ["none", "small"]
    else:
        cfg  = e2_8_config(seed=args.seed)
        caps = [args.capacity] if args.capacity else CRITIC_CAPACITIES

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
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.resume_from:
        cfg.resume_from = args.resume_from
        print(
            "[E2.8] WARNING: --resume-from was passed but run_e2_8.py does not "
            "yet support resume. Continuing without resuming. The flag is "
            "accepted only for SLURM script compatibility."
        )
    if args.reward_model_capacity is not None:
        cfg.reward_model_capacity = args.reward_model_capacity
    if args.reward_model_name is not None:
        cfg.reward_model_name = args.reward_model_name
    if args.reward_blend_alpha is not None:
        cfg.reward_blend_alpha = args.reward_blend_alpha
    if args.reward_model_reuse_reference:
        cfg.reward_model_reuse_reference = True

    run_e2_8(cfg, caps)
