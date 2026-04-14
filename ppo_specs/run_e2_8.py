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
) -> dict:
    """
    Train PPO with one critic capacity; return summary metrics.

    Starts from fresh pretrained weights every call so runs are independent.
    """
    print(f"\n{'='*55}")
    print(f"  Critic capacity: {capacity.upper()}")
    print(f"{'='*55}")

    cfg = copy_config(
        base_config,
        critic_capacity=capacity,
        experiment_name=f"ppo_e2_8_{capacity}_seed{base_config.seed}",
    )

    trainer = load_ppo_trainer(cfg, device)
    logger  = ExperimentLogger(cfg.experiment_name, cfg.output_dir)

    accuracy_curve: list[tuple[int, float]] = []
    ev_samples:     list[float] = []   # critic error εV per eval step
    bias_samples:   list[float] = []   # advantage bias per eval step

    for step in range(cfg.n_steps):
        batch_p  = cycle_batch(train_prompts, step, cfg.batch_size)
        batch_gt = cycle_batch(train_gts,     step, cfg.batch_size)

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
            print(
                f"  step {step:3d} | test_acc={test_acc:.3f} "
                f"| εV={ev:.4f} | bias={bias:.4f} "
                f"| kl={metrics['kl_divergence']:.4f}"
            )

    logger.save()

    final_acc  = trainer.evaluate(test_prompts, test_gts, n_eval=cfg.final_eval_size)
    mean_ev    = float(np.nanmean(ev_samples))   if ev_samples   else float("nan")
    mean_bias  = float(np.nanmean(bias_samples)) if bias_samples else float("nan")

    print(f"\n  {capacity}: final_acc={final_acc:.3f}  εV={mean_ev:.4f}  bias={mean_bias:.4f}")

    return {
        "capacity":        capacity,
        "final_accuracy":  final_acc,
        "mean_ev":         mean_ev,
        "mean_bias":       mean_bias,
        "accuracy_curve":  accuracy_curve,   # [(rollouts, acc), …]
    }


# ── Full sweep ────────────────────────────────────────────────────────────────

def run_e2_8(config: PPOConfig, capacities: list[str]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Full reproducibility: seed all RNGs including transformers' internal state
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    transformers_set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    print(f"[E2.8] Device: {device}")

    # ── Shared data ───────────────────────────────────────────────────────────
    print("[E2.8] Loading GSM8K …")
    train_ds = load_gsm8k("train", n_samples=config.n_train_samples, seed=config.seed)
    test_ds  = load_gsm8k("test",  n_samples=config.n_test_samples)

    train_gts     = [ex["ground_truth"] for ex in train_ds]
    test_gts      = [ex["ground_truth"] for ex in test_ds]

    # Load a temporary trainer first so we can use its tokenizer's chat template
    # for prompt formatting (L12). Reused below for MC baseline estimation.
    tmp_cfg     = copy_config(config, critic_capacity="none")
    tmp_trainer = load_ppo_trainer(tmp_cfg, device)

    train_prompts = [
        format_prompt_with_template(ex["question"], tmp_trainer.tokenizer) for ex in train_ds
    ]
    test_prompts  = [
        format_prompt_with_template(ex["question"], tmp_trainer.tokenizer) for ex in test_ds
    ]

    # ── MC baselines (shared reference, estimated once) ───────────────────────
    n_mc = 10 if config.n_steps <= 10 else 50
    ref_p  = train_prompts[:5]
    ref_gt = train_gts[:5]
    print(f"[E2.8] Estimating MC baselines ({n_mc} samples × {len(ref_p)} prompts) …")
    mc_baselines = estimate_mc_advantages(
        tmp_trainer.model, tmp_trainer.tokenizer,
        ref_p, ref_gt, tmp_trainer.reward_fn,
        n_samples=n_mc,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        device=str(device),
    )
    # Free VRAM before sweep: move model to CPU first, then GC
    import gc
    tmp_trainer.model.cpu()
    del tmp_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[E2.8] MC baselines:", {k[-30:]: f"{v:.3f}" for k, v in mc_baselines.items()})

    # ── Capacity sweep ────────────────────────────────────────────────────────
    results = []
    for cap in capacities:
        # Reset RNG state before each capacity for fair comparison
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
        )
        results.append(result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    out_path = Path(config.output_dir) / "e2_8_sweep_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[E2.8] Sweep summary saved to {out_path}")
    print("\n=== E2.8 Crossover Summary ===")
    print(f"{'Capacity':<10} {'Final Acc':<12} {'εV (RMSE)':<12} {'Adv Bias':<10}")
    print("-" * 46)
    for r in results:
        print(
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
    args = parser.parse_args()

    if args.local_test:
        cfg = local_test_config()
        cfg.seed = args.seed
        caps = [args.capacity] if args.capacity else ["none", "small"]
    else:
        cfg  = e2_8_config(seed=args.seed)
        caps = [args.capacity] if args.capacity else CRITIC_CAPACITIES

    run_e2_8(cfg, caps)
