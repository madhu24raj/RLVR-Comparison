"""
Shared utilities for ppo_specs run scripts.

Centralises helpers that would otherwise be duplicated across
run_e2_7.py and run_e2_8.py.
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from ppo_specs.advantage import estimate_mc_advantages


# ── Batch cycling ─────────────────────────────────────────────────────────────

def cycle_batch(items: list, step: int, batch_size: int) -> list:
    """
    Return a contiguous slice of length batch_size, wrapping around on overflow.

    Used to cycle through training data without reshuffling each step.

    Args:
        items:      Full list to slice from
        step:       Current training step (0-indexed)
        batch_size: Number of items to return

    Returns:
        List of length batch_size
    """
    n = len(items)
    start = (step * batch_size) % n
    if start + batch_size <= n:
        return items[start : start + batch_size]
    return items[start:] + items[: (start + batch_size) - n]


# ── MC baseline estimation ────────────────────────────────────────────────────

def setup_mc_baselines(
    trainer,
    train_prompts: list[str],
    train_gts: list[str],
    n_steps: int,
    max_new_tokens: int,
    device: torch.device,
    n_ref_prompts: int = 5,
) -> dict[str, float]:
    """
    Estimate Monte Carlo baselines on a small reference set.

    Selects the first n_ref_prompts from the training set, then calls
    estimate_mc_advantages with a sample count scaled to the run size:
      - n_steps <= 10  →  10 samples  (local smoke tests)
      - n_steps > 10   →  50 samples  (development runs; raise to 1000 on cluster)

    Args:
        trainer:        PPOTrainer (model + tokenizer already loaded)
        train_prompts:  Full list of training prompt strings
        train_gts:      Matching ground-truth answers
        n_steps:        Total training steps (used to decide sample count)
        max_new_tokens: Generation length per sample
        device:         Torch device
        n_ref_prompts:  Number of reference prompts to estimate baselines for

    Returns:
        Dict mapping prompt string → MC mean reward
    """
    n_mc = 10 if n_steps <= 10 else 50
    ref_prompts = train_prompts[:n_ref_prompts]
    ref_gts = train_gts[:n_ref_prompts]

    print(
        f"[MC] Estimating baselines: {n_mc} samples × {len(ref_prompts)} prompts …"
    )
    mc_baselines = estimate_mc_advantages(
        trainer.model,
        trainer.tokenizer,
        ref_prompts,
        ref_gts,
        trainer.reward_fn,
        n_samples=n_mc,
        max_new_tokens=max_new_tokens,
        device=str(device),
    )
    print("[MC] Baselines:", {k[-30:]: f"{v:.3f}" for k, v in mc_baselines.items()})
    return mc_baselines
