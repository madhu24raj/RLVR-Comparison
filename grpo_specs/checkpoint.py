"""
Checkpoint save/load for the from-scratch GRPO trainer.

GRPO has no critic, so a checkpoint is simpler than PPO's: policy model +
optimizer + training counters + logger log + RNG states. Saves are atomic
(write to a temp dir, then rename) and rotated (keep last K).

The generic, trainer-agnostic helpers (`find_latest_checkpoint`,
`restore_rng_states`, `GracefulExitHandler`) are reused from
ppo_specs/checkpoint.py rather than duplicated.

Designed for unreliable runtimes: point `checkpoint_dir` at a mounted Drive on
Colab, or let Slurm's SIGUSR1 trigger a graceful save before preemption.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Reuse the generic helpers (no PPO-specific state in these three).
from ppo_specs.checkpoint import (  # noqa: F401  (re-exported for callers)
    find_latest_checkpoint,
    restore_rng_states,
    GracefulExitHandler,
)


def save_grpo_checkpoint(
    trainer,                 # GRPOTrainer instance
    step: int,
    config,                  # GRPOConfig
    logger,                  # ExperimentLogger
    checkpoint_dir: str,
    keep_checkpoints: int = 3,
) -> str:
    """Save a complete GRPO training checkpoint atomically."""
    ckpt_name = f"checkpoint_step_{step:06d}"
    ckpt_path = Path(checkpoint_dir) / ckpt_name
    tmp_path = Path(checkpoint_dir) / f".tmp_{ckpt_name}"

    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Policy model + tokenizer (a full HF dir so resume can from_pretrained it).
    model_dir = tmp_path / "model"
    trainer.model.save_pretrained(str(model_dir))
    trainer.tokenizer.save_pretrained(str(model_dir))

    # Optimizer (if the trainer keeps a persistent one — DPO does not, since it
    # spins up a fresh TRL trainer per step, so there's nothing to persist).
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), str(tmp_path / "optimizer.pt"))

    # Training counters (task/model recorded for a resume sanity check).
    training_state = {
        "step": step,
        "total_rollouts": trainer.total_rollouts,
        "seed": config.seed,
        "task": config.task,
        "model_name": config.model_name,
    }
    with open(tmp_path / "training_state.json", "w") as f:
        json.dump(training_state, f, indent=2)

    # Logger log (the metrics curve so far).
    with open(tmp_path / "logger_state.json", "w") as f:
        json.dump(logger.log, f, indent=2)

    # RNG states for reproducible continuation.
    rng_states = {
        "torch_rng": torch.random.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_states["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(rng_states, str(tmp_path / "rng_states.pt"))

    # Atomic publish.
    if ckpt_path.exists():
        shutil.rmtree(ckpt_path)
    tmp_path.rename(ckpt_path)
    print(f"[Checkpoint] Saved step {step} -> {ckpt_path}")

    if keep_checkpoints > 0:
        _rotate_checkpoints(str(checkpoint_dir), keep_checkpoints)

    return str(ckpt_path)


def load_grpo_checkpoint(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    """Load a GRPO checkpoint. Returns the state needed to resume.

    The model itself is reloaded by the trainer factory from
    ``state["model_path"]``; here we return the path plus optimizer/logger/RNG.
    Optimizer state is loaded to CPU and moved per-tensor by load_state_dict.
    """
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(ckpt / "training_state.json") as f:
        training_state = json.load(f)
    with open(ckpt / "logger_state.json") as f:
        logger_log = json.load(f)

    opt_path = ckpt / "optimizer.pt"
    optimizer_state = (
        torch.load(str(opt_path), map_location="cpu", weights_only=True)
        if opt_path.exists() else None
    )
    rng_states = torch.load(
        str(ckpt / "rng_states.pt"), map_location="cpu", weights_only=False
    )

    return {
        "step": training_state["step"],
        "total_rollouts": training_state["total_rollouts"],
        "task": training_state.get("task"),
        "model_name": training_state.get("model_name"),
        "model_path": str(ckpt / "model"),
        "optimizer_state_dict": optimizer_state,
        "logger_log": logger_log,
        "rng_states": rng_states,
    }


def _rotate_checkpoints(checkpoint_dir: str, keep: int) -> None:
    ckpt_dir = Path(checkpoint_dir)
    dirs = sorted(
        [d for d in ckpt_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint_step_")],
        key=lambda d: int(d.name.split("_")[-1]),
    )
    while len(dirs) > keep:
        oldest = dirs.pop(0)
        shutil.rmtree(oldest)
        print(f"[Checkpoint] Rotated out {oldest.name}")
