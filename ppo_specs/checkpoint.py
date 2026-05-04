"""
Checkpoint save/load utilities for PPO training.

Supports:
  - Atomic saves (write to tmp dir, then rename)
  - Checkpoint rotation (keep last K)
  - Full state: model, critic, optimizers, RNG states, logger, training counters
  - Config compatibility verification on resume
  - Signal handling for emergency checkpoints
"""

import json
import os
import signal
import shutil
import random
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ppo_specs.config import PPOConfig


def save_checkpoint(
    trainer,            # PPOTrainer instance
    step: int,
    config: PPOConfig,
    logger,             # ExperimentLogger
    checkpoint_dir: str,
    keep_checkpoints: int = 3,
    accelerator=None,   # Optional Accelerator for DDP unwrapping
) -> str:
    """Save a complete training checkpoint atomically.

    When ``accelerator`` is provided, models are unwrapped via
    ``accelerator.unwrap_model(...)`` before serialization so the saved
    state_dict has clean (un-prefixed) keys that load cleanly into either
    DDP-wrapped or unwrapped trainers. Without unwrapping, a DDP-wrapped
    module's ``state_dict()`` returns keys prefixed with ``"module."``,
    which a plain (unwrapped) trainer cannot ``load_state_dict`` directly.

    Backward compatibility: when ``accelerator`` is ``None`` (the default),
    behavior is identical to the pre-DDP code path — ``trainer.model`` and
    ``trainer.critic`` are serialized as-is.

    Rank-0 gating contract (caller responsibility):
        This function does NOT internally check ``accelerator.is_main_process``.
        Gating writes inside the function would deadlock multi-proc runs:
        non-rank-0 processes would early-return before reaching the next
        collective barrier while rank 0 is still writing. Callers in DDP
        runs MUST gate the call themselves and place a barrier before it::

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(trainer, ..., accelerator=accelerator)

        The optimizer state_dict is already correct under DDP (Accelerate-
        prepared optimizers expose the underlying optimizer's state_dict
        directly), so optimizer serialization needs no special handling.
    """
    def _unwrap(module):
        if accelerator is not None:
            return accelerator.unwrap_model(module)
        return module

    ckpt_name = f"checkpoint_step_{step:06d}"
    ckpt_path = Path(checkpoint_dir) / ckpt_name
    tmp_path = Path(checkpoint_dir) / f".tmp_{ckpt_name}"

    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Model weights (unwrap DDP if needed so saved keys have no 'module.' prefix)
    model_dir = tmp_path / "model"
    _unwrap(trainer.model).save_pretrained(str(model_dir))
    trainer.tokenizer.save_pretrained(str(model_dir))

    # Critic (unwrap DDP if needed)
    torch.save(_unwrap(trainer.critic).state_dict(), str(tmp_path / "critic.pt"))

    # Optimizers
    torch.save(trainer.policy_optimizer.state_dict(), str(tmp_path / "policy_optimizer.pt"))
    if trainer.critic_optimizer is not None:
        torch.save(trainer.critic_optimizer.state_dict(), str(tmp_path / "critic_optimizer.pt"))

    # Training state
    training_state = {
        "step": step,
        "total_rollouts": trainer.total_rollouts,
        "trainer_step": trainer.step,
        "seed": config.seed,
        "config_hash": _config_hash(config),
    }
    with open(tmp_path / "training_state.json", "w") as f:
        json.dump(training_state, f, indent=2)

    # Logger
    with open(tmp_path / "logger_state.json", "w") as f:
        json.dump(logger.log, f, indent=2)

    # RNG states
    rng_states = {
        "torch_rng": torch.random.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_states["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(rng_states, str(tmp_path / "rng_states.pt"))

    # Config snapshot
    with open(tmp_path / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Atomic rename
    if ckpt_path.exists():
        shutil.rmtree(ckpt_path)
    tmp_path.rename(ckpt_path)
    print(f"[Checkpoint] Saved step {step} -> {ckpt_path}")

    # Rotation
    if keep_checkpoints > 0:
        _rotate_checkpoints(str(checkpoint_dir), keep_checkpoints)

    return str(ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    config: PPOConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Load checkpoint and return all state needed to resume."""
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Verify config compatibility
    with open(ckpt / "config.json") as f:
        saved_config = json.load(f)
    saved_hash = json.load(open(ckpt / "training_state.json"))["config_hash"]
    current_hash = _config_hash(config)
    if saved_hash != current_hash:
        print(f"[Checkpoint] WARNING: Config hash mismatch. Saved: {saved_hash}, Current: {current_hash}")
        print(f"  Saved model_name: {saved_config.get('model_name')}, Current: {config.model_name}")

    # Training state
    with open(ckpt / "training_state.json") as f:
        training_state = json.load(f)

    # Logger state
    with open(ckpt / "logger_state.json") as f:
        logger_log = json.load(f)

    # Optimizer states
    # Load optimizer state to CPU first to avoid 2x optimizer-state transient
    # on GPU during load_state_dict (would cause 96 GB peak for 8B AdamW).
    # torch's optim.load_state_dict copies tensors to the optimizer's
    # parameter device per-tensor, so this is correct AND memory-safe.
    policy_opt_state = torch.load(str(ckpt / "policy_optimizer.pt"), map_location="cpu", weights_only=True)
    critic_opt_path = ckpt / "critic_optimizer.pt"
    critic_opt_state = torch.load(str(critic_opt_path), map_location="cpu", weights_only=True) if critic_opt_path.exists() else None

    # Critic state
    critic_state = torch.load(str(ckpt / "critic.pt"), map_location="cpu", weights_only=True)

    # RNG states
    rng_states = torch.load(str(ckpt / "rng_states.pt"), map_location="cpu", weights_only=False)

    return {
        "step": training_state["step"],
        "total_rollouts": training_state["total_rollouts"],
        "trainer_step": training_state.get("trainer_step", training_state["step"]),
        "model_path": str(ckpt / "model"),
        "critic_state_dict": critic_state,
        "policy_optimizer_state_dict": policy_opt_state,
        "critic_optimizer_state_dict": critic_opt_state,
        "logger_log": logger_log,
        "rng_states": rng_states,
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in a directory."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    dirs = sorted(
        [d for d in ckpt_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint_step_")],
        key=lambda d: int(d.name.split("_")[-1]),
    )
    return str(dirs[-1]) if dirs else None


def restore_rng_states(rng_states: Dict[str, Any]) -> None:
    """Restore all RNG states from a checkpoint."""
    torch.random.set_rng_state(rng_states["torch_rng"])
    np.random.set_state(rng_states["numpy_rng"])
    random.setstate(rng_states["python_rng"])
    if "cuda_rng" in rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_states["cuda_rng"])


def _config_hash(config: PPOConfig) -> str:
    # Use getattr with defaults so older PPOConfig instances (without these
    # fields) and pre-existing checkpoints continue to load.
    key_fields = {
        "model_name": config.model_name,
        "critic_capacity": config.critic_capacity,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "clip_epsilon": config.clip_epsilon,
        "reward_mode": getattr(config, "reward_mode", "deterministic"),
        "reference_kl_coeff": getattr(config, "reference_kl_coeff", 0.0),
        "n_ppo_epochs": config.n_ppo_epochs,
        "kl_coeff": config.kl_coeff,
        "reward_model_capacity": getattr(config, "reward_model_capacity", "none"),
        "reward_model_name": getattr(config, "reward_model_name", None),
        "reward_model_dtype": getattr(config, "reward_model_dtype", "auto"),
        "reward_model_reuse_reference": getattr(config, "reward_model_reuse_reference", False),
        "reward_blend_alpha": getattr(config, "reward_blend_alpha", 1.0),
        "reward_score_activation": getattr(config, "reward_score_activation", "sigmoid"),
        "optimizer_8bit": getattr(config, "optimizer_8bit", False),
        "reference_quant": getattr(config, "reference_quant", "none"),
    }
    raw = json.dumps(key_fields, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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


class GracefulExitHandler:
    """Handles SIGTERM/SIGINT for emergency checkpoint saves.

    Usage:
        handler = GracefulExitHandler()
        for step in range(n_steps):
            train(...)
            if handler.should_exit:
                save_checkpoint(...)
                break
    """
    def __init__(self):
        self.should_exit = False
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)
        # SLURM sends SIGUSR1 ahead of preemption (configured via
        # `#SBATCH --signal=B:SIGUSR1@<seconds>`). Linux only — Windows
        # does not define SIGUSR1, so we guard with hasattr to keep the
        # local-dev / Windows test environment working.
        self._original_sigusr1 = None
        if hasattr(signal, "SIGUSR1"):
            self._original_sigusr1 = signal.getsignal(signal.SIGUSR1)
            signal.signal(signal.SIGUSR1, self._handler)

    def _handler(self, signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[GracefulExit] Received {sig_name}. Will save checkpoint after current step.")
        self.should_exit = True

    def restore_signals(self):
        signal.signal(signal.SIGTERM, self._original_sigterm)
        signal.signal(signal.SIGINT, self._original_sigint)
        if hasattr(signal, "SIGUSR1") and self._original_sigusr1 is not None:
            signal.signal(signal.SIGUSR1, self._original_sigusr1)
