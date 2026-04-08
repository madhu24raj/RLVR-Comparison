# Checkpointing, Fault Tolerance, and Observability Spec

**Status:** Implemented  
**Author:** Reliability Engineering  
**Date:** 2026-04-08  
**Scope:** `ppo_specs/` PPO training pipeline (E2.7 and E2.8 experiments)

### Implementation Status

The core checkpointing system is implemented in `ppo_specs/checkpoint.py`:

| Feature | Status |
|---------|--------|
| Atomic saves (write to tmp dir, then rename) | **Implemented** |
| Checkpoint rotation (keep last K) | **Implemented** |
| Full state save (model, critic, optimizers, RNG, logger, counters) | **Implemented** |
| Config compatibility verification on resume | **Implemented** |
| Signal handling (`GracefulExitHandler` for SIGTERM/SIGINT) | **Implemented** |
| `find_latest_checkpoint` for auto-resume | **Implemented** |
| `restore_rng_states` for reproducibility | **Implemented** |
| Config fields (`checkpoint_every`, `keep_checkpoints`, `checkpoint_dir`, `resume_from`) | **Implemented** in `config.py` |
| W&B integration | **Spec only** -- config fields exist (`use_wandb`, `wandb_project`, etc.) but logger wrapper not yet implemented |
| Observability dashboard | **Spec only** |

Tests: `tests/test_checkpoint.py` verifies save/load/resume and checkpoint rotation.

---

## 1. Current State Audit

### 1.1 Checkpointing: None

The pipeline has **zero checkpoint or resume capability**. Relevant code paths:

- `PPOTrainer.__init__` initialises `self.step = 0` and `self.total_rollouts = 0` unconditionally.
- `run_e2_7.py` and `run_e2_8.py` call `load_ppo_trainer()` which always loads fresh pretrained weights from HuggingFace.
- `ExperimentLogger` starts with an empty `self.log: list[dict] = []` every run.
- No filesystem scan for prior checkpoints. No `--resume` flag.

### 1.2 State Lost on Crash

A crash or preemption at step N destroys:

| Component | Location in Code | Impact |
|-----------|-----------------|--------|
| Policy model weights | `PPOTrainer.model` (fine-tuned CausalLM) | N gradient updates lost |
| Critic weights | `PPOTrainer.critic` (MLP or linear head) | Critic must relearn value function from scratch |
| Policy optimizer state | `PPOTrainer.policy_optimizer` (AdamW) | Momentum/variance accumulators lost; warm restart != cold restart |
| Critic optimizer state | `PPOTrainer.critic_optimizer` (AdamW, if trainable) | Same as above |
| Training counters | `PPOTrainer.step`, `PPOTrainer.total_rollouts` | Metrics mis-indexed on manual restart |
| Logged metrics | `ExperimentLogger.log` | Only persisted at end via `logger.save()` |
| RNG states | `torch`, `numpy`, `random`, CUDA RNG | Reproducibility broken; different data ordering on restart |

### 1.3 Runtime Estimates

| Experiment | Steps | Batch Size | Est. Time (A100) | Est. Time (V100) |
|------------|-------|------------|-------------------|-------------------|
| E2.7 | 200 | 16 | ~2 hours | ~4 hours |
| E2.8 (per capacity) | 150 | 16 | ~1.25 hours | ~2.5 hours |
| E2.8 (full sweep, 4 capacities) | 600 total | 16 | ~5 hours | ~10 hours |

A SLURM preemption at step 180 of E2.7 wastes ~1.8 hours. For E2.8 a crash during the third capacity run wastes ~3.75 hours.

### 1.4 Additional Gaps

- **No observability**: No wandb/tensorboard integration. Metrics are only visible in stdout and a final JSON dump.
- **No signal handling**: SLURM sends SIGTERM 30-120 seconds before SIGKILL. The pipeline does not catch this.
- **Logger flush**: `ExperimentLogger.save()` is only called once at the end of training. A crash loses all logged metrics.

---

## 2. Checkpoint Design

### 2.1 What to Save

```
checkpoint_step_0100/
    config.json              # PPOConfig as JSON (for compatibility verification)
    model/                   # HuggingFace save_pretrained format
        config.json
        model.safetensors    # (or pytorch_model.bin)
    critic.pt                # critic.state_dict()
    policy_optimizer.pt      # policy_optimizer.state_dict()
    critic_optimizer.pt      # critic_optimizer.state_dict() (if trainable critic)
    training_state.json      # step, total_rollouts, seed, config hash
    logger_state.json        # ExperimentLogger.log (all entries so far)
    rng_states.pt            # torch, numpy, random, CUDA RNG states
```

### 2.2 Checkpoint Frequency

- Default: every `checkpoint_every` steps (new config field, default 20 for E2.7, 25 for E2.8).
- Always checkpoint on the final step.
- Emergency checkpoint on SIGTERM/SIGINT (see Section 5).

### 2.3 Checkpoint Rotation

Keep the last `keep_checkpoints` checkpoints (default 3). Older checkpoints are deleted to save disk space. The emergency checkpoint does not count toward this limit.

### 2.4 Checkpoint Naming Convention

```
{output_dir}/checkpoints/{experiment_name}/checkpoint_step_{step:06d}/
```

Example: `results/checkpoints/ppo_e2_7_seed42/checkpoint_step_000100/`

### 2.5 Config Additions

```python
@dataclass
class PPOConfig:
    # ... existing fields ...

    # ── Checkpointing ───────────────────────────────────────────────────────
    checkpoint_every: int = 20          # save checkpoint every N steps
    keep_checkpoints: int = 3           # keep last K checkpoints (0 = keep all)
    checkpoint_dir: str = "results/checkpoints"
    resume_from: str = ""               # path to checkpoint dir, or "auto"

    # ── Wandb ───────────────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "rlvr-comparison"
    wandb_group: str = ""               # e.g. "e2_7" or "e2_8_sweep"
    wandb_run_name: str = ""            # defaults to experiment_name
    wandb_run_id: str = ""              # set for resume; empty = generate new
```

---

## 3. Checkpoint Save/Load Implementation

### 3.1 `save_checkpoint`

```python
"""
Checkpoint save/load utilities for PPO training.

Usage:
    from ppo_specs.checkpoint import save_checkpoint, load_checkpoint
"""

import json
import os
import shutil
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ppo_specs.config import PPOConfig
from eval.metrics import ExperimentLogger


def save_checkpoint(
    trainer,            # PPOTrainer instance
    step: int,
    config: PPOConfig,
    logger: ExperimentLogger,
    checkpoint_dir: str,
    keep_checkpoints: int = 3,
) -> str:
    """
    Save a complete training checkpoint to disk.

    Args:
        trainer:          PPOTrainer with model, critic, optimizers.
        step:             Current training step (0-indexed).
        config:           PPOConfig for this run.
        logger:           ExperimentLogger with metrics logged so far.
        checkpoint_dir:   Base directory for checkpoints of this experiment.
        keep_checkpoints: Number of recent checkpoints to retain (0 = keep all).

    Returns:
        Path to the saved checkpoint directory.
    """
    ckpt_name = f"checkpoint_step_{step:06d}"
    ckpt_path = Path(checkpoint_dir) / ckpt_name
    tmp_path = Path(checkpoint_dir) / f".tmp_{ckpt_name}"

    # Write to a temporary directory first, then atomically rename.
    # This prevents a partially-written checkpoint from being loaded
    # after a crash during save.
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Model weights (HuggingFace format for easy reload) ────────────
    model_dir = tmp_path / "model"
    trainer.model.save_pretrained(str(model_dir))
    trainer.tokenizer.save_pretrained(str(model_dir))

    # ── 2. Critic state dict ─────────────────────────────────────────────
    torch.save(trainer.critic.state_dict(), str(tmp_path / "critic.pt"))

    # ── 3. Optimizer states ──────────────────────────────────────────────
    torch.save(
        trainer.policy_optimizer.state_dict(),
        str(tmp_path / "policy_optimizer.pt"),
    )
    if trainer.critic_optimizer is not None:
        torch.save(
            trainer.critic_optimizer.state_dict(),
            str(tmp_path / "critic_optimizer.pt"),
        )

    # ── 4. Training state ────────────────────────────────────────────────
    training_state = {
        "step": step,
        "total_rollouts": trainer.total_rollouts,
        "seed": config.seed,
        "config_hash": _config_hash(config),
    }
    with open(tmp_path / "training_state.json", "w") as f:
        json.dump(training_state, f, indent=2)

    # ── 5. Logger state (all metrics so far) ─────────────────────────────
    with open(tmp_path / "logger_state.json", "w") as f:
        json.dump(logger.log, f, indent=2)

    # ── 6. RNG states ────────────────────────────────────────────────────
    rng_states = {
        "torch_rng": torch.random.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_states["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(rng_states, str(tmp_path / "rng_states.pt"))

    # ── 7. Config (for compatibility checking on resume) ─────────────────
    with open(tmp_path / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # ── Atomic rename ────────────────────────────────────────────────────
    if ckpt_path.exists():
        shutil.rmtree(ckpt_path)
    tmp_path.rename(ckpt_path)

    print(f"[Checkpoint] Saved step {step} -> {ckpt_path}")

    # ── Checkpoint rotation ──────────────────────────────────────────────
    if keep_checkpoints > 0:
        _rotate_checkpoints(checkpoint_dir, keep_checkpoints)

    return str(ckpt_path)


def _config_hash(config: PPOConfig) -> str:
    """Deterministic hash of config fields that must match on resume."""
    import hashlib
    key_fields = {
        "model_name": config.model_name,
        "critic_capacity": config.critic_capacity,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "clip_epsilon": config.clip_epsilon,
    }
    raw = json.dumps(key_fields, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _rotate_checkpoints(checkpoint_dir: str, keep: int) -> None:
    """Delete oldest checkpoints, keeping only the most recent `keep`."""
    ckpt_dir = Path(checkpoint_dir)
    # List only completed checkpoints (not tmp dirs or emergency)
    dirs = sorted(
        [d for d in ckpt_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint_step_")],
        key=lambda d: int(d.name.split("_")[-1]),
    )
    while len(dirs) > keep:
        oldest = dirs.pop(0)
        shutil.rmtree(oldest)
        print(f"[Checkpoint] Rotated out {oldest.name}")
```

### 3.2 `load_checkpoint`

```python
def load_checkpoint(
    ckpt_path: str,
    config: PPOConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load a checkpoint and return all state needed to resume training.

    Args:
        ckpt_path: Path to a checkpoint_step_NNNNNN directory.
        config:    PPOConfig for this run (used for compatibility check).
        device:    Target device (e.g. torch.device("cuda")).

    Returns:
        Dict with keys:
            "step"              : int     - step to resume FROM (next step = step + 1)
            "total_rollouts"    : int
            "model"             : AutoModelForCausalLM (on device)
            "tokenizer"         : AutoTokenizer
            "critic_state_dict" : dict
            "policy_optim_state": dict
            "critic_optim_state": dict or None
            "logger_log"        : list[dict]
            "rng_states"        : dict
            "saved_config"      : dict
    """
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── Compatibility check ──────────────────────────────────────────────
    with open(ckpt / "config.json") as f:
        saved_config = json.load(f)

    with open(ckpt / "training_state.json") as f:
        training_state = json.load(f)

    current_hash = _config_hash(config)
    saved_hash = training_state.get("config_hash", "")
    if current_hash != saved_hash:
        mismatches = []
        for key in ["model_name", "critic_capacity", "batch_size"]:
            saved_val = saved_config.get(key)
            current_val = getattr(config, key, None)
            if saved_val != current_val:
                mismatches.append(f"  {key}: saved={saved_val!r} vs current={current_val!r}")
        raise ValueError(
            f"Config mismatch between checkpoint and current config:\n"
            + "\n".join(mismatches)
            + "\nCheckpoint is incompatible. Use --resume-from with a matching config."
        )

    # ── Model + tokenizer ────────────────────────────────────────────────
    model_dir = str(ckpt / "model")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # ── Critic state dict ────────────────────────────────────────────────
    critic_state_dict = torch.load(
        str(ckpt / "critic.pt"), map_location=device, weights_only=True,
    )

    # ── Optimizer states ─────────────────────────────────────────────────
    policy_optim_state = torch.load(
        str(ckpt / "policy_optimizer.pt"), map_location=device, weights_only=True,
    )

    critic_optim_path = ckpt / "critic_optimizer.pt"
    critic_optim_state = None
    if critic_optim_path.exists():
        critic_optim_state = torch.load(
            str(critic_optim_path), map_location=device, weights_only=True,
        )

    # ── Logger state ─────────────────────────────────────────────────────
    with open(ckpt / "logger_state.json") as f:
        logger_log = json.load(f)

    # ── RNG states ───────────────────────────────────────────────────────
    rng_states = torch.load(
        str(ckpt / "rng_states.pt"), map_location="cpu", weights_only=False,
    )

    print(
        f"[Checkpoint] Loaded step {training_state['step']} "
        f"(total_rollouts={training_state['total_rollouts']}) from {ckpt_path}"
    )

    return {
        "step": training_state["step"],
        "total_rollouts": training_state["total_rollouts"],
        "model": model,
        "tokenizer": tokenizer,
        "critic_state_dict": critic_state_dict,
        "policy_optim_state": policy_optim_state,
        "critic_optim_state": critic_optim_state,
        "logger_log": logger_log,
        "rng_states": rng_states,
        "saved_config": saved_config,
    }


def restore_rng_states(rng_states: Dict[str, Any]) -> None:
    """Restore all RNG states for reproducibility after resume."""
    torch.random.set_rng_state(rng_states["torch_rng"])

    # numpy get_state/set_state uses a tuple; torch.save may convert to list
    np_state = rng_states["numpy_rng"]
    if isinstance(np_state, dict):
        # Handle the case where numpy state was saved as a dict
        np.random.set_state(tuple(np_state.values()))
    else:
        np.random.set_state(np_state)

    random.setstate(rng_states["python_rng"])

    if torch.cuda.is_available() and "cuda_rng" in rng_states:
        torch.cuda.set_rng_state_all(rng_states["cuda_rng"])


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory (for auto-resume).

    Returns:
        Path to the latest checkpoint dir, or None if no checkpoints exist.
    """
    ckpt_base = Path(checkpoint_dir)
    if not ckpt_base.exists():
        return None

    dirs = sorted(
        [d for d in ckpt_base.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint_step_")],
        key=lambda d: int(d.name.split("_")[-1]),
    )
    if not dirs:
        return None
    return str(dirs[-1])
```

---

## 4. Modified Training Loops

### 4.1 `run_e2_7.py` with Checkpoint and Wandb Integration

```python
"""
E2.7 with checkpointing, resume, and wandb support.

Usage:
    # Fresh run
    python ppo_specs/run_e2_7.py --seed 0

    # Resume from latest checkpoint (auto-detect)
    python ppo_specs/run_e2_7.py --seed 0 --resume auto

    # Resume from specific checkpoint
    python ppo_specs/run_e2_7.py --seed 0 --resume-from results/checkpoints/ppo_e2_7_seed0/checkpoint_step_000100

    # With wandb
    python ppo_specs/run_e2_7.py --seed 0 --wandb
"""

import sys, os, signal, argparse, json
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import random
import torch
import numpy as np
from transformers import set_seed as transformers_set_seed

from src.data import load_gsm8k, format_prompt
from eval.metrics import ExperimentLogger
from ppo_specs.config import PPOConfig, local_test_config, e2_7_config
from ppo_specs.ppo_trainer import load_ppo_trainer
from ppo_specs.advantage import estimate_mc_advantages, advantage_estimation_error
from ppo_specs.utils import cycle_batch
from ppo_specs.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    restore_rng_states,
    find_latest_checkpoint,
)


# ── Global flag for graceful shutdown ────────────────────────────────────────

_SHUTDOWN_REQUESTED = False

def _signal_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    sig_name = signal.Signals(signum).name
    print(f"\n[Signal] Received {sig_name}. Will save emergency checkpoint after current step.")
    _SHUTDOWN_REQUESTED = True


def run_e2_7(config: PPOConfig, compute_mc: bool = True, use_wandb: bool = False) -> None:
    global _SHUTDOWN_REQUESTED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[E2.7] Device: {device}")

    # ── Signal handling ──────────────────────────────────────────────────
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # ── Determine resume state ───────────────────────────────────────────
    start_step = 0
    resumed = False
    checkpoint_base = Path(config.checkpoint_dir) / config.experiment_name
    ckpt_state = None

    if config.resume_from == "auto":
        latest = find_latest_checkpoint(str(checkpoint_base))
        if latest:
            config.resume_from = latest
            print(f"[E2.7] Auto-detected checkpoint: {latest}")
        else:
            print("[E2.7] No checkpoint found; starting fresh.")
            config.resume_from = ""

    if config.resume_from:
        ckpt_state = load_checkpoint(config.resume_from, config, device)
        start_step = ckpt_state["step"] + 1
        resumed = True
        print(f"[E2.7] Resuming from step {start_step}")
    else:
        # Fresh start: seed all RNGs
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        transformers_set_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    # ── Data ─────────────────────────────────────────────────────────────
    print("[E2.7] Loading GSM8K ...")
    train_ds = load_gsm8k("train", n_samples=config.n_train_samples, seed=config.seed)
    test_ds  = load_gsm8k("test",  n_samples=200)
    train_prompts = [format_prompt(ex["question"]) for ex in train_ds]
    train_gts     = [ex["ground_truth"] for ex in train_ds]
    test_prompts  = [format_prompt(ex["question"]) for ex in test_ds]
    test_gts      = [ex["ground_truth"] for ex in test_ds]

    # ── Build or restore trainer ─────────────────────────────────────────
    if resumed and ckpt_state is not None:
        # Build trainer with restored model (already on device)
        from ppo_specs.critic import build_critic
        from src.rewards import gsm8k_reward

        hidden_size = ckpt_state["model"].config.hidden_size
        critic = build_critic(config.critic_capacity, hidden_size).to(device)
        critic.load_state_dict(ckpt_state["critic_state_dict"])

        trainer = PPOTrainer(
            config=config,
            model=ckpt_state["model"],
            tokenizer=ckpt_state["tokenizer"],
            critic=critic,
            reward_fn=gsm8k_reward,
            device=device,
        )
        # Restore optimizer states
        trainer.policy_optimizer.load_state_dict(ckpt_state["policy_optim_state"])
        if trainer.critic_optimizer and ckpt_state["critic_optim_state"]:
            trainer.critic_optimizer.load_state_dict(ckpt_state["critic_optim_state"])

        # Restore counters
        trainer.step = ckpt_state["step"]
        trainer.total_rollouts = ckpt_state["total_rollouts"]

        # Restore RNG
        restore_rng_states(ckpt_state["rng_states"])
    else:
        trainer = load_ppo_trainer(config, device)

    # ── Logger (restore or create) ───────────────────────────────────────
    logger = ExperimentLogger(config.experiment_name, config.output_dir)
    if resumed and ckpt_state is not None:
        logger.log = ckpt_state["logger_log"]

    # ── Wandb ────────────────────────────────────────────────────────────
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_kwargs = {
            "project": config.wandb_project or "rlvr-comparison",
            "group": config.wandb_group or "e2_7",
            "name": config.wandb_run_name or config.experiment_name,
            "config": asdict(config),
            "resume": "allow",  # allows resuming existing run
        }
        if config.wandb_run_id:
            wandb_kwargs["id"] = config.wandb_run_id
        elif resumed and ckpt_state:
            # Try to recover wandb run id from saved config
            saved_id = ckpt_state["saved_config"].get("wandb_run_id", "")
            if saved_id:
                wandb_kwargs["id"] = saved_id
        wandb_run = wandb.init(**wandb_kwargs)
        # Store run ID in config so it's saved in next checkpoint
        config.wandb_run_id = wandb_run.id

    # ── MC baselines ─────────────────────────────────────────────────────
    mc_baselines: dict = {}
    if compute_mc:
        n_mc = 10 if config.n_steps <= 10 else 50
        ref_p  = train_prompts[:5]
        ref_gt = train_gts[:5]
        print(f"[E2.7] Estimating MC baselines ({n_mc} samples x {len(ref_p)} prompts) ...")
        mc_baselines = estimate_mc_advantages(
            trainer.model, trainer.tokenizer,
            ref_p, ref_gt, trainer.reward_fn,
            n_samples=n_mc,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            device=str(device),
        )

    # ── Training loop ────────────────────────────────────────────────────
    reward_window: list[float] = []

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

        # ── Periodic evaluation ──────────────────────────────────────────
        if step % config.eval_every == 0:
            test_acc = trainer.evaluate(test_prompts, test_gts, n_eval=20)

            window = (reward_window[-config.eval_every:]
                      if len(reward_window) >= config.eval_every
                      else reward_window)
            stability = float(np.var(window))

            adv_error = None
            if mc_baselines:
                mc_vals = np.array(list(mc_baselines.values()))
                ref_p_for_eval = list(mc_baselines.keys())
                if config.critic_capacity == "none":
                    saved_rollouts = trainer.total_rollouts
                    ref_batch = trainer.generate_rollouts(ref_p_for_eval, train_gts[:5])
                    trainer.total_rollouts = saved_rollouts
                    ref_mean = float(ref_batch.rewards().mean())
                    est = np.full(len(mc_vals), ref_mean)
                else:
                    est = trainer._eval_critic_on_prompts(ref_p_for_eval)
                adv_error = advantage_estimation_error(est, mc_vals)

            log_entry = {
                "total_rollouts": metrics["total_rollouts"],
                "train_accuracy": metrics["accuracy"],
                "test_accuracy":  test_acc,
                "reward_variance": stability,
                "policy_loss":    metrics["policy_loss"],
                "critic_loss":    metrics["critic_loss"],
                "clip_fraction":  metrics["clip_fraction"],
            }
            if adv_error is not None:
                log_entry["advantage_error"] = adv_error

            logger.log_step(step, **log_entry)

            # ── Wandb logging ────────────────────────────────────────────
            if wandb_run:
                wandb_metrics = {
                    "step": step,
                    "train/reward_mean": metrics["mean_reward"],
                    "train/accuracy": metrics["accuracy"],
                    "train/policy_loss": metrics["policy_loss"],
                    "train/critic_loss": metrics["critic_loss"],
                    "train/kl_divergence": metrics["kl_divergence"],
                    "train/clip_fraction": metrics["clip_fraction"],
                    "train/mean_advantage": metrics["mean_advantage"],
                    "train/reward_variance": metrics["reward_variance"],
                    "eval/test_accuracy": test_acc,
                    "eval/stability_variance": stability,
                    "rollouts/total": metrics["total_rollouts"],
                }
                if adv_error is not None:
                    wandb_metrics["eval/advantage_error"] = adv_error
                wandb.log(wandb_metrics, step=step)

            print(
                f"    -> test_acc={test_acc:.3f} "
                f"| stability(var)={stability:.4f}"
                + (f" | adv_error={adv_error:.4f}" if adv_error is not None else "")
            )

        # ── Periodic checkpoint ──────────────────────────────────────────
        is_checkpoint_step = (
            (step > 0 and step % config.checkpoint_every == 0)
            or step == config.n_steps - 1    # always save on last step
        )
        if is_checkpoint_step:
            save_checkpoint(
                trainer, step, config, logger,
                checkpoint_dir=str(checkpoint_base),
                keep_checkpoints=config.keep_checkpoints,
            )
            # Flush logger to disk as well (independent of checkpoint)
            logger.save()

        # ── Emergency shutdown ───────────────────────────────────────────
        if _SHUTDOWN_REQUESTED:
            print("[E2.7] Saving emergency checkpoint before exit...")
            emergency_dir = str(checkpoint_base / "emergency")
            save_checkpoint(
                trainer, step, config, logger,
                checkpoint_dir=str(checkpoint_base),
                keep_checkpoints=0,  # don't rotate; save unconditionally
            )
            logger.save()
            if wandb_run:
                wandb.finish(exit_code=1)
            print(f"[E2.7] Emergency checkpoint saved. Exiting at step {step}.")
            sys.exit(0)

    logger.save()

    # ── Final evaluation ─────────────────────────────────────────────────
    final_acc = trainer.evaluate(test_prompts, test_gts, n_eval=50)
    print(f"\n[E2.7] Final test accuracy (PPO, {config.critic_capacity} critic): {final_acc:.3f}")

    if wandb_run:
        wandb.log({"eval/final_test_accuracy": final_acc})
        wandb.finish()


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataclasses import asdict
    from ppo_specs.ppo_trainer import PPOTrainer

    parser = argparse.ArgumentParser(description="E2.7: PPO head-to-head on GSM8K")
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mc", action="store_true")
    parser.add_argument(
        "--resume-from", type=str, default="",
        help="Path to checkpoint dir, or 'auto' to find latest.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()

    cfg = local_test_config() if args.local_test else e2_7_config(seed=args.seed)
    cfg.seed = args.seed
    cfg.checkpoint_every = args.checkpoint_every

    if args.resume_from:
        cfg.resume_from = args.resume_from

    run_e2_7(cfg, compute_mc=not args.no_mc, use_wandb=args.wandb)
```

---

## 5. Fault Tolerance Patterns

### 5.1 SIGTERM Handling (SLURM Preemption)

SLURM sends SIGTERM 30-120 seconds (configurable via `--signal`) before SIGKILL. The signal handler sets a global flag; the training loop checks it after each step and performs an orderly save.

```python
# Already integrated into the training loop above.
# Key behavior:
# 1. Signal sets _SHUTDOWN_REQUESTED = True
# 2. Current step completes (no mid-step interruption)
# 3. Emergency checkpoint is saved
# 4. Logger is flushed
# 5. Wandb run is marked as crashed (exit_code=1)
# 6. Process exits cleanly with code 0
```

SLURM job script pattern:

```bash
#!/bin/bash
#SBATCH --signal=B:SIGTERM@120     # send SIGTERM 120s before kill
#SBATCH --requeue                   # auto-requeue on preemption
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1

python ppo_specs/run_e2_7.py \
    --seed 0 \
    --resume-from auto \
    --wandb \
    --checkpoint-every 20
```

### 5.2 Atomic Checkpoints

The save function writes to a `.tmp_` directory first, then performs an atomic rename. This prevents the following failure mode:

1. Checkpoint write begins
2. Crash mid-write
3. Next resume finds a corrupted, partially-written checkpoint

With atomic rename, the checkpoint directory either exists fully or not at all.

### 5.3 Idempotent Step Execution

**Question: Can a step be safely re-run?**

**Yes, with caveats.**

- Each step generates rollouts, computes gradients, and updates weights. Re-running step N after a crash that happened mid-step will:
  - Re-generate rollouts (non-deterministic unless RNG state is restored)
  - Re-compute and re-apply the gradient update

- With RNG restoration from the checkpoint at step N-1, re-running step N produces identical results (bit-exact on CPU, near-identical on GPU due to non-deterministic CUDA ops).

- Without RNG restoration, step N will use different rollouts but the training is still correct -- just not bit-for-bit reproducible.

- The `total_rollouts` counter is restored from the checkpoint, so it remains consistent.

### 5.4 Elastic Training (Future)

For multi-node training, `torch.distributed.elastic` (torchrun) provides:

- Automatic worker restart on failure
- Dynamic scaling (add/remove workers)
- Rendezvous for rank assignment

This is out of scope for the current single-GPU pipeline but the checkpoint format is compatible: each rank would save its own shard, and `load_checkpoint` would load the appropriate shard.

```bash
# Future multi-GPU launch:
torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:29500 \
    ppo_specs/run_e2_7.py --seed 0 --resume-from auto
```

---

## 6. Wandb Integration Design

### 6.1 Project Structure

```
wandb project: rlvr-comparison
    group: e2_7           # all E2.7 runs
        run: ppo_e2_7_seed0
        run: ppo_e2_7_seed1
        run: ppo_e2_7_seed2
    group: e2_8_sweep      # all E2.8 runs
        run: ppo_e2_8_none_seed0
        run: ppo_e2_8_small_seed0
        run: ppo_e2_8_medium_seed0
        run: ppo_e2_8_large_seed0
```

### 6.2 Metrics Logged

| Metric Key | Source | Frequency |
|-----------|--------|-----------|
| `train/reward_mean` | `metrics["mean_reward"]` | Every eval step |
| `train/accuracy` | `metrics["accuracy"]` | Every eval step |
| `train/policy_loss` | `metrics["policy_loss"]` | Every eval step |
| `train/critic_loss` | `metrics["critic_loss"]` | Every eval step |
| `train/kl_divergence` | `metrics["kl_divergence"]` | Every eval step |
| `train/clip_fraction` | `metrics["clip_fraction"]` | Every eval step |
| `train/mean_advantage` | `metrics["mean_advantage"]` | Every eval step |
| `train/reward_variance` | `metrics["reward_variance"]` | Every eval step |
| `eval/test_accuracy` | `trainer.evaluate()` | Every eval step |
| `eval/stability_variance` | Rolling reward variance | Every eval step |
| `eval/advantage_error` | `advantage_estimation_error()` | Every eval step (if MC enabled) |
| `eval/final_test_accuracy` | Final evaluation | End of training |
| `rollouts/total` | `trainer.total_rollouts` | Every eval step |

### 6.3 Run Resumption

Wandb run IDs are stored in the checkpoint's `config.json` under `wandb_run_id`. On resume:

1. The saved `wandb_run_id` is loaded from the checkpoint.
2. `wandb.init(resume="allow", id=saved_id)` reconnects to the same run.
3. Metrics continue from where they left off (wandb deduplicates by step number).

### 6.4 Config Logging

The full `PPOConfig` is logged as `wandb.config` at init time, enabling hyperparameter comparison in the wandb dashboard.

---

## 7. Implementation Checklist

### Phase 1: Core Checkpointing (Priority: Critical)

- [ ] Create `ppo_specs/checkpoint.py` with `save_checkpoint`, `load_checkpoint`, `restore_rng_states`, `find_latest_checkpoint`
- [ ] Add checkpoint fields to `PPOConfig` (`checkpoint_every`, `keep_checkpoints`, `checkpoint_dir`, `resume_from`)
- [ ] Modify `run_e2_7.py` training loop to save/resume checkpoints
- [ ] Modify `run_e2_8.py` training loop to save/resume checkpoints (per-capacity)
- [ ] Add `--resume-from` CLI argument to both run scripts
- [ ] Flush `ExperimentLogger` at each checkpoint (not just at end)

### Phase 2: Fault Tolerance (Priority: High)

- [ ] Add SIGTERM/SIGINT signal handler with emergency checkpoint
- [ ] Test atomic checkpoint save (kill during save, verify no corruption)
- [ ] Add SLURM job script with `--signal` and `--requeue`
- [ ] Verify RNG state restoration produces bit-exact results on CPU

### Phase 3: Observability (Priority: Medium)

- [ ] Add wandb fields to `PPOConfig`
- [ ] Integrate `wandb.init()` and `wandb.log()` in `run_e2_7.py`
- [ ] Integrate `wandb.init()` and `wandb.log()` in `run_e2_8.py`
- [ ] Test wandb run resumption with checkpoint resume
- [ ] Add `--wandb` flag to CLI

### Phase 4: Testing (Priority: High)

- [ ] Unit test: `save_checkpoint` / `load_checkpoint` round-trip
- [ ] Unit test: checkpoint rotation (keep_checkpoints=2, save 5 checkpoints)
- [ ] Unit test: config compatibility check rejects mismatched model_name
- [ ] Integration test: train 10 steps, crash, resume, verify final metrics match uninterrupted run
- [ ] Integration test: SIGTERM during training triggers emergency save

---

## 8. Disk Space Estimates

| Component | Size per Checkpoint (Qwen2.5-0.5B) | Size (Llama-3-8B) |
|-----------|------------------------------------|--------------------|
| Model weights | ~1 GB (fp32) / ~500 MB (bf16) | ~16 GB (fp32) / ~8 GB (bf16) |
| Critic weights | 1-10 MB | 1-50 MB |
| Optimizer states (AdamW, 2 buffers per param) | ~2 GB (fp32) | ~32 GB (fp32) |
| RNG states | <1 MB | <1 MB |
| Logger + config | <1 MB | <1 MB |
| **Total per checkpoint** | **~3 GB** | **~48 GB** |
| **3 checkpoints (default rotation)** | **~9 GB** | **~144 GB** |

**Recommendation for Llama-3-8B on cluster:** Use bf16 for model weights and consider optimizer state compression (e.g., save only the model and reinitialize optimizer on resume if disk is constrained -- at the cost of losing momentum).

---

## 9. Open Questions

1. **Optimizer state on resume vs cold restart:** Should we always restore optimizer momentum, or is it acceptable to re-initialize AdamW on resume? Restoring is strictly better for training continuity but costs 2x model size in disk. For short runs (200 steps) the momentum is unlikely to be critical.

2. **MC baseline caching:** The MC baseline estimation (50 samples x 5 prompts) takes ~5 minutes. Should we cache it in the checkpoint? Currently it is re-computed on resume, which is wasteful but ensures it uses the restored model state.

3. **Wandb offline mode:** On clusters without internet access during jobs, `wandb.init(mode="offline")` saves logs locally and syncs later. Should this be the default?

4. **Multi-seed parallelism:** E2.7 with 3 seeds could run in parallel. Should the checkpoint directory structure account for this (it already does via `experiment_name` which includes the seed)?
