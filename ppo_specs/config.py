"""
PPO experiment configuration.

Default model: Qwen/Qwen2.5-0.5B-Instruct (500 M params).
Swap model_name to meta-llama/Meta-Llama-3-8B-Instruct on the cluster.
"""

from dataclasses import dataclass, field
import dataclasses


@dataclass
class PPOConfig:
    # ── Model ────────────────────────────────────────────────────────────────
    # 0.5 B for local verification; replace with Llama-3-8B on the cluster.
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ── PPO hyperparameters ──────────────────────────────────────────────────
    learning_rate: float = 1e-5
    critic_lr: float = 1e-4
    clip_epsilon: float = 0.2       # PPO surrogate clipping
    gamma: float = 0.99             # discount (single-step episodes → effectively 1)
    n_ppo_epochs: int = 1           # gradient steps per collected batch
    kl_coeff: float = 0.0           # optional KL penalty weight (off by default)

    # ── Rollout settings ─────────────────────────────────────────────────────
    # E2.7 spec: PPO uses 1 rollout per prompt (plus critic).
    n_rollouts_per_prompt: int = 1
    batch_size: int = 8             # prompts per training step
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True

    # ── Critic architecture (E2.8 sweep) ────────────────────────────────────
    # "none"   → REINFORCE with batch-mean baseline
    # "small"  → 2-layer MLP
    # "medium" → single linear head (same depth as LM head)
    # "large"  → deep MLP with 2× hidden width
    critic_capacity: str = "medium"

    # ── Training schedule ────────────────────────────────────────────────────
    n_steps: int = 100
    eval_every: int = 10
    log_every: int = 5

    # ── Data ─────────────────────────────────────────────────────────────────
    n_train_samples: int = 200
    seed: int = 42

    # ── Bookkeeping ──────────────────────────────────────────────────────────
    experiment_name: str = "ppo_default"
    output_dir: str = "results"


# ── Preset configs ────────────────────────────────────────────────────────────

def local_test_config() -> PPOConfig:
    """Minimal config to verify the pipeline runs end-to-end on a laptop."""
    return PPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        n_steps=5,
        batch_size=4,
        max_new_tokens=64,
        n_train_samples=20,
        eval_every=2,
        log_every=1,
        experiment_name="ppo_local_test",
    )


def e2_7_config(seed: int = 42) -> PPOConfig:
    """Config for E2.7 head-to-head on GSM8K (cluster scale)."""
    return PPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        n_steps=200,
        batch_size=16,
        max_new_tokens=256,
        n_train_samples=500,
        eval_every=20,
        log_every=5,
        seed=seed,
        experiment_name=f"ppo_e2_7_seed{seed}",
    )


def e2_8_config(critic_capacity: str = "medium", seed: int = 42) -> PPOConfig:
    """Config for one cell of the E2.8 critic-quality sweep."""
    return PPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        critic_capacity=critic_capacity,
        n_steps=150,
        batch_size=16,
        max_new_tokens=256,
        n_train_samples=500,
        eval_every=20,
        log_every=10,
        seed=seed,
        experiment_name=f"ppo_e2_8_{critic_capacity}_seed{seed}",
    )


def copy_config(cfg: PPOConfig, **overrides) -> PPOConfig:
    """Return a copy of cfg with selected fields overridden."""
    return dataclasses.replace(cfg, **overrides)
