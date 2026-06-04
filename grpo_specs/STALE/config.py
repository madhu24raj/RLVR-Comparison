"""
Configuration for GRPO training on RLVR tasks.

GRPO (Group Relative Policy Optimization) replaces PPO's learned critic
with a group-based advantage estimator: generate G completions per prompt,
compute advantage as (reward - group_mean) / group_std.
"""
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    # -- Model --
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # -- GRPO hyperparameters --
    learning_rate: float = 1e-5
    clip_epsilon: float = 0.2
    n_ppo_epochs: int = 4           # K gradient steps per collected batch
    kl_coeff: float = 0.0           # weight on per-step KL(pi_old || pi_new)
    reference_kl_coeff: float = 0.0  # weight on KL(pi_new || pi_ref)

    # -- Group sampling --
    n_rollouts_per_prompt: int = 4  # G: completions per prompt for advantage
    batch_size: int = 4             # prompts per training step (total = batch_size * G)
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True

    # -- Schedule --
    n_steps: int = 200
    eval_every: int = 20
    log_every: int = 5
    eval_size: int = 100
    final_eval_size: int = 500

    # -- Data --
    n_train_samples: int = 500
    n_test_samples: int = 500
    seed: int = 42

    # -- Task --
    task: str = "gsm8k"             # "gsm8k" | "humaneval" (see src/tasks/)

    # -- Bookkeeping --
    experiment_name: str = "grpo"
    output_dir: str = "results"
    torch_dtype: str = "auto"

    # -- Checkpointing (survive disconnects / Slurm time limits) --
    checkpoint_every: int = 0           # save every N steps (0 = disabled)
    keep_checkpoints: int = 3           # keep last K periodic checkpoints (0 = keep all)
    checkpoint_dir: str = "results/checkpoints"
    resume_from: str = ""               # checkpoint dir path, or "auto" for latest


# -- Preset configs --

def local_test_config() -> GRPOConfig:
    """Minimal config to verify the GRPO pipeline runs end-to-end."""
    return GRPOConfig(
        n_steps=5,
        batch_size=4,
        n_rollouts_per_prompt=4,
        max_new_tokens=256,
        n_train_samples=20,
        n_test_samples=50,
        eval_size=10,
        final_eval_size=20,
        eval_every=2,
        log_every=1,
        experiment_name="grpo_local_test",
    )


def e2_7_config(seed: int = 42) -> GRPOConfig:
    """Config for E2.7 head-to-head on GSM8K (cluster scale)."""
    return GRPOConfig(
        n_steps=200,
        batch_size=16,
        n_rollouts_per_prompt=8,
        max_new_tokens=384,
        n_train_samples=500,
        n_test_samples=500,
        eval_size=100,
        final_eval_size=500,
        eval_every=20,
        log_every=5,
        reference_kl_coeff=0.01,
        seed=seed,
        experiment_name=f"grpo_e2_7_seed{seed}",
    )
