"""
Configuration for iterative DPO training on RLVR tasks.

Mirrors grpo_specs/STALE/config.py so the three methods share a consistent
surface for the E2.7 head-to-head. DPO generates several completions per
prompt, scores them with the (binary) verifiable reward, and forms
chosen/rejected preference pairs (correct vs incorrect) for a TRL DPO update.
"""
from dataclasses import dataclass


@dataclass
class DPOConfig:
    # -- Model --
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # -- DPO hyperparameters --
    learning_rate: float = 1e-5
    beta: float = 0.1               # DPO KL-penalty strength (TRL DPOConfig.beta)

    # -- Rollouts / pair construction --
    n_rollouts_per_prompt: int = 4  # completions per prompt (need a correct AND
                                    # an incorrect one to form a pair)
    batch_size: int = 4             # prompts per training step
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
    experiment_name: str = "dpo"
    output_dir: str = "results"
    torch_dtype: str = "auto"

    # -- Checkpointing (survive disconnects / Slurm time limits) --
    checkpoint_every: int = 0
    keep_checkpoints: int = 3
    checkpoint_dir: str = "results/checkpoints"
    resume_from: str = ""


def local_test_config() -> DPOConfig:
    """Minimal config to verify the DPO pipeline runs end-to-end."""
    return DPOConfig(
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
        experiment_name="dpo_local_test",
    )


def e2_7_config(seed: int = 42) -> DPOConfig:
    """Config for E2.7 head-to-head (cluster scale)."""
    return DPOConfig(
        n_steps=200,
        batch_size=16,
        n_rollouts_per_prompt=4,
        max_new_tokens=384,
        n_train_samples=500,
        n_test_samples=500,
        eval_size=100,
        final_eval_size=500,
        eval_every=20,
        log_every=5,
        seed=seed,
        experiment_name=f"dpo_e2_7_seed{seed}",
    )
