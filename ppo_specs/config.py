"""
PPO experiment configuration.

Default model: Qwen/Qwen2.5-0.5B-Instruct (500 M params).
Swap model_name to meta-llama/Meta-Llama-3-8B-Instruct on the cluster.
"""

from dataclasses import dataclass, field
import dataclasses
from typing import Optional


@dataclass
class PPOConfig:
    """
    Unified configuration for PPO training on RLVR tasks.

    Field groups
    ────────────
    Model          model_name
    PPO            learning_rate, critic_lr, clip_epsilon, gamma,
                   n_ppo_epochs, kl_coeff, critic_loss_coeff, grad_clip_norm
    Rollout        n_rollouts_per_prompt, batch_size, max_new_tokens,
                   temperature, do_sample, max_prompt_length
    Critic (E2.8)  critic_capacity  ("none" | "small" | "medium" | "large")
    Schedule       n_steps, eval_every, log_every, eval_size, final_eval_size
    Data           n_train_samples, n_test_samples, seed
    Bookkeeping    experiment_name, output_dir
    """
    # ── Model ────────────────────────────────────────────────────────────────
    # 0.5 B for local verification; replace with Llama-3-8B on the cluster.
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ── PPO hyperparameters ──────────────────────────────────────────────────
    learning_rate: float = 1e-5
    critic_lr: float = 1e-4
    clip_epsilon: float = 0.2       # PPO surrogate clipping
    gamma: float = 1.0              # single-step episodes; gamma is unused
    n_ppo_epochs: int = 4           # PPO epochs per collected batch (TRL/InstructGPT
                                     # standard). With per-token PPO loss the surrogate
                                     # is well-behaved at K>=2; at K=1 the ratio is
                                     # identically 1.0 on the first pass so the printed
                                     # policy_loss is ~0 (gradient is still real;
                                     # see policy_grad_norm metric).
    kl_coeff: float = 0.0           # weight on per-step KL(pi_old || pi_new),
                                     # the within-batch trust-region penalty
    reference_kl_coeff: float = 0.0  # weight on KL(pi_new || pi_ref) where pi_ref
                                     # is a frozen copy of the model from before
                                     # training started. Anchors the policy against
                                     # drift away from the initial distribution.
                                     # Default 0 = off (loads no extra model).
                                     # Standard RLHF uses ~0.01-0.1. Setting > 0
                                     # roughly doubles VRAM (loads a second model).
    critic_loss_coeff: float = 0.5  # weight on critic MSE loss in total_loss
    grad_clip_norm: float = 1.0     # max gradient norm for policy and critic
    log_ratio_clip: float = 20.0    # clamp log-ratio before exp() to prevent overflow

    # ── Reward model settings ────────────────────────────────────────────────
    reward_mode: str = "deterministic"   # "deterministic" | "self_judge" | "combined"
    self_judge_weight: float = 0.5       # weight for self_judge in combined mode (0-1)
    self_judge_normalize: bool = False   # raw log-probs for wider reward range

    # ── Learned reward model (additive on top of reward_mode) ───────────────
    # See ppo_specs/specs/reward_model_integration.md. Defaults reproduce
    # today's behavior bit-identically (capacity="none", alpha=1.0).
    reward_model_capacity: str = "none"          # "none" | "small" | "large"
    reward_model_name: Optional[str] = None      # HF hub id or local path; required if capacity != "none"
    reward_model_dtype: str = "auto"             # "auto" | "bfloat16" | "float32"
    reward_model_reuse_reference: bool = False   # share weights with the frozen reference model
    reward_blend_alpha: float = 1.0              # final = alpha * rm + (1 - alpha) * gsm8k_reward
    reward_score_activation: str = "sigmoid"     # "sigmoid" | "tanh" | "none"

    # ── Rollout settings ─────────────────────────────────────────────────────
    # E2.7 spec: PPO uses 1 rollout per prompt (plus critic).
    n_rollouts_per_prompt: int = 1
    batch_size: int = 8             # prompts per training step
    max_new_tokens: int = 256
    max_prompt_length: int = 512    # truncation limit for prompt tokenization
    temperature: float = 0.7
    do_sample: bool = True
    eval_batch_size: int = 8        # sub-batch size for batched evaluation

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

    # Number of test prompts used for in-loop evaluation during training.
    # n=20 (the previous hardcoded value) gives stderr ~0.11 at p=0.5,
    # which dominates the convergence-curve signal. n=100 -> stderr ~0.05.
    # See specs/logic.md L15.
    eval_size: int = 100
    # Number of test prompts used for the single final evaluation. Should
    # be as large as the test set (or budget) allows, since this is what
    # gets reported.
    final_eval_size: int = 200

    # ── Data ─────────────────────────────────────────────────────────────────
    n_train_samples: int = 200
    # Number of test prompts to load. Must be >= max(eval_size, final_eval_size).
    n_test_samples: int = 500
    seed: int = 42

    # ── Bookkeeping ──────────────────────────────────────────────────────────
    experiment_name: str = "ppo_default"
    output_dir: str = "results"

    # ── Dtype and Memory ────────────────────────────────────────────────────
    torch_dtype: str = "auto"           # "auto" | "float32" | "bfloat16" — auto = bf16 on GPU, fp32 on CPU
    gradient_checkpointing: bool = False # enable gradient checkpointing (required for 8B+ models)

    # ── Memory optimizations (cluster scale) ────────────────────────────────
    # 8-bit AdamW (bitsandbytes). Saves ~48 GB at 8B. <0.5% accuracy loss.
    # Falls back to torch.optim.AdamW if bitsandbytes is not installed.
    optimizer_8bit: bool = False
    # Use torch.optim.AdamW(fused=True) on CUDA. Saves ~16 GB transient
    # at 8B during .step() vs the default foreach=True path.
    optimizer_fused: bool = False
    # Quantize the FROZEN reference model to save memory. Reference is
    # never trained so quantization does not affect gradient quality.
    # Values: "none" (bf16/fp32 default), "int8" (bnb LLM.int8), "nf4" (bnb 4-bit).
    reference_quant: str = "none"
    # Length-bucketed generation. Sort prompts by length, generate per
    # bucket. Reduces pad-waste by ~40%. Bucket size in samples.
    length_bucketed_generation: bool = False
    generation_bucket_size: int = 4

    # ── Checkpointing ───────────────────────────────────────────────────────
    checkpoint_every: int = 20          # save checkpoint every N steps (0 = disabled)
    keep_checkpoints: int = 3           # keep last K checkpoints (0 = keep all)
    checkpoint_dir: str = "results/checkpoints"
    resume_from: str = ""               # path to checkpoint dir, or "auto" for latest

    # ── Logging ─────────────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "rlvr-comparison"
    wandb_group: str = ""
    wandb_run_name: str = ""


CRITIC_CAPACITIES: list[str] = ["none", "small", "medium", "large"]


# ── Preset configs ────────────────────────────────────────────────────────────

def local_test_config() -> PPOConfig:
    """Minimal config to verify the pipeline runs end-to-end on a laptop."""
    return PPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        n_steps=5,
        batch_size=4,
        max_new_tokens=256,
        n_train_samples=20,
        n_test_samples=50,        # smoke test only needs a tiny test pool
        eval_size=10,             # tiny: pipeline check, not signal
        final_eval_size=20,
        eval_every=2,
        log_every=1,
        checkpoint_every=0,       # no checkpointing for smoke tests
        experiment_name="ppo_local_test",
    )


def local_test_self_judge_config() -> PPOConfig:
    """Minimal self-judge config for local pipeline verification."""
    cfg = local_test_config()
    cfg.reward_mode = "self_judge"
    cfg.reference_kl_coeff = 0.01
    cfg.experiment_name = "ppo_local_test_self_judge"
    return cfg


def e2_7_config(seed: int = 42) -> PPOConfig:
    """Config for E2.7 head-to-head on GSM8K (cluster scale)."""
    return PPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        n_steps=200,
        batch_size=16,
        max_new_tokens=384,       # bumped from 256: GSM8K CoT often needs > 256 tokens
        n_train_samples=500,
        n_test_samples=500,
        eval_size=100,            # stderr ~0.05 at p=0.5
        final_eval_size=500,      # full reported number
        eval_every=20,
        log_every=5,
        # Reference KL anchor (L14). 0.01 follows DeepSeekMath's RLVR
        # default lowered slightly for our smaller model scale. Anchors
        # the policy against drift away from the initial distribution
        # over 200 steps; without it we observed clip_fraction climb to
        # ~0.5 and per-token KL to ~1.0 by step 24 of a local run.
        # Doubles model VRAM (loads a frozen reference copy).
        reference_kl_coeff=0.01,
        seed=seed,
        experiment_name=f"ppo_e2_7_seed{seed}",
    )


def e2_7_self_judge_config(seed: int = 42) -> PPOConfig:
    """Config for E2.7 with self-judge ORM reward (cluster scale)."""
    cfg = e2_7_config(seed)
    cfg.reward_mode = "self_judge"
    cfg.experiment_name = f"ppo_e2_7_self_judge_seed{seed}"
    return cfg


def e2_7_combined_config(seed: int = 42, weight: float = 0.5) -> PPOConfig:
    """Config for E2.7 with combined (deterministic + self-judge) reward."""
    cfg = e2_7_config(seed)
    cfg.reward_mode = "combined"
    cfg.self_judge_weight = weight
    cfg.experiment_name = f"ppo_e2_7_combined_w{weight}_seed{seed}"
    return cfg


def e2_8_config(critic_capacity: str = "medium", seed: int = 42) -> PPOConfig:
    """Config for one cell of the E2.8 critic-quality sweep."""
    return PPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        critic_capacity=critic_capacity,
        n_steps=150,
        batch_size=16,
        max_new_tokens=384,       # bumped from 256: GSM8K CoT often needs > 256 tokens
        n_train_samples=500,
        n_test_samples=500,
        eval_size=100,
        final_eval_size=500,
        eval_every=20,
        log_every=10,
        reference_kl_coeff=0.01,  # see e2_7_config for justification
        seed=seed,
        experiment_name=f"ppo_e2_8_{critic_capacity}_seed{seed}",
    )


def copy_config(cfg: PPOConfig, **overrides) -> PPOConfig:
    """Return a copy of cfg with selected fields overridden."""
    return dataclasses.replace(cfg, **overrides)
