"""
Advantage estimation utilities for PPO.

Implements:
  compute_advantages      – per-batch advantage with critic or batch-mean baseline
  estimate_mc_advantages  – Monte Carlo ground-truth baseline via many rollouts
  advantage_estimation_error – |Â - A_MC| as required by E2.7 measurement (iv)
"""

import torch
import numpy as np
from typing import Callable, Dict, List, Optional


# ── Advantage computation ────────────────────────────────────────────────────

def compute_advantages(
    rewards: torch.Tensor,
    values: Optional[torch.Tensor],
    gamma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute advantages for single-step (terminal-reward) episodes.

    For RLVR the episode ends after one generation step, so GAE reduces to:
        A_i = r_i - V(s_i)

    When values is None (capacity="none"), the batch mean serves as baseline:
        A_i = r_i - mean(r)

    Args:
        rewards:   [B] binary rewards {0, 1}
        values:    [B] critic estimates, or None for REINFORCE baseline
        gamma:     discount factor (retained for API compatibility; effectively
                   unused in the single-step case)
        normalize: z-score advantages for gradient variance reduction

    Returns:
        advantages: [B] float tensor
    """
    if values is None:
        baseline = rewards.mean()
        advantages = rewards - baseline
    else:
        advantages = rewards - values.detach()

    if normalize and advantages.numel() > 1:
        std = advantages.std()
        if std > 1e-8:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)
        else:
            # All advantages identical → zero-center to produce all-zeros,
            # which is the correct zero-gradient signal.
            advantages = advantages - advantages.mean()

    return advantages


# ── Monte Carlo ground-truth estimation ──────────────────────────────────────

def estimate_mc_advantages(
    policy,
    tokenizer,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], float],
    n_samples: int = 50,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    device: str = "cpu",
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Estimate MC baselines with batched generation.

    Instead of generating one sample at a time, batches multiple samples
    for the same prompt together using input repetition.

    The MC baseline is the ground-truth value function under the current policy:
        V_MC(s) = (1/K) Σ_k r_k    where r_k ~ π(·|s)

    This serves as A_MC in measurement (iv) of E2.7:
        |Â - A_MC| = |V̂(s) - V_MC(s)|

    Args:
        policy:        Pretrained CausalLM (eval mode, no grad)
        tokenizer:     Matching tokenizer
        prompts:       Reference prompts to estimate baselines for
        ground_truths: Correct answers for each prompt
        reward_fn:     reward_fn(completion, ground_truth) -> float
        n_samples:     Rollouts per prompt (50 locally, 1000 on cluster)
        max_new_tokens: Max tokens to generate per rollout
        temperature:   Sampling temperature
        device:        Torch device string
        batch_size:    Micro-batch size for generation

    Returns:
        Dict[prompt -> MC mean reward]
    """
    policy.eval()
    mc_baselines: Dict[str, float] = {}

    # Ensure left-padding for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    with torch.no_grad():
        for prompt, gt in zip(prompts, ground_truths):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]

            sample_rewards = []
            # Process in micro-batches
            for start in range(0, n_samples, batch_size):
                n_batch = min(batch_size, n_samples - start)
                # Repeat the same prompt n_batch times
                batch_ids = enc["input_ids"].repeat(n_batch, 1)
                batch_mask = enc["attention_mask"].repeat(n_batch, 1)

                out = policy.generate(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

                for i in range(n_batch):
                    completion = tokenizer.decode(
                        out[i][prompt_len:], skip_special_tokens=True
                    )
                    sample_rewards.append(reward_fn(completion, gt))

            mc_baselines[prompt] = float(np.mean(sample_rewards))

    tokenizer.padding_side = original_padding_side
    return mc_baselines


# ── Advantage estimation error ────────────────────────────────────────────────

def advantage_estimation_error(
    estimated_baselines: np.ndarray,
    mc_baselines: np.ndarray,
) -> float:
    """
    Mean absolute advantage estimation error |Â - A_MC|.

    Since A_i = r_i - baseline_i and both Â and A_MC share the same r_i:
        |Â_i - A_MC_i| = |V̂(s_i) - V_MC(s_i)|

    This measures how accurately the method estimates the value function.
    - PPO's critic introduces irreducible bias when εV > 0
    - GRPO's group-mean error decays as O(1/√G)

    Args:
        estimated_baselines: [N] baselines from the method (critic or batch mean)
        mc_baselines:        [N] MC ground-truth baselines (from estimate_mc_advantages)

    Returns:
        Scalar mean absolute error
    """
    return float(np.mean(np.abs(estimated_baselines - mc_baselines)))


def critic_approximation_error(
    critic_values: np.ndarray,
    mc_baselines: np.ndarray,
) -> float:
    """
    Critic approximation error εV = RMSE(V̂(s), V_MC(s)).

    Used in E2.8 to locate the PPO–GRPO crossover point:
        ε*V ≈ (1 - γ) sqrt(σ*²(1 - 1/G) - σ*²_A)

    Args:
        critic_values: [N] value predictions from the critic
        mc_baselines:  [N] MC ground-truth baselines

    Returns:
        RMSE scalar
    """
    return float(np.sqrt(np.mean((critic_values - mc_baselines) ** 2)))
