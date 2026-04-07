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
    gamma: float = 0.99,
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

    if normalize and advantages.numel() > 1 and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Estimate the Monte Carlo baseline E[r | prompt] for each prompt.

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
        device:        Torch device string

    Returns:
        Dict[prompt -> MC mean reward]
    """
    policy.eval()
    mc_baselines: Dict[str, float] = {}

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

            for _ in range(n_samples):
                out = policy.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
                completion = tokenizer.decode(
                    out[0][prompt_len:], skip_special_tokens=True
                )
                sample_rewards.append(reward_fn(completion, gt))

            mc_baselines[prompt] = float(np.mean(sample_rewards))

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
