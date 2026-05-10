"""
GRPO (Group Relative Policy Optimization) trainer for RLVR on GSM8K.

GRPO is PPO without a learned critic. Instead of training a value network,
GRPO generates G completions per prompt and uses the group statistics
(mean, std) as the baseline for advantage estimation:

    A_ig = (R_ig - mean(R_g)) / std(R_g)

The policy loss uses the same per-token clipped surrogate as PPO.

Data flow:
    B prompts
        -> generate G completions each (B*G total)
        -> score all with reward_fn
        -> per-group advantage normalization
        -> per-token clipped surrogate loss (K epochs)
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_gsm8k, format_prompt_with_template
from src.rewards import gsm8k_reward
from eval.metrics import ExperimentLogger
from eval.metrics import accuracy as compute_accuracy

from grpo_specs.STALE.config import GRPOConfig
from shared.per_token_loss import (
    batched_per_token_log_probs,
    clipped_surrogate_loss,
    per_token_kl,
)


# -- Data structures --

@dataclass
class Rollout:
    prompt: str
    completion: str
    reward: float
    full_ids: List[int]
    prompt_len: int
    group_idx: int  # which prompt group this belongs to


@dataclass
class GroupRolloutBatch:
    rollouts: List[Rollout]
    n_groups: int           # number of unique prompts (B)
    group_size: int         # completions per prompt (G)

    def rewards(self) -> torch.Tensor:
        return torch.tensor([r.reward for r in self.rollouts], dtype=torch.float32)

    def group_rewards(self) -> List[List[float]]:
        """Rewards organized by group: [[r_00, r_01, ...], [r_10, ...], ...]."""
        groups = [[] for _ in range(self.n_groups)]
        for r in self.rollouts:
            groups[r.group_idx].append(r.reward)
        return groups


# -- GRPO Trainer --

class GRPOTrainer:
    """GRPO trainer for RLVR tasks with verifiable binary rewards."""

    def __init__(
        self,
        config: GRPOConfig,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_fn: Callable[[str, str], float],
        device: torch.device,
        reference_model: Optional[AutoModelForCausalLM] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.device = device

        self.reference_model = reference_model
        if reference_model is not None:
            reference_model.eval()
            for p in reference_model.parameters():
                p.requires_grad_(False)

        if config.reference_kl_coeff > 0 and reference_model is None:
            raise ValueError(
                f"reference_kl_coeff={config.reference_kl_coeff} > 0 requires "
                f"reference_model to be passed to GRPOTrainer.__init__."
            )

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self.total_rollouts = 0

    # -- Rollout generation --

    @torch.no_grad()
    def generate_group_rollouts(
        self,
        prompts: List[str],
        ground_truths: List[str],
    ) -> GroupRolloutBatch:
        """Generate G completions per prompt, score each with reward_fn."""
        self.model.eval()
        G = self.config.n_rollouts_per_prompt
        B = len(prompts)

        # Repeat each prompt G times for batched generation
        expanded_prompts = []
        expanded_gts = []
        group_indices = []
        for i, (p, gt) in enumerate(zip(prompts, ground_truths)):
            for _ in range(G):
                expanded_prompts.append(p)
                expanded_gts.append(gt)
                group_indices.append(i)

        # Tokenize all B*G prompts
        enc = self.tokenizer(
            expanded_prompts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

        # Generate
        out = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Build rollouts
        rollouts = []
        for i in range(B * G):
            pad_len = (enc["input_ids"][i] == self.tokenizer.pad_token_id).sum().item()
            real_start = pad_len
            prompt_len = prompt_lens[i]
            full_ids = out[i][real_start:]

            completion = self.tokenizer.decode(
                full_ids[prompt_len:], skip_special_tokens=True
            )
            reward = self.reward_fn(completion, expanded_gts[i])

            rollouts.append(Rollout(
                prompt=expanded_prompts[i],
                completion=completion,
                reward=reward,
                full_ids=full_ids.tolist(),
                prompt_len=prompt_len,
                group_idx=group_indices[i],
            ))

        self.total_rollouts += len(rollouts)
        return GroupRolloutBatch(
            rollouts=rollouts,
            n_groups=B,
            group_size=G,
        )

    # -- Group advantage estimation --

    def compute_group_advantages(
        self,
        batch: GroupRolloutBatch,
    ) -> torch.Tensor:
        """Per-sample advantage using group-relative normalization.

        For each prompt group g with G completions:
            A_ig = (R_ig - mean(R_g)) / max(std(R_g), eps)

        When all rewards in a group are identical (std=0), advantages
        are set to 0 for that group (no gradient signal, no NaN).

        Returns: [B*G] tensor of advantages.
        """
        group_rewards = batch.group_rewards()
        advantages = torch.zeros(len(batch.rollouts), dtype=torch.float32)

        for g, rewards_g in enumerate(group_rewards):
            rewards_t = torch.tensor(rewards_g, dtype=torch.float32)
            mean_g = rewards_t.mean()
            std_g = rewards_t.std()

            if std_g < 1e-8:
                # All rewards identical in this group. No signal.
                adv_g = torch.zeros_like(rewards_t)
            else:
                adv_g = (rewards_t - mean_g) / (std_g + 1e-8)

            # Place into the flat advantages tensor
            start = g * batch.group_size
            end = start + len(rewards_g)
            advantages[start:end] = adv_g

        return advantages.to(self.device)

    # -- Per-token log probs --

    def _per_token_log_probs(
        self,
        all_full_ids: List[List[int]],
        prompt_lens: List[int],
        model_override: Optional[AutoModelForCausalLM] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model = model_override if model_override is not None else self.model
        return batched_per_token_log_probs(
            model, all_full_ids, prompt_lens,
            self.tokenizer.pad_token_id, self.device,
        )

    # -- GRPO update --

    def grpo_update(
        self,
        batch: GroupRolloutBatch,
        precomputed_advantages: torch.Tensor,
        precomputed_old_per_token: torch.Tensor,
        precomputed_mask: torch.Tensor,
        precomputed_ref_per_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """One GRPO gradient step on the collected batch."""
        self.model.train()

        old_per_token = precomputed_old_per_token.detach()
        mask = precomputed_mask.detach()
        advantages = precomputed_advantages.detach()

        # New log probs (with grad)
        new_per_token, _ = self._per_token_log_probs(
            [r.full_ids for r in batch.rollouts],
            [r.prompt_len for r in batch.rollouts],
        )

        # Clipped surrogate loss
        policy_loss, clip_fraction, _ = clipped_surrogate_loss(
            new_per_token, old_per_token, advantages, mask,
            clip_epsilon=self.config.clip_epsilon,
        )

        # KL penalty (old vs new)
        kl = per_token_kl(old_per_token, new_per_token, mask)

        # Reference KL anchor
        kl_ref = torch.tensor(0.0, device=self.device)
        if (precomputed_ref_per_token is not None
                and self.config.reference_kl_coeff > 0):
            ref_per_token = precomputed_ref_per_token.detach()
            kl_ref = per_token_kl(new_per_token, ref_per_token, mask)

        # Combined loss
        total_loss = (policy_loss
                      + self.config.kl_coeff * kl
                      + self.config.reference_kl_coeff * kl_ref)

        self.optimizer.zero_grad()
        total_loss.backward()

        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0,
        )
        self.optimizer.step()

        rewards = batch.rewards().to(self.device)
        return {
            "policy_loss": policy_loss.item(),
            "policy_grad_norm": float(policy_grad_norm),
            "kl_divergence": kl.item(),
            "kl_ref_divergence": kl_ref.item(),
            "mean_reward": rewards.mean().item(),
            "reward_variance": rewards.var().item(),
            "mean_advantage": advantages.mean().item(),
            "clip_fraction": clip_fraction.item(),
        }

    # -- Train step (rollout + K updates) --

    def train_step(
        self,
        prompts: List[str],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """Single GRPO iteration: generate G rollouts per prompt, K updates."""
        batch = self.generate_group_rollouts(prompts, ground_truths)

        # Compute group advantages once
        advantages = self.compute_group_advantages(batch)

        # Freeze per-token old log probs once
        with torch.no_grad():
            fixed_old_per_token, fixed_mask = self._per_token_log_probs(
                [r.full_ids for r in batch.rollouts],
                [r.prompt_len for r in batch.rollouts],
            )
        fixed_old_per_token = fixed_old_per_token.detach()
        fixed_mask = fixed_mask.detach()

        # Freeze reference log probs once (if enabled)
        fixed_ref_per_token: Optional[torch.Tensor] = None
        if self.reference_model is not None and self.config.reference_kl_coeff > 0:
            with torch.no_grad():
                ref_per_token, _ = self._per_token_log_probs(
                    [r.full_ids for r in batch.rollouts],
                    [r.prompt_len for r in batch.rollouts],
                    model_override=self.reference_model,
                )
            fixed_ref_per_token = ref_per_token.detach()

        # K epochs
        all_metrics: List[Dict[str, float]] = []
        for epoch in range(self.config.n_ppo_epochs):
            metrics = self.grpo_update(
                batch,
                precomputed_advantages=advantages,
                precomputed_old_per_token=fixed_old_per_token,
                precomputed_mask=fixed_mask,
                precomputed_ref_per_token=fixed_ref_per_token,
            )
            all_metrics.append(metrics)

        # Average metrics over epochs
        avg = {}
        for key in all_metrics[0]:
            avg[key] = float(np.mean([m[key] for m in all_metrics]))

        # Add accuracy
        rewards = [r.reward for r in batch.rollouts]
        avg["accuracy"] = compute_accuracy(rewards)
        avg["total_rollouts"] = self.total_rollouts

        return avg

    # -- Evaluation --

    @torch.no_grad()
    def evaluate(
        self,
        prompts: List[str],
        ground_truths: List[str],
        n_eval: Optional[int] = None,
    ) -> float:
        """Evaluate policy accuracy with greedy decoding."""
        self.model.eval()
        if n_eval is not None:
            prompts = prompts[:n_eval]
            ground_truths = ground_truths[:n_eval]

        eval_batch_size = min(8, len(prompts))
        rewards: List[float] = []

        for start in range(0, len(prompts), eval_batch_size):
            batch_p = prompts[start:start + eval_batch_size]
            batch_gt = ground_truths[start:start + eval_batch_size]

            enc = self.tokenizer(
                batch_p, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            ).to(self.device)

            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for i, gt in enumerate(batch_gt):
                prompt_len = enc["attention_mask"][i].sum().item()
                pad_len = (enc["input_ids"][i] == self.tokenizer.pad_token_id).sum().item()
                real_start = pad_len
                completion = self.tokenizer.decode(
                    out[i][real_start + prompt_len:], skip_special_tokens=True
                )
                rewards.append(self.reward_fn(completion, gt))

        return compute_accuracy(rewards)


# -- Factory --

def load_grpo_trainer(config: GRPOConfig, device: torch.device) -> GRPOTrainer:
    """Load model, tokenizer, and optionally reference model."""
    if config.torch_dtype == "auto":
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif config.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    print(f"[GRPO] Loading model: {config.model_name} (device={device}, dtype={torch_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
    ).to(device)

    if getattr(config, '_gradient_checkpointing', False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("[GRPO] Gradient checkpointing enabled")

    reference_model = None
    if config.reference_kl_coeff > 0:
        print(f"[GRPO] reference_kl_coeff={config.reference_kl_coeff} > 0; "
              f"loading frozen reference model (doubles weight memory)")
        reference_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch_dtype,
        ).to(device)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad_(False)

    return GRPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reward_fn=gsm8k_reward,
        device=device,
        reference_model=reference_model,
    )
