"""
PPO Trainer for sequence-level RLVR on GSM8K.

Design notes
────────────
For terminal-reward RLVR every episode is a single step:
  state  s  = prompt tokens
  action a  = full generated response (sequence of tokens)
  reward r  = binary {0, 1} from verifiable answer checker

GAE therefore collapses to:  A_i = r_i - V(s_i)

The PPO-clip surrogate operates at the sequence level:
  ratio  ρ = exp( log π_θ(a|s) - log π_θ_old(a|s) )
  L_CLIP = E[ min( ρ A,  clip(ρ, 1-ε, 1+ε) A ) ]

The critic is trained simultaneously with MSE loss against the observed returns:
  L_V = (V̂(s) - r)²

Token-level log-probabilities are summed over the response to get
the sequence-level log π(a|s).  The generated token ids are stored
in the rollout buffer (avoiding re-tokenisation artefacts at the
prompt/response boundary).

Batched generation & bfloat16 support
──────────────────────────────────────
All per-sample loops have been converted to batched operations.
Left-padding is used for generation; log_softmax is always computed
in float32 for numerical stability.  Gradient checkpointing is
supported for large models.
"""

import sys
import os

# Make sibling packages importable when run as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_gsm8k, format_prompt
from src.rewards import gsm8k_reward
from eval.metrics import ExperimentLogger
from eval.metrics import accuracy as compute_accuracy

from ppo_specs.config import PPOConfig
from ppo_specs.critic import build_critic
from ppo_specs.advantage import compute_advantages


# ── Rollout data structures ───────────────────────────────────────────────────

@dataclass
class Rollout:
    prompt: str
    completion: str
    reward: float
    old_log_prob: float      # log π_θ_old(a|s), computed at generation time
    value: float             # V̂(s) from critic (or 0.0 for REINFORCE)
    full_ids: List[int]      # full token ids: [prompt | response]
    prompt_len: int          # number of prompt tokens


@dataclass
class RolloutBatch:
    rollouts: List[Rollout]

    def rewards(self) -> torch.Tensor:
        return torch.tensor([r.reward for r in self.rollouts], dtype=torch.float32)

    def old_log_probs(self) -> torch.Tensor:
        return torch.tensor([r.old_log_prob for r in self.rollouts], dtype=torch.float32)

    def values(self) -> torch.Tensor:
        return torch.tensor([r.value for r in self.rollouts], dtype=torch.float32)


# ── PPO Trainer ───────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    PPO trainer for RLVR tasks with verifiable binary rewards.

    Args:
        config    – PPOConfig
        model     – CausalLM (fine-tuned as the policy π_θ)
        tokenizer – Matching tokenizer
        critic    – Value function from ppo_specs.critic (any capacity)
        reward_fn – reward_fn(completion, ground_truth) -> float
        device    – torch.device
    """

    def __init__(
        self,
        config: PPOConfig,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        critic: nn.Module,
        reward_fn: Callable[[str, str], float],
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.critic = critic
        self.reward_fn = reward_fn
        self.device = device

        self.policy_optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate
        )
        if critic.is_trainable():
            self.critic_optimizer: Optional[torch.optim.Optimizer] = torch.optim.AdamW(
                critic.parameters(), lr=config.critic_lr
            )
        else:
            self.critic_optimizer = None

        self.logger = ExperimentLogger(config.experiment_name, config.output_dir)
        self.step = 0
        self.total_rollouts = 0

    # ── Rollout generation (batched) ─────────────────────────────────────────

    @torch.no_grad()
    def generate_rollouts(
        self,
        prompts: List[str],
        ground_truths: List[str],
    ) -> RolloutBatch:
        """
        Batched rollout generation: one generate() call, one forward pass
        for log-probs, one forward pass for critic values.
        """
        self.model.eval()
        B = len(prompts)

        # Batch tokenize with left-padding
        enc = self.tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()  # actual lengths per sample

        # Single batched generate
        out = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Build rollouts from batched output
        rollouts = []
        for i in range(B):
            pad_len = (enc["input_ids"][i] == self.tokenizer.pad_token_id).sum().item()
            # With left-padding, real prompt starts at pad_len
            real_start = pad_len
            prompt_len = prompt_lens[i]
            full_ids = out[i][real_start:]  # strip left-padding from output

            completion = self.tokenizer.decode(
                full_ids[prompt_len:], skip_special_tokens=True
            )
            reward = self.reward_fn(completion, ground_truths[i])

            rollouts.append(Rollout(
                prompt=prompts[i],
                completion=completion,
                reward=reward,
                old_log_prob=0.0,  # computed below in batch
                value=0.0,         # computed below in batch
                full_ids=full_ids.tolist(),
                prompt_len=prompt_len,
            ))

        # Batch compute old log probs
        old_log_probs = self._batched_sequence_log_probs(
            [r.full_ids for r in rollouts],
            [r.prompt_len for r in rollouts],
        )
        for i, r in enumerate(rollouts):
            r.old_log_prob = old_log_probs[i].item()

        # Batch compute critic values
        if self.critic.is_trainable():
            critic_values = self._batched_critic_values(prompts)
            for i, r in enumerate(rollouts):
                r.value = critic_values[i].item()

        self.total_rollouts += B
        return RolloutBatch(rollouts)

    # ── Batched helper methods ───────────────────────────────────────────────

    def _batched_sequence_log_probs(
        self,
        all_full_ids: List[List[int]],
        prompt_lens: List[int],
    ) -> torch.Tensor:
        """Compute sequence log probs for all samples in one forward pass."""
        # Pad to same length (right-pad with pad_token_id)
        max_len = max(len(ids) for ids in all_full_ids)
        padded = torch.full(
            (len(all_full_ids), max_len), self.tokenizer.pad_token_id,
            dtype=torch.long, device=self.device,
        )
        attention_mask = torch.zeros(
            len(all_full_ids), max_len,
            dtype=torch.long, device=self.device,
        )
        for i, ids in enumerate(all_full_ids):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1

        outputs = self.model(
            input_ids=padded, attention_mask=attention_mask, use_cache=False,
        )
        logits = outputs.logits.float()  # CRITICAL: compute log_softmax in fp32
        log_probs = torch.log_softmax(logits, dim=-1)

        # Extract per-sample response log probs
        result = []
        for i in range(len(all_full_ids)):
            pl = prompt_lens[i]
            seq_len = len(all_full_ids[i])
            if seq_len <= pl:
                result.append(torch.zeros(1, device=self.device))
                continue
            response_lp = log_probs[i, pl - 1 : seq_len - 1, :]  # [R, V]
            response_ids = padded[i, pl:seq_len]  # [R]
            token_lp = response_lp.gather(1, response_ids.unsqueeze(-1)).squeeze(-1)
            result.append(token_lp.sum().unsqueeze(0))

        return torch.cat(result)  # [B]

    @torch.no_grad()
    def _batched_critic_values(self, prompts: List[str]) -> torch.Tensor:
        """Evaluate critic V(s) on all prompts in one forward pass."""
        if not self.critic.is_trainable():
            return torch.zeros(len(prompts), device=self.device)

        enc = self.tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)

        outputs = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
        )

        # Get last non-padding token's hidden state per sample
        seq_lens = enc["attention_mask"].sum(dim=1) - 1  # index of last real token
        last_hidden = outputs.hidden_states[-1]  # [B, S, H]
        batch_idx = torch.arange(len(prompts), device=self.device)
        hidden_at_last = last_hidden[batch_idx, seq_lens, :]  # [B, H]

        return self.critic(hidden_at_last.float())  # critic stays fp32

    # ── Legacy single-sample methods (kept for backward compatibility) ───────

    def _sequence_log_prob(
        self,
        input_ids: torch.Tensor,  # [1, seq_len]
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Sum of per-token log probabilities for the response portion only.

        The language model produces logits[t] predicting token t+1, so:
            log π(a_t | s, a_<t) = log_softmax(logits[prompt_len + t - 1])[a_t]
        """
        outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits.float()  # upcast to fp32 for numerical stability
        log_probs = torch.log_softmax(logits, dim=-1)  # [1, L, V]

        response_ids = input_ids[:, prompt_len:]           # [1, R]
        if response_ids.shape[1] == 0:
            return torch.zeros(1, device=self.device)

        # logits at positions [prompt_len-1 : L-1] predict tokens [prompt_len : L]
        response_log_probs = log_probs[:, prompt_len - 1 : -1, :]  # [1, R, V]
        token_lp = response_log_probs.gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1)  # [1, R]

        return token_lp.sum(dim=-1)  # [1]

    @torch.no_grad()
    def _critic_value_no_grad(self, prompt_ids: torch.Tensor) -> float:
        """Extract last-token hidden state and run critic (no gradient)."""
        if not self.critic.is_trainable():
            return 0.0  # replaced by batch mean in compute_advantages

        outputs = self.model(
            input_ids=prompt_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, H]
        return self.critic(last_hidden.float()).item()

    # ── Critic evaluation on prompts (batched) ───────────────────────────────

    @torch.no_grad()
    def _eval_critic_on_prompts(self, prompts: List[str]) -> np.ndarray:
        """Batched critic evaluation on a list of prompts. Returns numpy array."""
        self.model.eval()
        self.critic.eval()

        if not self.critic.is_trainable():
            return np.zeros(len(prompts))

        enc = self.tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)

        outputs = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
        )

        # Extract last real token per sample
        # With left-padding, last real token is always at position -1
        last_hidden = outputs.hidden_states[-1]  # [B, S, H]
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(len(prompts), device=self.device)
        hidden_at_last = last_hidden[batch_idx, seq_lens, :]

        values = self.critic(hidden_at_last.float())
        return values.cpu().numpy()

    # ── PPO update ────────────────────────────────────────────────────────────

    def ppo_update(
        self,
        batch: RolloutBatch,
        precomputed_advantages: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        One PPO gradient step on the collected batch.

        Policy and critic losses are computed jointly but gradients are
        separated: critic hidden states are detached from the policy graph
        so L_V does not backpropagate into policy weights.

        Args:
            batch: Rollout data collected under pi_old.
            precomputed_advantages: If provided, these fixed advantages are
                used instead of recomputing from the (now-updated) critic.
                Standard PPO computes advantages once before the K-epoch
                optimisation loop (Schulman et al., 2017 Section 4).
        """
        self.model.train()

        rewards = batch.rewards().to(self.device)
        old_log_probs = batch.old_log_probs().to(self.device)

        # ── Critic forward pass (with grad, detached from policy) ─────────────
        critic_values, critic_loss = self._critic_forward(batch, rewards)

        # ── Advantages ───────────────────────────────────────────────────────
        if precomputed_advantages is not None:
            advantages = precomputed_advantages.detach()
        else:
            values_for_adv = critic_values.detach() if critic_values is not None else None
            advantages = compute_advantages(
                rewards,
                values_for_adv,
                gamma=self.config.gamma,
                normalize=True,
            )

        # ── Policy forward pass (with grad, batched) ─────────────────────────
        new_log_probs = self._policy_log_probs(batch)

        log_ratio = new_log_probs - old_log_probs.detach()
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(
            ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon
        )
        policy_loss = -torch.mean(torch.min(ratio * advantages, clipped * advantages))

        # ── KL divergence penalty ─────────────────────────────────────────────
        # KL(pi_old || pi_new) = E_{pi_old}[log pi_old - log pi_new]
        # This is the correct direction: penalises the new policy for
        # moving away from the old policy under which data was collected.
        kl = (old_log_probs.detach() - new_log_probs).mean()

        # ── Combined loss and backward ────────────────────────────────────────
        total_loss = (policy_loss
                      + self.config.critic_loss_coeff * critic_loss
                      + self.config.kl_coeff * kl)

        self.policy_optimizer.zero_grad()
        if self.critic_optimizer:
            self.critic_optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if self.critic.is_trainable():
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        if self.critic_optimizer:
            self.critic_optimizer.step()

        # Use detached ratio for metrics to avoid retaining the computation graph
        ratio_detached = ratio.detach()
        return {
            "policy_loss": policy_loss.item(),
            "critic_loss": critic_loss.item(),
            "kl_divergence": kl.item(),
            "mean_reward": rewards.mean().item(),
            "reward_variance": rewards.var().item(),
            "mean_advantage": advantages.mean().item(),
            "clip_fraction": ((ratio_detached - 1.0).abs() > self.config.clip_epsilon)
                             .float().mean().item(),
        }

    def _policy_log_probs(self, batch: RolloutBatch) -> torch.Tensor:
        """
        Batched policy log-probs WITH gradients for PPO surrogate loss.

        Reconstructs the padded batch from stored rollout full_ids, runs a
        single forward pass, and extracts per-sample sequence log-probs.
        """
        return self._batched_sequence_log_probs(
            [r.full_ids for r in batch.rollouts],
            [r.prompt_len for r in batch.rollouts],
        )

    def _critic_forward(
        self,
        batch: RolloutBatch,
        rewards: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Batched critic forward: one tokenization, one LM forward pass,
        one critic forward pass.

        Hidden states are detached from the policy computation graph so the
        critic loss does not affect policy weights.

        Returns:
            (critic_values [B] or None,  critic_loss scalar)
        """
        if not self.critic.is_trainable():
            return None, torch.tensor(0.0, device=self.device)

        self.critic.train()
        prompts = [r.prompt for r in batch.rollouts]

        enc = self.tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                use_cache=False,
                output_hidden_states=True,
            )

        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        last_hidden = outputs.hidden_states[-1]
        batch_idx = torch.arange(len(prompts), device=self.device)
        hidden_at_last = last_hidden[batch_idx, seq_lens, :].detach()

        critic_values = self.critic(hidden_at_last.float())
        critic_loss = torch.nn.functional.mse_loss(critic_values, rewards)
        return critic_values, critic_loss

    # ── Training loop ─────────────────────────────────────────────────────────

    def train_step(
        self,
        prompts: List[str],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """Single PPO iteration: collect rollouts → K gradient updates.

        Advantages are computed ONCE from the initial critic values and held
        fixed across all K PPO epochs, matching the standard PPO algorithm
        (Schulman et al., 2017).  Recomputing advantages each epoch with
        updated critic weights would introduce a moving optimisation target.
        """
        batch = self.generate_rollouts(prompts, ground_truths)

        # ── Compute advantages once, before any gradient updates ─────────
        rewards = batch.rewards().to(self.device)
        with torch.no_grad():
            critic_values_init, _ = self._critic_forward(batch, rewards)
        values_for_adv = critic_values_init.detach() if critic_values_init is not None else None
        fixed_advantages = compute_advantages(
            rewards,
            values_for_adv,
            gamma=self.config.gamma,
            normalize=True,
        )

        all_metrics: List[Dict[str, float]] = []
        for epoch in range(self.config.n_ppo_epochs):
            metrics = self.ppo_update(batch, precomputed_advantages=fixed_advantages)
            all_metrics.append(metrics)

        # Average scalar metrics over epochs
        aggregated = {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0]
        }
        aggregated["accuracy"] = compute_accuracy(
            [r.reward for r in batch.rollouts]
        )
        aggregated["total_rollouts"] = self.total_rollouts
        self.step += 1
        return aggregated

    @torch.no_grad()
    def evaluate(
        self,
        prompts: List[str],
        ground_truths: List[str],
        n_eval: int = 50,
    ) -> float:
        """Batched greedy decoding accuracy on the first n_eval prompts."""
        self.model.eval()
        eval_prompts = prompts[:n_eval]
        eval_gts = ground_truths[:n_eval]

        eval_batch_size = min(8, len(eval_prompts))
        rewards: List[float] = []

        for start in range(0, len(eval_prompts), eval_batch_size):
            batch_p = eval_prompts[start:start + eval_batch_size]
            batch_gt = eval_gts[start:start + eval_batch_size]

            enc = self.tokenizer(
                batch_p, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            ).to(self.device)

            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,  # greedy for deterministic eval
                pad_token_id=self.tokenizer.pad_token_id,
            )

            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
            for i in range(len(batch_p)):
                pad_len = (enc["input_ids"][i] == self.tokenizer.pad_token_id).sum().item()
                real_start = pad_len
                pl = prompt_lens[i]
                completion = self.tokenizer.decode(
                    out[i][real_start + pl:], skip_special_tokens=True
                )
                rewards.append(self.reward_fn(completion, batch_gt[i]))

        return compute_accuracy(rewards)


# ── Convenience loader ────────────────────────────────────────────────────────

def load_ppo_trainer(config: PPOConfig, device: torch.device) -> PPOTrainer:
    """
    Load model + tokenizer from HuggingFace, build critic, return PPOTrainer.

    Supports bfloat16 on GPU (via config.torch_dtype) and gradient
    checkpointing for large models.  The critic is always kept in float32
    for numerical stability.
    """
    # Determine dtype
    if config.torch_dtype == "auto":
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif config.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    print(f"[PPO] Loading model: {config.model_name} (device={device}, dtype={torch_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # REQUIRED for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch_dtype,
    ).to(device)

    # Enable gradient checkpointing for large models
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("[PPO] Gradient checkpointing enabled")

    hidden_size = model.config.hidden_size
    print(f"[PPO] Model hidden size: {hidden_size} | "
          f"Critic capacity: {config.critic_capacity}")

    # Keep critic in float32 even when model is bf16
    critic = build_critic(config.critic_capacity, hidden_size)
    critic = critic.to(device)  # stays float32

    return PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        critic=critic,
        reward_fn=gsm8k_reward,
        device=device,
    )
