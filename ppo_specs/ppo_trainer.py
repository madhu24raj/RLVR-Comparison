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
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_gsm8k, format_prompt
from src.rewards import (
    gsm8k_reward,
    extract_answer_from_completion,
    matches_boxed_format,
    make_reward_fn,
)
from eval.metrics import ExperimentLogger
from eval.metrics import accuracy as compute_accuracy

from ppo_specs.config import PPOConfig
from shared.per_token_loss import (
    batched_per_token_log_probs,
    clipped_surrogate_loss,
    per_token_kl,
)
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
    # Phase-1 reward-starvation diagnostics (populated at generation time).
    # Defaults keep existing positional Rollout(...) construction in tests working.
    parse_success: bool = False       # extract_answer_from_completion(...) returned a value
    format_match_boxed: bool = False  # completion contains \boxed{...}
    det_reward: float = 0.0          # deterministic gsm8k_reward (always computed for accuracy)


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
        reference_model: Optional[AutoModelForCausalLM] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.critic = critic
        self.reward_fn = reward_fn
        self.device = device

        # Reference model for KL anchoring (L14). When provided, the
        # PPO loss includes a `config.reference_kl_coeff * KL(pi_new || pi_ref)`
        # term that penalises the policy for drifting away from this
        # frozen snapshot. We force eval mode + no_grad on every parameter
        # so no gradient can flow back into it under any code path.
        self.reference_model = reference_model
        if reference_model is not None:
            reference_model.eval()
            for p in reference_model.parameters():
                p.requires_grad_(False)

        # Sanity: if reference_kl_coeff > 0, the user MUST pass a ref model.
        if config.reference_kl_coeff > 0 and reference_model is None:
            raise ValueError(
                f"reference_kl_coeff={config.reference_kl_coeff} > 0 requires "
                f"reference_model to be passed to PPOTrainer.__init__. "
                f"Use load_ppo_trainer() which handles this automatically, "
                f"or pass reference_model explicitly."
            )

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
            max_length=self.config.max_prompt_length, padding=True,
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
            # Always compute deterministic reward for accuracy reporting,
            # regardless of training reward mode.
            det_reward = gsm8k_reward(completion, ground_truths[i])

            # Phase-1 diagnostics: is the model producing parseable output?
            # Computed on the same completion string the reward sees, so rates
            # are directly comparable to reward_nonzero_rate.
            parse_success = extract_answer_from_completion(completion) is not None
            format_match_boxed = matches_boxed_format(completion)

            rollouts.append(Rollout(
                prompt=prompts[i],
                completion=completion,
                reward=reward,
                old_log_prob=0.0,  # computed below in batch
                value=0.0,         # computed below in batch
                full_ids=full_ids.tolist(),
                prompt_len=prompt_len,
                parse_success=parse_success,
                format_match_boxed=format_match_boxed,
                det_reward=det_reward,
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

    def _batched_per_token_log_probs(
        self,
        all_full_ids: List[List[int]],
        prompt_lens: List[int],
        model_override: Optional[AutoModelForCausalLM] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-token response log-probs for a batch.

        Delegates to shared.per_token_loss.batched_per_token_log_probs.
        """
        model = model_override if model_override is not None else self.model
        return batched_per_token_log_probs(
            model, all_full_ids, prompt_lens,
            self.tokenizer.pad_token_id, self.device,
        )

    def _batched_sequence_log_probs(
        self,
        all_full_ids: List[List[int]],
        prompt_lens: List[int],
    ) -> torch.Tensor:
        """Sequence-level log prob (sum of per-token response log probs).

        Kept for backward compatibility with existing tests that check
        sequence-level scalars. The PPO loss path no longer uses this --
        see _batched_per_token_log_probs and ppo_update.
        """
        per_token, mask = self._batched_per_token_log_probs(all_full_ids, prompt_lens)
        return (per_token * mask).sum(dim=-1)  # [B]

    @torch.no_grad()
    def _batched_critic_values(self, prompts: List[str]) -> torch.Tensor:
        """Evaluate critic V(s) on all prompts in one forward pass."""
        if not self.critic.is_trainable():
            return torch.zeros(len(prompts), device=self.device)

        hidden_at_last = self._extract_last_hidden(prompts)
        return self.critic(hidden_at_last.float())  # critic stays fp32

    # ── Shared hidden-state extraction ───────────────────────────────────────

    @torch.no_grad()
    def _extract_last_hidden(self, prompts: List[str]) -> torch.Tensor:
        """
        Tokenize prompts, run a no-grad LM forward pass, and return
        the hidden state at the last real (non-padding) token per sample.

        Returns: [B, H] float tensor (detached).
        """
        enc = self.tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=self.config.max_prompt_length, padding=True,
        ).to(self.device)

        outputs = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
        )

        last_hidden = outputs.hidden_states[-1]  # [B, S, H]
        seq_lens = enc["attention_mask"].sum(dim=1) - 1  # index of last real token
        batch_idx = torch.arange(len(prompts), device=self.device)
        return last_hidden[batch_idx, seq_lens, :]  # [B, H]

    # ── Critic evaluation on prompts (batched) ───────────────────────────────

    @torch.no_grad()
    def _eval_critic_on_prompts(self, prompts: List[str]) -> np.ndarray:
        """Batched critic evaluation on a list of prompts. Returns numpy array."""
        self.model.eval()
        self.critic.eval()

        if not self.critic.is_trainable():
            return np.zeros(len(prompts))

        hidden_at_last = self._extract_last_hidden(prompts)
        values = self.critic(hidden_at_last.float())
        return values.cpu().numpy()

    # ── PPO update ────────────────────────────────────────────────────────────

    def ppo_update(
        self,
        batch: RolloutBatch,
        precomputed_advantages: Optional[torch.Tensor] = None,
        precomputed_old_per_token_log_probs: Optional[torch.Tensor] = None,
        precomputed_response_mask: Optional[torch.Tensor] = None,
        precomputed_ref_per_token_log_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """One PPO gradient step on the collected batch (per-token loss).

        The PPO surrogate is computed at the *token* level, matching the
        TRL/InstructGPT formulation for autoregressive LMs:

            ratio_bt   = exp(new_log_probs_bt - old_log_probs_bt)
            unclipped  = ratio_bt * A_b
            clipped    = clip(ratio_bt, 1-eps, 1+eps) * A_b
            L_pg       = -mean_over_unmasked_tokens(min(unclipped, clipped))

        Sequence-level log ratios (which were used here previously) blow up
        with response length: even tiny per-token weight changes accumulate
        across hundreds of tokens. (Empirically: a K=4 run with the old
        sequence-level loss on this codebase produced kl_divergence ~ 232
        within 10 steps and immediate policy collapse.) The per-token
        formulation makes the clip actually do its job.

        For backward compatibility, callers may omit the `precomputed_*`
        kwargs; in that case we recompute everything from `batch`. The
        production path (train_step) precomputes them once before the
        K-epoch loop, matching standard PPO.

        Policy and critic losses are computed jointly but gradients are
        separated: critic hidden states are detached from the policy graph
        so L_V does not backpropagate into policy weights.
        """
        self.model.train()

        rewards = batch.rewards().to(self.device)

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

        # ── Per-token old log probs and response mask ────────────────────────
        if (precomputed_old_per_token_log_probs is not None
                and precomputed_response_mask is not None):
            old_per_token = precomputed_old_per_token_log_probs.detach()
            mask = precomputed_response_mask.detach()
        else:
            with torch.no_grad():
                old_per_token, mask = self._batched_per_token_log_probs(
                    [r.full_ids for r in batch.rollouts],
                    [r.prompt_len for r in batch.rollouts],
                )
            old_per_token = old_per_token.detach()
            mask = mask.detach()

        # ── Policy forward pass: per-token NEW log probs (with grad) ─────────
        # The response mask is purely a function of token positions, not policy
        # params, so we discard the mask from this call and reuse `mask`.
        new_per_token, _ = self._batched_per_token_log_probs(
            [r.full_ids for r in batch.rollouts],
            [r.prompt_len for r in batch.rollouts],
        )

        # ── Per-token PPO surrogate ──────────────────────────────────────────
        log_ratio = new_per_token - old_per_token            # [B, T]
        log_ratio = torch.clamp(log_ratio, -self.config.log_ratio_clip, self.config.log_ratio_clip)
        ratio = torch.exp(log_ratio)                         # [B, T]
        A_expanded = advantages.unsqueeze(-1)                # [B, 1] -> broadcasts over T

        unclipped = ratio * A_expanded
        clipped = torch.clamp(
            ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon
        ) * A_expanded
        per_token_pg = -torch.min(unclipped, clipped)        # [B, T]

        # Masked mean over the unmasked tokens (TRL standard)
        mask_sum = mask.sum().clamp(min=1.0)
        policy_loss = (per_token_pg * mask).sum() / mask_sum

        # ── KL penalty (per-token, masked) ───────────────────────────────────
        # KL(pi_old || pi_new) = E_{pi_old}[log pi_old - log pi_new]
        # NOTE: this is now PER-TOKEN KL, not sequence-level. Numbers from
        # this metric are NOT comparable to pre-refactor logs which used
        # sequence-level KL (which routinely hit ~10^2 per sequence in our
        # broken K=4 run). Per-token KL should stay << 1.0 in healthy PPO.
        kl_per_token = (old_per_token - new_per_token) * mask  # [B, T]
        kl = kl_per_token.sum() / mask_sum

        # ── Reference KL anchor (L14) ────────────────────────────────────────
        # KL(pi_new || pi_ref) where pi_ref is a frozen snapshot of the
        # initial policy. This penalises the policy for drifting away from
        # its starting distribution -- the standard RLHF "anchor" term.
        # Estimator: mean(new_log_prob - ref_log_prob) per token, masked.
        # We use this direction (KL(new||ref)) rather than KL(ref||new)
        # because our token sequences are sampled from the current policy,
        # making the KL(new||ref) Monte Carlo estimator unbiased
        # (TRL/InstructGPT convention).
        kl_ref = torch.tensor(0.0, device=self.device)
        if (precomputed_ref_per_token_log_probs is not None
                and self.config.reference_kl_coeff > 0):
            ref_per_token = precomputed_ref_per_token_log_probs.detach()
            kl_ref_per_token = (new_per_token - ref_per_token) * mask  # [B, T]
            kl_ref = kl_ref_per_token.sum() / mask_sum

        # ── Combined loss and backward ────────────────────────────────────────
        total_loss = (policy_loss
                      + self.config.critic_loss_coeff * critic_loss
                      + self.config.kl_coeff * kl
                      + self.config.reference_kl_coeff * kl_ref)

        self.policy_optimizer.zero_grad()
        if self.critic_optimizer:
            self.critic_optimizer.zero_grad()

        total_loss.backward()

        # Clip and capture grad norm BEFORE the optimizer step.
        # The metric reports the *pre-clip* L2 norm so it's informative even
        # when clipping engages. We need this because policy_loss is ~0 when
        # the new and old policies match exactly (same model state at start
        # of epoch 1) and is therefore not a useful health signal on its own.
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.grad_clip_norm,
        )
        if self.critic.is_trainable():
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.config.grad_clip_norm)
        self.policy_optimizer.step()
        if self.critic_optimizer:
            self.critic_optimizer.step()

        # Use detached ratio for metrics to avoid retaining the computation graph
        ratio_detached = ratio.detach()
        # clip_fraction is now per-token over unmasked positions
        clip_hits = ((ratio_detached - 1.0).abs() > self.config.clip_epsilon).float()
        clip_fraction = (clip_hits * mask).sum() / mask_sum

        return {
            "policy_loss": policy_loss.item(),
            "policy_grad_norm": float(policy_grad_norm),
            "critic_loss": critic_loss.item(),
            "kl_divergence": kl.item(),
            "kl_ref_divergence": kl_ref.item(),
            "mean_reward": rewards.mean().item(),
            "reward_variance": rewards.var().item(),
            "mean_advantage": advantages.mean().item(),
            "clip_fraction": clip_fraction.item(),
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

        hidden_at_last = self._extract_last_hidden(prompts).detach()

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

        Advantages AND per-token old log probs are computed ONCE before
        the K-epoch loop and held fixed across all K updates, matching
        the standard PPO algorithm (Schulman et al., 2017). Recomputing
        either each epoch with updated weights would introduce a moving
        optimisation target.

        The per-token old log probs are what make the K>=2 PPO ratios
        meaningful: by epoch 2, the model has moved, so new_log_probs
        differ from these frozen old_log_probs and the ratio is non-trivial.
        """
        batch = self.generate_rollouts(prompts, ground_truths)

        # ── Freeze advantages once, before any gradient updates ─────────
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

        # ── Freeze per-token old log probs once, before any gradient updates ─
        with torch.no_grad():
            fixed_old_per_token, fixed_response_mask = self._batched_per_token_log_probs(
                [r.full_ids for r in batch.rollouts],
                [r.prompt_len for r in batch.rollouts],
            )
        fixed_old_per_token = fixed_old_per_token.detach()
        fixed_response_mask = fixed_response_mask.detach()

        # ── Freeze per-token reference log probs once (L14 anchor) ──────────
        # Only computed when reference KL is enabled. The reference model
        # is frozen for the entire run, so its log probs depend only on the
        # rollout token ids -- not on the K-epoch loop iteration.
        fixed_ref_per_token: Optional[torch.Tensor] = None
        if self.reference_model is not None and self.config.reference_kl_coeff > 0:
            with torch.no_grad():
                ref_per_token, _ = self._batched_per_token_log_probs(
                    [r.full_ids for r in batch.rollouts],
                    [r.prompt_len for r in batch.rollouts],
                    model_override=self.reference_model,
                )
            fixed_ref_per_token = ref_per_token.detach()

        all_metrics: List[Dict[str, float]] = []
        for epoch in range(self.config.n_ppo_epochs):
            metrics = self.ppo_update(
                batch,
                precomputed_advantages=fixed_advantages,
                precomputed_old_per_token_log_probs=fixed_old_per_token,
                precomputed_response_mask=fixed_response_mask,
                precomputed_ref_per_token_log_probs=fixed_ref_per_token,
            )
            all_metrics.append(metrics)

        # Average scalar metrics over epochs
        aggregated = {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0]
        }
        aggregated["accuracy"] = compute_accuracy(
            [r.det_reward for r in batch.rollouts]
        )
        # Phase-1 reward-starvation diagnostics (batch-level rates).
        # reward_nonzero_rate equals accuracy under the current binary reward,
        # but is tracked as a separate column so it remains meaningful once
        # the reward source is swapped for a continuous learned RM (Phase 4).
        n = len(batch.rollouts)
        aggregated["parse_success_rate"] = (
            sum(r.parse_success for r in batch.rollouts) / n if n else 0.0
        )
        aggregated["format_match_rate"] = (
            sum(r.format_match_boxed for r in batch.rollouts) / n if n else 0.0
        )
        aggregated["reward_nonzero_rate"] = (
            sum(r.reward > 0 for r in batch.rollouts) / n if n else 0.0
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
        """Batched greedy decoding accuracy on the first n_eval prompts.

        Always uses deterministic gsm8k_reward for evaluation regardless
        of training reward mode — accuracy means "did you get the right
        answer", not "did the self-judge like your completion".
        """
        self.model.eval()
        eval_prompts = prompts[:n_eval]
        eval_gts = ground_truths[:n_eval]

        eval_batch_size = min(self.config.eval_batch_size, len(eval_prompts))
        rewards: List[float] = []

        for start in range(0, len(eval_prompts), eval_batch_size):
            batch_p = eval_prompts[start:start + eval_batch_size]
            batch_gt = eval_gts[start:start + eval_batch_size]

            enc = self.tokenizer(
                batch_p, return_tensors="pt", truncation=True,
                max_length=self.config.max_prompt_length, padding=True,
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
                rewards.append(gsm8k_reward(completion, batch_gt[i]))

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

    # Load reference model for KL anchoring (L14) when enabled.
    # NOTE: this loads a SECOND copy of the model -- roughly doubling
    # parameter memory. On a single GPU with an 8B model in bf16 this
    # is ~16 GB extra. We do not enable gradient checkpointing on the
    # reference model because it never sees gradients.
    reference_model = None
    if config.reference_kl_coeff > 0:
        print(f"[PPO] reference_kl_coeff={config.reference_kl_coeff} > 0; "
              f"loading frozen reference model (doubles weight memory)")
        reference_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch_dtype,
        ).to(device)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad_(False)

    reward_fn, diagnostic_fn = make_reward_fn(
        config, reference_model=reference_model, tokenizer=tokenizer,
    )

    return PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        critic=critic,
        reward_fn=reward_fn,
        device=device,
        reference_model=reference_model,
    ), diagnostic_fn
