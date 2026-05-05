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
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ppo_specs.reward_model import RewardModelScorer

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None  # graceful fallback if accelerate not installed

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


# ── Optimizer construction helper (memory_optimization §11.1, §11.5) ─────────

def _build_adamw(params, lr: float, *, use_8bit: bool, use_fused: bool):
    """Construct AdamW optimizer with optional 8-bit and fused kernel.

    use_8bit: bitsandbytes AdamW8bit (saves ~48 GB at 8B). Falls back to
              torch.optim.AdamW with a printed warning if bnb missing.
    use_fused: torch's fused=True kernel (CUDA only, saves ~16 GB transient).
    """
    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr)
        except ImportError:
            print(
                "[PPO] WARNING: optimizer_8bit=True but bitsandbytes not "
                "installed; falling back to torch.optim.AdamW. "
                "Install with: pip install bitsandbytes"
            )
    if use_fused and torch.cuda.is_available():
        return torch.optim.AdamW(params, lr=lr, fused=True)
    return torch.optim.AdamW(params, lr=lr)


def _shard_list(items: list, rank: int, world_size: int) -> list:
    """Return rank's contiguous slice of items.

    Caller MUST assert len(items) % world_size == 0 before calling.
    """
    per_rank = len(items) // world_size
    return items[rank * per_rank : (rank + 1) * per_rank]


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
        reward_fn – reward_fn(completion, ground_truth) -> float. This remains
                    the trainer's primary, baseline reward callable; custom
                    reward functions are the foundation of the trainer.
        device    – torch.device
        reward_model_scorer – Optional batched learned-RM scorer. When set
                    (i.e. ``config.reward_model_capacity != "none"``), it
                    takes precedence over ``reward_fn`` inside
                    ``generate_rollouts`` and ``evaluate`` — it is one
                    specific kind of custom reward function, supplied via
                    ``score_batch(prompts, completions, gts)`` instead of the
                    per-sample ``reward_fn(completion, gt)`` protocol. When
                    ``None`` (the default), the trainer's behavior is
                    bit-identical to today's per-sample ``reward_fn`` path.
                    The reported ``accuracy`` always comes from
                    ``gsm8k_reward`` regardless of which scorer trained
                    the policy.
    """

    def __init__(
        self,
        config: PPOConfig,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        critic: nn.Module,
        reward_fn: Callable[[str, str], float],
        device: Optional[torch.device] = None,
        *,
        accelerator: Optional["Accelerator"] = None,
        reference_model: Optional[AutoModelForCausalLM] = None,
        reward_model_scorer: Optional["RewardModelScorer"] = None,
    ):
        # Exactly one of device/accelerator must be provided.
        if (device is None) == (accelerator is None):
            raise ValueError(
                "PPOTrainer requires exactly one of `device` or `accelerator`. "
                f"Got device={device!r}, accelerator={accelerator!r}."
            )

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.critic = critic
        self.reward_fn = reward_fn
        self.reward_model_scorer = reward_model_scorer
        self.accelerator = accelerator
        if accelerator is not None:
            self.device = accelerator.device
        else:
            self.device = device

        # Cache critic.is_trainable() BEFORE accelerator.prepare(), because the
        # DDP wrapper proxies only `forward` and masks the method
        # (AttributeError at call sites otherwise).
        self._critic_trainable = critic.is_trainable()

        # Reference model for KL anchoring (L14). When provided, the
        # PPO loss includes a `config.reference_kl_coeff * KL(pi_new || pi_ref)`
        # term that penalises the policy for drifting away from this
        # frozen snapshot. We force eval mode + no_grad on every parameter
        # so no gradient can flow back into it under any code path.
        self.reference_model = reference_model
        if reference_model is not None:
            # Defensive assertion: must be frozen at construction time. Wrapping
            # an unfrozen reference in Accelerate later would silently sync
            # zero gradients on every step.
            assert all(not p.requires_grad for p in reference_model.parameters()), \
                "reference_model must be frozen (requires_grad=False on all params)"
            reference_model.eval()

        # Sanity: if reference_kl_coeff > 0, the user MUST pass a ref model.
        if config.reference_kl_coeff > 0 and reference_model is None:
            raise ValueError(
                f"reference_kl_coeff={config.reference_kl_coeff} > 0 requires "
                f"reference_model to be passed to PPOTrainer.__init__. "
                f"Use load_ppo_trainer() which handles this automatically, "
                f"or pass reference_model explicitly."
            )

        self.policy_optimizer = _build_adamw(
            model.parameters(), lr=config.learning_rate,
            use_8bit=config.optimizer_8bit, use_fused=config.optimizer_fused,
        )
        if self._critic_trainable:
            self.critic_optimizer: Optional[torch.optim.Optimizer] = _build_adamw(
                critic.parameters(), lr=config.critic_lr,
                use_8bit=False,  # critic is small; 8-bit overhead not worth it
                use_fused=config.optimizer_fused,
            )
        else:
            self.critic_optimizer = None

        # If running under Accelerate, prepare model+optimizer (DDP wrap on
        # multi-process). Single-process Accelerator keeps everything on
        # accelerator.device and is functionally equivalent to the legacy
        # path for our tests. Reference model stays UNWRAPPED.
        if accelerator is not None:
            self.model, self.policy_optimizer = accelerator.prepare(
                self.model, self.policy_optimizer,
            )
            if self._critic_trainable:
                self.critic, self.critic_optimizer = accelerator.prepare(
                    self.critic, self.critic_optimizer,
                )
            else:
                self.critic = self.critic.to(self.device)
            if reference_model is not None:
                self.reference_model = reference_model.to(self.device)

        self.logger = ExperimentLogger(config.experiment_name, config.output_dir)
        self.step = 0
        self.total_rollouts = 0

    @property
    def _is_ddp(self) -> bool:
        # Defensive getattr keeps tests that bypass __init__ (PPOTrainer.__new__)
        # working when they never set self.accelerator.
        acc = getattr(self, "accelerator", None)
        return acc is not None and acc.num_processes > 1

    def __getattr__(self, name):
        # Backstop for tests that construct via PPOTrainer.__new__ and never
        # run __init__: derive `_critic_trainable` from `self.critic` so the
        # internal call sites that replaced `self.critic.is_trainable()`
        # continue to work. Only fires when the attribute is genuinely missing
        # (Python only calls __getattr__ on AttributeError lookups).
        if name == "_critic_trainable":
            critic = self.__dict__.get("critic", None)
            if critic is not None and hasattr(critic, "is_trainable"):
                return critic.is_trainable()
        raise AttributeError(name)

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
        global_B = len(prompts)

        # When running under DDP (num_processes > 1), shard the prompts so each
        # rank generates only its local slice. The full RolloutBatch is gathered
        # back below before the PPO update so every rank computes loss on the
        # same global batch.
        if self._is_ddp:
            rank = self.accelerator.process_index
            ws = self.accelerator.num_processes
            assert global_B % ws == 0, (
                f"batch_size={global_B} not divisible by num_processes={ws}; "
                f"set config.batch_size accordingly."
            )
            local_prompts = _shard_list(prompts, rank, ws)
            local_gts = _shard_list(ground_truths, rank, ws)
            B = len(local_prompts)  # CRITICAL rebind for the inner loop
        else:
            local_prompts = prompts
            local_gts = ground_truths
            B = global_B

        # Batch tokenize with left-padding (on local shard)
        enc = self.tokenizer(
            local_prompts, return_tensors="pt", truncation=True,
            max_length=self.config.max_prompt_length, padding=True,
        ).to(self.device)
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()  # actual lengths per sample

        # Pick the model used for inference: under DDP, .generate() must be
        # called on the unwrapped module (the wrapped DDP module forward fails
        # under autoregressive sampling).
        gen_model = self.accelerator.unwrap_model(self.model) if self._is_ddp else self.model

        # Batched generate. P19: optionally length-bucket so each generate()
        # call has low intra-batch length variance, reducing pad-waste from
        # ~64% to ~25% (saves ~3 s/step at 8B). Output ordering is restored
        # to the original input order before downstream rollout construction.
        if self.config.length_bucketed_generation and B > self.config.generation_bucket_size:
            out = self._bucketed_generate(enc)
        else:
            out = gen_model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # P18: Compute pad lengths for all samples in a single GPU sync
        pad_mask = (enc["input_ids"] == self.tokenizer.pad_token_id)
        pad_lens_list = pad_mask.sum(dim=1).tolist()  # one sync, returns list

        # Pre-decode all completions (and stash full_ids) so we can either
        # (a) feed them to the batched RM scorer's score_batch() in one call,
        #     or
        # (b) hand each (completion, gt) pair to the per-sample reward_fn,
        #     bit-identically to the pre-RM trainer.
        completions: List[str] = []
        full_ids_per_sample: List[torch.Tensor] = []
        for i in range(B):
            pad_len = pad_lens_list[i]
            real_start = pad_len  # with left-padding, real prompt starts here
            prompt_len = prompt_lens[i]
            full_ids = out[i][real_start:]
            full_ids_per_sample.append(full_ids)
            completions.append(
                self.tokenizer.decode(
                    full_ids[prompt_len:], skip_special_tokens=True
                )
            )

        # Fast path: batched learned-RM scoring. Only fires when a learned
        # RM is configured (capacity != "none"). When None, the per-sample
        # self.reward_fn(...) loop below runs exactly as today (baseline
        # invariant: bit-identical metrics on the "none" tier).
        rm_scores_list: Optional[List[float]] = None
        if self.reward_model_scorer is not None:
            rm_scores = self.reward_model_scorer.score_batch(
                local_prompts,
                completions,
                local_gts,
            )
            rm_scores_list = rm_scores.detach().cpu().tolist()

        # Build rollouts from batched output
        rollouts = []
        for i in range(B):
            prompt_len = prompt_lens[i]
            full_ids = full_ids_per_sample[i]
            completion = completions[i]

            # Training reward: learned RM fast-path when configured, else
            # the per-sample self.reward_fn (which routes through
            # make_reward_fn / reward_mode / SelfJudgeRewardModel as today).
            if rm_scores_list is not None:
                reward = float(rm_scores_list[i])
            else:
                reward = self.reward_fn(completion, local_gts[i])
            # Always compute deterministic verifier reward for accuracy
            # reporting, regardless of training reward mode or whether a
            # learned RM is in use. This keeps the accuracy metric binary
            # and comparable across reward sources.
            det_reward = gsm8k_reward(completion, local_gts[i])

            # Phase-1 diagnostics: is the model producing parseable output?
            # Computed on the same completion string the reward sees, so rates
            # are directly comparable to reward_nonzero_rate.
            parse_success = extract_answer_from_completion(completion) is not None
            format_match_boxed = matches_boxed_format(completion)

            rollouts.append(Rollout(
                prompt=local_prompts[i],
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

        # Under DDP, gather rollouts so every rank holds the global batch.
        # all_gather_object pickles the Rollout dataclass list (small, ~few KB).
        if self._is_ddp:
            ws = self.accelerator.num_processes
            gathered = [None] * ws
            dist.all_gather_object(gathered, rollouts)
            rollouts = [r for shard in gathered for r in shard]  # rank-ordered

        # Batch compute old log probs (P18: one .cpu().tolist() instead of B per-sample syncs)
        old_log_probs = self._batched_sequence_log_probs(
            [r.full_ids for r in rollouts],
            [r.prompt_len for r in rollouts],
        )
        old_log_probs_list = old_log_probs.cpu().tolist()
        for i, r in enumerate(rollouts):
            r.old_log_prob = old_log_probs_list[i]

        # Batch compute critic values (P18: one .cpu().tolist() instead of B per-sample syncs)
        if self._critic_trainable:
            critic_values = self._batched_critic_values([r.prompt for r in rollouts])
            critic_values_list = critic_values.cpu().tolist()
            for i, r in enumerate(rollouts):
                r.value = critic_values_list[i]

        # Counter reflects the global rollout budget regardless of world size.
        self.total_rollouts += global_B
        return RolloutBatch(rollouts)

    @torch.no_grad()
    def _bucketed_generate(self, enc) -> torch.Tensor:
        """P19: length-bucketed generation.

        Sorts samples by prompt length (ascending), runs generate() per
        bucket of size config.generation_bucket_size, then restores the
        original input order. Each bucket has lower intra-batch length
        variance than the full batch, so pad-waste during the autoregressive
        loop drops substantially (~64% → ~25% on GSM8K-like distributions).

        Returns a [B, max_total_len] tensor in ORIGINAL input order, padded
        to the maximum total length across all buckets so downstream code
        (which strips left-padding by pad_token_id count) is unchanged.
        """
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        B = input_ids.shape[0]
        bucket_size = self.config.generation_bucket_size
        pad_id = self.tokenizer.pad_token_id

        # Sort by ascending prompt length (real-token count). Left-padded inputs
        # have lengths = attention_mask.sum(dim=1).
        prompt_lens = attention_mask.sum(dim=1)
        sort_idx = torch.argsort(prompt_lens)  # [B], indices into original
        inverse_idx = torch.argsort(sort_idx)  # [B], maps sorted -> original

        # Generate per bucket. Trim leading all-pad columns per bucket
        # to avoid feeding bucket-1's max-prompt padding into bucket-0's
        # generate() call. Re-prepend the trimmed pads after generate so
        # each row's left-pad count matches the ORIGINAL enc — this
        # preserves the invariant downstream code relies on:
        #     out[i][pad_lens_list[i]:] == [prompt_tokens, completion_tokens]
        # Unwrap under DDP: .generate() must run on the unwrapped module.
        gen_model = self.accelerator.unwrap_model(self.model) if self._is_ddp else self.model

        bucket_outputs: List[torch.Tensor] = []
        bucket_first_keeps: List[int] = []
        for start in range(0, B, bucket_size):
            sel = sort_idx[start : start + bucket_size]
            ids_b = input_ids[sel]
            mask_b = attention_mask[sel]
            keep = mask_b.sum(dim=0) > 0  # [S], True where any sample has a real token
            first_keep = int(keep.float().argmax().item()) if keep.any() else 0
            if first_keep > 0:
                ids_b = ids_b[:, first_keep:]
                mask_b = mask_b[:, first_keep:]
            out_b = gen_model.generate(
                input_ids=ids_b,
                attention_mask=mask_b,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                pad_token_id=pad_id,
            )
            bucket_outputs.append(out_b)
            bucket_first_keeps.append(first_keep)

        # Re-prepend trimmed pads so all bucket outputs have the original
        # left-pad structure (first S columns equal the original enc rows).
        # Right-pad to a common max length so we can stack.
        bucket_total_lens = [o.shape[1] + fk for o, fk in zip(bucket_outputs, bucket_first_keeps)]
        max_total_len = max(bucket_total_lens)
        restored_outputs: List[torch.Tensor] = []
        for o, fk in zip(bucket_outputs, bucket_first_keeps):
            n = o.shape[0]
            row_len = o.shape[1] + fk
            if fk > 0:
                left_pad = torch.full((n, fk), pad_id, dtype=o.dtype, device=o.device)
                o = torch.cat([left_pad, o], dim=1)
            if row_len < max_total_len:
                right_pad = torch.full(
                    (n, max_total_len - row_len), pad_id, dtype=o.dtype, device=o.device,
                )
                o = torch.cat([o, right_pad], dim=1)
            restored_outputs.append(o)
        sorted_out = torch.cat(restored_outputs, dim=0)  # [B, max_total_len]

        # Restore original input order so out[i] aligns with prompts[i].
        return sorted_out[inverse_idx]

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
        if not self._critic_trainable:
            return torch.zeros(len(prompts), device=self.device)

        hidden_at_last = self._extract_last_hidden(prompts)
        return self.critic(hidden_at_last.float())  # critic stays fp32

    # ── Shared hidden-state extraction ───────────────────────────────────────

    @torch.no_grad()
    def _extract_last_hidden(self, prompts: List[str]) -> torch.Tensor:
        """Tokenize prompts, run a no-grad LM forward pass, and return
        the hidden state at the last real (non-padding) token per sample.

        Uses a forward hook on the FINAL NORM layer (not the last decoder
        block) so the captured tensor matches `output_hidden_states[-1]`
        bitwise. Hooking the last decoder block instead returns the
        PRE-norm activation, which is numerically different (verified
        max-abs-diff ~163 on Qwen2.5-0.5B). Saves ~3 GB transient at 8B
        by avoiding the all-layers tuple allocation.

        Falls back to `output_hidden_states=True` for unsupported
        architectures where the final norm cannot be located.

        Returns: [B, H] float tensor (detached).
        """
        enc = self.tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=self.config.max_prompt_length, padding=True,
        ).to(self.device)

        final_norm = self._get_final_norm_layer()

        # Per §7.6.1: even though this method has @torch.no_grad(), the
        # decorator is bypassed when the model forward fires DDP's reducer
        # hooks. Wrap the call site explicitly so the reducer skips this
        # forward.
        with torch.no_grad():
            if final_norm is None:
                # Fallback: pay the all-layers allocation cost.
                outputs = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    use_cache=False,
                    output_hidden_states=True,
                )
                last_hidden = outputs.hidden_states[-1]
            else:
                captured = {}

                def _hook(module, inputs, output):
                    captured["hidden"] = output[0] if isinstance(output, tuple) else output

                handle = final_norm.register_forward_hook(_hook)
                try:
                    _ = self.model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        use_cache=False,
                    )
                finally:
                    handle.remove()
                last_hidden = captured["hidden"]  # [B, S, H]

        seq_lens = enc["attention_mask"].sum(dim=1) - 1  # index of last real token
        batch_idx = torch.arange(len(prompts), device=self.device)
        return last_hidden[batch_idx, seq_lens, :]  # [B, H]

    def _get_final_norm_layer(self):
        """Locate the post-decoder final norm whose output equals
        `model_output.hidden_states[-1]`.

        Returns the module if found, or None to trigger the
        output_hidden_states=True fallback.
        """
        m = self.model
        # Llama / Qwen / Mistral: m.model.norm (RMSNorm)
        if hasattr(m, "model") and hasattr(m.model, "norm"):
            return m.model.norm
        # GPT-NeoX / GPT-J style: m.transformer.final_layer_norm
        if hasattr(m, "transformer"):
            for attr in ("final_layer_norm", "ln_f"):
                if hasattr(m.transformer, attr):
                    return getattr(m.transformer, attr)
        return None

    def _get_last_decoder_layer(self):
        """Best-effort access to the model's last decoder block.

        Kept for compatibility with code that wants the pre-norm activation.
        Most callers should use _get_final_norm_layer instead so the
        captured hidden state matches output_hidden_states[-1].
        """
        m = self.model
        # Most HF causal LMs: model.model.layers (Llama, Qwen, Mistral)
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers[-1]
        # GPT-style: model.transformer.h
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h[-1]
        raise RuntimeError(
            f"Could not locate last decoder layer for {type(m).__name__}. "
            "Add support in _get_last_decoder_layer or fall back to "
            "output_hidden_states=True."
        )

    # ── Critic evaluation on prompts (batched) ───────────────────────────────

    @torch.no_grad()
    def _eval_critic_on_prompts(self, prompts: List[str]) -> np.ndarray:
        """Batched critic evaluation on a list of prompts. Returns numpy array."""
        self.model.eval()
        self.critic.eval()

        if not self._critic_trainable:
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
        is_first_epoch: bool = False,
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
        # P12: On epoch 0 with NO reference anchor, new_per_token is bitwise-
        # identical to old_per_token (no optimizer step has occurred yet) and
        # we can skip the redundant forward pass — the PPO surrogate ratio is
        # 1.0, kl(old||new)=0, and the critic still updates normally.
        # B1 fix: when reference_kl_coeff > 0, we MUST run a real forward on
        # epoch 0 too. Otherwise kl_ref = (clone(old) - ref) is constant w.r.t.
        # policy params, contributing zero gradient on epoch 0 and silently
        # losing 1/K of the reference-anchor signal (25% at K=4).
        skip_forward = is_first_epoch and self.config.reference_kl_coeff == 0
        if skip_forward:
            new_per_token = old_per_token.detach().clone()
        else:
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

        # If total_loss has no gradient, the entire epoch is a no-op
        # (e.g. P12 epoch-0 skip with capacity="none" and no reference KL).
        # Skip backward + step but still produce a clean metrics dict.
        if total_loss.requires_grad:
            if self._is_ddp:
                self.accelerator.backward(total_loss)
            else:
                total_loss.backward()

            # Clip and capture grad norm BEFORE the optimizer step.
            # The metric reports the *pre-clip* L2 norm so it's informative
            # even when clipping engages. We need this because policy_loss
            # is ~0 when new and old policies match exactly (same model
            # state at start of epoch 1).
            if self._is_ddp:
                policy_grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.grad_clip_norm,
                )
                if self._critic_trainable:
                    self.accelerator.clip_grad_norm_(
                        self.critic.parameters(), max_norm=self.config.grad_clip_norm,
                    )
            else:
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.grad_clip_norm,
                )
                if self._critic_trainable:
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), max_norm=self.config.grad_clip_norm,
                    )
            self.policy_optimizer.step()
            if self.critic_optimizer:
                self.critic_optimizer.step()
        else:
            # No-op epoch: zero gradient by construction. No backward, no step.
            policy_grad_norm = torch.tensor(0.0, device=self.device)

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
        if not self._critic_trainable:
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
            # P12: skip the redundant policy forward on epoch 0 (ratio is
            # identically 1.0 since no optimizer step has occurred). Guard
            # with K>=2 so K=1 still gets a real policy update.
            is_first_epoch = (epoch == 0 and self.config.n_ppo_epochs >= 2)
            metrics = self.ppo_update(
                batch,
                precomputed_advantages=fixed_advantages,
                precomputed_old_per_token_log_probs=fixed_old_per_token,
                precomputed_response_mask=fixed_response_mask,
                precomputed_ref_per_token_log_probs=fixed_ref_per_token,
                is_first_epoch=is_first_epoch,
            )
            all_metrics.append(metrics)

        # Average scalar metrics over epochs
        aggregated = {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0]
        }
        # Accuracy always reduces over the per-rollout deterministic verifier
        # reward (gsm8k_reward), which is populated unconditionally in
        # generate_rollouts. This keeps the metric binary and comparable
        # across reward sources — works for the "none" tier, the self-judge
        # path, the combined path, and the learned-RM path uniformly.
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
        actual_n_eval = len(eval_prompts)

        # Under DDP, shard the eval set across ranks; gather rewards at the end.
        # We pad the eval set up to a multiple of world_size so each rank gets
        # an equal slice (silent sample-drop bug otherwise: when n_eval is not
        # divisible by ws, _shard_list previously truncated the trailing
        # remainder). After gather we trim the duplicated padding back off.
        pad_count = 0
        if self._is_ddp:
            rank = self.accelerator.process_index
            ws = self.accelerator.num_processes
            pad_count = (ws - actual_n_eval % ws) % ws
            if pad_count > 0:
                # Repeat from the head of the eval set; these duplicates are
                # truncated after gather_for_metrics so they do not bias
                # accuracy.
                eval_prompts = eval_prompts + eval_prompts[:pad_count]
                eval_gts = eval_gts + eval_gts[:pad_count]
            eval_prompts = _shard_list(eval_prompts, rank, ws)
            eval_gts = _shard_list(eval_gts, rank, ws)

        # Self-judge wrapper indexes by completion order; without this call
        # reward_mode='self_judge' crashes mid-eval because the wrapper has no
        # questions context. Pass the LOCAL shard so questions and rewards
        # stay aligned per-rank.
        if hasattr(self.reward_fn, 'set_questions'):
            self.reward_fn.set_questions(eval_prompts)

        # Unwrap under DDP for .generate().
        gen_model = self.accelerator.unwrap_model(self.model) if self._is_ddp else self.model

        eval_batch_size = min(self.config.eval_batch_size, len(eval_prompts)) if eval_prompts else self.config.eval_batch_size
        # `accuracy` is ALWAYS sourced from gsm8k_reward, regardless of which
        # scorer trained the policy. This keeps the binary metric meaningful
        # under continuous learned-RM rewards. (See reward_model_integration.md
        # "Accuracy metric: gsm8k_reward stays the source of truth".)
        accuracy_rewards: List[float] = []

        for start in range(0, len(eval_prompts), eval_batch_size):
            batch_p = eval_prompts[start:start + eval_batch_size]
            batch_gt = eval_gts[start:start + eval_batch_size]

            enc = self.tokenizer(
                batch_p, return_tensors="pt", truncation=True,
                max_length=self.config.max_prompt_length, padding=True,
            ).to(self.device)

            out = gen_model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,  # greedy for deterministic eval
                pad_token_id=self.tokenizer.pad_token_id,
            )

            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
            batch_completions: List[str] = []
            for i in range(len(batch_p)):
                pad_len = (enc["input_ids"][i] == self.tokenizer.pad_token_id).sum().item()
                real_start = pad_len
                pl = prompt_lens[i]
                completion = self.tokenizer.decode(
                    out[i][real_start + pl:], skip_special_tokens=True
                )
                batch_completions.append(completion)

            # Note: when a learned RM is configured, we still SCORE training
            # rewards via score_batch in generate_rollouts, but the evaluate()
            # accuracy must come from gsm8k_reward (binary, comparable across
            # reward sources). We therefore call self.reward_fn ONLY in the
            # baseline path (where it routes through make_reward_fn /
            # gsm8k_reward / self_judge / combined as today). When
            # reward_model_scorer is set, we bypass reward_fn entirely here
            # and call gsm8k_reward directly so the metric is not contaminated
            # by self-judge log-likelihoods or RM scores.
            if self.reward_model_scorer is not None:
                for c, gt in zip(batch_completions, batch_gt):
                    accuracy_rewards.append(gsm8k_reward(c, gt))
            else:
                for c, gt in zip(batch_completions, batch_gt):
                    accuracy_rewards.append(self.reward_fn(c, gt))

        if self._is_ddp:
            local_rewards_t = torch.tensor(accuracy_rewards, device=self.device)
            all_rewards_t = self.accelerator.gather_for_metrics(local_rewards_t)
            all_rewards = all_rewards_t.cpu().tolist()
            # Trim the padding duplicates we added so accuracy is computed on
            # the original n_eval set.
            if pad_count > 0:
                all_rewards = all_rewards[:actual_n_eval]
            return compute_accuracy(all_rewards)
        return compute_accuracy(accuracy_rewards)


# ── Convenience loader ────────────────────────────────────────────────────────

def load_ppo_trainer(
    config: PPOConfig,
    device_or_accelerator: Union[torch.device, "Accelerator"],
) -> Tuple[PPOTrainer, Callable]:
    """
    Load model + tokenizer from HuggingFace, build critic, return PPOTrainer.

    Accepts either a torch.device (legacy single-process path) OR an
    Accelerator (DDP path). Detects by isinstance.

    Supports bfloat16 on GPU (via config.torch_dtype) and gradient
    checkpointing for large models.  The critic is always kept in float32
    for numerical stability.
    """
    # Conflict-resolution: a learned RM and a non-deterministic reward_mode
    # (self_judge or combined) both produce continuous rewards from a frozen
    # LM. Combining them adds a third reward source that has not been
    # specified or asked for. Reject early with a clear error.
    # (See reward_model_integration.md "Interaction with reward_mode
    # (orthogonality contract)".)
    if (config.reward_model_capacity != "none"
            and config.reward_mode != "deterministic"):
        raise ValueError(
            f"reward_model_capacity={config.reward_model_capacity!r} is "
            f"incompatible with reward_mode={config.reward_mode!r}. Use "
            f"reward_mode='deterministic' and tune reward_blend_alpha to mix "
            f"the learned RM with the gsm8k verifier."
        )

    # Detect whether the caller is opting into the Accelerate path.
    if Accelerator is not None and isinstance(device_or_accelerator, Accelerator):
        accelerator = device_or_accelerator
        device = accelerator.device
        is_ddp = True
    else:
        accelerator = None
        device = device_or_accelerator
        is_ddp = False

    # Rank-0 print helper. Under DDP every rank runs `load_ppo_trainer`;
    # without gating, every "[PPO] Loading model: ..." line fires N times.
    # The `accelerator.main_process_first` block below serialises the
    # actual HF load, so the prints aren't garbled — but they duplicate.
    def _print0(*args, **kwargs):
        if accelerator is None or accelerator.is_main_process:
            print(*args, **kwargs)

    # Determine dtype
    if config.torch_dtype == "auto":
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif config.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    _print0(f"[PPO] Loading model: {config.model_name} (device={device}, dtype={torch_dtype})")

    # Under DDP, every rank would otherwise hit HuggingFace concurrently on
    # cold cache: rate-limit hits, redundant downloads, on-disk cache thrash.
    # Wrap downloads in accelerator.main_process_first() so rank 0 populates
    # the cache first; other ranks wait on the implicit barrier and then read
    # from the now-warm cache. No-op on the legacy single-process path.
    if accelerator is not None:
        with accelerator.main_process_first():
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"  # REQUIRED for batched generation
            # Under DDP, accelerator.prepare() inside PPOTrainer.__init__
            # handles placement; do NOT call .to(device) on the policy here.
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                dtype=torch_dtype,
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # REQUIRED for batched generation
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch_dtype,
        )
        model = model.to(device)

    # Enable gradient checkpointing for large models
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        _print0("[PPO] Gradient checkpointing enabled")

    hidden_size = model.config.hidden_size
    _print0(f"[PPO] Model hidden size: {hidden_size} | "
            f"Critic capacity: {config.critic_capacity}")

    # Keep critic in float32 even when model is bf16
    critic = build_critic(config.critic_capacity, hidden_size)
    if not is_ddp:
        critic = critic.to(device)  # stays float32

    # Load reference model for KL anchoring (L14) when enabled.
    # NOTE: this loads a SECOND copy of the model -- roughly doubling
    # parameter memory. On a single GPU with an 8B model in bf16 this
    # is ~16 GB extra. We do not enable gradient checkpointing on the
    # reference model because it never sees gradients.
    reference_model = None
    if config.reference_kl_coeff > 0:
        _print0(f"[PPO] reference_kl_coeff={config.reference_kl_coeff} > 0; "
                f"loading frozen reference model (doubles weight memory)")

        # Optional bnb quantization (memory_optimization §11.2). Reference
        # model is never trained, so quantization is safe.
        quantization_config = None
        if config.reference_quant in ("int8", "nf4"):
            try:
                from transformers import BitsAndBytesConfig
                if config.reference_quant == "int8":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                else:  # nf4
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                    )
            except ImportError:
                _print0(
                    f"[PPO] WARNING: reference_quant='{config.reference_quant}' "
                    "but bitsandbytes not installed; loading reference at full "
                    "precision. Install with: pip install bitsandbytes"
                )

        # Same rationale as the policy load above: under DDP, gate the HF
        # download/cache through main_process_first to avoid concurrent
        # cache thrash + rate limits.
        if accelerator is not None:
            with accelerator.main_process_first():
                reference_model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    dtype=torch_dtype,
                    quantization_config=quantization_config,
                )
        else:
            reference_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                dtype=torch_dtype,
                quantization_config=quantization_config,
            )
        # When using bnb quantization, the model is already on its target device;
        # .to(device) would error. Only call .to() when no quant is active.
        if quantization_config is None:
            reference_model = reference_model.to(device)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad_(False)

    reward_fn, diagnostic_fn = make_reward_fn(
        config, reference_model=reference_model, tokenizer=tokenizer,
    )

    # Optionally build a learned reward-model scorer. Returns None when
    # config.reward_model_capacity == "none" — the trainer's fast path
    # only activates for non-"none" tiers, preserving baseline parity.
    from ppo_specs.reward_model import build_reward_model
    reward_model_scorer = build_reward_model(
        config,
        device,
        base_model=reference_model if config.reward_model_reuse_reference else None,
    )
    if reward_model_scorer is not None:
        # Defense in depth: a learned RM must be fully frozen so DDP's
        # reducer never tries to all-reduce its gradients (and so the
        # optimizer never updates it).
        assert all(not p.requires_grad for p in reward_model_scorer.parameters()), (
            "build_reward_model returned a scorer with trainable parameters; "
            "the RM must be frozen at load time."
        )

    if is_ddp:
        trainer = PPOTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=reward_fn,
            accelerator=accelerator,
            reference_model=reference_model,
            reward_model_scorer=reward_model_scorer,
        )
    else:
        trainer = PPOTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=reward_fn,
            device=device,
            reference_model=reference_model,
            reward_model_scorer=reward_model_scorer,
        )
    return trainer, diagnostic_fn
