"""
Tier-based learned Reward Model (RM) for PPO.

This module is an *additive* layer on top of the existing reward stack
(`gsm8k_reward`, `SelfJudgeRewardModel`, `_RewardFnWrapper`,
`make_reward_fn` in `src/rewards.py`). It is NOT a replacement for any
of those.  When `config.reward_model_capacity == "none"`, this module's
`NoneReward` produces float-identical output to a direct call to
`gsm8k_reward(completion, ground_truth)` per sample, preserving the
PPO baseline invariant.

When `reward_model_capacity != "none"`, a `LearnedRMScorer` wraps a
frozen `AutoModelForCausalLM`-shaped base model with a scalar value
head and an output-squashing activation. Optionally, the scorer is
further wrapped in a `BlendedScorer` to mix the learned signal with
the verifiable `gsm8k_reward` (per `reward_blend_alpha`).

Mirrors the structure of `ppo_specs/critic.py`: classes first, then a
single factory `build_reward_model(config, device, *, base_model=None)`,
then a `__main__` smoke that exercises only the `none` tier (no HF
download).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running as `python ppo_specs/reward_model.py` from the repo root
# (mirrors the bootstrap in ppo_specs/ppo_trainer.py:37-39).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.rewards import gsm8k_reward  # noqa: E402


# ── Base class ────────────────────────────────────────────────────────────────

class RewardModelScorer(nn.Module):
    """Common interface for all reward-model tiers.

    Concrete subclasses implement `score_batch`, which returns a
    `[B]`-shaped float32 tensor — one scalar reward per (prompt,
    completion) pair.

    `ground_truths` is optional in the signature for backward-compat
    with preference-trained RMs that ignore the ground truth, but is
    REQUIRED by the `none` tier (which falls through to `gsm8k_reward`).
    """

    def score_batch(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> torch.Tensor:  # [B] float32
        raise NotImplementedError


# ── `none` tier ───────────────────────────────────────────────────────────────

class NoneReward(RewardModelScorer):
    """Capacity `none`: thin wrapper around the deterministic verifier.

    Produces float-identical output to a per-sample loop calling
    `gsm8k_reward(completion, ground_truth)`. This is the baseline-
    invariant path: when the trainer routes through `NoneReward`, its
    reward signal is bit-identical to today's deterministic reward.
    """

    def __init__(self):
        super().__init__()
        # No learnable parameters, but keep a non-trainable dummy so the
        # module is still a valid `nn.Module` for downstream `.to(device)`
        # calls and parameter iteration.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def score_batch(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if ground_truths is None:
            raise ValueError(
                "ground_truths required for capacity='none' tier"
            )
        scores = [
            gsm8k_reward(c, gt) for c, gt in zip(completions, ground_truths)
        ]
        return torch.tensor(scores, dtype=torch.float32, device=self._dummy.device)


# ── learned tier ──────────────────────────────────────────────────────────────

class LearnedRMScorer(RewardModelScorer):
    """Learned reward model: frozen LM base + linear value head + activation.

    Architecture (mirrors the TRL value-head convention and the
    existing `_extract_last_hidden` pattern in `ppo_specs/ppo_trainer.py`):

      score = activation( value_head( hidden_state_at_last_real_token(
                              prompt ++ completion ) ) )

    Both `base_model` and `value_head` are frozen (`requires_grad=False`)
    at construction time. The trainer never updates them.

    The class accepts an optional `tokenizer`; the spec note in the
    integration doc explicitly calls out that "if the RM uses a different
    tokenizer than the policy, the `score_batch` implementation must
    re-tokenize from strings". The factory `build_reward_model` loads the
    matching tokenizer for `config.reward_model_name` and threads it in.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        activation: str,
        tokenizer=None,
        max_length: int = 1024,
    ):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(hidden_size, 1)
        # Match the base model's parameter dtype so the matmul in the
        # value head doesn't trigger an autocast / dtype-mismatch error.
        try:
            base_param = next(base_model.parameters())
            self.value_head.to(dtype=base_param.dtype)
        except StopIteration:
            pass
        if activation not in ("sigmoid", "tanh", "none"):
            raise ValueError(
                f"reward_score_activation must be 'sigmoid' | 'tanh' | 'none'; "
                f"got {activation!r}"
            )
        self.activation = activation
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Freeze every parameter. The reward model is never trained.
        for p in self.parameters():
            p.requires_grad_(False)

        assert all(not p.requires_grad for p in self.parameters()), (
            "LearnedRMScorer parameters must all be frozen at construction time"
        )

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        return x

    @torch.no_grad()
    def score_batch(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError(
                "LearnedRMScorer requires a tokenizer; build via "
                "`build_reward_model(...)` so the matching tokenizer is loaded."
            )

        # Concatenate prompt and completion at the string level so the
        # RM tokenizer applies its own word-piece boundary, regardless
        # of what tokenizer the policy used.
        full_texts = [p + c for p, c in zip(prompts, completions)]

        # Left-padding so the LAST token of every sample is a real
        # (non-pad) token. This mirrors the policy tokenizer's
        # `padding_side = "left"` setting and lets us extract the
        # final-token hidden state without indexing on a per-row basis.
        prev_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        finally:
            self.tokenizer.padding_side = prev_padding_side

        device = next(self.base_model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, S, H]

        # With left-padding the last real token sits at index S-1.
        # We still index via the attention mask for robustness against
        # tokenizers that special-case left-padding (some prepend a BOS
        # so the last real token isn't strictly S-1).
        seq_lens = attention_mask.sum(dim=1) - 1  # [B] index of last real tok
        batch_idx = torch.arange(last_hidden.shape[0], device=device)
        last_token_hidden = last_hidden[batch_idx, seq_lens, :]  # [B, H]

        # Value head + activation, then upcast to float32 for the
        # downstream PPO advantage normalization which assumes fp32.
        raw = self.value_head(last_token_hidden).squeeze(-1)  # [B]
        squashed = self._apply_activation(raw)
        return squashed.to(dtype=torch.float32)


# ── blended tier ──────────────────────────────────────────────────────────────

class BlendedScorer(RewardModelScorer):
    """Convex combination of a learned scorer and the `none`-tier verifier.

    `final_reward = alpha * learned.score_batch(...) + (1 - alpha) *
                    none_scorer.score_batch(...)`

    `alpha == 1.0` short-circuits to the learned scorer (the factory
    avoids constructing a `BlendedScorer` in that case). `alpha == 0.0`
    is exactly today's deterministic reward.
    """

    def __init__(
        self,
        learned: RewardModelScorer,
        none_scorer: NoneReward,
        alpha: float,
    ):
        super().__init__()
        self.learned = learned
        self.none_scorer = none_scorer
        self.alpha = float(alpha)

    def score_batch(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if ground_truths is None and self.alpha < 1.0:
            raise ValueError(
                "ground_truths required when reward_blend_alpha < 1.0 "
                "(BlendedScorer falls through to gsm8k_reward for the "
                "(1 - alpha) component)."
            )
        learned_scores = self.learned.score_batch(prompts, completions, ground_truths)
        det_scores = self.none_scorer.score_batch(prompts, completions, ground_truths)
        # Both branches return float32 already; cast defensively in case
        # a future learned scorer returns bf16.
        learned_scores = learned_scores.to(dtype=torch.float32)
        det_scores = det_scores.to(dtype=torch.float32, device=learned_scores.device)
        return self.alpha * learned_scores + (1.0 - self.alpha) * det_scores


# ── factory ───────────────────────────────────────────────────────────────────

def build_reward_model(
    config,
    device: torch.device,
    *,
    base_model: Optional[nn.Module] = None,
) -> Optional[RewardModelScorer]:
    """Build the reward model described by `config`.

    Mirrors `ppo_specs/critic.py:build_critic`. Returns a frozen,
    `eval()`-mode `RewardModelScorer` placed on `device`, OR returns
    ``None`` when ``config.reward_model_capacity == "none"``.

    Returning ``None`` for the "none" tier keeps the PPO baseline
    invariant trivially satisfied: the trainer checks
    ``self.reward_model_scorer is not None`` to decide whether to run
    the batched-RM fast path, so under capacity="none" the existing
    per-sample ``self.reward_fn(...)`` loop runs exactly as today —
    bit-identical metrics. The ``NoneReward`` class itself stays as a
    stable, importable test double (used by RM-TEST and any external
    code that wants the wrapped-gsm8k_reward behavior in the
    ``RewardModelScorer`` shape).

    For non-"none" capacities, loads / reuses a base LM, wraps it in
    ``LearnedRMScorer``, and optionally further wraps in ``BlendedScorer``.

    Args:
        config: A ``PPOConfig`` (or duck-typed equivalent).
        device: Target device.
        base_model: Optional pre-loaded base model for weight reuse
            (e.g. the frozen reference model). Honored only when
            ``config.reward_model_reuse_reference is True``.

    Returns:
        A ``RewardModelScorer`` ready for inference, or ``None`` when
        ``capacity == "none"``.
    """
    capacity = str(getattr(config, "reward_model_capacity", "none")).strip().lower()

    if capacity == "none":
        # Baseline-invariant path: signal "no learned RM configured" by
        # returning None. The trainer keeps using its per-sample reward_fn.
        return None

    # ─── learned tier ─────────────────────────────────────────────────────────
    # Local imports so the `none`-tier path (and the `__main__` smoke) does
    # not pay the transformers import cost.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rm_name = getattr(config, "reward_model_name", None)
    if rm_name is None:
        raise ValueError(
            f"reward_model_capacity={capacity!r} requires "
            "reward_model_name to be set (HF hub id or local path)."
        )

    # Resolve dtype. "auto" → bf16 on CUDA, fp32 on CPU (mirrors the
    # policy/reference dtype resolution at ppo_trainer.py:1064-1070).
    dtype_str = str(getattr(config, "reward_model_dtype", "auto")).strip().lower()
    if dtype_str == "auto":
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype_str == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(
            f"reward_model_dtype must be 'auto' | 'bfloat16' | 'float32'; "
            f"got {dtype_str!r}"
        )

    # Decide between reuse and fresh load.
    reuse = bool(getattr(config, "reward_model_reuse_reference", False))
    if reuse and base_model is not None:
        rm_base = base_model
    else:
        if reuse and base_model is None:
            # User asked for reuse but no base was provided (e.g. reference_kl_coeff=0
            # so load_ppo_trainer never built a reference_model). Loading fresh
            # silently would defeat the memory-saving intent — surface it loudly.
            raise ValueError(
                "reward_model_reuse_reference=True but no base_model was supplied "
                "to build_reward_model(). This usually means reference_kl_coeff=0, "
                "so load_ppo_trainer did not create a reference_model. Either set "
                "reference_kl_coeff > 0, or set reward_model_reuse_reference=False "
                "and accept the extra memory cost of loading a fresh RM base."
            )
        rm_base = AutoModelForCausalLM.from_pretrained(
            rm_name,
            dtype=torch_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(rm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    activation = str(getattr(config, "reward_score_activation", "sigmoid"))

    learned = LearnedRMScorer(
        rm_base,
        rm_base.config.hidden_size,
        activation,
        tokenizer=tokenizer,
    )

    alpha = float(getattr(config, "reward_blend_alpha", 1.0))
    if alpha < 1.0:
        scorer = BlendedScorer(learned, NoneReward(), alpha)
    else:
        scorer = learned

    scorer = scorer.to(device)
    scorer.eval()

    # Defense in depth: if a future caller mutates `requires_grad` on
    # the base model, this assertion fires before the trainer ever
    # tries to compute gradients on the RM.
    assert all(not p.requires_grad for p in scorer.parameters()), (
        "build_reward_model returned a scorer with trainable parameters; "
        "the RM must be fully frozen."
    )
    return scorer


# ── self-test (no HF download) ────────────────────────────────────────────────

if __name__ == "__main__":
    # Hand-rolled batch: two correct, one wrong. The `none` tier wraps
    # `gsm8k_reward`, so we expect [1.0, 1.0, 0.0].
    completions = [
        "Working through it... \\boxed{42}",
        "Step 1: 5*3=15\nStep 2: 15+10=25\n#### 25",
        "Some random text that never commits to an answer.",
    ]
    ground_truths = ["42", "25", "100"]
    prompts = ["q0", "q1", "q2"]  # ignored by the `none` tier

    scorer = NoneReward()
    out = scorer.score_batch(prompts, completions, ground_truths)
    print(f"NoneReward.score_batch -> {out} (shape={list(out.shape)}, dtype={out.dtype})")

    # Float-identical parity check vs direct gsm8k_reward calls.
    direct = [gsm8k_reward(c, gt) for c, gt in zip(completions, ground_truths)]
    direct_t = torch.tensor(direct, dtype=torch.float32)
    print(f"gsm8k_reward direct    -> {direct_t}")
    assert torch.equal(out, direct_t), (
        f"NoneReward parity broken: {out} vs {direct_t}"
    )
    print("parity ok (float-identical to gsm8k_reward)")
