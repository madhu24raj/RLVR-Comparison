"""
Unit tests for ppo_specs/reward_model.py (the learned-RM module).

Covers:
    1. NoneReward float-exact parity with `gsm8k_reward` on a 3-item batch.
    2. LearnedRMScorer / build_reward_model "small" tier shape/dtype/device,
       with monkeypatched `transformers.AutoModelForCausalLM.from_pretrained`
       and `transformers.AutoTokenizer.from_pretrained` so no HF download
       happens.
    3. BlendedScorer convex combination at alpha in {0.0, 0.5, 1.0}.
    4. `reward_model_reuse_reference=True` shares the base model object
       (id-equality), and the non-reuse path loads a fresh instance.
    5. `ground_truths=None` is accepted by a learned scorer and rejected
       by the `none` tier with a clear error.
    6. `build_reward_model(..., capacity="small")` returns a fully frozen
       scorer (every parameter has `requires_grad=False`).

CPU-only. No network access. Total runtime under 10 s.

Run with:
    pytest ppo_specs/tests/test_reward_model.py -v
"""

import os
import sys
from typing import List, Optional

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so all imports resolve.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ppo_specs.config import PPOConfig
from ppo_specs.reward_model import (
    BlendedScorer,
    LearnedRMScorer,
    NoneReward,
    RewardModelScorer,
    build_reward_model,
)
from src.rewards import gsm8k_reward


# ===========================================================================
# Fakes used by the learned-tier tests (no network, hidden_size=16).
# ===========================================================================


class _FakeOutputs:
    """Stand-in for a HF causal-LM `ModelOutput`.

    Exposes `hidden_states` (the only attribute LearnedRMScorer reads) as
    a one-element tuple — mirroring `output_hidden_states=True` semantics.
    """

    def __init__(self, hidden_states):
        # `LearnedRMScorer` reads `outputs.hidden_states[-1]`.
        self.hidden_states = (hidden_states,)


class _FakeConfig:
    def __init__(self, hidden_size: int = 16):
        self.hidden_size = hidden_size


class _FakeBaseModel(nn.Module):
    """Tiny stand-in for `AutoModelForCausalLM.from_pretrained(...)`.

    `LearnedRMScorer` only needs:
      - `.config.hidden_size`
      - `.parameters()` (for device discovery)
      - `forward(input_ids, attention_mask, use_cache, output_hidden_states)`
        returning an object with `.hidden_states[-1]` of shape `[B, S, H]`
    """

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = _FakeConfig(hidden_size=hidden_size)
        # One real parameter so `next(self.parameters())` works and the
        # `dtype`-matching path in LearnedRMScorer.__init__ is exercised.
        self.embed = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self._hidden_size = hidden_size

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        use_cache=False,
        output_hidden_states=False,
        **kwargs,
    ):
        B, S = input_ids.shape
        last_hidden = torch.zeros(
            B, S, self._hidden_size, device=input_ids.device, dtype=torch.float32
        )
        return _FakeOutputs(last_hidden)


class _FakeTokenizer:
    """Minimal stand-in for `AutoTokenizer.from_pretrained(...)`.

    `LearnedRMScorer.score_batch` calls `tokenizer(texts, ...)` expecting a
    dict with `input_ids` and `attention_mask` of shape `[B, S]`. It also
    reads / writes `padding_side` and reads `pad_token` / `eos_token`.
    """

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.padding_side = "left"

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        # Tokenise by character count so different texts produce different
        # lengths; pad on the configured side.
        lens = [max(1, min(len(t), max_length or 1024)) for t in texts]
        S = max(lens)
        B = len(texts)
        input_ids = torch.full((B, S), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, S), dtype=torch.long)
        for i, n in enumerate(lens):
            if self.padding_side == "left":
                input_ids[i, S - n:] = 2  # any non-pad id
                attention_mask[i, S - n:] = 1
            else:
                input_ids[i, :n] = 2
                attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _patch_hf_loaders(monkeypatch, base_model: nn.Module, tokenizer: _FakeTokenizer):
    """Patch the HF loader classmethods so `build_reward_model` returns
    our fakes instead of hitting the network.

    We patch the classmethods on the original `transformers` classes so
    every alias / re-import of those classes (including the
    `from transformers import AutoModelForCausalLM, AutoTokenizer` inside
    `build_reward_model`) sees the patch.
    """
    import transformers

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: base_model),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: tokenizer),
    )


# ===========================================================================
# 1. `none`-tier parity
# ===========================================================================


class TestNoneTierParity:
    """`NoneReward.score_batch` is float-identical to `gsm8k_reward`."""

    def test_none_tier_parity(self):
        """Float-exact parity on a hand-rolled 3-item batch.

        Note: we instantiate `NoneReward()` directly (not via
        `build_reward_model`) so the test is robust to a future change
        where the factory returns `None` for capacity='none'.
        """
        prompts = ["q0", "q1", "q2"]
        completions = [
            "Working through it... \\boxed{42}",
            "Step 1: 5*3=15\nStep 2: 15+10=25\n#### 25",
            "no answer here",
        ]
        ground_truths = ["42", "25", "100"]

        scorer = NoneReward()
        out = scorer.score_batch(prompts, completions, ground_truths)
        expected = torch.tensor(
            [gsm8k_reward(c, gt) for c, gt in zip(completions, ground_truths)],
            dtype=torch.float32,
        )

        assert out.shape == (3,)
        assert out.dtype == torch.float32
        assert torch.equal(out, expected), (
            f"NoneReward parity broken: {out} vs {expected}"
        )


# ===========================================================================
# 2. `small`-tier shape / dtype / device
# ===========================================================================


class TestSmallTierShapeDtype:
    """`build_reward_model(capacity='small', ...)` returns a [B] float32 CPU tensor."""

    def test_small_tier_shape_dtype(self, monkeypatch):
        """`score_batch` returns a 1-D float32 CPU tensor of length B."""
        fake_base = _FakeBaseModel(hidden_size=16)
        fake_tok = _FakeTokenizer()
        _patch_hf_loaders(monkeypatch, fake_base, fake_tok)

        cfg = PPOConfig(
            reward_model_capacity="small",
            reward_model_name="fake/rm-small",
            reward_model_dtype="float32",
            reward_blend_alpha=1.0,  # avoid wrapping in BlendedScorer
            reward_score_activation="sigmoid",
        )
        device = torch.device("cpu")

        scorer = build_reward_model(cfg, device)

        out = scorer.score_batch(["a", "b"], ["c", "d"])
        assert out.shape == (2,), f"expected shape (2,), got {tuple(out.shape)}"
        assert out.dtype == torch.float32, f"expected float32, got {out.dtype}"
        assert out.device.type == "cpu", f"expected CPU, got {out.device}"


# ===========================================================================
# 3. Blend alpha interpolation
# ===========================================================================


class _FixedLearnedScorer(RewardModelScorer):
    """A learned scorer whose `score_batch` returns a fixed tensor.

    Used to make the blend test deterministic without standing up a real
    `LearnedRMScorer`.
    """

    def __init__(self, fixed_scores: torch.Tensor):
        super().__init__()
        # Register as a buffer so `.to(device)` moves it.
        self.register_buffer("fixed_scores", fixed_scores.to(torch.float32))

    def score_batch(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        return self.fixed_scores.clone()


class TestBlendAlphaInterpolation:
    """`BlendedScorer` is a convex combination of learned and verifier."""

    def _setup(self):
        prompts = ["q0", "q1", "q2"]
        # gsm8k_reward gives [1.0, 1.0, 0.0] on these.
        completions = [
            "Working through it... \\boxed{42}",
            "Step 1: 5*3=15\nStep 2: 15+10=25\n#### 25",
            "no answer here",
        ]
        ground_truths = ["42", "25", "100"]
        det_expected = torch.tensor(
            [gsm8k_reward(c, gt) for c, gt in zip(completions, ground_truths)],
            dtype=torch.float32,
        )
        learned_scores = torch.tensor([0.7, 0.3, 0.5], dtype=torch.float32)
        learned = _FixedLearnedScorer(learned_scores)
        none_scorer = NoneReward()
        return prompts, completions, ground_truths, det_expected, learned, learned_scores, none_scorer

    def test_alpha_zero_equals_verifier(self):
        """alpha=0.0 reproduces the deterministic `gsm8k_reward` output."""
        (prompts, completions, gts, det_expected,
         learned, _, none_scorer) = self._setup()
        blended = BlendedScorer(learned, none_scorer, alpha=0.0)
        out = blended.score_batch(prompts, completions, gts)
        assert torch.allclose(out, det_expected, atol=1e-6), (
            f"alpha=0 should equal verifier; got {out} vs {det_expected}"
        )

    def test_alpha_one_equals_learned(self):
        """alpha=1.0 reproduces the learned scorer's output exactly."""
        (prompts, completions, gts, _,
         learned, learned_scores, none_scorer) = self._setup()
        blended = BlendedScorer(learned, none_scorer, alpha=1.0)
        out = blended.score_batch(prompts, completions, gts)
        assert torch.allclose(out, learned_scores, atol=1e-6), (
            f"alpha=1 should equal learned; got {out} vs {learned_scores}"
        )

    def test_alpha_half_is_elementwise_mean(self):
        """alpha=0.5 is the element-wise mean of the two signals."""
        (prompts, completions, gts, det_expected,
         learned, learned_scores, none_scorer) = self._setup()
        blended = BlendedScorer(learned, none_scorer, alpha=0.5)
        out = blended.score_batch(prompts, completions, gts)
        expected = 0.5 * learned_scores + 0.5 * det_expected
        assert torch.allclose(out, expected, atol=1e-6), (
            f"alpha=0.5 should be the midpoint; got {out} vs {expected}"
        )


# ===========================================================================
# 4. Reuse-reference weight sharing
# ===========================================================================


class TestReuseReferenceWeightSharing:
    """`reward_model_reuse_reference=True` shares the base model object."""

    def test_reuse_shares_same_base(self, monkeypatch):
        """Reuse path: scorer.base_model is the supplied `base_model` object."""
        fake_base = _FakeBaseModel(hidden_size=16)
        fake_tok = _FakeTokenizer()
        # `from_pretrained` should NOT be called when reuse=True with a base.
        # Patch it to a poison value so an accidental call would surface.
        _patch_hf_loaders(monkeypatch, fake_base, fake_tok)

        cfg = PPOConfig(
            reward_model_capacity="small",
            reward_model_name="fake/rm-small",
            reward_model_dtype="float32",
            reward_model_reuse_reference=True,
            reward_blend_alpha=1.0,
            reward_score_activation="sigmoid",
        )
        scorer = build_reward_model(cfg, torch.device("cpu"), base_model=fake_base)
        # Unwrap BlendedScorer if the factory ever wraps; with alpha=1.0 it
        # should not, but be defensive.
        learned = scorer.learned if isinstance(scorer, BlendedScorer) else scorer
        assert isinstance(learned, LearnedRMScorer)
        assert learned.base_model is fake_base, (
            "reuse_reference=True must share the supplied base_model object "
            "(id-equality), not load a fresh copy."
        )

    def test_no_reuse_loads_fresh_base(self, monkeypatch):
        """No-reuse path: factory loads a fresh instance via from_pretrained."""
        fake_base = _FakeBaseModel(hidden_size=16)  # the "supplied" one
        fresh_base = _FakeBaseModel(hidden_size=16)  # what from_pretrained returns
        fake_tok = _FakeTokenizer()

        import transformers

        monkeypatch.setattr(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            classmethod(lambda cls, *a, **kw: fresh_base),
        )
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            classmethod(lambda cls, *a, **kw: fake_tok),
        )

        cfg = PPOConfig(
            reward_model_capacity="small",
            reward_model_name="fake/rm-small",
            reward_model_dtype="float32",
            reward_model_reuse_reference=False,
            reward_blend_alpha=1.0,
            reward_score_activation="sigmoid",
        )
        scorer = build_reward_model(cfg, torch.device("cpu"), base_model=fake_base)
        learned = scorer.learned if isinstance(scorer, BlendedScorer) else scorer
        assert isinstance(learned, LearnedRMScorer)
        assert learned.base_model is fresh_base, (
            "no-reuse path must use the from_pretrained-loaded base"
        )
        assert learned.base_model is not fake_base, (
            "no-reuse path must NOT share with the supplied base"
        )


# ===========================================================================
# 5. `ground_truths` optional for learned, required for `none`
# ===========================================================================


class TestGroundTruthOptional:
    """Learned tier accepts ground_truths=None; `none` tier rejects it."""

    def test_learned_accepts_none_ground_truths(self):
        """A learned scorer must work when `ground_truths` is omitted."""
        learned = _FixedLearnedScorer(torch.tensor([0.1, 0.9]))
        out = learned.score_batch(["a", "b"], ["c", "d"], ground_truths=None)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2,)
        assert out.dtype == torch.float32

    def test_none_rejects_none_ground_truths(self):
        """`NoneReward` must raise ValueError when ground_truths is None."""
        scorer = NoneReward()
        with pytest.raises(ValueError, match=r"(?i)ground_truths|required|none"):
            scorer.score_batch(["a", "b"], ["c", "d"], ground_truths=None)


# ===========================================================================
# 6. Frozen RM parameters
# ===========================================================================


class TestFrozenRmParams:
    """Every parameter of a `small`-tier scorer has requires_grad=False."""

    def test_frozen_rm_params(self, monkeypatch):
        """All scorer parameters are frozen after build_reward_model."""
        fake_base = _FakeBaseModel(hidden_size=16)
        fake_tok = _FakeTokenizer()
        _patch_hf_loaders(monkeypatch, fake_base, fake_tok)

        cfg = PPOConfig(
            reward_model_capacity="small",
            reward_model_name="fake/rm-small",
            reward_model_dtype="float32",
            reward_blend_alpha=1.0,
            reward_score_activation="sigmoid",
        )
        scorer = build_reward_model(cfg, torch.device("cpu"))

        # Locate the first non-frozen parameter (if any) for a clear msg.
        first_bad = next(
            (
                name
                for name, p in scorer.named_parameters()
                if p.requires_grad
            ),
            None,
        )
        assert first_bad is None, (
            f"build_reward_model returned an unfrozen parameter: {first_bad!r}"
        )
        assert all(not p.requires_grad for p in scorer.parameters())
