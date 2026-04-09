"""
Unit tests for the per-token PPO loss formula.

This file is the FAST regression net for the per-token PPO refactor.
Most of these tests use raw synthetic tensors with no model load, so
the fast suite stays fast (microseconds per test).

The tests are derived from the mathematical contract of the PPO-clip
surrogate at the token level (Schulman et al. 2017, adapted for
autoregressive LMs as in TRL/InstructGPT):

    log_ratio_bt = new_log_probs_bt - old_log_probs_bt
    ratio_bt     = exp(log_ratio_bt)
    unclipped    = ratio_bt * A_b                       # broadcast A over T
    clipped      = clip(ratio_bt, 1-eps, 1+eps) * A_b
    per_token_pg = -min(unclipped, clipped)             # [B, T]
    L_pg         = (per_token_pg * mask).sum() / mask.sum()

Properties this file checks (each maps to a class below):

  1. At ratio = 1 (degenerate first-epoch case), the loss equals
     -masked_mean(broadcast(A)).
  2. The loss is invariant under arbitrary changes to *masked* positions.
  3. Clipping engages exactly when the per-token ratio leaves
     [1 - eps, 1 + eps].
  4. The gradient direction is correct: increasing log_prob of a token
     with positive advantage decreases the loss; with negative
     advantage, increases it.
  5. The all-masked degenerate case (mask.sum() == 0) does not produce
     NaN or division-by-zero.
  6. The synthetic per-token loss matches PPOTrainer.ppo_update on the
     same inputs (slow integration test, gated on model load).

Run with:
    pytest ppo_specs/tests/test_per_token_loss.py -v
    pytest ppo_specs/tests/test_per_token_loss.py -v -m "not slow"
"""
from __future__ import annotations

import sys
import os

import pytest
import torch

# Make the repo importable when running directly
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Reference implementation -- the formula under test, written in isolation
# from PPOTrainer.ppo_update so we can verify the production path against it.
# ─────────────────────────────────────────────────────────────────────────────


def reference_per_token_ppo_loss(
    new_log_probs: torch.Tensor,   # [B, T]
    old_log_probs: torch.Tensor,   # [B, T]
    advantages: torch.Tensor,      # [B]
    mask: torch.Tensor,            # [B, T] -- 1 = real token, 0 = padding
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """The mathematical PPO-clip surrogate at the token level, masked-mean.

    This is intentionally NOT a copy-paste from ppo_trainer.py -- it's
    written from the mathematical definition. If both produce the same
    output on the same inputs, that's evidence the production code is
    correct (or wrong in the same way, but that's much less likely than
    one path being wrong).
    """
    log_ratio = (new_log_probs - old_log_probs).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)                              # [B, T]
    A = advantages.unsqueeze(-1)                              # [B, 1]
    unclipped = ratio * A
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * A
    per_token_pg = -torch.min(unclipped, clipped)             # [B, T]
    mask_sum = mask.sum().clamp(min=1.0)
    return (per_token_pg * mask).sum() / mask_sum


# ─────────────────────────────────────────────────────────────────────────────
# 1. At ratio = 1, loss collapses to -masked_mean(broadcast(A))
# ─────────────────────────────────────────────────────────────────────────────


class TestRatioOneDegenerate:
    """When new_log_probs == old_log_probs, ratio = 1 everywhere.

    The PPO clip is inactive (1 is inside [1-eps, 1+eps]), so the loss
    reduces to -masked_mean(A_broadcast). This is the first-epoch case
    that produces the misleading 'policy_loss == 0' artifact when A is
    z-score normalised over the batch.
    """

    def test_ratio_one_zero_advantages(self):
        """All-zero advantages → loss = 0."""
        B, T = 4, 5
        new_lp = torch.zeros(B, T)
        old_lp = torch.zeros(B, T)
        A = torch.zeros(B)
        mask = torch.ones(B, T)
        loss = reference_per_token_ppo_loss(new_lp, old_lp, A, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_ratio_one_uniform_advantages(self):
        """All-equal advantages = 0.5: loss = -0.5 (since ratio*A = 0.5)."""
        B, T = 4, 5
        lp = torch.full((B, T), -1.0)
        A = torch.full((B,), 0.5)
        mask = torch.ones(B, T)
        loss = reference_per_token_ppo_loss(lp, lp, A, mask)
        assert loss.item() == pytest.approx(-0.5, abs=1e-6)

    def test_ratio_one_zero_mean_advantages(self):
        """Z-scored advantages (mean=0): masked mean of broadcast is 0 → loss=0.

        This is the case that makes policy_loss read ~0 in the K=1 first
        epoch even though the gradient is nonzero. Documented in the
        ppo_update docstring; we test the *value* here.
        """
        B, T = 4, 5
        lp = torch.full((B, T), -1.0)
        # Symmetric advantages summing to zero
        A = torch.tensor([+1.0, -1.0, +0.5, -0.5])
        mask = torch.ones(B, T)
        loss = reference_per_token_ppo_loss(lp, lp, A, mask)
        # broadcast: [-1, +1, -0.5, +0.5] per row, mean across all 20 entries
        # = (1 - 1 + 0.5 - 0.5) * 5 / 20 = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mask invariance: padding positions never affect the loss
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskInvariance:
    """The loss must depend only on log probs / advantages at unmasked positions.

    This is the property that justifies right-padding sequences of unequal
    length: padding tokens contribute 0 to the loss regardless of what
    log prob value they hold.
    """

    def test_changing_masked_log_probs_does_not_affect_loss(self):
        B, T = 3, 6
        new_lp = torch.tensor([
            [-0.5, -0.7, -1.0, 0.0, 0.0, 0.0],
            [-0.3, -0.4, -0.5, -0.6, 0.0, 0.0],
            [-0.8, -0.9, -1.1, -1.2, -1.3, -1.4],
        ])
        old_lp = torch.tensor([
            [-0.4, -0.6, -0.9, 0.0, 0.0, 0.0],
            [-0.2, -0.3, -0.4, -0.5, 0.0, 0.0],
            [-0.7, -0.8, -1.0, -1.1, -1.2, -1.3],
        ])
        A = torch.tensor([0.5, -0.3, 0.1])
        mask = torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ], dtype=torch.float32)

        loss_a = reference_per_token_ppo_loss(new_lp, old_lp, A, mask)

        # Now scribble arbitrary garbage at masked positions
        new_lp_b = new_lp.clone()
        new_lp_b[0, 3:] = 999.9    # row 0 has 3 padding positions
        new_lp_b[1, 4:] = -1234.5  # row 1 has 2 padding positions
        old_lp_b = old_lp.clone()
        old_lp_b[0, 3:] = -50.0
        old_lp_b[1, 4:] = +50.0

        loss_b = reference_per_token_ppo_loss(new_lp_b, old_lp_b, A, mask)
        assert loss_b.item() == pytest.approx(loss_a.item(), abs=1e-6)

    def test_mask_count_normalises_correctly(self):
        """Adding more 0s to the mask should NOT change per-token loss
        (because the divisor is mask.sum(), not B*T).

        Construct a case where we add a fully-masked row -- the loss
        should be unchanged because the new row contributes 0 to both
        numerator and denominator (after the clamp(min=1) safeguard
        only kicks in when EVERY row is masked, which we test elsewhere).
        """
        B, T = 2, 4
        new_lp = torch.tensor([
            [-0.5, -0.6, -0.7, -0.8],
            [-0.3, -0.4, -0.5, -0.6],
        ])
        old_lp = torch.tensor([
            [-0.4, -0.5, -0.6, -0.7],
            [-0.2, -0.3, -0.4, -0.5],
        ])
        A = torch.tensor([0.5, -0.5])
        mask = torch.ones(B, T)

        loss_small = reference_per_token_ppo_loss(new_lp, old_lp, A, mask)

        # Add a fully-padded row at the bottom
        new_lp3 = torch.cat([new_lp, torch.zeros(1, T)], dim=0)
        old_lp3 = torch.cat([old_lp, torch.zeros(1, T)], dim=0)
        A3 = torch.cat([A, torch.tensor([0.0])])
        mask3 = torch.cat([mask, torch.zeros(1, T)], dim=0)

        loss_large = reference_per_token_ppo_loss(new_lp3, old_lp3, A3, mask3)
        assert loss_large.item() == pytest.approx(loss_small.item(), abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Clipping behaviour
# ─────────────────────────────────────────────────────────────────────────────


class TestClipping:
    """The PPO clip should engage exactly when the per-token ratio leaves
    [1 - eps, 1 + eps], and only on the side that prevents *increasing*
    the surrogate.
    """

    def test_no_clip_when_ratio_inside_band(self):
        """Ratio = 1.1 with eps=0.2 → 1.1 ∈ [0.8, 1.2] → no clip.

        With positive advantage, loss should equal -1.1 * A (no clipping).
        """
        B, T = 1, 1
        # log_ratio = 0.0953 → ratio ≈ 1.10
        new_lp = torch.tensor([[0.0]])
        old_lp = torch.tensor([[-0.0953]])
        A = torch.tensor([1.0])
        mask = torch.ones(B, T)
        loss = reference_per_token_ppo_loss(new_lp, old_lp, A, mask, clip_epsilon=0.2)
        assert loss.item() == pytest.approx(-1.10, abs=0.01)

    def test_clip_engages_above_band_with_positive_advantage(self):
        """Ratio = 2.0, eps = 0.2, A = +1.

        unclipped = 2.0 * 1 = 2.0
        clipped   = 1.2 * 1 = 1.2
        min(2.0, 1.2) = 1.2 → loss = -1.2 (clip engaged on upside)
        """
        # log_ratio = ln(2) ≈ 0.693
        new_lp = torch.tensor([[0.693]])
        old_lp = torch.tensor([[0.0]])
        A = torch.tensor([1.0])
        mask = torch.ones(1, 1)
        loss = reference_per_token_ppo_loss(new_lp, old_lp, A, mask, clip_epsilon=0.2)
        assert loss.item() == pytest.approx(-1.2, abs=1e-3)

    def test_clip_does_not_engage_above_band_with_negative_advantage(self):
        """Ratio = 2.0, eps = 0.2, A = -1.

        unclipped = 2.0 * -1 = -2.0
        clipped   = 1.2 * -1 = -1.2
        min(-2.0, -1.2) = -2.0 → loss = +2.0  (NOT clipped: PPO uses
        the WORSE of the two so the policy is penalised for moving in
        the wrong direction).
        """
        new_lp = torch.tensor([[0.693]])
        old_lp = torch.tensor([[0.0]])
        A = torch.tensor([-1.0])
        mask = torch.ones(1, 1)
        loss = reference_per_token_ppo_loss(new_lp, old_lp, A, mask, clip_epsilon=0.2)
        assert loss.item() == pytest.approx(+2.0, abs=1e-3)

    def test_clip_engages_below_band_with_negative_advantage(self):
        """Ratio = 0.5, eps = 0.2, A = -1.

        unclipped = 0.5 * -1 = -0.5
        clipped   = 0.8 * -1 = -0.8   (0.5 clamped up to 0.8)
        min(-0.5, -0.8) = -0.8 → loss = +0.8 (clip engaged on downside)
        """
        new_lp = torch.tensor([[-0.693]])  # log(0.5)
        old_lp = torch.tensor([[0.0]])
        A = torch.tensor([-1.0])
        mask = torch.ones(1, 1)
        loss = reference_per_token_ppo_loss(new_lp, old_lp, A, mask, clip_epsilon=0.2)
        assert loss.item() == pytest.approx(+0.8, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gradient direction
# ─────────────────────────────────────────────────────────────────────────────


class TestGradientDirection:
    """Increasing the log prob of a token should:
       - DECREASE the loss when its sample's advantage is POSITIVE
       - INCREASE the loss when its sample's advantage is NEGATIVE
    These are the basic policy-gradient correctness properties.
    """

    def test_positive_advantage_loss_decreases_when_log_prob_increases(self):
        """∂L/∂(new_lp) < 0 when A > 0 (and ratio is in the unclipped band)."""
        new_lp = torch.tensor([[-1.0]], requires_grad=True)
        old_lp = torch.tensor([[-1.0]])
        A = torch.tensor([1.0])
        mask = torch.ones(1, 1)
        loss = reference_per_token_ppo_loss(new_lp, old_lp, A, mask)
        loss.backward()
        # Gradient w.r.t. new_lp should be negative (so SGD on new_lp
        # decreases loss by increasing new_lp).
        assert new_lp.grad.item() < 0, (
            f"Expected ∂L/∂new_lp < 0 with A=+1, got {new_lp.grad.item()}"
        )

    def test_negative_advantage_loss_increases_when_log_prob_increases(self):
        """∂L/∂(new_lp) > 0 when A < 0."""
        new_lp = torch.tensor([[-1.0]], requires_grad=True)
        old_lp = torch.tensor([[-1.0]])
        A = torch.tensor([-1.0])
        mask = torch.ones(1, 1)
        loss = reference_per_token_ppo_loss(new_lp, old_lp, A, mask)
        loss.backward()
        assert new_lp.grad.item() > 0, (
            f"Expected ∂L/∂new_lp > 0 with A=-1, got {new_lp.grad.item()}"
        )

    def test_zero_advantage_zero_gradient(self):
        """∂L/∂(new_lp) = 0 when A = 0 (no learning signal)."""
        new_lp = torch.tensor([[-1.0]], requires_grad=True)
        old_lp = torch.tensor([[-1.0]])
        A = torch.tensor([0.0])
        mask = torch.ones(1, 1)
        loss = reference_per_token_ppo_loss(new_lp, old_lp, A, mask)
        loss.backward()
        assert new_lp.grad.item() == pytest.approx(0.0, abs=1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Degenerate all-masked case
# ─────────────────────────────────────────────────────────────────────────────


class TestAllMaskedDegenerate:
    """If every token is masked (mask.sum() == 0), the formula must not
    produce NaN/inf or divide by zero. The clamp(min=1) on mask_sum is
    what guarantees this.
    """

    def test_all_masked_loss_is_finite_zero(self):
        B, T = 2, 3
        new_lp = torch.full((B, T), -0.5)
        old_lp = torch.full((B, T), -0.7)  # ratio != 1
        A = torch.tensor([1.0, -1.0])
        mask = torch.zeros(B, T)
        loss = reference_per_token_ppo_loss(new_lp, old_lp, A, mask)
        assert torch.isfinite(loss).item(), f"Got non-finite loss {loss}"
        # With all-zero mask, numerator is 0 and denominator is clamped
        # to 1, so loss is exactly 0.
        assert loss.item() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integration: production ppo_update agrees with the reference formula
# ─────────────────────────────────────────────────────────────────────────────
#
# This test loads the real model so it's marked @slow. The fast suite
# above already covers the loss formula correctness; this is the
# additional check that PPOTrainer.ppo_update has not drifted from the
# reference implementation.


@pytest.mark.slow
class TestProductionPpoUpdateMatchesReference:
    """ppo_update should produce a policy_loss equal to what the reference
    formula computes on the same per-token log probs, advantages, and mask.
    """

    def test_ppo_update_loss_matches_reference(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ppo_specs.config import PPOConfig
        from ppo_specs.critic import build_critic
        from ppo_specs.ppo_trainer import PPOTrainer, Rollout, RolloutBatch
        from src.rewards import gsm8k_reward

        device = torch.device("cpu")
        MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)

        cfg = PPOConfig(
            model_name=MODEL,
            batch_size=2,
            n_ppo_epochs=1,
            critic_capacity="none",
            max_new_tokens=8,
        )
        critic = build_critic("none", model.config.hidden_size).to(device)
        trainer = PPOTrainer(cfg, model, tok, critic, gsm8k_reward, device)

        # Construct two simple rollouts with known full_ids and prompt_lens
        prompt = "Solve: 2 + 2 ="
        comp = " The answer is 4."
        prompt_ids = tok.encode(prompt, add_special_tokens=True)
        comp_ids = tok.encode(comp, add_special_tokens=False)

        rollouts = [
            Rollout(
                prompt=prompt,
                completion=comp,
                reward=1.0,
                old_log_prob=0.0,  # will be overwritten
                value=0.0,
                full_ids=prompt_ids + comp_ids,
                prompt_len=len(prompt_ids),
            ),
            Rollout(
                prompt=prompt,
                completion=comp,
                reward=0.0,
                old_log_prob=0.0,
                value=0.0,
                full_ids=prompt_ids + comp_ids,
                prompt_len=len(prompt_ids),
            ),
        ]
        batch = RolloutBatch(rollouts)

        # Capture per-token log probs / mask BEFORE the update (these
        # are what the reference formula will use)
        with torch.no_grad():
            per_token_lp, mask = trainer._batched_per_token_log_probs(
                [r.full_ids for r in rollouts],
                [r.prompt_len for r in rollouts],
            )
        old_per_token = per_token_lp.detach().clone()
        frozen_mask = mask.detach().clone()

        # Compute advantages exactly as ppo_update would (with the
        # same z-score normalisation, no critic since capacity=none)
        rewards = batch.rewards().to(device)
        from ppo_specs.advantage import compute_advantages
        advantages = compute_advantages(rewards, None, gamma=1.0, normalize=True).detach()

        # Reference loss: ratio = 1 because new == old in this fresh
        # state, so the loss should equal -masked_mean(broadcast(A)).
        ref_loss = reference_per_token_ppo_loss(
            old_per_token, old_per_token, advantages, frozen_mask,
            clip_epsilon=cfg.clip_epsilon,
        )

        # Production loss: call ppo_update with the precomputed inputs
        metrics = trainer.ppo_update(
            batch,
            precomputed_advantages=advantages,
            precomputed_old_per_token_log_probs=old_per_token,
            precomputed_response_mask=frozen_mask,
        )
        prod_loss = metrics["policy_loss"]

        # Both should be ~0 (ratio=1 + symmetric normalised advantages),
        # and they should match each other to high precision.
        assert ref_loss.item() == pytest.approx(prod_loss, abs=1e-5), (
            f"reference loss {ref_loss.item():.6e} != "
            f"production loss {prod_loss:.6e}"
        )
