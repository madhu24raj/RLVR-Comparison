"""
Comprehensive tests for advantage estimation and critic modules.

Covers:
  - compute_advantages (basic, normalization, gradient flow, edge cases)
  - estimate_mc_advantages (with mocked model/tokenizer)
  - advantage_estimation_error
  - critic_approximation_error
  - Critic architectures (none/small/medium/large) via build_critic
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from ppo_specs.advantage import (
    advantage_estimation_error,
    compute_advantages,
    critic_approximation_error,
    estimate_mc_advantages,
)
from ppo_specs.critic import (
    LargeCriticMLP,
    MediumCriticHead,
    REINFORCEBaseline,
    SmallCriticMLP,
    build_critic,
)


# ============================================================================
# 1. compute_advantages -- Basic Cases
# ============================================================================

class TestComputeAdvantagesBasic:
    """Basic correctness tests for compute_advantages."""

    def test_reinforce_baseline_values_none(self):
        """With values=None (REINFORCE), advantages = r - mean(r)."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = compute_advantages(rewards, values=None, normalize=False)
        expected = rewards - rewards.mean()
        assert torch.allclose(adv, expected), f"Expected {expected}, got {adv}"

    def test_learned_critic_values(self):
        """With critic values, advantages = r - V(s).detach()."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.5, 2.5])
        adv = compute_advantages(rewards, values=values, normalize=False)
        expected = rewards - values
        assert torch.allclose(adv, expected)

    def test_values_are_detached(self):
        """Values should be detached -- no gradient flows through critic."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
        adv = compute_advantages(rewards, values=values, normalize=False)
        # The advantages should not require grad (values were detached)
        assert not adv.requires_grad

    def test_all_zero_rewards(self):
        """All-zero rewards should yield all-zero advantages."""
        rewards = torch.zeros(5)
        adv = compute_advantages(rewards, values=None, normalize=False)
        assert torch.allclose(adv, torch.zeros(5))

    def test_all_same_rewards_reinforce(self):
        """All-same rewards with REINFORCE baseline should yield zeros."""
        rewards = torch.full((6,), 0.7)
        adv = compute_advantages(rewards, values=None, normalize=False)
        assert torch.allclose(adv, torch.zeros(6), atol=1e-7)

    def test_binary_rewards(self):
        """Binary rewards [0, 0, 1, 1] should produce correct advantages."""
        rewards = torch.tensor([0.0, 0.0, 1.0, 1.0])
        adv = compute_advantages(rewards, values=None, normalize=False)
        mean_r = 0.5
        expected = torch.tensor([-0.5, -0.5, 0.5, 0.5])
        assert torch.allclose(adv, expected)


# ============================================================================
# 2. compute_advantages -- Normalization
# ============================================================================

class TestComputeAdvantagesNormalization:
    """Tests for the normalization behaviour (full z-score: subtract mean, divide by std)."""

    def test_normalize_full_zscore(self):
        """Normalization should apply full z-score: (A - mean) / (std + eps).

        Standard PPO (OpenAI baselines, SB3, "37 Implementation Details of PPO")
        uses full z-score normalization for advantages.
        """
        rewards = torch.tensor([0.0, 0.0, 1.0, 1.0])
        adv_raw = compute_advantages(rewards, values=None, normalize=False)
        adv_norm = compute_advantages(rewards, values=None, normalize=True)

        raw_std = adv_raw.std()
        expected = (adv_raw - adv_raw.mean()) / (raw_std + 1e-8)
        assert torch.allclose(adv_norm, expected, atol=1e-6), (
            f"Expected {expected}, got {adv_norm}"
        )

    def test_normalized_mean_is_zero(self):
        """After z-score normalization, mean of advantages should be ~0."""
        rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
        adv_norm = compute_advantages(rewards, values=None, normalize=True)
        assert abs(adv_norm.mean().item()) < 1e-6, (
            f"Expected mean ~0, got {adv_norm.mean().item()}"
        )

    def test_normalize_false_returns_raw(self):
        """normalize=False should return unscaled advantages."""
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        adv = compute_advantages(rewards, values=values, normalize=False)
        expected = rewards - values
        assert torch.allclose(adv, expected)

    def test_normalization_skipped_when_std_near_zero(self):
        """When std < 1e-8, normalization should be skipped."""
        rewards = torch.full((4,), 3.0)
        adv = compute_advantages(rewards, values=None, normalize=True)
        # All same rewards -> advantages are all zero -> std is 0 -> skip norm
        assert torch.allclose(adv, torch.zeros(4), atol=1e-7)

    def test_normalization_skipped_for_single_element(self):
        """Single-element tensor (numel() <= 1) should skip normalization."""
        rewards = torch.tensor([5.0])
        values = torch.tensor([2.0])
        adv = compute_advantages(rewards, values=values, normalize=True)
        expected = torch.tensor([3.0])
        assert torch.allclose(adv, expected)

    def test_normalized_std_approximately_one(self):
        """After normalization, the std of advantages should be ~1.0."""
        rewards = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        adv = compute_advantages(rewards, values=None, normalize=True)
        assert abs(adv.std().item() - 1.0) < 0.1, (
            f"Expected std ~1.0, got {adv.std().item()}"
        )


# ============================================================================
# 3. compute_advantages -- Gradient Flow
# ============================================================================

class TestComputeAdvantagesGradientFlow:
    """Tests that gradients are properly detached."""

    def test_advantages_have_no_gradient(self):
        """Advantages should not carry gradients from values."""
        rewards = torch.tensor([1.0, 0.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        adv = compute_advantages(rewards, values=values, normalize=False)
        assert not adv.requires_grad

    def test_detach_prevents_gradient_flow(self):
        """Even when values require grad, backward through advantages fails."""
        rewards = torch.tensor([1.0, 0.0])
        values = torch.tensor([0.3, 0.7], requires_grad=True)
        adv = compute_advantages(rewards, values=values, normalize=False)

        # adv does not require grad, so we cannot call .backward() on it
        # This is the correct behaviour: no gradient flows back to critic
        assert not adv.requires_grad
        assert values.grad is None  # no grad was computed


# ============================================================================
# 4. compute_advantages -- Edge Cases
# ============================================================================

class TestComputeAdvantagesEdgeCases:
    """Edge-case robustness tests."""

    def test_very_large_rewards(self):
        """Should handle very large reward values without overflow."""
        rewards = torch.tensor([1e6, 1e6, 0.0, 0.0])
        adv = compute_advantages(rewards, values=None, normalize=False)
        expected = rewards - rewards.mean()
        assert torch.allclose(adv, expected)

    def test_negative_rewards(self):
        """Should handle negative rewards correctly."""
        rewards = torch.tensor([-1.0, -2.0, 3.0, 0.0])
        adv = compute_advantages(rewards, values=None, normalize=False)
        expected = rewards - rewards.mean()
        assert torch.allclose(adv, expected)

    def test_gamma_parameter_accepted(self):
        """gamma parameter is accepted but does not affect single-step computation."""
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        adv1 = compute_advantages(rewards, values=None, gamma=0.5, normalize=False)
        adv2 = compute_advantages(rewards, values=None, gamma=0.99, normalize=False)
        assert torch.allclose(adv1, adv2)


# ============================================================================
# 5. estimate_mc_advantages
# ============================================================================

class TestEstimateMCAdvantages:
    """Tests for Monte Carlo advantage estimation with mocked model."""

    @staticmethod
    def _make_mock_policy_and_tokenizer(fixed_reward: float = 0.5):
        """Create a mock policy model and tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 0
        # tokenizer() returns a dict-like with input_ids
        enc_result = MagicMock()
        enc_result.__getitem__ = lambda self, key: torch.tensor([[1, 2, 3]])
        enc_result.to = MagicMock(return_value=enc_result)
        tokenizer.return_value = enc_result
        tokenizer.decode = MagicMock(return_value="42")

        policy = MagicMock()
        policy.eval = MagicMock()
        # generate must return [B, seq_len] matching the input batch size
        def _mock_generate(input_ids, **kwargs):
            B = input_ids.shape[0]
            return torch.tensor([[1, 2, 3, 4, 5]]).repeat(B, 1)
        policy.generate = MagicMock(side_effect=_mock_generate)

        return policy, tokenizer

    def test_returns_dict_per_prompt(self):
        """Returned dict should have one entry per prompt."""
        policy, tokenizer = self._make_mock_policy_and_tokenizer()
        reward_fn = MagicMock(return_value=1.0)
        prompts = ["What is 1+1?", "What is 2+2?"]
        ground_truths = ["2", "4"]

        result = estimate_mc_advantages(
            policy, tokenizer, prompts, ground_truths,
            reward_fn, n_samples=3, max_new_tokens=16,
        )
        assert isinstance(result, dict)
        assert len(result) == 2
        assert set(result.keys()) == set(prompts)

    def test_mc_values_between_0_and_1_binary(self):
        """MC values for binary rewards should be in [0, 1]."""
        policy, tokenizer = self._make_mock_policy_and_tokenizer()
        # Alternate between 0 and 1
        call_count = {"n": 0}
        def binary_reward(completion, gt):
            call_count["n"] += 1
            return float(call_count["n"] % 2)

        result = estimate_mc_advantages(
            policy, tokenizer, ["prompt1"], ["answer1"],
            binary_reward, n_samples=10,
        )
        val = result["prompt1"]
        assert 0.0 <= val <= 1.0, f"MC value {val} not in [0, 1]"

    def test_more_samples_gives_stable_estimate(self):
        """Higher n_samples should give a more stable (lower variance) estimate."""
        policy, tokenizer = self._make_mock_policy_and_tokenizer()

        rng = np.random.RandomState(42)
        def noisy_reward(completion, gt):
            return float(rng.choice([0.0, 1.0]))

        # Run multiple independent estimations, compare variance
        variances = {}
        for n_samples in [5, 50]:
            estimates = []
            for seed in range(10):
                rng = np.random.RandomState(seed)
                result = estimate_mc_advantages(
                    policy, tokenizer, ["p"], ["a"],
                    noisy_reward, n_samples=n_samples,
                )
                estimates.append(result["p"])
            variances[n_samples] = np.var(estimates)

        # More samples should generally give lower variance
        # Allow some tolerance since this is stochastic
        assert variances[50] <= variances[5] + 0.05, (
            f"Expected var(50 samples)={variances[50]:.4f} <= "
            f"var(5 samples)={variances[5]:.4f} + tolerance"
        )


# ============================================================================
# 6. advantage_estimation_error
# ============================================================================

class TestAdvantageEstimationError:
    """Tests for the MAE-based advantage estimation error."""

    def test_identical_baselines(self):
        """Identical estimated and MC baselines should give error 0."""
        a = np.array([0.5, 0.5, 0.5])
        assert advantage_estimation_error(a, a) == pytest.approx(0.0)

    def test_known_offset(self):
        """Constant offset should equal that offset."""
        est = np.array([1.0, 2.0, 3.0])
        mc = np.array([1.5, 2.5, 3.5])
        assert advantage_estimation_error(est, mc) == pytest.approx(0.5)

    def test_symmetry(self):
        """error(a, b) should equal error(b, a)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 1.5, 4.0])
        assert advantage_estimation_error(a, b) == pytest.approx(
            advantage_estimation_error(b, a)
        )

    def test_mixed_positive_negative_errors(self):
        """Should handle a mix of positive and negative differences."""
        est = np.array([1.0, 3.0, 2.0])
        mc = np.array([2.0, 1.0, 2.0])
        # |1-2| + |3-1| + |2-2| = 1 + 2 + 0 = 3, mean = 1.0
        assert advantage_estimation_error(est, mc) == pytest.approx(1.0)


# ============================================================================
# 7. critic_approximation_error
# ============================================================================

class TestCriticApproximationError:
    """Tests for RMSE-based critic approximation error."""

    def test_identical_values(self):
        """Identical critic and MC values should give RMSE 0."""
        v = np.array([0.3, 0.7, 0.5])
        assert critic_approximation_error(v, v) == pytest.approx(0.0)

    def test_known_values(self):
        """Verify RMSE formula with known values."""
        critic_vals = np.array([1.0, 2.0, 3.0])
        mc_vals = np.array([1.0, 2.0, 4.0])
        # MSE = (0 + 0 + 1) / 3 = 1/3, RMSE = sqrt(1/3)
        expected = math.sqrt(1.0 / 3.0)
        assert critic_approximation_error(critic_vals, mc_vals) == pytest.approx(
            expected, abs=1e-7
        )

    def test_returns_rmse_not_mse(self):
        """Should return RMSE (with sqrt), not raw MSE."""
        critic_vals = np.array([0.0, 0.0])
        mc_vals = np.array([1.0, 1.0])
        # MSE = 1.0, RMSE = 1.0 (coincidentally same here)
        # Use a case where they differ:
        critic_vals2 = np.array([0.0, 0.0])
        mc_vals2 = np.array([2.0, 2.0])
        # MSE = 4.0, RMSE = 2.0
        result = critic_approximation_error(critic_vals2, mc_vals2)
        assert result == pytest.approx(2.0)
        assert result != pytest.approx(4.0)  # MSE would be 4.0


# ============================================================================
# 8. Critic Architecture Tests
# ============================================================================

class TestCriticArchitectures:
    """Tests for the four critic capacities and the build_critic factory."""

    HIDDEN_SIZES = [896, 4096]  # Qwen2.5-0.5B and Llama-scale
    BATCH_SIZE = 4

    # -- Factory tests --------------------------------------------------------

    @pytest.mark.parametrize("capacity", ["none", "small", "medium", "large"])
    def test_build_critic_all_capacities(self, capacity):
        """build_critic should return a valid module for all four capacities."""
        critic = build_critic(capacity, hidden_size=896)
        assert isinstance(critic, nn.Module)

    def test_build_critic_unknown_raises(self):
        """build_critic should raise ValueError for unknown capacity."""
        with pytest.raises(ValueError, match="Unknown critic capacity"):
            build_critic("extra_large", hidden_size=896)

    # -- Output shape tests ---------------------------------------------------

    @pytest.mark.parametrize("capacity", ["none", "small", "medium", "large"])
    @pytest.mark.parametrize("hidden_size", [896, 4096])
    def test_output_shape(self, capacity, hidden_size):
        """Output should be [batch_size] for every architecture and hidden_size."""
        batch = self.BATCH_SIZE
        critic = build_critic(capacity, hidden_size)
        h = torch.randn(batch, hidden_size)
        out = critic(h)
        assert out.shape == (batch,), f"Expected ({batch},), got {out.shape}"

    # -- is_trainable tests ---------------------------------------------------

    def test_reinforce_not_trainable(self):
        """REINFORCEBaseline.is_trainable() should return False."""
        critic = build_critic("none", hidden_size=896)
        assert not critic.is_trainable()

    @pytest.mark.parametrize("capacity", ["small", "medium", "large"])
    def test_learned_critics_trainable(self, capacity):
        """Learned critics should report is_trainable() == True."""
        critic = build_critic(capacity, hidden_size=896)
        assert critic.is_trainable()

    # -- REINFORCEBaseline specifics ------------------------------------------

    def test_reinforce_returns_zeros(self):
        """REINFORCEBaseline should always return a zero tensor."""
        critic = REINFORCEBaseline()
        h = torch.randn(8, 512)
        out = critic(h)
        assert torch.allclose(out, torch.zeros(8))

    def test_reinforce_no_trainable_parameters(self):
        """REINFORCEBaseline should have zero trainable parameters."""
        critic = REINFORCEBaseline()
        n_trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)
        assert n_trainable == 0

    # -- Parameter count sanity -----------------------------------------------

    def test_parameter_counts_ordering(self):
        """Parameter counts should follow: none < medium < small < large."""
        hidden = 896
        counts = {}
        for cap in ("none", "small", "medium", "large"):
            c = build_critic(cap, hidden)
            counts[cap] = sum(p.numel() for p in c.parameters() if p.requires_grad)

        assert counts["none"] == 0
        assert counts["medium"] > 0
        # medium is a single linear (hidden+1 params), small has a hidden layer
        assert counts["small"] > counts["medium"]
        assert counts["large"] > counts["small"]

    @pytest.mark.parametrize("hidden_size", [896, 4096])
    def test_parameter_counts_reasonable(self, hidden_size):
        """Parameter counts should scale reasonably with hidden_size."""
        for cap in ("small", "medium", "large"):
            critic = build_critic(cap, hidden_size)
            n_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
            assert n_params > 0, f"{cap} critic should have trainable parameters"
            # Sanity: no critic should exceed 200M params for hidden_size<=4096
            assert n_params < 200_000_000, f"{cap} critic has too many params: {n_params}"

    # -- Gradient flow through trainable critics ------------------------------

    @pytest.mark.parametrize("capacity", ["small", "medium", "large"])
    def test_gradient_flows_through_trainable_critics(self, capacity):
        """Backward pass should produce gradients for trainable critic params."""
        hidden = 896
        critic = build_critic(capacity, hidden)
        h = torch.randn(4, hidden)
        out = critic(h)
        loss = out.sum()
        loss.backward()

        grads_found = False
        for p in critic.parameters():
            if p.requires_grad and p.grad is not None:
                grads_found = True
                break
        assert grads_found, f"No gradients found in {capacity} critic after backward"

    # -- Forward pass with various hidden sizes -------------------------------

    @pytest.mark.parametrize("capacity", ["small", "medium", "large"])
    @pytest.mark.parametrize("hidden_size", [896, 4096])
    def test_forward_pass_various_hidden(self, capacity, hidden_size):
        """Forward pass should succeed for typical backbone hidden sizes."""
        critic = build_critic(capacity, hidden_size)
        h = torch.randn(2, hidden_size)
        out = critic(h)
        assert out.shape == (2,)
        assert torch.isfinite(out).all(), "Output contains non-finite values"
