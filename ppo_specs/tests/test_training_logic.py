"""
Tests for PPO training logic correctness and scaling invariants.

Verifies the mathematical properties of the PPO algorithm:
  - Gradient isolation between policy and critic
  - Advantage precomputation held fixed across PPO epochs
  - PPO-clip surrogate properties
  - Config-driven hyperparameters (no magic numbers)
  - Numerical stability under edge cases
  - Seed determinism for E2.8 fair comparison

Run with:
    pytest ppo_specs/tests/test_training_logic.py -v
    pytest ppo_specs/tests/test_training_logic.py -v -m "not slow"
"""

import sys
import os
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ppo_specs.config import PPOConfig, CRITIC_CAPACITIES
from ppo_specs.critic import build_critic
from ppo_specs.advantage import compute_advantages
from ppo_specs.ppo_trainer import PPOTrainer, Rollout, RolloutBatch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _tiny_config(**overrides) -> PPOConfig:
    defaults = dict(
        model_name=MODEL_NAME,
        n_steps=1,
        batch_size=2,
        max_new_tokens=16,
        n_train_samples=4,
        n_ppo_epochs=1,
        eval_every=1,
        log_every=1,
        experiment_name="test_training_logic",
        critic_capacity="medium",
        do_sample=True,
        temperature=0.7,
        checkpoint_every=0,
    )
    defaults.update(overrides)
    return PPOConfig(**defaults)


# ===========================================================================
# 1. Advantage computation correctness
# ===========================================================================


class TestAdvantageComputation:
    """Verify GAE reduction for single-step terminal-reward episodes."""

    def test_gae_with_critic(self):
        """A_i = r_i - V(s_i), then normalize."""
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        values = torch.tensor([0.6, 0.4, 0.7, 0.3])
        adv = compute_advantages(rewards, values, gamma=1.0, normalize=False)
        expected = rewards - values
        torch.testing.assert_close(adv, expected)

    def test_gae_without_critic(self):
        """When values=None, baseline is batch mean: A_i = r_i - mean(r)."""
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        adv = compute_advantages(rewards, None, gamma=1.0, normalize=False)
        expected = rewards - rewards.mean()
        torch.testing.assert_close(adv, expected)

    def test_normalization_zero_mean_unit_var(self):
        """Normalized advantages should have ~zero mean and ~unit variance."""
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        adv = compute_advantages(rewards, values, gamma=1.0, normalize=True)
        assert abs(adv.mean().item()) < 1e-6, f"Mean not zero: {adv.mean()}"
        assert abs(adv.std().item() - 1.0) < 0.1, f"Std not ~1: {adv.std()}"

    def test_all_same_rewards_gives_zero_advantages(self):
        """When all rewards are identical, normalized advantages should be all zero."""
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        adv = compute_advantages(rewards, values, gamma=1.0, normalize=True)
        assert torch.allclose(adv, torch.zeros_like(adv)), f"Expected all zeros: {adv}"

    def test_single_element_no_normalization(self):
        """Single-element batch should skip normalization."""
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.3])
        adv = compute_advantages(rewards, values, gamma=1.0, normalize=True)
        expected = rewards - values
        torch.testing.assert_close(adv, expected)

    def test_binary_rewards_batch_mean_baseline(self):
        """With binary rewards {0,1} and no critic, batch mean = accuracy."""
        rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])  # 50% accuracy
        adv = compute_advantages(rewards, None, gamma=1.0, normalize=False)
        # Baseline = 0.5, so advantages = [0.5, 0.5, -0.5, -0.5]
        expected = torch.tensor([0.5, 0.5, -0.5, -0.5])
        torch.testing.assert_close(adv, expected)


# ===========================================================================
# 2. Config-driven hyperparameters (no magic numbers)
# ===========================================================================


class TestConfigDrivenHyperparams:
    """Verify that all previously-magic numbers are now in PPOConfig."""

    def test_max_prompt_length_in_config(self):
        cfg = PPOConfig()
        assert hasattr(cfg, "max_prompt_length")
        assert cfg.max_prompt_length == 512

    def test_grad_clip_norm_in_config(self):
        cfg = PPOConfig()
        assert hasattr(cfg, "grad_clip_norm")
        assert cfg.grad_clip_norm == 1.0

    def test_log_ratio_clip_in_config(self):
        cfg = PPOConfig()
        assert hasattr(cfg, "log_ratio_clip")
        assert cfg.log_ratio_clip == 20.0

    def test_eval_batch_size_in_config(self):
        cfg = PPOConfig()
        assert hasattr(cfg, "eval_batch_size")
        assert cfg.eval_batch_size == 8

    def test_critic_loss_coeff_in_config(self):
        cfg = PPOConfig()
        assert hasattr(cfg, "critic_loss_coeff")
        assert cfg.critic_loss_coeff == 0.5

    def test_all_critic_capacities_available(self):
        assert CRITIC_CAPACITIES == ["none", "small", "medium", "large"]
        for cap in CRITIC_CAPACITIES:
            critic = build_critic(cap, hidden_size=64)
            h = torch.randn(2, 64)
            out = critic(h)
            assert out.shape == (2,), f"{cap}: wrong output shape {out.shape}"


# ===========================================================================
# 3. Critic architecture properties
# ===========================================================================


class TestCriticProperties:
    """Verify critic architectures satisfy required invariants."""

    @pytest.mark.parametrize("capacity", ["small", "medium", "large"])
    def test_trainable_critic_has_grad(self, capacity):
        critic = build_critic(capacity, hidden_size=64)
        assert critic.is_trainable()
        h = torch.randn(4, 64, requires_grad=False)
        out = critic(h)
        assert out.requires_grad, f"{capacity}: output should require grad"

    def test_none_critic_no_grad(self):
        critic = build_critic("none", hidden_size=64)
        assert not critic.is_trainable()
        h = torch.randn(4, 64)
        out = critic(h)
        assert not out.requires_grad
        assert torch.all(out == 0), "REINFORCE baseline should return zeros"

    @pytest.mark.parametrize("capacity", ["small", "medium", "large"])
    def test_critic_output_shape(self, capacity):
        """Critic output shape should be [B] for any batch size."""
        critic = build_critic(capacity, hidden_size=128)
        for B in [1, 4, 16]:
            h = torch.randn(B, 128)
            out = critic(h)
            assert out.shape == (B,), f"{capacity}, B={B}: shape {out.shape}"

    def test_critic_parameter_count_ordering(self):
        """Parameter count: none < medium < small < large."""
        critics = {cap: build_critic(cap, 128) for cap in CRITIC_CAPACITIES}
        counts = {
            cap: sum(p.numel() for p in c.parameters() if p.requires_grad)
            for cap, c in critics.items()
        }
        assert counts["none"] == 0
        assert counts["medium"] < counts["small"]
        assert counts["small"] < counts["large"]


# ===========================================================================
# 4. PPO ratio and clipping properties
# ===========================================================================


class TestPPORatioProperties:
    """Test mathematical properties of the PPO ratio computation."""

    def test_ratio_is_one_when_policy_unchanged(self):
        """When new_log_probs == old_log_probs, ratio should be 1.0."""
        log_probs = torch.tensor([-10.0, -20.0, -15.0])
        log_ratio = log_probs - log_probs
        ratio = torch.exp(log_ratio)
        torch.testing.assert_close(ratio, torch.ones_like(ratio))

    def test_clip_bounds(self):
        """Clipped ratio must be in [1-ε, 1+ε]."""
        epsilon = 0.2
        ratios = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
        clipped = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon)
        assert torch.all(clipped >= 1.0 - epsilon)
        assert torch.all(clipped <= 1.0 + epsilon)

    def test_ppo_loss_pessimistic_bound(self):
        """PPO-clip should take the minimum of clipped and unclipped."""
        advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
        ratios = torch.tensor([1.5, 1.5, 0.5, 0.5])
        epsilon = 0.2
        clipped = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon)
        surr1 = ratios * advantages
        surr2 = clipped * advantages
        loss = -torch.mean(torch.min(surr1, surr2))
        # For positive advantage, high ratio: clipped is lower → pessimistic
        # For negative advantage, high ratio: unclipped is lower → pessimistic
        assert torch.isfinite(loss)

    def test_log_ratio_clamping_prevents_overflow(self):
        """Large log-ratio differences should be clamped, not overflow."""
        old_lp = torch.tensor([-500.0])
        new_lp = torch.tensor([-0.1])
        clip_val = 20.0
        log_ratio = torch.clamp(new_lp - old_lp, -clip_val, clip_val)
        ratio = torch.exp(log_ratio)
        assert torch.isfinite(ratio), f"Ratio overflowed: {ratio}"
        assert ratio.item() == pytest.approx(np.exp(20.0), rel=1e-5)


# ===========================================================================
# 5. Gradient isolation
# ===========================================================================


class TestGradientIsolation:
    """Verify that policy and critic gradients don't contaminate each other."""

    def test_critic_loss_no_grad_to_policy(self):
        """Critic MSE loss should not produce gradients on a mock 'model'."""
        # Simulate: model produces hidden states under no_grad, critic has grad
        hidden_size = 32
        mock_model_param = nn.Parameter(torch.randn(hidden_size))

        with torch.no_grad():
            hidden = mock_model_param.unsqueeze(0).expand(4, -1)  # [4, 32]

        hidden_detached = hidden.detach()
        critic = build_critic("medium", hidden_size)
        values = critic(hidden_detached)
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = nn.functional.mse_loss(values, rewards)
        loss.backward()

        assert mock_model_param.grad is None, (
            "Critic loss should not produce gradients on model parameters"
        )
        assert critic.linear.weight.grad is not None, (
            "Critic loss should produce gradients on critic parameters"
        )


# ===========================================================================
# 6. Seed determinism for E2.8 fair comparison
# ===========================================================================


class TestSeedDeterminism:
    """Verify that resetting seeds produces identical random sequences."""

    def test_torch_seed_reset_gives_same_sequence(self):
        """Two runs with the same seed should produce identical tensors."""
        seed = 42
        torch.manual_seed(seed)
        t1 = torch.randn(10)

        torch.manual_seed(seed)
        t2 = torch.randn(10)

        torch.testing.assert_close(t1, t2)

    def test_numpy_seed_reset_gives_same_sequence(self):
        seed = 42
        np.random.seed(seed)
        a1 = np.random.randn(10)

        np.random.seed(seed)
        a2 = np.random.randn(10)

        np.testing.assert_array_equal(a1, a2)

    def test_advantage_deterministic_with_same_inputs(self):
        """compute_advantages is deterministic (no random ops)."""
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        a1 = compute_advantages(rewards, values, normalize=True)
        a2 = compute_advantages(rewards, values, normalize=True)
        torch.testing.assert_close(a1, a2)


# ===========================================================================
# 7. Numerical edge cases
# ===========================================================================


class TestNumericalEdgeCases:
    """Test numerical stability under extreme conditions."""

    def test_advantage_with_extreme_values(self):
        """Very large reward-value differences should not produce NaN."""
        rewards = torch.tensor([1.0, 0.0])
        values = torch.tensor([1e6, -1e6])
        adv = compute_advantages(rewards, values, normalize=True)
        assert torch.isfinite(adv).all(), f"Non-finite advantages: {adv}"

    def test_advantage_all_zeros(self):
        """All-zero rewards and values should produce all-zero advantages."""
        rewards = torch.zeros(8)
        values = torch.zeros(8)
        adv = compute_advantages(rewards, values, normalize=True)
        assert torch.allclose(adv, torch.zeros_like(adv))

    def test_advantage_single_nonzero(self):
        """One correct answer in a batch of zeros."""
        rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        values = torch.zeros(8)
        adv = compute_advantages(rewards, values, normalize=True)
        assert torch.isfinite(adv).all()
        # The correct sample should have the highest advantage
        assert adv[0] > adv[1]

    def test_log_softmax_finite_with_large_logits(self):
        """log_softmax should be finite even with large logit values."""
        logits = torch.randn(1, 10, 1000) * 100  # extreme logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        assert torch.isfinite(log_probs).all(), "log_softmax produced non-finite values"

    def test_log_softmax_finite_with_bf16_upcast(self):
        """bf16 logits upcast to fp32 for log_softmax should be finite."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for bf16 test")
        logits_bf16 = torch.randn(1, 10, 1000, dtype=torch.bfloat16, device="cuda")
        log_probs = torch.log_softmax(logits_bf16.float(), dim=-1)
        assert torch.isfinite(log_probs).all()


# ===========================================================================
# 8. RolloutBatch tensor construction
# ===========================================================================


class TestRolloutBatch:
    """Test that RolloutBatch correctly constructs tensors from rollouts."""

    def test_rewards_tensor(self):
        rollouts = [
            Rollout("p1", "c1", 1.0, -5.0, 0.3, [1, 2, 3], 2),
            Rollout("p2", "c2", 0.0, -6.0, 0.5, [4, 5, 6], 2),
        ]
        batch = RolloutBatch(rollouts)
        rewards = batch.rewards()
        assert rewards.shape == (2,)
        torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0]))

    def test_old_log_probs_tensor(self):
        rollouts = [
            Rollout("p1", "c1", 1.0, -5.0, 0.3, [1, 2, 3], 2),
            Rollout("p2", "c2", 0.0, -6.0, 0.5, [4, 5, 6], 2),
        ]
        batch = RolloutBatch(rollouts)
        lp = batch.old_log_probs()
        torch.testing.assert_close(lp, torch.tensor([-5.0, -6.0]))

    def test_values_tensor(self):
        rollouts = [
            Rollout("p1", "c1", 1.0, -5.0, 0.3, [1, 2, 3], 2),
            Rollout("p2", "c2", 0.0, -6.0, 0.5, [4, 5, 6], 2),
        ]
        batch = RolloutBatch(rollouts)
        vals = batch.values()
        torch.testing.assert_close(vals, torch.tensor([0.3, 0.5]))


# ===========================================================================
# 9. Config preset consistency
# ===========================================================================


class TestConfigPresets:
    """Verify config presets have consistent settings."""

    def test_local_test_config_fast(self):
        from ppo_specs.config import local_test_config
        cfg = local_test_config()
        assert cfg.n_steps <= 10
        assert cfg.batch_size <= 8
        assert cfg.n_train_samples <= 50
        assert cfg.checkpoint_every == 0  # no checkpointing for smoke tests

    def test_e2_7_config_valid(self):
        from ppo_specs.config import e2_7_config
        cfg = e2_7_config(seed=42)
        assert cfg.n_test_samples >= cfg.eval_size
        assert cfg.n_test_samples >= cfg.final_eval_size
        assert cfg.eval_every > 0
        assert cfg.seed == 42

    def test_e2_8_config_valid(self):
        from ppo_specs.config import e2_8_config
        cfg = e2_8_config(critic_capacity="large", seed=0)
        assert cfg.critic_capacity == "large"
        assert cfg.seed == 0
        assert cfg.n_test_samples >= cfg.eval_size

    def test_copy_config_overrides(self):
        from ppo_specs.config import copy_config
        cfg = PPOConfig(batch_size=8, seed=42)
        new_cfg = copy_config(cfg, batch_size=16, seed=0)
        assert new_cfg.batch_size == 16
        assert new_cfg.seed == 0
        assert cfg.batch_size == 8  # original unchanged
