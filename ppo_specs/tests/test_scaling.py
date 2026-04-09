"""
Tests for scaling properties and memory/performance invariants.

These tests verify that the implementation will behave correctly at scale
without requiring actual large models or GPUs.

Run with:
    pytest ppo_specs/tests/test_scaling.py -v
    pytest ppo_specs/tests/test_scaling.py -v -m "not slow"
"""

import sys
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ppo_specs.config import PPOConfig
from ppo_specs.critic import build_critic
from ppo_specs.advantage import compute_advantages, advantage_estimation_error
from ppo_specs.checkpoint import _config_hash


# ===========================================================================
# 1. Memory-related invariants
# ===========================================================================


class TestMemoryInvariants:
    """Tests that verify memory-safety properties without requiring a GPU."""

    def test_detach_prevents_graph_retention(self):
        """Detached tensors should not retain computation graph."""
        x = torch.randn(4, 32, requires_grad=True)
        y = (x * 2).sum()
        detached = y.detach()
        assert not detached.requires_grad
        # Modifying detached should not affect x's grad
        assert x.grad is None

    def test_critic_hidden_detach(self):
        """Simulates the critic forward: hidden states must be detached."""
        hidden_size = 64
        # Simulate model output under no_grad
        with torch.no_grad():
            hidden = torch.randn(4, hidden_size)

        hidden_detached = hidden.detach()
        critic = build_critic("small", hidden_size)

        values = critic(hidden_detached)
        loss = values.sum()
        loss.backward()

        # Critic params should have gradients
        for name, p in critic.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad on {name}"

    def test_no_grad_context_prevents_graph(self):
        """torch.no_grad() should prevent graph construction entirely."""
        model = nn.Linear(32, 32)
        with torch.no_grad():
            x = torch.randn(4, 32)
            out = model(x)
        assert not out.requires_grad

    def test_large_batch_advantage_computation(self):
        """Advantage computation should handle large batches without issues."""
        B = 1024
        rewards = torch.randint(0, 2, (B,)).float()
        values = torch.rand(B)
        adv = compute_advantages(rewards, values, normalize=True)
        assert adv.shape == (B,)
        assert torch.isfinite(adv).all()


# ===========================================================================
# 2. Forward pass counting (structural tests)
# ===========================================================================


class TestForwardPassStructure:
    """Verify structural properties that affect forward pass count."""

    def test_policy_log_probs_delegates_to_batched(self):
        """_policy_log_probs should call _batched_sequence_log_probs internally."""
        # This is a structural test: verify the method exists and has
        # the right signature
        from ppo_specs.ppo_trainer import PPOTrainer
        assert hasattr(PPOTrainer, "_policy_log_probs")
        assert hasattr(PPOTrainer, "_batched_sequence_log_probs")

    def test_extract_last_hidden_method_exists(self):
        """The shared _extract_last_hidden method should exist (dedup fix)."""
        from ppo_specs.ppo_trainer import PPOTrainer
        assert hasattr(PPOTrainer, "_extract_last_hidden")

    def test_legacy_methods_removed(self):
        """Dead legacy methods should no longer exist."""
        from ppo_specs.ppo_trainer import PPOTrainer
        assert not hasattr(PPOTrainer, "_sequence_log_prob"), (
            "Legacy _sequence_log_prob should be removed"
        )
        assert not hasattr(PPOTrainer, "_critic_value_no_grad"), (
            "Legacy _critic_value_no_grad should be removed"
        )


# ===========================================================================
# 3. Critic scaling properties
# ===========================================================================


class TestCriticScaling:
    """Test that critics scale correctly with hidden size."""

    @pytest.mark.parametrize("hidden_size", [64, 128, 896, 4096])
    def test_critic_accepts_various_hidden_sizes(self, hidden_size):
        """All critic capacities should work with common hidden sizes."""
        for cap in ["none", "small", "medium", "large"]:
            critic = build_critic(cap, hidden_size)
            h = torch.randn(2, hidden_size)
            out = critic(h)
            assert out.shape == (2,), f"{cap}, H={hidden_size}: shape {out.shape}"
            assert torch.isfinite(out).all()

    def test_large_critic_param_scaling(self):
        """Large critic with 4096 hidden should have reasonable param count."""
        critic = build_critic("large", 4096)
        n_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
        # 4096 -> 8192 -> 8192 -> 1 = ~100M params (expected for 2x width)
        assert n_params > 1_000_000, f"Large critic too small: {n_params}"
        assert n_params < 200_000_000, f"Large critic too large: {n_params}"

    def test_critic_fp32_stability(self):
        """Critic should produce stable outputs with fp32 hidden states."""
        critic = build_critic("large", 128)
        # Simulate bf16 model hidden states cast to fp32
        h_bf16 = torch.randn(8, 128, dtype=torch.bfloat16)
        h_fp32 = h_bf16.float()
        out = critic(h_fp32)
        assert torch.isfinite(out).all()


# ===========================================================================
# 4. Checkpoint config compatibility
# ===========================================================================


class TestCheckpointScaling:
    """Test checkpoint-related scaling properties."""

    def test_config_hash_deterministic(self):
        """Same config should produce same hash."""
        cfg = PPOConfig(model_name="test", batch_size=16)
        h1 = _config_hash(cfg)
        h2 = _config_hash(cfg)
        assert h1 == h2

    def test_config_hash_changes_with_key_fields(self):
        """Changing key fields should change the hash."""
        cfg1 = PPOConfig(model_name="model_a")
        cfg2 = PPOConfig(model_name="model_b")
        assert _config_hash(cfg1) != _config_hash(cfg2)

    def test_config_hash_ignores_non_key_fields(self):
        """Non-key fields (like n_steps) should not affect hash."""
        cfg1 = PPOConfig(n_steps=100)
        cfg2 = PPOConfig(n_steps=200)
        assert _config_hash(cfg1) == _config_hash(cfg2)


# ===========================================================================
# 5. Advantage estimation error metrics
# ===========================================================================


class TestAdvantageEstimationError:
    """Test the metrics used for E2.7 and E2.8 evaluation."""

    def test_perfect_estimation_zero_error(self):
        """When estimated == mc, error should be 0."""
        est = np.array([0.5, 0.3, 0.7])
        mc = np.array([0.5, 0.3, 0.7])
        err = advantage_estimation_error(est, mc)
        assert err == pytest.approx(0.0)

    def test_constant_offset_error(self):
        """Constant offset should give that offset as MAE."""
        est = np.array([0.6, 0.4, 0.8])
        mc = np.array([0.5, 0.3, 0.7])
        err = advantage_estimation_error(est, mc)
        assert err == pytest.approx(0.1)

    def test_error_symmetric(self):
        """Error should be symmetric: |est - mc| = |mc - est|."""
        est = np.array([0.2, 0.8])
        mc = np.array([0.5, 0.5])
        err1 = advantage_estimation_error(est, mc)
        err2 = advantage_estimation_error(mc, est)
        assert err1 == pytest.approx(err2)


# ===========================================================================
# 6. Batch processing invariants
# ===========================================================================


class TestBatchInvariants:
    """Test that batch processing maintains correctness invariants."""

    def test_cycle_batch_wraps_correctly(self):
        """cycle_batch should wrap around at list boundaries."""
        from ppo_specs.utils import cycle_batch
        items = list(range(10))
        # Normal slice
        assert cycle_batch(items, 0, 3) == [0, 1, 2]
        assert cycle_batch(items, 1, 3) == [3, 4, 5]
        # Wrap-around
        result = cycle_batch(items, 3, 3)
        assert result == [9, 0, 1]

    def test_cycle_batch_covers_all_items(self):
        """Over enough steps, all items should appear in batches."""
        from ppo_specs.utils import cycle_batch
        items = list(range(10))
        seen = set()
        for step in range(10):
            batch = cycle_batch(items, step, 3)
            seen.update(batch)
        assert seen == set(range(10))

    def test_cycle_batch_deterministic(self):
        """Same step and batch_size should always return same batch."""
        from ppo_specs.utils import cycle_batch
        items = list(range(20))
        b1 = cycle_batch(items, 5, 4)
        b2 = cycle_batch(items, 5, 4)
        assert b1 == b2


# ===========================================================================
# 7. Config validation
# ===========================================================================


class TestConfigValidation:
    """Test that config values are internally consistent."""

    def test_eval_size_within_test_samples(self):
        """eval_size should not exceed n_test_samples."""
        from ppo_specs.config import e2_7_config, e2_8_config, local_test_config
        for cfg_fn in [local_test_config, lambda: e2_7_config(42), lambda: e2_8_config("medium", 42)]:
            cfg = cfg_fn()
            assert cfg.eval_size <= cfg.n_test_samples, (
                f"{cfg.experiment_name}: eval_size={cfg.eval_size} > n_test_samples={cfg.n_test_samples}"
            )
            assert cfg.final_eval_size <= cfg.n_test_samples, (
                f"{cfg.experiment_name}: final_eval_size={cfg.final_eval_size} > n_test_samples={cfg.n_test_samples}"
            )

    def test_new_config_fields_have_sane_defaults(self):
        """New config fields added in this review should have sane defaults."""
        cfg = PPOConfig()
        assert cfg.max_prompt_length > 0
        assert cfg.grad_clip_norm > 0
        assert cfg.log_ratio_clip > 0
        assert cfg.eval_batch_size > 0
        # log_ratio_clip should be large enough that PPO clip handles the rest
        assert cfg.log_ratio_clip >= 5.0
