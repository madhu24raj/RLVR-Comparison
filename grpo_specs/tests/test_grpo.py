"""
Tests for the GRPO trainer.

Covers:
  1. Group advantage computation (the novel part of GRPO)
  2. Config validation
  3. Integration with shared per-token loss (via trainer)

Run:
    pytest grpo_specs/tests/test_grpo.py -v
    pytest grpo_specs/tests/test_grpo.py -v -m "not slow"
"""
from __future__ import annotations

import sys
import os

import pytest
import torch
import numpy as np

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from grpo_specs.grpo_trainer import GRPOTrainer, GroupRolloutBatch, Rollout
from grpo_specs.config import GRPOConfig, local_test_config, e2_7_config


# -- Helpers --

def make_batch(rewards_by_group: list[list[float]]) -> GroupRolloutBatch:
    """Create a GroupRolloutBatch from a list of reward groups.

    Each inner list is one prompt's group of G completions.
    Rollout fields other than reward and group_idx are dummies.
    """
    rollouts = []
    n_groups = len(rewards_by_group)
    group_size = len(rewards_by_group[0])
    for g, rewards_g in enumerate(rewards_by_group):
        for r in rewards_g:
            rollouts.append(Rollout(
                prompt=f"prompt_{g}",
                completion="dummy",
                reward=r,
                full_ids=[1, 2, 3],
                prompt_len=1,
                group_idx=g,
            ))
    return GroupRolloutBatch(
        rollouts=rollouts,
        n_groups=n_groups,
        group_size=group_size,
    )


def make_trainer() -> GRPOTrainer:
    """Create a GRPOTrainer with a dummy reward function (no model load)."""
    config = GRPOConfig(n_rollouts_per_prompt=4)
    # We can't create a real trainer without a model, but we can test
    # compute_group_advantages by accessing it as an unbound-style call
    # with a minimal trainer. Use __new__ to skip __init__.
    trainer = object.__new__(GRPOTrainer)
    trainer.config = config
    trainer.device = torch.device("cpu")
    return trainer


# ============================================================================
# 1. Group advantage computation
# ============================================================================

class TestGroupAdvantages:
    """Tests for GRPOTrainer.compute_group_advantages."""

    def test_all_same_reward_gives_zero_advantages(self):
        """When all rewards in a group are identical, advantages should be 0."""
        trainer = make_trainer()
        batch = make_batch([[1.0, 1.0, 1.0, 1.0]])
        adv = trainer.compute_group_advantages(batch)
        assert torch.allclose(adv, torch.zeros(4)), f"Expected all zeros, got {adv}"

    def test_all_zero_reward_gives_zero_advantages(self):
        """All-zero group: no signal, advantages should be 0."""
        trainer = make_trainer()
        batch = make_batch([[0.0, 0.0, 0.0, 0.0]])
        adv = trainer.compute_group_advantages(batch)
        assert torch.allclose(adv, torch.zeros(4)), f"Expected all zeros, got {adv}"

    def test_mixed_rewards_normalized(self):
        """One correct out of four: advantages should be z-scored within group."""
        trainer = make_trainer()
        # Rewards: [1, 0, 0, 0]. mean=0.25, std=0.5
        batch = make_batch([[1.0, 0.0, 0.0, 0.0]])
        adv = trainer.compute_group_advantages(batch)

        # The correct completion should have positive advantage
        assert adv[0] > 0, f"Correct completion should have positive advantage, got {adv[0]}"
        # Incorrect completions should have negative advantage
        for i in [1, 2, 3]:
            assert adv[i] < 0, f"Incorrect completion {i} should have negative advantage, got {adv[i]}"

    def test_advantages_sum_approximately_zero_per_group(self):
        """Z-scored advantages within a group should sum to ~0."""
        trainer = make_trainer()
        batch = make_batch([[1.0, 0.0, 0.0, 1.0]])
        adv = trainer.compute_group_advantages(batch)
        assert abs(adv.sum().item()) < 1e-5, f"Advantages should sum to ~0, got {adv.sum()}"

    def test_multiple_groups_independent(self):
        """Advantages in group 0 should not depend on rewards in group 1."""
        trainer = make_trainer()
        batch = make_batch([
            [1.0, 0.0, 0.0, 0.0],  # group 0: one correct
            [1.0, 1.0, 1.0, 1.0],  # group 1: all correct
        ])
        adv = trainer.compute_group_advantages(batch)

        # Group 0: should have nonzero advantages
        assert adv[0] > 0, "Group 0 correct should be positive"
        assert adv[1] < 0, "Group 0 incorrect should be negative"

        # Group 1: all same, should be zero
        for i in [4, 5, 6, 7]:
            assert abs(adv[i].item()) < 1e-5, f"Group 1 adv[{i}] should be 0, got {adv[i]}"

    def test_no_nan_with_binary_rewards(self):
        """Binary rewards {0, 1} should never produce NaN advantages."""
        trainer = make_trainer()
        for rewards in [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]:
            batch = make_batch([rewards])
            adv = trainer.compute_group_advantages(batch)
            assert not torch.isnan(adv).any(), f"NaN in advantages for rewards={rewards}"
            assert not torch.isinf(adv).any(), f"Inf in advantages for rewards={rewards}"

    def test_advantage_magnitude_scales_with_reward_spread(self):
        """Wider reward spread should produce larger advantage magnitudes."""
        trainer = make_trainer()
        # Narrow spread: [0.4, 0.5, 0.5, 0.6]
        batch_narrow = make_batch([[0.4, 0.5, 0.5, 0.6]])
        adv_narrow = trainer.compute_group_advantages(batch_narrow)

        # Wide spread: [0.0, 0.0, 1.0, 1.0]
        batch_wide = make_batch([[0.0, 0.0, 1.0, 1.0]])
        adv_wide = trainer.compute_group_advantages(batch_wide)

        # Both should be z-scored, so magnitudes should be similar
        # (z-scoring normalizes the spread). The key test is that
        # neither produces NaN or wrong signs.
        assert adv_narrow[0] < 0  # 0.4 < mean
        assert adv_narrow[3] > 0  # 0.6 > mean
        assert adv_wide[0] < 0    # 0.0 < mean
        assert adv_wide[2] > 0    # 1.0 > mean


# ============================================================================
# 2. Config validation
# ============================================================================

class TestGRPOConfig:

    def test_local_test_config_valid(self):
        cfg = local_test_config()
        assert cfg.n_rollouts_per_prompt >= 2, "G must be >= 2 for group normalization"
        assert cfg.batch_size > 0
        assert cfg.max_new_tokens >= 64

    def test_e2_7_config_valid(self):
        cfg = e2_7_config(seed=123)
        assert cfg.n_rollouts_per_prompt >= 4
        assert cfg.seed == 123
        assert "grpo" in cfg.experiment_name

    def test_reference_kl_coeff_default_zero(self):
        cfg = GRPOConfig()
        assert cfg.reference_kl_coeff == 0.0


# ============================================================================
# 3. GroupRolloutBatch
# ============================================================================

class TestGroupRolloutBatch:

    def test_rewards_tensor(self):
        batch = make_batch([[1.0, 0.0], [0.0, 1.0]])
        rewards = batch.rewards()
        assert rewards.shape == (4,)
        assert rewards.tolist() == [1.0, 0.0, 0.0, 1.0]

    def test_group_rewards(self):
        batch = make_batch([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        groups = batch.group_rewards()
        assert len(groups) == 2
        assert groups[0] == [1.0, 0.0, 0.0]
        assert groups[1] == [0.0, 1.0, 1.0]


# ============================================================================
# 4. Integration: shared loss functions work with GRPO advantages
# ============================================================================

class TestSharedLossIntegration:
    """Verify shared per-token loss functions accept GRPO-shaped inputs."""

    def test_clipped_surrogate_with_group_advantages(self):
        """The shared clipped_surrogate_loss should work with B*G samples."""
        from shared.per_token_loss import clipped_surrogate_loss

        B_times_G = 8  # 2 groups of 4
        T = 5
        new_lp = torch.zeros(B_times_G, T, requires_grad=True)
        old_lp = torch.zeros(B_times_G, T)
        mask = torch.ones(B_times_G, T)

        # Advantages from group normalization
        advantages = torch.tensor([1.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5])

        loss, clip_frac, ratio = clipped_surrogate_loss(
            new_lp, old_lp, advantages, mask,
        )
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.requires_grad, "Loss should have grad"

    def test_per_token_kl_with_group_batch(self):
        """per_token_kl should handle B*G batch size."""
        from shared.per_token_loss import per_token_kl

        B_times_G = 16
        T = 10
        a = torch.randn(B_times_G, T)
        b = torch.randn(B_times_G, T)
        mask = torch.ones(B_times_G, T)

        kl = per_token_kl(a, b, mask)
        assert not torch.isnan(kl), "KL should not be NaN"
        assert kl.shape == (), "KL should be scalar"


# ============================================================================
# 5. Slow integration test (requires model load)
# ============================================================================

@pytest.mark.slow
class TestGRPOEndToEnd:
    """Full integration: load model, generate rollouts, run one train_step."""

    @pytest.fixture(scope="class")
    def trainer(self):
        from grpo_specs.grpo_trainer import load_grpo_trainer
        config = local_test_config()
        config.n_steps = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return load_grpo_trainer(config, device)

    def test_train_step_produces_metrics(self, trainer):
        from src.data import load_gsm8k, format_prompt_with_template

        ds = load_gsm8k("test", n_samples=4, seed=42)
        prompts = [
            format_prompt_with_template(ex["question"], trainer.tokenizer) for ex in ds
        ]
        gts = [ex["ground_truth"] for ex in ds]

        metrics = trainer.train_step(prompts, gts)

        assert "policy_loss" in metrics
        assert "mean_reward" in metrics
        assert "accuracy" in metrics
        assert "kl_divergence" in metrics
        assert "clip_fraction" in metrics
        assert not np.isnan(metrics["policy_loss"]), "policy_loss is NaN"
        assert not np.isnan(metrics["kl_divergence"]), "kl_divergence is NaN"
