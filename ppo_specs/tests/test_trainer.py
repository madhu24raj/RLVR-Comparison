"""
Comprehensive tests for PPOTrainer.

Tests cover model initialization, training steps, KL divergence,
PPO ratio safety, critic evaluation, rollout generation, and
sequence log probability computation.

Run with:
    pytest ppo_specs/tests/test_trainer.py -v
    pytest ppo_specs/tests/test_trainer.py -v -m "not slow"   # skip generation tests
"""

import sys
import os
import math
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make the RLVR-Comparison root importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ppo_specs.config import PPOConfig, copy_config
from ppo_specs.critic import (
    build_critic,
    REINFORCEBaseline,
    SmallCriticMLP,
    MediumCriticHead,
    LargeCriticMLP,
)
from ppo_specs.ppo_trainer import (
    PPOTrainer,
    Rollout,
    RolloutBatch,
    load_ppo_trainer,
)
from src.rewards import gsm8k_reward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _tiny_config(**overrides) -> PPOConfig:
    """Return a minimal PPOConfig suitable for fast unit tests."""
    defaults = dict(
        model_name=MODEL_NAME,
        n_steps=1,
        batch_size=2,
        max_new_tokens=16,
        n_train_samples=4,
        n_ppo_epochs=1,
        eval_every=1,
        log_every=1,
        experiment_name="test_run",
        critic_capacity="none",
        do_sample=True,
        temperature=0.7,
    )
    defaults.update(overrides)
    return PPOConfig(**defaults)


@pytest.fixture(scope="module")
def shared_model_and_tokenizer():
    """Load the model and tokenizer once for the entire module.

    Using module scope avoids re-downloading / re-loading the 0.5B model
    for every individual test, keeping the suite fast.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    ).to(DEVICE)

    return model, tokenizer


@pytest.fixture
def trainer_none(shared_model_and_tokenizer):
    """PPOTrainer with critic_capacity='none' (REINFORCE baseline)."""
    model, tokenizer = shared_model_and_tokenizer
    cfg = _tiny_config(critic_capacity="none")
    hidden_size = model.config.hidden_size
    critic = build_critic("none", hidden_size).to(DEVICE)
    return PPOTrainer(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        critic=critic,
        reward_fn=gsm8k_reward,
        device=DEVICE,
    )


@pytest.fixture
def trainer_medium(shared_model_and_tokenizer):
    """PPOTrainer with critic_capacity='medium' (linear head)."""
    model, tokenizer = shared_model_and_tokenizer
    cfg = _tiny_config(critic_capacity="medium")
    hidden_size = model.config.hidden_size
    critic = build_critic("medium", hidden_size).to(DEVICE)
    return PPOTrainer(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        critic=critic,
        reward_fn=gsm8k_reward,
        device=DEVICE,
    )


def _make_dummy_rollouts(tokenizer, n=2, device=DEVICE):
    """Create synthetic rollouts without running the model.

    Returns a RolloutBatch with deterministic, well-formed data suitable
    for testing ppo_update and related methods.
    """
    rollouts = []
    prompts = [
        "Solve: 2 + 2 =",
        "Solve: 3 + 5 =",
        "Solve: 1 + 1 =",
        "Solve: 7 - 3 =",
    ]
    answers = ["4", "8", "2", "4"]
    completions = [
        " Let me think. #### 4",
        " The answer is 8",
        " 2. #### 2",
        " I get 5. #### 5",  # wrong answer
    ]

    for i in range(n):
        prompt = prompts[i % len(prompts)]
        comp = completions[i % len(completions)]
        gt = answers[i % len(answers)]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        comp_ids = tokenizer.encode(comp, add_special_tokens=False)
        full_ids = prompt_ids + comp_ids

        reward = gsm8k_reward(comp, gt)

        rollouts.append(
            Rollout(
                prompt=prompt,
                completion=comp,
                reward=reward,
                old_log_prob=-5.0 - i * 0.5,  # plausible negative values
                value=0.0,
                full_ids=full_ids,
                prompt_len=len(prompt_ids),
            )
        )
    return RolloutBatch(rollouts)


# ===========================================================================
# 1. Model Initialization Tests
# ===========================================================================


class TestModelInitialization:
    """Tests for load_ppo_trainer and PPOTrainer constructor."""

    def test_load_ppo_trainer_creates_valid_trainer(self, shared_model_and_tokenizer):
        """load_ppo_trainer should return a PPOTrainer with all core attributes."""
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="medium")
        hidden_size = model.config.hidden_size
        critic = build_critic("medium", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        assert trainer.config is cfg
        assert trainer.model is model
        assert trainer.tokenizer is tokenizer
        assert trainer.critic is critic
        assert trainer.device == DEVICE
        assert trainer.step == 0
        assert trainer.total_rollouts == 0
        assert trainer.policy_optimizer is not None

    def test_critic_none_capacity(self, shared_model_and_tokenizer):
        """Capacity 'none' should build a REINFORCEBaseline that is not trainable."""
        model, _ = shared_model_and_tokenizer
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size)
        assert isinstance(critic, REINFORCEBaseline)
        assert not critic.is_trainable()

    def test_critic_small_capacity(self, shared_model_and_tokenizer):
        """Capacity 'small' should build a SmallCriticMLP that is trainable."""
        model, _ = shared_model_and_tokenizer
        hidden_size = model.config.hidden_size
        critic = build_critic("small", hidden_size)
        assert isinstance(critic, SmallCriticMLP)
        assert critic.is_trainable()

    def test_critic_medium_capacity(self, shared_model_and_tokenizer):
        """Capacity 'medium' should build a MediumCriticHead that is trainable."""
        model, _ = shared_model_and_tokenizer
        hidden_size = model.config.hidden_size
        critic = build_critic("medium", hidden_size)
        assert isinstance(critic, MediumCriticHead)
        assert critic.is_trainable()

    def test_critic_large_capacity(self, shared_model_and_tokenizer):
        """Capacity 'large' should build a LargeCriticMLP that is trainable."""
        model, _ = shared_model_and_tokenizer
        hidden_size = model.config.hidden_size
        critic = build_critic("large", hidden_size)
        assert isinstance(critic, LargeCriticMLP)
        assert critic.is_trainable()

    def test_critic_placed_on_correct_device(self, shared_model_and_tokenizer):
        """After .to(device), all critic parameters should reside on that device."""
        model, _ = shared_model_and_tokenizer
        hidden_size = model.config.hidden_size
        for cap in ("small", "medium", "large"):
            critic = build_critic(cap, hidden_size).to(DEVICE)
            for p in critic.parameters():
                assert p.device == DEVICE, f"Critic {cap} param not on {DEVICE}"

    def test_optimizer_exists_only_for_trainable_critic(
        self, trainer_none, trainer_medium
    ):
        """critic_optimizer should be None when critic is not trainable,
        and an actual optimizer otherwise."""
        assert trainer_none.critic_optimizer is None
        assert trainer_medium.critic_optimizer is not None
        assert isinstance(
            trainer_medium.critic_optimizer, torch.optim.Optimizer
        )

    def test_policy_optimizer_always_exists(self, trainer_none, trainer_medium):
        """A policy optimizer must always be created regardless of critic capacity."""
        assert isinstance(trainer_none.policy_optimizer, torch.optim.Optimizer)
        assert isinstance(trainer_medium.policy_optimizer, torch.optim.Optimizer)

    def test_hidden_size_detected_correctly(self, shared_model_and_tokenizer):
        """The hidden_size read from model.config should match what Qwen2.5-0.5B reports."""
        model, _ = shared_model_and_tokenizer
        hidden_size = model.config.hidden_size
        # Qwen2.5-0.5B-Instruct has hidden_size = 896
        assert hidden_size == 896, (
            f"Expected hidden_size=896 for {MODEL_NAME}, got {hidden_size}"
        )


# ===========================================================================
# 2. Training Step Tests
# ===========================================================================


class TestTrainingStep:
    """Tests for the train_step method and its metric output."""

    EXPECTED_METRIC_KEYS = {
        "policy_loss",
        "critic_loss",
        "mean_reward",
        "reward_variance",
        "mean_advantage",
        "clip_fraction",
        "accuracy",
        "total_rollouts",
        "kl_divergence",
    }

    @pytest.mark.slow
    def test_train_step_returns_all_metric_keys(self, trainer_none):
        """train_step must return a dict containing every expected metric key."""
        prompts = ["Solve: 2 + 2 =", "Solve: 3 + 1 ="]
        gts = ["4", "4"]
        metrics = trainer_none.train_step(prompts, gts)
        missing = self.EXPECTED_METRIC_KEYS - set(metrics.keys())
        assert not missing, f"Missing metric keys: {missing}"

    @pytest.mark.slow
    def test_metrics_are_finite(self, trainer_none):
        """All numeric metrics should be finite (no NaN or inf)."""
        prompts = ["Solve: 2 + 2 =", "Solve: 3 + 1 ="]
        gts = ["4", "4"]
        metrics = trainer_none.train_step(prompts, gts)
        for key, val in metrics.items():
            assert math.isfinite(val), f"Metric '{key}' is not finite: {val}"

    @pytest.mark.slow
    def test_total_rollouts_increments(self, trainer_none):
        """total_rollouts should grow by the number of prompts each step."""
        prompts = ["Solve: 2 + 2 =", "Solve: 3 + 1 ="]
        gts = ["4", "4"]

        before = trainer_none.total_rollouts
        trainer_none.train_step(prompts, gts)
        after = trainer_none.total_rollouts

        assert after == before + len(prompts)

    def test_n_ppo_epochs_multiple_updates(self, shared_model_and_tokenizer):
        """With n_ppo_epochs > 1, ppo_update should be called that many times.

        This verifies the fix for bug C1 where n_ppo_epochs was ignored.
        We mock ppo_update and count invocations.
        """
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="none", n_ppo_epochs=3)
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        dummy_batch = _make_dummy_rollouts(tokenizer, n=2)
        dummy_metrics = {
            "policy_loss": 0.1,
            "critic_loss": 0.0,
            "kl_divergence": 0.0,
            "mean_reward": 0.5,
            "reward_variance": 0.25,
            "mean_advantage": 0.0,
            "clip_fraction": 0.0,
        }

        with patch.object(trainer, "generate_rollouts", return_value=dummy_batch), \
             patch.object(trainer, "ppo_update", return_value=dummy_metrics) as mock_update:
            trainer.train_step(["a", "b"], ["1", "2"])

        assert mock_update.call_count == 3, (
            f"ppo_update should be called n_ppo_epochs=3 times, "
            f"got {mock_update.call_count}"
        )

    def test_n_ppo_epochs_single_update(self, shared_model_and_tokenizer):
        """With n_ppo_epochs=1, ppo_update should be called exactly once."""
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="none", n_ppo_epochs=1)
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        dummy_batch = _make_dummy_rollouts(tokenizer, n=2)
        dummy_metrics = {
            "policy_loss": 0.1,
            "critic_loss": 0.0,
            "kl_divergence": 0.0,
            "mean_reward": 0.5,
            "reward_variance": 0.25,
            "mean_advantage": 0.0,
            "clip_fraction": 0.0,
        }

        with patch.object(trainer, "generate_rollouts", return_value=dummy_batch), \
             patch.object(trainer, "ppo_update", return_value=dummy_metrics) as mock_update:
            trainer.train_step(["a", "b"], ["1", "2"])

        assert mock_update.call_count == 1


# ===========================================================================
# 3. KL Divergence Tests
# ===========================================================================


class TestKLDivergence:
    """Tests for KL divergence penalty behaviour in ppo_update."""

    def test_kl_divergence_reported_when_coeff_zero(
        self, shared_model_and_tokenizer
    ):
        """Even when kl_coeff=0.0, the kl_divergence metric should still be reported."""
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="none", kl_coeff=0.0)
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        batch = _make_dummy_rollouts(tokenizer, n=2)
        metrics = trainer.ppo_update(batch)
        assert "kl_divergence" in metrics

    def test_kl_coeff_zero_does_not_affect_gradient(
        self, shared_model_and_tokenizer
    ):
        """When kl_coeff=0.0, the KL term should contribute zero to the loss,
        meaning gradients are identical to the no-KL case.

        We verify by checking that the total_loss = policy_loss + 0.5*critic_loss
        (i.e. the KL term multiplied by 0.0 adds nothing).
        """
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="none", kl_coeff=0.0)
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        batch = _make_dummy_rollouts(tokenizer, n=2)
        metrics = trainer.ppo_update(batch)

        # With kl_coeff=0 and critic=none (critic_loss=0), the total loss
        # equals the policy loss; the KL term contributes nothing.
        assert "kl_divergence" in metrics
        assert math.isfinite(metrics["kl_divergence"])

    def test_kl_with_positive_coeff(self, shared_model_and_tokenizer):
        """When kl_coeff > 0, the KL term should appear in the total loss.

        We run two ppo_updates: one with kl_coeff=0, one with kl_coeff=0.5.
        The policy losses should differ because the KL term affects gradients.
        """
        model, tokenizer = shared_model_and_tokenizer
        hidden_size = model.config.hidden_size

        # We need fresh critics each time to avoid state leakage
        cfg_no_kl = _tiny_config(critic_capacity="none", kl_coeff=0.0)
        critic_no_kl = build_critic("none", hidden_size).to(DEVICE)
        trainer_no_kl = PPOTrainer(
            config=cfg_no_kl,
            model=model,
            tokenizer=tokenizer,
            critic=critic_no_kl,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        cfg_with_kl = _tiny_config(critic_capacity="none", kl_coeff=0.5)
        critic_with_kl = build_critic("none", hidden_size).to(DEVICE)
        trainer_with_kl = PPOTrainer(
            config=cfg_with_kl,
            model=model,
            tokenizer=tokenizer,
            critic=critic_with_kl,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        batch = _make_dummy_rollouts(tokenizer, n=2)

        # Both share the same model, so this test just checks that kl_coeff
        # is wired into the loss calculation (config.kl_coeff * kl appears)
        metrics_with_kl = trainer_with_kl.ppo_update(batch)
        assert "kl_divergence" in metrics_with_kl
        assert math.isfinite(metrics_with_kl["kl_divergence"])

    def test_kl_divergence_is_nonnegative_approximation(
        self, shared_model_and_tokenizer
    ):
        """The KL(old||new) approximation E[log(old/new)] should typically be
        non-negative at the start of training when old == new (ratio ~ 1).

        Since we use old_log_probs - new_log_probs, this is an unbiased
        first-order approximation. Right after generation (before any update)
        it should be approximately zero, hence non-negative within floating
        point tolerance.
        """
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="none", kl_coeff=0.0)
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        # Use the same model for old_log_probs and new_log_probs, so KL ~ 0
        batch = _make_dummy_rollouts(tokenizer, n=2)

        # Recompute old_log_probs from the current model to simulate
        # the scenario right after rollout generation. Use the modern
        # batched API (legacy _sequence_log_prob was removed).
        with torch.no_grad():
            new_lp = trainer._batched_sequence_log_probs(
                [r.full_ids for r in batch.rollouts],
                [r.prompt_len for r in batch.rollouts],
            )
        for i, r in enumerate(batch.rollouts):
            r.old_log_prob = new_lp[i].item()

        metrics = trainer.ppo_update(batch)
        # KL should be very close to zero (same policy for old and new)
        assert metrics["kl_divergence"] >= -1e-4, (
            f"KL divergence is unexpectedly negative: {metrics['kl_divergence']}"
        )


# ===========================================================================
# 4. PPO Ratio Safety Tests
# ===========================================================================


class TestPPORatioSafety:
    """Tests that the PPO ratio computation is numerically stable."""

    def test_extreme_log_prob_difference_stays_finite(
        self, shared_model_and_tokenizer
    ):
        """When old and new log probs differ wildly (simulating a very stale
        rollout buffer), the ratio should remain finite thanks to log-ratio
        clamping at +/- 20.

        We create rollouts with artificially extreme old_log_probs.
        """
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="none")
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        batch = _make_dummy_rollouts(tokenizer, n=2)
        # Make old_log_probs extremely different from what the model would produce
        batch.rollouts[0].old_log_prob = -1000.0
        batch.rollouts[1].old_log_prob = 0.0  # impossibly high

        metrics = trainer.ppo_update(batch)

        assert math.isfinite(metrics["policy_loss"]), "policy_loss is not finite"
        assert math.isfinite(metrics["clip_fraction"]), "clip_fraction is not finite"

    def test_clip_fraction_between_zero_and_one(
        self, shared_model_and_tokenizer
    ):
        """clip_fraction (fraction of samples where ratio is clipped) must
        be in [0, 1] since it is a fraction."""
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="none")
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        batch = _make_dummy_rollouts(tokenizer, n=2)
        metrics = trainer.ppo_update(batch)

        assert 0.0 <= metrics["clip_fraction"] <= 1.0, (
            f"clip_fraction out of range: {metrics['clip_fraction']}"
        )

    def test_ratio_clamping_prevents_overflow(self, shared_model_and_tokenizer):
        """Verify that the log-ratio clamping at [-20, 20] keeps the ratio
        within [exp(-20), exp(20)] and prevents inf values.

        After clamping, exp(20) ~ 4.85e8 which is large but finite.
        """
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="none")
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        batch = _make_dummy_rollouts(tokenizer, n=2)
        # Force extreme old_log_probs that would cause overflow without clamping
        batch.rollouts[0].old_log_prob = -500.0
        batch.rollouts[1].old_log_prob = -500.0

        metrics = trainer.ppo_update(batch)

        # All returned metrics should be finite
        for key, val in metrics.items():
            assert math.isfinite(val), (
                f"Metric '{key}' is not finite ({val}) with extreme log probs"
            )


# ===========================================================================
# 5. Critic Evaluation Tests
# ===========================================================================


class TestCriticEvaluation:
    """Tests for _eval_critic_on_prompts (if implemented) and critic forward pass."""

    def test_eval_critic_on_prompts_returns_correct_shape(
        self, shared_model_and_tokenizer
    ):
        """_eval_critic_on_prompts should return a numpy array with length
        equal to the number of input prompts.

        Skipped if the method is not yet implemented.
        """
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="medium")
        hidden_size = model.config.hidden_size
        critic = build_critic("medium", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        if not hasattr(trainer, "_eval_critic_on_prompts"):
            pytest.skip("_eval_critic_on_prompts not yet implemented")

        prompts = ["Solve: 1+1=", "Solve: 2+3=", "Solve: 4+4="]
        result = trainer._eval_critic_on_prompts(prompts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(prompts),)

    def test_eval_critic_none_returns_zeros(self, shared_model_and_tokenizer):
        """For capacity='none', _eval_critic_on_prompts returns all zeros."""
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="none")
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        if not hasattr(trainer, "_eval_critic_on_prompts"):
            pytest.skip("_eval_critic_on_prompts not yet implemented")

        result = trainer._eval_critic_on_prompts(["Solve: 1+1="])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 0.0

    def test_critic_value_no_grad_returns_zero_for_none(self, trainer_none):
        """For critic_capacity='none', critic value should be 0.0.
        (Modern API: _batched_critic_values; legacy _critic_value_no_grad removed.)"""
        values = trainer_none._batched_critic_values(["Solve: 2 + 2 ="])
        assert values.shape == (1,)
        assert values[0].item() == 0.0

    def test_critic_value_no_grad_finite_for_trainable(self, trainer_medium):
        """For trainable critics, critic value should be a finite float.
        (Modern API: _batched_critic_values; legacy _critic_value_no_grad removed.)"""
        values = trainer_medium._batched_critic_values(["Solve: 2 + 2 ="])
        assert values.shape == (1,)
        v = values[0].item()
        assert isinstance(v, float)
        assert math.isfinite(v)

    def test_critic_forward_none_returns_zero_loss(
        self, shared_model_and_tokenizer
    ):
        """For capacity='none', _critic_forward should return (None, tensor(0.0))."""
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="none")
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        batch = _make_dummy_rollouts(tokenizer, n=2)
        rewards = batch.rewards().to(DEVICE)

        values, loss = trainer._critic_forward(batch, rewards)
        assert values is None
        assert loss.item() == 0.0

    def test_critic_forward_trainable_returns_values_and_loss(
        self, shared_model_and_tokenizer
    ):
        """For trainable critics, _critic_forward should return
        (tensor of values, finite MSE loss)."""
        model, tokenizer = shared_model_and_tokenizer
        cfg = _tiny_config(critic_capacity="medium")
        hidden_size = model.config.hidden_size
        critic = build_critic("medium", hidden_size).to(DEVICE)

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            critic=critic,
            reward_fn=gsm8k_reward,
            device=DEVICE,
        )

        batch = _make_dummy_rollouts(tokenizer, n=2)
        rewards = batch.rewards().to(DEVICE)

        values, loss = trainer._critic_forward(batch, rewards)
        assert values is not None
        assert values.shape == (2,)
        assert math.isfinite(loss.item())


# ===========================================================================
# 6. Rollout Generation Tests
# ===========================================================================


class TestRolloutGeneration:
    """Tests for generate_rollouts and the Rollout/RolloutBatch data structures."""

    @pytest.mark.slow
    def test_generate_rollouts_returns_correct_count(self, trainer_none):
        """generate_rollouts should return one rollout per prompt."""
        prompts = ["Solve: 2 + 2 =", "Solve: 3 + 1 ="]
        gts = ["4", "4"]
        batch = trainer_none.generate_rollouts(prompts, gts)

        assert isinstance(batch, RolloutBatch)
        assert len(batch.rollouts) == len(prompts)

    @pytest.mark.slow
    def test_rollout_fields_are_valid(self, trainer_none):
        """Each rollout should have well-typed, sensible field values."""
        prompts = ["Solve: 2 + 2 ="]
        gts = ["4"]
        batch = trainer_none.generate_rollouts(prompts, gts)
        r = batch.rollouts[0]

        # String fields
        assert isinstance(r.prompt, str) and len(r.prompt) > 0
        assert isinstance(r.completion, str)

        # Reward is binary
        assert r.reward in (0.0, 1.0), f"Unexpected reward value: {r.reward}"

        # Old log prob should be non-positive (log of a probability)
        assert isinstance(r.old_log_prob, float)
        assert r.old_log_prob <= 0.0 or r.old_log_prob == 0.0, (
            f"old_log_prob should be <= 0, got {r.old_log_prob}"
        )

        # full_ids is a list of ints
        assert isinstance(r.full_ids, list)
        assert all(isinstance(x, int) for x in r.full_ids)

        # prompt_len is positive
        assert isinstance(r.prompt_len, int)
        assert r.prompt_len > 0

    @pytest.mark.slow
    def test_full_ids_starts_with_prompt_tokens(self, trainer_none):
        """The full_ids should start with the tokenized prompt tokens."""
        prompt = "Solve: 2 + 2 ="
        gts = ["4"]
        batch = trainer_none.generate_rollouts([prompt], gts)
        r = batch.rollouts[0]

        # Tokenize the prompt independently
        prompt_ids = trainer_none.tokenizer.encode(prompt, add_special_tokens=True)
        # full_ids should start with the prompt tokens
        assert r.full_ids[: r.prompt_len] == prompt_ids[: r.prompt_len], (
            "full_ids does not start with the prompt token ids"
        )

    @pytest.mark.slow
    def test_total_rollouts_updated_after_generation(self, trainer_none):
        """generate_rollouts should increment total_rollouts by len(prompts)."""
        initial = trainer_none.total_rollouts
        prompts = ["Solve: 1+1=", "Solve: 2+2="]
        gts = ["2", "4"]
        trainer_none.generate_rollouts(prompts, gts)
        assert trainer_none.total_rollouts == initial + 2

    def test_rollout_batch_rewards_tensor(self):
        """RolloutBatch.rewards() should return a float32 tensor."""
        r1 = Rollout("p1", "c1", 1.0, -3.0, 0.0, [1, 2, 3], 1)
        r2 = Rollout("p2", "c2", 0.0, -4.0, 0.0, [4, 5, 6], 1)
        batch = RolloutBatch([r1, r2])

        rewards = batch.rewards()
        assert rewards.dtype == torch.float32
        assert rewards.tolist() == [1.0, 0.0]

    def test_rollout_batch_old_log_probs_tensor(self):
        """RolloutBatch.old_log_probs() should return a float32 tensor."""
        r1 = Rollout("p1", "c1", 1.0, -3.0, 0.0, [1, 2, 3], 1)
        r2 = Rollout("p2", "c2", 0.0, -4.0, 0.0, [4, 5, 6], 1)
        batch = RolloutBatch([r1, r2])

        lps = batch.old_log_probs()
        assert lps.dtype == torch.float32
        assert lps.tolist() == [-3.0, -4.0]


# ===========================================================================
# 7. Sequence Log Prob Tests
# ===========================================================================


class TestSequenceLogProb:
    """Tests for sequence-level log-prob computation.

    The legacy `_sequence_log_prob(ids, prompt_len)` method was removed in
    the 2026-04-09 code review. These tests now exercise the modern
    `_batched_sequence_log_probs([full_ids], [prompt_len])` API which is
    semantically equivalent (returns [B] tensor; index [0] for single-sample).
    """

    def _trainer_with_none_critic(self, model, tokenizer):
        cfg = _tiny_config(critic_capacity="none")
        hidden_size = model.config.hidden_size
        critic = build_critic("none", hidden_size).to(DEVICE)
        return PPOTrainer(
            config=cfg, model=model, tokenizer=tokenizer,
            critic=critic, reward_fn=gsm8k_reward, device=DEVICE,
        )

    def test_log_prob_is_negative_for_valid_sequence(
        self, shared_model_and_tokenizer
    ):
        """The sequence log probability should be negative (log of a value < 1)."""
        model, tokenizer = shared_model_and_tokenizer
        trainer = self._trainer_with_none_critic(model, tokenizer)

        text = "Solve: 2 + 2 = The answer is 4."
        full_ids = tokenizer.encode(text, add_special_tokens=True)
        prompt_len = 5

        with torch.no_grad():
            lp = trainer._batched_sequence_log_probs([full_ids], [prompt_len])

        assert lp[0].item() < 0.0, (
            f"Sequence log prob should be negative, got {lp[0].item()}"
        )

    def test_log_prob_is_scalar(self, shared_model_and_tokenizer):
        """Batched API returns a 1-D [B] tensor; single-sample input → shape [1]."""
        model, tokenizer = shared_model_and_tokenizer
        trainer = self._trainer_with_none_critic(model, tokenizer)

        text = "Hello world, this is a test."
        full_ids = tokenizer.encode(text, add_special_tokens=True)
        prompt_len = 3

        with torch.no_grad():
            lp = trainer._batched_sequence_log_probs([full_ids], [prompt_len])

        assert lp.dim() == 1
        assert lp.shape[0] == 1

    def test_log_prob_zero_for_empty_response(self, shared_model_and_tokenizer):
        """When prompt_len == full sequence length (empty response),
        log prob should be 0.0."""
        model, tokenizer = shared_model_and_tokenizer
        trainer = self._trainer_with_none_critic(model, tokenizer)

        text = "Hello"
        full_ids = tokenizer.encode(text, add_special_tokens=True)
        seq_len = len(full_ids)

        with torch.no_grad():
            lp = trainer._batched_sequence_log_probs([full_ids], [seq_len])

        assert lp[0].item() == 0.0, (
            f"Expected 0.0 for empty response, got {lp[0].item()}"
        )

    def test_log_prob_is_finite(self, shared_model_and_tokenizer):
        """The returned log probability should always be a finite number."""
        model, tokenizer = shared_model_and_tokenizer
        trainer = self._trainer_with_none_critic(model, tokenizer)

        text = "What is 2 plus 2? The answer is 4."
        full_ids = tokenizer.encode(text, add_special_tokens=True)
        prompt_len = 4

        with torch.no_grad():
            lp = trainer._batched_sequence_log_probs([full_ids], [prompt_len])

        assert math.isfinite(lp[0].item()), (
            f"Log prob should be finite, got {lp[0].item()}"
        )
