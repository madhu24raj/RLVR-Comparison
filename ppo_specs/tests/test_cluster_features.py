"""
Tests for cluster-critical features: dtype handling, gradient checkpointing,
and related configuration.

These tests verify that:
  - auto dtype selection picks bfloat16 on GPU, float32 on CPU
  - explicit dtype overrides work correctly
  - the critic always stays in float32 regardless of model dtype
  - log_softmax is computed in float32 (no -inf / NaN for bf16 models)
  - gradient checkpointing enables/disables correctly
  - end-to-end training steps work in both float32 and bfloat16
  - new config fields have correct defaults
  - preset configs include all fields

Run with:
    pytest ppo_specs/tests/test_cluster_features.py -v
    pytest ppo_specs/tests/test_cluster_features.py -v -m "not slow"  # skip model tests
"""

import sys
import os
import copy
import math

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

from ppo_specs.config import (
    PPOConfig,
    copy_config,
    local_test_config,
    e2_7_config,
    e2_8_config,
)
from ppo_specs.critic import build_critic
from ppo_specs.ppo_trainer import (
    PPOTrainer,
    Rollout,
    RolloutBatch,
    load_ppo_trainer,
)
from src.rewards import gsm8k_reward


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
HAS_CUDA = torch.cuda.is_available()


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
        experiment_name="test_cluster",
        critic_capacity="medium",
        do_sample=True,
        temperature=0.7,
    )
    defaults.update(overrides)
    return PPOConfig(**defaults)


def _make_dummy_rollouts(tokenizer, n=2, device=DEVICE):
    """Create synthetic rollouts without running the model."""
    prompts = [
        "Solve: 2 + 2 =",
        "Solve: 3 + 5 =",
    ]
    completions = [
        " Let me think. #### 4",
        " The answer is 8. #### 8",
    ]
    ground_truths = ["4", "8"]

    rollouts = []
    for i in range(n):
        idx = i % len(prompts)
        prompt = prompts[idx]
        comp = completions[idx]
        gt = ground_truths[idx]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        comp_ids = tokenizer.encode(comp, add_special_tokens=False)
        full_ids = prompt_ids + comp_ids

        reward = gsm8k_reward(comp, gt)
        rollouts.append(
            Rollout(
                prompt=prompt,
                completion=comp,
                reward=reward,
                old_log_prob=-5.0 - i * 0.5,
                value=0.0,
                full_ids=full_ids,
                prompt_len=len(prompt_ids),
            )
        )
    return RolloutBatch(rollouts)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (expensive model loads)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fp32_trainer():
    """Load a trainer with float32 on CPU. Reused across the module."""
    cfg = _tiny_config(torch_dtype="float32")
    device = DEVICE
    return load_ppo_trainer(cfg, device)


@pytest.fixture(scope="module")
def gpu_bf16_trainer():
    """Load a trainer with bfloat16 on GPU. Skipped if no GPU."""
    if not HAS_CUDA:
        pytest.skip("CUDA not available")
    cfg = _tiny_config(torch_dtype="bfloat16")
    device = torch.device("cuda")
    return load_ppo_trainer(cfg, device)


@pytest.fixture(scope="module")
def gpu_auto_trainer():
    """Load a trainer with torch_dtype='auto' on GPU. Skipped if no GPU."""
    if not HAS_CUDA:
        pytest.skip("CUDA not available")
    cfg = _tiny_config(torch_dtype="auto")
    device = torch.device("cuda")
    return load_ppo_trainer(cfg, device)


@pytest.fixture(scope="module")
def checkpointing_trainer():
    """Load a trainer with gradient checkpointing enabled on CPU."""
    cfg = _tiny_config(torch_dtype="float32", gradient_checkpointing=True)
    device = DEVICE
    return load_ppo_trainer(cfg, device)


# ===========================================================================
# 1. Auto dtype selection
# ===========================================================================

class TestAutoDtypeSelection:
    """Verify torch_dtype='auto' logic and explicit dtype overrides."""

    @pytest.mark.slow
    def test_auto_cpu_selects_float32(self, fp32_trainer):
        """torch_dtype='auto' with CPU device should select float32."""
        # fp32_trainer uses explicit float32; test the auto logic directly
        cfg = _tiny_config(torch_dtype="auto")
        device = DEVICE
        trainer = load_ppo_trainer(cfg, device)
        param = next(trainer.model.parameters())
        assert param.dtype == torch.float32, (
            f"auto on CPU should be float32, got {param.dtype}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_auto_cuda_selects_bfloat16(self, gpu_auto_trainer):
        """torch_dtype='auto' with CUDA device should select bfloat16."""
        param = next(gpu_auto_trainer.model.parameters())
        assert param.dtype == torch.bfloat16, (
            f"auto on CUDA should be bfloat16, got {param.dtype}"
        )

    @pytest.mark.slow
    def test_explicit_float32(self, fp32_trainer):
        """torch_dtype='float32' should always produce float32 parameters."""
        param = next(fp32_trainer.model.parameters())
        assert param.dtype == torch.float32

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_explicit_bfloat16(self, gpu_bf16_trainer):
        """torch_dtype='bfloat16' should always produce bfloat16 parameters."""
        param = next(gpu_bf16_trainer.model.parameters())
        assert param.dtype == torch.bfloat16


# ===========================================================================
# 2. Model loads in correct dtype
# ===========================================================================

class TestModelDtype:
    """Verify ALL model parameters are in the expected dtype."""

    @pytest.mark.slow
    def test_all_params_float32(self, fp32_trainer):
        """Every parameter should be float32 when loaded with float32."""
        for name, p in fp32_trainer.model.named_parameters():
            assert p.dtype == torch.float32, (
                f"Parameter {name} has dtype {p.dtype}, expected float32"
            )

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_all_params_bfloat16(self, gpu_bf16_trainer):
        """Every parameter should be bfloat16 when loaded with bfloat16."""
        for name, p in gpu_bf16_trainer.model.named_parameters():
            assert p.dtype == torch.bfloat16, (
                f"Parameter {name} has dtype {p.dtype}, expected bfloat16"
            )


# ===========================================================================
# 3. Critic stays float32 regardless of model dtype
# ===========================================================================

class TestCriticDtype:
    """The critic must always be float32 -- bf16 would lose value precision."""

    @pytest.mark.slow
    def test_critic_float32_with_fp32_model(self, fp32_trainer):
        """Critic should be float32 when model is float32."""
        for name, p in fp32_trainer.critic.named_parameters():
            if p.requires_grad:
                assert p.dtype == torch.float32, (
                    f"Critic param {name} has dtype {p.dtype}, expected float32"
                )

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_critic_float32_with_bf16_model(self, gpu_bf16_trainer):
        """Critic MUST remain float32 even when model is bfloat16."""
        for name, p in gpu_bf16_trainer.critic.named_parameters():
            if p.requires_grad:
                assert p.dtype == torch.float32, (
                    f"Critic param {name} has dtype {p.dtype}, expected float32. "
                    "Critic in bf16 would lose precision for value estimates."
                )


# ===========================================================================
# 4. log_softmax is computed in float32
# ===========================================================================

class TestLogSoftmaxPrecision:
    """Verify that _batched_sequence_log_probs produces finite results.

    If log_softmax were computed in bfloat16 without upcasting, rare tokens
    would produce -inf.  The test passes if all results are finite.
    """

    @pytest.mark.slow
    def test_log_probs_finite_fp32(self, fp32_trainer):
        """log probs should be finite in float32 mode."""
        batch = _make_dummy_rollouts(fp32_trainer.tokenizer, n=2, device=DEVICE)
        log_probs = fp32_trainer._batched_sequence_log_probs(
            [r.full_ids for r in batch.rollouts],
            [r.prompt_len for r in batch.rollouts],
        )
        assert torch.isfinite(log_probs).all(), (
            f"Non-finite log probs in fp32: {log_probs}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_log_probs_finite_bf16(self, gpu_bf16_trainer):
        """log probs should be finite in bfloat16 mode (fp32 upcast in code)."""
        device = torch.device("cuda")
        batch = _make_dummy_rollouts(gpu_bf16_trainer.tokenizer, n=2, device=device)
        log_probs = gpu_bf16_trainer._batched_sequence_log_probs(
            [r.full_ids for r in batch.rollouts],
            [r.prompt_len for r in batch.rollouts],
        )
        assert torch.isfinite(log_probs).all(), (
            f"Non-finite log probs in bf16 -- log_softmax may not be upcasted: {log_probs}"
        )


# ===========================================================================
# 5. Gradient checkpointing enables correctly
# ===========================================================================

class TestGradientCheckpointingEnabled:
    """Verify gradient checkpointing works when enabled."""

    @pytest.mark.slow
    def test_checkpointing_flag_set(self, checkpointing_trainer):
        """model.is_gradient_checkpointing should be True."""
        assert checkpointing_trainer.model.is_gradient_checkpointing, (
            "Gradient checkpointing should be enabled but is_gradient_checkpointing is False"
        )

    @pytest.mark.slow
    def test_train_step_with_checkpointing(self, checkpointing_trainer):
        """A train_step should complete without error under checkpointing."""
        prompts = ["Solve: 2 + 2 =", "Solve: 3 + 5 ="]
        ground_truths = ["4", "8"]
        metrics = checkpointing_trainer.train_step(prompts, ground_truths)
        assert "policy_loss" in metrics
        assert math.isfinite(metrics["policy_loss"])

    @pytest.mark.slow
    def test_gradients_exist_after_backward_with_checkpointing(self, checkpointing_trainer):
        """After a training step, gradients should exist on model parameters."""
        # Run a ppo_update to populate gradients
        batch = _make_dummy_rollouts(
            checkpointing_trainer.tokenizer, n=2, device=DEVICE
        )
        checkpointing_trainer.model.train()
        checkpointing_trainer.ppo_update(batch)

        has_grad = False
        for name, p in checkpointing_trainer.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                break

        assert has_grad, (
            "No gradients found after backward pass with gradient checkpointing enabled"
        )


# ===========================================================================
# 6. Gradient checkpointing disabled by default
# ===========================================================================

class TestGradientCheckpointingDisabled:
    """Verify checkpointing is off when not requested."""

    @pytest.mark.slow
    def test_checkpointing_off_by_default(self, fp32_trainer):
        """Default trainer should not have gradient checkpointing enabled."""
        flag = getattr(fp32_trainer.model, "is_gradient_checkpointing", False)
        assert not flag, (
            "Gradient checkpointing should be disabled by default"
        )


# ===========================================================================
# 7. Training step works with bfloat16 (end-to-end)
# ===========================================================================

class TestTrainStepBfloat16:
    """End-to-end training with bfloat16 on GPU."""

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_train_step_bf16_metrics_finite(self, gpu_bf16_trainer):
        """All metrics from a bf16 train_step should be finite."""
        prompts = ["Solve: 2 + 2 =", "Solve: 3 + 5 ="]
        ground_truths = ["4", "8"]
        metrics = gpu_bf16_trainer.train_step(prompts, ground_truths)

        for key, val in metrics.items():
            if isinstance(val, float):
                assert math.isfinite(val), (
                    f"Metric {key}={val} is not finite in bf16 training"
                )

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_train_step_bf16_params_updated(self, gpu_bf16_trainer):
        """Parameters should change after a gradient update in bf16."""
        # Snapshot a few parameters
        snapshots = {}
        for name, p in gpu_bf16_trainer.model.named_parameters():
            if p.requires_grad:
                snapshots[name] = p.data.clone()
                if len(snapshots) >= 3:
                    break

        prompts = ["Solve: 2 + 2 =", "Solve: 3 + 5 ="]
        ground_truths = ["4", "8"]
        gpu_bf16_trainer.train_step(prompts, ground_truths)

        changed = False
        for name, p in gpu_bf16_trainer.model.named_parameters():
            if name in snapshots and not torch.equal(p.data, snapshots[name]):
                changed = True
                break

        assert changed, "No parameters changed after bf16 train_step"


# ===========================================================================
# 8. Training step works with float32 (baseline)
# ===========================================================================

class TestTrainStepFloat32:
    """End-to-end training with float32 on CPU."""

    @pytest.mark.slow
    def test_train_step_fp32_metrics_finite(self, fp32_trainer):
        """All metrics from a fp32 train_step should be finite."""
        prompts = ["Solve: 2 + 2 =", "Solve: 3 + 5 ="]
        ground_truths = ["4", "8"]
        metrics = fp32_trainer.train_step(prompts, ground_truths)

        for key, val in metrics.items():
            if isinstance(val, float):
                assert math.isfinite(val), (
                    f"Metric {key}={val} is not finite in fp32 training"
                )

    @pytest.mark.slow
    def test_train_step_fp32_params_updated(self, fp32_trainer):
        """Parameters should change after a gradient update in fp32."""
        snapshots = {}
        for name, p in fp32_trainer.model.named_parameters():
            if p.requires_grad:
                snapshots[name] = p.data.clone()
                if len(snapshots) >= 3:
                    break

        prompts = ["Solve: 2 + 2 =", "Solve: 3 + 5 ="]
        ground_truths = ["4", "8"]
        fp32_trainer.train_step(prompts, ground_truths)

        changed = False
        for name, p in fp32_trainer.model.named_parameters():
            if name in snapshots and not torch.equal(p.data, snapshots[name]):
                changed = True
                break

        assert changed, "No parameters changed after fp32 train_step"


# ===========================================================================
# 9. Left-padding is set correctly
# ===========================================================================

class TestLeftPadding:
    """Tokenizer must use left-padding for batched generation."""

    @pytest.mark.slow
    def test_padding_side_left(self, fp32_trainer):
        """Tokenizer padding_side must be 'left'."""
        assert fp32_trainer.tokenizer.padding_side == "left", (
            f"Expected padding_side='left', got '{fp32_trainer.tokenizer.padding_side}'"
        )


# ===========================================================================
# 10. New config fields have correct defaults
# ===========================================================================

class TestConfigDefaults:
    """Verify that PPOConfig default values are correct for new fields."""

    def test_torch_dtype_default(self):
        cfg = PPOConfig()
        assert cfg.torch_dtype == "auto"

    def test_gradient_checkpointing_default(self):
        cfg = PPOConfig()
        assert cfg.gradient_checkpointing is False

    def test_checkpoint_every_default(self):
        cfg = PPOConfig()
        assert cfg.checkpoint_every == 20

    def test_keep_checkpoints_default(self):
        cfg = PPOConfig()
        assert cfg.keep_checkpoints == 3

    def test_resume_from_default(self):
        cfg = PPOConfig()
        assert cfg.resume_from == ""

    def test_critic_loss_coeff_default(self):
        cfg = PPOConfig()
        assert cfg.critic_loss_coeff == 0.5


# ===========================================================================
# 11. Config presets work with new fields
# ===========================================================================

class TestConfigPresets:
    """Verify that preset configs produce valid PPOConfig with all fields."""

    def _assert_has_all_fields(self, cfg: PPOConfig):
        """Check that all expected fields exist and have valid types."""
        assert isinstance(cfg.torch_dtype, str)
        assert cfg.torch_dtype in ("auto", "float32", "bfloat16")
        assert isinstance(cfg.gradient_checkpointing, bool)
        assert isinstance(cfg.checkpoint_every, int)
        assert isinstance(cfg.keep_checkpoints, int)
        assert isinstance(cfg.resume_from, str)
        assert isinstance(cfg.critic_loss_coeff, float)

    def test_local_test_config(self):
        cfg = local_test_config()
        self._assert_has_all_fields(cfg)
        assert cfg.n_steps == 5
        assert cfg.batch_size == 4

    def test_e2_7_config(self):
        cfg = e2_7_config()
        self._assert_has_all_fields(cfg)
        assert cfg.n_steps == 200

    def test_e2_8_config(self):
        cfg = e2_8_config()
        self._assert_has_all_fields(cfg)
        assert cfg.critic_capacity == "medium"

    def test_e2_8_config_with_capacity(self):
        for cap in ("none", "small", "medium", "large"):
            cfg = e2_8_config(critic_capacity=cap)
            assert cfg.critic_capacity == cap
            self._assert_has_all_fields(cfg)

    def test_copy_config_new_field_override(self):
        """copy_config should allow overriding new fields."""
        base = PPOConfig()
        modified = copy_config(
            base,
            torch_dtype="bfloat16",
            gradient_checkpointing=True,
            checkpoint_every=50,
            keep_checkpoints=5,
            resume_from="/tmp/ckpt",
            critic_loss_coeff=1.0,
        )
        assert modified.torch_dtype == "bfloat16"
        assert modified.gradient_checkpointing is True
        assert modified.checkpoint_every == 50
        assert modified.keep_checkpoints == 5
        assert modified.resume_from == "/tmp/ckpt"
        assert modified.critic_loss_coeff == 1.0
        # Original unchanged
        assert base.torch_dtype == "auto"
        assert base.gradient_checkpointing is False
