"""
Tests for the checkpoint save/load/resume system.

Covers:
  - Directory structure and file completeness after save
  - Full state round-trip (model weights, critic weights, optimizers, RNG)
  - Checkpoint rotation (keep last K)
  - Atomic save guarantees (no leftover .tmp_ dirs)
  - find_latest_checkpoint logic
  - RNG state preservation
  - Config hash stability and sensitivity
  - GracefulExitHandler signal handling
  - ExperimentLogger state preservation
  - Resume offset (start_step = saved step + 1)

Run with:
    pytest ppo_specs/tests/test_checkpoint.py -v
    pytest ppo_specs/tests/test_checkpoint.py -v -m "not slow"   # skip model-loading tests
"""

import json
import os
import random
import signal
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from ppo_specs.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    restore_rng_states,
    _config_hash,
    _rotate_checkpoints,
    GracefulExitHandler,
)
from ppo_specs.critic import build_critic, MediumCriticHead
from ppo_specs.ppo_trainer import PPOTrainer
from eval.metrics import ExperimentLogger
from src.rewards import gsm8k_reward


# ---------------------------------------------------------------------------
# Constants & helpers
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
        experiment_name="test_checkpoint",
        critic_capacity="medium",
        do_sample=True,
        temperature=0.7,
    )
    defaults.update(overrides)
    return PPOConfig(**defaults)


def _make_mock_trainer(config: PPOConfig = None):
    """Build a mock trainer with real critic & optimizer state dicts.

    Avoids loading the full LLM -- uses a tiny nn.Linear as a stand-in
    for the model so we can test checkpoint I/O quickly.
    """
    if config is None:
        config = _tiny_config()

    hidden_size = 32  # tiny stand-in

    # Minimal model stand-in
    model = nn.Linear(hidden_size, hidden_size)
    model.save_pretrained = MagicMock()  # HF-style save

    # Tokenizer stand-in
    tokenizer = MagicMock()
    tokenizer.save_pretrained = MagicMock()

    # Real critic
    critic = MediumCriticHead(hidden_size).to(DEVICE)

    # Real optimizers
    policy_optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr)

    trainer = MagicMock()
    trainer.model = model
    trainer.tokenizer = tokenizer
    trainer.critic = critic
    trainer.policy_optimizer = policy_optimizer
    trainer.critic_optimizer = critic_optimizer
    trainer.total_rollouts = 0
    trainer.step = 0

    return trainer


# ---------------------------------------------------------------------------
# Module-scoped fixture for real model (expensive -- only for @slow tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_model_and_tokenizer():
    """Load the model and tokenizer once for the entire module."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    ).to(DEVICE)

    return model, tokenizer


@pytest.fixture(scope="module")
def real_trainer(shared_model_and_tokenizer):
    """PPOTrainer with a real model (module-scoped to avoid reloading)."""
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
    return trainer


# ---------------------------------------------------------------------------
# 1. save_checkpoint creates correct directory structure
# ---------------------------------------------------------------------------

class TestSaveCheckpointStructure:
    def test_creates_all_expected_files(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        logger = ExperimentLogger("test", str(tmp_path / "logs"))

        ckpt_dir = str(tmp_path / "checkpoints")
        ckpt_path = save_checkpoint(
            trainer, step=0, config=config, logger=logger,
            checkpoint_dir=ckpt_dir, keep_checkpoints=3,
        )

        ckpt = Path(ckpt_path)
        assert ckpt.exists(), "Checkpoint directory should exist"

        expected_files = [
            "model",  # directory (model.save_pretrained called)
            "critic.pt",
            "policy_optimizer.pt",
            "training_state.json",
            "logger_state.json",
            "rng_states.pt",
            "config.json",
        ]
        # model/ is mocked so it won't be a real dir -- check the rest
        for name in expected_files:
            if name == "model":
                # save_pretrained is mocked; just verify it was called
                trainer.model.save_pretrained.assert_called_once()
                continue
            assert (ckpt / name).exists(), f"Missing expected file: {name}"

    def test_training_state_json_contents(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        trainer.total_rollouts = 20
        trainer.step = 5
        logger = ExperimentLogger("test", str(tmp_path / "logs"))

        ckpt_path = save_checkpoint(
            trainer, step=5, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        with open(Path(ckpt_path) / "training_state.json") as f:
            state = json.load(f)

        assert state["step"] == 5
        assert state["total_rollouts"] == 20
        assert "config_hash" in state
        assert len(state["config_hash"]) == 16  # sha256[:16]


# ---------------------------------------------------------------------------
# 2. load_checkpoint restores all state
# ---------------------------------------------------------------------------

class TestLoadCheckpoint:
    def test_load_returns_correct_step_and_rollouts(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        trainer.total_rollouts = 20
        trainer.step = 5
        logger = ExperimentLogger("test", str(tmp_path / "logs"))

        ckpt_path = save_checkpoint(
            trainer, step=5, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        loaded = load_checkpoint(ckpt_path, config, DEVICE)
        assert loaded["step"] == 5
        assert loaded["total_rollouts"] == 20

    def test_load_contains_critic_state_dict(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        logger = ExperimentLogger("test", str(tmp_path / "logs"))

        ckpt_path = save_checkpoint(
            trainer, step=0, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        loaded = load_checkpoint(ckpt_path, config, DEVICE)
        assert "critic_state_dict" in loaded
        assert isinstance(loaded["critic_state_dict"], dict)

    def test_load_contains_optimizer_states(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        logger = ExperimentLogger("test", str(tmp_path / "logs"))

        ckpt_path = save_checkpoint(
            trainer, step=0, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        loaded = load_checkpoint(ckpt_path, config, DEVICE)
        assert "policy_optimizer_state_dict" in loaded
        assert "critic_optimizer_state_dict" in loaded
        assert loaded["policy_optimizer_state_dict"] is not None
        assert loaded["critic_optimizer_state_dict"] is not None


# ---------------------------------------------------------------------------
# 3. Round-trip: save then load preserves model weights
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRoundTripModelWeights:
    def test_model_weights_preserved(self, tmp_path, real_trainer):
        config = real_trainer.config
        logger = ExperimentLogger("test", str(tmp_path / "logs"))

        # Grab original weights
        original_params = {
            name: p.clone()
            for name, p in real_trainer.model.named_parameters()
        }

        ckpt_path = save_checkpoint(
            real_trainer, step=0, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        loaded = load_checkpoint(ckpt_path, config, DEVICE)

        # Load model from the saved path
        from transformers import AutoModelForCausalLM

        restored_model = AutoModelForCausalLM.from_pretrained(
            loaded["model_path"], torch_dtype=torch.float32
        ).to(DEVICE)

        for name, orig_param in original_params.items():
            restored_param = dict(restored_model.named_parameters())[name]
            assert torch.equal(orig_param, restored_param), (
                f"Model parameter {name} differs after round-trip"
            )


# ---------------------------------------------------------------------------
# 4. Round-trip: save then load preserves critic weights
# ---------------------------------------------------------------------------

class TestRoundTripCriticWeights:
    def test_critic_weights_preserved(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)

        # Set critic weights to known values
        with torch.no_grad():
            for p in trainer.critic.parameters():
                p.fill_(3.14)

        original_state = {
            k: v.clone() for k, v in trainer.critic.state_dict().items()
        }

        logger = ExperimentLogger("test", str(tmp_path / "logs"))
        ckpt_path = save_checkpoint(
            trainer, step=0, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        loaded = load_checkpoint(ckpt_path, config, DEVICE)

        for key, orig_tensor in original_state.items():
            loaded_tensor = loaded["critic_state_dict"][key]
            assert torch.equal(orig_tensor, loaded_tensor), (
                f"Critic param {key} differs after round-trip"
            )


# ---------------------------------------------------------------------------
# 5. Checkpoint rotation
# ---------------------------------------------------------------------------

class TestCheckpointRotation:
    def test_keeps_only_last_k_checkpoints(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        logger = ExperimentLogger("test", str(tmp_path / "logs"))
        ckpt_dir = str(tmp_path / "ckpts")

        for step in range(5):
            save_checkpoint(
                trainer, step=step, config=config, logger=logger,
                checkpoint_dir=ckpt_dir, keep_checkpoints=3,
            )

        ckpt_root = Path(ckpt_dir)
        remaining = sorted(
            [d.name for d in ckpt_root.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint_step_")]
        )

        assert len(remaining) == 3
        assert "checkpoint_step_000002" in remaining
        assert "checkpoint_step_000003" in remaining
        assert "checkpoint_step_000004" in remaining

    def test_old_checkpoints_deleted(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        logger = ExperimentLogger("test", str(tmp_path / "logs"))
        ckpt_dir = str(tmp_path / "ckpts")

        for step in range(5):
            save_checkpoint(
                trainer, step=step, config=config, logger=logger,
                checkpoint_dir=ckpt_dir, keep_checkpoints=3,
            )

        ckpt_root = Path(ckpt_dir)
        assert not (ckpt_root / "checkpoint_step_000000").exists()
        assert not (ckpt_root / "checkpoint_step_000001").exists()


# ---------------------------------------------------------------------------
# 6. Atomic save (crash safety)
# ---------------------------------------------------------------------------

class TestAtomicSave:
    def test_no_tmp_directories_remain(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        logger = ExperimentLogger("test", str(tmp_path / "logs"))
        ckpt_dir = str(tmp_path / "ckpts")

        save_checkpoint(
            trainer, step=0, config=config, logger=logger,
            checkpoint_dir=ckpt_dir,
        )

        ckpt_root = Path(ckpt_dir)
        tmp_dirs = [d for d in ckpt_root.iterdir() if d.name.startswith(".tmp_")]
        assert len(tmp_dirs) == 0, f"Leftover .tmp_ dirs found: {tmp_dirs}"

    def test_final_checkpoint_is_complete(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        logger = ExperimentLogger("test", str(tmp_path / "logs"))
        ckpt_dir = str(tmp_path / "ckpts")

        ckpt_path = save_checkpoint(
            trainer, step=0, config=config, logger=logger,
            checkpoint_dir=ckpt_dir,
        )

        ckpt = Path(ckpt_path)
        assert ckpt.exists()
        assert (ckpt / "critic.pt").exists()
        assert (ckpt / "policy_optimizer.pt").exists()
        assert (ckpt / "training_state.json").exists()
        assert (ckpt / "logger_state.json").exists()
        assert (ckpt / "rng_states.pt").exists()
        assert (ckpt / "config.json").exists()


# ---------------------------------------------------------------------------
# 7. find_latest_checkpoint
# ---------------------------------------------------------------------------

class TestFindLatestCheckpoint:
    def test_returns_latest_step(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        logger = ExperimentLogger("test", str(tmp_path / "logs"))
        ckpt_dir = str(tmp_path / "ckpts")

        for step in [10, 20, 30]:
            save_checkpoint(
                trainer, step=step, config=config, logger=logger,
                checkpoint_dir=ckpt_dir, keep_checkpoints=0,
            )

        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None
        assert latest.endswith("checkpoint_step_000030")

    def test_empty_dir_returns_none(self, tmp_path):
        empty_dir = tmp_path / "empty_ckpts"
        empty_dir.mkdir()
        assert find_latest_checkpoint(str(empty_dir)) is None

    def test_nonexistent_dir_returns_none(self, tmp_path):
        assert find_latest_checkpoint(str(tmp_path / "does_not_exist")) is None


# ---------------------------------------------------------------------------
# 8. restore_rng_states
# ---------------------------------------------------------------------------

class TestRestoreRngStates:
    def test_rng_restoration_produces_same_sequence(self, tmp_path):
        # Set a known seed
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)

        # Save RNG states
        rng_states = {
            "torch_rng": torch.random.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }

        # Generate some numbers (advances the RNG)
        expected_torch = torch.randn(5)
        expected_numpy = np.random.randn(5)
        expected_python = [random.random() for _ in range(5)]

        # Restore to the saved state
        restore_rng_states(rng_states)

        # Generate again -- should match
        actual_torch = torch.randn(5)
        actual_numpy = np.random.randn(5)
        actual_python = [random.random() for _ in range(5)]

        assert torch.equal(expected_torch, actual_torch), "Torch RNG not restored"
        np.testing.assert_array_equal(expected_numpy, actual_numpy)
        assert expected_python == actual_python, "Python RNG not restored"

    def test_rng_round_trip_through_checkpoint(self, tmp_path):
        """Save RNG via checkpoint, corrupt state, restore, verify."""
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        logger = ExperimentLogger("test", str(tmp_path / "logs"))

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        ckpt_path = save_checkpoint(
            trainer, step=0, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        # Advance RNG
        torch.randn(100)
        np.random.randn(100)
        [random.random() for _ in range(100)]

        # Load and restore
        loaded = load_checkpoint(ckpt_path, config, DEVICE)
        restore_rng_states(loaded["rng_states"])

        # The sequences from this point should match what they would
        # have been right after the save
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Re-capture the state at save time, then generate
        save_rng = {
            "torch_rng": torch.random.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        expected_t = torch.randn(5)

        restore_rng_states(loaded["rng_states"])
        actual_t = torch.randn(5)

        assert torch.equal(expected_t, actual_t)


# ---------------------------------------------------------------------------
# 9. Config hash verification
# ---------------------------------------------------------------------------

class TestConfigHash:
    def test_same_config_same_hash(self):
        cfg1 = _tiny_config()
        cfg2 = _tiny_config()
        assert _config_hash(cfg1) == _config_hash(cfg2)

    def test_changed_model_name_different_hash(self):
        cfg1 = _tiny_config(model_name="Qwen/Qwen2.5-0.5B-Instruct")
        cfg2 = _tiny_config(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
        assert _config_hash(cfg1) != _config_hash(cfg2)

    def test_changed_seed_same_hash(self):
        """seed is not part of the hash -- changing it should not affect the hash."""
        cfg1 = _tiny_config(seed=42)
        cfg2 = _tiny_config(seed=999)
        assert _config_hash(cfg1) == _config_hash(cfg2)

    def test_changed_learning_rate_different_hash(self):
        cfg1 = _tiny_config(learning_rate=1e-5)
        cfg2 = _tiny_config(learning_rate=3e-4)
        assert _config_hash(cfg1) != _config_hash(cfg2)

    def test_changed_batch_size_different_hash(self):
        cfg1 = _tiny_config(batch_size=2)
        cfg2 = _tiny_config(batch_size=16)
        assert _config_hash(cfg1) != _config_hash(cfg2)


# ---------------------------------------------------------------------------
# 10. GracefulExitHandler
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGTERM signal handling is not reliable on Windows",
)
class TestGracefulExitHandler:
    def test_should_exit_starts_false(self):
        handler = GracefulExitHandler()
        try:
            assert handler.should_exit is False
        finally:
            handler.restore_signals()

    def test_handler_sets_should_exit_true(self):
        handler = GracefulExitHandler()
        try:
            # Simulate signal delivery by calling the handler directly
            handler._handler(signal.SIGTERM, None)
            assert handler.should_exit is True
        finally:
            handler.restore_signals()

    def test_restore_signals_works(self):
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        handler = GracefulExitHandler()
        # Signals should now point to handler._handler
        assert signal.getsignal(signal.SIGTERM) is not original_sigterm

        handler.restore_signals()
        # Signals should be restored
        assert signal.getsignal(signal.SIGTERM) is original_sigterm
        assert signal.getsignal(signal.SIGINT) is original_sigint

    def test_sigint_also_triggers_exit(self):
        handler = GracefulExitHandler()
        try:
            handler._handler(signal.SIGINT, None)
            assert handler.should_exit is True
        finally:
            handler.restore_signals()


class TestGracefulExitHandlerWindows:
    """Test what we can on Windows (direct _handler invocation)."""

    def test_should_exit_starts_false(self):
        handler = GracefulExitHandler()
        try:
            assert handler.should_exit is False
        finally:
            handler.restore_signals()

    def test_direct_handler_call_sets_exit(self):
        handler = GracefulExitHandler()
        try:
            # Use SIGINT which is available on Windows
            handler._handler(signal.SIGINT, None)
            assert handler.should_exit is True
        finally:
            handler.restore_signals()


# ---------------------------------------------------------------------------
# 11. ExperimentLogger state preservation
# ---------------------------------------------------------------------------

class TestLoggerStatePreservation:
    def test_logger_log_round_trips(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)

        logger = ExperimentLogger("test", str(tmp_path / "logs"))
        logger.log_step(0, accuracy=0.25, mean_reward=0.25)
        logger.log_step(1, accuracy=0.50, mean_reward=0.50)
        logger.log_step(2, accuracy=0.75, mean_reward=0.75)

        original_log = list(logger.log)  # shallow copy

        ckpt_path = save_checkpoint(
            trainer, step=2, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        loaded = load_checkpoint(ckpt_path, config, DEVICE)

        assert "logger_log" in loaded
        assert len(loaded["logger_log"]) == 3
        assert loaded["logger_log"] == original_log

    def test_logger_log_content_fidelity(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)

        logger = ExperimentLogger("test", str(tmp_path / "logs"))
        logger.log_step(5, policy_loss=-0.01, kl_divergence=0.003)

        ckpt_path = save_checkpoint(
            trainer, step=5, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        loaded = load_checkpoint(ckpt_path, config, DEVICE)
        entry = loaded["logger_log"][0]
        assert entry["step"] == 5
        assert abs(entry["policy_loss"] - (-0.01)) < 1e-9
        assert abs(entry["kl_divergence"] - 0.003) < 1e-9


# ---------------------------------------------------------------------------
# 12. Resume skips completed steps
# ---------------------------------------------------------------------------

class TestResumeOffset:
    def test_start_step_is_saved_step_plus_one(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        trainer.step = 2
        trainer.total_rollouts = 12
        logger = ExperimentLogger("test", str(tmp_path / "logs"))

        ckpt_path = save_checkpoint(
            trainer, step=2, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        loaded = load_checkpoint(ckpt_path, config, DEVICE)
        start_step = loaded["step"] + 1
        assert start_step == 3

    def test_find_and_resume_from_latest(self, tmp_path):
        config = _tiny_config()
        trainer = _make_mock_trainer(config)
        logger = ExperimentLogger("test", str(tmp_path / "logs"))
        ckpt_dir = str(tmp_path / "ckpts")

        # Simulate 3 training steps with checkpoints
        for step in range(3):
            trainer.step = step
            trainer.total_rollouts = (step + 1) * 4
            save_checkpoint(
                trainer, step=step, config=config, logger=logger,
                checkpoint_dir=ckpt_dir, keep_checkpoints=0,
            )

        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None

        loaded = load_checkpoint(latest, config, DEVICE)
        assert loaded["step"] == 2
        assert loaded["total_rollouts"] == 12
        assert loaded["step"] + 1 == 3  # resume from step 3


# ---------------------------------------------------------------------------
# Slow integration: save/load with real model
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRealModelCheckpoint:
    def test_save_and_load_with_real_trainer(self, tmp_path, real_trainer):
        config = real_trainer.config
        logger = real_trainer.logger
        logger.log_step(0, accuracy=0.1)

        ckpt_path = save_checkpoint(
            real_trainer, step=0, config=config, logger=logger,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )

        loaded = load_checkpoint(ckpt_path, config, DEVICE)
        assert loaded["step"] == 0
        assert "model_path" in loaded
        assert Path(loaded["model_path"]).exists()
        assert loaded["logger_log"] == logger.log
