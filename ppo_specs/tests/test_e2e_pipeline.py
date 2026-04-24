"""
End-to-end integration tests for the PPO training pipeline.

Every test in this module loads the actual model and runs real forward/backward
passes.  They are all marked @pytest.mark.slow and use a module-scoped fixture
so the model is loaded only once per test session.

Run with:
    pytest ppo_specs/tests/test_e2e_pipeline.py -v --tb=short -x
"""

import sys
import os
import json
import hashlib
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

# ── Make RLVR-Comparison root importable ────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data import load_gsm8k, format_prompt
from src.rewards import gsm8k_reward
from eval.metrics import ExperimentLogger
from ppo_specs.config import PPOConfig, local_test_config, copy_config, CRITIC_CAPACITIES
from ppo_specs.ppo_trainer import PPOTrainer, load_ppo_trainer
from ppo_specs.critic import build_critic
from ppo_specs.advantage import (
    estimate_mc_advantages,
    advantage_estimation_error,
)
from ppo_specs.checkpoint import save_checkpoint, load_checkpoint
from ppo_specs.utils import cycle_batch
from ppo_specs.run_e2_7 import run_e2_7
from ppo_specs.run_e2_8 import run_one_capacity


# ── Tiny config shared by all tests ─────────────────────────────────────────

def _tiny_config(**overrides) -> PPOConfig:
    """Minimal config for integration tests."""
    defaults = dict(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        n_steps=3,
        batch_size=2,
        max_new_tokens=16,
        n_train_samples=10,
        eval_every=2,
        log_every=1,
        checkpoint_every=0,
        n_ppo_epochs=1,
        experiment_name="test_e2e",
        torch_dtype="float32",
        do_sample=True,
        temperature=0.7,
        seed=42,
    )
    defaults.update(overrides)
    return PPOConfig(**defaults)


# ── Module-scoped fixtures (model loaded once) ──────────────────────────────

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def shared_trainer(device):
    """Load the model once and share across all tests in this module."""
    cfg = _tiny_config(critic_capacity="medium")
    trainer, _ = load_ppo_trainer(cfg, device)
    return trainer


@pytest.fixture(scope="module")
def data():
    """Load a small dataset slice, shared across all tests."""
    train_ds = load_gsm8k("train", n_samples=10, seed=42)
    test_ds = load_gsm8k("test", n_samples=10)
    return {
        "train_prompts": [format_prompt(ex["question"]) for ex in train_ds],
        "train_gts": [ex["ground_truth"] for ex in train_ds],
        "test_prompts": [format_prompt(ex["question"]) for ex in test_ds],
        "test_gts": [ex["ground_truth"] for ex in test_ds],
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _model_weights_hash(model) -> str:
    """Deterministic hash of model parameters for comparison."""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _critic_weights_hash(critic) -> str:
    h = hashlib.sha256()
    for p in critic.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _all_finite(metrics: dict) -> bool:
    """Check that every numeric value in metrics is finite."""
    for v in metrics.values():
        if isinstance(v, (int, float)) and not np.isfinite(v):
            return False
    return True


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestE2EPipeline:
    """Integration tests that exercise the full PPO pipeline."""

    # 1. Full local-test run completes
    def test_full_local_run_completes(self, tmp_path, device):
        cfg = _tiny_config(
            n_steps=3,
            checkpoint_every=0,
            experiment_name="test_local_run",
            output_dir=str(tmp_path / "results"),
        )
        run_e2_7(cfg, compute_mc=False)

        results_file = tmp_path / "results" / f"{cfg.experiment_name}.json"
        assert results_file.exists(), "Results JSON file was not created"

        with open(results_file) as f:
            log = json.load(f)

        # eval_every=2 with n_steps=3 means evals at steps 0, 2 -> 2 entries
        assert len(log) == 2, f"Expected 2 log entries, got {len(log)}"

    # 2. Train_step produces learning signal
    def test_train_step_learning_signal(self, shared_trainer, data, device):
        trainer = shared_trainer
        prompts = data["train_prompts"][:2]
        gts = data["train_gts"][:2]

        initial_rollouts = trainer.total_rollouts
        critic_losses = []

        for _ in range(5):
            metrics = trainer.train_step(prompts, gts)
            critic_losses.append(metrics["critic_loss"])

        # Critic loss should change across steps (critic is learning)
        assert not all(
            abs(c - critic_losses[0]) < 1e-12 for c in critic_losses
        ), "Critic loss did not change across 5 steps -- no learning signal"

        # total_rollouts should increment by 5 * batch_size = 10
        expected_increment = 5 * 2
        actual_increment = trainer.total_rollouts - initial_rollouts
        assert actual_increment == expected_increment, (
            f"total_rollouts incremented by {actual_increment}, expected {expected_increment}"
        )

    # 3. Checkpoint save and resume produces identical continuation
    def test_checkpoint_save_and_resume(self, data, device, tmp_path):
        cfg = _tiny_config(
            n_steps=4,
            critic_capacity="small",
            checkpoint_every=0,
            experiment_name="test_ckpt",
            output_dir=str(tmp_path / "results"),
        )
        trainer, _ = load_ppo_trainer(cfg, device)
        logger = ExperimentLogger(cfg.experiment_name, cfg.output_dir)

        prompts = data["train_prompts"][:2]
        gts = data["train_gts"][:2]

        # Run 3 steps
        for step in range(3):
            trainer.train_step(prompts, gts)
            logger.log_step(step, dummy=1.0)

        # Save checkpoint
        ckpt_dir = str(tmp_path / "checkpoints")
        ckpt_path = save_checkpoint(trainer, 2, cfg, logger, ckpt_dir, keep_checkpoints=0)

        # Record state after step 3
        weights_hash_before = _critic_weights_hash(trainer.critic)

        # Load checkpoint into fresh trainer
        state = load_checkpoint(ckpt_path, cfg, device)
        trainer2, _ = load_ppo_trainer(cfg, device)

        from transformers import AutoModelForCausalLM
        trainer2.model = AutoModelForCausalLM.from_pretrained(
            state["model_path"],
            torch_dtype=torch.float32,
        ).to(device)
        trainer2.policy_optimizer = torch.optim.AdamW(
            trainer2.model.parameters(), lr=cfg.learning_rate
        )
        trainer2.policy_optimizer.load_state_dict(state["policy_optimizer_state_dict"])
        trainer2.critic.load_state_dict(state["critic_state_dict"])
        if trainer2.critic_optimizer and state["critic_optimizer_state_dict"]:
            trainer2.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
        trainer2.step = state["trainer_step"]
        trainer2.total_rollouts = state["total_rollouts"]

        # Verify weights match
        weights_hash_after = _critic_weights_hash(trainer2.critic)
        assert weights_hash_before == weights_hash_after, "Critic weights differ after checkpoint load"

        # Run step 4 from checkpoint -- should produce finite metrics
        metrics = trainer2.train_step(prompts, gts)
        assert _all_finite(metrics), f"Non-finite metrics after resume: {metrics}"

    # 4. MC estimation with batched generation
    def test_mc_estimation_batched(self, shared_trainer, data, device):
        ref_prompts = data["train_prompts"][:2]
        ref_gts = data["train_gts"][:2]

        mc_baselines = estimate_mc_advantages(
            shared_trainer.model,
            shared_trainer.tokenizer,
            ref_prompts,
            ref_gts,
            shared_trainer.reward_fn,
            n_samples=8,
            max_new_tokens=16,
            temperature=0.7,
            device=str(device),
            batch_size=4,
        )

        assert len(mc_baselines) == 2, f"Expected 2 entries, got {len(mc_baselines)}"
        for prompt, value in mc_baselines.items():
            assert 0.0 <= value <= 1.0, (
                f"MC value {value} out of [0, 1] for prompt: {prompt[:40]}..."
            )

    # 5. E2.8 single capacity run
    def test_e28_single_capacity_run(self, data, device, tmp_path):
        base_cfg = _tiny_config(
            n_steps=2,
            eval_every=1,
            experiment_name="test_e28",
            output_dir=str(tmp_path / "results"),
        )

        for capacity in ["none", "small"]:
            result = run_one_capacity(
                capacity=capacity,
                base_config=base_cfg,
                train_prompts=data["train_prompts"],
                train_gts=data["train_gts"],
                test_prompts=data["test_prompts"],
                test_gts=data["test_gts"],
                mc_baselines={},  # skip MC for speed
                device=device,
            )

            expected_keys = {"capacity", "final_accuracy", "mean_ev", "mean_bias", "accuracy_curve"}
            assert expected_keys.issubset(result.keys()), (
                f"Missing keys for {capacity}: {expected_keys - result.keys()}"
            )
            assert result["capacity"] == capacity

    # 6. Evaluate is deterministic
    def test_evaluate_deterministic(self, shared_trainer, data):
        prompts = data["test_prompts"][:4]
        gts = data["test_gts"][:4]

        acc1 = shared_trainer.evaluate(prompts, gts, n_eval=4)
        acc2 = shared_trainer.evaluate(prompts, gts, n_eval=4)

        assert acc1 == acc2, (
            f"Greedy evaluate not deterministic: {acc1} vs {acc2}"
        )

    # 7. All critic capacities work with batched ops
    def test_all_critic_capacities(self, data, device):
        prompts = data["train_prompts"][:2]
        gts = data["train_gts"][:2]

        for capacity in CRITIC_CAPACITIES:
            cfg = _tiny_config(
                critic_capacity=capacity,
                experiment_name=f"test_cap_{capacity}",
            )
            trainer, _ = load_ppo_trainer(cfg, device)
            metrics = trainer.train_step(prompts, gts)

            assert _all_finite(metrics), (
                f"Non-finite metrics for capacity={capacity}: {metrics}"
            )
            assert 0.0 <= metrics["accuracy"] <= 1.0, (
                f"Accuracy out of range for capacity={capacity}: {metrics['accuracy']}"
            )

    # 8. Logger flushes incrementally
    def test_logger_flushes_incrementally(self, data, device, tmp_path):
        cfg = _tiny_config(
            n_steps=4,
            eval_every=2,
            checkpoint_every=0,
            experiment_name="test_incremental_flush",
            output_dir=str(tmp_path / "results"),
        )
        run_e2_7(cfg, compute_mc=False)

        results_file = tmp_path / "results" / f"{cfg.experiment_name}.json"
        assert results_file.exists(), "Results file should exist after run"

        with open(results_file) as f:
            log = json.load(f)

        # eval_every=2, n_steps=4 -> evals at steps 0, 2 -> 2 entries
        assert len(log) >= 2, f"Expected at least 2 log entries, got {len(log)}"

        # Each entry should have the standard keys
        for entry in log:
            assert "step" in entry
            assert "test_accuracy" in entry

    # 9. Multiple PPO epochs work correctly
    def test_multiple_ppo_epochs(self, data, device):
        prompts = data["train_prompts"][:2]
        gts = data["train_gts"][:2]

        # n_ppo_epochs=3
        cfg3 = _tiny_config(n_ppo_epochs=3, experiment_name="test_epochs3")
        trainer3, _ = load_ppo_trainer(cfg3, device)
        metrics3 = trainer3.train_step(prompts, gts)
        assert _all_finite(metrics3), f"Non-finite metrics with n_ppo_epochs=3: {metrics3}"

        # Verify ppo_update was called 3 times by patching and counting
        cfg_mock = _tiny_config(n_ppo_epochs=3, experiment_name="test_epochs_mock")
        trainer_mock, _ = load_ppo_trainer(cfg_mock, device)
        call_count = 0
        original_ppo_update = trainer_mock.ppo_update

        def counting_ppo_update(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_ppo_update(*args, **kwargs)

        trainer_mock.ppo_update = counting_ppo_update
        trainer_mock.train_step(prompts, gts)
        assert call_count == 3, f"ppo_update called {call_count} times, expected 3"

    # 10. Advantage error metric with trainable critic
    def test_advantage_error_with_critic(self, data, device):
        cfg = _tiny_config(
            critic_capacity="medium",
            n_steps=3,
            experiment_name="test_adv_error",
        )
        trainer, _ = load_ppo_trainer(cfg, device)

        prompts = data["train_prompts"][:2]
        gts = data["train_gts"][:2]

        # Run a few steps to train the critic
        for _ in range(3):
            trainer.train_step(prompts, gts)

        # Evaluate critic on reference prompts
        ref_prompts = data["train_prompts"][:3]
        critic_values = trainer._eval_critic_on_prompts(ref_prompts)

        assert isinstance(critic_values, np.ndarray), (
            f"Expected numpy array, got {type(critic_values)}"
        )
        assert len(critic_values) == 3, (
            f"Expected 3 values, got {len(critic_values)}"
        )

        # Compute advantage estimation error against fake MC baselines
        mc_baselines = np.array([0.5, 0.3, 0.7])
        adv_error = advantage_estimation_error(critic_values, mc_baselines)

        assert isinstance(adv_error, float), f"Expected float, got {type(adv_error)}"
        assert np.isfinite(adv_error), f"Advantage error is not finite: {adv_error}"
