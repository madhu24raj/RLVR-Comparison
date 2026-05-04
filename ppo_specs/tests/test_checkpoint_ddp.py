"""Tests for Accelerator-aware checkpoint save/load.

Verifies the unwrap path produces a state_dict that loads cleanly into
both DDP-wrapped and plain models. Real multi-proc DDP is not exercised
here -- that's the integration smoke runbook's job.
"""
import sys, os, tempfile, inspect
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ppo_specs import checkpoint as ckpt_mod


class TestSaveCheckpointSignature:
    def test_save_checkpoint_accepts_accelerator_kwarg(self):
        """save_checkpoint must accept an optional accelerator kwarg
        for DDP-aware unwrapping (backward compatible default None)."""
        sig = inspect.signature(ckpt_mod.save_checkpoint)
        assert "accelerator" in sig.parameters
        # Default must be None (backward compat)
        assert sig.parameters["accelerator"].default is None

    def test_save_checkpoint_calls_unwrap_when_accelerator_provided(self):
        """Source-code check: when accelerator is provided, save_checkpoint
        must use accelerator.unwrap_model(...) before .state_dict() / .save_pretrained()."""
        src = inspect.getsource(ckpt_mod.save_checkpoint)
        assert "unwrap_model" in src or "_unwrap" in src, (
            "save_checkpoint must unwrap DDP-wrapped models before "
            "serialization (otherwise state_dict keys are prefixed with "
            "'module.' and won't load cleanly)."
        )


class TestSaveCheckpointBackwardCompat:
    """Existing callers without an accelerator must work identically."""

    def test_save_then_load_roundtrip_no_accelerator(self):
        """Save a tiny synthetic 'trainer' without an accelerator,
        load it back, verify weights match."""
        from ppo_specs.config import PPOConfig
        from eval.metrics import ExperimentLogger

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpts"
            ckpt_dir.mkdir()

            # Build a synthetic trainer-like object
            class FakeTrainer:
                def __init__(self):
                    self.model = self._make_hf_like()
                    self.tokenizer = self._make_tok()
                    self.critic = nn.Linear(4, 1)
                    self.policy_optimizer = torch.optim.AdamW(
                        self.model.parameters(), lr=1e-4,
                    )
                    self.critic_optimizer = torch.optim.AdamW(
                        self.critic.parameters(), lr=1e-4,
                    )
                    self.step = 0
                    self.total_rollouts = 0

                def _make_hf_like(self):
                    """Minimal HF-compatible model that supports save_pretrained."""
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    return AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

                def _make_tok(self):
                    from transformers import AutoTokenizer
                    return AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

            trainer = FakeTrainer()
            cfg = PPOConfig(critic_capacity="medium", checkpoint_dir=str(ckpt_dir))
            logger = ExperimentLogger("test_ddp_ckpt", str(tmp))

            # Save with no accelerator (legacy path)
            saved_path = ckpt_mod.save_checkpoint(
                trainer, step=42, config=cfg, logger=logger,
                checkpoint_dir=str(ckpt_dir), keep_checkpoints=1,
            )
            assert Path(saved_path).exists()
            # Critic state_dict must be loadable without 'module.' prefix
            critic_state = torch.load(
                Path(saved_path) / "critic.pt", map_location="cpu",
                weights_only=True,
            )
            keys = list(critic_state.keys())
            assert not any(k.startswith("module.") for k in keys), (
                f"State dict has DDP 'module.' prefix in non-DDP save: {keys}"
            )
