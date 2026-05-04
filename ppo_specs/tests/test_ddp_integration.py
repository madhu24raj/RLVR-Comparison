"""Tests for PPOTrainer Accelerator integration (CPU, single-proc).

Real multi-proc DDP smoke is exercised by accelerate launch in a separate
runbook; these tests verify the Accelerator-aware code paths run correctly
when num_processes=1 (i.e. DDP semantics with a trivial world).
"""
import sys, os, inspect, pytest, torch, torch.nn as nn

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ppo_specs.config import PPOConfig
from ppo_specs.ppo_trainer import PPOTrainer, _shard_list, _build_adamw
from ppo_specs.critic import build_critic
from src.rewards import gsm8k_reward


class TestShardListHelper:
    def test_even_split(self):
        assert _shard_list([1, 2, 3, 4], 0, 2) == [1, 2]
        assert _shard_list([1, 2, 3, 4], 1, 2) == [3, 4]

    def test_single_rank(self):
        assert _shard_list([1, 2, 3], 0, 1) == [1, 2, 3]

    def test_full_coverage(self):
        items = list(range(8))
        ws = 4
        seen = []
        for r in range(ws):
            seen.extend(_shard_list(items, r, ws))
        assert seen == items


class TestPPOTrainerSignature:
    def test_init_requires_exactly_one_of_device_accelerator(self):
        cfg = PPOConfig(critic_capacity="none")
        critic = build_critic("none", 32)
        # Mock model and tokenizer (won't be called)
        m = nn.Linear(4, 4)
        t = type("T", (), {"pad_token_id": 0, "eos_token_id": 0, "padding_side": "left"})()

        # Both None -> error
        with pytest.raises(ValueError):
            PPOTrainer(config=cfg, model=m, tokenizer=t, critic=critic,
                       reward_fn=gsm8k_reward, device=None, accelerator=None)

    def test_init_accepts_device_legacy_path(self):
        """Legacy: device=cpu, accelerator=None -> no DDP code paths engaged."""
        cfg = PPOConfig(critic_capacity="none")
        critic = build_critic("none", 32)
        m = nn.Linear(32, 32)
        # Need a tokenizer-like object with required attrs:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        if t.pad_token is None:
            t.pad_token = t.eos_token
        device = torch.device("cpu")
        trainer = PPOTrainer(config=cfg, model=m, tokenizer=t, critic=critic,
                             reward_fn=gsm8k_reward, device=device)
        assert trainer.device == device
        assert trainer.accelerator is None
        assert trainer._is_ddp is False
        assert trainer._critic_trainable is False  # cached at init


class TestRefModelFrozenAssertion:
    def test_assertion_fires_on_unfrozen_reference(self):
        """Reference model must be fully frozen; assertion catches mistakes."""
        from transformers import AutoTokenizer
        cfg = PPOConfig(critic_capacity="none", reference_kl_coeff=0.0)
        critic = build_critic("none", 32)
        m = nn.Linear(32, 32)
        ref = nn.Linear(32, 32)  # NOT frozen -- params have requires_grad=True by default
        t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        if t.pad_token is None:
            t.pad_token = t.eos_token

        with pytest.raises(AssertionError, match="frozen"):
            PPOTrainer(
                config=cfg, model=m, tokenizer=t, critic=critic,
                reward_fn=gsm8k_reward, device=torch.device("cpu"),
                reference_model=ref,
            )


class TestAcceleratorPath:
    """Smoke test for Accelerator path with num_processes=1.

    We can't easily run real multi-proc here, but we can verify that an
    Accelerator with ws=1 produces a working trainer.
    """
    def test_accelerator_with_ws_1_works(self):
        try:
            from accelerate import Accelerator
        except ImportError:
            pytest.skip("accelerate not installed")
        from transformers import AutoTokenizer
        cfg = PPOConfig(critic_capacity="none", batch_size=2)
        critic = build_critic("none", 32)
        m = nn.Linear(32, 32)
        t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        if t.pad_token is None:
            t.pad_token = t.eos_token

        acc = Accelerator()  # num_processes=1 in this test process
        trainer = PPOTrainer(
            config=cfg, model=m, tokenizer=t, critic=critic,
            reward_fn=gsm8k_reward, accelerator=acc,
        )
        assert trainer.accelerator is acc
        assert trainer.device == acc.device
        # _is_ddp is False because num_processes=1
        assert trainer._is_ddp is False
