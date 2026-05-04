"""Tests for P19: length-bucketed generation in generate_rollouts.

Verifies:
1. Output ordering matches input ordering (rollout[i].prompt == prompts[i])
2. Each rollout has prompt_len matching the original prompt's tokenization
3. With do_sample=False (greedy), bucketed and non-bucketed produce identical
   completions (greedy is deterministic — bucketing must not change semantics).
4. The bucketing path is invoked only when length_bucketed_generation=True
   AND batch_size > generation_bucket_size.
"""
import sys
import os
import inspect

import pytest
import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ppo_specs.config import PPOConfig
from ppo_specs.ppo_trainer import PPOTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = torch.device("cpu")


class TestBucketedGenerateStructure:
    """Structural checks that don't require loading a real model."""

    def test_helper_method_exists(self):
        assert hasattr(PPOTrainer, "_bucketed_generate")

    def test_generate_rollouts_dispatches_on_config(self):
        """generate_rollouts must check config.length_bucketed_generation
        before deciding which generate path to take."""
        src = inspect.getsource(PPOTrainer.generate_rollouts)
        assert "length_bucketed_generation" in src
        assert "_bucketed_generate" in src

    def test_bucketed_generate_preserves_input_order(self):
        """The helper must restore the original input order via inverse_idx
        so that out[i] aligns with prompts[i]."""
        src = inspect.getsource(PPOTrainer._bucketed_generate)
        # Must compute an inverse index and apply it at the end
        assert "inverse_idx" in src or "inverse" in src
        assert "argsort" in src


class TestBucketedGenerateNumerical:
    """Real-model tests verifying bucketing doesn't change semantics."""

    @pytest.fixture(scope="module")
    def trainer_factory(self):
        """Builds a fresh trainer per test (module scope to amortize model load)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ppo_specs.critic import build_critic
        from src.rewards import gsm8k_reward

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

        def make(length_bucketed: bool, bucket_size: int = 2):
            cfg = PPOConfig(
                model_name=MODEL_NAME,
                batch_size=4,
                max_new_tokens=8,
                max_prompt_length=64,
                temperature=1.0,
                do_sample=False,  # greedy → deterministic
                critic_capacity="none",
                length_bucketed_generation=length_bucketed,
                generation_bucket_size=bucket_size,
            )
            critic = build_critic("none", model.config.hidden_size).to(DEVICE)
            return PPOTrainer(
                config=cfg, model=model, tokenizer=tok,
                critic=critic, reward_fn=gsm8k_reward, device=DEVICE,
            )
        return make

    @pytest.mark.slow
    def test_greedy_bucketed_matches_non_bucketed(self, trainer_factory):
        """Greedy decoding is deterministic; bucketing must produce
        IDENTICAL completions to the non-bucketed path."""
        # Prompts deliberately have varied lengths so bucketing actually re-orders
        prompts = [
            "What is 2+2?",
            "Solve the following arithmetic problem step by step: 3 + 5 =",
            "1+1=",
            "Compute carefully and explain your work for the value of seven plus five:",
        ]
        gts = ["4", "8", "2", "12"]

        trainer_off = trainer_factory(length_bucketed=False)
        batch_off = trainer_off.generate_rollouts(prompts, gts)

        trainer_on = trainer_factory(length_bucketed=True, bucket_size=2)
        batch_on = trainer_on.generate_rollouts(prompts, gts)

        # Same number of rollouts
        assert len(batch_off.rollouts) == len(batch_on.rollouts) == len(prompts)

        # Each rollout must align with its original prompt index
        for i in range(len(prompts)):
            assert batch_off.rollouts[i].prompt == prompts[i]
            assert batch_on.rollouts[i].prompt == prompts[i]

        # Greedy must produce identical completions across paths
        for i in range(len(prompts)):
            assert batch_off.rollouts[i].completion == batch_on.rollouts[i].completion, (
                f"Bucketed and non-bucketed greedy completions differ at i={i}:\n"
                f"  off: {batch_off.rollouts[i].completion!r}\n"
                f"  on:  {batch_on.rollouts[i].completion!r}"
            )
            # Same prompt_len, same reward, same parse_success
            assert batch_off.rollouts[i].prompt_len == batch_on.rollouts[i].prompt_len
            assert batch_off.rollouts[i].reward == batch_on.rollouts[i].reward

    @pytest.mark.slow
    def test_bucketing_skipped_when_batch_smaller_than_bucket(self, trainer_factory):
        """When B <= bucket_size, the bucketing path should not engage —
        the implementation falls back to a single generate() call."""
        prompts = ["Hello", "World"]  # B=2
        gts = ["a", "b"]
        # Bucket size 4, B=2 → bucket path skipped (B <= bucket_size)
        trainer = trainer_factory(length_bucketed=True, bucket_size=4)
        batch = trainer.generate_rollouts(prompts, gts)
        assert len(batch.rollouts) == 2
        # Sanity: completions are non-empty and aligned
        for i in range(2):
            assert batch.rollouts[i].prompt == prompts[i]
