"""
Tests for batched PPO trainer operations.

Verifies that batched implementations (log-probs, critic values, generation,
evaluation) produce results consistent with single-sample legacy methods and
satisfy correctness invariants.

Run with:
    pytest ppo_specs/tests/test_batched_ops.py -v
    pytest ppo_specs/tests/test_batched_ops.py -v -m "not slow"   # skip GPU/generation tests
"""

import sys
import os
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

from ppo_specs.config import PPOConfig
from ppo_specs.critic import build_critic
from ppo_specs.ppo_trainer import (
    PPOTrainer,
    Rollout,
    RolloutBatch,
)
from src.rewards import gsm8k_reward


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Prompts of varying length for consistency tests
VARYING_PROMPTS = [
    "What is 2+2?",
    "Solve the following math problem step by step: 3 + 5 =",
    "Please compute the result of adding seven and twelve together and show your work.",
    "5-1=",
]
GROUND_TRUTHS = ["4", "8", "19", "4"]


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
        experiment_name="test_batched_ops",
        critic_capacity="medium",
        do_sample=True,
        temperature=0.7,
    )
    defaults.update(overrides)
    return PPOConfig(**defaults)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (load model once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_model_and_tokenizer():
    """Load the model and tokenizer once for the entire module."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch_dtype
    ).to(DEVICE)

    return model, tokenizer


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def trainer_small(shared_model_and_tokenizer):
    """PPOTrainer with critic_capacity='small' (2-layer MLP)."""
    model, tokenizer = shared_model_and_tokenizer
    cfg = _tiny_config(critic_capacity="small")
    hidden_size = model.config.hidden_size
    critic = build_critic("small", hidden_size).to(DEVICE)
    return PPOTrainer(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        critic=critic,
        reward_fn=gsm8k_reward,
        device=DEVICE,
    )


@pytest.fixture(scope="module")
def trainer_large(shared_model_and_tokenizer):
    """PPOTrainer with critic_capacity='large' (deep MLP)."""
    model, tokenizer = shared_model_and_tokenizer
    cfg = _tiny_config(critic_capacity="large")
    hidden_size = model.config.hidden_size
    critic = build_critic("large", hidden_size).to(DEVICE)
    return PPOTrainer(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        critic=critic,
        reward_fn=gsm8k_reward,
        device=DEVICE,
    )


def _make_dummy_rollouts(tokenizer, prompts, completions, ground_truths, device=DEVICE):
    """Create synthetic rollouts from given prompts and completions."""
    rollouts = []
    for i, (prompt, comp, gt) in enumerate(zip(prompts, completions, ground_truths)):
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


# ===========================================================================
# 1. Batched vs single log-prob consistency
# ===========================================================================


class TestBatchedVsSingleLogProb:
    """The critical correctness test: batched log-probs must be self-consistent.

    Note: The legacy _sequence_log_prob (no attention mask) and the batched
    _batched_sequence_log_probs (with attention mask) produce slightly different
    results because padding changes attention patterns.  The correct comparison
    is batched-N vs N × batched-1 (both use attention masks).
    """

    @pytest.mark.slow
    def test_batched_matches_individual_batched(self, trainer_medium, shared_model_and_tokenizer):
        """
        _batched_sequence_log_probs on N samples must match running
        _batched_sequence_log_probs on each sample individually (batch=1).

        This verifies that padding in the batch does not corrupt per-sample
        log-probs, which is critical for a correct PPO ratio.
        """
        _, tokenizer = shared_model_and_tokenizer
        prompts = VARYING_PROMPTS[:3]

        all_full_ids = []
        prompt_lens = []
        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            comp_ids = tokenizer.encode(" The answer is 4.", add_special_tokens=False)
            all_full_ids.append(ids + comp_ids)
            prompt_lens.append(len(ids))

        # Individual batched calls (batch_size=1 each, no cross-sample padding)
        individual_results = []
        with torch.no_grad():
            for full_ids, pl in zip(all_full_ids, prompt_lens):
                lp = trainer_medium._batched_sequence_log_probs([full_ids], [pl])
                individual_results.append(lp.item())
        individual_tensor = torch.tensor(individual_results, dtype=torch.float32)

        # Full batched call (all samples together, with padding)
        with torch.no_grad():
            batched_tensor = trainer_medium._batched_sequence_log_probs(
                all_full_ids, prompt_lens
            ).cpu().float()

        torch.testing.assert_close(
            batched_tensor, individual_tensor, atol=1e-4, rtol=1e-4,
        )

    @pytest.mark.slow
    def test_batched_matches_individual_four_prompts(self, trainer_medium, shared_model_and_tokenizer):
        """Same consistency test with 4 prompts of highly varying lengths."""
        _, tokenizer = shared_model_and_tokenizer
        prompts = VARYING_PROMPTS  # all 4

        all_full_ids = []
        prompt_lens = []
        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            comp_ids = tokenizer.encode(" Result: 42 #### 42", add_special_tokens=False)
            all_full_ids.append(ids + comp_ids)
            prompt_lens.append(len(ids))

        individual_results = []
        with torch.no_grad():
            for full_ids, pl in zip(all_full_ids, prompt_lens):
                lp = trainer_medium._batched_sequence_log_probs([full_ids], [pl])
                individual_results.append(lp.item())
        individual_tensor = torch.tensor(individual_results, dtype=torch.float32)

        with torch.no_grad():
            batched_tensor = trainer_medium._batched_sequence_log_probs(
                all_full_ids, prompt_lens
            ).cpu().float()

        torch.testing.assert_close(
            batched_tensor, single_tensor, atol=1e-4, rtol=1e-4,
        )


# ===========================================================================
# 2. Batched generation produces valid rollouts
# ===========================================================================


class TestBatchedGenerationRollouts:
    """Verify generate_rollouts produces well-formed rollouts."""

    @pytest.mark.slow
    def test_generate_rollouts_basic_structure(self, trainer_none):
        """Each rollout from generate_rollouts must have valid fields."""
        prompts = VARYING_PROMPTS[:3]
        gts = GROUND_TRUTHS[:3]

        before_rollouts = trainer_none.total_rollouts
        batch = trainer_none.generate_rollouts(prompts, gts)

        assert len(batch.rollouts) == len(prompts), (
            f"Expected {len(prompts)} rollouts, got {len(batch.rollouts)}"
        )

        for i, r in enumerate(batch.rollouts):
            # Non-empty completion
            assert isinstance(r.completion, str), f"Rollout {i}: completion is not a string"
            assert len(r.completion) > 0, f"Rollout {i}: empty completion"

            # Reward is binary {0, 1}
            assert r.reward in (0.0, 1.0), (
                f"Rollout {i}: reward={r.reward}, expected 0.0 or 1.0"
            )

            # old_log_prob should be negative (log-probability of a sequence)
            assert r.old_log_prob < 0.0, (
                f"Rollout {i}: old_log_prob={r.old_log_prob}, expected < 0"
            )

            # full_ids is non-empty
            assert len(r.full_ids) > 0, f"Rollout {i}: empty full_ids"

            # prompt_len is positive
            assert r.prompt_len > 0, f"Rollout {i}: prompt_len={r.prompt_len}"

            # full_ids should be longer than prompt_len (there is a completion)
            assert len(r.full_ids) >= r.prompt_len, (
                f"Rollout {i}: full_ids length {len(r.full_ids)} < prompt_len {r.prompt_len}"
            )

        # total_rollouts counter incremented
        assert trainer_none.total_rollouts == before_rollouts + len(prompts)


# ===========================================================================
# 3. Left-padding correctness
# ===========================================================================


class TestLeftPaddingCorrectness:
    """Verify left-padding for batched tokenization works correctly."""

    @pytest.mark.slow
    def test_left_padding_structure(self, shared_model_and_tokenizer):
        """
        Left-padding must place pad tokens on the left for shorter prompts,
        with attention_mask zeros where padding is, and sum of attention_mask
        matching individual tokenization lengths.
        """
        _, tokenizer = shared_model_and_tokenizer

        prompts = VARYING_PROMPTS[:3]  # varying lengths

        # Individual tokenization lengths (no padding)
        individual_lens = []
        for p in prompts:
            ids = tokenizer.encode(p, add_special_tokens=True)
            individual_lens.append(len(ids))

        # Batched tokenization with left-padding
        enc = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        B, S = input_ids.shape

        # Check prompt_lens from attention_mask match individual lengths
        batched_lens = attention_mask.sum(dim=1).tolist()
        for i, (bl, il) in enumerate(zip(batched_lens, individual_lens)):
            assert bl == il, (
                f"Prompt {i}: batched attention_mask sum={bl} != individual len={il}"
            )

        # The shortest prompt should have padding on the left
        min_len = min(individual_lens)
        max_len = max(individual_lens)
        if min_len < max_len:
            # Find a sample that is shorter than max
            short_idx = individual_lens.index(min_len)
            pad_count = S - min_len

            # Left-padding: first `pad_count` tokens should be pad_token_id
            for j in range(pad_count):
                assert input_ids[short_idx, j].item() == tokenizer.pad_token_id, (
                    f"Expected pad token at position {j} for shortest prompt"
                )

            # Attention mask should be 0 where padding is
            for j in range(pad_count):
                assert attention_mask[short_idx, j].item() == 0, (
                    f"Expected attention_mask=0 at pad position {j}"
                )

            # Attention mask should be 1 after padding
            for j in range(pad_count, S):
                assert attention_mask[short_idx, j].item() == 1, (
                    f"Expected attention_mask=1 at real-token position {j}"
                )


# ===========================================================================
# 4. Batched critic values consistency
# ===========================================================================


class TestBatchedCriticValues:
    """Batched critic values must match single-sample critic evaluation."""

    @pytest.mark.slow
    def test_batched_critic_matches_single_medium(self, trainer_medium, shared_model_and_tokenizer):
        """Medium critic: batched vs individual (batch=1) should match."""
        _, tokenizer = shared_model_and_tokenizer
        prompts = VARYING_PROMPTS[:3]

        # Batched critic values (all at once)
        with torch.no_grad():
            batched_vals = trainer_medium._batched_critic_values(prompts).cpu()

        # Individual batched calls (batch=1 each, no cross-sample padding)
        individual_vals = []
        for p in prompts:
            with torch.no_grad():
                v = trainer_medium._batched_critic_values([p])
            individual_vals.append(v.item())
        individual_tensor = torch.tensor(individual_vals, dtype=torch.float32)

        torch.testing.assert_close(
            batched_vals.float(), individual_tensor, atol=1e-4, rtol=1e-4,
        )

    @pytest.mark.slow
    def test_batched_critic_matches_single_small(self, trainer_small, shared_model_and_tokenizer):
        """Small critic: batched vs individual (batch=1) should match."""
        _, tokenizer = shared_model_and_tokenizer
        prompts = VARYING_PROMPTS[:3]

        with torch.no_grad():
            batched_vals = trainer_small._batched_critic_values(prompts).cpu()

        individual_vals = []
        for p in prompts:
            with torch.no_grad():
                v = trainer_small._batched_critic_values([p])
            individual_vals.append(v.item())
        individual_tensor = torch.tensor(individual_vals, dtype=torch.float32)

        torch.testing.assert_close(
            batched_vals.float(), single_tensor, atol=1e-4, rtol=1e-4,
        )

    @pytest.mark.slow
    def test_batched_critic_matches_single_large(self, trainer_large, shared_model_and_tokenizer):
        """Large critic: batched vs individual (batch=1) should match."""
        _, tokenizer = shared_model_and_tokenizer
        prompts = VARYING_PROMPTS[:3]

        with torch.no_grad():
            batched_vals = trainer_large._batched_critic_values(prompts).cpu()

        individual_vals = []
        for p in prompts:
            with torch.no_grad():
                v = trainer_large._batched_critic_values([p])
            individual_vals.append(v.item())
        individual_tensor = torch.tensor(individual_vals, dtype=torch.float32)

        torch.testing.assert_close(
            batched_vals.float(), individual_tensor, atol=1e-4, rtol=1e-4,
        )

    @pytest.mark.slow
    def test_none_critic_returns_zeros(self, trainer_none):
        """critic_capacity='none': _batched_critic_values must return all zeros."""
        prompts = VARYING_PROMPTS[:3]
        with torch.no_grad():
            vals = trainer_none._batched_critic_values(prompts).cpu()

        expected = torch.zeros(len(prompts))
        torch.testing.assert_close(vals.float(), expected, atol=0.0, rtol=0.0)


# ===========================================================================
# 5. Batched evaluate correctness
# ===========================================================================


class TestBatchedEvaluate:
    """Evaluate should produce a deterministic float in [0, 1]."""

    @pytest.mark.slow
    def test_evaluate_returns_valid_accuracy(self, trainer_none):
        """evaluate() must return a float between 0.0 and 1.0."""
        prompts = [
            "What is 2+2?",
            "What is 3+1?",
            "What is 5+5?",
            "What is 1+1?",
            "What is 10-3?",
        ]
        gts = ["4", "4", "10", "2", "7"]
        acc = trainer_none.evaluate(prompts, gts, n_eval=5)
        assert isinstance(acc, float), f"Expected float, got {type(acc)}"
        assert 0.0 <= acc <= 1.0, f"Accuracy {acc} outside [0, 1]"

    @pytest.mark.slow
    def test_evaluate_is_deterministic(self, trainer_none):
        """Greedy decoding makes evaluate() deterministic -- two calls, same result."""
        prompts = [
            "What is 2+2?",
            "What is 3+1?",
            "What is 5+5?",
            "What is 1+1?",
            "What is 10-3?",
        ]
        gts = ["4", "4", "10", "2", "7"]
        acc1 = trainer_none.evaluate(prompts, gts, n_eval=5)
        acc2 = trainer_none.evaluate(prompts, gts, n_eval=5)
        assert acc1 == acc2, f"evaluate() not deterministic: {acc1} vs {acc2}"


# ===========================================================================
# 6. fp32 log_softmax verification
# ===========================================================================


class TestFp32LogSoftmax:
    """Verify that log_softmax in fp32 avoids -inf and NaN even with bf16 model."""

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batched_log_probs_finite_bf16(self, trainer_medium, shared_model_and_tokenizer):
        """
        With a bf16 model, _batched_sequence_log_probs must produce finite values.
        This validates the .float() cast on logits before log_softmax.
        """
        _, tokenizer = shared_model_and_tokenizer
        prompts = VARYING_PROMPTS[:3]

        all_full_ids = []
        prompt_lens = []
        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            comp_ids = tokenizer.encode(" The answer is 42.", add_special_tokens=False)
            all_full_ids.append(ids + comp_ids)
            prompt_lens.append(len(ids))

        with torch.no_grad():
            result = trainer_medium._batched_sequence_log_probs(
                all_full_ids, prompt_lens
            )

        assert result.shape == (len(prompts),)
        assert torch.isfinite(result).all(), (
            f"Non-finite log-probs detected: {result.tolist()}"
        )
        # Log-probs should be negative
        assert (result < 0).all(), (
            f"Expected all negative log-probs, got: {result.tolist()}"
        )

    @pytest.mark.slow
    def test_batched_log_probs_no_nan(self, trainer_medium, shared_model_and_tokenizer):
        """Even on CPU (fp32), results must be finite with no NaN."""
        _, tokenizer = shared_model_and_tokenizer
        prompts = VARYING_PROMPTS

        all_full_ids = []
        prompt_lens = []
        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            comp_ids = tokenizer.encode(" Step 1: compute. #### 7", add_special_tokens=False)
            all_full_ids.append(ids + comp_ids)
            prompt_lens.append(len(ids))

        with torch.no_grad():
            result = trainer_medium._batched_sequence_log_probs(
                all_full_ids, prompt_lens
            )

        assert not torch.isnan(result).any(), f"NaN in log-probs: {result.tolist()}"
        assert not torch.isinf(result).any(), f"Inf in log-probs: {result.tolist()}"


# ===========================================================================
# 7. _policy_log_probs matches batched version
# ===========================================================================


class TestPolicyLogProbsMatchesBatched:
    """_policy_log_probs must be equivalent to _batched_sequence_log_probs."""

    @pytest.mark.slow
    def test_policy_log_probs_finite_and_matches_batched(
        self, trainer_none, shared_model_and_tokenizer
    ):
        """
        _policy_log_probs(batch) should return a [B] tensor of finite values
        that equals _batched_sequence_log_probs called with the same inputs.
        """
        _, tokenizer = shared_model_and_tokenizer

        prompts = VARYING_PROMPTS[:3]
        completions = [" The answer is 4.", " 8 is the result.", " nineteen."]
        gts = GROUND_TRUTHS[:3]

        batch = _make_dummy_rollouts(tokenizer, prompts, completions, gts, device=DEVICE)

        # _policy_log_probs uses _batched_sequence_log_probs internally
        policy_lp = trainer_none._policy_log_probs(batch)

        assert policy_lp.shape == (len(prompts),), (
            f"Expected shape ({len(prompts)},), got {policy_lp.shape}"
        )
        assert torch.isfinite(policy_lp).all(), (
            f"Non-finite _policy_log_probs: {policy_lp.tolist()}"
        )

        # Direct call to _batched_sequence_log_probs with same data
        direct_lp = trainer_none._batched_sequence_log_probs(
            [r.full_ids for r in batch.rollouts],
            [r.prompt_len for r in batch.rollouts],
        )

        torch.testing.assert_close(
            policy_lp.detach().cpu().float(),
            direct_lp.detach().cpu().float(),
            atol=1e-6, rtol=1e-6,
        )


# ===========================================================================
# 8. _critic_forward batched correctness
# ===========================================================================


class TestCriticForwardBatched:
    """_critic_forward must return correct shapes and types."""

    @pytest.mark.slow
    def test_critic_forward_trainable(self, trainer_medium, shared_model_and_tokenizer):
        """
        For a trainable critic, _critic_forward returns (tensor[B], scalar loss).
        Values must be finite and loss non-negative.
        """
        _, tokenizer = shared_model_and_tokenizer

        prompts = VARYING_PROMPTS[:3]
        completions = [" 4", " 8", " 19"]
        gts = GROUND_TRUTHS[:3]

        batch = _make_dummy_rollouts(tokenizer, prompts, completions, gts, device=DEVICE)
        rewards = batch.rewards().to(DEVICE)

        values, loss = trainer_medium._critic_forward(batch, rewards)

        # values is a [B] tensor
        assert values is not None, "Trainable critic should return values, not None"
        assert values.shape == (len(prompts),), (
            f"Expected values shape ({len(prompts)},), got {values.shape}"
        )
        assert torch.isfinite(values).all(), (
            f"Non-finite critic values: {values.tolist()}"
        )

        # loss is a scalar
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"Non-finite critic loss: {loss.item()}"
        assert loss.item() >= 0.0, f"Critic loss should be >= 0, got {loss.item()}"

    @pytest.mark.slow
    def test_critic_forward_none(self, trainer_none, shared_model_and_tokenizer):
        """
        For critic_capacity='none', _critic_forward returns (None, tensor(0.0)).
        """
        _, tokenizer = shared_model_and_tokenizer

        prompts = VARYING_PROMPTS[:2]
        completions = [" 4", " 8"]
        gts = GROUND_TRUTHS[:2]

        batch = _make_dummy_rollouts(tokenizer, prompts, completions, gts, device=DEVICE)
        rewards = batch.rewards().to(DEVICE)

        values, loss = trainer_none._critic_forward(batch, rewards)

        assert values is None, f"Expected None values for 'none' critic, got {values}"
        assert loss.item() == 0.0, f"Expected loss=0.0 for 'none' critic, got {loss.item()}"


# ===========================================================================
# 9. _eval_critic_on_prompts batched
# ===========================================================================


class TestEvalCriticOnPrompts:
    """_eval_critic_on_prompts must return correctly shaped numpy arrays."""

    @pytest.mark.slow
    def test_eval_critic_shape_and_finite(self, trainer_medium):
        """Trainable critic: returns (n_prompts,) array of finite values."""
        prompts = VARYING_PROMPTS[:3]
        result = trainer_medium._eval_critic_on_prompts(prompts)

        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
        assert result.shape == (len(prompts),), (
            f"Expected shape ({len(prompts)},), got {result.shape}"
        )
        assert np.all(np.isfinite(result)), f"Non-finite values: {result}"

    @pytest.mark.slow
    def test_eval_critic_none_returns_zeros(self, trainer_none):
        """'none' critic: all values should be exactly zero."""
        prompts = VARYING_PROMPTS
        result = trainer_none._eval_critic_on_prompts(prompts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(prompts),)
        np.testing.assert_array_equal(result, np.zeros(len(prompts)))

    @pytest.mark.slow
    def test_eval_critic_multiple_capacities(
        self, trainer_small, trainer_medium, trainer_large
    ):
        """All trainable critics produce finite arrays of correct shape."""
        prompts = VARYING_PROMPTS[:3]

        for trainer, label in [
            (trainer_small, "small"),
            (trainer_medium, "medium"),
            (trainer_large, "large"),
        ]:
            result = trainer._eval_critic_on_prompts(prompts)
            assert result.shape == (len(prompts),), (
                f"{label} critic: wrong shape {result.shape}"
            )
            assert np.all(np.isfinite(result)), (
                f"{label} critic: non-finite values {result}"
            )
