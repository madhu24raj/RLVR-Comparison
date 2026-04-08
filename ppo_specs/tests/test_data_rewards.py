"""
Comprehensive tests for data loading, reward computation, evaluation metrics,
and utility functions in the RLVR-Comparison project.

Covers:
    1. GSM8K data loading and prompt formatting  (src/data.py)
    2. Reward functions and answer extraction     (src/rewards.py)
    3. Evaluation metrics and logging             (eval/metrics.py)
    4. Batch cycling and MC baseline utilities     (ppo_specs/utils.py)
    5. End-to-end data pipeline integration
"""

import sys
import os
import json
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so all imports resolve.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data import (
    extract_answer,
    format_prompt,
    format_prompt_with_template,
    load_gsm8k,
    get_experiment_subset,
)
from src.rewards import (
    extract_answer_from_completion,
    normalize_number,
    gsm8k_reward,
    batch_reward,
    trl_reward_fn,
)
from eval.metrics import (
    accuracy,
    reward_variance,
    advantage_estimation_error,
    compute_mc_advantage,
    ExperimentLogger,
)
from ppo_specs.utils import cycle_batch


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GSM8K Data Loading  (src/data.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractAnswer:
    """Tests for extract_answer (parsing GSM8K ground-truth answer strings)."""

    def test_standard_format(self):
        """Standard GSM8K answer ending with '#### 42'."""
        assert extract_answer("Some steps\n#### 42") == "42"

    def test_commas_removed(self):
        """Comma-separated numbers like '#### 1,234' become '1234'."""
        assert extract_answer("Steps\n#### 1,234") == "1234"

    def test_no_marker_fallback(self):
        """When no #### marker is present, return stripped full text."""
        assert extract_answer("  just a number 7  ") == "just a number 7"

    def test_multiple_hashes(self):
        """Only the part after the last #### is returned."""
        text = "Step #### wrong\n#### 99"
        assert extract_answer(text) == "99"


@pytest.mark.slow
class TestLoadGsm8k:
    """Tests that require downloading the GSM8K dataset (marked slow)."""

    def test_returns_expected_columns(self):
        """load_gsm8k adds a 'ground_truth' column alongside question/answer."""
        ds = load_gsm8k("train", n_samples=5, seed=0)
        for col in ("question", "answer", "ground_truth"):
            assert col in ds.column_names, f"Missing column: {col}"

    def test_n_samples_subset_size(self):
        """Requesting n_samples=5 yields exactly 5 rows."""
        ds = load_gsm8k("train", n_samples=5, seed=0)
        assert len(ds) == 5

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical subsets."""
        ds1 = load_gsm8k("train", n_samples=5, seed=123)
        ds2 = load_gsm8k("train", n_samples=5, seed=123)
        assert ds1["question"] == ds2["question"]

    def test_different_seeds_differ(self):
        """Different seeds produce different subsets (with high probability)."""
        ds1 = load_gsm8k("train", n_samples=5, seed=1)
        ds2 = load_gsm8k("train", n_samples=5, seed=999)
        assert ds1["question"] != ds2["question"]


class TestFormatPrompt:
    """Tests for format_prompt (plain-text prompt builder)."""

    def test_contains_question(self):
        """Formatted prompt must contain the original question text."""
        prompt = format_prompt("What is 2+2?")
        assert "What is 2+2?" in prompt

    def test_expected_structure(self):
        """Prompt follows the 'system\\n\\nQuestion: ...\\n\\nAnswer:' layout."""
        prompt = format_prompt("What is 2+2?")
        assert "Question: What is 2+2?" in prompt
        assert prompt.endswith("Answer:")

    def test_custom_system_prompt(self):
        """Custom system prompt is used when provided."""
        prompt = format_prompt("Q", system_prompt="Custom instruction")
        assert prompt.startswith("Custom instruction")


class TestFormatPromptWithTemplate:
    """Tests for format_prompt_with_template (tokenizer-aware formatting)."""

    def test_fallback_when_tokenizer_is_none(self):
        """Falls back to plain format when tokenizer is None."""
        plain = format_prompt("What is 2+2?")
        templated = format_prompt_with_template("What is 2+2?", tokenizer=None)
        assert templated == plain

    def test_fallback_when_no_apply_method(self):
        """Falls back when tokenizer lacks apply_chat_template."""
        fake_tok = MagicMock(spec=[])  # no apply_chat_template attr
        result = format_prompt_with_template("Q?", tokenizer=fake_tok)
        assert "Question: Q?" in result

    def test_uses_chat_template_when_available(self):
        """Calls tokenizer.apply_chat_template when present."""
        fake_tok = MagicMock()
        fake_tok.apply_chat_template.return_value = "<|im_start|>user\nQ?<|im_end|>"
        result = format_prompt_with_template("Q?", tokenizer=fake_tok)
        assert result == "<|im_start|>user\nQ?<|im_end|>"
        fake_tok.apply_chat_template.assert_called_once()

    def test_fallback_on_template_exception(self):
        """Falls back to plain format if apply_chat_template raises."""
        fake_tok = MagicMock()
        fake_tok.apply_chat_template.side_effect = RuntimeError("boom")
        result = format_prompt_with_template("Q?", tokenizer=fake_tok)
        assert "Question: Q?" in result

    @pytest.mark.slow
    def test_with_real_tokenizer(self):
        """Uses the Qwen2.5-0.5B-Instruct tokenizer to apply a real template."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        question = "What is 7 times 8?"
        result = format_prompt_with_template(question, tokenizer=tok)
        # The real template should include the question and be different from plain.
        assert question in result
        assert result != format_prompt(question)

    @pytest.mark.slow
    def test_templated_prompt_contains_question(self):
        """Templated prompt retains the original question regardless of format."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        question = "If Sally has 3 apples and buys 5 more, how many does she have?"
        result = format_prompt_with_template(question, tokenizer=tok)
        assert question in result


@pytest.mark.slow
class TestGetExperimentSubset:
    """Tests for get_experiment_subset (train/test split helper)."""

    def test_returns_train_and_test(self):
        """Returns a 2-tuple of (train_dataset, test_dataset)."""
        train, test = get_experiment_subset(n=5, seed=42)
        assert len(train) == 5
        assert len(test) > 0  # GSM8K test split is non-empty


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Reward Functions  (src/rewards.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractAnswerFromCompletion:
    """Tests for extract_answer_from_completion (model output parser)."""

    def test_hash_format(self):
        """Standard #### N format."""
        assert extract_answer_from_completion("blah\n#### 42") == "42"

    def test_boxed_format(self):
        """LaTeX \\boxed{N} format."""
        assert extract_answer_from_completion("result is \\boxed{15}") == "15"

    def test_the_answer_is_format(self):
        """Natural language 'the answer is N' format."""
        assert extract_answer_from_completion("So the answer is 99") == "99"

    def test_fallback_last_number(self):
        """Falls back to the last number when no pattern matches."""
        assert extract_answer_from_completion("step 1: 10, step 2: 20") == "20"

    def test_no_numbers_returns_none(self):
        """Returns None when the completion has no numbers at all."""
        assert extract_answer_from_completion("no numbers here") is None

    def test_commas_in_hash_format(self):
        """Commas are stripped from numbers in #### format."""
        assert extract_answer_from_completion("#### 1,234") == "1234"

    def test_negative_number(self):
        """Negative numbers are extracted correctly."""
        assert extract_answer_from_completion("#### -7") == "-7"


class TestNormalizeNumber:
    """Tests for normalize_number (string -> float conversion)."""

    def test_integer(self):
        """Plain integer string."""
        assert normalize_number("42") == 42.0

    def test_float(self):
        """Float string."""
        assert normalize_number("3.14") == pytest.approx(3.14)

    def test_comma_separated(self):
        """Comma-separated number like '1,000'."""
        assert normalize_number("1,000") == 1000.0

    def test_invalid_returns_none(self):
        """Non-numeric string returns None."""
        assert normalize_number("abc") is None

    def test_none_input_returns_none(self):
        """None input returns None."""
        assert normalize_number(None) is None

    def test_negative(self):
        """Negative number string."""
        assert normalize_number("-5") == -5.0


class TestGsm8kReward:
    """Tests for gsm8k_reward (binary scoring of a single completion)."""

    def test_correct_answer(self):
        """Correct answer yields reward 1.0."""
        assert gsm8k_reward("#### 42", "42") == 1.0

    def test_wrong_answer(self):
        """Wrong answer yields reward 0.0."""
        assert gsm8k_reward("#### 100", "42") == 0.0

    def test_no_answer_extractable(self):
        """No extractable number yields reward 0.0."""
        assert gsm8k_reward("no numbers anywhere", "42") == 0.0

    def test_float_tolerance(self):
        """Floating-point-equivalent answers compare as equal (42.0 vs 42)."""
        assert gsm8k_reward("#### 42.0", "42") == 1.0

    def test_negative_number_correct(self):
        """Correct negative answer yields 1.0."""
        assert gsm8k_reward("#### -3", "-3") == 1.0

    def test_negative_number_wrong(self):
        """Incorrect negative answer yields 0.0."""
        assert gsm8k_reward("#### -3", "3") == 0.0


class TestBatchReward:
    """Tests for batch_reward (vectorised scoring)."""

    def test_correct_list(self):
        """Returns correct per-element rewards."""
        completions = ["#### 1", "#### 2", "#### 3"]
        truths = ["1", "2", "99"]
        result = batch_reward(completions, truths)
        assert result == [1.0, 1.0, 0.0]

    def test_length_matches_input(self):
        """Output length matches input length."""
        completions = ["#### 1"] * 7
        truths = ["1"] * 7
        assert len(batch_reward(completions, truths)) == 7


class TestTrlRewardFn:
    """Tests for trl_reward_fn (TRL GRPOTrainer-compatible signature)."""

    def test_signature_accepts_kwargs(self):
        """Function accepts extra **kwargs without error."""
        result = trl_reward_fn(
            ["#### 5"],
            ["5"],
            extra_field="ignored",
        )
        assert result == [1.0]

    def test_returns_list(self):
        """Return type is a list."""
        result = trl_reward_fn(["#### 1", "#### 2"], ["1", "3"])
        assert isinstance(result, list)
        assert result == [1.0, 0.0]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Evaluation Metrics  (eval/metrics.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAccuracy:
    """Tests for accuracy (fraction of correct completions)."""

    def test_all_correct(self):
        """All rewards > 0.5 gives accuracy 1.0."""
        assert accuracy([1.0, 1.0, 1.0]) == 1.0

    def test_all_wrong(self):
        """All rewards 0.0 gives accuracy 0.0."""
        assert accuracy([0.0, 0.0, 0.0]) == 0.0

    def test_mixed(self):
        """Mixed rewards give the correct fraction."""
        assert accuracy([1.0, 0.0, 1.0, 0.0]) == pytest.approx(0.5)

    def test_empty_list(self):
        """Empty reward list returns 0.0."""
        assert accuracy([]) == 0.0


class TestRewardVariance:
    """Tests for reward_variance (per-step variance)."""

    def test_known_variances(self):
        """Computed variances match expected values."""
        rewards = [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        result = reward_variance(rewards)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.25)
        assert result[2] == pytest.approx(0.0)

    def test_output_length(self):
        """Output list has one entry per step."""
        rewards = [[0.0], [1.0], [0.5]]
        assert len(reward_variance(rewards)) == 3


class TestAdvantageEstimationError:
    """Tests for advantage_estimation_error (MAE between estimates and MC)."""

    def test_known_values(self):
        """MAE of [1,2,3] vs [1,1,1] = mean(|0,1,2|) = 1.0."""
        est = np.array([1.0, 2.0, 3.0])
        mc = np.array([1.0, 1.0, 1.0])
        assert advantage_estimation_error(est, mc) == pytest.approx(1.0)

    def test_identical_is_zero(self):
        """Identical arrays give error 0.0."""
        a = np.array([5.0, 10.0])
        assert advantage_estimation_error(a, a) == pytest.approx(0.0)


class TestComputeMcAdvantage:
    """Tests for compute_mc_advantage (MC baseline estimation)."""

    def test_correct_baselines(self):
        """Baseline for each prompt equals the mean reward of its rollouts."""
        rewards_per_prompt = {
            "p1": [0.0, 1.0, 1.0],
            "p2": [0.0, 0.0],
        }
        baselines = compute_mc_advantage(rewards_per_prompt)
        assert baselines["p1"] == pytest.approx(2.0 / 3.0)
        assert baselines["p2"] == pytest.approx(0.0)

    def test_returns_correct_count(self):
        """Number of entries equals number of prompts."""
        rewards_per_prompt = {"a": [1.0], "b": [0.5], "c": [0.0]}
        baselines = compute_mc_advantage(rewards_per_prompt)
        assert len(baselines) == 3


class TestExperimentLogger:
    """Tests for ExperimentLogger (JSON metric logging)."""

    def test_creates_output_directory(self):
        """Logger creates the output directory if it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "subdir", "deeper")
            logger = ExperimentLogger("test", output_dir=out)
            assert os.path.isdir(out)

    def test_log_step_and_save(self):
        """log_step + save produces valid JSON with correct entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("run1", output_dir=tmpdir)
            logger.log_step(0, accuracy=0.5, loss=1.2)
            logger.log_step(1, accuracy=0.7, loss=0.9)
            logger.save()

            path = os.path.join(tmpdir, "run1.json")
            assert os.path.isfile(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 2
            assert data[0]["step"] == 0
            assert data[1]["accuracy"] == pytest.approx(0.7)

    def test_load_reads_back(self):
        """load() restores the saved log entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger("run2", output_dir=tmpdir)
            logger.log_step(5, reward=0.8)
            logger.save()

            logger2 = ExperimentLogger("run2", output_dir=tmpdir)
            loaded = logger2.load()
            assert len(loaded) == 1
            assert loaded[0]["step"] == 5
            assert loaded[0]["reward"] == pytest.approx(0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Utils  (ppo_specs/utils.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCycleBatch:
    """Tests for cycle_batch (contiguous slice with wrap-around)."""

    def test_basic_slicing(self):
        """Step 0 returns the first batch_size items."""
        items = [0, 1, 2, 3, 4]
        assert cycle_batch(items, step=0, batch_size=2) == [0, 1]

    def test_second_step(self):
        """Step 1 returns the next slice."""
        items = [0, 1, 2, 3, 4]
        assert cycle_batch(items, step=1, batch_size=2) == [2, 3]

    def test_wrapping(self):
        """Wraps around the end of the list."""
        items = [0, 1, 2, 3, 4]
        # step=2, batch=2 -> start=4, need wrap
        result = cycle_batch(items, step=2, batch_size=2)
        assert result == [4, 0]

    def test_batch_equals_length(self):
        """batch_size == len(items) returns the full list from the right offset."""
        items = [0, 1, 2]
        assert cycle_batch(items, step=0, batch_size=3) == [0, 1, 2]

    def test_batch_larger_than_list(self):
        """batch_size > len(items) wraps around and returns correct items."""
        items = [0, 1, 2]
        result = cycle_batch(items, step=0, batch_size=5)
        assert len(result) == 5
        assert result == [0, 1, 2, 0, 1]

    def test_deterministic_for_same_step(self):
        """Same step always returns the same batch."""
        items = list(range(10))
        a = cycle_batch(items, step=3, batch_size=4)
        b = cycle_batch(items, step=3, batch_size=4)
        assert a == b

    def test_covers_all_items(self):
        """Cycling through enough steps covers every item at least once."""
        items = list(range(7))
        seen = set()
        for step in range(7):
            seen.update(cycle_batch(items, step=step, batch_size=1))
        assert seen == set(items)


class TestSetupMcBaselines:
    """Tests for setup_mc_baselines (with mocked trainer)."""

    def test_returns_dict_with_correct_count(self):
        """Returns a dict with one entry per reference prompt."""
        mock_trainer = MagicMock()
        mock_trainer.reward_fn = MagicMock()

        prompts = [f"prompt_{i}" for i in range(10)]
        gts = [str(i) for i in range(10)]

        fake_baselines = {p: 0.5 for p in prompts[:3]}

        with patch(
            "ppo_specs.utils.estimate_mc_advantages",
            return_value=fake_baselines,
        ):
            from ppo_specs.utils import setup_mc_baselines
            import torch

            result = setup_mc_baselines(
                trainer=mock_trainer,
                train_prompts=prompts,
                train_gts=gts,
                n_steps=5,
                max_new_tokens=64,
                device=torch.device("cpu"),
                n_ref_prompts=3,
            )
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result.values())


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Data Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestDataPipelineIntegration:
    """End-to-end integration tests spanning data loading through reward scoring."""

    def test_load_format_reward_pipeline(self):
        """load_gsm8k -> format_prompt -> gsm8k_reward pipeline runs without error."""
        ds = load_gsm8k("train", n_samples=5, seed=0)
        for row in ds:
            prompt = format_prompt(row["question"])
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            # Simulate a correct completion using the ground truth
            completion = f"The answer is #### {row['ground_truth']}"
            reward = gsm8k_reward(completion, row["ground_truth"])
            assert reward == 1.0

    def test_ground_truths_are_valid_numbers(self):
        """All ground_truth values from load_gsm8k are parseable numbers."""
        ds = load_gsm8k("train", n_samples=5, seed=0)
        for row in ds:
            num = normalize_number(row["ground_truth"])
            assert num is not None, (
                f"ground_truth '{row['ground_truth']}' could not be parsed"
            )

    def test_format_prompt_non_empty_with_question(self):
        """format_prompt output is a non-empty string containing the question."""
        ds = load_gsm8k("train", n_samples=5, seed=0)
        for row in ds:
            prompt = format_prompt(row["question"])
            assert len(prompt) > 0
            assert row["question"] in prompt
