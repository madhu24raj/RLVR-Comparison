"""
Reward functions for RLVR on GSM8K.

Parses model completions, extracts the final numerical answer,
and compares against ground truth. Returns binary reward (0 or 1).
"""

import re


def extract_answer_from_completion(completion: str) -> str:
    """Extract the final numerical answer from a model completion.

    Strict extraction: only formats where the model has explicitly
    committed to an answer count.

      - "#### 42"           (GSM8K standard format -- the format we
                             instruct the model to use in the system prompt)
      - "\\boxed{42}"       (LaTeX format)
      - "The answer is 42"  (anchored natural language)

    Returns None if no answer is found in any of these formats.

    NOTE: We deliberately do NOT fall back to "last number in completion".
    That fallback gives reward to outputs that contain the right number
    by accident (intermediate calculations, problem restatement, etc.)
    and undermines the verifiability that justifies RLVR. Spurious reward
    is worse than no reward: it teaches the model to emit numbers without
    structuring its output. See ppo_specs/specs/logic.md L13.
    """
    # 1) #### format. Take the LAST match -- if the model emits multiple
    #    #### markers (e.g. inside its reasoning), we want the final one,
    #    consistent with data.py:extract_answer which uses split("####")[-1].
    hash_matches = re.findall(r"####\s*(-?[\d,]+\.?\d*)", completion)
    if hash_matches:
        return hash_matches[-1].replace(",", "")

    # 2) \boxed{} format
    match = re.search(r"\\boxed\{([^}]+)\}", completion)
    if match:
        return match.group(1).strip()

    # 3) "(the )answer is X" -- anchored on natural language, low FP rate.
    match = re.search(
        r"(?:the\s+)?answer\s+is\s*:?\s*(-?[\d,]+\.?\d*)",
        completion,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "")

    return None


def matches_boxed_format(completion: str) -> bool:
    """True iff the completion contains a `\\boxed{...}` anchor.

    Diagnostic helper for reward-starvation debugging: we ask the model
    to emit `\\boxed{}` in the system prompt (higher parse rate than
    `####` on Qwen2.5-0.5B), so this rate tracks whether the model is
    obeying the prompt format. Distinct from `extract_answer_from_completion`,
    which also accepts `####` and "the answer is" forms.
    """
    return re.search(r"\\boxed\{[^}]+\}", completion) is not None


def normalize_number(s: str) -> float:
    """Normalize a number string for comparison.

    Handles integers, floats, and comma-separated numbers.
    """
    if s is None:
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def gsm8k_reward(completion: str, ground_truth: str) -> float:
    """Score a model completion against the ground truth.

    Args:
        completion: Model-generated text
        ground_truth: The correct answer string (already extracted from GSM8K)

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    predicted = extract_answer_from_completion(completion)
    if predicted is None:
        return 0.0

    pred_num = normalize_number(predicted)
    truth_num = normalize_number(ground_truth)

    if pred_num is None or truth_num is None:
        return 0.0

    # Compare with small tolerance for floating point
    return 1.0 if abs(pred_num - truth_num) < 1e-6 else 0.0


def batch_reward(completions: list[str], ground_truths: list[str]) -> list[float]:
    """Score a batch of completions."""
    return [gsm8k_reward(c, gt) for c, gt in zip(completions, ground_truths)]


# --- TRL-compatible reward function ---
# TRL's GRPOTrainer expects a reward function with signature:
#   reward_fn(completions: list[str], **kwargs) -> list[float]
# The ground truths are passed via the dataset's 'ground_truth' column.

def trl_reward_fn(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """Reward function compatible with TRL's GRPOTrainer.

    Usage with GRPOTrainer:
        trainer = GRPOTrainer(
            ...,
            reward_funcs=trl_reward_fn,
            ...
        )
    """
    return batch_reward(completions, ground_truth)


if __name__ == "__main__":
    # Sanity check with example completions
    test_cases = [
        ("Let me solve this step by step.\n3 + 4 = 7\n#### 7", "7"),
        ("The answer is 42", "42"),
        ("Working through it... I get \\boxed{15}", "15"),
        ("Some random text with no answer", "10"),
        ("#### 100", "200"),  # Wrong answer
        ("Step 1: 5*3=15\nStep 2: 15+10=25\n#### 25", "25"),
        # L13: strict extraction -- these should now return None / reward 0,
        # because the model never explicitly committed to an answer.
        ("Bob has 5 apples and 3 oranges, so 8 fruits total.", "8"),
        ("First I compute 6*7=42, then 42-2=40.", "40"),
    ]

    print("Reward function tests:")
    for completion, truth in test_cases:
        pred = extract_answer_from_completion(completion)
        reward = gsm8k_reward(completion, truth)
        print(f"  Predicted: {str(pred):>6s} | Truth: {truth:>6s} | Reward: {reward}")
