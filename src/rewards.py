"""
Reward functions for RLVR on GSM8K.

Parses model completions, extracts the final numerical answer,
and compares against ground truth. Returns binary reward (0 or 1).
"""

import math
import re
from typing import List

import torch


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


class SelfJudgeRewardModel:
    """Continuous reward signal using a frozen reference model's log-likelihood.

    Scores how likely the reference model considers the completion given the
    question. Uses mean log-probability of completion tokens as the raw signal,
    optionally sigmoid-normalized to [0, 1].
    """

    def __init__(self, model, tokenizer, normalize: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.normalize = normalize
        # Ensure pad token exists for tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def score(self, question: str, completion: str) -> float:
        """Score a single question-completion pair.

        Returns the mean log-probability of the completion tokens, optionally
        sigmoid-normalized to [0, 1]. Empty completions return 0.5 (normalized)
        or 0.0 (unnormalized).
        """
        if not completion:
            return 0.5 if self.normalize else 0.0

        # Tokenize question and full sequence
        question_ids = self.tokenizer.encode(question, add_special_tokens=False)
        full_text = question + completion
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Number of completion tokens
        n_completion = len(full_ids) - len(question_ids)
        if n_completion <= 0:
            return 0.5 if self.normalize else 0.0

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=next(self.model.parameters()).device)
        outputs = self.model(input_ids)
        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

        # Log-probabilities for each next-token prediction
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Extract log-probs for the completion tokens only.
        # Position i predicts token i+1. Completion tokens start at index
        # len(question_ids) in the full sequence.
        completion_start = len(question_ids)
        completion_log_probs = []
        for i in range(completion_start, len(full_ids)):
            # logits at position i-1 predict token at position i
            token_id = full_ids[i]
            lp = log_probs[i - 1, token_id].item()
            completion_log_probs.append(lp)

        mean_lp = sum(completion_log_probs) / len(completion_log_probs)

        if self.normalize:
            # Sigmoid normalization: shifts so typical log-probs (-3 to -1)
            # map to usable 0.05-0.73 range
            return 1.0 / (1.0 + math.exp(-(mean_lp + 2.0)))
        return mean_lp

    @torch.no_grad()
    def batch_score(self, questions: list[str], completions: list[str]) -> list[float]:
        """Score a batch of (question, completion) pairs in ONE forward pass.

        Phase 1 fused implementation per
        ppo_specs/specs/reward_model_integration.md "Performance gap":
        replaces the previous per-sample loop (B × `score()`) with a single
        batched forward through the reference model and a vectorized
        completion-mask aggregation. At 8B with B=16 this saves ~1.4 s/step
        of pure single-sample inference (~5 minutes per 200-step run).

        Falls back to the per-sample path on three edge cases that the
        fused vector math does not handle cleanly:
          - empty `completions` list (B == 0)
          - any completion is empty (`""`) — would yield n_completion=0
          - a question + completion pair tokenizes to the same length as
            the question alone (no completion tokens at all)

        The per-sample fallback returns the same values as the unbatched
        `score()` method, including normalization handling.

        Returns a list of floats (length B), matching the legacy signature.
        """
        if not completions:
            return []

        # Pre-tokenize each (q, c) pair to capture the prompt boundary.
        # We tokenize the full text and the question separately so the
        # completion-token region is unambiguous regardless of how the
        # tokenizer handles whitespace at the boundary.
        question_ids_per: list[list[int]] = []
        full_ids_per: list[list[int]] = []
        completion_lens: list[int] = []
        for q, c in zip(questions, completions):
            if not c:
                # Edge case: empty completion — fall back to per-sample
                # path for the entire batch (rare; preserves bit-identical
                # output for completeness).
                return [self.score(q, c) for q, c in zip(questions, completions)]
            qid = self.tokenizer.encode(q, add_special_tokens=False)
            fid = self.tokenizer.encode(q + c, add_special_tokens=False)
            n_comp = len(fid) - len(qid)
            if n_comp <= 0:
                return [self.score(q, c) for q, c in zip(questions, completions)]
            question_ids_per.append(qid)
            full_ids_per.append(fid)
            completion_lens.append(n_comp)

        B = len(full_ids_per)
        max_len = max(len(ids) for ids in full_ids_per)
        pad_id = self.tokenizer.pad_token_id
        device = next(self.model.parameters()).device

        padded = torch.full(
            (B, max_len), pad_id, dtype=torch.long, device=device,
        )
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(full_ids_per):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, :len(ids)] = 1

        # ONE batched forward pass. The reference model is frozen; outer
        # @torch.no_grad() decorator already prevents graph allocation.
        outputs = self.model(input_ids=padded, attention_mask=attention_mask)
        # logits[b, t, v] predicts token at position t+1 — fp32 cast
        # preserves the same numerics as the per-sample path's log_softmax.
        logits = outputs.logits.float()

        # Build a [B, max_len] mask over completion-token positions.
        # Token at position t in `padded` is a completion token iff
        #   len(question_ids[i]) <= t < len(full_ids[i])
        # We score the LOG-PROB of token at position t, which is computed
        # from logits[t-1]; the cross-entropy below handles that shift.
        positions = torch.arange(max_len, device=device).unsqueeze(0)  # [1, T]
        q_lens = torch.tensor(
            [len(q) for q in question_ids_per], dtype=torch.long, device=device,
        ).unsqueeze(1)  # [B, 1]
        full_lens = torch.tensor(
            [len(f) for f in full_ids_per], dtype=torch.long, device=device,
        ).unsqueeze(1)  # [B, 1]
        completion_mask = ((positions >= q_lens) & (positions < full_lens)).float()

        # Use cross_entropy(reduction='none') for fused log_softmax + gather.
        # Predict token at position t from logits[t-1]; ignore the very first
        # token (position 0) which has no predictor.
        target = padded[:, 1:].contiguous()                    # [B, T-1]
        pred_logits = logits[:, :-1, :].contiguous()           # [B, T-1, V]
        token_lp = -torch.nn.functional.cross_entropy(
            pred_logits.view(-1, pred_logits.size(-1)),
            target.view(-1),
            reduction="none",
        ).view(B, max_len - 1)                                  # [B, T-1]

        # Align mask with the [T-1] predicted-token axis: a token at
        # position t (1-indexed in padded) is predicted by logits[t-1].
        comp_mask_shifted = completion_mask[:, 1:]              # [B, T-1]
        masked_lp = token_lp * comp_mask_shifted
        denom = comp_mask_shifted.sum(dim=1).clamp(min=1.0)     # [B]
        mean_lp = masked_lp.sum(dim=1) / denom                  # [B]

        if self.normalize:
            # Sigmoid normalization: shift so typical log-probs (-3..-1)
            # land in 0.05-0.73 — same scaling as `score()`.
            scores = torch.sigmoid(mean_lp + 2.0)
        else:
            scores = mean_lp

        return scores.detach().cpu().tolist()


class _RewardFnWrapper:
    """Stateful reward function that uses SelfJudgeRewardModel for scoring.

    Tracks an internal index counter that auto-increments on each call.
    Call set_questions() before each rollout batch to provide question context.
    """

    def __init__(self, judge: SelfJudgeRewardModel, deterministic_fn=None, weight: float = 1.0):
        """
        Args:
            judge: SelfJudgeRewardModel instance for self-judge scoring.
            deterministic_fn: If provided, blend deterministic and self-judge scores.
            weight: Weight for self-judge score in combined mode (0-1).
                    deterministic gets (1 - weight).
        """
        self._judge = judge
        self._deterministic_fn = deterministic_fn
        self._weight = weight
        self._questions: list[str] = []
        self._idx: int = 0

    def set_questions(self, questions: list[str]) -> None:
        """Set the question context for the current rollout batch and reset index."""
        self._questions = questions
        self._idx = 0

    def __call__(self, completion: str, ground_truth: str) -> float:
        """Score a completion, indexing into questions for self-judge context."""
        question = self._questions[self._idx % len(self._questions)]
        self_judge_score = self._judge.score(question, completion)

        if self._deterministic_fn is not None:
            det_score = self._deterministic_fn(completion, ground_truth)
            result = (1.0 - self._weight) * det_score + self._weight * self_judge_score
        else:
            result = self_judge_score

        self._idx += 1
        return float(result)


def make_reward_fn(config, reference_model=None, tokenizer=None):
    """Factory that returns (reward_fn, diagnostic_fn) based on config.reward_mode.

    Modes:
        "deterministic" -> (gsm8k_reward, None)
        "self_judge"    -> (_RewardFnWrapper, gsm8k_reward)
        "combined"      -> (_RewardFnWrapper with blending, gsm8k_reward)

    Args:
        config: PPOConfig with reward_mode, self_judge_weight, self_judge_normalize.
        reference_model: Frozen reference model (required for self_judge/combined).
        tokenizer: Tokenizer for the reference model.

    Returns:
        (reward_fn, diagnostic_fn) tuple.
    """
    mode = config.reward_mode

    if mode == "deterministic":
        return gsm8k_reward, None

    if reference_model is None:
        raise ValueError(
            "reference_model is required for reward_mode "
            f"'{mode}'. Pass a frozen model."
        )

    judge = SelfJudgeRewardModel(
        reference_model, tokenizer, normalize=config.self_judge_normalize
    )

    if mode == "self_judge":
        wrapper = _RewardFnWrapper(judge, deterministic_fn=None, weight=0.0)
        return wrapper, gsm8k_reward

    if mode == "combined":
        wrapper = _RewardFnWrapper(
            judge,
            deterministic_fn=gsm8k_reward,
            weight=config.self_judge_weight,
        )
        return wrapper, gsm8k_reward

    raise ValueError(f"Unknown reward_mode: '{mode}'")


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
