"""
GSM8K task spec.

This is a thin WRAPPER around the existing src/data.py and src/rewards.py.
No logic is moved or duplicated, so GSM8K behavior is byte-identical to the
pre-refactor code path:

    load           = src.data.load_gsm8k
    format_prompt  = src.data.format_prompt_with_template (on example["question"])
    reward         = src.rewards.gsm8k_reward
    diagnostics    = parse-success / boxed-format checks from src.rewards
"""

from __future__ import annotations

from typing import Any, Dict

from src.data import load_gsm8k, format_prompt_with_template
from src.rewards import (
    gsm8k_reward,
    extract_answer_from_completion,
    matches_boxed_format,
)
from src.tasks.base import TaskSpec


def _format_prompt(example: Dict, tokenizer: Any) -> str:
    return format_prompt_with_template(example["question"], tokenizer)


def _diagnostics(completion: str) -> Dict[str, Any]:
    return {
        "parse_success": extract_answer_from_completion(completion) is not None,
        "format_match_boxed": matches_boxed_format(completion),
    }


def build_gsm8k_task() -> TaskSpec:
    return TaskSpec(
        name="gsm8k",
        eval_split="test",
        load=load_gsm8k,
        format_prompt=_format_prompt,
        reward=gsm8k_reward,
        diagnostics=_diagnostics,
    )
