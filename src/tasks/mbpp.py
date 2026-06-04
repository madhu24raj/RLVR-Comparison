"""
MBPP loader (training data for the code task).

MBPP ("Mostly Basic Python Problems") ships ~974 verifiable problems across
splits (full config: train≈374, test=500, validation=90, prompt=10). Each item
has an NL `text`, a reference `code`, a `test_list` of bare `assert` statements,
and optional `test_setup_code`. There is no `entry_point` column, so we derive
it from the asserts (the function the tests call), falling back to the reference
code's first top-level def.

Emits the unified schema (columns `question`, `ground_truth`) shared with GSM8K
and HumanEval; `ground_truth` is the JSON payload from src/tasks/code.py with
style="mbpp" (bare-assert harness).
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional

from src.tasks import code as code_task

_ASSERT_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _entry_point_from_tests(test_list: List[str], code: str) -> str:
    """Best-effort function name: the first identifier called in the asserts.

    MBPP asserts look like `assert similar_elements((3,4),(4,5)) == (4,)`.
    We take the first call expression in the first assert. Falls back to the
    first top-level def in the reference code.
    """
    for test in test_list:
        # Drop the leading "assert " so we match the function under test, not
        # the assert keyword.
        body = test.strip()
        if body.startswith("assert"):
            body = body[len("assert"):]
        m = _ASSERT_CALL_RE.search(body)
        if m:
            return m.group(1)
    # Fallback: first top-level def in the reference solution.
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
    except SyntaxError:
        pass
    return "solution"


def _build_question(text: str, test_list: List[str]) -> str:
    """Prompt text: the NL description plus a couple of sample asserts so the
    model knows the exact function name and signature to produce (standard MBPP
    prompting convention)."""
    sample = "\n".join(test_list[:3])
    return (
        f"{text.strip()}\n\n"
        f"Your code should define a function that passes these tests:\n{sample}"
    )


def load_mbpp(split: str = "train", n_samples: Optional[int] = None, seed: int = 42):
    """Load MBPP (full config) into the unified schema."""
    from datasets import load_dataset  # lazy: only .load() needs `datasets`

    # Canonical namespaced repo id (bare "mbpp" is a legacy script name that
    # newer datasets/huggingface_hub reject). "full" config = ~974 problems.
    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)

    def _to_unified(ex):
        test_list = ex["test_list"]
        code = ex.get("code", "") or ""
        entry_point = _entry_point_from_tests(test_list, code)
        return {
            "question": _build_question(ex["text"], test_list),
            "ground_truth": code_task.encode_payload(
                entry_point=entry_point,
                test="\n".join(test_list),
                prompt_prefix="",
                style="mbpp",
                test_setup=ex.get("test_setup_code", "") or "",
            ),
        }

    ds = ds.map(_to_unified, remove_columns=ds.column_names)

    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    return ds
