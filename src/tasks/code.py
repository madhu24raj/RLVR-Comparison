"""
Shared machinery for code-generation tasks (HumanEval, MBPP).

Responsibilities:
  - prompt formatting (instruct the model to complete a Python function)
  - completion sanitizing (strip markdown fences / prose, keep the code)
  - the ground_truth JSON payload codec (entry_point + test harness + prefix)
  - program assembly for the two test styles:
        "humaneval": prompt prefix + completion + `test` (defines check) + check(entry_point)
        "mbpp":      completion + test_setup + assert statements
  - code_reward(completion, ground_truth) -> 1.0 / 0.0  (canonical pass@1)
  - code_diagnostics(completion) -> cheap text-only health checks

Pure-ish module: imports only stdlib + src.tasks.code_exec (also stdlib). It does
NOT import torch / datasets, so it stays testable in a minimal environment.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Optional

from src.tasks.code_exec import check_correctness, DEFAULT_TIMEOUT

CODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the function below. "
    "Write only Python code: output the full function definition and nothing "
    "else — no explanations, no markdown fences, no example usage."
)


# ── ground_truth payload codec ───────────────────────────────────────────────
# For code tasks the uniform `ground_truth` string carries a JSON payload so it
# can flow through the (completion, ground_truth) reward contract and through a
# TRL/Arrow dataset column unchanged.

def encode_payload(
    entry_point: str,
    test: str,
    prompt_prefix: str = "",
    style: str = "humaneval",
    test_setup: str = "",
) -> str:
    return json.dumps(
        {
            "entry_point": entry_point,
            "test": test,
            "prompt_prefix": prompt_prefix,
            "style": style,
            "test_setup": test_setup,
        }
    )


def decode_payload(ground_truth: str) -> Dict[str, Any]:
    return json.loads(ground_truth)


# ── prompt formatting ─────────────────────────────────────────────────────────

def format_prompt(example: Dict, tokenizer: Any = None) -> str:
    """Format a code-task example as a prompt.

    `example["question"]` holds the user-facing problem text (a function
    signature + docstring for HumanEval; an NL description + sample asserts for
    MBPP — assembled by the respective loaders). Uses the model chat template
    when available, mirroring src.data.format_prompt_with_template.
    """
    question = example["question"]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"{CODE_SYSTEM_PROMPT}\n\n{question}\n"


# ── completion sanitizing ─────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def sanitize_completion(completion: str) -> str:
    """Extract runnable Python from a raw model completion.

    Handles the common chat-model habits:
      - fenced ```python ... ``` blocks (take the first fenced block)
      - leading prose before the code
      - trailing prose / example usage after the function
    Best-effort: returns the original text if no better extraction is found.
    """
    if completion is None:
        return ""

    text = completion

    # 1) If there are markdown fences, prefer the first fenced block.
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip("\n")

    # 2) Strip a dangling opening fence with no close (```python\n...).
    stripped = re.sub(r"^\s*```(?:python|py)?\s*\n", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"\n```\s*$", "", stripped)

    return stripped.strip("\n")


def _defines(code: str, name: str) -> bool:
    """True if `code` parses and defines a top-level function `name`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in tree.body
    )


# ── program assembly ──────────────────────────────────────────────────────────

def assemble_program(payload: Dict[str, Any], completion: str) -> str:
    """Build a self-running program: candidate code + test harness.

    Two styles:
      humaneval: the canonical protocol. The HF `prompt` (signature+docstring)
        is the prefix; the model's completion is the body. We re-prepend the
        prefix unless the completion already redefines `entry_point` (chat models
        often emit the whole function). Then append `test` (defines check) and a
        trailing `check(entry_point)` call.
      mbpp: the completion must define the function itself (the prompt gives the
        signature via sample asserts). Append test_setup + the assert list.
    """
    entry_point = payload["entry_point"]
    style = payload.get("style", "humaneval")
    candidate = sanitize_completion(completion)

    if style == "humaneval":
        prefix = payload.get("prompt_prefix", "")
        if _defines(candidate, entry_point):
            # Model emitted a full definition — use it as-is.
            body = candidate
        else:
            # Model emitted only the body (continuation of the signature).
            body = prefix + candidate
        test = payload["test"]
        return f"{body}\n\n{test}\n\ncheck({entry_point})\n"

    # mbpp style: bare asserts.
    test_setup = payload.get("test_setup", "")
    test = payload["test"]  # newline-joined assert statements
    parts = [candidate]
    if test_setup:
        parts.append(test_setup)
    parts.append(test)
    return "\n\n".join(parts) + "\n"


# ── reward + diagnostics ──────────────────────────────────────────────────────

def code_reward(
    completion: str, ground_truth: str, timeout: float = DEFAULT_TIMEOUT
) -> float:
    """Canonical pass@1 reward: 1.0 if the assembled program passes its tests."""
    payload = decode_payload(ground_truth)
    program = assemble_program(payload, completion)
    result = check_correctness(program, payload.get("entry_point"), timeout=timeout)
    return 1.0 if result["passed"] else 0.0


def code_diagnostics(completion: str) -> Dict[str, Any]:
    """Cheap, text-only health checks (no execution)."""
    code = sanitize_completion(completion)
    try:
        ast.parse(code)
        syntax_ok = True
    except SyntaxError:
        syntax_ok = False
    has_def = bool(re.search(r"(?m)^\s*def\s+\w+\s*\(", code))
    return {"syntax_ok": syntax_ok, "has_def": has_def}
