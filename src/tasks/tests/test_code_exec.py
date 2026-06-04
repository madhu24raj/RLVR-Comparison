"""
Offline tests for the code sandbox and code reward (no model, no datasets).

Runnable with the system interpreter:  python3 -m pytest src/tasks/tests/test_code_exec.py
These exercise only stdlib + src/tasks/{code_exec,code}.py.
"""

import os
import sys

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.tasks.code_exec import check_correctness
from src.tasks import code as code_task


# ── raw executor ──────────────────────────────────────────────────────────────

def test_correct_program_passes():
    program = (
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n"
        "    assert candidate(-1, 1) == 0\n"
        "\n"
        "check(add)\n"
    )
    r = check_correctness(program, "add", timeout=5.0)
    assert r["passed"] is True
    assert r["diagnostic"] == "passed"


def test_wrong_program_fails():
    program = (
        "def add(a, b):\n"
        "    return a - b\n"
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n"
        "check(add)\n"
    )
    r = check_correctness(program, "add", timeout=5.0)
    assert r["passed"] is False
    assert r["diagnostic"] == "failed"


def test_infinite_loop_times_out():
    program = "while True:\n    pass\n"
    r = check_correctness(program, None, timeout=2.0)
    assert r["passed"] is False
    assert r["diagnostic"] == "timeout"


def test_syntax_error_detected():
    program = "def broken(:\n    return 1\n"
    r = check_correctness(program, "broken", timeout=5.0)
    assert r["passed"] is False
    assert r["diagnostic"] == "syntax_error"


def test_runtime_exception_is_exec_error():
    program = "raise ValueError('boom')\n"
    r = check_correctness(program, None, timeout=5.0)
    assert r["passed"] is False
    assert r["diagnostic"] == "exec_error"
    assert "ValueError" in r["detail"]


def test_malicious_file_delete_is_neutralized(tmp_path):
    """A candidate that tries to delete a real file must not succeed, and the
    file must survive (os.remove is disabled inside the sandbox)."""
    victim = tmp_path / "victim.txt"
    victim.write_text("important")
    program = f"import os\nos.remove({str(victim)!r})\n"
    r = check_correctness(program, None, timeout=5.0)
    # os.remove is set to None -> calling None(...) raises TypeError (exec_error).
    assert r["passed"] is False
    assert r["diagnostic"] in ("exec_error", "failed")
    assert victim.exists(), "sandbox must not allow file deletion"
    assert victim.read_text() == "important"


# ── code_reward / payload / assembly ─────────────────────────────────────────

def _humaneval_payload():
    # Mimics a HumanEval row: prompt prefix (signature+docstring), a `test` that
    # defines check(), and an entry_point.
    prefix = (
        "def add(a, b):\n"
        '    """Return the sum of a and b."""\n'
    )
    test = (
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n"
        "    assert candidate(10, -4) == 6\n"
    )
    return code_task.encode_payload(
        entry_point="add", test=test, prompt_prefix=prefix, style="humaneval"
    )


def test_code_reward_humaneval_body_only():
    """Completion is just the body (model continues the signature)."""
    gt = _humaneval_payload()
    completion = "    return a + b\n"
    assert code_task.code_reward(completion, gt) == 1.0


def test_code_reward_humaneval_full_def_with_fence():
    """Completion redefines the whole function inside a markdown fence."""
    gt = _humaneval_payload()
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    assert code_task.code_reward(completion, gt) == 1.0


def test_code_reward_humaneval_wrong():
    gt = _humaneval_payload()
    completion = "    return a - b\n"
    assert code_task.code_reward(completion, gt) == 0.0


def test_code_reward_mbpp_style():
    """MBPP uses bare asserts and the model must define the function itself."""
    test_list = [
        "assert double(2) == 4",
        "assert double(0) == 0",
        "assert double(-3) == -6",
    ]
    gt = code_task.encode_payload(
        entry_point="double",
        test="\n".join(test_list),
        prompt_prefix="",
        style="mbpp",
        test_setup="",
    )
    good = "def double(x):\n    return x * 2\n"
    bad = "def double(x):\n    return x + 2\n"
    assert code_task.code_reward(good, gt) == 1.0
    assert code_task.code_reward(bad, gt) == 0.0


def test_sanitize_strips_fences():
    raw = "Sure! Here is the code:\n```python\ndef f():\n    return 1\n```\nHope that helps!"
    assert code_task.sanitize_completion(raw) == "def f():\n    return 1"


def test_diagnostics():
    assert code_task.code_diagnostics("def f():\n    return 1")["syntax_ok"] is True
    assert code_task.code_diagnostics("def f():\n    return 1")["has_def"] is True
    assert code_task.code_diagnostics("def broken(:\n  x")["syntax_ok"] is False
