"""
Task abstraction for RLVR experiments.

A `TaskSpec` bundles everything that is task-specific behind one uniform
interface so the PPO / GRPO / DPO trainers can stay task-agnostic:

    load(split, n_samples, seed) -> HF Dataset with columns `question`, `ground_truth`
    format_prompt(example, tokenizer) -> str
    reward(completion, ground_truth) -> float         (the UNIFORM reward contract)
    diagnostics(completion) -> dict                    (cheap, text-only health checks)

`ground_truth` is whatever the reward fn needs:
  - GSM8K: a bare numeric string ("7").
  - Code tasks: a JSON-encoded payload (entry_point / test harness / prompt prefix),
    decoded only inside the code reward fn. See src/tasks/code.py.

The registry is intentionally LAZY: each task module is imported only when its
task is requested. This keeps the code task importable in environments without
torch (src/rewards.py imports torch at module top, and the GSM8K task wraps it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TaskSpec:
    """Task-specific behavior behind a uniform interface.

    Attributes:
        name: Registry key, e.g. "gsm8k" or "humaneval".
        eval_split: The split name used for evaluation (passed to `load`).
        load: (split, n_samples=None, seed=42) -> HF Dataset with columns
            `question` and `ground_truth`.
        format_prompt: (example, tokenizer) -> prompt string.
        reward: (completion, ground_truth) -> float. The uniform reward contract
            shared by every trainer.
        diagnostics: (completion) -> dict of cheap, text-only health checks
            (booleans/floats) that the generic ExperimentLogger aggregates into rates.
    """

    name: str
    eval_split: str
    load: Callable[..., Any]
    format_prompt: Callable[[Dict, Any], str]
    reward: Callable[[str, str], float]
    diagnostics: Callable[[str], Dict[str, Any]]


# ── Lazy registry ───────────────────────────────────────────────────────────
# Map task name -> zero-arg factory. Factories import their module lazily so a
# caller that only wants the code task never imports the GSM8K stack (torch).

def _build_gsm8k() -> TaskSpec:
    from src.tasks.gsm8k import build_gsm8k_task
    return build_gsm8k_task()


def _build_humaneval() -> TaskSpec:
    from src.tasks.humaneval import build_humaneval_task
    return build_humaneval_task()


_REGISTRY: Dict[str, Callable[[], TaskSpec]] = {
    "gsm8k": _build_gsm8k,
    "humaneval": _build_humaneval,
}


def get_task(name: str) -> TaskSpec:
    """Return the TaskSpec for `name`. Raises ValueError on unknown task."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown task '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return factory()


def available_tasks() -> List[str]:
    """Sorted list of registered task names."""
    return sorted(_REGISTRY)
