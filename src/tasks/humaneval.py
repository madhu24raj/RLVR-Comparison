"""
HumanEval task spec (code generation).

HumanEval is evaluation-only (164 problems, no train split), so the "humaneval"
task trains on MBPP and evaluates on HumanEval — the standard RLVR-for-code
setup. The TaskSpec.load dispatches:

    split == "train"  -> MBPP train  (src/tasks/mbpp.py)
    split == "test"   -> HumanEval   (the 164 canonical problems)

Both loaders emit the unified schema (columns `question`, `ground_truth`) used by
every trainer, where `ground_truth` is the JSON payload from src/tasks/code.py.
"""

from __future__ import annotations

from typing import Optional

from src.tasks.base import TaskSpec
from src.tasks import code as code_task


def load_humaneval(split: str = "test", n_samples: Optional[int] = None, seed: int = 42):
    """Load HumanEval (openai_humaneval) into the unified schema.

    Columns out: `question` (the prompt = signature + docstring) and
    `ground_truth` (JSON payload: entry_point, test, prompt_prefix, style).
    """
    from datasets import load_dataset  # lazy: only .load() needs `datasets`

    # HumanEval only ships a "test" split.
    ds = load_dataset("openai_humaneval", split="test")

    def _to_unified(ex):
        return {
            "question": ex["prompt"],
            "ground_truth": code_task.encode_payload(
                entry_point=ex["entry_point"],
                test=ex["test"],
                prompt_prefix=ex["prompt"],
                style="humaneval",
            ),
        }

    ds = ds.map(_to_unified, remove_columns=ds.column_names)

    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    return ds


def _load(split: str = "train", n_samples: Optional[int] = None, seed: int = 42):
    if split == "train":
        from src.tasks.mbpp import load_mbpp

        return load_mbpp("train", n_samples=n_samples, seed=seed)
    # eval / test split -> HumanEval
    return load_humaneval("test", n_samples=n_samples, seed=seed)


def build_humaneval_task() -> TaskSpec:
    return TaskSpec(
        name="humaneval",
        eval_split="test",
        load=_load,
        format_prompt=code_task.format_prompt,
        reward=code_task.code_reward,
        diagnostics=code_task.code_diagnostics,
    )
