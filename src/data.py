"""
GSM8K data loading and preprocessing for RLVR experiments.

Loads the dataset, extracts ground-truth answers, and provides
a configurable subset for quick prototyping (default: 100 prompts).
"""

from datasets import load_dataset


def extract_answer(answer_text: str) -> str:
    """Extract the final numerical answer from a GSM8K answer string.

    GSM8K answers end with '#### <number>'. This extracts that number.
    """
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip()


def load_gsm8k(split: str = "train", n_samples: int = None, seed: int = 42):
    """Load GSM8K dataset with optional subset selection.

    Args:
        split: 'train' or 'test'
        n_samples: If set, randomly sample this many prompts (default: all)
        seed: Random seed for reproducible subset selection

    Returns:
        Dataset with columns: question, answer, ground_truth
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    # Add extracted ground-truth answer as a column
    ds = ds.map(lambda x: {"ground_truth": extract_answer(x["answer"])})

    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))

    return ds


def format_prompt(question: str, system_prompt: str = None) -> str:
    """Format a GSM8K question as a plain-text prompt.

    Used by callers that don't have a tokenizer with apply_chat_template,
    or that want a plain-text format. Kept consistent with
    format_prompt_with_template's default system prompt so the two
    produce identical fallback strings.
    """
    if system_prompt is None:
        # Same default system prompt as format_prompt_with_template -- see
        # the comment there for the empirical justification (Qwen2.5-0.5B
        # ignores #### but emits \boxed{} unprompted).
        system_prompt = (
            "Solve this math problem step by step. "
            "Put your final numerical answer in \\boxed{}."
        )
    return f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"


def format_prompt_with_template(question: str, tokenizer=None, system_prompt: str = None) -> str:
    """Format a GSM8K question using the model's chat template.

    Falls back to plain text format if tokenizer is None or has no chat template.
    """
    if system_prompt is None:
        # We ask for \boxed{} (LaTeX) rather than #### because measured on
        # untrained Qwen2.5-0.5B-Instruct (2026-04-08, 20 GSM8K test prompts):
        #   - "Put your answer after ####": 0/20 emitted ####, 3/20 emitted
        #     \boxed{} unprompted, parse rate 15%, reward 15%.
        #   - "Put your answer in \boxed{}":  0/20 emitted ####, 9/20 emitted
        #     \boxed{}, parse rate 45%, reward 25%.
        # The model has a strong prior toward LaTeX from its math fine-tuning
        # data. Our reward parser accepts both formats, so requesting the
        # format the model already wants to produce is a pure win.
        system_prompt = (
            "Solve this math problem step by step. "
            "Put your final numerical answer in \\boxed{}."
        )

    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass  # fall back to plain format

    return f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"


def get_experiment_subset(n: int = 100, seed: int = 42):
    """Get the standard 100-prompt subset used across all experiments.

    Returns both train subset and full test set.
    """
    train = load_gsm8k("train", n_samples=n, seed=seed)
    test = load_gsm8k("test")  # Full test set for evaluation
    return train, test


if __name__ == "__main__":
    # Quick sanity check
    train, test = get_experiment_subset(100)
    print(f"Train subset: {len(train)} prompts")
    print(f"Test set: {len(test)} prompts")
    print(f"\nExample prompt:\n{format_prompt(train[0]['question'])}")
    print(f"\nGround truth: {train[0]['ground_truth']}")
