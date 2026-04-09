"""
Diagnostic: prompt + reward starvation check.

Two checks in one script:
  A. Print one chat-templated GSM8K prompt so a human can verify the
     template is well-formed (system prompt present, no double BOS, etc.).
  B. Generate N completions on real GSM8K test prompts with the untrained
     model and report how often each answer-format marker (####, \\boxed{},
     "answer is") appears, plus the strict-extraction parse rate and the
     correct-answer rate.

Why this exists: PPO needs nonzero reward signal on at least some
completions, otherwise the policy gradient is zero and the model never
learns. Strict reward extraction (see specs/logic.md L13) raises this
bar by refusing to credit "the right number appeared somewhere" -- so
we want to confirm empirically that the chosen prompt + reward parser
combination produces a workable parse rate before committing to a
training run.

Usage:
    python scripts/check_prompt_and_starvation.py
    python scripts/check_prompt_and_starvation.py --n-prompts 50
    python scripts/check_prompt_and_starvation.py --model Qwen/Qwen2.5-1.5B-Instruct

History (Qwen2.5-0.5B-Instruct, 20 prompts, seed=42, T=0.7):
    "answer after ####":  parse 15%, reward 15% -- model emits 0/20 ####.
    "answer in \\boxed{}": parse 45%, reward 25% -- model emits 9/20 boxed.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Make src/ importable when running this script directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_gsm8k, format_prompt_with_template
from src.rewards import extract_answer_from_completion, gsm8k_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model id (default: Qwen2.5-0.5B-Instruct)")
    p.add_argument("--n-prompts", type=int, default=20,
                   help="Number of test prompts to evaluate (default: 20)")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Generation length (default: 256, matches PPOConfig base default)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (default: 0.7)")
    p.add_argument("--seed", type=int, default=42,
                   help="GSM8K subset seed (default: 42)")
    return p.parse_args()


def load_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return tok, model


def inspect_template(tok, sample_question: str) -> None:
    """Check A: print one templated prompt so a human can eyeball it."""
    print("=" * 70)
    print("A: TEMPLATED PROMPT INSPECTION")
    print("=" * 70)
    print(f"\n[raw question]\n{sample_question}\n")
    templated = format_prompt_with_template(sample_question, tok)
    print(f"[templated prompt -- {len(templated)} chars]")
    print("-" * 70)
    print(templated)
    print("-" * 70)
    print(f"\n[tokenized] {len(tok(templated)['input_ids'])} tokens")


def starvation_check(tok, model, prompts, ground_truths,
                     max_new_tokens: int, temperature: float) -> dict:
    """Check B: generate completions and tally format-marker hit rates."""
    print()
    print("=" * 70)
    print(f"B: STARVATION CHECK ({len(prompts)} prompts, sampled at T={temperature})")
    print("=" * 70)

    counters = {"hash": 0, "boxed": 0, "answer_is": 0, "extracted": 0, "correct": 0}

    for i, (p, gt) in enumerate(zip(prompts, ground_truths)):
        enc = tok(p, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tok.eos_token_id,
            )
        completion = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        has_hash = bool(re.search(r"####\s*-?[\d,]+", completion))
        has_boxed = bool(re.search(r"\\boxed\{", completion))
        has_answer_is = bool(re.search(r"answer\s+is", completion, re.IGNORECASE))
        extracted = extract_answer_from_completion(completion)
        reward = gsm8k_reward(completion, gt)

        counters["hash"] += has_hash
        counters["boxed"] += has_boxed
        counters["answer_is"] += has_answer_is
        counters["extracted"] += extracted is not None
        counters["correct"] += int(reward)

        print(
            f"  {i:2d} | gt={gt:>8s} "
            f"| ####={'Y' if has_hash else 'N'} "
            f"| boxed={'Y' if has_boxed else 'N'} "
            f"| ans_is={'Y' if has_answer_is else 'N'} "
            f"| extracted={str(extracted):>8s} "
            f"| reward={reward}"
        )
    return counters


def report(counters: dict, n: int) -> None:
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"#### marker present:      {counters['hash']}/{n}  ({100*counters['hash']/n:.0f}%)")
    print(f"\\boxed marker present:    {counters['boxed']}/{n}  ({100*counters['boxed']/n:.0f}%)")
    print(f"'answer is' present:      {counters['answer_is']}/{n}  ({100*counters['answer_is']/n:.0f}%)")
    print(f"Strict extraction got #:  {counters['extracted']}/{n}  ({100*counters['extracted']/n:.0f}%)")
    print(f"Correct (reward=1):       {counters['correct']}/{n}  ({100*counters['correct']/n:.0f}%)")
    print()

    parse_rate = counters["extracted"] / n
    if parse_rate < 0.10:
        print(f"STARVATION RISK: only {100*parse_rate:.0f}% of completions parse.")
        print("PPO will get near-zero gradient signal. Fix prompt or do SFT warmup.")
    elif parse_rate < 0.30:
        print(f"LOW PARSE RATE: {100*parse_rate:.0f}% of completions parse.")
        print("PPO will work but slowly. Consider strengthening the prompt.")
    else:
        print(f"OK: {100*parse_rate:.0f}% of completions parse via strict extraction.")


def main() -> None:
    args = parse_args()
    print(f"[load] model: {args.model}")
    tok, model = load_model(args.model)

    print(f"[load] GSM8K test ({args.n_prompts} prompts, seed={args.seed})")
    ds = load_gsm8k("test", n_samples=args.n_prompts, seed=args.seed)

    inspect_template(tok, ds[0]["question"])

    prompts = [format_prompt_with_template(ex["question"], tok) for ex in ds]
    ground_truths = [ex["ground_truth"] for ex in ds]
    counters = starvation_check(
        tok, model, prompts, ground_truths,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    report(counters, len(prompts))


if __name__ == "__main__":
    main()
