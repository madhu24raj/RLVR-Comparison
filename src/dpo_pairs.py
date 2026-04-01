"""
DPO preference pair construction from verifiable rewards.

For RLVR tasks like GSM8K, we have scalar rewards (0 or 1) for each
completion. DPO needs preference pairs (o+, o-). This module constructs
those pairs by pairing correct completions with incorrect ones.

This is a key design choice highlighted in the project paper (Section 2.2.3):
DPO requires paired comparisons, not individual scalar rewards. For verifiable
tasks, pairs must be constructed synthetically.
"""

import random
from dataclasses import dataclass


@dataclass
class PreferencePair:
    prompt: str
    chosen: str      # o+ (correct completion)
    rejected: str    # o- (incorrect completion)


def construct_pairs_from_batch(
    prompts: list[str],
    completions: list[str],
    rewards: list[float],
    strategy: str = "random",
    seed: int = None,
) -> list[PreferencePair]:
    """Construct DPO preference pairs from a batch of scored completions.

    For each prompt, pairs each correct completion (reward=1) with an
    incorrect completion (reward=0) from the same prompt.

    Args:
        prompts: List of prompt strings
        completions: List of model completions
        rewards: List of binary rewards (0.0 or 1.0)
        strategy: How to pair. Options:
            - "random": Pair each correct with a random incorrect (default)
            - "all": Generate all possible correct-incorrect pairs
        seed: Random seed for reproducibility

    Returns:
        List of PreferencePair objects ready for DPO training
    """
    if seed is not None:
        random.seed(seed)

    # Group completions by prompt
    prompt_groups: dict[str, dict[str, list[str]]] = {}
    for prompt, completion, reward in zip(prompts, completions, rewards):
        if prompt not in prompt_groups:
            prompt_groups[prompt] = {"correct": [], "incorrect": []}
        if reward > 0.5:
            prompt_groups[prompt]["correct"].append(completion)
        else:
            prompt_groups[prompt]["incorrect"].append(completion)

    pairs = []
    for prompt, groups in prompt_groups.items():
        correct = groups["correct"]
        incorrect = groups["incorrect"]

        # Need at least one of each to form a pair
        if not correct or not incorrect:
            continue

        if strategy == "all":
            # All possible pairs (can be large: |correct| * |incorrect|)
            for c in correct:
                for i in incorrect:
                    pairs.append(PreferencePair(prompt=prompt, chosen=c, rejected=i))

        elif strategy == "random":
            # Each correct paired with one random incorrect
            for c in correct:
                i = random.choice(incorrect)
                pairs.append(PreferencePair(prompt=prompt, chosen=c, rejected=i))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return pairs


def pairs_to_dataset(pairs: list[PreferencePair]) -> dict[str, list[str]]:
    """Convert PreferencePair list to a dict format for TRL's DPOTrainer.

    Returns dict with keys: prompt, chosen, rejected
    """
    return {
        "prompt": [p.prompt for p in pairs],
        "chosen": [p.chosen for p in pairs],
        "rejected": [p.rejected for p in pairs],
    }


if __name__ == "__main__":
    # Example usage
    prompts = ["What is 2+2?"] * 4
    completions = [
        "2+2=4\n#### 4",
        "2+2=5\n#### 5",
        "Let me think... 4!\n#### 4",
        "The answer is 3\n#### 3",
    ]
    rewards = [1.0, 0.0, 1.0, 0.0]

    pairs = construct_pairs_from_batch(prompts, completions, rewards, seed=42)
    print(f"Generated {len(pairs)} preference pairs:")
    for p in pairs:
        print(f"  Chosen: {p.chosen[:40]}...")
        print(f"  Rejected: {p.rejected[:40]}...")
        print()
