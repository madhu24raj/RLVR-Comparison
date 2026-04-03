"""
Experiment 2.8: PPO Critic Architecture Sweep

Runs PPO with different critic sizes (none, small, medium, large) and
compares advantage estimation error against GRPO's group-based baseline.

For each critic configuration, runs 3 seeds and logs:
  - Test accuracy at each step
  - Reward mean/variance (training stability)
  - Policy and value loss
  - KL divergence

Usage:
    python scripts/train_exp_2_8.py
    python scripts/train_exp_2_8.py --critic_config medium --seed 42  # Single run
"""

import argparse
import yaml
from pathlib import Path

from src.ppo_trainer import run_ppo, evaluate_on_test


def load_config(config_path: str = "configs/ppo.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single(config: dict, critic_config: str, seed: int):
    """Run a single PPO training with the given critic and seed."""
    experiment_name = f"exp_2_8_ppo_{critic_config}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"{'='*60}\n")

    logger = run_ppo(
        model_name=config["model"]["name"],
        critic_config=critic_config,
        n_prompts=config["data"]["n_prompts"],
        n_steps=config["training"]["n_steps"],
        batch_size=config["training"]["batch_size"],
        mini_batch_size=config["training"]["mini_batch_size"],
        max_new_tokens=config["training"]["max_new_tokens"],
        learning_rate=config["training"]["learning_rate"],
        seed=seed,
        experiment_name=experiment_name,
    )

    # Evaluate on test set
    model_path = f"models/{experiment_name}"
    test_acc = evaluate_on_test(
        model_path=model_path,
        max_new_tokens=config["training"]["max_new_tokens"],
        seed=seed,
    )

    return logger, test_acc


def run_full_sweep(config: dict):
    """Run the full E2.8 critic sweep: all configs x all seeds."""
    critic_configs = config["exp_2_8"]["critic_configs"]
    seeds = config["exp_2_8"]["seeds"]

    results = {}

    for critic_config in critic_configs:
        results[critic_config] = []
        for seed in seeds:
            logger, test_acc = run_single(config, critic_config, seed)
            results[critic_config].append({
                "seed": seed,
                "test_accuracy": test_acc,
                "log": logger.log,
            })

    # Print summary
    print(f"\n{'='*60}")
    print("E2.8 RESULTS SUMMARY")
    print(f"{'='*60}")
    for critic_config in critic_configs:
        accs = [r["test_accuracy"] for r in results[critic_config]]
        mean_acc = sum(accs) / len(accs)
        print(f"  {critic_config:>8s}: {mean_acc:.1%} (seeds: {[f'{a:.1%}' for a in accs]})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2.8: PPO Critic Architecture Sweep")
    parser.add_argument("--config", default="configs/ppo.yaml", help="Config file path")
    parser.add_argument("--critic_config", default=None, help="Run single critic config (none/small/medium/large)")
    parser.add_argument("--seed", type=int, default=None, help="Run single seed")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.critic_config and args.seed is not None:
        # Single run
        run_single(config, args.critic_config, args.seed)
    elif args.critic_config:
        # All seeds for one critic config
        for seed in config["exp_2_8"]["seeds"]:
            run_single(config, args.critic_config, seed)
    else:
        # Full sweep
        run_full_sweep(config)
