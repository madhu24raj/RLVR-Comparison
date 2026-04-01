"""
Evaluation metrics for RLVR experiments.

Implements the measurements specified in the project paper:
- Final test accuracy at fixed rollout budget
- Training stability (reward variance across iterations)
- Convergence speed (accuracy vs rollout count)
- Advantage estimation error |Â - A_MC| vs Monte Carlo ground truth
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt


def accuracy(rewards: list[float]) -> float:
    """Fraction of completions that received reward=1."""
    if not rewards:
        return 0.0
    return sum(r > 0.5 for r in rewards) / len(rewards)


def reward_variance(rewards_per_step: list[list[float]]) -> list[float]:
    """Variance of reward at each training step (training stability)."""
    return [float(np.var(rewards)) for rewards in rewards_per_step]


def advantage_estimation_error(
    estimated_advantages: np.ndarray,
    mc_advantages: np.ndarray,
) -> float:
    """Mean absolute error between estimated and MC ground-truth advantages.

    This is measurement (iv) from E2.7:
        |Â - A_MC| where A_MC is estimated by 1000 Monte Carlo rollouts.
    """
    return float(np.mean(np.abs(estimated_advantages - mc_advantages)))


def compute_mc_advantage(
    rewards_per_prompt: dict[str, list[float]],
) -> dict[str, float]:
    """Compute Monte Carlo ground-truth advantage for each (prompt, completion).

    For each prompt, the MC advantage of completion i is:
        A_MC_i = r_i - mean(all rewards for this prompt)

    In practice, we estimate this with many rollouts (ideally 1000 per prompt).
    """
    mc_advantages = {}
    for prompt, rewards in rewards_per_prompt.items():
        mean_reward = np.mean(rewards)
        mc_advantages[prompt] = mean_reward  # The baseline
    return mc_advantages


# --- Logging ---

class ExperimentLogger:
    """Simple JSON logger for tracking metrics across training."""

    def __init__(self, experiment_name: str, output_dir: str = "results"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log: list[dict] = []

    def log_step(self, step: int, **metrics):
        """Log metrics for a training step."""
        entry = {"step": step, **metrics}
        self.log.append(entry)

    def save(self):
        """Save log to JSON file."""
        path = self.output_dir / f"{self.experiment_name}.json"
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"Saved {len(self.log)} entries to {path}")

    def load(self, path: str = None):
        """Load log from JSON file."""
        if path is None:
            path = self.output_dir / f"{self.experiment_name}.json"
        with open(path) as f:
            self.log = json.load(f)
        return self.log


# --- Plotting ---

def plot_convergence(
    logs: dict[str, list[dict]],
    metric: str = "accuracy",
    title: str = "Convergence: Accuracy vs Rollout Count",
    save_path: str = None,
):
    """Plot convergence curves for multiple methods.

    Args:
        logs: Dict mapping method name -> list of step dicts
        metric: Which metric to plot (must be a key in step dicts)
        title: Plot title
        save_path: If set, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"PPO": "#e74c3c", "GRPO": "#3498db", "DPO": "#2ecc71"}

    for method, steps in logs.items():
        x = [s["step"] for s in steps if metric in s]
        y = [s[metric] for s in steps if metric in s]
        color = colors.get(method, None)
        ax.plot(x, y, label=method, linewidth=2, color=color)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_advantage_error(
    errors: dict[str, list[float]],
    group_sizes=None,
    save_path: str = None,
):
    """Plot advantage estimation error for PPO vs GRPO.

    For E2.8: shows PPO's irreducible critic bias vs GRPO's O(1/sqrt(G)) decay.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if "PPO" in errors:
        # PPO should show a flat bias floor
        for label, errs in errors.items():
            if label.startswith("PPO"):
                ax.axhline(y=np.mean(errs), linestyle="--", label=label, alpha=0.8)

    if "GRPO" in errors and group_sizes:
        # GRPO should decay as O(1/sqrt(G))
        ax.plot(group_sizes, errors["GRPO"], "o-", label="GRPO", linewidth=2, color="#3498db")

        # Theoretical O(1/sqrt(G)) reference line
        scale = errors["GRPO"][0] * np.sqrt(group_sizes[0])
        theoretical = [scale / np.sqrt(g) for g in group_sizes]
        ax.plot(group_sizes, theoretical, "--", label="O(1/√G) reference", alpha=0.5, color="gray")

    ax.set_xlabel("Group Size G", fontsize=12)
    ax.set_ylabel("Advantage Estimation Error", fontsize=12)
    ax.set_title("Advantage Estimation: PPO Bias vs GRPO Group Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Quick test
    logger = ExperimentLogger("test_run")
    for i in range(10):
        logger.log_step(i, accuracy=0.3 + i * 0.05, reward_mean=0.3 + i * 0.04)
    logger.save()
    print("Logger test passed.")
