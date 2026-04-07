"""
Modular critic (value) network for PPO.

The critic estimates V(s) — how good a given state (prompt + partial completion)
is. PPO uses this to compute advantages: A = R - V(s), which tells the policy
how much better (or worse) an action was compared to expectation.

For E2.8, we sweep over different critic sizes to measure how critic capacity
affects advantage estimation accuracy and downstream task performance.
"""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """MLP value head that attaches to a language model's hidden states.

    Takes the last hidden state from the base model and maps it to a
    scalar value estimate through configurable hidden layers.

    Args:
        input_dim: Dimension of the base model's hidden states
            (e.g., 3584 for Qwen2.5-7B, 4096 for Llama-3-8B)
        hidden_sizes: List of hidden layer dimensions.
            [] = linear (no hidden layers)
            [64] = small critic
            [128, 64] = medium critic
            [256, 128, 64] = large critic
        dropout: Dropout rate between layers
    """

    def __init__(
        self,
        input_dim: int = 3584,  # Qwen2.5-7B hidden size
        hidden_sizes: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final projection to scalar value
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute value estimates from hidden states.

        Args:
            hidden_states: Shape (batch_size, hidden_dim) — typically the
                last token's hidden state from the base model.

        Returns:
            Shape (batch_size,) — scalar value estimate per example.
        """
        return self.network(hidden_states).squeeze(-1)


# Predefined critic configurations for E2.8 sweep
CRITIC_CONFIGS = {
    "none": [],              # No hidden layers — linear projection only
    "small": [64],           # 1 hidden layer
    "medium": [128, 64],     # 2 hidden layers (default)
    "large": [256, 128, 64], # 3 hidden layers
}


def build_critic(config_name: str = "medium", input_dim: int = 3584, dropout: float = 0.1) -> ValueHead:
    """Build a critic network from a named configuration.

    Args:
        config_name: One of 'none', 'small', 'medium', 'large'
        input_dim: Base model hidden dimension
        dropout: Dropout rate

    Returns:
        ValueHead module
    """
    if config_name not in CRITIC_CONFIGS:
        raise ValueError(f"Unknown critic config: {config_name}. Choose from {list(CRITIC_CONFIGS.keys())}")

    hidden_sizes = CRITIC_CONFIGS[config_name]
    return ValueHead(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=dropout)


if __name__ == "__main__":
    # Test all critic configurations
    print("Critic configurations:\n")
    for name, hidden_sizes in CRITIC_CONFIGS.items():
        critic = build_critic(name)
        n_params = sum(p.numel() for p in critic.parameters())
        print(f"  {name:>8s}: hidden_sizes={str(hidden_sizes):>18s}  params={n_params:,}")

    # Test forward pass
    print("\nForward pass test:")
    critic = build_critic("medium")
    dummy_input = torch.randn(4, 3584)  # batch of 4, Qwen hidden dim
    values = critic(dummy_input)
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {values.shape}")
    print(f"  Values: {values.tolist()}")
