"""
Critic (value function) architectures for PPO – E2.8 capacity sweep.

Four capacities as specified in the assignment:
  (a) none   – no critic; REINFORCE with batch-mean baseline
  (b) small  – 2-layer MLP on the last-token hidden state
  (c) medium – single linear head (same depth as the policy's LM head)
  (d) large  – deep MLP with 2× hidden width (~2× parameter count of (c))

All critics share the same forward API:
    value: Tensor[batch] = critic(hidden_state: Tensor[batch, hidden_size])

The `is_trainable()` method lets the trainer skip critic optimisation for (a).
"""

import torch
import torch.nn as nn


# ── Individual architectures ─────────────────────────────────────────────────

class REINFORCEBaseline(nn.Module):
    """
    Capacity (a): no learned critic.

    Returns zeros so the trainer can always call critic(h) uniformly.
    The trainer replaces the zeros with the batch-mean reward as baseline.
    """
    def __init__(self):
        super().__init__()
        # A dummy non-trainable parameter keeps the module valid for
        # optimizer construction even when is_trainable() returns False.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(hidden_state.shape[0], device=hidden_state.device)

    def is_trainable(self) -> bool:
        return False


class SmallCriticMLP(nn.Module):
    """
    Capacity (b): 2-layer MLP.

    hidden_size → 256 → ReLU → 1
    """
    def __init__(self, hidden_size: int, mid_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state).squeeze(-1)

    def is_trainable(self) -> bool:
        return True


class MediumCriticHead(nn.Module):
    """
    Capacity (c): single linear projection – same depth as the LM head.

    hidden_size → 1
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_state).squeeze(-1)

    def is_trainable(self) -> bool:
        return True


class LargeCriticMLP(nn.Module):
    """
    Capacity (d): deep MLP with 2× hidden width.

    hidden_size → 2*hidden_size → GELU → 2*hidden_size → GELU → 1
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        w = hidden_size * 2
        self.net = nn.Sequential(
            nn.Linear(hidden_size, w),
            nn.GELU(),
            nn.Linear(w, w),
            nn.GELU(),
            nn.Linear(w, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state).squeeze(-1)

    def is_trainable(self) -> bool:
        return True


# ── Factory ───────────────────────────────────────────────────────────────────

def build_critic(capacity: str, hidden_size: int) -> nn.Module:
    """
    Instantiate a critic by capacity name.

    Args:
        capacity:    "none" | "small" | "medium" | "large"
        hidden_size: Hidden dimension of the backbone LM (e.g. 896 for Qwen2.5-0.5B)

    Returns:
        nn.Module with .forward(hidden_state) → Tensor[batch] and .is_trainable() → bool
    """
    capacity = capacity.strip().lower()
    if capacity == "none":
        return REINFORCEBaseline()
    elif capacity == "small":
        return SmallCriticMLP(hidden_size)
    elif capacity == "medium":
        return MediumCriticHead(hidden_size)
    elif capacity == "large":
        return LargeCriticMLP(hidden_size)
    else:
        raise ValueError(
            f"Unknown critic capacity: {capacity!r}. "
            "Choose from: none, small, medium, large"
        )


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    hidden_size = 896  # Qwen2.5-0.5B hidden dim
    batch = 4
    h = torch.randn(batch, hidden_size)

    for cap in ("none", "small", "medium", "large"):
        critic = build_critic(cap, hidden_size)
        out = critic(h)
        n_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
        print(f"{cap:8s} | output shape: {list(out.shape)} | trainable params: {n_params:,}")
