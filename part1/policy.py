import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(
        self,
        state_space: int,
        action_space: int,
        hidden_dim: int = 64,
        init_sigma: float = 0.5,
    ):
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space

        self.actor_net = nn.Sequential(
            nn.Linear(self.state_space, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.action_space),
        )

        self.sigma = nn.Parameter(
            torch.full((self.action_space,), init_sigma, dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.zeros_(m.bias)

    @property
    def current_sigma(self) -> float:
        with torch.no_grad():
            return F.softplus(self.sigma).mean().item()

    def forward(self, state: torch.Tensor) -> Normal:
        action_mean = self.actor_net(state)
        sigma = F.softplus(self.sigma)
        return Normal(loc=action_mean, scale=sigma)


class Critic(nn.Module):
    def __init__(self, state_space: int, hidden_dim: int = 64):
        super().__init__()

        self.state_space = state_space

        self.critic_net = nn.Sequential(
            nn.Linear(self.state_space, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic_net(state)
