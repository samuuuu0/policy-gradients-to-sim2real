from abc import ABC, abstractmethod

import numpy as np
import torch
from policy import Actor


class Agent(ABC):
    def __init__(
        self,
        actor: Actor,
        device: str = "cpu",
        actor_lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.device = device
        self.actor = actor.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma

        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.action_log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.next_states: list[torch.Tensor] = []
        self.dones: list[bool] = []

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

    def store_outcome(
        self,
        state: np.ndarray,
        action_log_prob: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.states.append(torch.from_numpy(state).float().to(self.device))
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.next_states.append(torch.from_numpy(next_state).float().to(self.device))
        self.dones.append(done)

    def get_action(
        self, state: np.ndarray, evaluation: bool = False
    ) -> tuple[np.ndarray, torch.Tensor | None]:
        state_t = torch.from_numpy(state).float().to(self.device)
        dist = self.actor(state_t)

        if evaluation:
            action = dist.mean
            log_prob = None
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        action_np = action.detach().cpu().numpy()
        return action_np, log_prob

    @abstractmethod
    def update_polict(self) -> dict[str, float]:
        pass
