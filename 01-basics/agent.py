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
    def update_policy(self) -> dict[str, float]:
        pass


class REINFORCE(Agent):
    def __init__(
        self,
        actor: Actor,
        baseline: float = 0.0,
        use_normalization: bool = False,
        **kwargs,
    ):
        super().__init__(actor, **kwargs)
        self.baseline = baseline
        self.use_normalization = use_normalization

    def _discounted_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        discounted_r = torch.zeros_like(rewards)
        running_add = 0.0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def update_policy(self) -> dict[str, float]:
        log_probs = torch.stack(self.action_log_probs)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)

        discounted_returns = self._discounted_rewards(rewards)
        returns_to_use = discounted_returns - self.baseline

        if self.use_normalization:
            returns_to_use = (returns_to_use - returns_to_use.mean()) / (
                returns_to_use.std + 1e-8
            )

        loss = -torch.sum(log_probs * returns_to_use)

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.clear_memory()
        return {"actor_loss": loss.item()}
