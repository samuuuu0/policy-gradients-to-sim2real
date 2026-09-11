from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from policy import Actor, Critic


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

    def _discount_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        discounted_r = torch.zeros_like(rewards)
        running_add = 0.0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def update_policy(self) -> dict[str, float]:
        log_probs = torch.stack(self.action_log_probs)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)

        discounted_returns = self._discount_rewards(rewards)
        returns_to_use = discounted_returns - self.baseline

        if self.use_normalization:
            returns_to_use = (returns_to_use - returns_to_use.mean()) / (
                returns_to_use.std() + 1e-8
            )

        loss = -torch.sum(log_probs * returns_to_use)

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.clear_memory()
        return {
            "actor_loss": loss.item(),
            "mean_sigma": self.actor.current_sigma
        }


class ActorCritic(Agent):
    def __init__(self, actor: Actor, critic: Critic, critic_lr: float = 2e-3, **kwargs):
        super().__init__(actor, **kwargs)

        self.critic = critic.to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def update_policy(self) -> dict[str, float]:
        states = torch.stack(self.states)
        next_states = torch.stack(self.next_states)
        log_probs = torch.stack(self.action_log_probs)

        rewards = torch.tensor(
            self.rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        dones = torch.tensor(
            self.dones, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        log_probs = log_probs.unsqueeze(1)

        values = self.critic(states)
        next_values = self.critic(next_states)

        td_targets = rewards + self.gamma * next_values * (1 - dones)
        td_targets = td_targets.detach()

        advantages = td_targets - values

        critic_loss = F.mse_loss(values, td_targets)

        actor_loss = -(log_probs * advantages.detach()).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.clear_memory()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_value": values.mean().item(),
            "mean_sigma": self.actor.current_sigma
        }


def make_agent(
    algo_name: str, actor: Actor, critic: Critic | None = None, **kwargs
) -> Agent:
    baseline = kwargs.pop("baseline", 0.0)
    use_normalization = kwargs.pop("use_normalization", False)

    critic_lr = kwargs.pop("critic_lr", 2e-3)

    if algo_name == "reinforce":
        return REINFORCE(
            actor, baseline=baseline, use_normalization=use_normalization, **kwargs
        )
    elif algo_name == "actor-critic":
        return ActorCritic(actor, critic, critic_lr, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
