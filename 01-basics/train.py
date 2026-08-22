import argparse
import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from agent import make_agent
from policy import Actor, Critic


def parse_args():
    parser = argparse.ArgumentParser(description="Hopper Training")
    parser.add_argument("--algo", type=str, choices=["reinforce", "actor-critic"])

    # REINFORCE parameters
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--use-norm", action="store_true")

    # Training loop parameters
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="exp_default")
    return parser.parse_args()


def set_seed(env: gym.Env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def training():
    args = parse_args()

    env = gym.make("Hopper-v4")

    set_seed(env, args.seed)

    log_dir = Path("logs")
    model_dir = Path("models")
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Network initialization
    actor = Actor(state_space=state_dim, action_space=action_dim)
    critic = Critic(state_space=state_dim)

    # Agent (factory pattern)
    agent = make_agent(
        algo_name=args.algo,
        actor=actor,
        critic=critic,
        actor_lr=args.actor_lr,
        baseline=args.baseline,
        use_normalization=args.use_norm,
    )

    csv_path = log_dir / f"{args.exp_name}.csv"

    best_reward = -float("inf")

    with csv_path.open(mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["episode", "reward", "actor_loss", "critic_loss", "mean_value"]
        )

        for ep in range(1, args.epochs + 1):
            if ep == 1:
                state, _ = env.reset(seed=args.seed)
            else:
                state, _ = env.reset()

            episode_reward = 0.0
            done = False
            while not done:
                action, log_prob = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_outcome(state, log_prob, reward, next_state, done)

                state = next_state
                episode_reward += reward

            metrics = agent.update_policy()
            actor_loss = metrics.get("actor_loss", 0.0)
            critic_loss = metrics.get("critic_loss", 0.0)
            mean_value = metrics.get("mean_value", 0.0)

            writer.writerow([ep, episode_reward, actor_loss, critic_loss, mean_value])

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(agent.actor.state_dict(), model_dir / f"{args.exp_name}.pth")

            if ep % 50 == 0:
                print(
                    f"Episode {ep:4d}/{args.epochs} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"A_Loss: {actor_loss:8.2f} | "
                    f"C_Loss: {critic_loss:8.2f} | "
                    f"Mean Value: {mean_value:8.2f}"
                )

        env.close()


if __name__ == "__main__":
    training()
