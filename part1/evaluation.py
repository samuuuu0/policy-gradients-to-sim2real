import argparse
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from agent import make_agent
from policy import Actor, Critic


def parse_args():
    parser = argparse.ArgumentParser(description="Hopper Evaluation")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--algo", type=str, choices=["reinforce", "actor-critic"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate():
    args = parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Unable to find {model_path.resolve()}")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    env = gym.make("Hopper-v4", render_mode="human" if args.render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_space=state_dim, action_space=action_dim)

    state_dict = torch.load(model_path, map_location="cpu")
    actor.load_state_dict(state_dict)

    actor.eval()

    critic = Critic(state_space=state_dim) if args.algo == "actor-critic" else None

    agent = make_agent(args.algo, actor=actor, critic=critic, device="cpu")

    print("=" * 65)
    print(f"Evaluation | Model: {model_path.name} | Algorithm: {args.algo}")
    print(f"Environment: Hopper-v4 | Episodes: {args.episodes} | Render: {args.render}")
    print("=" * 65)

    rewards = []

    with torch.no_grad():
        for ep in range(1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + ep)
            done = False
            episode_reward = 0.0

            while not done:
                action, _ = agent.get_action(state, evaluation=True)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)
            print(
                f"Episode {ep:2d}/{args.episodes} | Total reward: {episode_reward:7.2f}"
            )

    env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print("-" * 65)
    print("Final Results:")
    print(f"Medium Reward : {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Min / Max    : {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print("-" * 65)


if __name__ == "__main__":
    evaluate()
