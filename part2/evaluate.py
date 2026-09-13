import argparse
import os

import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import PPO, SAC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC/PPO on PandaPush-v3")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"])
    parser.add_argument(
        "--env-type", type=str, default="target", choices=["source", "target"]
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main(
    model_path: str,
    algo: str,
    n_episodes: int,
    deterministic: bool,
    render: bool,
    env_type: str,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model."
        )

    render_mode = "human" if render else None
    env_kwargs = {"type": env_type, "reward_type": "dense"}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = gym.make("PandaPush-v3", **env_kwargs)

    print(f"Loading {algo.upper()} model from {model_path}...")
    if algo == "ppo":
        model = PPO.load(model_path, env=env)
    elif algo == "sac":
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    episode_returns = []
    successes = []

    print(f"Starting evaluation on {env_type} environment for {n_episodes} episodes...")

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)

        episode_returns.append(episode_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        if episode & 10 == 0 or episode == n_episodes:
            print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n" + "=" * 40)
    print("=== Evaluation Summary ===")
    print("=" * 40)
    print(f"Algorithm    : {algo.upper()}")
    print(f"Tested on    : {env_type.upper()} environment")
    print(f"Episodes     : {n_episodes}")
    print(f"Mean return  : {returns.mean():.3f} +/- {returns.std():.3f}")
    print(f"Min return   : {returns.min():.3f}")
    print(f"Max return   : {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model_path,
        algo=args.algo,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
    )
