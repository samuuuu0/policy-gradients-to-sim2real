import argparse
import os
import time

import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import SAC
from wrappers import SuccessWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC/PPO on PandaPush-v3")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--env-type", type=str, default="target", choices=["source", "target"]
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model file not found: {args.model_path}. "
            "Make sure you saved your trained model."
        )

    args.render_mode = "human" if args.render else None
    env_kwargs = {"type": args.env_type, "reward_type": "dense"}
    if args.render_mode is not None:
        env_kwargs["render_mode"] = args.render_mode

    env = gym.make("PandaPush-v3", **env_kwargs)
    env = SuccessWrapper(env, hold_steps=10)

    print(f"Loading SAC model from {args.model_path}")
    model = SAC.load(args.model_path, env=env)

    episode_returns = []
    successes = []

    print(
        f"Starting evaluation on {args.env_type} environment for {args.episodes} episodes..."
    )

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action, _states = model.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)

            if args.render_mode:
                time.sleep(0.02)

        episode_returns.append(episode_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        if episode % 10 == 0 or episode == args.episodes:
            print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n" + "=" * 40)
    print("=== Evaluation Summary ===")
    print("=" * 40)
    print("Algorithm    : SAC")
    print(f"Tested on    : {args.env_type.upper()} environment")
    print(f"Episodes     : {args.episodes}")
    print(f"Mean return  : {returns.mean():.3f} +/- {returns.std():.3f}")
    print(f"Min return   : {returns.min():.3f}")
    print(f"Max return   : {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
