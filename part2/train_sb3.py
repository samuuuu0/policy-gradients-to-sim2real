import argparse

import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO, SAC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
    parser.add_argument("--env-type", type=str, default="source", choices=["source", "target"])
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"])
    parser.add_argument("--sampling-strategy", type=str, choices=["none", "udr", "adr"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        render_mode="human",
        type=args.env_type,
        reward_type="dense",
    )

    if args.algo == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1)
    elif args.algo == "sac":
        model = SAC("MultiInputPolicy", env, verbose=1)
    else:
        raise ValueError(f"Algorithm not found {args.algo}")

    model.learn(total_timesteps=args.timesteps)

    # TODO: add randomization wrapper here

    save_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"

    model.save(save_name)
    env.close()


if __name__ == "__main__":
    main()
