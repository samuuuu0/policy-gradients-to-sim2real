import argparse
from pathlib import Path

import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC/PPO on PandaPush-v3")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"])
    parser.add_argument(
        "--env-type", type=str, default="source", choices=["source", "target"]
    )
    parser.add_argument("--sampling-strategy", type=str, choices=["none", "udr", "adr"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    log_dir = Path("logs/part2")
    model_dir = Path("models/part2")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    exp_name = f"{args.algo}_{args.env_type}_{args.sampling_strategy}_s{args.seed}"
    print(f"Starting Training: {exp_name}")

    n_envs = 8
    env = make_vec_env(
        "PandaPush-v3",
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"type": args.env_type, "reward_type": "dense"},
    )

    # env = gym.make("PandaPush-v3", type=args.env_type, reward_type="dense")
    # env = Monitor(env)

    # TODO: domain randomization wrapper (task 6)

    eval_env = gym.make("PandaPush-v3", type=args.env_type, reward_type="dense")
    eval_env = Monitor(eval_env)

    if args.algo == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1, seed=args.seed, device="cpu")
    elif args.algo == "sac":
        model = SAC("MultiInputPolicy", env, verbose=1, seed=args.seed, device="cpu")
    else:
        raise ValueError(f"Algorithm not found {args.algo}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / exp_name),
        log_path=str(log_dir / exp_name),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    print(
        f"Training finished. Best model saved in {model_dir / exp_name / 'best_model.zip'}"
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
