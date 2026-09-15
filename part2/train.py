import argparse
from pathlib import Path
import yaml

import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from wrappers import SuccessWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC/PPO on PandaPush-v3")
    parser.add_argument("--config", type=str, default="../configs/part2_sac.yaml")
    parser.add_argument("--env-type", type=str, choices=["source", "target"])
    parser.add_argument("--sampling-strategy", type=str, choices=["none", "udr", "adr"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    log_dir = Path("logs/part2")
    model_dir = Path("models/part2")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    exp_name = f"{config['algo']}_{args.env_type}_{args.sampling_strategy}_s{args.seed}"
    print(f"Starting Training: {exp_name}")

    def make_env(**kwargs):
        e = gym.make("PandaPush-v3", **kwargs)
        e = SuccessWrapper(e, hold_steps=10)
        return e

    n_envs = config.get("n_envs", 8)
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"type": args.env_type, "reward_type": "dense"},
    )

    eval_env = gym.make("PandaPush-v3", type=args.env_type, reward_type="dense")
    eval_env = SuccessWrapper(eval_env, hold_steps=10)
    eval_env = Monitor(eval_env)

    hyperparams = config.get("hyperparams", {})

    if config['algo'] == "ppo":
        model = PPO("MultiInputPolicy", env, verbose=1, seed=args.seed, device="cpu")
    elif config['algo'] == "sac":
        model = SAC("MultiInputPolicy", env, verbose=1, seed=args.seed, device="cpu", **hyperparams)
    else:
        raise ValueError(f"Algorithm not found {config['algo']}")

    eval_freq = max(10_000 // n_envs, 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / exp_name),
        log_path=str(log_dir / exp_name),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=config["timesteps"], callback=eval_callback)

    print(f"Best model saved in {model_dir / exp_name / 'best_model.zip'}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
