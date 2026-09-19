import argparse
from pathlib import Path

import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wrappers import SuccessWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC/PPO on PandaPush-v3")
    parser.add_argument(
        "--env-type", type=str, default="source", choices=["source", "target"]
    )
    parser.add_argument("--sampling-strategy", type=str, choices=["udr", "adr"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    TIMESTEPS = 1_000_000
    N_ENVS = 8

    sac_hyperparams = {
        "learning_rate": 0.001,
        "buffer_size": 1_000_000,
        "batch_size": 1024,
        "train_freq": 64,
        "gradient_steps": 64,
        "ent_coef": "auto_0.1",
        "gamma": 0.95,
        "tau": 0.005,
        "policy_kwargs": {"net_arch": [256, 256, 256]},
    }

    log_dir = Path("logs/part2")
    model_dir = Path("models/part2")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    exp_name = f"sac_{args.env_type}_{args.sampling_strategy}_s{args.seed}"
    print(f"Starting SAC Training: {exp_name}")

    def make_custom_env(**kwargs):
        e = gym.make("PandaPush-v3", **kwargs)
        # TODO: add domain randomization here
        e = SuccessWrapper(e, hold_steps=10)
        return e

    train_env = make_vec_env(
        make_custom_env,
        n_envs=N_ENVS,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"type": args.env_type, "reward_type": "dense"},
    )

    eval_env = make_vec_env(
        make_custom_env,
        n_envs=1,
        seed=args.seed,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"type": args.env_type, "reward_type": "dense"},
    )

    model = SAC(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        device="cpu",
        tensorboard_log=str(log_dir / "tensorboard"),
        **sac_hyperparams,
    )

    eval_freq = max(10_000 // N_ENVS, 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / exp_name),
        log_path=str(log_dir / exp_name),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, progress_bar=True)

    print(f"Best model saved in {model_dir / exp_name / 'best_model.zip'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
