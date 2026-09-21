import gymnasium as gym
import numpy as np


class DomainRandWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        mass_range: tuple = (1.0, 5.0),
        mode: str = "none",
        adr_threshold: float = 0.8,
        adr_step: float = 0.2,
        adr_window: int = 20,
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range

        self.mass_min_limit, self.mass_max_limit = mass_range

        self.mass_min = self.mass_min_limit
        if self.mode == "adr":
            self.mass_max = self.mass_min_limit
        else:
            self.mass_max = self.mass_max_limit

        self.last_sample_type = "fixed"

        # ADR setup
        if self.mode == "adr":
            self.threshold = adr_threshold
            self.window = adr_window
            self.step_size = adr_step
            self.success_buffer = np.zeros(self.window, dtype=float)
            self.buffer_ptr = 0
            self.buffer_full = False

    def _sample_mass(self):
        if self.mode == "none":
            self.last_sample_type = "fixed"
            return None
        elif self.mode in ["adr", "udr"]:
            self.last_sample_type = "uniform"
            return np.random.uniform(self.mass_min, self.mass_max)
        else:
            raise ValueError(f"Sampling strategy {self.mode} is not implemented.")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if done and self.mode == "adr":
            is_success = info.get("is_success", False)

            self.success_buffer[self.buffer_ptr] = is_success
            self.buffer_ptr += 1

            if self.buffer_ptr >= self.window:
                self.buffer_ptr = 0
                self.buffer_full = True

            if self.buffer_full:
                rate = self.success_buffer.mean()
                if rate > self.threshold:
                    old_mass = self.mass_max
                    self.mass_max = min(
                        self.mass_max + self.step_size, self.mass_max_limit
                    )

                    if self.mass_max > old_mass:
                        self.success_buffer.fill(0.0)
                        self.buffer_ptr = 0
                        self.buffer_full = False

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        new_mass = self._sample_mass()

        if new_mass is not None:
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

        return super().reset(**kwargs)


class SuccessWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, hold_steps: int = 10):
        super().__init__(env)
        self.hold_steps = hold_steps
        self.current_hold_streak = 0

    def reset(self, **kwargs):
        self.current_hold_streak = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        is_naive_success = info.get("is_success", False)

        if is_naive_success:
            self.current_hold_streak += 1
        else:
            self.current_hold_streak = 0

        actual_success = bool(self.current_hold_streak >= self.hold_steps)
        info["is_success"] = actual_success

        if terminated and not actual_success:
            terminated = False

        if actual_success:
            terminated = True

        return obs, reward, terminated, truncated, info
