import gymnasium as gym


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
