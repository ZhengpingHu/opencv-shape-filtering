# dqn_train.py

import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from env_yolo_kalman import FeatureEnv
from gymnasium import spaces
import numpy as np

class DiscreteFeatureWrapper(gym.Env):
    """
    将 FeatureEnv 封装为标准 Gym Env（符合 Stable-Baselines3 要求）
    """
    def __init__(self, model_path, fps=5.0, gravity=-3.5, render=False):
        super().__init__()
        self.env = FeatureEnv(model_path, "DQN-Env", fps, gravity, launch_env=render)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        state = self.env.reset()
        return state, {}

    def step(self, action):
        state, reward, done = self.env.step(action)
        return state, reward, done, False, {}

    def render(self):
        pass

    def close(self):
        self.env.close()

if __name__ == "__main__":
    env = DiscreteFeatureWrapper("best.pt", fps=5.0, gravity=-3.5, render=False)
    check_env(env)

    vec_env = DummyVecEnv([lambda: env])

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500
    )

    model.learn(total_timesteps=200_000)
    model.save("trained/dqn_feature_env")
