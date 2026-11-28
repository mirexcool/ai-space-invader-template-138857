from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TemporalFrameStack(gym.Wrapper):

    def __init__(self, env, frames=8, stride=5):
        super().__init__(env)

        self.frames = frames
        self.stride = stride
        self.history_size = (frames - 1) * stride + 1

        self.history = deque(maxlen=self.history_size)

        obs_size = env.observation_space.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size * frames,),
            dtype=np.float32,
        )

    def _stack(self):
        indices = range(
            0, self.history_size, self.stride
        )

        return np.concatenate(
            [self.history[i] for i in indices]
        ).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.history.clear()

        for _ in range(self.history_size):
            self.history.append(obs.copy())

        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        self.history.append(obs.copy())

        return (
            self._stack(),
            reward,
            terminated,
            truncated,
            info,
        )
