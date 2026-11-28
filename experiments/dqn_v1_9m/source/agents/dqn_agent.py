import argparse
import asyncio
from collections import deque

import numpy as np
import torch
from stable_baselines3 import DQN

from agents.base_agent import BaseAgent
from training.features import encode_state
from training.dodge import choose_dodge_direction


class DQNAgent(BaseAgent):

    def __init__(self, server_uri, model_path):
        super().__init__(server_uri)

        torch.set_num_threads(1)

        self.model = DQN.load(model_path, device="cpu")

        self.frames = 8
        self.stride = 5
        self.history_size = 36
        self.history = deque(maxlen=self.history_size)

        self.actions = {
            0: None,
            1: {"action": "move", "direction": "WEST"},
            2: {"action": "move", "direction": "EAST"},
            3: {"action": "shoot"}
        }

        print(f"Loaded model: {model_path}")
        print(f"Observation: {self.model.observation_space}")
        print(f"Actions: {self.model.action_space}")

    async def deliberate(self):

        state = self.current_state

        if state.get("game_over"):
            self.history.clear()
            return None

        obs = encode_state(state)

        if not self.history:
            for _ in range(self.history_size):
                self.history.append(obs.copy())
        else:
            self.history.append(obs.copy())

        stacked = np.concatenate([
            self.history[i]
            for i in range(0, self.history_size, self.stride)
        ]).astype(np.float32)

        action, _ = self.model.predict(
            stacked,
            deterministic=True
        )

        action = int(action.item())

        if action == 4:
            direction = choose_dodge_direction(state)
            if direction is not None:
                return {"action": "move", "direction": direction}
            return None

        return self.actions[action]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--uri",
        default="ws://localhost:8780/ws"
    )

    parser.add_argument(
        "--model",
        default=(
            "models/dqn_progressive_f8_s5_dodge0_seed42/"
            "best/best_model.zip"
        )
    )

    args = parser.parse_args()

    asyncio.run(
        DQNAgent(args.uri, args.model).run()
    )
