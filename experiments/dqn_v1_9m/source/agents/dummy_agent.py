__author__ = "Mário Antunes"
__version__ = "1.0.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import asyncio
import logging
import argparse
from typing import Optional, Dict, Any

import numpy as np
from agents.base_agent import BaseAgent

class DummyAgent(BaseAgent):
    def __init__(self, *args: Any, seed: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # seed=None draws from system entropy
        self.rng = np.random.default_rng(seed)

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        if not self.current_state or self.current_state.get("game_over"):
            return None
        
        valid_actions = self.current_state.get("valid_actions", [])
        if valid_actions:
            # Randomly choose one of the valid actions or choose to do nothing (None)
            choices = [*valid_actions, None]
            action = choices[int(self.rng.integers(len(choices)))]
            if action:
                logging.info(f"Deliberated action: {action}")
            return action

        # Fallback to legacy random choice if valid_actions is not provided by server
        choice = ["move_west", "move_east", "shoot", "idle"][int(self.rng.integers(4))]
        action = None
        if choice == "move_west":
            action = {"action": "move", "direction": "WEST"}
        elif choice == "move_east":
            action = {"action": "move", "direction": "EAST"}
        elif choice == "shoot":
            action = {"action": "shoot"}
        
        if action:
            logging.info(f"Deliberated action (fallback): {action}")
        return action

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy random agent")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed (default: system random)")
    parser.add_argument("--uri", "-u", type=str, default="ws://localhost:8765/ws", help="Server WebSocket URI")
    args = parser.parse_args()
    asyncio.run(DummyAgent(server_uri=args.uri, seed=args.seed).run())
