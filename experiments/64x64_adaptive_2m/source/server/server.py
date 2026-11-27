__author__ = "Mário Antunes"
__version__ = "1.0.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import argparse
import logging

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aigf import interface
from . import logic


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - SPACE_INVADERS - %(levelname)s - %(message)s"
)

logging.getLogger("websockets").setLevel(logging.WARNING)
log = logging.getLogger("space_invaders")

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_FPS = 30


class SpaceInvadersGameServer(interface.GameInterface):

    def __init__(
        self,
        seed: int | None = None,
        fps: int = DEFAULT_FPS,
        difficulty: str = "classic"
    ) -> None:

        super().__init__(
            is_real_time=True,
            fps=fps,
            static_dirs=[str(VIEWER_DIR)]
        )

        self.game = logic.SpaceInvaders(
            fps=fps,
            seed=seed,
            difficulty=difficulty
        )

        self.player_id: int | None = None

        log.info(f"Difficulty mode: {difficulty}")

    async def on_player_connect(self, player_id: int) -> None:

        log.info(f"Player {player_id} connected.")

        if self.player_id is None:
            self.player_id = player_id
            self.state = interface.GameState.RUNNING
        else:
            log.warning(
                f"Extra player {player_id} connected. "
                "Only one player supported."
            )

    async def on_player_disconnect(self, player_id: int) -> None:

        log.info(f"Player {player_id} disconnected.")

        if self.player_id == player_id:
            self.player_id = None
            self.state = interface.GameState.LOBBY
            self.game.reset_game()

    async def on_start_sim(self) -> None:

        if self.player_id is not None:
            self.state = interface.GameState.RUNNING
        else:
            log.warning(
                "Cannot start simulation: no agent connected."
            )

    async def on_reset_sim(self) -> None:

        self.game.reset_game()
        self.state = interface.GameState.LOBBY

        log.info("Game reset.")

    async def process_action(
        self,
        player_id: int,
        action: dict[str, Any]
    ) -> None:

        log.debug(
            f"Processing action from {player_id}: {action}"
        )

        if (
            self.state != interface.GameState.RUNNING
            or player_id != self.player_id
        ):
            return

        act_type = action.get("action")

        if act_type == "move":

            direction = action.get("direction")

            if isinstance(direction, str):
                self.game.move_player(direction)

        elif act_type == "shoot":

            self.game.shoot_laser()

    async def tick(self, dt: float) -> None:

        if (
            self.state != interface.GameState.RUNNING
            or self.player_id is None
        ):
            return

        try:

            self.game.update(dt)

            if self.game.game_over:

                log.info(
                    f"Game Over! "
                    f"Score: {self.game.score}, "
                    f"Wave: {self.game.wave}"
                )

                self.state = interface.GameState.LOBBY

        except Exception:

            log.exception(
                "Game update failed: "
                "wave=%s, score=%s, dt=%s",
                self.game.wave,
                self.game.score,
                dt
            )

            raise

    def get_state(self) -> dict[str, Any]:

        try:

            # Include attack patterns for alien icons.
            state = self.game.get_viewer_state()

            state["player_id"] = self.player_id

            return state

        except Exception:

            log.exception("Failed to build game state")

            raise

    def get_setup_payload(self) -> dict[str, Any]:

        return {
            "width": self.game.width,
            "height": self.game.height
        }


def parse_args(
    argv: Sequence[str] | None = None
) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Space Invaders Game Server"
    )

    parser.add_argument(
        "--host",
        "-H",
        type=str,
        default=DEFAULT_HOST,
        help="Host address"
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=DEFAULT_PORT,
        help="Port to run on"
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed"
    )

    parser.add_argument(
        "--fps",
        "-f",
        type=int,
        default=DEFAULT_FPS,
        help="Simulation frames per second"
    )

    parser.add_argument(
        "--difficulty",
        choices=["classic", "progressive", "hardcore"],
        default="classic",
        help="Alien attack difficulty mode"
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:

    args = parse_args(argv)

    from aigf.main import run_app

    if args.seed is None:
        log.info("No seed given: using system random.")
    else:
        log.info(f"Using seed {args.seed}.")

    run_app(
        SpaceInvadersGameServer(
            seed=args.seed,
            fps=args.fps,
            difficulty=args.difficulty
        ),
        host=args.host,
        port=args.port
    )


if __name__ == "__main__":

    main()