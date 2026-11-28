import gymnasium as gym
import numpy as np

from gymnasium import spaces
from server.logic import SpaceInvaders
from training.features import encode_state, OBS_SIZE
from training.dodge import choose_dodge_direction

class SpaceInvadersEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        difficulty="progressive",
        max_steps=18000,
        use_dodge=True,
        shot_penalty=0.0,
        ignore_score_rollback=False,
    ):
        super().__init__()

        self.difficulty = difficulty
        self.max_steps = max_steps
        self.dt = 1.0 / 30.0

        self.shot_penalty = shot_penalty
        self.ignore_score_rollback = ignore_score_rollback
        self.use_dodge = use_dodge
        self.action_space = spaces.Discrete(
            5 if use_dodge else 4
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

        self.game = None
        self.steps = 0

    def _get_obs(self):
        return encode_state(self.game.get_state())

    def _get_info(self):
        return {
            "score": self.game.score,
            "wave": self.game.wave,
            "lives": self.game.lives,
            "steps": self.steps,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        game_seed = int(
            self.np_random.integers(0, 2**31 - 1)
        )

        self.game = SpaceInvaders(
            seed=game_seed,
            difficulty=self.difficulty,
        )

        self.steps = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        old_score = self.game.score
        old_lives = self.game.lives
        shots_fired = 0

        if action == 1:
            self.game.move_player("WEST")

        elif action == 2:
            self.game.move_player("EAST")

        elif action == 3:
            lasers_before = len(self.game.lasers)
            self.game.shoot_laser()
            shots_fired = len(self.game.lasers) - lasers_before

        elif action == 4 and self.use_dodge:
            direction = choose_dodge_direction(
                self.game.get_state()
            )

            if direction is not None:
                self.game.move_player(direction)

        self.game.update(self.dt)
        self.steps += 1

        score_delta = self.game.score - old_score
        lives_lost = old_lives - self.game.lives

        if self.ignore_score_rollback and lives_lost > 0:
            score_delta = max(0, score_delta)
        edge_distance = min(
            self.game.player_x,
            self.game.width - self.game.player_x
        ) / self.game.width

        edge_penalty = 0.1 * max(
            0.0, 1.0 - edge_distance / 0.2
        ) ** 2
        reward = (
            score_delta / 5.0
            - 10.0 * lives_lost
            - 0.005
            - self.shot_penalty * shots_fired
            - edge_penalty
        )

        terminated = bool(self.game.game_over)

        truncated = (
            self.steps >= self.max_steps
            and not terminated
        )

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(),
        )

    def close(self):
        self.game = None