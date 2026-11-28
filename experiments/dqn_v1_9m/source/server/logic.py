__author__ = "Mário Antunes"
__version__ = "1.0.0"
__email__ = "mario.antunes@ua.pt"
__status__ = "Development"

import math
from typing import Any

import numpy as np

# Difficulty tuning (all values grow with the wave number and the score).
BASE_DIVE_SPEED = 11.0
MAX_DIVE_SPEED = 18.0
BASE_DIVE_COOLDOWN = 2.2
MIN_DIVE_COOLDOWN = 0.8
MAX_SIMULTANEOUS_DIVERS = 4

# Progressive attack patterns (unlocked cumulatively by wave).
# P0 Classic homing: tracks player above y=3, then locks target.
# P1 Late homing: tracks player until y=2.0.
# P2 Zigzag: alternating lateral velocity during descent.
# P3 Sine wave: smooth oscillating lateral velocity.
# P4 Feint: moves away from target mid-dive, then returns.
# P5 Accelerating dive: vertical speed increases with dive progress.
# P6 Laser dodge: reacts to approaching lasers with lateral evasion.
# P7 Adaptive hunter: late homing, oscillation, acceleration and evasion.
ATTACK_PATTERNS = (
    "classic_homing", "late_homing", "zigzag", "sine_wave",
    "feint", "accelerating_dive", "laser_dodge", "adaptive_hunter",
)

# Unlock waves for progressive mode; classic and hardcore override these.
PATTERN_UNLOCK_WAVES = (1, 3, 5, 7, 10, 13, 16, 20)
DIFFICULTY_MODES = ("classic", "progressive", "hardcore")

class Laser:
    def __init__(self, x: float, y: float, speed: float = 8.0):
        self.x = x
        self.y = y
        self.speed = speed
        self.active = True

    def update(self, dt: float):
        self.y += self.speed * dt

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "active": self.active
        }


class Alien:
    def __init__(self, index: int, base_x: float, base_y: float):
        self.index = index
        self.base_x = base_x
        self.base_y = base_y
        
        # Desynchronized horizontal figure-8 amplitudes, frequency and phase
        self.amp_x = 0.8 + 0.1 * (index % 3)
        self.amp_y = 0.5 + 0.05 * (index % 2)
        self.freq = 1.0 + 0.15 * (index % 4)
        self.phase = (2.0 * math.pi / 10.0) * index
        
        # Dynamic float positions
        self.x = base_x
        self.y = base_y
        
        # Diving state variables
        self.is_diving = False
        self.target_x = 0.0
        self.dive_speed = 5.0
        self.pattern = "classic_homing"
        self.dive_start_y = base_y
        self.dive_elapsed = 0.0
        
        self.active = True

    def update(self, dt: float, time_elapsed: float, base_shift_x: float = 0.0, player_x: float | None = None, difficulty_scale: float = 1.0, lasers=(), width: int = 11):
        if not self.active:
            return

        if not self.is_diving:
            # Figure-8 parametric motion + global shift
            self.x = (self.base_x + base_shift_x) + self.amp_x * math.sin(self.freq * difficulty_scale * time_elapsed + self.phase)
            self.y = self.base_y + self.amp_y * math.sin(2.0 * self.freq * difficulty_scale * time_elapsed + self.phase)
        else:
            self.dive_elapsed += dt
            pattern = self.pattern
            progress = max(0.0, min(1.0, 1.0 - self.y / max(0.1, self.dive_start_y)))
            side = 1.0 if self.index % 2 == 0 else -1.0

            # Classic homing remains identical to the original physics.
            lock_y = 2.0 if pattern in ("late_homing", "adaptive_hunter") else 3.0
            if player_x is not None and self.y > lock_y:
                self.target_x = player_x

            speed = self.dive_speed
            if pattern in ("accelerating_dive", "adaptive_hunter"):
                speed = min(MAX_DIVE_SPEED, speed * (0.85 + 0.30 * progress))

            vy = -speed
            time_to_ground = max(0.1, self.y / speed)
            vx = (self.target_x - self.x) / time_to_ground

            # Lateral maneuvers fade near ground to allow final interception.
            envelope = min(1.0, max(0.0, self.y / 2.5))
            if pattern == "zigzag":
                direction = 1.0 if int(self.dive_elapsed / 0.30) % 2 == 0 else -1.0
                vx += side * direction * speed * 0.40 * envelope
            elif pattern in ("sine_wave", "adaptive_hunter"):
                vx += side * speed * 0.35 * math.sin(2 * math.pi * 1.5 * self.dive_elapsed) * envelope
            elif pattern == "feint" and 0.35 <= progress < 0.55:
                vx -= (side if abs(self.target_x - self.x) < 0.1 else math.copysign(1.0, self.target_x - self.x)) * speed * 0.45 * envelope

            if pattern in ("laser_dodge", "adaptive_hunter") and self.y > 2.5:
                # React only to an active laser below the alien and approaching it.
                threats = [l for l in lasers if l.active and 0 <= self.y - l.y <= 2.0 and abs(l.x - self.x) < 0.8]
                if threats:
                    threat = min(threats, key=lambda l: self.y - l.y)
                    dodge = side if abs(self.x - threat.x) < 0.05 else math.copysign(1.0, self.x - threat.x)
                    if self.x + dodge * speed * dt < 0 or self.x + dodge * speed * dt > width - 1:
                        dodge = -dodge
                    vx += dodge * speed * 0.60 * envelope

            max_vx = speed * 1.5
            vx = max(-max_vx, min(max_vx, vx))
            self.x += vx * dt
            if pattern != "classic_homing":
                self.x = max(0.0, min(width - 1.0, self.x))
            self.y += vy * dt

    def start_dive(self, player_x: float, speed: float = 8.0, pattern: str = "classic_homing"):
        self.is_diving = True
        self.target_x = player_x
        self.dive_speed = speed
        self.pattern = pattern
        self.dive_start_y = self.y
        self.dive_elapsed = 0.0

    def reset_dive(self):
        self.is_diving = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "is_diving": self.is_diving,
            "active": self.active
        }

class SpaceInvaders:
    def __init__(self, width: int = 11, height: int = 11, fps: int = 30, seed: int | None = None, difficulty: str = "classic"):
        """Creates a game.

        Args:
            width: Number of columns.
            height: Number of rows.
            fps: Frames per second used by the server loop.
            seed: Seed of the numpy random generator. ``None`` draws the seed
                from the operating system entropy, so every game differs. With
                a seed, every reset replays the same alien dive sequence.
        """
        if difficulty not in DIFFICULTY_MODES:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        self.difficulty = difficulty
        self.width = width
        self.height = height
        self.fps = fps
        self.seed = seed
        self.high_score = 0
        self.reset_game()

    def available_patterns(self) -> tuple[str, ...]:
        if self.difficulty == "classic":
            return ATTACK_PATTERNS[:1]
        if self.difficulty == "hardcore":
            return ATTACK_PATTERNS
        return tuple(p for p, wave in zip(ATTACK_PATTERNS, PATTERN_UNLOCK_WAVES)
                     if self.wave >= wave)

    def reset_game(self):
        self.rng = np.random.default_rng(self.seed)
        self.wave = 1
        self.player_x: float = float(self.width // 2)
        self.player_y: float = 0.0
        self.lives = 3
        self.score = 0
        self.checkpoint_score = 0
        self.game_over = False
        
        self.aliens: list[Alien] = []
        self.lasers: list[Laser] = []
        
        self.time_elapsed = 0.0
        self.dive_cooldown = BASE_DIVE_COOLDOWN
        self.time_since_last_dive = 0.0
        
        self.shoot_cooldown = 0.5
        self.time_since_last_shoot = self.shoot_cooldown
        
        self._init_aliens()

    def _init_aliens(self):
        self.aliens = []
        
        # 5 aliens per row; a third row joins from wave 3 on.
        # Columns centered nicely: 2.0, 3.5, 5.0, 6.5, 8.0
        columns = [2.0, 3.5, 5.0, 6.5, 8.0]
        rows = [8.0, 9.0] if self.wave < 3 else [7.0, 8.0, 9.0]
        
        index = 0
        for y_base in rows:
            for x_base in columns:
                self.aliens.append(Alien(index, x_base, y_base))
                index += 1

    def dive_speed(self) -> float:
        """Speed of newly diving aliens: grows with score and wave."""
        return min(MAX_DIVE_SPEED, BASE_DIVE_SPEED + 0.01 * self.score + 0.5 * (self.wave - 1))

    def dive_cooldown_now(self) -> float:
        """Seconds between dives: shrinks with score and wave."""
        return max(MIN_DIVE_COOLDOWN, self.dive_cooldown - 0.005 * self.score - 0.2 * (self.wave - 1))

    def max_divers(self) -> int:
        """Aliens allowed to dive at the same time: one more every two waves."""
        return min(MAX_SIMULTANEOUS_DIVERS, 1 + (self.wave - 1) // 2)

    def move_player(self, direction: str):
        if self.game_over:
            return

        if direction == "WEST":
            if self.player_x > 0.0:
                self.player_x -= 1.0
        elif direction == "EAST" and self.player_x < self.width - 1.0:
            self.player_x += 1.0

    def shoot_laser(self):
        if self.game_over:
            return

        if self.time_since_last_shoot >= self.shoot_cooldown:
            # Firing laser from the player's front
            self.lasers.append(Laser(self.player_x, 1.0, speed=8.0))
            self.time_since_last_shoot = 0.0

    def _die(self):
        self.lives -= 1
        self.score = self.checkpoint_score
        self.player_x = float(self.width // 2)
        self.lasers = []
        
        # Reset any diving alien
        for alien in self.aliens:
            alien.reset_dive()
            
        if self.lives <= 0:
            self.game_over = True

    def update(self, dt: float):
        if self.game_over:
            return

        self.time_elapsed += dt
        self.time_since_last_shoot += dt
        self.time_since_last_dive += dt

        # 1. Update Lasers
        active_lasers = []
        for laser in self.lasers:
            laser.update(dt)
            if laser.y < self.height:
                active_lasers.append(laser)
        self.lasers = active_lasers

        # 2. Update Aliens
        # Difficulty scale follows the current dive speed
        current_dive_speed = self.dive_speed()
        difficulty_scale = current_dive_speed / BASE_DIVE_SPEED
        
        sweep_offset = 1.5 * math.sin(1.2 * difficulty_scale * self.time_elapsed)
        for alien in self.aliens:
            alien.update(dt, self.time_elapsed, base_shift_x=sweep_offset, player_x=self.player_x, difficulty_scale=difficulty_scale, lasers=self.lasers, width=self.width)

        # 3. Schedule Alien Dives
        if self.time_since_last_dive >= self.dive_cooldown_now():
            divers = sum(1 for a in self.aliens if a.active and a.is_diving)
            available_aliens = [a for a in self.aliens if a.active and not a.is_diving]
            if available_aliens and divers < self.max_divers():
                diving_alien = available_aliens[int(self.rng.integers(len(available_aliens)))]
                # Avoid consuming extra RNG values in classic mode.
                patterns = self.available_patterns()
                pattern = patterns[0] if len(patterns) == 1 else patterns[int(self.rng.integers(len(patterns)))]
                diving_alien.start_dive(self.player_x, speed=current_dive_speed, pattern=pattern)
            self.time_since_last_dive = 0.0

        # 4. Check Laser-Alien Collisions (Distance < 0.8) using numpy
        if self.lasers and self.aliens:
            active_aliens = [a for a in self.aliens if a.active]
            if active_aliens:
                laser_coords = np.array([[las.x, las.y] for las in self.lasers])
                alien_coords = np.array([[a.x, a.y] for a in active_aliens])
                
                diff = laser_coords[:, np.newaxis, :] - alien_coords[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=-1))
                
                collisions = dists < 0.8
                laser_hit_idx, alien_hit_idx = np.where(collisions)
                
                hit_lasers = set()
                hit_aliens = set()
                for l_idx, a_idx in zip(laser_hit_idx, alien_hit_idx):
                    if l_idx not in hit_lasers and a_idx not in hit_aliens:
                        active_aliens[a_idx].active = False
                        active_aliens[a_idx].is_diving = False
                        self.lasers[l_idx].active = False
                        hit_lasers.add(l_idx)
                        hit_aliens.add(a_idx)
                        self.score += 10
                        self.high_score = max(self.high_score, self.score)
                self.lasers = [las for las in self.lasers if las.active]

        # 5. Check Alien-Player Collisions (Distance < 0.8) and Alien-Bottom Misses using numpy
        diving_aliens = [a for a in self.aliens if a.active and a.is_diving]
        if diving_aliens:
            alien_coords = np.array([[a.x, a.y] for a in diving_aliens])
            player_coord = np.array([self.player_x, self.player_y])
            
            dists = np.sqrt(np.sum((alien_coords - player_coord)**2, axis=-1))
            if np.any(dists < 0.8):
                self._die()
                return
                
            for alien in diving_aliens:
                if alien.y <= 0.0:
                    alien.reset_dive()
                    self.score = max(self.checkpoint_score, self.score - 10)

        # 6. Check Wave Cleared
        active_count = len([a for a in self.aliens if a.active])
        if active_count == 0:
            self.score += 100
            self.high_score = max(self.high_score, self.score)
            self.checkpoint_score = self.score
            # Respawn a harder wave
            self.wave += 1
            self._init_aliens()

    def get_state(self) -> dict[str, Any]:
        valid_actions = []
        if not self.game_over:
            if self.player_x > 0.0:
                valid_actions.append({"action": "move", "direction": "WEST"})
            if self.player_x < self.width - 1.0:
                valid_actions.append({"action": "move", "direction": "EAST"})
            if self.time_since_last_shoot >= self.shoot_cooldown:
                valid_actions.append({"action": "shoot"})

        return {
            "difficulty": self.difficulty,
            "width": self.width,
            "height": self.height,
            "player_x": self.player_x,
            "player_y": self.player_y,
            "lives": self.lives,
            "wave": self.wave,
            "score": self.score,
            "high_score": self.high_score,
            "game_over": self.game_over,
            "lasers": [las.to_dict() for las in self.lasers],
            "aliens": [a.to_dict() for a in self.aliens if a.active],
            "valid_actions": valid_actions
        }
    def get_viewer_state(self) -> dict[str, Any]:
        """Return the game state with visual attack pattern metadata.

        Includes attack patterns for rendering. Recipients depend on server routing.
        """
        state = self.get_state()

        state["aliens"] = [
            {
                **alien.to_dict(),
                "pattern": (
                    alien.pattern
                    if alien.is_diving
                    else "formation"
                )
            }
            for alien in self.aliens
            if alien.active
        ]
        return state
