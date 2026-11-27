import asyncio
import math
import json
import time
from agents.base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):

    def __init__(self, server_uri="ws://localhost:8765/ws"):
        super().__init__()

        self.previous_positions = {}
        self.previous_state = None
        self.tick = 0
        self.shot_id = 0
        self.target_id = None

        self.log_file = open(
            "agent_analysis_v5.jsonl",
            "w",
            buffering=1
        )

    def log(self, event, **data):
        record = {
            "timestamp": time.time(),
            "tick": self.tick,
            "event": event,
            **data
        }

        self.log_file.write(json.dumps(record) + "\n")

    def move(self, px, target_x, reason):

        if abs(target_x - px) <= 0.5:
            return None

        direction = "EAST" if target_x > px else "WEST"

        self.log(
            "MOVE",
            player_x=px,
            target_x=target_x,
            direction=direction,
            reason=reason
        )

        return {
            "action": "move",
            "direction": direction
        }

    async def deliberate(self):

        state = self.current_state

        if not state:
            return None

        self.tick += 1

        px = state["player_x"]
        py = state["player_y"]

        width = state.get("width", 11)
        center_x = (width - 1) / 2

        wave = state.get("wave", 1)
        score = state.get("score", 0)
        lives = state.get("lives", 0)
        lasers = state.get("lasers", [])

        self.log("STATE", state=state)

        if self.previous_state:

            old = self.previous_state

            if lives < old["lives"]:

                self.log(
                    "LIFE_LOST",
                    wave=wave,
                    score=score,
                    lives=lives,
                    last_shot_id=self.shot_id
                )

                print(
                    f"LIFE LOST | Wave={wave} "
                    f"| Score={score} | Lives={lives}",
                    flush=True
                )

                self.previous_positions.clear()
                self.target_id = None

            if score != old["score"]:
                self.log(
                    "SCORE_CHANGE",
                    old_score=old["score"],
                    new_score=score,
                    delta=score - old["score"]
                )

            if wave != old["wave"]:
                self.log(
                    "WAVE_CHANGE",
                    old_wave=old["wave"],
                    new_wave=wave
                )

                self.previous_positions.clear()
                self.target_id = None

        self.previous_state = {
            "wave": wave,
            "score": score,
            "lives": lives
        }

        if state.get("game_over"):

            self.log(
                "GAME_OVER",
                score=score,
                wave=wave,
                shots=self.shot_id
            )

            self.previous_positions.clear()
            self.target_id = None

            return None

        current_positions = {}
        targets = []

        # Parse all active enemies
        for a in state.get("aliens", []):

            if not a["active"]:
                continue

            idx = a["index"]
            x = a["x"]
            y = a["y"]

            current_positions[idx] = (x, y)

            if not a["is_diving"]:
                continue

            previous = self.previous_positions.get(idx)

            if previous is None:
                continue

            vx = x - previous[0]
            vy = y - previous[1]

            if vy >= 0:
                continue

            dy = y - py
            dx = abs(x - px)

            if dy <= 0:
                continue

            speed = math.hypot(vx, vy)

            angle = math.degrees(
                math.atan2(abs(vx), abs(vy))
            )

            ticks_to_player = dy / abs(vy)

            # Predict enemy position in two updates
            predicted_x = x + vx * 2
            predicted_y = y + vy * 2

            # Predicted position at player height
            collision_x = x + vx * ticks_to_player

            # Interception position
            intercept_x = max(
                0,
                min(width - 1, predicted_x)
            )

            target = {
                "index": idx,
                "wave": wave,
                "player_x": px,
                "enemy_x": x,
                "enemy_y": y,
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "angle": angle,
                "dx": dx,
                "dy": dy,
                "ticks": ticks_to_player,
                "predicted_x": predicted_x,
                "predicted_y": predicted_y,
                "collision_x": collision_x,
                "intercept_x": intercept_x
            }

            targets.append(target)

            self.log(
                "TARGET_EVALUATION",
                **target
            )

        self.previous_positions = current_positions

        # No attacking enemies: return to center
        if not targets:

            self.target_id = None

            return self.move(
                px,
                center_x,
                "RETURN_TO_CENTER"
            )

        # Keep tracking the same enemy if possible
        target = next(
            (
                t for t in targets
                if t["index"] == self.target_id
            ),
            None
        )

        nearest = min(
            targets,
            key=lambda t: t["ticks"]
        )

        # Switch target if another enemy is more urgent
        if (
            target is None
            or nearest["ticks"] + 2 < target["ticks"]
        ):
            target = nearest

        self.target_id = target["index"]

        angle = target["angle"]
        dx = target["dx"]
        dy = target["dy"]
        ticks = target["ticks"]

        intercept_x = target["intercept_x"]

        max_dy = min(
            3.0,
            1.5 + 0.15 * (wave - 1)
        )

        # Adaptive close-range shooting
        if angle <= 10:
            adjusted_dy = max_dy
            adjusted_dx = 1.0

        elif angle <= 20:
            adjusted_dy = min(max_dy, 2.0)
            adjusted_dx = 0.8

        elif angle <= 30:
            adjusted_dy = min(max_dy, 1.5)
            adjusted_dx = 0.6

        elif angle <= 40:
            adjusted_dy = min(max_dy, 1.1)
            adjusted_dx = 0.5

        else:
            adjusted_dy = min(max_dy, 0.9)
            adjusted_dx = 0.4

        valid_actions = state.get("valid_actions")

        shoot_allowed = (
            valid_actions is None
            or {"action": "shoot"} in valid_actions
        )

        can_shoot = (
            not lasers
            and shoot_allowed
            and 0 < dy <= adjusted_dy
            and dx <= adjusted_dx
        )

        self.log(
            "DECISION",
            target=target,
            adjusted_dx=adjusted_dx,
            adjusted_dy=adjusted_dy,
            shoot_allowed=shoot_allowed,
            can_shoot=can_shoot,
            laser_active=bool(lasers)
        )

        # Priority 1: shoot
        if can_shoot:

            self.shot_id += 1

            self.log(
                "SHOOT",
                shot_id=self.shot_id,
                target=target,
                adjusted_dx=adjusted_dx,
                adjusted_dy=adjusted_dy
            )

            print(
                f"SHOOT #{self.shot_id} "
                f"| Wave={wave} "
                f"| PlayerX={px:.3f} "
                f"| Enemy={target['index']} "
                f"| x={target['enemy_x']:.3f} "
                f"| y={target['enemy_y']:.3f} "
                f"| vx={target['vx']:.3f} "
                f"| vy={target['vy']:.3f} "
                f"| Angle={angle:.1f}° "
                f"| dx={dx:.3f} "
                f"| dy={dy:.3f} "
                f"| Ticks={ticks:.2f}",
                flush=True
            )

            return {"action": "shoot"}

        # Wait for the active laser to finish
        if lasers:
            return None

        # Priority 2: intercept approaching enemy
        if ticks > 2:

            action = self.move(
                px,
                intercept_x,
                "INTERCEPT_ENEMY"
            )

            if action:

                if (
                    valid_actions is None
                    or action in valid_actions
                ):
                    return action

        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri",
        default="ws://localhost:8765/ws"
    )
    args = parser.parse_args()

    asyncio.run(RuleBasedAgent(server_uri=args.uri).run())