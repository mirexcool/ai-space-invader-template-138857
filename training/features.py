import numpy as np

MAX_ALIENS = 15
MAX_LASERS = 3
OBS_SIZE = 4 + MAX_ALIENS * 4 + MAX_LASERS * 3


def encode_state(state: dict) -> np.ndarray:
    width = state["width"]
    height = state["height"]

    can_shoot = any(
        action["action"] == "shoot"
        for action in state["valid_actions"]
    )

    # Player features
    features = [
        state["player_x"] / width,
        state["lives"] / 3.0,
        min(state["wave"] / 100.0, 1.0),
        float(can_shoot),
    ]

    # Alien features
    aliens = {
        alien["index"]: alien
        for alien in state["aliens"]
        if alien["active"]
    }

    for index in range(MAX_ALIENS):
        alien = aliens.get(index)

        if alien is None:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([
                alien["x"] / width,
                alien["y"] / height,
                float(alien["is_diving"]),
                1.0,
            ])

    # Laser features
    lasers = sorted(
        (laser for laser in state["lasers"]
         if laser["active"]),
        key=lambda laser: laser["y"]
    )

    for index in range(MAX_LASERS):
        if index >= len(lasers):
            features.extend([0.0, 0.0, 0.0])
        else:
            laser = lasers[index]

            features.extend([
                laser["x"] / width,
                laser["y"] / height,
                1.0,
            ])

    return np.asarray(features, dtype=np.float32)