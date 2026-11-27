def choose_dodge_direction(state):
    px = state["player_x"]
    py = state["player_y"]

    threats = [
        alien
        for alien in state["aliens"]
        if alien["active"]
        and alien["is_diving"]
        and 0 <= alien["y"] - py <= 3.0
    ]

    if not threats:
        return None

    def danger(x):
        return sum(
            1.0 / (
                0.1
                + (alien["x"] - x) ** 2
                + 0.25 * (alien["y"] - py) ** 2
            )
            for alien in threats
        )

    candidates = [(None, px)]

    for action in state["valid_actions"]:
        if action["action"] != "move":
            continue

        direction = action["direction"]

        if direction == "WEST":
            candidates.append(("WEST", px - 1.0))
        elif direction == "EAST":
            candidates.append(("EAST", px + 1.0))

    direction, _ = min(
        candidates,
        key=lambda candidate: danger(candidate[1])
    )

    return direction
