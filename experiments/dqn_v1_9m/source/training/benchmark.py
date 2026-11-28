import random
import time

from server.logic import SpaceInvaders

start = time.perf_counter()
total_steps = 0
scores = []

for seed in range(100):
    game = SpaceInvaders(seed=seed, difficulty="classic")

    for step in range(18000):
        state = game.get_state()
        actions = state["valid_actions"]

        if actions:
            action = random.choice(actions)

            if action["action"] == "move":
                game.move_player(action["direction"])
            elif action["action"] == "shoot":
                game.shoot_laser()

        game.update(1 / 30)
        total_steps += 1

        if game.game_over:
            break

    scores.append(game.score)

elapsed = time.perf_counter() - start

print(f"Games: {len(scores)}")
print(f"Steps: {total_steps}")
print(f"Time: {elapsed:.2f} s")
print(f"Steps/sec: {total_steps / elapsed:,.0f}")
print(f"Average score: {sum(scores) / len(scores):.2f}")
print(f"Max score: {max(scores)}")
