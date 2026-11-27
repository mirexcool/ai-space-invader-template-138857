import asyncio
import time
import os
import csv
import argparse
import statistics
from contextlib import redirect_stdout

from server.logic import SpaceInvaders
from agents.rule_based_agent import RuleBasedAgent


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--difficulty", default="classic")
    parser.add_argument("--max-steps", type=int, default=18000)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    results = []
    total_steps = 0
    start = time.perf_counter()

    for seed in range(args.seed_start,
                      args.seed_start + args.games):

        game = SpaceInvaders(
            seed=seed,
            difficulty=args.difficulty
        )

        agent = RuleBasedAgent.__new__(RuleBasedAgent)

        # Prevent overwriting existing analysis logs.
        from unittest.mock import patch
        with patch("builtins.open",
                   side_effect=lambda *a, **kw: os.fdopen(
                       os.open(os.devnull, os.O_WRONLY), "w"
                   )):
            agent.__init__()

        agent.log = lambda *a, **kw: None
        agent.player_id = 1
        initial_lives = game.lives

        with open(os.devnull, "w") as silent:
            with redirect_stdout(silent):
                for step in range(args.max_steps):
                    agent.current_state = game.get_viewer_state()
                    agent.current_state["player_id"] = 1

                    action = await agent.deliberate()

                    if action:
                        if action["action"] == "move":
                            game.move_player(action["direction"])
                        elif action["action"] == "shoot":
                            game.shoot_laser()

                    game.update(1 / 30)
                    total_steps += 1

                    if game.game_over:
                        break

        agent.log_file.close()

        result = {
            "seed": seed,
            "score": game.score,
            "waves_cleared": game.wave - 1,
            "lives_lost": initial_lives - game.lives,
            "steps": step + 1,
            "survived": not game.game_over
        }

        results.append(result)

        print(
            f"Seed={seed:4d} | "
            f"Score={game.score:6d} | "
            f"Wave={game.wave:3d} | "
            f"Lives={game.lives} | "
            f"Survived={not game.game_over}"
        )

    elapsed = time.perf_counter() - start
    scores = [r["score"] for r in results]
    waves = [r["waves_cleared"] for r in results]

    print("\n========== SUMMARY ==========")
    print(f"Difficulty: {args.difficulty}")
    print(f"Games: {len(results)}")
    print(f"Time: {elapsed:.2f} s")
    print(f"Steps/sec: {total_steps / elapsed:,.0f}")
    print(f"Average score: {statistics.mean(scores):.2f}")
    print(f"Median score: {statistics.median(scores):.2f}")
    print(f"Min score: {min(scores)}")
    print(f"Max score: {max(scores)}")
    print(f"Average waves: {statistics.mean(waves):.2f}")
    print(f"Min waves: {min(waves)}")
    print(f"Max waves: {max(waves)}")

    for target in [10, 20, 50]:
        rate = sum(w >= target for w in waves) / len(waves)
        print(f"Cleared {target} waves: {rate:.1%}")

    survival = sum(r["survived"] for r in results)
    print(f"Survival rate: {survival / len(results):.1%}")

    print("\nWorst 5 games:")
    for r in sorted(results, key=lambda r: r["score"])[:5]:
        print(r)

    filename = f"training/rule_results_{args.difficulty}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())
