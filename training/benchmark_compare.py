import asyncio
import sys
import csv
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import DQN

from agents.rule_based_agent import RuleBasedAgent
from training.environment import SpaceInvadersEnv
from training.frame_stack import TemporalFrameStack

GAMES = 1000
SEED_START = 1_000_000
MAX_STEPS = 18_000

MODEL = (
    "models/dqn_progressive_f8_s5_dodge1_seed42"
    "_sp0.05_rb1_g0.999_lr5e-05_net64x64"
    "_quadedge_adaptive/best/best_model.zip"
)

OUT = Path("results/benchmark_rule_1000")
OUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(1)


async def benchmark(kind, model=None):
    env = TemporalFrameStack(
        SpaceInvadersEnv(
            difficulty="progressive",
            max_steps=MAX_STEPS,
            use_dodge=True,
            shot_penalty=0.05,
            ignore_score_rollback=True,
        ),
        frames=8,
        stride=5,
    )

    rows = []
    started = time.perf_counter()

    for i in range(GAMES):
        seed = SEED_START + i
        obs, info = env.reset(seed=seed)
        game = env.unwrapped.game

        if kind == "rule":
            agent = RuleBasedAgent.__new__(RuleBasedAgent)

            real_open = open
            with patch("builtins.open") as mock:
                mock.side_effect = lambda *a, **kw: real_open(
                    "/dev/null", "w"
                )
                agent.__init__("ws://localhost:8780/ws")

            agent.log = lambda *a, **kw: None
            agent.player_id = 1

        for step in range(1, MAX_STEPS + 1):
            if kind == "dqn":
                action, _ = model.predict(
                    obs, deterministic=True
                )
                obs, _, terminated, truncated, info = env.step(
                    int(action.item())
                )
                if terminated or truncated:
                    break
            else:
                agent.current_state = game.get_viewer_state()
                agent.current_state["player_id"] = 1

                action = await agent.deliberate()

                if action:
                    if action["action"] == "move":
                        game.move_player(action["direction"])
                    elif action["action"] == "shoot":
                        game.shoot_laser()

                game.update(1 / 30)

                if game.game_over:
                    break

        row = {
            "seed": seed,
            "score": game.score,
            "waves_cleared": game.wave - 1,
            "lives_lost": 3 - game.lives,
            "steps": step,
            "survival_sec": step / 30,
            "survived": int(not game.game_over),
        }

        rows.append(row)

        if (i + 1) % 100 == 0:
            print(
                f"{kind}: {i+1}/{GAMES} | "
                f"mean score="
                f"{np.mean([r['score'] for r in rows]):.2f}",
                flush=True,
            )

    with open(OUT / f"{kind}.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    env.close()

    print(
        f"{kind} completed in "
        f"{time.perf_counter()-started:.2f}s"
    )
    return rows


async def main():
    if "--rule-only" in sys.argv:
        await benchmark("rule")
        return

    model = DQN.load(MODEL, device="cpu")
    print("Model:", MODEL)
    print("Training steps:", model.num_timesteps)

    dqn = await benchmark("dqn", model)
    rule = await benchmark("rule")

    print("\n===== FINAL COMPARISON =====")
    print(f"{'Metric':25} {'DQN':>12} {'Rule':>12}")

    for name, key, func in [
        ("Mean score", "score", np.mean),
        ("Median score", "score", np.median),
        ("Min score", "score", np.min),
        ("Max score", "score", np.max),
        ("Score std", "score", np.std),
        ("Mean waves cleared", "waves_cleared", np.mean),
        ("Max waves cleared", "waves_cleared", np.max),
        ("Mean survival (sec)", "survival_sec", np.mean),
        ("Mean lives lost", "lives_lost", np.mean),
        ("Survived (%)", "survived",
         lambda x: 100 * np.mean(x)),
        ("Reached wave 10 (%)", "waves_cleared",
         lambda x: 100 * np.mean(np.array(x) >= 9)),
        ("Reached wave 20 (%)", "waves_cleared",
         lambda x: 100 * np.mean(np.array(x) >= 19)),
    ]:
        a = func([r[key] for r in dqn])
        b = func([r[key] for r in rule])
        print(f"{name:25} {a:12.2f} {b:12.2f}")

    a = np.array([r["score"] for r in dqn])
    b = np.array([r["score"] for r in rule])

    print("\n===== PAIRED COMPARISON =====")
    print(f"DQN wins: {100*np.mean(a>b):.2f}%")
    print(f"Rule wins: {100*np.mean(b>a):.2f}%")
    print(f"Draws: {100*np.mean(a==b):.2f}%")
    print(f"Mean score difference: {np.mean(a-b):.2f}")

    if b.mean() != 0:
        print(f"DQN / Rule score: {100*a.mean()/b.mean():.2f}%")

    print("Results saved to:", OUT)


if __name__ == "__main__":
    asyncio.run(main())
