import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN

from training.environment import SpaceInvadersEnv
from training.frame_stack import TemporalFrameStack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=1000000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/dqn_benchmark"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    root = Path("models/dqn_progressive_f8_s5_dodge1_seed42")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    models = sorted(
        (root / "checkpoints").glob("*.zip"),
        key=lambda p: int(p.name.split("_")[1])
    )

    models += [
        root / "best/best_model.zip",
        root / "final.zip"
    ]

    if args.model is not None:
        if not args.model.is_file():
            raise FileNotFoundError(args.model)
        models = [args.model]

    env = TemporalFrameStack(
        SpaceInvadersEnv(
            difficulty="progressive",
            use_dodge=True
        ),
        frames=8,
        stride=5
    )

    summary = []

    for path in models:
        print(f"\n===== {path.stem} =====", flush=True)

        model = DQN.load(path, device="cpu")
        rows = []

        for i in range(args.games):
            seed = args.seed_start + i
            obs, info = env.reset(seed=seed)

            reward_total = 0.0
            actions = np.zeros(5, dtype=int)

            while True:
                action, _ = model.predict(
                    obs, deterministic=True
                )
                action = int(action.item())
                actions[action] += 1

                obs, reward, terminated, truncated, info = env.step(action)
                reward_total += reward

                if terminated or truncated:
                    break

            rows.append({
                "seed": seed,
                "score": info["score"],
                "waves_cleared": info["wave"] - 1,
                "lives_lost": 3 - info["lives"],
                "steps": info["steps"],
                "survival_sec": info["steps"] / 30,
                "reward": reward_total,
                "wait": actions[0],
                "west": actions[1],
                "east": actions[2],
                "shoot": actions[3],
                "dodge": actions[4],
                "survived": int(truncated)
            })

            if (i + 1) % 100 == 0:
                print(
                    f"{i+1}/{args.games} | "
                    f"Mean score: "
                    f"{np.mean([r['score'] for r in rows]):.2f}",
                    flush=True
                )

        with open(output / f"{path.stem}.csv", "w") as f:
            writer = csv.DictWriter(
                f, fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

        scores = np.array([r["score"] for r in rows])

        result = {
            "model": path.stem,
            "games": args.games,
            "mean_score": scores.mean(),
            "median_score": np.median(scores),
            "std_score": scores.std(),
            "min_score": scores.min(),
            "max_score": scores.max(),
            "mean_waves": np.mean(
                [r["waves_cleared"] for r in rows]
            ),
            "mean_survival": np.mean(
                [r["survival_sec"] for r in rows]
            ),
            "mean_reward": np.mean(
                [r["reward"] for r in rows]
            )
        }

        total_steps = sum(r["steps"] for r in rows)

        for action in ["wait", "west", "east", "shoot", "dodge"]:
            result[f"{action}_pct"] = (
                100 * sum(r[action] for r in rows) / total_steps
            )

        summary.append(result)

        for key, value in result.items():
            print(f"{key}: {value}")

        with open(output / "summary.csv", "w") as f:
            writer = csv.DictWriter(
                f, fieldnames=summary[0].keys()
            )
            writer.writeheader()
            writer.writerows(summary)

    env.close()


if __name__ == "__main__":
    main()
