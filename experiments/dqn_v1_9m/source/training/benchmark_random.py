import argparse
import csv
import time
from pathlib import Path

import numpy as np

from training.environment import SpaceInvadersEnv

parser = argparse.ArgumentParser()
parser.add_argument("--difficulty", default="classic",
                    choices=["classic", "progressive", "hardcore"])
parser.add_argument("--games", type=int, default=1000)
parser.add_argument("--dodge", action="store_true")
parser.add_argument("--seed-start", type=int, default=0)
args = parser.parse_args()

env = SpaceInvadersEnv(
    difficulty=args.difficulty,
    use_dodge=args.dodge,
)

results = []
start = time.perf_counter()

for seed in range(args.seed_start, args.seed_start + args.games):
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    total_reward = 0
    dodge_count = 0

    while True:
        action = env.action_space.sample()

        if action == 4:
            dodge_count += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    row = {
        "seed": seed,
        "score": info["score"],
        "waves_cleared": info["wave"] - 1,
        "lives_lost": 3 - info["lives"],
        "steps": info["steps"],
        "survival_sec": info["steps"] / 30,
        "reward": total_reward,
        "dodge_count": dodge_count,
        "survived": not terminated,
    }

    results.append(row)

    print(
        f"Seed: {seed:4d} | "
        f"Score: {row['score']:5d} | "
        f"Waves: {row['waves_cleared']:3d} | "
        f"Survival: {row['survival_sec']:.2f}s | "
        f"Dodge: {dodge_count}"
    )

Path("results").mkdir(exist_ok=True)

name = "random_dodge" if args.dodge else "random"
difficulty = args.difficulty

if difficulty != "classic":
    difficulty += "_easy"

path = f"results/{name}_{difficulty}_{args.games}games.csv"

with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

scores = [r["score"] for r in results]

print("\n===== SUMMARY =====")
print("Difficulty:", args.difficulty)
print("DODGE:", args.dodge)
print("Games:", len(results))
print("Mean score:", np.mean(scores))
print("Median score:", np.median(scores))
print("Min score:", min(scores))
print("Max score:", max(scores))
print("Std score:", np.std(scores, ddof=1))
print("Mean waves:", np.mean([r["waves_cleared"] for r in results]))
print("Mean survival:", np.mean([r["survival_sec"] for r in results]))
print("Mean reward:", np.mean([r["reward"] for r in results]))
print("Mean dodge count:", np.mean([r["dodge_count"] for r in results]))
print("Survived (%):", 100 * np.mean([r["survived"] for r in results]))
print("Execution time:", round(time.perf_counter() - start, 2))
print("Saved:", path)

env.close()
