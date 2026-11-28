import argparse
import csv
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--run", default=None)
args = parser.parse_args()

plt.ion()

fig, axes = plt.subplots(
    3, 2, figsize=(15, 10), sharex=True
)

axes = axes.ravel()
window = 10


def read_csv(path, skip_header=False):
    if not path.exists():
        return []

    with path.open() as f:
        if skip_header:
            next(f, None)

        return list(csv.DictReader(f))


def column(rows, name):
    return np.asarray([
        float(r[name]) for r in rows
    ])


def plot_metric(ax, x, y, title, smooth=True):
    ax.clear()
    ax.set_title(title)
    ax.grid(alpha=0.2)

    if len(y) == 0:
        return

    ax.plot(x, y, alpha=0.5, label="Raw")

    if smooth and len(y) >= window:
        avg = np.convolve(
            y,
            np.ones(window) / window,
            mode="valid",
        )

        ax.plot(
            x[window-1:], avg,
            linewidth=2,
            label="Moving average",
        )

    ax.legend(fontsize=8)


while plt.fignum_exists(fig.number):
    try:
        if args.run:
            root = Path(args.run)
        else:
            files = list(
                Path("results").rglob("adaptive.csv")
            )

            if not files:
                plt.pause(5)
                continue

            root = max(
                files,
                key=lambda p: p.stat().st_mtime
            ).parent

        path = root / "eval" / "evaluations.npz"

        if not path.exists():
            plt.pause(5)
            continue

        with np.load(path) as data:
            steps = data["timesteps"] / 1e6
            rewards = data["results"].mean(axis=1)
            survival = data["ep_lengths"].mean(axis=1) / 30

            lengths = data["ep_lengths"]

        plot_metric(
            axes[1], steps, rewards,
            "Evaluation reward"
        )

        plot_metric(
            axes[2], steps, survival,
            "Survival (seconds)"
        )

        eval_rows = read_csv(
            root / "eval_monitor.monitor.csv",
            skip_header=True,
        )

        scores = []

        if eval_rows and "score" in eval_rows[0]:
            offset = 0

            for episode_lengths in lengths:
                count = len(episode_lengths)
                batch = eval_rows[offset:offset + count]

                if len(batch) != count:
                    break

                scores.append(
                    np.mean(column(batch, "score"))
                )

                offset += count

        plot_metric(
            axes[0],
            steps[:len(scores)],
            scores,
            "Original game score",
        )

        rows = read_csv(root / "adaptive.csv")

        if rows:
            x = column(rows, "step") / 1e6
            lr = column(rows, "actual_lr")
            scheduled = column(rows, "scheduled_lr")
            epsilon = column(rows, "epsilon")
            loss = column(rows, "loss")

            axes[3].clear()

            axes[3].plot(
                x, lr,
                label="Actual optimizer LR",
                linewidth=2,
            )

            axes[3].plot(
                x, scheduled,
                "--",
                alpha=0.6,
                label="Scheduled LR",
            )

            axes[3].set_yscale("log")
            axes[3].set_title("Learning rate")
            axes[3].grid(alpha=0.2)
            axes[3].legend(fontsize=8)

            plot_metric(
                axes[4], x, epsilon,
                "Exploration rate",
                smooth=False,
            )

            plot_metric(
                axes[5], x, loss,
                "Training loss",
            )

            previous = "normal"

            for row in rows:
                mode = row["mode"]

                if mode != previous:
                    position = float(row["step"]) / 1e6

                    for ax in axes:
                        ax.axvline(
                            position,
                            linestyle=":",
                            alpha=0.4,
                        )

                    axes[3].annotate(
                        mode,
                        (position, float(row["actual_lr"])),
                        fontsize=8,
                        rotation=45,
                    )

                    previous = mode

        axes[0].axhline(
            3413.23,
            linestyle="--",
            alpha=0.5,
            label="Rule-Based Easy",
        )

        axes[0].legend(fontsize=8)

        axes[4].set_xlabel("Training steps (millions)")
        axes[5].set_xlabel("Training steps (millions)")

        fig.suptitle(root.name, fontsize=11)
        fig.tight_layout()
        fig.canvas.draw_idle()

    except (
        OSError, ValueError, EOFError,
        BadZipFile, KeyError, TypeError
    ):
        pass

    plt.pause(5)
