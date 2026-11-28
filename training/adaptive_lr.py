import csv
from pathlib import Path

import numpy as np
import torch
from training.safety import TrainingSafetyError

from stable_baselines3.common.callbacks import BaseCallback


class MutableSchedule:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, progress_remaining):
        return self.value


class AdaptiveLRCallback(BaseCallback):

    def __init__(
        self,
        eval_callback,
        model_dir,
        result_dir,
        base_lr=0.00005,
        warmup=3_000_000,
    ):
        super().__init__(verbose=1)

        self.eval_callback = eval_callback
        self.model_dir = Path(model_dir)
        self.result_dir = Path(result_dir)

        self.base_lr = base_lr
        self.min_lr = base_lr * 0.2
        self.max_lr = base_lr * 3
        self.boost_lr = base_lr * 5

        self.warmup = warmup
        self.patience = 500_000
        self.cooldown = 250_000

        self.lr = MutableSchedule(base_lr)
        self.eps = MutableSchedule(0.05)

        self.mode = "normal"
        self.mode_start = 0
        self.cooldown_until = 0

        self.best_reward = None
        self.last_improvement = warmup
        self.last_eval = 0
        self.history = []
        self.eps_attached = False

    def _on_training_start(self):
        self.model.lr_schedule = self.lr

        self.csv_path = self.result_dir / "adaptive.csv"

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow([
                    "step",
                    "scheduled_lr",
                    "actual_lr",
                    "epsilon",
                    "mode",
                    "loss",
                ])

        print("[ADAPTIVE] Controller attached", flush=True)

    def _backup(self, step):
        path = self.model_dir / "adaptive_backup"
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path / "model"))
        self.model.save_replay_buffer(
            str(path / "replay_buffer.pkl")
        )

        print(f"[BACKUP] step={step:,}", flush=True)

    def _start_mode(self, mode, step):
        self._backup(step)

        self.mode = mode
        self.mode_start = step

        if mode == "boost":
            self.lr.value = self.boost_lr
            self.eps.value = 0.15
        else:
            self.lr.value = self.min_lr
            self.eps.value = 0.05

        print(
            f"[ADAPTIVE] {mode.upper()} "
            f"step={step:,} lr={self.lr.value:.7f}",
            flush=True,
        )

    def _finish_mode(self, step):
        self.mode = "normal"
        self.lr.value = self.base_lr
        self.eps.value = 0.05

        self.cooldown_until = step + self.cooldown
        self.last_improvement = step

        print(
            f"[ADAPTIVE] NORMAL step={step:,}",
            flush=True,
        )

    def _update_mode(self, step):
        elapsed = step - self.mode_start

        if self.mode == "boost":
            if elapsed < 100_000:
                self.lr.value = self.boost_lr
                self.eps.value = 0.15

            elif elapsed < 250_000:
                p = (elapsed - 100_000) / 150_000

                self.lr.value = (
                    self.boost_lr
                    + p * (self.base_lr - self.boost_lr)
                )

                self.eps.value = 0.15 - 0.10 * p

            else:
                self._finish_mode(step)

        elif self.mode == "stabilize":
            if elapsed >= 150_000:
                self._finish_mode(step)

    def _adapt_trend(self):
        if len(self.history) < 10:
            return

        previous = np.mean(self.history[-10:-5])
        current = np.mean(self.history[-5:])

        trend = (
            (current - previous)
            / max(abs(previous), 10.0)
        )

        if trend > 0.10:
            factor = 1.20
        elif trend > 0.03:
            factor = 1.10
        elif trend < -0.10:
            factor = 0.50
        elif trend < -0.03:
            factor = 0.75
        else:
            factor = 1.0

        old = self.lr.value

        self.lr.value = float(np.clip(
            old * factor,
            self.min_lr,
            self.max_lr,
        ))

        if old != self.lr.value:
            print(
                f"[LR TREND] {trend:+.2%} "
                f"{old:.7f} -> {self.lr.value:.7f}",
                flush=True,
            )

    def _oscillating(self):
        if len(self.history) < 20:
            return False

        values = np.asarray(self.history[-20:])
        scale = max(abs(values.mean()), 10.0)

        changes = np.diff(values)

        significant = changes[
            np.abs(changes) >= 0.02 * scale
        ]

        reversals = np.sum(
            significant[1:] * significant[:-1] < 0
        )

        return (
            np.ptp(values) / scale > 0.20
            and reversals >= 4
            and abs(values[-1] - values[0])
                / scale < 0.05
        )

    def _check_finite(self):
        with torch.no_grad():
            for p in self.model.policy.parameters():
                if not torch.isfinite(p).all().item():
                    return False

        return True

    def _log(self, step):
        optimizer = self.model.policy.optimizer

        actual_lr = optimizer.param_groups[0]["lr"]

        loss = self.logger.name_to_value.get(
            "train/loss", float("nan")
        )

        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                step,
                self.lr.value,
                actual_lr,
                self.model.exploration_rate,
                self.mode,
                loss,
            ])

    def _on_step(self):
        step = self.num_timesteps

        if step >= self.warmup and not self.eps_attached:
            self.model.exploration_schedule = self.eps
            self.eps_attached = True
            self.last_improvement = step

            print(
                "[ADAPTIVE] Exploration control enabled",
                flush=True,
            )

        self._update_mode(step)

        results = self.eval_callback.evaluations_results

        if len(results) > self.last_eval:
            self.last_eval = len(results)

            reward = float(np.mean(results[-10:]))
            self.history.append(reward)

            if not np.isfinite(reward):
                raise TrainingSafetyError("Invalid evaluation reward")

            if step >= self.warmup and len(results) >= 10:

                threshold = max(
                    2.0,
                    abs(self.best_reward or 0) * 0.05
                )

                if (
                    self.best_reward is None
                    or reward >= self.best_reward + threshold
                ):
                    self.best_reward = reward
                    self.last_improvement = step

                plateau = (
                    step - self.last_improvement
                    >= self.patience
                )

                if (
                    self.mode == "normal"
                    and step >= self.cooldown_until
                ):
                    if self._oscillating():
                        self._start_mode("stabilize", step)

                    elif plateau:
                        self._start_mode("boost", step)

                    elif self.last_eval % 5 == 0:
                        self._adapt_trend()

            print(
                f"[ADAPTIVE] step={step:,} "
                f"mode={self.mode} "
                f"reward={reward:.2f} "
                f"lr={self.lr.value:.7f}",
                flush=True,
            )

        if step % 5_000 == 0:
            if not self._check_finite():
                print(
                    "[SAFETY] NaN/Inf detected.",
                    flush=True,
                )
                return False

            self._log(step)

        self.logger.record(
            "adaptive/current_lr", self.lr.value
        )

        return True
