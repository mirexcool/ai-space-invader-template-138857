import csv
from pathlib import Path

import numpy as np
import torch
from training.safety import TrainingSafetyError
from training.performance_recovery import (
    PerformanceRollback,
    save_performance_checkpoint,
)


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
        self.max_lr = base_lr * 2
        self.boost_lr = base_lr * 2
        self.lr_levels = [1.2, 1.5, 1.8, 2.0]
        self.eps_levels = [0.08, 0.10, 0.15, 0.20]
        self.trial_steps = 250_000
        self.trial_level = 0
        self.trial_baseline = None
        self.next_parameter = "explore"
        self.trial_history = []
        self.trial_rewards = []
        self.trial_started = None

        self.warmup = warmup
        self.patience = 300_000
        self.cooldown = 100_000

        self.lr = MutableSchedule(base_lr)
        self.eps = MutableSchedule(0.05)

        self.mode = "normal"
        self.mode_start = 0
        self.cooldown_until = 0

        self.best_reward = None
        self.degradation_since = None
        self.degradation_patience = 250_000
        self.performance_rollbacks = 0

        self.last_improvement = warmup
        self.last_eval = 0
        self.history = []
        self.eval_window = []
        self.eps_attached = False

    def _on_training_start(self):
        self.model.lr_schedule = self.lr
        self.last_improvement = max(
            self.warmup, self.model.num_timesteps
        )

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

    def _start_trial(self, mode, step, reward, level=0):

        self.mode = mode
        self.mode_start = step
        self.trial_level = level
        self.trial_baseline = reward
        self.trial_rewards = []
        self.trial_started = step

        self.lr.value = self.base_lr
        self.eps.value = 0.05

        if mode == "boost":
            self.lr.value = (
                self.base_lr * self.lr_levels[level]
            )
        elif mode == "explore":
            self.eps.value = self.eps_levels[level]

        print(
            f"[TRIAL] {mode.upper()} "
            f"level={level + 1} step={step:,} "
            f"baseline={reward:.2f} "
            f"lr={self.lr.value:.8f} "
            f"eps={self.eps.value:.3f} "
                f"stagnation={step - self.last_improvement:,} "
                f"best={self.best_reward}",
            flush=True,
        )

    def _finish_mode(self, step):

        self.mode = "normal"
        self.lr.value = self.base_lr
        self.eps.value = 0.05

        self.cooldown_until = step + self.cooldown
        self.last_improvement = step

        self.trial_rewards = []
        self.trial_started = None

        print(
            f"[ADAPTIVE] NORMAL step={step:,}",
            flush=True,
        )

    def _evaluate_trial(self, step, raw_reward):

        self.trial_rewards.append(raw_reward)

        if step - self.mode_start < self.trial_steps:
            return

        if len(self.trial_rewards) < 10:
            return

        reward = float(np.mean(self.trial_rewards[-10:]))
        baseline = self.trial_baseline

        improvement = (
            (reward - baseline)
            / max(abs(baseline), 10.0)
        )

        mode = self.mode
        level = self.trial_level

        self.trial_history.append({
            "step": step,
            "mode": mode,
            "level": level + 1,
            "before": baseline,
            "after": reward,
            "improvement": improvement,
        })

        with (self.result_dir / "trials.csv").open(
            "a", newline=""
        ) as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    "step", "mode", "level",
                    "before", "after", "improvement"
                ])
            writer.writerow([
                step, mode, level + 1,
                baseline, reward, improvement
            ])

        print(
            f"[TRIAL RESULT] {mode.upper()} "
            f"level={level + 1} "
            f"improvement={improvement:+.2%}",
            flush=True,
        )

        if improvement >= 0.05 and level < 3:

            print("[TRIAL] Reinforcing success", flush=True)

            self._start_trial(
                mode, step, reward, level + 1
            )
            return

        if improvement <= -0.20:

            self.mode = "stabilize"
            self.mode_start = step
            self.lr.value = self.base_lr * 0.5
            self.eps.value = 0.05

            print(
                "[TRIAL] Degradation: stabilizing",
                flush=True,
            )
            return

        self.next_parameter = (
            "boost" if mode == "explore" else "explore"
        )

        self._finish_mode(step)

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

        if self.mode == "stabilize":
            if step - self.mode_start >= 250_000:
                self._finish_mode(step)

        results = self.eval_callback.evaluations_results

        if len(results) > self.last_eval:

            self.last_eval = len(results)

            raw_reward = float(np.mean(results[-1]))

            self.eval_window.append(raw_reward)
            self.eval_window = self.eval_window[-10:]

            reward = float(np.mean(self.eval_window))

            if not np.isfinite(reward):
                raise TrainingSafetyError(
                    "Invalid evaluation reward"
                )

            self.history.append(reward)

            if step >= self.warmup and len(self.eval_window) >= 10:

                threshold = max(
                    2.0,
                    abs(self.best_reward or 0) * 0.05
                )

                if (
                    self.best_reward is None
                    or (
                        reward >= self.best_reward + threshold
                        and raw_reward >= reward
                    )
                ):
                    save_performance_checkpoint(
                        self.model,
                        self.model_dir,
                        reward,
                    )

                    self.best_reward = reward
                    self.degradation_since = None
                    self.last_improvement = step

                    print(
                        f"[PROGRESS] New MA10 best={reward:.2f}",
                        flush=True,
                    )

                degradation = (
                    self.best_reward - reward
                ) / max(abs(self.best_reward), 10.0)

                if degradation >= 0.20:

                    if self.degradation_since is None:
                        self.degradation_since = step

                    elif (
                        step - self.degradation_since
                        >= self.degradation_patience
                    ):
                        raise PerformanceRollback(
                            f"MA10 degradation={degradation:.2%}"
                        )

                else:
                    self.degradation_since = None

                if self.mode in ("boost", "explore"):

                    self._evaluate_trial(step, raw_reward)

                elif (
                    self.mode == "normal"
                    and step >= self.cooldown_until
                    and step - self.last_improvement
                        >= self.patience
                ):

                    self._backup(step)

                    self._start_trial(
                        self.next_parameter,
                        step,
                        reward,
                    )

            print(
                f"[ADAPTIVE] step={step:,} "
                f"mode={self.mode} "
                f"reward={reward:.2f} "
                f"lr={self.lr.value:.8f} "
                f"eps={self.eps.value:.3f} "
                f"stagnation={step - self.last_improvement:,} "
                f"best={self.best_reward}",
                flush=True,
            )

        if step % 5_000 == 0:

            if not self._check_finite():
                raise TrainingSafetyError(
                    "NaN/Inf detected"
                )

            self._log(step)

        self.logger.record(
            "adaptive/current_lr", self.lr.value
        )

        self.logger.record(
            "adaptive/current_epsilon", self.eps.value
        )

        return True
