import csv
import json
import numpy as np

from training.adaptive_lr import AdaptiveLRCallback
from training.safety import TrainingSafetyError
from training.performance_recovery import (
    save_performance_checkpoint,
    PerformanceRollback,
)


class AdaptiveV5Callback(AdaptiveLRCallback):

    CONFIG = {
        "ma_window": 10,
        "record_growth": 0.10,
        "plateau_patience": 500_000,
        "plateau_trend": 0.02,
        "stabilize_steps": 250_000,
        "boost_hold": 100_000,
        "boost_decay": 150_000,
        "boost_assessment": 250_000,
        "recovery_drop": 0.20,
        "recovery_steps": 250_000,
        "critical_drop": 0.40,
        "critical_evals": 5,
        "cooldown": 250_000,
        "boost_order": ["ER", "LR", "BOTH"],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.score_history = []
        self.best_ma = None
        self.record_ma = None
        self.boost_index = 0
        self.boost_type = None
        self.pending_boost = None
        self.critical_count = 0
        self.performance_stop = False

        self.boost_stats = {
            k: [] for k in self.CONFIG["boost_order"]
        }

    def _on_training_start(self):
        super()._on_training_start()

        self.boost_path = self.result_dir / "boost_history.csv"
        self.score_path = self.result_dir / "controller_scores.csv"

        with (self.result_dir / "controller_config.json").open("w") as f:
            json.dump(self.CONFIG, f, indent=4)

        for path, header in [
            (
                self.boost_path,
                ["start", "end", "type", "ma_before",
                 "ma_after", "change_pct"]
            ),
            (
                self.score_path,
                ["step", "score", "ma_score", "best_ma",
                 "record_ma", "mode", "lr", "epsilon"]
            ),
        ]:
            if not path.exists():
                with path.open("w", newline="") as f:
                    csv.writer(f).writerow(header)

    def _read_score(self):
        path = self.result_dir / "eval_monitor.monitor.csv"

        with path.open() as f:
            next(f)
            rows = list(csv.DictReader(f))

        n = self.eval_callback.n_eval_episodes

        if len(rows) < n:
            raise RuntimeError("Incomplete evaluation scores")

        score = float(np.mean([
            float(row["score"]) for row in rows[-n:]
        ]))

        if not np.isfinite(score):
            raise TrainingSafetyError("Invalid evaluation score")

        return score

    def _save_weights(self, name):
        path = self.model_dir / "controller"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path / name))

    def _start_v5_mode(self, mode, step):
        self.mode = mode
        self.mode_start = step

        if mode == "stabilize":
            self.lr.value = self.min_lr
            self.eps.value = 0.05

        elif mode == "recovery":
            self.lr.value = self.base_lr * 0.4
            self.eps.value = 0.05

        print(
            f"[V5] {mode.upper()} step={step:,}",
            flush=True
        )

    def _start_boost(self, step, ma):
        order = self.CONFIG["boost_order"]

        if all(self.boost_stats[k] for k in order):
            # Prefer historically successful intervention.
            # Every third trial still explores the alternatives.
            if self.boost_index % 3 == 0:
                kind = order[(self.boost_index // 3) % 3]
            else:
                kind = max(
                    order,
                    key=lambda k: np.mean(self.boost_stats[k][-3:])
                )
        else:
            kind = order[self.boost_index % 3]

        self.boost_index += 1
        self.boost_type = kind
        self.mode = "boost"
        self.mode_start = step

        self.pending_boost = {
            "type": kind,
            "start": step,
            "before": ma,
            "assess_at": step + 500_000,
        }

        self._save_weights(f"pre_boost_{step}")

        self._update_mode(step)

        print(
            f"[V5 BOOST] {kind} step={step:,} "
            f"MA={ma:.2f}",
            flush=True
        )

    def _update_mode(self, step):
        elapsed = step - self.mode_start

        if self.mode == "boost":
            if elapsed >= 250_000:
                self._finish_mode(step)
                return

            strength = (
                1.0 if elapsed < 100_000
                else 1.0 - (elapsed - 100_000) / 150_000
            )

            lr_strength = (
                strength
                if self.boost_type in ("LR", "BOTH")
                else 0.0
            )

            er_strength = (
                strength
                if self.boost_type in ("ER", "BOTH")
                else 0.0
            )

            self.lr.value = (
                self.base_lr
                + (self.boost_lr - self.base_lr) * lr_strength
            )

            self.eps.value = 0.05 + 0.10 * er_strength

        elif self.mode in ("stabilize", "recovery"):
            if elapsed >= 250_000:
                self._finish_mode(step)

    def _assess_boost(self, step, ma):
        trial = self.pending_boost

        if trial is None or step < trial["assess_at"]:
            return

        before = trial["before"]
        change = (ma - before) / max(abs(before), 1.0)

        self.boost_stats[trial["type"]].append(change)

        with self.boost_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                trial["start"],
                step,
                trial["type"],
                before,
                ma,
                change * 100,
            ])

        print(
            f"[V5 BOOST RESULT] {trial['type']} "
            f"{before:.2f} -> {ma:.2f} "
            f"({change:+.2%})",
            flush=True
        )

        self.pending_boost = None

    def _process_score(self, step, score):
        self.score_history.append(score)

        if len(self.score_history) < 10:
            return

        ma = float(np.mean(self.score_history[-10:]))

        if step < self.warmup:
            return

        if self.best_ma is None:
            self.best_ma = ma
            self.record_ma = ma
            self.last_improvement = step
            self._save_weights("best_ma")
            save_performance_checkpoint(
                self.model, self.model_dir, ma
            )

        if ma > self.best_ma:
            self.best_ma = ma
            self._save_weights("best_ma")

        record = ma >= self.record_ma * 1.10

        if record:
            self.record_ma = ma
            self.last_improvement = step

            save_performance_checkpoint(
                self.model, self.model_dir, ma
            )

        self._assess_boost(step, ma)

        drop = (
            (self.best_ma - ma)
            / max(abs(self.best_ma), 1.0)
        )

        if drop >= 0.40:
            self.critical_count += 1
        else:
            self.critical_count = 0

        if self.critical_count >= 5:
            self.performance_stop = True

            print(
                "[V5] Critical sustained degradation. "
                "Best MA checkpoint preserved.",
                flush=True
            )

        elif self.mode == "normal":

            if record:
                self._start_v5_mode("stabilize", step)

            elif (
                step >= self.cooldown_until
                and self.pending_boost is None
            ):
                if drop >= 0.20:
                    self._start_v5_mode("recovery", step)

                elif len(self.score_history) >= 20:
                    previous = np.mean(
                        self.score_history[-20:-10]
                    )

                    trend = (
                        (ma - previous)
                        / max(abs(previous), 1.0)
                    )

                    plateau = (
                        step - self.last_improvement >= 500_000
                        and abs(trend) <= 0.02
                    )

                    if plateau:
                        self._start_boost(step, ma)

        with self.score_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                step, score, ma, self.best_ma,
                self.record_ma, self.mode,
                self.lr.value, self.eps.value,
            ])

        print(
            f"[V5] step={step:,} "
            f"score={score:.2f} MA={ma:.2f} "
            f"best={self.best_ma:.2f} "
            f"drop={drop:.1%} mode={self.mode} "
            f"LR={self.lr.value:.7f} "
            f"ER={self.eps.value:.3f}",
            flush=True
        )

    def _on_step(self):
        step = self.num_timesteps

        if step >= self.warmup and not self.eps_attached:
            self.model.exploration_schedule = self.eps
            self.eps_attached = True
            self.last_improvement = step

        self._update_mode(step)

        results = self.eval_callback.evaluations_results

        if len(results) > self.last_eval:
            self.last_eval = len(results)
            self._process_score(step, self._read_score())

        if step % 5_000 == 0:
            if not self._check_finite():
                raise TrainingSafetyError("NaN/Inf detected")

            self._log(step)

        self.logger.record(
            "adaptive/current_lr", self.lr.value
        )

        if self.performance_stop:
            raise PerformanceRollback(
                "Sustained MA score degradation"
            )

        return True
