import shutil
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class RotatingCheckpointCallback(BaseCallback):

    def __init__(self, save_path, save_freq=1_000_000, keep=2):
        super().__init__()
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.keep = keep

    def _on_training_start(self):
        self.save_path.mkdir(parents=True, exist_ok=True)

        if not any(self.save_path.glob("step_*")):
            self._save(self.model.num_timesteps)

    def _on_step(self):
        step = self.num_timesteps

        if step % self.save_freq != 0:
            return True

        self._save(step)
        return True

    def _save(self, step):
        target = self.save_path / f"step_{step:012d}"
        temporary = self.save_path / f".tmp_{step:012d}"

        if target.exists():
            raise FileExistsError(target)

        if temporary.exists():
            shutil.rmtree(temporary)

        temporary.mkdir()

        self.model.save(str(temporary / "model.zip"))

        self.model.save_replay_buffer(
            str(temporary / "replay_buffer.pkl")
        )

        temporary.rename(target)

        print(f"[CHECKPOINT] Saved step={step:,}", flush=True)

        checkpoints = sorted(
            p for p in self.save_path.glob("step_*")
            if p.is_dir()
        )

        for old in checkpoints[:-self.keep]:
            shutil.rmtree(old)
            print(f"[CHECKPOINT] Removed {old.name}", flush=True)

        return True
