import json
import shutil
from pathlib import Path

from stable_baselines3 import DQN


def save_performance_checkpoint(model, model_dir, reward):

    root = Path(model_dir) / "performance"
    root.mkdir(parents=True, exist_ok=True)

    step = model.num_timesteps
    target = root / f"step_{step:012d}"
    temporary = root / f".tmp_{step:012d}"

    if temporary.exists():
        shutil.rmtree(temporary)

    temporary.mkdir()

    model.save(str(temporary / "model.zip"))
    model.save_replay_buffer(
        str(temporary / "replay_buffer.pkl")
    )

    with (temporary / "metadata.json").open("w") as f:
        json.dump({
            "step": step,
            "reward": float(reward),
        }, f, indent=4)

    temporary.rename(target)

    for old in root.glob("step_*"):
        if old != target:
            shutil.rmtree(old)

    print(
        f"[PERFORMANCE SAVE] step={step:,} "
        f"reward={reward:.2f}",
        flush=True,
    )


def restore_performance_checkpoint(model, model_dir):

    root = Path(model_dir) / "performance"
    checkpoints = sorted(
        root.glob("step_*"), reverse=True
    )

    if not checkpoints:
        raise RuntimeError(
            "No performance checkpoint available"
        )

    checkpoint = checkpoints[0]

    restored = DQN.load(
        checkpoint / "model.zip",
        device=model.device,
    )

    restored.load_replay_buffer(
        checkpoint / "replay_buffer.pkl"
    )

    model.policy.load_state_dict(
        restored.policy.state_dict()
    )

    model.policy.optimizer.load_state_dict(
        restored.policy.optimizer.state_dict()
    )

    model.replay_buffer = restored.replay_buffer

    del restored

    # Prevent mixing a transition from the current
    # rollout with the restored replay buffer.
    model._last_obs = None

    print(
        f"[PERFORMANCE ROLLBACK] "
        f"restored={checkpoint.name} "
        f"current_step={model.num_timesteps:,}",
        flush=True,
    )


class PerformanceRollback(RuntimeError):
    pass
