from pathlib import Path

from stable_baselines3 import DQN


def restore_checkpoint(model_dir, env, device):
    recovery_dir = Path(model_dir) / "recovery"

    checkpoints = sorted(
        recovery_dir.glob("step_*"),
        reverse=True,
    )

    for checkpoint in checkpoints:
        try:
            model = DQN.load(
                checkpoint / "model.zip",
                env=env,
                device=device,
            )

            model.load_replay_buffer(
                checkpoint / "replay_buffer.pkl"
            )

            print(
                f"[ROLLBACK] Restored {checkpoint.name}",
                flush=True,
            )

            return model

        except Exception as e:
            print(
                f"[ROLLBACK] Cannot restore {checkpoint}: {e}",
                flush=True,
            )

    raise RuntimeError("No usable recovery checkpoint")
