import math
import os
import torch

from stable_baselines3.common.callbacks import BaseCallback


class TrainingSafetyError(RuntimeError):
    pass


class SafetyCallback(BaseCallback):

    def __init__(self, check_freq=1000):
        super().__init__()
        self.check_freq = check_freq

    def _on_step(self):
        fail_at = int(os.environ.get("DQN_TEST_FAIL_AT", "0"))

        if fail_at and self.num_timesteps >= fail_at:
            os.environ.pop("DQN_TEST_FAIL_AT", None)

            print("[TEST] Injecting NaN", flush=True)

            with torch.no_grad():
                next(self.model.policy.parameters()).fill_(float("nan"))

        if self.num_timesteps % self.check_freq != 0:
            return True

        with torch.no_grad():
            for name, param in self.model.policy.named_parameters():
                if not torch.isfinite(param).all().item():
                    raise TrainingSafetyError(
                        f"Invalid parameter: {name}"
                    )

            optimizer = self.model.policy.optimizer

            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        if not torch.isfinite(value).all().item():
                            raise TrainingSafetyError(
                                f"Invalid optimizer state: {key}"
                            )

            loss = self.logger.name_to_value.get("train/loss")

            if loss is not None and not math.isfinite(float(loss)):
                raise TrainingSafetyError(
                    f"Invalid loss: {loss}"
                )

        return True
