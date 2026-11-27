import argparse
import json
from pathlib import Path

import torch

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)

from training.environment import SpaceInvadersEnv
from training.frame_stack import TemporalFrameStack
from training.adaptive_lr import AdaptiveLRCallback


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--difficulty", default="progressive")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--dodge", action="store_true")
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument("--shot-penalty", type=float, default=0.0)
    parser.add_argument("--ignore-score-rollback", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--replay-buffer", type=str, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    name = (
        f"dqn_{args.difficulty}"
        f"_f{args.frames}_s{args.stride}"
        f"_dodge{int(args.dodge)}"
        f"_seed{args.seed}"
    )

    name += (
        f"_sp{args.shot_penalty:g}"
        f"_rb{int(args.ignore_score_rollback)}"
        f"_g{args.gamma:g}"
    )

    name += f"_lr{args.learning_rate:g}_net64x64_quadedge_adaptive"

    if args.resume:
        name += f"_resume_{Path(args.resume).stem}_{args.steps}"

    model_dir = Path("models") / name
    result_dir = Path("results") / name

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    def make_env(filename=None):
        env = SpaceInvadersEnv(
            difficulty=args.difficulty,
            use_dodge=args.dodge,
            shot_penalty=args.shot_penalty,
            ignore_score_rollback=args.ignore_score_rollback,
        )

        if args.frames > 1:
            env = TemporalFrameStack(
                env,
                frames=args.frames,
                stride=args.stride,
            )

        return Monitor(env, filename=filename, info_keywords=("score", "wave", "lives"))

    env = make_env(str(result_dir / "train"))
    eval_env = make_env(str(result_dir / "eval_monitor"))

    eval_env.reset(seed=100000)

    print("\n===== TRAINING CONFIG =====")
    print("Experiment:", name)
    print("Observation:", env.observation_space.shape)
    print("Actions:", env.action_space)
    print("Frames:", args.frames)
    print("Stride:", args.stride)
    print("History seconds:", (args.frames - 1) * args.stride / 30)
    print("Device:", args.device)
    print("Steps:", args.steps)
    print("===========================\n")

    if args.resume:
        if not args.replay_buffer:
            parser.error("--resume requires --replay-buffer")

        model = DQN.load(
            args.resume,
            env=env,
            device=args.device,
        )

        model.load_replay_buffer(args.replay_buffer)

        model.exploration_initial_eps = 0.05
        model.exploration_final_eps = 0.05
        model.exploration_rate = 0.05
        model.exploration_schedule = lambda progress: 0.05

        print("Resumed from:", args.resume)
        print("Previous steps:", model.num_timesteps)
        print("Replay buffer size:", model.replay_buffer.size())
        print("Target steps:", model.num_timesteps + args.steps)
        print("Epsilon:", model.exploration_rate)

    else:
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=500_000,
            learning_starts=10_000,
            batch_size=128,
            gamma=args.gamma,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[64, 64]),
            tensorboard_log="results/tensorboard",
            device=args.device,
            seed=args.seed,
            verbose=1,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(result_dir / "eval"),
        eval_freq=25_000,
        n_eval_episodes=100,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="dqn",
        save_replay_buffer=False,
    )

    replay_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="dqn",
        save_replay_buffer=True,
    )

    adaptive_callback = AdaptiveLRCallback(
        eval_callback=eval_callback,
        model_dir=model_dir,
        result_dir=result_dir,
        base_lr=args.learning_rate,
        warmup=int(args.steps * 0.3),
    )

    model.learn(
        total_timesteps=args.steps,
        reset_num_timesteps=not bool(args.resume),
        callback=[
            eval_callback,
            checkpoint_callback,
            replay_callback,
            adaptive_callback,
        ],
        log_interval=20,
        tb_log_name=name,
    )

    model.save(str(model_dir / "final"))
    model.save_replay_buffer(
        str(model_dir / "final_replay_buffer.pkl")
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
