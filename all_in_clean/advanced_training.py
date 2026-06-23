"""
Advanced training launcher for Othello DQN.

Automates experiment folder creation, parameter documentation,
checkpoint management, best-model tracking, and resume support.

Usage:
    # New experiment
    uv run python advanced_training.py \\
        --load_model_path models/guided_per_dqn_6_best_overnight.pth

    # Resume existing experiment
    uv run python advanced_training.py --experiment_dir experiments_001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import torch

from train_vs_minimax import train


class _Tee:
    """Duplicates writes to both a file handle and stdout."""
    def __init__(self, path: str):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data: str) -> None:
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def _resolve_experiment_dir(base_dir: str) -> str:
    n = 1
    while True:
        name = f"{base_dir}_{n:03d}"
        if not os.path.exists(name):
            os.makedirs(name)
            return name
        n += 1


def _read_base_model_info(path: str) -> dict:
    info = {"path": path, "sha256": None, "config": None, "train_steps": None}
    if not path or not os.path.isfile(path):
        return info

    with open(path, "rb") as f:
        info["sha256"] = hashlib.sha256(f.read()).hexdigest()[:16]

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return info

    info["train_steps"] = ckpt.get("train_steps")
    info["config"] = ckpt.get("config")
    return info


def _build_setup(
    experiment_dir: str,
    load_model_path: Optional[str],
    train_kwargs: dict,
) -> dict:
    base_model = _read_base_model_info(load_model_path) if load_model_path else None

    relevant = {k: v for k, v in train_kwargs.items()
                if k not in ("model_path",)}
    return {
        "experiment_dir": experiment_dir,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "training_params": relevant,
    }


def _save_setup(setup: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(setup, f, indent=2, default=str)
        f.write("\n")


def _load_resume_state(exp_dir: str) -> Optional[dict]:
    path = os.path.join(exp_dir, "results", "resume_state.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Advanced DQN training with experiment tracking.")

    mode = p.add_argument_group("mode (exactly one)")
    mode.add_argument("--base_dir", type=str, default="Training/experiments",
                      help="Base path for NEW experiment folder (auto-incremented).")
    mode.add_argument("--experiment_dir", type=str, default=None,
                      help="EXISTING experiment dir to resume.")

    # --- forwardable training args (override defaults or resume params) ---
    tg = p.add_argument_group("training overrides")
    tg.add_argument("--load_model_path",     type=str,   default=None)
    tg.add_argument("--board_size",          type=int,   default=6)
    tg.add_argument("--num_episodes",        type=int,   default=200_000)
    tg.add_argument("--minimax_max_depth",   type=int,   default=5)
    tg.add_argument("--minimax_time_limit",  type=float, default=1.0)
    tg.add_argument("--final_minimax_weight", type=float, default=0.60)
    tg.add_argument("--minimax_start_progress", type=float, default=0.05)
    tg.add_argument("--minimax_full_progress", type=float, default=0.30)
    tg.add_argument("--epsilon_start",       type=float, default=0.05)
    tg.add_argument("--epsilon_end",         type=float, default=0.01)
    tg.add_argument("--epsilon_decay",       type=float, default=0.9995)
    tg.add_argument("--learning_rate",       type=float, default=5e-4)
    tg.add_argument("--gamma",               type=float, default=0.99)
    tg.add_argument("--batch_size",          type=int,   default=128)
    tg.add_argument("--buffer_capacity",     type=int,   default=50_000)
    tg.add_argument("--target_update_freq",  type=int,   default=500)
    tg.add_argument("--learning_starts",     type=int,   default=1_000)
    tg.add_argument("--heuristic_weight",    type=float, default=0.2)
    tg.add_argument("--use_per",             action="store_true", default=True)
    tg.add_argument("--per_alpha",           type=float, default=0.6)
    tg.add_argument("--per_beta_start",      type=float, default=0.4)
    tg.add_argument("--per_beta_frames",     type=int,   default=100_000)
    tg.add_argument("--tau",                 type=float, default=0.005)
    tg.add_argument("--double_dqn",          action="store_true", default=True)
    tg.add_argument("--no_double_dqn",       action="store_true")
    tg.add_argument("--max_minutes",         type=float, default=1000000)

    tg.add_argument("--eval_every",          type=int,   default=1000)
    tg.add_argument("--save_every",          type=int,   default=1000)
    tg.add_argument("--print_every",         type=int,   default=50)
    tg.add_argument("--record_game_eps",     type=str,   default="5000,10000,20000,30000,40000,50000,100000,150000")
    tg.add_argument("--n_record_games",      type=int,   default=3)
    tg.add_argument("--seed",                type=int,   default=42)
    tg.add_argument("--profile",             action="store_true")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ---------------------------------------------------------------- #
    #  Mode: new experiment or resume                                   #
    # ---------------------------------------------------------------- #
    if args.experiment_dir:
        exp_dir = args.experiment_dir
        if not os.path.isdir(exp_dir):
            print(f"Error: experiment directory '{exp_dir}' not found.")
            return

        resume = _load_resume_state(exp_dir)
        if resume is None:
            print(f"No resume_state.json found in {exp_dir}. "
                  f"Cannot resume.")
            return

        last_ep = resume["episode"]
        load_from = resume.get("model_path", None)

        # if the resume model doesn't exist, try model_checkpoints
        if load_from is None or not os.path.isfile(load_from):
            fallback = os.path.join(exp_dir, "model_checkpoints",
                                    f"model_ep{last_ep}.pth")
            if os.path.isfile(fallback):
                load_from = fallback
            else:
                print(f"Neither model at '{load_from}' nor checkpoint "
                      f"'{fallback}' found. Cannot resume.")
                return

        model_checkpoints_dir = os.path.join(exp_dir, "model_checkpoints")
        os.makedirs(model_checkpoints_dir, exist_ok=True)

        results_dir = os.path.join(exp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        model_path = os.path.join(results_dir, "latest.pth")
        best_model_path = os.path.join(exp_dir, "best_model.pth")

        log_path = os.path.join(exp_dir, "training.log")
        tee = _Tee(log_path)
        sys.stdout = tee

        start_episode = last_ep + 1
        epsilon_start = resume.get("epsilon", args.epsilon_start)

        print(f"Resuming experiment: {exp_dir}/")
        print(f"  from episode {last_ep}  (next: {start_episode})")
        print(f"  epsilon={epsilon_start:.4f}")
        print(f"  loading model: {load_from}")
        print()

        train_kwargs = dict(
            board_size=args.board_size,
            num_episodes=args.num_episodes,
            start_episode=start_episode,
            epsilon_start=epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            minimax_max_depth=args.minimax_max_depth,
            minimax_time_limit=args.minimax_time_limit,
            final_minimax_weight=args.final_minimax_weight,
            minimax_start_progress=args.minimax_start_progress,
            minimax_full_progress=args.minimax_full_progress,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_capacity=args.buffer_capacity,
            target_update_freq=args.target_update_freq,
            learning_starts=args.learning_starts,
            double_dqn=not args.no_double_dqn,
            heuristic_weight=args.heuristic_weight,
            use_per=args.use_per,
            per_alpha=args.per_alpha,
            per_beta_start=args.per_beta_start,
            per_beta_frames=args.per_beta_frames,
            tau=args.tau,
            model_path=model_path,
            load_model_path=load_from,
            checkpoint_dir=model_checkpoints_dir,
            best_model_path=best_model_path,
            max_minutes=args.max_minutes,
            print_every=args.print_every,
            eval_every=args.eval_every,
            save_every=args.save_every,
            seed=args.seed,
            profile=args.profile,
            record_game_eps=args.record_game_eps,
            n_record_games=args.n_record_games,
        )

        # update setup.json with resume info
        setup = _build_setup(exp_dir, load_from, {**train_kwargs,
                             "load_model_path": load_from})
        setup["resumed_from"] = {
            "previous_episode": last_ep,
            "previous_epsilon": resume.get("epsilon"),
        }
        setup_path = os.path.join(exp_dir, "setup.json")
        _save_setup(setup, setup_path)
        print(f"Setup updated -> {setup_path}")

        train(**train_kwargs)

        print(f"\nResumed experiment complete: {exp_dir}/")

        sys.stdout = tee.stdout
        tee.close()

    elif args.base_dir:
        exp_dir = _resolve_experiment_dir(args.base_dir)
        print(f"Experiment directory: {exp_dir}/")

        model_checkpoints_dir = os.path.join(exp_dir, "model_checkpoints")
        os.makedirs(model_checkpoints_dir, exist_ok=True)

        results_dir = os.path.join(exp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        model_path = os.path.join(results_dir, "latest.pth")
        best_model_path = os.path.join(exp_dir, "best_model.pth")

        log_path = os.path.join(exp_dir, "training.log")
        tee = _Tee(log_path)
        sys.stdout = tee

        train_kwargs = dict(
            board_size=args.board_size,
            num_episodes=args.num_episodes,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            minimax_max_depth=args.minimax_max_depth,
            minimax_time_limit=args.minimax_time_limit,
            final_minimax_weight=args.final_minimax_weight,
            minimax_start_progress=args.minimax_start_progress,
            minimax_full_progress=args.minimax_full_progress,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_capacity=args.buffer_capacity,
            target_update_freq=args.target_update_freq,
            learning_starts=args.learning_starts,
            double_dqn=not args.no_double_dqn,
            heuristic_weight=args.heuristic_weight,
            use_per=args.use_per,
            per_alpha=args.per_alpha,
            per_beta_start=args.per_beta_start,
            per_beta_frames=args.per_beta_frames,
            tau=args.tau,
            model_path=model_path,
            load_model_path=args.load_model_path,
            checkpoint_dir=model_checkpoints_dir,
            best_model_path=best_model_path,
            max_minutes=args.max_minutes,
            print_every=args.print_every,
            eval_every=args.eval_every,
            save_every=args.save_every,
            seed=args.seed,
            profile=args.profile,
            record_game_eps=args.record_game_eps,
            n_record_games=args.n_record_games,
        )

        setup = _build_setup(exp_dir, args.load_model_path, train_kwargs)
        setup_path = os.path.join(exp_dir, "setup.json")
        _save_setup(setup, setup_path)
        print(f"Setup saved -> {setup_path}")

        print(f"Results           -> {results_dir}/")
        print(f"Model checkpoints -> {model_checkpoints_dir}/")
        print(f"Best model        -> {best_model_path}")
        print()

        train(**train_kwargs)

        print(f"\nExperiment complete: {exp_dir}/")
        print(f"  setup.json    -> {setup_path}")
        print(f"  best model    -> {best_model_path}")
        print(f"  checkpoints   -> {model_checkpoints_dir}/")

        sys.stdout = tee.stdout
        tee.close()

    else:
        print("Specify --experiment_dir <dir> to resume, "
              "or --base_dir <name> to start a new experiment.")


if __name__ == "__main__":
    main()
