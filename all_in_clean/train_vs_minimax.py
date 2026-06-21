"""
train_vs_minimax.py — Train a DQN agent against the fast C++ minimax solver.

Uses the same DQNAgent infrastructure as train.py but with the new
FastMinimaxAgent as the primary opponent.  Supports all DQN variants:

    Classic DQN:    python train_vs_minimax.py
    Guided (HDQN):  python train_vs_minimax.py --heuristic_weight 0.2
    Prioritized:    python train_vs_minimax.py --use_per
    Guided+PER:     python train_vs_minimax.py --use_per --heuristic_weight 0.2

The opponent curriculum starts weak (random/greedy) and progressively
introduces the fast bitboard minimax so the agent learns incrementally.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time as time_module
from typing import Dict, Optional

import numpy as np
import torch

from environment import OthelloEnv
from evaluation import evaluate_fair
from agent import DQNAgent


# ------------------------------------------------------------------ #
#  Opponents                                                          #
# ------------------------------------------------------------------ #

def _import_opponents():
    from agents.random_agent import RandomAgent
    from agents.greedy_agent import GreedyAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.cpp_minimax_agent import FastMinimaxAgent

    return {
        "random":        RandomAgent,
        "greedy":        GreedyAgent,
        "heuristic":     HeuristicAgent,
        "fast_minimax":  FastMinimaxAgent,
    }


def _minimax_factory(board_size: int, time_limit: float):
    """Factory that returns a FastMinimaxAgent instance (matches the
    ``agent_b_class(board_size)`` signature used by evaluate_fair)."""
    from agents.cpp_minimax_agent import FastMinimaxAgent
    return FastMinimaxAgent(board_size=board_size, time_limit=time_limit)


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_final_reward(winner: int, agent_player: int) -> float:
    if winner == agent_player:
        return 1.0
    if winner == 0:
        return 0.0
    return -1.0


def _make_opponent(opponent_type: str, board_size: int,
                   opponent_classes: Dict, minimax_time_limit: float):
    cls = opponent_classes[opponent_type]
    if opponent_type == "fast_minimax":
        return cls(board_size=board_size, time_limit=minimax_time_limit)
    return cls(board_size=board_size)


def _choose_opponent_from_curriculum(
    episode: int,
    num_episodes: int,
) -> str:
    progress = episode / num_episodes

    if progress < 0.25:
        names =   ["random", "greedy", "heuristic"]
        weights = [0.50,     0.35,     0.15]
    elif progress < 0.50:
        names =   ["random", "greedy", "heuristic", "fast_minimax"]
        weights = [0.20,     0.25,     0.30,        0.25]
    elif progress < 0.75:
        names =   ["random", "greedy", "heuristic", "fast_minimax"]
        weights = [0.10,     0.15,     0.30,        0.45]
    else:
        names =   ["random", "greedy", "heuristic", "fast_minimax"]
        weights = [0.05,     0.10,     0.20,        0.65]

    return random.choices(names, weights=weights, k=1)[0]


def _epsilon_for_opponent(
    base_epsilon: float,
    opponent_type: str,
) -> float:
    floors = {
        "greedy": 0.15,
        "heuristic": 0.15,
        "fast_minimax": 0.20,
    }
    return max(base_epsilon, floors.get(opponent_type, 0.0))


# ------------------------------------------------------------------ #
#  Main training function                                             #
# ------------------------------------------------------------------ #

def train(
    # --- environment ---
    board_size: int = 6,
    # --- episodes ---
    num_episodes: int = 3_000,
    # --- epsilon schedule ---
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    # --- opponent ---
    opponent_type: str = "curriculum",
    minimax_time_limit: float = 1.0,
    # --- agent hyper-parameters ---
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 50_000,
    target_update_freq: int = 500,
    learning_starts: int = 1_000,
    double_dqn: bool = True,
    heuristic_weight: float = 0.0,
    use_per: bool = False,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_frames: int = 100_000,
    # --- I/O ---
    model_path: str = "models/minimax_trained.pth",
    load_model_path: Optional[str] = None,
    # --- logging ---
    print_every: int = 50,
    eval_every: int = 200,
    save_every: int = 500,
    seed: int = 42,
) -> Dict:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    set_seed(seed)

    opponent_classes = _import_opponents()
    env = OthelloEnv(board_size=board_size)

    agent = DQNAgent(
        board_size=board_size,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        learning_starts=learning_starts,
        double_dqn=double_dqn,
        heuristic_weight=heuristic_weight,
        use_per=use_per,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        per_beta_frames=per_beta_frames,
    )

    if load_model_path is not None:
        agent.load(load_model_path)
        print(f"Loaded weights from: {load_model_path}")

    agent_label = (
        f"{'guided_' if heuristic_weight > 0 else ''}"
        f"{'per_' if use_per else ''}"
        f"dqn"
    )
    print(f"Training: {agent_label} | board={board_size}x{board_size} | "
          f"opponent={opponent_type} | episodes={num_episodes} | "
          f"minimax_time_limit={minimax_time_limit}s")

    epsilon = epsilon_start
    rewards_history, win_history, loss_history = [], [], []
    beta_history, td_error_history = [], []
    last_train_info = None
    wins = draws = losses = 0

    opponent_stats = {
        name: {"games": 0, "wins": 0, "draws": 0, "losses": 0}
        for name in opponent_classes
    }

    total_game_time = 0.0
    total_games = 0

    # ============================================================== #
    #  Episode loop                                                    #
    # ============================================================== #
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False

        if opponent_type == "curriculum":
            cur_opp_type = _choose_opponent_from_curriculum(
                episode, num_episodes)
        else:
            cur_opp_type = opponent_type

        opponent = _make_opponent(cur_opp_type, board_size,
                                  opponent_classes, minimax_time_limit)

        cur_epsilon = _epsilon_for_opponent(epsilon, cur_opp_type)
        opponent_stats[cur_opp_type]["games"] += 1

        agent_player = 1 if episode % 2 == 0 else -1

        if env.current_player != agent_player:
            obs, _, done, _ = env.step(opponent.select_action(obs))

        episode_reward = 0.0
        t0 = time_module.perf_counter()

        while not done:
            state_obs = obs
            action = agent.select_action(state_obs, epsilon=cur_epsilon)
            obs_after_agent, _, done, info = env.step(action)

            if done:
                reward = compute_final_reward(info["winner"], agent_player)
                agent.store_transition(
                    state_obs, action, reward, obs_after_agent, True)
                train_info = agent.train_step()
                if train_info is not None:
                    last_train_info = train_info
                    loss_history.append(train_info["loss"])
                    if use_per:
                        beta_history.append(train_info["beta"])
                        td_error_history.append(train_info["mean_td_error"])
                episode_reward = reward
                break

            obs_after_opp, _, done, info = env.step(
                opponent.select_action(obs_after_agent))

            if done:
                reward = compute_final_reward(info["winner"], agent_player)
            else:
                reward = 0.0

            agent.store_transition(
                state_obs, action, reward, obs_after_opp, done)
            train_info = agent.train_step()
            if train_info is not None:
                last_train_info = train_info
                loss_history.append(train_info["loss"])
                if use_per:
                    beta_history.append(train_info["beta"])
                    td_error_history.append(train_info["mean_td_error"])

            obs = obs_after_opp
            episode_reward = reward

        game_time = time_module.perf_counter() - t0
        total_game_time += game_time
        total_games += 1

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        rewards_history.append(episode_reward)
        if episode_reward > 0:
            wins += 1
            win_history.append(1.0)
            opponent_stats[cur_opp_type]["wins"] += 1
        elif episode_reward < 0:
            losses += 1
            win_history.append(0.0)
            opponent_stats[cur_opp_type]["losses"] += 1
        else:
            draws += 1
            win_history.append(0.0)
            opponent_stats[cur_opp_type]["draws"] += 1

        # --- console log ---
        if episode % print_every == 0 or episode == 1:
            avg_r = np.mean(rewards_history[-100:])
            avg_w = np.mean(win_history[-100:])
            avg_time = total_game_time / max(total_games, 1)

            if last_train_info is None:
                extra = "learning not started"
            else:
                extra = f"loss={last_train_info['loss']:.4f}"
                if use_per:
                    extra += (
                        f" | beta={last_train_info['beta']:.3f}"
                        f" | td={last_train_info['mean_td_error']:.4f}"
                    )

            print(
                f"[{episode:>5}/{num_episodes}] "
                f"opp={cur_opp_type:<13} eps={epsilon:.3f} | "
                f"avg_r={avg_r:.3f} win%={avg_w:.3f} | "
                f"{extra} | W/D/L={wins}/{draws}/{losses} | "
                f"time={avg_time:.1f}s/game"
            )

        # --- periodic evaluation ---
        if episode % eval_every == 0:
            print("\n-- Evaluation (no exploration) --")
            agent.q_net.eval()

            for opp_name, opp_class in opponent_classes.items():
                if opp_name == "fast_minimax":
                    n_games = 50
                    result = evaluate_fair(
                        agent_a=agent,
                        agent_b_class=lambda bs: _minimax_factory(
                            bs, max(0.25, minimax_time_limit * 0.5)),
                        board_size=board_size,
                        n_games=n_games,
                    )
                else:
                    n_games = 100
                    result = evaluate_fair(
                        agent_a=agent,
                        agent_b_class=opp_class,
                        board_size=board_size,
                        n_games=n_games,
                    )
                print(
                    f"  {opp_name:<13} score={result['score']:.3f} "
                    f"win={result['win_rate']:.3f} "
                    f"W/D/L={result['wins']}/{result['draws']}/{result['losses']}"
                )

            agent.q_net.train()
            print()

        # --- periodic save ---
        if episode % save_every == 0:
            agent.save(model_path)
            print(f"  [saved → {model_path}]")

    # --- final save ---
    agent.save(model_path)
    print(f"\nModel saved -> {model_path}")

    print("\nResults by opponent type:")
    for opp_name, stats in opponent_stats.items():
        if stats["games"] > 0:
            print(
                f"  {opp_name:<13} games={stats['games']:>5}  "
                f"W/D/L={stats['wins']}/{stats['draws']}/{stats['losses']}"
            )

    return {
        "agent": agent,
        "rewards": rewards_history,
        "wins": win_history,
        "losses": loss_history,
        "betas": beta_history,
        "td_errors": td_error_history,
    }


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a DQN agent against the fast C++ minimax solver.")

    p.add_argument("--board_size",          type=int,   default=6)
    p.add_argument("--num_episodes",        type=int,   default=3_000)
    p.add_argument("--epsilon_start",       type=float, default=1.0)
    p.add_argument("--epsilon_end",         type=float, default=0.05)
    p.add_argument("--epsilon_decay",       type=float, default=0.999)
    p.add_argument("--opponent_type",       type=str,   default="curriculum",
                   choices=["curriculum", "random", "greedy",
                            "heuristic", "fast_minimax"])
    p.add_argument("--minimax_time_limit",  type=float, default=1.0,
                   help="Seconds per move for FastMinimaxAgent.")
    p.add_argument("--learning_rate",       type=float, default=1e-3)
    p.add_argument("--gamma",               type=float, default=0.99)
    p.add_argument("--batch_size",          type=int,   default=64)
    p.add_argument("--buffer_capacity",     type=int,   default=50_000)
    p.add_argument("--target_update_freq",  type=int,   default=500)
    p.add_argument("--learning_starts",     type=int,   default=1_000)
    p.add_argument("--heuristic_weight",    type=float, default=0.0,
                   help="Heuristic bonus weight. 0 = classic DQN.")
    p.add_argument("--use_per",             action="store_true",
                   help="Prioritized Experience Replay (PDQN).")
    p.add_argument("--per_alpha",           type=float, default=0.6)
    p.add_argument("--per_beta_start",      type=float, default=0.4)
    p.add_argument("--per_beta_frames",     type=int,   default=100_000)
    p.add_argument("--model_path",          type=str,
                   default="models/minimax_trained.pth")
    p.add_argument("--load_model_path",     type=str,   default=None)
    p.add_argument("--print_every",         type=int,   default=50)
    p.add_argument("--eval_every",          type=int,   default=200)
    p.add_argument("--save_every",          type=int,   default=500)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--no_double_dqn",       action="store_true",
                   help="Disable Double DQN.")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        board_size=args.board_size,
        num_episodes=args.num_episodes,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        opponent_type=args.opponent_type,
        minimax_time_limit=args.minimax_time_limit,
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
        model_path=args.model_path,
        load_model_path=args.load_model_path,
        print_every=args.print_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
    )
