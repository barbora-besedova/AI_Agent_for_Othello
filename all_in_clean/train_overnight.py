"""
train_overnight.py — Overnight training run for the best DQN agent.

Loads the best-performing model (Guided PER, board_size=6) and continues
training with heavy self-play to further improve against all opponents.

Usage:
    python train_overnight.py          # default: 30,000 episodes
    python train_overnight.py --num_episodes 50000

The script saves checkpoints and evaluation logs to models/.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time as time_module
from typing import Dict, Optional

import numpy as np
import torch

from environment import OthelloEnv
from evaluation import evaluate_fair
from agent import DQNAgent

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _import_opponents():
    from agents.random_agent import RandomAgent
    from agents.greedy_agent import GreedyAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.minimax_agent import MinimaxAgent
    return {
        "random": RandomAgent,
        "greedy": GreedyAgent,
        "heuristic": HeuristicAgent,
        "minimax": MinimaxAgent,
    }


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


def _make_opponent(opponent_type: str, board_size: int, opponent_classes: Dict):
    if opponent_type not in opponent_classes:
        raise ValueError(f"Unknown opponent_type '{opponent_type}'.")
    return opponent_classes[opponent_type](board_size)


def _choose_opponent_from_curriculum(
    episode: int, num_episodes: int, include_self_play: bool = True,
) -> str:
    """
    Three-phase curriculum biased toward self-play for continuous improvement.
    Minimax is kept at >=20% throughout to prevent catastrophic forgetting of
    the minimax playing style — the main cause of late-phase oscillation.
    """
    progress = episode / num_episodes
    if progress < 0.30:
        names = ["random", "greedy", "heuristic", "minimax", "self-play"]
        weights = [0.05, 0.05, 0.10, 0.20, 0.60]
    elif progress < 0.70:
        names = ["random", "greedy", "heuristic", "minimax", "self-play"]
        weights = [0.05, 0.05, 0.05, 0.20, 0.65]
    else:
        names = ["random", "greedy", "heuristic", "minimax", "self-play"]
        weights = [0.00, 0.00, 0.05, 0.25, 0.70]
    if not include_self_play:
        names = names[:-1]
        weights = weights[:-1]
    return random.choices(names, weights=weights, k=1)[0]


def _epsilon_for_opponent(base_epsilon: float, opponent_type: str) -> float:
    """Ensure a minimum exploration rate against strong opponents."""
    floors = {"greedy": 0.15, "heuristic": 0.15, "minimax": 0.20}
    return max(base_epsilon, floors.get(opponent_type, 0.0))


def train_overnight(
    board_size: int = 6,
    num_episodes: int = 30000,
    epsilon_start: float = 0.2,
    epsilon_end: float = 0.02,
    epsilon_decay: float = 0.9999,
    load_model_path: str | None = "models/guided_per_dqn_6.pth",
    model_save_prefix: str = "guided_per_dqn_6_overnight",
    learning_rate: float = 5e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 100_000,
    target_update_freq: int = 1000,
    learning_starts: int = 1000,
    use_per: bool = True,
    heuristic_weight: float = 0.2,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_frames: int = 200_000,
    self_play_update_freq: int = 500,
    print_every: int = 200,
    eval_every: int = 1000,
    save_every: int = 2000,
    seed: int = 42,
    eval_random_plies: int = 2,
) -> Dict:
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
        double_dqn=True,
        heuristic_weight=heuristic_weight,
        use_per=use_per,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        per_beta_frames=per_beta_frames,
    )

    if load_model_path and os.path.exists(load_model_path):
        agent.load(load_model_path, load_optimizer=False)
        print(f"Loaded weights from: {load_model_path}")
        print(f"  (train_steps={agent.train_steps})")

    self_play_agent = copy.deepcopy(agent)
    self_play_agent.q_net.eval()
    self_play_agent.target_net.eval()

    print(f"Overnight training: board={board_size}x{board_size} | episodes={num_episodes}")
    print(f"  learning_rate={learning_rate} | epsilon: {epsilon_start} -> {epsilon_end}")
    print(f"  use_per={use_per} | heuristic_weight={heuristic_weight}")

    epsilon = epsilon_start
    rewards_history, win_history, loss_history = [], [], []
    last_train_info = None
    wins, draws, losses = 0, 0, 0

    opponent_stats = {
        name: {"games": 0, "wins": 0, "draws": 0, "losses": 0}
        for name in [*opponent_classes.keys(), "self-play"]
    }

    eval_log_path = os.path.join(MODELS_DIR, f"{model_save_prefix}_eval_log.csv")
    with open(eval_log_path, "w") as f:
        f.write("episode,random,greedy,heuristic,minimax\n")

    start_time = time_module.perf_counter()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False

        cur_opp_type = _choose_opponent_from_curriculum(episode, num_episodes)

        if cur_opp_type == "self-play":
            opponent = self_play_agent
        else:
            opponent = _make_opponent(cur_opp_type, board_size, opponent_classes)

        cur_epsilon = _epsilon_for_opponent(epsilon, cur_opp_type)

        agent_player = 1 if episode % 2 == 0 else -1
        if env.current_player != agent_player:
            obs, _, done, _ = env.step(opponent.select_action(obs))

        episode_reward = 0.0

        while not done:
            state_obs = obs
            action = agent.select_action(state_obs, epsilon=cur_epsilon)
            obs_after_agent, _, done, info = env.step(action)

            if done:
                reward = compute_final_reward(info["winner"], agent_player)
                agent.store_transition(state_obs, action, reward, obs_after_agent, True)
                train_info = agent.train_step()
                if train_info is not None:
                    last_train_info = train_info
                    loss_history.append(train_info["loss"])
                episode_reward = reward
                break

            obs_after_opp, _, done, info = env.step(opponent.select_action(obs_after_agent))
            reward = compute_final_reward(info["winner"], agent_player) if done else 0.0
            agent.store_transition(state_obs, action, reward, obs_after_opp, done)
            train_info = agent.train_step()
            if train_info is not None:
                last_train_info = train_info
                loss_history.append(train_info["loss"])
            obs = obs_after_opp
            episode_reward = reward

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
        opponent_stats[cur_opp_type]["games"] += 1

        if episode % print_every == 0:
            avg_r = np.mean(rewards_history[-200:])
            avg_w = np.mean(win_history[-200:])
            elapsed = time_module.perf_counter() - start_time
            eps_per_sec = episode / elapsed if elapsed > 0 else 0

            extra = ""
            if last_train_info:
                extra = f"loss={last_train_info['loss']:.4f}"
                if use_per:
                    extra += f" | beta={last_train_info['beta']:.3f} | td={last_train_info['mean_td_error']:.4f}"

            print(
                f"[{episode:>6}/{num_episodes}] "
                f"eps={epsilon:.3f} | avg_r={avg_r:.3f} win%={avg_w:.3f} | "
                f"{extra} | W/D/L={wins}/{draws}/{losses} | "
                f"{eps_per_sec:.1f}ep/s | {elapsed/3600:.1f}h elapsed"
            )

        if episode % eval_every == 0:
            print(f"\n── Evaluation at episode {episode} ──")
            agent.q_net.eval()

            eval_scores = {}
            with torch.no_grad():
                for opp_name, opp_class in opponent_classes.items():
                    result = evaluate_fair(
                        agent_a=agent,
                        agent_b_class=opp_class,
                        board_size=board_size,
                        n_games=100,
                        random_opening_plies=eval_random_plies,
                    )
                    eval_scores[opp_name] = result["score"]
                    print(
                        f"  {opp_name:<12} score={result['score']:.3f} "
                        f"W/D/L={result['wins']}/{result['draws']}/{result['losses']}"
                    )

            with open(eval_log_path, "a") as f:
                f.write(f"{episode},{eval_scores['random']:.3f},{eval_scores['greedy']:.3f},"
                        f"{eval_scores['heuristic']:.3f},{eval_scores['minimax']:.3f}\n")

            agent.q_net.train()
            print()

        if episode % save_every == 0:
            save_path = os.path.join(MODELS_DIR, f"{model_save_prefix}_ep{episode}.pth")
            agent.save(save_path)
            print(f"  [saved → {os.path.basename(save_path)}]")

        if episode % self_play_update_freq == 0:
            self_play_agent.q_net.load_state_dict(agent.q_net.state_dict())
            self_play_agent.target_net.load_state_dict(agent.q_net.state_dict())
            self_play_agent.q_net.eval()
            self_play_agent.target_net.eval()

    # Final save
    final_path = os.path.join(MODELS_DIR, f"{model_save_prefix}_final.pth")
    agent.save(final_path)
    total_time = time_module.perf_counter() - start_time
    print(f"\nModel saved → {os.path.basename(final_path)}")
    print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")

    print("\nResults by opponent type:")
    for opp_name, stats in opponent_stats.items():
        if stats["games"] > 0:
            print(f"  {opp_name:<12} games={stats['games']:>5} "
                  f"W/D/L={stats['wins']}/{stats['draws']}/{stats['losses']}")

    return {
        "agent": agent,
        "rewards": rewards_history,
        "wins": win_history,
        "losses": loss_history,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Overnight training for Othello DQN agent.")
    p.add_argument("--num_episodes", type=int, default=30000,
                   help="Number of training episodes (default: 30000)")
    p.add_argument("--load_model", type=str, default="models/guided_per_dqn_6.pth",
                   help="Path to pretrained model (default: best model)")
    p.add_argument("--epsilon_start", type=float, default=0.2)
    p.add_argument("--epsilon_end", type=float, default=0.02)
    p.add_argument("--epsilon_decay", type=float, default=0.9999)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--learning_starts", type=int, default=1000)
    p.add_argument("--buffer_capacity", type=int, default=100000)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_random_plies", type=int, default=2,
                   help="Random opening moves per eval game (default: 2)")
    args = p.parse_args()

    train_overnight(
        num_episodes=args.num_episodes,
        load_model_path=args.load_model,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        learning_starts=args.learning_starts,
        buffer_capacity=args.buffer_capacity,
        print_every=args.print_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        eval_random_plies=args.eval_random_plies,
    )
