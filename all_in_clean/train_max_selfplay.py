"""
This script maximize self-play training


train.py — Unified training script for a single DQN-family agent.

Supports all combinations of:
    • Classic DQN        (use_per=False, heuristic_weight=0)
    • Guided DQN (HDQN)  (heuristic_weight > 0)
    • Prioritized DQN    (use_per=True)
    • Self-play          (opponent_type="mix" or "self-play")
    • Curriculum mixing  (opponent_type="mix")

Quick-start examples
--------------------
# Classic DQN vs random opponent
python train.py

# Guided DQN with curriculum mix
python train.py --heuristic_weight 0.2 --opponent_type mix

# Prioritized DQN with mix
python train.py --use_per --opponent_type mix

# Guided + Prioritized
python train.py --use_per --heuristic_weight 0.2 --opponent_type mix

# Load a pre-trained checkpoint and fine-tune
python train.py --load_model_path models/my_agent.pth --num_episodes 2000
"""

from __future__ import annotations

import argparse
import copy
import os
import random
from typing import Dict, Optional

import numpy as np
import torch

from environment import OthelloEnv
from evaluation import evaluate_fair
from agent import DQNAgent

# ------------------------------------------------------------------ #
#  Lazy imports for rule-based opponents                              #
# ------------------------------------------------------------------ #

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


def _make_opponent(opponent_type: str, board_size: int, opponent_classes: Dict):
    """Instantiate a rule-based opponent by name."""
    if opponent_type not in opponent_classes:
        raise ValueError(
            f"Unknown opponent_type '{opponent_type}'. "
            f"Valid: {list(opponent_classes.keys())}"
        )
    return opponent_classes[opponent_type](board_size)


def _choose_opponent_from_curriculum(
    episode: int,
    num_episodes: int,
    include_self_play: bool = True,
) -> str:
    """
    Three-phase curriculum that gradually increases opponent strength.

    Phase 1 (0–30 %):   mostly random / greedy
    Phase 2 (30–70 %):  balanced mix, minimax appears
    Phase 3 (70–100 %): strong opponents dominate
    """
    progress = episode / num_episodes

    if progress < 0.30:
        names =   ["random", "greedy", "heuristic", "minimax", "self-play"]
        weights = [0.05,     0.05,     0.10,        0.20,       0.60]
    elif progress < 0.70:
        names =   ["random", "greedy", "heuristic", "minimax", "self-play"]
        weights = [0.05,     0.05,     0.05,        0.15,       0.70]
    else:
        names =   ["random", "greedy", "heuristic", "minimax", "self-play"]
        weights = [0.00,     0.00,     0.05,        0.15,       0.80]

    if not include_self_play:
        names = names[:-1]
        weights = weights[:-1]

    return random.choices(names, weights=weights, k=1)[0]


def _epsilon_for_opponent(
    base_epsilon: float,
    opponent_type: str,
) -> float:
    """Ensure a minimum exploration rate against strong opponents."""
    floors = {"greedy": 0.15, "heuristic": 0.15, "minimax": 0.20}
    return max(base_epsilon, floors.get(opponent_type, 0.0))


# ------------------------------------------------------------------ #
#  Main training function                                             #
# ------------------------------------------------------------------ #

def train(
    # --- environment ---
    board_size: int = 5,
    # --- episodes ---
    num_episodes: int = 5_000,
    # --- epsilon schedule ---
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    # --- opponent ---
    opponent_type: str = "mix",          # "random"|"greedy"|"heuristic"|"minimax"|"mix"|"self-play"
    # --- agent hyper-parameters ---
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 50_000,
    target_update_freq: int = 500,
    learning_starts: int = 1_000,
    double_dqn: bool = True,
    heuristic_weight: float = 0.0,       # >0 → guided/HDQN mode
    use_per: bool = False,               # True → PER/PDQN mode
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_frames: int = 100_000,
    # --- I/O ---
    model_path: str = "models/othello_agent.pth",
    load_model_path: Optional[str] = None,
    # --- self-play update interval ---
    self_play_update_freq: int = 500,
    # --- logging ---
    print_every: int = 100,
    eval_every: int = 250,
    save_every: int = 500,
    seed: int = 42,
) -> Dict:
    """
    Train a DQNAgent and return a dict with the trained agent and histories.

    The agent type is fully determined by the combination of flags:
        heuristic_weight=0, use_per=False  → Classic DQN
        heuristic_weight>0, use_per=False  → Guided DQN (HDQN)
        heuristic_weight=0, use_per=True   → Prioritized DQN (PDQN)
        heuristic_weight>0, use_per=True   → Guided + Prioritized
    """
    # ---- set-up ----
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

    # Optional: load weights from an existing checkpoint
    if load_model_path is not None:
        agent.load(load_model_path)
        print(f"Loaded weights from: {load_model_path}")

    # Frozen snapshot used for self-play
    self_play_agent = copy.deepcopy(agent)
    self_play_agent.q_net.eval()
    self_play_agent.target_net.eval()

    agent_label = (
        f"{'guided_' if heuristic_weight > 0 else ''}"
        f"{'per_' if use_per else ''}"
        f"dqn | hw={heuristic_weight} | per={use_per}"
    )
    print(f"Training: {agent_label} | opponent={opponent_type} | episodes={num_episodes}")

    # ---- histories ----
    epsilon = epsilon_start
    rewards_history, win_history, loss_history = [], [], []
    beta_history, td_error_history = [], []
    last_train_info = None
    wins = draws = losses = 0

    opponent_stats = {
        name: {"games": 0, "wins": 0, "draws": 0, "losses": 0}
        for name in [*opponent_classes.keys(), "self-play"]
    }

    # ================================================================== #
    #  Episode loop                                                        #
    # ================================================================== #
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False

        # --- choose opponent for this episode ---
        if opponent_type == "mix":
            cur_opp_type = _choose_opponent_from_curriculum(episode, num_episodes)
        else:
            cur_opp_type = opponent_type

        if cur_opp_type == "self-play":
            opponent = self_play_agent
        else:
            opponent = _make_opponent(cur_opp_type, board_size, opponent_classes)

        cur_epsilon = _epsilon_for_opponent(epsilon, cur_opp_type)
        opponent_stats[cur_opp_type]["games"] += 1

        # Alternate which color the agent plays
        agent_player = 1 if episode % 2 == 0 else -1

        # If agent is black (-1), opponent (player 1) moves first
        if env.current_player != agent_player:
            obs, _, done, _ = env.step(opponent.select_action(obs))

        episode_reward = 0.0

        # --- inner game loop ---
        while not done:
            state_obs = obs

            action = agent.select_action(state_obs, epsilon=cur_epsilon)
            obs_after_agent, _, done, info = env.step(action)

            # Game ended immediately after agent's move
            if done:
                reward = compute_final_reward(info["winner"], agent_player)

                agent.store_transition(state_obs, action, reward, obs_after_agent, True)
                train_info = agent.train_step()
                if train_info is not None:
                    last_train_info = train_info
                    loss_history.append(train_info["loss"])
                    if use_per:
                        beta_history.append(train_info["beta"])
                        td_error_history.append(train_info["mean_td_error"])

                episode_reward = reward
                break

            # Opponent responds
            obs_after_opp, _, done, info = env.step(opponent.select_action(obs_after_agent))

            if done:
                reward = compute_final_reward(info["winner"], agent_player)
            else:
                reward = 0.0

            agent.store_transition(state_obs, action, reward, obs_after_opp, done)
            train_info = agent.train_step()
            if train_info is not None:
                last_train_info = train_info
                loss_history.append(train_info["loss"])
                if use_per:
                    beta_history.append(train_info["beta"])
                    td_error_history.append(train_info["mean_td_error"])

            obs = obs_after_opp
            episode_reward = reward

        # --- epsilon decay ---
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # --- stats ---
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
        if episode % print_every == 0:
            avg_r = np.mean(rewards_history[-100:])
            avg_w = np.mean(win_history[-100:])

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
                f"[{episode:>6}/{num_episodes}] "
                f"opp={cur_opp_type:<10} ε={epsilon:.3f} | "
                f"avg_r={avg_r:.3f} win%={avg_w:.3f} | "
                f"{extra} | W/D/L={wins}/{draws}/{losses}"
            )

        # --- periodic evaluation ---
        if episode % eval_every == 0:
            print("\n── Evaluation (no exploration) ──")
            agent.q_net.eval()

            for opp_name, opp_class in opponent_classes.items():
                result = evaluate_fair(
                    agent_a=agent,
                    agent_b_class=opp_class,
                    board_size=board_size,
                    n_games=100,
                )
                print(
                    f"  {opp_name:<10} score={result['score']:.3f} "
                    f"win={result['win_rate']:.3f} "
                    f"W/D/L={result['wins']}/{result['draws']}/{result['losses']}"
                )

            agent.q_net.train()
            print()

        # --- periodic save ---
        if episode % save_every == 0:
            agent.save(model_path)

        # --- self-play snapshot update ---
        if episode % self_play_update_freq == 0:
            self_play_agent.q_net.load_state_dict(agent.q_net.state_dict())
            self_play_agent.target_net.load_state_dict(agent.q_net.state_dict())
            self_play_agent.q_net.eval()
            self_play_agent.target_net.eval()

    # --- final save ---
    agent.save(model_path)
    print(f"\nModel saved → {model_path}")

    if opponent_type == "mix":
        print("\nResults by opponent type:")
        for opp_name, stats in opponent_stats.items():
            if stats["games"] > 0:
                print(
                    f"  {opp_name:<12} games={stats['games']:>5} "
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
#  CLI entry-point                                                    #
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a DQN-family Othello agent.")

    p.add_argument("--board_size",          type=int,   default=5)
    p.add_argument("--num_episodes",        type=int,   default=5_000)
    p.add_argument("--epsilon_start",       type=float, default=1.0)
    p.add_argument("--epsilon_end",         type=float, default=0.05)
    p.add_argument("--epsilon_decay",       type=float, default=0.999)
    p.add_argument("--opponent_type",       type=str,   default="mix",
                   choices=["random", "greedy", "heuristic", "minimax", "mix", "self-play"])
    p.add_argument("--learning_rate",       type=float, default=1e-3)
    p.add_argument("--gamma",               type=float, default=0.99)
    p.add_argument("--batch_size",          type=int,   default=64)
    p.add_argument("--buffer_capacity",     type=int,   default=50_000)
    p.add_argument("--target_update_freq",  type=int,   default=500)
    p.add_argument("--learning_starts",     type=int,   default=1_000)
    p.add_argument("--heuristic_weight",    type=float, default=0.0,
                   help="Heuristic bonus weight. 0 = classic DQN, >0 = guided (HDQN).")
    p.add_argument("--use_per",             action="store_true",
                   help="Use Prioritized Experience Replay (PDQN).")
    p.add_argument("--per_alpha",           type=float, default=0.6)
    p.add_argument("--per_beta_start",      type=float, default=0.4)
    p.add_argument("--per_beta_frames",     type=int,   default=100_000)
    p.add_argument("--model_path",          type=str,   default="models/othello_agent.pth")
    p.add_argument("--load_model_path",     type=str,   default=None)
    p.add_argument("--self_play_update_freq", type=int, default=500)
    p.add_argument("--print_every",         type=int,   default=100)
    p.add_argument("--eval_every",          type=int,   default=250)
    p.add_argument("--save_every",          type=int,   default=500)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--no_double_dqn",       action="store_true",
                   help="Disable Double DQN (not recommended).")

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
        self_play_update_freq=args.self_play_update_freq,
        print_every=args.print_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
    )
