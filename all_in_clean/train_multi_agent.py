"""
train_multi_agent.py — Train multiple DQN-family agents simultaneously.

Each episode randomly selects one trainable agent as the learner and pits it
against a randomly sampled opponent from the full opponent pool (including the
other trainable agents).

Usage examples
--------------
# Default: trains classic DQN, guided DQN, and PER DQN simultaneously
python train_multi_agent.py

# Custom: only two agents, 10 000 episodes
python train_multi_agent.py --num_episodes 10000 --agents dqn guided

# Load saved weights for specific agents
python train_multi_agent.py --load_dqn models/dqn.pth --load_guided models/guided.pth
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from environment import OthelloEnv
from agent import DQNAgent


# ------------------------------------------------------------------ #
#  Helpers                                                            #
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


def _call_select_action(agent, observation, epsilon=None):
    """Works with both DQNAgent (accepts epsilon) and rule-based agents (do not)."""
    if epsilon is None:
        return agent.select_action(observation)
    try:
        return agent.select_action(observation, epsilon=epsilon)
    except TypeError:
        return agent.select_action(observation)


def _store_transition(agent, obs, action, reward, next_obs, done):
    if hasattr(agent, "store_transition"):
        agent.store_transition(obs, action, reward, next_obs, done)
    elif hasattr(agent, "replay_buffer"):
        agent.replay_buffer.push(
            agent.encode_state(obs),
            action,
            reward,
            agent.encode_state(next_obs) if next_obs is not None else None,
            done,
            next_obs["legal_actions"] if next_obs is not None else [],
        )


def _train_step(agent):
    if hasattr(agent, "train_step"):
        agent.train_step()


def _update_target(agent):
    if hasattr(agent, "update_target_network"):
        agent.update_target_network()
    elif hasattr(agent, "target_net") and hasattr(agent, "q_net"):
        agent.target_net.load_state_dict(agent.q_net.state_dict())


def _save_agent(agent, path: str, board_size: int):
    if hasattr(agent, "save"):
        agent.save(path)
    else:
        torch.save(
            {"board_size": board_size, "model_state_dict": agent.q_net.state_dict()},
            path,
        )


# ------------------------------------------------------------------ #
#  Default agent factory                                              #
# ------------------------------------------------------------------ #

def make_default_agents(board_size: int, selected: List[str]) -> Dict[str, DQNAgent]:
    """
    Create a dict of trainable agents based on the selected names.

    Available names:
        "dqn"     — Classic double DQN
        "guided"  — Guided (heuristic) DQN  (heuristic_weight=0.2)
        "per"     — Prioritized DQN
        "guided_per" — Guided + Prioritized
    """
    factories = {
        "dqn": lambda: DQNAgent(board_size, use_per=False, heuristic_weight=0.0),
        "guided": lambda: DQNAgent(board_size, use_per=False, heuristic_weight=0.2),
        "per": lambda: DQNAgent(board_size, use_per=True, heuristic_weight=0.0),
        "guided_per": lambda: DQNAgent(board_size, use_per=True, heuristic_weight=0.2),
    }

    agents = {}
    for name in selected:
        if name not in factories:
            raise ValueError(
                f"Unknown agent '{name}'. Choose from: {list(factories.keys())}"
            )
        agents[name] = factories[name]()

    return agents


# ------------------------------------------------------------------ #
#  Main training function                                             #
# ------------------------------------------------------------------ #

def train_multi_agent(
    trainable_agents: Optional[Dict[str, Any]] = None,
    board_size: int = 5,
    num_episodes: int = 5_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.998,
    model_dir: str = "models/multi_agent",
    target_update_every: int = 500,
    print_every: int = 100,
    agent_names: Optional[List[str]] = None,
    load_paths: Optional[Dict[str, str]] = None,
    seed: int = 42,
) -> Dict:
    """
    Train multiple agents simultaneously via random episode assignment.

    Args:
        trainable_agents:  Pre-built agent dict (overrides agent_names).
        board_size:        Board side length.
        num_episodes:      Total training episodes.
        epsilon_start/end/decay: ε-greedy schedule (shared across all agents).
        model_dir:         Directory where checkpoints are saved.
        target_update_every: Hard-update target nets every N episodes.
        print_every:       Console log interval.
        agent_names:       Which agent types to create (used when
                           trainable_agents is None). Default: all four types.
        load_paths:        Optional dict mapping agent name → checkpoint path.
        seed:              Random seed.
    """
    os.makedirs(model_dir, exist_ok=True)
    set_seed(seed)

    opponent_classes = _import_opponents()

    # ---- build agents ----
    if trainable_agents is None:
        if agent_names is None:
            agent_names = ["dqn", "guided", "per", "guided_per"]
        trainable_agents = make_default_agents(board_size, agent_names)

    # Load optional pre-trained weights
    if load_paths:
        for name, path in load_paths.items():
            if name in trainable_agents and path and os.path.exists(path):
                trainable_agents[name].load(path)
                print(f"Loaded {name} from {path}")

    # ---- opponent pool ----
    # Rule-based opponents are lambdas to create a fresh instance each episode.
    opponent_pool: Dict[str, Callable] = {
        name: (lambda cls=cls: cls(board_size))
        for name, cls in opponent_classes.items()
    }
    # Add trainable agents themselves (they won't play against their own instance)
    for name, ag in trainable_agents.items():
        opponent_pool[name] = lambda ag=ag: ag

    env = OthelloEnv(board_size=board_size)
    epsilon = epsilon_start

    # Per-agent statistics
    stats: Dict[str, Dict] = {
        name: {
            "games": 0, "wins": 0, "draws": 0, "losses": 0,
            "rewards": [],
            "opponents": {},
        }
        for name in trainable_agents
    }

    # ================================================================== #
    #  Episode loop                                                        #
    # ================================================================== #
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False

        # Pick a random learner
        learner_name = random.choice(list(trainable_agents.keys()))
        learner = trainable_agents[learner_name]

        # Pick a random opponent (not the exact same instance)
        possible_opps = [k for k in opponent_pool if k != learner_name]
        opp_name = random.choice(possible_opps)
        opponent = opponent_pool[opp_name]()

        # Track per-learner opponent stats
        if opp_name not in stats[learner_name]["opponents"]:
            stats[learner_name]["opponents"][opp_name] = {
                "games": 0, "wins": 0, "draws": 0, "losses": 0
            }

        # Alternate colors
        learner_player = 1 if episode % 2 == 0 else -1

        # If learner is black (-1), opponent (player 1) moves first
        if env.current_player != learner_player:
            first = opponent if env.current_player != learner_player else learner
            obs, _, done, _ = env.step(_call_select_action(first, obs))

        episode_reward = 0.0

        while not done:
            state_obs = obs

            action = _call_select_action(learner, state_obs, epsilon=epsilon)
            obs_after_learner, _, done, info = env.step(action)

            if done:
                reward = compute_final_reward(info["winner"], learner_player)
                _store_transition(learner, state_obs, action, reward, obs_after_learner, True)
                _train_step(learner)
                episode_reward = reward
                break

            opp_action = _call_select_action(opponent, obs_after_learner)
            obs_after_opp, _, done, info = env.step(opp_action)

            reward = (
                compute_final_reward(info["winner"], learner_player)
                if done else 0.0
            )

            _store_transition(learner, state_obs, action, reward, obs_after_opp, done)
            _train_step(learner)

            obs = obs_after_opp
            episode_reward = reward

        # ε decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update stats
        stats[learner_name]["games"] += 1
        stats[learner_name]["rewards"].append(episode_reward)
        stats[learner_name]["opponents"][opp_name]["games"] += 1

        if episode_reward > 0:
            stats[learner_name]["wins"] += 1
            stats[learner_name]["opponents"][opp_name]["wins"] += 1
        elif episode_reward < 0:
            stats[learner_name]["losses"] += 1
            stats[learner_name]["opponents"][opp_name]["losses"] += 1
        else:
            stats[learner_name]["draws"] += 1
            stats[learner_name]["opponents"][opp_name]["draws"] += 1

        # Console log
        if episode % print_every == 0:
            recent = stats[learner_name]["rewards"][-100:]
            print(
                f"[{episode:>6}/{num_episodes}] "
                f"learner={learner_name:<12} opp={opp_name:<10} "
                f"ε={epsilon:.3f} | "
                f"reward={episode_reward:.1f} | "
                f"avg_r_100={np.mean(recent):.3f}"
            )

        # Hard-update all target networks
        if episode % target_update_every == 0:
            for ag in trainable_agents.values():
                _update_target(ag)

    # ---- save all agents ----
    for name, ag in trainable_agents.items():
        save_path = os.path.join(model_dir, f"{name}.pth")
        _save_agent(ag, save_path, board_size)
        print(f"Saved {name} → {save_path}")

    # ---- final summary ----
    print("\n── Final statistics ──")
    for name, s in stats.items():
        print(
            f"  {name:<14} games={s['games']:>5} "
            f"W/D/L={s['wins']}/{s['draws']}/{s['losses']}"
        )

    return {"agents": trainable_agents, "stats": stats}


# ------------------------------------------------------------------ #
#  CLI entry-point                                                    #
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-agent DQN Othello training.")

    p.add_argument("--board_size",           type=int,   default=5)
    p.add_argument("--num_episodes",         type=int,   default=5_000)
    p.add_argument("--epsilon_start",        type=float, default=1.0)
    p.add_argument("--epsilon_end",          type=float, default=0.05)
    p.add_argument("--epsilon_decay",        type=float, default=0.998)
    p.add_argument("--model_dir",            type=str,   default="models/multi_agent")
    p.add_argument("--target_update_every",  type=int,   default=500)
    p.add_argument("--print_every",          type=int,   default=100)
    p.add_argument("--seed",                 type=int,   default=42)
    p.add_argument(
        "--agents", nargs="+",
        default=["dqn", "guided", "per", "guided_per"],
        choices=["dqn", "guided", "per", "guided_per"],
        help="Which agent types to train simultaneously.",
    )
    p.add_argument("--load_dqn",             type=str,   default=None)
    p.add_argument("--load_guided",          type=str,   default=None)
    p.add_argument("--load_per",             type=str,   default=None)
    p.add_argument("--load_guided_per",      type=str,   default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    load_paths = {
        "dqn":        args.load_dqn,
        "guided":     args.load_guided,
        "per":        args.load_per,
        "guided_per": args.load_guided_per,
    }
    load_paths = {k: v for k, v in load_paths.items() if v is not None}

    train_multi_agent(
        board_size=args.board_size,
        num_episodes=args.num_episodes,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        model_dir=args.model_dir,
        target_update_every=args.target_update_every,
        print_every=args.print_every,
        agent_names=args.agents,
        load_paths=load_paths,
        seed=args.seed,
    )
