"""
Shared diagnostics for DQN training.

Provides:
    MetricsLogger  — CSV logging of scalar metrics
    record_game   — plays a game and returns a full transcript dict
    save_transcript — saves a transcript dict to JSON
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from environment import OthelloEnv


class MetricsLogger:
    """Write scalar metrics to a CSV file, one row per call to ``log``."""

    def __init__(self, path: str, fieldnames: List[str]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self.file = open(path, "w", encoding="utf-8")
        self.file.write(",".join(fieldnames) + "\n")
        self.file.flush()

    def log(self, **kwargs) -> None:
        vals = [str(kwargs.get(f, "")) for f in self.fieldnames]
        self.file.write(",".join(vals) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def record_game(
    agent,
    opponent,
    board_size: int = 6,
    record_q: bool = True,
    random_opening_plies: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Play one game and return a full transcript.

    ``agent`` plays as player 1, ``opponent`` as player -1.
    To record both colours, call twice swapping the arguments.

    Returns
    -------
    dict with keys:
        moves      — list of per-turn dicts
        winner     — 1 | -1 | 0
        disc_diff  — final disc margin from agent's perspective
        n_moves    — total moves played
        agent_player — which colour the agent played (1 or -1)
    """
    env = OthelloEnv(board_size=board_size)
    obs = env.reset()
    done = False

    for _ in range(random_opening_plies):
        if done:
            break
        legal = obs["legal_actions"]
        pa = obs.get("pass_action", board_size * board_size)
        non_pass = [a for a in legal if a != pa]
        action = list(np.random.choice(non_pass if non_pass else legal, 1))[0]
        obs, _, done, _ = env.step(action)

    moves: List[Dict] = []
    agent_player = 1

    turn = 0
    while not done:
        turn += 1
        if env.current_player == agent_player:
            mover = "agent"
            q_values = None
            if record_q and device is not None and hasattr(agent, "q_net"):
                state = np.expand_dims(obs["board"].astype(np.float32), axis=0)
                st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values = agent.q_net(st).squeeze(0).cpu().numpy().tolist()
            try:
                action = agent.select_action(obs, epsilon=0.0)
            except TypeError:
                action = agent.select_action(obs)
        else:
            mover = "opponent"
            q_values = None
            action = opponent.select_action(obs)

        entry = {
            "turn": turn,
            "mover": mover,
            "player": int(env.current_player),
            "board": obs["board"].tolist(),
            "action": int(action),
            "legal_actions": obs["legal_actions"],
        }
        if q_values is not None:
            entry["q_values"] = q_values
        moves.append(entry)

        obs, _, done, info = env.step(action)

    disc_diff = int(info["disc_diff"])
    if agent_player == -1:
        disc_diff = -disc_diff

    return {
        "moves": moves,
        "winner": info["winner"],
        "disc_diff": disc_diff,
        "n_moves": turn,
        "agent_player": agent_player,
    }


def save_transcript(transcript: Dict[str, Any], path: str) -> None:
    """Save a transcript dict to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, default=str)
