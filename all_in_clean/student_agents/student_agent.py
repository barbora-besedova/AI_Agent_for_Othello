"""
student_agent.py — Submission wrapper for the UPV Othello competition.

Satisfies the exact interface required by the assignment:

    agent = StudentAgent(board_size=6, checkpoint_path="models/best.pth")
    action = agent.select_action(observation)

The class wraps the trained DQNAgent (Guided + PER variant by default)
and exposes only the methods required by the evaluator.

No training occurs here — the model is loaded from disk and used in
inference mode only.
"""

from __future__ import annotations

import os
import sys

# ── Ensure project root is importable ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

from q_network import QNetwork


class StudentAgent:
    """
    Final competition agent.

    Parameters
    ----------
    board_size : int
        Side length of the board (e.g. 5 for a 5×5 game).
    checkpoint_path : str | None
        Path to the saved .pth checkpoint produced by DQNAgent.save().
        If None, the agent falls back to a simple greedy heuristic so
        that it is still valid (though weak) without a model file.
    """

    def __init__(
        self,
        board_size: int,
        checkpoint_path: str | None = None,
    ):
        self.board_size = board_size
        self.checkpoint_path = checkpoint_path
        self.action_size = board_size * board_size + 1

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load the Q-network
        self.q_net = QNetwork(board_size).to(self.device)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self._load(checkpoint_path)
        else:
            # No model available — warn silently and run without weights
            # (falls back to heuristic inside select_action)
            self.q_net = None

        if self.q_net is not None:
            self.q_net.eval()

    # ------------------------------------------------------------------ #
    #  Required interface methods                                          #
    # ------------------------------------------------------------------ #

    def encode_state(self, observation: dict) -> np.ndarray:
        """
        Convert an observation dict to a (1, H, W) float32 numpy array.

        The board is always relative to the current player
        (1 = my pieces, -1 = opponent, 0 = empty), which is exactly
        what the environment already provides in observation["board"].
        """
        board = observation["board"].astype(np.float32)
        return np.expand_dims(board, axis=0)   # (1, H, W)

    def select_action(self, observation: dict) -> int:
        """
        Choose a legal action for the current board state.

        Always returns a value from observation["legal_actions"].
        No epsilon-greedy exploration — pure greedy inference.
        """
        legal_actions: list[int] = observation["legal_actions"]

        # Safety: should never be empty, but guard anyway
        if not legal_actions:
            return observation["pass_action"]

        # Only pass available → pass
        if legal_actions == [observation["pass_action"]]:
            return observation["pass_action"]

        # ── Neural network inference ──────────────────────────────
        if self.q_net is not None:
            state = self.encode_state(observation)
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)                        # (1, 1, H, W)

            with torch.no_grad():
                q_values = self.q_net(state_tensor).squeeze(0).cpu().numpy()

            # Mask illegal actions with -inf
            masked_q = np.full(self.action_size, -np.inf, dtype=np.float32)
            masked_q[legal_actions] = q_values[legal_actions]

            best_action = int(np.argmax(masked_q))

            # Safety fallback (should not trigger)
            if best_action not in legal_actions:
                best_action = legal_actions[0]

            return best_action

        # ── Fallback: heuristic (no model loaded) ─────────────────
        return self._heuristic_action(observation, legal_actions)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load(self, path: str) -> None:
        """Load weights from a DQNAgent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # DQNAgent.save() stores the dict with "model_state_dict"
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint.get("state_dict", checkpoint),
            )
        else:
            state_dict = checkpoint

        self.q_net.load_state_dict(state_dict)

    def _heuristic_action(
        self,
        observation: dict,
        legal_actions: list[int],
    ) -> int:
        """
        Simple positional heuristic used as a fallback when no model
        is available. Scores corners > edges > rest, penalises
        dangerous squares adjacent to empty corners.
        """
        n    = self.board_size
        last = n - 1
        board = observation["board"]
        pass_action = observation["pass_action"]

        corners   = {(0, 0), (0, last), (last, 0), (last, last)}
        dangerous = {
            (0, 1): (0, 0),       (1, 0): (0, 0),       (1, 1): (0, 0),
            (0, last-1): (0, last),(1, last): (0, last),  (1, last-1): (0, last),
            (last-1, 0): (last,0), (last, 1): (last, 0),  (last-1, 1): (last, 0),
            (last-1, last): (last, last), (last, last-1): (last, last),
            (last-1, last-1): (last, last),
        }

        best_action = legal_actions[0]
        best_score  = -float("inf")

        for action in legal_actions:
            if action == pass_action:
                score = -1.0
            else:
                row, col = divmod(action, n)
                score = 0.0
                if (row, col) in corners:
                    score += 5.0
                if row in (0, last) or col in (0, last):
                    score += 1.0
                if (row, col) in dangerous:
                    cr, cc = dangerous[(row, col)]
                    if board[cr][cc] == 0:
                        score -= 4.0

            if score > best_score:
                best_score  = score
                best_action = action

        return int(best_action)
