"""
student_agent_q_search.py — Competition submission.

Minimax tree search where leaf nodes are evaluated by the trained Q-network
instead of a hand-crafted heuristic (Q-value tree search).

    agent = StudentAgent(board_size=6, checkpoint_path="models/best.pth")
    action = agent.select_action(observation)
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Identical to q_network.py — must match training architecture."""

    def __init__(self, board_size: int, channels: int = 64, hidden_size: int = 128):
        super().__init__()
        self.output_size = board_size * board_size + 1
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * board_size * board_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, self.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(start_dim=1))


class StudentAgent:

    def __init__(
        self,
        board_size: int,
        checkpoint_path: str | None = None,
        depth: int = 2,
    ):
        self.board_size  = board_size
        self.pass_action = board_size * board_size
        self.action_size = board_size * board_size + 1
        self.depth       = depth
        self.device      = torch.device("cpu")

        self.q_net = QNetwork(board_size).to(self.device)
        self._model_loaded = False

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.q_net.load_state_dict(state_dict)
            self._model_loaded = True

        self.q_net.eval()

        # minimax helpers
        self._dirs    = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        last = board_size - 1
        self._corners = {(0,0),(0,last),(last,0),(last,last)}
        self._edges   = {(r,c) for i in range(board_size)
                         for r,c in [(0,i),(last,i),(i,0),(i,last)]}

    def encode_state(self, observation: dict) -> np.ndarray:
        return np.expand_dims(observation["board"].astype(np.float32), axis=0)

    def select_action(self, observation: dict) -> int:
        legal: List[int] = observation["legal_actions"]
        if not legal:
            return self.pass_action
        if legal == [self.pass_action]:
            return self.pass_action

        if not self._model_loaded:
            return self._heuristic(observation, legal)

        board  = observation["board_abs"]
        player = observation["current_player"]

        best_action = legal[0]
        best_score  = -np.inf

        for action in legal:
            if action == self.pass_action:
                continue
            next_board = self._apply(board, player, action)
            score = self._search(next_board, player, self.depth - 1,
                                 -np.inf, np.inf, False)
            if score > best_score:
                best_score  = score
                best_action = action

        return best_action

    # ── Q-network leaf evaluation ────────────────────────────────────

    def _q_eval(self, board: np.ndarray, player: int) -> float:
        """
        Ask the Q-network: how good is this position for `player`?

        The board is stored as absolute values (1 / -1 / 0).
        We convert it to relative (from player's perspective) before
        passing to the network, exactly as during training.
        """
        board_relative = board * player          # flip if player == -1
        state = board_relative.astype(np.float32)
        state_t = torch.tensor(state, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()
        # best reachable Q-value = estimated value of the position
        return float(q_values.max())

    # ── minimax with alpha-beta ──────────────────────────────────────

    def _search(self, board: np.ndarray, player: int, depth: int,
                alpha: float, beta: float, maximizing: bool) -> float:
        cur   = player if maximizing else -player
        moves = self._legal(board, cur)

        if not moves:
            opp = self._legal(board, -cur)
            if not opp:                          # terminal state
                my = int(np.sum(board == player))
                op = int(np.sum(board == -player))
                return 500.0 * (1 if my > op else (-1 if my < op else 0))
            # forced pass — switch sides, keep depth
            return self._search(board, player, depth, alpha, beta, not maximizing)

        if depth == 0:
            # leaf node — evaluate with Q-network instead of hand-crafted heuristic
            return self._q_eval(board, player)

        if maximizing:
            best = -np.inf
            for a in moves:
                nb   = self._apply(board, cur, a)
                val  = self._search(nb, player, depth - 1, alpha, beta, False)
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        else:
            best = np.inf
            for a in moves:
                nb   = self._apply(board, cur, a)
                val  = self._search(nb, player, depth - 1, alpha, beta, True)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best

    # ── board helpers ────────────────────────────────────────────────

    def _inb(self, r: int, c: int) -> bool:
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def _legal(self, board: np.ndarray, player: int) -> List[int]:
        moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != 0:
                    continue
                for dr, dc in self._dirs:
                    nr, nc = r + dr, c + dc
                    if not self._inb(nr, nc) or board[nr, nc] != -player:
                        continue
                    nr += dr; nc += dc
                    while self._inb(nr, nc) and board[nr, nc] == -player:
                        nr += dr; nc += dc
                    if self._inb(nr, nc) and board[nr, nc] == player:
                        moves.append(r * self.board_size + c)
                        break
        return moves

    def _apply(self, board: np.ndarray, player: int, action: int) -> np.ndarray:
        nb = board.copy()
        r, c = divmod(action, self.board_size)
        nb[r, c] = player
        for dr, dc in self._dirs:
            cands = []
            nr, nc = r + dr, c + dc
            while self._inb(nr, nc) and nb[nr, nc] == -player:
                cands.append((nr, nc)); nr += dr; nc += dc
            if cands and self._inb(nr, nc) and nb[nr, nc] == player:
                for fr, fc in cands:
                    nb[fr, fc] = player
        return nb

    # ── heuristic fallback (no checkpoint) ──────────────────────────

    def _heuristic(self, observation: dict, legal: List[int]) -> int:
        n, last = self.board_size, self.board_size - 1
        board = observation["board"]
        x_sq  = {(0,1):(0,0),(1,0):(0,0),(1,1):(0,0),
                 (0,last-1):(0,last),(1,last):(0,last),(1,last-1):(0,last),
                 (last-1,0):(last,0),(last,1):(last,0),(last-1,1):(last,0),
                 (last-1,last):(last,last),(last,last-1):(last,last),
                 (last-1,last-1):(last,last)}
        best_a, best_s = legal[0], -np.inf
        for a in legal:
            if a == self.pass_action:
                s = -1.0
            else:
                r, c = divmod(a, n)
                s = 5.0 if (r,c) in self._corners else \
                   -4.0 if (r,c) in x_sq and board[x_sq[(r,c)][0]][x_sq[(r,c)][1]] == 0 else \
                    1.0 if r in (0,last) or c in (0,last) else 0.0
            if s > best_s: best_s, best_a = s, a
        return int(best_a)
