"""
student_agent_hybrid.py — Competition submission.

Minimax tree search where leaf nodes are evaluated by a combination
of the trained Q-network and a hand-crafted positional heuristic.

    leaf_score = q_weight * Q_net(position) + (1 - q_weight) * heuristic(position)

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
        q_weight: float = 0.7,      # váha Q-sítě v listovém ohodnocení
    ):
        self.board_size  = board_size
        self.pass_action = board_size * board_size
        self.action_size = board_size * board_size + 1
        self.depth       = depth
        self.q_weight    = q_weight
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
        self._dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
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
            return self._heuristic_action(observation, legal)

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

    # ── listové ohodnocení: Q-síť + heuristika ──────────────────────

    def _leaf_eval(self, board: np.ndarray, player: int) -> float:
        """
        Kombinuje Q-síť a ruční heuristiku pro ohodnocení listového uzlu.

        q_weight=0.7 → síť dominuje, heuristika zachrání taktické chyby
        q_weight=0.0 → čistá heuristika (pokud síť není načtena)
        q_weight=1.0 → čistá Q-síť (stejné jako q_search varianta)
        """
        h_score = self._heuristic_eval(board, player)

        if not self._model_loaded:
            return h_score

        # Q-síť: převeď na relativní board (z pohledu hráče) jako při tréninku
        board_relative = (board * player).astype(np.float32)
        state_t = torch.tensor(board_relative, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()
        q_score = float(q_values.max())

        return self.q_weight * q_score + (1.0 - self.q_weight) * h_score

    def _heuristic_eval(self, board: np.ndarray, player: int) -> float:
        """Ruční poziční heuristika: rohy, hrany, mobilita."""
        score = 0.0
        for r in range(self.board_size):
            for c in range(self.board_size):
                v = float(board[r, c] * player)
                if   (r, c) in self._corners: v *= 8.0
                elif (r, c) in self._edges:   v *= 2.0
                score += v
        score += 2.0 * (len(self._legal(board, player)) - len(self._legal(board, -player)))
        return score

    # ── minimax s alpha-beta ─────────────────────────────────────────

    def _search(self, board: np.ndarray, player: int, depth: int,
                alpha: float, beta: float, maximizing: bool) -> float:
        cur   = player if maximizing else -player
        moves = self._legal(board, cur)

        if not moves:
            opp = self._legal(board, -cur)
            if not opp:                          # terminální stav
                my = int(np.sum(board == player))
                op = int(np.sum(board == -player))
                return 500.0 * (1 if my > op else (-1 if my < op else 0))
            return self._search(board, player, depth, alpha, beta, not maximizing)

        if depth == 0:
            return self._leaf_eval(board, player)   # ← Q-síť + heuristika

        if maximizing:
            best = -np.inf
            for a in moves:
                nb    = self._apply(board, cur, a)
                val   = self._search(nb, player, depth - 1, alpha, beta, False)
                best  = max(best, val)
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

    def _heuristic_action(self, observation: dict, legal: List[int]) -> int:
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
