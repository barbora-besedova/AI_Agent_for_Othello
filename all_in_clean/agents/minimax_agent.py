# ======================================================================
# Minimax Agent  (with alpha-beta pruning, optional)
# ======================================================================

import numpy as np
import random

class MinimaxAgent:
    """
    Minimax with alpha-beta pruning and a simple heuristic evaluation.
    Depth 4 is fast enough for 5×5.
    """

    def __init__(self, board_size: int = 5, depth: int = 4):
        self.board_size = board_size
        self.depth = depth
        self.pass_action = board_size * board_size

        n = board_size - 1
        self.corner_cells = {(0,0),(0,n),(n,0),(n,n)}
        self.edge_cells = set()
        for i in range(board_size):
            self.edge_cells.update({(0,i),(n,i),(i,0),(i,n)})

        DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        self.directions = DIRECTIONS

    def _in_bounds(self, r, c):
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def _legal_moves(self, board, player):
        moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != 0:
                    continue
                for dr, dc in self.directions:
                    nr, nc = r + dr, c + dc
                    if not self._in_bounds(nr, nc) or board[nr, nc] != -player:
                        continue
                    nr += dr; nc += dc
                    while self._in_bounds(nr, nc) and board[nr, nc] == -player:
                        nr += dr; nc += dc
                    if self._in_bounds(nr, nc) and board[nr, nc] == player:
                        moves.append(r * self.board_size + c)
                        break
        return moves

    def _apply(self, board, player, action):
        nb = board.copy()
        if action == self.pass_action:
            return nb
        r = action // self.board_size
        c = action % self.board_size
        nb[r, c] = player
        for dr, dc in self.directions:
            candidate = []
            nr, nc = r + dr, c + dc
            while self._in_bounds(nr, nc) and nb[nr, nc] == -player:
                candidate.append((nr, nc))
                nr += dr; nc += dc
            if candidate and self._in_bounds(nr, nc) and nb[nr, nc] == player:
                for fr, fc in candidate:
                    nb[fr, fc] = player
        return nb

    def _evaluate(self, board, player):
        """Heuristic: disc count + corner bonus + mobility."""
        score = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                v = board[r, c] * player
                if (r, c) in self.corner_cells:
                    v *= 10
                elif (r, c) in self.edge_cells:
                    v *= 2
                score += v
        my_moves = len(self._legal_moves(board, player))
        opp_moves = len(self._legal_moves(board, -player))
        score += 3 * (my_moves - opp_moves)
        return score

    def _minimax(self, board, player, depth, alpha, beta, maximizing):
        legal = self._legal_moves(board, player if maximizing else -player)
        if not legal:
            opp_legal = self._legal_moves(board, -player if maximizing else player)
            if not opp_legal:
                # Terminal
                my = np.sum(board == player)
                opp = np.sum(board == -player)
                return 1000 * (1 if my > opp else (-1 if my < opp else 0))
            # Forced pass
            return self._minimax(board, player, depth,
                                 alpha, beta, not maximizing)

        if depth == 0:
            return self._evaluate(board, player)

        if maximizing:
            best = -np.inf
            for action in legal:
                nb = self._apply(board, player, action)
                val = self._minimax(nb, player, depth - 1, alpha, beta, False)
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        else:
            best = np.inf
            for action in legal:
                nb = self._apply(board, -player, action)
                val = self._minimax(nb, player, depth - 1, alpha, beta, True)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best

    def select_action(self, observation: dict) -> int:
        legal = observation["legal_actions"]
        if self.pass_action in legal and len(legal) == 1:
            return self.pass_action

        # board_abs is absolute; we need the player's absolute color
        board = observation["board_abs"]
        player = observation["current_player"]

        best_val = -np.inf
        best_actions = []

        for action in legal:
            if action == self.pass_action:
                continue

            nb = self._apply(board, player, action)

            val = self._minimax(
                nb,
                player,
                self.depth - 1,
                -np.inf,
                np.inf,
                False
            )

            if val > best_val:
                best_val = val
                best_actions = [action]

            elif val == best_val:
                best_actions.append(action)

        return random.choice(best_actions)