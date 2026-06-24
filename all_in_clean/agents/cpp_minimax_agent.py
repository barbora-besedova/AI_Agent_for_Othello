"""
C++ negamax solver agent for 6x6 Othello + practical iterative-deepening fallback.

Provides two agents:

1. ``CppSolverAgent`` — uses the compiled C++ extension for PERFECT play
   (solves to terminal).  Practical for endgame positions (< ~20 empties).

2. ``FastMinimaxAgent`` — pure-Python bitboard minimax with iterative
   deepening and configurable time limit.  Fast enough for full-game training
   against a DQN agent.  Supports any board size (4–8).

Both implement ``select_action(observation)`` and are drop-in replacements
for the agents in ``agents/``.
"""

import time
import math
import random

from cpp_solver.solver_backend import (
    Solver as CppSolver,
    apply_move as bb_apply,
    legal_actions as bb_legal,
    BACKEND,
)


# =========================================================================
# CppSolverAgent — perfect solver (terminal search, may be slow in opening)
# =========================================================================

class CppSolverAgent:
    """Perfect-play 6x6 solver using the C++ negamax engine.

    Solves every position to terminal (game-theoretically optimal).
    May be slow from the opening (30+ empties).

    Parameters
    ----------
    board_size : int
        Must be 6.
    use_wld : bool
        Narrow-window WLD mode (faster, sign-correct).  If False, uses
        full-window EXACT mode (slower, exact disc margin).
    tt_bits : int
        TT size = 2**tt_bits slots (default 22 = 4M).
    ordering : bool
        Enable move ordering.
    cpp_threshold : int
        Use the C++ solver for positions with this many or fewer empties.
        For positions with more empties, falls back to iterative deepening.
    time_limit : float
        Time limit per move for the iterative-deepening fallback (seconds).
    """
    def __init__(self, board_size: int, use_wld: bool = True,
                 tt_bits: int = 22, ordering: bool = True,
                 cpp_threshold: int = 18, time_limit: float = 2.0):
        if board_size != 6:
            raise ValueError(
                f"CppSolverAgent requires board_size=6 (got {board_size}). "
                "Use FastMinimaxAgent for other sizes."
            )
        self.board_size = board_size
        self.pass_action = board_size * board_size
        self._use_wld = use_wld
        self._cpp_threshold = cpp_threshold
        self._time_limit = time_limit
        self._solver = CppSolver(tt_bits=tt_bits)
        self._solver.set_ordering(ordering)
        self._backend = BACKEND
        self._fast = FastMinimaxAgent(board_size=board_size,
                                      time_limit=time_limit)

    def _obs_to_bitboards(self, observation: dict) -> tuple[int, int]:
        board = observation["board_abs"]
        player = observation["current_player"]
        N = self.board_size
        me = 0
        opp = 0
        for r in range(N):
            row = board[r]
            for c in range(N):
                v = int(row[c])
                if v == 0:
                    continue
                bit = 1 << (r * N + c)
                if v == player:
                    me |= bit
                else:
                    opp |= bit
        return me, opp

    def _count_empties(self, observation: dict) -> int:
        return int((observation["board_abs"] == 0).sum())

    def select_action(self, observation: dict) -> int:
        if self._count_empties(observation) <= self._cpp_threshold:
            return self._solve_perfect(observation)
        return self._fast.select_action(observation)

    def _solve_perfect(self, observation: dict) -> int:
        legal = observation["legal_actions"]
        pass_action = observation["pass_action"]
        if pass_action in legal and len(legal) == 1:
            return pass_action

        me, opp = self._obs_to_bitboards(observation)

        best_val = -10**9
        best_actions = []
        for action in legal:
            if action == pass_action:
                continue
            me2, opp2 = bb_apply(me, opp, action)
            if self._use_wld:
                val = -self._solver.solve_wld(opp2, me2)
            else:
                val = -self._solver.solve_exact(opp2, me2)
            if val > best_val:
                best_val = val
                best_actions = [action]
            elif val == best_val:
                best_actions.append(action)

        if not best_actions:
            return random.choice(legal)
        return random.choice(best_actions)


# =========================================================================
# FastMinimaxAgent — iterative-deepening bitboard minimax (training-ready)
# =========================================================================

class FastMinimaxAgent:
    """Fast bitboard minimax agent with iterative deepening and time limit.

    Uses pure-Python bitboard engine.  Supports any board size (4–8).
    Designed for practical full-game use during Q-learning training.

    Parameters
    ----------
    board_size : int
        Board side length (4–8).
    max_depth : int or None
        Maximum search depth.  None → search until time runs out or the
        game tree is fully resolved.
    time_limit : float
        Seconds per move.  The search stops at the next layer boundary.
    ordering : bool
        Move ordering (corners first + previous-best first).
    debug : bool
        Print per-layer timing statistics.
    """
    def __init__(self, board_size: int = 6, max_depth: int | None = None,
                 time_limit: float = 2.0, ordering: bool = True,
                 debug: bool = False):
        self.board_size = board_size
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.ordering = ordering
        self.debug = debug
        self.pass_action = board_size * board_size
        self.rng = random.Random()

        N = board_size
        self.full = (1 << (N * N)) - 1
        self.dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        self.shifts = []
        for dr, dc in self.dirs:
            s = dr * N + dc
            m = 0
            for r in range(N):
                for c in range(N):
                    if 0 <= r + dr < N and 0 <= c + dc < N:
                        m |= 1 << (r * N + c)
            self.shifts.append((s, m))

        self.sq_weight = self._build_square_weights(N)

    # ---- bitboard helpers ----
    def _shift(self, x, s, m):
        x &= m
        return (x << s) & self.full if s > 0 else (x >> (-s))

    def _gen_moves(self, me, opp):
        full = self.full
        empty = full & ~(me | opp)
        moves = 0
        for s, m in self.shifts:
            x = ((me & m) << s) & full & opp if s > 0 else ((me & m) >> (-s)) & opp
            for _ in range(self.board_size - 1):
                if not x:
                    break
                x |= (((x & m) << s) & full & opp) if s > 0 else (((x & m) >> (-s)) & opp)
            shifted = ((x & m) << s) & full if s > 0 else ((x & m) >> (-s))
            moves |= shifted & empty
        return moves

    def _apply(self, me, opp, pos):
        flips = 0
        for s, m in self.shifts:
            line = 0
            x = self._shift(pos, s, m) & opp
            while x:
                line |= x
                nxt = self._shift(x, s, m)
                if nxt & me:
                    flips |= line
                    break
                if not (nxt & opp):
                    break
                x = nxt
        return (me | pos | flips), (opp & ~flips)

    def _build_square_weights(self, N):
        w = [[1] * N for _ in range(N)]
        n = N - 1
        for (r, c) in [(0,0),(0,n),(n,0),(n,n)]:
            w[r][c] = 50
        for (cr, cc) in [(0,0),(0,n),(n,0),(n,n)]:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r, c = cr + dr, cc + dc
                    if 0 <= r < N and 0 <= c < N and w[r][c] != 50:
                        w[r][c] = -8
        flat = {}
        for r in range(N):
            for c in range(N):
                flat[1 << (r*N+c)] = w[r][c]
        return flat

    def _heuristic(self, me, opp):
        diff = me.bit_count() - opp.bit_count()
        my_moves = self._gen_moves(me, opp).bit_count()
        op_moves = self._gen_moves(opp, me).bit_count()
        mob = my_moves - op_moves
        pos = 0
        x = me
        while x:
            lsb = x & -x
            pos += self.sq_weight.get(lsb, 1)
            x ^= lsb
        x = opp
        while x:
            lsb = x & -x
            pos -= self.sq_weight.get(lsb, 1)
            x ^= lsb
        return diff + 2 * mob + pos

    def _ordered_moves(self, moves, best_first=0):
        ms = []
        x = moves
        while x:
            lsb = x & -x
            ms.append(lsb)
            x ^= lsb
        if self.ordering:
            ms.sort(key=lambda b: self.sq_weight.get(b, 1), reverse=True)
        if best_first and best_first in ms:
            ms.remove(best_first)
            ms.insert(0, best_first)
        return ms

    class _Timeout(Exception):
        pass

    def _negamax(self, me, opp, depth, alpha, beta):
        self.nodes += 1
        if self.deadline and (self.nodes & 2047) == 0 and time.perf_counter() > self.deadline:
            raise self._Timeout
        moves = self._gen_moves(me, opp)
        if moves == 0:
            if self._gen_moves(opp, me) == 0:
                diff = me.bit_count() - opp.bit_count()
                if diff > 0: return 1_000_000 + diff
                if diff < 0: return -1_000_000 + diff
                return 0
            return -self._negamax(opp, me, depth - 1, -beta, -alpha)
        if depth <= 0:
            return self._heuristic(me, opp)
        best = -10**9
        for mv in self._ordered_moves(moves):
            nme, nopp = self._apply(me, opp, mv)
            val = -self._negamax(nopp, nme, depth - 1, -beta, -alpha)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def _search_root(self, me, opp, depth, prev_best):
        moves = self._gen_moves(me, opp)
        if moves == 0:
            return self.pass_action, 0
        best_move, best_val = None, -10**9
        alpha, beta = -10**9, 10**9
        for mv in self._ordered_moves(moves, best_first=prev_best):
            nme, nopp = self._apply(me, opp, mv)
            val = -self._negamax(nopp, nme, depth - 1, -beta, -alpha)
            if val > best_val:
                best_val, best_move = val, mv
            if best_val > alpha:
                alpha = best_val
        return best_move, best_val

    def _obs_to_bitboards(self, observation):
        board = observation["board_abs"]
        player = observation["current_player"]
        N = self.board_size
        me = opp = 0
        for r in range(N):
            for c in range(N):
                v = int(board[r, c])
                if v == 0:
                    continue
                bit = 1 << (r * N + c)
                if v == player:
                    me |= bit
                else:
                    opp |= bit
        return me, opp

    def select_action(self, observation: dict) -> int:
        legal = observation["legal_actions"]
        pass_action = observation["pass_action"]
        if pass_action in legal and len(legal) == 1:
            return pass_action

        me, opp = self._obs_to_bitboards(observation)
        empties = self.full.bit_count() - (me | opp).bit_count()
        target = self.max_depth if self.max_depth is not None else empties

        self.deadline = time.perf_counter() + self.time_limit if self.time_limit else None

        best_move_bit = 0
        best_action = legal[0] if legal[0] != pass_action else pass_action

        for depth in range(1, target + 1):
            self.nodes = 0
            t0 = time.perf_counter()
            try:
                mv, val = self._search_root(me, opp, depth, best_move_bit)
            except self._Timeout:
                break
            best_move_bit = mv
            if mv == pass_action:
                best_action = pass_action
            else:
                best_action = mv.bit_length() - 1
            if abs(val) >= 1_000_000:
                break

        return best_action


__all__ = [
    "CppSolverAgent",
    "FastMinimaxAgent",
]
