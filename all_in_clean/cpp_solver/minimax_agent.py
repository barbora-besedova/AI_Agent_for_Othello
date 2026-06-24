"""
Bitboard minimax agent for Othello/Reversi with alpha-beta pruning.

Design notes
------------
* Board is packed into a single Python int: bit (row*N + col). For N<=8 this
  fits comfortably (<=64 bits). Move generation and flipping are pure bit ops.
* Search is NEGAMAX + alpha-beta. Negamax keeps the code symmetric: the value
  is always from the side-to-move's perspective, child = -search(child).
* ITERATIVE DEEPENING: we search depth 1, then 2, ... up to `max_depth`
  (or until the game tree is fully resolved). This is what makes per-layer
  timing meaningful and is also good practice: the best move from depth d-1
  is searched first at depth d, which sharpens pruning.
* No tree is stored. Recursion holds only the current root-to-leaf path on the
  call stack (O(depth) memory). An optional transposition table can be enabled
  for speed, but it is OFF by default to honour "store only what is necessary".
* TIME BUDGET: if `time_limit` is set, a partially-finished layer is abandoned
  and the best move from the last *fully completed* layer is returned. This
  turns the agent into a usable anytime player even when full depth is hopeless.

The agent only consumes the observation dict your OthelloEnv produces:
    board_abs, current_player, legal_actions, pass_action, board_size.
"""

import time
import math
import random

INF = math.inf
WIN_BASE = 1_000_000  # terminal scores dominate any heuristic value


class _Timeout(Exception):
    pass


class MinimaxAgent:
    def __init__(self, board_size=6, max_depth=None, time_limit=None,
                 debug=False, use_tt=False, ordering=True, seed=None):
        """
        board_size : N (4..8)
        max_depth  : plies to search. None  -> search to the end of the game
                     (the "lowest layer": terminal nodes). This is the default.
        time_limit : seconds per move. None -> no limit (search until max_depth
                     or terminal, which for the opening may never return).
        debug      : print a per-layer timing/stats table for each move.
        use_tt     : enable a transposition table (faster, uses memory).
        ordering   : enable move ordering (corners first + previous-best first).
        """
        self.N = board_size
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.debug = debug
        self.use_tt = use_tt
        self.ordering = ordering
        self.pass_action = board_size * board_size
        self.rng = random.Random(seed)

        N = board_size
        self.full = (1 << (N * N)) - 1
        self.dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        # Per-direction (shift, origin_mask). origin_mask marks cells whose
        # neighbour in that direction is still on the board, which kills wrap.
        self.shifts = []
        for dr, dc in self.dirs:
            s = dr * N + dc
            m = 0
            for r in range(N):
                for c in range(N):
                    if 0 <= r + dr < N and 0 <= c + dc < N:
                        m |= 1 << (r * N + c)
            self.shifts.append((s, m))

        # Static square weights for move ordering (corners great, X/C squares bad).
        self.sq_weight = self._build_square_weights(N)

        # Stats for the most recent select_action call.
        self.last_stats = []

    # ---------- bit helpers ----------
    def _shift(self, x, s, m):
        x &= m
        return (x << s) & self.full if s > 0 else (x >> (-s))

    def _gen_moves(self, me, opp):
        full = self.full
        empty = full & ~(me | opp)
        moves = 0
        for s, m in self.shifts:
            # walk: opp cells adjacent to `me` along this direction, extended
            x = ((me & m) << s) & full & opp if s > 0 else ((me & m) >> (-s)) & opp
            for _ in range(self.N - 1):
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

    # ---------- evaluation ----------
    def _build_square_weights(self, N):
        w = [[1] * N for _ in range(N)]
        n = N - 1
        for (r, c) in [(0,0),(0,n),(n,0),(n,n)]:
            w[r][c] = 50                       # corners are very strong
        # squares adjacent to a corner are dangerous (they give the corner away)
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
        # disc differential + cheap positional + mobility, from side-to-move view
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

    def _terminal_score(self, me, opp):
        diff = me.bit_count() - opp.bit_count()
        if diff > 0:
            return WIN_BASE + diff
        if diff < 0:
            return -WIN_BASE + diff
        return 0

    # ---------- search ----------
    def _ordered_moves(self, moves, best_first=0):
        ms = []
        x = moves
        while x:
            lsb = x & -x
            ms.append(lsb)
            x ^= lsb
        if self.ordering:
            ms.sort(key=lambda b: self.sq_weight.get(b, 1), reverse=True)
        if best_first:
            if best_first in ms:
                ms.remove(best_first)
            ms.insert(0, best_first)
        return ms

    def _negamax(self, me, opp, depth, alpha, beta):
        self.nodes += 1
        if self.deadline and (self.nodes & 2047) == 0 and time.perf_counter() > self.deadline:
            raise _Timeout
        moves = self._gen_moves(me, opp)
        if moves == 0:
            if self._gen_moves(opp, me) == 0:
                return self._terminal_score(me, opp)   # both pass -> game over
            return -self._negamax(opp, me, depth - 1, -beta, -alpha)  # forced pass
        if depth <= 0:
            return self._heuristic(me, opp)
        if self.use_tt:
            key = (me, opp, depth)
            hit = self.tt.get(key)
            if hit is not None:
                return hit
        best = -INF
        for mv in self._ordered_moves(moves):
            nme, nopp = self._apply(me, opp, mv)
            val = -self._negamax(nopp, nme, depth - 1, -beta, -alpha)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self.cutoffs += 1
                break
        if self.use_tt:
            self.tt[(me, opp, depth)] = best
        return best

    def _search_root(self, me, opp, depth, prev_best):
        moves = self._gen_moves(me, opp)
        if moves == 0:
            return self.pass_action, 0
        best_move, best_val = None, -INF
        alpha, beta = -INF, INF
        for mv in self._ordered_moves(moves, best_first=prev_best):
            nme, nopp = self._apply(me, opp, mv)
            val = -self._negamax(nopp, nme, depth - 1, -beta, -alpha)
            if val > best_val:
                best_val, best_move = val, mv
            if best_val > alpha:
                alpha = best_val
        return best_move, best_val

    # ---------- public API ----------
    def _obs_to_bitboards(self, observation):
        board = observation["board_abs"]
        player = observation["current_player"]
        N = self.N
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

    def select_action(self, observation):
        legal = observation["legal_actions"]
        if self.pass_action in legal and len(legal) == 1:
            self.last_stats = []
            return self.pass_action

        me, opp = self._obs_to_bitboards(observation)
        empties = self.full.bit_count() - (me | opp).bit_count()
        # Default: search to the lowest layer (terminal). Bounded by #empties.
        target = self.max_depth if self.max_depth is not None else empties

        self.deadline = (time.perf_counter() + self.time_limit) if self.time_limit else None
        self.tt = {}
        self.last_stats = []

        best_move_bit, prev_best = None, 0
        best_action = legal[0] if legal[0] != self.pass_action else self.pass_action

        for depth in range(1, target + 1):
            self.nodes = 0
            self.cutoffs = 0
            t0 = time.perf_counter()
            try:
                mv, val = self._search_root(me, opp, depth, prev_best)
            except _Timeout:
                self.last_stats.append((depth, None, None, self.nodes,
                                        self.cutoffs, time.perf_counter() - t0, False))
                break
            dt = time.perf_counter() - t0
            best_move_bit = mv
            prev_best = mv if isinstance(mv, int) and mv != self.pass_action else 0
            if mv == self.pass_action:
                best_action = self.pass_action
            else:
                best_action = (mv.bit_length() - 1)
            self.last_stats.append((depth, best_action, val, self.nodes,
                                    self.cutoffs, dt, True))
            # If a forced result (win/loss/draw proven) is found, no point going deeper.
            if abs(val) >= WIN_BASE:
                break

        if self.debug:
            self._print_stats(observation, target)
        return best_action

    def _print_stats(self, observation, target):
        N = self.N
        print(f"\n[debug] move for player {observation['current_player']}  "
              f"empties={self.full.bit_count() - sum(1 for r in range(N) for c in range(N) if observation['board_abs'][r,c]!=0)}  "
              f"target_depth={target}  tt={'on' if self.use_tt else 'off'}")
        print(f"  {'depth':>5} {'best':>6} {'value':>10} {'nodes':>12} "
              f"{'cutoffs':>10} {'time_s':>10} {'nodes/s':>12} {'done':>5}")
        for (d, a, v, n, cut, dt, done) in self.last_stats:
            nps = n / dt if dt > 0 else 0
            astr = "pass" if a == self.pass_action else (f"{a//N},{a%N}" if a is not None else "-")
            vstr = "-" if v is None else (f"{v:+d}" if abs(v) < WIN_BASE else ("WIN" if v > 0 else "LOSS"))
            print(f"  {d:>5} {astr:>6} {vstr:>10} {n:>12,} {cut:>10,} "
                  f"{dt:>10.4f} {nps:>12,.0f} {str(done):>5}")
