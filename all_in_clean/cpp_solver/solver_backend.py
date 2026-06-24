"""solver_backend.py
================================================================================
The SINGLE public import surface for the 6x6 Othello solver (plan §3.1, §10
step 4). Everything downstream (step 5's book builder and StudentAgent) imports
from here and never touches `othello_cpp` directly.

Behaviour
---------
* Prefer the compiled C++ backend: try ``import othello_cpp`` and re-export its
  ``Solver`` + free functions; ``BACKEND == "cpp"``.
* If that import fails (no/incompatible .so/.pyd), or if the env var
  ``OTHELLO_FORCE_PYTHON=1`` is set (the test hook for the fallback path), fall
  back to a pure-Python solver that WRAPS minimax_agent's already-validated
  exact search; ``BACKEND == "python"``.
* The fallback is SILENT, LOGGED ONCE, and NEVER FATAL: importing this module
  always succeeds. A single ``logging.warning`` is emitted when the fallback is
  selected; nothing is printed and nothing is raised.

Public surface (byte-identical across both backends; masks are mover-frame
uint64, cell = row*6+col, pass = 36):

    Solver(tt_bits=22)
        .solve_exact(me, opp) -> int      # signed disc margin (sign == WLD)
        .solve_wld(me, opp)   -> int      # sign-correct value
        .clear_tt()
        .set_use_tt(on: bool)
        .set_ordering(on: bool)
        .nodes                            # readable: nodes of last solve
    gen_moves(me, opp)        -> int      # bitmask of legal placing squares
    legal_actions(me, opp)    -> list[int]
    apply_move(me, opp, cell) -> (me2, opp2)
    is_terminal(me, opp)      -> bool
    count(x)                  -> int
    Symmetry operations (step 5a):
    permute_mask(mask, transform)     -> int
    apply_symmetry(me, opp, t)       -> (me2, opp2)
    canonicalize(me, opp)            -> (canon_me, canon_opp, transform)
    inverse_transform(cell, t)       -> int
    pack_key(me, opp)                -> int
    unpack_key(key)                  -> (me, opp)
    N_SYMS                            -> 8
    BACKEND                   -> "cpp" | "python"

Speed reality (NOT a bug): pure-Python exact solving caps at ~14-16 empties in
15 s (plan §2). On the fallback path the live solver therefore CANNOT clear the
E_exact=18 book frontier in time -- that is expected and is exactly why the
runtime agent (step 5) keeps a heuristic safety move. The fallback's job here is
correctness-parity with the oracle + never-crash, not speed.
"""

import logging
import os
import sys

_log = logging.getLogger("othello.solver_backend")

# Ensure this directory is on sys.path so the .so/.pyd and sibling modules are
# found regardless of where the importer lives.
_solver_dir = os.path.dirname(os.path.abspath(__file__))
if _solver_dir not in sys.path:
    sys.path.insert(0, _solver_dir)

# On Windows, the .pyd may depend on python3.dll / python314.dll and MinGW
# runtime DLLs.  Ensure those directories are in the DLL search path.
if sys.platform == "win32":
    _python_dll_dir = os.path.dirname(sys.executable)
    if os.path.isdir(_python_dll_dir):
        os.add_dll_directory(_python_dll_dir)
    # Also try Strawberry Perl / msys2 / conda mingw paths.
    for _cand in [r"D:\Strawberry\c\bin"]:
        if os.path.isdir(_cand):
            os.add_dll_directory(_cand)

# Board constants (must match environment.py / bitboard6.hpp).
CELLS = 36
PASS = CELLS
_FULL = (1 << CELLS) - 1

_FORCE_PYTHON = os.environ.get("OTHELLO_FORCE_PYTHON", "0") == "1"


# ===========================================================================
# Pure-Python fallback (plan §4 of the step-4 prompt). REUSES the validated
# minimax_agent search; it does not reimplement or edit it, so value-parity
# with the step-2 oracle holds BY CONSTRUCTION.
# ===========================================================================
def _build_python_surface():
    import math

    from minimax_agent import MinimaxAgent, WIN_BASE

    # One shared agent for the stateless free functions (bit helpers only).
    _helper = MinimaxAgent(board_size=6, use_tt=False, ordering=True)

    def _empties(me, opp):
        return CELLS - bin(me | opp).count("1")

    class _PySolver:
        """Mirror of othello_cpp.Solver, backed by minimax_agent._negamax.

        Each solve replays cmd_solveoracle's exact recipe (full window,
        use_tt=False, depth = 2*empties + 8) and the SAME win-base -> margin
        conversion, so margins are identical to the differential oracle.

        tt_bits / set_use_tt are accepted for API parity but the fallback keeps
        the agent's TT OFF on purpose: minimax_agent's depth-keyed cache stores
        fail-soft bounds without bound flags, so it is only SOUND with TT off
        (see differential.cmd_solveoracle). Correctness wins over speed here.
        """

        def __init__(self, tt_bits=22):
            self._tt_bits = tt_bits          # accepted, intentionally unused
            self._ordering = True
            # Reused across moves (parity with the C++ persistent solver); the
            # per-solve state is reset each call exactly like the oracle.
            self._agent = MinimaxAgent(board_size=6, use_tt=False,
                                       ordering=self._ordering)
            self.nodes = 0

        def set_use_tt(self, on):
            # No-op by design: enabling the agent's unsound TT could corrupt
            # values. Recorded only so the surface matches the C++ Solver.
            self._tt_requested = bool(on)

        def set_ordering(self, on):
            self._ordering = bool(on)
            self._agent.ordering = bool(on)

        def clear_tt(self):
            self._agent.tt = {}

        def solve_exact(self, me, opp):
            a = self._agent
            depth = 2 * _empties(me, opp) + 8
            a.nodes = 0
            a.cutoffs = 0
            a.deadline = None
            a.tt = {}
            v = a._negamax(me, opp, depth, -math.inf, math.inf)
            self.nodes = a.nodes
            # win-base score -> signed margin: copied verbatim from
            # differential.cmd_solveoracle (do not "simplify").
            if v > 0:
                return int(v - WIN_BASE)
            elif v < 0:
                return int(v + WIN_BASE)
            else:
                return 0

        def solve_wld(self, me, opp):
            # The exact margin's SIGN is the WLD answer; magnitude is irrelevant
            # to WLD callers. No narrow window -- the fallback optimises for
            # correctness, not pruning speed.
            return self.solve_exact(me, opp)

    def _gen_moves(me, opp):
        return _helper._gen_moves(me, opp)

    def _legal_actions(me, opp):
        m = _helper._gen_moves(me, opp)
        if m == 0:
            return [PASS]
        out = []
        while m:
            lsb = m & -m
            out.append(lsb.bit_length() - 1)   # ascending cell indices
            m ^= lsb
        return out

    def _apply_move(me, opp, cell):
        # mover's frame, no side swap -- matches bb6::apply_move_cell.
        return _helper._apply(me, opp, 1 << cell)

    def _is_terminal(me, opp):
        return _helper._gen_moves(me, opp) == 0 and \
            _helper._gen_moves(opp, me) == 0

    def _count(x):
        return bin(x & _FULL).count("1")

    return {
        "Solver": _PySolver,
        "gen_moves": _gen_moves,
        "legal_actions": _legal_actions,
        "apply_move": _apply_move,
        "is_terminal": _is_terminal,
        "count": _count,
    }


# ===========================================================================
# Backend selection: try C++ first (unless forced), else fall back. Importing
# this module ALWAYS succeeds; the fallback warns exactly once.
# ===========================================================================
BACKEND = None
Solver = None
gen_moves = legal_actions = apply_move = is_terminal = count = None
permute_mask = apply_symmetry = canonicalize = None
inverse_transform = pack_key = unpack_key = None
N_SYMS = 8

_have_cpp = False
_cpp = None
if not _FORCE_PYTHON:
    try:
        import othello_cpp as _cpp
    except ImportError:
        _cpp = None

if _cpp is not None:
    try:
        Solver = _cpp.Solver
        gen_moves = _cpp.gen_moves
        legal_actions = _cpp.legal_actions
        apply_move = _cpp.apply_move
        is_terminal = _cpp.is_terminal
        count = _cpp.count
        _have_cpp = True
    except AttributeError as _e:
        _log.warning("solver_backend: C++ module corrupted (missing %s)", _e)
        _cpp = None

# Symmetry: prefer C++ bindings, fall back to validated symmetry.py.
import symmetry as _sym
try:
    if _cpp is not None:
        permute_mask = _cpp.permute_mask
        apply_symmetry = _cpp.apply_symmetry
        canonicalize = _cpp.canonicalize
        inverse_transform = _cpp.inverse_transform
        pack_key = _cpp.pack_key
        unpack_key = _cpp.unpack_key
        _sym_ok = True
    else:
        _sym_ok = False
except AttributeError:
    _sym_ok = False

if not _sym_ok:
    permute_mask = _sym.permute_mask
    apply_symmetry = _sym.apply_symmetry
    canonicalize = _sym.canonicalize
    inverse_transform = _sym.inverse_transform
    pack_key = _sym.pack_key
    unpack_key = _sym.unpack_key

if _have_cpp:
    BACKEND = "cpp"
else:
    _surface = _build_python_surface()
    Solver = _surface["Solver"]
    gen_moves = _surface["gen_moves"]
    legal_actions = _surface["legal_actions"]
    apply_move = _surface["apply_move"]
    is_terminal = _surface["is_terminal"]
    count = _surface["count"]
    BACKEND = "python"
    _reason = "forced via OTHELLO_FORCE_PYTHON=1" if _FORCE_PYTHON \
        else "othello_cpp extension not importable"
    _log.warning(
        "solver_backend: using pure-Python fallback (%s). It is value-correct "
        "but slow (exact solving caps ~14-16 empties in 15 s) and cannot clear "
        "the E_exact=18 frontier live; step 5 keeps a heuristic safety move.",
        _reason,
    )

__all__ = [
    "Solver", "gen_moves", "legal_actions", "apply_move", "is_terminal",
    "count", "BACKEND", "CELLS", "PASS",
    "permute_mask", "apply_symmetry", "canonicalize", "inverse_transform",
    "pack_key", "unpack_key", "N_SYMS",
]
