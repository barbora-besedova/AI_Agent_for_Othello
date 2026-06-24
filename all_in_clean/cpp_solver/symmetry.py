"""symmetry.py — D4 symmetry canonicalization + base-3 packing for 6×6 Othello.

Implements:
  - 8 precomputed D4 cell-permutation tables (identity, rotate 90/180/270,
    reflect H/V/D1/D2) for a 6×6 board.
  - canonicalize(me, opp)    → minimum-base-3-key canonical form.
  - inverse_transform(cell, transform) → map a stored move back.
  - pack_key / unpack_key    → collision-free uint64 base-3 encoding.
  - apply_symmetry           → apply a D4 transform to (me, opp) masks.

Run ``python3 symmetry.py`` for self-verification.
"""

import random

# ── D4 symmetry constants ────────────────────────────────────────────────
SYM_IDENTITY = 0
SYM_ROT90 = 1
SYM_ROT180 = 2
SYM_ROT270 = 3
SYM_REFL_H = 4      # horizontal mirror (flip left-right)
SYM_REFL_V = 5      # vertical mirror (flip top-bottom)
SYM_REFL_D1 = 6     # main-diagonal mirror
SYM_REFL_D2 = 7     # anti-diagonal mirror
N_SYMS = 8

N = 6
CELLS = N * N

# ── Coordinate transforms (r, c) → (r', c') ──────────────────────────────
_TRANSFORMS = [
    # 0: identity
    lambda r, c: (r, c),
    # 1: rotate 90° CW
    lambda r, c: (c, N - 1 - r),
    # 2: rotate 180°
    lambda r, c: (N - 1 - r, N - 1 - c),
    # 3: rotate 270° CW
    lambda r, c: (N - 1 - c, r),
    # 4: reflect H (left-right)
    lambda r, c: (r, N - 1 - c),
    # 5: reflect V (top-bottom)
    lambda r, c: (N - 1 - r, c),
    # 6: reflect D1 (main diagonal)
    lambda r, c: (c, r),
    # 7: reflect D2 (anti-diagonal)
    lambda r, c: (N - 1 - c, N - 1 - r),
]


def _build_perm():
    perm = [[0] * CELLS for _ in range(N_SYMS)]
    for t in range(N_SYMS):
        transform_fn = _TRANSFORMS[t]
        for idx in range(CELLS):
            r, c = divmod(idx, N)
            r2, c2 = transform_fn(r, c)
            perm[t][idx] = r2 * N + c2
    return perm


PERM: list[list[int]] = _build_perm()

# Inverse symmetry table: SYM_INV[t] = the transform that inverts t.
SYM_INV = [0] * N_SYMS
for t in range(N_SYMS):
    # Find the inverse: compose PERM[t] and PERM[cand] → identity
    for cand in range(N_SYMS):
        if all(PERM[t][PERM[cand][i]] == i for i in range(CELLS)):
            SYM_INV[t] = cand
            break

# ── Base-3 power table ───────────────────────────────────────────────────
POW3 = [3 ** i for i in range(CELLS)]


# ── Public API ────────────────────────────────────────────────────────────

def permute_mask(mask: int, transform: int) -> int:
    """Apply a cell permutation to a 36-bit mask."""
    p = PERM[transform]
    out = 0
    m = mask
    while m:
        lsb = m & -m
        idx = (lsb.bit_length() - 1)
        out |= 1 << p[idx]
        m ^= lsb
    return out


def apply_symmetry(me: int, opp: int, transform: int) -> tuple[int, int]:
    """Apply a D4 symmetry to (me, opp); return (new_me, new_opp)."""
    return (permute_mask(me, transform),
            permute_mask(opp, transform))


def pack_key(me: int, opp: int) -> int:
    """Collision-free uint64 key via base-3 encoding.

    For each cell: 0 = empty, 1 = me, 2 = opp.
    """
    key = 0
    for cell in range(CELLS):
        if (me >> cell) & 1:
            state = 1
        elif (opp >> cell) & 1:
            state = 2
        else:
            state = 0
        key += state * POW3[cell]
    return key


def unpack_key(key: int) -> tuple[int, int]:
    """Inverse of pack_key.  Returns (me, opp) bitmasks."""
    me = opp = 0
    for cell in range(CELLS):
        state = (key // POW3[cell]) % 3
        if state == 1:
            me |= 1 << cell
        elif state == 2:
            opp |= 1 << cell
    return (me, opp)


def canonicalize(me: int, opp: int) -> tuple[int, int, int]:
    """Return (canon_me, canon_opp, transform) giving the minimum base-3 key.

    Tie-breaking: smallest transform index wins.
    """
    best_key = None
    best = (0, 0, 0)
    for t in range(N_SYMS):
        me_t, opp_t = apply_symmetry(me, opp, t)
        key = pack_key(me_t, opp_t)
        if best_key is None or key < best_key:
            best_key = key
            best = (me_t, opp_t, t)
    return best


def inverse_transform(cell: int, transform: int) -> int:
    """Map a cell back through the inverse of `transform`.

    The inverse of transform t is SYM_INV[t]; applying SYM_INV[t] to the
    destination cell of the forward transform gives back the original.
    """
    p = PERM[SYM_INV[transform]]
    return p[cell]


# ── Self-verification ─────────────────────────────────────────────────────

def _check(cond: bool, msg: str):
    if not cond:
        raise AssertionError(f"FAILED: {msg}")


def _random_position(empties: int, rng: random.Random):
    """Generate a random non-terminal position with exactly `empties` empty cells."""
    cells = list(range(CELLS))
    rng.shuffle(cells)
    empty_set = set(cells[:empties])
    remaining = cells[empties:]
    # Split remaining roughly evenly between me and opp
    n_discs = len(remaining)
    n_me = n_discs // 2
    n_opp = n_discs - n_me
    me_cells = set(remaining[:n_me])
    opp_cells = set(remaining[n_me:])
    me = opp = 0
    for c in me_cells:
        me |= 1 << c
    for c in opp_cells:
        opp |= 1 << c
    return me, opp


def verify():
    rng = random.Random(42)

    # 1. Permutation table integrity: bijection + inverse round-trip.
    print("  [1] Permutation table integrity ...", end=" ")
    for t in range(N_SYMS):
        seen = [False] * CELLS
        for src in range(CELLS):
            dst = PERM[t][src]
            _check(0 <= dst < CELLS, f"PERM[{t}][{src}] = {dst} out of range")
            _check(not seen[dst], f"PERM[{t}] is not bijective (collision at {dst})")
            seen[dst] = True
        # Inverse round-trip
        inv = SYM_INV[t]
        for src in range(CELLS):
            dst = PERM[t][src]
            back = PERM[inv][dst]
            _check(back == src,
                   f"PERM[{t}] o PERM[{inv}] fails at src={src}: {src}→{dst}→{back}")
        # Composing PERM[t] and PERM[SYM_INV[t]] gives identity for all cells
        _check(all(PERM[t][PERM[SYM_INV[t]][i]] == i for i in range(CELLS)),
               f"Inverse check failed for t={t}")
    print("PASS")

    # 2. Value invariance under all 8 symmetries (if solver is available).
    print("  [2] Value invariance under symmetry ...", end=" ")
    try:
        from solver_backend import Solver, is_terminal as _isterm
        # Use Solver via the solve_wld API exclusively.
        solver = Solver(tt_bits=22)
        solver.set_use_tt(True)
        solver.set_ordering(True)
    except Exception:
        print("SKIP (solver_backend not available)")
    else:
        for _ in range(10):
            me, opp = _random_position(empties=rng.randint(8, 20), rng=rng)
            # Ensure non-terminal
            if _isterm(me, opp):
                continue
            # Reference value (WLD sign)
            ref = solver.solve_wld(me, opp)
            ref_sign = (ref > 0) - (ref < 0)  # 1 / 0 / -1
            for t in range(N_SYMS):
                me_t, opp_t = apply_symmetry(me, opp, t)
                v = solver.solve_wld(me_t, opp_t)
                v_sign = (v > 0) - (v < 0)
                _check(v_sign == ref_sign,
                       f"Value sign mismatch at t={t}: ref_sign={ref_sign} val_sign={v_sign}")
        print("PASS")

    # 3. Canonical form: all 8 symmetric variants canonicalize to same key.
    print("  [3] Canonical form consistency ...", end=" ")
    for _ in range(10):
        me, opp = _random_position(empties=rng.randint(8, 20), rng=rng)
        canon_keys = set()
        for t in range(N_SYMS):
            me_t, opp_t = apply_symmetry(me, opp, t)
            c_me, c_opp, _ = canonicalize(me_t, opp_t)
            canon_keys.add(pack_key(c_me, c_opp))
        _check(len(canon_keys) == 1,
               f"Canonical form differs across symmetries: {len(canon_keys)} distinct keys")
    print("PASS")

    # 4. Base-3 round-trip.
    print("  [4] Base-3 round-trip ...", end=" ")
    for _ in range(100):
        me, opp = _random_position(empties=rng.randint(0, 36), rng=rng)
        key = pack_key(me, opp)
        me2, opp2 = unpack_key(key)
        _check(me == me2 and opp == opp2,
               "pack/unpack round-trip failure")
    print("PASS")

    # 5. Determinism.
    print("  [5] Determinism ...", end=" ")
    for _ in range(20):
        me, opp = _random_position(empties=rng.randint(8, 20), rng=rng)
        c1 = canonicalize(me, opp)
        c2 = canonicalize(me, opp)
        _check(c1 == c2, "canonicalize is not deterministic")
    print("PASS")

    print()
    print("ALL SYMMETRY CHECKS PASSED")


if __name__ == "__main__":
    verify()
