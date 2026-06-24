// bitboard6.hpp
// ---------------------------------------------------------------------------
// 6x6 Othello bitboard core (plan §4 technique 1, §10 step 1).
//
// Scope of this file: ONLY the board representation and the primitive
// operations search/book/pybind11 will later be built on top of:
//   - move generation
//   - flip application
//   - pass detection / terminal detection
//   - disc count
// No search, no transposition table, no bindings. Those are later steps.
//
// Representation
// --------------
//   A position is two 64-bit masks (me, opp), ALWAYS from the side-to-move's
//   perspective (me = side to move). Bit index = row*6 + col, matching the
//   competition action encoding (action a -> row = a/6, col = a%6). Only the
//   low 36 bits are ever used; bits 36..63 are always zero.
//
//   apply_move() returns the new (me, opp) STILL IN THE MOVER'S FRAME (it does
//   not swap sides). Swapping to the opponent's turn is just (opp', me') and is
//   the caller's concern. Keeping apply side-frame-stable makes the differential
//   test unambiguous.
//
// 6x6-native masks (NOT 8x8 constants adapted down)
// -------------------------------------------------
//   Vertical shift is +/-6, diagonals +/-5 and +/-7, horizontal +/-1. Each
//   direction carries a precomputed source mask = "cells whose neighbour in
//   this direction is still on the board". That mask is the column/row-edge
//   guard: masking the source BEFORE shifting kills wraparound across the left
//   /right edges and off the top/bottom. The masks are generated from the 6x6
//   geometry at compile time (constexpr), so there are no hand-typed hex
//   constants to get wrong.
//
// Portability (plan §3.1)
// -----------------------
//   No GCC/Clang-only builtins are used directly. popcount and bitscan are
//   wrapped behind a shim that branches on _MSC_VER vs GCC/Clang, with a plain
//   C++ fallback. This same source is intended to compile under both g++ (WSL)
//   and MSVC (native Windows) later; the shim is set up now, not retrofitted.
// ---------------------------------------------------------------------------
#ifndef BITBOARD6_HPP
#define BITBOARD6_HPP

#include <cstdint>
#include <vector>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

namespace bb6 {

// ---------- platform shim: popcount + count-trailing-zeros -----------------
// Wrapped so the bitboard core stays portable across MSVC and GCC/Clang.
inline int popcount64(uint64_t x) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(x));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int c = 0;                       // Kernighan fallback
    while (x) { x &= x - 1; ++c; }
    return c;
#endif
}

// Index of the least-significant set bit. Precondition: x != 0.
inline int ctz64(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int n = 0;                       // loop fallback
    while (!(x & 1ULL)) { x >>= 1; ++n; }
    return n;
#endif
}

// ---------- board geometry -------------------------------------------------
constexpr int      N     = 6;
constexpr int      CELLS = N * N;                          // 36
constexpr uint64_t FULL  = (uint64_t(1) << CELLS) - 1;     // low 36 bits set
constexpr int      PASS  = CELLS;                          // pass action == 36

constexpr uint64_t cell_bit(int r, int c) { return uint64_t(1) << (r * N + c); }

// One ray direction: shift amount s = dr*N + dc, plus the source mask of cells
// whose (r+dr, c+dc) neighbour is on the board.
struct Dir { int s; uint64_t mask; };

constexpr Dir make_dir(int dr, int dc) {
    uint64_t m = 0;
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            if (r + dr >= 0 && r + dr < N && c + dc >= 0 && c + dc < N)
                m |= cell_bit(r, c);
    return Dir{ dr * N + dc, m };
}

// 8 directions. Shifts work out to: N=-6 S=+6 E=+1 W=-1 NE=-5 NW=-7 SE=+7 SW=+5.
constexpr Dir DIRS[8] = {
    make_dir(-1, -1), make_dir(-1, 0), make_dir(-1, 1),
    make_dir( 0, -1),                  make_dir( 0, 1),
    make_dir( 1, -1), make_dir( 1, 0), make_dir( 1, 1),
};

// Mask the source by the edge guard, then shift one step in direction d.
inline uint64_t shift_dir(uint64_t x, const Dir& d) {
    x &= d.mask;
    return d.s > 0 ? ((x << d.s) & FULL) : (x >> (-d.s));
}

// ---------- primitive operations -------------------------------------------

// Bitmask of all legal placing squares for the side to move (0 if none -> pass).
inline uint64_t gen_moves(uint64_t me, uint64_t opp) {
    const uint64_t empty = FULL & ~(me | opp);
    uint64_t moves = 0;
    for (const Dir& d : DIRS) {
        // opponent cells directly adjacent to one of my pieces along d ...
        uint64_t x = shift_dir(me, d) & opp;
        // ... extended along the contiguous opponent run (>= N-3 steps needed;
        // N-1 over-provisions and is idempotent once the run ends).
        for (int i = 0; i < N - 1; ++i)
            x |= shift_dir(x, d) & opp;
        // one further step lands on the bracketing square, which must be empty.
        moves |= shift_dir(x, d) & empty;
    }
    return moves;
}

// Place the disc at single-bit `pos` and flip the bracketed opponent discs.
// Output (me_out, opp_out) stays in the MOVER's frame (no side swap).
inline void apply_move(uint64_t me, uint64_t opp, uint64_t pos,
                       uint64_t& me_out, uint64_t& opp_out) {
    uint64_t flips = 0;
    for (const Dir& d : DIRS) {
        uint64_t line = 0;
        uint64_t x = shift_dir(pos, d) & opp;   // first neighbour must be opp
        while (x) {
            line |= x;
            uint64_t nxt = shift_dir(x, d);
            if (nxt & me) { flips |= line; break; }  // bracket closed by my disc
            if (!(nxt & opp)) break;                 // empty/off-board -> no flip
            x = nxt;
        }
    }
    me_out  = me | pos | flips;
    opp_out = opp & ~flips;
}

// Convenience overload taking a cell index 0..35.
inline void apply_move_cell(uint64_t me, uint64_t opp, int cell,
                            uint64_t& me_out, uint64_t& opp_out) {
    apply_move(me, opp, uint64_t(1) << cell, me_out, opp_out);
}

inline bool has_move(uint64_t me, uint64_t opp)  { return gen_moves(me, opp) != 0; }

// Terminal iff neither side has a placing move.
inline bool is_terminal(uint64_t me, uint64_t opp) {
    return gen_moves(me, opp) == 0 && gen_moves(opp, me) == 0;
}

inline int count(uint64_t x) { return popcount64(x); }

// Legal actions in the env's encoding: ascending cell indices, or just {PASS}
// when no placing move exists (mirrors OthelloEnv.get_legal_actions exactly).
inline std::vector<int> legal_actions(uint64_t me, uint64_t opp) {
    uint64_t m = gen_moves(me, opp);
    std::vector<int> out;
    if (m == 0) { out.push_back(PASS); return out; }
    while (m) {                          // ctz walk yields ascending indices
        out.push_back(ctz64(m));
        m &= m - 1;
    }
    return out;
}

} // namespace bb6

#endif // BITBOARD6_HPP
