// symmetry.hpp
// ---------------------------------------------------------------------------
// D4 symmetry canonicalization + base-3 packing for 6x6 Othello (plan §4
// technique 5, §10 step 5a).  Header-only, built ON TOP OF bitboard6.hpp.
//
// Provides:
//   - 8 precomputed D4 cell-permutation tables (identity / rot90 / rot180 /
//     rot270 / reflect-H / reflect-V / reflect-D1 / reflect-D2).
//   - apply_symmetry(me, opp, t)  -> transformed (me, opp)
//   - canonicalize(me, opp)       -> (canon_me, canon_opp, transform)
//   - inverse_transform(cell, t)  -> original cell after reverse symmetry
//   - pack_key(me, opp)           -> collision-free base-3 uint64 key
//   - unpack_key(key)             -> (me, opp) bitmasks
//
// All tables are computed at compile time (constexpr).  Permutation tables
// are bijections over 0..35 and satisfy  PERM[t][PERM[SYM_INV[t]][i]] == i.
#ifndef SYMMETRY_HPP
#define SYMMETRY_HPP

#include "bitboard6.hpp"

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>

namespace sym {

constexpr int N_SYMS = 8;   // D4 group size

// ==========================================================================
// Compile-time table builders
// ==========================================================================

// Coordinate transform (r, c) -> destination cell index for transform t.
constexpr int perm_cell(int t, int idx) {
    constexpr int N = 6;
    int r = idx / N, c = idx % N;
    switch (t) {
    case 0: return r * N + c;                       // identity
    case 1: return c * N + (N - 1 - r);             // rotate 90 CW
    case 2: return (N - 1 - r) * N + (N - 1 - c);   // rotate 180
    case 3: return (N - 1 - c) * N + r;             // rotate 270 CW
    case 4: return r * N + (N - 1 - c);             // reflect H
    case 5: return (N - 1 - r) * N + c;             // reflect V
    case 6: return c * N + r;                       // reflect D1
    case 7: return (N - 1 - c) * N + (N - 1 - r);   // reflect D2
    default: return 0;
    }
}

// 8 x 36 permutation table.
constexpr auto build_perm() {
    std::array<std::array<int, bb6::CELLS>, N_SYMS> p{};
    for (int t = 0; t < N_SYMS; ++t)
        for (int i = 0; i < bb6::CELLS; ++i)
            p[t][i] = perm_cell(t, i);
    return p;
}
constexpr auto PERM = build_perm();

// Inverse table: SYM_INV[t] = the transform that inverts t.
constexpr auto build_inv() {
    std::array<int, N_SYMS> inv{};
    for (int t = 0; t < N_SYMS; ++t) {
        for (int cand = 0; cand < N_SYMS; ++cand) {
            bool ok = true;
            for (int i = 0; i < bb6::CELLS; ++i) {
                if (PERM[t][PERM[cand][i]] != i) { ok = false; break; }
            }
            if (ok) { inv[t] = cand; break; }
        }
    }
    return inv;
}
constexpr auto SYM_INV = build_inv();

// Base-3 powers: POW3[i] = 3^i  (fits in uint64 for i < 36).
constexpr auto build_pow3() {
    std::array<uint64_t, bb6::CELLS> p{};
    p[0] = 1;
    for (int i = 1; i < bb6::CELLS; ++i)
        p[i] = p[i - 1] * uint64_t(3);
    return p;
}
constexpr auto POW3 = build_pow3();

// ==========================================================================
// Public API  (all inline — called from hot path)
// ==========================================================================

// Apply a cell permutation to a 36-bit mask.
inline uint64_t permute_mask(uint64_t mask, int transform) {
    const auto& p = PERM[transform];
    uint64_t out = 0;
    uint64_t m = mask;
    while (m) {
        int idx = bb6::ctz64(m);
        out |= uint64_t(1) << p[idx];
        m &= m - 1;
    }
    return out;
}

// Apply a D4 symmetry to the full position; return (new_me, new_opp).
inline std::pair<uint64_t, uint64_t>
apply_symmetry(uint64_t me, uint64_t opp, int transform) {
    return { permute_mask(me, transform),
             permute_mask(opp, transform) };
}

// Collision-free uint64 key via base-3 encoding.
//   state per cell: 0 = empty, 1 = me, 2 = opp.
inline uint64_t pack_key(uint64_t me, uint64_t opp) {
    uint64_t key = 0;
    for (int cell = 0; cell < bb6::CELLS; ++cell) {
        int state = 0;
        if ((me >> cell) & 1) state = 1;
        else if ((opp >> cell) & 1) state = 2;
        key += uint64_t(state) * POW3[cell];
    }
    return key;
}

// Inverse of pack_key.
inline std::pair<uint64_t, uint64_t> unpack_key(uint64_t key) {
    uint64_t me = 0, opp = 0;
    for (int cell = 0; cell < bb6::CELLS; ++cell) {
        int state = int((key / POW3[cell]) % 3);
        if (state == 1) me |= uint64_t(1) << cell;
        else if (state == 2) opp |= uint64_t(1) << cell;
    }
    return { me, opp };
}

// Return (canon_me, canon_opp, transform) giving the minimum base-3 key.
// Tie-breaking: smallest transform index wins.
inline std::tuple<uint64_t, uint64_t, int>
canonicalize(uint64_t me, uint64_t opp) {
    uint64_t best_key = UINT64_MAX;
    uint64_t best_me = 0, best_opp = 0;
    int best_t = 0;
    for (int t = 0; t < N_SYMS; ++t) {
        auto [mt, ot] = apply_symmetry(me, opp, t);
        uint64_t key = pack_key(mt, ot);
        if (key < best_key) {
            best_key = key;
            best_me = mt;
            best_opp = ot;
            best_t = t;
        }
    }
    return { best_me, best_opp, best_t };
}

// Map a cell back through the inverse of `transform`.
// The inverse of transform t is SYM_INV[t]; applying PERM[SYM_INV[t]] to the
// forward-mapped cell recovers the original index.
inline int inverse_transform(int cell, int transform) {
    return PERM[SYM_INV[transform]][cell];
}

} // namespace sym

#endif // SYMMETRY_HPP
