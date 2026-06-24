// solver.hpp
// ---------------------------------------------------------------------------
// 6x6 Othello exact / WLD solver (plan §10 step 2; techniques 2,3,4,6).
//
// Built ON TOP OF the frozen bitboard core (bitboard6.hpp). This file adds
// search only; it never reimplements move-gen/flip/terminal logic and never
// touches the core. All board primitives come from namespace bb6.
//
// What this provides
// ------------------
//   - Negamax + alpha-beta over (me,opp) masks, mover-relative, solve-to-
//     terminal (NO depth cap: the search bottoms out only at double-pass
//     terminals, so every value is a true game-theoretic value).
//   - EXACT mode:  full window [-INF,+INF]. Terminal score = signed disc diff
//                  count(me)-count(opp). Root value = optimal margin under
//                  optimal play. Sign of the margin == win/loss/draw.
//   - WLD mode:    narrow window [-1,+1]. Proves only the SIGN of the value
//                  (win>0 / draw=0 / loss<0) and prunes far harder.
//                  (Why sign-equivalent to EXACT: the agent's win-first score
//                  f(diff)=sign(diff)*WIN_BASE+diff is strictly monotincreasing
//                  in diff for |diff|<=36<WIN_BASE, so maximising margin and
//                  maximising win-first agree on value AND play. The narrow
//                  [-1,+1] window straddles 0, so the returned value's sign
//                  matches the exact margin's sign on every position.)
//   - Zobrist transposition table: std::mt19937_64 with a FIXED seed
//     (deterministic). Indexed by the Zobrist hash of (me,opp); each slot
//     stores the FULL (me,opp) for verification, so hash collisions can never
//     produce a wrong answer (plan §6: "use Zobrist ... verify"). Stores
//     value + bound flag (exact/lower/upper) + best move; replace-by-depth
//     (depth proxy = empty-square count: bigger subtrees are kept).
//   - Move ordering: TT best-move first -> corner priority -> fastest-first
//     (minimise opponent mobility). Parity ordering deliberately omitted
//     (optional in the plan; defer).
//
// Correctness contract honoured here
// ----------------------------------
//   * TT bounds are used only for an immediate cutoff/return against the
//     ORIGINAL window; the window is never narrowed-then-continued. This makes
//     the fail-soft bound flag unambiguous and guarantees
//         value(TT on) == value(TT off)
//     for any window (verified empirically by the harness). With a full window
//     at the root this means the EXACT root value is always the true value.
//
// Portability: reuses bb6::popcount64 / bb6::ctz64 only. No new raw builtins.
// ---------------------------------------------------------------------------
#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "bitboard6.hpp"
#include "symmetry.hpp"

#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <atomic>

namespace solver {

constexpr int INF = 1000;  // > any |margin| (<=36); sentinel for full window.

enum Bound : uint8_t { B_NONE = 0, B_EXACT = 1, B_LOWER = 2, B_UPPER = 3 };

struct TTEntry {
    uint64_t me  = 0;
    uint64_t opp = 0;
    int16_t  value     = 0;
    uint8_t  empties   = 0;   // replace-by-depth proxy
    uint8_t  flag      = B_NONE;
    int8_t   best_move = -1;
};

// 6x6 corners: (0,0)=0 (0,5)=5 (5,0)=30 (5,5)=35.
inline bool is_corner(int cell) {
    return cell == 0 || cell == 5 || cell == 30 || cell == 35;
}

class Solver {
public:
    // tt_bits: log2 of the number of TT slots. seed: FIXED for determinism.
    explicit Solver(unsigned tt_bits = 22, uint64_t seed = 0x9E3779B97F4A7C15ULL) {
        init_zobrist(seed);   // mt19937_64 with a fixed seed -> deterministic
        resize_tt(tt_bits);
    }

    void resize_tt(unsigned tt_bits) {
        tt_bits_ = tt_bits;
        tt_mask_ = (uint64_t(1) << tt_bits_) - 1;
        tt_.assign(size_t(1) << tt_bits_, TTEntry{});
    }

    void clear_tt() {
        std::fill(tt_.begin(), tt_.end(), TTEntry{});
    }

    void set_use_tt(bool on) { use_tt_ = on; }
    void set_ordering(bool on) { ordering_ = on; }
    bool sym_tt_ = true;
    void set_sym_tt(bool on) { sym_tt_ = on; }

    void request_stop() { stop_ = true; }
    void reset_stop() { stop_ = false; }
    bool stopped() const { return stop_; }

    uint64_t nodes = 0;
    uint64_t tt_hit_exact = 0;
    uint64_t tt_hit_symmetric = 0;

    // Solve to terminal. Returns the optimal disc margin (count(me)-count(opp))
    // under optimal play, from the side-to-move's perspective. Sign == WLD.
    int solve_exact(uint64_t me, uint64_t opp) {
        nodes = 0;
        tt_hit_exact = 0;
        tt_hit_symmetric = 0;
        return negamax(me, opp, -INF, +INF);
    }

    // Solve only the sign of the value with a hard-pruning narrow window.
    // Returns a value whose SIGN (>0 win / ==0 draw / <0 loss) is correct;
    // its magnitude is not meaningful beyond the sign.
    int solve_wld(uint64_t me, uint64_t opp) {
        nodes = 0;
        tt_hit_exact = 0;
        tt_hit_symmetric = 0;
        return negamax(me, opp, -1, +1);
    }

private:
    // ---- Zobrist ----
    uint64_t zob_[bb6::CELLS][2];

    void init_zobrist(uint64_t seed) {
        std::mt19937_64 rng(seed);
        for (int c = 0; c < bb6::CELLS; ++c) {
            zob_[c][0] = rng();
            zob_[c][1] = rng();
        }
    }

    uint64_t hash_pos(uint64_t me, uint64_t opp) const {
        uint64_t h = 0;
        uint64_t x = me;
        while (x) { int b = bb6::ctz64(x); h ^= zob_[b][0]; x &= x - 1; }
        x = opp;
        while (x) { int b = bb6::ctz64(x); h ^= zob_[b][1]; x &= x - 1; }
        return h;
    }

    // ---- TT ----
    std::vector<TTEntry> tt_;
    unsigned tt_bits_ = 22;
    uint64_t tt_mask_ = (uint64_t(1) << 22) - 1;
    bool use_tt_ = true;
    bool ordering_ = true;
    std::atomic<bool> stop_{false};

    inline int empties_of(uint64_t me, uint64_t opp) const {
        return bb6::CELLS - bb6::popcount64(me | opp);
    }

    // ---- search ----
    int negamax(uint64_t me, uint64_t opp, int alpha, int beta) {
        ++nodes;
        if (stop_) return alpha;

        uint64_t canon_me = me, canon_opp = opp;
        int sym_t = 0;
        if (sym_tt_) {
            std::tie(canon_me, canon_opp, sym_t) = sym::canonicalize(me, opp);
        }

        uint64_t moves = bb6::gen_moves(me, opp);
        if (moves == 0) {
            if (bb6::gen_moves(opp, me) == 0) {
                return bb6::count(me) - bb6::count(opp);
            }
            return -negamax(opp, me, -beta, -alpha);
        }

        const int alpha0 = alpha;
        const int beta0  = beta;
        const int emp    = empties_of(me, opp);

        int tt_move = -1;
        if (use_tt_) {
            const TTEntry& e = tt_[hash_pos(canon_me, canon_opp) & tt_mask_];
            if (e.flag != B_NONE && e.me == canon_me && e.opp == canon_opp) {
                if (e.me == me && e.opp == opp) {
                    ++tt_hit_exact;
                } else {
                    ++tt_hit_symmetric;
                }
                if (e.flag == B_EXACT) return e.value;
                if (e.flag == B_LOWER && e.value >= beta) return e.value;
                if (e.flag == B_UPPER && e.value <= alpha) return e.value;
                int tt_move_canon = e.best_move;
                tt_move = (tt_move_canon >= 0 && sym_tt_)
                          ? sym::inverse_transform(tt_move_canon, sym_t)
                          : tt_move_canon;
            }
        }

        struct Cand { int mv; uint64_t cme, copp; int key; };
        Cand cand[bb6::CELLS];
        int nc = 0;
        {
            uint64_t m = moves;
            while (m) {
                int mv = bb6::ctz64(m);
                m &= m - 1;
                uint64_t meO, oppO;
                bb6::apply_move_cell(me, opp, mv, meO, oppO);
                int key = 0;
                if (ordering_) {
                    int opp_mob = bb6::popcount64(bb6::gen_moves(oppO, meO));
                    key = (is_corner(mv) ? 1000 : 0) - opp_mob;
                }
                cand[nc++] = Cand{ mv, oppO, meO, key };
            }
        }
        if (ordering_ && nc > 1) {
            std::sort(cand, cand + nc,
                      [](const Cand& a, const Cand& b) { return a.key > b.key; });
        }
        if (tt_move >= 0) {
            for (int i = 0; i < nc; ++i) {
                if (cand[i].mv == tt_move) {
                    Cand t = cand[i];
                    for (int j = i; j > 0; --j) cand[j] = cand[j - 1];
                    cand[0] = t;
                    break;
                }
            }
        }

        int best = -INF, best_move = -1;
        for (int i = 0; i < nc; ++i) {
            if (stop_) break;
            int val = -negamax(cand[i].cme, cand[i].copp, -beta, -alpha);
            if (val > best) { best = val; best_move = cand[i].mv; }
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;
        }

        if (use_tt_) {
            Bound f = (best <= alpha0) ? B_UPPER
                    : (best >= beta0)  ? B_LOWER
                                       : B_EXACT;
            TTEntry& slot = tt_[hash_pos(canon_me, canon_opp) & tt_mask_];
            if (slot.flag == B_NONE ||
                (slot.me == canon_me && slot.opp == canon_opp) ||
                emp >= slot.empties) {
                slot.me        = canon_me;
                slot.opp       = canon_opp;
                slot.value     = static_cast<int16_t>(best);
                slot.empties   = static_cast<uint8_t>(emp);
                slot.flag      = static_cast<uint8_t>(f);
                slot.best_move = (best_move >= 0 && sym_tt_)
                                 ? static_cast<int8_t>(sym::PERM[sym_t][best_move])
                                 : static_cast<int8_t>(best_move);
            }
        }
        return best;
    }
};

}  // namespace solver

#endif  // SOLVER_HPP
