// bindings.cpp
// ---------------------------------------------------------------------------
// pybind11 bindings for the FROZEN 6x6 Othello engine (plan §3, §10 step 4).
//
// This is the FFI boundary and nothing else. It only #includes the frozen
// headers and calls into bb6:: / solver::Solver. It does NOT reimplement any
// board logic and does NOT edit either header (frozen contract, step-4 §2).
//
// Convention exposed to Python (must match environment.py / bitboard6.hpp):
//   * positions are two uint64 masks (me, opp) ALWAYS in the mover's frame;
//   * cell index = row*6 + col;  pass action = 36;
//   * only the low 36 bits of a mask are ever set (bits 36..63 always zero).
//
// Mask marshaling is the single highest-risk bug class here (step-4 §6), so
// every mask crosses the boundary as an explicit uint64_t: a Python int in
// [0, 2^36) round-trips losslessly with no sign/width ambiguity. Bar (b) of
// run_bindings.sh re-checks this end-to-end by diffing solve_exact margins
// through Python against the independent oracle.
// ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // std::vector<int> -> list, std::pair -> tuple

#include <cstdint>
#include <utility>

#include "bitboard6.hpp"
#include "solver.hpp"
#include "symmetry.hpp"

namespace py = pybind11;

// apply a placing move by cell index; return the new (me, opp) in the mover's
// frame (no side swap -- matches bb6::apply_move_cell exactly).
static std::pair<uint64_t, uint64_t>
py_apply_move(uint64_t me, uint64_t opp, int cell) {
    uint64_t me_out = 0, opp_out = 0;
    bb6::apply_move_cell(me, opp, cell, me_out, opp_out);
    return {me_out, opp_out};
}

PYBIND11_MODULE(othello_cpp, m) {
    m.doc() = "Compiled bindings for the frozen 6x6 Othello bitboard solver "
              "(bb6 primitives + solver::Solver). Masks are mover-frame uint64, "
              "cell = row*6+col, pass = 36.";

    // Backend tag + board constants (let callers avoid hardcoding 36).
    m.attr("__backend__") = "cpp";
    m.attr("CELLS")       = bb6::CELLS;   // 36
    m.attr("PASS")        = bb6::PASS;    // 36

    // --- the solver class (persistent TT reusable across moves) -------------
    py::class_<solver::Solver>(m, "Solver")
        .def(py::init<unsigned>(), py::arg("tt_bits") = 22u,
             "Construct a solver with 2**tt_bits TT slots. The Zobrist seed is "
             "fixed inside solver.hpp, so results are deterministic.")
        .def("solve_exact", &solver::Solver::solve_exact,
             py::arg("me"), py::arg("opp"),
             "Solve to terminal; return the optimal signed disc margin "
             "count(me)-count(opp) under optimal play (sign == WLD).")
        .def("solve_wld", &solver::Solver::solve_wld,
             py::arg("me"), py::arg("opp"),
             "Solve only the SIGN of the value with a narrow [-1,+1] window "
             "(>0 win / ==0 draw / <0 loss); magnitude is not meaningful.")
        .def("clear_tt", &solver::Solver::clear_tt,
             "Zero the transposition table (does not change the Zobrist seed).")
        .def("set_use_tt", &solver::Solver::set_use_tt, py::arg("on"))
        .def("set_ordering", &solver::Solver::set_ordering, py::arg("on"))
        .def_readonly("nodes", &solver::Solver::nodes,
                      "Nodes visited by the most recent solve_* call.");

    // --- bb6 primitives (interchangeable with the Python fallback) ----------
    m.def("gen_moves", &bb6::gen_moves, py::arg("me"), py::arg("opp"),
          "Bitmask of all legal placing squares for the side to move "
          "(0 -> the only legal action is pass).");
    m.def("legal_actions", &bb6::legal_actions, py::arg("me"), py::arg("opp"),
          "Ascending cell indices of legal placing moves, or [36] (PASS) when "
          "none exist -- mirrors OthelloEnv.get_legal_actions.");
    m.def("apply_move", &py_apply_move,
          py::arg("me"), py::arg("opp"), py::arg("cell"),
          "Place at `cell` and flip; return (me2, opp2) still in the mover's "
          "frame.");
    m.def("is_terminal", &bb6::is_terminal, py::arg("me"), py::arg("opp"),
          "True iff neither side has a placing move (double-pass terminal).");
    m.def("count", &bb6::count, py::arg("x"),
          "Popcount of a mask.");

    // --- symmetry operations (step 5a) ------------------------------------
    m.attr("N_SYMS") = sym::N_SYMS;

    m.def("permute_mask", &sym::permute_mask,
          py::arg("mask"), py::arg("transform"),
          "Apply a D4 cell permutation to a 36-bit mask.");
    m.def("apply_symmetry", &sym::apply_symmetry,
          py::arg("me"), py::arg("opp"), py::arg("transform"),
          "Apply a D4 symmetry to (me, opp); return (new_me, new_opp).");
    m.def("canonicalize", &sym::canonicalize,
          py::arg("me"), py::arg("opp"),
          "Return (canon_me, canon_opp, transform) with min base-3 key.");
    m.def("inverse_transform", &sym::inverse_transform,
          py::arg("cell"), py::arg("transform"),
          "Map a cell back through the inverse of a symmetry.");
    m.def("pack_key", &sym::pack_key,
          py::arg("me"), py::arg("opp"),
          "Collision-free uint64 key: base-3 encoding of (me, opp).");
    m.def("unpack_key", &sym::unpack_key,
          py::arg("key"),
          "Inverse of pack_key; return (me, opp) bitmasks.");
}
