# setup.py
# ---------------------------------------------------------------------------
# One build description for the `othello_cpp` extension that works under BOTH
# toolchains the project targets (plan §3.1):
#   * g++  on WSL/Linux  -> othello_cpp.cpython-<ver>-<arch>-linux-gnu.so
#   * MSVC on Windows    -> othello_cpp.cp<ver>-win_amd64.pyd
#
# Usage (either is fine):
#   python setup.py build_ext --inplace      # drops the module next to this file
#   pip install -e .                          # editable install
#
# Build-both rationale: the tournament machine's OS/arch is unconfirmed
# (step-4 §7 open item 3), so the agent must run from whichever artifact the
# grading box can load -- and if it can load NEITHER, solver_backend.py falls
# back to pure Python (it never hard-depends on this .so/.pyd).
#
# Warning flags: the project compiles its own headers under
# `-Wall -Wextra -Wpedantic` (see run_solver.sh) and they are clean there. We
# deliberately pass only `-Wall -Wextra` to THIS translation unit: pybind11's
# `PYBIND11_MODULE` macro is not ISO-pedantic (variadic-macro warning under
# -Wpedantic), and that noise is third-party, not ours. Our code stays clean.
# ---------------------------------------------------------------------------
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if sys.platform == "win32":
    # MSVC: /O2 optimisation; /W3 is MSVC's sensible warning level.
    extra_compile_args = ["/O2", "/W3"]
else:
    # g++ / clang (WSL/Linux/macOS): match the project's optimisation, and the
    # subset of its warning flags that is clean through pybind11's headers.
    extra_compile_args = ["-O2", "-Wall", "-Wextra"]

ext_modules = [
    Pybind11Extension(
        "othello_cpp",
        sources=["bindings.cpp"],
        cxx_std=17,                       # plan §3.1: C++17 on both toolchains
        extra_compile_args=extra_compile_args,
        # bindings.cpp #includes the frozen headers from this directory.
        include_dirs=["."],
    ),
]

setup(
    name="othello_cpp",
    version="0.1.0",
    description="pybind11 bindings for the frozen 6x6 Othello bitboard solver",
    long_description=(
        "Exposes the frozen bb6 primitives and solver::Solver (exact / WLD "
        "negamax + Zobrist TT) to Python. Pure binding layer; no board logic "
        "is reimplemented here."
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
