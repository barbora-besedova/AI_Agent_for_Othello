# C++ Negamax Solver Integration

This folder documents the C++ negamax solver that was ported from the `OthelloCpp/` project into `all_in_clean/` for use as a training opponent for the DQN-based agents.

---

## Architecture

```
all_in_clean/
  cpp_solver/                    # C++ solver engine
    othello_cpp.pyd              #  Windows .pyd (compiled for Python 3.14)
    othello_cpp.cpython-*-linux-gnu.so  # Linux/WSL .so
    bitboard6.hpp                #  6x6 bitboard primitives (move gen, flip, terminal)
    solver.hpp                   #  Negamax + alpha-beta + Zobrist transposition table
    symmetry.hpp                 #  D4 symmetry canonicalization (TT efficiency)
    bindings.cpp                 #  pybind11 bridge exposing Solver + bb6 to Python
    setup.py                     #  Build script for both MSVC and MinGW
    solver_backend.py            #  Python facade: tries C++ first, falls back to pure-Python
    minimax_agent.py             #  Pure-Python bitboard minimax (fallback, also standalone)
    symmetry.py                  #  Pure-Python symmetry module (fallback)
    __init__.py

  agents/
    cpp_minimax_agent.py         #  Two agent wrappers with select_action(obs) interface:
                                   - FastMinimaxAgent  (iterative deepening + time limit)
                                   - CppSolverAgent    (perfect terminal solver)

  train_vs_minimax.py            # Training script that uses FastMinimaxAgent as opponent
```

### Backend selection (`solver_backend.py`)

```
import othello_cpp  ── success ──> BACKEND = "cpp"
                    ── failure ──> BACKEND = "python" (falls back to minimax_agent.py)
```

The module always loads; the fallback is silent (one `logging.warning`). The `BACKEND` constant tells you which engine is active.

### Two-layer search (solver.hpp)

The C++ solver performs a **negamax search with alpha-beta pruning**:

- **EXACT mode** (`solve_exact`): full window `[-INF, +INF]`, returns signed disc margin `count(me)-count(opp)` under optimal play.
- **WLD mode** (`solve_wld`): narrow window `[-1, +1]`, returns a value whose sign is correct (`>0` win, `==0` draw, `<0` loss). Much faster because the narrow window prunes more aggressively.

Both modes search **to terminal** (no depth limit). Search bottoms out only at double-pass terminal positions. The transposition table uses Zobrist hashing with full position verification (collisions cannot produce wrong answers).

---

## Agent classes (`agents/cpp_minimax_agent.py`)

### FastMinimaxAgent (recommended for training)

```
FastMinimaxAgent(board_size=6, max_depth=None, time_limit=2.0, ordering=True)
```

- **Iterative deepening**: searches depth 1, 2, ... up to `max_depth` (or until terminal).
- **Time limit**: if `time_limit` seconds is exceeded, the search stops at the next completed depth and returns the best move from that depth.
- **Board size**: supports **4–8** (not just 6x6).
- **Heuristic**: disc differential + corner/edge/X-square weights + mobility.

This is the practical choice for DQN training. Even with `time_limit=1.0` it plays a strong game.

### CppSolverAgent (perfect play, endgame only)

```
CppSolverAgent(board_size=6, use_wld=True, tt_bits=22, cpp_threshold=18, time_limit=2.0)
```

- Uses the C++ solver directly for positions with **≤ `cpp_threshold` empty squares** (default 18).
- For positions with more empties, falls back to `FastMinimaxAgent`.
- Only supports **6x6** (the C++ bitboard is hardcoded for 6x6).
- **ccp_threshold controls the speed/strength tradeoff**: lower = faster (more iterative deepening), higher = stronger (more perfect play).

---

## Training script (`train_vs_minimax.py`)

### Quick start

From the `all_in_clean/` directory:

```bash
# Classic DQN vs minimax curriculum (default)
python train_vs_minimax.py

# Guided DQN (heuristic bonus)
python train_vs_minimax.py --heuristic_weight 0.2

# Prioritized DQN
python train_vs_minimax.py --use_per

# Guided + Prioritized DQN
python train_vs_minimax.py --use_per --heuristic_weight 0.2

# Resume from checkpoint
python train_vs_minimax.py --load_model_path models/checkpoint.pth

# Adjust minimax strength / speed
python train_vs_minimax.py --minimax_time_limit 0.5   # faster, weaker
python train_vs_minimax.py --minimax_time_limit 3.0   # slower, stronger
```

### Opponent curriculum

The training curriculum automatically scales opponent strength across 4 phases:

| Phase | Episodes % | Opponent mix |
|---|---|---|
| 1 (warm-up) | 0–25% | 50% random, 35% greedy, 15% heuristic |
| 2 (early) | 25–50% | 20% random, 25% greedy, 30% heuristic, **25% fast_minimax** |
| 3 (mid) | 50–75% | 10% random, 15% greedy, 30% heuristic, **45% fast_minimax** |
| 4 (late) | 75–100% | 5% random, 10% greedy, 20% heuristic, **65% fast_minimax** |

The agent alternates colors each episode (player 1 / player -1) for balanced training.

### Epsilon schedule

- Starts at 1.0 (full exploration), decays by factor 0.999 per episode
- Floors: greedy/heuristic → 0.15, fast_minimax → 0.20 (more exploration vs strong opponents)

### Evaluation

Every `--eval_every` episodes (default 200), the script runs 100 games (50 for minimax) against each opponent type **without exploration** (`epsilon=0`) and reports win rate, score, and W/D/L.

### All CLI options

```
--board_size           Board size (default: 6)
--num_episodes         Number of episodes (default: 3000)
--epsilon_start        Initial epsilon (default: 1.0)
--epsilon_end          Final epsilon (default: 0.05)
--epsilon_decay        Epsilon decay rate (default: 0.999)
--opponent_type        'curriculum' | 'random' | 'greedy' | 'heuristic' | 'fast_minimax'
--minimax_time_limit   Seconds per move for minimax (default: 1.0)
--learning_rate        Adam learning rate (default: 1e-3)
--gamma                Discount factor (default: 0.99)
--batch_size           Minibatch size (default: 64)
--buffer_capacity      Replay buffer size (default: 50000)
--target_update_freq   Target network update interval (default: 500)
--learning_starts      Transitions before training starts (default: 1000)
--heuristic_weight     Heuristic bonus weight; 0 = classic DQN, >0 = guided (default: 0.0)
--use_per              Enable Prioritized Experience Replay
--per_alpha            PER priority exponent (default: 0.6)
--per_beta_start       PER IS exponent start (default: 0.4)
--per_beta_frames      PER beta annealing frames (default: 100000)
--model_path           Save/load path (default: models/minimax_trained.pth)
--load_model_path      Resume from checkpoint (default: none)
--print_every          Log interval (default: 50)
--eval_every           Evaluation interval (default: 200)
--save_every           Save interval (default: 500)
--seed                 Random seed (default: 42)
--no_double_dqn        Disable Double DQN
```

---

## Rebuilding the C++ extension

The `.pyd` is already built for Windows Python 3.14 (MinGW-w64). If you need to rebuild:

### Windows (MinGW-w64)

```bash
pip install pybind11 setuptools
cd all_in_clean/cpp_solver
g++ -shared -O2 -Wall -Wextra -std=c++17
    -I.
    -IC:/Users/cube/AppData/Local/Programs/Python/Python314/include
    -IC:/Users/cube/AppData/Local/Programs/Python/Python314/Lib/site-packages/pybind11/include
    -LC:/Users/cube/AppData/Local/Programs/Python/Python314/libs
    bindings.cpp -lpython314
    -o othello_cpp.pyd
    -static-libstdc++ -static-libgcc
```

### WSL/Linux (g++)

```bash
pip install pybind11 setuptools
cd all_in_clean/cpp_solver
c++ -O2 -Wall -Wextra -std=c++17 -fPIC
    -I.
    $(python3 -m pybind11 --includes)
    bindings.cpp
    -o othello_cpp$(python3-config --extension-suffix)
    -shared
```

---

## File reference

| File in `all_in_clean/` | Lines | What it does |
|---|---|---|
| `cpp_solver/bitboard6.hpp` | 186 | 6x6 bitboard core: move gen, flip, terminal, popcount |
| `cpp_solver/solver.hpp` | 269 | Negamax + alpha-beta + Zobrist TT (exact + WLD modes) |
| `cpp_solver/symmetry.hpp` | 165 | D4 canonicalization + base-3 packing (TT efficiency) |
| `cpp_solver/bindings.cpp` | 108 | pybind11 bridge: exposes Solver and bb6 primitives |
| `cpp_solver/solver_backend.py` | 267 | Python facade: prefers C++, falls back to pure-Python |
| `cpp_solver/minimax_agent.py` | 300 | Pure-Python bitboard minimax (iterative deepening, time limits) |
| `cpp_solver/symmetry.py` | 268 | Pure-Python symmetry module |
| `agents/cpp_minimax_agent.py` | 347 | FastMinimaxAgent + CppSolverAgent wrappers |
| `train_vs_minimax.py` | 449 | DQN training script with minimax curriculum |

---

## Notes

- **Board size**: The C++ solver (`bitboard6.hpp`) is hardcoded for **6x6**. `FastMinimaxAgent` supports 4–8.
- **Python version**: The `.pyd` was compiled for Python 3.14 (Windows). The `.so` is for Linux/WSL Python 3.14.
- **Speed**: FastMinimaxAgent with `time_limit=1.0` completes a full game in ~10–20 s. Training 3000 episodes takes approximately 8–16 hours.
- **DQN variants**: All combinations of classic/guided/prioritized are supported through CLI flags.
