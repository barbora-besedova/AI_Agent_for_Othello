# Othello DQN Agent — Project Documentation

## Overview

This project implements a **Deep Reinforcement Learning** agent that learns to play **Othello** (also known as Reversi). The agent uses **Deep Q-Learning (DQN)** with several enhancements (double DQN, prioritized experience replay, heuristic-guided learning, self-play) to learn strong strategies by playing against itself and rule-based opponents.

The codebase was built as part of a university assignment at the **Universitat Politecnica de Valencia (UPV)**. The goal is to create an agent that can be submitted to an all-against-all tournament. The final agent loads a trained model from disk and selects actions during inference — no further training occurs in the competition phase.

---

## Project Goal

Build, train, evaluate, and submit an Othello AI agent that:

- Plays Othello on a configurable board size (4×4 up to 8×8, typically **5×5** or **6×6**).
- Uses **Deep Q-Learning** to learn from self-play and against scripted opponents.
- Achieves high win rates against **random**, **greedy**, **heuristic**, and **minimax** opponents.
- Satisfies the fixed `StudentAgent` interface required for the tournament competition.

---

## Methods Used

| Method | Description |
|---|---|
| **Deep Q-Learning (DQN)** | Neural network approximates Q-values for each possible action. Uses epsilon-greedy exploration, replay buffer, and target network. |
| **Double DQN** | Decouples action selection (online network) from value estimation (target network) to reduce Q-value overestimation. |
| **Prioritized Experience Replay (PER)** | Samples transitions with higher TD-error more frequently, focusing learning on "surprising" states. |
| **Heuristic-Guided Learning** | Blends Q-learning loss with a heuristic loss (preferring corners/edges) to guide early training. |
| **Self-Play** | Agent plays against a frozen copy of itself, updated periodically. This prevents overfitting to a single opponent and forces continuous improvement. |
| **Curriculum Learning** | During training, the opponent mix shifts over time (more self-play in later phases) to progressively challenge the agent. |

---

## Directory Structure

```
├── all_in_clean/              # Main codebase
│   ├── environment.py         # Othello game rules and environment
│   ├── q_network.py           # Neural network architecture
│   ├── replay_buffer.py       # Uniform experience replay
│   ├── prioritized_replay_buffer.py  # Prioritized experience replay
│   ├── agent.py               # DQNAgent (all DQN variants)
│   ├── diagnostics.py         # Shared diagnostics (logging, game recording)
│   ├── evaluation.py          # Evaluation functions
│   ├── train.py               # Unified training script
│   ├── train_overnight.py     # Self-play-heavy overnight training
│   ├── train_multi_agent.py   # Multi-agent comparison training
│   ├── train_max_selfplay.py  # Max self-play training
│   ├── train_vs_minimax.py    # Training specifically against minimax
│   ├── evaluate_models.py     # Evaluate all saved models
│   ├── eval_overnight.py      # Evaluate overnight checkpoints
│   ├── smoke_test.py          # Quick verification script
│   ├── play.py                # Play Othello in terminal
│   ├── agents/                # Rule-based opponents
│   │   ├── random_agent.py
│   │   ├── greedy_agent.py
│   │   ├── heuristic_agent.py
│   │   ├── minimax_agent.py
│   │   ├── cpp_minimax_agent.py
│   │   └── student_agent.py
│   ├── student_agents/        # Competition-ready wrappers
│   │   └── student_agent.py
│   ├── cpp_solver/            # C++ Othello solver (native code)
│   ├── models/                # Saved model checkpoints
│   └── *.ipynb                # Jupyter notebooks
├── docs/
│   ├── Task(Professor)/       # Original assignment description
│   ├── Documentation/         # This folder
│   └── Training_History/      # Training run logs and analysis
└── .venv/                     # Python virtual environment
```

---

## File-by-File Documentation

### Core Game Logic

#### `environment.py` (327 lines)
The **Othello game engine**. Implements the full game rules:

- **`OthelloEnv`** class: board initialization, legal move computation, disc flipping, win/draw detection.
- Board representation: `board_abs` (1=black, -1=white, 0=empty) and relative board from current player's perspective.
- Observations include `board`, `board_abs`, `legal_actions`, `pass_action`, `current_player`, `board_size`, `turn_count`.
- Supports board sizes from 4×4 to 8×8 (default 6×6).

**Key method**: `step(action)` applies a move, flips discs, switches turns, and returns the new observation.

**Used by**: Every training script, evaluation, and play.

---

### Neural Network

#### `q_network.py` (42 lines)
Defines the **Q-network architecture**:

```
Input:  (batch, 1, H, W)  — board representation
  → Conv2D(1→64, 3×3, pad=1) + ReLU
  → Conv2D(64→64, 3×3, pad=1) + ReLU
  → Flatten
  → Linear(64*H*W → 128) + ReLU
  → Linear(128 → H*W+1)   — Q-values per action (+ pass)
```

The final layer outputs one Q-value per board cell plus one for the pass action.

**Used by**: `agent.py` (DQNAgent creates a QNetwork internally), `student_agents/student_agent.py`.

---

### Replay Buffers

#### `replay_buffer.py` (34 lines)
**Uniform experience replay buffer**. Stores transitions `(state, action, reward, next_state, done, next_legal_actions)` in a `deque` and samples uniformly at random.

**Used by**: DQN variants without PER.

#### `prioritized_replay_buffer.py` (94 lines)
**Prioritized Experience Replay buffer** (Schaul et al., 2016). Transitions with higher TD-error are sampled more frequently. Includes importance-sampling weights to correct the bias introduced by non-uniform sampling.

**Parameters**: `alpha` (prioritization strength, default 0.6), `beta` (annealed from 0.4 to 1.0 over training).

**Used by**: DQN variants that set `use_per=True`.

---

### Agent

#### `agent.py` (425 lines)
The **central agent class** `DQNAgent`. Supports all DQN variants:

- **Classic DQN**: Basic DQN with uniform replay.
- **Double DQN**: Uses online network to select actions, target network to evaluate them.
- **PER DQN**: Uses prioritized replay buffer.
- **Guided DQN**: Adds a heuristic auxiliary loss to encourage corner/edge play.
- **Guided PER DQN**: Combines all features (PER + heuristic + double DQN).

**Key components**:
- `select_action(obs, epsilon)` — epsilon-greedy action selection.
- `store_transition(...)` — stores experience in replay buffer.
- `train_step()` — samples batch, computes Q-loss (or guided loss), backpropagates. Returns a dict with `loss`, `mean_q` (mean predicted Q-value across batch), `grad_norm` (gradient norm before clipping), `mean_td_error`, `beta`, `mean_is_weight`.
- `save(path)` / `load(path)` — saves/loads model weights, optimizer state, training counters.
- `schedule_beta(step)` — anneals PER beta over training.

**Default parameters**:
- `learning_rate = 5e-4`, `gamma = 0.99`, `batch_size = 64`
- `buffer_capacity = 50_000`, `target_update_freq = 1_000`
- `tau = 0.005` (soft target update when used)

**Used by**: All training scripts and `play.py`.

---

### Evaluation

#### `evaluation.py` (151 lines)
Functions to **evaluate agents against each other**:

- **`play_game(env, agent_a, agent_b)`** — plays one game between two agents, alternating who plays first.
- **`evaluate_fair(agent_a, agent_b_class, board_size, n_games)`** — plays `n_games` (default 100), alternating colors, and returns `{score, wins, draws, losses}`.

**Used by**: `evaluate_models.py`, `eval_overnight.py`, all training scripts (periodic eval).

---

### Diagnostics

#### `diagnostics.py` (130 lines)
Shared utilities for **training diagnostics**, usable by any training script:

- **`MetricsLogger(path, fieldnames)`** — writes scalar metrics to a CSV file. One row per `log(**kwargs)` call, auto-flushed. Used by `train_vs_minimax.py` for the comprehensive `_train.csv` log and per-eval `_eval_ep*.csv` logs.
- **`record_game(agent, opponent, board_size, record_q, device)`** — plays a full game between two agents and returns a **transcript dict** with per-move: board state, action taken, legal actions, Q-values (if `record_q=True`), mover identity. Also returns winner, disc diff, and total moves.
- **`save_transcript(transcript, path)`** — saves a transcript dict to a JSON file.

**Example transcript (abbreviated)**:
```python
{
  "moves": [
    {"turn": 1, "mover": "agent", "player": 1,
     "board": [[...]], "action": 19, "legal_actions": [...],
     "q_values": [0.12, -0.34, ...]},
    ...
  ],
  "winner": 1, "disc_diff": 4, "n_moves": 32, "agent_player": 1
}
```

**Used by**: `train_vs_minimax.py` (game recording at key episodes).

---

### Rule-Based Opponents

All in `agents/` directory. These are the **opponents** used during training and evaluation.

#### `agents/random_agent.py` (9 lines)
Chooses a random legal move. Baseline opponent — a trained agent should easily beat it.

#### `agents/greedy_agent.py` (101 lines)
Selects the move that flips the **most opponent discs** immediately. Simple but beats random. Useful for early training.

#### `agents/heuristic_agent.py` (97 lines)
Selects the move maximizing a positional heuristic: corners (highest value) > edges > interior. Also penalizes moves adjacent to empty corners.

#### `agents/minimax_agent.py` (153 lines)
**Minimax search with alpha-beta pruning**. Default depth = 3. Evaluates leaf states with a disc-count heuristic. This is the strongest scripted opponent and the main benchmark.

#### `agents/cpp_minimax_agent.py` (362 lines)
Python wrapper around a **C++ minimax solver** (native binary in `cpp_solver/`). Much faster than the pure Python minimax. Used as the primary opponent during `train_vs_minimax.py` runs. See `docs/cpp_minimax_solver/README.md` for build instructions.

---

### Competition Agent

#### `student_agents/student_agent.py` (194 lines)
The **competition submission wrapper**. Satisfies the exact interface required by the tournament:

```python
agent = StudentAgent(board_size=6, checkpoint_path="othello_agent.pt")
action = agent.select_action(observation)
```

- Loads a `QNetwork` from a saved checkpoint (`.pth` file).
- Masks illegal actions with `-inf` before argmax.
- Falls back to a heuristic agent if no checkpoint is provided.
- **No training occurs** — pure inference.

**This is the file to submit for the competition.**

---

### Training Scripts

#### `train.py` (468 lines)
**Unified training script**. The main training harness with extensive command-line options:

```
python train.py --board_size 6 --num_episodes 5000 \
  --use_per --heuristic_weight 0.2 --double_dqn \
  --opponent_mix random:0.1 greedy:0.2 heuristic:0.3 minimax:0.4
```

**Key features**:
- Configurable opponent mix (weighted random selection per episode).
- Epsilon-greedy exploration with decay.
- Replay buffer warmup before learning starts.
- Periodic evaluation and checkpoint saving.
- Result logging and visualization.

**Supports all DQN variants**: `--use_per`, `--heuristic_weight`, `--double_dqn`.

#### `train_overnight.py` (337 lines)
**Self-play-heavy overnight training**. Used for long, unattended training runs. Loads a pretrained model and continues training with:

- **Three-phase curriculum**: Phase 1 (0-30%): 60% self-play → Phase 2 (30-70%): 70% self-play → Phase 3 (70-100%): 80% self-play.
- Frozen self-play opponent updated every 500 episodes.
- Evaluation every 1000 episodes (100 games vs each opponent).
- Checkpoint saving every 5000 episodes.
- Starts with low epsilon (0.2) to preserve existing knowledge.

```
python train_overnight.py --num_episodes 30000
```

#### `train_multi_agent.py` (384 lines)
Trains **multiple agents simultaneously** with different hyperparameters, then evaluates and ranks them. Useful for hyperparameter search.

```
python train_multi_agent.py --num_episodes 2000 --num_agents 4
```

#### `train_max_selfplay.py` (471 lines)
Training with **maximum self-play focus**. The agent plays almost exclusively against itself with periodic frozen-copy updates.

```
python train_max_selfplay.py --num_episodes 10000 --board_size 6
```

#### `train_vs_minimax.py` (590+ lines)
Training focused on **beating the minimax opponent**. Uses a curriculum opponent mix that progressively introduces the fast C++ minimax. The primary training script for the `guided_per_dqn` agent used in the `against_minimax` experiments.

**Key features**:
- **Curriculum opponent schedule**: starts weak (random/greedy/heuristic), shifts to ~65% minimax by late training.
- **ε-greedy exploration** with opponent-aware epsilon floors (minimax floor = 0.20).
- **Periodic evaluation** every `--eval_every` episodes, with **per-color breakdown** for minimax (as black vs as white, revealing color asymmetry).
- **Full game recording** at configurable episodes (`--record_game_eps`, default `1200,3000`), saves Q-values and board states to JSON.
- **Comprehensive CSV logging**: `{model_path}_train.csv` (episode, epsilon, reward, win%, loss, mean_q, grad_norm, td_error) and per-eval `_eval_ep*.csv` (per-opponent, per-color).
- **Profile mode** (`--profile`) prints timing breakdown of agent forward pass / opponent search / env step / train step.
- **Time limit** (`--max_minutes`) for graceful stop-and-save.

```
python train_vs_minimax.py --use_per --heuristic_weight 0.2 \
  --minimax_max_depth 3 --model_path models/against_minimax/minimax_trained.pth \
  --load_model_path models/guided_per_dqn_6_best_overnight.pth \
  --max_minutes 180 --profile
```

**Additional CLI flags**:
| Flag | Default | Description |
|---|---|---|
| `--minimax_max_depth` | `None` | Fixed search depth for minimax opponent |
| `--profile` | `False` | Print timing breakdown % every log interval |
| `--record_game_eps` | `1200,3000` | Comma-separated episode numbers to record full games |
| `--n_record_games` | `3` | Games to record per checkpoint (agent as both colors) |

---

### Evaluation and Analysis

#### `evaluate_models.py` (79 lines)
**Benchmark all saved models** against all rule-based opponents (random, greedy, heuristic, minimax). Plays 100 games per opponent, prints a ranking table.

```
python evaluate_models.py
```

#### `eval_overnight.py` (58 lines)
**Evaluate overnight training checkpoints** specifically. Tests each saved checkpoint from `train_overnight.py` and compares against the original model.

```
python eval_overnight.py
```

#### `smoke_test.py` (73 lines)
**Quick verification** that the codebase works correctly. Tests:
- Imports (all modules)
- GPU availability
- Q-network forward pass
- Agent action selection
- Model save/load
- Evaluation function

```
python smoke_test.py
```

---

### Interactive Play

#### `play.py` (357 lines)
**Play Othello in the terminal** against a trained DQN agent or a rule-based opponent.

```
# Play against a trained model
python play.py --model models/guided_per_dqn_6.pth

# Play against minimax
python play.py --opponent minimax

# Two-player mode
python play.py --opponent human
```

Uses terminal colors and displays legal moves. Supports board sizes 4-8.

---

### Models

All trained models are saved in `models/` as `.pth` files (PyTorch state dictionaries).

**Naming convention**: `<variant>_<board_size>.pth`
- `dqn_6.pth` — Classic DQN, board_size=6
- `guided_dqn_6.pth` — Guided DQN, board_size=6
- `per_dqn_6.pth` — PER DQN, board_size=6
- `guided_per_dqn_6.pth` — Guided + PER DQN, board_size=6 (best model)
- `guided_per_dqn_6_best_overnight.pth` — Best overnight checkpoint
- `_overnight_ep*.pth` — Checkpoints saved during overnight training
- `multi/` — Multi-agent training results

**Checkpoint format**:
```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_steps": train_steps,
    "board_size": board_size,
}
```

---

## How to Run

### Prerequisites
```bash
# Create virtual environment (if not exists)
python -m venv .venv
# Activate (Linux/WSL)
source .venv/bin/activate
# Or (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install torch numpy
```

### Verify the codebase works
```bash
cd all_in_clean
python smoke_test.py
```

### Train a new model
```bash
# Quick test (200 episodes)
python train.py --board_size 6 --num_episodes 200 --use_per --heuristic_weight 0.2

# Full training (5000 episodes with self-play curriculum)
python train_overnight.py --num_episodes 30000
```

### Training diagnostics (--profile)

The `--profile` flag prints a **timing breakdown** every log interval:

```
prof: agent=1% opp=0% env=1% train=97% store=0%
```

This identifies the bottleneck — typically **`train` (backpropagation)** dominates at ~97%.

The training CSV (`{model_path}_train.csv`) records key metrics per interval:
- **`mean_q`** — mean predicted Q-value. A sudden spike signals **value blow-up** (divergence), pointing at a too-high learning rate.
- **`grad_norm`** — gradient norm before clipping. Spikes indicate **instability**; flat-and-low is healthy.
- **`mean_td_error`** — average TD error magnitude (PER priority proxy).

### Interpreting per-color win rates

The evaluation CSV (`{model_path}_eval_ep*.csv`) breaks minimax results by colour:

```
fast_minimax,as_black,9,2,14,0.360,0.400
fast_minimax,as_white,6,3,16,0.240,0.300
```

A large gap (e.g. 36% as black vs 24% as white) reveals **colour asymmetry** — common in Othello DQN agents. Training with balanced colours (alternating `agent_player` each episode) mitigates this.

### Analysing recorded games

Full game transcripts are saved to `{model_dir}/games/ep{episode}_game{n}.json`. Each move includes board state, action, and Q-values, allowing post-hoc analysis of:
- Does the agent lose narrowly (close disc diff) or get crushed?
- Does it go wrong in the **opening**, **midgame**, or **endgame**?
- Are Q-values well-calibrated (high Q for good moves, low for blunders)?

### Evaluate models
```bash
python evaluate_models.py
```

### Play against the agent
```bash
python play.py --model models/guided_per_dqn_6_best_overnight.pth --board_size 6
```

### Prepare competition submission
```bash
# Copy the best model to the student_agents folder
cp models/guided_per_dqn_6_best_overnight.pth student_agents/othello_agent.pt

# The student_agent.py file is the submission file
```

---

## Training Configuration Summary

| Parameter | Typical Value | Description |
|---|---|---|
| `board_size` | 6 | Board side length |
| `num_episodes` | 5,000–30,000 | Total training episodes |
| `learning_rate` | 5e-4 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 64 | Transitions per training step |
| `buffer_capacity` | 50,000–100,000 | Replay buffer size |
| `target_update_freq` | 1,000 | Steps between target network syncs |
| `epsilon_start` | 1.0 (fresh) / 0.2 (pretrained) | Initial exploration rate |
| `epsilon_end` | 0.02 | Final exploration rate |
| `epsilon_decay` | 0.9995–0.9999 | Per-episode decay multiplier |
| `use_per` | True | Enable prioritized replay |
| `heuristic_weight` | 0.2 | Weight of heuristic auxiliary loss |
| `double_dqn` | True | Enable double DQN |
| `per_alpha` | 0.6 | Prioritization exponent |
| `per_beta_start` | 0.4 | Initial importance-sampling weight |
| `minimax_max_depth` | None (unlimited) | Fixed search depth for minimax |
| `max_minutes` | None (no limit) | Graceful stop-and-save after N minutes |
| `profile` | False | Print timing breakdown % |
| `record_game_eps` | `1200,3000` | Episodes to record full game transcripts |
| `n_record_games` | 3 | Games per recording checkpoint |

---

## Best Model Results

The best performing model is **Guided PER (board_size=6)**, further improved by overnight training:

| Opponent | Win Rate |
|---|---|
| Random | 0.950 |
| Greedy | 1.000 |
| Heuristic | 1.000 |
| Minimax | 0.810 |
| **Average** | **0.940** |

Peak performance during training reached **0.980** vs minimax (at episodes 16,000 and 18,000).

---

## Competition Interface

The final submission must satisfy the following interface:

```python
class StudentAgent:
    def __init__(self, board_size: int, checkpoint_path: str | None = None):
        ...
    def select_action(self, observation: dict) -> int:
        ...
```

- `observation["board"]` — relative board (1=my piece, -1=opponent, 0=empty)
- `observation["legal_actions"]` — list of valid action indices
- Action `board_size * board_size` = pass action
- Must return a legal action (illegal action = automatic loss)

See `student_agents/student_agent.py` for the implementation.
