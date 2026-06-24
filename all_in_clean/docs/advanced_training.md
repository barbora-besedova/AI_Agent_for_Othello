# Advanced Training Launcher

`advanced_training.py` is a user-friendly wrapper around `train_vs_minimax.py`
that automates experiment tracking, checkpoint management, and resumability.

---

## Quick Start

```bash
# New experiment (default: Training/experiments_001, _002, ...)
uv run python advanced_training.py \
    --load_model_path models/guided_per_dqn_6_best_overnight.pth

# Resume an interrupted experiment
uv run python advanced_training.py --experiment_dir Training/experiments_001
```

---

## Folder Structure

```
Training/
└── experiments_001/
    ├── setup.json                   ← all training params + base model info
    ├── best_model.pth               ← model with highest minimax eval score
    ├── resume_state.json            ← episode + epsilon for resuming
    ├── model_checkpoints/
    │   ├── model_ep1000.pth         ← preserved copies at each eval interval
    │   └── model_ep2000.pth
    └── results/
        ├── latest.pth               ← most recent model (overwritten)
        ├── latest_train.csv         ← training metrics (loss, Q, grad norm)
        ├── latest_eval_ep1000.csv   ← evaluation results per opponent
        └── games/                   ← recorded game transcripts (JSON)
```

### What goes where

| File | Purpose | Created |
|---|---|---|
| `setup.json` | Full parameter documentation for scientific comparison | Once at start |
| `best_model.pth` | Keeps the best-performing model by minimax score | Updated after each eval |
| `resume_state.json` | Saves episode counter + epsilon for clean resume | Every save interval |
| `model_checkpoints/model_epN.pth` | Historical checkpoint at episode N | Every save interval |
| `results/latest.pth` | Current model (used internally for CSV path derivation) | Every save interval |
| `results/latest_train.csv` | Training metrics over time | Incremented each print step |
| `results/latest_eval_epN.csv` | Per-opponent eval at episode N | Every eval interval |
| `results/games/*.json` | Full move-by-move game transcripts | At specified episodes |

---

## Configuration

### Default Parameters

| Argument | Default | Description |
|---|---|---|
| `--base_dir` | `Training/experiments` | Base path for new experiment folders |
| `--num_episodes` | 30 000 | Total training episodes |
| `--board_size` | 6 | Board size (4–8) |
| `--minimax_max_depth` | 5 | Minimax search depth cap |
| `--minimax_time_limit` | 1.0 s | Fallback time limit per move |
| `--heuristic_weight` | 0.2 | Guided DQN bonus weight |
| `--use_per` | True | Prioritized Experience Replay |
| `--batch_size` | 512 | Mini-batch size |
| `--learning_rate` | 1e-3 | Adam learning rate |
| `--epsilon_start` | 0.05 | Exploration rate start |
| `--epsilon_end` | 0.01 | Exploration rate floor |
| `--epsilon_decay` | 0.9995 | Per-episode decay multiplier |
| `--eval_every` | 1000 | Evaluation + checkpoint frequency |
| `--save_every` | 1000 | Model save frequency |
| `--max_minutes` | 148 | Graceful stop after N minutes |
| `--double_dqn` | True | Double DQN target |

All arguments from `train_vs_minimax.py` are forwardable and override the
defaults above.

---

## Usage Modes

### New Experiment

```bash
uv run python advanced_training.py \
    --load_model_path path/to/base_model.pth \
    --heuristic_weight 0.2 \
    --use_per
```

- Creates `Training/experiments_001/` (or `_002`, `_003`, ... auto-incremented)
- Reads the base model checkpoint for SHA256 + config documentation
- Saves `setup.json` with all parameters
- Runs training

You can also specify a custom base path:

```bash
uv run python advanced_training.py \
    --base_dir Training/my_custom_name \
    --load_model_path path/to/model.pth
```

This creates `Training/my_custom_name_001/`.

### Resume Experiment

```bash
uv run python advanced_training.py \
    --experiment_dir Training/experiments_001
```

- Reads `resume_state.json` to get last completed episode + epsilon
- Loads the checkpoint from `model_checkpoints/model_ep{N}.pth`
- Continues training from episode `N + 1`
- Appends resume metadata to `setup.json`

You can also override parameters on resume (e.g., change `--max_minutes`):

```bash
uv run python advanced_training.py \
    --experiment_dir Training/experiments_001 \
    --max_minutes 300
```

---

## Inner Workings

### Experiment Lifecycle

1. **Startup**
   - Parse CLI args
   - Determine experiment directory (new: auto-increment / resume: read state)
   - Load base model, compute SHA256 hash, extract config
   - Save `setup.json` with all params + base model info
   - Create subdirectories: `model_checkpoints/`, `results/`

2. **Training** (delegates to `train_vs_minimax.train()`)
   - Creates DQN agent with specified config
   - Loads base model weights
   - Runs episode loop with opponent curriculum
   - Logs metrics to CSV every `print_every` episodes

3. **Checkpointing** (every `save_every` / `eval_every` episodes)
   - Saves model to `results/latest.pth`
   - Copies to `model_checkpoints/model_ep{N}.pth` (preserved)
   - Saves `resume_state.json` with current episode + epsilon
   - Runs evaluation against all opponents
   - If minimax score improved, saves `best_model.pth`

4. **Shutdown**
   - On `--max_minutes` expiry: saves model + closes logs gracefully
   - On crash: last checkpoint + `resume_state.json` are intact for resume
   - On manual kill (`pkill -f advanced_training`): same — `resume_state.json`
     has the last save point

### Resume Mechanics

The resume system relies on three files:

```
resume_state.json
├── episode      → last fully completed episode
├── epsilon      → exploration rate at that point
└── model_path   → path to the model checkpoint

model_checkpoints/model_ep{N}.pth   → model weights + optimizer + config
setup.json                          → original training parameters
```

On resume:
1. Read `resume_state.json` → get `episode`, `epsilon`
2. Load model from `model_checkpoints/model_ep{episode}.pth`
3. Set `start_episode = episode + 1`, `epsilon_start = epsilon`
4. Run training loop from `start_episode` onward
5. `setup.json` is updated with a `resumed_from` entry

### Minimax Depth Cap

By default, minimax is capped at depth 5 (not time-limited). This makes
training dramatically faster:

| Setting | Time per minimax game | Time for 30k episodes (65% minimax) |
|---|---|---|
| Time-limited (1s/move) | ~15 s | ~81 h |
| Depth-capped (depth 5) | <0.1 s | <30 min |

The depth-5 minimax is still a strong opponent for a 6×6 board — it can solve
most mid-game positions and will only struggle in the opening or highly
complex endgame lines.

---

## Comparing Experiments

Each experiment folder is fully self-documenting. To compare runs:

```bash
cat Training/experiments_001/setup.json
cat Training/experiments_002/setup.json
```

Key fields for comparison:

- `base_model.sha256` — which starting weights were used
- `base_model.config` — heuristic_weight, use_per, etc. from base model
- `training_params.*` — all training hyperparameters
- `timestamp` — when the experiment started
- `resumed_from` — present if experiment was resumed

The CSV metrics (`results/latest_train.csv`, `results/latest_eval_ep*.csv`)
contain the actual performance data for plotting.
