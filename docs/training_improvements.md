# Training Improvements — 2026-06-22

Fixes applied to address evaluation determinism, self-play cycling, and value-estimation stability.

## 1. Evaluation Determinism (the 0.500 artifact)

**Problem:** Against deterministic opponents (Greedy, Heuristic) the agent's argmax policy produced identical games within each color. 100-game evaluations collapsed to 2 data points — one per color — making scores only ever `1.000` (win both) or `0.500` (win one).

**Fix:** `play_game()` now plays `random_opening_plies=2` random moves before the agents take control (`evaluation.py:78-85`). This creates unique board states for each game, so `n_games=100` produces 100 distinct trajectories.

Random tie-breaking was also added to the agent's action selection (`agent.py:178-180`, `agent.py:196-209`). When multiple actions share the maximum Q-value (or guided score), one is chosen uniformly at random instead of always picking the first. This injects diversity at eval time and in self-play without hurting performance — all max-Q actions are equally good by definition.

| Metric | Before | After |
|---|---|---|
| Unique games vs Greedy (n=100) | 2 | 100 |
| Unique games vs Heuristic (n=100) | 2 | 100 |
| Greedy/Heuristic eval scores | Only 0.500, 1.000 | Continuous range |

All training scripts (`train.py`, `train_overnight.py`) and evaluation scripts (`eval_overnight.py`, `evaluate_models.py`) pass `random_opening_plies=2` to `evaluate_fair()`. This is now the default in `evaluate_fair()` itself.

## 2. Curriculum — Prevent Minimax Forgetting

**Problem:** Phase 3 of the overnight training curriculum (`train_overnight.py`) used 80% self-play and only 15% minimax. At high self-play ratios the agent specialized against its own style and lost the ability to beat minimax — visible as late-phase oscillation in the eval log.

**Fix:** Minimax is now kept at **≥20%** in all phases, and self-play is capped at **70%** in phase 3:

| Phase | Episodes | Random | Greedy | Heuristic | **Minimax** | Self-Play |
|---|---|---|---|---|---|---|
| 1 (0-30%) | 0–9k | 5% | 5% | 10% | **20%** | 60% |
| 2 (30-70%) | 9k–21k | 5% | 5% | 5% | **20%** | 65% |
| 3 (70-100%) | 21k–30k | 0% | 0% | 5% | **25%** | 70% |

## 3. Epsilon Floor for Strong Opponents

**Problem:** `train_overnight.py` decayed epsilon uniformly regardless of opponent. Against minimax at late training, epsilon could drop to 0.02, effectively eliminating exploration when facing the strongest opponent.

**Fix:** `_epsilon_for_opponent()` (`train_overnight.py:97-100`) floors epsilon at 0.15 for greedy/heuristic and 0.20 for minimax, matching the existing behavior in `train.py`.

## 4. Polyak Soft Target Updates

**Problem:** Hard target-network updates every N steps (the classic DQN approach) produce abrupt target shifts, contributing to the "deadly triad" instability (bootstrapping + function approximation + off-policy), especially during self-play.

**Fix:** A new `tau` parameter (default `0.005`) enables Polyak averaging (`agent.py:401-406`):

```
θ_target ← τ · θ_online + (1 − τ) · θ_target
```

Applied **every train step** instead of a periodic hard copy. This smooths the target trajectory and reduces oscillation. Set `tau=0.0` to restore classic hard updates.

The `tau` value is saved in checkpoints for reproducibility.

## Files Modified

| File | Changes |
|---|---|
| `evaluation.py` | Random opening plies in `play_game()` and `evaluate_fair()` |
| `agent.py` | Random tie-breaking in `select_action()` and `_guided_argmax()`, Polyak `tau` parameter, soft-update logic |
| `train_overnight.py` | Curriculum weights adjusted, `_epsilon_for_opponent()` added, `--eval_random_plies` CLI arg |
| `train.py` | `eval_random_plies` parameter and CLI arg, wired to `evaluate_fair()` |
| `eval_overnight.py` | `random_opening_plies=2` passed to `evaluate_fair()` |
| `evaluate_models.py` | `random_opening_plies=2` passed to `evaluate_fair()` |
