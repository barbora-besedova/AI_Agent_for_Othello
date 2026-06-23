"""
eval_overnight.py — Evaluate all training checkpoints from experiments_002
against scripted opponents (Random, Heuristic, FastMinimax depth 3)
and plot winrate vs training episodes.
"""

import sys, os, glob, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt

from evaluation import evaluate_fair
from agent import DQNAgent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.cpp_minimax_agent import FastMinimaxAgent

MODELS_DIR = os.path.join(
    os.path.dirname(__file__),
    "Training", "experiments_002", "model_checkpoints",
)

N_GAMES = 100
BOARD_SIZE = 6
RANDOM_OPENING_PLIES = 2

OPPONENTS = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "minimax_d3": lambda bs: FastMinimaxAgent(board_size=bs, max_depth=3),
}

ckpts = sorted(
    glob.glob(os.path.join(MODELS_DIR, "model_ep*.pth")),
    key=lambda p: int(
        re.search(r"model_ep(\d+)\.pth", os.path.basename(p)).group(1)
    ),
)

print(f"Found {len(ckpts)} checkpoints in {MODELS_DIR}\n")

episodes = []
results = {name: [] for name in OPPONENTS}

for ckpt in ckpts:
    ep = int(re.search(r"model_ep(\d+)\.pth", os.path.basename(ckpt)).group(1))
    episodes.append(ep)

    agent = DQNAgent(board_size=BOARD_SIZE, use_per=True, heuristic_weight=0.2)
    agent.load(ckpt)
    agent.q_net.eval()

    print(f"\n--- Episode {ep} ({os.path.basename(ckpt)}) ---")
    for opp_name, opp_fn in OPPONENTS.items():
        res = evaluate_fair(
            agent, opp_fn,
            board_size=BOARD_SIZE,
            n_games=N_GAMES,
            random_opening_plies=RANDOM_OPENING_PLIES,
        )
        score = res["score"]
        results[opp_name].append(score)
        wr = res["win_rate"]
        print(f"  vs {opp_name:<14s}  score={score:.3f}  win_rate={wr:.3f}  "
              f"({res['wins']}W/{res['draws']}D/{res['losses']}L)")

header = f"{'Episode':<10}" + "".join(f"{name:<14s}" for name in OPPONENTS)
print("\n\n" + "=" * (10 + 14 * len(OPPONENTS)))
print(header)
print("-" * (10 + 14 * len(OPPONENTS)))
for i, ep in enumerate(episodes):
    row = f"{ep:<10d}" + "".join(f"{results[name][i]:<14.3f}" for name in OPPONENTS)
    print(row)
print("=" * (10 + 14 * len(OPPONENTS)))

plt.figure(figsize=(10, 6))
for opp_name in OPPONENTS:
    plt.plot(episodes, results[opp_name], marker="o", label=opp_name)
plt.xlabel("Training Episodes")
plt.ylabel("Win Rate (score)")
plt.title("Model Win Rate vs Scripted Opponents Over Training")
plt.legend()
plt.grid(True)
out_path = os.path.join(os.path.dirname(__file__), "eval_winrate_vs_episodes.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {out_path}")
