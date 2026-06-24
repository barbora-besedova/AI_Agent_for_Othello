"""
evaluate_models.py — Evaluate all saved models against rule-based opponents.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

from evaluation import evaluate_fair
from agent import DQNAgent
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent

OPPONENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "heuristic": HeuristicAgent,
    "minimax": MinimaxAgent,
}

# All model files with inferred board size
# Files ending in _6.pth or _6_02.pth are board_size=6
# Files ending in _5.pth are board_size=5
# Files without suffix from multi/ are board_size=5
MODEL_DEFS = [
    ("Classic DQN (bs=6)", os.path.join(MODELS_DIR, "dqn_6.pth"), 6, False, 0.0),
    ("Classic DQN (bs=5)", os.path.join(MODELS_DIR, "dqn.pth"), 5, False, 0.0),
    ("Guided DQN (bs=6)", os.path.join(MODELS_DIR, "guided_dqn_6.pth"), 6, False, 0.2),
    ("Guided DQN (bs=5)", os.path.join(MODELS_DIR, "guided_dqn_5.pth"), 5, False, 0.2),
    ("PER DQN (bs=6)", os.path.join(MODELS_DIR, "per_dqn_6.pth"), 6, True, 0.0),
    ("PER DQN (bs=5)", os.path.join(MODELS_DIR, "per_dqn_5.pth"), 5, True, 0.0),
    ("Guided PER (bs=6)", os.path.join(MODELS_DIR, "guided_per_dqn_6.pth"), 6, True, 0.2),
    ("Guided PER (bs=5)", os.path.join(MODELS_DIR, "guided_per_dqn_5.pth"), 5, True, 0.2),
]

# Also check multi-agent models
MULTI_DIR = os.path.join(MODELS_DIR, "multi")
if os.path.isdir(MULTI_DIR):
    for fname in os.listdir(MULTI_DIR):
        if fname.endswith(".pth"):
            bs = 6  # multi-agent uses board_size=5 default
            MODEL_DEFS.append((f"Multi {fname.replace('.pth','')}", os.path.join(MULTI_DIR, fname), bs, False, 0.0))

def load_agent(path, board_size, use_per, heuristic_weight):
    agent = DQNAgent(board_size=board_size, use_per=use_per, heuristic_weight=heuristic_weight)
    agent.load(path)
    agent.q_net.eval()
    return agent

print(f"{'Model':<28} {'Board':<6} {'random':<8} {'greedy':<8} {'heuristic':<10} {'minimax':<9} {'avg':<6}")
print("-" * 78)

results = []
for name, path, bs, use_per, hw in MODEL_DEFS:
    if not os.path.exists(path):
        continue
    try:
        agent = load_agent(path, bs, use_per, hw)
        scores = {}
        for opp_name, opp_cls in OPPONENTS.items():
            res = evaluate_fair(agent, opp_cls, board_size=bs, n_games=100, random_opening_plies=2)
            scores[opp_name] = res["score"]
        avg = np.mean(list(scores.values()))
        results.append((avg, name, bs, scores, path))
        pcs = f"{scores['random']:<8.3f} {scores['greedy']:<8.3f} {scores['heuristic']:<10.3f} {scores['minimax']:<9.3f} {avg:<6.3f}"
        print(f"{name:<28} {bs:<6} {pcs}")
    except Exception as e:
        print(f"{name:<28} {bs:<6} ERROR: {e}")

if results:
    results.sort(key=lambda x: x[0], reverse=True)
    print("\n── Ranking ──")
    for i, (avg, name, bs, scores, path) in enumerate(results, 1):
        print(f"  {i}. {name:<24} avg={avg:.3f} minimax={scores['minimax']:.3f} path={path}")
