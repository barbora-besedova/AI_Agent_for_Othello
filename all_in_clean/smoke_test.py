"""
smoke_test.py — Quick verification that all imports, GPU, and basic training work.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")

ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

from environment import OthelloEnv
from evaluation import evaluate_fair, play_game
from agent import DQNAgent
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent

print("All imports OK")

env = OthelloEnv(board_size=6)
obs = env.reset()
print(f"Env OK: board shape={obs['board'].shape}, legal={len(obs['legal_actions'])}")

for cls_name, cls in [("Random", RandomAgent), ("Greedy", GreedyAgent),
                       ("Heuristic", HeuristicAgent), ("Minimax", MinimaxAgent)]:
    ag = cls(6)
    a = ag.select_action(obs)
    assert a in obs["legal_actions"]
    print(f"  {cls_name}Agent OK -> action={a}")

agent = DQNAgent(board_size=6, use_per=True, heuristic_weight=0.0)
print(f"DQNAgent created on {agent.device}")

state = agent.encode_state(obs)
t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
with torch.no_grad():
    q = agent.q_net(t)
print(f"Q-network forward OK: {q.shape}")

a = agent.select_action(obs, epsilon=1.0)
assert a in obs["legal_actions"]
print(f"Action selection OK: {a}")

agent.store_transition(obs, a, 0.0, None, True)
ti = agent.train_step()
print(f"Train step: {ti}")

result = evaluate_fair(agent, RandomAgent, board_size=6, n_games=20)
print(f"Evaluation vs Random: {result}")

winner = play_game(agent, RandomAgent(6), board_size=6)
print(f"Play game winner: {winner}")

save_path = os.path.join(MODELS_DIR, "smoke_test.pth")
agent.save(save_path)
print("Save OK")

agent2 = DQNAgent(board_size=6, use_per=True, heuristic_weight=0.0)
agent2.load(save_path)
print("Load OK")

result2 = evaluate_fair(agent2, RandomAgent, board_size=6, n_games=10)
print(f"Loaded eval: {result2}")

print("\n✓ All smoke tests passed!")
