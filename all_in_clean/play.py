"""
play.py — Play Othello in the terminal against a DQN agent (or another rule-based agent).

Usage
-----
# Player (X) vs. DQN agent (O)
python play.py --model models/dqn.pth

# Player (O) vs. DQN agent (X) — agent starts
python play.py --model models/guided_dqn.pth --human_color -1

# Player vs. Minimax
python play.py --opponent minimax

# Player vs. Player (no agent)
python play.py --opponent human

# Different board size
python play.py --model models/dqn.pth --board_size 6

Controls
--------
Moves are entered as  "row col"  (space-separated), e.g.  "2 3"
If you must pass, enter  "p"  or  "pass"
"""

from __future__ import annotations

import argparse
import os
import sys

# Adjust path to your project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from environment import OthelloEnv
from agent import DQNAgent

# ── Terminal colors ────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RED    = "\033[91m"
_GRAY   = "\033[90m"


# ─────────────────────────────────────────────────────────────────────────────
#  Board rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_board(
    env: OthelloEnv,
    human_color: int,
    highlight_legal: bool = True,
) -> None:
    """Draws the board to the terminal with color highlighting."""
    n = env.board_size
    legal = set(env.get_legal_actions(env.current_player))
    pass_action = env.pass_action

    # Legal moves as a (row, col) set
    legal_cells = set()
    for a in legal:
        if a != pass_action:
            legal_cells.add(divmod(a, n))

    human_sym = "X" if human_color == 1 else "O"
    agent_sym = "O" if human_color == 1 else "X"
    cur_sym   = "X" if env.current_player == 1 else "O"
    is_human_turn = env.current_player == human_color

    print()
    print(f"  Tah: {env.turn_count}   "
          f"Player X={_GREEN}X{_RESET}  Agent={_YELLOW}O{_RESET}   "
          f"Na tahu: {_BOLD}{cur_sym}{_RESET}"
          f"{'  ← TY' if is_human_turn else '  ← AGENT'}")

    # Piece counts
    x_count = int(np.sum(env.board_abs == 1))
    o_count = int(np.sum(env.board_abs == -1))
    print(f"  X: {x_count} pieces   O: {o_count} pieces")
    print()

    # Column header
    header = "     " + "  ".join(f"{_GRAY}{c}{_RESET}" for c in range(n))
    print(header)
    print(f"    {'─' * (3 * n + 1)}")

    for row in range(n):
        line = f"  {_GRAY}{row}{_RESET} │"
        for col in range(n):
            val = int(env.board_abs[row, col])
            if val == 1:
                cell = f" {_GREEN}X{_RESET}"
            elif val == -1:
                cell = f" {_YELLOW}O{_RESET}"
            elif highlight_legal and (row, col) in legal_cells and is_human_turn:
                cell = f" {_CYAN}·{_RESET}"   # legal player move
            else:
                cell = f" {_GRAY}.{_RESET}"
            line += cell + " "
        print(line)

    print()
    if is_human_turn and highlight_legal:
        legal_coords = sorted(legal_cells)
        coords_str = "  ".join(f"{r},{c}" for r, c in legal_coords)
        if pass_action in legal and not legal_cells:
            coords_str = "(you must pass)"
        print(f"  {_CYAN}Legal moves: {coords_str}{_RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Read player move
# ─────────────────────────────────────────────────────────────────────────────

def get_human_action(env: OthelloEnv) -> int:
    """Interactively reads the player's move from stdin."""
    n = env.board_size
    legal = env.get_legal_actions(env.current_player)
    pass_action = env.pass_action

    # If the only legal move is pass, automatically pass
    if legal == [pass_action]:
        print(f"  {_YELLOW}You have no legal moves — automatic pass.{_RESET}")
        input("  Press Enter...")
        return pass_action

    while True:
        try:
            raw = input(f"  {_BOLD}Your move (row col) or 'p' to pass:{_RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGame over.")
            sys.exit(0)

        if raw in ("p", "pass"):
            if pass_action in legal:
                return pass_action
            print(f"  {_RED}Pass is not legal — you have regular moves.{_RESET}")
            continue

        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print(f"  {_RED}Enter two numbers separated by a space, e.g. '2 3'.{_RESET}")
            continue

        try:
            row, col = int(parts[0]), int(parts[1])
        except ValueError:
            print(f"  {_RED}Invalid input — enter integer numbers.{_RESET}")
            continue

        if not (0 <= row < n and 0 <= col < n):
            print(f"  {_RED}Coordinates out of board range (0–{n-1}).{_RESET}")
            continue

        action = row * n + col
        if action not in legal:
            print(f"  {_RED}Square ({row},{col}) is not a legal move.{_RESET}")
            continue

        return action


# ─────────────────────────────────────────────────────────────────────────────
#  Main game loop
# ─────────────────────────────────────────────────────────────────────────────

def play_game(
    env: OthelloEnv,
    human_color: int,           # 1 nebo -1
    agent,                      # DQNAgent nebo rule-based agent, nebo None (human vs human)
    agent_epsilon: float = 0.0,
) -> None:
    """Plays one game."""
    obs = env.reset()
    done = False

    human_sym = "X" if human_color == 1 else "O"
    agent_sym = "O" if human_color == 1 else "X"

    if agent is None:
        print(f"\n  {_BOLD}Player 1 (X) vs. Player 2 (O){_RESET}")
    else:
        print(f"\n  {_BOLD}You ({_GREEN}{human_sym}{_BOLD}) vs. Agent ({_YELLOW}{agent_sym}{_BOLD}){_RESET}")
    print()

    while not done:
        render_board(env, human_color)
        is_human = (env.current_player == human_color) or (agent is None)

        if is_human:
            action = get_human_action(env)
        else:
            print(f"  {_YELLOW}Agent is thinking...{_RESET}")
            if hasattr(agent, "select_action"):
                action = agent.select_action(obs, epsilon=agent_epsilon) \
                    if hasattr(agent.select_action, "__code__") and \
                       "epsilon" in agent.select_action.__code__.co_varnames \
                    else agent.select_action(obs)
            else:
                action = agent.select_action(obs)

            n = env.board_size
            pass_action = env.pass_action
            if action == pass_action:
                print(f"  Agent passes.")
            else:
                row, col = divmod(action, n)
                print(f"  Agent plays: ({row}, {col})")
            print()

        obs, _, done, info = env.step(action)

    # Final state
    render_board(env, human_color, highlight_legal=False)
    winner = info["winner"]

    print("  " + "═" * 36)
    if winner == 0:
        print(f"  {_BOLD}Draw!{_RESET}")
    elif winner == human_color:
        print(f"  {_BOLD}{_GREEN}You won! 🎉{_RESET}")
    elif agent is None:
        sym = "X" if winner == 1 else "O"
        print(f"  {_BOLD}Player {sym} won!{_RESET}")
    else:
        print(f"  {_BOLD}{_RED}Agent won.{_RESET}")

    x_count = int(np.sum(env.board_abs == 1))
    o_count = int(np.sum(env.board_abs == -1))
    print(f"  Score: X={x_count}  O={o_count}")
    print("  " + "═" * 36)
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Load agent
# ─────────────────────────────────────────────────────────────────────────────

def load_agent(
    model_path: str | None,
    opponent_type: str,
    board_size: int,
):
    """
    Returns an agent based on the request.
    If model_path is provided, loads DQNAgent from file.
    Otherwise returns a rule-based agent based on opponent_type, or None (human vs human).
    """
    if model_path is not None:
        if not os.path.exists(model_path):
            print(f"{_RED}Model file not found: {model_path}{_RESET}")
            sys.exit(1)

        agent = DQNAgent(board_size=board_size)
        agent.load(model_path)
        agent.q_net.eval()
        print(f"  Loaded DQN model: {model_path}")
        return agent

    if opponent_type == "human":
        return None

    try:
        from agents.random_agent    import RandomAgent
        from agents.greedy_agent    import GreedyAgent
        from agents.heuristic_agent import HeuristicAgent
        from agents.minimax_agent   import MinimaxAgent
    except ImportError as e:
        print(f"{_RED}Agent import failed: {e}{_RESET}")
        sys.exit(1)

    classes = {
        "random":    RandomAgent,
        "greedy":    GreedyAgent,
        "heuristic": HeuristicAgent,
        "minimax":   MinimaxAgent,
    }
    if opponent_type not in classes:
        print(f"{_RED}Unknown opponent_type '{opponent_type}'. "
              f"Options: {list(classes.keys())} or 'human'.{_RESET}")
        sys.exit(1)

    agent = classes[opponent_type](board_size)
    print(f"  Opponent: {opponent_type}")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
#  Main function
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Othello against a DQN agent or a rule-based opponent."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a saved DQN agent .pth file.",
    )
    parser.add_argument(
        "--opponent", type=str, default="random",
        choices=["random", "greedy", "heuristic", "minimax", "human"],
        help="Type of opponent when --model is not provided. 'human' = player vs. player.",
    )
    parser.add_argument(
        "--human_color", type=int, default=1, choices=[1, -1],
        help="Human player color: 1 = X (starts), -1 = O (second).",
    )
    parser.add_argument(
        "--board_size", type=int, default=6,
        help="Board size (4–8, default 6).",
    )
    parser.add_argument(
        "--agent_epsilon", type=float, default=0.0,
        help="Agent epsilon (0 = greedy, >0 = slightly random).",
    )
    args = parser.parse_args()

    env   = OthelloEnv(board_size=args.board_size)
    agent = load_agent(args.model, args.opponent, args.board_size)

    print()
    print(f"  {_BOLD}═══ Othello {args.board_size}×{args.board_size} ═══{_RESET}")
    print(f"  X = player 1 (starts)   O = player 2")
    print(f"  {_CYAN}·{_RESET} = your legal moves")
    print()

    while True:
        play_game(
            env=env,
            human_color=args.human_color,
            agent=agent,
            agent_epsilon=args.agent_epsilon,
        )

        try:
            again = input("  Play again? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if again not in ("y", "yes", "a", "ano"):
            break

    print(f"\n  {_BOLD}Goodbye!{_RESET}\n")


if __name__ == "__main__":
    main()
