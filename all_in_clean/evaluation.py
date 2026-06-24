"""
from environments.environment import OthelloEnv

def play_game(agent_1, agent_2, board_size=6):
    env = OthelloEnv(board_size=board_size)
    obs = env.reset()
    done = False

    while not done:
        if env.current_player == 1:
            action = agent_1.select_action(obs)
        else:
            action = agent_2.select_action(obs)

        obs, reward, done, info = env.step(action)

    return info["winner"], info["disc_diff"]


def evaluate_fair(agent_a_class, agent_b_class, board_size=6, n_games=200):
    a_wins = 0
    b_wins = 0
    draws = 0

    for _ in range(n_games // 2):
        agent_a = agent_a_class(board_size)
        agent_b = agent_b_class(board_size)

        winner, _ = play_game(agent_a, agent_b, board_size)

        if winner == 1:
            a_wins += 1
        elif winner == -1:
            b_wins += 1
        else:
            draws += 1

    for _ in range(n_games // 2):
        agent_b = agent_b_class(board_size)
        agent_a = agent_a_class(board_size)

        winner, _ = play_game(agent_b, agent_a, board_size)

        if winner == 1:
            b_wins += 1
        elif winner == -1:
            a_wins += 1
        else:
            draws += 1

    return {
        "agent_a_wins": a_wins,
        "agent_b_wins": b_wins,
        "draws": draws,
        "n_games": n_games,
    }
"""

import random

from environment import OthelloEnv


def play_game(agent_1, agent_2, board_size=6, random_opening_plies=2):
    """
    Plays one complete game.

    agent_1 always plays as player 1.
    agent_2 always plays as player -1.
    """
    env = OthelloEnv(board_size=board_size)
    observation = env.reset()
    done = False

    # Random opening plies — creates genuine game diversity against deterministic
    # opponents (Greedy, Heuristic) so that n_games actually generates n distinct
    # trajectories instead of just 2 (one per color).
    for _ in range(random_opening_plies):
        if done:
            break
        legal = observation["legal_actions"]
        pass_action = observation.get("pass_action", board_size * board_size)
        non_pass = [a for a in legal if a != pass_action]
        action = random.choice(non_pass if non_pass else legal)
        observation, _, done, info = env.step(action)

    while not done:
        if env.current_player == 1:
            action = agent_1.select_action(observation)
        else:
            action = agent_2.select_action(observation)

        observation, _, done, info = env.step(action)

    return info["winner"]


def evaluate_fair(
    agent_a,
    agent_b_class,
    board_size=6,
    n_games=200,
    random_opening_plies=2,
):
    """
    Evaluates an existing instance of agent_a against agent_b_class.

    Half the games agent_a plays as player 1.
    The other half agent_a plays as player -1.
    """
    if n_games <= 0:
        raise ValueError("n_games must be positive.")

    a_wins = 0
    a_losses = 0
    draws = 0

    games_as_first = n_games // 2
    games_as_second = n_games - games_as_first

    # Agent A plays as player 1.
    for _ in range(games_as_first):
        agent_b = agent_b_class(board_size)

        winner = play_game(
            agent_1=agent_a,
            agent_2=agent_b,
            board_size=board_size,
            random_opening_plies=random_opening_plies,
        )

        if winner == 1:
            a_wins += 1
        elif winner == -1:
            a_losses += 1
        else:
            draws += 1

    # Agent A plays as player -1.
    for _ in range(games_as_second):
        agent_b = agent_b_class(board_size)

        winner = play_game(
            agent_1=agent_b,
            agent_2=agent_a,
            board_size=board_size,
            random_opening_plies=random_opening_plies,
        )

        if winner == -1:
            a_wins += 1
        elif winner == 1:
            a_losses += 1
        else:
            draws += 1

    win_rate = a_wins / n_games

    # Score where win = 1, draw = 0.5, loss = 0.
    score = (a_wins + 0.5 * draws) / n_games

    return {
        "wins": a_wins,
        "draws": draws,
        "losses": a_losses,
        "n_games": n_games,
        "win_rate": win_rate,
        "score": score,
    }