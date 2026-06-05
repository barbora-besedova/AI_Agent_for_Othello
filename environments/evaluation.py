from environments.environment import OthelloEnv


def play_game(agent_1, agent_2, board_size=5):
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


def evaluate_fair(agent_a_class, agent_b_class, board_size=5, n_games=200):
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