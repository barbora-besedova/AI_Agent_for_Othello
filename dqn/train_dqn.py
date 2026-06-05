import os
import numpy as np

from environments.environment import OthelloEnv
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.heuristic_agent import HeuristicAgent
from dqn.dqn_agent import DQNAgent


def compute_final_reward(winner, dqn_player):
    if winner == dqn_player:
        return 1.0
    elif winner == 0:
        return 0.0
    else:
        return -1.0


def make_opponent(opponent_type, board_size):
    if opponent_type == "random":
        return RandomAgent(board_size)

    if opponent_type == "greedy":
        return GreedyAgent(board_size)

    if opponent_type == "heuristic":
        return HeuristicAgent(board_size)

    raise ValueError(
        "Unknown opponent_type. Use 'random', 'greedy', or 'heuristic'."
    )

def train_dqn(
    board_size=5,
    num_episodes=5000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    model_path="models/othello_dqn.pth",
    opponent_type="random",
):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Training DQN against: {opponent_type}")

    env = OthelloEnv(board_size=board_size)
    agent = DQNAgent(board_size=board_size)

    epsilon = epsilon_start

    rewards_history = []
    win_history = []

    wins = 0
    losses = 0
    draws = 0

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False

        opponent = make_opponent(opponent_type, board_size)

        # Střídáme, jestli DQN hraje jako 1 nebo -1.
        dqn_player = 1 if episode % 2 == 0 else -1

        # Pokud DQN hraje jako -1, první tah musí udělat soupeř.
        if env.current_player != dqn_player:
            opponent_action = opponent.select_action(obs)
            obs, _, done, info = env.step(opponent_action)

        episode_reward = 0.0

        while not done:
            # Teď by měl být na tahu DQN.
            state_obs = obs

            action = agent.select_action(state_obs, epsilon=epsilon)

            obs_after_dqn, _, done, info = env.step(action)

            if done:
                winner = info["winner"]
                reward = compute_final_reward(winner, dqn_player)
                next_obs = obs_after_dqn

                agent.store_transition(
                    state_obs,
                    action,
                    reward,
                    next_obs,
                    done,
                )

                agent.train_step()
                episode_reward = reward
                break

            # Soupeř odpoví.
            opponent_action = opponent.select_action(obs_after_dqn)
            obs_after_opponent, _, done, info = env.step(opponent_action)

            if done:
                winner = info["winner"]
                reward = compute_final_reward(winner, dqn_player)
            else:
                reward = 0.0

            # Toto je důležité:
            # next_obs je stav po tahu soupeře, tedy další stav pro DQN.
            agent.store_transition(
                state_obs,
                action,
                reward,
                obs_after_opponent,
                done,
            )

            agent.train_step()

            obs = obs_after_opponent
            episode_reward = reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        rewards_history.append(episode_reward)

        if episode_reward > 0:
            wins += 1
            win_history.append(1)
        elif episode_reward < 0:
            losses += 1
            win_history.append(0)
        else:
            draws += 1
            win_history.append(0.5)

        if episode % 100 == 0:
            recent_rewards = rewards_history[-100:]
            recent_winrate = win_history[-100:]

            print(
                f"Episode {episode}/{num_episodes} | "
                f"opponent={opponent_type} | "
                f"epsilon={epsilon:.3f} | "
                f"avg_reward_100={np.mean(recent_rewards):.3f} | "
                f"winrate_100={np.mean(recent_winrate):.3f} | "
                f"W/D/L={wins}/{draws}/{losses}"
            )

        if episode % 500 == 0:
            agent.save(model_path)

    agent.save(model_path)
    print(f"Model saved to: {model_path}")

    return agent, rewards_history, win_history