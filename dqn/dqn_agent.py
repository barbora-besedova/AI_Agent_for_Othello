import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.q_network import QNetwork
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        board_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_freq: int = 500,
        device: str | None = None,
    ):
        self.board_size = board_size
        self.action_size = board_size * board_size + 1
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_net = QNetwork(board_size).to(self.device)
        self.target_net = QNetwork(board_size).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.train_steps = 0

    def encode_state(self, observation: dict):
        board = observation["board"].astype(np.float32)
        state = board.flatten()
        return state

    def select_action(self, observation: dict, epsilon: float = 0.0) -> int:
        legal_actions = observation["legal_actions"]

        if len(legal_actions) == 0:
            return observation["pass_action"]

        if random.random() < epsilon:
            return random.choice(legal_actions)

        state = self.encode_state(observation)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze(0).cpu().numpy()

        masked_q_values = np.full(self.action_size, -np.inf)
        masked_q_values[legal_actions] = q_values[legal_actions]

        action = int(np.argmax(masked_q_values))

        if action not in legal_actions:
            action = legal_actions[0]

        return action

    def store_transition(self, observation, action, reward, next_observation, done):
        state = self.encode_state(observation)

        if next_observation is None:
            next_state = np.zeros_like(state, dtype=np.float32)
            next_legal_actions = []
        else:
            next_state = self.encode_state(next_observation)
            next_legal_actions = next_observation["legal_actions"]

        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done,
            next_legal_actions,
        )

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = np.array([item[0] for item in batch], dtype=np.float32)
        actions = np.array([item[1] for item in batch], dtype=np.int64)
        rewards = np.array([item[2] for item in batch], dtype=np.float32)
        next_states = np.array([item[3] for item in batch], dtype=np.float32)
        dones = np.array([item[4] for item in batch], dtype=np.float32)
        next_legal_actions_batch = [item[5] for item in batch]

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states_tensor)
        current_q_values = q_values.gather(1, actions_tensor).squeeze(1)

        with torch.no_grad():
            next_q_values_all = self.target_net(next_states_tensor).cpu().numpy()
            max_next_q_values = []

            for i, legal_actions in enumerate(next_legal_actions_batch):
                if dones[i] or len(legal_actions) == 0:
                    max_next_q_values.append(0.0)
                else:
                    legal_q_values = next_q_values_all[i][legal_actions]
                    max_next_q_values.append(float(np.max(legal_q_values)))

            max_next_q_values_tensor = torch.tensor(
                max_next_q_values,
                dtype=torch.float32,
                device=self.device,
            )

            target_q_values = rewards_tensor + self.gamma * (1.0 - dones_tensor) * max_next_q_values_tensor

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        torch.save(
            {
                "board_size": self.board_size,
                "model_state_dict": self.q_net.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["model_state_dict"])
        self.q_net.eval()
        self.target_net.eval()