import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_legal_actions):
        self.buffer.append(
            (state, action, reward, next_state, done, next_legal_actions)
        )

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)