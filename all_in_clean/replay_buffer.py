import random
from collections import deque
from typing import List, Tuple, Any


class ReplayBuffer:
    """
    Uniform experience replay buffer.

    Each transition is stored as:
        (state, action, reward, next_state, done, next_legal_actions)
    """

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def get_state(self) -> dict:
        return {"buffer": list(self.buffer)}

    def set_state(self, state: dict) -> None:
        self.buffer = deque(state["buffer"], maxlen=self.capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_legal_actions,
    ) -> None:
        self.buffer.append(
            (state, action, reward, next_state, done, next_legal_actions)
        )

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
