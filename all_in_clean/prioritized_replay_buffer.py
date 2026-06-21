import numpy as np
from typing import List, Tuple


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (Schaul et al., 2016).

    Each transition is stored as:
        (state, action, reward, next_state, done, next_legal_actions)

    New transitions receive the current maximum priority so they are
    sampled at least once quickly.

    Args:
        capacity:  Maximum number of transitions to store.
        alpha:     Exponent controlling how much prioritization is used.
                   0 = uniform, 1 = full prioritization.
        epsilon:   Small constant added to every TD-error to avoid zero priority.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        epsilon: float = 1e-5,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon

        self.buffer: list = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_legal_actions,
    ) -> None:
        transition = (state, action, reward, next_state, done, next_legal_actions)

        max_priority = (
            1.0 if len(self.buffer) == 0
            else self.priorities[: len(self.buffer)].max()
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        current_size = len(self.buffer)

        priorities = self.priorities[:current_size]
        scaled = priorities ** self.alpha
        probabilities = scaled / scaled.sum()

        indices = np.random.choice(
            current_size,
            size=batch_size,
            replace=False,
            p=probabilities,
        )

        batch = [self.buffer[i] for i in indices]

        # Importance-sampling correction weights.
        weights = (current_size * probabilities[indices]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return batch, indices, weights

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        self.priorities[indices] = np.abs(td_errors) + self.epsilon

    def __len__(self) -> int:
        return len(self.buffer)
