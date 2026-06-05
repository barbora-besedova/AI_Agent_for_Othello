"""
agent_submission_template.py

This is the file students should submit after training.

They may train with Q-learning / Deep Q-learning in a separate notebook or script.
For the competition, no training should happen here. This file only loads the
trained model and selects actions.
"""

from __future__ import annotations

import random
import numpy as np

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Example neural network for a Deep Q-Learning agent.

    IMPORTANT:
    Students must use here the SAME architecture they used during training.
    If they trained a different network, they must replace this class with
    their own architecture.
    torch.save(
    {
        "model_state_dict": model.state_dict(),
        "board_size": board_size
    },
    "othello_agent.pt"
    )
    """

    def __init__(self, board_size: int):
        super().__init__()

        input_size = board_size * board_size
        output_size = board_size * board_size + 1  # board positions + pass action

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)


class StudentAgent:
    """
    Required public interface.

    The teacher/referee will instantiate the class as:

        agent = StudentAgent(board_size=6, checkpoint_path="my_checkpoint.pt")

    and will repeatedly call:

        action = agent.select_action(observation)
    """

    def __init__(self, board_size: int, checkpoint_path: str | None = None):
        self.board_size = board_size
        self.pass_action = board_size * board_size

        self.rng = random.Random(0)

        # Device fixed to CPU for evaluation.
        # This avoids problems if the teacher's computer has no GPU.
        self.device = torch.device("cpu")

        # Create the model architecture.
        self.model = DQN(board_size).to(self.device)

        if checkpoint_path is None:
            raise ValueError("A checkpoint_path is required for evaluation.")

        # Load the trained PyTorch checkpoint.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        # Recommended saving format:
        #
        # torch.save({
        #     "model_state_dict": model.state_dict(),
        #     "board_size": board_size
        # }, "othello_agent.pt")
        #
        # But we also allow the checkpoint to be directly a state_dict.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        # Evaluation mode: disables training-specific behavior.
        self.model.eval()

    def encode_observation(self, observation: dict) -> torch.Tensor:
        """
        Convert the observation into a tensor usable by the neural network.

        The board is relative:
             1 = my pieces
            -1 = opponent pieces
             0 = empty cells

        Shape returned:
            (1, board_size * board_size)
        """
        board = observation["board"]

        x = board.reshape(-1).astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32, device=self.device)

        return x.unsqueeze(0)

    def select_action(self, observation: dict) -> int:
        """
        Return exactly one integer action.

        Valid actions:
            0 ... board_size*board_size - 1: position row, col
            board_size*board_size: pass

        The action must be legal. The referee will end the game as a loss
        if the returned action is illegal.
        """
        legal_actions = observation["legal_actions"]

        # Safety fallback.
        if len(legal_actions) == 0:
            return self.pass_action

        x = self.encode_observation(observation)

        with torch.no_grad():
            q_values = self.model(x)[0].cpu().numpy()

        # Important:
        # We do NOT take np.argmax(q_values), because that may be illegal.
        # We only choose among legal actions.
        best_action = max(
            legal_actions,
            key=lambda action: q_values[action]
        )

        return int(best_action)