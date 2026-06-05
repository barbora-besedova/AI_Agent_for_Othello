import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, board_size: int, hidden_size: int = 128):
        super().__init__()

        self.board_size = board_size
        self.input_size = board_size * board_size
        self.output_size = board_size * board_size + 1  # +1 for pass action

        self.net = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size),
        )

    def forward(self, x):
        return self.net(x)