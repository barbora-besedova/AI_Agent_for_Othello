import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Convolutional Q-network shared by all DQN variants.

    Input:  (batch, 1, board_size, board_size)  — board relative to current player
    Output: (batch, board_size * board_size + 1) — Q-values incl. pass action
    """

    def __init__(
        self,
        board_size: int,
        channels: int = 64,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.board_size = board_size
        self.output_size = board_size * board_size + 1

        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flattened_size = channels * board_size * board_size

        self.head = nn.Sequential(
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.head(x)
