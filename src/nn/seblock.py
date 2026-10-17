import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        s = self.fc(x)
        return x * s
