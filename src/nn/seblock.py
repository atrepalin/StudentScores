import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels: int, reduction: int = 4):
        """
        Args:
            channels (int): number of input channels
            reduction (int): reduction factor
        """
        super().__init__()

        hidden = max(channels // reduction, 8)

        # Standard squeeze-and-excitation block
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # As we use SEBlock for tabular data, we don't need to use average pooling
        s = self.fc(x)
        return x * s
