import torch
import torch.nn as nn

from src.nn.seblock import SEBlock


class ResBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        reduction: int = 4,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock(dim, reduction)

    def forward(self, x: torch.Tensor):
        residual = x

        out = self.norm1(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout(out)

        out = self.se(out)

        return residual + out
