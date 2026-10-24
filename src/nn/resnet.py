import torch
import torch.nn as nn

from src.nn.resblock import ResBlock


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
        head_dims: tuple = (64, 16),
        output_dim: int = 1,
    ):
        """
        Args:
            input_dim (int): input dimension
            hidden_dim (int): hidden dimension
            n_blocks (int): number of blocks
            dropout (float): dropout rate
            head_dims (tuple): head dimensions
            output_dim (int): output dimension
        """
        super().__init__()

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.dropout_in = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                ResBlock(
                    dim=hidden_dim,
                    dropout=dropout,
                    reduction=4,
                )
                for _ in range(n_blocks)
            ]
        )

        head_layers = []
        in_features = hidden_dim

        for head_dim in head_dims:
            head_layers.extend(
                [
                    nn.Linear(in_features, head_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = head_dim

        head_layers.append(nn.Linear(in_features, output_dim))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.dropout_in(x)

        x = self.blocks(x)
        x = self.head(x)

        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x
