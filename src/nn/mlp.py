import torch
from torch import nn


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return (
            torch.squeeze(x, dim=self.dim) if self.dim is not None else torch.squeeze(x)
        )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_features = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, output_dim))
        layers.append(Squeeze(-1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
