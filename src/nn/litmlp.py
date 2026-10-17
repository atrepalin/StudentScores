from src.nn.litmodel import LitModel
from src.nn.mlp import MLP


class LitMLP(LitModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        min_lr: float = 1e-6,
        scheduler_patience: int = 5,
    ):
        super().__init__(
            model=MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                output_dim=1,
            ),
            lr=lr,
            weight_decay=weight_decay,
            min_lr=min_lr,
            scheduler_patience=scheduler_patience,
        )
