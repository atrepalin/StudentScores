from src.nn.litmodel import LitModel
from src.nn.mlp import MLP


class LitMLP(LitModel):
    """MLP model for regression."""

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
        """
        Args:
            input_dim (int): input dimension
            hidden_dims (tuple): hidden dimensions
            dropout (float): dropout rate
            lr (float): learning rate
            weight_decay (float): weight decay
            min_lr (float): minimum learning rate
            scheduler_patience (int): patience for learning rate scheduler
        """
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
