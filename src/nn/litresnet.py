from src.nn.litmodel import LitModel
from src.nn.resnet import TabularResNet


class LitTabResNet(LitModel):
    """ResNet model for regression"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
        head_dims: tuple = (64, 16),
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        min_lr: float = 1e-6,
        scheduler_patience: int = 5,
    ):
        """
        Args:
            input_dim (int): input dimension
            hidden_dim (int): hidden dimension
            n_blocks (int): number of blocks
            dropout (float): dropout rate
            head_dims (tuple): head dimensions
            lr (float): learning rate
            weight_decay (float): weight decay
            min_lr (float): minimum learning rate
            scheduler_patience (int): patience for learning rate scheduler
        """
        super().__init__(
            model=TabularResNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_blocks=n_blocks,
                dropout=dropout,
                head_dims=head_dims,
                output_dim=1,
            ),
            lr=lr,
            weight_decay=weight_decay,
            min_lr=min_lr,
            scheduler_patience=scheduler_patience,
        )
