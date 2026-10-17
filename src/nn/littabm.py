import torch
from tabm import TabM

from src.nn.litmodel import LitModel


class LitTabM(LitModel):
    def __init__(
        self,
        input_dim: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        min_lr: float = 1e-6,
        scheduler_patience: int = 5,
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 512,
        dropout: float = 0.1,
        arch_type: str = "tabm",
    ):
        super().__init__(
            model=TabM.make(
                n_num_features=input_dim,
                d_out=1,
                k=k,
                n_blocks=n_blocks,
                d_block=d_block,
                dropout=dropout,
                arch_type=arch_type,
            ),
            lr=lr,
            weight_decay=weight_decay,
            min_lr=min_lr,
            scheduler_patience=scheduler_patience,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).mean(dim=1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch

        predictions = self.model(x)
        targets = y.reshape(-1, 1, 1)

        loss = self.criterion(
            predictions,
            targets.expand_as(predictions),
        )

        mean_prediction = predictions.mean(dim=1)
        self.train_rmse.update(mean_prediction.detach(), y.reshape(-1, 1))

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch

        predictions = self(x)
        targets = y.reshape(-1, 1)

        loss = self.criterion(predictions, targets)

        self.val_rmse.update(predictions, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
            prog_bar=True,
        )
