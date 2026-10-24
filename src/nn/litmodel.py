import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.regression import MeanSquaredError


class LitModel(L.LightningModule):
    """Lightning module for regression"""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        min_lr: float = 1e-6,
        scheduler_patience: int = 5,
    ):
        """
        Args:
            model (nn.Module): model to train
            lr (float): learning rate
            weight_decay (float): weight decay
            min_lr (float): minimum learning rate
            scheduler_patience (int): patience for learning rate scheduler
        """
        super().__init__()

        # Store kwargs
        self.save_hyperparameters(ignore="model")

        self.model = model

        # Use MSE as loss
        self.criterion = nn.MSELoss()

        # Use RMSE as metric
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)

    def forward(self, x: torch.Tensor):
        """Model forward pass"""
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Training step

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): batch
            batch_idx (int): batch index
        """
        x, y = batch

        pred = self(x)
        loss = self.criterion(pred, y)

        self.train_rmse.update(pred.detach(), y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self):
        """Log train RMSE at the end of each epoch"""
        self.log("train_rmse", self.train_rmse.compute(), prog_bar=True)
        self.train_rmse.reset()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Validation step

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): batch
            batch_idx (int): batch index
        """
        x, y = batch

        pred = self(x)
        loss = self.criterion(pred, y)

        self.val_rmse.update(pred, y)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        """Log val RMSE at the end of each epoch"""
        self.log("val_rmse", self.val_rmse.compute(), prog_bar=True)
        self.val_rmse.reset()

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""

        # Use AdamW with weight decay
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Use ReduceLROnPlateau scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.hparams.scheduler_patience,
            min_lr=self.hparams.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_rmse",
                "interval": "epoch",
                "frequency": 1,
            },
        }
