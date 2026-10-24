import logging
import os
import tempfile
import warnings

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset

from src.nn.litmodel import LitModel

os.environ["LT_CLOUD"] = "0"

warnings.filterwarnings("ignore")

logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class LightningTabularRegressor(RegressorMixin, BaseEstimator):
    """
    Lightning based tabular regressor

    Uses sklearn-like API to fit and predict
    """

    def __init__(
        self,
        model: type[LitModel],
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=200,
        patience=20,
        scheduler_patience=5,
        min_lr=1e-6,
        num_workers=0,
        seed=42,
        device="cpu",
        numeric_cols=None,
        verbose=True,
        **kwargs,
    ):
        """
        Args:
            model (type[LitModel]): model to train
            lr (float): learning rate
            weight_decay (float): weight decay
            batch_size (int): batch size
            max_epochs (int): maximum number of epochs
            patience (int): patience for early stopping
            scheduler_patience (int): patience for learning rate scheduler
            min_lr (float): minimum learning rate
            num_workers (int): number of workers
            seed (int): random seed
            device (str): device to train on
            numeric_cols (list): list of numeric columns
            verbose (bool): whether to print logs
            **kwargs (dict): model arguments
        """
        self.kwargs = kwargs
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.scheduler_patience = scheduler_patience
        self.min_lr = min_lr
        self.num_workers = num_workers
        self.seed = seed
        self.numeric_cols = numeric_cols
        self.verbose = verbose
        self.device = torch.device(device)

    @staticmethod
    def _target_array(y):
        """Convert target to numpy array"""

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()

        return np.asarray(y, dtype=np.float32).reshape(-1)

    def _prepare_fit(self, X: pd.DataFrame):
        """
        Prepare data for training

        Convert to numpy array
        """
        X = X.copy()

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        if self.numeric_cols is None:
            self.numeric_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        else:
            self.numeric_cols_ = list(self.numeric_cols)

        missing = set(self.numeric_cols_) - set(X.columns)
        if missing:
            raise ValueError(f"Numeric columns not found: {sorted(missing)}")

        return X.astype(np.float32)

    def _prepare_predict(self, X: pd.DataFrame):
        """Prepare data for prediction"""
        X = X.copy()

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Feature columns or their order differ from fit().")

        return X.astype(np.float32)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.DataFrame | None = None,
    ):
        """
        Train model

        Args:
            X (pd.DataFrame): training features
            y (pd.DataFrame): training targets
            X_valid (pd.DataFrame | None): validation features
            y_valid (pd.DataFrame | None): validation targets
        """

        # Seed pytorch for reproducibility
        L.seed_everything(self.seed, workers=True, verbose=False)

        X = X.copy()
        y = self._target_array(y)

        if X_valid is None or y_valid is None:
            raise ValueError("Pass X_valid and y_valid for early stopping.")

        X_valid = X_valid.copy()
        y_valid = self._target_array(y_valid)

        if len(X) != len(y):
            raise ValueError("X and y have different lengths.")

        if len(X_valid) != len(y_valid):
            raise ValueError("X_valid and y_valid have different lengths.")

        # Preprocess data
        X_train_processed = self._prepare_fit(X)
        X_valid_processed = self._prepare_predict(X_valid)

        # Instantiate model using kwargs
        self.model_ = self.model(
            **self.kwargs,
            input_dim=X.shape[1],
            lr=self.lr,
            weight_decay=self.weight_decay,
            min_lr=self.min_lr,
            scheduler_patience=self.scheduler_patience,
        )

        # Create dataloaders
        train_ds = TensorDataset(
            torch.tensor(X_train_processed.to_numpy(), dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

        valid_ds = TensorDataset(
            torch.tensor(X_valid_processed.to_numpy(), dtype=torch.float32),
            torch.tensor(y_valid, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        checkpoint_dir = tempfile.mkdtemp(prefix="lightning_best_")

        # Save best model to temp checkpoint directory
        # Best model is saved based on validation rmse
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            monitor="val_rmse",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        )

        # Early stopping on validation rmse
        early_stopping = EarlyStopping(
            monitor="val_rmse",
            mode="min",
            patience=self.patience,
            min_delta=1e-4,
            verbose=False,
        )

        # Train model with early stopping and checkpoint
        # Use deterministic mode for reproducibility
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[early_stopping, checkpoint],
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=self.verbose,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
        )

        trainer.fit(
            self.model_,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

        best_path = checkpoint.best_model_path

        # Restore best model
        if best_path:
            best_state = torch.load(
                best_path,
                map_location="cpu",
                weights_only=True,
            )["state_dict"]
            self.model_.load_state_dict(best_state)

        self.best_val_rmse_ = float(checkpoint.best_model_score)

        self.model_.to(self.device)
        self.model_.eval()

        return self

    def predict(self, X: pd.DataFrame):
        """
        Make predictions

        Args:
            X (pd.DataFrame): features

        Returns:
            numpy.ndarray
        """
        X = pd.DataFrame(X).copy()
        X_processed = self._prepare_predict(X)

        # Create dataloader
        dataset = TensorDataset(
            torch.tensor(X_processed.to_numpy(), dtype=torch.float32)
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        predictions = []

        self.model_.eval()

        with torch.inference_mode():
            for (xb,) in loader:
                xb = xb.to(self.device)
                predictions.append(self.model_(xb).cpu().numpy())

        return np.concatenate(predictions).reshape(-1)
