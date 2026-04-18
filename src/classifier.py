# Contains the Lightning modules for downstream classifiers and regressors.

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from mamba_ssm import Mamba
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError, R2Score

class CNN1DClassifier(pl.LightningModule):
    """
    1D-CNN Baseline for Bearing Fault Classification (CWRU / Paderborn).
    Optimized for extracting high-frequency spatial-temporal features.
    """
    def __init__(self, num_classes: int, input_channels: int = 1, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Architecture: 5 Conv blocks (filters 16→64→128→256→256) + GAP + FC
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 16, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv1d(16, 64, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self._test_preds = []
        self._test_targets = []

    def _prepare_input(self, x):
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected rank-2 or rank-3 input, received shape {tuple(x.shape)}")

        expected_channels = int(self.hparams.input_channels)
        if x.shape[1] == expected_channels:
            return x
        if x.shape[2] == expected_channels:
            return x.transpose(1, 2)
        raise ValueError(
            f"Unable to align channels for CNN input. Expected {expected_channels} channels, "
            f"received shape {tuple(x.shape)}."
        )

    def extract_features(self, x):
        x = self._prepare_input(x)
        return self.feature_extractor(x).squeeze(-1)

    def forward(self, x):
        return self.classifier(self.extract_features(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.train_acc(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self._test_preds.append(preds.detach().cpu())
        self._test_targets.append(y.detach().cpu())

    def on_test_epoch_start(self):
        self._test_preds = []
        self._test_targets = []

    def on_test_epoch_end(self):
        if not self._test_preds:
            return

        y_pred = torch.cat(self._test_preds).numpy()
        y_true = torch.cat(self._test_targets).numpy()

        metrics = {
            "test_acc": float(accuracy_score(y_true, y_pred)),
            "test_balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "test_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "test_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "test_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "test_mcc": float(matthews_corrcoef(y_true, y_pred)),
        }

        for name, value in metrics.items():
            self.log(name, value, prog_bar=name in {"test_acc", "test_f1_macro"}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


class LSTMRegressor(pl.LightningModule):
    """
    LSTM Baseline for Remaining Useful Life (RUL) Regression (CMAPSS / FEMTO).
    Optimized for capturing long-term temporal degradation dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 100,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        target_scale: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.target_scale = float(target_scale)
        if self.target_scale <= 0:
            raise ValueError(f"target_scale must be positive, received {target_scale}.")

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        # Metrics
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self._test_preds = []
        self._test_targets = []

    def _restore_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.target_scale

    def extract_features(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]

    def forward(self, x):
        # Input shape expected: (Batch, Sequence_Length, Features)
        return self.regressor(self.extract_features(x)).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.huber_loss(y_hat, y.float())

        self.train_rmse(self._restore_scale(y_hat), self._restore_scale(y.float()))
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_rmse', self.train_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.huber_loss(y_hat, y.float())

        preds_real = self._restore_scale(y_hat)
        y_real = self._restore_scale(y.float())
        self.val_rmse(preds_real, y_real)
        self.val_mae(preds_real, y_real)
        self.val_r2(preds_real, y_real)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_rmse', self.val_rmse, prog_bar=True, sync_dist=True)
        self.log('val_mae', self.val_mae, sync_dist=True)
        self.log('val_r2', self.val_r2, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        preds_real = self._restore_scale(y_hat)
        y_real = self._restore_scale(y.float())

        self._test_preds.append(preds_real.detach().cpu())
        self._test_targets.append(y_real.detach().cpu())

    def on_test_epoch_start(self):
        self._test_preds = []
        self._test_targets = []

    def on_test_epoch_end(self):
        if not self._test_preds:
            return

        y_pred = torch.cat(self._test_preds).numpy()
        y_true = torch.cat(self._test_targets).numpy()

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        denom = np.maximum(np.abs(y_true), 1e-8)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

        smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
        smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / smape_denom) * 100.0)

        metrics = {
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_mape": mape,
            "test_smape": smape,
        }

        for name, value in metrics.items():
            self.log(name, value, prog_bar=name in {"test_rmse", "test_mae"}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class CNN1DRegressor(pl.LightningModule):
    """
    1D-CNN regressor for long-sequence RUL forecasting.

    This baseline emphasizes local receptive fields and serves as the
    convolutional comparison against the sequence-model baselines.
    """

    def __init__(
        self,
        input_channels: int = 1,
        learning_rate: float = 1e-3,
        target_scale: float = 1.0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.target_scale = float(target_scale)
        if self.target_scale <= 0:
            raise ValueError(f"target_scale must be positive, received {target_scale}.")

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self._test_preds = []
        self._test_targets = []

    def _restore_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.target_scale

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected rank-3 input, received shape {tuple(x.shape)}")
        return x.transpose(1, 2).contiguous()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(self._prepare_input(x)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor_head(self.feature_extractor(self._prepare_input(x))).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.huber_loss(y_hat, y.float())

        self.train_rmse(self._restore_scale(y_hat), self._restore_scale(y.float()))
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.huber_loss(y_hat, y.float())

        preds_real = self._restore_scale(y_hat)
        y_real = self._restore_scale(y.float())
        self.val_rmse(preds_real, y_real)
        self.val_mae(preds_real, y_real)
        self.val_r2(preds_real, y_real)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_rmse", self.val_rmse, prog_bar=True, sync_dist=True)
        self.log("val_mae", self.val_mae, sync_dist=True)
        self.log("val_r2", self.val_r2, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        preds_real = self._restore_scale(y_hat)
        y_real = self._restore_scale(y.float())
        self._test_preds.append(preds_real.detach().cpu())
        self._test_targets.append(y_real.detach().cpu())

    def on_test_epoch_start(self):
        self._test_preds = []
        self._test_targets = []

    def on_test_epoch_end(self):
        if not self._test_preds:
            return

        y_pred = torch.cat(self._test_preds).numpy()
        y_true = torch.cat(self._test_targets).numpy()

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        denom = np.maximum(np.abs(y_true), 1e-8)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

        smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
        smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / smape_denom) * 100.0)

        metrics = {
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_mape": mape,
            "test_smape": smape,
        }

        for name, value in metrics.items():
            self.log(name, value, prog_bar=name in {"test_rmse", "test_mae"}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class TransformerRegressor(pl.LightningModule):
    """
    Patch-transformer regressor for long-horizon RUL forecasting.

    Patching keeps attention tractable on 2048/2560-length sequences while exposing
    a quadratic-attention baseline for the paper comparison.
    """

    def __init__(
        self,
        input_channels: int = 1,
        patch_size: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        target_scale: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.target_scale = float(target_scale)
        self.patch_size = int(patch_size)
        self.input_channels = int(input_channels)
        if self.target_scale <= 0:
            raise ValueError(f"target_scale must be positive, received {target_scale}.")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, received {patch_size}.")

        self.patch_proj = nn.Linear(self.patch_size * self.input_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self._test_preds = []
        self._test_targets = []

    def _restore_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.target_scale

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected rank-3 input, received shape {tuple(x.shape)}")

        batch_size, seq_len, channels = x.shape
        if channels != self.input_channels:
            raise ValueError(
                f"Expected input_channels={self.input_channels}, received last dimension {channels}."
            )
        if seq_len % self.patch_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by patch_size={self.patch_size}."
            )

        num_patches = seq_len // self.patch_size
        return x.contiguous().view(batch_size, num_patches, self.patch_size * channels)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._patchify(x)
        x = self.patch_proj(x)
        x = self.transformer(x)
        return x[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x)).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.huber_loss(y_hat, y.float())

        self.train_rmse(self._restore_scale(y_hat), self._restore_scale(y.float()))
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.huber_loss(y_hat, y.float())

        preds_real = self._restore_scale(y_hat)
        y_real = self._restore_scale(y.float())
        self.val_rmse(preds_real, y_real)
        self.val_mae(preds_real, y_real)
        self.val_r2(preds_real, y_real)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_rmse", self.val_rmse, prog_bar=True, sync_dist=True)
        self.log("val_mae", self.val_mae, sync_dist=True)
        self.log("val_r2", self.val_r2, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        preds_real = self._restore_scale(y_hat)
        y_real = self._restore_scale(y.float())
        self._test_preds.append(preds_real.detach().cpu())
        self._test_targets.append(y_real.detach().cpu())

    def on_test_epoch_start(self):
        self._test_preds = []
        self._test_targets = []

    def on_test_epoch_end(self):
        if not self._test_preds:
            return

        y_pred = torch.cat(self._test_preds).numpy()
        y_true = torch.cat(self._test_targets).numpy()

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        denom = np.maximum(np.abs(y_true), 1e-8)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

        smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
        smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / smape_denom) * 100.0)

        metrics = {
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_mape": mape,
            "test_smape": smape,
        }

        for name, value in metrics.items():
            self.log(name, value, prog_bar=name in {"test_rmse", "test_mae"}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class MambaRULRegressor(pl.LightningModule):
    """
    State-space regressor for long-window RUL forecasting.

    This is the active downstream regressor for the FEMTO/XJTU-SY pivot and is
    designed to remain stable on 2048-2560 step windows where the LSTM baseline
    tends to collapse to the mean.
    """

    def __init__(
        self,
        input_channels: int = 1,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        learning_rate: float = 1e-3,
        target_scale: float = 1.0,
        context_channels: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.target_scale = float(target_scale)
        self.input_channels = int(input_channels)
        self.context_channels = int(context_channels)
        if self.target_scale <= 0:
            raise ValueError(f"target_scale must be positive, received {target_scale}.")
        self.input_proj = nn.Linear(self.input_channels, d_model)
        self.mamba_block = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.regressor_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self._test_preds = []
        self._test_targets = []

    def _restore_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.target_scale

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x)
        hidden = self.mamba_block(hidden)
        return hidden[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rul_prediction = self.regressor_head(self.extract_features(x))
        return rul_prediction.squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.huber_loss(y_hat, y.float())

        self.train_rmse(self._restore_scale(y_hat), self._restore_scale(y.float()))
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.huber_loss(y_hat, y.float())

        preds_real = self._restore_scale(y_hat)
        y_real = self._restore_scale(y.float())
        self.val_rmse(preds_real, y_real)
        self.val_mae(preds_real, y_real)
        self.val_r2(preds_real, y_real)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_rmse", self.val_rmse, prog_bar=True, sync_dist=True)
        self.log("val_mae", self.val_mae, sync_dist=True)
        self.log("val_r2", self.val_r2, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        preds_real = self._restore_scale(y_hat)
        y_real = self._restore_scale(y.float())

        self._test_preds.append(preds_real.detach().cpu())
        self._test_targets.append(y_real.detach().cpu())

    def on_test_epoch_start(self):
        self._test_preds = []
        self._test_targets = []

    def on_test_epoch_end(self):
        if not self._test_preds:
            return

        y_pred = torch.cat(self._test_preds).numpy()
        y_true = torch.cat(self._test_targets).numpy()

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        denom = np.maximum(np.abs(y_true), 1e-8)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

        smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
        smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / smape_denom) * 100.0)

        metrics = {
            "test_mse": mse,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_mape": mape,
            "test_smape": smape,
        }

        for name, value in metrics.items():
            self.log(name, value, prog_bar=name in {"test_rmse", "test_mae"}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
