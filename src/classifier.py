# Contains the PyTorch classes for `CNN1D` and `LSTMClassifier`

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, MeanSquaredError, MeanAbsoluteError

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
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        # Input shape expected: (Batch, Channels, Sequence_Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.shape[-1] != x.shape[1] and x.shape[1] > 10:
            # Swap axes if input is (Batch, Seq_Len, Channels)
            x = x.transpose(1, 2)
            
        features = self.feature_extractor(x)
        features = features.squeeze(-1)
        return self.classifier(features)

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
        
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log('test_acc', self.test_acc)
        self.log('test_f1', self.test_f1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


class LSTMRegressor(pl.LightningModule):
    """
    LSTM Baseline for Remaining Useful Life (RUL) Regression (CMAPSS / FEMTO).
    Optimized for capturing long-term temporal degradation dependencies.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 100, num_layers: int = 2, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
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
        
        self.criterion = nn.MSELoss()
        
        # Metrics
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()

    def forward(self, x):
        # Input shape expected: (Batch, Sequence_Length, Features)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step for RUL prediction
        last_step_features = lstm_out[:, -1, :]
        return self.regressor(last_step_features).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())
        
        self.train_rmse(y_hat, y.float())
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_rmse', self.train_rmse, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())
        
        self.val_rmse(y_hat, y.float())
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_rmse', self.val_rmse, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        self.test_rmse(y_hat, y.float())
        self.test_mae(y_hat, y.float())
        self.log('test_rmse', self.test_rmse)
        self.log('test_mae', self.test_mae)

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