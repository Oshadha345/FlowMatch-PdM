# datasets/cwru_data_loader.py
import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, Subset


class CWRUDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for CWRU Bearing Fault Classification.
    Loads pre-processed .npz created by notebooks/01_dataset_analysis.ipynb.
    DE signals | Window=2048 | Step=512 | Z-score | 10 classes.
    """
    def __init__(self, npz_path: str = "datasets/processed/cwru_processed.npz",
                 window_size: int = 2048, batch_size: int = 512, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.npz_path = npz_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_classes = 10

    def prepare_data(self):
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(
                f"{self.npz_path} not found. Run notebooks/01_dataset_analysis.ipynb first."
            )

    def setup(self, stage=None):
        data = np.load(self.npz_path)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val,   y_val   = data["X_val"],   data["y_val"]
        X_test,  y_test  = data["X_test"],  data["y_test"]

        self.train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        self.val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        self.test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )
        print(f"[CWRU] Loaded from {self.npz_path}  "
              f"Train: {len(self.train_ds)} | Val: {len(self.val_ds)} | Test: {len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def get_minority_dataset(self) -> Subset:
        """Extracts strictly fault classes (labels 1-9) for generative models."""
        fault_indices = [i for i, (_, y) in enumerate(self.train_ds) if y.item() != 0]
        print(f"[CWRU] Extracted {len(fault_indices)} minority (fault) samples.")
        return Subset(self.train_ds, fault_indices)