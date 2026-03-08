# datasets/cwru_data_loader.py
import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, Subset


class CWRUDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for CWRU Bearing Fault Classification.
    Loads individual .npy files created by notebooks/01_dataset_analysis.ipynb.
    DE signals | Window=2048 | Step=512 | Z-score | 10 classes.
    Uses np.load(mmap_mode='r') for near-instant loading.
    """
    def __init__(self, data_dir: str = "datasets/processed/cwru",
                 window_size: int = 2048, batch_size: int = 512, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_classes = 10

    def prepare_data(self):
        required = os.path.join(self.data_dir, "X_train.npy")
        if not os.path.exists(required):
            raise FileNotFoundError(
                f"{self.data_dir}/ not found. Run notebooks/01_dataset_analysis.ipynb first."
            )

    def setup(self, stage=None):
        def _load(name):
            return np.load(os.path.join(self.data_dir, f"{name}.npy"), mmap_mode='r')

        X_train, y_train = _load("X_train"), _load("y_train")
        X_val,   y_val   = _load("X_val"),   _load("y_val")
        X_test,  y_test  = _load("X_test"),  _load("y_test")

        self.train_ds = TensorDataset(
            torch.tensor(np.array(X_train), dtype=torch.float32),
            torch.tensor(np.array(y_train), dtype=torch.long),
        )
        self.val_ds = TensorDataset(
            torch.tensor(np.array(X_val), dtype=torch.float32),
            torch.tensor(np.array(y_val), dtype=torch.long),
        )
        self.test_ds = TensorDataset(
            torch.tensor(np.array(X_test), dtype=torch.float32),
            torch.tensor(np.array(y_test), dtype=torch.long),
        )
        print(f"[CWRU] Loaded from {self.data_dir}/  "
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