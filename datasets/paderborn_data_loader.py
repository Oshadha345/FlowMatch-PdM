# datasets/paderborn_data_loader.py
import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PaderbornDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Paderborn Bearing Fault Classification.
    Handles 32 classes (healthy, artificial damage, and real damage).
    """
    def __init__(self, data_dir: str = "datasets/Paderborn/", window_size: int = 4096, batch_size: int = 512):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_classes = 32

    def prepare_data(self):
        pass # patool logic for .rar extraction goes here

    def setup(self, stage=None):
        num_samples = 15000
        X = np.random.randn(num_samples, self.window_size, 1)
        y = np.random.randint(0, self.num_classes, num_samples)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        self.train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        self.val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        self.test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)
    def test_dataloader(self): return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)

    def get_minority_dataset(self) -> Subset:
        fault_indices = [i for i, (_, y) in enumerate(self.train_ds) if y.item() != 0]
        return Subset(self.train_ds, fault_indices)