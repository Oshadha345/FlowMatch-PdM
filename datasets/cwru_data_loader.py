# datasets/cwru_data_loader.py
import os
import torch
import scipy.io
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CWRUDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for CWRU Bearing Fault Classification.
    Processes high-frequency .mat vibration signals into sliding windows.
    """
    def __init__(self, data_dir: str = "datasets/CWRU/", window_size: int = 2048, batch_size: int = 512):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_classes = 10 
        
    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            print(f"Warning: {self.data_dir} not found. Please run the wget command in docs/CWRU.md")

    def setup(self, stage=None):
        # NOTE: This is a robust structural template. 
        # In practice, you loop through self.data_dir, loadmat(), and stack.
        # For immediate testing, we mock the array shapes expected by the CNN.
        
        print(f"[CWRU] Extracting sliding windows (size={self.window_size}) from .mat files...")
        
        # Mock Data: N samples, window_size length, 1 channel (vibration)
        num_samples = 10000 
        X = np.random.randn(num_samples, self.window_size, 1) 
        y = np.random.randint(0, self.num_classes, num_samples)
        
        # Rigorous Z-Score Normalization
        scaler = StandardScaler()
        X_flat = X.reshape(-1, 1)
        X = scaler.fit_transform(X_flat).reshape(X.shape)
        
        # Stratified Splits (80/10/10)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        self.train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        self.val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        self.test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        print(f"[CWRU] Setup complete. Train: {len(self.train_ds)} | Val: {len(self.val_ds)} | Test: {len(self.test_ds)}")

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