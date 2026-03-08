# datasets/rul_data_loader.py
import pytorch_lightning as pl
import rul_datasets
from torch.utils.data import DataLoader, Subset
import numpy as np

class FlowMatchRULDataModule(pl.LightningDataModule):
    """
    Unified PyTorch Lightning DataModule for Engine and Bearing RUL Tracks.
    Powered by the `rul-datasets` library for automated downloading, 
    windowing, and feature scaling.
    """
    def __init__(self, dataset_name: str, fd: int = 1, window_size: int = 30, batch_size: int = 512):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name.upper()
        self.fd = fd
        self.window_size = window_size
        self.batch_size = batch_size
        
        # Initialize the correct reader based on the dataset
        self.reader = self._get_reader()
        
        # RulDataModule handles the internal PyTorch Dataset creation
        self.rul_dm = rul_datasets.RulDataModule(self.reader, batch_size=self.batch_size)
        
        # These will be populated during setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _get_reader(self):
        """Instantiates the specific reader with physical constraints."""
        if self.dataset_name == "CMAPSS":
            # max_rul=125 implements the piecewise-linear RUL clipping standard in literature
            return rul_datasets.CmapssReader(self.fd, window_size=self.window_size, max_rul=125)
        elif self.dataset_name == "N-CMAPSS":
            return rul_datasets.NCmapssReader(self.fd, window_size=self.window_size)
        elif self.dataset_name == "FEMTO":
            return rul_datasets.FemtoReader(self.fd, window_size=self.window_size)
        elif self.dataset_name == "XJTU-SY":
            return rul_datasets.XjtuSyReader(self.fd, window_size=self.window_size)
        else:
            raise ValueError(f"Unsupported RUL dataset: {self.dataset_name}")

    def prepare_data(self):
        """Downloads the data to disk if it doesn't already exist."""
        self.reader.prepare_data()

    def setup(self, stage=None):
        """Applies normalization (e.g., MinMax/Z-score) and splits the data."""
        self.rul_dm.setup(stage)
        
        if stage == 'fit' or stage is None:
            self.train_ds = self.rul_dm.to_dataset("dev")
            self.val_ds = self.rul_dm.to_dataset("val")
        if stage == 'test' or stage is None:
            self.test_ds = self.rul_dm.to_dataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def get_minority_dataset(self, rul_threshold_ratio: float = 0.2) -> Subset:
        """
        Extracts the minority class (heavily degraded states) for generative training.
        This is critical for Phase 2: Generative Training with SS-FlowMatch.
        
        Args:
            rul_threshold_ratio: Fraction of max life considered "degraded" (default: last 20%).
        """
        if self.train_ds is None:
            raise RuntimeError("You must call setup('fit') before extracting the minority dataset.")
            
        # Determine the maximum RUL in the training set to calculate the threshold
        max_rul = 125.0 if self.dataset_name == "CMAPSS" else max([y.item() for _, y in self.train_ds])
        threshold = rul_threshold_ratio * max_rul
        
        # Isolate indices where the machine is near failure
        degraded_indices = [i for i, (_, y) in enumerate(self.train_ds) if y.item() <= threshold]
        
        print(f"[{self.dataset_name}] Extracted {len(degraded_indices)} minority samples (RUL <= {threshold:.1f})")
        return Subset(self.train_ds, degraded_indices)