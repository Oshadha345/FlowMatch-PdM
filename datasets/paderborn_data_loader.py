# datasets/paderborn_data_loader.py
import os
from pathlib import Path
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset


class _MemmapNpyDataset(Dataset):
    """Lazy dataset over .npy memmaps to avoid loading full arrays into RAM."""

    def __init__(self, x_path: str, y_path: str):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if len(self.x) != len(self.y):
            raise ValueError(f"Feature/label size mismatch: {x_path} vs {y_path}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.x[idx], dtype=np.float32, copy=True))
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


class PaderbornDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Paderborn Bearing Fault Classification.
    Loads individual .npy files created by notebooks/01_dataset_analysis.ipynb.
    Window=4096 | Z-score | 31 classes.
    Uses np.load(mmap_mode='r') for near-instant loading.
    """
    def __init__(self, data_dir: str = "datasets/processed/paderborn",
                 window_size: int = 4096, batch_size: int = 512, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_classes = 31
        self._resolved_data_dir = None

    def _resolve_data_dir(self) -> str:
        if self._resolved_data_dir is not None:
            return self._resolved_data_dir

        repo_root = Path(__file__).resolve().parents[1]
        configured = Path(self.data_dir)
        candidates = []

        # 1) explicit absolute/relative as provided by user/caller
        candidates.append(configured)

        # 2) repo-relative (FlowMatch-PdM/<data_dir>)
        if not configured.is_absolute():
            candidates.append(repo_root / configured)

        # 3) workspace-root relative (<workspace>/<data_dir>)
        if not configured.is_absolute():
            candidates.append(repo_root.parent / configured)

        seen = set()
        for cand in candidates:
            cand = cand.resolve()
            if str(cand) in seen:
                continue
            seen.add(str(cand))
            if (cand / "X_train.npy").exists():
                self._resolved_data_dir = str(cand)
                return self._resolved_data_dir

        checked = "\n  - " + "\n  - ".join(sorted(seen))
        raise FileNotFoundError(
            "Paderborn processed .npy files not found. Checked paths:"
            f"{checked}\nRun notebooks/01_dataset_analysis.ipynb preprocessing cell first."
        )

    def prepare_data(self):
        self._resolve_data_dir()

    def setup(self, stage=None):
        data_dir = self._resolve_data_dir()

        self.train_ds = _MemmapNpyDataset(
            os.path.join(data_dir, "X_train.npy"),
            os.path.join(data_dir, "y_train.npy"),
        )
        self.val_ds = _MemmapNpyDataset(
            os.path.join(data_dir, "X_val.npy"),
            os.path.join(data_dir, "y_val.npy"),
        )
        self.test_ds = _MemmapNpyDataset(
            os.path.join(data_dir, "X_test.npy"),
            os.path.join(data_dir, "y_test.npy"),
        )
        print(f"[Paderborn] Loaded from {data_dir}/  "
              f"Train: {len(self.train_ds)} | Val: {len(self.val_ds)} | Test: {len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def get_minority_dataset(self) -> Subset:
        """Extract minority class samples based on actual train split counts."""
        if not hasattr(self, "train_ds"):
            self.setup(stage="fit")

        y_train = np.asarray(self.train_ds.y)
        classes, counts = np.unique(y_train, return_counts=True)
        min_count = counts.min()
        minority_labels = classes[counts == min_count]
        minority_indices = np.where(np.isin(y_train, minority_labels))[0].tolist()

        print(
            f"[Paderborn] Minority label(s): {minority_labels.tolist()} "
            f"with {int(min_count)} samples each. Extracted {len(minority_indices)} samples."
        )
        return Subset(self.train_ds, minority_indices)