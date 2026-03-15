# datasets/rul_data_loader.py
from typing import Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import rul_datasets
import torch
from torch.utils.data import DataLoader, Dataset, Subset


_RUL_DATASET_DEFAULTS: Dict[str, Dict[str, Optional[int]]] = {
    "CMAPSS": {"window_size": 30, "max_rul": 125},
    "N-CMAPSS": {"window_size": 50, "max_rul": None},
    "FEMTO": {"window_size": 2560, "max_rul": None},
    "XJTU-SY": {"window_size": 2048, "max_rul": None},
}


class _WindowFirstRulDataset(Dataset):
    """Wrap `rul_datasets.RulDataset` and expose windows as [window, features]."""

    def __init__(self, base_dataset: Dataset, window_size: int, num_features: int):
        self.base_dataset = base_dataset
        self.window_size = window_size
        self.num_features = num_features

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, target = self.base_dataset[index]

        if features.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D feature tensor, received shape {tuple(features.shape)}."
            )

        if tuple(features.shape) == (self.num_features, self.window_size):
            features = features.transpose(0, 1)
        elif tuple(features.shape) != (self.window_size, self.num_features):
            if features.shape[-1] == self.window_size:
                features = features.transpose(-1, -2)
            elif features.shape[-2] != self.window_size:
                raise RuntimeError(
                    "Unable to normalize RUL tensor orientation. "
                    f"Expected {(self.num_features, self.window_size)} or "
                    f"{(self.window_size, self.num_features)}, got {tuple(features.shape)}."
                )

        if tuple(features.shape) != (self.window_size, self.num_features):
            raise RuntimeError(
                "RUL feature normalization failed. "
                f"Expected {(self.window_size, self.num_features)}, got {tuple(features.shape)}."
            )

        return features.contiguous(), torch.as_tensor(target, dtype=torch.float32)


class FlowMatchRULDataModule(pl.LightningDataModule):
    """
    Unified Lightning DataModule for CMAPSS, N-CMAPSS, FEMTO, and XJTU-SY.

    The wrapped `rul_datasets` package stores features as [features, window].
    This module converts them back to [window, features] so all downstream
    sequence models receive batch-first tensors: [batch, window, features].
    """

    def __init__(
        self,
        dataset_name: str,
        fd: int = 1,
        window_size: Optional[int] = None,
        batch_size: int = 512,
    ):
        super().__init__()
        self.dataset_name = dataset_name.upper()
        self.fd = fd
        self.batch_size = batch_size

        defaults = self._get_dataset_defaults()
        self.window_size = int(window_size or defaults["window_size"])
        self.max_rul = defaults["max_rul"]

        self.save_hyperparameters()

        self.reader = self._get_reader()
        self.rul_dm = rul_datasets.RulDataModule(self.reader, batch_size=self.batch_size)

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self._split_shapes: Dict[str, Tuple[int, int]] = {}

    def _get_dataset_defaults(self) -> Dict[str, Optional[int]]:
        if self.dataset_name not in _RUL_DATASET_DEFAULTS:
            raise ValueError(f"Unsupported RUL dataset: {self.dataset_name}")
        return _RUL_DATASET_DEFAULTS[self.dataset_name]

    def _get_reader(self):
        if self.dataset_name == "CMAPSS":
            return rul_datasets.CmapssReader(
                self.fd,
                window_size=self.window_size,
                max_rul=int(self.max_rul),
            )
        if self.dataset_name == "N-CMAPSS":
            return rul_datasets.NCmapssReader(self.fd, window_size=self.window_size)
        if self.dataset_name == "FEMTO":
            return rul_datasets.FemtoReader(self.fd, window_size=self.window_size)
        if self.dataset_name == "XJTU-SY":
            return rul_datasets.XjtuSyReader(self.fd, window_size=self.window_size)
        raise ValueError(f"Unsupported RUL dataset: {self.dataset_name}")

    def prepare_data(self):
        self.reader.prepare_data()

    def setup(self, stage=None):
        self.rul_dm.setup(stage)

        if stage == "fit" or stage is None:
            self.train_ds = self._wrap_dataset("dev")
            self.val_ds = self._wrap_dataset("val")
        if stage == "test" or stage is None:
            self.test_ds = self._wrap_dataset("test")

    def _infer_split_shape(self, split: str) -> Tuple[int, int]:
        if split in self._split_shapes:
            return self._split_shapes[split]

        features_per_run, _ = self.rul_dm.data[split]
        for run_features in features_per_run:
            if len(run_features) == 0:
                continue
            window_size = int(run_features.shape[1])
            num_features = int(run_features.shape[2])
            self._split_shapes[split] = (window_size, num_features)
            return self._split_shapes[split]

        raise RuntimeError(f"Split '{split}' is empty; unable to infer window/feature dimensions.")

    def _wrap_dataset(self, split: str) -> Dataset:
        window_size, num_features = self._infer_split_shape(split)
        if window_size != self.window_size:
            raise RuntimeError(
                f"{self.dataset_name} {split} split window size mismatch: "
                f"expected {self.window_size}, got {window_size}."
            )
        return _WindowFirstRulDataset(
            self.rul_dm.to_dataset(split),
            window_size=window_size,
            num_features=num_features,
        )

    def train_dataloader(self):
        if self.train_ds is None:
            raise RuntimeError("Call setup('fit') before requesting the training dataloader.")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            raise RuntimeError("Call setup('fit') before requesting the validation dataloader.")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            raise RuntimeError("Call setup('test') before requesting the test dataloader.")
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def _flatten_split_targets(self, split: str) -> np.ndarray:
        _, targets_per_run = self.rul_dm.data[split]
        flattened = [np.asarray(run_targets, dtype=np.float32).reshape(-1) for run_targets in targets_per_run]
        flattened = [run_targets for run_targets in flattened if run_targets.size > 0]
        if not flattened:
            raise RuntimeError(f"Split '{split}' does not contain any targets.")
        return np.concatenate(flattened, axis=0)

    def get_minority_dataset(self, rul_threshold_ratio: float = 0.2) -> Subset:
        """
        Extract the last `rul_threshold_ratio` fraction of life as the minority subset.

        For CMAPSS this uses the clipped SOTA max RUL of 125. For FEMTO and the
        remaining RUL datasets, the threshold is derived from the maximum target in
        the development split after reader preprocessing.
        """
        if self.train_ds is None:
            raise RuntimeError("You must call setup('fit') before extracting the minority dataset.")
        if not 0.0 < rul_threshold_ratio <= 1.0:
            raise ValueError("rul_threshold_ratio must be in the interval (0, 1].")

        train_targets = self._flatten_split_targets("dev")
        if self.dataset_name == "CMAPSS":
            max_rul = float(self.max_rul)
        else:
            max_rul = float(train_targets.max())

        threshold = float(rul_threshold_ratio) * max_rul
        degraded_indices = np.flatnonzero(train_targets <= threshold).tolist()

        print(
            f"[{self.dataset_name}] Extracted {len(degraded_indices)} minority samples "
            f"(RUL <= {threshold:.4f})"
        )
        return Subset(self.train_ds, degraded_indices)
