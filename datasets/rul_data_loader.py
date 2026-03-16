# datasets/rul_data_loader.py
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import rul_datasets
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


_RUL_DATASET_DEFAULTS: Dict[str, Dict[str, Optional[int]]] = {
    "CMAPSS": {"window_size": 30, "max_rul": 125},
    "N-CMAPSS": {"window_size": 50, "max_rul": None},
    "FEMTO": {"window_size": 2560, "max_rul": None},
    "XJTU-SY": {"window_size": 2048, "max_rul": None},
}


class _WindowFirstRulDataset(Dataset):
    """Wrap `rul_datasets.RulDataset` and expose windows as [window, features]."""

    def __init__(self, base_dataset: Dataset, window_size: int, num_features: int, target_scale: float = 1.0):
        if target_scale <= 0:
            raise ValueError(f"target_scale must be positive, received {target_scale}.")

        self.base_dataset = base_dataset
        self.window_size = window_size
        self.num_features = num_features
        self.target_scale = float(target_scale)

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

        scaled_target = torch.as_tensor(target, dtype=torch.float32) / self.target_scale
        return features.contiguous(), scaled_target


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
        fd: Union[int, Sequence[int]] = 1,
        window_size: Optional[int] = None,
        batch_size: int = 512,
    ):
        super().__init__()
        self.dataset_name = dataset_name.upper()
        self.fd = self._normalize_fd_list(fd)
        self.batch_size = batch_size

        defaults = self._get_dataset_defaults()
        self.window_size = int(window_size or defaults["window_size"])
        self.max_rul = defaults["max_rul"]
        self.max_rul_val = 1.0

        self.save_hyperparameters()

        self.readers = self._get_reader()
        self.rul_dms = [rul_datasets.RulDataModule(reader, batch_size=self.batch_size) for reader in self.readers]

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def _normalize_fd_list(self, fd: Union[int, Sequence[int]]) -> List[int]:
        if isinstance(fd, Sequence) and not isinstance(fd, (str, bytes)):
            fd_list = [int(item) for item in fd]
        else:
            fd_list = [int(fd)]

        fd_list = list(dict.fromkeys(fd_list))
        if not fd_list:
            raise ValueError("At least one RUL operating condition must be provided.")
        return fd_list

    def _get_dataset_defaults(self) -> Dict[str, Optional[int]]:
        if self.dataset_name not in _RUL_DATASET_DEFAULTS:
            raise ValueError(f"Unsupported RUL dataset: {self.dataset_name}")
        return _RUL_DATASET_DEFAULTS[self.dataset_name]

    def _get_reader(self):
        readers = []
        for fd_idx in self.fd:
            if self.dataset_name == "CMAPSS":
                readers.append(
                    rul_datasets.CmapssReader(
                        fd_idx,
                        window_size=self.window_size,
                        max_rul=int(self.max_rul) if self.max_rul is not None else None,
                    )
                )
                continue
            if self.dataset_name == "N-CMAPSS":
                readers.append(rul_datasets.NCmapssReader(fd_idx, window_size=self.window_size))
                continue
            if self.dataset_name == "FEMTO":
                readers.append(rul_datasets.FemtoReader(fd_idx, window_size=self.window_size))
                continue
            if self.dataset_name == "XJTU-SY":
                readers.append(rul_datasets.XjtuSyReader(fd_idx, window_size=self.window_size))
                continue
            raise ValueError(f"Unsupported RUL dataset: {self.dataset_name}")
        return readers

    def prepare_data(self):
        for reader in self.readers:
            reader.prepare_data()

    def setup(self, stage=None):
        for rul_dm in self.rul_dms:
            rul_dm.setup(stage)

        if stage == "fit" or stage is None:
            self.max_rul_val = self._compute_target_scale()
            self.train_ds = self._wrap_split("dev")
            self.val_ds = self._wrap_split("val")
        if stage == "test" or stage is None:
            if self.max_rul_val <= 0:
                self.max_rul_val = self._compute_target_scale()
            self.test_ds = self._wrap_split("test")

    def _compute_target_scale(self) -> float:
        for rul_dm in self.rul_dms:
            rul_dm.setup(stage="fit")

        train_targets = self._flatten_split_targets("dev")
        max_rul_val = float(train_targets.max())
        if max_rul_val <= 0:
            raise RuntimeError("Encountered non-positive max_rul_val while preparing scaled RUL targets.")
        return max_rul_val

    def _infer_split_shape(self, rul_dm, split: str) -> Tuple[int, int]:
        features_per_run, _ = rul_dm.data[split]
        for run_features in features_per_run:
            if len(run_features) == 0:
                continue
            window_size = int(run_features.shape[1])
            num_features = int(run_features.shape[2])
            return window_size, num_features

        raise RuntimeError(f"Split '{split}' is empty; unable to infer window/feature dimensions.")

    def _wrap_split(self, split: str) -> Dataset:
        wrapped_datasets: List[Dataset] = []
        expected_shape: Optional[Tuple[int, int]] = None

        for rul_dm, fd_idx in zip(self.rul_dms, self.fd):
            window_size, num_features = self._infer_split_shape(rul_dm, split)
            if window_size != self.window_size:
                raise RuntimeError(
                    f"{self.dataset_name} FD{fd_idx} {split} split window size mismatch: "
                    f"expected {self.window_size}, got {window_size}."
                )

            split_shape = (window_size, num_features)
            if expected_shape is None:
                expected_shape = split_shape
            elif expected_shape != split_shape:
                raise RuntimeError(
                    f"Incompatible feature shape across operating conditions for {self.dataset_name} {split}: "
                    f"expected {expected_shape}, got {split_shape} for FD{fd_idx}."
                )

            wrapped_datasets.append(
                _WindowFirstRulDataset(
                    rul_dm.to_dataset(split),
                    window_size=window_size,
                    num_features=num_features,
                    target_scale=self.max_rul_val,
                )
            )

        if len(wrapped_datasets) == 1:
            return wrapped_datasets[0]
        return ConcatDataset(wrapped_datasets)

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
        flattened = []
        for rul_dm in self.rul_dms:
            _, targets_per_run = rul_dm.data[split]
            flattened.extend(
                np.asarray(run_targets, dtype=np.float32).reshape(-1)
                for run_targets in targets_per_run
                if np.asarray(run_targets).size > 0
            )

        if not flattened:
            raise RuntimeError(f"Split '{split}' does not contain any targets.")
        return np.concatenate(flattened, axis=0)

    def get_minority_dataset(self, rul_threshold_ratio: float = 0.2) -> Subset:
        """
        Extract the last `rul_threshold_ratio` fraction of life as the minority subset.

        The threshold is derived from the maximum clipped RUL value across the
        concatenated development split.
        """
        if self.train_ds is None:
            raise RuntimeError("You must call setup('fit') before extracting the minority dataset.")
        if not 0.0 < rul_threshold_ratio <= 1.0:
            raise ValueError("rul_threshold_ratio must be in the interval (0, 1].")

        train_targets = self._flatten_split_targets("dev")
        threshold = float(rul_threshold_ratio) * float(self.max_rul_val)
        degraded_indices = np.flatnonzero(train_targets <= threshold).tolist()

        print(
            f"[{self.dataset_name}] Extracted {len(degraded_indices)} minority samples "
            f"(RUL <= {threshold:.4f}) across FD(s) {self.fd}"
        )
        return Subset(self.train_ds, degraded_indices)
