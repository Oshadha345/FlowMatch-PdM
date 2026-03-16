# datasets/rul_data_loader.py
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import rul_datasets
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


_RUL_DATASET_DEFAULTS: Dict[str, Dict[str, object]] = {
    "CMAPSS": {
        "window_size": 30,
        "max_rul": 125,
        "conditions": [1, 2, 3, 4],
    },
    "N-CMAPSS": {
        "window_size": 50,
        "max_rul": None,
        "conditions": [1, 2, 3, 4, 5, 6, 7, 8],
    },
    "FEMTO": {
        "window_size": 2560,
        "max_rul": None,
        "conditions": [1, 2, 3],
    },
    "XJTU-SY": {
        "window_size": 2048,
        "max_rul": None,
        "conditions": [1, 2, 3],
    },
}

_AVAILABLE_FDS: Dict[str, List[int]] = {
    "CMAPSS": [1, 2, 3, 4],
    "N-CMAPSS": [1, 2, 3, 4, 5, 6, 7],
    "FEMTO": [1, 2, 3],
    "XJTU-SY": [1, 2, 3],
}

_FEMTO_DEFAULT_SPLITS: Dict[int, Dict[str, List[int]]] = {
    1: {"dev": [1, 2], "val": [3], "test": [4, 5, 6, 7]},
    2: {"dev": [1, 2], "val": [3], "test": [4, 5, 6, 7]},
    3: {"dev": [1], "val": [2], "test": [3]},
}


@dataclass(frozen=True)
class _ReaderSpec:
    fd: int
    run_split_dist: Optional[Dict[str, List[int]]] = None
    label: Optional[str] = None


class _WindowFirstRulDataset(Dataset):
    """Wrap `rul_datasets.RulDataset` and expose windows as [window, features]."""

    def __init__(
        self,
        base_dataset: Dataset,
        window_size: int,
        num_features: int,
        target_scale: float = 1.0,
        static_context: Optional[Sequence[float]] = None,
    ):
        if target_scale <= 0:
            raise ValueError(f"target_scale must be positive, received {target_scale}.")

        self.base_dataset = base_dataset
        self.window_size = window_size
        self.num_features = num_features
        self.target_scale = float(target_scale)
        self.static_context = None if static_context is None else torch.tensor(static_context, dtype=torch.float32)

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

        if self.static_context is not None:
            context = self.static_context.unsqueeze(0).expand(self.window_size, -1)
            features = torch.cat([features, context], dim=-1)

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
        conditions: Optional[Sequence[Union[int, str]]] = None,
        fd: Optional[Union[int, Sequence[Union[int, str]]]] = None,
        window_size: Optional[int] = None,
        batch_size: int = 128,
        append_condition_features: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name.upper()

        defaults = self._get_dataset_defaults()
        resolved_conditions = conditions if conditions is not None else fd
        self.conditions = self._normalize_conditions(resolved_conditions or defaults["conditions"])
        self.fd = self.conditions
        self.batch_size = int(batch_size)
        self.window_size = int(window_size or defaults["window_size"])
        self.append_condition_features = bool(append_condition_features)
        self.max_rul = defaults["max_rul"]
        self.target_scale = 1.0
        self.max_rul_val = self.target_scale

        self.save_hyperparameters()

        self.reader_specs = self._build_reader_specs()
        self.condition_feature_map = self._build_condition_feature_map()
        self.readers = self._build_readers()
        self.rul_dms = [rul_datasets.RulDataModule(reader, batch_size=self.batch_size) for reader in self.readers]

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def _get_dataset_defaults(self) -> Dict[str, object]:
        if self.dataset_name not in _RUL_DATASET_DEFAULTS:
            raise ValueError(f"Unsupported RUL dataset: {self.dataset_name}")
        return _RUL_DATASET_DEFAULTS[self.dataset_name]

    def _normalize_conditions(self, conditions: Union[int, str, Sequence[Union[int, str]]]) -> List[Union[int, str]]:
        if isinstance(conditions, Sequence) and not isinstance(conditions, (str, bytes)):
            raw_conditions = list(conditions)
        else:
            raw_conditions = [conditions]

        normalized: List[Union[int, str]] = []
        for condition in raw_conditions:
            if isinstance(condition, str):
                stripped = condition.strip()
                if stripped.isdigit():
                    normalized.append(int(stripped))
                else:
                    normalized.append(stripped)
            else:
                normalized.append(int(condition))

        normalized = list(dict.fromkeys(normalized))
        if not normalized:
            raise ValueError("At least one RUL operating condition must be provided.")
        return normalized

    def _validate_fd_conditions(self, fds: Sequence[int]) -> List[int]:
        available = _AVAILABLE_FDS[self.dataset_name]
        valid = [fd for fd in fds if fd in available]
        invalid = [fd for fd in fds if fd not in available]

        if invalid:
            print(
                f"[{self.dataset_name}] Skipping unsupported conditions {invalid}. "
                f"Available conditions: {available}"
            )
        if not valid:
            raise ValueError(
                f"No valid conditions supplied for {self.dataset_name}. "
                f"Requested {list(fds)}, available {available}."
            )
        return valid

    def _build_reader_specs(self) -> List[_ReaderSpec]:
        if self.dataset_name != "FEMTO":
            numeric_conditions = [int(condition) for condition in self.conditions]
            valid_fds = self._validate_fd_conditions(numeric_conditions)
            return [_ReaderSpec(fd=fd, label=f"FD{fd}") for fd in valid_fds]

        if all(isinstance(condition, int) for condition in self.conditions):
            valid_fds = self._validate_fd_conditions([int(condition) for condition in self.conditions])
            return [_ReaderSpec(fd=fd, label=f"FD{fd}") for fd in valid_fds]

        bearing_pattern = re.compile(r"^Bearing(?P<fd>\d+)_(?P<run>\d+)$", re.IGNORECASE)
        grouped_runs: Dict[int, List[int]] = {}
        labels_by_fd: Dict[int, List[str]] = {}

        for condition in self.conditions:
            if not isinstance(condition, str):
                raise ValueError(f"Unsupported FEMTO condition type: {type(condition)!r}")

            match = bearing_pattern.match(condition)
            if match is None:
                raise ValueError(
                    "FEMTO conditions must be integer FD identifiers or labels like 'Bearing1_1'. "
                    f"Received '{condition}'."
                )

            fd = int(match.group("fd"))
            run_idx = int(match.group("run"))
            if fd not in _FEMTO_DEFAULT_SPLITS:
                raise ValueError(f"Unsupported FEMTO condition '{condition}'.")

            grouped_runs.setdefault(fd, []).append(run_idx)
            labels_by_fd.setdefault(fd, []).append(condition)

        specs: List[_ReaderSpec] = []
        for fd in sorted(grouped_runs):
            selected_runs = sorted(set(grouped_runs[fd]))
            default_splits = _FEMTO_DEFAULT_SPLITS[fd]

            if any(run_idx not in sum(default_splits.values(), []) for run_idx in selected_runs):
                raise ValueError(f"Unsupported FEMTO run selection for FD{fd}: {selected_runs}")

            run_split_dist = {
                "dev": selected_runs,
                "val": [run_idx for run_idx in default_splits["val"] if run_idx not in selected_runs],
                "test": list(default_splits["test"]),
            }
            if not run_split_dist["val"]:
                warnings.warn(
                    "Custom FEMTO run selection consumed the canonical validation run for "
                    f"FD{fd}. Validation for this condition will be empty. "
                    f"Requested runs: {selected_runs}; default split: {default_splits}.",
                    stacklevel=2,
                )
            specs.append(
                _ReaderSpec(
                    fd=fd,
                    run_split_dist=run_split_dist,
                    label=",".join(labels_by_fd[fd]),
                )
            )

        return specs

    def _build_readers(self):
        readers = []
        for spec in self.reader_specs:
            if self.dataset_name == "CMAPSS":
                readers.append(
                    rul_datasets.CmapssReader(
                        spec.fd,
                        window_size=self.window_size,
                        max_rul=int(self.max_rul) if self.max_rul is not None else None,
                    )
                )
            elif self.dataset_name == "N-CMAPSS":
                readers.append(
                    rul_datasets.NCmapssReader(
                        spec.fd,
                        window_size=self.window_size,
                    )
                )
            elif self.dataset_name == "FEMTO":
                readers.append(
                    rul_datasets.FemtoReader(
                        spec.fd,
                        window_size=self.window_size,
                        run_split_dist=spec.run_split_dist,
                    )
                )
            elif self.dataset_name == "XJTU-SY":
                readers.append(
                    rul_datasets.XjtuSyReader(
                        spec.fd,
                        window_size=self.window_size,
                    )
                )
            else:
                raise ValueError(f"Unsupported RUL dataset: {self.dataset_name}")
        return readers

    def _build_condition_feature_map(self) -> Dict[str, Optional[np.ndarray]]:
        if not self.append_condition_features:
            return {str(index): None for index in range(len(self.reader_specs))}

        one_hot = np.eye(len(self.reader_specs), dtype=np.float32)
        return {str(index): one_hot[index] for index in range(len(self.reader_specs))}

    def prepare_data(self):
        for reader in self.readers:
            reader.prepare_data()

    def setup(self, stage=None):
        for rul_dm in self.rul_dms:
            rul_dm.setup(stage)

        if stage == "fit" or stage is None:
            self.target_scale = self._compute_target_scale()
            self.max_rul_val = self.target_scale
            self.train_ds = self._wrap_split("dev")
            self.val_ds = self._wrap_split("val")

        if stage == "test" or stage is None:
            if self.target_scale <= 0:
                self.target_scale = self._compute_target_scale()
                self.max_rul_val = self.target_scale
            self.test_ds = self._wrap_split("test")

    def _split_is_empty(self, rul_dm, split: str) -> bool:
        features_per_run, _ = rul_dm.data[split]
        return not any(len(run_features) > 0 for run_features in features_per_run)

    def _compute_target_scale(self) -> float:
        raw_train_targets = self._flatten_split_targets("dev")
        target_scale = float(np.max(np.abs(raw_train_targets)))
        if target_scale <= 0:
            raise RuntimeError("Encountered non-positive target_scale while preparing scaled RUL targets.")
        return target_scale

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

        for index, (spec, rul_dm) in enumerate(zip(self.reader_specs, self.rul_dms)):
            if self._split_is_empty(rul_dm, split):
                continue

            window_size, num_features = self._infer_split_shape(rul_dm, split)
            if window_size != self.window_size:
                raise RuntimeError(
                    f"{self.dataset_name} condition '{spec.label or spec.fd}' {split} split window size mismatch: "
                    f"expected {self.window_size}, got {window_size}."
                )

            split_shape = (window_size, num_features)
            if expected_shape is None:
                expected_shape = split_shape
            elif expected_shape != split_shape:
                raise RuntimeError(
                    f"Incompatible feature shape across conditions for {self.dataset_name} {split}: "
                    f"expected {expected_shape}, got {split_shape} for condition '{spec.label or spec.fd}'."
                )

            wrapped_datasets.append(
                _WindowFirstRulDataset(
                    rul_dm.to_dataset(split),
                    window_size=window_size,
                    num_features=num_features,
                    target_scale=self.target_scale,
                    static_context=self.condition_feature_map[str(index)],
                )
            )

        if not wrapped_datasets:
            raise RuntimeError(f"No non-empty datasets available for {self.dataset_name} split '{split}'.")
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
            for run_targets in targets_per_run:
                run_targets = np.asarray(run_targets, dtype=np.float32).reshape(-1)
                if run_targets.size > 0:
                    flattened.append(run_targets)

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
        threshold = float(rul_threshold_ratio) * float(self.target_scale)
        degraded_indices = np.flatnonzero(train_targets <= threshold).tolist()

        print(
            f"[{self.dataset_name}] Extracted {len(degraded_indices)} minority samples "
            f"(RUL <= {threshold:.4f}) across conditions {self.conditions}"
        )
        return Subset(self.train_ds, degraded_indices)
