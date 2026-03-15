# src/utils/data_helper.py
import pytorch_lightning as pl

from datasets.cwru_data_loader import CWRUDataModule
from datasets.demadics_data_loader import DEMADICSDataModule
from datasets.paderborn_data_loader import PaderbornDataModule
from datasets.rul_data_loader import FlowMatchRULDataModule


_DATASET_NAME_ALIASES = {
    "CMAPSS": "CMAPSS",
    "C-MAPSS": "CMAPSS",
    "N-CMAPSS": "N-CMAPSS",
    "NCMAPSS": "N-CMAPSS",
    "FEMTO": "FEMTO",
    "XJTU-SY": "XJTU-SY",
    "XJTU_SY": "XJTU-SY",
    "XJTUSY": "XJTU-SY",
    "CWRU": "CWRU",
    "PADERBORN": "Paderborn",
    "DEMADICS": "DEMADICS",
}


def canonicalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().upper().replace("_", "-")
    if normalized not in _DATASET_NAME_ALIASES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported datasets: "
            f"{sorted(set(_DATASET_NAME_ALIASES.values()))}."
        )
    return _DATASET_NAME_ALIASES[normalized]


def get_dataset_config(config: dict, dataset_name: str) -> dict:
    canonical_name = canonicalize_dataset_name(dataset_name)
    datasets_cfg = config.get("datasets", {})
    if canonical_name not in datasets_cfg:
        raise KeyError(f"Dataset config for '{canonical_name}' not found in configs/default_config.yaml.")
    return datasets_cfg[canonical_name]


def get_data_module(track: str, dataset_name: str, **kwargs) -> pl.LightningDataModule:
    """
    Factory function to instantiate the correct LightningDataModule for a given track.
    """
    track = track.lower().strip()
    canonical_name = canonicalize_dataset_name(dataset_name)

    if track in ["engine_rul", "bearing_rul"]:
        supported_rul = ["CMAPSS", "N-CMAPSS", "FEMTO", "XJTU-SY"]
        if canonical_name not in supported_rul:
            raise ValueError(
                f"Dataset {canonical_name} is not supported for RUL tracks. "
                f"Choose from {supported_rul}."
            )
        return FlowMatchRULDataModule(dataset_name=canonical_name, **kwargs)

    if track == "bearing_fault":
        if canonical_name == "CWRU":
            return CWRUDataModule(**kwargs)
        if canonical_name == "Paderborn":
            return PaderbornDataModule(**kwargs)
        if canonical_name == "DEMADICS":
            return DEMADICSDataModule(**kwargs)
        raise ValueError(
            f"Dataset {canonical_name} is not supported for classification. "
            "Choose CWRU, Paderborn, or DEMADICS."
        )

    raise ValueError(
        f"Unknown track: '{track}'. Use 'engine_rul', 'bearing_rul', or 'bearing_fault'."
    )
