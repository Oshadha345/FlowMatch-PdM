# src/utils/data_helper.py
import pytorch_lightning as pl

_DATASET_NAME_ALIASES = {
    "FEMTO": "FEMTO",
    "XJTU-SY": "XJTU-SY",
    "XJTU_SY": "XJTU-SY",
    "XJTUSY": "XJTU-SY",
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

    if track != "bearing_rul":
        raise ValueError(
            f"Unsupported track '{track}' for the current pivot. "
            "Only 'bearing_rul' is active."
        )

    supported_rul = ["FEMTO", "XJTU-SY"]
    if canonical_name not in supported_rul:
        raise ValueError(
            f"Dataset {canonical_name} is not supported for the current pivot. "
            f"Choose from {supported_rul}."
        )

    from datasets.rul_data_loader import FlowMatchRULDataModule

    return FlowMatchRULDataModule(dataset_name=canonical_name, **kwargs)
