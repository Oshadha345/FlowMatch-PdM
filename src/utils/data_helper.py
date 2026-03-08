# src/utils/data_helper.py
import pytorch_lightning as pl
from datasets.rul_data_loader import FlowMatchRULDataModule
from datasets.cwru_data_loader import CWRUDataModule
from datasets.paderborn_data_loader import PaderbornDataModule


def get_data_module(track: str, dataset_name: str, **kwargs) -> pl.LightningDataModule:
    """
    Factory function to instantiate the correct LightningDataModule for a given track.
    """
    track = track.lower().strip()
    dataset_name = dataset_name.upper().strip()

    # Track 1 & 2: Remaining Useful Life (Regression)
    if track in ["engine_rul", "bearing_rul"]:
        supported_rul = ["CMAPSS", "N-CMAPSS", "FEMTO", "XJTU-SY"]
        if dataset_name not in supported_rul:
            raise ValueError(f"Dataset {dataset_name} is not supported for RUL tracks. Choose from {supported_rul}.")
        return FlowMatchRULDataModule(dataset_name=dataset_name, **kwargs)

    # Track 3: Bearing Fault (Classification)
    elif track == "bearing_fault":
        if dataset_name == "CWRU":
            return CWRUDataModule(**kwargs)
        elif dataset_name == "PADERBORN":
            return PaderbornDataModule(**kwargs)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported for classification. Choose CWRU or Paderborn.")

    else:
        raise ValueError(f"Unknown track: '{track}'. Use 'engine_rul', 'bearing_rul', or 'bearing_fault'.")