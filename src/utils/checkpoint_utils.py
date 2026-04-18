from pathlib import Path
from typing import Dict, List, Tuple, Type, Union

import pytorch_lightning as pl
import torch


_PROFILE_ARTIFACT_NAMES = {"total_ops", "total_params"}


def _is_profile_artifact_key(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in _PROFILE_ARTIFACT_NAMES


def strip_profile_artifact_keys(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    cleaned_state_dict: Dict[str, torch.Tensor] = {}
    removed_keys: List[str] = []

    for key, value in state_dict.items():
        if _is_profile_artifact_key(key):
            removed_keys.append(key)
            continue
        cleaned_state_dict[key] = value

    return cleaned_state_dict, removed_keys


def load_lightning_module_checkpoint(
    module_cls: Type[pl.LightningModule],
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> pl.LightningModule:
    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint.get("state_dict")

    if not isinstance(hyper_parameters, dict):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not contain a usable 'hyper_parameters' payload for {module_cls.__name__}."
        )
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path} does not contain a valid 'state_dict'.")

    cleaned_state_dict, removed_keys = strip_profile_artifact_keys(state_dict)
    model = module_cls(**hyper_parameters)
    incompatible = model.load_state_dict(cleaned_state_dict, strict=False)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {module_cls.__name__}:\n\t"
            f"Missing key(s): {sorted(incompatible.missing_keys)}\n\t"
            f"Unexpected key(s): {sorted(incompatible.unexpected_keys)}"
        )

    if removed_keys:
        print(
            f"[CheckpointLoader] Ignored {len(removed_keys)} profiling artifact key(s) "
            f"from {Path(checkpoint_path).name}."
        )

    return model
