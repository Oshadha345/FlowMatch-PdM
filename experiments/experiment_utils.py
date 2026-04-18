"""
Shared utilities for reviewer experiments.
Provides logging, metrics collection, and common data helpers.
"""
import csv
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

EXPERIMENT_ROOT = Path(__file__).resolve().parent / "irai_2026"
FIGURE_DIR = EXPERIMENT_ROOT / "figures"
TABLE_DIR = EXPERIMENT_ROOT / "tables"
LOG_DIR = EXPERIMENT_ROOT / "logs"

for _d in (EXPERIMENT_ROOT, FIGURE_DIR, TABLE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_csv(rows: List[Dict], path: Path):
    """Save a list of dicts as CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Experiments] Saved CSV: {path}")


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[Experiments] Saved JSON: {path}")


def log_experiment(name: str, config: Dict, metrics: Dict, runtime_s: float):
    """Append a single experiment result to the master log."""
    entry = {
        "timestamp": timestamp(),
        "experiment": name,
        "runtime_seconds": round(runtime_s, 2),
        **{f"cfg_{k}": v for k, v in config.items()},
        **{f"metric_{k}": v for k, v in metrics.items()},
    }
    log_path = LOG_DIR / "experiment_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def compute_generation_metrics(
    real: np.ndarray, synthetic: np.ndarray
) -> Dict[str, float]:
    """Compute basic generation quality metrics between real and synthetic data.
    
    Both arrays should be [N, seq_len, features].
    """
    metrics = {}
    
    # Flatten to [N, seq_len * features] for MMD
    real_flat = real.reshape(real.shape[0], -1)
    synth_flat = synthetic.reshape(synthetic.shape[0], -1)
    
    # Mean/std statistics
    metrics["real_mean"] = float(np.mean(real))
    metrics["synth_mean"] = float(np.mean(synthetic))
    metrics["real_std"] = float(np.std(real))
    metrics["synth_std"] = float(np.std(synthetic))
    metrics["mean_abs_diff"] = float(np.abs(np.mean(real) - np.mean(synthetic)))
    metrics["std_abs_diff"] = float(np.abs(np.std(real) - np.std(synthetic)))
    
    # Power spectrum similarity
    try:
        real_fft = np.abs(np.fft.rfft(real, axis=1))
        synth_fft = np.abs(np.fft.rfft(synthetic, axis=1))
        ps_real = np.mean(real_fft ** 2, axis=(0, 2))
        ps_synth = np.mean(synth_fft ** 2, axis=(0, 2))
        metrics["power_spectrum_l2"] = float(np.sqrt(np.mean((ps_real - ps_synth) ** 2)))
        metrics["power_spectrum_corr"] = float(np.corrcoef(ps_real, ps_synth)[0, 1]) if len(ps_real) > 1 else 0.0
    except Exception:
        metrics["power_spectrum_l2"] = float("nan")
        metrics["power_spectrum_corr"] = float("nan")
    
    # MMD with RBF kernel (subsample for speed)
    try:
        max_n = min(500, len(real_flat), len(synth_flat))
        r = real_flat[np.random.choice(len(real_flat), max_n, replace=False)]
        s = synth_flat[np.random.choice(len(synth_flat), max_n, replace=False)]
        metrics["mmd"] = float(_mmd_rbf(r, s))
    except Exception:
        metrics["mmd"] = float("nan")
    
    return metrics


def _mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    """MMD with median heuristic bandwidth."""
    from scipy.spatial.distance import cdist
    
    xx = cdist(x, x, "sqeuclidean")
    yy = cdist(y, y, "sqeuclidean")
    xy = cdist(x, y, "sqeuclidean")
    
    all_dists = np.concatenate([xx.ravel(), yy.ravel(), xy.ravel()])
    median_dist = np.median(all_dists[all_dists > 0])
    if median_dist <= 0:
        median_dist = 1.0
    gamma = 1.0 / (2 * median_dist)
    
    kxx = np.exp(-gamma * xx).mean()
    kyy = np.exp(-gamma * yy).mean()
    kxy = np.exp(-gamma * xy).mean()
    return float(kxx + kyy - 2 * kxy)


def load_config(config_path: str = "configs/default_config.yaml") -> Dict:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_minority_data(
    dataset_name: str,
    window_size: int,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load minority (degraded) data for a dataset."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils.data_helper import get_data_module, get_dataset_config
    
    dataset_cfg = get_dataset_config(config, dataset_name)
    dm = get_data_module(
        track="bearing_rul",
        dataset_name=dataset_name,
        conditions=dataset_cfg.get("conditions", dataset_cfg.get("fd_list", 1)),
        window_size=window_size,
        batch_size=64,
        append_condition_features=dataset_cfg.get("append_condition_features", False),
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    
    rul_ratio = config["datasets"]["minority_rul_ratio"]
    try:
        minority_ds = dm.get_minority_dataset(rul_threshold_ratio=rul_ratio)
    except TypeError:
        minority_ds = dm.get_minority_dataset()
    
    xs, ys = [], []
    for i in range(len(minority_ds)):
        x, y = minority_ds[i]
        xs.append(x.numpy())
        ys.append(float(y))
    
    return np.array(xs), np.array(ys)
