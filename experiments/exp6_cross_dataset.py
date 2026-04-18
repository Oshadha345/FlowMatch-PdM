#!/usr/bin/env python3
"""
Experiment 6: Cross-Dataset Validation.

Runs the FlowMatch-PdM pipeline on both FEMTO and XJTU-SY datasets
with a unified interface. Validates that the pipeline generalizes
across datasets with different characteristics.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_utils import (
    FIGURE_DIR, TABLE_DIR,
    compute_generation_metrics, get_device, get_minority_data,
    load_config, log_experiment, save_csv, save_json,
)
from flowmatchPdM.flowmatch_pdm import FlowMatchPdM


DATASETS = ["FEMTO", "XJTU-SY"]


def run_cross_dataset(
    datasets: list = None,
    epochs: int = 30,
    seed: int = 42,
    config_path: str = "configs/default_config.yaml",
):
    if datasets is None:
        datasets = DATASETS
    
    config = load_config(config_path)
    device = get_device()
    results = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"[CrossDataset] Dataset={dataset}")
        print(f"{'='*60}")
        
        window_size = config["datasets"][dataset]["window_size"]
        
        row = {
            "dataset": dataset,
            "window_size": window_size,
            "seed": seed,
            "epochs": epochs,
        }
        
        try:
            pl.seed_everything(seed, workers=True)
            
            real_data, real_targets = get_minority_data(dataset, window_size, config)
            input_dim = real_data.shape[-1]
            n_samples = real_data.shape[0]
            
            row["n_minority_samples"] = n_samples
            row["input_dim"] = input_dim
            row["target_min"] = float(real_targets.min())
            row["target_max"] = float(real_targets.max())
            row["target_mean"] = float(real_targets.mean())
            
            print(f"[CrossDataset] {dataset}: {n_samples} samples, ws={window_size}, dim={input_dim}")
            
            # Build model
            model_cfg = config["generative"]["flowmatch_pdm"].copy()
            model_cfg["epochs"] = epochs
            model_cfg["euler_steps"] = min(model_cfg.get("euler_steps", 200), 100)
            
            model = FlowMatchPdM(input_dim=input_dim, window_size=window_size, config=model_cfg)
            n_params = sum(p.numel() for p in model.parameters())
            row["num_params"] = n_params
            
            ds = TensorDataset(
                torch.from_numpy(real_data).float(),
                torch.from_numpy(real_targets).float(),
            )
            loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True, num_workers=2)
            
            trainer = pl.Trainer(
                max_epochs=epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                precision=32,
                enable_checkpointing=False,
                logger=False,
            )
            
            t0 = time.time()
            trainer.fit(model, train_dataloaders=loader)
            train_time = time.time() - t0
            
            # Generate
            model.eval()
            model.to(device)
            num_gen = min(len(ds), 256)
            cond = torch.from_numpy(real_targets[:num_gen]).float()
            
            t0 = time.time()
            with torch.no_grad():
                synthetic = model.generate(cond, num_gen).cpu().numpy()
            gen_time = time.time() - t0
            
            metrics = compute_generation_metrics(real_data[:num_gen], synthetic)
            
            row.update({
                "train_time_s": round(train_time, 2),
                "gen_time_s": round(gen_time, 2),
                "status": "success",
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
            })
            
            log_experiment("cross_dataset", row, metrics, train_time)
            
        except Exception as e:
            print(f"[CrossDataset] FAILED {dataset}: {e}")
            traceback.print_exc()
            row.update({"status": "failed", "error": str(e)})
        
        results.append(row)
    
    save_csv(results, TABLE_DIR / "cross_dataset.csv")
    save_json(results, TABLE_DIR / "cross_dataset.json")
    _plot_cross_dataset(results)
    
    return results


def _plot_cross_dataset(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        return
    
    names = [r["dataset"] for r in successful]
    mmd = [r.get("mmd", float("nan")) for r in successful]
    ps = [r.get("power_spectrum_l2", float("nan")) for r in successful]
    ws = [r.get("window_size", 0) for r in successful]
    ns = [r.get("n_minority_samples", 0) for r in successful]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Cross-Dataset Validation", fontsize=13, fontweight="bold")
    
    x = np.arange(len(names))
    colors = ["tab:blue", "tab:orange"]
    
    axes[0].bar(x, mmd, color=colors[:len(x)], alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("MMD")
    axes[0].set_title("Generation Quality")
    axes[0].grid(axis="y", alpha=0.3)
    
    axes[1].bar(x, ps, color=colors[:len(x)], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("PS L2")
    axes[1].set_title("Spectral Fidelity")
    axes[1].grid(axis="y", alpha=0.3)
    
    axes[2].bar(x, ws, color=colors[:len(x)], alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names)
    axes[2].set_ylabel("Window Size")
    axes[2].set_title("Sequence Length")
    axes[2].grid(axis="y", alpha=0.3)
    
    axes[3].bar(x, ns, color=colors[:len(x)], alpha=0.8)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(names)
    axes[3].set_ylabel("Samples")
    axes[3].set_title("Minority Set Size")
    axes[3].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"cross_dataset.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    run_cross_dataset(
        datasets=args.datasets,
        epochs=args.epochs,
        seed=args.seed,
        config_path=args.config,
    )
