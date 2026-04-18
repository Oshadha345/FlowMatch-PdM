#!/usr/bin/env python3
"""
Experiment 1: Scaling Study over Sequence Length.

Sweeps sequence length [128, 256, 512, 1024, 2048, 2560] for the FlowMatch-PdM
generator, measuring generative quality and downstream utility at each length.
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
    EXPERIMENT_ROOT, FIGURE_DIR, TABLE_DIR,
    compute_generation_metrics, get_device, get_minority_data,
    load_config, log_experiment, save_csv, save_json, timestamp,
)
from flowmatchPdM.flowmatch_pdm import FlowMatchPdM
from src.baselines import TimeFlow


LENGTHS = [128, 256, 512, 1024, 2048, 2560]


def _crop_or_pad(data: np.ndarray, target_len: int) -> np.ndarray:
    """Crop or zero-pad data along the sequence dimension.
    
    data: [N, seq_len, features]
    """
    current_len = data.shape[1]
    if current_len == target_len:
        return data
    elif current_len > target_len:
        # Crop from the end (keep most recent / degraded portion)
        return data[:, -target_len:, :]
    else:
        # Zero-pad at the beginning
        pad_width = target_len - current_len
        return np.pad(data, ((0, 0), (pad_width, 0), (0, 0)), mode="constant")


def run_scaling_study(
    dataset: str = "FEMTO",
    model_name: str = "FlowMatch",
    lengths: list = None,
    epochs: int = 30,
    seed: int = 42,
    config_path: str = "configs/default_config.yaml",
):
    if lengths is None:
        lengths = LENGTHS
    
    config = load_config(config_path)
    device = get_device()
    results = []
    
    # Load full data at native resolution
    native_ws = config["datasets"][dataset]["window_size"]
    print(f"[Scaling] Loading {dataset} at native window_size={native_ws}")
    real_data, real_targets = get_minority_data(dataset, native_ws, config)
    print(f"[Scaling] Loaded {real_data.shape[0]} minority samples, shape={real_data.shape}")
    
    for length in lengths:
        print(f"\n{'='*60}")
        print(f"[Scaling] Length={length}, Dataset={dataset}, Model={model_name}")
        print(f"{'='*60}")
        
        row = {
            "dataset": dataset,
            "model": model_name,
            "seq_length": length,
            "seed": seed,
            "epochs": epochs,
        }
        
        try:
            pl.seed_everything(seed, workers=True)
            
            # Crop/pad data to target length
            data = _crop_or_pad(real_data, length)
            targets = real_targets
            input_dim = data.shape[-1]
            
            # Prepare data loader
            ds = TensorDataset(
                torch.from_numpy(data).float(),
                torch.from_numpy(targets).float(),
            )
            loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True, num_workers=2)
            
            # Build model
            model_cfg = config["generative"]["flowmatch_pdm"].copy()
            model_cfg["epochs"] = epochs
            model_cfg["euler_steps"] = min(model_cfg.get("euler_steps", 200), 100)  # Faster for sweep
            
            # Adjust patch size if needed
            patch_size = model_cfg.get("patch_size", 32)
            if length < patch_size:
                model_cfg["patch_size"] = max(4, length // 4)
            elif length % patch_size != 0:
                # Find largest divisor <= patch_size
                for ps in range(patch_size, 0, -1):
                    if length % ps == 0:
                        model_cfg["patch_size"] = ps
                        break
            
            if model_name == "FlowMatch":
                model = FlowMatchPdM(input_dim=input_dim, window_size=length, config=model_cfg)
            elif model_name == "TimeFlow":
                model = TimeFlow(
                    input_dim=input_dim, window_size=length,
                    hidden_dim=128, euler_steps=100, lr=1e-3,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            # Train
            trainer = pl.Trainer(
                max_epochs=epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                precision=32,
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=True,
            )
            
            t0 = time.time()
            trainer.fit(model, train_dataloaders=loader)
            train_time = time.time() - t0
            
            # Generate
            model.eval()
            model.to(device)
            num_gen = min(len(ds), 256)
            cond = torch.from_numpy(targets[:num_gen]).float()
            
            t0 = time.time()
            with torch.no_grad():
                if model_name == "FlowMatch":
                    synthetic = model.generate(cond, num_gen).cpu().numpy()
                else:
                    synthetic = model.generate(num_gen).cpu().numpy()
            gen_time = time.time() - t0
            
            # Evaluate
            real_eval = data[:num_gen]
            metrics = compute_generation_metrics(real_eval, synthetic)
            
            row.update({
                "train_time_s": round(train_time, 2),
                "gen_time_s": round(gen_time, 2),
                "num_samples": num_gen,
                "status": "success",
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
            })
            
            log_experiment("scaling_study", row, metrics, train_time)
            
        except Exception as e:
            print(f"[Scaling] FAILED at length={length}: {e}")
            traceback.print_exc()
            row.update({"status": "failed", "error": str(e)})
        
        results.append(row)
        print(f"[Scaling] Result: {row}")
    
    # Save results
    save_csv(results, TABLE_DIR / "scaling_study.csv")
    save_json(results, TABLE_DIR / "scaling_study.json")
    
    # Generate figure
    _plot_scaling(results)
    
    return results


def _plot_scaling(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        print("[Scaling] No successful runs to plot.")
        return
    
    lengths = [r["seq_length"] for r in successful]
    mmd = [r.get("mmd", float("nan")) for r in successful]
    ps_l2 = [r.get("power_spectrum_l2", float("nan")) for r in successful]
    train_t = [r.get("train_time_s", 0) for r in successful]
    gen_t = [r.get("gen_time_s", 0) for r in successful]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Scaling Study: Performance vs Sequence Length", fontsize=14, fontweight="bold")
    
    axes[0, 0].plot(lengths, mmd, "o-", color="tab:blue", linewidth=2)
    axes[0, 0].set_xlabel("Sequence Length")
    axes[0, 0].set_ylabel("MMD (lower=better)")
    axes[0, 0].set_title("Maximum Mean Discrepancy")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(lengths, ps_l2, "s-", color="tab:orange", linewidth=2)
    axes[0, 1].set_xlabel("Sequence Length")
    axes[0, 1].set_ylabel("Power Spectrum L2")
    axes[0, 1].set_title("Frequency Domain Fidelity")
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(lengths, train_t, "^-", color="tab:green", linewidth=2)
    axes[1, 0].set_xlabel("Sequence Length")
    axes[1, 0].set_ylabel("Training Time (s)")
    axes[1, 0].set_title("Training Cost")
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(lengths, gen_t, "d-", color="tab:red", linewidth=2)
    axes[1, 1].set_xlabel("Sequence Length")
    axes[1, 1].set_ylabel("Generation Time (s)")
    axes[1, 1].set_title("Generation Cost")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"scaling_study.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Scaling] Figure saved to {FIGURE_DIR / 'scaling_study.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FEMTO")
    parser.add_argument("--model", type=str, default="FlowMatch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths", type=int, nargs="+", default=None)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    run_scaling_study(
        dataset=args.dataset,
        model_name=args.model,
        lengths=args.lengths,
        epochs=args.epochs,
        seed=args.seed,
        config_path=args.config,
    )
