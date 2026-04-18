#!/usr/bin/env python3
"""
Experiment 2: Frequency Ablation Study.

Compares FlowMatch-PdM performance on:
  - raw (unfiltered) signals
  - low-pass filtered signals (various cutoff frequencies)
  - downsampled signals
Shows whether failure is tied to high-frequency content.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from scipy import signal as sp_signal
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_utils import (
    FIGURE_DIR, TABLE_DIR,
    compute_generation_metrics, get_device, get_minority_data,
    load_config, log_experiment, save_csv, save_json,
)
from flowmatchPdM.flowmatch_pdm import FlowMatchPdM


# Frequency variants: (name, type, param)
FREQ_VARIANTS = [
    ("raw", "none", None),
    ("lowpass_0.3", "lowpass", 0.3),      # Cutoff at 0.3 * Nyquist
    ("lowpass_0.1", "lowpass", 0.1),      # Cutoff at 0.1 * Nyquist
    ("lowpass_0.05", "lowpass", 0.05),    # Aggressive low-pass
    ("bandpass_0.05_0.3", "bandpass", (0.05, 0.3)),
    ("downsample_2x", "downsample", 2),
    ("downsample_4x", "downsample", 4),
]


def apply_frequency_transform(
    data: np.ndarray, variant_type: str, param
) -> np.ndarray:
    """Apply frequency-domain transform to vibration data.
    
    data: [N, seq_len, features]
    Returns transformed data of the same shape (after potential resizing).
    """
    if variant_type == "none":
        return data
    
    N, seq_len, features = data.shape
    out = np.zeros_like(data)
    
    for i in range(N):
        for f in range(features):
            sig = data[i, :, f]
            if variant_type == "lowpass":
                cutoff = float(param)
                # Butterworth low-pass filter
                sos = sp_signal.butter(4, cutoff, btype="low", output="sos")
                out[i, :, f] = sp_signal.sosfiltfilt(sos, sig)
            
            elif variant_type == "bandpass":
                low, high = param
                sos = sp_signal.butter(4, [low, high], btype="band", output="sos")
                out[i, :, f] = sp_signal.sosfiltfilt(sos, sig)
            
            elif variant_type == "downsample":
                factor = int(param)
                downsampled = sp_signal.decimate(sig, factor)
                # Upsample back to original length via interpolation
                out[i, :, f] = np.interp(
                    np.linspace(0, 1, seq_len),
                    np.linspace(0, 1, len(downsampled)),
                    downsampled,
                )
            else:
                out[i, :, f] = sig
    
    return out


def run_frequency_ablation(
    dataset: str = "FEMTO",
    variants: list = None,
    epochs: int = 30,
    seed: int = 42,
    config_path: str = "configs/default_config.yaml",
):
    if variants is None:
        variants = FREQ_VARIANTS
    
    config = load_config(config_path)
    device = get_device()
    results = []
    
    window_size = config["datasets"][dataset]["window_size"]
    real_data, real_targets = get_minority_data(dataset, window_size, config)
    print(f"[FreqAblation] Loaded {real_data.shape[0]} samples, shape={real_data.shape}")
    
    for name, vtype, param in variants:
        print(f"\n{'='*60}")
        print(f"[FreqAblation] Variant={name}, Dataset={dataset}")
        print(f"{'='*60}")
        
        row = {
            "dataset": dataset,
            "variant": name,
            "variant_type": vtype,
            "param": str(param),
            "seed": seed,
            "epochs": epochs,
            "window_size": window_size,
        }
        
        try:
            pl.seed_everything(seed, workers=True)
            
            # Apply transform
            data = apply_frequency_transform(real_data, vtype, param)
            input_dim = data.shape[-1]
            
            # Dataloader
            ds = TensorDataset(
                torch.from_numpy(data).float(),
                torch.from_numpy(real_targets).float(),
            )
            loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True, num_workers=2)
            
            # Build model
            model_cfg = config["generative"]["flowmatch_pdm"].copy()
            model_cfg["epochs"] = epochs
            model_cfg["euler_steps"] = min(model_cfg.get("euler_steps", 200), 100)
            
            model = FlowMatchPdM(input_dim=input_dim, window_size=window_size, config=model_cfg)
            
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
            cond = torch.from_numpy(real_targets[:num_gen]).float()
            
            t0 = time.time()
            with torch.no_grad():
                synthetic = model.generate(cond, num_gen).cpu().numpy()
            gen_time = time.time() - t0
            
            # Metrics
            metrics = compute_generation_metrics(data[:num_gen], synthetic)
            
            # Extra: measure high-freq energy ratio
            real_fft = np.abs(np.fft.rfft(data[:num_gen], axis=1))
            synth_fft = np.abs(np.fft.rfft(synthetic, axis=1))
            n_freq = real_fft.shape[1]
            hf_boundary = n_freq // 2  # upper half of frequency spectrum
            real_hf_energy = np.mean(real_fft[:, hf_boundary:, :] ** 2)
            synth_hf_energy = np.mean(synth_fft[:, hf_boundary:, :] ** 2)
            real_total_energy = np.mean(real_fft ** 2)
            synth_total_energy = np.mean(synth_fft ** 2)
            
            metrics["real_hf_energy_ratio"] = float(real_hf_energy / max(real_total_energy, 1e-12))
            metrics["synth_hf_energy_ratio"] = float(synth_hf_energy / max(synth_total_energy, 1e-12))
            
            row.update({
                "train_time_s": round(train_time, 2),
                "gen_time_s": round(gen_time, 2),
                "status": "success",
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
            })
            
            log_experiment("freq_ablation", row, metrics, train_time)
            
        except Exception as e:
            print(f"[FreqAblation] FAILED variant={name}: {e}")
            traceback.print_exc()
            row.update({"status": "failed", "error": str(e)})
        
        results.append(row)
    
    save_csv(results, TABLE_DIR / "frequency_ablation.csv")
    save_json(results, TABLE_DIR / "frequency_ablation.json")
    _plot_frequency_ablation(results)
    
    return results


def _plot_frequency_ablation(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        return
    
    names = [r["variant"] for r in successful]
    mmd = [r.get("mmd", float("nan")) for r in successful]
    ps = [r.get("power_spectrum_l2", float("nan")) for r in successful]
    hf_real = [r.get("real_hf_energy_ratio", 0) for r in successful]
    hf_synth = [r.get("synth_hf_energy_ratio", 0) for r in successful]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Frequency Ablation: Generation Quality vs Frequency Content", fontsize=13, fontweight="bold")
    
    x = np.arange(len(names))
    
    axes[0].bar(x, mmd, color="tab:blue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("MMD")
    axes[0].set_title("MMD by Variant")
    axes[0].grid(axis="y", alpha=0.3)
    
    axes[1].bar(x, ps, color="tab:orange", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Power Spectrum L2")
    axes[1].set_title("Spectral Fidelity by Variant")
    axes[1].grid(axis="y", alpha=0.3)
    
    w = 0.35
    axes[2].bar(x - w/2, hf_real, w, label="Real", color="tab:green", alpha=0.8)
    axes[2].bar(x + w/2, hf_synth, w, label="Synthetic", color="tab:red", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("HF Energy Ratio")
    axes[2].set_title("High-Frequency Energy")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"frequency_ablation.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FEMTO")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    run_frequency_ablation(
        dataset=args.dataset,
        epochs=args.epochs,
        seed=args.seed,
        config_path=args.config,
    )
