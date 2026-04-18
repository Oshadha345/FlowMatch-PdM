#!/usr/bin/env python3
"""
Experiment 7: Failure Visualization.

Produces paper-ready diagnostic plots:
  1. Real vs Synthetic waveform examples
  2. Frequency spectrum comparison
  3. Phase misalignment / temporal drift
  4. Performance vs length curves (from scaling study)
  5. Performance vs frequency-reduction curves (from freq ablation)
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_utils import (
    FIGURE_DIR, TABLE_DIR,
    get_device, get_minority_data, load_config,
)
from flowmatchPdM.flowmatch_pdm import FlowMatchPdM


def _train_quick_model(real_data, real_targets, window_size, input_dim, config, seed=42, epochs=30):
    """Train a quick FlowMatch model and return it + synthetic data."""
    pl.seed_everything(seed, workers=True)
    
    model_cfg = config["generative"]["flowmatch_pdm"].copy()
    model_cfg["epochs"] = epochs
    model_cfg["euler_steps"] = min(model_cfg.get("euler_steps", 200), 100)
    
    model = FlowMatchPdM(input_dim=input_dim, window_size=window_size, config=model_cfg)
    
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
    trainer.fit(model, train_dataloaders=loader)
    
    model.eval()
    num_gen = min(len(ds), 256)
    cond = torch.from_numpy(real_targets[:num_gen]).float()
    with torch.no_grad():
        synthetic = model.generate(cond, num_gen).cpu().numpy()
    
    return model, synthetic, real_data[:num_gen]


def plot_waveform_comparison(real, synthetic, n_examples=4, dataset_name=""):
    """Plot real vs synthetic waveform examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    n = min(n_examples, len(real), len(synthetic))
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
    fig.suptitle(f"Real vs Synthetic Waveforms ({dataset_name})", fontsize=14, fontweight="bold")
    
    for i in range(n):
        # Real
        ax_r = axes[i, 0] if n > 1 else axes[0]
        ax_r.plot(real[i, :, 0], color="tab:blue", linewidth=0.5, alpha=0.9)
        ax_r.set_ylabel(f"Sample {i+1}")
        if i == 0:
            ax_r.set_title("Real", fontsize=12)
        ax_r.grid(True, alpha=0.2)
        
        # Synthetic
        ax_s = axes[i, 1] if n > 1 else axes[1]
        ax_s.plot(synthetic[i, :, 0], color="tab:orange", linewidth=0.5, alpha=0.9)
        if i == 0:
            ax_s.set_title("Synthetic", fontsize=12)
        ax_s.grid(True, alpha=0.2)
    
    if n > 1:
        axes[-1, 0].set_xlabel("Time Step")
        axes[-1, 1].set_xlabel("Time Step")
    
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"waveform_comparison_{dataset_name}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum_comparison(real, synthetic, dataset_name=""):
    """Plot frequency spectrum comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Average power spectrum
    real_fft = np.abs(np.fft.rfft(real[:, :, 0], axis=1))
    synth_fft = np.abs(np.fft.rfft(synthetic[:, :, 0], axis=1))
    
    real_ps = np.mean(real_fft ** 2, axis=0)
    synth_ps = np.mean(synth_fft ** 2, axis=0)
    
    freqs = np.arange(len(real_ps))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"Frequency Spectrum Analysis ({dataset_name})", fontsize=13, fontweight="bold")
    
    # Power spectrum overlay
    axes[0].semilogy(freqs, real_ps + 1e-12, color="tab:blue", linewidth=1.5, label="Real", alpha=0.8)
    axes[0].semilogy(freqs, synth_ps + 1e-12, color="tab:orange", linewidth=1.5, label="Synthetic", alpha=0.8)
    axes[0].set_xlabel("Frequency Bin")
    axes[0].set_ylabel("Power (log)")
    axes[0].set_title("Power Spectrum")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Spectral difference
    diff = np.abs(real_ps - synth_ps)
    axes[1].bar(freqs, diff, color="tab:red", alpha=0.6, width=1.0)
    axes[1].set_xlabel("Frequency Bin")
    axes[1].set_ylabel("|Real - Synthetic|")
    axes[1].set_title("Spectral Difference")
    axes[1].grid(True, alpha=0.3)
    
    # Cumulative spectral energy
    real_cumul = np.cumsum(real_ps) / max(np.sum(real_ps), 1e-12)
    synth_cumul = np.cumsum(synth_ps) / max(np.sum(synth_ps), 1e-12)
    axes[2].plot(freqs, real_cumul, color="tab:blue", linewidth=2, label="Real")
    axes[2].plot(freqs, synth_cumul, color="tab:orange", linewidth=2, label="Synthetic")
    axes[2].set_xlabel("Frequency Bin")
    axes[2].set_ylabel("Cumulative Energy")
    axes[2].set_title("Cumulative Spectral Energy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"spectrum_comparison_{dataset_name}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_phase_drift(real, synthetic, dataset_name=""):
    """Analyze phase misalignment / temporal drift."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import correlate
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Phase & Temporal Drift Analysis ({dataset_name})", fontsize=13, fontweight="bold")
    
    n_samples = min(100, len(real), len(synthetic))
    
    # Cross-correlation lag distribution
    lags = []
    correlations = []
    for i in range(n_samples):
        r = real[i, :, 0]
        s = synthetic[i, :, 0]
        r = (r - np.mean(r)) / max(np.std(r), 1e-8)
        s = (s - np.mean(s)) / max(np.std(s), 1e-8)
        corr = correlate(r, s, mode="full")
        lag = np.argmax(corr) - (len(r) - 1)
        lags.append(lag)
        correlations.append(float(np.max(corr) / len(r)))
    
    axes[0, 0].hist(lags, bins=30, color="tab:blue", alpha=0.7, edgecolor="black")
    axes[0, 0].axvline(0, color="red", ls="--", label="Zero lag")
    axes[0, 0].set_xlabel("Lag (samples)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Cross-Correlation Peak Lag")
    axes[0, 0].legend()
    
    axes[0, 1].hist(correlations, bins=30, color="tab:green", alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Max Correlation")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Peak Cross-Correlation")
    
    # Autocorrelation comparison
    real_ac = np.mean([np.correlate(real[i, :, 0], real[i, :, 0], mode="full") for i in range(n_samples)], axis=0)
    synth_ac = np.mean([np.correlate(synthetic[i, :, 0], synthetic[i, :, 0], mode="full") for i in range(n_samples)], axis=0)
    
    center = len(real_ac) // 2
    lag_range = min(200, center)
    ac_lags = np.arange(-lag_range, lag_range + 1)
    
    axes[1, 0].plot(ac_lags, real_ac[center - lag_range:center + lag_range + 1],
                    color="tab:blue", linewidth=1.5, label="Real", alpha=0.8)
    axes[1, 0].plot(ac_lags, synth_ac[center - lag_range:center + lag_range + 1],
                    color="tab:orange", linewidth=1.5, label="Synthetic", alpha=0.8)
    axes[1, 0].set_xlabel("Lag")
    axes[1, 0].set_ylabel("Autocorrelation")
    axes[1, 0].set_title("Mean Autocorrelation")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Envelope comparison (rolling RMS)
    win = min(64, real.shape[1] // 8)
    if win > 0:
        r_env = np.sqrt(np.convolve(np.mean(real[:, :, 0] ** 2, axis=0), np.ones(win)/win, mode="same"))
        s_env = np.sqrt(np.convolve(np.mean(synthetic[:, :, 0] ** 2, axis=0), np.ones(win)/win, mode="same"))
        axes[1, 1].plot(r_env, color="tab:blue", linewidth=1.5, label="Real")
        axes[1, 1].plot(s_env, color="tab:orange", linewidth=1.5, label="Synthetic")
        axes[1, 1].set_xlabel("Time Step")
        axes[1, 1].set_ylabel("RMS Envelope")
        axes[1, 1].set_title("Energy Envelope")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"phase_drift_{dataset_name}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_from_saved_results():
    """Generate summary plots from previously saved experiment CSVs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv
    
    # Performance vs length (from scaling study)
    scaling_path = TABLE_DIR / "scaling_study.csv"
    if scaling_path.exists():
        with open(scaling_path) as f:
            rows = list(csv.DictReader(f))
        successful = [r for r in rows if r.get("status") == "success"]
        if successful:
            lengths = [int(r["seq_length"]) for r in successful]
            mmd = [float(r.get("mmd", "nan")) for r in successful]
            ps = [float(r.get("power_spectrum_l2", "nan")) for r in successful]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle("Performance vs Sequence Length", fontsize=13, fontweight="bold")
            
            ax1.plot(lengths, mmd, "o-", color="tab:blue", linewidth=2, markersize=8)
            ax1.set_xlabel("Sequence Length")
            ax1.set_ylabel("MMD (lower=better)")
            ax1.set_title("Distributional Distance")
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(lengths, ps, "s-", color="tab:orange", linewidth=2, markersize=8)
            ax2.set_xlabel("Sequence Length")
            ax2.set_ylabel("Power Spectrum L2")
            ax2.set_title("Spectral Fidelity")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            for fmt in ("png", "pdf"):
                fig.savefig(FIGURE_DIR / f"perf_vs_length.{fmt}", dpi=200, bbox_inches="tight")
            plt.close(fig)
    
    # Performance vs frequency reduction
    freq_path = TABLE_DIR / "frequency_ablation.csv"
    if freq_path.exists():
        with open(freq_path) as f:
            rows = list(csv.DictReader(f))
        successful = [r for r in rows if r.get("status") == "success"]
        if successful:
            names = [r["variant"] for r in successful]
            mmd = [float(r.get("mmd", "nan")) for r in successful]
            hf_real = [float(r.get("real_hf_energy_ratio", "0")) for r in successful]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            fig.suptitle("Performance vs Frequency Content", fontsize=13, fontweight="bold")
            
            x = np.arange(len(names))
            ax1.bar(x, mmd, color="tab:blue", alpha=0.8)
            ax1.set_xticks(x)
            ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax1.set_ylabel("MMD")
            ax1.set_title("Generation Quality")
            ax1.grid(axis="y", alpha=0.3)
            
            ax2.bar(x, hf_real, color="tab:orange", alpha=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel("HF Energy Ratio")
            ax2.set_title("High-Frequency Energy in Input")
            ax2.grid(axis="y", alpha=0.3)
            
            plt.tight_layout()
            for fmt in ("png", "pdf"):
                fig.savefig(FIGURE_DIR / f"perf_vs_frequency.{fmt}", dpi=200, bbox_inches="tight")
            plt.close(fig)


def run_failure_visualization(
    dataset: str = "FEMTO",
    epochs: int = 30,
    seed: int = 42,
    config_path: str = "configs/default_config.yaml",
):
    config = load_config(config_path)
    window_size = config["datasets"][dataset]["window_size"]
    real_data, real_targets = get_minority_data(dataset, window_size, config)
    input_dim = real_data.shape[-1]
    
    print(f"[Viz] Training model for visualization (epochs={epochs})...")
    model, synthetic, real_eval = _train_quick_model(
        real_data, real_targets, window_size, input_dim, config, seed, epochs
    )
    
    print("[Viz] Generating waveform comparison...")
    plot_waveform_comparison(real_eval, synthetic, n_examples=4, dataset_name=dataset)
    
    print("[Viz] Generating spectrum comparison...")
    plot_spectrum_comparison(real_eval, synthetic, dataset_name=dataset)
    
    print("[Viz] Generating phase drift analysis...")
    plot_phase_drift(real_eval, synthetic, dataset_name=dataset)
    
    print("[Viz] Generating plots from saved experiment results...")
    plot_from_saved_results()
    
    print(f"[Viz] All figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FEMTO")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    run_failure_visualization(
        dataset=args.dataset,
        epochs=args.epochs,
        seed=args.seed,
        config_path=args.config,
    )
