#!/usr/bin/env python3
"""
Experiment 4: Alternative Model Families.

Adds lightweight baselines to broaden the claim beyond one model family:
  - Autoregressive LSTM baseline
  - Spectral/Frequency-domain baseline (FFT-based generation)
  - VAE baseline (already exists in baselines.py as TimeVAE)
  - TimeFlow (flow-based, simpler architecture)
All use the same preprocessing and evaluation interface.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_utils import (
    FIGURE_DIR, TABLE_DIR,
    compute_generation_metrics, get_device, get_minority_data,
    load_config, log_experiment, save_csv, save_json,
)
from flowmatchPdM.flowmatch_pdm import FlowMatchPdM
from src.baselines import TimeVAE, TimeFlow


# ── Lightweight Autoregressive Baseline ──────────────────────────────
class AutoregressiveLSTM(pl.LightningModule):
    """Simple autoregressive LSTM for sequence generation."""
    
    def __init__(self, input_dim: int, window_size: int, hidden_dim: int = 128,
                 num_layers: int = 2, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)
    
    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.float()
        # Teacher-forced: predict next step from current
        x_in = x[:, :-1, :]
        x_target = x[:, 1:, :]
        pred = self(x_in)
        loss = F.mse_loss(pred, x_target)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    @torch.no_grad()
    def generate(self, num_samples: int, conditions=None):
        """Autoregressive generation."""
        device = self.device
        ws = self.hparams.window_size
        dim = self.hparams.input_dim
        
        # Start with random seed token
        x = torch.randn(num_samples, 1, dim, device=device)
        outputs = [x]
        
        hidden = None
        for t in range(ws - 1):
            out, hidden = self.lstm(x, hidden)
            next_step = self.head(out[:, -1:, :])
            outputs.append(next_step)
            x = next_step
        
        return torch.cat(outputs, dim=1)


# ── Spectral/Frequency-domain Baseline ───────────────────────────────
class SpectralGenerator(pl.LightningModule):
    """Generate signals by learning in the frequency domain."""
    
    def __init__(self, input_dim: int, window_size: int, latent_dim: int = 64,
                 hidden_dim: int = 256, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Number of FFT coefficients
        self.n_fft = window_size // 2 + 1
        fft_flat_dim = self.n_fft * input_dim * 2  # real + imag
        
        # Encoder: time-domain → latent
        self.encoder = nn.Sequential(
            nn.Linear(window_size * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: latent → frequency domain → time domain
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fft_flat_dim),
        )
    
    def encode(self, x):
        B, T, C = x.shape
        flat = x.reshape(B, -1)
        h = self.encoder(flat)
        return self.mu_head(h), self.logvar_head(h)
    
    def decode(self, z):
        B = z.shape[0]
        fft_flat = self.decoder(z)
        # Reshape to complex coefficients
        fft_flat = fft_flat.reshape(B, self.n_fft, self.hparams.input_dim, 2)
        fft_complex = torch.complex(fft_flat[..., 0], fft_flat[..., 1])
        # IRFFT back to time domain
        signal = torch.fft.irfft(fft_complex, n=self.hparams.window_size, dim=1)
        return signal
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decode(z), mu, logvar
    
    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.float()
        recon, mu, logvar = self(x)
        
        # Time-domain reconstruction loss
        recon_loss = F.mse_loss(recon, x)
        
        # Frequency-domain loss
        fft_real = torch.fft.rfft(x, dim=1)
        fft_recon = torch.fft.rfft(recon, dim=1)
        spectral_loss = F.l1_loss(torch.abs(fft_recon), torch.abs(fft_real))
        
        # KL divergence
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + 0.1 * spectral_loss + 0.05 * kld
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    @torch.no_grad()
    def generate(self, num_samples: int, conditions=None):
        z = torch.randn(num_samples, self.hparams.latent_dim, device=self.device)
        return self.decode(z)


# ── Main experiment ──────────────────────────────────────────────────
MODEL_REGISTRY = {
    "FlowMatch": "flowmatch",
    "TimeFlow": "timeflow",
    "TimeVAE": "vae",
    "AutoregressiveLSTM": "autoregressive",
    "SpectralGenerator": "spectral",
}


def _build_model(name, input_dim, window_size, config):
    model_cfg = config["generative"].get("flowmatch_pdm", {})
    
    if name == "FlowMatch":
        cfg = model_cfg.copy()
        cfg["euler_steps"] = min(cfg.get("euler_steps", 200), 100)
        return FlowMatchPdM(input_dim=input_dim, window_size=window_size, config=cfg)
    elif name == "TimeFlow":
        return TimeFlow(input_dim=input_dim, window_size=window_size,
                       hidden_dim=128, euler_steps=100, lr=1e-3)
    elif name == "TimeVAE":
        return TimeVAE(input_dim=input_dim, window_size=window_size,
                      latent_dim=32, hidden_dim=128, lr=1e-3)
    elif name == "AutoregressiveLSTM":
        return AutoregressiveLSTM(input_dim=input_dim, window_size=window_size,
                                  hidden_dim=128, lr=1e-3)
    elif name == "SpectralGenerator":
        return SpectralGenerator(input_dim=input_dim, window_size=window_size,
                                latent_dim=64, hidden_dim=256, lr=1e-3)
    else:
        raise ValueError(f"Unknown model: {name}")


def run_alternative_baselines(
    dataset: str = "FEMTO",
    models: list = None,
    epochs: int = 30,
    seed: int = 42,
    config_path: str = "configs/default_config.yaml",
):
    if models is None:
        models = list(MODEL_REGISTRY.keys())
    
    config = load_config(config_path)
    device = get_device()
    results = []
    
    window_size = config["datasets"][dataset]["window_size"]
    real_data, real_targets = get_minority_data(dataset, window_size, config)
    input_dim = real_data.shape[-1]
    print(f"[AltBaselines] Loaded {real_data.shape[0]} samples")
    
    ds = TensorDataset(
        torch.from_numpy(real_data).float(),
        torch.from_numpy(real_targets).float(),
    )
    loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True, num_workers=2)
    num_gen = min(len(ds), 256)
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"[AltBaselines] Model={model_name}, Dataset={dataset}")
        print(f"{'='*60}")
        
        row = {
            "dataset": dataset,
            "model": model_name,
            "model_family": MODEL_REGISTRY.get(model_name, "unknown"),
            "seed": seed,
            "epochs": epochs,
        }
        
        try:
            pl.seed_everything(seed, workers=True)
            
            model = _build_model(model_name, input_dim, window_size, config)
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            row["num_params"] = n_params
            
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
            cond = torch.from_numpy(real_targets[:num_gen]).float()
            
            t0 = time.time()
            with torch.no_grad():
                if model_name == "FlowMatch":
                    synthetic = model.generate(cond, num_gen).cpu().numpy()
                else:
                    synthetic = model.generate(num_gen).cpu().numpy()
            gen_time = time.time() - t0
            
            metrics = compute_generation_metrics(real_data[:num_gen], synthetic)
            
            row.update({
                "train_time_s": round(train_time, 2),
                "gen_time_s": round(gen_time, 2),
                "status": "success",
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
            })
            
            log_experiment("alt_baselines", row, metrics, train_time)
            
        except Exception as e:
            print(f"[AltBaselines] FAILED {model_name}: {e}")
            traceback.print_exc()
            row.update({"status": "failed", "error": str(e)})
        
        results.append(row)
    
    save_csv(results, TABLE_DIR / "alternative_baselines.csv")
    save_json(results, TABLE_DIR / "alternative_baselines.json")
    _plot_baselines(results)
    
    return results


def _plot_baselines(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        return
    
    names = [r["model"] for r in successful]
    mmd = [r.get("mmd", float("nan")) for r in successful]
    ps = [r.get("power_spectrum_l2", float("nan")) for r in successful]
    params = [r.get("num_params", 0) / 1e6 for r in successful]  # in millions
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Alternative Model Families Comparison", fontsize=13, fontweight="bold")
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    x = np.arange(len(names))
    
    axes[0].bar(x, mmd, color=colors[:len(x)], alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("MMD (lower=better)")
    axes[0].set_title("Generation Quality")
    axes[0].grid(axis="y", alpha=0.3)
    
    axes[1].bar(x, ps, color=colors[:len(x)], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("PS L2 (lower=better)")
    axes[1].set_title("Spectral Fidelity")
    axes[1].grid(axis="y", alpha=0.3)
    
    axes[2].bar(x, params, color=colors[:len(x)], alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[2].set_ylabel("Parameters (M)")
    axes[2].set_title("Model Size")
    axes[2].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"alternative_baselines.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FEMTO")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    run_alternative_baselines(
        dataset=args.dataset,
        models=args.models,
        epochs=args.epochs,
        seed=args.seed,
        config_path=args.config,
    )
