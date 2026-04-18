#!/usr/bin/env python3
"""
Experiment 3: ODE Solver Sensitivity Study.

Evaluates the FlowMatch-PdM generator under different ODE solvers:
  - Euler (various step counts)
  - RK4 (Runge-Kutta 4th order)
  - Adaptive Dopri5 (if torchdiffeq is available)
Compares stability, runtime, divergence, and output quality.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
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


SOLVER_CONFIGS = [
    {"name": "euler_50",   "solver": "euler", "steps": 50},
    {"name": "euler_100",  "solver": "euler", "steps": 100},
    {"name": "euler_200",  "solver": "euler", "steps": 200},
    {"name": "euler_500",  "solver": "euler", "steps": 500},
    {"name": "rk4_25",     "solver": "rk4",   "steps": 25},
    {"name": "rk4_50",     "solver": "rk4",   "steps": 50},
    {"name": "rk4_100",    "solver": "rk4",   "steps": 100},
    {"name": "dopri5",     "solver": "dopri5", "steps": None},  # adaptive
]


def _generate_euler(model, conditions, num_samples, steps, device):
    """Manual Euler integration."""
    if conditions.dim() == 1:
        conditions = conditions.unsqueeze(-1)
    conditions = conditions.to(device).float()
    
    x_t = model._sample_base_distribution(conditions, num_samples)
    dt = 1.0 / max(steps, 1)
    
    for step in range(steps):
        t_val = step / max(steps - 1, 1)
        t_tensor = torch.full((num_samples, 1), t_val, device=device, dtype=x_t.dtype)
        v_t = model.forward(t_tensor, x_t, conditions)
        v_t = torch.clamp(v_t, min=-model.field_clamp, max=model.field_clamp)
        x_t = x_t + v_t * dt
    
    return x_t


def _generate_rk4(model, conditions, num_samples, steps, device):
    """4th-order Runge-Kutta integration."""
    if conditions.dim() == 1:
        conditions = conditions.unsqueeze(-1)
    conditions = conditions.to(device).float()
    
    x_t = model._sample_base_distribution(conditions, num_samples)
    dt = 1.0 / max(steps, 1)
    clamp = model.field_clamp
    
    for step in range(steps):
        t_val = step * dt
        t1 = torch.full((num_samples, 1), t_val, device=device, dtype=x_t.dtype)
        t2 = torch.full((num_samples, 1), t_val + 0.5 * dt, device=device, dtype=x_t.dtype)
        t3 = torch.full((num_samples, 1), t_val + dt, device=device, dtype=x_t.dtype)
        
        k1 = torch.clamp(model.forward(t1, x_t, conditions), -clamp, clamp)
        k2 = torch.clamp(model.forward(t2, x_t + 0.5 * dt * k1, conditions), -clamp, clamp)
        k3 = torch.clamp(model.forward(t2, x_t + 0.5 * dt * k2, conditions), -clamp, clamp)
        k4 = torch.clamp(model.forward(t3, x_t + dt * k3, conditions), -clamp, clamp)
        
        x_t = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return x_t


def _generate_dopri5(model, conditions, num_samples, device):
    """Adaptive Dormand-Prince integration via torchdiffeq."""
    try:
        from torchdiffeq import odeint
    except ImportError:
        raise RuntimeError("torchdiffeq not available for Dopri5 solver")
    
    if conditions.dim() == 1:
        conditions = conditions.unsqueeze(-1)
    conditions = conditions.to(device).float()
    
    x0 = model._sample_base_distribution(conditions, num_samples)
    
    class ODEFunc(torch.nn.Module):
        def __init__(self, flow_model, cond):
            super().__init__()
            self.flow_model = flow_model
            self.cond = cond
        
        def forward(self, t, x):
            batch = x.shape[0]
            t_tensor = torch.full((batch, 1), t.item(), device=x.device, dtype=x.dtype)
            v = self.flow_model.forward(t_tensor, x, self.cond)
            return torch.clamp(v, -self.flow_model.field_clamp, self.flow_model.field_clamp)
    
    func = ODEFunc(model, conditions)
    t_span = torch.tensor([0.0, 1.0], device=device)
    
    solution = odeint(func, x0, t_span, method="dopri5", rtol=1e-5, atol=1e-5)
    return solution[-1]


def run_solver_sensitivity(
    dataset: str = "FEMTO",
    epochs: int = 30,
    seed: int = 42,
    config_path: str = "configs/default_config.yaml",
):
    config = load_config(config_path)
    device = get_device()
    results = []
    
    window_size = config["datasets"][dataset]["window_size"]
    real_data, real_targets = get_minority_data(dataset, window_size, config)
    input_dim = real_data.shape[-1]
    print(f"[Solver] Loaded {real_data.shape[0]} samples")
    
    # Train a single model
    pl.seed_everything(seed, workers=True)
    model_cfg = config["generative"]["flowmatch_pdm"].copy()
    model_cfg["epochs"] = epochs
    model_cfg["euler_steps"] = 200
    
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
    
    print(f"[Solver] Training model for {epochs} epochs...")
    trainer.fit(model, train_dataloaders=loader)
    model.eval()
    model.to(device)
    
    num_gen = min(len(ds), 256)
    cond = torch.from_numpy(real_targets[:num_gen]).float()
    real_eval = real_data[:num_gen]
    
    # Test each solver
    for scfg in SOLVER_CONFIGS:
        name = scfg["name"]
        solver = scfg["solver"]
        steps = scfg["steps"]
        
        print(f"\n[Solver] Testing {name} (solver={solver}, steps={steps})")
        
        row = {
            "dataset": dataset,
            "solver_name": name,
            "solver_type": solver,
            "steps": steps if steps else "adaptive",
            "seed": seed,
        }
        
        try:
            pl.seed_everything(seed, workers=True)
            
            t0 = time.time()
            with torch.no_grad():
                if solver == "euler":
                    synthetic = _generate_euler(model, cond, num_gen, steps, device).cpu().numpy()
                elif solver == "rk4":
                    synthetic = _generate_rk4(model, cond, num_gen, steps, device).cpu().numpy()
                elif solver == "dopri5":
                    synthetic = _generate_dopri5(model, cond, num_gen, device).cpu().numpy()
                else:
                    raise ValueError(f"Unknown solver: {solver}")
            gen_time = time.time() - t0
            
            # Check for NaN / Inf
            has_nan = bool(np.isnan(synthetic).any())
            has_inf = bool(np.isinf(synthetic).any())
            max_abs = float(np.nanmax(np.abs(synthetic)))
            
            if has_nan or has_inf:
                print(f"[Solver] WARNING: {name} produced NaN={has_nan}, Inf={has_inf}")
                # Replace NaN/Inf for metric computation
                synthetic = np.nan_to_num(synthetic, nan=0.0, posinf=1e6, neginf=-1e6)
            
            metrics = compute_generation_metrics(real_eval, synthetic)
            metrics["has_nan"] = has_nan
            metrics["has_inf"] = has_inf
            metrics["max_abs_value"] = max_abs
            metrics["diverged"] = has_nan or has_inf or max_abs > 1e4
            
            row.update({
                "gen_time_s": round(gen_time, 4),
                "status": "success",
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
            })
            
            log_experiment("solver_sensitivity", row, metrics, gen_time)
            
        except Exception as e:
            print(f"[Solver] FAILED {name}: {e}")
            traceback.print_exc()
            row.update({"status": "failed", "error": str(e)})
        
        results.append(row)
    
    save_csv(results, TABLE_DIR / "solver_sensitivity.csv")
    save_json(results, TABLE_DIR / "solver_sensitivity.json")
    _plot_solver_sensitivity(results)
    
    return results


def _plot_solver_sensitivity(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        return
    
    names = [r["solver_name"] for r in successful]
    mmd = [r.get("mmd", float("nan")) for r in successful]
    gen_t = [r.get("gen_time_s", 0) for r in successful]
    diverged = [r.get("diverged", False) for r in successful]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ODE Solver Sensitivity", fontsize=13, fontweight="bold")
    
    x = np.arange(len(names))
    colors = ["tab:red" if d else "tab:blue" for d in diverged]
    
    axes[0].bar(x, mmd, color=colors, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("MMD")
    axes[0].set_title("Quality (MMD, lower=better)")
    axes[0].grid(axis="y", alpha=0.3)
    
    axes[1].bar(x, gen_t, color="tab:green", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Generation Runtime")
    axes[1].grid(axis="y", alpha=0.3)
    
    max_abs = [r.get("max_abs_value", 0) for r in successful]
    axes[2].bar(x, max_abs, color="tab:purple", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Max |x|")
    axes[2].set_title("Stability (Max Absolute Value)")
    axes[2].set_yscale("log")
    axes[2].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"solver_sensitivity.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FEMTO")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    run_solver_sensitivity(
        dataset=args.dataset,
        epochs=args.epochs,
        seed=args.seed,
        config_path=args.config,
    )
