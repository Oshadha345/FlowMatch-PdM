#!/usr/bin/env python3
"""
Experiment 5: TSTR Robustness with Multiple Seeds.

Runs Train-on-Synthetic, Test-on-Real evaluation across multiple seeds
and computes mean, std, and bootstrap confidence intervals.
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
from src.evaluation import TSTR_Evaluation


SEEDS = [42, 123, 456, 789, 1024]


def _assign_rul_classes(targets: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Bin RUL values into discrete classes for TSTR."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(targets, percentiles)
    bin_edges[-1] += 1e-6  # ensure max is included
    classes = np.digitize(targets, bin_edges[1:-1])
    return classes.astype(np.int64)


def run_tstr_robustness(
    dataset: str = "FEMTO",
    seeds: list = None,
    gen_epochs: int = 30,
    config_path: str = "configs/default_config.yaml",
):
    if seeds is None:
        seeds = SEEDS
    
    config = load_config(config_path)
    device = get_device()
    
    window_size = config["datasets"][dataset]["window_size"]
    real_data, real_targets = get_minority_data(dataset, window_size, config)
    input_dim = real_data.shape[-1]
    print(f"[TSTR] Loaded {real_data.shape[0]} samples")
    
    # Create discrete labels for TSTR classification
    real_classes = _assign_rul_classes(real_targets, n_bins=4)
    unique_classes = np.unique(real_classes)
    print(f"[TSTR] Created {len(unique_classes)} RUL classes: {unique_classes}")
    
    all_results = []
    tstr_metrics_per_seed = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[TSTR] Seed={seed}, Dataset={dataset}")
        print(f"{'='*60}")
        
        row = {"dataset": dataset, "seed": seed, "gen_epochs": gen_epochs}
        
        try:
            pl.seed_everything(seed, workers=True)
            
            # Train generator
            model_cfg = config["generative"]["flowmatch_pdm"].copy()
            model_cfg["epochs"] = gen_epochs
            model_cfg["euler_steps"] = min(model_cfg.get("euler_steps", 200), 100)
            
            model = FlowMatchPdM(input_dim=input_dim, window_size=window_size, config=model_cfg)
            
            ds = TensorDataset(
                torch.from_numpy(real_data).float(),
                torch.from_numpy(real_targets).float(),
            )
            loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True, num_workers=2)
            
            trainer = pl.Trainer(
                max_epochs=gen_epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                precision=32,
                enable_checkpointing=False,
                logger=False,
            )
            
            t0 = time.time()
            trainer.fit(model, train_dataloaders=loader)
            train_time = time.time() - t0
            
            # Generate synthetic data
            model.eval()
            num_gen = len(ds)
            cond = torch.from_numpy(real_targets[:num_gen]).float()
            
            with torch.no_grad():
                synthetic = model.generate(cond, num_gen).cpu().numpy()
            
            # Assign classes to synthetic data (same binning as real)
            synth_classes = _assign_rul_classes(real_targets[:num_gen], n_bins=4)
            
            # Generation quality
            gen_metrics = compute_generation_metrics(real_data[:num_gen], synthetic)
            
            # TSTR Evaluation
            save_dir = TABLE_DIR / f"tstr_seed_{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            tstr_eval = TSTR_Evaluation(
                save_dir=str(save_dir),
                batch_size=64,
                epochs=20,
                device=device,
            )
            
            tstr_metrics = tstr_eval.run(
                synthetic_data=synthetic,
                synthetic_targets=synth_classes,
                real_data=real_data,
                real_targets=real_classes,
                filename_prefix=f"tstr_seed_{seed}",
            )
            
            row.update({
                "train_time_s": round(train_time, 2),
                "status": "success",
                **{f"gen_{k}": round(v, 6) if isinstance(v, float) else v for k, v in gen_metrics.items()},
                **{f"tstr_{k}": v for k, v in tstr_metrics.items()},
            })
            
            tstr_metrics_per_seed.append(tstr_metrics)
            log_experiment("tstr_robustness", row, {**gen_metrics, **tstr_metrics}, train_time)
            
        except Exception as e:
            print(f"[TSTR] FAILED seed={seed}: {e}")
            traceback.print_exc()
            row.update({"status": "failed", "error": str(e)})
        
        all_results.append(row)
    
    # Aggregate statistics
    summary = _aggregate_tstr(tstr_metrics_per_seed, dataset)
    
    save_csv(all_results, TABLE_DIR / "tstr_robustness.csv")
    save_json(all_results, TABLE_DIR / "tstr_robustness.json")
    save_json(summary, TABLE_DIR / "tstr_robustness_summary.json")
    _plot_tstr(all_results, summary)
    
    return all_results, summary


def _aggregate_tstr(metrics_list, dataset):
    """Compute mean, std, and bootstrap CIs for TSTR metrics."""
    if not metrics_list:
        return {}
    
    numeric_keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))]
    
    summary = {"dataset": dataset, "n_seeds": len(metrics_list)}
    
    for key in numeric_keys:
        vals = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(arr))
        summary[f"{key}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        
        # Bootstrap 95% CI
        if len(arr) >= 3:
            n_boot = 1000
            rng = np.random.default_rng(42)
            boot_means = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))
            summary[f"{key}_ci95_low"] = ci_low
            summary[f"{key}_ci95_high"] = ci_high
    
    # Gate pass rate
    gate_vals = [m.get("gate_passed", False) for m in metrics_list]
    summary["gate_pass_rate"] = float(sum(gate_vals)) / max(len(gate_vals), 1)
    
    return summary


def _plot_tstr(results, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        return
    
    seeds = [r["seed"] for r in successful]
    
    # Extract key TSTR metrics
    tstr_f1 = [r.get("tstr_tstr_f1_macro", r.get("tstr_f1_macro", 0)) for r in successful]
    tstr_ba = [r.get("tstr_tstr_balanced_accuracy", r.get("tstr_balanced_accuracy", 0)) for r in successful]
    trtr_f1 = [r.get("tstr_trtr_f1_macro", r.get("trtr_f1_macro", 0)) for r in successful]
    gen_mmd = [r.get("gen_mmd", 0) for r in successful]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("TSTR Robustness Across Seeds", fontsize=13, fontweight="bold")
    
    x = np.arange(len(seeds))
    
    # TSTR F1
    axes[0, 0].bar(x, tstr_f1, color="tab:blue", alpha=0.8, label="TSTR")
    axes[0, 0].bar(x, trtr_f1, color="tab:gray", alpha=0.3, label="TRTR (ref)")
    if "tstr_f1_macro_mean" in summary:
        axes[0, 0].axhline(summary["tstr_f1_macro_mean"], color="red", ls="--", label="Mean")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([str(s) for s in seeds])
    axes[0, 0].set_xlabel("Seed")
    axes[0, 0].set_ylabel("F1 Macro")
    axes[0, 0].set_title("TSTR vs TRTR F1")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.3)
    
    # TSTR Balanced Accuracy
    axes[0, 1].bar(x, tstr_ba, color="tab:orange", alpha=0.8)
    if "tstr_balanced_accuracy_mean" in summary:
        axes[0, 1].axhline(summary["tstr_balanced_accuracy_mean"], color="red", ls="--", label="Mean")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(s) for s in seeds])
    axes[0, 1].set_xlabel("Seed")
    axes[0, 1].set_ylabel("Balanced Accuracy")
    axes[0, 1].set_title("TSTR Balanced Accuracy")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    # Gen quality (MMD)
    axes[1, 0].bar(x, gen_mmd, color="tab:green", alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([str(s) for s in seeds])
    axes[1, 0].set_xlabel("Seed")
    axes[1, 0].set_ylabel("MMD")
    axes[1, 0].set_title("Generation Quality (MMD)")
    axes[1, 0].grid(axis="y", alpha=0.3)
    
    # Summary box
    axes[1, 1].axis("off")
    summary_text = "TSTR Summary\n" + "=" * 30 + "\n"
    summary_text += f"Seeds: {len(seeds)}\n"
    summary_text += f"Gate pass rate: {summary.get('gate_pass_rate', 'N/A'):.1%}\n"
    for key in ["tstr_f1_macro_mean", "tstr_f1_macro_std", "tstr_balanced_accuracy_mean", "tstr_balanced_accuracy_std"]:
        if key in summary:
            label = key.replace("tstr_", "").replace("_", " ").title()
            summary_text += f"{label}: {summary[key]:.4f}\n"
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, family="monospace", verticalalignment="center",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"tstr_robustness.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FEMTO")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--gen_epochs", type=int, default=30)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    run_tstr_robustness(
        dataset=args.dataset,
        seeds=args.seeds,
        gen_epochs=args.gen_epochs,
        config_path=args.config,
    )
