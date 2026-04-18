#!/usr/bin/env python3
"""
Master Experiment Runner for IRAI 2026 Reviewer Checks.

Runs all 7 experiments sequentially with error isolation:
  1. Scaling study over sequence length
  2. Frequency ablation
  3. Solver sensitivity
  4. Alternative model families
  5. TSTR robustness (multi-seed)
  6. Cross-dataset validation
  7. Failure visualization

Usage:
    python experiments/run_all_experiments.py
    python experiments/run_all_experiments.py --experiments 1 3 7
    python experiments/run_all_experiments.py --dataset FEMTO --epochs 30
"""
import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_utils import (
    EXPERIMENT_ROOT, FIGURE_DIR, TABLE_DIR, LOG_DIR,
    save_json, timestamp,
)


def run_all(
    experiments: list = None,
    dataset: str = "FEMTO",
    epochs: int = 30,
    seed: int = 42,
    config_path: str = "configs/default_config.yaml",
):
    if experiments is None:
        experiments = [1, 2, 3, 4, 5, 6, 7]
    
    overall_start = time.time()
    command_log = []
    status_log = {}
    
    print(f"\n{'#'*70}")
    print(f"# IRAI 2026 REVIEWER EXPERIMENT SUITE")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"# Dataset: {dataset}, Epochs: {epochs}, Seed: {seed}")
    print(f"# Experiments: {experiments}")
    print(f"# Output: {EXPERIMENT_ROOT}")
    print(f"{'#'*70}\n")
    
    # ── Experiment 1: Scaling Study ──────────────────────────────────
    if 1 in experiments:
        print(f"\n{'━'*70}")
        print("EXPERIMENT 1: Scaling Study over Sequence Length")
        print(f"{'━'*70}")
        cmd = f"python experiments/exp1_scaling_study.py --dataset {dataset} --epochs {epochs} --seed {seed} --config {config_path}"
        command_log.append(cmd)
        try:
            from experiments.exp1_scaling_study import run_scaling_study
            t0 = time.time()
            run_scaling_study(dataset=dataset, epochs=epochs, seed=seed, config_path=config_path)
            status_log["exp1_scaling"] = {"status": "success", "runtime_s": round(time.time() - t0, 1)}
        except Exception as e:
            print(f"[FAILED] Exp 1: {e}")
            traceback.print_exc()
            status_log["exp1_scaling"] = {"status": "failed", "error": str(e)}
    
    # ── Experiment 2: Frequency Ablation ─────────────────────────────
    if 2 in experiments:
        print(f"\n{'━'*70}")
        print("EXPERIMENT 2: Frequency Ablation")
        print(f"{'━'*70}")
        cmd = f"python experiments/exp2_frequency_ablation.py --dataset {dataset} --epochs {epochs} --seed {seed} --config {config_path}"
        command_log.append(cmd)
        try:
            from experiments.exp2_frequency_ablation import run_frequency_ablation
            t0 = time.time()
            run_frequency_ablation(dataset=dataset, epochs=epochs, seed=seed, config_path=config_path)
            status_log["exp2_frequency"] = {"status": "success", "runtime_s": round(time.time() - t0, 1)}
        except Exception as e:
            print(f"[FAILED] Exp 2: {e}")
            traceback.print_exc()
            status_log["exp2_frequency"] = {"status": "failed", "error": str(e)}
    
    # ── Experiment 3: Solver Sensitivity ─────────────────────────────
    if 3 in experiments:
        print(f"\n{'━'*70}")
        print("EXPERIMENT 3: ODE Solver Sensitivity")
        print(f"{'━'*70}")
        cmd = f"python experiments/exp3_solver_sensitivity.py --dataset {dataset} --epochs {epochs} --seed {seed} --config {config_path}"
        command_log.append(cmd)
        try:
            from experiments.exp3_solver_sensitivity import run_solver_sensitivity
            t0 = time.time()
            run_solver_sensitivity(dataset=dataset, epochs=epochs, seed=seed, config_path=config_path)
            status_log["exp3_solver"] = {"status": "success", "runtime_s": round(time.time() - t0, 1)}
        except Exception as e:
            print(f"[FAILED] Exp 3: {e}")
            traceback.print_exc()
            status_log["exp3_solver"] = {"status": "failed", "error": str(e)}
    
    # ── Experiment 4: Alternative Baselines ──────────────────────────
    if 4 in experiments:
        print(f"\n{'━'*70}")
        print("EXPERIMENT 4: Alternative Model Families")
        print(f"{'━'*70}")
        cmd = f"python experiments/exp4_alternative_baselines.py --dataset {dataset} --epochs {epochs} --seed {seed} --config {config_path}"
        command_log.append(cmd)
        try:
            from experiments.exp4_alternative_baselines import run_alternative_baselines
            t0 = time.time()
            run_alternative_baselines(dataset=dataset, epochs=epochs, seed=seed, config_path=config_path)
            status_log["exp4_baselines"] = {"status": "success", "runtime_s": round(time.time() - t0, 1)}
        except Exception as e:
            print(f"[FAILED] Exp 4: {e}")
            traceback.print_exc()
            status_log["exp4_baselines"] = {"status": "failed", "error": str(e)}
    
    # ── Experiment 5: TSTR Robustness ────────────────────────────────
    if 5 in experiments:
        print(f"\n{'━'*70}")
        print("EXPERIMENT 5: TSTR Robustness (Multi-Seed)")
        print(f"{'━'*70}")
        cmd = f"python experiments/exp5_tstr_robustness.py --dataset {dataset} --gen_epochs {epochs} --config {config_path}"
        command_log.append(cmd)
        try:
            from experiments.exp5_tstr_robustness import run_tstr_robustness
            t0 = time.time()
            run_tstr_robustness(dataset=dataset, gen_epochs=epochs, config_path=config_path)
            status_log["exp5_tstr"] = {"status": "success", "runtime_s": round(time.time() - t0, 1)}
        except Exception as e:
            print(f"[FAILED] Exp 5: {e}")
            traceback.print_exc()
            status_log["exp5_tstr"] = {"status": "failed", "error": str(e)}
    
    # ── Experiment 6: Cross-Dataset ──────────────────────────────────
    if 6 in experiments:
        print(f"\n{'━'*70}")
        print("EXPERIMENT 6: Cross-Dataset Validation")
        print(f"{'━'*70}")
        cmd = f"python experiments/exp6_cross_dataset.py --epochs {epochs} --seed {seed} --config {config_path}"
        command_log.append(cmd)
        try:
            from experiments.exp6_cross_dataset import run_cross_dataset
            t0 = time.time()
            run_cross_dataset(epochs=epochs, seed=seed, config_path=config_path)
            status_log["exp6_cross_dataset"] = {"status": "success", "runtime_s": round(time.time() - t0, 1)}
        except Exception as e:
            print(f"[FAILED] Exp 6: {e}")
            traceback.print_exc()
            status_log["exp6_cross_dataset"] = {"status": "failed", "error": str(e)}
    
    # ── Experiment 7: Failure Visualization ──────────────────────────
    if 7 in experiments:
        print(f"\n{'━'*70}")
        print("EXPERIMENT 7: Failure Visualization")
        print(f"{'━'*70}")
        cmd = f"python experiments/exp7_failure_visualization.py --dataset {dataset} --epochs {epochs} --seed {seed} --config {config_path}"
        command_log.append(cmd)
        try:
            from experiments.exp7_failure_visualization import run_failure_visualization
            t0 = time.time()
            run_failure_visualization(dataset=dataset, epochs=epochs, seed=seed, config_path=config_path)
            status_log["exp7_visualization"] = {"status": "success", "runtime_s": round(time.time() - t0, 1)}
        except Exception as e:
            print(f"[FAILED] Exp 7: {e}")
            traceback.print_exc()
            status_log["exp7_visualization"] = {"status": "failed", "error": str(e)}
    
    # ── Final Summary ────────────────────────────────────────────────
    total_time = time.time() - overall_start
    
    final_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_runtime_s": round(total_time, 1),
        "dataset": dataset,
        "epochs": epochs,
        "seed": seed,
        "experiments_run": experiments,
        "experiment_status": status_log,
        "command_log": command_log,
    }
    
    save_json(final_summary, EXPERIMENT_ROOT / "experiment_summary.json")
    
    # Save command log
    with open(EXPERIMENT_ROOT / "command_log.txt", "w") as f:
        f.write(f"# IRAI 2026 Reviewer Experiment Commands\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Conda env: flowmatch_pdm\n")
        f.write(f"# Working dir: FlowMatch-PdM/\n\n")
        f.write(f"conda activate flowmatch_pdm\n\n")
        for cmd in command_log:
            f.write(cmd + "\n")
        f.write(f"\n# Or run all at once:\n")
        f.write(f"python experiments/run_all_experiments.py --dataset {dataset} --epochs {epochs} --seed {seed}\n")
    
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT SUITE COMPLETE")
    print(f"# Total runtime: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"# Results: {EXPERIMENT_ROOT}")
    n_success = sum(1 for v in status_log.values() if v.get("status") == "success")
    n_total = len(status_log)
    print(f"# Status: {n_success}/{n_total} experiments succeeded")
    for name, status in status_log.items():
        icon = "✓" if status.get("status") == "success" else "✗"
        runtime = status.get("runtime_s", "N/A")
        print(f"#   {icon} {name}: {status.get('status')} ({runtime}s)")
    print(f"{'#'*70}\n")
    
    return final_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all IRAI 2026 reviewer experiments")
    parser.add_argument("--experiments", type=int, nargs="+", default=None,
                       help="Experiment numbers to run (1-7). Default: all")
    parser.add_argument("--dataset", type=str, default="FEMTO")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Training epochs per experiment (reduced for fast sweep)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    run_all(
        experiments=args.experiments,
        dataset=args.dataset,
        epochs=args.epochs,
        seed=args.seed,
        config_path=args.config,
    )
