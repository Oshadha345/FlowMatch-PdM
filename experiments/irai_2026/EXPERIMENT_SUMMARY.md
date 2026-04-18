# IRAI 2026 Reviewer Experiment Suite — Results Summary

**Model**: FlowMatch-PdM (Patch-based Bidirectional Mamba + Flow Matching)  
**Primary Dataset**: FEMTO (bearing RUL, window_size=2560)  
**Training**: 30 epochs per model, seed=42  
**GPU**: NVIDIA RTX A6000  
**Total Runtime**: ~23 min (1385s for exps 1,2,4,5,6,7) + exp3 from prior run

---

## Experiment 1: Scaling Study over Sequence Length

Sweeps FlowMatch-PdM across {128, 256, 512, 1024, 2048, 2560} sequence lengths.

| Length | MMD ↓ | PSD Corr ↑ | Train (s) | Gen (s) |
|--------|-------|-----------|-----------|---------|
| 128    | 0.199 | 0.907     | 14.5      | 1.5     |
| 256    | 0.279 | 0.683     | 18.5      | 1.4     |
| 512    | 0.229 | 0.692     | 16.1      | 1.5     |
| 1024   | 0.235 | 0.928     | 14.7      | 1.7     |
| 2048   | **0.127** | 0.828 | 19.6      | 2.2     |
| 2560   | 0.305 | **0.964** | 17.5      | 2.4     |

**Key Finding**: Best MMD at L=2048; best spectral correlation at native L=2560. Model handles all lengths stably with sub-linear generation cost scaling.

---

## Experiment 2: Frequency Ablation

Tests whether FlowMatch-PdM captures high-frequency content or primarily low-frequency structure.

| Variant | MMD ↓ | PSD Corr ↑ | Real HF% | Synth HF% |
|---------|-------|-----------|----------|-----------|
| raw | 0.367 | 0.807 | 32.8% | 76.4% |
| lowpass_0.3 | 0.205 | 0.961 | 0.04% | 2.3% |
| lowpass_0.1 | **0.077** | 0.965 | 0.06% | 1.6% |
| lowpass_0.05 | 0.109 | **0.984** | 0.05% | 4.3% |
| bandpass | 0.211 | 0.106 | 0.01% | 11.7% |
| downsample_2x | 0.153 | 0.862 | 1.1% | 2.3% |
| downsample_4x | 0.191 | 0.976 | 0.2% | 2.8% |

**Key Finding**: Model performs significantly better on low-pass filtered signals (MMD drops from 0.367→0.077). Synthetic HF energy ratio exceeds real data (76.4% vs 32.8% on raw), suggesting the harmonic prior injects excess high-frequency content. Low-pass at 0.1×Nyquist gives optimal fidelity.

---

## Experiment 3: Solver Sensitivity

Trains one model, then tests 8 ODE solver configurations at inference.

| Solver | Steps | MMD ↓ | Gen Time (s) |
|--------|-------|-------|-------------|
| euler_50 | 50 | **0.403** | **1.22** |
| euler_100 | 100 | 0.406 | 2.41 |
| euler_200 | 200 | 0.408 | 4.79 |
| euler_500 | 500 | 0.409 | 12.14 |
| rk4_25 | 25 | 0.408 | 2.45 |
| rk4_50 | 50 | 0.410 | 4.89 |
| rk4_100 | 100 | 0.410 | 9.76 |
| dopri5 | adaptive | 0.410 | 3.34 |

**Key Finding**: All solvers produce nearly identical quality (MMD range: 0.403–0.410). Euler with 50 steps is Pareto-optimal: fastest (1.2s) and lowest MMD. Higher-order solvers (RK4, dopri5) offer no advantage, confirming the learned vector field is smooth enough for low-order integration.

---

## Experiment 4: Alternative Model Families

Compares FlowMatch-PdM against 4 alternative generators on FEMTO.

| Model | MMD ↓ | PSD Corr ↑ | Train (s) | Gen (s) |
|-------|-------|-----------|-----------|---------|
| **FlowMatch-PdM** | 0.215 | 0.968 | 17.5 | 2.3 |
| TimeFlow | **0.032** | 0.972 | 3.5 | 0.1 |
| TimeVAE | 0.214 | 0.972 | 35.2 | 0.01 |
| AutoregressiveLSTM | 0.181 | 0.972 | 35.7 | 3.2 |
| SpectralGenerator | 0.309 | **0.976** | 6.4 | 0.01 |

**Key Finding**: TimeFlow achieves surprisingly low MMD (0.032) on FEMTO with 30-epoch training. FlowMatch-PdM is competitive on spectral correlation (0.968) but has room for improvement on distributional metrics at reduced epochs. SpectralGenerator leads on PSD correlation by design.

---

## Experiment 5: TSTR Robustness (Multi-Seed)

Trains FlowMatch-PdM with 5 seeds {42, 123, 456, 789, 1024}, evaluates TSTR classification (4-class RUL binning).

| Metric | Mean ± Std | 95% CI |
|--------|-----------|--------|
| TSTR Accuracy | 0.270 ± 0.023 | [0.257, 0.291] |
| TSTR Balanced Acc | 0.269 ± 0.024 | [0.254, 0.290] |
| TSTR F1-macro | 0.198 ± 0.072 | [0.144, 0.255] |
| TRTR Accuracy | 0.395 ± 0.027 | [0.374, 0.415] |
| TRTR Balanced Acc | 0.396 ± 0.027 | [0.376, 0.417] |
| Relative F1 (TSTR/TRTR) | 0.523 ± 0.217 | [0.360, 0.701] |
| Gate Pass Rate | 0% (threshold: 0.8) |

**Key Finding**: TSTR accuracy is consistently above random (0.25 for 4 classes) across all seeds but does not reach the 80% gate threshold relative to TRTR. Variance is moderate (std ~0.023). The relative F1 of ~52% indicates synthetic data captures roughly half the class-discriminative information.

---

## Experiment 6: Cross-Dataset Validation

Trains and evaluates FlowMatch-PdM on both FEMTO and XJTU-SY.

| Dataset | MMD ↓ | PSD Corr ↑ | Train (s) | Gen (s) |
|---------|-------|-----------|-----------|---------|
| FEMTO (ws=2560) | 0.238 | **0.962** | 19.1 | 2.4 |
| XJTU-SY (ws=2048) | **0.049** | 0.299 | 18.2 | 2.0 |

**Key Finding**: FlowMatch-PdM achieves much lower MMD on XJTU-SY (0.049 vs 0.238) but significantly worse spectral correlation (0.299 vs 0.962). This suggests the model adapts differently to each dataset's spectral characteristics — XJTU-SY has better distributional match but poorer frequency fidelity.

---

## Experiment 7: Failure Visualization

Generates diagnostic plots for qualitative analysis:
- **Waveform comparison**: Side-by-side real vs synthetic time-series (4 examples)
- **Spectrum comparison**: Power spectrum overlay, spectral difference, cumulative energy
- **Phase drift analysis**: Cross-correlation lag, autocorrelation, RMS envelope

All figures saved as PNG + PDF in `figures/`.

---

## Output Artifacts

| Type | Path | Count |
|------|------|-------|
| Figures (PNG+PDF) | `figures/` | 24 files (12 pairs) |
| Tables (CSV+JSON) | `tables/` | 14 files (7 pairs) |
| TSTR per-seed | `tables/tstr_seed_*/` | 10 files |
| Logs | `logs/` | 3 files |
| Summary | `experiment_summary.json` | 1 file |
| Command log | `command_log.txt` | 1 file |

---

## Reproducibility

```bash
# Full suite (skip exp3 if solver results already exist)
conda activate flowmatch_pdm
cd FlowMatch-PdM
python experiments/run_all_experiments.py --dataset FEMTO --epochs 30 --seed 42

# Individual experiments
python experiments/exp1_scaling_study.py --dataset FEMTO --epochs 30
python experiments/exp2_frequency_ablation.py --dataset FEMTO --epochs 30
python experiments/exp3_solver_sensitivity.py --dataset FEMTO --epochs 30
python experiments/exp4_alternative_baselines.py --dataset FEMTO --epochs 30
python experiments/exp5_tstr_robustness.py --dataset FEMTO --gen_epochs 30
python experiments/exp6_cross_dataset.py --epochs 30
python experiments/exp7_failure_visualization.py --dataset FEMTO --epochs 30
```
