# 📊 Experiment Results Ledger

Log only successful runs here. This file is the single source of truth for baseline tables, generator fidelity tables, augmented-classifier results, ablations, and the final sweep winner.

---

## Logging Rules

- Record the exact `run_id` for every populated row.
- For generator rows, log the generator `run_id`.
- For augmented-classifier rows, log both the generator source `run_id` and the classifier `run_id` in the notes column when they differ.
- Treat `evaluation_results/classifier_metrics.json` as the authoritative classifier artifact.
- Treat `evaluation_results/metrics.json` as the authoritative generator artifact.

---

## 🟢 Table 1: Phase 0 - Baseline Classifiers

*Source: `evaluation_results/classifier_metrics.json` from baseline `train_classifier.py` runs.*

| Dataset | Track | Primary Metric | Score | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **CMAPSS** | Engine RUL | RMSE (↓) |  |  |
| **N-CMAPSS** | Engine RUL | RMSE (↓) | - | - |
| **FEMTO** | Bearing RUL | RMSE (↓) | - | - |
| **XJTU-SY** | Bearing RUL | RMSE (↓) | - | - |
| **CWRU** | Fault Classification | F1 Macro (↑) | 1 | run_20260316_111110 |
| **DEMADICS** | Fault Classification | F1 Macro (↑) | 0.9667774086378739 | run_20260316_112649 |
| **Paderborn** | Fault Classification | F1 Macro (↑) | - | - |

---

## 🔵 Table 2: Phase 2 - Generator Fidelity On Primary Datasets

*Source: `evaluation_results/metrics.json` from generator runs.*

### CMAPSS

| Model | FTSD (↓) | MMD (↓) | Disc. Score (↓) | TSTR MAE (↓) | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TimeVAE | - | - | - | - | - |
| TimeGAN | - | - | - | - | - |
| COTGAN | - | - | - | - | - |
| FaultDiffusion | - | - | - | - | - |
| DiffusionTS | - | - | - | - | - |
| TimeFlow | - | - | - | - | - |
| **FlowMatch-PdM** | **-** | **-** | **-** | **-** | **-** |

### CWRU

| Model | FTSD (↓) | MMD (↓) | Disc. Score (↓) | TSTR MAE (↓) | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TimeVAE | - | - | - | - | - |
| TimeGAN | - | - | - | - | - |
| COTGAN | - | - | - | - | - |
| FaultDiffusion | - | - | - | - | - |
| DiffusionTS | - | - | - | - | - |
| TimeFlow | - | - | - | - | - |
| **FlowMatch-PdM** | **-** | **-** | **-** | **-** | **-** |

### DEMADICS

| Model | FTSD (↓) | MMD (↓) | Disc. Score (↓) | TSTR MAE (↓) | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TimeVAE | - | - | - | - | - |
| TimeGAN | - | - | - | - | - |
| COTGAN | - | - | - | - | - |
| FaultDiffusion | - | - | - | - | - |
| DiffusionTS | - | - | - | - | - |
| TimeFlow | - | - | - | - | - |
| **FlowMatch-PdM** | **-** | **-** | **-** | **-** | **-** |

---

## 🟣 Table 3: Phase 1 And Phase 2 - Downstream Utility

*Source: `evaluation_results/classifier_metrics.json` from augmented classifier runs.*

| Augmentation Method | CMAPSS (RMSE ↓) | CWRU (F1 Macro ↑) | DEMADICS (F1 Macro ↑) | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **None (Phase 0 Baseline)** | - | - | - | - |
| + Noise | - | - | - | - |
| + SMOTE | - | - | - | - |
| + TimeVAE | - | - | - | - |
| + TimeGAN | - | - | - | - |
| + COTGAN | - | - | - | - |
| + FaultDiffusion | - | - | - | - |
| + DiffusionTS | - | - | - | - |
| + TimeFlow | - | - | - | - |
| **+ FlowMatch-PdM** | **-** | **-** | **-** | **-** |

---

## 🟠 Table 4: Phase 3 - Secondary-Dataset Generalization

*Source: `evaluation_results/classifier_metrics.json` from secondary-dataset augmented classifier runs.*

| Dataset (Metric) | Baseline | Top 1 Model / Score | Top 2 Model / Score | Top 3 Model / Score | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **N-CMAPSS** (RMSE ↓) | - | - | - | - | - |
| **FEMTO** (RMSE ↓) | - | - | - | - | - |
| **XJTU-SY** (RMSE ↓) | - | - | - | - | - |
| **Paderborn** (F1 Macro ↑) | - | - | - | - | - |

---

## 🔴 Table 5: Phase 4 - FlowMatch-PdM Ablations On CMAPSS

*Source: generator `evaluation_results/metrics.json` plus augmented-classifier `evaluation_results/classifier_metrics.json`.*

| Variant | FTSD (↓) | MMD (↓) | TSTR MAE (↓) | Downstream RMSE (↓) | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full FlowMatch-PdM** | - | - | - | - | - |
| No Prior | - | - | - | - | - |
| No TCCM | - | - | - | - | - |
| No LAP | - | - | - | - | - |

---

## 🏁 Table 6: Phase 5 - W&B Sweep And Final CMAPSS Run

| Item | Value | Run ID / Notes |
| :--- | :--- | :--- |
| Sweep config path | `configs/sweep_flowmatch_cmapss.yaml` | - |
| Sweep winner model | FlowMatch-PdM | - |
| Sweep winner FTSD | - | - |
| Sweep winner TSTR MAE | - | - |
| Final proof-of-concept run status | - | - |
| Final proof-of-concept generator run | - | - |
| Final proof-of-concept classifier run | - | - |

---

## Final Freeze Checklist

- [ ] Table 1 is complete
- [ ] Table 2 is complete
- [ ] Table 3 is complete and the Top 1 / Top 2 / Top 3 order is frozen
- [ ] Table 4 is complete
- [ ] Table 5 is complete
- [ ] Table 6 is complete
- [ ] Every populated row includes the exact `run_id`
