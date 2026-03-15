# 📊 Experiment Results Ledger

Log only successful runs here. This file is the single source of truth for the paper tables, comparison charts, and final model ranking. Never overwrite a score without confirming the exact `run_id`.

---

## Logging Rules

- Write the metric exactly as it appears in the terminal or JSON artifact.
- Record the matching `run_id` for every row.
- For Phase 2 and Phase 3 generator work, record both the generator `run_id` and the downstream augmented-classifier `run_id` in the notes column when they differ.
- Freeze the Top 1, Top 2, and Top 3 ranking only after Table 3 is complete.

---

## 🟢 Table 1: Phase 0 - Pure Baselines

*Source: terminal output and `evaluation_results/phase0_metrics.json` from `train_classifier.py`.*
*Metric rule: RMSE for RUL tracks. F1-score for classification tracks.*

| Dataset | Track | Metric | Baseline Score | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **CMAPSS** | Engine RUL | RMSE (↓) | - | - |
| **N-CMAPSS** | Engine RUL | RMSE (↓) | - | - |
| **FEMTO** | Bearing RUL | RMSE (↓) | - | - |
| **XJTU-SY** | Bearing RUL | RMSE (↓) | - | - |
| **CWRU** | Fault Classification | F1-Score (↑) | - | - |
| **DEMADICS** | Fault Classification | F1-Score (↑) | - | - |
| **Paderborn** | Fault Classification | F1-Score (↑) | - | - |

---

## 🔵 Table 2: Phase 2 - Generative Fidelity On Primary Datasets

*Source: `evaluation_results/metrics.txt` and `evaluation_results/metrics.json` from `run_evaluation.py`.*
*Lower is better for FTSD, MMD, Discriminative Score, and TSTR MAE.*

### CMAPSS

| Model | FTSD (↓) | MMD (↓) | Disc. Score (↓) | TSTR MAE (↓) | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TimeVAE | - | - | - | - | - |
| TimeGAN | - | - | - | - | - |
| COTGAN | - | - | - | - | - |
| FaultDiffusion | - | - | - | - | - |
| DiffusionTS | - | - | - | - | - |
| TimeFlow | - | - | - | - | - |
| **FlowMatch-PdM (CLI: `FlowMatch`)** | **-** | **-** | **-** | **-** | **-** |

### CWRU

| Model | FTSD (↓) | MMD (↓) | Disc. Score (↓) | TSTR MAE (↓) | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TimeVAE | - | - | - | - | - |
| TimeGAN | - | - | - | - | - |
| COTGAN | - | - | - | - | - |
| FaultDiffusion | - | - | - | - | - |
| DiffusionTS | - | - | - | - | - |
| TimeFlow | - | - | - | - | - |
| **FlowMatch-PdM (CLI: `FlowMatch`)** | **-** | **-** | **-** | **-** | **-** |

### DEMADICS

| Model | FTSD (↓) | MMD (↓) | Disc. Score (↓) | TSTR MAE (↓) | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TimeVAE | - | - | - | - | - |
| TimeGAN | - | - | - | - | - |
| COTGAN | - | - | - | - | - |
| FaultDiffusion | - | - | - | - | - |
| DiffusionTS | - | - | - | - | - |
| TimeFlow | - | - | - | - | - |
| **FlowMatch-PdM (CLI: `FlowMatch`)** | **-** | **-** | **-** | **-** | **-** |

---

## 🟣 Table 3: Phase 1 And Phase 2 - Downstream Utility

*Source: terminal output and `evaluation_results/phase3_metrics.json` from `train_classifier_aug.py`.*
*Question: how much does classical or synthetic augmentation improve the Phase 0 baseline?*

| Augmentation Method | CMAPSS (RMSE ↓) | CWRU (F1-Score ↑) | DEMADICS (F1-Score ↑) | Run ID / Notes |
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

## 🟠 Table 4: Phase 3 - Scalability And Generalization

*Source: terminal output and `evaluation_results/phase3_metrics.json` from `train_classifier_aug.py` on the secondary datasets.*
*Use this table to freeze the final Top 1, Top 2, and Top 3 ranking from Table 3.*

| Dataset (Metric) | Phase 0 Baseline | Top 1 Model / Score | Top 2 Model / Score | Top 3 Model / Score | Run ID / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **N-CMAPSS** (RMSE ↓) | - | - | - | - | - |
| **FEMTO** (RMSE ↓) | - | - | - | - | - |
| **XJTU-SY** (RMSE ↓) | - | - | - | - | - |
| **Paderborn** (F1-Score ↑) | - | - | - | - | - |

---

## 🔴 Table 5: Phase 5 - Final CMAPSS Proof-Of-Concept

*Source: the final rerun after the ledger is frozen.*

| Item | Final Value | Run ID / Notes |
| :--- | :--- | :--- |
| Phase 0 baseline score | - | - |
| Final generator model | FlowMatch-PdM | - |
| Generator evaluation FTSD | - | - |
| Generator evaluation MMD | - | - |
| Generator evaluation Disc. Score | - | - |
| Generator evaluation TSTR MAE | - | - |
| Final augmented classifier score | - | - |
| Final proof-of-concept status | - | - |

---

## Final Freeze Checklist

- [ ] Table 1 is complete
- [ ] Table 2 is complete for `CMAPSS`, `CWRU`, and `DEMADICS`
- [ ] Table 3 is complete and the winner ranking is frozen
- [ ] Table 4 is complete for all secondary datasets
- [ ] Table 5 contains the final CMAPSS rerun
- [ ] Every populated row includes the exact `run_id`
