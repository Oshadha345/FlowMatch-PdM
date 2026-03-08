# 📊 Experiment Results Ledger

*Log all successful runs here to maintain a single source of truth for the final manuscript.*

## Table 1: Generative Fidelity (Phase 2)
*Metrics: Fréchet Time Series Distance (FTSD) and Maximum Mean Discrepancy (MMD). Lower is better.*

| Dataset | Model | FTSD (↓) | MMD (↓) |
| :--- | :--- | :--- | :--- |
| **CMAPSS** | TimeVAE | - | - |
| | TimeGAN | - | - |
| | Diffusion-TS | - | - |
| | TimeFlow | - | - |
| | **FlowMatch-PdM (Ours)** | **-** | **-** |
| **CWRU** | TimeGAN | - | - |
| | Diffusion-TS | - | - |
| | **FlowMatch-PdM (Ours)** | **-** | **-** |

---

## Table 2: Discriminative & Predictive Scores (TSTR)
*Metrics: Disc. Score ($|0.5 - \text{Acc}|$, closer to 0 is better). TSTR (Predictive MAE, lower is better).*

| Dataset | Model | Discriminative Score (↓) | Predictive Score / TSTR (↓) |
| :--- | :--- | :--- | :--- |
| **CMAPSS** | TimeGAN | - | - |
| | Diffusion-TS | - | - |
| | **FlowMatch-PdM** | **-** | **-** |

---

## Table 3: Downstream Task Performance (Phase 1 vs Phase 3)
*Does adding synthetic data actually help the classifier? Metric: RMSE for Engine RUL, Accuracy/F1 for Bearing Fault.*

| Dataset (Task) | Baseline (No Aug) | + SMOTE | + TimeGAN | + Diffusion-TS | + FlowMatch-PdM (Ours) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CMAPSS (RMSE)** | - | - | - | - | **-** |
| **N-CMAPSS (RMSE)** | - | N/A | N/A | - | **-** |
| **FEMTO (RMSE)** | - | N/A | N/A | - | **-** |
| **XJTU-SY (RMSE)** | - | N/A | N/A | - | **-** |
| **CWRU (F1-Score)** | - | - | - | - | **-** |
| **Paderborn (F1-Score)** | - | N/A | N/A | - | **-** |

*(Note: N/A indicates datasets where we only ran the top-3 models to save compute time).*