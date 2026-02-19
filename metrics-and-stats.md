# 📊 Metrics & Statistics - Member 2

## 🎯 Primary Objective

Build the universal testing suite that will evaluate how realistic the synthetic data is, and implement the classical/statistical augmentation baselines.

---

## 📋 Specific Tasks

### 📝 Core Implementation
- Write the comprehensive `evaluation.py` script with:
  - TSTR pipeline
  - Context-FID mathematical calculation
  - t-SNE/PCA plotting functions
- Implement statistical and time-domain baselines: **SMOTE** and **Jittering/Time-Warping**

### ⚠️ Note
> This evaluation scripts will be the central tool everyone else uses to test their generative models.

---

## 🔧 Technical Requirements

### Classical Baselines
- Implement SMOTE (adapted for flattened time-series)
- Implement Time-Domain Augmentations (Jittering, Time-Warping)
- Generate synthetic data using these methods

### TSTR Pipeline
- Write the "Train on Synthetic, Test on Real" pipeline
- Ingest synthetic dataset
- Train Member 1's classifiers on synthetic data
- Output metrics when tested on the hidden real dataset

### Visual Metrics
- Implement functions to flatten sequences
- Generate t-SNE and PCA scatter plots
- Compare real vs. synthetic distributions

### Mathematical Metrics
- Implement Context-FID (FTSD) calculation
- Measure mathematical distance between real and fake feature distributions

---

## 📦 Key Deliverables

| File | Components |
|------|------------|
| `src/baselines.py` | `SMOTE_augmenter`, `Jitter_Warp_augmenter` |
| `src/evaluation.py` | `run_tstr()`, `plot_tsne()`, `calculate_ftsd()` |

---

## 🔗 Dependencies

- ✅ Requires `src/classifiers.py` from Member 1 for the TSTR pipeline
- ⚠️ **Critical:** The rest of the team depends on your `evaluation.py` to test their generative models