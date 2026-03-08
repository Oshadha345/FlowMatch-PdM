# 04 — Script & Module Reference

> Internal "Technical Bible" for FlowMatch-PdM.  
> Updated: 2026-03-08

---

## Root Scripts

### `train_classifier.py`
Phase 0 baseline trainer. Instantiates the **CNN1DClassifier** (bearing fault) or **LSTMRegressor** (RUL) from `src/classifier.py`, wires up the correct `DataModule` via `data_helper.get_data_module()`, and runs a full PyTorch Lightning `fit → test` loop. All outputs (checkpoints, W&B logs, run configs) are routed through **SessionManager** into an isolated `results/<track>/<dataset>/<model>/run<N>_<ts>/` directory so every experiment is fully reproducible.

### `train_generator.py`
Phase 2 generative training. Loads the **minority / degraded subset** from the `DataModule`, instantiates one of five generative models (TimeVAE, TimeGAN, DiffusionTS, TimeFlow, or FlowMatch-PdM), and trains it exclusively on that minority data. For the FlowMatch-PdM model it additionally attaches the **LayerAdaptivePruningCallback**. Checkpoints are saved to `best_models_generator/` inside the SessionManager run directory.

### `run_evaluation.py`
Phase 2/3 evaluation. Loads a previously-trained generator from its checkpoint, generates synthetic time-series at the same sample count as the real minority set, and runs the full **TimeSeriesEvaluator** suite (FTSD, MMD, Discriminative Score, Predictive Score, PCA/t-SNE plots, KDE). Synthetic and real `.npy` arrays plus a `metrics.txt` report are persisted in `generator_datas/` and `evaluation_results/`.

### `train_classifier_aug.py`
Phase 1 (classical augmentation) **and** Phase 3 (mixed generative augmentation). Supports three modes: `--aug smote` applies SMOTE oversampling, `--aug noise` applies Gaussian jittering, and `--run_id <ID>` loads pre-generated synthetic `.npy` data and concatenates it with the real minority samples to create a balanced "Augmented Dataset." The augmented set replaces the training loader while validation and test loaders remain untouched. Uses **SessionManager** for output isolation and **WandbLogger** for experiment tracking.

---

## `flowmatchPdM/` — Core Generative Architecture

### `flowmatchPdM/flowmatch_pdm.py` — `FlowMatchPdM`
The main `LightningModule` implementing **State-Space Flow Matching**. It composes three novel sub-modules: (1) a **DynamicHarmonicPrior** that replaces standard Gaussian noise with a physics-informed oscillatory base distribution, (2) a stack of **BidirectionalMambaBlock** layers as the ODE vector-field estimator, and (3) a **TCCMManifoldLoss** penalty with $\lambda=10$ enforcing monotonic degradation. LAP forward-hooks are registered on every Mamba block to track per-channel activation norms for downstream pruning. Generation is performed by solving the learned ODE from the harmonic prior to the data manifold via Euler integration.

### `flowmatchPdM/model/mamba_backbone.py` — `BidirectionalMambaBlock`
**Bidirectional State-Space Model.** Processes the sequence both forward and backward through two independent `Mamba` SSM modules, concatenates the outputs, fuses via a learned linear gate, and adds a residual connection with LayerNorm. This captures both past-context and future-context degradation information.

### `flowmatchPdM/model/harmonic_prior.py` — `DynamicHarmonicPrior`
**Physics-Informed Base Distribution.** A small MLP estimates amplitude $A$ and frequency $f$ from operating conditions (e.g., motor load, health index), then generates $A \sin(2\pi f t + \phi) + \varepsilon$, where $\phi$ is a random phase and $\varepsilon \sim \mathcal{N}(0, 0.1)$. This injects domain knowledge about bearing vibration harmonics directly into the flow-matching sampling path.

### `flowmatchPdM/model/tccm_loss.py` — `TCCMManifoldLoss`
**Time-Conditioned Contraction Matching.** A penalty term ($\lambda=10$) that penalises positive implied health-change in the predicted vector field, enforcing the physical constraint that degradation is monotonically non-increasing. Implemented via a ReLU gate on the mean field projection.

### `flowmatchPdM/model/lap.py` — `LayerAdaptivePruningCallback`
**Layer-Adaptive Pruning** ($\alpha=0.2$, $\beta=0.1$). A Lightning Callback that monitors per-channel L1-norm activations accumulated during each epoch. Once the activation variance stabilises below a threshold it enters the "stable phase" and begins zeroing out channels whose load falls below the $\alpha$ and $\beta$ criteria, producing a sparse, efficient Mamba backbone at convergence.

---

## `src/` — Training Infrastructure

### `src/classifier.py`
Defines two Lightning classifiers. **CNN1DClassifier** is a 5-block 1D-CNN (filters 16 → 64 → 128 → 256 → 256) with BatchNorm, MaxPool, Global Average Pooling, and a 2-layer FC head — used for bearing fault classification. **LSTMRegressor** is a 2-layer LSTM (hidden=100) with a regression head — used for RUL prediction on CMAPSS/FEMTO data.

### `src/baselines.py`
Contains **ClassicalAugmenter** (SMOTE + Jittering static methods) and four deep generative baselines: **TimeVAE** (recurrent VAE with ELBO), **TimeGAN** (adversarial LSTM generator/discriminator), **DiffusionTS** (DDPM with 1D-CNN denoiser), and **TimeFlow** (standard Flow Matching with an MLP vector field — no Mamba, no physics prior).

### `src/evaluation.py`
**TimeSeriesEvaluator** class implementing five gold-standard synthetic time-series metrics: FTSD (Fréchet distance), MMD (kernel-based distribution distance), Discriminative Score (real-vs-fake classifier), Predictive Score (TSTR MAE), plus PCA/t-SNE and KDE visualisations.

### `src/utils/data_helper.py`
Factory function `get_data_module()` that routes `(track, dataset_name)` to the correct `LightningDataModule`: **FlowMatchRULDataModule** for engine/bearing RUL, **CWRUDataModule** for CWRU bearing faults, **PaderbornDataModule** for Paderborn bearing faults.

### `src/utils/logger_utils.py`
**SessionManager**: creates an isolated `results/<track>/<dataset>/<model>/run<N>_<ts>/` directory tree with sub-folders for generator checkpoints, classifier checkpoints, synthetic data, and evaluation results. Also dumps the exact YAML config for each run. **setup_wandb_logger()**: creates a `WandbLogger` pointing to the SessionManager's run directory. **JSONMetricsTracker**: Lightning Callback that serialises training/test metrics to JSON for LaTeX table generation.

---

## `datasets/` — Data Modules

### `datasets/cwru_data_loader.py`
`CWRUDataModule` — loads the pre-processed `cwru_processed.npz` (DE signals, window=2048, Z-score, 10 classes) and serves stratified train/val/test `DataLoader`s. Exposes `get_minority_dataset()` for generative training.

### `datasets/paderborn_data_loader.py`
`PaderbornDataModule` — loads `paderborn_processed.npz` (window=4096, Z-score, 32 classes). Same interface as CWRU.

### `datasets/rul_data_loader.py`
`FlowMatchRULDataModule` — wraps the `rul-datasets` library to handle CMAPSS, N-CMAPSS, FEMTO, and XJTU-SY with automatic download, windowing, normalisation, and minority extraction via `get_minority_dataset(rul_threshold_ratio)`.
