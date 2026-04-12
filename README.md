[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# FlowMatch-PdM

**State-Space Flow Matching with Dynamic Harmonic Priors for Physics-Aware Synthetic Fault Generation in Predictive Maintenance.**

FlowMatch-PdM is a conditional flow matching framework that combines a bidirectional Mamba backbone, physics-informed harmonic priors, a Temporal-Condition Consistency Manifold (TCCM) loss, and Layer-Adaptive Pruning (LAP) to generate high-fidelity synthetic degradation time series for predictive maintenance data augmentation.

## Architecture

The model integrates four key components:

1. **Dynamic Harmonic Prior** — replaces Gaussian noise with a condition-driven harmonic base distribution
2. **Bidirectional Mamba Backbone** — state-space model for efficient sequence-level vector field estimation
3. **TCCM Loss** — enforces temporal-condition consistency on the learned vector field
4. **Layer-Adaptive Pruning** — dynamically prunes Mamba blocks based on activation statistics

See [docs/02_architecture_and_math.md](docs/02_architecture_and_math.md) for the full mathematical formulation.

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd FlowMatch-PdM
conda env create -f environment.yml
conda activate flowmatch_pdm
pip install -r requirements.txt

# 2. Run the full automated pipeline
bash launch.sh

# 3. Monitor progress in another terminal
python scripts/check_results.py --watch

# 4. View results after completion
cat results/final_report/ranking_summary.md
```

## Results

Results are populated automatically by the orchestrator into `docs/03_result_logger.md` and `results/final_report/`.

### Phase 0 Baselines (Pre-computed)

| Dataset | Track | Metric | Score |
|---------|-------|--------|-------|
| CMAPSS | Engine RUL | RMSE ↓ | 16.52 |
| CWRU | Fault Classification | F1 Macro ↑ | 1.000 |
| DEMADICS | Fault Classification | F1 Macro ↑ | 0.967 |
| Paderborn | Fault Classification | F1 Macro ↑ | 0.999 |

Full comparison tables (including all 7 generators × 3 primary datasets) are generated after training completes.

## Project Structure

```
FlowMatch-PdM/
├── orchestrate.py              # Master automation pipeline
├── launch.sh                   # tmux launcher
├── pipeline_state.json         # Resumable state (single source of truth)
├── train_classifier.py         # Train baseline / augmented classifiers
├── train_generator.py          # Train generative models
├── run_evaluation.py           # Re-evaluate existing runs
├── configs/
│   ├── default_config.yaml     # All model hyperparameters
│   └── sweep_flowmatch_cmapss.yaml  # W&B Bayesian sweep config
├── flowmatchPdM/               # FlowMatch-PdM model implementation
│   ├── flowmatch_pdm.py        # Main LightningModule
│   └── model/                  # Mamba backbone, harmonic prior, TCCM, LAP
├── src/
│   ├── baselines.py            # 6 baseline generators
│   ├── classifier.py           # CNN1DClassifier, LSTMRegressor
│   └── evaluation.py           # FTSD, MMD, discriminative/predictive scores
├── scripts/
│   ├── check_results.py        # Live progress dashboard
│   └── resume_from.py          # Reset failed steps for re-execution
├── docs/
│   ├── 01_experiment_plan.md
│   ├── 02_architecture_and_math.md
│   ├── 03_result_logger.md     # Auto-populated results ledger
│   ├── HOW_TO_TRAIN.md         # Manual training guide
│   └── dataset_acquisition.md
└── results/                    # All training outputs (gitignored)
```

## Datasets

| Dataset | Track | Source | Acquisition |
|---------|-------|--------|-------------|
| C-MAPSS | Engine RUL | NASA | Auto-download via `rul-datasets` |
| N-CMAPSS | Engine RUL | NASA | Auto-download via `rul-datasets` |
| FEMTO | Bearing RUL | IEEE PHM 2012 | Auto-download via `rul-datasets` |
| XJTU-SY | Bearing RUL | Xi'an Jiaotong Univ. | Auto-download via `rul-datasets` |
| CWRU | Bearing Fault | Case Western Reserve | Preprocessed `.npy` files |
| DEMADICS | Bearing Fault | Lublin Univ. | Preprocessed `.npy` files |
| Paderborn | Bearing Fault | Paderborn Univ. | Preprocessed `.npy` files |

See [docs/dataset_acquisition.md](docs/dataset_acquisition.md) for detailed instructions.

## Hardware Requirements

- **Minimum:** 1× NVIDIA GPU with 24 GB VRAM (e.g., RTX 3090)
- **Recommended:** 1× NVIDIA GPU with 48 GB VRAM (e.g., RTX A6000)
- **Storage:** ~50 GB for all datasets + results
- **RAM:** 32 GB system memory
- **Tested on:** 2× NVIDIA RTX A6000 (48 GB), Ubuntu 22.04, CUDA 12.1



## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
