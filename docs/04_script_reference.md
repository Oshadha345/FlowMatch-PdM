# 04 - Script Reference

This file is the CLI and module reference for the current repository state.

Phase ownership:
- Phase 0: `train_classifier.py`
- Phase 1: `train_classifier_aug.py --aug ...`
- Phase 2: `train_generator.py` -> `run_evaluation.py` -> `train_classifier_aug.py --gen_model ...`
- Phase 3: same loop on the secondary datasets
- Phase 5: `run_all.sh` proof-of-concept rerun

## 1. Top-Level Scripts

## `train_classifier.py`

Purpose:
- Phase 0 baseline training and test

Exact CLI:

```bash
python train_classifier.py \
  --track <engine_rul|bearing_rul|bearing_fault> \
  --dataset <DATASET> \
  --model <baseline|LSTMRegressor|CNN1DClassifier> \
  [--run_id <CUSTOM_RUN_ID>] \
  [--config configs/default_config.yaml]
```

Notes:
- `baseline` resolves to `LSTMRegressor` for RUL tracks
- `baseline` resolves to `CNN1DClassifier` for `bearing_fault`
- writes to `results/<track>/<dataset>/<resolved_model>/run_<timestamp>/`

## `train_generator.py`

Purpose:
- Phase 2 minority-subset generative training

Exact CLI:

```bash
python train_generator.py \
  --track <engine_rul|bearing_rul|bearing_fault> \
  --dataset <DATASET> \
  --model <TimeVAE|TimeGAN|DiffusionTS|TimeFlow|COTGAN|FaultDiffusion|FlowMatch> \
  [--run_id <CUSTOM_RUN_ID>] \
  [--config configs/default_config.yaml]
```

Notes:
- uses `get_minority_dataset()` from the dataset module
- saves checkpoints into `best_models_generator/`
- `FlowMatch` is the CLI name for FlowMatch-PdM
- `FlowMatch` additionally attaches `LayerAdaptivePruningCallback`
- the FlowMatch-PdM config block is `generative.flowmatch_pdm`

## `run_evaluation.py`

Purpose:
- Generate synthetic arrays from a trained generator run
- evaluate them against real minority data
- persist metrics and plots
- this is the evaluation stage used inside Phase 2 and Phase 3

Exact CLI:

```bash
python run_evaluation.py \
  --track <engine_rul|bearing_rul|bearing_fault> \
  --dataset <DATASET> \
  --model <TimeVAE|TimeGAN|DiffusionTS|TimeFlow|COTGAN|FaultDiffusion|FlowMatch> \
  [--run_id <GENERATOR_RUN_ID>] \
  [--config configs/default_config.yaml]
```

Alias:

```bash
python run_evaluation.py --track ... --dataset ... --gen_model FlowMatch --run_id ...
```

Notes:
- if `--run_id` is omitted, the latest generator run is used
- the generator checkpoint is loaded from the same run directory
- the Phase 0 baseline classifier/regressor for the same dataset is auto-resolved and used as the FTSD feature extractor
- outputs are written into the generator run itself

Exact `run_id` lookup example:

```bash
python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch').name)"
```

## `train_classifier_aug.py`

Purpose:
- Phase 1 classical augmentation
- Phase 2 and Phase 3 synthetic-data augmentation

Exact CLI:

```bash
python train_classifier_aug.py \
  --track <engine_rul|bearing_rul|bearing_fault> \
  --dataset <DATASET> \
  --model <baseline|LSTMRegressor|CNN1DClassifier> \
  [--aug <noise|smote>] \
  [--gen_model <TimeVAE|TimeGAN|DiffusionTS|TimeFlow|COTGAN|FaultDiffusion|FlowMatch>] \
  [--run_id <GENERATOR_RUN_ID>] \
  [--config configs/default_config.yaml]
```

Mode behavior:
- `--aug noise`: duplicates the training set with Gaussian jittering
- `--aug smote`: classification only
- `--gen_model ... --run_id ...`: loads `synthetic_data.npy` and `synthetic_targets.npy` from the generator run and concatenates them with the real training set

Notes:
- validation and test always use the original datamodule splits
- synthetic mode reads from `results/<track>/<dataset>/<gen_model>/<run_id>/generator_datas/`

## `run_all.sh`

Purpose:
- proof-of-concept full pipeline for `CMAPSS`

Exact usage:

```bash
./run_all.sh
```

Environment overrides:

```bash
PYTHON_BIN=/home/buddhiw/miniconda3/envs/flowmatch_pdm/bin/python ./run_all.sh
GEN_MODELS="FlowMatch COTGAN FaultDiffusion" ./run_all.sh
DATASET=CMAPSS TRACK=engine_rul ./run_all.sh
```

Behavior:
- runs Phase 0 baseline once
- trains each requested generator
- resolves the latest generator `run_id`
- runs evaluation
- runs synthetic-augmented classifier retraining
- by default it is a FlowMatch-PdM CMAPSS proof-of-concept

## 2. Utility Modules

## `src/utils/logger_utils.py`

Key APIs:
- `SessionManager(...)`
- `SessionManager.from_existing(...)`
- `resolve_model_root(...)`
- `resolve_run_dir(...)`
- `resolve_checkpoint(...)`
- `setup_wandb_logger(...)`
- `JSONMetricsTracker(...)`

Important current behavior:
- run folders are `run_<timestamp>`
- CSV logging is always on
- W&B logging is optional via `logging.use_wandb`

## `src/utils/data_helper.py`

Key APIs:
- `canonicalize_dataset_name(...)`
- `get_dataset_config(...)`
- `get_data_module(...)`

Supported datasets:
- `CMAPSS`
- `N-CMAPSS`
- `FEMTO`
- `XJTU-SY`
- `CWRU`
- `Paderborn`
- `DEMADICS`

## 3. Dataset Modules

## `datasets/rul_data_loader.py`

Supported:
- `CMAPSS`
- `N-CMAPSS`
- `FEMTO`
- `XJTU-SY`

Current guarantees:
- windows returned as `[batch, window, features]`
- `setup("fit")` maps `dev -> train` and `val -> validation`
- minority extraction uses `target <= 0.2 * max_rul`

## `datasets/cwru_data_loader.py`

Current contract:
- loads `datasets/processed/cwru/*.npy`
- windows are `(2048, 1)`
- labels are `torch.int64`
- minority extraction is smallest train class

## `datasets/paderborn_data_loader.py`

Current contract:
- loads `datasets/processed/paderborn/*.npy`
- windows are `(4096, 1)`
- labels are `torch.int64`
- minority extraction is smallest train class

## `datasets/demadics_data_loader.py`

Current contract:
- loads `datasets/processed/demadics/*.npy`
- windows are `(2048, 32)`
- labels are `torch.int64`
- minority extraction is smallest train class

## 4. Model Modules

## `src/classifier.py`

Contains:
- `LSTMRegressor`
- `CNN1DClassifier`

Important current behavior:
- `CNN1DClassifier` now handles multi-channel inputs correctly
- both models expose a valid forward path for the top-level scripts
- both models are loadable from saved `.ckpt` files
- both models expose feature extraction needed by evaluation, directly or indirectly

## `src/baselines.py`

Contains:
- `ClassicalAugmenter`
- `TimeVAE`
- `TimeGAN`
- `DiffusionTS`
- `FaultDiffusion`
- `TimeFlow`
- `COTGAN`

Important current behavior:
- every generator exposes `generate(...)`
- `TimeGAN` and `COTGAN` use manual optimization
- `FaultDiffusion` is the sequence-diffusion baseline used for long bearing windows

## `src/evaluation.py`

Contains:
- `TimeSeriesEvaluator`
- `RealSyntheticGRU`
- `NextStepGRU`

Current outputs:
- `metrics.txt`
- `metrics.json`
- `projection_pca_tsne.png`
- `marginal_kde.png`

## 5. Notebook Reference

## `notebooks/01_dataset_analysis.ipynb`

Purpose:
- rebuild DEMADICS processed arrays if needed
- verify one training batch from every supported dataset
- validate dtype, tensor shape, and minority subset size

Run it before large training cycles when dataset preprocessing changes.
