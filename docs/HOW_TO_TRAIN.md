# FlowMatch-PdM: Complete Training Guide

Copy-paste-ready commands to run the full experiment from scratch.

---

## 0. Prerequisites

### Environment Setup

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate flowmatch_pdm

# 2. Install special packages (mamba-ssm, causal-conv1d, rul-datasets)
pip install -r requirements.txt

# If mamba-ssm / causal-conv1d fail to build, install from the local wheels:
pip install whl/causal_conv1d-1.5.0.post8+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install whl/mamba_ssm-2.2.4+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### GPU Check

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### Dataset Check

```bash
# RUL datasets (auto-download via rul-datasets)
python -c "
import rul_datasets
for name, cls in [('CMAPSS', rul_datasets.CmapssReader)]:
    r = cls(fd=1); r.prepare_data(); print(f'{name}: OK')
"

# Classification datasets (preprocessed .npy files)
python -c "
from pathlib import Path
for ds in ['cwru', 'paderborn', 'demadics']:
    p = Path('datasets/processed') / ds / 'X_train.npy'
    print(f'{ds}: {\"OK\" if p.exists() else \"MISSING\"}')"
```

### Preflight Notebook

```bash
python -m jupyter nbconvert --to notebook --execute \
  notebooks/01_dataset_analysis.ipynb \
  --output 01_dataset_analysis.executed.ipynb \
  --output-dir notebooks \
  --ExecutePreprocessor.timeout=0 \
  --ExecutePreprocessor.kernel_name=python3
```

Confirm the executed notebook ends with:
- `Supported loader readiness: GO`
- `Full requested roster readiness: GO`

---

## 1. Phase 0 — Baseline Classifiers

Each command trains a baseline classifier and auto-evaluates it.
Outputs land in `results/<track>/<dataset>/<classifier>/run_<timestamp>/`.

### Already Completed

| Dataset | Track | Score | Run ID |
|---------|-------|-------|--------|
| CMAPSS | engine_rul | RMSE 16.52 | run_20260316_125116 |
| CWRU | bearing_fault | F1 1.0 | run_20260316_111110 |
| DEMADICS | bearing_fault | F1 0.967 | run_20260316_112649 |
| Paderborn | bearing_fault | F1 0.999 | run_20260316_154854 |

### Remaining

```bash
python train_classifier.py --track engine_rul --dataset N-CMAPSS --model baseline
python train_classifier.py --track bearing_rul --dataset FEMTO --model baseline
python train_classifier.py --track bearing_rul --dataset XJTU-SY --model baseline
```

---

## 2. Phase 1 — Classical Augmentation

### 1A. Noise / Jittering

```bash
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --aug noise
python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --aug noise
python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --aug noise
```

### 1B. SMOTE (classification tracks only)

```bash
python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --aug smote
python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --aug smote
python train_classifier.py --track bearing_fault --dataset Paderborn --model baseline --aug smote
```

---

## 3. Phase 2 — Generator Training and Augmentation

For each generator × dataset combination: (1) train the generator, (2) capture its run ID,
(3) train the augmented classifier using synthetic data from that generator.

### Helper: Capture RUN_ID

After each `train_generator.py` call, capture the latest run ID:

```bash
RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('TRACK', 'DATASET', 'MODEL').name)")"
echo "$RUN_ID"
```

### 2A. TimeVAE

```bash
# CMAPSS
python train_generator.py --track engine_rul --dataset CMAPSS --model TimeVAE
CMAPSS_TIMEVAE_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'TimeVAE').name)")"
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model TimeVAE --source_run_id "$CMAPSS_TIMEVAE_RUN_ID"

# CWRU
python train_generator.py --track bearing_fault --dataset CWRU --model TimeVAE
CWRU_TIMEVAE_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'TimeVAE').name)")"
python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model TimeVAE --source_run_id "$CWRU_TIMEVAE_RUN_ID"

# DEMADICS
python train_generator.py --track bearing_fault --dataset DEMADICS --model TimeVAE
DEMADICS_TIMEVAE_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'TimeVAE').name)")"
python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model TimeVAE --source_run_id "$DEMADICS_TIMEVAE_RUN_ID"
```

### 2B. TimeGAN

```bash
# CMAPSS
python train_generator.py --track engine_rul --dataset CMAPSS --model TimeGAN
CMAPSS_TIMEGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'TimeGAN').name)")"
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model TimeGAN --source_run_id "$CMAPSS_TIMEGAN_RUN_ID"

# CWRU
python train_generator.py --track bearing_fault --dataset CWRU --model TimeGAN
CWRU_TIMEGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'TimeGAN').name)")"
python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model TimeGAN --source_run_id "$CWRU_TIMEGAN_RUN_ID"

# DEMADICS
python train_generator.py --track bearing_fault --dataset DEMADICS --model TimeGAN
DEMADICS_TIMEGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'TimeGAN').name)")"
python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model TimeGAN --source_run_id "$DEMADICS_TIMEGAN_RUN_ID"
```

### 2C. COTGAN

```bash
# CMAPSS
python train_generator.py --track engine_rul --dataset CMAPSS --model COTGAN
CMAPSS_COTGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'COTGAN').name)")"
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model COTGAN --source_run_id "$CMAPSS_COTGAN_RUN_ID"

# CWRU
python train_generator.py --track bearing_fault --dataset CWRU --model COTGAN
CWRU_COTGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'COTGAN').name)")"
python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model COTGAN --source_run_id "$CWRU_COTGAN_RUN_ID"

# DEMADICS
python train_generator.py --track bearing_fault --dataset DEMADICS --model COTGAN
DEMADICS_COTGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'COTGAN').name)")"
python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model COTGAN --source_run_id "$DEMADICS_COTGAN_RUN_ID"
```

### 2D. FaultDiffusion

```bash
# CMAPSS
python train_generator.py --track engine_rul --dataset CMAPSS --model FaultDiffusion
CMAPSS_FAULTDIFFUSION_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FaultDiffusion').name)")"
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FaultDiffusion --source_run_id "$CMAPSS_FAULTDIFFUSION_RUN_ID"

# CWRU
python train_generator.py --track bearing_fault --dataset CWRU --model FaultDiffusion
CWRU_FAULTDIFFUSION_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'FaultDiffusion').name)")"
python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model FaultDiffusion --source_run_id "$CWRU_FAULTDIFFUSION_RUN_ID"

# DEMADICS
python train_generator.py --track bearing_fault --dataset DEMADICS --model FaultDiffusion
DEMADICS_FAULTDIFFUSION_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'FaultDiffusion').name)")"
python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model FaultDiffusion --source_run_id "$DEMADICS_FAULTDIFFUSION_RUN_ID"
```

### 2E. DiffusionTS

```bash
# CMAPSS
python train_generator.py --track engine_rul --dataset CMAPSS --model DiffusionTS
CMAPSS_DIFFUSIONTS_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'DiffusionTS').name)")"
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model DiffusionTS --source_run_id "$CMAPSS_DIFFUSIONTS_RUN_ID"

# CWRU
python train_generator.py --track bearing_fault --dataset CWRU --model DiffusionTS
CWRU_DIFFUSIONTS_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'DiffusionTS').name)")"
python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model DiffusionTS --source_run_id "$CWRU_DIFFUSIONTS_RUN_ID"

# DEMADICS
python train_generator.py --track bearing_fault --dataset DEMADICS --model DiffusionTS
DEMADICS_DIFFUSIONTS_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'DiffusionTS').name)")"
python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model DiffusionTS --source_run_id "$DEMADICS_DIFFUSIONTS_RUN_ID"
```

### 2F. TimeFlow

```bash
# CMAPSS
python train_generator.py --track engine_rul --dataset CMAPSS --model TimeFlow
CMAPSS_TIMEFLOW_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'TimeFlow').name)")"
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model TimeFlow --source_run_id "$CMAPSS_TIMEFLOW_RUN_ID"

# CWRU
python train_generator.py --track bearing_fault --dataset CWRU --model TimeFlow
CWRU_TIMEFLOW_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'TimeFlow').name)")"
python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model TimeFlow --source_run_id "$CWRU_TIMEFLOW_RUN_ID"

# DEMADICS
python train_generator.py --track bearing_fault --dataset DEMADICS --model TimeFlow
DEMADICS_TIMEFLOW_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'TimeFlow').name)")"
python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model TimeFlow --source_run_id "$DEMADICS_TIMEFLOW_RUN_ID"
```

### 2G. FlowMatch-PdM

```bash
# CMAPSS
python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch
CMAPSS_FLOWMATCH_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch').name)")"
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FlowMatch --source_run_id "$CMAPSS_FLOWMATCH_RUN_ID"

# CWRU
python train_generator.py --track bearing_fault --dataset CWRU --model FlowMatch
CWRU_FLOWMATCH_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'FlowMatch').name)")"
python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model FlowMatch --source_run_id "$CWRU_FLOWMATCH_RUN_ID"

# DEMADICS
python train_generator.py --track bearing_fault --dataset DEMADICS --model FlowMatch
DEMADICS_FLOWMATCH_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'FlowMatch').name)")"
python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model FlowMatch --source_run_id "$DEMADICS_FLOWMATCH_RUN_ID"
```

---

## 4. Phase 3 — Secondary Datasets

Run after Phase 2 is complete. Freeze the Top 1/2/3 generators from `docs/03_result_logger.md` Table 2 + Table 3, then:

```bash
# Set the top models (example — replace with actual winners)
export TOP_1="FlowMatch"
export TOP_2="COTGAN"
export TOP_3="FaultDiffusion"

# For each top model × secondary dataset:
for MODEL in "$TOP_1" "$TOP_2" "$TOP_3"; do
  for SPEC in "engine_rul:N-CMAPSS" "bearing_rul:FEMTO" "bearing_rul:XJTU-SY" "bearing_fault:Paderborn"; do
    TRACK="${SPEC%%:*}"
    DATASET="${SPEC##*:}"
    echo "=== $MODEL on $DATASET ($TRACK) ==="
    python train_generator.py --track "$TRACK" --dataset "$DATASET" --model "$MODEL"
    RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('$TRACK', '$DATASET', '$MODEL').name)")"
    python train_classifier.py --track "$TRACK" --dataset "$DATASET" --model baseline --gen_model "$MODEL" --source_run_id "$RUN_ID"
  done
done
```

---

## 5. Phase 4 — Ablations

FlowMatch-PdM ablations on CMAPSS:

```bash
# No Prior (Gaussian noise instead of harmonic prior)
python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch --ablation no_prior
ABL_NO_PRIOR_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch_ablation_no_prior').name)")"
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FlowMatch --gen_ablation no_prior --source_run_id "$ABL_NO_PRIOR_RUN_ID"

# No TCCM (disable temporal-condition consistency manifold loss)
python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch --ablation no_tccm
ABL_NO_TCCM_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch_ablation_no_tccm').name)")"
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FlowMatch --gen_ablation no_tccm --source_run_id "$ABL_NO_TCCM_RUN_ID"

# No LAP (disable layer-adaptive pruning callback)
python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch --ablation no_lap
ABL_NO_LAP_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch_ablation_no_lap').name)")"
python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FlowMatch --gen_ablation no_lap --source_run_id "$ABL_NO_LAP_RUN_ID"
```

---

## 6. Phase 5 — W&B Sweep + run_all.sh

### W&B Bayesian Sweep

```bash
# Create the sweep (returns a SWEEP_ID)
wandb sweep configs/sweep_flowmatch_cmapss.yaml

# Launch the agent (replace <SWEEP_ID> with the returned ID)
wandb agent <SWEEP_ID>
```

### End-to-End Proof-of-Concept

```bash
# Default: CMAPSS with FlowMatch + COTGAN + FaultDiffusion
./run_all.sh

# Custom generator set:
GEN_MODELS="FlowMatch TimeVAE TimeGAN" ./run_all.sh

# Custom dataset:
TRACK=bearing_fault DATASET=CWRU ./run_all.sh
```

---

## Estimated Runtimes

All estimates are for a single NVIDIA RTX A6000 (48 GB).

| Task | CMAPSS | CWRU | DEMADICS |
|------|--------|------|----------|
| Baseline classifier | ~15 min | ~10 min | ~10 min |
| TimeVAE (200 ep) | ~15 min | ~20 min | ~15 min |
| TimeGAN (200 ep) | ~25 min | ~30 min | ~25 min |
| DiffusionTS (200 ep) | ~30 min | ~40 min | ~30 min |
| FaultDiffusion (200 ep) | ~35 min | ~45 min | ~35 min |
| TimeFlow (200 ep) | ~20 min | ~25 min | ~20 min |
| COTGAN (200 ep) | ~30 min | ~35 min | ~30 min |
| FlowMatch-PdM (200 ep) | ~25 min | ~35 min | ~25 min |
| Augmented classifier | ~15 min | ~10 min | ~10 min |
| **Full Phase 2 (7 gen × 3 ds)** | **~10 GPU-hours** | | |
| **Full experiment (all phases)** | **~20–25 GPU-hours** | | |

---

## How to Read Results

### Directory Structure

```
results/<track>/<dataset>/<model>/run_<timestamp>/
├── best_model_classifier/          # Classifier checkpoints
├── best_models_generator/          # Generator checkpoints
├── generator_datas/
│   ├── synthetic_data.npy          # Generated time series
│   ├── synthetic_targets.npy       # Corresponding labels/RUL values
│   ├── real_minority_data.npy      # Real minority subset
│   └── generation_manifest.json    # Generation metadata
├── evaluation_results/
│   ├── classifier_metrics.json     # Accuracy, F1, RMSE, MAE, R²
│   ├── metrics.json                # Generator: FTSD, MMD, disc_score, TSTR
│   ├── metrics.txt                 # Human-readable summary
│   ├── projection_pca_tsne.png     # PCA + t-SNE of real vs synthetic
│   ├── marginal_kde.png            # Per-feature KDE overlay
│   ├── classifier_confusion_matrix.png
│   └── classifier_regression_diagnostics.png
├── logs/                           # TensorBoard / W&B logs
└── manifest.json                   # Run metadata
```

### Key JSON Files

| File | Contains | Key Fields |
|------|----------|------------|
| `classifier_metrics.json` | Downstream task metrics | `test_accuracy`, `test_f1_macro`, `test_rmse`, `test_mae`, `test_r2` |
| `metrics.json` | Generator fidelity metrics | `ftsd`, `mmd`, `discriminative_score`, `predictive_score_mae` |

### Identifying the Best Model

1. Compare FTSD (lower is better) across generators in `metrics.json`
2. Compare downstream task metric in `classifier_metrics.json` for
   augmented classifiers
3. Log all results in `docs/03_result_logger.md` Tables 2 and 3
4. The best model has the best combination of low FTSD and good
   downstream improvement over the Phase 0 baseline

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2` hangs | `devices: "auto"` in config detects 2 GPUs | Use `CUDA_VISIBLE_DEVICES=0` prefix or set `devices: 1` in config |
| `TypeError: '<=' not supported between instances of 'float' and 'str'` in FlowMatchPdM | PyYAML parses `1e-3` as string | Fixed: `configure_optimizers` now casts to `float()`. Also use `0.001` notation in YAML |
| CUDA OOM during generation on CWRU | CWRU window_size=2048 creates large tensors during LSTM decode for 1500+ samples | Use the GPU with the most free VRAM (`nvidia-smi`), or reduce `evaluation.max_samples` |
| `import torch` takes >60 s | Slow NFS filesystem on shared cluster | Normal — wait for first import, subsequent imports use cache |
