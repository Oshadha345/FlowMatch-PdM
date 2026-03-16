# 🚀 FlowMatch-PdM: Master Experiment Plan

This is the exact execution ledger for the current repo. The workflow is now:

- `train_classifier.py` trains and auto-evaluates baseline or augmented classifiers
- `train_generator.py` trains and auto-evaluates generators
- `run_evaluation.py` re-evaluates existing runs on demand with `--eval_mode classifier|generator`

Every checkbox below is meant to map to a real run folder under `results/<track>/<dataset>/<model>/run_<timestamp>/`.

---

## Preflight

- [x] `cd /home/buddhiw/flowmatch/FlowMatch-PdM`
- [x] `conda activate flowmatch_pdm`
- [x] `chmod +x run_all.sh`
- [x] `python -m jupyter nbconvert --to notebook --execute notebooks/01_dataset_analysis.ipynb --output 01_dataset_analysis.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=0 --ExecutePreprocessor.kernel_name=python3`
- [x] Confirm the executed notebook ends with `Supported loader readiness: GO`
- [x] Confirm the executed notebook ends with `Full requested roster readiness: GO`

---

## Phase 0: Baseline Classifiers

**Goal:** pure downstream baselines with automatic classifier evaluation.

### Engine RUL

- [x] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline`
- [ ] `python train_classifier.py --track engine_rul --dataset N-CMAPSS --model baseline`

### Bearing RUL

- [ ] `python train_classifier.py --track bearing_rul --dataset FEMTO --model baseline`
- [ ] `python train_classifier.py --track bearing_rul --dataset XJTU-SY --model baseline`

### Fault Classification

- [x] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline`
- [x] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline`
- [x] `python train_classifier.py --track bearing_fault --dataset Paderborn --model baseline`

### Phase 0 Checks

- [ ] Every baseline run wrote `best_model_classifier/*.ckpt`
- [ ] Every baseline run wrote `evaluation_results/phase0_metrics.json`
- [ ] Every baseline run wrote `evaluation_results/classifier_metrics.json`
- [ ] Every baseline run wrote either `evaluation_results/classifier_confusion_matrix.png` or `evaluation_results/classifier_regression_diagnostics.png`

---

## Phase 1: Classical Augmentation Baselines

**Goal:** classical augmentation with the same classifier training entrypoint.

### 1A. Noise / Jittering

- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --aug noise`
- [ ] `python train_classifier.py --track engine_rul --dataset N-CMAPSS --model baseline --aug noise`
- [ ] `python train_classifier.py --track bearing_rul --dataset FEMTO --model baseline --aug noise`
- [ ] `python train_classifier.py --track bearing_rul --dataset XJTU-SY --model baseline --aug noise`
- [ ] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --aug noise`
- [ ] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --aug noise`
- [ ] `python train_classifier.py --track bearing_fault --dataset Paderborn --model baseline --aug noise`

### 1B. SMOTE

- [ ] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --aug smote`
- [ ] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --aug smote`
- [ ] `python train_classifier.py --track bearing_fault --dataset Paderborn --model baseline --aug smote`

### Phase 1 Checks

- [ ] Every augmented classifier run wrote `augmentation_summary.json`
- [ ] Every augmented classifier run wrote `evaluation_results/phase3_metrics.json`
- [ ] Every augmented classifier run wrote `evaluation_results/classifier_metrics.json`

---

## Phase 2: Generative Proving Ground

**Goal:** train every generator on the three primary datasets. Generator training now auto-runs synthetic evaluation and writes FTSD, MMD, discriminative score, and TSTR artifacts before the augmented classifier step.

Primary datasets:
- `CMAPSS`
- `CWRU`
- `DEMADICS`

### `RUN_ID` Resolver

Use this exact command after each generator training run:

```bash
RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('TRACK', 'DATASET', 'MODEL_FOLDER').name)")"
echo "$RUN_ID"
```

For non-ablation runs, `MODEL_FOLDER` is the same as the CLI `--model`.
For FlowMatch ablations, `MODEL_FOLDER` is `FlowMatch_ablation_<ablation_name>`.

### 2A. TimeVAE

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeVAE`
- [ ] `CMAPSS_TIMEVAE_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'TimeVAE').name)")"; echo "$CMAPSS_TIMEVAE_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model TimeVAE --source_run_id "$CMAPSS_TIMEVAE_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model TimeVAE`
- [ ] `CWRU_TIMEVAE_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'TimeVAE').name)")"; echo "$CWRU_TIMEVAE_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model TimeVAE --source_run_id "$CWRU_TIMEVAE_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model TimeVAE`
- [ ] `DEMADICS_TIMEVAE_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'TimeVAE').name)")"; echo "$DEMADICS_TIMEVAE_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model TimeVAE --source_run_id "$DEMADICS_TIMEVAE_RUN_ID"`

### 2B. TimeGAN

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeGAN`
- [ ] `CMAPSS_TIMEGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'TimeGAN').name)")"; echo "$CMAPSS_TIMEGAN_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model TimeGAN --source_run_id "$CMAPSS_TIMEGAN_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model TimeGAN`
- [ ] `CWRU_TIMEGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'TimeGAN').name)")"; echo "$CWRU_TIMEGAN_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model TimeGAN --source_run_id "$CWRU_TIMEGAN_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model TimeGAN`
- [ ] `DEMADICS_TIMEGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'TimeGAN').name)")"; echo "$DEMADICS_TIMEGAN_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model TimeGAN --source_run_id "$DEMADICS_TIMEGAN_RUN_ID"`

### 2C. COTGAN

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model COTGAN`
- [ ] `CMAPSS_COTGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'COTGAN').name)")"; echo "$CMAPSS_COTGAN_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model COTGAN --source_run_id "$CMAPSS_COTGAN_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model COTGAN`
- [ ] `CWRU_COTGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'COTGAN').name)")"; echo "$CWRU_COTGAN_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model COTGAN --source_run_id "$CWRU_COTGAN_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model COTGAN`
- [ ] `DEMADICS_COTGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'COTGAN').name)")"; echo "$DEMADICS_COTGAN_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model COTGAN --source_run_id "$DEMADICS_COTGAN_RUN_ID"`

### 2D. FaultDiffusion

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model FaultDiffusion`
- [ ] `CMAPSS_FAULTDIFFUSION_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FaultDiffusion').name)")"; echo "$CMAPSS_FAULTDIFFUSION_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FaultDiffusion --source_run_id "$CMAPSS_FAULTDIFFUSION_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model FaultDiffusion`
- [ ] `CWRU_FAULTDIFFUSION_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'FaultDiffusion').name)")"; echo "$CWRU_FAULTDIFFUSION_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model FaultDiffusion --source_run_id "$CWRU_FAULTDIFFUSION_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model FaultDiffusion`
- [ ] `DEMADICS_FAULTDIFFUSION_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'FaultDiffusion').name)")"; echo "$DEMADICS_FAULTDIFFUSION_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model FaultDiffusion --source_run_id "$DEMADICS_FAULTDIFFUSION_RUN_ID"`

### 2E. DiffusionTS

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model DiffusionTS`
- [ ] `CMAPSS_DIFFUSIONTS_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'DiffusionTS').name)")"; echo "$CMAPSS_DIFFUSIONTS_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model DiffusionTS --source_run_id "$CMAPSS_DIFFUSIONTS_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model DiffusionTS`
- [ ] `CWRU_DIFFUSIONTS_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'DiffusionTS').name)")"; echo "$CWRU_DIFFUSIONTS_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model DiffusionTS --source_run_id "$CWRU_DIFFUSIONTS_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model DiffusionTS`
- [ ] `DEMADICS_DIFFUSIONTS_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'DiffusionTS').name)")"; echo "$DEMADICS_DIFFUSIONTS_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model DiffusionTS --source_run_id "$DEMADICS_DIFFUSIONTS_RUN_ID"`

### 2F. TimeFlow

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeFlow`
- [ ] `CMAPSS_TIMEFLOW_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'TimeFlow').name)")"; echo "$CMAPSS_TIMEFLOW_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model TimeFlow --source_run_id "$CMAPSS_TIMEFLOW_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model TimeFlow`
- [ ] `CWRU_TIMEFLOW_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'TimeFlow').name)")"; echo "$CWRU_TIMEFLOW_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model TimeFlow --source_run_id "$CWRU_TIMEFLOW_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model TimeFlow`
- [ ] `DEMADICS_TIMEFLOW_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'TimeFlow').name)")"; echo "$DEMADICS_TIMEFLOW_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model TimeFlow --source_run_id "$DEMADICS_TIMEFLOW_RUN_ID"`

### 2G. FlowMatch-PdM

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch`
- [ ] `CMAPSS_FLOWMATCH_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch').name)")"; echo "$CMAPSS_FLOWMATCH_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FlowMatch --source_run_id "$CMAPSS_FLOWMATCH_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model FlowMatch`
- [ ] `CWRU_FLOWMATCH_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'FlowMatch').name)")"; echo "$CWRU_FLOWMATCH_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline --gen_model FlowMatch --source_run_id "$CWRU_FLOWMATCH_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model FlowMatch`
- [ ] `DEMADICS_FLOWMATCH_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'FlowMatch').name)")"; echo "$DEMADICS_FLOWMATCH_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model FlowMatch --source_run_id "$DEMADICS_FLOWMATCH_RUN_ID"`

### Phase 2 Checks

- [ ] Every generator run wrote `best_models_generator/*.ckpt`
- [ ] Every generator run wrote `generator_datas/synthetic_data.npy`
- [ ] Every generator run wrote `generator_datas/synthetic_targets.npy`
- [ ] Every generator run wrote `evaluation_results/metrics.txt`
- [ ] Every generator run wrote `evaluation_results/metrics.json`
- [ ] Every generator run wrote `evaluation_results/projection_pca_tsne.png`
- [ ] Every generator run wrote `evaluation_results/marginal_kde.png`
- [ ] Every augmented classifier run wrote `evaluation_results/classifier_metrics.json`

---

## Phase 3: Secondary-Dataset Generalization

Freeze the Top 1, Top 2, and Top 3 models in `docs/03_result_logger.md`, then export them:

- [ ] `read -r -p "Enter Top 1 model: " TOP_1; export TOP_1`
- [ ] `read -r -p "Enter Top 2 model: " TOP_2; export TOP_2`
- [ ] `read -r -p "Enter Top 3 model: " TOP_3; export TOP_3`

### Rank 1

- [ ] `python train_generator.py --track engine_rul --dataset N-CMAPSS --model "$TOP_1"`
- [ ] `TOP1_NCMAPSS_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'N-CMAPSS', os.environ['TOP_1']).name)")"; echo "$TOP1_NCMAPSS_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset N-CMAPSS --model baseline --gen_model "$TOP_1" --source_run_id "$TOP1_NCMAPSS_RUN_ID"`
- [ ] `python train_generator.py --track bearing_rul --dataset FEMTO --model "$TOP_1"`
- [ ] `TOP1_FEMTO_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'FEMTO', os.environ['TOP_1']).name)")"; echo "$TOP1_FEMTO_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_rul --dataset FEMTO --model baseline --gen_model "$TOP_1" --source_run_id "$TOP1_FEMTO_RUN_ID"`
- [ ] `python train_generator.py --track bearing_rul --dataset XJTU-SY --model "$TOP_1"`
- [ ] `TOP1_XJTU_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'XJTU-SY', os.environ['TOP_1']).name)")"; echo "$TOP1_XJTU_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_rul --dataset XJTU-SY --model baseline --gen_model "$TOP_1" --source_run_id "$TOP1_XJTU_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset Paderborn --model "$TOP_1"`
- [ ] `TOP1_PADERBORN_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'Paderborn', os.environ['TOP_1']).name)")"; echo "$TOP1_PADERBORN_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset Paderborn --model baseline --gen_model "$TOP_1" --source_run_id "$TOP1_PADERBORN_RUN_ID"`

### Rank 2

- [ ] `python train_generator.py --track engine_rul --dataset N-CMAPSS --model "$TOP_2"`
- [ ] `TOP2_NCMAPSS_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'N-CMAPSS', os.environ['TOP_2']).name)")"; echo "$TOP2_NCMAPSS_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset N-CMAPSS --model baseline --gen_model "$TOP_2" --source_run_id "$TOP2_NCMAPSS_RUN_ID"`
- [ ] `python train_generator.py --track bearing_rul --dataset FEMTO --model "$TOP_2"`
- [ ] `TOP2_FEMTO_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'FEMTO', os.environ['TOP_2']).name)")"; echo "$TOP2_FEMTO_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_rul --dataset FEMTO --model baseline --gen_model "$TOP_2" --source_run_id "$TOP2_FEMTO_RUN_ID"`
- [ ] `python train_generator.py --track bearing_rul --dataset XJTU-SY --model "$TOP_2"`
- [ ] `TOP2_XJTU_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'XJTU-SY', os.environ['TOP_2']).name)")"; echo "$TOP2_XJTU_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_rul --dataset XJTU-SY --model baseline --gen_model "$TOP_2" --source_run_id "$TOP2_XJTU_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset Paderborn --model "$TOP_2"`
- [ ] `TOP2_PADERBORN_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'Paderborn', os.environ['TOP_2']).name)")"; echo "$TOP2_PADERBORN_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset Paderborn --model baseline --gen_model "$TOP_2" --source_run_id "$TOP2_PADERBORN_RUN_ID"`

### Rank 3

- [ ] `python train_generator.py --track engine_rul --dataset N-CMAPSS --model "$TOP_3"`
- [ ] `TOP3_NCMAPSS_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'N-CMAPSS', os.environ['TOP_3']).name)")"; echo "$TOP3_NCMAPSS_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset N-CMAPSS --model baseline --gen_model "$TOP_3" --source_run_id "$TOP3_NCMAPSS_RUN_ID"`
- [ ] `python train_generator.py --track bearing_rul --dataset FEMTO --model "$TOP_3"`
- [ ] `TOP3_FEMTO_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'FEMTO', os.environ['TOP_3']).name)")"; echo "$TOP3_FEMTO_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_rul --dataset FEMTO --model baseline --gen_model "$TOP_3" --source_run_id "$TOP3_FEMTO_RUN_ID"`
- [ ] `python train_generator.py --track bearing_rul --dataset XJTU-SY --model "$TOP_3"`
- [ ] `TOP3_XJTU_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'XJTU-SY', os.environ['TOP_3']).name)")"; echo "$TOP3_XJTU_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_rul --dataset XJTU-SY --model baseline --gen_model "$TOP_3" --source_run_id "$TOP3_XJTU_RUN_ID"`
- [ ] `python train_generator.py --track bearing_fault --dataset Paderborn --model "$TOP_3"`
- [ ] `TOP3_PADERBORN_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'Paderborn', os.environ['TOP_3']).name)")"; echo "$TOP3_PADERBORN_RUN_ID"`
- [ ] `python train_classifier.py --track bearing_fault --dataset Paderborn --model baseline --gen_model "$TOP_3" --source_run_id "$TOP3_PADERBORN_RUN_ID"`

---

## Phase 4: FlowMatch-PdM Ablations

**Goal:** run real ablation variants on CMAPSS with the same automatic generator evaluation and augmented-classifier evaluation workflow.

### No Prior

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch --ablation no_prior`
- [ ] `ABL_NO_PRIOR_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch_ablation_no_prior').name)")"; echo "$ABL_NO_PRIOR_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FlowMatch --gen_ablation no_prior --source_run_id "$ABL_NO_PRIOR_RUN_ID"`

### No TCCM

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch --ablation no_tccm`
- [ ] `ABL_NO_TCCM_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch_ablation_no_tccm').name)")"; echo "$ABL_NO_TCCM_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FlowMatch --gen_ablation no_tccm --source_run_id "$ABL_NO_TCCM_RUN_ID"`

### No LAP

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch --ablation no_lap`
- [ ] `ABL_NO_LAP_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch_ablation_no_lap').name)")"; echo "$ABL_NO_LAP_RUN_ID"`
- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FlowMatch --gen_ablation no_lap --source_run_id "$ABL_NO_LAP_RUN_ID"`

---

## Phase 5: W&B Sweep And Final Proof-Of-Concept

### 5A. CMAPSS FlowMatch-PdM Sweep

- [ ] `wandb sweep configs/sweep_flowmatch_cmapss.yaml`
- [ ] `wandb agent <SWEEP_ID>`

### 5B. Final CMAPSS Proof-Of-Concept

- [ ] `./run_all.sh`

### Optional Manual Re-Evaluation Commands

- [ ] `python run_evaluation.py --eval_mode generator --track engine_rul --dataset CMAPSS --model FlowMatch --run_id "$CMAPSS_FLOWMATCH_RUN_ID"`
- [ ] `python run_evaluation.py --eval_mode classifier --track engine_rul --dataset CMAPSS --model baseline`
- [ ] `python run_evaluation.py --eval_mode classifier --track engine_rul --dataset CMAPSS --model baseline --source_gen_model FlowMatch --source_run_id "$CMAPSS_FLOWMATCH_RUN_ID"`

### Final Checks

- [ ] Generator runs always contain `generator_datas/` and `evaluation_results/metrics.json`
- [ ] Classifier runs always contain `evaluation_results/classifier_metrics.json`
- [ ] Phase 4 ablation results are logged in `docs/03_result_logger.md`
- [ ] Sweep winner and proof-of-concept run IDs are logged in `docs/03_result_logger.md`
