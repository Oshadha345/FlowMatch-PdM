# 🚀 FlowMatch-PdM: Master Experiment Plan

Welcome to the lab. This is the exact command ledger for the current repository state. Run the phases in order and tick the boxes only after the corresponding artifacts exist under `results/<track>/<dataset>/<model>/run_<timestamp>/`.

---

## Phase Map

- Phase 0: `train_classifier.py`
- Phase 1: `train_classifier_aug.py --aug ...`
- Phase 2: `train_generator.py` -> `run_evaluation.py` -> `train_classifier_aug.py --gen_model ...`
- Phase 3: same Phase 2 loop on the secondary datasets with the top 3 Phase 2 models
- Phase 4: consolidate the finished runs into `docs/03_result_logger.md`
- Phase 5: rerun the final CMAPSS proof-of-concept with the frozen pipeline

---

## Preflight

- [ ] `cd /home/buddhiw/flowmatch/FlowMatch-PdM`
- [ ] `conda activate flowmatch_pdm`
- [ ] `chmod +x run_all.sh`
- [ ] `python -m jupyter nbconvert --to notebook --execute notebooks/01_dataset_analysis.ipynb --output 01_dataset_analysis.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=0 --ExecutePreprocessor.kernel_name=python3`
- [ ] Confirm the executed notebook ends with `Supported loader readiness: GO`
- [ ] Confirm the executed notebook ends with `Full requested roster readiness: GO`

---

## Phase 0: The Empirical Foundation

**Goal:** establish the pure baseline performance of the downstream models before any augmentation.

### Engine RUL

- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline`
- [ ] `python train_classifier.py --track engine_rul --dataset N-CMAPSS --model baseline`

### Bearing RUL

- [ ] `python train_classifier.py --track bearing_rul --dataset FEMTO --model baseline`
- [ ] `python train_classifier.py --track bearing_rul --dataset XJTU-SY --model baseline`

### Fault Classification

- [ ] `python train_classifier.py --track bearing_fault --dataset CWRU --model baseline`
- [ ] `python train_classifier.py --track bearing_fault --dataset DEMADICS --model baseline`
- [ ] `python train_classifier.py --track bearing_fault --dataset Paderborn --model baseline`

### Phase 0 Checks

- [ ] Every Phase 0 run wrote `best_model_classifier/*.ckpt`
- [ ] Every Phase 0 run wrote `evaluation_results/phase0_metrics.json`
- [ ] Every Phase 0 run wrote `run_manifest.json`
- [ ] Table 1 in `docs/03_result_logger.md` is filled

---

## Phase 1: Classical Augmentation Baselines

**Goal:** measure whether classical augmentation improves downstream training.

### 1A. Noise / Jittering

- [ ] `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --model baseline --aug noise`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset N-CMAPSS --model baseline --aug noise`
- [ ] `python train_classifier_aug.py --track bearing_rul --dataset FEMTO --model baseline --aug noise`
- [ ] `python train_classifier_aug.py --track bearing_rul --dataset XJTU-SY --model baseline --aug noise`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset CWRU --model baseline --aug noise`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --model baseline --aug noise`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset Paderborn --model baseline --aug noise`

### 1B. SMOTE

`SMOTE` is classification-only in the current repo.

- [ ] `python train_classifier_aug.py --track bearing_fault --dataset CWRU --model baseline --aug smote`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --model baseline --aug smote`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset Paderborn --model baseline --aug smote`

### Phase 1 Checks

- [ ] Every Phase 1 run wrote `augmentation_summary.json`
- [ ] Every Phase 1 run wrote `best_model_classifier/*.ckpt`
- [ ] Every Phase 1 run wrote `evaluation_results/phase3_metrics.json`
- [ ] Table 3 rows for `+ Noise` and `+ SMOTE` are filled in `docs/03_result_logger.md`

---

## Phase 2: Generative Proving Ground

**Goal:** train every generator on the three primary datasets, evaluate synthetic fidelity, and retrain the downstream model with synthetic data.

Primary datasets:
- `CMAPSS`
- `CWRU`
- `DEMADICS`

### 2A. TimeVAE

#### CMAPSS

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeVAE`
- [ ] `CMAPSS_TIMEVAE_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'TimeVAE').name)")"; echo "$CMAPSS_TIMEVAE_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset CMAPSS --model TimeVAE --run_id "$CMAPSS_TIMEVAE_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --model baseline --gen_model TimeVAE --run_id "$CMAPSS_TIMEVAE_RUN_ID"`

#### CWRU

- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model TimeVAE`
- [ ] `CWRU_TIMEVAE_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'TimeVAE').name)")"; echo "$CWRU_TIMEVAE_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset CWRU --model TimeVAE --run_id "$CWRU_TIMEVAE_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset CWRU --model baseline --gen_model TimeVAE --run_id "$CWRU_TIMEVAE_RUN_ID"`

#### DEMADICS

- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model TimeVAE`
- [ ] `DEMADICS_TIMEVAE_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'TimeVAE').name)")"; echo "$DEMADICS_TIMEVAE_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset DEMADICS --model TimeVAE --run_id "$DEMADICS_TIMEVAE_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model TimeVAE --run_id "$DEMADICS_TIMEVAE_RUN_ID"`

### 2B. TimeGAN

#### CMAPSS

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeGAN`
- [ ] `CMAPSS_TIMEGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'TimeGAN').name)")"; echo "$CMAPSS_TIMEGAN_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset CMAPSS --model TimeGAN --run_id "$CMAPSS_TIMEGAN_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --model baseline --gen_model TimeGAN --run_id "$CMAPSS_TIMEGAN_RUN_ID"`

#### CWRU

- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model TimeGAN`
- [ ] `CWRU_TIMEGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'TimeGAN').name)")"; echo "$CWRU_TIMEGAN_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset CWRU --model TimeGAN --run_id "$CWRU_TIMEGAN_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset CWRU --model baseline --gen_model TimeGAN --run_id "$CWRU_TIMEGAN_RUN_ID"`

#### DEMADICS

- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model TimeGAN`
- [ ] `DEMADICS_TIMEGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'TimeGAN').name)")"; echo "$DEMADICS_TIMEGAN_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset DEMADICS --model TimeGAN --run_id "$DEMADICS_TIMEGAN_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model TimeGAN --run_id "$DEMADICS_TIMEGAN_RUN_ID"`

### 2C. COTGAN

#### CMAPSS

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model COTGAN`
- [ ] `CMAPSS_COTGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'COTGAN').name)")"; echo "$CMAPSS_COTGAN_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset CMAPSS --model COTGAN --run_id "$CMAPSS_COTGAN_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --model baseline --gen_model COTGAN --run_id "$CMAPSS_COTGAN_RUN_ID"`

#### CWRU

- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model COTGAN`
- [ ] `CWRU_COTGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'COTGAN').name)")"; echo "$CWRU_COTGAN_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset CWRU --model COTGAN --run_id "$CWRU_COTGAN_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset CWRU --model baseline --gen_model COTGAN --run_id "$CWRU_COTGAN_RUN_ID"`

#### DEMADICS

- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model COTGAN`
- [ ] `DEMADICS_COTGAN_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'COTGAN').name)")"; echo "$DEMADICS_COTGAN_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset DEMADICS --model COTGAN --run_id "$DEMADICS_COTGAN_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model COTGAN --run_id "$DEMADICS_COTGAN_RUN_ID"`

### 2D. FaultDiffusion

#### CMAPSS

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model FaultDiffusion`
- [ ] `CMAPSS_FAULTDIFFUSION_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FaultDiffusion').name)")"; echo "$CMAPSS_FAULTDIFFUSION_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset CMAPSS --model FaultDiffusion --run_id "$CMAPSS_FAULTDIFFUSION_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FaultDiffusion --run_id "$CMAPSS_FAULTDIFFUSION_RUN_ID"`

#### CWRU

- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model FaultDiffusion`
- [ ] `CWRU_FAULTDIFFUSION_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'FaultDiffusion').name)")"; echo "$CWRU_FAULTDIFFUSION_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset CWRU --model FaultDiffusion --run_id "$CWRU_FAULTDIFFUSION_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset CWRU --model baseline --gen_model FaultDiffusion --run_id "$CWRU_FAULTDIFFUSION_RUN_ID"`

#### DEMADICS

- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model FaultDiffusion`
- [ ] `DEMADICS_FAULTDIFFUSION_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'FaultDiffusion').name)")"; echo "$DEMADICS_FAULTDIFFUSION_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset DEMADICS --model FaultDiffusion --run_id "$DEMADICS_FAULTDIFFUSION_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model FaultDiffusion --run_id "$DEMADICS_FAULTDIFFUSION_RUN_ID"`

### 2E. DiffusionTS

#### CMAPSS

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model DiffusionTS`
- [ ] `CMAPSS_DIFFUSIONTS_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'DiffusionTS').name)")"; echo "$CMAPSS_DIFFUSIONTS_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset CMAPSS --model DiffusionTS --run_id "$CMAPSS_DIFFUSIONTS_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --model baseline --gen_model DiffusionTS --run_id "$CMAPSS_DIFFUSIONTS_RUN_ID"`

#### CWRU

- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model DiffusionTS`
- [ ] `CWRU_DIFFUSIONTS_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'DiffusionTS').name)")"; echo "$CWRU_DIFFUSIONTS_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset CWRU --model DiffusionTS --run_id "$CWRU_DIFFUSIONTS_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset CWRU --model baseline --gen_model DiffusionTS --run_id "$CWRU_DIFFUSIONTS_RUN_ID"`

#### DEMADICS

- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model DiffusionTS`
- [ ] `DEMADICS_DIFFUSIONTS_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'DiffusionTS').name)")"; echo "$DEMADICS_DIFFUSIONTS_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset DEMADICS --model DiffusionTS --run_id "$DEMADICS_DIFFUSIONTS_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model DiffusionTS --run_id "$DEMADICS_DIFFUSIONTS_RUN_ID"`

### 2F. TimeFlow

#### CMAPSS

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeFlow`
- [ ] `CMAPSS_TIMEFLOW_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'TimeFlow').name)")"; echo "$CMAPSS_TIMEFLOW_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset CMAPSS --model TimeFlow --run_id "$CMAPSS_TIMEFLOW_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --model baseline --gen_model TimeFlow --run_id "$CMAPSS_TIMEFLOW_RUN_ID"`

#### CWRU

- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model TimeFlow`
- [ ] `CWRU_TIMEFLOW_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'TimeFlow').name)")"; echo "$CWRU_TIMEFLOW_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset CWRU --model TimeFlow --run_id "$CWRU_TIMEFLOW_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset CWRU --model baseline --gen_model TimeFlow --run_id "$CWRU_TIMEFLOW_RUN_ID"`

#### DEMADICS

- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model TimeFlow`
- [ ] `DEMADICS_TIMEFLOW_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'TimeFlow').name)")"; echo "$DEMADICS_TIMEFLOW_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset DEMADICS --model TimeFlow --run_id "$DEMADICS_TIMEFLOW_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model TimeFlow --run_id "$DEMADICS_TIMEFLOW_RUN_ID"`

### 2G. FlowMatch-PdM

The CLI model name is `FlowMatch`. It resolves to the FlowMatch-PdM implementation and the `generative.flowmatch_pdm` config block.

#### CMAPSS

- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch`
- [ ] `CMAPSS_FLOWMATCH_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch').name)")"; echo "$CMAPSS_FLOWMATCH_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset CMAPSS --model FlowMatch --run_id "$CMAPSS_FLOWMATCH_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FlowMatch --run_id "$CMAPSS_FLOWMATCH_RUN_ID"`

#### CWRU

- [ ] `python train_generator.py --track bearing_fault --dataset CWRU --model FlowMatch`
- [ ] `CWRU_FLOWMATCH_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'CWRU', 'FlowMatch').name)")"; echo "$CWRU_FLOWMATCH_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset CWRU --model FlowMatch --run_id "$CWRU_FLOWMATCH_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset CWRU --model baseline --gen_model FlowMatch --run_id "$CWRU_FLOWMATCH_RUN_ID"`

#### DEMADICS

- [ ] `python train_generator.py --track bearing_fault --dataset DEMADICS --model FlowMatch`
- [ ] `DEMADICS_FLOWMATCH_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'DEMADICS', 'FlowMatch').name)")"; echo "$DEMADICS_FLOWMATCH_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset DEMADICS --model FlowMatch --run_id "$DEMADICS_FLOWMATCH_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --model baseline --gen_model FlowMatch --run_id "$DEMADICS_FLOWMATCH_RUN_ID"`

### Phase 2 Checks

- [ ] Every generator run wrote `best_models_generator/*.ckpt`
- [ ] Every evaluation run wrote `generator_datas/synthetic_data.npy`
- [ ] Every evaluation run wrote `generator_datas/synthetic_targets.npy`
- [ ] Every evaluation run wrote `evaluation_results/metrics.txt`
- [ ] Every evaluation run wrote `evaluation_results/metrics.json`
- [ ] Every evaluation run wrote `evaluation_results/projection_pca_tsne.png`
- [ ] Every evaluation run wrote `evaluation_results/marginal_kde.png`
- [ ] Every synthetic augmentation run wrote `evaluation_results/phase3_metrics.json`
- [ ] Tables 2 and 3 in `docs/03_result_logger.md` are filled for `CMAPSS`, `CWRU`, and `DEMADICS`

---

## Phase 3: Scalability And Generalization

**Goal:** rerun the same generator-evaluation-augmentation loop on the secondary datasets using the top 3 models selected from Phase 2.

Before running this phase, freeze the ranking in `docs/03_result_logger.md` and export the winners once in the shell:

- [ ] `read -r -p "Enter Top 1 model: " TOP_1; export TOP_1`
- [ ] `read -r -p "Enter Top 2 model: " TOP_2; export TOP_2`
- [ ] `read -r -p "Enter Top 3 model: " TOP_3; export TOP_3`

### Rank 1 Model

#### N-CMAPSS

- [ ] `python train_generator.py --track engine_rul --dataset N-CMAPSS --model "$TOP_1"`
- [ ] `TOP1_NCMAPSS_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'N-CMAPSS', os.environ['TOP_1']).name)")"; echo "$TOP1_NCMAPSS_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset N-CMAPSS --model "$TOP_1" --run_id "$TOP1_NCMAPSS_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset N-CMAPSS --model baseline --gen_model "$TOP_1" --run_id "$TOP1_NCMAPSS_RUN_ID"`

#### FEMTO

- [ ] `python train_generator.py --track bearing_rul --dataset FEMTO --model "$TOP_1"`
- [ ] `TOP1_FEMTO_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'FEMTO', os.environ['TOP_1']).name)")"; echo "$TOP1_FEMTO_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_rul --dataset FEMTO --model "$TOP_1" --run_id "$TOP1_FEMTO_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_rul --dataset FEMTO --model baseline --gen_model "$TOP_1" --run_id "$TOP1_FEMTO_RUN_ID"`

#### XJTU-SY

- [ ] `python train_generator.py --track bearing_rul --dataset XJTU-SY --model "$TOP_1"`
- [ ] `TOP1_XJTU_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'XJTU-SY', os.environ['TOP_1']).name)")"; echo "$TOP1_XJTU_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_rul --dataset XJTU-SY --model "$TOP_1" --run_id "$TOP1_XJTU_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_rul --dataset XJTU-SY --model baseline --gen_model "$TOP_1" --run_id "$TOP1_XJTU_RUN_ID"`

#### Paderborn

- [ ] `python train_generator.py --track bearing_fault --dataset Paderborn --model "$TOP_1"`
- [ ] `TOP1_PADERBORN_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'Paderborn', os.environ['TOP_1']).name)")"; echo "$TOP1_PADERBORN_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset Paderborn --model "$TOP_1" --run_id "$TOP1_PADERBORN_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset Paderborn --model baseline --gen_model "$TOP_1" --run_id "$TOP1_PADERBORN_RUN_ID"`

### Rank 2 Model

#### N-CMAPSS

- [ ] `python train_generator.py --track engine_rul --dataset N-CMAPSS --model "$TOP_2"`
- [ ] `TOP2_NCMAPSS_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'N-CMAPSS', os.environ['TOP_2']).name)")"; echo "$TOP2_NCMAPSS_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset N-CMAPSS --model "$TOP_2" --run_id "$TOP2_NCMAPSS_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset N-CMAPSS --model baseline --gen_model "$TOP_2" --run_id "$TOP2_NCMAPSS_RUN_ID"`

#### FEMTO

- [ ] `python train_generator.py --track bearing_rul --dataset FEMTO --model "$TOP_2"`
- [ ] `TOP2_FEMTO_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'FEMTO', os.environ['TOP_2']).name)")"; echo "$TOP2_FEMTO_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_rul --dataset FEMTO --model "$TOP_2" --run_id "$TOP2_FEMTO_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_rul --dataset FEMTO --model baseline --gen_model "$TOP_2" --run_id "$TOP2_FEMTO_RUN_ID"`

#### XJTU-SY

- [ ] `python train_generator.py --track bearing_rul --dataset XJTU-SY --model "$TOP_2"`
- [ ] `TOP2_XJTU_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'XJTU-SY', os.environ['TOP_2']).name)")"; echo "$TOP2_XJTU_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_rul --dataset XJTU-SY --model "$TOP_2" --run_id "$TOP2_XJTU_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_rul --dataset XJTU-SY --model baseline --gen_model "$TOP_2" --run_id "$TOP2_XJTU_RUN_ID"`

#### Paderborn

- [ ] `python train_generator.py --track bearing_fault --dataset Paderborn --model "$TOP_2"`
- [ ] `TOP2_PADERBORN_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'Paderborn', os.environ['TOP_2']).name)")"; echo "$TOP2_PADERBORN_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset Paderborn --model "$TOP_2" --run_id "$TOP2_PADERBORN_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset Paderborn --model baseline --gen_model "$TOP_2" --run_id "$TOP2_PADERBORN_RUN_ID"`

### Rank 3 Model

#### N-CMAPSS

- [ ] `python train_generator.py --track engine_rul --dataset N-CMAPSS --model "$TOP_3"`
- [ ] `TOP3_NCMAPSS_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'N-CMAPSS', os.environ['TOP_3']).name)")"; echo "$TOP3_NCMAPSS_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset N-CMAPSS --model "$TOP_3" --run_id "$TOP3_NCMAPSS_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset N-CMAPSS --model baseline --gen_model "$TOP_3" --run_id "$TOP3_NCMAPSS_RUN_ID"`

#### FEMTO

- [ ] `python train_generator.py --track bearing_rul --dataset FEMTO --model "$TOP_3"`
- [ ] `TOP3_FEMTO_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'FEMTO', os.environ['TOP_3']).name)")"; echo "$TOP3_FEMTO_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_rul --dataset FEMTO --model "$TOP_3" --run_id "$TOP3_FEMTO_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_rul --dataset FEMTO --model baseline --gen_model "$TOP_3" --run_id "$TOP3_FEMTO_RUN_ID"`

#### XJTU-SY

- [ ] `python train_generator.py --track bearing_rul --dataset XJTU-SY --model "$TOP_3"`
- [ ] `TOP3_XJTU_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_rul', 'XJTU-SY', os.environ['TOP_3']).name)")"; echo "$TOP3_XJTU_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_rul --dataset XJTU-SY --model "$TOP_3" --run_id "$TOP3_XJTU_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_rul --dataset XJTU-SY --model baseline --gen_model "$TOP_3" --run_id "$TOP3_XJTU_RUN_ID"`

#### Paderborn

- [ ] `python train_generator.py --track bearing_fault --dataset Paderborn --model "$TOP_3"`
- [ ] `TOP3_PADERBORN_RUN_ID="$(python -c "import os; from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('bearing_fault', 'Paderborn', os.environ['TOP_3']).name)")"; echo "$TOP3_PADERBORN_RUN_ID"`
- [ ] `python run_evaluation.py --track bearing_fault --dataset Paderborn --model "$TOP_3" --run_id "$TOP3_PADERBORN_RUN_ID"`
- [ ] `python train_classifier_aug.py --track bearing_fault --dataset Paderborn --model baseline --gen_model "$TOP_3" --run_id "$TOP3_PADERBORN_RUN_ID"`

### Phase 3 Checks

- [ ] Every secondary generator run wrote `best_models_generator/*.ckpt`
- [ ] Every secondary evaluation run wrote `evaluation_results/metrics.txt`
- [ ] Every secondary evaluation run wrote `evaluation_results/metrics.json`
- [ ] Every secondary synthetic augmentation run wrote `evaluation_results/phase3_metrics.json`
- [ ] Table 4 in `docs/03_result_logger.md` is filled

---

## Phase 4: Results Consolidation

**Goal:** freeze the experiment record before the final rerun.

- [ ] Table 1 in `docs/03_result_logger.md` is complete
- [ ] Table 2 in `docs/03_result_logger.md` is complete
- [ ] Table 3 in `docs/03_result_logger.md` is complete
- [ ] Table 4 in `docs/03_result_logger.md` is complete
- [ ] The final Top 1, Top 2, and Top 3 model names are written into the ledger
- [ ] Every result entry includes the exact `run_id`

---

## Phase 5: Final CMAPSS Proof-Of-Concept Rerun

**Goal:** rerun the full FlowMatch-PdM pipeline once more on CMAPSS after the ledger is frozen.

### Manual End-To-End Flow

- [ ] `python train_classifier.py --track engine_rul --dataset CMAPSS --model baseline`
- [ ] `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch`
- [ ] `FINAL_CMAPSS_FLOWMATCH_RUN_ID="$(python -c "from src.utils.logger_utils import resolve_run_dir; print(resolve_run_dir('engine_rul', 'CMAPSS', 'FlowMatch').name)")"; echo "$FINAL_CMAPSS_FLOWMATCH_RUN_ID"`
- [ ] `python run_evaluation.py --track engine_rul --dataset CMAPSS --model FlowMatch --run_id "$FINAL_CMAPSS_FLOWMATCH_RUN_ID"`
- [ ] `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --model baseline --gen_model FlowMatch --run_id "$FINAL_CMAPSS_FLOWMATCH_RUN_ID"`

### Automated Proof-Of-Concept

- [ ] `./run_all.sh`

### Final Checks

- [ ] The final CMAPSS generator run contains `generator_datas/synthetic_data.npy`
- [ ] The final CMAPSS generator run contains `generator_datas/synthetic_targets.npy`
- [ ] The final CMAPSS evaluation run contains `evaluation_results/metrics.txt`
- [ ] The final CMAPSS augmentation run contains `evaluation_results/phase3_metrics.json`
- [ ] The final CMAPSS `run_id` values are copied into `docs/03_result_logger.md`
