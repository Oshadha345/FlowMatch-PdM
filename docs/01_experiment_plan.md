# 🚀 FlowMatch-PdM: Master Experiment Plan

Welcome to the lab! This document contains the exact sequential commands to execute our 5-Phase research plan. Check these off as you complete them to track your progress accurately.

---

## Phase 0: The Empirical Foundation (Baseline Classifiers - No Augmentation)
**Goal:** Establish the pure baseline performance (floor and ceiling) of each classifier on all 7 datasets before introducing any synthetic data. `train_classifier.py` automatically runs testing at the end.

### Engine RUL Track (LSTM Regression)
- [ ] **CMAPSS:** `python train_classifier.py --track engine_rul --dataset CMAPSS`
- [ ] **N-CMAPSS:** `python train_classifier.py --track engine_rul --dataset N-CMAPSS`

### Bearing RUL Track (LSTM Regression)
- [ ] **FEMTO:** `python train_classifier.py --track bearing_rul --dataset FEMTO`
- [ ] **XJTU-SY:** `python train_classifier.py --track bearing_rul --dataset XJTU-SY`

### Fault Classification Track (1D-CNN)
- [ ] **CWRU:** `python train_classifier.py --track bearing_fault --dataset CWRU`
- [ ] **DEMADICS:** `python train_classifier.py --track bearing_fault --dataset DEMADICS`
- [ ] **PADERBORN:** `python train_classifier.py --track bearing_fault --dataset PADERBORN`

---

## Phase 1: Classical Augmentation Baselines (Primary Datasets)
**Goal:** Test standard statistical augmentation techniques on the Primary datasets (CMAPSS, CWRU, DEMADICS).

### 1A: SMOTE (Synthetic Minority Over-sampling)
- [ ] **CMAPSS (LSTM):** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --aug smote`
- [ ] **CWRU (1D-CNN):** `python train_classifier_aug.py --track bearing_fault --dataset CWRU --aug smote`
- [ ] **DEMADICS (1D-CNN):** `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --aug smote`

### 1B: Jittering (Gaussian Noise Addition)
- [ ] **CMAPSS (LSTM):** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --aug noise`
- [ ] **CWRU (1D-CNN):** `python train_classifier_aug.py --track bearing_fault --dataset CWRU --aug noise`
- [ ] **DEMADICS (1D-CNN):** `python train_classifier_aug.py --track bearing_fault --dataset DEMADICS --aug noise`

---

## Phase 2: Generative Proving Ground (Primary Datasets Only)
**Goal:** Pit all deep generative methods against each other on the 3 primary datasets. Measure fidelity (FTSD, MMD) and utility (Augmented Training).
*Note: Run these command blocks sequentially for each model.*

### 2A: TimeVAE
- [ ] **Train:** `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeVAE`
- [ ] **Eval:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model TimeVAE --run_id <RUN_ID>`
- [ ] **Aug Classify:** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
*(Repeat above 3 steps for `--track bearing_fault --dataset CWRU` and `--dataset DEMADICS`)*

### 2B: TimeGAN
- [ ] **Train:** `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeGAN`
- [ ] **Eval:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model TimeGAN --run_id <RUN_ID>`
- [ ] **Aug Classify:** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
*(Repeat for CWRU and DEMADICS)*

### 2C: COTGan
- [ ] **Train:** `python train_generator.py --track engine_rul --dataset CMAPSS --model COTGan`
- [ ] **Eval:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model COTGan --run_id <RUN_ID>`
- [ ] **Aug Classify:** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
*(Repeat for CWRU and DEMADICS)*

### 2D: Fault Diffusion
- [ ] **Train:** `python train_generator.py --track engine_rul --dataset CMAPSS --model FaultDiffusion`
- [ ] **Eval:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model FaultDiffusion --run_id <RUN_ID>`
- [ ] **Aug Classify:** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
*(Repeat for CWRU and DEMADICS)*

### 2E: Diffusion-TS
- [ ] **Train:** `python train_generator.py --track engine_rul --dataset CMAPSS --model DiffusionTS`
- [ ] **Eval:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model DiffusionTS --run_id <RUN_ID>`
- [ ] **Aug Classify:** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
*(Repeat for CWRU and DEMADICS)*

### 2F: TimeFlow
- [ ] **Train:** `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeFlow`
- [ ] **Eval:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model TimeFlow --run_id <RUN_ID>`
- [ ] **Aug Classify:** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
*(Repeat for CWRU and DEMADICS)*

### 2G: FlowMatch-PdM (Our Architecture)
- [ ] **Train:** `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch`
- [ ] **Eval:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model FlowMatch --run_id <RUN_ID>`
- [ ] **Aug Classify:** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
*(Repeat for CWRU and DEMADICS)*

---

## Phase 3: Scalability & Generalization (Secondary Datasets)
**Goal:** Take the Top 3 models from Phase 2 (based on highest Aug Classify scores) and deploy them on the massive generalization datasets (N-CMAPSS, FEMTO, XJTU-SY, Paderborn).

### Rank #1 Model (Expected: FlowMatch-PdM)
- [ ] **Train Gen (N-CMAPSS):** `python train_generator.py --track engine_rul --dataset N-CMAPSS --model <TOP_1>`
- [ ] **Eval & Aug Classify (N-CMAPSS):** Run `run_evaluation.py` then `train_classifier_aug.py` with the `<RUN_ID>`
- [ ] **Train Gen (FEMTO):** `python train_generator.py --track bearing_rul --dataset FEMTO --model <TOP_1>`
- [ ] **Eval & Aug Classify (FEMTO):** Run `run_evaluation.py` then `train_classifier_aug.py`
- [ ] **Train Gen (XJTU-SY):** `python train_generator.py --track bearing_rul --dataset XJTU-SY --model <TOP_1>`
- [ ] **Eval & Aug Classify (XJTU-SY):** Run `run_evaluation.py` then `train_classifier_aug.py`
- [ ] **Train Gen (Paderborn):** `python train_generator.py --track bearing_fault --dataset PADERBORN --model <TOP_1>`
- [ ] **Eval & Aug Classify (Paderborn):** Run `run_evaluation.py` then `train_classifier_aug.py`

### Rank #2 Model
- [ ] **Execute Pipeline:** Repeat the 8 steps above using `--model <TOP_2>`

### Rank #3 Model
- [ ] **Execute Pipeline:** Repeat the 8 steps above using `--model <TOP_3>`

---

## 🔬 Phase 4: The Ablation Study (Deconstructing the Magic)
**Goal:** Prove why our architecture works by selectively disabling our novel components on a representative dataset (e.g., CMAPSS).

- [ ] **w/o Harmonic Prior (Pure Noise):** `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch --ablation no_prior` 
  *(Then Eval & Aug Classify)*
- [ ] **w/o TCCM Loss (No Manifold Constraint):** `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch --ablation no_tccm` 
  *(Then Eval & Aug Classify)*
- [ ] **w/o LAP (No Dynamic Pruning):** `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch --ablation no_lap` 
  *(Then Eval & Aug Classify)*

---

## 🎛️ Phase 5: Final Polish & Hyperparameter Sweeps (W&B)
**Goal:** Squeeze out the final 2% accuracy for the final IEEE manuscript tables via Bayesian Optimization.

- [ ] **Initialize Sweep:** `wandb sweep configs/sweep_config.yaml`
- [ ] **Run W&B Agent:** `wandb agent <sweep_id_provided_by_step_1>`