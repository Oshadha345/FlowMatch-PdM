# 🚀 FlowMatch-PdM: Master Experiment Plan

Welcome to the lab! This document contains the exact sequential commands to execute our research plan. Check these off as you complete them to track your progress accurately.

---

## Phase 0: Baseline Classifiers (No Augmentation)

**Goal:** Establish the pure baseline performance of each classifier before introducing any synthetic data.

**Action:** Run these commands from the root directory.

---

### Engine RUL Track (LSTM)

- [ ] **Train:**  
`python train_classifier.py --track engine_rul --dataset CMAPSS`

- [ ] **Evaluate:**  
`python run_evaluation.py --track engine_rul --dataset CMAPSS --model baseline`

- [ ] **Train:**  
`python train_classifier.py --track engine_rul --dataset N-CMAPSS`

- [ ] **Evaluate:**  
`python run_evaluation.py --track engine_rul --dataset N-CMAPSS --model baseline`

---

### Bearing RUL Track (LSTM)

- [ ] **Train:**  
`python train_classifier.py --track bearing_rul --dataset FEMTO`

- [ ] **Evaluate:**  
`python run_evaluation.py --track bearing_rul --dataset FEMTO --model baseline`

- [ ] **Train:**  
`python train_classifier.py --track bearing_rul --dataset XJTU-SY`

- [ ] **Evaluate:**  
`python run_evaluation.py --track bearing_rul --dataset XJTU-SY --model baseline`

---

### Bearing Fault Classification Track (1D-CNN)

- [ ] **Train:**  
`python train_classifier.py --track bearing_fault --dataset CWRU`

- [ ] **Evaluate:**  
`python run_evaluation.py --track bearing_fault --dataset CWRU --model baseline`

- [ ] **Train:**  
`python train_classifier.py --track bearing_fault --dataset PADERBORN`

- [ ] **Evaluate:**  
`python run_evaluation.py --track bearing_fault --dataset PADERBORN --model baseline`

---

## Phase 1: Classical Augmentation Baselines

**Goal:** Test standard statistical augmentation techniques on heavyweight datasets (CMAPSS & CWRU).

*(Uses the `train_classifier_aug.py` script to inject augmented samples during training.)*

---

### 1A: SMOTE (Synthetic Minority Over-sampling Technique)

#### CMAPSS / LSTM

- [ ] **Train + Augment:**  
`python train_classifier_aug.py --track engine_rul --dataset CMAPSS --aug smote`

- [ ] **Evaluate:**  
`python run_evaluation.py --track engine_rul --dataset CMAPSS --model smote`

#### CWRU / 1D-CNN

- [ ] **Train + Augment:**  
`python train_classifier_aug.py --track bearing_fault --dataset CWRU --aug smote`

- [ ] **Evaluate:**  
`python run_evaluation.py --track bearing_fault --dataset CWRU --model smote`

---

### 1B: Jittering (Gaussian Noise Addition)

#### CMAPSS / LSTM

- [ ] **Train + Augment:**  
`python train_classifier_aug.py --track engine_rul --dataset CMAPSS --aug noise`

- [ ] **Evaluate:**  
`python run_evaluation.py --track engine_rul --dataset CMAPSS --model noise`

#### CWRU / 1D-CNN

- [ ] **Train + Augment:**  
`python train_classifier_aug.py --track bearing_fault --dataset CWRU --aug noise`

- [ ] **Evaluate:**  
`python run_evaluation.py --track bearing_fault --dataset CWRU --model noise`

---

## Phase 2: Generative Training & Evaluation (Heavyweight Datasets)
**Goal:** Train, evaluate, and test the downstream classification improvement for each deep generative model on CMAPSS and CWRU, one by one.

### 2A: TimeVAE
- [ ] **Train Generator:** `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeVAE`
- [ ] **Train Generator:** `python train_generator.py --track bearing_fault --dataset CWRU --model TimeVAE`
- [ ] **Evaluate:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model TimeVAE --run_id <RUN_ID>`
- [ ] **Evaluate:** `python run_evaluation.py --track bearing_fault --dataset CWRU --model TimeVAE --run_id <RUN_ID>`
- [ ] **Aug Classify (LSTM):** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
- [ ] **Aug Classify (1D-CNN):** `python train_classifier_aug.py --track bearing_fault --dataset CWRU --run_id <RUN_ID>`

### 2B: TimeGAN

- [ ] **Train Generator:** `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeGAN`
- [ ] **Train Generator:** `python train_generator.py --track bearing_fault --dataset CWRU --model TimeGAN`
- [ ] **Evaluate:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model TimeGAN --run_id <RUN_ID>`
- [ ] **Evaluate:** `python run_evaluation.py --track bearing_fault --dataset CWRU --model TimeGAN --run_id <RUN_ID>`
- [ ] **Aug Classify (LSTM):** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
- [ ] **Aug Classify (1D-CNN):** `python train_classifier_aug.py --track bearing_fault --dataset CWRU --run_id <RUN_ID>`

### 2C: Diffusion-TS
- [ ] **Train Generator:** `python train_generator.py --track engine_rul --dataset CMAPSS --model DiffusionTS`
- [ ] **Train Generator:** `python train_generator.py --track bearing_fault --dataset CWRU --model DiffusionTS`
- [ ] **Evaluate:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model DiffusionTS --run_id <RUN_ID>`
- [ ] **Evaluate:** `python run_evaluation.py --track bearing_fault --dataset CWRU --model DiffusionTS --run_id <RUN_ID>`
- [ ] **Aug Classify (LSTM):** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
- [ ] **Aug Classify (1D-CNN):** `python train_classifier_aug.py --track bearing_fault --dataset CWRU --run_id <RUN_ID>`

### 2D: TimeFlow (Standard Flow Matching)
- [ ] **Train Generator:** `python train_generator.py --track engine_rul --dataset CMAPSS --model TimeFlow`
- [ ] **Train Generator:** `python train_generator.py --track bearing_fault --dataset CWRU --model TimeFlow`
- [ ] **Evaluate:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model TimeFlow --run_id <RUN_ID>`
- [ ] **Evaluate:** `python run_evaluation.py --track bearing_fault --dataset CWRU --model TimeFlow --run_id <RUN_ID>`
- [ ] **Aug Classify (LSTM):** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
- [ ] **Aug Classify (1D-CNN):** `python train_classifier_aug.py --track bearing_fault --dataset CWRU --run_id <RUN_ID>`

### 2E: FlowMatch-PdM (Our Architecture)
- [ ] **Train Generator:** `python train_generator.py --track engine_rul --dataset CMAPSS --model FlowMatch`
- [ ] **Train Generator:** `python train_generator.py --track bearing_fault --dataset CWRU --model FlowMatch`
- [ ] **Evaluate:** `python run_evaluation.py --track engine_rul --dataset CMAPSS --model FlowMatch --run_id <RUN_ID>`
- [ ] **Evaluate:** `python run_evaluation.py --track bearing_fault --dataset CWRU --model FlowMatch --run_id <RUN_ID>`
- [ ] **Aug Classify (LSTM):** `python train_classifier_aug.py --track engine_rul --dataset CMAPSS --run_id <RUN_ID>`
- [ ] **Aug Classify (1D-CNN):** `python train_classifier_aug.py --track bearing_fault --dataset CWRU --run_id <RUN_ID>`

---

## Phase 3: Generalization (Top 3 Generative Models)
**Goal:** Compare the evaluation results from Phase 2. Select the absolute Top 3 models and run them on the remaining generalization datasets (N-CMAPSS, FEMTO, XJTU-SY, Paderborn).

### Rank #1 Model (Expected: FlowMatch-PdM)
- [ ] **Train Generators:** Run `train_generator.py` for N-CMAPSS, FEMTO, XJTU-SY, and Paderborn using `--model <TOP_1_MODEL>`.
- [ ] **Evaluate:** Run `run_evaluation.py` for all 4 runs.
- [ ] **Aug Classify (LSTM):** Run `train_classifier_aug.py` for N-CMAPSS, FEMTO, and XJTU-SY using the respective `<RUN_ID>`.
- [ ] **Aug Classify (1D-CNN):** Run `train_classifier_aug.py` for Paderborn using the respective `<RUN_ID>`.

### Rank #2 Model (Expected: TimeFlow or DiffusionTS)
- [ ] **Train Generators:** Run `train_generator.py` for N-CMAPSS, FEMTO, XJTU-SY, and Paderborn using `--model <TOP_2_MODEL>`.
- [ ] **Evaluate:** Run `run_evaluation.py` for all 4 runs.
- [ ] **Aug Classify (LSTM):** Run `train_classifier_aug.py` for N-CMAPSS, FEMTO, and XJTU-SY using the respective `<RUN_ID>`.
- [ ] **Aug Classify (1D-CNN):** Run `train_classifier_aug.py` for Paderborn using the respective `<RUN_ID>`.

### Rank #3 Model
- [ ] **Train Generators:** Run `train_generator.py` for N-CMAPSS, FEMTO, XJTU-SY, and Paderborn using `--model <TOP_3_MODEL>`.
- [ ] **Evaluate:** Run `run_evaluation.py` for all 4 runs.
- [ ] **Aug Classify (LSTM):** Run `train_classifier_aug.py` for N-CMAPSS, FEMTO, and XJTU-SY using the respective `<RUN_ID>`.
- [ ] **Aug Classify (1D-CNN):** Run `train_classifier_aug.py` for Paderborn using the respective `<RUN_ID>`.

---


## 🎛️ Phase 4: Final Polish & Hyperparameter Sweeps (W&B)
**Goal:** Squeeze out the final 2% accuracy for our FlowMatch-PdM model for the final submission.
1. **Initialize Sweep:** `wandb sweep configs/sweep_config.yaml`
2. **Run W&B Agent:** `wandb agent <sweep_id_provided_by_step_1>`