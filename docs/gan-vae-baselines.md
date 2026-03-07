# 🎯 Standard Deep Generative Baselines

**Role:** Standard Deep Generative Baselines

**Primary Objective:** Implement the foundational deep learning generative baselines (TimeGAN and TimeVAE) to establish a high bar for the novel Flow Matching model to beat.

---

## 📋 Your Specific Tasks

### Core Responsibilities

- ✅ Implement TimeGAN and TimeVAE
- ✅ Write the training loops for these models using Member 1's data loaders
- ✅ Generate synthetic data from these models and pass them through Member 2's evaluation scripts to record baseline performance

### Architecture Implementation 🏗️

Build the network architectures for:
- **TimeGAN:** Generator, discriminator, embedder, and recovery networks
- **TimeVAE:** Encoder, decoder, and latent space sampling

### Training Loops 🔄

- Write the training scripts that import Member 1's DataLoaders
- Ensure the models correctly condition on the fault labels (e.g., generating specific CWRU fault types)

### Data Generation 📊

- Once trained, write a sampling function that outputs synthetic datasets in the exact shape and format expected by Member 2's evaluation scripts

### Evaluation 📈

- Run your synthetic data through Member 2's `evaluation.py`
- Record the TSTR and Context-FID scores

---

## 📦 Key Deliverables

| Deliverable | Description |
|---|---|
| `src/baselines.py` (Part 2) | Add TimeGAN and TimeVAE classes |
| `notebooks/02_TimeGAN_TimeVAE_Training.ipynb` | Document training losses and hyperparameter choices |

---

## 🔗 Dependencies

- 📥 **From Member 1:** `src/data_loader.py` (required to start training)
- 📥 **From Member 2:** `src/evaluation.py` (required to test your results)