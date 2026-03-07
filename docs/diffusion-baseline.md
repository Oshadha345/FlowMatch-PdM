# 🎯 Advanced Generative Baseline (Diffusion)

## 📌 Core Responsibilities

- 🔬 **Focus entirely on implementing Diffusion-TS** — Diffusion models are mathematically and computationally heavy, requiring dedicated attention.
- ⏱️ **Tune the noise scheduling and temporal dependencies**
- 📊 **Generate synthetic diffusion data and evaluate it**

## 🎯 Primary Objective

Implement the most complex baseline, **Diffusion-TS**, which is currently state-of-the-art in many generative tasks, providing the ultimate benchmark for the novel CFM approach.

## ✅ Your Specific Tasks

### 🏗️ Diffusion Architecture
Implement the forward (noise-adding) and reverse (denoising) diffusion processes specifically tailored for time-series data (Diffusion-TS).

### 📈 Noise Scheduling
Implement and tune the noise schedule (e.g., linear or cosine) to ensure the model captures:
- High-frequency nuances of the CWRU dataset
- Long-term dependencies of C-MAPSS

### 🔄 Conditional Sampling
Ensure the reverse diffusion process is conditioned on fault labels to generate class-specific synthetic data.

### 🔍 Evaluation
Output synthetic data and run it through the evaluation suite.

## 📦 Key Deliverables

| Deliverable | Description |
|---|---|
| `src/diffusion.py` | Diffusion-TS architecture and sampling methods |
| `notebooks/03_Diffusion_Training.ipynb` | Complex training loop and generation times documentation |

## 🔗 Dependencies

- ✏️ Member 1's `data_loader.py`
- ✏️ Member 2's `evaluation.py`