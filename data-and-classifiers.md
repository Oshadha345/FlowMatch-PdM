# 🎯 Primary Objective
Build the foundational data pipelines for the NASA C-MAPSS and CWRU datasets and establish the baseline classification metrics that all generative models must try to beat.

---

## 📋 Core Responsibilities
- 📥 Download, clean, and preprocess the NASA C-MAPSS and CWRU datasets
- 📦 Write `data_loader.py` so everyone else can easily import the training tensors
- 🤖 Implement and train the baseline classifiers: 1D-CNN for CWRU and LSTM for C-MAPSS
- 📊 Save the baseline classification metrics (the "floor" to beat)

---

## ✅ Your Specific Tasks

### 📊 Data Acquisition & Preprocessing
- Download the C-MAPSS and CWRU datasets
- Implement sliding window techniques for the time-series data
- Handle missing values and normalize the sensor readings

### 🔧 PyTorch DataLoaders
- Create robust, reusable PyTorch Dataset and DataLoader classes
- The rest of the team will import these to train their generative models

### 🧠 Baseline Classifiers
- Implement an **LSTM** optimized for the slow degradation sequences in C-MAPSS
- Implement a **1D-CNN** optimized for the high-frequency spatial-temporal vibration data in CWRU

### 📈 Baseline Metrics
- Train these classifiers on raw (unaugmented) data
- Record the baseline $F_1$-score, Precision, and Recall

---

## 📦 Key Deliverables
- ✔️ **`src/data_loader.py`**: Must contain functions like `get_cmapss_loaders()` and `get_cwru_loaders()`
- ✔️ **`src/classifiers.py`**: Contains the PyTorch classes for `CNN1D` and `LSTMClassifier`
- ✔️ **`notebooks/01_EDA_and_Baselines.ipynb`**: A notebook documenting the baseline classifier performance

---

## ⚠️ Dependencies
🔴 **You are the blocker for the team.** Members 3, 4, and 5 cannot train their models until `data_loader.py` is merged into `dev`.