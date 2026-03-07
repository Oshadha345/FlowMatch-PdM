# 🌊 Conditional Flow Matching

## 🎯 Primary Objective
Build, train, and optimize the core novelty of this research paper: the **Conditional Flow Matching (CFM)** architecture for industrial time-series data.

## 📋 Your Specific Tasks

- **🧠 CFM Architecture**: Implement the neural network to regress the vector field. You will likely need to design a temporal-aware backbone (like a 1D-UNet or a sequence model) to handle the sensor data.

- **⚙️ ODE Solver Integration**: Integrate an ODE solver (e.g., using `torchdiffeq`) to simulate the probability flow ODE during the sampling/inference phase.

- **🏷️ Label Conditioning**: Strictly enforce fault-label conditioning so the CFM model can map the base distribution (Gaussian noise) to the specific target distribution (e.g., a specific C-MAPSS degradation level).

- **🏆 Beat the Baselines**: Iteratively tune your model until your TSTR and Context-FID scores surpass the metrics recorded by Members 3 and 4.

## 📦 Key Deliverables

- **`src/flow_matching.py`**: The core code for the novel architecture.
- **`notebooks/04_CFM_Training_and_Results.ipynb`**: The primary notebook that will form the basis of the research paper's results section.

## 🔗 Dependencies
Requires Member 1's `data_loader.py` and Member 2's `evaluation.py`. You will also be directly comparing your metrics against the results from Members 3 and 4.

## ✅ Core Responsibilities

1. Build the primary **Conditional Flow Matching (CFM)** architecture.
2. Implement the vector field regression and the ODE solver for inference.
3. Ensure the model correctly conditions on the fault labels from the datasets.
4. Iterate and tune the CFM model to beat the metrics established by Members 3 and 4.