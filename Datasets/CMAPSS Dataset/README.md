# C-MAPSS Dataset for FlowMatch-PdM

This document defines how the C-MAPSS data is used in the **FlowMatch-PdM** project from a data engineering and deep learning pipeline perspective.

## 1) Purpose in this project

For C-MAPSS, we model multivariate run-to-failure engine trajectories and use synthetic data generation to improve downstream predictive performance.

- **Primary generator (ours):** Conditional Flow Matching (CFM)
- **Generative baselines:** TimeGAN, Diffusion-TS, TimeVAE
- **Statistical baseline:** SMOTE
- **Signal-domain augmentation baseline:** Jittering / Time-Warping
- **Downstream classifier for C-MAPSS:** LSTM (long-term temporal dependency modeling)

## 2) Dataset summary

The original C-MAPSS benchmark contains four subsets with different operating conditions and fault complexity:

| Subset | Train trajectories | Test trajectories | Operating conditions | Fault modes |
|---|---:|---:|---|---|
| FD001 | 100 | 100 | 1 (Sea Level) | 1 (HPC Degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC Degradation) |
| FD003 | 100 | 100 | 1 (Sea Level) | 2 (HPC + Fan Degradation) |
| FD004 | 248 | 249 | 6 | 2 (HPC + Fan Degradation) |

Each row is one engine cycle snapshot. Data is multivariate, noisy, and collected over complete degradation trajectories (train) and truncated trajectories (test).

## 3) Files and expected semantics

For each subset `FD00X`:

- `train_FD00X.txt`: full run-to-failure trajectories
- `test_FD00X.txt`: trajectories ending before failure
- `RUL_FD00X.txt`: true remaining useful life labels for each test engine at its final observed cycle

Reference file in this folder:

- `readme.txt`: original NASA competition-style description used as the source of truth for raw format and benchmark context

## 4) Raw schema (26 columns)

The raw text files are space-delimited and map to:

1. unit number (engine id)
2. time in cycles
3. operational setting 1
4. operational setting 2
5. operational setting 3
6. sensor measurement 1
7. sensor measurement 2
8. ...
26. sensor measurement 26

## 5) Data engineering contract

To keep the pipeline reproducible across all augmentation methods, follow this processing contract.

### 5.1 Ingestion

- Parse with whitespace delimiter and robust handling of trailing spaces.
- Assign deterministic column names (`unit_id`, `cycle`, `op_setting_1..3`, `sensor_1..21` or full sensor index naming based on parser mapping).
- Enforce dtypes:
	- `unit_id`, `cycle` as integers
	- operational settings and sensors as float32

### 5.2 Cleaning and ordering

- Sort by (`unit_id`, `cycle`) strictly.
- Drop empty columns introduced by irregular spacing.
- Remove exact duplicate rows if any appear after parse.
- Validate monotonic cycle progression per `unit_id`.

### 5.3 Label construction

- **Train RUL:** for each engine trajectory,
	- `RUL = max_cycle(unit_id) - cycle`
- **Test RUL target:** use `RUL_FD00X.txt` value aligned to each engine’s last observed cycle.
- Optional capped-RUL target (common in literature): `RUL_capped = min(RUL, cap_value)`; keep both raw and capped when experimenting.

### 5.4 Feature processing

- Apply normalization fit **only on training split** (per subset).
- Persist scaler artifacts for reproducibility.
- Keep operational settings and sensors available for model ablations.
- Track dropped/constant features in metadata rather than silently removing them.

### 5.5 Windowing

- Convert trajectories into fixed-length sequences for model input.
- Typical config fields:
	- `window_size`
	- `stride`
	- `horizon`
	- `padding_mode` (if sequence shorter than window)
- Preserve mapping from each window to (`unit_id`, end_cycle, label).

## 6) DL pipeline role of CFM and baselines

### 6.1 Core idea

CFM is used to generate synthetic multivariate sensor windows conditioned on context (e.g., operating conditions, degradation stage, or class/RUL bin strategy from experiment config).

### 6.2 Benchmark matrix

Synthetic data generation methods to compare:

1. Conditional Flow Matching (FlowMatch-PdM, primary)
2. TimeGAN
3. Diffusion-TS
4. TimeVAE
5. SMOTE (tabular/statistical baseline after feature representation choice)
6. Jittering / Time-Warping (time-domain augmentation baseline)

Each method should plug into the same downstream training and evaluation protocol to ensure fair comparison.

### 6.3 Downstream model for C-MAPSS

- **Model:** LSTM
- **Why:** robust for long-term temporal dependencies in degradation trajectories
- **Input:** windowed multivariate sequences
- **Output (task-dependent):** classification target defined by experiment (e.g., health state / RUL bin)

## 7) Recommended experiment protocol (C-MAPSS)

1. Select subset(s): FD001–FD004.
2. Build baseline real-only training set.
3. Train each synthetic generator on train split only.
4. Generate synthetic windows with matched class/condition strategy.
5. Construct augmented training sets (real + synthetic) at controlled ratios.
6. Train identical LSTM downstream model per method.
7. Evaluate on unchanged real test split.
8. Report both predictive metrics and synthetic quality diagnostics.

## 8) Data quality checks (must-pass)

- No leakage between train/test engines.
- No future-cycle leakage inside windows.
- RUL alignment sanity check for every test engine.
- Per-feature min/max/mean drift report before and after augmentation.
- Sequence length and label distribution report per subset.

## 9) Reproducibility requirements

- Fixed random seeds for:
	- data split and window sampling
	- generator training
	- synthetic sampling
	- downstream LSTM training
- Version and store:
	- raw source hashes
	- preprocessing config
	- scaler parameters
	- experiment config used for each benchmark run

## 10) Suggested artifact structure

Recommended output structure for data engineering handoff:

- `processed/cmapss/FD00X/train_windows.*`
- `processed/cmapss/FD00X/val_windows.*`
- `processed/cmapss/FD00X/test_windows.*`
- `artifacts/cmapss/FD00X/scalers/*`
- `artifacts/cmapss/FD00X/metadata.json`
- `artifacts/cmapss/FD00X/synthetic/<method_name>/*`

Use any storage format your pipeline standardizes on (Parquet/NPZ/PT), but keep schema and metadata consistent across all methods.

## 11) Citation

A. Saxena, K. Goebel, D. Simon, and N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, Oct 2008.

