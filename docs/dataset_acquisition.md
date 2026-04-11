# Dataset Acquisition Guide

This document lists every dataset used in the FlowMatch-PdM experiment,
with exact download instructions, expected paths, and license information.

---

## Primary Datasets (Phase 0 + Phase 2)

### 1. C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

| Field | Value |
|-------|-------|
| **Track** | `engine_rul` |
| **Source** | NASA Prognostics Data Repository |
| **License** | Public domain (US Government work) |
| **Handled by** | `rul-datasets` Python package (auto-download) |

The `rul_data_loader.py` uses `rul_datasets.CmapssReader` which automatically
downloads and caches the data under `~/.rul-datasets/`. No manual download needed.

```bash
# Verify availability
python -c "import rul_datasets; r = rul_datasets.CmapssReader(fd=1); r.prepare_data(); print('OK')"
```

### 2. CWRU (Case Western Reserve University Bearing Data)

| Field | Value |
|-------|-------|
| **Track** | `bearing_fault` |
| **Source** | https://engineering.case.edu/bearingdatacenter |
| **License** | Academic use |
| **Expected path** | `datasets/processed/cwru/` (X_train.npy, y_train.npy, etc.) |

Pre-processing is done by `notebooks/01_dataset_analysis.ipynb`. Raw `.mat` files
should be placed under `datasets/cwru_raw/raw/`. The notebook windows, z-scores,
and splits the data into train/val/test `.npy` files.

If raw files are missing, download the 12kHz Drive End (DE) vibration data
for all four fault sizes (Normal, B007, B014, B021, IR007, IR014, IR021,
OR007@6, OR014@6, OR021@6) from the CWRU Bearing Data Center website.

### 3. DEMADICS (Damadics Benchmark)

| Field | Value |
|-------|-------|
| **Track** | `bearing_fault` |
| **Source** | Lublin University of Technology / DAMADICS consortium |
| **License** | Research use |
| **Expected path** | `datasets/processed/demadics/` (X_train.npy, y_train.npy, etc.) |

Pre-processing is done by `notebooks/01_dataset_analysis.ipynb` with helper
`src/utils/demadics_preprocessing.py`. Raw text files should be in
`datasets/damadics_raw/Lublin_all_data/`.

5 classes: Normal (0), f16 (1), f17 (2), f18 (3), f19 (4).

---

## Secondary Datasets (Phase 3 only)

### 4. N-CMAPSS (New CMAPSS)

| Field | Value |
|-------|-------|
| **Track** | `engine_rul` |
| **Source** | NASA Prognostics Data Repository |
| **License** | Public domain (US Government work) |
| **URL** | https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6 |
| **Handled by** | `rul-datasets` (`NCmapssReader`, auto-download) |

```bash
python -c "import rul_datasets; r = rul_datasets.NCmapssReader(fd=1); r.prepare_data(); print('OK')"
```

### 5. FEMTO (PRONOSTIA)

| Field | Value |
|-------|-------|
| **Track** | `bearing_rul` |
| **Source** | IEEE PHM 2012 Prognostic Challenge |
| **License** | Research use |
| **URL** | https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/ |
| **Handled by** | `rul-datasets` (`FemtoReader`, auto-download) |

```bash
python -c "import rul_datasets; r = rul_datasets.FemtoReader(fd=1); r.prepare_data(); print('OK')"
```

### 6. XJTU-SY (Xi'an Jiaotong University Bearing Dataset)

| Field | Value |
|-------|-------|
| **Track** | `bearing_rul` |
| **Source** | Xi'an Jiaotong University & Changxing Sumyoung Technology |
| **License** | Research use |
| **URL** | https://biaowang.tech/xjtu-sy-bearing-datasets/ |
| **Handled by** | `rul-datasets` (`XjtuSyReader`, auto-download) |

```bash
python -c "import rul_datasets; r = rul_datasets.XjtuSyReader(fd=1); r.prepare_data(); print('OK')"
```

### 7. Paderborn (Paderborn University Bearing Dataset)

| Field | Value |
|-------|-------|
| **Track** | `bearing_fault` |
| **Source** | Paderborn University, Chair of Design and Drive Technology |
| **License** | CC BY 4.0 |
| **URL** | https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter |
| **Expected path** | `datasets/processed/paderborn/` (X_train.npy, etc.) |

Pre-processing is done by `notebooks/01_dataset_analysis.ipynb`. Raw `.mat` files
should be in `datasets/paderborn_raw/mat_files/<condition_folder>/`.

31 bearing conditions (6 healthy K001-K006 + 25 damage classes).

---

## Quick Verification

Run all checks at once:

```bash
conda activate flowmatch_pdm
cd FlowMatch-PdM

# RUL datasets (auto-managed by rul-datasets)
python -c "
import rul_datasets
for name, cls in [('CMAPSS', rul_datasets.CmapssReader),
                  ('N-CMAPSS', rul_datasets.NCmapssReader),
                  ('FEMTO', rul_datasets.FemtoReader),
                  ('XJTU-SY', rul_datasets.XjtuSyReader)]:
    try:
        r = cls(fd=1); r.prepare_data(); print(f'{name}: OK')
    except Exception as e:
        print(f'{name}: FAIL - {e}')
"

# Classification datasets (preprocessed .npy required)
python -c "
from pathlib import Path
for ds in ['cwru', 'paderborn', 'demadics']:
    p = Path('datasets/processed') / ds / 'X_train.npy'
    print(f'{ds}: {\"OK\" if p.exists() else \"MISSING - run notebook preprocessing\"}')"
```

## Full Preflight Notebook

The authoritative dataset check is the preprocessing notebook:

```bash
python -m jupyter nbconvert --to notebook --execute \
  notebooks/01_dataset_analysis.ipynb \
  --output 01_dataset_analysis.executed.ipynb \
  --output-dir notebooks \
  --ExecutePreprocessor.timeout=0 \
  --ExecutePreprocessor.kernel_name=python3
```

It must end with:
- `Supported loader readiness: GO`
- `Full requested roster readiness: GO`
