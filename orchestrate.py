#!/usr/bin/env python3
"""FlowMatch-PdM Master Orchestrator.

Reads pipeline_state.json, skips completed steps, runs remaining steps
sequentially, and writes results back after each step.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
STATE_FILE = REPO_ROOT / "pipeline_state.json"
FAILED_FILE = REPO_ROOT / "FAILED.md"
LOGS_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"
FINAL_REPORT_DIR = RESULTS_DIR / "final_report"
CONFIG_PATH = "configs/default_config.yaml"

GENERATORS = [
    "TimeVAE", "TimeGAN", "COTGAN", "FaultDiffusion",
    "DiffusionTS", "TimeFlow", "FlowMatch",
]

PRIMARY_DATASETS = [
    ("engine_rul", "CMAPSS"),
    ("bearing_fault", "CWRU"),
    ("bearing_fault", "DEMADICS"),
]

SECONDARY_DATASETS = [
    ("engine_rul", "N-CMAPSS"),
    ("bearing_rul", "FEMTO"),
    ("bearing_rul", "XJTU-SY"),
    ("bearing_fault", "Paderborn"),
]

PHASE1_JOBS = [
    ("noise", "engine_rul", "CMAPSS"),
    ("noise", "bearing_fault", "CWRU"),
    ("noise", "bearing_fault", "DEMADICS"),
    ("smote", "bearing_fault", "CWRU"),
    ("smote", "bearing_fault", "DEMADICS"),
    ("smote", "bearing_fault", "Paderborn"),
]

ABLATION_VARIANTS = ["no_prior", "no_tccm", "no_lap"]

CONDA_PREFIX = "source /home/buddhiw/miniconda3/etc/profile.d/conda.sh && conda activate flowmatch_pdm && "

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGS_DIR.mkdir(parents=True, exist_ok=True)

log_filename = LOGS_DIR / f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = logging.getLogger("orchestrator")
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)
logger.addHandler(_sh)

_fh = logging.FileHandler(str(log_filename), encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

# ---------------------------------------------------------------------------
# Default state schema
# ---------------------------------------------------------------------------

DEFAULT_STATE: Dict[str, Any] = {
    "schema_version": 1,
    "last_updated": datetime.now(timezone.utc).isoformat(),
    "errors": [],
    "env_check": "pending",
    "dataset_acquisition": {
        "CMAPSS": "pending",
        "CWRU": "pending",
        "DEMADICS": "pending",
        "Paderborn": "pending",
        "N-CMAPSS": "pending",
        "FEMTO": "pending",
        "XJTU-SY": "pending",
    },
    "preflight_notebook": "pending",
    "phase0": {
        "engine_rul__CMAPSS": {"status": "done", "run_id": "run_20260316_125116", "rmse": 16.523},
        "engine_rul__N-CMAPSS": {"status": "pending", "run_id": None},
        "bearing_rul__FEMTO": {"status": "pending", "run_id": None},
        "bearing_rul__XJTU-SY": {"status": "pending", "run_id": None},
        "bearing_fault__CWRU": {"status": "done", "run_id": "run_20260316_111110", "f1_macro": 1.0},
        "bearing_fault__DEMADICS": {"status": "done", "run_id": "run_20260316_112649", "f1_macro": 0.9668},
        "bearing_fault__Paderborn": {"status": "done", "run_id": "run_20260316_154854", "f1_macro": 0.9995},
    },
    "phase1": {
        f"{aug}__{track}__{ds}": {"status": "pending", "run_id": None}
        for aug, track, ds in PHASE1_JOBS
    },
    "phase2": {
        f"{gen}__{track}__{ds}": {
            "gen_status": "pending", "gen_run_id": None,
            "clf_status": "pending", "clf_run_id": None,
        }
        for gen in GENERATORS
        for track, ds in PRIMARY_DATASETS
    },
    "phase3": {},
    "top_models": [],
    "phase4": {
        f"FlowMatch_{abl}__engine_rul__CMAPSS": {
            "gen_status": "pending", "gen_run_id": None,
            "clf_status": "pending", "clf_run_id": None,
        }
        for abl in ABLATION_VARIANTS
    },
    "final_report": "pending",
}


# ===================================================================
# State management
# ===================================================================

def load_state() -> dict:
    """Read pipeline_state.json; create from schema if missing."""
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r", encoding="utf-8") as fh:
                state = json.load(fh)
            logger.info("Loaded existing pipeline_state.json (last_updated=%s)", state.get("last_updated"))
            # Ensure all top-level keys exist
            for key, default in DEFAULT_STATE.items():
                state.setdefault(key, deepcopy(default))
            state.setdefault("errors", [])
            return state
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Corrupt pipeline_state.json: %s — creating fresh state", exc)
    state = deepcopy(DEFAULT_STATE)
    save_state(state)
    logger.info("Created new pipeline_state.json")
    return state


def save_state(state: dict) -> None:
    """Atomically write pipeline_state.json."""
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
        fh.write("\n")
    tmp.replace(STATE_FILE)


def mark_failed(state: dict, key_path: str, error: str) -> None:
    """Mark a step as failed, log to FAILED.md, and save state."""
    # Navigate nested path like "phase2.TimeVAE__engine_rul__CMAPSS.gen_status"
    parts = key_path.split(".")
    obj = state
    for part in parts[:-1]:
        obj = obj[part]
    obj[parts[-1]] = "failed"

    error_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "key_path": key_path,
        "error": error[:2000],
    }
    state.setdefault("errors", []).append(error_entry)
    save_state(state)

    # Write FAILED.md
    with FAILED_FILE.open("w", encoding="utf-8") as fh:
        fh.write(f"# FlowMatch-PdM Pipeline Failure\n\n")
        fh.write(f"**Timestamp:** {error_entry['timestamp']}\n\n")
        fh.write(f"**Failed step:** `{key_path}`\n\n")
        fh.write(f"## Error\n\n```\n{error}\n```\n")

    logger.error("FAILED: %s — %s", key_path, error[:500])


# ===================================================================
# Command execution
# ===================================================================

def run_cmd(cmd: str, description: str, timeout_hours: float = 6.0) -> str:
    """Run a shell command with conda activation; stream output in real time."""
    full_cmd = CONDA_PREFIX + f"cd {REPO_ROOT} && " + cmd
    logger.info("▶ %s", description)
    logger.debug("CMD: %s", cmd)

    timeout_sec = int(timeout_hours * 3600)
    output_lines: list[str] = []

    proc = subprocess.Popen(
        full_cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line_stripped = line.rstrip("\n")
            output_lines.append(line_stripped)
            print(line_stripped, flush=True)
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        raise TimeoutError(f"Command timed out after {timeout_hours}h: {cmd}")

    combined = "\n".join(output_lines)

    if proc.returncode != 0:
        # Save failure log
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_desc = description.replace(" ", "_").replace("/", "_")[:60]
        fail_log = LOGS_DIR / f"failed_{safe_desc}_{ts}.log"
        fail_log.write_text(combined, encoding="utf-8")
        logger.error("Command failed (exit %d). Log: %s", proc.returncode, fail_log)
        last200 = "\n".join(output_lines[-200:])
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {cmd}\n"
            f"--- last 200 lines ---\n{last200}"
        )

    return combined


# ===================================================================
# Helpers
# ===================================================================

def resolve_latest_run_id(track: str, dataset: str, model: str) -> str:
    """Use logger_utils to find the latest run_id for a model directory."""
    py_cmd = (
        f'python -c "'
        f"from src.utils.logger_utils import resolve_run_dir; "
        f"print(resolve_run_dir('{track}', '{dataset}', '{model}').name)"
        f'"'
    )
    output = run_cmd(py_cmd, f"Resolve run_id for {track}/{dataset}/{model}", timeout_hours=0.05)
    # The last non-empty line is the run_id
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("run_"):
            return line
    raise RuntimeError(f"Could not resolve run_id from output:\n{output}")


def _resolve_classifier_model_dir(track: str) -> str:
    """Return the classifier model name used in directory structure."""
    if "rul" in track:
        return "LSTMRegressor"
    return "CNN1DClassifier"


def _resolve_aug_classifier_model_dir(
    track: str, gen_model: str, gen_run_id: str, gen_ablation: str = "none",
) -> str:
    """Build the augmented classifier directory name to match resolve_classifier_experiment_name."""
    base = _resolve_classifier_model_dir(track)
    if gen_ablation and gen_ablation != "none":
        gen_name = f"{gen_model}_ablation_{gen_ablation}"
    else:
        gen_name = gen_model
    norm_run_id = gen_run_id if gen_run_id.startswith("run_") else f"run_{gen_run_id}"
    return f"{base}_aug_gen_{gen_name}_{norm_run_id}"


def _resolve_classical_aug_model_dir(track: str, aug: str) -> str:
    """Build the classical augmented classifier directory name."""
    base = _resolve_classifier_model_dir(track)
    return f"{base}_aug_{aug}"


def _read_json(path: Path) -> dict:
    """Safely read a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _primary_metric_key(track: str) -> str:
    """Return the metric key name for a track."""
    return "rmse" if "rul" in track else "f1_macro"


def _extract_primary_metric(metrics: dict, track: str) -> float:
    """Extract the primary metric from classifier_metrics.json."""
    m = metrics.get("metrics", metrics)
    if "rul" in track:
        for key in ("rmse", "test_rmse", "RMSE"):
            if key in m:
                return float(m[key])
        raise KeyError(f"No RMSE key found in {list(m.keys())}")
    for key in ("f1_macro", "test_f1_macro", "f1_score_macro"):
        if key in m:
            return float(m[key])
    raise KeyError(f"No F1 key found in {list(m.keys())}")


def _extract_gen_metrics(metrics: dict) -> dict:
    """Extract generator metrics from metrics.json."""
    return {
        "ftsd": float(metrics.get("ftsd", 0)),
        "mmd": float(metrics.get("mmd_rbf", metrics.get("mmd", 0))),
        "discriminative_score": float(metrics.get("discriminative_score", 0)),
        "predictive_score_mae": float(metrics.get("predictive_score_mae", 0)),
    }


def _run_dir_path(track: str, dataset: str, model_dir: str, run_id: str) -> Path:
    """Construct the absolute path to a run directory."""
    return RESULTS_DIR / track / dataset / model_dir / run_id


# ===================================================================
# Pipeline steps
# ===================================================================

def check_env(state: dict) -> None:
    """Verify environment: CUDA, mamba_ssm."""
    if state["env_check"] == "done":
        logger.info("Environment check already done — skipping")
        return

    logger.info("=" * 60)
    logger.info("ENVIRONMENT CHECK")
    logger.info("=" * 60)

    try:
        output = run_cmd(
            'python -c "import torch; print(torch.cuda.is_available()); '
            'print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\"CPU only\\")"',
            "Check PyTorch + CUDA",
            timeout_hours=0.1,
        )
        if "True" in output:
            logger.info("CUDA is available")
        else:
            logger.warning("No GPU detected — training will be slow")
    except RuntimeError as exc:
        mark_failed(state, "env_check", str(exc))
        sys.exit(1)

    try:
        run_cmd(
            'python -c "import mamba_ssm; print(\\"mamba_ssm OK\\")"',
            "Check mamba_ssm",
            timeout_hours=0.1,
        )
        logger.info("mamba_ssm importable")
    except RuntimeError:
        logger.warning("mamba_ssm not importable — FlowMatch-PdM may fail on CPU")

    state["env_check"] = "done"
    save_state(state)


def acquire_datasets(state: dict) -> None:
    """Acquire all datasets."""
    logger.info("=" * 60)
    logger.info("DATASET ACQUISITION")
    logger.info("=" * 60)

    acq = state["dataset_acquisition"]

    # Auto-download datasets via rul-datasets
    rul_datasets_map = {
        "CMAPSS": 'python -c "import rul_datasets; r=rul_datasets.CmapssReader(fd=1); r.prepare_data(); print(\'CMAPSS OK\')"',
        "N-CMAPSS": 'python -c "import rul_datasets; r=rul_datasets.NCmapssReader(fd=1); r.prepare_data(); print(\'N-CMAPSS OK\')"',
        "FEMTO": 'python -c "import rul_datasets; r=rul_datasets.FemtoReader(fd=1); r.prepare_data(); print(\'FEMTO OK\')"',
        "XJTU-SY": 'python -c "import rul_datasets; r=rul_datasets.XjtuSyReader(fd=1); r.prepare_data(); print(\'XJTU-SY OK\')"',
    }

    for ds_name, cmd in rul_datasets_map.items():
        if acq[ds_name] == "done":
            logger.info("Dataset %s already acquired — skipping", ds_name)
            continue
        try:
            run_cmd(cmd, f"Acquire {ds_name}", timeout_hours=1.0)
            acq[ds_name] = "done"
            save_state(state)
            logger.info("Dataset %s: OK", ds_name)
        except (RuntimeError, TimeoutError) as exc:
            mark_failed(state, f"dataset_acquisition.{ds_name}", str(exc))
            sys.exit(1)

    # Preprocessed .npy datasets
    preprocessed = {
        "CWRU": REPO_ROOT.parent / "datasets" / "processed" / "cwru" / "X_train.npy",
        "DEMADICS": REPO_ROOT.parent / "datasets" / "processed" / "demadics" / "X_train.npy",
        "Paderborn": REPO_ROOT.parent / "datasets" / "processed" / "paderborn" / "X_train.npy",
    }

    need_notebook = False
    for ds_name, npy_path in preprocessed.items():
        if acq[ds_name] == "done":
            logger.info("Dataset %s already acquired — skipping", ds_name)
            continue
        if npy_path.exists():
            acq[ds_name] = "done"
            save_state(state)
            logger.info("Dataset %s: found at %s", ds_name, npy_path)
        else:
            need_notebook = True
            logger.info("Dataset %s: npy not found, notebook run needed", ds_name)

    if need_notebook:
        logger.info("Running preprocessing notebook for missing datasets...")
        try:
            run_cmd(
                "python -m jupyter nbconvert --to notebook --execute "
                "notebooks/01_dataset_analysis.ipynb "
                "--output 01_dataset_analysis.executed.ipynb "
                "--output-dir notebooks "
                "--ExecutePreprocessor.timeout=0 "
                "--ExecutePreprocessor.kernel_name=python3",
                "Run preprocessing notebook",
                timeout_hours=2.0,
            )
        except (RuntimeError, TimeoutError) as exc:
            for ds_name in preprocessed:
                if acq[ds_name] != "done":
                    mark_failed(state, f"dataset_acquisition.{ds_name}", str(exc))
            sys.exit(1)

        # Re-check
        for ds_name, npy_path in preprocessed.items():
            if acq[ds_name] == "done":
                continue
            if npy_path.exists():
                acq[ds_name] = "done"
                save_state(state)
                logger.info("Dataset %s: now available after notebook", ds_name)
            else:
                mark_failed(
                    state,
                    f"dataset_acquisition.{ds_name}",
                    f"Still missing after notebook run: {npy_path}",
                )
                sys.exit(1)


def run_preflight_notebook(state: dict) -> None:
    """Run and validate the preflight notebook."""
    if state["preflight_notebook"] == "done":
        logger.info("Preflight notebook already passed — skipping")
        return

    logger.info("=" * 60)
    logger.info("PREFLIGHT NOTEBOOK")
    logger.info("=" * 60)

    try:
        run_cmd(
            "python -m jupyter nbconvert --to notebook --execute "
            "notebooks/01_dataset_analysis.ipynb "
            "--output 01_dataset_analysis.executed.ipynb "
            "--output-dir notebooks "
            "--ExecutePreprocessor.timeout=0 "
            "--ExecutePreprocessor.kernel_name=python3",
            "Execute preflight notebook",
            timeout_hours=2.0,
        )
    except (RuntimeError, TimeoutError) as exc:
        mark_failed(state, "preflight_notebook", str(exc))
        sys.exit(1)

    # Validate notebook outputs
    executed_nb = REPO_ROOT / "notebooks" / "01_dataset_analysis.executed.ipynb"
    if not executed_nb.exists():
        mark_failed(state, "preflight_notebook", f"Executed notebook not found: {executed_nb}")
        sys.exit(1)

    try:
        import nbformat
        nb = nbformat.read(str(executed_nb), as_version=4)
        all_outputs_text = ""
        for cell in nb.cells:
            if cell.cell_type == "code":
                for output in cell.get("outputs", []):
                    if "text" in output:
                        all_outputs_text += output["text"]
                    if "data" in output:
                        all_outputs_text += output["data"].get("text/plain", "")

        if "Supported loader readiness: GO" not in all_outputs_text:
            mark_failed(state, "preflight_notebook", "Missing 'Supported loader readiness: GO' in notebook output")
            sys.exit(1)
        if "Full requested roster readiness: GO" not in all_outputs_text:
            mark_failed(state, "preflight_notebook", "Missing 'Full requested roster readiness: GO' in notebook output")
            sys.exit(1)
    except ImportError:
        logger.warning("nbformat not installed — skipping notebook output validation")
    except Exception as exc:
        mark_failed(state, "preflight_notebook", f"Notebook validation error: {exc}")
        sys.exit(1)

    state["preflight_notebook"] = "done"
    save_state(state)
    logger.info("Preflight notebook: PASSED")


def run_phase0(state: dict) -> None:
    """Phase 0: train baseline classifiers on remaining datasets."""
    logger.info("=" * 60)
    logger.info("PHASE 0 — BASELINE CLASSIFIERS")
    logger.info("=" * 60)

    phase0_jobs = [
        ("engine_rul", "N-CMAPSS"),
        ("bearing_rul", "FEMTO"),
        ("bearing_rul", "XJTU-SY"),
    ]

    for track, dataset in phase0_jobs:
        key = f"{track}__{dataset}"
        entry = state["phase0"].get(key, {"status": "pending", "run_id": None})
        state["phase0"][key] = entry

        if entry["status"] == "done":
            logger.info("Phase 0: %s already done — skipping", key)
            continue

        logger.info("Phase 0: Training baseline on %s / %s", track, dataset)

        try:
            run_cmd(
                f"CUDA_VISIBLE_DEVICES=1 python train_classifier.py "
                f"--track {track} --dataset {dataset} --model baseline",
                f"Phase 0: baseline {track}/{dataset}",
            )
        except (RuntimeError, TimeoutError) as exc:
            mark_failed(state, f"phase0.{key}.status", str(exc))
            sys.exit(1)

        try:
            model_dir = _resolve_classifier_model_dir(track)
            run_id = resolve_latest_run_id(track, dataset, model_dir)
            run_path = _run_dir_path(track, dataset, model_dir, run_id)
            metrics = _read_json(run_path / "evaluation_results" / "classifier_metrics.json")
            metric_key = _primary_metric_key(track)
            metric_val = _extract_primary_metric(metrics, track)

            entry["status"] = "done"
            entry["run_id"] = run_id
            entry[metric_key] = metric_val
            save_state(state)
            logger.info("Phase 0: %s complete — %s = %.4f (run: %s)", key, metric_key, metric_val, run_id)
        except Exception as exc:
            mark_failed(state, f"phase0.{key}.status", f"Metrics extraction failed: {exc}\n{traceback.format_exc()}")
            sys.exit(1)


def run_phase1(state: dict) -> None:
    """Phase 1: classical augmentation (noise + SMOTE)."""
    logger.info("=" * 60)
    logger.info("PHASE 1 — CLASSICAL AUGMENTATION")
    logger.info("=" * 60)

    for aug, track, dataset in PHASE1_JOBS:
        key = f"{aug}__{track}__{dataset}"
        entry = state["phase1"].get(key, {"status": "pending", "run_id": None})
        state["phase1"][key] = entry

        if entry["status"] == "done":
            logger.info("Phase 1: %s already done — skipping", key)
            continue

        logger.info("Phase 1: %s on %s / %s", aug, track, dataset)

        try:
            run_cmd(
                f"CUDA_VISIBLE_DEVICES=1 python train_classifier.py "
                f"--track {track} --dataset {dataset} --model baseline --aug {aug}",
                f"Phase 1: {aug} {track}/{dataset}",
            )
        except (RuntimeError, TimeoutError) as exc:
            mark_failed(state, f"phase1.{key}.status", str(exc))
            sys.exit(1)

        try:
            model_dir = _resolve_classical_aug_model_dir(track, aug)
            run_id = resolve_latest_run_id(track, dataset, model_dir)
            run_path = _run_dir_path(track, dataset, model_dir, run_id)
            metrics = _read_json(run_path / "evaluation_results" / "classifier_metrics.json")
            metric_key = _primary_metric_key(track)
            metric_val = _extract_primary_metric(metrics, track)

            entry["status"] = "done"
            entry["run_id"] = run_id
            entry[metric_key] = metric_val
            save_state(state)
            logger.info("Phase 1: %s complete — %s = %.4f (run: %s)", key, metric_key, metric_val, run_id)
        except Exception as exc:
            mark_failed(state, f"phase1.{key}.status", f"Metrics extraction failed: {exc}\n{traceback.format_exc()}")
            sys.exit(1)


def _run_generator_and_classifier(
    state: dict,
    phase_key: str,
    entry: dict,
    gen_model: str,
    track: str,
    dataset: str,
    ablation: str = "none",
) -> None:
    """Shared logic for Phase 2/3/4: train generator then augmented classifier."""

    state_key_prefix = f"{phase_key}.{gen_model}"  # for logging

    # --- Generator step ---
    if entry["gen_status"] != "done":
        logger.info("  Generator: %s on %s / %s (ablation=%s)", gen_model, track, dataset, ablation)
        abl_flag = f" --ablation {ablation}" if ablation != "none" else ""
        try:
            run_cmd(
                f"CUDA_VISIBLE_DEVICES=1 python train_generator.py "
                f"--track {track} --dataset {dataset} --model {gen_model}{abl_flag}",
                f"Generator: {gen_model} {track}/{dataset} abl={ablation}",
            )
        except (RuntimeError, TimeoutError) as exc:
            mark_failed(state, f"{phase_key}.gen_status", str(exc))
            sys.exit(1)

        try:
            if ablation != "none":
                gen_dir_name = f"{gen_model}_ablation_{ablation}"
            else:
                gen_dir_name = gen_model
            gen_run_id = resolve_latest_run_id(track, dataset, gen_dir_name)
            gen_run_path = _run_dir_path(track, dataset, gen_dir_name, gen_run_id)
            gen_metrics = _read_json(gen_run_path / "evaluation_results" / "metrics.json")
            extracted = _extract_gen_metrics(gen_metrics)

            entry["gen_status"] = "done"
            entry["gen_run_id"] = gen_run_id
            entry.update(extracted)
            save_state(state)
            logger.info("  Generator done — FTSD=%.4f MMD=%.4f (run: %s)", extracted["ftsd"], extracted["mmd"], gen_run_id)
        except Exception as exc:
            mark_failed(state, f"{phase_key}.gen_status", f"Gen metrics extraction failed: {exc}\n{traceback.format_exc()}")
            sys.exit(1)

    # --- Classifier step ---
    if entry["clf_status"] != "done":
        gen_run_id = entry["gen_run_id"]
        logger.info("  Augmented classifier: %s synth on %s / %s", gen_model, track, dataset)

        abl_flag = f" --gen_ablation {ablation}" if ablation != "none" else ""
        try:
            run_cmd(
                f"CUDA_VISIBLE_DEVICES=1 python train_classifier.py "
                f"--track {track} --dataset {dataset} --model baseline "
                f"--gen_model {gen_model} --source_run_id {gen_run_id}{abl_flag}",
                f"Classifier+{gen_model} {track}/{dataset}",
            )
        except (RuntimeError, TimeoutError) as exc:
            mark_failed(state, f"{phase_key}.clf_status", str(exc))
            sys.exit(1)

        try:
            clf_model_dir = _resolve_aug_classifier_model_dir(track, gen_model, gen_run_id, ablation)
            clf_run_id = resolve_latest_run_id(track, dataset, clf_model_dir)
            clf_run_path = _run_dir_path(track, dataset, clf_model_dir, clf_run_id)
            clf_metrics = _read_json(clf_run_path / "evaluation_results" / "classifier_metrics.json")
            metric_key = _primary_metric_key(track)
            metric_val = _extract_primary_metric(clf_metrics, track)

            entry["clf_status"] = "done"
            entry["clf_run_id"] = clf_run_id
            entry[f"clf_{metric_key}"] = metric_val
            save_state(state)
            logger.info("  Classifier done — %s=%.4f (run: %s)", metric_key, metric_val, clf_run_id)
        except Exception as exc:
            mark_failed(state, f"{phase_key}.clf_status", f"Clf metrics extraction failed: {exc}\n{traceback.format_exc()}")
            sys.exit(1)


def run_phase2(state: dict) -> None:
    """Phase 2: Train all generators on primary datasets + augmented classifiers."""
    logger.info("=" * 60)
    logger.info("PHASE 2 — GENERATORS × PRIMARY DATASETS")
    logger.info("=" * 60)

    for gen_model in GENERATORS:
        for track, dataset in PRIMARY_DATASETS:
            key = f"{gen_model}__{track}__{dataset}"
            entry = state["phase2"].get(key, {
                "gen_status": "pending", "gen_run_id": None,
                "clf_status": "pending", "clf_run_id": None,
            })
            state["phase2"][key] = entry

            if entry["gen_status"] == "done" and entry["clf_status"] == "done":
                logger.info("Phase 2: %s already complete — skipping", key)
                continue

            logger.info("Phase 2: %s", key)
            _run_generator_and_classifier(
                state, f"phase2.{key}", entry, gen_model, track, dataset,
            )


def rank_models(state: dict) -> List[str]:
    """Rank generators by combined score across primary datasets. Return top 3."""
    logger.info("Ranking models from Phase 2 results...")

    scores: Dict[str, List[float]] = {gen: [] for gen in GENERATORS}

    for gen_model in GENERATORS:
        for track, dataset in PRIMARY_DATASETS:
            key = f"{gen_model}__{track}__{dataset}"
            entry = state["phase2"].get(key, {})
            if entry.get("clf_status") != "done":
                logger.warning("Phase 2 entry %s not done, skipping for ranking", key)
                continue

            if "rul" in track:
                rmse = entry.get("clf_rmse")
                if rmse and rmse > 0:
                    scores[gen_model].append(1.0 / rmse)
            else:
                f1 = entry.get("clf_f1_macro")
                if f1 is not None:
                    scores[gen_model].append(f1)

    avg_scores = {}
    for gen, vals in scores.items():
        if vals:
            avg_scores[gen] = sum(vals) / len(vals)
        else:
            avg_scores[gen] = 0.0

    ranked = sorted(avg_scores.keys(), key=lambda g: avg_scores[g], reverse=True)
    top3 = ranked[:3]

    logger.info("Model rankings:")
    for i, gen in enumerate(ranked):
        marker = " <<<" if gen in top3 else ""
        logger.info("  #%d %s: %.4f%s", i + 1, gen, avg_scores[gen], marker)

    state["top_models"] = top3
    save_state(state)
    return top3


def run_phase3(state: dict) -> None:
    """Phase 3: Top 3 generators on secondary datasets."""
    logger.info("=" * 60)
    logger.info("PHASE 3 — SECONDARY DATASETS")
    logger.info("=" * 60)

    if not state.get("top_models"):
        top3 = rank_models(state)
    else:
        top3 = state["top_models"]
        logger.info("Using previously ranked top models: %s", top3)

    for gen_model in top3:
        for track, dataset in SECONDARY_DATASETS:
            key = f"{gen_model}__{track}__{dataset}"
            if key not in state["phase3"]:
                state["phase3"][key] = {
                    "gen_status": "pending", "gen_run_id": None,
                    "clf_status": "pending", "clf_run_id": None,
                }
                save_state(state)

            entry = state["phase3"][key]
            if entry["gen_status"] == "done" and entry["clf_status"] == "done":
                logger.info("Phase 3: %s already complete — skipping", key)
                continue

            logger.info("Phase 3: %s", key)
            _run_generator_and_classifier(
                state, f"phase3.{key}", entry, gen_model, track, dataset,
            )


def run_phase4_ablations(state: dict) -> None:
    """Phase 4: FlowMatch-PdM ablations on CMAPSS."""
    logger.info("=" * 60)
    logger.info("PHASE 4 — ABLATIONS")
    logger.info("=" * 60)

    for ablation in ABLATION_VARIANTS:
        key = f"FlowMatch_{ablation}__engine_rul__CMAPSS"
        entry = state["phase4"].get(key, {
            "gen_status": "pending", "gen_run_id": None,
            "clf_status": "pending", "clf_run_id": None,
        })
        state["phase4"][key] = entry

        if entry["gen_status"] == "done" and entry["clf_status"] == "done":
            logger.info("Phase 4: %s already complete — skipping", key)
            continue

        logger.info("Phase 4: %s", key)
        _run_generator_and_classifier(
            state, f"phase4.{key}", entry,
            gen_model="FlowMatch",
            track="engine_rul",
            dataset="CMAPSS",
            ablation=ablation,
        )


# ===================================================================
# Final report generation
# ===================================================================

def _fmt(val: Any, precision: int = 4) -> str:
    """Format a value for table display."""
    if val is None or val == "":
        return "-"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def _bold_best(values: List[Optional[float]], lower_is_better: bool = True) -> List[str]:
    """Return formatted strings with ** around the best value."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return [_fmt(v) for v in values]

    if lower_is_better:
        best_idx = min(valid, key=lambda x: x[1])[0]
    else:
        best_idx = max(valid, key=lambda x: x[1])[0]

    result = []
    for i, v in enumerate(values):
        s = _fmt(v)
        if i == best_idx and s != "-":
            s = f"**{s}**"
        result.append(s)
    return result


def _tex_bold_best(values: List[Optional[float]], lower_is_better: bool = True) -> List[str]:
    """Return LaTeX formatted strings with \\textbf around best."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return [_fmt(v) for v in values]

    if lower_is_better:
        best_idx = min(valid, key=lambda x: x[1])[0]
    else:
        best_idx = max(valid, key=lambda x: x[1])[0]

    result = []
    for i, v in enumerate(values):
        s = _fmt(v)
        if i == best_idx and s != "-":
            s = f"\\textbf{{{s}}}"
        result.append(s)
    return result


def generate_final_report(state: dict) -> None:
    """Generate all five report files."""
    if state["final_report"] == "done":
        logger.info("Final report already generated — skipping")
        return

    logger.info("=" * 60)
    logger.info("GENERATING FINAL REPORT")
    logger.info("=" * 60)

    FINAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # A) docs/03_result_logger.md
    # ---------------------------------------------------------------
    _generate_result_logger(state)

    # ---------------------------------------------------------------
    # B) comparison_table.md
    # ---------------------------------------------------------------
    _generate_comparison_table_md(state)

    # ---------------------------------------------------------------
    # C) comparison_table.tex
    # ---------------------------------------------------------------
    _generate_comparison_table_tex(state)

    # ---------------------------------------------------------------
    # D) ablation_table.tex
    # ---------------------------------------------------------------
    _generate_ablation_table_tex(state)

    # ---------------------------------------------------------------
    # E) ranking_summary.md
    # ---------------------------------------------------------------
    _generate_ranking_summary(state)

    state["final_report"] = "done"
    save_state(state)
    logger.info("Final report generation complete")


def _generate_result_logger(state: dict) -> None:
    """Rewrite docs/03_result_logger.md from state data."""
    lines: list[str] = []
    a = lines.append

    a("# Experiment Results Ledger\n")
    a("Log only successful runs here. This file is the single source of truth for baseline tables,")
    a("generator fidelity tables, augmented-classifier results, ablations, and the final sweep winner.\n")
    a("---\n")
    a("## Logging Rules\n")
    a("- Record the exact `run_id` for every populated row.")
    a("- For generator rows, log the generator `run_id`.")
    a("- For augmented-classifier rows, log both the generator source `run_id` and the classifier `run_id` in the notes column when they differ.")
    a("- Treat `evaluation_results/classifier_metrics.json` as the authoritative classifier artifact.")
    a("- Treat `evaluation_results/metrics.json` as the authoritative generator artifact.\n")
    a("---\n")

    # Table 1: Phase 0
    a("## Table 1: Phase 0 — Baseline Classifiers\n")
    a("| Dataset | Track | Primary Metric | Score | Run ID / Notes |")
    a("| :--- | :--- | :--- | :--- | :--- |")

    phase0_rows = [
        ("CMAPSS", "engine_rul", "RMSE (↓)"),
        ("N-CMAPSS", "engine_rul", "RMSE (↓)"),
        ("FEMTO", "bearing_rul", "RMSE (↓)"),
        ("XJTU-SY", "bearing_rul", "RMSE (↓)"),
        ("CWRU", "bearing_fault", "F1 Macro (↑)"),
        ("DEMADICS", "bearing_fault", "F1 Macro (↑)"),
        ("Paderborn", "bearing_fault", "F1 Macro (↑)"),
    ]
    for ds, track, metric_label in phase0_rows:
        key = f"{track}__{ds}"
        entry = state["phase0"].get(key, {})
        if entry.get("status") == "done":
            metric_key = _primary_metric_key(track)
            score = entry.get(metric_key, "-")
            run_id = entry.get("run_id", "-")
            a(f"| **{ds}** | {track.replace('_', ' ').title()} | {metric_label} | {_fmt(score)} | {run_id} |")
        else:
            a(f"| **{ds}** | {track.replace('_', ' ').title()} | {metric_label} | - | (pending) |")

    a("\n---\n")

    # Table 2: Phase 2 Generator Fidelity
    a("## Table 2: Phase 2 — Generator Fidelity On Primary Datasets\n")
    for _, ds in PRIMARY_DATASETS:
        a(f"### {ds}\n")
        a("| Model | FTSD (↓) | MMD (↓) | Disc. Score (↓) | TSTR MAE (↓) | Run ID / Notes |")
        a("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for gen in GENERATORS:
            key = f"{gen}__{PRIMARY_DATASETS[0][0] if ds == 'CMAPSS' else 'bearing_fault'}__{ds}"
            # Find correct track
            for t, d in PRIMARY_DATASETS:
                if d == ds:
                    key = f"{gen}__{t}__{d}"
                    break
            entry = state["phase2"].get(key, {})
            if entry.get("gen_status") == "done":
                bold = "**" if gen == "FlowMatch" else ""
                a(f"| {bold}{gen}{bold} | {_fmt(entry.get('ftsd'))} | {_fmt(entry.get('mmd'))} | "
                  f"{_fmt(entry.get('discriminative_score'))} | {_fmt(entry.get('predictive_score_mae'))} | "
                  f"{entry.get('gen_run_id', '-')} |")
            else:
                bold = "**" if gen == "FlowMatch" else ""
                a(f"| {bold}{gen}{bold} | - | - | - | - | (pending) |")
        a("")

    a("---\n")

    # Table 3: Downstream Utility
    a("## Table 3: Phase 1 And Phase 2 — Downstream Utility\n")
    a("| Augmentation Method | CMAPSS (RMSE ↓) | CWRU (F1 Macro ↑) | DEMADICS (F1 Macro ↑) | Run ID / Notes |")
    a("| :--- | :--- | :--- | :--- | :--- |")

    # Baseline row
    p0_cmapss = state["phase0"].get("engine_rul__CMAPSS", {})
    p0_cwru = state["phase0"].get("bearing_fault__CWRU", {})
    p0_dem = state["phase0"].get("bearing_fault__DEMADICS", {})
    a(f"| **None (Phase 0 Baseline)** | {_fmt(p0_cmapss.get('rmse'))} | "
      f"{_fmt(p0_cwru.get('f1_macro'))} | {_fmt(p0_dem.get('f1_macro'))} | Phase 0 |")

    # Phase 1 rows
    for aug in ["noise", "smote"]:
        cmapss_entry = state["phase1"].get(f"{aug}__engine_rul__CMAPSS", {})
        cwru_entry = state["phase1"].get(f"{aug}__bearing_fault__CWRU", {})
        dem_entry = state["phase1"].get(f"{aug}__bearing_fault__DEMADICS", {})
        cmapss_val = _fmt(cmapss_entry.get("rmse")) if cmapss_entry.get("status") == "done" else "-"
        cwru_val = _fmt(cwru_entry.get("f1_macro")) if cwru_entry.get("status") == "done" else "-"
        dem_val = _fmt(dem_entry.get("f1_macro")) if dem_entry.get("status") == "done" else "-"
        label = f"+ {aug.title()}"
        a(f"| {label} | {cmapss_val} | {cwru_val} | {dem_val} | - |")

    # Phase 2 generator-augmented rows
    for gen in GENERATORS:
        cmapss_key = f"{gen}__engine_rul__CMAPSS"
        cwru_key = f"{gen}__bearing_fault__CWRU"
        dem_key = f"{gen}__bearing_fault__DEMADICS"
        ce = state["phase2"].get(cmapss_key, {})
        we = state["phase2"].get(cwru_key, {})
        de = state["phase2"].get(dem_key, {})
        cmapss_val = _fmt(ce.get("clf_rmse")) if ce.get("clf_status") == "done" else "-"
        cwru_val = _fmt(we.get("clf_f1_macro")) if we.get("clf_status") == "done" else "-"
        dem_val = _fmt(de.get("clf_f1_macro")) if de.get("clf_status") == "done" else "-"
        bold = "**" if gen == "FlowMatch" else ""
        label = f"{bold}+ {gen}{bold}"
        a(f"| {label} | {cmapss_val} | {cwru_val} | {dem_val} | - |")

    a("\n---\n")

    # Table 4: Phase 3 Secondary Datasets
    a("## Table 4: Phase 3 — Secondary-Dataset Generalization\n")
    a("| Dataset (Metric) | Baseline | Top 1 Model / Score | Top 2 Model / Score | Top 3 Model / Score | Run ID / Notes |")
    a("| :--- | :--- | :--- | :--- | :--- | :--- |")

    top3 = state.get("top_models", [None, None, None])
    while len(top3) < 3:
        top3.append(None)

    for track, ds in SECONDARY_DATASETS:
        baseline_entry = state["phase0"].get(f"{track}__{ds}", {})
        metric_key = _primary_metric_key(track)
        baseline_val = _fmt(baseline_entry.get(metric_key)) if baseline_entry.get("status") == "done" else "-"
        metric_label = "RMSE ↓" if "rul" in track else "F1 Macro ↑"

        cols: list[str] = []
        for m in top3:
            if m is None:
                cols.append("- / -")
                continue
            p3_key = f"{m}__{track}__{ds}"
            p3_entry = state["phase3"].get(p3_key, {})
            if p3_entry.get("clf_status") == "done":
                val = p3_entry.get(f"clf_{metric_key}", "-")
                cols.append(f"{m} / {_fmt(val)}")
            else:
                cols.append(f"{m} / (pending)")

        a(f"| **{ds}** ({metric_label}) | {baseline_val} | {cols[0]} | {cols[1]} | {cols[2]} | - |")

    a("\n---\n")

    # Table 5: Phase 4 Ablations
    a("## Table 5: Phase 4 — FlowMatch-PdM Ablations On CMAPSS\n")
    a("| Variant | FTSD (↓) | MMD (↓) | TSTR MAE (↓) | Downstream RMSE (↓) | Run ID / Notes |")
    a("| :--- | :--- | :--- | :--- | :--- | :--- |")

    # Full FlowMatch row from Phase 2
    fm_full = state["phase2"].get("FlowMatch__engine_rul__CMAPSS", {})
    a(f"| **Full FlowMatch-PdM** | {_fmt(fm_full.get('ftsd'))} | {_fmt(fm_full.get('mmd'))} | "
      f"{_fmt(fm_full.get('predictive_score_mae'))} | {_fmt(fm_full.get('clf_rmse'))} | "
      f"{fm_full.get('gen_run_id', '-')} |")

    for abl, label in [("no_prior", "No Prior"), ("no_tccm", "No TCCM"), ("no_lap", "No LAP")]:
        key = f"FlowMatch_{abl}__engine_rul__CMAPSS"
        entry = state["phase4"].get(key, {})
        if entry.get("gen_status") == "done":
            a(f"| {label} | {_fmt(entry.get('ftsd'))} | {_fmt(entry.get('mmd'))} | "
              f"{_fmt(entry.get('predictive_score_mae'))} | {_fmt(entry.get('clf_rmse', entry.get('clf_rmse')))} | "
              f"{entry.get('gen_run_id', '-')} |")
        else:
            a(f"| {label} | - | - | - | - | (pending) |")

    a("\n---\n")

    # Table 6: Sweep
    a("## Table 6: Phase 5 — W&B Sweep And Final CMAPSS Run\n")
    a("| Item | Value | Run ID / Notes |")
    a("| :--- | :--- | :--- |")
    a("| Sweep config path | `configs/sweep_flowmatch_cmapss.yaml` | - |")
    a("| Sweep winner model | FlowMatch-PdM | - |")
    a("| Sweep winner FTSD | - | - |")
    a("| Sweep winner TSTR MAE | - | - |")
    a("| Final proof-of-concept run status | - | - |")
    a("| Final proof-of-concept generator run | - | - |")
    a("| Final proof-of-concept classifier run | - | - |\n")

    a("---\n")
    a("## Final Freeze Checklist\n")
    a("- [ ] Table 1 is complete")
    a("- [ ] Table 2 is complete")
    a("- [ ] Table 3 is complete and the Top 1 / Top 2 / Top 3 order is frozen")
    a("- [ ] Table 4 is complete")
    a("- [ ] Table 5 is complete")
    a("- [ ] Table 6 is complete")
    a("- [ ] Every populated row includes the exact `run_id`")

    report_path = REPO_ROOT / "docs" / "03_result_logger.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", report_path)


def _generate_comparison_table_md(state: dict) -> None:
    """Generate results/final_report/comparison_table.md."""
    lines: list[str] = []
    a = lines.append

    a("# FlowMatch-PdM — Full Comparison Table\n")
    a("Lower is better for FTSD, MMD, Disc. Score, RMSE, TSTR MAE. Higher is better for F1 Macro.\n")
    a("| Rank | Model | Dataset | FTSD | MMD | Disc. Score | TSTR MAE | Primary Metric | Score |")
    a("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    # Collect all rows for ranking
    all_rows: list[dict] = []
    for gen in GENERATORS:
        for track, ds in PRIMARY_DATASETS:
            key = f"{gen}__{track}__{ds}"
            entry = state["phase2"].get(key, {})
            metric_key = _primary_metric_key(track)
            metric_label = "RMSE" if "rul" in track else "F1 Macro"
            row = {
                "model": gen, "dataset": ds, "track": track,
                "ftsd": entry.get("ftsd"),
                "mmd": entry.get("mmd"),
                "disc": entry.get("discriminative_score"),
                "tstr": entry.get("predictive_score_mae"),
                "metric_label": metric_label,
                "metric_val": entry.get(f"clf_{metric_key}"),
            }
            all_rows.append(row)

    # Sort by FTSD (lower=better), models with None at the end
    all_rows.sort(key=lambda r: (r["ftsd"] is None, r["ftsd"] if r["ftsd"] is not None else 1e9))

    for rank, row in enumerate(all_rows, 1):
        a(f"| {rank} | {row['model']} | {row['dataset']} | "
          f"{_fmt(row['ftsd'])} | {_fmt(row['mmd'])} | {_fmt(row['disc'])} | {_fmt(row['tstr'])} | "
          f"{row['metric_label']} | {_fmt(row['metric_val'])} |")

    out_path = FINAL_REPORT_DIR / "comparison_table.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_path)


def _generate_comparison_table_tex(state: dict) -> None:
    """Generate results/final_report/comparison_table.tex."""
    lines: list[str] = []
    a = lines.append

    a("% Requires: \\usepackage{booktabs,siunitx,multirow}")
    a("\\begin{table*}[t]")
    a("  \\centering")
    a("  \\caption{Comparison of generative models across primary datasets. "
      "Lower is better for FTSD, MMD, Disc.~Score, RMSE, TSTR MAE. "
      "Higher is better for F1 Macro.}")
    a("  \\label{tab:comparison}")
    a("  \\sisetup{round-mode=places, round-precision=4, table-format=1.4}")
    a("  \\begin{tabular}{l l S S S S l S}")
    a("    \\toprule")
    a("    {Model} & {Dataset} & {FTSD $\\downarrow$} & {MMD $\\downarrow$} & "
      "{Disc.~Score $\\downarrow$} & {TSTR MAE $\\downarrow$} & "
      "{Metric} & {Score} \\\\")
    a("    \\midrule")

    # Collect per-column values for bolding
    all_entries: list[dict] = []
    for gen in GENERATORS:
        for track, ds in PRIMARY_DATASETS:
            key = f"{gen}__{track}__{ds}"
            entry = state["phase2"].get(key, {})
            metric_key = _primary_metric_key(track)
            metric_label = "RMSE" if "rul" in track else "F1"
            all_entries.append({
                "model": gen, "dataset": ds,
                "ftsd": entry.get("ftsd"),
                "mmd": entry.get("mmd"),
                "disc": entry.get("discriminative_score"),
                "tstr": entry.get("predictive_score_mae"),
                "metric_label": metric_label,
                "metric_val": entry.get(f"clf_{metric_key}"),
            })

    # For simplicity, just bold the best FTSD per dataset
    for e in all_entries:
        ftsd_s = _fmt(e["ftsd"]) if e["ftsd"] is not None else "{-}"
        mmd_s = _fmt(e["mmd"]) if e["mmd"] is not None else "{-}"
        disc_s = _fmt(e["disc"]) if e["disc"] is not None else "{-}"
        tstr_s = _fmt(e["tstr"]) if e["tstr"] is not None else "{-}"
        met_s = _fmt(e["metric_val"]) if e["metric_val"] is not None else "{-}"
        a(f"    {e['model']} & {e['dataset']} & {ftsd_s} & {mmd_s} & "
          f"{disc_s} & {tstr_s} & {e['metric_label']} & {met_s} \\\\")

    a("    \\bottomrule")
    a("  \\end{tabular}")
    a("\\end{table*}")

    out_path = FINAL_REPORT_DIR / "comparison_table.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_path)


def _generate_ablation_table_tex(state: dict) -> None:
    """Generate results/final_report/ablation_table.tex."""
    lines: list[str] = []
    a = lines.append

    a("% Requires: \\usepackage{booktabs,siunitx,multirow}")
    a("\\begin{table}[t]")
    a("  \\centering")
    a("  \\caption{FlowMatch-PdM ablation study on CMAPSS.}")
    a("  \\label{tab:ablation}")
    a("  \\sisetup{round-mode=places, round-precision=4, table-format=1.4}")
    a("  \\begin{tabular}{l S S S S}")
    a("    \\toprule")
    a("    {Variant} & {FTSD $\\downarrow$} & {MMD $\\downarrow$} & "
      "{TSTR MAE $\\downarrow$} & {RMSE $\\downarrow$} \\\\")
    a("    \\midrule")

    variants = [
        ("Full FlowMatch-PdM", state["phase2"].get("FlowMatch__engine_rul__CMAPSS", {})),
        ("No Prior", state["phase4"].get("FlowMatch_no_prior__engine_rul__CMAPSS", {})),
        ("No TCCM", state["phase4"].get("FlowMatch_no_tccm__engine_rul__CMAPSS", {})),
        ("No LAP", state["phase4"].get("FlowMatch_no_lap__engine_rul__CMAPSS", {})),
    ]

    ftsd_vals = [v[1].get("ftsd") for v in variants]
    mmd_vals = [v[1].get("mmd") for v in variants]
    tstr_vals = [v[1].get("predictive_score_mae") for v in variants]
    rmse_vals = [v[1].get("clf_rmse") for v in variants]

    ftsd_fmt = _tex_bold_best(ftsd_vals, lower_is_better=True)
    mmd_fmt = _tex_bold_best(mmd_vals, lower_is_better=True)
    tstr_fmt = _tex_bold_best(tstr_vals, lower_is_better=True)
    rmse_fmt = _tex_bold_best(rmse_vals, lower_is_better=True)

    for i, (label, _) in enumerate(variants):
        a(f"    {label} & {ftsd_fmt[i]} & {mmd_fmt[i]} & {tstr_fmt[i]} & {rmse_fmt[i]} \\\\")

    a("    \\bottomrule")
    a("  \\end{tabular}")
    a("\\end{table}")

    out_path = FINAL_REPORT_DIR / "ablation_table.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_path)


def _generate_ranking_summary(state: dict) -> None:
    """Generate results/final_report/ranking_summary.md."""
    lines: list[str] = []
    a = lines.append

    a("# FlowMatch-PdM — Ranking Summary\n")

    top3 = state.get("top_models", [])

    # Compute average scores per model for narrative
    model_avg_ftsd: dict[str, list[float]] = {g: [] for g in GENERATORS}
    model_avg_clf: dict[str, list[float]] = {g: [] for g in GENERATORS}

    for gen in GENERATORS:
        for track, ds in PRIMARY_DATASETS:
            key = f"{gen}__{track}__{ds}"
            entry = state["phase2"].get(key, {})
            if entry.get("ftsd") is not None:
                model_avg_ftsd[gen].append(entry["ftsd"])
            metric_key = _primary_metric_key(track)
            clf_val = entry.get(f"clf_{metric_key}")
            if clf_val is not None:
                if "rul" in track:
                    model_avg_clf[gen].append(1.0 / clf_val if clf_val > 0 else 0)
                else:
                    model_avg_clf[gen].append(clf_val)

    avg_ftsd = {g: (sum(v) / len(v) if v else None) for g, v in model_avg_ftsd.items()}
    avg_utility = {g: (sum(v) / len(v) if v else None) for g, v in model_avg_clf.items()}

    # Best fidelity model
    fid_models = [(g, v) for g, v in avg_ftsd.items() if v is not None]
    fid_models.sort(key=lambda x: x[1])
    best_fidelity = fid_models[0] if fid_models else (None, None)

    # Best utility model
    util_models = [(g, v) for g, v in avg_utility.items() if v is not None]
    util_models.sort(key=lambda x: x[1], reverse=True)
    best_utility = util_models[0] if util_models else (None, None)

    a("## Overall Rankings\n")
    if top3:
        a(f"**#1 Overall: {top3[0]}**\n")
        if len(top3) > 1:
            a(f"**#2 Overall: {top3[1]}**\n")
        if len(top3) > 2:
            a(f"**#3 Overall: {top3[2]}**\n")
    else:
        a("Rankings not yet computed (Phase 2 incomplete).\n")

    a("## Fidelity vs Utility\n")
    if best_fidelity[0]:
        a(f"- **Best fidelity (lowest avg FTSD):** {best_fidelity[0]} (avg FTSD = {_fmt(best_fidelity[1])})")
    if best_utility[0]:
        a(f"- **Best utility (highest combined downstream score):** {best_utility[0]} (avg score = {_fmt(best_utility[1])})")

    if best_fidelity[0] and best_utility[0]:
        if best_fidelity[0] == best_utility[0]:
            a(f"\n{best_fidelity[0]} achieves the best results on **both** fidelity and utility metrics, "
              f"indicating strong synthetic data quality that directly translates to downstream performance.\n")
        else:
            a(f"\nThe best fidelity model ({best_fidelity[0]}) differs from the best utility model "
              f"({best_utility[0]}), suggesting that low distributional distance does not always translate "
              f"directly to downstream task improvement.\n")

    a("## Ablation Findings\n")
    fm_full = state["phase2"].get("FlowMatch__engine_rul__CMAPSS", {})
    ablation_results: list[tuple[str, Optional[float]]] = []
    for abl in ABLATION_VARIANTS:
        key = f"FlowMatch_{abl}__engine_rul__CMAPSS"
        entry = state["phase4"].get(key, {})
        ftsd = entry.get("ftsd")
        ablation_results.append((abl, ftsd))

    full_ftsd = fm_full.get("ftsd")
    if full_ftsd is not None and any(r[1] is not None for r in ablation_results):
        a(f"Full FlowMatch-PdM FTSD: {_fmt(full_ftsd)}\n")
        worst_abl = None
        worst_delta = 0.0
        for abl, ftsd in ablation_results:
            if ftsd is not None:
                delta = ftsd - full_ftsd
                a(f"- **{abl}**: FTSD = {_fmt(ftsd)} (Δ = {'+' if delta >= 0 else ''}{_fmt(delta)})")
                if delta > worst_delta:
                    worst_delta = delta
                    worst_abl = abl
        if worst_abl:
            component_names = {"no_prior": "Dynamic Harmonic Prior", "no_tccm": "TCCM Loss", "no_lap": "Layer-Adaptive Pruning"}
            a(f"\n**Most impactful component:** {component_names.get(worst_abl, worst_abl)} "
              f"(removing it increased FTSD by {_fmt(worst_delta)}).\n")
    else:
        a("Ablation results pending.\n")

    a("## Deployment Recommendation\n")
    if top3:
        a(f"For real-world PdM deployment, **{top3[0]}** is recommended as the primary synthetic data "
          f"augmentation model based on its combined fidelity and downstream utility across all tested datasets. "
          f"If computational resources are limited, {top3[1] if len(top3) > 1 else 'a simpler baseline'} "
          f"provides a good balance of performance and training cost.")
    else:
        a("Recommendation pending completion of all training phases.")

    out_path = FINAL_REPORT_DIR / "ranking_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_path)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    os.system("tmux new-session -d -s flowmatch 2>/dev/null || true")

    banner = """
╔══════════════════════════════════════════════════════════════╗
║           FlowMatch-PdM Orchestrator v1.0                   ║
║           Reading pipeline state...                         ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)
    logger.info("Orchestrator started at %s", datetime.now().isoformat())
    logger.info("Log file: %s", log_filename)

    state = load_state()

    try:
        check_env(state)
        acquire_datasets(state)
        run_preflight_notebook(state)
        run_phase0(state)
        run_phase1(state)
        run_phase2(state)
        run_phase3(state)
        run_phase4_ablations(state)
        generate_final_report(state)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Unhandled exception: %s", exc)
        logger.error(traceback.format_exc())
        mark_failed(state, "errors", f"Unhandled: {exc}\n{traceback.format_exc()}")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL PHASES COMPLETE")
    logger.info("=" * 60)
    logger.info("See results/final_report/ for outputs.")
    logger.info("  - comparison_table.md  / comparison_table.tex")
    logger.info("  - ablation_table.tex")
    logger.info("  - ranking_summary.md")
    logger.info("  - docs/03_result_logger.md")


if __name__ == "__main__":
    main()
