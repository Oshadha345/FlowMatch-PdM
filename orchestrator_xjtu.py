from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = "XJTU-SY"
TRACK = "bearing_rul"
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/default_config.yaml")
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)
RESULTS_ROOT = ROOT / "results" / TRACK / DATASET
RUN_PREFIX = "run_"
JOB_FILTER = os.environ.get("JOB_FILTER", "").strip()


def normalize_run_id(run_id: str) -> str:
    return run_id if run_id.startswith(RUN_PREFIX) else f"{RUN_PREFIX}{run_id}"


def run_exists(*run_ids: str) -> bool:
    if not RESULTS_ROOT.exists():
        return False
    normalized = {normalize_run_id(run_id) for run_id in run_ids if run_id}
    for candidate in RESULTS_ROOT.rglob("*"):
        if candidate.is_dir() and candidate.name in normalized:
            return True
    return False


def run_has_artifact(relative_path: str, *run_ids: str) -> bool:
    if not RESULTS_ROOT.exists():
        return False
    normalized = {normalize_run_id(run_id) for run_id in run_ids if run_id}
    for candidate in RESULTS_ROOT.rglob("*"):
        if candidate.is_dir() and candidate.name in normalized and (candidate / relative_path).exists():
            return True
    return False


def expected_artifact_for_cmd(cmd: list[str]) -> str | None:
    if len(cmd) < 2:
        return None
    script_name = Path(cmd[1]).name
    if script_name in {"train_generator.py", "train_generator_raw.py"}:
        return "evaluation_results/metrics.json"
    if script_name in {"train_classifier_aug.py", "train_classifier.py"}:
        return "evaluation_results/classifier_metrics.json"
    return None


def select_jobs(all_jobs: list[dict]) -> list[dict]:
    if not JOB_FILTER:
        return all_jobs
    tokens = [token.strip().lower() for token in JOB_FILTER.split(",") if token.strip()]
    if not tokens:
        return all_jobs
    return [job for job in all_jobs if all(token in job["name"].lower() for token in tokens)]


def execute_job(name: str, run_id: str, cmd: list[str], skip_aliases: list[str] | None = None) -> bool:
    aliases = [run_id, *(skip_aliases or [])]
    expected_artifact = expected_artifact_for_cmd(cmd)
    if expected_artifact is not None:
        if run_has_artifact(expected_artifact, *aliases):
            print(f"Skipping {name}: Already completed")
            return True
        if run_exists(*aliases):
            print(f"Resuming {name}: found prior run directory but '{expected_artifact}' is still missing")
    elif run_exists(*aliases):
        print(f"Skipping {name}: Already completed")
        return True

    print(f"Running {name}")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, env=os.environ.copy())
        return True
    except FileNotFoundError as exc:
        print(f"Failed {name}: {exc}")
    except subprocess.CalledProcessError as exc:
        print(f"Failed {name}: exit code {exc.returncode}")
    return False


jobs = [
    {
        "name": "XJTU-SY Raw CNN1D",
        "run_id": "dual_orch_xjtu_sy_cnn1d_raw_20260415",
        "cmd": [
            PYTHON_BIN,
            "train_classifier_aug.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--model",
            "raw",
            "--eval_model",
            "cnn1d",
            "--epochs",
            "20",
            "--output_run_id",
            "dual_orch_xjtu_sy_cnn1d_raw_20260415",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY Raw Transformer",
        "run_id": "dual_orch_xjtu_sy_transformer_raw_20260415",
        "cmd": [
            PYTHON_BIN,
            "train_classifier_aug.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--model",
            "raw",
            "--eval_model",
            "transformer",
            "--epochs",
            "20",
            "--output_run_id",
            "dual_orch_xjtu_sy_transformer_raw_20260415",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY Raw FlowMatch Generator",
        "run_id": "pivot_rul_gpu1_flowmatch_20260414_xjtu_sy_flowmatch",
        "cmd": [
            PYTHON_BIN,
            "train_generator_raw.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--model",
            "FlowMatch",
            "--run_id",
            "pivot_rul_gpu1_flowmatch_20260414_xjtu_sy_flowmatch",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY Raw FlowMatch Mamba TSTR",
        "run_id": "pivot_rul_gpu1_flowmatch_20260414_xjtu_sy_flowmatch_mamba_tstr",
        "cmd": [
            PYTHON_BIN,
            "train_classifier_aug.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--gen_model",
            "FlowMatch",
            "--source_run_id",
            "pivot_rul_gpu1_flowmatch_20260414_xjtu_sy_flowmatch",
            "--eval_model",
            "mamba",
            "--output_run_id",
            "pivot_rul_gpu1_flowmatch_20260414_xjtu_sy_flowmatch_mamba_tstr",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY Raw FlowMatch no_tccm",
        "run_id": "pivot_rul_gpu1_flowmatch_20260414_xjtu_sy_flowmatch_no_tccm",
        "skip_aliases": [
            "pivot_rul_gpu1_flowmatch_20260414_femto_flowmatch_no_tccm",
        ],
        "cmd": [
            PYTHON_BIN,
            "train_generator_raw.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--model",
            "FlowMatch",
            "--ablation",
            "no_tccm",
            "--run_id",
            "pivot_rul_gpu1_flowmatch_20260414_xjtu_sy_flowmatch_no_tccm",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY Raw FlowMatch no_tccm Mamba TSTR",
        "run_id": "pivot_rul_gpu1_flowmatch_20260414_xjtu_sy_flowmatch_no_tccm_mamba_tstr",
        "cmd": [
            PYTHON_BIN,
            "train_classifier_aug.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--gen_model",
            "FlowMatch",
            "--gen_ablation",
            "no_tccm",
            "--source_run_id",
            "pivot_rul_gpu1_flowmatch_20260414_femto_flowmatch_no_tccm",
            "--eval_model",
            "mamba",
            "--output_run_id",
            "pivot_rul_gpu1_flowmatch_20260414_xjtu_sy_flowmatch_no_tccm_mamba_tstr",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY Raw FlowMatch no_lap",
        "run_id": "dual_orch_xjtu_sy_flowmatch_no_lap_20260415",
        "cmd": [
            PYTHON_BIN,
            "train_generator_raw.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--model",
            "FlowMatch",
            "--ablation",
            "no_lap",
            "--run_id",
            "dual_orch_xjtu_sy_flowmatch_no_lap_20260415",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY Raw FlowMatch no_lap Mamba TSTR",
        "run_id": "dual_orch_xjtu_sy_flowmatch_no_lap_20260415_mamba_tstr",
        "cmd": [
            PYTHON_BIN,
            "train_classifier_aug.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--gen_model",
            "FlowMatch",
            "--gen_ablation",
            "no_lap",
            "--source_run_id",
            "dual_orch_xjtu_sy_flowmatch_no_lap_20260415",
            "--eval_model",
            "mamba",
            "--output_run_id",
            "dual_orch_xjtu_sy_flowmatch_no_lap_20260415_mamba_tstr",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY Raw FlowMatch no_harmonic_prior",
        "run_id": "dual_orch_xjtu_sy_flowmatch_no_harmonic_prior_20260415",
        "cmd": [
            PYTHON_BIN,
            "train_generator_raw.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--model",
            "FlowMatch",
            "--ablation",
            "no_harmonic_prior",
            "--run_id",
            "dual_orch_xjtu_sy_flowmatch_no_harmonic_prior_20260415",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY Raw FlowMatch no_harmonic_prior Mamba TSTR",
        "run_id": "dual_orch_xjtu_sy_flowmatch_no_harmonic_prior_20260415_mamba_tstr",
        "cmd": [
            PYTHON_BIN,
            "train_classifier_aug.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--gen_model",
            "FlowMatch",
            "--gen_ablation",
            "no_prior",
            "--source_run_id",
            "dual_orch_xjtu_sy_flowmatch_no_harmonic_prior_20260415",
            "--eval_model",
            "mamba",
            "--output_run_id",
            "dual_orch_xjtu_sy_flowmatch_no_harmonic_prior_20260415_mamba_tstr",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY CropFlow Generator",
        "run_id": "patchflow_xjtu_test_20260415",
        "cmd": [
            PYTHON_BIN,
            "train_generator.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--model",
            "CropFlow",
            "--run_id",
            "patchflow_xjtu_test_20260415",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY CropFlow Mamba TSTR",
        "run_id": "patchflow_xjtu_test_20260415_mamba_tstr",
        "cmd": [
            PYTHON_BIN,
            "train_classifier_aug.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--gen_model",
            "CropFlow",
            "--source_run_id",
            "patchflow_xjtu_test_20260415",
            "--eval_model",
            "mamba",
            "--output_run_id",
            "patchflow_xjtu_test_20260415_mamba_tstr",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY CropFlow no_tccm",
        "run_id": "patchflow_xjtu_test_20260415_no_tccm",
        "cmd": [
            PYTHON_BIN,
            "train_generator.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--model",
            "CropFlow",
            "--ablation",
            "no_tccm",
            "--run_id",
            "patchflow_xjtu_test_20260415_no_tccm",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY CropFlow no_lap",
        "run_id": "patchflow_xjtu_test_20260415_no_lap",
        "cmd": [
            PYTHON_BIN,
            "train_generator.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--model",
            "CropFlow",
            "--ablation",
            "no_lap",
            "--run_id",
            "patchflow_xjtu_test_20260415_no_lap",
            "--config",
            CONFIG_PATH,
        ],
    },
    {
        "name": "XJTU-SY CropFlow no_harmonic_prior",
        "run_id": "patchflow_xjtu_test_20260415_no_harmonic_prior",
        "cmd": [
            PYTHON_BIN,
            "train_generator.py",
            "--track",
            TRACK,
            "--dataset",
            DATASET,
            "--model",
            "CropFlow",
            "--ablation",
            "no_prior",
            "--run_id",
            "patchflow_xjtu_test_20260415_no_harmonic_prior",
            "--config",
            CONFIG_PATH,
        ],
    },
]


def main() -> int:
    failures: list[str] = []
    active_jobs = select_jobs(jobs)
    if JOB_FILTER:
        print(f"Applying JOB_FILTER={JOB_FILTER!r}: {len(active_jobs)} job(s) selected")
    for job in active_jobs:
        ok = execute_job(
            name=job["name"],
            run_id=job["run_id"],
            cmd=job["cmd"],
            skip_aliases=job.get("skip_aliases"),
        )
        if not ok:
            failures.append(job["name"])

    if failures:
        print("Completed with failures:")
        for name in failures:
            print(f" - {name}")
        return 1

    print("All XJTU-SY jobs processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
