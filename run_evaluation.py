import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import yaml

from flowmatchPdM.flowmatch_pdm import FlowMatchPdM
from src.baselines import COTGAN, DiffusionTS, FaultDiffusion, TimeFlow, TimeGAN, TimeVAE
from src.classifier import CNN1DClassifier, LSTMRegressor
from src.evaluation import SupervisedTaskEvaluator, TimeSeriesEvaluator
from src.utils.data_helper import get_data_module, get_dataset_config
from src.utils.logger_utils import (
    SessionManager,
    resolve_checkpoint,
    resolve_classifier_experiment_name,
    resolve_experiment_model_name,
)


CLASSIFIER_CHOICES = ["baseline", "LSTMRegressor", "CNN1DClassifier"]
GENERATOR_CHOICES = ["TimeVAE", "TimeGAN", "DiffusionTS", "TimeFlow", "COTGAN", "FaultDiffusion", "FlowMatch"]
EVAL_MODE_CHOICES = ["generator", "classifier"]


def _resolve_classifier_name(track: str, requested_model: str) -> str:
    default_model = "LSTMRegressor" if "rul" in track else "CNN1DClassifier"
    model_name = default_model if requested_model == "baseline" else requested_model
    if model_name not in {"LSTMRegressor", "CNN1DClassifier"}:
        raise ValueError(f"Unsupported classifier model '{requested_model}'.")
    if "rul" in track and model_name != "LSTMRegressor":
        raise ValueError(f"Track '{track}' requires LSTMRegressor, received '{model_name}'.")
    if track == "bearing_fault" and model_name != "CNN1DClassifier":
        raise ValueError(f"Track '{track}' requires CNN1DClassifier, received '{model_name}'.")
    return model_name


def _get_minority_dataset(dm, rul_threshold_ratio: float):
    try:
        return dm.get_minority_dataset(rul_threshold_ratio=rul_threshold_ratio)
    except TypeError:
        return dm.get_minority_dataset()


def _collect_dataset_arrays(dataset) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for x, y in dataset:
        xs.append(x.detach().cpu().numpy())
        if torch.is_tensor(y):
            ys.append(y.detach().cpu().numpy())
        else:
            ys.append(np.asarray(y))

    x_array = np.stack(xs).astype(np.float32)
    y_array = np.stack(ys)
    return x_array, y_array


def _load_generator(model_name: str, checkpoint_path: Path, input_dim: int, window_size: int, config: dict):
    if model_name == "TimeVAE":
        return TimeVAE.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size)
    if model_name == "TimeGAN":
        return TimeGAN.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size)
    if model_name == "DiffusionTS":
        return DiffusionTS.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size)
    if model_name == "TimeFlow":
        return TimeFlow.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size)
    if model_name == "COTGAN":
        return COTGAN.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size)
    if model_name == "FaultDiffusion":
        return FaultDiffusion.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size)
    if model_name == "FlowMatch":
        return FlowMatchPdM.load_from_checkpoint(
            str(checkpoint_path),
            input_dim=input_dim,
            window_size=window_size,
            config=config["generative"]["flowmatch_pdm"],
        )
    raise ValueError(f"Unsupported generator model: {model_name}")


def _load_classifier(model_name: str, checkpoint_path: Path):
    if model_name == "LSTMRegressor":
        return LSTMRegressor.load_from_checkpoint(str(checkpoint_path))
    if model_name == "CNN1DClassifier":
        return CNN1DClassifier.load_from_checkpoint(str(checkpoint_path))
    raise ValueError(f"Unsupported classifier model: {model_name}")


def _load_reference_model(track: str, dataset: str):
    reference_model_name = _resolve_classifier_name(track, "baseline")
    reference_session = SessionManager.from_existing(track, dataset, reference_model_name)
    checkpoint_path = resolve_checkpoint(Path(reference_session.paths["root"]), "best_model_classifier")
    model = _load_classifier(reference_model_name, checkpoint_path)
    model.eval()
    return reference_model_name, reference_session, checkpoint_path, model


def evaluate_generator_run(
    track: str,
    dataset: str,
    model: str,
    run_id: Optional[str] = None,
    config_path: Optional[str] = None,
    ablation: str = "none",
):
    if model not in GENERATOR_CHOICES:
        raise ValueError(f"Unsupported generator model '{model}'.")

    experiment_model_name = resolve_experiment_model_name(model, ablation)
    generator_session = SessionManager.from_existing(track, dataset, experiment_model_name, run_id=run_id)
    run_root = Path(generator_session.paths["root"])

    session_config_path = generator_session.config_path
    resolved_config_path = session_config_path if session_config_path.exists() else Path(config_path or "")
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Unable to locate configuration for evaluation: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_cfg = get_dataset_config(config, dataset)
    dm = get_data_module(
        track=track,
        dataset_name=dataset,
        window_size=dataset_cfg["window_size"],
        batch_size=config.get("evaluation", {}).get("batch_size", 128),
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    minority_ds = _get_minority_dataset(dm, config["datasets"]["minority_rul_ratio"])
    real_data, real_targets = _collect_dataset_arrays(minority_ds)
    sample_x, _ = dm.train_ds[0]
    input_dim = int(sample_x.shape[-1])
    window_size = int(dataset_cfg["window_size"])

    manifest = {}
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}

    checkpoint_path = (
        Path(manifest.get("best_model_path", ""))
        if manifest.get("best_model_path")
        else resolve_checkpoint(run_root, "best_models_generator")
    )
    generator = _load_generator(model, checkpoint_path, input_dim, window_size, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.eval().to(device)

    conditions = torch.tensor(real_targets, dtype=torch.float32, device=device)
    if conditions.dim() == 1:
        conditions = conditions.unsqueeze(-1)

    with torch.no_grad():
        if model == "FlowMatch":
            synthetic = generator.generate(conditions=conditions, num_samples=len(real_data))
        else:
            synthetic = generator.generate(num_samples=len(real_data), conditions=conditions)
    synthetic_data = synthetic.detach().cpu().numpy().astype(np.float32)

    generator_session.save_numpy("generator_datas/synthetic_data.npy", synthetic_data)
    generator_session.save_numpy("generator_datas/synthetic_targets.npy", real_targets)
    generator_session.save_numpy("generator_datas/real_minority_data.npy", real_data)
    generator_session.save_numpy("generator_datas/real_minority_targets.npy", real_targets)

    reference_model_name, reference_session, reference_ckpt, reference_model = _load_reference_model(track, dataset)
    evaluator = TimeSeriesEvaluator(
        real_data=real_data,
        synthetic_data=synthetic_data,
        save_dir=str(generator_session.paths["evaluation_results"]),
        feature_extractor=reference_model,
        batch_size=config.get("evaluation", {}).get("batch_size", 128),
        max_samples=config.get("evaluation", {}).get("max_samples", 2048),
        discriminative_epochs=config.get("evaluation", {}).get("discriminative_epochs", 20),
        predictive_epochs=config.get("evaluation", {}).get("predictive_epochs", 20),
    )
    metrics = evaluator.run_full_suite()

    generator_session.write_json(
        "generator_datas/generation_manifest.json",
        {
            "generator_model": model,
            "generator_run_id": generator_session.run_id,
            "generator_checkpoint": str(checkpoint_path),
            "reference_model": reference_model_name,
            "reference_run_id": reference_session.run_id,
            "reference_checkpoint": str(reference_ckpt),
            "num_generated_samples": int(len(synthetic_data)),
            "window_size": window_size,
            "input_dim": input_dim,
            "ablation": ablation,
        },
    )
    generator_session.update_manifest(
        {
            "evaluation_mode": "generator",
            "evaluation_complete": True,
            "generator_checkpoint": str(checkpoint_path),
            "reference_model": reference_model_name,
            "reference_checkpoint": str(reference_ckpt),
            "evaluation_metrics": metrics,
            "ablation": ablation,
        }
    )

    print(f"[Evaluation:generator] Synthetic arrays saved under {generator_session.paths['generator_datas']}")
    print(f"[Evaluation:generator] Evaluation saved under {generator_session.paths['evaluation_results']}")
    return metrics, generator_session


def evaluate_classifier_run(
    track: str,
    dataset: str,
    model: str,
    run_id: Optional[str] = None,
    config_path: Optional[str] = None,
    aug: str = "none",
    gen_model: Optional[str] = None,
    source_run_id: Optional[str] = None,
    gen_ablation: str = "none",
):
    classifier_model_name = _resolve_classifier_name(track, model)
    classifier_session_name = resolve_classifier_experiment_name(
        classifier_model_name,
        aug=aug,
        gen_model=gen_model,
        gen_run_id=source_run_id,
        gen_ablation=gen_ablation,
    )
    classifier_session = SessionManager.from_existing(track, dataset, classifier_session_name, run_id=run_id)

    session_config_path = classifier_session.config_path
    resolved_config_path = session_config_path if session_config_path.exists() else Path(config_path or "")
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Unable to locate configuration for evaluation: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_cfg = get_dataset_config(config, dataset)
    is_rul = "rul" in track
    default_batch_size = config["classifier"]["lstm"]["batch_size"] if is_rul else config["classifier"]["cnn1d"]["batch_size"]
    dm = get_data_module(
        track=track,
        dataset_name=dataset,
        window_size=dataset_cfg["window_size"],
        batch_size=config.get("evaluation", {}).get("batch_size", default_batch_size),
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    manifest = {}
    manifest_path = Path(classifier_session.paths["root"]) / "run_manifest.json"
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}

    checkpoint_path = (
        Path(manifest.get("best_model_path", ""))
        if manifest.get("best_model_path")
        else resolve_checkpoint(Path(classifier_session.paths["root"]), "best_model_classifier")
    )
    classifier = _load_classifier(classifier_model_name, checkpoint_path)
    evaluator = SupervisedTaskEvaluator(
        model=classifier,
        task_type="regression" if is_rul else "classification",
        save_dir=str(classifier_session.paths["evaluation_results"]),
    )
    metrics = evaluator.evaluate(dm.test_dataloader(), filename_prefix="classifier")
    classifier_session.update_manifest(
        {
            "evaluation_mode": "classifier",
            "evaluation_complete": True,
            "classifier_checkpoint": str(checkpoint_path),
            "classifier_evaluation_metrics": metrics,
            "augmentation_mode": aug,
            "generator_source_model": gen_model,
            "generator_source_run_id": source_run_id,
            "generator_source_ablation": gen_ablation,
        }
    )

    print(f"[Evaluation:classifier] Metrics saved under {classifier_session.paths['evaluation_results']}")
    return metrics, classifier_session


def main():
    parser = argparse.ArgumentParser(description="Evaluate either a classifier/regressor run or a generator run.")
    parser.add_argument("--eval_mode", type=str, default="generator", choices=EVAL_MODE_CHOICES)
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, or bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, FEMTO, CWRU, etc.")
    parser.add_argument("--model", "--gen_model", dest="model", type=str, required=True, help="Classifier or generator model name.")
    parser.add_argument("--run_id", type=str, default=None, help="Run identifier to evaluate.")
    parser.add_argument("--config", type=str, default=None, help="Optional fallback config path.")
    parser.add_argument("--aug", type=str, default="none", choices=["none", "noise", "smote"], help="Classifier augmentation mode.")
    parser.add_argument("--source_run_id", type=str, default=None, help="Generator run identifier used to create augmented classifier data.")
    parser.add_argument("--source_gen_model", type=str, default=None, help="Generator model used to create augmented classifier data.")
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "no_prior", "no_tccm", "no_lap"], help="Generator ablation variant.")
    parser.add_argument("--gen_ablation", type=str, default="none", choices=["none", "no_prior", "no_tccm", "no_lap"], help="Generator ablation used for augmented classifier data.")
    args = parser.parse_args()

    if args.eval_mode == "generator":
        evaluate_generator_run(
            track=args.track,
            dataset=args.dataset,
            model=args.model,
            run_id=args.run_id,
            config_path=args.config,
            ablation=args.ablation,
        )
        return

    evaluate_classifier_run(
        track=args.track,
        dataset=args.dataset,
        model=args.model,
        run_id=args.run_id,
        config_path=args.config,
        aug=args.aug,
        gen_model=args.source_gen_model,
        source_run_id=args.source_run_id,
        gen_ablation=args.gen_ablation,
    )


if __name__ == "__main__":
    main()
