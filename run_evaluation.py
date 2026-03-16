import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

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


GENERATOR_CHOICES = ["TimeVAE", "TimeGAN", "DiffusionTS", "TimeFlow", "COTGAN", "FaultDiffusion", "FlowMatch"]
CLASSIFIER_CHOICES = ["baseline", "LSTMRegressor", "CNN1DClassifier"]


def _resolve_classifier_name(track: str, requested_model: str) -> str:
    if requested_model not in CLASSIFIER_CHOICES:
        raise ValueError(f"Unsupported classifier model: {requested_model}. Choose from {CLASSIFIER_CHOICES}.")

    default_model = "LSTMRegressor" if "rul" in track else "CNN1DClassifier"
    model_name = default_model if requested_model == "baseline" else requested_model

    if "rul" in track and model_name != "LSTMRegressor":
        raise ValueError(f"Track '{track}' requires LSTMRegressor, received '{model_name}'.")
    if track == "bearing_fault" and model_name != "CNN1DClassifier":
        raise ValueError(f"Track '{track}' requires CNN1DClassifier, received '{model_name}'.")
    return model_name


def _load_run_config(session: SessionManager, fallback_config_path: Optional[str]) -> dict:
    if getattr(session, "config", None):
        return session.config

    if fallback_config_path is None:
        raise FileNotFoundError("Run configuration is unavailable and no fallback config path was provided.")

    config_path = Path(fallback_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Unable to locate configuration for evaluation: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_manifest(session: SessionManager) -> dict:
    if not session.manifest_path.exists():
        return {}

    with session.manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_best_checkpoint(session: SessionManager, checkpoint_group: str) -> Path:
    manifest = _read_manifest(session)
    best_model_path = manifest.get("best_model_path")

    if best_model_path:
        checkpoint_path = Path(best_model_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = session.run_dir / checkpoint_path
        if checkpoint_path.exists():
            return checkpoint_path

    return resolve_checkpoint(session.run_dir, checkpoint_group)


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
    common_kwargs = {"map_location": "cpu"}

    if model_name == "TimeVAE":
        return TimeVAE.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size, **common_kwargs)
    if model_name == "TimeGAN":
        return TimeGAN.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size, **common_kwargs)
    if model_name == "DiffusionTS":
        return DiffusionTS.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size, **common_kwargs)
    if model_name == "TimeFlow":
        return TimeFlow.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size, **common_kwargs)
    if model_name == "COTGAN":
        return COTGAN.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size, **common_kwargs)
    if model_name == "FaultDiffusion":
        return FaultDiffusion.load_from_checkpoint(str(checkpoint_path), input_dim=input_dim, window_size=window_size, **common_kwargs)
    if model_name == "FlowMatch":
        return FlowMatchPdM.load_from_checkpoint(
            str(checkpoint_path),
            input_dim=input_dim,
            window_size=window_size,
            config=config["generative"]["flowmatch_pdm"],
            **common_kwargs,
        )
    raise ValueError(f"Unsupported generator model: {model_name}")


def _load_classifier(model_name: str, checkpoint_path: Path):
    if model_name == "LSTMRegressor":
        model = LSTMRegressor.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    elif model_name == "CNN1DClassifier":
        model = CNN1DClassifier.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    else:
        raise ValueError(f"Unsupported classifier model: {model_name}")
    model.eval()
    return model


def _load_reference_model(track: str, dataset: str):
    reference_model_name = "LSTMRegressor" if "rul" in track else "CNN1DClassifier"
    reference_session = SessionManager.from_existing(track, dataset, reference_model_name)
    checkpoint_path = _resolve_best_checkpoint(reference_session, "best_model_classifier")
    model = _load_classifier(reference_model_name, checkpoint_path)
    return reference_model_name, reference_session, checkpoint_path, model


def evaluate_generator_run(
    track: str,
    dataset: str,
    model: str,
    run_id: Optional[str] = None,
    config_path: Optional[str] = None,
    ablation: str = "none",
) -> Tuple[Dict[str, float], SessionManager]:
    if model not in GENERATOR_CHOICES:
        raise ValueError(f"Unsupported generator model: {model}. Choose from {GENERATOR_CHOICES}.")

    experiment_model_name = resolve_experiment_model_name(model, ablation)
    generator_session = SessionManager.from_existing(track, dataset, experiment_model_name, run_id=run_id)
    config = _load_run_config(generator_session, config_path)

    dataset_cfg = get_dataset_config(config, dataset)
    dm = get_data_module(
        track=track,
        dataset_name=dataset,
        fd=dataset_cfg.get("fd_list", 1),
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

    checkpoint_path = _resolve_best_checkpoint(generator_session, "best_models_generator")
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
        },
    )
    generator_session.update_manifest(
        {
            "evaluation_complete": True,
            "generator_checkpoint": str(checkpoint_path),
            "reference_model": reference_model_name,
            "reference_checkpoint": str(reference_ckpt),
            "evaluation_metrics": metrics,
        }
    )

    print(f"[Phase 3] Synthetic arrays saved under {generator_session.paths['generator_datas']}")
    print(f"[Phase 3] Evaluation saved under {generator_session.paths['evaluation_results']}")
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
) -> Tuple[Dict[str, float], SessionManager]:
    if bool(gen_model) ^ bool(source_run_id):
        raise ValueError("Classifier reevaluation for generator augmentation requires both gen_model and source_run_id.")
    if aug != "none" and gen_model is not None:
        raise ValueError("Classifier reevaluation targets either classical augmentation or generator augmentation, not both.")

    resolved_model_name = _resolve_classifier_name(track, model)
    experiment_model_name = resolve_classifier_experiment_name(
        resolved_model_name,
        aug=aug,
        gen_model=gen_model,
        gen_run_id=source_run_id,
        gen_ablation=gen_ablation,
    )
    classifier_session = SessionManager.from_existing(track, dataset, experiment_model_name, run_id=run_id)
    config = _load_run_config(classifier_session, config_path)

    dataset_cfg = get_dataset_config(config, dataset)
    is_rul = "rul" in track
    batch_size = config["classifier"]["lstm"]["batch_size"] if is_rul else config["classifier"]["cnn1d"]["batch_size"]

    dm = get_data_module(
        track=track,
        dataset_name=dataset,
        fd=dataset_cfg.get("fd_list", 1),
        window_size=dataset_cfg["window_size"],
        batch_size=batch_size,
    )
    dm.prepare_data()
    dm.setup(stage=None)

    checkpoint_path = _resolve_best_checkpoint(classifier_session, "best_model_classifier")
    classifier = _load_classifier(resolved_model_name, checkpoint_path)
    evaluator = SupervisedTaskEvaluator(
        model=classifier,
        task_type="regression" if is_rul else "classification",
        save_dir=str(classifier_session.paths["evaluation_results"]),
    )
    metrics = evaluator.evaluate(dm.test_dataloader(), filename_prefix="classifier")

    classifier_session.update_manifest(
        {
            "classifier_evaluation_complete": True,
            "classifier_checkpoint": str(checkpoint_path),
            "classifier_evaluation_metrics": metrics,
        }
    )

    print(f"[Classifier Eval] Metrics saved under {classifier_session.paths['evaluation_results']}")
    return metrics, classifier_session


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate an existing classifier or generator run.")
    parser.add_argument("--eval_mode", type=str, default="generator", choices=["classifier", "generator"])
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, or bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, FEMTO, CWRU, etc.")
    parser.add_argument("--model", type=str, required=True, help="Classifier or generator model name.")
    parser.add_argument("--run_id", type=str, default=None, help="Run identifier to re-evaluate. Defaults to latest.")
    parser.add_argument("--aug", type=str, default="none", help="Classifier classical augmentation mode.")
    parser.add_argument("--source_gen_model", "--gen_model", dest="source_gen_model", type=str, default=None, help="Source generator model for synthetic classifier augmentation.")
    parser.add_argument("--source_run_id", type=str, default=None, help="Source generator run identifier for synthetic classifier augmentation.")
    parser.add_argument("--ablation", type=str, default="none", help="Generator ablation variant.")
    parser.add_argument("--gen_ablation", type=str, default="none", help="Source generator ablation variant for classifier reevaluation.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Fallback config path when the run config is unavailable.")
    args = parser.parse_args()

    if args.eval_mode == "classifier":
        metrics, session = evaluate_classifier_run(
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
    else:
        metrics, session = evaluate_generator_run(
            track=args.track,
            dataset=args.dataset,
            model=args.model,
            run_id=args.run_id,
            config_path=args.config,
            ablation=args.ablation,
        )

    print(f"[Evaluation] Run: {session.run_dir}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
