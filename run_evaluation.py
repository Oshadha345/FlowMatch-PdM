import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml

from flowmatchPdM.flowmatch_pdm import FlowMatchPdM
from src.baselines import COTGAN , DiffusionTS, FaultDiffusion , TimeFlow , TimeGAN , TimeVAE
from src.classifier import CNN1DClassifier , LSTMRegressor
from src.evaluation import TimeSeriesEvaluator
from src.utils.data_helper import get_data_module , get_dataset_config
from src.utils.logger_utils import SessionManager , resolve_checkpoint


GENERATOR_CHOICES = ["TimeVAE", "TimeGAN", "DiffusionTS", "TimeFlow", "COTGAN", "FaultDiffusion", "FlowMatch"]
GENERATOR_CONFIG_MAP = {
    "TimeVAE": "timevae",
    "TimeGAN": "timegan",
    "DiffusionTS": "diffusion",
    "TimeFlow": "timeflow",
    "COTGAN": "cotgan",
    "FaultDiffusion": "faultdiffusion",
    "FlowMatch": "flowmatch_pdm",
}


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


def _load_reference_model(track: str, dataset: str):
    reference_model_name = "LSTMRegressor" if "rul" in track else "CNN1DClassifier"
    reference_session = SessionManager.from_existing(track, dataset, reference_model_name)
    checkpoint_path = resolve_checkpoint(Path(reference_session.paths["root"]), "best_model_classifier")
    if reference_model_name == "LSTMRegressor":
        model = LSTMRegressor.load_from_checkpoint(str(checkpoint_path))
    else:
        model = CNN1DClassifier.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    return reference_model_name, reference_session, checkpoint_path, model


def main():
    parser = argparse.ArgumentParser(description="Phase 3: generate synthetic arrays and run the evaluation suite.")
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, or bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, FEMTO, CWRU, etc.")
    parser.add_argument("--model", "--gen_model", dest="model", type=str, required=True, choices=GENERATOR_CHOICES)
    parser.add_argument("--run_id", type=str, default=None, help="Generator run identifier. Defaults to latest.")
    parser.add_argument("--config", type=str, default=None, help="Optional fallback config path.")
    args = parser.parse_args()

    generator_session = SessionManager.from_existing(args.track, args.dataset, args.model, run_id=args.run_id)
    run_root = Path(generator_session.paths["root"])
    config_path = Path(generator_session.config_path if generator_session.config_path.exists() else args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Unable to locate configuration for evaluation: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dataset_cfg = get_dataset_config(config, args.dataset)
    dm = get_data_module(
        track=args.track,
        dataset_name=args.dataset,
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

    checkpoint_path = Path(manifest.get("best_model_path", "")) if manifest.get("best_model_path") else resolve_checkpoint(run_root, "best_models_generator")
    generator = _load_generator(args.model, checkpoint_path, input_dim, window_size, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.eval().to(device)

    conditions = torch.tensor(real_targets, dtype=torch.float32, device=device)
    if conditions.dim() == 1:
        conditions = conditions.unsqueeze(-1)

    with torch.no_grad():
        if args.model == "FlowMatch":
            synthetic = generator.generate(conditions=conditions, num_samples=len(real_data))
        else:
            synthetic = generator.generate(num_samples=len(real_data), conditions=conditions)
    synthetic_data = synthetic.detach().cpu().numpy().astype(np.float32)

    generator_session.save_numpy("generator_datas/synthetic_data.npy", synthetic_data)
    generator_session.save_numpy("generator_datas/synthetic_targets.npy", real_targets)
    generator_session.save_numpy("generator_datas/real_minority_data.npy", real_data)
    generator_session.save_numpy("generator_datas/real_minority_targets.npy", real_targets)

    reference_model_name, reference_session, reference_ckpt, reference_model = _load_reference_model(args.track, args.dataset)
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
            "generator_model": args.model,
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


if __name__ == "__main__":
    main()
