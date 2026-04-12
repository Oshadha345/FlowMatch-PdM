import argparse
import math
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, TensorDataset

from run_evaluation import evaluate_classifier_run
from src.baselines import ClassicalAugmenter
from src.classifier import CNN1DClassifier, LSTMRegressor
from src.evaluation import TSTR_Evaluation
from src.utils.data_helper import get_data_module, get_dataset_config
from src.utils.logger_utils import (
    JSONMetricsTracker,
    SessionManager,
    resolve_classifier_experiment_name,
    resolve_experiment_model_name,
    setup_wandb_logger,
)

try:
    import wandb
except Exception:  # pragma: no cover - wandb is optional at runtime
    wandb = None


CLASSIFIER_CHOICES = ["baseline", "LSTMRegressor", "CNN1DClassifier"]
AUGMENTATION_CHOICES = ["none", "noise", "smote"]
DEFAULT_TSTR_MIN_RELATIVE_F1 = 0.8
DEFAULT_TSTR_MIN_RELATIVE_BALANCED_ACC = 0.8


class _RealSyntheticBatchSampler(BatchSampler):
    """
    Batch sampler that anchors each epoch on real samples and injects a controlled
    number of synthetic samples into every batch.
    """

    def __init__(self, real_size: int, synthetic_size: int, batch_size: int, aug_ratio: float):
        if real_size <= 0:
            raise ValueError("real_size must be positive.")
        if synthetic_size < 0:
            raise ValueError("synthetic_size must be non-negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0.0 <= aug_ratio < 1.0:
            raise ValueError(f"aug_ratio must lie in [0, 1), received {aug_ratio}.")

        self.real_size = int(real_size)
        self.synthetic_size = int(synthetic_size)
        self.batch_size = int(batch_size)
        self.aug_ratio = float(aug_ratio)

        synthetic_per_batch = int(round(self.batch_size * self.aug_ratio))
        if self.synthetic_size > 0 and self.aug_ratio > 0.0 and synthetic_per_batch == 0:
            synthetic_per_batch = 1
        if synthetic_per_batch >= self.batch_size:
            synthetic_per_batch = self.batch_size - 1

        self.synthetic_per_batch = max(synthetic_per_batch, 0)
        self.real_per_batch = self.batch_size - self.synthetic_per_batch
        if self.real_per_batch <= 0:
            raise ValueError("Resolved real_per_batch to zero; reduce aug_ratio or increase batch_size.")

        self.num_batches = max(int(math.ceil(self.real_size / self.real_per_batch)), 1)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        real_indices = torch.randperm(self.real_size, dtype=torch.long)
        required_real = self.num_batches * self.real_per_batch
        if required_real > self.real_size:
            pad_size = required_real - self.real_size
            pad = torch.randint(self.real_size, (pad_size,), dtype=torch.long)
            real_indices = torch.cat([real_indices, pad], dim=0)

        synthetic_indices = None
        if self.synthetic_per_batch > 0:
            required_synthetic = self.num_batches * self.synthetic_per_batch
            synthetic_indices = torch.randint(self.synthetic_size, (required_synthetic,), dtype=torch.long)

        for batch_idx in range(self.num_batches):
            start_real = batch_idx * self.real_per_batch
            end_real = start_real + self.real_per_batch
            real_batch = real_indices[start_real:end_real]

            if synthetic_indices is None:
                batch = real_batch
            else:
                start_syn = batch_idx * self.synthetic_per_batch
                end_syn = start_syn + self.synthetic_per_batch
                synthetic_batch = synthetic_indices[start_syn:end_syn] + self.real_size
                batch = torch.cat([real_batch, synthetic_batch], dim=0)
                batch = batch[torch.randperm(batch.numel(), dtype=torch.long)]

            yield batch.tolist()


def _resolve_classifier_name(track: str, requested_model: str) -> str:
    default_model = "LSTMRegressor" if "rul" in track else "CNN1DClassifier"
    model_name = default_model if requested_model == "baseline" else requested_model

    if "rul" in track and model_name != "LSTMRegressor":
        raise ValueError(f"Track '{track}' requires LSTMRegressor, received '{model_name}'.")
    if track == "bearing_fault" and model_name != "CNN1DClassifier":
        raise ValueError(f"Track '{track}' requires CNN1DClassifier, received '{model_name}'.")
    return model_name


def _build_model(track: str, model_name: str, config: dict, input_dim: int, num_classes: int, target_scale: float = 1.0):
    if model_name == "LSTMRegressor":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=config["classifier"]["lstm"]["hidden_dim"],
            num_layers=config["classifier"]["lstm"]["num_layers"],
            learning_rate=config["classifier"]["lstm"]["lr"],
            target_scale=target_scale,
        )
    if model_name == "CNN1DClassifier":
        return CNN1DClassifier(
            num_classes=num_classes,
            input_channels=input_dim,
            learning_rate=config["classifier"]["cnn1d"]["lr"],
        )
    raise ValueError(f"Unsupported classifier model: {model_name}")


def _resolve_trainer_precision(config: dict, is_rul: bool):
    precision_cfg = config["trainer"].get("precision", "16-mixed")
    if isinstance(precision_cfg, dict):
        return precision_cfg["rul" if is_rul else "fault"]
    return 32 if is_rul else precision_cfg


def _collect_dataset_tensors(dataset):
    xs = []
    ys = []
    for x, y in dataset:
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu() if torch.is_tensor(y) else torch.tensor(y))
    return torch.stack(xs), torch.stack(ys)


def _resolve_aug_ratio(requested_aug_ratio, real_count: int, synthetic_count: int) -> float:
    if synthetic_count <= 0:
        return 0.0
    if requested_aug_ratio is None:
        return float(synthetic_count) / float(real_count + synthetic_count)

    aug_ratio = float(requested_aug_ratio)
    if not 0.0 <= aug_ratio < 1.0:
        raise ValueError(f"--aug_ratio must lie in [0, 1), received {requested_aug_ratio}.")
    return aug_ratio


def _run_tstr_gate(
    generator_session: SessionManager,
    synthetic_x: np.ndarray,
    synthetic_y: np.ndarray,
    config: dict,
) -> dict:
    real_x_path = Path(generator_session.paths["generator_datas"]) / "real_minority_data.npy"
    real_y_path = Path(generator_session.paths["generator_datas"]) / "real_minority_targets.npy"
    if not real_x_path.exists() or not real_y_path.exists():
        raise FileNotFoundError(
            "TSTR gate requires the generator evaluation artifacts "
            f"'{real_x_path.name}' and '{real_y_path.name}'. Re-run generator evaluation first."
        )

    real_x = np.load(real_x_path).astype(np.float32)
    real_y = np.load(real_y_path)
    tstr = TSTR_Evaluation(
        save_dir=str(generator_session.paths["evaluation_results"]),
        batch_size=config.get("evaluation", {}).get("batch_size", 128),
        epochs=config.get("evaluation", {}).get("tstr_epochs", 20),
        min_relative_f1=config.get("evaluation", {}).get("tstr_min_relative_f1", DEFAULT_TSTR_MIN_RELATIVE_F1),
        min_relative_balanced_accuracy=config.get("evaluation", {}).get(
            "tstr_min_relative_balanced_accuracy",
            DEFAULT_TSTR_MIN_RELATIVE_BALANCED_ACC,
        ),
    )
    return tstr.run(
        synthetic_data=synthetic_x,
        synthetic_targets=synthetic_y,
        real_data=real_x,
        real_targets=real_y,
        filename_prefix="tstr_gate",
    )


def _load_generator_augmented_dataset(
    dm,
    track: str,
    dataset: str,
    gen_model: str,
    source_run_id: str,
    gen_ablation: str,
    requested_aug_ratio,
    config: dict,
):
    generator_model_name = resolve_experiment_model_name(gen_model, gen_ablation)
    generator_session = SessionManager.from_existing(track, dataset, generator_model_name, run_id=source_run_id)
    gen_dir = Path(generator_session.paths["generator_datas"])
    x_path = gen_dir / "synthetic_data.npy"
    y_path = gen_dir / "synthetic_targets.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Synthetic data artifact not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Synthetic target artifact not found: {y_path}")

    synthetic_x = np.load(x_path).astype(np.float32)
    synthetic_y = np.load(y_path)
    label_dtype = dm.train_ds[0][1].dtype if torch.is_tensor(dm.train_ds[0][1]) else torch.float32
    synthetic_x_tensor = torch.from_numpy(synthetic_x)

    if torch.is_floating_point(dm.train_ds[0][1]):
        synthetic_y_tensor = torch.tensor(synthetic_y, dtype=torch.float32).reshape(-1)
    else:
        synthetic_y_tensor = torch.tensor(synthetic_y, dtype=label_dtype).reshape(-1)

    synthetic_ds = TensorDataset(synthetic_x_tensor, synthetic_y_tensor)
    resolved_aug_ratio = _resolve_aug_ratio(requested_aug_ratio, len(dm.train_ds), len(synthetic_ds))

    tstr_metrics = None
    if track == "bearing_fault":
        tstr_metrics = _run_tstr_gate(generator_session, synthetic_x, synthetic_y, config)
        if not bool(tstr_metrics.get("gate_passed", False)):
            raise RuntimeError(
                "TSTR gate rejected synthetic augmentation "
                f"for {dataset}/{gen_model}. Metrics: {tstr_metrics}"
            )

    mixed_ds = ConcatDataset([dm.train_ds, synthetic_ds])
    batch_sampler = _RealSyntheticBatchSampler(
        real_size=len(dm.train_ds),
        synthetic_size=len(synthetic_ds),
        batch_size=dm.batch_size,
        aug_ratio=resolved_aug_ratio,
    )
    summary = {
        "mode": "generator",
        "source_run_id": generator_session.run_id,
        "source_model": gen_model,
        "source_experiment_model": generator_model_name,
        "source_ablation": gen_ablation,
        "requested_aug_ratio": None if requested_aug_ratio is None else float(requested_aug_ratio),
        "resolved_aug_ratio": resolved_aug_ratio,
        "real_per_batch": batch_sampler.real_per_batch,
        "synthetic_per_batch": batch_sampler.synthetic_per_batch,
        "synthetic_count": int(len(synthetic_ds)),
        "real_count": int(len(dm.train_ds)),
        "total_count": int(len(mixed_ds)),
    }
    if tstr_metrics is not None:
        summary["tstr_gate"] = tstr_metrics
    return mixed_ds, summary, batch_sampler


def _build_training_dataset(dm, args, config):
    dm.prepare_data()
    dm.setup(stage="fit")

    if args.source_run_id or args.gen_model:
        if not args.gen_model or not args.source_run_id:
            raise ValueError("Generator augmentation requires both --gen_model and --source_run_id.")
        if args.aug != "none":
            raise ValueError("Use either classical augmentation (--aug noise/smote) or generator augmentation, not both.")
        return _load_generator_augmented_dataset(
            dm,
            args.track,
            args.dataset,
            args.gen_model,
            args.source_run_id,
            args.gen_ablation,
            args.aug_ratio,
            config,
        )

    x_train, y_train = _collect_dataset_tensors(dm.train_ds)
    x_np = x_train.numpy()
    y_np = y_train.numpy()

    if args.aug == "smote":
        if "rul" in args.track:
            raise ValueError("SMOTE augmentation is only valid for classification tracks.")
        x_aug, y_aug = ClassicalAugmenter.apply_smote(
            x_np,
            y_np,
            k_neighbors=config.get("classical", {}).get("smote", {}).get("k_neighbors", 5),
        )
        dataset = TensorDataset(
            torch.tensor(x_aug, dtype=torch.float32),
            torch.tensor(y_aug, dtype=torch.long),
        )
        return dataset, {
            "mode": "smote",
            "real_count": int(len(dm.train_ds)),
            "augmented_count": int(len(dataset)),
        }, None

    if args.aug == "noise":
        sigma = config.get("classical", {}).get("jittering", {}).get("sigma", 0.05)
        x_noise = ClassicalAugmenter.apply_jittering(x_np, sigma=sigma)
        x_mixed = np.concatenate([x_np, x_noise], axis=0)
        y_mixed = np.concatenate([y_np, y_np], axis=0)
        y_dtype = torch.float32 if torch.is_floating_point(y_train) else torch.long
        dataset = TensorDataset(
            torch.tensor(x_mixed, dtype=torch.float32),
            torch.tensor(y_mixed, dtype=y_dtype),
        )
        return dataset, {
            "mode": "noise",
            "sigma": sigma,
            "real_count": int(len(dm.train_ds)),
            "augmented_count": int(len(dataset)),
        }, None

    return dm.train_ds, {
        "mode": "none",
        "real_count": int(len(dm.train_ds)),
        "augmented_count": int(len(dm.train_ds)),
    }, None


def _build_train_loader(train_ds, batch_size: int, batch_sampler):
    if batch_sampler is not None:
        return DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True,
        )

    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train baseline or augmented classifiers/regressors.")
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, or bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, FEMTO, CWRU, etc.")
    parser.add_argument("--model", type=str, default="baseline", choices=CLASSIFIER_CHOICES)
    parser.add_argument("--run_id", type=str, default=None, help="Optional output run identifier for the classifier run.")
    parser.add_argument("--aug", type=str, default="none", choices=AUGMENTATION_CHOICES, help="Classical augmentation mode.")
    parser.add_argument("--gen_model", type=str, default=None, help="Generator model name for synthetic augmentation.")
    parser.add_argument("--source_run_id", type=str, default=None, help="Generator run identifier for synthetic augmentation.")
    parser.add_argument("--gen_ablation", type=str, default="none", choices=["none", "no_prior", "no_tccm", "no_lap"], help="Generator ablation variant for synthetic augmentation.")
    parser.add_argument("--aug_ratio", type=float, default=None, help="Target synthetic batch fraction for generator augmentation, e.g. 0.2.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging for this run.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.use_wandb:
        config.setdefault("logging", {})["use_wandb"] = True

    pl.seed_everything(config["seed"], workers=True)

    dataset_cfg = get_dataset_config(config, args.dataset)
    model_name = _resolve_classifier_name(args.track, args.model)
    is_rul = "rul" in args.track
    batch_size = int(dataset_cfg["batch_size"])

    dm = get_data_module(
        track=args.track,
        dataset_name=args.dataset,
        conditions=dataset_cfg.get("conditions", dataset_cfg.get("fd_list", 1)),
        window_size=dataset_cfg["window_size"],
        batch_size=batch_size,
        append_condition_features=dataset_cfg.get("append_condition_features", False),
    )
    train_ds, augmentation_summary, batch_sampler = _build_training_dataset(dm, args, config)
    train_loader = _build_train_loader(train_ds, batch_size=batch_size, batch_sampler=batch_sampler)

    dm.prepare_data()
    dm.setup(stage="fit")
    sample_x, _ = dm.train_ds[0]
    input_dim = int(sample_x.shape[-1])
    num_classes = int(getattr(dm, "num_classes", dataset_cfg.get("num_classes", 1)))
    target_scale = float(getattr(dm, "target_scale", 1.0))
    model = _build_model(args.track, model_name, config, input_dim, num_classes, target_scale=target_scale)

    session_model_name = resolve_classifier_experiment_name(
        model_name,
        aug=args.aug,
        gen_model=args.gen_model,
        gen_run_id=args.source_run_id,
        gen_ablation=args.gen_ablation,
    )
    session = SessionManager(
        track=args.track,
        dataset=args.dataset,
        model_name=session_model_name,
        config=config,
        run_id=args.run_id,
    )
    paths = session.get_paths()

    checkpoint_callback = ModelCheckpoint(
        dirpath=paths["best_model_classifier"],
        filename=f"{model_name}-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=config["trainer"]["patience"], mode="min")
    metrics_filename = "phase0_metrics.json" if augmentation_summary["mode"] == "none" else "phase3_metrics.json"
    metrics_tracker = JSONMetricsTracker(output_path=str(Path(paths["evaluation_results"]) / metrics_filename))
    phase_label = "Phase0" if augmentation_summary["mode"] == "none" else "PhaseAug"
    loggers = setup_wandb_logger(f"{phase_label}_{args.dataset}_{session_model_name}", config, save_dir=paths["logs"])

    epochs = config["classifier"]["lstm"]["epochs"] if is_rul else config["classifier"]["cnn1d"]["epochs"]
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        precision=_resolve_trainer_precision(config, is_rul),
        logger=loggers,
        callbacks=[early_stop, checkpoint_callback, metrics_tracker],
        log_every_n_steps=10,
    )

    print(f"[Classifier] Training {model_name} on {args.dataset} with augmentation mode '{augmentation_summary['mode']}'")
    print(f"[Classifier] Outputs: {paths['root']}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dm.val_dataloader())
    dm.setup(stage="test")
    trainer.test(model=model, dataloaders=dm.test_dataloader(), ckpt_path="best")

    best_model_path = checkpoint_callback.best_model_path
    if augmentation_summary["mode"] != "none":
        session.write_json("augmentation_summary.json", augmentation_summary)
    session.update_manifest(
        {
            "phase": "phase_0" if augmentation_summary["mode"] == "none" else "phase_augmented_classifier",
            "best_model_path": best_model_path,
            "classifier_model": model_name,
            "input_dim": input_dim,
            "window_size": dataset_cfg["window_size"],
            "augmentation": augmentation_summary,
        }
    )
    evaluation_metrics, _ = evaluate_classifier_run(
        track=args.track,
        dataset=args.dataset,
        model=model_name,
        run_id=session.run_id,
        config_path=args.config,
        aug=args.aug,
        gen_model=args.gen_model,
        source_run_id=args.source_run_id,
        gen_ablation=args.gen_ablation,
    )
    if config.get("logging", {}).get("use_wandb", False) and wandb is not None and wandb.run is not None:
        wandb.log({f"classifier_evaluation/{key}": value for key, value in evaluation_metrics.items()})
    print(f"[Classifier] Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
