import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from src.baselines import ClassicalAugmenter
from src.classifier import CNN1DClassifier, LSTMRegressor
from src.utils.data_helper import get_data_module, get_dataset_config
from src.utils.logger_utils import JSONMetricsTracker, SessionManager, setup_wandb_logger


CLASSIFIER_CHOICES = ["baseline", "LSTMRegressor", "CNN1DClassifier"]


def _resolve_classifier_name(track: str, requested_model: str) -> str:
    default_model = "LSTMRegressor" if "rul" in track else "CNN1DClassifier"
    model_name = default_model if requested_model == "baseline" else requested_model
    if "rul" in track and model_name != "LSTMRegressor":
        raise ValueError(f"Track '{track}' requires LSTMRegressor, received '{model_name}'.")
    if track == "bearing_fault" and model_name != "CNN1DClassifier":
        raise ValueError(f"Track '{track}' requires CNN1DClassifier, received '{model_name}'.")
    return model_name


def _build_model(track: str, model_name: str, config: dict, input_dim: int, num_classes: int):
    if model_name == "LSTMRegressor":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=config["classifier"]["lstm"]["hidden_dim"],
            num_layers=config["classifier"]["lstm"]["num_layers"],
            learning_rate=config["classifier"]["lstm"]["lr"],
        )
    return CNN1DClassifier(
        num_classes=num_classes,
        input_channels=input_dim,
        learning_rate=config["classifier"]["cnn1d"]["lr"],
    )


def _collect_dataset_tensors(dataset):
    xs = []
    ys = []
    for x, y in dataset:
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu() if torch.is_tensor(y) else torch.tensor(y))
    return torch.stack(xs), torch.stack(ys)


def _load_generator_augmented_dataset(dm, args):
    generator_session = SessionManager.from_existing(args.track, args.dataset, args.gen_model, run_id=args.run_id)
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
    mixed_ds = ConcatDataset([dm.train_ds, synthetic_ds])
    summary = {
        "mode": "generator",
        "source_run_id": generator_session.run_id,
        "source_model": args.gen_model,
        "synthetic_count": int(len(synthetic_ds)),
        "real_count": int(len(dm.train_ds)),
        "total_count": int(len(mixed_ds)),
    }
    return mixed_ds, summary


def _build_augmented_dataset(dm, args, config):
    dm.prepare_data()
    dm.setup(stage="fit")

    if args.run_id:
        return _load_generator_augmented_dataset(dm, args)

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
        }

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
        }

    return dm.train_ds, {
        "mode": "none",
        "real_count": int(len(dm.train_ds)),
        "augmented_count": int(len(dm.train_ds)),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1/3: train classifier on classical or synthetic augmentation.")
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, or bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, FEMTO, CWRU, etc.")
    parser.add_argument("--model", type=str, default="baseline", choices=CLASSIFIER_CHOICES)
    parser.add_argument("--gen_model", type=str, default="FlowMatch", help="Generator model folder for synthetic augmentation.")
    parser.add_argument("--run_id", type=str, default=None, help="Generator run identifier for synthetic augmentation.")
    parser.add_argument("--aug", type=str, default=None, choices=["smote", "noise"], help="Classical augmentation mode.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    pl.seed_everything(config["seed"], workers=True)

    dataset_cfg = get_dataset_config(config, args.dataset)
    model_name = _resolve_classifier_name(args.track, args.model)
    is_rul = "rul" in args.track
    batch_size = config["classifier"]["lstm"]["batch_size"] if is_rul else config["classifier"]["cnn1d"]["batch_size"]

    dm = get_data_module(
        track=args.track,
        dataset_name=args.dataset,
        window_size=dataset_cfg["window_size"],
        batch_size=batch_size,
    )
    aug_train_ds, augmentation_summary = _build_augmented_dataset(dm, args, config)
    aug_train_loader = DataLoader(
        aug_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    dm.prepare_data()
    dm.setup(stage="fit")
    sample_x, _ = dm.train_ds[0]
    input_dim = int(sample_x.shape[-1])
    num_classes = int(getattr(dm, "num_classes", dataset_cfg.get("num_classes", 1)))
    model = _build_model(args.track, model_name, config, input_dim, num_classes)

    aug_tag = args.aug or (f"gen_{args.gen_model}_{args.run_id}" if args.run_id else "none")
    session = SessionManager(
        track=args.track,
        dataset=args.dataset,
        model_name=f"{model_name}_aug_{aug_tag}",
        config=config,
    )
    paths = session.get_paths()

    checkpoint_callback = ModelCheckpoint(
        dirpath=paths["best_model_classifier"],
        filename=f"{model_name}-aug-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=config["trainer"]["patience"], mode="min")
    metrics_tracker = JSONMetricsTracker(output_path=str(Path(paths["evaluation_results"]) / "phase3_metrics.json"))
    loggers = setup_wandb_logger(f"Phase3_{args.dataset}_{model_name}_{aug_tag}", config, save_dir=paths["logs"])

    epochs = config["classifier"]["lstm"]["epochs"] if is_rul else config["classifier"]["cnn1d"]["epochs"]
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        precision=config["trainer"]["precision"],
        logger=loggers,
        callbacks=[early_stop, checkpoint_callback, metrics_tracker],
        log_every_n_steps=10,
    )

    print(f"[Phase 3] Training {model_name} with augmentation mode '{augmentation_summary['mode']}'")
    print(f"[Phase 3] Outputs: {paths['root']}")
    trainer.fit(model, train_dataloaders=aug_train_loader, val_dataloaders=dm.val_dataloader())
    trainer.test(model=model, dataloaders=dm.test_dataloader(), ckpt_path="best")

    session.write_json("augmentation_summary.json", augmentation_summary)
    session.update_manifest(
        {
            "phase": "phase_3",
            "classifier_model": model_name,
            "augmentation": augmentation_summary,
            "best_model_path": checkpoint_callback.best_model_path,
            "input_dim": input_dim,
            "window_size": dataset_cfg["window_size"],
        }
    )
    print(f"[Phase 3] Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
