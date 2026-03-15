import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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
    if model_name == "CNN1DClassifier":
        return CNN1DClassifier(
            num_classes=num_classes,
            input_channels=input_dim,
            learning_rate=config["classifier"]["cnn1d"]["lr"],
        )
    raise ValueError(f"Unsupported classifier model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Train baseline classifiers/regressors.")
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, or bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, FEMTO, CWRU, etc.")
    parser.add_argument("--model", type=str, default="baseline", choices=CLASSIFIER_CHOICES)
    parser.add_argument("--run_id", type=str, default=None, help="Optional output run identifier.")
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
    dm.prepare_data()
    dm.setup(stage="fit")

    sample_x, _ = dm.train_ds[0]
    input_dim = int(sample_x.shape[-1])
    num_classes = int(getattr(dm, "num_classes", dataset_cfg.get("num_classes", 1)))
    model = _build_model(args.track, model_name, config, input_dim, num_classes)

    session = SessionManager(
        track=args.track,
        dataset=args.dataset,
        model_name=model_name,
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
    metrics_tracker = JSONMetricsTracker(output_path=str(Path(paths["evaluation_results"]) / "phase0_metrics.json"))
    loggers = setup_wandb_logger(f"Phase0_{args.dataset}_{model_name}", config, save_dir=paths["logs"])

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

    print(f"[Phase 0] Training {model_name} on {args.dataset} ({args.track})")
    print(f"[Phase 0] Outputs: {paths['root']}")
    trainer.fit(model, datamodule=dm)
    trainer.test(model=model, datamodule=dm, ckpt_path="best")

    best_model_path = checkpoint_callback.best_model_path
    session.update_manifest(
        {
            "phase": "phase_0",
            "best_model_path": best_model_path,
            "classifier_model": model_name,
            "input_dim": input_dim,
            "window_size": dataset_cfg["window_size"],
        }
    )
    print(f"[Phase 0] Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
