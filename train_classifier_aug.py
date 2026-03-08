# train_classifier_aug.py
"""
Phase 1 & 3: Train Classifiers with Mixed Data Augmentation.

Modes:
  --aug smote   : SMOTE oversampling of minority classes (classical baseline).
  --aug noise   : Gaussian jittering augmentation (classical baseline).
  --run_id ID   : Load FlowMatch-PdM / baseline generator synthetic data from
                  results/.../[run_id]/generator_datas/synthetic_data.npy and
                  concatenate with real minority samples.

Results are routed through SessionManager for full reproducibility.
"""
import argparse
import os
import yaml

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from src.utils.data_helper import get_data_module
from src.utils.logger_utils import SessionManager, setup_wandb_logger
from src.classifier import CNN1DClassifier, LSTMRegressor
from src.baselines import ClassicalAugmenter


def _build_augmented_dataset(dm, args, config):
    """
    Build a balanced training dataset by mixing real data with augmented/synthetic
    samples. Returns a new TensorDataset ready for the classifier.
    """
    dm.prepare_data()
    dm.setup(stage="fit")

    # Real training tensors
    X_train = torch.stack([x for x, _ in dm.train_ds])
    y_train = torch.stack([y for _, y in dm.train_ds])

    # --- Classical augmentation modes ---
    if args.aug in ("smote", "noise"):
        X_np = X_train.numpy()
        y_np = y_train.numpy()

        if args.aug == "smote":
            k = config.get("classical", {}).get("smote", {}).get("k_neighbors", 5)
            X_aug, y_aug = ClassicalAugmenter.apply_smote(X_np, y_np, k_neighbors=k)
            print(f"[Aug] SMOTE: {X_np.shape[0]} → {X_aug.shape[0]} samples")
        else:  # noise / jittering
            sigma = config.get("classical", {}).get("jittering", {}).get("sigma", 0.05)
            X_jittered = ClassicalAugmenter.apply_jittering(X_np, sigma=sigma)
            X_aug = np.concatenate([X_np, X_jittered], axis=0)
            y_aug = np.concatenate([y_np, y_np], axis=0)
            print(f"[Aug] Jittering (σ={sigma}): {X_np.shape[0]} → {X_aug.shape[0]} samples")

        return TensorDataset(
            torch.tensor(X_aug, dtype=torch.float32),
            torch.tensor(y_aug, dtype=torch.long),
        )

    # --- Generative augmentation mode (Phase 3: Mixed) ---
    if args.run_id:
        # Locate generator model name from run directory structure
        # Convention: results/<track>/<dataset>/<model_name>/<run_id>/
        gen_model = args.gen_model if args.gen_model else "FlowMatch"
        syn_dir = os.path.join(
            "results", args.track, args.dataset, gen_model,
            args.run_id, "generator_datas",
        )
        syn_path = os.path.join(syn_dir, "synthetic_data.npy")
        if not os.path.exists(syn_path):
            raise FileNotFoundError(f"Synthetic data not found at {syn_path}")

        synthetic = np.load(syn_path).astype(np.float32)
        print(f"[Aug] Loaded {synthetic.shape[0]} synthetic samples from {syn_path}")

        # Extract real minority samples
        minority_sub = dm.get_minority_dataset()
        X_min = torch.stack([x for x, _ in minority_sub])
        y_min = torch.stack([y for _, y in minority_sub])

        # Assign minority label to synthetic samples (use most common minority label)
        unique_labels, counts = torch.unique(y_min, return_counts=True)
        dominant_label = unique_labels[counts.argmax()].item()

        # If synthetic data has matching batch size, pair labels from minority set
        n_syn = synthetic.shape[0]
        if n_syn <= len(y_min):
            y_syn = y_min[:n_syn]
        else:
            # Repeat minority labels cyclically to cover all synthetic samples
            repeats = (n_syn // len(y_min)) + 1
            y_syn = y_min.repeat(repeats)[:n_syn]

        X_syn = torch.tensor(synthetic, dtype=torch.float32)

        # Concatenate: full real training set + synthetic minority
        X_mixed = torch.cat([X_train, X_syn], dim=0)
        y_mixed = torch.cat([y_train, y_syn], dim=0)
        print(f"[Aug] Mixed dataset: {X_train.shape[0]} real + {n_syn} synthetic = {X_mixed.shape[0]} total")

        return TensorDataset(X_mixed, y_mixed)

    # Fallback: no augmentation, return original
    print("[Aug] No augmentation specified — using raw training data.")
    return dm.train_ds


def main():
    parser = argparse.ArgumentParser(description="Phase 1/3: Train Classifier with Augmentation")
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, or bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, CWRU, etc.")
    parser.add_argument("--aug", type=str, default=None, choices=["smote", "noise"],
                        help="Classical augmentation strategy")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Generator run-id to load synthetic .npy from results/")
    parser.add_argument("--gen_model", type=str, default="FlowMatch",
                        help="Name of the generator model folder under results/ (default: FlowMatch)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config["seed"], workers=True)

    # Determine track-specific settings
    is_lstm = "rul" in args.track
    window_size = config["datasets"]["window_size_engine"] if is_lstm else config["datasets"]["window_size_bearing"]
    batch_size = config["classifier"]["lstm"]["batch_size"] if is_lstm else config["classifier"]["cnn1d"]["batch_size"]

    # DataModule (used for val/test splits and minority extraction)
    dm = get_data_module(track=args.track, dataset_name=args.dataset,
                         window_size=window_size, batch_size=batch_size)

    # Build augmented training dataset
    aug_train_ds = _build_augmented_dataset(dm, args, config)
    aug_train_loader = DataLoader(aug_train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)

    # Ensure val/test are ready
    dm.prepare_data()
    dm.setup(stage="fit")

    # Instantiate classifier
    aug_tag = args.aug or f"gen_{args.run_id}" if args.run_id else "none"
    if args.track == "engine_rul":
        model = LSTMRegressor(
            input_dim=14,
            hidden_dim=config["classifier"]["lstm"]["hidden_dim"],
            num_layers=config["classifier"]["lstm"]["num_layers"],
            learning_rate=config["classifier"]["lstm"]["lr"],
        )
    elif args.track == "bearing_rul":
        model = LSTMRegressor(
            input_dim=2,
            hidden_dim=config["classifier"]["lstm"]["hidden_dim"],
            num_layers=config["classifier"]["lstm"]["num_layers"],
            learning_rate=config["classifier"]["lstm"]["lr"],
        )
    elif args.track == "bearing_fault":
        num_classes = 10 if args.dataset.upper() == "CWRU" else 32
        model = CNN1DClassifier(
            num_classes=num_classes,
            learning_rate=config["classifier"]["cnn1d"]["lr"],
        )

    # Session Manager — routes all outputs to an isolated run folder
    model_tag = f"{model.__class__.__name__}_aug_{aug_tag}"
    session = SessionManager(
        track=args.track, dataset=args.dataset,
        model_name=model_tag, config=config,
    )
    paths = session.get_paths()

    # Logger & callbacks
    experiment_name = f"AugClassify_{args.dataset}_{aug_tag}"
    wandb_logger = setup_wandb_logger(experiment_name, config, save_dir=paths["root"])

    early_stop = EarlyStopping(monitor="val_loss", patience=config["trainer"]["patience"], mode="min")
    checkpoint_cb = ModelCheckpoint(
        dirpath=paths["best_model_classifier"],
        filename="best-aug-classifier",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    epochs = config["classifier"]["lstm"]["epochs"] if is_lstm else config["classifier"]["cnn1d"]["epochs"]

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        precision=config["trainer"]["precision"],
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint_cb],
        log_every_n_steps=10,
    )

    # Train with augmented data, validate/test on original splits
    print(f"\n[START] Training {model.__class__.__name__} with aug={aug_tag} on {args.dataset}")
    print(f"[LOGS]  Routing outputs to: {paths['root']}")

    trainer.fit(
        model,
        train_dataloaders=aug_train_loader,
        val_dataloaders=dm.val_dataloader(),
    )

    print(f"\n[TEST] Evaluating on unseen test set...")
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == "__main__":
    main()
