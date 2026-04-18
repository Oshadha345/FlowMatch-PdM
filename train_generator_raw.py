import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from flowmatchPdM.flowmatch_pdm_raw_backup import FlowMatchPdMRawBackup
from run_evaluation import evaluate_generator_run
from flowmatchPdM.model.lap import LayerAdaptivePruningCallback
from src.utils.data_helper import get_data_module, get_dataset_config
from src.utils.logger_utils import (
    checkpoint_completed_training,
    SessionManager,
    resolve_resume_checkpoint,
    resolve_trainer_runtime,
    session_exists,
    setup_wandb_logger,
)

try:
    import wandb
except Exception:  # pragma: no cover - wandb is optional at runtime
    wandb = None


ABLATION_CHOICES = ["none", "no_prior", "no_harmonic_prior", "no_tccm", "no_lap"]


def _normalize_ablation(ablation: str) -> str:
    if ablation == "no_harmonic_prior":
        return "no_prior"
    return ablation


def _resolve_generator_precision(config: dict):
    precision_cfg = config["trainer"].get("precision", "16-mixed")
    if isinstance(precision_cfg, dict):
        return precision_cfg.get("generator", precision_cfg.get("fault", "16-mixed"))
    return precision_cfg


def _get_minority_dataset(dm, rul_threshold_ratio: float):
    try:
        return dm.get_minority_dataset(rul_threshold_ratio=rul_threshold_ratio)
    except TypeError:
        return dm.get_minority_dataset()


def _apply_generator_ablation(config: dict, ablation: str) -> dict:
    run_config = deepcopy(config)
    run_config.setdefault("experiment", {})["generator_ablation"] = ablation

    model_cfg = run_config["generative"]["flowmatch_pdm"]
    model_cfg["use_harmonic_prior"] = True
    model_cfg["use_tccm"] = True
    model_cfg["use_lap"] = True

    if ablation == "no_prior":
        model_cfg["use_harmonic_prior"] = False
    elif ablation == "no_tccm":
        model_cfg["use_tccm"] = False
        model_cfg["tccm_lambda"] = 0.0
    elif ablation == "no_lap":
        model_cfg["use_lap"] = False

    return run_config


def _apply_model_overrides(config: dict, args) -> dict:
    model_cfg = config["generative"]["flowmatch_pdm"]

    if args.lr is not None:
        model_cfg["lr"] = float(args.lr)
    if args.batch_size is not None:
        model_cfg["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        model_cfg["epochs"] = int(args.epochs)
    if args.euler_steps is not None:
        model_cfg["euler_steps"] = int(args.euler_steps)
    if args.mamba_d_model is not None:
        model_cfg["mamba_d_model"] = int(args.mamba_d_model)
    if args.mamba_d_state is not None:
        model_cfg["mamba_d_state"] = int(args.mamba_d_state)
    if args.tccm_lambda is not None:
        model_cfg["tccm_lambda"] = float(args.tccm_lambda)
    if args.lap_threshold is not None:
        model_cfg["lap_threshold"] = float(args.lap_threshold)

    return config


def _resolve_generator_batch_size(dataset_cfg: dict, model_cfg: dict) -> int:
    return int(dataset_cfg.get("generator_batch_size", model_cfg["batch_size"]))


def _load_best_model(checkpoint_path: str, input_dim: int, window_size: int, model_cfg: dict):
    return FlowMatchPdMRawBackup.load_from_checkpoint(
        checkpoint_path,
        input_dim=input_dim,
        window_size=window_size,
        config=model_cfg,
        map_location="cpu",
    )


def _generate_and_save_artifacts(session: SessionManager, dataset: str, config: dict, checkpoint_path: str, input_dim: int):
    dataset_cfg = get_dataset_config(config, dataset)
    dm = get_data_module(
        track="bearing_rul",
        dataset_name=dataset,
        conditions=dataset_cfg.get("conditions", dataset_cfg.get("fd_list", 1)),
        window_size=dataset_cfg["window_size"],
        batch_size=config.get("evaluation", {}).get("batch_size", 128),
        append_condition_features=dataset_cfg.get("append_condition_features", False),
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    minority_ds = _get_minority_dataset(dm, config["datasets"]["minority_rul_ratio"])

    real_x = []
    real_y = []
    for x_item, y_item in minority_ds:
        real_x.append(x_item.detach().cpu().numpy())
        if torch.is_tensor(y_item):
            real_y.append(y_item.detach().cpu().numpy())
        else:
            real_y.append(np.asarray(y_item))

    real_data = np.stack(real_x).astype(np.float32)
    real_targets = np.stack(real_y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["generative"]["flowmatch_pdm"]
    model = _load_best_model(checkpoint_path, input_dim=input_dim, window_size=dataset_cfg["window_size"], model_cfg=model_cfg)
    model.eval().to(device)

    conditions = torch.tensor(real_targets, dtype=torch.float32, device=device)
    if conditions.dim() == 1:
        conditions = conditions.unsqueeze(-1)

    with torch.no_grad():
        synthetic = model.generate(conditions=conditions, num_samples=len(real_targets)).detach().cpu().numpy().astype(np.float32)

    session.save_numpy("generator_datas/synthetic_data.npy", synthetic)
    session.save_numpy("generator_datas/synthetic_targets.npy", real_targets)
    session.save_numpy("generator_datas/real_minority_data.npy", real_data)
    session.save_numpy("generator_datas/real_minority_targets.npy", real_targets)
    session.write_json(
        "generator_datas/generation_manifest.json",
        {
            "generator_model": "FlowMatch",
            "generator_variant": "raw_backup",
            "generator_run_id": session.run_id,
            "generator_checkpoint": checkpoint_path,
            "num_generated_samples": int(len(real_targets)),
            "window_size": int(dataset_cfg["window_size"]),
            "input_dim": int(input_dim),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Train the preserved raw FlowMatch generator on the minority/degraded subset.")
    parser.add_argument("--track", type=str, required=True, help="Active pivot track: bearing_rul")
    parser.add_argument("--dataset", type=str, required=True, help="Active pivot datasets: FEMTO or XJTU-SY")
    parser.add_argument("--model", type=str, default="FlowMatch", choices=["FlowMatch"])
    parser.add_argument("--run_id", type=str, default=None, help="Run identifier. Reuses and resumes the session if it already exists.")
    parser.add_argument("--ablation", type=str, default="none", choices=ABLATION_CHOICES, help="Raw FlowMatch ablation variant.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging for this run.")
    parser.add_argument("--lr", type=float, default=None, help="Override generator learning rate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override generator batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override generator epochs.")
    parser.add_argument("--euler_steps", type=int, default=None, help="Override flow integration steps.")
    parser.add_argument("--mamba_d_model", type=int, default=None, help="Override raw FlowMatch Mamba model width.")
    parser.add_argument("--mamba_d_state", type=int, default=None, help="Override raw FlowMatch Mamba state size.")
    parser.add_argument("--tccm_lambda", type=float, default=None, help="Override raw FlowMatch TCCM weight.")
    parser.add_argument("--lap_threshold", type=float, default=None, help="Override raw FlowMatch LAP threshold.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    ablation = _normalize_ablation(args.ablation)
    experiment_model_name = "FlowMatch" if ablation == "none" else f"FlowMatch_ablation_{ablation}"
    resuming_existing_session = bool(args.run_id) and session_exists(
        args.track,
        args.dataset,
        experiment_model_name,
        args.run_id,
    )

    if resuming_existing_session:
        session = SessionManager.from_existing(
            track=args.track,
            dataset=args.dataset,
            model_name=experiment_model_name,
            run_id=args.run_id,
        )
        base_config = deepcopy(session.config)
        print(f"[Raw Phase 2] Reusing existing session: {session.run_dir}")
    else:
        with Path(args.config).open("r", encoding="utf-8") as handle:
            base_config = yaml.safe_load(handle)
        session = None

    config = _apply_generator_ablation(base_config, ablation)
    config = _apply_model_overrides(config, args)
    if args.use_wandb:
        config.setdefault("logging", {})["use_wandb"] = True
    pl.seed_everything(config["seed"], workers=True)

    model_cfg = config["generative"]["flowmatch_pdm"]
    dataset_cfg = get_dataset_config(config, args.dataset)
    model_cfg["batch_size"] = _resolve_generator_batch_size(dataset_cfg, model_cfg)

    if session is None:
        session = SessionManager(
            track=args.track,
            dataset=args.dataset,
            model_name=experiment_model_name,
            config=config,
            run_id=args.run_id,
        )
    else:
        session.write_config(config)
    paths = session.get_paths()

    dm = get_data_module(
        track=args.track,
        dataset_name=args.dataset,
        conditions=dataset_cfg.get("conditions", dataset_cfg.get("fd_list", 1)),
        window_size=dataset_cfg["window_size"],
        batch_size=model_cfg["batch_size"],
        append_condition_features=dataset_cfg.get("append_condition_features", False),
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    minority_ds = _get_minority_dataset(dm, config["datasets"]["minority_rul_ratio"])
    minority_loader = DataLoader(
        minority_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    sample_x, _ = dm.train_ds[0]
    input_dim = int(sample_x.shape[-1])
    model = FlowMatchPdMRawBackup(input_dim=input_dim, window_size=dataset_cfg["window_size"], config=model_cfg)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=paths["best_models_generator"],
        filename="FlowMatch-best",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    if model_cfg.get("use_lap", True):
        callbacks.append(LayerAdaptivePruningCallback(alpha=0.2, beta=0.1, stability_threshold=0.05))

    loggers = setup_wandb_logger(f"RawPhase2_{args.dataset}_{experiment_model_name}", config, save_dir=paths["logs"])
    trainer_runtime = resolve_trainer_runtime(config)
    trainer = pl.Trainer(
        max_epochs=model_cfg["epochs"],
        accelerator=trainer_runtime["accelerator"],
        devices=trainer_runtime["devices"],
        precision=_resolve_generator_precision(config),
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    print(f"[Raw Phase 2] Training FlowMatch on minority subset of {args.dataset} (ablation={ablation})")
    print(f"[Raw Phase 2] Outputs: {paths['root']}")
    resume_checkpoint = resolve_resume_checkpoint(session.run_dir, "best_models_generator")
    resume_ckpt_path = None
    skip_training = False
    if resume_checkpoint is not None:
        if checkpoint_completed_training(resume_checkpoint, model_cfg["epochs"]):
            skip_training = True
            print(f"[Raw Phase 2] Existing checkpoint already reached max_epochs={model_cfg['epochs']}. Skipping fit.")
        else:
            resume_ckpt_path = str(resume_checkpoint)
            print(f"[Raw Phase 2] Resuming fit from checkpoint: {resume_checkpoint}")

    if not skip_training:
        trainer.fit(model, train_dataloaders=minority_loader, ckpt_path=resume_ckpt_path)

    manifest = session.read_manifest()
    best_model_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path or manifest.get("best_model_path")
    if not best_model_path:
        resolved_checkpoint = resolve_resume_checkpoint(session.run_dir, "best_models_generator")
        best_model_path = str(resolved_checkpoint) if resolved_checkpoint is not None else ""
    else:
        best_model_path = Path(best_model_path)
        if not best_model_path.is_absolute():
            best_model_path = session.run_dir / best_model_path
        best_model_path = str(best_model_path)

    if best_model_path:
        _generate_and_save_artifacts(
            session=session,
            dataset=args.dataset,
            config=config,
            checkpoint_path=best_model_path,
            input_dim=input_dim,
        )

    session.update_manifest(
        {
            "phase": "phase_2_raw",
            "generator_model": "FlowMatch",
            "generator_variant": "raw_backup",
            "best_model_path": best_model_path,
            "minority_size": len(minority_ds),
            "input_dim": input_dim,
            "window_size": dataset_cfg["window_size"],
            "ablation": ablation,
            "resumed_existing_session": resuming_existing_session,
            "training_skipped": skip_training,
        }
    )
    evaluation_metrics, _ = evaluate_generator_run(
        track=args.track,
        dataset=args.dataset,
        model="FlowMatch",
        run_id=session.run_id,
        config_path=args.config,
        ablation=ablation,
    )
    if config.get("logging", {}).get("use_wandb", False) and wandb is not None and wandb.run is not None:
        wandb.log({"raw_generator/minority_size": len(minority_ds)})
        wandb.log({f"evaluation/{key}": value for key, value in evaluation_metrics.items()})
    print(f"[Raw Phase 2] Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
