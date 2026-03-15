import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from flowmatchPdM.flowmatch_pdm import FlowMatchPdM
from flowmatchPdM.model.lap import LayerAdaptivePruningCallback
from src.baselines import COTGAN, DiffusionTS, FaultDiffusion, TimeFlow, TimeGAN, TimeVAE
from src.utils.data_helper import get_data_module, get_dataset_config
from src.utils.logger_utils import SessionManager, setup_wandb_logger


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


def _build_generator(model_name: str, model_cfg: dict, input_dim: int, window_size: int):
    if model_name == "TimeVAE":
        return TimeVAE(
            input_dim=input_dim,
            window_size=window_size,
            latent_dim=model_cfg["latent_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            lr=float(model_cfg["lr"]),
        )
    if model_name == "TimeGAN":
        return TimeGAN(
            input_dim=input_dim,
            window_size=window_size,
            hidden_dim=model_cfg["hidden_dim"],
            noise_dim=model_cfg["noise_dim"],
            lr=float(model_cfg["lr"]),
        )
    if model_name == "DiffusionTS":
        return DiffusionTS(
            input_dim=input_dim,
            window_size=window_size,
            timesteps=model_cfg["timesteps"],
            lr=float(model_cfg["lr"]),
            base_channels=model_cfg.get("base_channels", 64),
            num_heads=model_cfg.get("num_heads", 4),
        )
    if model_name == "TimeFlow":
        return TimeFlow(
            input_dim=input_dim,
            window_size=window_size,
            hidden_dim=model_cfg["hidden_dim"],
            euler_steps=model_cfg["euler_steps"],
            lr=float(model_cfg["lr"]),
        )
    if model_name == "COTGAN":
        return COTGAN(
            input_dim=input_dim,
            window_size=window_size,
            hidden_dim=model_cfg["hidden_dim"],
            noise_dim=model_cfg["noise_dim"],
            lr=float(model_cfg["lr"]),
            sinkhorn_eps=float(model_cfg["sinkhorn_eps"]),
            sinkhorn_iters=int(model_cfg["sinkhorn_iters"]),
            martingale_weight=float(model_cfg["martingale_weight"]),
            causal_weight=float(model_cfg["causal_weight"]),
            critic_projection_dim=int(model_cfg["critic_projection_dim"]),
        )
    if model_name == "FaultDiffusion":
        return FaultDiffusion(
            input_dim=input_dim,
            window_size=window_size,
            timesteps=model_cfg["timesteps"],
            lr=float(model_cfg["lr"]),
            base_channels=model_cfg.get("base_channels", 96),
            num_heads=model_cfg.get("num_heads", 4),
            diversity_weight=float(model_cfg.get("diversity_weight", 0.05)),
        )
    if model_name == "FlowMatch":
        return FlowMatchPdM(input_dim=input_dim, window_size=window_size, config=model_cfg)
    raise ValueError(f"Unsupported generative model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: train a generator on the minority/degraded subset.")
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, or bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, FEMTO, CWRU, etc.")
    parser.add_argument("--model", type=str, required=True, choices=GENERATOR_CHOICES)
    parser.add_argument("--run_id", type=str, default=None, help="Optional output run identifier.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    pl.seed_everything(config["seed"], workers=True)

    model_cfg = config["generative"][GENERATOR_CONFIG_MAP[args.model]]
    dataset_cfg = get_dataset_config(config, args.dataset)

    session = SessionManager(
        track=args.track,
        dataset=args.dataset,
        model_name=args.model,
        config=config,
        run_id=args.run_id,
    )
    paths = session.get_paths()

    dm = get_data_module(
        track=args.track,
        dataset_name=args.dataset,
        window_size=dataset_cfg["window_size"],
        batch_size=model_cfg["batch_size"],
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
    model = _build_generator(args.model, model_cfg, input_dim, dataset_cfg["window_size"])

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=paths["best_models_generator"],
        filename=f"{args.model}-best",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    if args.model == "FlowMatch":
        callbacks.append(LayerAdaptivePruningCallback(alpha=0.2, beta=0.1, stability_threshold=0.05))

    loggers = setup_wandb_logger(f"Phase2_{args.dataset}_{args.model}", config, save_dir=paths["logs"])
    trainer = pl.Trainer(
        max_epochs=model_cfg["epochs"],
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        precision=config["trainer"]["precision"],
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    print(f"[Phase 2] Training {args.model} on minority subset of {args.dataset}")
    print(f"[Phase 2] Outputs: {paths['root']}")
    trainer.fit(model, train_dataloaders=minority_loader)

    best_model_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    session.update_manifest(
        {
            "phase": "phase_2",
            "generator_model": args.model,
            "best_model_path": best_model_path,
            "minority_size": len(minority_ds),
            "input_dim": input_dim,
            "window_size": dataset_cfg["window_size"],
        }
    )
    print(f"[Phase 2] Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
