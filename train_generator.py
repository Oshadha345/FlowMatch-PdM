# src/train_generator.py
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.utils.data_helper import get_data_module
from src.utils.logger_utils import SessionManager, setup_wandb_logger

# 1. Actual Imports of the Models and Callbacks
from src.baselines import TimeGAN, TimeVAE, DiffusionTS, TimeFlow
from flowmatchPdM.flowmatch_model import FlowMatchPdM
from flowmatchPdM.LAP import LayerAdaptivePruningCallback

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train Generative Models on Minority Data")
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, or bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, CWRU, etc.")
    parser.add_argument("--model", type=str, required=True, help="TimeGAN, TimeVAE, DiffusionTS, TimeFlow, FlowMatch")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    # Map CLI model name to config keys
    model_key_map = {
        "TimeVAE": "timevae",
        "TimeGAN": "timegan",
        "DiffusionTS": "diffusion",
        "TimeFlow": "timeflow",
        "FlowMatch": "ss_flowmatch"
    }
    
    if args.model not in model_key_map:
        raise ValueError(f"Unknown generative model: {args.model}. Choose from {list(model_key_map.keys())}")
        
    model_cfg = config['generative'][model_key_map[args.model]]

    # --------------------------------------------------------------------------
    # NEW: Initialize Session Manager & Route Output Paths
    # --------------------------------------------------------------------------
    session = SessionManager(track=args.track, dataset=args.dataset, model_name=args.model, config=config)
    paths = session.get_paths()

    # 2. Load DataModule and Extract Minority Class
    window_size = config['datasets']['window_size_engine'] if 'rul' in args.track else config['datasets']['window_size_bearing']
    
    dm = get_data_module(track=args.track, dataset_name=args.dataset, window_size=window_size, batch_size=model_cfg['batch_size'])
    dm.prepare_data()
    dm.setup(stage='fit')
    
    minority_ds = dm.get_minority_dataset(rul_threshold_ratio=config['datasets']['rul_threshold'])
    minority_loader = DataLoader(minority_ds, batch_size=model_cfg['batch_size'], shuffle=True, num_workers=4)

    # 3. Instantiate Generative Model
    if args.track == "engine_rul":
        input_dim = 14
    elif args.track == "bearing_rul":
        input_dim = 2
    else:
        input_dim = 1

    print(f"\n⚙️ Instantiating {args.model} Generator for {args.dataset} (Input Dim: {input_dim}, Window: {window_size})...")
    
    callbacks = []

    if args.model == "TimeVAE":
        model = TimeVAE(input_dim=input_dim, window_size=window_size, latent_dim=model_cfg['latent_dim'], hidden_dim=model_cfg['hidden_dim'], lr=float(model_cfg['lr']))
    elif args.model == "TimeGAN":
        model = TimeGAN(input_dim=input_dim, window_size=window_size, hidden_dim=model_cfg['hidden_dim'], noise_dim=model_cfg['noise_dim'], lr=float(model_cfg['lr']))
    elif args.model == "DiffusionTS":
        model = DiffusionTS(input_dim=input_dim, window_size=window_size, timesteps=model_cfg['timesteps'], lr=float(model_cfg['lr']))
    elif args.model == "TimeFlow":
        model = TimeFlow(input_dim=input_dim, window_size=window_size, hidden_dim=model_cfg['hidden_dim'], euler_steps=model_cfg['euler_steps'], lr=float(model_cfg['lr']))
    elif args.model == "FlowMatch":
        model = FlowMatchPdM(input_dim=input_dim, window_size=window_size, config=model_cfg)
        
        lap_callback = LayerAdaptivePruningCallback(
            alpha=0.2, 
            beta=0.1, 
            stability_threshold=0.05
        )
        callbacks.append(lap_callback)

    # 4. Setup Logging and Checkpointing using Session Manager
    experiment_name = f"Generator_{args.dataset}_{args.model}"
    wandb_logger = setup_wandb_logger(experiment_name, config, save_dir=paths['root'])
    
    # Checkpoints now explicitly drop into the generator subfolder
    checkpoint_callback = ModelCheckpoint(
        dirpath=paths['best_models_generator'],
        filename=f"{args.model}-best",
        save_top_k=1,
        monitor="train_loss" if args.model != "TimeGAN" else "g_loss", 
        mode="min"
    )
    callbacks.append(checkpoint_callback)

    # 5. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=model_cfg['epochs'], 
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        precision=config['trainer']['precision'],
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10
    )

    # 6. Execute Training
    print(f"\n🚀 [START] Training {args.model} purely on {args.dataset} anomaly/degradation data...")
    print(f"📂 [LOGS] Routing all outputs to: {paths['root']}")
    trainer.fit(model, train_dataloaders=minority_loader)
    
    print(f"\n✅ [SUCCESS] Generator weights saved to {paths['best_models_generator']}")

if __name__ == "__main__":
    main()