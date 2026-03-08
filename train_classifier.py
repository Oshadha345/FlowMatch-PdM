# src/train_classifier.py
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.data_helper import get_data_module
from src.utils.logger_utils import SessionManager, setup_wandb_logger
from src.classifier import LSTMRegressor, CNN1DClassifier

def main():
    # 1. Parse CLI Arguments
    parser = argparse.ArgumentParser(description="Phase 1: Train Baseline Classifiers")
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, or bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, CWRU, etc.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    # 2. Load Global Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    # 3. Instantiate DataModule
    window_size = config['datasets']['window_size_engine'] if 'rul' in args.track else config['datasets']['window_size_bearing']
    
    # We pull batch size from the specific classifier config now
    is_lstm = 'rul' in args.track
    batch_size = config['classifier']['lstm']['batch_size'] if is_lstm else config['classifier']['cnn1d']['batch_size']
    
    dm = get_data_module(
        track=args.track, 
        dataset_name=args.dataset, 
        window_size=window_size, 
        batch_size=batch_size
    )
    
    # 4. Instantiate Model
    if args.track == "engine_rul":
        model = LSTMRegressor(
            input_dim=14, 
            hidden_dim=config['classifier']['lstm']['hidden_dim'], 
            num_layers=config['classifier']['lstm']['num_layers'],
            learning_rate=config['classifier']['lstm']['lr']
        )
    elif args.track == "bearing_rul":
        model = LSTMRegressor(
            input_dim=2,
            hidden_dim=config['classifier']['lstm']['hidden_dim'], 
            num_layers=config['classifier']['lstm']['num_layers'],
            learning_rate=config['classifier']['lstm']['lr']
        )
    elif args.track == "bearing_fault":
        num_classes = 10 if args.dataset.upper() == "CWRU" else 32
        model = CNN1DClassifier(
            num_classes=num_classes,
            learning_rate=config['classifier']['cnn1d']['lr']
        )
    
    # --------------------------------------------------------------------------
    # 5. Initialize Session Manager & Route Output Paths
    # --------------------------------------------------------------------------
    session = SessionManager(track=args.track, dataset=args.dataset, model_name=model.__class__.__name__, config=config)
    paths = session.get_paths()

    # Setup Loggers & Callbacks using the generated run directory
    experiment_name = f"Classify_{args.dataset}_{model.__class__.__name__}"
    wandb_logger = setup_wandb_logger(experiment_name, config, save_dir=paths['root'])
    
    early_stop = EarlyStopping(monitor="val_loss", patience=config['trainer']['patience'], mode="min")
    
    # Route the checkpoint directly to the designated folder
    checkpoint_callback = ModelCheckpoint(
        dirpath=paths['best_model_classifier'], 
        filename="best-classifier",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    # 6. Initialize Trainer
    epochs = config['classifier']['lstm']['epochs'] if is_lstm else config['classifier']['cnn1d']['epochs']
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        precision=config['trainer']['precision'],
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint_callback],
        log_every_n_steps=10
    )

    # 7. Execute Training & Testing Pipeline
    print(f"\n🚀 [START] Training {model.__class__.__name__} on {args.dataset} ({args.track})...")
    print(f"📂 [LOGS] Routing all outputs to: {paths['root']}")
    trainer.fit(model, datamodule=dm)
    
    print(f"\n🧪 [TEST] Evaluating on unseen test set...")
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()