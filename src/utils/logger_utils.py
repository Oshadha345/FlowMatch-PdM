# src/utils/logger_utils.py
import json
import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# src/utils/logger_utils.py
import os
import yaml
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger

class SessionManager:
    """
    Centralized file I/O and session tracker.
    Builds the isolated run directory structure for every experiment.
    """
    def __init__(self, track: str, dataset: str, model_name: str, config: dict):
        self.track = track
        self.dataset = dataset
        self.model_name = model_name
        self.config = config
        
        # Determine the next run number based on existing folders
        self.base_dir = os.path.join("results", self.track, self.dataset, self.model_name)
        os.makedirs(self.base_dir, exist_ok=True)
        
        existing_runs = [d for d in os.listdir(self.base_dir) if d.startswith("run")]
        next_run_idx = len(existing_runs) + 1
        
        # Timestamp: e.g., 20260308_1427
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_id = f"run{next_run_idx}_{timestamp}"
        
        # Create full paths
        self.run_dir = os.path.join(self.base_dir, self.run_id)
        
        self.paths = {
            "root": self.run_dir,
            "best_models_generator": os.path.join(self.run_dir, "best_models_generator"),
            "best_model_classifier": os.path.join(self.run_dir, "best_model_classifier"),
            "generator_datas": os.path.join(self.run_dir, "generator_datas"),
            "evaluation_results": os.path.join(self.run_dir, "evaluation_results")
        }
        
        # Build directories
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
            
        # Dump the exact config used for this run to pin down hyperparams
        self.config_path = os.path.join(self.run_dir, "run_configs.yaml")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        print(f"\n📂 [SessionManager] Initialized {self.run_dir}")

    def get_paths(self):
        return self.paths

def setup_wandb_logger(experiment_name: str, config: dict, save_dir: str):
    """Initializes Weights & Biases logging inside the specific run directory."""
    return WandbLogger(
        project=config.get('project_name', 'FlowMatch-PdM'),
        name=experiment_name,
        save_dir=save_dir,
        config=config,
        log_model=False # We handle model saving via PyTorch Lightning Callbacks locally
    )



class JSONMetricsTracker(pl.Callback):
    """
    Lightning Callback that automatically saves validation and test metrics
    to a JSON file for easy LaTeX table generation later.
    """
    def __init__(self, track: str, dataset_name: str, model_name: str):
        super().__init__()
        self.track = track
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.metrics_dir = os.path.join("results", "metrics", track, dataset_name)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.metrics_dir, f"{model_name}_{timestamp}.json")
        self.history = {"val_loss": [], "train_loss": []}

    def on_validation_epoch_end(self, trainer, pl_module):
        # Capture validation loss from the LightningModule's logged metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.history["val_loss"].append(val_loss.item())

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.history["train_loss"].append(train_loss.item())

    def on_test_end(self, trainer, pl_module):
        # Dump the final test metrics and epoch history to JSON
        test_metrics = {k: v.item() for k, v in trainer.callback_metrics.items() if "test" in k}
        
        output_data = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "track": self.track,
            "test_results": test_metrics,
            "training_curve": self.history
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"\n[Logging] Metrics perfectly saved to {self.filepath}")