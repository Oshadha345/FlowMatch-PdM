import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.loggers import WandbLogger
except Exception:  # pragma: no cover - wandb is optional at runtime
    WandbLogger = None


RUN_ID_PREFIX = "run_"
RUN_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def _results_root() -> Path:
    return Path("results")


def _normalize_run_id(run_id: Optional[str]) -> Optional[str]:
    if run_id is None:
        return None
    return run_id if run_id.startswith(RUN_ID_PREFIX) else f"{RUN_ID_PREFIX}{run_id}"


def _latest_run_dir(model_root: Path) -> Path:
    candidates = sorted(
        [path for path in model_root.iterdir() if path.is_dir() and path.name.startswith(RUN_ID_PREFIX)],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {model_root}")
    return candidates[-1]


def resolve_model_root(track: str, dataset: str, model_name: str) -> Path:
    return _results_root() / track / dataset / model_name


def resolve_experiment_model_name(model_name: str, ablation: Optional[str] = None) -> str:
    if not ablation or ablation == "none":
        return model_name
    return f"{model_name}_ablation_{ablation}"


def resolve_classifier_experiment_name(
    model_name: str,
    aug: Optional[str] = None,
    gen_model: Optional[str] = None,
    gen_run_id: Optional[str] = None,
    gen_ablation: Optional[str] = None,
) -> str:
    if gen_model and gen_run_id:
        gen_name = resolve_experiment_model_name(gen_model, gen_ablation)
        normalized_gen_run_id = _normalize_run_id(gen_run_id) or gen_run_id
        aug_tag = f"gen_{gen_name}_{normalized_gen_run_id}"
        return f"{model_name}_aug_{aug_tag}"
    if aug and aug != "none":
        return f"{model_name}_aug_{aug}"
    return model_name


def resolve_run_dir(track: str, dataset: str, model_name: str, run_id: Optional[str] = None) -> Path:
    model_root = resolve_model_root(track, dataset, model_name)
    if not model_root.exists():
        raise FileNotFoundError(f"Model directory not found: {model_root}")

    normalized_run_id = _normalize_run_id(run_id)
    if normalized_run_id is not None:
        run_dir = model_root / normalized_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    return _latest_run_dir(model_root)


def resolve_checkpoint(run_dir: Path, checkpoint_group: str) -> Path:
    checkpoint_dir = run_dir / checkpoint_group
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    ckpts = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    return ckpts[-1]


class SessionManager:
    """
    Centralized run directory manager used by all training and evaluation scripts.
    """

    def __init__(
        self,
        track: str,
        dataset: str,
        model_name: str,
        config: Dict[str, Any],
        run_id: Optional[str] = None,
        existing_ok: bool = True,
    ):
        self.track = track
        self.dataset = dataset
        self.model_name = model_name
        self.config = config

        timestamp = datetime.now().strftime(RUN_TIMESTAMP_FMT)
        self.run_id = _normalize_run_id(run_id) or f"{RUN_ID_PREFIX}{timestamp}"
        self.base_dir = resolve_model_root(track, dataset, model_name)
        self.run_dir = self.base_dir / self.run_id

        self.paths = {
            "root": self.run_dir,
            "logs": self.run_dir / "logs",
            "best_models_generator": self.run_dir / "best_models_generator",
            "best_model_classifier": self.run_dir / "best_model_classifier",
            "generator_datas": self.run_dir / "generator_datas",
            "evaluation_results": self.run_dir / "evaluation_results",
        }

        self.base_dir.mkdir(parents=True, exist_ok=True)
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=existing_ok)

        self.config_path = self.run_dir / "run_configs.yaml"
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        self.manifest_path = self.run_dir / "run_manifest.json"
        self._write_manifest(
            {
                "track": track,
                "dataset": dataset,
                "model_name": model_name,
                "run_id": self.run_id,
                "created_at": timestamp,
            }
        )

        print(f"[SessionManager] Initialized {self.run_dir}")

    @classmethod
    def from_existing(cls, track: str, dataset: str, model_name: str, run_id: Optional[str] = None) -> "SessionManager":
        run_dir = resolve_run_dir(track, dataset, model_name, run_id=run_id)
        config_path = run_dir / "run_configs.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Run config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        manager = cls.__new__(cls)
        manager.track = track
        manager.dataset = dataset
        manager.model_name = model_name
        manager.config = config
        manager.base_dir = run_dir.parent
        manager.run_id = run_dir.name
        manager.run_dir = run_dir
        manager.paths = {
            "root": run_dir,
            "logs": run_dir / "logs",
            "best_models_generator": run_dir / "best_models_generator",
            "best_model_classifier": run_dir / "best_model_classifier",
            "generator_datas": run_dir / "generator_datas",
            "evaluation_results": run_dir / "evaluation_results",
        }
        manager.config_path = config_path
        manager.manifest_path = run_dir / "run_manifest.json"
        for path in manager.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return manager

    def _read_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _write_manifest(self, payload: Dict[str, Any]) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def update_manifest(self, payload: Dict[str, Any]) -> None:
        manifest = self._read_manifest()
        manifest.update(payload)
        self._write_manifest(manifest)

    def write_json(self, relative_path: str, payload: Dict[str, Any]) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return target

    def write_text(self, relative_path: str, content: str) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def save_numpy(self, relative_path: str, array: Any) -> Path:
        import numpy as np

        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, array)
        return target

    def get_paths(self) -> Dict[str, str]:
        return {key: str(value) for key, value in self.paths.items()}


def setup_wandb_logger(experiment_name: str, config: Dict[str, Any], save_dir: str):
    """
    Returns a trainer-compatible logger bundle.
    CSV logging is always enabled under the run directory.
    W&B is optional and controlled by config.logging.use_wandb.
    """

    loggers = [
        CSVLogger(save_dir=save_dir, name="csv_logs", version=""),
    ]

    use_wandb = bool(config.get("logging", {}).get("use_wandb", False))
    if use_wandb and WandbLogger is not None:
        loggers.append(
            WandbLogger(
                project=config.get("project_name", "FlowMatch-PdM"),
                name=experiment_name,
                save_dir=save_dir,
                config=config,
                log_model=False,
            )
        )

    return loggers if len(loggers) > 1 else loggers[0]


class JSONMetricsTracker(pl.Callback):
    """
    Lightning callback that stores epoch history and final test metrics inside a run directory.
    """

    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = Path(output_path)
        self.history = {"train_loss": [], "val_loss": []}

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.history["train_loss"].append(float(train_loss.detach().cpu()))

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.history["val_loss"].append(float(val_loss.detach().cpu()))

    def on_test_end(self, trainer, pl_module):
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if not key.startswith("test_"):
                continue
            if hasattr(value, "detach"):
                metrics[key] = float(value.detach().cpu())
            else:
                metrics[key] = float(value)

        payload = {
            "test_metrics": metrics,
            "history": self.history,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
