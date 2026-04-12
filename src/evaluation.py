import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class RealSyntheticGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.head(hidden[:, -1])


class NextStepGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.head(hidden)


class _TSTRSequenceClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(x)
        return self.head(hidden[:, -1])


class TSTR_Evaluation:
    """
    Train-on-synthetic, test-on-real gate for downstream classification readiness.

    The gate compares a lightweight classifier trained on synthetic labels against a
    matched classifier trained on a held-out split of real data. Synthetic data only
    passes when it retains enough of the real decision boundary to approach the
    real-trained reference performance.
    """

    def __init__(
        self,
        save_dir: str,
        batch_size: int = 128,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        hidden_dim: int = 64,
        min_relative_f1: float = 0.8,
        min_relative_balanced_accuracy: float = 0.8,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.hidden_dim = int(hidden_dim)
        self.min_relative_f1 = float(min_relative_f1)
        self.min_relative_balanced_accuracy = float(min_relative_balanced_accuracy)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

    def _write_report(self, filename_prefix: str, metrics: Dict[str, float], extra_payload: Optional[dict] = None):
        lines = [f"{filename_prefix.replace('_', ' ').title()} Evaluation"]
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool, np.floating)):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")

        (self.save_dir / f"{filename_prefix}_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        payload = {"metrics": metrics}
        if extra_payload:
            payload.update(extra_payload)
        (self.save_dir / f"{filename_prefix}_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _build_loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.int64)),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _fit_classifier(self, model: nn.Module, x_train: np.ndarray, y_train: np.ndarray):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        loader = self._build_loader(x_train, y_train, shuffle=True)

        model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    def _evaluate_classifier(self, model: nn.Module, x_eval: np.ndarray, y_eval: np.ndarray) -> Dict[str, float]:
        loader = self._build_loader(x_eval, y_eval, shuffle=False)
        preds = []
        probs = []
        targets = []

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                logits = model(batch_x.to(self.device))
                prob = torch.softmax(logits, dim=1)
                preds.append(torch.argmax(prob, dim=1).cpu().numpy())
                probs.append(prob.cpu().numpy())
                targets.append(batch_y.numpy())

        y_true = np.concatenate(targets, axis=0)
        y_pred = np.concatenate(preds, axis=0)
        y_prob = np.concatenate(probs, axis=0)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "cross_entropy": float(log_loss(y_true, y_prob, labels=np.unique(y_true))),
        }

    def run(
        self,
        synthetic_data: np.ndarray,
        synthetic_targets: np.ndarray,
        real_data: np.ndarray,
        real_targets: np.ndarray,
        filename_prefix: str = "tstr",
    ) -> Dict[str, float]:
        synthetic_data = np.asarray(synthetic_data, dtype=np.float32)
        synthetic_targets = np.asarray(synthetic_targets).reshape(-1)
        real_data = np.asarray(real_data, dtype=np.float32)
        real_targets = np.asarray(real_targets).reshape(-1)

        if synthetic_data.ndim != 3 or real_data.ndim != 3:
            raise ValueError("TSTR_Evaluation expects [batch, window, features] arrays for both synthetic and real data.")
        if len(synthetic_data) != len(synthetic_targets):
            raise ValueError("Synthetic features and targets must have the same length.")
        if len(real_data) != len(real_targets):
            raise ValueError("Real features and targets must have the same length.")

        synthetic_label_values = np.unique(synthetic_targets)
        real_label_values = np.unique(real_targets)

        if synthetic_label_values.size < 2:
            metrics = {
                "gate_passed": False,
                "tstr_applicable": False,
                "failure_reason": "synthetic_targets_have_fewer_than_two_classes",
                "synthetic_num_classes": int(synthetic_label_values.size),
                "real_num_classes": int(real_label_values.size),
            }
            self._write_report(filename_prefix, metrics)
            return metrics

        if real_label_values.size < 2:
            metrics = {
                "gate_passed": False,
                "tstr_applicable": False,
                "failure_reason": "real_targets_have_fewer_than_two_classes",
                "synthetic_num_classes": int(synthetic_label_values.size),
                "real_num_classes": int(real_label_values.size),
            }
            self._write_report(filename_prefix, metrics)
            return metrics

        label_values = np.unique(np.concatenate([synthetic_targets, real_targets], axis=0))
        label_to_index = {label: index for index, label in enumerate(label_values.tolist())}
        y_syn = np.asarray([label_to_index[label] for label in synthetic_targets], dtype=np.int64)
        y_real = np.asarray([label_to_index[label] for label in real_targets], dtype=np.int64)

        if np.min(np.bincount(y_real)) < 2:
            metrics = {
                "gate_passed": False,
                "tstr_applicable": False,
                "failure_reason": "real_targets_do_not_support_stratified_reference_split",
                "synthetic_num_classes": int(synthetic_label_values.size),
                "real_num_classes": int(real_label_values.size),
            }
            self._write_report(filename_prefix, metrics)
            return metrics

        input_dim = int(synthetic_data.shape[-1])
        num_classes = int(len(label_values))

        tstr_model = _TSTRSequenceClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dim=self.hidden_dim).to(self.device)
        self._fit_classifier(tstr_model, synthetic_data, y_syn)
        tstr_metrics = self._evaluate_classifier(tstr_model, real_data, y_real)

        x_real_train, x_real_test, y_real_train, y_real_test = train_test_split(
            real_data,
            y_real,
            test_size=0.3,
            random_state=42,
            stratify=y_real,
        )
        trtr_model = _TSTRSequenceClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dim=self.hidden_dim).to(self.device)
        self._fit_classifier(trtr_model, x_real_train, y_real_train)
        trtr_metrics = self._evaluate_classifier(trtr_model, x_real_test, y_real_test)

        relative_f1 = float(tstr_metrics["f1_macro"] / max(trtr_metrics["f1_macro"], 1e-8))
        relative_balanced_accuracy = float(
            tstr_metrics["balanced_accuracy"] / max(trtr_metrics["balanced_accuracy"], 1e-8)
        )
        gate_passed = (
            relative_f1 >= self.min_relative_f1
            and relative_balanced_accuracy >= self.min_relative_balanced_accuracy
        )

        metrics = {
            "gate_passed": bool(gate_passed),
            "tstr_applicable": True,
            "synthetic_num_classes": int(synthetic_label_values.size),
            "real_num_classes": int(real_label_values.size),
            "tstr_accuracy": tstr_metrics["accuracy"],
            "tstr_balanced_accuracy": tstr_metrics["balanced_accuracy"],
            "tstr_f1_macro": tstr_metrics["f1_macro"],
            "tstr_mcc": tstr_metrics["mcc"],
            "tstr_cross_entropy": tstr_metrics["cross_entropy"],
            "trtr_accuracy": trtr_metrics["accuracy"],
            "trtr_balanced_accuracy": trtr_metrics["balanced_accuracy"],
            "trtr_f1_macro": trtr_metrics["f1_macro"],
            "trtr_mcc": trtr_metrics["mcc"],
            "trtr_cross_entropy": trtr_metrics["cross_entropy"],
            "relative_f1_macro": relative_f1,
            "relative_balanced_accuracy": relative_balanced_accuracy,
            "min_relative_f1": self.min_relative_f1,
            "min_relative_balanced_accuracy": self.min_relative_balanced_accuracy,
        }

        self._write_report(
            filename_prefix,
            metrics,
            extra_payload={
                "num_synthetic_samples": int(len(synthetic_data)),
                "num_real_samples": int(len(real_data)),
                "label_values": label_values.tolist(),
            },
        )
        return metrics


class SupervisedTaskEvaluator:
    def __init__(
        self,
        model: nn.Module,
        task_type: str,
        save_dir: str,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.task_type = task_type
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

        if self.task_type not in {"regression", "classification"}:
            raise ValueError(f"Unsupported task_type '{self.task_type}'.")

    def _collect_predictions(self, dataloader: DataLoader):
        self.model = self.model.to(self.device)
        self.model.eval()
        target_scale = float(getattr(self.model, "target_scale", 1.0))

        targets = []
        predictions = []
        probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(self.device)
                outputs = self.model(x)

                if self.task_type == "classification":
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    probabilities.append(probs.detach().cpu().numpy())
                    predictions.append(preds.detach().cpu().numpy())
                    targets.append(y.detach().cpu().numpy())
                else:
                    preds_real = outputs.detach().cpu().reshape(-1) * target_scale
                    y_real = y.detach().cpu().float().reshape(-1) * target_scale
                    predictions.append(preds_real.numpy())
                    targets.append(y_real.numpy())

        y_true = np.concatenate(targets, axis=0)
        y_pred = np.concatenate(predictions, axis=0)
        y_prob = np.concatenate(probabilities, axis=0) if probabilities else None
        return y_true, y_pred, y_prob

    def _write_report(self, filename_prefix: str, metrics: Dict[str, float], extra_payload: Optional[dict] = None):
        lines = [f"{filename_prefix.replace('_', ' ').title()} Evaluation"]
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating)):
                lines.append(f"{key}: {float(value):.6f}")
            else:
                lines.append(f"{key}: {value}")

        (self.save_dir / f"{filename_prefix}_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        payload = {"metrics": metrics}
        if extra_payload:
            payload.update(extra_payload)
        (self.save_dir / f"{filename_prefix}_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _plot_regression_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray):
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(y_true, y_pred, s=18, alpha=0.5, color="#0b5d8f")
        low = min(float(y_true.min()), float(y_pred.min()))
        high = max(float(y_true.max()), float(y_pred.max()))
        axes[0].plot([low, high], [low, high], linestyle="--", color="#d95f02", linewidth=2)
        axes[0].set_title("Prediction vs Ground Truth")
        axes[0].set_xlabel("Ground Truth")
        axes[0].set_ylabel("Prediction")

        sns.histplot(residuals, bins=40, kde=True, ax=axes[1], color="#d95f02")
        axes[1].set_title("Residual Distribution")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(self.save_dir / "classifier_regression_diagnostics.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        labels = np.unique(np.concatenate([y_true, y_pred], axis=0))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        display_labels = self.class_names if self.class_names and len(self.class_names) == len(labels) else [str(label) for label in labels]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=display_labels, yticklabels=display_labels)
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        normalized = matrix / row_sums
        sns.heatmap(normalized, annot=True, fmt=".2f", cmap="Blues", ax=axes[1], xticklabels=display_labels, yticklabels=display_labels)
        axes[1].set_title("Normalized Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")

        fig.tight_layout()
        fig.savefig(self.save_dir / "classifier_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def evaluate(self, dataloader: DataLoader, filename_prefix: str = "classifier") -> Dict[str, float]:
        y_true, y_pred, y_prob = self._collect_predictions(dataloader)

        if self.task_type == "regression":
            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_true, y_pred))
            median_ae = float(median_absolute_error(y_true, y_pred))
            max_err = float(max_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            explained_variance = float(explained_variance_score(y_true, y_pred))
            denom = np.maximum(np.abs(y_true), 1e-8)
            mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
            smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
            smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / smape_denom) * 100.0)

            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "median_ae": median_ae,
                "max_error": max_err,
                "r2": r2,
                "explained_variance": explained_variance,
                "mape": mape,
                "smape": smape,
            }
            self._plot_regression_diagnostics(y_true, y_pred)
            self._write_report(
                filename_prefix,
                metrics,
                extra_payload={
                    "task_type": self.task_type,
                    "num_samples": int(len(y_true)),
                },
            )
            return metrics

        cross_entropy = float(log_loss(y_true, y_prob, labels=np.unique(y_true))) if y_prob is not None else float("nan")

        metrics = {
            "cross_entropy": cross_entropy,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        }

        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        self._plot_confusion_matrix(y_true, y_pred)
        self._write_report(
            filename_prefix,
            metrics,
            extra_payload={
                "task_type": self.task_type,
                "num_samples": int(len(y_true)),
                "classification_report": report,
            },
        )
        return metrics


class TimeSeriesEvaluator:
    def __init__(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        save_dir: str,
        feature_extractor: Optional[nn.Module] = None,
        batch_size: int = 128,
        max_samples: int = 2048,
        discriminative_epochs: int = 20,
        predictive_epochs: int = 20,
    ):
        self.real_data = np.asarray(real_data, dtype=np.float32)
        self.synthetic_data = np.asarray(synthetic_data, dtype=np.float32)
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.discriminative_epochs = discriminative_epochs
        self.predictive_epochs = predictive_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.real_data.ndim != 3 or self.synthetic_data.ndim != 3:
            raise ValueError("Expected 3D arrays with shape [batch, window, features].")

        limit = min(len(self.real_data), len(self.synthetic_data), self.max_samples)
        self.real_data = self.real_data[:limit]
        self.synthetic_data = self.synthetic_data[:limit]
        self.n_samples, self.window_size, self.input_dim = self.real_data.shape

        self.real_flat = self.real_data.reshape(self.n_samples, -1)
        self.synthetic_flat = self.synthetic_data.reshape(self.n_samples, -1)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _tensor_loader(self, array: np.ndarray, shuffle: bool = False) -> DataLoader:
        tensor = torch.from_numpy(array.astype(np.float32))
        return DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=shuffle)

    def _extract_deep_features(self, array: np.ndarray) -> np.ndarray:
        if self.feature_extractor is None:
            return array.reshape(len(array), -1)

        model = self.feature_extractor.to(self.device)
        model.eval()
        outputs = []
        with torch.no_grad():
            for (batch,) in self._tensor_loader(array):
                batch = batch.to(self.device)
                if hasattr(model, "extract_features"):
                    feats = model.extract_features(batch)
                else:
                    feats = model(batch)
                outputs.append(feats.detach().cpu().reshape(batch.size(0), -1).numpy())
        return np.concatenate(outputs, axis=0)

    def _stable_covariance(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)
        cov = np.atleast_2d(cov)
        cov += np.eye(cov.shape[0], dtype=np.float64) * 1e-6
        return mean.astype(np.float64), cov.astype(np.float64)

    def calculate_ftsd(self, real_features: np.ndarray, synthetic_features: np.ndarray) -> float:
        mu_r, cov_r = self._stable_covariance(real_features)
        mu_s, cov_s = self._stable_covariance(synthetic_features)
        mean_term = np.sum((mu_r - mu_s) ** 2)
        covmean = sqrtm(cov_r @ cov_s)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(mean_term + np.trace(cov_r + cov_s - 2.0 * covmean))

    def calculate_mmd(self) -> float:
        x = torch.from_numpy(self.real_flat)
        y = torch.from_numpy(self.synthetic_flat)
        combined = torch.cat([x, y], dim=0)
        distances = torch.cdist(combined, combined, p=2).pow(2)
        positive_distances = distances[distances > 0]
        median = torch.tensor(1.0) if positive_distances.numel() == 0 else torch.median(positive_distances)
        bandwidth = torch.clamp(median, min=1e-6)

        k_xx = torch.exp(-torch.cdist(x, x, p=2).pow(2) / (2.0 * bandwidth))
        k_yy = torch.exp(-torch.cdist(y, y, p=2).pow(2) / (2.0 * bandwidth))
        k_xy = torch.exp(-torch.cdist(x, y, p=2).pow(2) / (2.0 * bandwidth))

        n = x.size(0)
        m = y.size(0)
        mmd = (
            (k_xx.sum() - torch.diagonal(k_xx).sum()) / max(n * (n - 1), 1)
            + (k_yy.sum() - torch.diagonal(k_yy).sum()) / max(m * (m - 1), 1)
            - 2.0 * k_xy.mean()
        )
        return float(max(mmd.item(), 0.0))

    def calculate_discriminative_score(self) -> Tuple[float, float]:
        x = np.concatenate([self.real_data, self.synthetic_data], axis=0)
        y = np.concatenate([np.ones(len(self.real_data)), np.zeros(len(self.synthetic_data))], axis=0)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model = RealSyntheticGRU(self.input_dim).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test.astype(np.float32))),
            batch_size=self.batch_size,
            shuffle=False,
        )

        model.train()
        for _ in range(self.discriminative_epochs):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(-1)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits = model(batch_x.to(self.device))
                preds.append((torch.sigmoid(logits) >= 0.5).cpu().numpy())
                targets.append(batch_y.numpy())

        y_pred = np.concatenate(preds, axis=0).reshape(-1)
        y_true = np.concatenate(targets, axis=0).reshape(-1)
        accuracy = float((y_pred == y_true).mean())
        score = abs(0.5 - accuracy)
        return float(score), accuracy

    def calculate_predictive_score(self) -> float:
        x_syn = self.synthetic_data[:, :-1, :]
        y_syn = self.synthetic_data[:, 1:, :]
        x_real = self.real_data[:, :-1, :]
        y_real = self.real_data[:, 1:, :]

        model = NextStepGRU(self.input_dim).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.L1Loss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_syn), torch.from_numpy(y_syn)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_real), torch.from_numpy(y_real)),
            batch_size=self.batch_size,
            shuffle=False,
        )

        model.train()
        for _ in range(self.predictive_epochs):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        errors = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                pred = model(batch_x.to(self.device)).cpu()
                errors.append(torch.mean(torch.abs(pred - batch_y), dim=(1, 2)).numpy())
        return float(np.concatenate(errors).mean())

    def plot_pca_tsne(self, real_features: np.ndarray, synthetic_features: np.ndarray):
        combined = np.concatenate([real_features, synthetic_features], axis=0)
        labels = np.concatenate([np.zeros(len(real_features)), np.ones(len(synthetic_features))], axis=0)
        scaled = StandardScaler().fit_transform(combined)

        pca = PCA(n_components=2, random_state=42).fit_transform(scaled)
        perplexity = max(2, min(30, max(len(combined) // 8, 2), len(combined) - 1))
        tsne = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(scaled)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        palettes = [("Real", "#0b5d8f"), ("Synthetic", "#d95f02")]

        for ax, coords, title in zip(axes, [pca, tsne], ["PCA", "t-SNE"]):
            for label_value, (label_name, color) in enumerate(palettes):
                mask = labels == label_value
                ax.scatter(coords[mask, 0], coords[mask, 1], s=20, alpha=0.5, label=label_name, color=color)
            ax.set_title(f"{title} Projection")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend()

        fig.tight_layout()
        fig.savefig(self.save_dir / "projection_pca_tsne.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_marginal_kde(self):
        feature_count = min(4, self.input_dim)
        fig, axes = plt.subplots(1, feature_count, figsize=(5 * feature_count, 4.5), squeeze=False)
        axes = axes.reshape(-1)

        for feature_idx in range(feature_count):
            real_feature = self.real_data[:, :, feature_idx].reshape(-1)
            synthetic_feature = self.synthetic_data[:, :, feature_idx].reshape(-1)
            sns.kdeplot(real_feature, ax=axes[feature_idx], label="Real", color="#0b5d8f", fill=True, alpha=0.25)
            sns.kdeplot(
                synthetic_feature,
                ax=axes[feature_idx],
                label="Synthetic",
                color="#d95f02",
                fill=True,
                alpha=0.25,
            )
            axes[feature_idx].set_title(f"Feature {feature_idx}")
            axes[feature_idx].set_xlabel("Value")
            axes[feature_idx].set_ylabel("Density")
            axes[feature_idx].legend()

        fig.tight_layout()
        fig.savefig(self.save_dir / "marginal_kde.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def run_full_suite(self) -> Dict[str, float]:
        real_features = self._extract_deep_features(self.real_data)
        synthetic_features = self._extract_deep_features(self.synthetic_data)

        metrics = {
            "ftsd": self.calculate_ftsd(real_features, synthetic_features),
            "mmd_rbf": self.calculate_mmd(),
        }
        metrics["discriminative_score"], metrics["discriminative_accuracy"] = self.calculate_discriminative_score()
        metrics["predictive_score_mae"] = self.calculate_predictive_score()

        self.plot_pca_tsne(real_features, synthetic_features)
        self.plot_marginal_kde()

        report = "\n".join(
            [
                "FlowMatch-PdM Synthetic Time-Series Evaluation",
                f"Samples compared: {self.n_samples}",
                f"Window size: {self.window_size}",
                f"Feature dimension: {self.input_dim}",
                f"FTSD: {metrics['ftsd']:.6f}",
                f"MMD (RBF): {metrics['mmd_rbf']:.6f}",
                f"Discriminative Score: {metrics['discriminative_score']:.6f}",
                f"Discriminative Accuracy: {metrics['discriminative_accuracy']:.6f}",
                f"Predictive Score (TSTR MAE): {metrics['predictive_score_mae']:.6f}",
            ]
        )
        (self.save_dir / "metrics.txt").write_text(report + "\n", encoding="utf-8")
        (self.save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metrics
