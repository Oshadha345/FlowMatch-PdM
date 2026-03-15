import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
