# src/evaluation.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.linalg import sqrtm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# Ad-Hoc Networks for Discriminative and Predictive Scores
# ==============================================================================
class RNNClassifier(nn.Module):
    """Simple GRU used to distinguish Real vs Synthetic data."""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h = self.rnn(x)
        return self.sigmoid(self.linear(h[-1]))

class RNNPredictor(nn.Module):
    """Simple GRU used to predict the next time step (TSTR metric)."""
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.linear(out[:, -1, :])

# ==============================================================================
# Master Evaluator Class
# ==============================================================================
class TimeSeriesEvaluator:
    """
    Comprehensive evaluation suite for synthetic time-series data.
    Implements the 5 gold-standard metrics for Generative Time Series.
    """
    def __init__(self, real_data: np.ndarray, synthetic_data: np.ndarray, save_dir: str):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure dimensions match
        assert self.real_data.shape == self.synthetic_data.shape, "Real and Synthetic datasets must have the same shape."
        self.N, self.window_size, self.input_dim = self.real_data.shape
        
        # Flatten for spatial/statistical analysis: (N, Window * Features)
        self.real_flat = real_data.reshape(self.N, -1)
        self.syn_flat = synthetic_data.reshape(self.N, -1)

    # --------------------------------------------------------------------------
    # 1. Fréchet Time Series Distance (FTSD)
    # --------------------------------------------------------------------------
    def calculate_ftsd(self, real_features: np.ndarray = None, synthetic_features: np.ndarray = None) -> float:
        """
        Calculates FTSD. If deep features aren't provided, falls back to raw flattened sequences.
        """
        r_feats = real_features if real_features is not None else self.real_flat
        s_feats = synthetic_features if synthetic_features is not None else self.syn_flat
        
        mu_r, sigma_r = r_feats.mean(axis=0), np.cov(r_feats, rowvar=False)
        mu_s, sigma_s = s_feats.mean(axis=0), np.cov(s_feats, rowvar=False)

        ssdiff = np.sum((mu_r - mu_s)**2.0)
        covmean = sqrtm(sigma_r.dot(sigma_s))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        ftsd = ssdiff + np.trace(sigma_r + sigma_s - 2.0 * covmean)
        return float(ftsd)

    # --------------------------------------------------------------------------
    # 2. Maximum Mean Discrepancy (MMD)
    # --------------------------------------------------------------------------
    def calculate_mmd(self, kernel_mul=2.0, kernel_num=5):
        """Measures the distance between distributions using RBF kernels."""
        limit = min(2000, len(self.real_flat)) # Memory cap
        X = torch.tensor(self.real_flat[:limit], dtype=torch.float32)
        Y = torch.tensor(self.syn_flat[:limit], dtype=torch.float32)
        
        xx, yy, zz = torch.mm(X, X.t()), torch.mm(Y, Y.t()), torch.mm(X, Y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx, dyy, dxy = rx.t() + rx - 2. * xx, ry.t() + ry - 2. * yy, rx.t() + ry - 2. * zz
        
        bandwidths = [kernel_mul ** i for i in range(kernel_num)]
        XX, YY, XY = torch.zeros_like(xx), torch.zeros_like(yy), torch.zeros_like(zz)
        
        for a in bandwidths:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
            
        mmd = torch.mean(XX + YY - 2. * XY)
        return mmd.item()

    # --------------------------------------------------------------------------
    # 3. Discriminative Score (Post-hoc Classifier)
    # --------------------------------------------------------------------------
    def calculate_discriminative_score(self, epochs=30):
        """
        Trains an RNN to distinguish real vs. synthetic. 
        Score = |0.5 - Accuracy|. Lower is better (0 means perfectly indistinguishable).
        """
        print("[Evaluator] Computing Discriminative Score...")
        X = np.vstack([self.real_data, self.synthetic_data])
        y = np.array([1]*self.N + [0]*self.N) # 1: Real, 0: Synthetic
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        
        model = RNNClassifier(input_dim=self.input_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(X_train_t)
            loss = criterion(preds, y_train_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            test_preds = torch.round(model(X_test_t)).cpu().numpy()
            
        acc = accuracy_score(y_test, test_preds)
        score = abs(0.5 - acc)
        return score, acc

    # --------------------------------------------------------------------------
    # 4. Predictive Score (Train on Synthetic, Test on Real)
    # --------------------------------------------------------------------------
    def calculate_predictive_score(self, epochs=30):
        """
        TSTR metric: Predicts step t+1 given 0..t.
        Trains on Synthetic data, evaluates MAE on Real data. Lower is better.
        """
        print("[Evaluator] Computing Predictive Score (TSTR)...")
        # Task: use X[:, :-1, :] to predict X[:, -1, :]
        X_syn, Y_syn = self.synthetic_data[:, :-1, :], self.synthetic_data[:, -1, :]
        X_real, Y_real = self.real_data[:, :-1, :], self.real_data[:, -1, :]
        
        X_syn_t = torch.tensor(X_syn, dtype=torch.float32).to(self.device)
        Y_syn_t = torch.tensor(Y_syn, dtype=torch.float32).to(self.device)
        X_real_t = torch.tensor(X_real, dtype=torch.float32).to(self.device)
        
        model = RNNPredictor(input_dim=self.input_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.L1Loss() # MAE
        
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(X_syn_t)
            loss = criterion(preds, Y_syn_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            real_preds = model(X_real_t).cpu().numpy()
            
        mae = mean_absolute_error(Y_real, real_preds)
        return mae

    # --------------------------------------------------------------------------
    # 5. Visualizations (PCA, t-SNE, Distributions)
    # --------------------------------------------------------------------------
    def plot_pca_tsne(self):
        """Generates rigorous 2D projections to assess manifold overlap."""
        print("[Evaluator] Computing PCA and t-SNE projections...")
        combined = np.vstack([self.real_flat, self.syn_flat])
        labels = np.array([0]*self.N + [1]*self.N)
        
        pca_results = PCA(n_components=2).fit_transform(combined)
        tsne_results = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42).fit_transform(combined)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for ax, results, title in zip(axes, [pca_results, tsne_results], ['PCA Projection', 't-SNE Projection']):
            ax.scatter(results[labels==0, 0], results[labels==0, 1], alpha=0.3, label='Real', color='blue')
            ax.scatter(results[labels==1, 0], results[labels==1, 1], alpha=0.3, label='Synthetic', color='red')
            ax.set_title(title)
            ax.legend()

        save_path = os.path.join(self.save_dir, "dimensionality_reduction.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_marginal_distributions(self):
        """Plots KDE for a random feature to verify statistical overlap."""
        print("[Evaluator] Plotting marginal distributions...")
        # Select the first feature of the final timestep as a representative sample
        real_feat = self.real_data[:, -1, 0]
        syn_feat = self.synthetic_data[:, -1, 0]
        
        plt.figure(figsize=(8, 6))
        sns.kdeplot(real_feat, label='Real Data', color='blue', fill=True, alpha=0.3)
        sns.kdeplot(syn_feat, label='Synthetic Data', color='red', fill=True, alpha=0.3)
        plt.title('Kernel Density Estimation: Real vs Synthetic (Feature 0, Last Timestep)')
        plt.legend()
        
        save_path = os.path.join(self.save_dir, "marginal_distributions.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # --------------------------------------------------------------------------
    # Execution
    # --------------------------------------------------------------------------
    def run_full_suite(self, real_features=None, synthetic_features=None):
        print("\n" + "="*50)
        print("🔍 Commencing Comprehensive Synthetic Data Evaluation")
        print("="*50)
        
        ftsd_score = self.calculate_ftsd(real_features, synthetic_features)
        mmd_score = self.calculate_mmd()
        disc_score, disc_acc = self.calculate_discriminative_score()
        pred_score = self.calculate_predictive_score()
        
        self.plot_pca_tsne()
        self.plot_marginal_distributions()
        
        report = (
            f"--- EVALUATION METRICS ---\n"
            f"Fréchet Time Series Distance (FTSD): {ftsd_score:.4f} \n"
            f"Maximum Mean Discrepancy (MMD):      {mmd_score:.4f} \n"
            f"Discriminative Score (|0.5 - Acc|):  {disc_score:.4f} (Accuracy: {disc_acc:.4f})\n"
            f"Predictive Score (TSTR MAE):         {pred_score:.4f} \n"
        )
        
        print(f"\n{report}")
        
        with open(os.path.join(self.save_dir, "metrics.txt"), "w") as f:
            f.write(report)
            
        print(f"✅ Evaluation complete. Plots and metrics saved to {self.save_dir}")