# Add SMOTE_augmenter and Jitter_Warp_augmenter.
# Add TimeGAN and TimeVAE classes
# Diffusion-TS architecture and sampling methods
# Timeflow architecture and sampling methods

# src/baselines.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math

# ==============================================================================
# 1. Classical Augmentation: SMOTE & Jittering
# ==============================================================================
class ClassicalAugmenter:
    """
    Wrapper for standard non-parametric data augmentation.
    Does not require PyTorch training.
    """
    @staticmethod
    def apply_smote(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5):
        """
        Applies Synthetic Minority Over-sampling Technique.
        Requires flattening the time-series window first.
        """
        from imblearn.over_sampling import SMOTE
        batch_size, window_size, features = X.shape
        
        # Flatten: (Batch, Window * Features)
        X_flat = X.reshape(batch_size, -1)
        
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y)
        
        # Reshape back to sequence
        X_resampled = X_resampled_flat.reshape(-1, window_size, features)
        return X_resampled, y_resampled

    @staticmethod
    def apply_jittering(X: np.ndarray, sigma: float = 0.05):
        """
        Adds Gaussian noise to the time series (Jittering).
        """
        noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
        return X + noise



# ==============================================================================
# 2. TimeVAE: Recurrent Variational Autoencoder
# ==============================================================================
class TimeVAE(pl.LightningModule):
    """
    Recurrent VAE for Time-Series Generation.
    Optimizes ELBO: Reconstruction Loss + KL Divergence.
    """
    def __init__(self, input_dim: int, window_size: int, latent_dim: int = 32, hidden_dim: int = 64, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        
        # Encoder: Extracts sequence context
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Reconstructs sequence from latent vector
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n[-1] # Take last hidden state
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        # Repeat the hidden state across the window size
        h_seq = h.unsqueeze(1).repeat(1, self.window_size, 1)
        out, _ = self.decoder_lstm(h_seq)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, list) else batch
        recon_x, mu, logvar = self(x)
        
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        # KL Divergence: D_KL(Q(z|X) || P(z))
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + (0.1 * kld_loss) # Beta-VAE weighting
        
        self.log('train_recon_loss', recon_loss, prog_bar=True)
        self.log('train_kld_loss', kld_loss, prog_bar=True)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def generate(self, num_samples: int):
        """Generates purely synthetic data from isotropic Gaussian noise."""
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)


# ==============================================================================
# 3. TimeGAN (Simplified Continuous Adversarial Baseline)
# ==============================================================================
class TimeGAN(pl.LightningModule):
    """
    A unified Recurrent GAN tailored for time-series.
    Uses an LSTM Generator and an LSTM Discriminator.
    """
    def __init__(self, input_dim: int, window_size: int, hidden_dim: int = 64, noise_dim: int = 32, lr: float = 2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False # Manual control for GAN alternating updates
        
        self.generator = nn.Sequential(
            nn.LSTM(noise_dim, hidden_dim, batch_first=True),
        )
        self.generator_out = nn.Linear(hidden_dim, input_dim)
        
        self.discriminator = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, batch_first=True),
        )
        self.discriminator_out = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward_g(self, z):
        out, _ = self.generator(z)
        return self.generator_out(out)
        
    def forward_d(self, x):
        out, _ = self.discriminator(x)
        return self.discriminator_out(out[:, -1, :]) # Discriminate based on the final state

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, list) else batch
        opt_g, opt_d = self.optimizers()
        batch_size = x.size(0)
        
        # 1. Train Discriminator
        opt_d.zero_grad()
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        z = torch.randn(batch_size, self.hparams.window_size, self.hparams.noise_dim, device=self.device)
        fake_data = self.forward_g(z)
        
        d_loss_real = F.binary_cross_entropy(self.forward_d(x), real_labels)
        d_loss_fake = F.binary_cross_entropy(self.forward_d(fake_data.detach()), fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # 2. Train Generator
        opt_g.zero_grad()
        g_loss = F.binary_cross_entropy(self.forward_d(fake_data), real_labels)
        self.manual_backward(g_loss)
        opt_g.step()
        
        self.log('d_loss', d_loss, prog_bar=True)
        self.log('g_loss', g_loss, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return [opt_g, opt_d], []


# ==============================================================================
# 4. Diffusion-TS (DDPM specific for Time-Series)
# ==============================================================================
class DiffusionTS(pl.LightningModule):
    """
    Time-Series Denoising Diffusion Probabilistic Model.
    Employs a 1D-CNN UNet-style architecture to reverse the diffusion process.
    """
    def __init__(self, input_dim: int, window_size: int, timesteps: int = 1000, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.timesteps = timesteps
        
        # Simple 1D Denoising Network (approximating epsilon_theta)
        self.denoiser = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, input_dim, kernel_size=3, padding=1)
        )
        # Time embedding
        self.time_embed = nn.Embedding(timesteps, window_size)

        # Precompute DDPM noise schedule variables
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer('alpha_bar', alpha_bar)

    def forward(self, x, t):
        # Time conditioning injection
        t_emb = self.time_embed(t).unsqueeze(1) # (Batch, 1, Window_Size)
        x_in = x.transpose(1, 2) # Switch to (Batch, Channels, Seq) for Conv1d
        out = self.denoiser(x_in) + t_emb
        return out.transpose(1, 2)

    def training_step(self, batch, batch_idx):
        x0 = batch[0] if isinstance(batch, list) else batch
        batch_size = x0.size(0)
        
        # Sample random time steps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # Generate noise
        noise = torch.randn_like(x0)
        
        # Calculate forward diffusion q(x_t | x_0)
        a_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        xt = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * noise
        
        # Predict the noise
        predicted_noise = self(xt, t)
        
        # Loss: Mean Squared Error between actual noise and predicted noise
        loss = F.mse_loss(noise, predicted_noise)
        self.log('diffusion_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def generate(self, num_samples: int):
        """Reverses the diffusion process to generate synthetic data from pure noise."""
        device = next(self.parameters()).device
        x = torch.randn(num_samples, self.hparams.window_size, self.hparams.input_dim, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            predicted_noise = self(x, t)
            
            # Simplified reverse step (DDIM-style fast sampling abstraction)
            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
            x = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            
        return x



# ==============================================================================
# 5. TimeFlow (Standard Flow Matching Baseline)
# Ref: "TimeFlow: Towards Stochastic-Aware and Efficient Time Series Generation"
# ==============================================================================
from torchdiffeq import odeint

class TimeFlow(pl.LightningModule):
    """
    Standard Flow Matching for Time Series.
    Learns a vector field mapping from standard Gaussian noise to data.
    Uses a standard Transformer/MLP backbone (NOT Mamba).
    """
    def __init__(self, input_dim: int, window_size: int, hidden_dim: int = 64, euler_steps: int = 100, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.window_size = window_size
        self.euler_steps = euler_steps
        
        # Standard Vector Field Estimator (v_theta)
        # In the original paper, this is often an MLP or lightweight Transformer.
        # We use a 1D-CNN/MLP hybrid here for baseline efficiency.
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), # +1 for time feature
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, x):
        """
        Predicts the velocity field at time t.
        Input x shape: (Batch, Window, Features)
        """
        # Expand scalar time t to match x dimensions for concatenation
        t_tensor = torch.full((x.shape[0], x.shape[1], 1), t.item(), device=self.device)
        x_with_time = torch.cat([x, t_tensor], dim=-1)
        return self.net(x_with_time)

    def training_step(self, batch, batch_idx):
        x1 = batch[0] if isinstance(batch, list) else batch # Target data distribution
        batch_size = x1.size(0)
        
        # 1. Sample standard Gaussian noise (x0)
        x0 = torch.randn_like(x1)
        
        # 2. Sample random time t in [0, 1]
        t = torch.rand(batch_size, 1, 1, device=self.device)
        
        # 3. Construct the probability flow path (Linear interpolation)
        # x_t = (1 - t) * x0 + t * x1
        xt = (1 - t) * x0 + t * x1
        
        # 4. Target vector field (u_t)
        ut = x1 - x0
        
        # 5. Predict vector field
        # Note: We pass t.mean() as a scalar to the forward pass for simplicity in this baseline
        vt = self.forward(t.mean(), xt)
        
        # 6. Flow Matching Loss
        loss = F.mse_loss(vt, ut)
        self.log('timeflow_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def generate(self, num_samples: int):
        """
        Generates synthetic data by solving the ODE using the Euler method.
        """
        device = next(self.parameters()).device
        
        # Start at standard Gaussian noise
        x0 = torch.randn(num_samples, self.window_size, self.input_dim, device=device)
        
        # Define integration time points [0, 1]
        t_span = torch.linspace(0.0, 1.0, self.euler_steps, device=device)
        
        # Solve the ODE: dx/dt = v_theta(t, x)
        # Note: torchdiffeq's odeint expects a function signature func(t, x)
        print(f"[TimeFlow] Solving ODE with {self.euler_steps} steps...")
        trajectory = odeint(self.forward, x0, t_span, method='euler')
        
        # The final state in the trajectory is the generated data
        generated_data = trajectory[-1]
        return generated_data