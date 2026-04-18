import math
from typing import Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_sequence(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0].float()
    return batch.float()


def _group_norm(num_channels: int) -> nn.GroupNorm:
    for num_groups in (8, 4, 2, 1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


def _broadcast_time_scalar(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if t.dim() == 0:
        t = t.unsqueeze(0)
    if t.dim() == 1:
        t = t[:, None]
    return t.to(dtype=x.dtype, device=x.device)


class ClassicalAugmenter:
    @staticmethod
    def apply_smote(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5):
        from imblearn.over_sampling import SMOTE

        batch_size, window_size, features = X.shape
        X_flat = X.reshape(batch_size, -1)
        X_resampled, y_resampled = SMOTE(k_neighbors=k_neighbors, random_state=42).fit_resample(X_flat, y)
        return X_resampled.reshape(-1, window_size, features), y_resampled

    @staticmethod
    def apply_smote_regression(
        X: np.ndarray,
        y: np.ndarray,
        k_neighbors: int = 5,
        n_samples: int | None = None,
        random_state: int = 42,
    ):
        """
        Generate SMOTE-style interpolated minority windows for regression targets.

        This is a lightweight SMOTER-inspired variant that interpolates between
        nearest-neighbour low-RUL windows and linearly blends their targets.
        """

        from sklearn.neighbors import NearestNeighbors

        batch_size, window_size, features = X.shape
        if batch_size < 2:
            raise ValueError("Regression SMOTE requires at least two source samples.")

        X_flat = X.reshape(batch_size, -1).astype(np.float32)
        y_array = np.asarray(y, dtype=np.float32)
        y_flat = y_array.reshape(batch_size, -1)

        resolved_neighbors = max(1, min(int(k_neighbors), batch_size - 1))
        nn = NearestNeighbors(n_neighbors=resolved_neighbors + 1)
        nn.fit(X_flat)
        neighbor_indices = nn.kneighbors(return_distance=False)

        rng = np.random.default_rng(random_state)
        synth_count = int(n_samples or batch_size)
        synth_x = np.empty((synth_count, X_flat.shape[1]), dtype=np.float32)
        synth_y = np.empty((synth_count, y_flat.shape[1]), dtype=np.float32)

        for idx in range(synth_count):
            anchor_idx = int(rng.integers(0, batch_size))
            candidates = neighbor_indices[anchor_idx, 1:]
            if candidates.size == 0:
                neighbor_idx = anchor_idx
            else:
                neighbor_idx = int(rng.choice(candidates))

            mix = float(rng.random())
            synth_x[idx] = X_flat[anchor_idx] + mix * (X_flat[neighbor_idx] - X_flat[anchor_idx])
            synth_y[idx] = y_flat[anchor_idx] + mix * (y_flat[neighbor_idx] - y_flat[anchor_idx])

        synthetic_x = synth_x.reshape(synth_count, window_size, features)
        synthetic_y = synth_y.reshape(synth_count, *y_array.shape[1:])
        return synthetic_x, synthetic_y.astype(y_array.dtype, copy=False)

    @staticmethod
    def apply_jittering(X: np.ndarray, sigma: float = 0.05):
        return X + np.random.normal(loc=0.0, scale=sigma, size=X.shape)


class TimeVAE(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        beta_kl: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.output_head = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.encoder(x)
        h = hidden[-1]
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        repeated = self.decoder_input(z).unsqueeze(1).repeat(1, self.hparams.window_size, 1)
        decoded, _ = self.decoder(repeated)
        return self.output_head(decoded)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def training_step(self, batch, batch_idx):
        x = _extract_sequence(batch)
        reconstruction, mu, logvar = self(x)
        recon_loss = F.mse_loss(reconstruction, x)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.hparams.beta_kl * kld
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("train_kld", kld, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

    @torch.no_grad()
    def generate(self, num_samples: int, conditions: Optional[torch.Tensor] = None):
        del conditions
        z = torch.randn(num_samples, self.hparams.latent_dim, device=self.device)
        return self.decode(z)


class _SequenceGenerator(nn.Module):
    def __init__(self, noise_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(noise_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(noise)
        return self.proj(hidden)


class _SequenceDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(x)
        return self.head(hidden[:, -1])


class TimeGAN(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        hidden_dim: int = 96,
        noise_dim: int = 32,
        lr: float = 2e-4,
        feature_matching_weight: float = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = _SequenceGenerator(noise_dim, hidden_dim, input_dim)
        self.discriminator = _SequenceDiscriminator(input_dim, hidden_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.hparams.window_size, self.hparams.noise_dim, device=self.device)

    def _feature_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.mean(dim=(0, 1)), x.std(dim=(0, 1))

    def training_step(self, batch, batch_idx):
        real = _extract_sequence(batch)
        batch_size = real.size(0)
        opt_g, opt_d = self.optimizers()

        real_labels = torch.full((batch_size, 1), 0.9, device=self.device)
        fake_labels = torch.zeros((batch_size, 1), device=self.device)

        noise = self._sample_noise(batch_size)
        fake = self.generator(noise)

        opt_d.zero_grad()
        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake.detach())
        d_loss = 0.5 * (
            self.loss_fn(d_real, real_labels) + self.loss_fn(d_fake, fake_labels)
        )
        self.manual_backward(d_loss)
        opt_d.step()

        opt_g.zero_grad()
        noise = self._sample_noise(batch_size)
        fake = self.generator(noise)
        adv_loss = self.loss_fn(self.discriminator(fake), torch.ones_like(real_labels))
        real_mean, real_std = self._feature_stats(real)
        fake_mean, fake_std = self._feature_stats(fake)
        moment_loss = F.mse_loss(fake_mean, real_mean) + F.mse_loss(fake_std, real_std)
        g_loss = adv_loss + self.hparams.feature_matching_weight * moment_loss
        self.manual_backward(g_loss)
        opt_g.step()

        total_loss = g_loss + d_loss
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("g_loss", g_loss, on_step=False, on_epoch=True)
        self.log("d_loss", d_loss, on_step=False, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9))
        return [opt_g, opt_d], []

    @torch.no_grad()
    def generate(self, num_samples: int, conditions: Optional[torch.Tensor] = None):
        del conditions
        return self.generator(self._sample_noise(num_samples))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=t.device).float()
        exponent = exponent / max(half_dim - 1, 1)
        freqs = torch.exp(exponent)
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock1D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = _group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x).transpose(1, 2)
        x, _ = self.attn(x, x, x, need_weights=False)
        return residual + x.transpose(1, 2)


class AdapterBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.gelu(self.pointwise(self.depthwise(x)))
        return x + self.gate(delta) * delta


class UNet1DDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        base_channels: int = 64,
        time_dim: int = 256,
        num_heads: int = 4,
        use_adapters: bool = False,
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock1D(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResidualBlock1D(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.down3 = ResidualBlock1D(base_channels * 4, base_channels * 4, time_dim)

        self.mid1 = ResidualBlock1D(base_channels * 4, base_channels * 4, time_dim)
        self.mid_attn = AttentionBlock1D(base_channels * 4, num_heads=num_heads)
        self.mid2 = ResidualBlock1D(base_channels * 4, base_channels * 4, time_dim)

        self.upsample2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock1D(base_channels * 4, base_channels * 2, time_dim)
        self.upsample1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock1D(base_channels * 2, base_channels, time_dim)
        self.out_norm = _group_norm(base_channels)
        self.out_proj = nn.Conv1d(base_channels, input_dim, kernel_size=3, padding=1)

        self.adapter1 = AdapterBlock1D(base_channels) if use_adapters else nn.Identity()
        self.adapter2 = AdapterBlock1D(base_channels * 2) if use_adapters else nn.Identity()
        self.adapter3 = AdapterBlock1D(base_channels * 4) if use_adapters else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(t)
        x = x.transpose(1, 2)

        x0 = self.input_proj(x)
        d1 = self.adapter1(self.down1(x0, time_emb))
        d2 = self.adapter2(self.down2(self.downsample1(d1), time_emb))
        d3 = self.adapter3(self.down3(self.downsample2(d2), time_emb))

        mid = self.mid2(self.mid_attn(self.mid1(d3, time_emb)), time_emb)

        u2 = self.upsample2(mid)
        # Pad/trim to match skip connection when window_size is not divisible by 4
        if u2.shape[2] != d2.shape[2]:
            u2 = F.pad(u2, (0, d2.shape[2] - u2.shape[2]))
        u2 = self.up2(torch.cat([u2, d2], dim=1), time_emb)
        u1 = self.upsample1(u2)
        if u1.shape[2] != d1.shape[2]:
            u1 = F.pad(u1, (0, d1.shape[2] - u1.shape[2]))
        u1 = self.up1(torch.cat([u1, d1], dim=1), time_emb)

        out = self.out_proj(F.silu(self.out_norm(u1)))
        return out.transpose(1, 2)


class _BaseDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        timesteps: int = 200,
        lr: float = 2e-4,
        base_channels: int = 64,
        num_heads: int = 4,
        use_adapters: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.network = UNet1DDenoiser(
            input_dim=input_dim,
            base_channels=base_channels,
            num_heads=num_heads,
            use_adapters=use_adapters,
        )

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(x_t, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return (x_t - sqrt_one_minus * pred_noise) / torch.clamp(sqrt_alpha, min=1e-6)

    def training_step(self, batch, batch_idx):
        x0 = _extract_sequence(batch)
        batch_size = x0.size(0)
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self(x_t, t)
        loss = F.mse_loss(pred_noise, noise)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

    @torch.no_grad()
    def generate(self, num_samples: int, conditions: Optional[torch.Tensor] = None):
        del conditions
        x = torch.randn(
            num_samples,
            self.hparams.window_size,
            self.hparams.input_dim,
            device=self.device,
        )

        for step in reversed(range(self.hparams.timesteps)):
            t = torch.full((num_samples,), step, device=self.device, dtype=torch.long)
            pred_noise = self(x, t)
            alpha_t = self.alphas[t].view(-1, 1, 1)
            alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1)
            mean = (x - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_t)
            if step > 0:
                noise = torch.randn_like(x)
                variance = torch.sqrt(torch.clamp(self.posterior_variance[t].view(-1, 1, 1), min=1e-8))
                x = mean + variance * noise
            else:
                x = mean
        return x


class DiffusionTS(_BaseDiffusionModel):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        timesteps: int = 200,
        lr: float = 2e-4,
        base_channels: int = 64,
        num_heads: int = 4,
    ):
        super().__init__(
            input_dim=input_dim,
            window_size=window_size,
            timesteps=timesteps,
            lr=lr,
            base_channels=base_channels,
            num_heads=num_heads,
            use_adapters=False,
        )


class FaultDiffusion(_BaseDiffusionModel):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        timesteps: int = 200,
        lr: float = 2e-4,
        base_channels: int = 96,
        num_heads: int = 4,
        diversity_weight: float = 0.05,
    ):
        super().__init__(
            input_dim=input_dim,
            window_size=window_size,
            timesteps=timesteps,
            lr=lr,
            base_channels=base_channels,
            num_heads=num_heads,
            use_adapters=True,
        )
        self.diversity_weight = diversity_weight

    def training_step(self, batch, batch_idx):
        x0 = _extract_sequence(batch)
        batch_size = x0.size(0)

        t1 = torch.randint(0, self.hparams.timesteps, (batch_size,), device=self.device)
        t2 = torch.randint(0, self.hparams.timesteps, (batch_size,), device=self.device)
        noise1 = torch.randn_like(x0)
        noise2 = torch.randn_like(x0)
        x_t1 = self.q_sample(x0, t1, noise1)
        x_t2 = self.q_sample(x0, t2, noise2)
        pred1 = self(x_t1, t1)
        pred2 = self(x_t2, t2)

        denoise_loss = 0.5 * (F.mse_loss(pred1, noise1) + F.mse_loss(pred2, noise2))
        x0_hat1 = self.predict_x0(x_t1, t1, pred1)
        x0_hat2 = self.predict_x0(x_t2, t2, pred2)

        diversity_real = torch.pdist(x0.flatten(1), p=2)
        diversity_fake = torch.pdist(0.5 * (x0_hat1 + x0_hat2).flatten(1), p=2)
        if diversity_fake.numel() == 0:
            diversity_bonus = torch.tensor(0.0, device=self.device)
        else:
            denom = diversity_real.mean().detach() if diversity_real.numel() else torch.tensor(1.0, device=self.device)
            diversity_bonus = diversity_fake.mean() / torch.clamp(denom, min=1e-6)

        loss = denoise_loss - self.diversity_weight * torch.clamp(diversity_bonus, min=0.0, max=5.0)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_denoise_loss", denoise_loss, on_step=False, on_epoch=True)
        self.log("train_diversity_bonus", diversity_bonus, on_step=False, on_epoch=True)
        return loss


class TimeFlow(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        hidden_dim: int = 128,
        euler_steps: int = 100,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = _broadcast_time_scalar(t, x)
        if t.size(0) == 1 and x.size(0) > 1:
            t = t.repeat(x.size(0), 1)
        time_feature = t.unsqueeze(1).expand(-1, x.size(1), -1)
        return self.net(torch.cat([x, time_feature], dim=-1))

    def training_step(self, batch, batch_idx):
        x1 = _extract_sequence(batch)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.size(0), device=self.device)
        t_view = t.view(-1, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0
        pred_velocity = self(t, x_t)
        loss = F.mse_loss(pred_velocity, target_velocity)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

    @torch.no_grad()
    def generate(self, num_samples: int, conditions: Optional[torch.Tensor] = None):
        del conditions
        dt = 1.0 / max(self.hparams.euler_steps, 1)
        x = torch.randn(
            num_samples,
            self.hparams.window_size,
            self.hparams.input_dim,
            device=self.device,
        )
        for step in range(self.hparams.euler_steps):
            t = torch.full((num_samples,), step * dt, device=self.device)
            x = x + dt * self(t, x)
        return x


class _CausalCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, projection_dim: int):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.proj = nn.Linear(hidden_dim, projection_dim)
        self.martingale_head = nn.Linear(hidden_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.rnn(x)
        return self.proj(hidden), self.martingale_head(hidden)


class COTGAN(pl.LightningModule):
    """
    Causal OT GAN with mixed Sinkhorn divergence and martingale regularization.
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        hidden_dim: int = 128,
        noise_dim: int = 32,
        lr: float = 2e-4,
        sinkhorn_eps: float = 0.1,
        sinkhorn_iters: int = 30,
        martingale_weight: float = 10.0,
        causal_weight: float = 1.0,
        critic_projection_dim: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = _SequenceGenerator(noise_dim, hidden_dim, input_dim)
        self.critic = _CausalCritic(input_dim, hidden_dim, critic_projection_dim)

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.hparams.window_size, self.hparams.noise_dim, device=self.device)

    def _causal_cost(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        hx, mx = self.critic(x)
        hy, my = self.critic(y)

        pairwise_data_cost = ((x[:, None] - y[None, :]) ** 2).mean(dim=(2, 3))
        pairwise_feature_cost = ((hx[:, None] - hy[None, :]) ** 2).mean(dim=(2, 3))

        delta_mx = mx[:, 1:] - mx[:, :-1]
        hy_prefix = hy[:, :-1]
        causal_term = torch.einsum("btk,stk->bs", delta_mx, hy_prefix)
        causal_term = causal_term / max(delta_mx.size(1) * delta_mx.size(2), 1)

        return pairwise_data_cost + pairwise_feature_cost + self.hparams.causal_weight * causal_term

    def _sinkhorn(self, cost: torch.Tensor) -> torch.Tensor:
        n, m = cost.shape
        mu = torch.full((n,), 1.0 / n, device=cost.device, dtype=cost.dtype)
        nu = torch.full((m,), 1.0 / m, device=cost.device, dtype=cost.dtype)
        kernel = torch.exp(-cost / self.hparams.sinkhorn_eps).clamp_min(1e-9)
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        for _ in range(self.hparams.sinkhorn_iters):
            u = mu / torch.clamp(kernel @ v, min=1e-9)
            v = nu / torch.clamp(kernel.t() @ u, min=1e-9)

        transport = u[:, None] * kernel * v[None, :]
        return torch.sum(transport * cost)

    def _sinkhorn_divergence(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cost_xy = self._causal_cost(x, y)
        cost_xx = self._causal_cost(x, x)
        cost_yy = self._causal_cost(y, y)
        return self._sinkhorn(cost_xy) - 0.5 * self._sinkhorn(cost_xx) - 0.5 * self._sinkhorn(cost_yy)

    def _martingale_penalty(self, x: torch.Tensor) -> torch.Tensor:
        _, martingale = self.critic(x)
        increments = martingale[:, 1:] - martingale[:, :-1]
        centered_mean = increments.mean(dim=0)
        variance = martingale.var(dim=(0, 1), unbiased=False)
        return (centered_mean.abs() / torch.sqrt(torch.clamp(variance, min=1e-6))).mean()

    def training_step(self, batch, batch_idx):
        real = _extract_sequence(batch)
        batch_size = real.size(0)
        opt_g, opt_c = self.optimizers()

        opt_c.zero_grad()
        fake = self.generator(self._sample_noise(batch_size)).detach()
        critic_objective = self._sinkhorn_divergence(real, fake)
        martingale_penalty = 0.5 * (self._martingale_penalty(real) + self._martingale_penalty(fake))
        critic_loss = -critic_objective + self.hparams.martingale_weight * martingale_penalty
        self.manual_backward(critic_loss)
        opt_c.step()

        opt_g.zero_grad()
        fake = self.generator(self._sample_noise(batch_size))
        generator_loss = self._sinkhorn_divergence(real, fake)
        self.manual_backward(generator_loss)
        opt_g.step()

        total_loss = critic_loss + generator_loss
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cotgan_generator_loss", generator_loss, on_step=False, on_epoch=True)
        self.log("cotgan_critic_loss", critic_loss, on_step=False, on_epoch=True)
        self.log("cotgan_martingale_penalty", martingale_penalty, on_step=False, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9))
        opt_c = torch.optim.AdamW(self.critic.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9))
        return [opt_g, opt_c], []

    @torch.no_grad()
    def generate(self, num_samples: int, conditions: Optional[torch.Tensor] = None):
        del conditions
        return self.generator(self._sample_noise(num_samples))
