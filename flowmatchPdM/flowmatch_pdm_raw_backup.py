# Backup copy of the pre-patch-flow FlowMatch-PdM generator.
#
# This file preserves the original raw-sequence ODE implementation that was active
# immediately before the Patch-Flow Mamba architectural overhaul. It is kept as a
# recovery/reference artifact and is intentionally isolated from the active import path.

import functools

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from flowmatchPdM.model.harmonic_prior import DynamicHarmonicPrior
from flowmatchPdM.model.mamba_backbone import BidirectionalMambaBlock


class RawTCCMManifoldLossBackup(nn.Module):
    """
    Original raw-wave TCCM penalty used before the energy-envelope upgrade.
    """

    def __init__(self, lambda_weight: float = 10.0):
        super().__init__()
        self.lambda_weight = float(lambda_weight)

    def forward(self, predicted_vector_field, current_health_index):
        del current_health_index
        implied_health_change = torch.mean(predicted_vector_field, dim=[1, 2])
        manifold_violation = torch.relu(implied_health_change)
        return self.lambda_weight * torch.mean(manifold_violation)


class FlowMatchPdMRawBackup(pl.LightningModule):
    """
    Pre-patch-flow FlowMatch-PdM preserved for reference.

    Architecture:
    - raw-sequence bidirectional Mamba vector field
    - harmonic prior initialization
    - raw-wave TCCM penalty
    - `torchdiffeq.odeint(..., method="euler")` generation path
    """

    def __init__(self, input_dim: int, window_size: int, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.window_size = window_size
        self.d_model = config.get("mamba_d_model", 128)
        self.euler_steps = config.get("euler_steps", 100)
        self.use_harmonic_prior = bool(config.get("use_harmonic_prior", True))
        self.use_tccm = bool(config.get("use_tccm", True))
        self.use_lap = bool(config.get("use_lap", True))

        self.harmonic_prior = DynamicHarmonicPrior(condition_dim=1, window_size=window_size)

        self.input_proj = nn.Linear(input_dim + 1, self.d_model)
        self.mamba_blocks = nn.ModuleList(
            [
                BidirectionalMambaBlock(
                    d_model=self.d_model,
                    d_state=config.get("mamba_d_state", 16),
                )
                for _ in range(3)
            ]
        )
        self.output_proj = nn.Linear(self.d_model, input_dim)
        self.tccm_loss = RawTCCMManifoldLossBackup(lambda_weight=config.get("tccm_lambda", 10.0))

        self.mamba_activations = {}
        self.mamba_pruning_masks = {}
        if self.use_lap:
            self._register_lap_hooks()

    def _register_lap_hooks(self):
        for i, block in enumerate(self.mamba_blocks):
            layer_name = f"mamba_block_{i}"
            hook_fn = functools.partial(self._lap_hook, layer_name)
            block.register_forward_hook(hook_fn)

    def _lap_hook(self, layer_name: str, module: nn.Module, inp: tuple, out: torch.Tensor):
        del module, inp
        l1_norm = out.abs().sum(dim=(0, 1)).detach()

        if layer_name not in self.mamba_activations:
            self.mamba_activations[layer_name] = l1_norm
        else:
            self.mamba_activations[layer_name] += l1_norm

        if layer_name in self.mamba_pruning_masks:
            mask = self.mamba_pruning_masks[layer_name]
            out = out * mask.view(1, 1, -1)

        return out

    def reset_mamba_activations(self):
        self.mamba_activations = {}

    def forward(self, t, x, conditions=None):
        del conditions
        t_tensor = torch.full((x.shape[0], x.shape[1], 1), t.item(), device=self.device)
        x_in = torch.cat([x, t_tensor], dim=-1)

        hidden = self.input_proj(x_in)
        for block in self.mamba_blocks:
            hidden = block(hidden)

        return self.output_proj(hidden)

    def training_step(self, batch, batch_idx):
        del batch_idx
        x1, cond = batch
        batch_size = x1.size(0)

        if cond.dim() == 1:
            cond = cond.unsqueeze(-1).float()

        if self.use_harmonic_prior:
            x0 = self.harmonic_prior(cond, batch_size, self.device)
            if x0.shape[-1] != x1.shape[-1]:
                x0 = x0.repeat(1, 1, x1.shape[-1])
        else:
            x0 = torch.randn_like(x1)

        t = torch.rand(batch_size, 1, 1, device=self.device)
        xt = (1 - t) * x0 + t * x1
        ut = x1 - x0

        vt = self.forward(t.mean(), xt, cond)
        fm_loss = F.mse_loss(vt, ut)
        if self.use_tccm:
            tccm_penalty = self.tccm_loss(vt, cond)
        else:
            tccm_penalty = torch.zeros((), device=self.device, dtype=fm_loss.dtype)
        loss = fm_loss + tccm_penalty

        self.log("train_fm_loss", fm_loss, prog_bar=True)
        self.log("train_tccm_penalty", tccm_penalty, prog_bar=True)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.config.get("lr", 1e-3)))

    @torch.no_grad()
    def generate(self, conditions, num_samples: int):
        if self.use_harmonic_prior and conditions is not None:
            x0 = self.harmonic_prior(conditions, num_samples, self.device)
            if x0.shape[-1] != self.input_dim:
                x0 = x0.repeat(1, 1, self.input_dim)
        else:
            x0 = torch.randn(num_samples, self.window_size, self.input_dim, device=self.device)

        t_span = torch.linspace(0.0, 1.0, self.euler_steps, device=self.device)
        trajectory = odeint(lambda t, x: self.forward(t, x, conditions), x0, t_span, method="euler")
        return trajectory[-1]
