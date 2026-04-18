# The main LightningModule for FlowMatch-PdM

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import functools

from flowmatchPdM.model.mamba_backbone import PatchMambaVectorField
from flowmatchPdM.model.harmonic_prior import DynamicHarmonicPrior
from flowmatchPdM.model.tccm_loss import TCCMManifoldLoss

class FlowMatchPdM(pl.LightningModule):
    """
    FlowMatch-PdM for Predictive Maintenance.
    Integrates Bidirectional Mamba, Dynamic Harmonic Priors, TCCM, and LAP.
    """
    def __init__(self, input_dim: int, window_size: int, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.window_size = window_size
        self.d_model = config.get('mamba_d_model', 128)
        self.d_state = config.get('mamba_d_state', 16)
        self.euler_steps = config.get('euler_steps', 100)
        self.patch_size = int(config.get('patch_size', 32))
        self.field_clamp = float(config.get('field_clamp', 10.0))
        self.spectral_lambda = float(config.get('spectral_lambda', 0.1))
        self.use_harmonic_prior = bool(config.get('use_harmonic_prior', True))
        self.use_tccm = bool(config.get('use_tccm', True))

        # 1. Physics-Informed Prior
        # Assumes condition_dim is 1 for this example (e.g., motor load or HI)
        self.harmonic_prior = DynamicHarmonicPrior(condition_dim=1, window_size=window_size)

        # 2. Patch-Flow Mamba Vector Field Estimator
        self.vector_field = PatchMambaVectorField(
            raw_seq_len=window_size,
            patch_size=self.patch_size,
            channels=input_dim,
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=int(config.get("d_conv", 4)),
            expand=int(config.get("expand", 2)),
            num_layers=int(config.get("num_mamba_layers", 3)),
        )
        self.mamba_blocks = self.vector_field.mamba_blocks

        # 3. Constraint & Pruning Infrastructure
        self.tccm_loss = TCCMManifoldLoss(
            lambda_weight=config.get('tccm_lambda', 0.1),
            patch_size=self.patch_size,
        )
        
        # LAP Tracking Dictionaries
        self.mamba_activations = {}
        self.mamba_pruning_masks = {}
        
        # Register the invisible hooks on our Mamba layers
        self._register_lap_hooks()

    def _register_lap_hooks(self):
        """Attaches the invisible LAP sensor to every Mamba block."""
        for i, block in enumerate(self.mamba_blocks):
            layer_name = f"mamba_block_{i}"
            # functools.partial allows us to pass the layer_name into the standard PyTorch hook signature
            hook_fn = functools.partial(self._lap_hook, layer_name)
            block.register_forward_hook(hook_fn)
            print(f"[FlowMatch-PdM] Registered LAP Forward Hook on {layer_name}")

    def _lap_hook(self, layer_name: str, module: nn.Module, inp: tuple, out: torch.Tensor):
        """
        The invisible sensor:
        1. Accumulates the L1 Norm of the activations.
        2. Applies the hard pruning mask if the stable phase has triggered.
        """
        # 1. Accumulate L1 norm across Batch and Sequence dims, leaving per-channel totals
        # out shape: (Batch, Seq, d_model)
        l1_norm = out.abs().sum(dim=(0, 1)).detach()
        
        if layer_name not in self.mamba_activations:
            self.mamba_activations[layer_name] = l1_norm
        else:
            self.mamba_activations[layer_name] += l1_norm

        # 2. Apply Mask (Dynamic Pruning)
        if layer_name in self.mamba_pruning_masks:
            mask = self.mamba_pruning_masks[layer_name]
            # Reshape mask to broadcast across Batch and Sequence dimensions
            out = out * mask.view(1, 1, -1) 

        return out

    def reset_mamba_activations(self):
        """Called by LAP.py at the end of each epoch to reset counters."""
        self.mamba_activations = {}

    def forward(self, t, x, conditions=None):
        """
        Predicts the velocity field v_theta(t, x).
        """
        return self.vector_field(x, t, conditions)

    def _sample_base_distribution(
        self,
        conditions: torch.Tensor,
        batch_size: int,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_harmonic_prior:
            if reference is not None:
                return torch.randn_like(reference)
            return torch.randn(batch_size, self.window_size, self.input_dim, device=self.device)

        x0 = self.harmonic_prior(conditions, batch_size, self.device)
        if x0.shape[-1] != self.input_dim:
            x0 = x0.repeat(1, 1, self.input_dim)
        if reference is not None:
            x0 = x0.to(dtype=reference.dtype)
        return x0

    def training_step(self, batch, batch_idx):
        # x1: Target real degradation data, cond: e.g., Health Index or Load
        x1, cond = batch
        batch_size = x1.size(0)
        
        # Ensure condition is 2D
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1).float()
        else:
            cond = cond.float()
            
        # 1. Base Distribution: Dynamic Conditioned Harmonic Prior
        x0 = self._sample_base_distribution(cond, batch_size, reference=x1)
        
        # 2. Sample random time t in [0, 1]
        t = torch.rand(batch_size, 1, 1, device=self.device)
        
        # 3. Construct the probability flow path
        xt = (1 - t) * x0 + t * x1
        ut = x1 - x0 # Target vector field
        
        # 4. Predict vector field via Mamba
        vt = self.forward(t.view(batch_size, 1), xt, cond)
        
        # 5. Calculate Standard Flow Matching Loss
        fm_loss = F.mse_loss(vt, ut)
        fft_pred = torch.fft.rfft(vt, dim=1)
        fft_target = torch.fft.rfft(ut, dim=1)
        spectral_loss = F.l1_loss(torch.abs(fft_pred), torch.abs(fft_target))

        # 6. Apply Energy-Envelope TCCM Penalty
        if self.use_tccm:
            tccm_penalty = self.tccm_loss(vt, cond)
        else:
            tccm_penalty = vt.new_zeros(())

        loss = fm_loss + (self.spectral_lambda * spectral_loss) + tccm_penalty
        
        self.log('train_fm_loss', fm_loss, prog_bar=True)
        self.log('train_spectral_loss', spectral_loss, prog_bar=False)
        self.log('train_tccm_penalty', tccm_penalty, prog_bar=True)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.config.get('lr', 1e-3)))

    @torch.no_grad()
    def generate(self, conditions, num_samples: int):
        """
        Generates synthetic degradation data by solving the ODE.
        """
        if conditions.dim() == 1:
            conditions = conditions.unsqueeze(-1)
        conditions = conditions.to(self.device).float()

        x_t = self._sample_base_distribution(conditions, num_samples)
        dt = 1.0 / max(self.euler_steps, 1)

        print(f"[FlowMatch-PdM] Solving patched Mamba Euler flow over {self.euler_steps} steps...")
        for step in range(self.euler_steps):
            t_value = step / max(self.euler_steps - 1, 1)
            t_tensor = torch.full((num_samples, 1), t_value, device=self.device, dtype=x_t.dtype)
            v_t = self.forward(t_tensor, x_t, conditions)
            v_t = torch.clamp(v_t, min=-self.field_clamp, max=self.field_clamp)
            x_t = x_t + v_t * dt

        return x_t
    
