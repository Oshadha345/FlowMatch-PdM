# The main LightningModule for FlowMatch-PdM

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchdiffeq import odeint
import functools

from flowmatchPdM.model.mamba_backbone import BidirectionalMambaBlock
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
        self.euler_steps = config.get('euler_steps', 100)
        
        # 1. Physics-Informed Prior
        # Assumes condition_dim is 1 for this example (e.g., motor load or HI)
        self.harmonic_prior = DynamicHarmonicPrior(condition_dim=1, window_size=window_size)
        
        # 2. Vector Field Estimator (Mamba Backbone)
        # Project input to Mamba dimension
        self.input_proj = nn.Linear(input_dim + 1, self.d_model) # +1 for time scalar
        
        # Stack multiple Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            BidirectionalMambaBlock(d_model=self.d_model, d_state=config.get('mamba_d_state', 16))
            for _ in range(3) # 3 layers of Mamba
        ])
        
        # Project back to data dimension
        self.output_proj = nn.Linear(self.d_model, input_dim)
        
        # 3. Constraint & Pruning Infrastructure
        self.tccm_loss = TCCMManifoldLoss(lambda_weight=config.get('tccm_lambda', 10.0))
        
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
        # Inject time scalar into features
        t_tensor = torch.full((x.shape[0], x.shape[1], 1), t.item(), device=self.device)
        x_in = torch.cat([x, t_tensor], dim=-1)
        
        hidden = self.input_proj(x_in)
        for block in self.mamba_blocks:
            hidden = block(hidden)
            
        return self.output_proj(hidden)

    def training_step(self, batch, batch_idx):
        # x1: Target real degradation data, cond: e.g., Health Index or Load
        x1, cond = batch
        batch_size = x1.size(0)
        
        # Ensure condition is 2D
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1).float()
            
        # 1. Base Distribution: Dynamic Conditioned Harmonic Prior
        # This replaces standard pure noise!
        x0 = self.harmonic_prior(cond, batch_size, self.device)
        
        # Ensure x0 dimensions match x1 (CWRU is 1D, CMAPSS is 14D)
        if x0.shape[-1] != x1.shape[-1]:
            # Simple padding/projection if harmonic prior is strictly 1D but data is multi-variate
            x0 = x0.repeat(1, 1, x1.shape[-1])
        
        # 2. Sample random time t in [0, 1]
        t = torch.rand(batch_size, 1, 1, device=self.device)
        
        # 3. Construct the probability flow path
        xt = (1 - t) * x0 + t * x1
        ut = x1 - x0 # Target vector field
        
        # 4. Predict vector field via Mamba
        vt = self.forward(t.mean(), xt, cond)
        
        # 5. Calculate Standard Flow Matching Loss
        fm_loss = F.mse_loss(vt, ut)
        
        # 6. Apply TCCM Hyper-Manifold Penalty (Enforce physics constraint)
        # We pass the predicted vector field and the condition (RUL/HI)
        tccm_penalty = self.tccm_loss(vt, cond)
        
        loss = fm_loss + tccm_penalty
        
        self.log('train_fm_loss', fm_loss, prog_bar=True)
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
        # Start at our physics-informed harmonic prior, NOT pure noise
        x0 = self.harmonic_prior(conditions, num_samples, self.device)
        if x0.shape[-1] != self.input_dim:
            x0 = x0.repeat(1, 1, self.input_dim)
            
        t_span = torch.linspace(0.0, 1.0, self.euler_steps, device=self.device)
        
        # Solve the ODE using standard Euler method
        print(f"[FlowMatch-PdM] Solving Mamba ODE over {self.euler_steps} steps...")
        trajectory = odeint(lambda t, x: self.forward(t, x, conditions), x0, t_span, method='euler')
        
        return trajectory[-1]
    
