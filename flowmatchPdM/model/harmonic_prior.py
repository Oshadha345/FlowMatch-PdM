# Physics-informed noise initialization

# flowmatchPdM/harmonic_prior.py
import torch
import torch.nn as nn
import math

class DynamicHarmonicPrior(nn.Module):
    """
    Physics-Informed Prior for FlowMatch-PdM.
    Replaces standard N(0,I) noise with a stochastic harmonic oscillator
    conditioned on machine operating parameters.
    """
    def __init__(self, condition_dim: int, window_size: int, hidden_dim: int = 32):
        super().__init__()
        self.window_size = window_size
        
        # Neural network to predict frequency (f) and amplitude (A) from conditions
        self.param_estimator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2) # Outputs [Amplitude, Frequency]
        )

    def forward(self, conditions, batch_size: int, device: torch.device):
        """
        Generates the base noise distribution p_0(x).
        Args:
            conditions: Tensor of operating conditions (e.g., motor load) shape (Batch, Cond_Dim)
        """
        # 1. Estimate physics parameters
        # params shape: (Batch, 2)
        params = self.param_estimator(conditions)
        
        # Softplus ensures Amplitude and Frequency are strictly positive
        A = torch.nn.functional.softplus(params[:, 0]).unsqueeze(1) # (Batch, 1)
        f = torch.nn.functional.softplus(params[:, 1]).unsqueeze(1) # (Batch, 1)
        
        # 2. Create time vector t
        t = torch.linspace(0, 1, self.window_size, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        # 3. Generate Harmonic Base Signal: A * sin(2*pi*f*t + phase)
        phase = torch.rand(batch_size, 1, device=device) * 2 * math.pi
        harmonic_signal = A * torch.sin(2 * math.pi * f * t + phase)
        
        # 4. Add small stochastic noise epsilon ~ N(0, 0.1)
        epsilon = torch.randn_like(harmonic_signal) * 0.1
        
        # Output shape: (Batch, Window_Size, 1 channel)
        return (harmonic_signal + epsilon).unsqueeze(-1)