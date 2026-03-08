# Bidirectional SSM

# flowmatchPdM/mamba_backbone.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional State-Space Model for FlowMatch-PdM.
    Processes sequences forward and backward to capture degradation context.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # Forward SSM
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Backward SSM
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Fusion gate
        self.fusion_linear = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Input: (Batch, Sequence_Length, d_model)
        Returns: Fused bidirectional context (Batch, Sequence_Length, d_model)
        """
        # Forward pass
        out_fwd = self.mamba_fwd(x)
        
        # Backward pass (flip sequence dimension)
        x_flipped = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_flipped)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        # Concatenate and fuse
        fused = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.fusion_linear(fused)
        
        # Residual connection and normalization
        return self.norm(x + out)