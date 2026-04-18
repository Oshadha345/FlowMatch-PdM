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


class PatchMambaVectorField(nn.Module):
    """
    Patch-wise bidirectional Mamba vector field for long-horizon vibration generation.

    The raw sequence is compressed into latent patch tokens, processed in latent space,
    and then expanded back to the original temporal resolution. This reduces Euler-step
    stiffness on 2048/2560-length AC vibration windows.
    """

    def __init__(
        self,
        raw_seq_len: int,
        patch_size: int = 32,
        channels: int = 1,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 3,
    ):
        super().__init__()
        if raw_seq_len % patch_size != 0:
            raise ValueError(
                f"patch_size={patch_size} must evenly divide raw_seq_len={raw_seq_len}."
            )

        self.raw_seq_len = int(raw_seq_len)
        self.patch_size = int(patch_size)
        self.num_patches = self.raw_seq_len // self.patch_size
        self.channels = int(channels)

        self.patch_embedding = nn.Linear(self.patch_size * self.channels, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.mamba_blocks = nn.ModuleList(
            [
                BidirectionalMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.head = nn.Linear(d_model, self.patch_size * self.channels)

    def _prepare_time(self, t, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=dtype)
        else:
            t = t.to(device=device, dtype=dtype)

        if t.dim() == 0:
            t = t.expand(batch_size).unsqueeze(-1)
        elif t.dim() == 1:
            if t.numel() == 1:
                t = t.expand(batch_size).unsqueeze(-1)
            else:
                t = t.reshape(batch_size, 1)
        else:
            t = t.reshape(batch_size, -1)
            if t.shape[-1] != 1:
                t = t[:, :1]
        return t

    def forward(self, x_t: torch.Tensor, t, cond=None) -> torch.Tensor:
        del cond
        if x_t.dim() != 3:
            raise ValueError(f"Expected [batch, seq, channels] input, received shape {tuple(x_t.shape)}.")

        batch_size, seq_len, channels = x_t.shape
        if seq_len != self.raw_seq_len:
            raise ValueError(f"Expected seq_len={self.raw_seq_len}, received {seq_len}.")
        if channels != self.channels:
            raise ValueError(f"Expected channels={self.channels}, received {channels}.")

        x_patched = x_t.contiguous().view(batch_size, self.num_patches, self.patch_size * channels)
        latent_x = self.patch_embedding(x_patched)

        t_emb = self.time_mlp(self._prepare_time(t, batch_size, x_t.device, x_t.dtype)).unsqueeze(1)
        latent_x = latent_x + t_emb

        for block in self.mamba_blocks:
            latent_x = block(latent_x)

        v_patched = self.head(latent_x)
        return v_patched.contiguous().view(batch_size, seq_len, channels)
