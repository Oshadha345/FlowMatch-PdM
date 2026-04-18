# Degradation Manifold constraint penalty

# flowmatchPdM/tccm_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TCCMManifoldLoss(nn.Module):
    """
    Energy-envelope TCCM penalty.

    The raw AC waveform is first mapped to a smooth energy envelope so the loss
    acts on macro degradation structure instead of phase-sensitive oscillations.
    """
    def __init__(self, lambda_weight: float = 0.1, patch_size: int = 32):
        super().__init__()
        self.lambda_weight = float(lambda_weight)
        self.patch_size = int(patch_size)
        self.envelope_stride = max(1, self.patch_size // 2)
        self.envelope_padding = max(0, self.patch_size // 4)

    def forward(self, predicted_vector_field, current_health_index):
        """
        Args:
            predicted_vector_field: Predicted vector field with shape [B, L, C].
            current_health_index: Normalized RUL condition in [0, 1].
        """
        if self.lambda_weight <= 0.0:
            return predicted_vector_field.new_zeros(())

        if predicted_vector_field.dim() != 3:
            raise ValueError(
                "TCCM expects vector fields with shape [batch, seq, channels], "
                f"received {tuple(predicted_vector_field.shape)}."
            )

        field = predicted_vector_field.transpose(1, 2).contiguous()  # [B, C, L]
        energy = torch.abs(field)
        envelope = F.avg_pool1d(
            energy,
            kernel_size=self.patch_size,
            stride=self.envelope_stride,
            padding=self.envelope_padding,
        )

        diff = envelope[:, :, 1:] - envelope[:, :, :-1]
        penalty = torch.relu(-diff)

        if current_health_index is not None:
            health = current_health_index.float().reshape(current_health_index.shape[0], -1).mean(dim=-1)
            health_weight = 1.0 + (1.0 - health)
            penalty = penalty.mean(dim=(1, 2)) * health_weight
            return self.lambda_weight * penalty.mean()

        return self.lambda_weight * penalty.mean()
