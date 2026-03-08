# Degradation Manifold constraint penalty

# flowmatchPdM/tccm_loss.py
import torch
import torch.nn as nn

class TCCMManifoldLoss(nn.Module):
    """
    Time-Conditioned Contraction Matching Penalty.
    Enforces monotonic degradation (RUL must decrease or stay flat).
    """
    def __init__(self, lambda_weight: float = 10.0):
        super().__init__()
        self.lambda_weight = lambda_weight

    def forward(self, predicted_vector_field, current_health_index):
        """
        Args:
            predicted_vector_field: The output of the FlowMatch ODE.
            current_health_index: The normalized RUL condition (0 to 1).
        """
        # We assume the vector field corresponds to the rate of change of the sensor data.
        # In a full implementation, you map the vector field to the Health Index manifold.
        # Here we mock the projection: if the implied health change (delta_HI) is positive, penalize.
        
        # Mock projection: assume the last dimension of the field correlates to health state
        implied_health_change = torch.mean(predicted_vector_field, dim=[1, 2])
        
        # ReLU acts as the penalty function: max(0, delta_HI)
        # If delta_HI is negative (degrading), penalty is 0. 
        # If delta_HI is positive (healing), penalty spikes.
        manifold_violation = torch.relu(implied_health_change)
        
        loss = self.lambda_weight * torch.mean(manifold_violation)
        return loss