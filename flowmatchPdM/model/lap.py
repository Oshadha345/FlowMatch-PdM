# flowmatchPdM/LAP.py
import torch
import pytorch_lightning as pl
from typing import Dict

class LayerAdaptivePruningCallback(pl.Callback):
    """
    LAP: Monitors Mamba block activations and dynamically prunes 
    underutilized channels once the degradation manifold reaches a stable phase.
    """
    def __init__(self, alpha: float = 0.2, beta: float = 0.1, stability_threshold: float = 0.05):
        super().__init__()
        self.alpha = alpha  
        self.beta = beta    
        self.stability_threshold = stability_threshold
        
        self.is_stable = False
        self.previous_activations: Dict[str, torch.Tensor] = {}
        self.pruning_masks: Dict[str, torch.Tensor] = {}

    def on_train_epoch_end(self, trainer, pl_module):
        if not hasattr(pl_module, 'mamba_activations'):
            return

        current_activations = pl_module.mamba_activations 
        
        if not self.is_stable:
            self._check_stability(current_activations, trainer.current_epoch)
        
        if self.is_stable:
            self._apply_lap(current_activations, pl_module)
            
        pl_module.reset_mamba_activations()

    def _check_stability(self, current_activations, epoch):
        if not self.previous_activations:
            self.previous_activations = {k: v.clone() for k, v in current_activations.items()}
            return

        max_variance = 0.0
        for layer_name, current_acts in current_activations.items():
            prev_acts = self.previous_activations[layer_name]
            variance = torch.max(torch.abs(current_acts - prev_acts) / (prev_acts + 1e-8)).item()
            max_variance = max(max_variance, variance)

        if max_variance < self.stability_threshold:
            self.is_stable = True
            print(f"\n[LAP Logger] >>> STABLE PHASE ACHIEVED at Epoch {epoch} <<<")
        else:
            self.previous_activations = {k: v.clone() for k, v in current_activations.items()}

    def _apply_lap(self, current_activations, pl_module):
        total_channels = 0
        pruned_channels = 0

        for layer_name, acts in current_activations.items():
            num_channels = acts.numel()
            total_channels += num_channels
            
            sorted_acts, sorted_indices = torch.sort(acts)
            total_load = acts.sum()
            average_load = total_load / num_channels
            
            cumulative_acts = torch.cumsum(sorted_acts, dim=0)
            beta_threshold_idx = torch.searchsorted(cumulative_acts, self.beta * total_load).item()
            candidate_indices = sorted_indices[:beta_threshold_idx]
            
            if len(candidate_indices) == 0:
                continue

            candidate_acts = acts[candidate_indices]
            alpha_mask = candidate_acts <= (self.alpha * average_load)
            final_prune_indices = candidate_indices[alpha_mask]
            
            num_pruned = len(final_prune_indices)
            pruned_channels += num_pruned

            if layer_name not in self.pruning_masks:
                self.pruning_masks[layer_name] = torch.ones(num_channels, device=acts.device)
            
            self.pruning_masks[layer_name][final_prune_indices] = 0.0
            pl_module.mamba_pruning_masks[layer_name] = self.pruning_masks[layer_name]
            
            if num_pruned > 0:
                print(f"[LAP Logger] {layer_name}: Pruned {num_pruned}/{num_channels} channels.")