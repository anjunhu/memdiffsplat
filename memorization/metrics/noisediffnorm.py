import torch
import numpy as np
from typing import Dict
from .base import BaseMetric

class NoiseDiffNormMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "Noise_Difference_Norm"
        
    @property
    def metric_type(self) -> str:
        return "per_seed"
        
    @property
    def requires_intermediates(self) -> bool:
        return True

    def measure(self, intermediates: Dict = None, **kwargs) -> Dict:
        """Calculates the noise difference norm for a single generation."""
        if intermediates is None:
            raise ValueError("NoiseDiffNormMetric requires intermediates dict")
            
        uncond_noise, text_noise = intermediates['uncond_noise'], intermediates['text_noise']
        
        # This is s_delta in the paper's notation
        noise_diffs = [(tn - un) for un, tn in zip(uncond_noise, text_noise)]
        
        # The metric is the L2 norm of the score difference, averaged over steps
        norm_traj = [d.norm(p=2).item() for d in noise_diffs]
        norm_mean = np.mean(norm_traj)
        
        return {
            "noise_diff_norm_mean": norm_mean,
            "noise_diff_norm_traj": norm_traj
        }