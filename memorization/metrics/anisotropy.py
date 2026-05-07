"""
Anisotropy-based memorization detection metric.
Reference: Asthana & Belagiannis, "Detecting and Mitigating Memorization in
Diffusion Models through Anisotropy of the Log-Probability", ICLR 2026.
https://github.com/rohanasthana/memorization-anisotropy
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from .base import BaseMetric


class AnisotropyMetric(BaseMetric):
    """
    Combines isotropic norm (high-noise) with anisotropic angular alignment
    (low-noise) for memorization detection.  Eq. 14 in Asthana & Belagiannis.

    M(x_T, c) = γ₁ · cos(s_θ(x,t≈0), s_θ^Δ(x,t≈0,c))
              + γ₂ · ‖s_θ^Δ(x,t≈T,c)‖
    """

    def __init__(self, gamma1: float = 1.0, gamma2: float = 1.0):
        super().__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    @property
    def name(self) -> str:
        return "Anisotropy_Score"

    @property
    def metric_type(self) -> str:
        return "per_seed"

    @property
    def requires_intermediates(self) -> bool:
        return True

    def measure(self, intermediates: Dict = None, **kwargs) -> Dict:
        if intermediates is None:
            raise ValueError("AnisotropyMetric requires intermediates dict")

        uncond_noise = intermediates['uncond_noise']
        text_noise = intermediates['text_noise']

        # --- isotropic component: norm at t≈T (first denoising step = high noise) ---
        diff_first = text_noise[0] - uncond_noise[0]
        norm_iso = diff_first.norm(p=2).item()

        # --- anisotropic component: cosine sim at t≈0 (last denoising step = low noise) ---
        uc_last = uncond_noise[-1].flatten()
        diff_last = (text_noise[-1] - uncond_noise[-1]).flatten()
        cos_aniso = F.cosine_similarity(uc_last.unsqueeze(0),
                                        diff_last.unsqueeze(0)).item()

        score = self.gamma1 * cos_aniso + self.gamma2 * norm_iso

        return {
            "anisotropy_score": score,
            "cosine_aniso": cos_aniso,
            "norm_iso": norm_iso,
        }
