"""
SAIL — Sharpness-Aware Initialization for unLearning (portable version).
Operates via DiffusionModelAdapter; no model-specific imports.
"""
import torch
from .adapter import DiffusionModelAdapter


class SAILOptimizer:
    """
    Finds an initial noise x_T that minimises the sharpness of the score
    difference, making memorised content harder to reproduce.

    Algorithm 2 from the SAIL paper.
    """

    def __init__(self, adapter: DiffusionModelAdapter,
                 optim_steps: int = 20, lr: float = 0.05,
                 alpha: float = 0.05, delta: float = 1e-3,
                 early_stop_threshold: float = 8.0):
        self.adapter = adapter
        self.optim_steps = optim_steps
        self.lr = lr
        self.alpha = alpha
        self.delta = delta
        self.early_stop_threshold = early_stop_threshold

    def _score_diff(self, x_t, t, c, uc):
        return self.adapter.apply_model(x_t, t, c) - self.adapter.apply_model(x_t, t, uc)

    def optimize_noise(self, prompt: str, seed: int = 0) -> torch.Tensor:
        """Return an optimised x_T for the given prompt."""
        device = self.adapter.device
        gen = torch.Generator(device=device).manual_seed(seed)
        x_T = self.adapter.randn_latent(1, generator=gen).requires_grad_(True)

        text_emb = self.adapter.encode_text([prompt])
        null_emb = self.adapter.null_embedding(1)
        c = self.adapter.make_conditioning(text_emb)
        uc = self.adapter.make_conditioning(null_emb)
        t = torch.tensor([999], device=device)

        opt = torch.optim.Adam([x_T], lr=self.lr)

        for _ in range(self.optim_steps):
            opt.zero_grad()
            s = self._score_diff(x_T, t, c, uc)
            with torch.no_grad():
                norm = s.norm()
                if norm < 1e-8:
                    break
                perturb = self.delta * s / norm
            s_perturbed = self._score_diff(x_T + perturb, t, c, uc)
            loss = ((s_perturbed - s) ** 2).sum() + self.alpha * (x_T ** 2).sum()
            if loss.item() < self.early_stop_threshold:
                break
            loss.backward()
            opt.step()

        return x_T.detach()
