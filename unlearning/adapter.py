"""
DiffusionModelAdapter — thin ABC that unlearning methods program against.
Each model family provides a concrete subclass.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class DiffusionModelAdapter(ABC):
    """
    Minimal interface required by all portable unlearning methods.
    Concrete adapters wrap MVDream, SD, LaVie, CogVideo, DiffSplat, etc.
    """

    # ------------------------------------------------------------------
    # Text conditioning
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Return text embeddings [B, L, D]."""
        ...

    @abstractmethod
    def null_embedding(self, batch_size: int) -> torch.Tensor:
        """Return unconditional (empty-string) embedding [B, L, D]."""
        ...

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    @abstractmethod
    def apply_model(self, x_t: torch.Tensor, t: torch.Tensor,
                    conditioning: Dict) -> torch.Tensor:
        """Single UNet forward pass. Returns predicted noise."""
        ...

    @abstractmethod
    def make_conditioning(self, text_emb: torch.Tensor) -> Dict:
        """Wrap a text embedding tensor into the model's conditioning dict."""
        ...

    # ------------------------------------------------------------------
    # Architecture introspection (for UCE / SubspacePruner)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_attention_layers(self, layer_type: str = "cross") -> Dict[str, nn.Module]:
        """
        Return {name: module} for attention projection layers.
        layer_type: 'cross' → to_k / to_v in attn2
                    'ffn'   → ff.net.2 (for SubspacePruner)
        """
        ...

    # ------------------------------------------------------------------
    # Latent space
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def latent_shape(self) -> Tuple[int, ...]:
        """(C, H, W) of the latent space, e.g. (4, 32, 32)."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def randn_latent(self, batch_size: int, generator=None) -> torch.Tensor:
        return torch.randn(
            (batch_size, *self.latent_shape),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
