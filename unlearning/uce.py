"""
UCE — Unified Concept Editing (portable version).
Operates via DiffusionModelAdapter; no model-specific imports.
Reference: Gandikota et al., WACV 2024.
"""
from copy import deepcopy
from typing import Dict, List
import torch
import torch.nn as nn
from .adapter import DiffusionModelAdapter


class UCEEditor:
    """
    Closed-form weight update for cross-attention to/k and to/v layers.
    Erases edit_concepts while guiding toward guide_concepts and
    preserving preserve_concepts.
    """

    def __init__(self, adapter: DiffusionModelAdapter):
        self.adapter = adapter
        self._original_weights: Dict[str, torch.Tensor] = {}
        self._backup()

    def _backup(self):
        for name, layer in self.adapter.get_attention_layers("cross").items():
            self._original_weights[name] = layer.weight.detach().clone()

    def _concept_embedding(self, concept: str) -> torch.Tensor:
        """Return the last meaningful token embedding for a concept string."""
        emb = self.adapter.encode_text([concept])  # [1, L, D]
        # Use the last non-padding token (index -2 to skip EOS)
        return emb[:, -2, :]  # [1, D]

    @torch.no_grad()
    def erase_concept(self,
                      edit_concepts: List[str],
                      guide_concepts: List[str],
                      preserve_concepts: List[str],
                      erase_scale: float = 1.0,
                      preserve_scale: float = 0.5,
                      lamb: float = 0.5):
        """Apply the UCE closed-form weight update in-place."""
        layers = self.adapter.get_attention_layers("cross")

        # Pre-compute all embeddings
        embeds = {c: self._concept_embedding(c)
                  for c in edit_concepts + guide_concepts + preserve_concepts}

        # Pre-compute guide outputs from the *current* (original) weights
        guide_outputs: Dict[str, List[torch.Tensor]] = {}
        for concept in guide_concepts + preserve_concepts:
            t = embeds[concept].T  # [D, 1]
            guide_outputs[concept] = [layer(embeds[concept]).T
                                      for layer in layers.values()]

        for i, (name, layer) in enumerate(layers.items()):
            W = layer.weight.data.clone()
            dtype = W.dtype
            W = W.float()

            mat1 = lamb * W
            mat2 = lamb * torch.eye(W.shape[1], device=W.device)

            for ec, gc in zip(edit_concepts, guide_concepts):
                c = embeds[ec].T.float()
                v_star = guide_outputs[gc][i].float()
                mat1 += erase_scale * (v_star @ c.T)
                mat2 += erase_scale * (c @ c.T)

            for pc in preserve_concepts:
                c = embeds[pc].T.float()
                v_star = guide_outputs[pc][i].float()
                mat1 += preserve_scale * (v_star @ c.T)
                mat2 += preserve_scale * (c @ c.T)

            try:
                W_new = mat1 @ torch.inverse(mat2)
            except torch.linalg.LinAlgError:
                W_new = mat1 @ torch.pinverse(mat2)

            layer.weight.data = W_new.to(dtype)

    def restore(self):
        """Restore original weights."""
        for name, layer in self.adapter.get_attention_layers("cross").items():
            if name in self._original_weights:
                layer.weight.data.copy_(self._original_weights[name])
