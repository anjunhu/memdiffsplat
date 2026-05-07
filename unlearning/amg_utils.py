"""
AMG utilities — Nearest-Neighbour search for Anti-Memorisation Guidance.
Model-agnostic: operates on PIL images and CLIP embeddings.
"""
from typing import List, Optional
import torch
import torch.nn.functional as F
from PIL import Image


class NearestNeighborSearch:
    """
    Maintains a CLIP-embedded reference set and finds nearest neighbours
    for AMG guidance during sampling.
    """

    def __init__(self, clip_model=None, clip_processor=None, device: str = "cuda"):
        """
        Pass a HuggingFace CLIP model+processor, or leave None to use
        openai/clip-vit-base-patch32 loaded lazily.
        """
        self.device = device
        self._clip_model = clip_model
        self._clip_processor = clip_processor
        self._ref_embeddings: Optional[torch.Tensor] = None  # (N, D)
        self._ref_captions: List[str] = []

    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32").to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32")

    @torch.no_grad()
    def _embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        self._load_clip()
        inputs = self._clip_processor(images=images, return_tensors="pt").to(self.device)
        feats = self._clip_model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)

    def precompute_training_embeddings(self, images: List[Image.Image],
                                       captions: List[str]):
        """Embed reference images and store for NN lookup."""
        if not images:
            return
        self._ref_embeddings = self._embed_images(images)
        self._ref_captions = captions
        print(f"AMG: indexed {len(images)} reference images.")

    @torch.no_grad()
    def find_nearest(self, query_image: Image.Image, top_k: int = 1):
        """Return (similarity, caption) for the top-k nearest neighbours."""
        if self._ref_embeddings is None:
            return []
        q = self._embed_images([query_image])
        sims = (q @ self._ref_embeddings.T).squeeze(0)
        vals, idxs = sims.topk(min(top_k, len(self._ref_captions)))
        return [(vals[i].item(), self._ref_captions[idxs[i]]) for i in range(len(idxs))]

    @torch.no_grad()
    def guidance_direction(self, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Return a unit vector pointing away from the nearest reference embedding.
        query_emb: (1, D) normalised CLIP image embedding.
        """
        if self._ref_embeddings is None:
            return torch.zeros_like(query_emb)
        sims = (query_emb @ self._ref_embeddings.T).squeeze(0)
        nn_emb = self._ref_embeddings[sims.argmax()]
        direction = query_emb.squeeze(0) - nn_emb
        norm = direction.norm()
        return (direction / norm) if norm > 1e-8 else direction
