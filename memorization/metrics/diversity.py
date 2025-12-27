# DiffSplat/memorization/metrics/diversity.py

import os
import itertools
import numpy as np
from PIL import Image
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from .base import BaseMetric

class DiversityMetric(BaseMetric):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        sscd_model_path = "sscd_disc_mixup.torchscript.pt"
        if not os.path.exists(sscd_model_path):
            raise FileNotFoundError(
                f"SSCD model not found at {sscd_model_path}. "
                "Please download it first using: "
                "'wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt'"
            )
        
        self.sscd_model = torch.jit.load(sscd_model_path).to(self.device)
        self.preprocess_sscd = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Preprocessing for the L2 distance metric
        self.preprocess_l2 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    @property
    def name(self) -> str:
        return "Image_Diversity"

    @property
    def metric_type(self) -> str:
        return "per_prompt_across_seeds"

    def _compute_noise_difference_image(self, uncond_noise: List[torch.Tensor], text_noise: List[torch.Tensor]) -> torch.Tensor:
        """Computes the noise difference image δ = p_θ(x_T, T, y) - x_T"""
        noise_diffs = [tn - un for un, tn in zip(uncond_noise, text_noise)]
        return torch.stack(noise_diffs).mean(dim=0)

    def _normalize_for_ssim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range for SSIM computation."""
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        if tensor_max > tensor_min:
            return (tensor - tensor_min) / (tensor_max - tensor_min)
        else:
            return torch.zeros_like(tensor)

    def _compute_ssim_score(self, noise_diff_images: List[torch.Tensor]) -> float:
        """Compute SSIM score between pairs of noise difference images."""
        if len(noise_diff_images) < 2:
            return 0.0

        ssim_scores = []
        for i, j in itertools.combinations(range(len(noise_diff_images)), 2):
            delta_i = self._normalize_for_ssim(noise_diff_images[i])
            delta_j = self._normalize_for_ssim(noise_diff_images[j])
            
            if len(delta_i.shape) == 3:
                delta_i = delta_i.unsqueeze(0)
                delta_j = delta_j.unsqueeze(0)
            
            try:
                ssim_val = structural_similarity_index_measure(
                    delta_i.to(self.device), delta_j.to(self.device), data_range=1.0
                )
                ssim_scores.append(ssim_val.item())
            except Exception:
                continue
        
        return np.mean(ssim_scores) if ssim_scores else 0.0

    def _get_tiles(self, image_tensor: torch.Tensor, tile_size: int = 128) -> torch.Tensor:
        """Divides an image tensor into non-overlapping tiles."""
        # A 256x256 image with 128x128 tiles results in 4 tiles.
        # This function is flexible for other sizes as well.
        assert image_tensor.dim() == 3, "Input tensor must be C x H x W"
        c, h, w = image_tensor.shape
        assert h % tile_size == 0 and w % tile_size == 0, "Image dimensions must be divisible by tile_size"
        
        tiles = image_tensor.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
        # Reshape to (num_tiles, C, H, W)
        tiles = tiles.contiguous().reshape(c, -1, tile_size, tile_size).permute(1, 0, 2, 3)
        return tiles

    @torch.no_grad()
    def measure(self, images: List[Image.Image], intermediates: Optional[List[Dict]] = None, **kwargs) -> Dict:
        """
        Calculates diversity scores for a set of images from a single prompt using two metrics.

        Args:
            images (List[Image.Image]): The list of generated images for one prompt.
        
        Returns:
            Dict: A dictionary with the following keys:
                  - "median_sscd_similarity": Median similarity from the SSCD model. Lower indicates more diversity.
                  - "min_tiled_l2_distance": The minimum pairwise tiled L2 distance. Lower indicates more memorization/less diversity.
        """
        if len(images) < 2:
            return {
                "median_sscd_similarity": 1.0,
                "min_tiled_l2_distance": 0.0,
                "ssim_noise_diff": 0.0,
            } 

        # Metric 1: SSCD-based Similarity (Original)
        preprocessed_sscd_images = torch.stack([self.preprocess_sscd(img) for img in images]).to(self.device)
        embeddings = self.sscd_model(preprocessed_sscd_images)
        similarity_matrix = pairwise_cosine_similarity(embeddings)
        tril_mask = torch.tril(torch.ones_like(similarity_matrix), diagonal=-1).bool()
        sim_scores = similarity_matrix[tril_mask]
        median_sscd_similarity = sim_scores.median().item()
        
        # Metric 2: Tiled L2 Distance [cite: 803, 805]
        # This metric is based on Carlini et al. (2023) as described in the paper.
        # It measures the pixel-wise L2 distance between pairs of images on tiled sections.
        preprocessed_l2_images = torch.stack([self.preprocess_l2(img) for img in images]).to(self.device)
        
        pairwise_distances = []
        for img1_tensor, img2_tensor in itertools.combinations(preprocessed_l2_images, 2):
            # The paper divides images into 128x128 tiles[cite: 805].
            tiles1 = self._get_tiles(img1_tensor, tile_size=128)
            tiles2 = self._get_tiles(img2_tensor, tile_size=128)
            
            # Reshape the difference tensor so that each tile's data is a flat vector
            diff = tiles1 - tiles2
            diff_flat = diff.reshape(diff.size(0), -1) # Shape: (num_tiles, C*H*W)

            # Compute the L2 norm along the flattened dimension for each tile
            tile_distances = torch.linalg.norm(diff_flat, dim=1)
            
            # The distance between the two images is the mean of their tile distances.
            # This is a robust way to represent the overall difference.
            mean_distance = tile_distances.mean().item()
            pairwise_distances.append(mean_distance)
            
        min_tiled_l2_distance = min(pairwise_distances) if pairwise_distances else 0.0

        results = {
            "median_sscd_similarity": median_sscd_similarity,
            "min_tiled_l2_distance": min_tiled_l2_distance,
        }
        
        # Metric 3: SSIM of Noise Differences
        if intermediates is not None and len(intermediates) >= 2:
            try:
                noise_diff_images = []
                for intermediates in intermediates:
                    if 'uncond_noise' in intermediates and 'text_noise' in intermediates:
                        noise_diff = self._compute_noise_difference_image(
                            intermediates['uncond_noise'], intermediates['text_noise']
                        )
                        noise_diff_images.append(noise_diff)
                
                if len(noise_diff_images) >= 2:
                    results["ssim_noise_diff"] = self._compute_ssim_score(noise_diff_images)
                else:
                    results["ssim_noise_diff"] = 0.0
            except Exception as e:
                print(f"[DiversityMetric] Warning: Could not compute SSIM of noise differences: {e}")
                raise e
        else:
            results["ssim_noise_diff"] = 0.0

        return results