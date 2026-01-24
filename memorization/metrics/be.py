# DiffSplat/memorization/metrics/be.py

import os
import re
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .base import BaseMetric

class BrightEndingMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "BrightEnding_LD_Score"
        
    @property
    def metric_type(self) -> str:
        return "per_seed"
        
    @property
    def requires_intermediates(self) -> bool:
        return True
        
    @property
    def requires_attention_maps(self) -> bool:
        return True

    @torch.no_grad()
    def measure(self, intermediates: Dict = None, controller = None, 
                attention_map_dir = None, **kwargs) -> Dict:
        """
        Calculates the Localized Detection (LD) score using dimmem core logic:
        1. Tries to get the BE map from the live controller's cache.
        2. If that fails, falls back to loading the map from disk.
        3. Uses raw attention scores as weights (no normalization).
        4. Proper CFG handling by taking conditional half.
        """
        if intermediates is None:
            return {"error": "BrightEndingMetric requires intermediates dict"}
            
        uncond_noise = intermediates.get('uncond_noise', [])
        text_noise = intermediates.get('text_noise', [])
        
        if not uncond_noise or not text_noise:
            return {"error": "No noise data available"}
            
        # Global magnitude (unchanged from dimmem)
        d_global = torch.mean(torch.stack([d.norm(p=2) for d in [(tn - un) for tn, un in zip(text_noise, uncond_noise)]])).item()
        
        be_map = None
        
        # --- Stage 1: Try to get map from live controller cache (for baseline run) ---
        if controller and hasattr(controller, 'get_attention_maps'):
            try:
                # Get attention maps from the final timestep
                available_timesteps = list(controller.attention_store.keys()) if hasattr(controller, 'attention_store') else []
                if available_timesteps:
                    final_timestep = max(available_timesteps)
                    final_step_maps = controller.get_attention_maps(timestep=final_timestep)
                    
                    if final_step_maps:
                        # DiffSplat uses different layer naming - adapt to available layers
                        layer_candidates = ["down_0_attn1", "down_1_attn1", "down_2_attn1", "down_3_attn1"]
                        attn_maps = []
                        
                        for layer_name in layer_candidates:
                            if layer_name in final_step_maps:
                                attn_map = final_step_maps[layer_name]
                                # Handle CFG by taking conditional half
                                if attn_map.shape[0] % 2 == 0:
                                    cond_attn = attn_map[attn_map.shape[0] // 2:]
                                else:
                                    cond_attn = attn_map
                                attn_maps.append(cond_attn)
                        
                        if len(attn_maps) >= 2:
                            # Average the first two available layers (like dimmem down_0 + down_1)
                            final_attn_map = (attn_maps[0] + attn_maps[1]) / 2.0
                            be_map_raw = final_attn_map[:, :, -1]  # Last token
                            
                            # Reshape assuming 8 heads and square spatial dimensions
                            if len(be_map_raw.shape) == 2:
                                num_heads = 8
                                spatial_dim = int((be_map_raw.shape[1] // num_heads) ** 0.5)
                                if spatial_dim * spatial_dim * num_heads == be_map_raw.shape[1]:
                                    be_map = be_map_raw.reshape(-1, num_heads, spatial_dim, spatial_dim).mean(1)
                                else:
                                    # Fallback: treat as spatial map directly
                                    spatial_dim = int(be_map_raw.shape[1] ** 0.5)
                                    if spatial_dim * spatial_dim == be_map_raw.shape[1]:
                                        be_map = be_map_raw.reshape(-1, spatial_dim, spatial_dim).mean(0)
                        elif len(attn_maps) == 1:
                            # Use single layer if only one available
                            be_map_raw = attn_maps[0][:, :, -1]
                            spatial_dim = int(be_map_raw.shape[1] ** 0.5)
                            if spatial_dim * spatial_dim == be_map_raw.shape[1]:
                                be_map = be_map_raw.reshape(-1, spatial_dim, spatial_dim).mean(0)
                        
                        print(f"[BrightEndingMetric] Extracted BE map from controller: {be_map.shape if be_map is not None else 'None'}")
            except Exception as e:
                print(f"[BrightEndingMetric] Error extracting from controller: {e}")
        
        # --- Stage 2: Fallback to loading from disk (for mitigated run or if cache fails) ---
        if be_map is None and attention_map_dir and os.path.exists(attention_map_dir):
            print("[BrightEndingMetric] Warning: BE map not found in live cache. Attempting to load from disk...")
            all_final_step_maps = glob.glob(os.path.join(attention_map_dir, "step49_*_token*.png"))
            if all_final_step_maps:
                last_token_idx = max([int(re.search(r"_token(\d+)\.png", f).group(1)) for f in all_final_step_maps])
                map_files = glob.glob(os.path.join(attention_map_dir, f"step49_*_token{last_token_idx:02d}.png"))
                if map_files:
                    all_maps_from_disk = [torch.from_numpy(np.array(Image.open(f).convert("L")) / 255.0).float() for f in map_files]
                    be_map = torch.stack(all_maps_from_disk).mean(dim=0)
        
        if be_map is None:
            print("[BrightEndingMetric] Warning: Could not find BE map from either live cache or disk. Returning only d_score.")
            return {"d_score": d_global}

        # Use raw attention scores as weights (NO normalization - key difference from original)
        # This follows dimmem logic exactly
        
        # Compute noise differences
        noise_diff_traj = [(tn - un) for tn, un in zip(text_noise, uncond_noise)]
        
        # Ensure be_map has batch dimension for interpolation
        if len(be_map.shape) == 2:
            be_map = be_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif len(be_map.shape) == 3:
            be_map = be_map.unsqueeze(0)  # [1, B, H, W] -> assume B=1
        
        # Keep everything on same device and use raw attention weights
        masked_diffs = []
        for diff in noise_diff_traj:
            # Interpolate BE map to match diff spatial dimensions
            be_map_resized = F.interpolate(
                be_map.to(diff.device), 
                size=diff.shape[2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).unsqueeze(1)
            
            # Element-wise multiplication and L2 norm
            print(f"[BrightEndingMetric] BE map resized shape: {be_map_resized.shape}, Noise diff shape: {diff.shape}")
            masked_diff = (diff * be_map_resized).norm(p=2)
            masked_diffs.append(masked_diff)
        
        # LD = (1/T) * sum(||(...) ○ m||_2) / (1/N * sum(m_i))
        T = len(masked_diffs)  # Number of timesteps
        numerator = torch.mean(torch.stack(masked_diffs))  # (1/T) * sum(...)
        
        # Use the resized be_map for denominator calculation (mean of attention weights)
        be_map_for_denom = F.interpolate(
            be_map.to(noise_diff_traj[0].device), 
            size=noise_diff_traj[0].shape[2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
        denominator = torch.mean(be_map_for_denom) + 1e-6  # (1/N * sum(m_i))
        
        ld_score = numerator / denominator
        
        print(f"[BrightEndingMetric] LD Score: {ld_score.item()}, D Score: {d_global}, BE Mean: {torch.mean(be_map).item()}")
        return {
            "ld_score": ld_score.item(), 
            "d_score": d_global, 
            "be_intensity": torch.mean(be_map).item()
        }