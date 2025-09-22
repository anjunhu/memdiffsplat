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
        Calculates the Localized Detection (LD) score using a hybrid approach:
        1. Tries to get the BE map from the live controller's cache.
        2. If that fails, falls back to loading the map from disk.
        """
        if intermediates is None:
            raise ValueError("BrightEndingMetric requires intermediates dict")
            
        uncond_noise = intermediates.get('uncond_noise', [])
        text_noise = intermediates.get('text_noise', [])
        
        if not uncond_noise or not text_noise:
            return {"error": "No noise data available"}
            
        d_global = torch.mean(torch.stack([d.norm(p=2) for d in [(tn - un) for tn, un in zip(text_noise, uncond_noise)]])).item()
        
        be_map = None
        
        # --- Stage 1: Try to get map from live controller cache (for baseline run) ---
        if controller is not None:
            try:
                # Get attention maps from the last timestep (step 49)
                final_timestep_maps = controller.get_attention_maps(timestep=49)
                
                if final_timestep_maps:
                    print(f"[BE Metric] Found attention maps at timestep 49: {list(final_timestep_maps.keys())}")
                    # 'down_0_attn1', 'down', 'down_0_attn2', 'down_1_attn1', 'down_1_attn2', 'down_2_attn1', 'down_2_attn_2', 'down_3_attn1', 'down_3_attn2']
                    # Try to get maps from down_0 and down_1 layers (32x32 and 16x16 resolution)
                    attn_map1 = final_timestep_maps.get('down_0_attn1')  # 32x32
                    attn_map2 = final_timestep_maps.get('down_1_attn1')  # 16x16
                    
                    if attn_map1 is not None or attn_map2 is not None:
                        # Use available maps, prioritizing higher resolution
                        if attn_map1 is not None and attn_map2 is not None:
                            # Interpolate down_0 to match down_1 size and average
                            # Handle half precision for interpolation
                            map1_for_interp = attn_map1.float() if attn_map1.dtype == torch.float16 else attn_map1
                            map1_resized = F.interpolate(
                                map1_for_interp.unsqueeze(0), 
                                size=attn_map2.shape[1:], 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze(0)
                            # Convert back to original dtype and average
                            map1_resized = map1_resized.to(attn_map1.dtype)
                            final_attn_map = (map1_resized + attn_map2) / 2.0
                        else:
                            final_attn_map = attn_map1 if attn_map1 is not None else attn_map2
                        
                        # Extract attention to the last token (typically end-of-text token)
                        be_map_raw = final_attn_map[:, :, -1]  # Shape: [batch, seq_len]
                        
                        # Get spatial dimensions from controller for proper reshaping
                        batch_size, seq_len = be_map_raw.shape
                        
                        # Determine spatial dimensions based on which layer we're using
                        spatial_dims = None
                        if attn_map1 is not None:
                            spatial_dims = controller.get_spatial_dims_for_layer('down_0_attn1')
                        elif attn_map2 is not None:
                            spatial_dims = controller.get_spatial_dims_for_layer('down_1_attn1')
                        
                        if spatial_dims is None:
                            # Fallback to square assumption
                            spatial_dim = int(seq_len**0.5)
                            spatial_dims = (spatial_dim, spatial_dim)
                            print(f"[BE Metric] Warning: Using square assumption for spatial dims: {spatial_dims}")
                        
                        spatial_h, spatial_w = spatial_dims
                        
                        # Average across batch dimension if needed
                        if batch_size > 1:
                            # For CFG, take only the conditional part (second half)
                            cond_batch_size = batch_size // 2
                            be_map_raw = be_map_raw[cond_batch_size:]
                        
                        be_map = be_map_raw.reshape(-1, spatial_h, spatial_w).mean(0)
                        print(f"[BE Metric] Successfully extracted BE map from controller cache, shape: {be_map.shape}")
                        
                    else:
                        print(f"[BE Metric] No suitable attention maps found. Available: {list(final_timestep_maps.keys())}")
                        
                else:
                    print("[BE Metric] No attention maps found at timestep 49")
                    
            except Exception as e:
                print(f"[BE Metric] Error extracting attention maps from controller: {e}")
                raise e

        # --- Stage 2: Fallback to loading from disk (for mitigated run or if cache fails) ---
        if be_map is None and attention_map_dir and os.path.exists(attention_map_dir):
            print("[BE Metric] Attention map not found in live cache. Attempting to load from disk...")
            try:
                all_final_step_maps = glob.glob(os.path.join(attention_map_dir, "step49_*_token*.png"))
                if all_final_step_maps:
                    last_token_idx = max([int(re.search(r"_token(\d+)\.png", f).group(1)) for f in all_final_step_maps])
                    map_files = glob.glob(os.path.join(attention_map_dir, f"step49_*_token{last_token_idx:02d}.png"))
                    
                    if map_files:
                        all_maps_from_disk = [torch.from_numpy(np.array(Image.open(f).convert("L")) / 255.0).float() for f in map_files]
                        be_map = torch.stack(all_maps_from_disk).mean(dim=0)
                        if uncond_noise:
                            be_map = be_map.to(uncond_noise[0].device)
                        print(f"[BE Metric] Loaded BE map from disk, shape: {be_map.shape}")
            except Exception as e:
                print(f"[BE Metric] Error loading attention maps from disk: {e}")
                raise e
        
        if be_map is None:
            print("[BE Metric] Could not find BE map from either live cache or disk. Returning only d_score.")
            return {"d_score": d_global, "error": "No attention maps available"}

        # --- Final Calculation ---
        # Convert to float32 for numerical stability
        if be_map.dtype == torch.float16:
            be_map = be_map.float()
        
        # Get noise difference trajectory
        noise_diff_traj = [(tn - un) for tn, un in zip(text_noise, uncond_noise)]
        
        # Ensure proper dimensions for interpolation
        if len(be_map.shape) == 2:
            be_map = be_map.unsqueeze(0).unsqueeze(0)
        elif len(be_map.shape) == 3:
            be_map = be_map.unsqueeze(0)
            
        masked_diffs = []
        for diff in noise_diff_traj:
            try:
                # Handle video tensors vs image tensors
                if len(diff.shape) == 5:
                    diff_2d = diff.mean(dim=2)
                else:
                    diff_2d = diff
                    
                # Interpolate BE map to match noise difference spatial dimensions
                be_map_for_interp = be_map.float()
                interpolated_map = F.interpolate(be_map_for_interp, size=diff_2d.shape[2:], mode='bilinear', align_corners=False)
                
                # Apply mask and compute norm
                diff_2d_float = diff_2d.detach().cpu().float()
                interpolated_map_float = interpolated_map.squeeze(0).float()
                masked_diff = (diff_2d_float * interpolated_map_float).norm(p=2)
                masked_diffs.append(masked_diff)
            except Exception:
                continue
        
        if not masked_diffs:
            return {"d_score": d_global, "error": "Could not process noise differences"}
            
        # Correct LD formula: mean of masked norms divided by mean of mask values
        numerator = torch.mean(torch.stack(masked_diffs))
        denominator = torch.mean(be_map.float()) * be_map.numel()  # N * (1/N * Σm_i) = Σm_i
        ld_score = numerator / (denominator + 1e-6)
        
        return {
            "ld_score": ld_score.item(), 
            "d_score": d_global, 
            "be_intensity": torch.mean(be_map.float()).item(),
        }