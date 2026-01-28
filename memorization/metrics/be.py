# DiffSplat/memorization/metrics/be.py

import os
import re
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, Optional, List

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

    def _find_down_block_maps(self, final_step_maps: Dict) -> List[torch.Tensor]:
        """
        Find attention maps from down blocks, handling various naming conventions.
        Returns a list of attention maps from down blocks.
        """
        down_maps = []
        
        # Priority 1: Look for aggregated down_0, down_1 keys (like DanceGRPO/MVDream)
        for i in range(4):
            key = f"down_{i}"
            if key in final_step_maps:
                down_maps.append((i, final_step_maps[key]))
        
        # Priority 2: Look for keys with pattern down_*_t*_attn2
        if not down_maps:
            for key, value in final_step_maps.items():
                if key.startswith('down_') and 'attn2' in key:
                    # Extract block index from key like "down_0_t0_attn2"
                    match = re.match(r'down_(\d+)', key)
                    if match:
                        block_idx = int(match.group(1))
                        down_maps.append((block_idx, value))
        
        # Priority 3: Look for aggregated 'down' key
        if not down_maps and 'down' in final_step_maps:
            down_maps.append((0, final_step_maps['down']))
        
        # Sort by block index and return just the tensors
        down_maps.sort(key=lambda x: x[0])
        return [m[1] for m in down_maps]

    def _extract_be_map(self, attn_maps: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Extract BE map from attention maps.
        Handles various tensor shapes and averaging across maps.
        """
        if not attn_maps:
            return None
        
        # Use first two maps if available (like dimmem)
        maps_to_use = attn_maps[:2] if len(attn_maps) >= 2 else attn_maps
        
        processed_maps = []
        target_shape = None
        
        for attn_map in maps_to_use:
            # Handle CFG by taking conditional half
            if attn_map.shape[0] % 2 == 0 and attn_map.shape[0] > 1:
                attn_map = attn_map[attn_map.shape[0] // 2:]
            
            # Extract last token attention: shape is (B, Q, K) where K is num tokens
            if attn_map.dim() == 3:
                be_map_raw = attn_map[:, :, -1]  # (B, Q) - last token
            elif attn_map.dim() == 4:
                # Shape might be (B, V, Q, K) for multi-view
                be_map_raw = attn_map[:, :, :, -1].mean(dim=1)  # Average views, take last token
            else:
                continue
            
            B = be_map_raw.shape[0]
            Q = be_map_raw.shape[1]
            
            # Try to reshape to spatial dimensions
            latent_dim = int(Q ** 0.5)
            if latent_dim * latent_dim == Q:
                spatial_map = be_map_raw.reshape(B, latent_dim, latent_dim).mean(0)
                
                if target_shape is None:
                    target_shape = spatial_map.shape
                    processed_maps.append(spatial_map)
                elif spatial_map.shape == target_shape:
                    processed_maps.append(spatial_map)
                else:
                    # Resize to match target shape
                    resized = F.interpolate(
                        spatial_map.unsqueeze(0).unsqueeze(0),
                        size=target_shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    processed_maps.append(resized)
        
        if not processed_maps:
            return None
        
        # Average all processed maps
        be_map = torch.stack(processed_maps).mean(0)
        return be_map

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
        if controller and hasattr(controller, 'attention_store'):
            try:
                available_timesteps = list(controller.attention_store.keys())
                print(f"[BrightEndingMetric] Available timesteps in controller: {available_timesteps}")
                
                if available_timesteps:
                    # Use the final timestep (smallest value for step index)
                    final_timestep = min(available_timesteps)
                    final_step_maps = controller.attention_store.get(final_timestep, {})
                    
                    print(f"[BrightEndingMetric] Final timestep {final_timestep} has layers: {list(final_step_maps.keys())}")
                    
                    if final_step_maps:
                        # Find down block attention maps
                        down_maps = self._find_down_block_maps(final_step_maps)
                        
                        if down_maps:
                            print(f"[BrightEndingMetric] Found {len(down_maps)} down block maps")
                            for i, m in enumerate(down_maps):
                                print(f"[BrightEndingMetric]   Map {i}: shape {m.shape}")
                            
                            be_map = self._extract_be_map(down_maps)
                            
                            if be_map is not None:
                                print(f"[BrightEndingMetric] Extracted BE map: {be_map.shape}")
                        else:
                            print(f"[BrightEndingMetric] No down block maps found in keys: {list(final_step_maps.keys())}")
                                        
            except Exception as e:
                print(f"[BrightEndingMetric] Error extracting from controller: {e}")
                import traceback
                traceback.print_exc()
        
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
            masked_diff = (diff * be_map_resized).norm(p=2)
            masked_diffs.append(masked_diff)
        
        # LD = (1/T) * sum(||(...) ○ m||_2) / (1/N * sum(m_i))
        T = len(masked_diffs)
        numerator = torch.mean(torch.stack(masked_diffs))
        
        # Use the resized be_map for denominator calculation
        be_map_for_denom = F.interpolate(
            be_map.to(noise_diff_traj[0].device), 
            size=noise_diff_traj[0].shape[2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
        denominator = torch.mean(be_map_for_denom) + 1e-6
        
        ld_score = numerator / denominator
        
        print(f"[BrightEndingMetric] LD Score: {ld_score.item()}, D Score: {d_global}, BE Mean: {torch.mean(be_map).item()}")
        return {
            "ld_score": ld_score.item(), 
            "d_score": d_global, 
            "be_intensity": torch.mean(be_map).item()
        }
