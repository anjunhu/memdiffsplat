import torch
from typing import Dict, Optional
from .base import BaseMetric

class XAttnEntropyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "CrossAttention_Entropy"
        
    @property
    def metric_type(self) -> str:
        return "per_seed"
        
    @property
    def requires_attention_maps(self) -> bool:
        return True

    @torch.no_grad()
    def measure(self, controller = None, **kwargs) -> Dict:
        """
        Calculates the attention entropy, as per Ren et al. (2024).
        We use the faster version (E_t=T^l) from the paper.
        """
        if controller is None:
            print("[XAttnEntropyMetric] No controller provided!")
            return {"entropy": 0, "error": "No controller provided"}

        try:
            # Get attention maps from the first timestep (step 0)
            first_timestep_maps = controller.get_attention_maps(timestep=999)
            
            if not first_timestep_maps:
                print("[XAttnEntropyMetric] No attention maps found at timestep 0")
                available_timesteps = list(controller.attention_store.keys())
                print(f"[XAttnEntropyMetric] Available timesteps: {available_timesteps}")
                return {"entropy": 0, "error": f"No attention maps at timestep 0. Available: {available_timesteps}"}

            print(f"[XAttnEntropyMetric] Found attention maps at timestep 0: {list(first_timestep_maps.keys())}")
            
            # Target the 4th down layer (down_3) as per the paper
            target_layer_name = "down_3_attn1"
            attn_maps = first_timestep_maps.get(target_layer_name, None)
            
            if attn_maps is None:
                # Try alternative layer names or use the deepest available layer
                layer_candidates = ["down_3_attn2", "down_2_attn1", "down_2_attn2", 
                                    "down_1_attn1", "down_1_attn2", "down_0_attn1", "down_0_attn2", ]
                for candidate in layer_candidates:
                    if candidate in first_timestep_maps:
                        attn_maps = first_timestep_maps[candidate]
                        target_layer_name = candidate
                        print(f"[XAttnEntropyMetric] Using layer '{candidate}' instead of 'down_3'")
                        break
                        
            if attn_maps is None:
                available_keys = list(first_timestep_maps.keys())
                print(f"[XAttnEntropyMetric] Could not find suitable attention layer. Available: {available_keys}")
                return {"entropy": 0, "error": f"No suitable attention layer found. Available: {available_keys}"}
            
            print(f"[XAttnEntropyMetric] Using attention maps from layer '{target_layer_name}', shape: {attn_maps.shape}")
            
            # Handle CFG: only use the conditional part of the batch
            if attn_maps.shape[0] > 1:
                # For CFG, the second half is the conditional part
                cond_batch_size = attn_maps.shape[0] // 2
                cond_attn_map = attn_maps[cond_batch_size:]
            else:
                cond_attn_map = attn_maps
            
            print(f"[XAttnEntropyMetric] Conditional attention map shape: {cond_attn_map.shape}")
            
            # Average across spatial locations (patches) and batch dimension to get per-token scores
            # Shape should be [batch, spatial_locations, num_tokens]
            avg_per_token_scores = cond_attn_map.mean(dim=(0, 1))  # Average over batch and spatial dims
            
            print(f"[XAttnEntropyMetric] Per-token scores shape: {avg_per_token_scores.shape}")
            print(f"[XAttnEntropyMetric] Per-token scores: {avg_per_token_scores}")
            
            # Normalize to create a probability distribution
            score_sum = avg_per_token_scores.sum()
            if score_sum == 0:
                print("[XAttnEntropyMetric] Warning: All attention scores are zero")
                return {"entropy": 0, "error": "All attention scores are zero"}
                
            dist = avg_per_token_scores / (score_sum + 1e-9)
            
            # Calculate entropy: H = -sum(p * log(p))
            entropy = -torch.sum(dist * torch.log(dist + 1e-9)).item()
            
            print(f"[XAttnEntropyMetric] Computed entropy: {entropy}")
            
            return {
                "entropy": entropy,
                "num_tokens": avg_per_token_scores.shape[0],
                "layer_used": target_layer_name
            }
            
        except Exception as e:
            print(f"[XAttnEntropyMetric] Error computing entropy: {e}")
            import traceback
            traceback.print_exc()
            return {"entropy": 0, "error": str(e)}