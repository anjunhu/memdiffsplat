# DiffSplat/memorization/metrics/ca.py

import torch
from typing import Dict, Optional, List, Union
from .base import BaseMetric

class XAttnEntropyMetric(BaseMetric):
    """
    Cross-Attention Entropy Metric implementing both MemAttn variants:
    - CAE-E: Length-16 vector of entropy at t=T across all cross-attention modules
    - CAE-D: Two-term metric across final timesteps with summary token delta
    
    From Ren et al. (2024) - MemAttn paper.
    Replicates dimmem core logic for DiffSplat pipeline.
    """
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
    def measure(self, controller = None, model = None, prompt = None, **kwargs) -> Dict:
        """
        Calculates CAE-E and CAE-D metrics following dimmem implementation.
        
        Required kwargs:
            controller: AttentionController with recorded attention maps
            
        Optional kwargs:
            model: The diffusion model (for tokenization)
            prompt: Original text prompt (for tokenization to determine prompt length)
                   If provided, enables accurate summary token identification for CAE-D
        
        Returns:
            {
                "cae-e": list,       # Length-16 vector of entropies across all modules at t=T
                "cae-d": float,      # Two-term metric across final timesteps
            }
        """
        if controller is None:
            print(f"[{self.name}] No controller!")
            return {"cae-e": [0.0] * 16, "cae-d": 0.0}

        # Check available timesteps
        available_steps = []
        if hasattr(controller, 'attention_store'):
            available_steps = sorted(controller.attention_store.keys())
        elif hasattr(controller, 'get_attention_maps'):
            # Try to get available timesteps from controller
            try:
                test_maps = controller.get_attention_maps()
                if test_maps:
                    available_steps = [0]  # Assume at least one timestep available
            except:
                pass
        
        num_steps = len(available_steps)
        
        if num_steps == 0:
            print(f"[{self.name}] Warning: No timesteps available!")
            return {"cae-e": [0.0] * 16, "cae-d": 0.0}
        
        print(f"[{self.name}] Controller Cached Steps: {available_steps}")
        
        # Determine prompt length from tokenization if prompt and model provided
        prompt_length = None
        if prompt is not None and model is not None:
            try:
                # Try different tokenizer access patterns for DiffSplat
                tokenizer = None
                if hasattr(model, 'cond_stage_model') and hasattr(model.cond_stage_model, 'tokenizer'):
                    tokenizer = model.cond_stage_model.tokenizer
                elif hasattr(model, 'tokenizer'):
                    tokenizer = model.tokenizer
                elif hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'tokenizer'):
                    tokenizer = model.text_encoder.tokenizer
                
                if tokenizer:
                    tokens = tokenizer.encode(prompt)
                    if isinstance(tokens, torch.Tensor):
                        # Find actual prompt length (exclude padding tokens)
                        # Common padding token IDs: 0, 49407 (CLIP EOS), etc.
                        non_padding_mask = (tokens != 0) & (tokens != 49407)
                        prompt_length = non_padding_mask.sum().item()
                    elif isinstance(tokens, list):
                        # If it's a list, count non-padding tokens
                        prompt_length = len([t for t in tokens if t not in [0, 49407]])
                    
                    print(f"[{self.name}] Detected prompt length: {prompt_length}")
            except Exception as e:
                print(f"[{self.name}] Warning: Could not tokenize prompt: {e}")
        
        # DiffSplat multi-view handling parameters
        num_views = 4
        
        # ============================================================
        # CAE-E: Vectorized entropy across all modules at t=T
        # ============================================================
        first_step_maps = None
        if hasattr(controller, 'get_attention_maps'):
            first_step_maps = controller.get_attention_maps(timestep=available_steps[0])
        elif hasattr(controller, 'attention_store'):
            first_step_maps = controller.attention_store.get(available_steps[0], {})
        
        if not first_step_maps:
            print(f"[{self.name}] No attention maps found at first timestep")
            return {"cae-e": [0.0] * 16, "cae-d": 0.0}
        
        # Sort layer keys to ensure consistent ordering
        # Expected naming for DiffSplat: down_0_attn1, down_1_attn1, etc.
        sorted_layer_keys = sorted(first_step_maps.keys())
        
        cae_e = []
        
        for layer_key in sorted_layer_keys:
            attn_maps = first_step_maps[layer_key]
            
            # Extract conditional part (for CFG)
            cond_attn = attn_maps[attn_maps.shape[0] // 2:]
            
            # DiffSplat multi-view handling
            if cond_attn.shape[0] % num_views == 0 and cond_attn.shape[0] > 1:
                batch_size = cond_attn.shape[0] // num_views
                cond_attn = cond_attn.reshape(batch_size, num_views, *cond_attn.shape[1:])
                cond_attn = cond_attn.mean(dim=1)
            
            # Calculate entropy for this module
            avg_per_token = cond_attn.mean(dim=(0, 1))
            dist = avg_per_token / (avg_per_token.sum() + 1e-9)
            entropy = -torch.sum(dist * torch.log(dist + 1e-9)).item()
            cae_e.append(entropy)
        
        # Pad to length 16 if needed
        while len(cae_e) < 16:
            cae_e.append(0.0)
        
        # Truncate to length 16 if we have more
        cae_e = cae_e[:16]
        
        print(f"[{self.name}] CAE-E: Computed entropy for {len(sorted_layer_keys)} modules")
        print(f"[{self.name}] Layer ordering: {sorted_layer_keys}")
        
        # ============================================================
        # CAE-D: Final timesteps, all layers, two-term formula
        # ============================================================
        
        if num_steps == 1:
            print(f"[{self.name}] CAE-D: Not available (only 1 timestep)")
            return {"cae-e": cae_e, "cae-d": 0.0}
        
        # Only compute CAE-D if we have prompt length information
        if prompt_length is None:
            print(f"[{self.name}] CAE-D: Skipping (no prompt length available)")
            return {"cae-e": cae_e, "cae-d": 0.0}
        
        # Select final 20% of timesteps (like detect.py uses final 10 of 50)
        num_final_steps = max(1, int(num_steps * 0.2))
        final_step_indices = available_steps[-num_final_steps:]
        
        print(f"[{self.name}] CAE-D using final {num_final_steps} timesteps: {final_step_indices}")
        
        # Collect attention maps from ALL layers for final timesteps
        all_entropies = []
        all_summary_entropies = []
        
        for step_idx in final_step_indices:
            step_entropies = []
            step_summary_entropies = []
            
            # Get attention maps for this timestep
            step_maps = None
            if hasattr(controller, 'get_attention_maps'):
                step_maps = controller.get_attention_maps(timestep=step_idx)
            elif hasattr(controller, 'attention_store'):
                step_maps = controller.attention_store.get(step_idx, {})
            
            if not step_maps:
                continue
            
            # Aggregate across all available layers
            for layer_key in step_maps.keys():
                attn_maps = step_maps[layer_key]
                
                # Extract conditional part
                cond_attn = attn_maps[attn_maps.shape[0] // 2:]
                
                # DiffSplat multi-view handling
                if cond_attn.shape[0] % num_views == 0 and cond_attn.shape[0] > 1:
                    batch_size = cond_attn.shape[0] // num_views
                    cond_attn = cond_attn.reshape(batch_size, num_views, *cond_attn.shape[1:])
                    cond_attn = cond_attn.mean(dim=1)
                
                # Calculate entropy on all tokens (first term)
                avg_per_token = cond_attn.mean(dim=(0, 1))
                dist = avg_per_token / (avg_per_token.sum() + 1e-9)
                entropy_all = -torch.sum(dist * torch.log(dist + 1e-9)).item()
                step_entropies.append(entropy_all)
                
                # Calculate entropy on summary tokens only (second term)
                summary_tokens = avg_per_token[prompt_length:]
                
                if summary_tokens.numel() > 0:
                    dist_summary = summary_tokens / (summary_tokens.sum() + 1e-9)
                    entropy_summary = -torch.sum(dist_summary * torch.log(dist_summary + 1e-9)).item()
                    step_summary_entropies.append(entropy_summary)
            
            # Average across layers for this timestep
            if step_entropies:
                all_entropies.append(sum(step_entropies) / len(step_entropies))
            if step_summary_entropies:
                all_summary_entropies.append(sum(step_summary_entropies) / len(step_summary_entropies))
        
        # Get summary entropy at first timestep for delta calculation
        first_summary_entropies = []
        for layer_key in first_step_maps.keys():
            attn_maps = first_step_maps[layer_key]
            cond_attn = attn_maps[attn_maps.shape[0] // 2:]
            
            if cond_attn.shape[0] % num_views == 0 and cond_attn.shape[0] > 1:
                batch_size = cond_attn.shape[0] // num_views
                cond_attn = cond_attn.reshape(batch_size, num_views, *cond_attn.shape[1:])
                cond_attn = cond_attn.mean(dim=1)
            
            avg_per_token = cond_attn.mean(dim=(0, 1))
            summary_tokens = avg_per_token[prompt_length:]
            
            if summary_tokens.numel() > 0:
                dist_summary = summary_tokens / (summary_tokens.sum() + 1e-9)
                entropy_summary = -torch.sum(dist_summary * torch.log(dist_summary + 1e-9)).item()
                first_summary_entropies.append(entropy_summary)
        
        summary_entropy_first = None
        if first_summary_entropies:
            summary_entropy_first = sum(first_summary_entropies) / len(first_summary_entropies)
        
        # Calculate CAE-D: two terms
        if not all_entropies:
            print(f"[{self.name}] CAE-D: Failed to compute")
            return {"cae-e": cae_e, "cae-d": 0.0}
        
        # First term: average entropy across final timesteps
        term1 = sum(all_entropies) / len(all_entropies)
        
        # Second term: average |E_summary_t - E_summary_T|
        term2 = 0.0
        if all_summary_entropies and summary_entropy_first is not None:
            deltas = [abs(e - summary_entropy_first) for e in all_summary_entropies]
            term2 = sum(deltas) / len(deltas)
        
        cae_d = term1 + term2
        
        print(f"[{self.name}] CAE-D: {cae_d:.4f} (term1={term1:.4f}, term2={term2:.4f})")
        
        return {
            "cae-e": cae_e,
            "cae-d": cae_d,
        }