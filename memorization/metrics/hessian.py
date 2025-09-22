import torch
import torch.nn.functional as F
from typing import Dict
from .base import BaseMetric

class HessianMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "Hessian_SAIL_Metric"

    @property
    def metric_type(self) -> str:
        return "per_seed"
        
    @property
    def requires_intermediates(self) -> bool:
        return True
        
    @property
    def requires_model(self) -> bool:
        return True
        
    @property
    def requires_contexts(self) -> bool:
        return True

    def _prepare_unet_input(self, latents, model, encoder_hidden_states):
        """
        Prepare input for DiffSplat UNet by formatting for multi-view processing.
        
        Args:
            latents: Base latent tensor (B, 4, H, W) or (B*V, 4, H, W)
            model: The UNet model
            encoder_hidden_states: Text embeddings
            
        Returns:
            Prepared input tensor with correct format for multi-view UNet
        """
        from einops import rearrange
        
        # Determine number of views from model configuration or default to 4
        num_views = getattr(model.config, 'num_views', 4)
        
        # If latents are single-view, replicate for multi-view
        if latents.shape[0] == 1:
            # Replicate single latent for each view
            latents = latents.repeat(num_views, 1, 1, 1)  # (V, 4, H, W)
        
        # Ensure we have the right batch structure for multi-view
        if latents.shape[0] % num_views == 0:
            batch_size = latents.shape[0] // num_views
            # Reshape to multi-view format: (B*V, C, H, W) -> (B, V, C, H, W)
            latents = rearrange(latents, "(b v) c h w -> b v c h w", v=num_views)
        else:
            # Single batch, replicate across views
            batch_size = 1
            latents = latents.unsqueeze(0)  # Add batch dimension if needed
            if latents.shape[1] != num_views:
                latents = latents.repeat(1, num_views, 1, 1, 1)
        
        # Check if model expects additional channels
        expected_channels = model.conv_in.weight.shape[1]
        current_channels = latents.shape[2]  # Channel dimension in (B, V, C, H, W)
        
        if expected_channels > current_channels:
            additional_channels_needed = expected_channels - current_channels
            
            # Add zero-filled channels for missing inputs (plucker, mask, etc.)
            b, v, c, h, w = latents.shape
            additional_channels = torch.zeros(
                b, v, additional_channels_needed, h, w,
                device=latents.device, dtype=latents.dtype
            )
            
            latents = torch.cat([latents, additional_channels], dim=2)
        
        # Handle view concatenation conditions if needed
        if hasattr(model.config, 'view_concat_condition') and model.config.view_concat_condition:
            # For view concat condition, we might need to add condition views
            # For simplicity, we'll add zero views as condition
            zero_condition_views = torch.zeros_like(latents[:, :1, ...])  # (B, 1, C, H, W)
            latents = torch.cat([zero_condition_views, latents], dim=1)  # (B, V+1, C, H, W)
            num_views += 1
        
        # Add binary mask if required
        if hasattr(model.config, 'input_concat_binary_mask') and model.config.input_concat_binary_mask:
            b, v, c, h, w = latents.shape
            if hasattr(model.config, 'view_concat_condition') and model.config.view_concat_condition:
                # First view (condition) gets 0 mask, rest get 1 mask
                mask_channels = torch.cat([
                    torch.zeros(b, 1, 1, h, w, device=latents.device, dtype=latents.dtype),  # Condition view
                    torch.ones(b, v-1, 1, h, w, device=latents.device, dtype=latents.dtype)   # Generation views
                ], dim=1)
            else:
                # All views get 1 mask
                mask_channels = torch.ones(b, v, 1, h, w, device=latents.device, dtype=latents.dtype)
            
            latents = torch.cat([latents, mask_channels], dim=2)
        
        # Flatten back to (B*V, C, H, W) format for UNet input
        latents = rearrange(latents, "b v c h w -> (b v) c h w")
        
        # Ensure encoder_hidden_states matches the multi-view format
        if encoder_hidden_states.shape[0] == 1:
            # Replicate text embeddings for each view
            encoder_hidden_states = encoder_hidden_states.repeat(latents.shape[0], 1, 1)
        elif encoder_hidden_states.shape[0] != latents.shape[0]:
            # Adjust to match batch size
            target_batch = latents.shape[0]
            current_batch = encoder_hidden_states.shape[0]
            if target_batch % current_batch == 0:
                repeat_factor = target_batch // current_batch
                encoder_hidden_states = encoder_hidden_states.repeat(repeat_factor, 1, 1)
            else:
                # Fallback: just repeat to match
                encoder_hidden_states = encoder_hidden_states.repeat(target_batch, 1, 1)
        
        return latents, encoder_hidden_states

    @torch.no_grad()
    def measure(self, intermediates: Dict = None, model = None, 
                conditioning_context = None, unconditioning_context = None, **kwargs) -> Dict:
        """Calculates the fast Hessian metric and returns per-pixel magnitudes."""
        if intermediates is None or model is None:
            raise ValueError("HessianMetric requires intermediates dict and model")
        if conditioning_context is None or unconditioning_context is None:
            raise ValueError("HessianMetric requires conditioning and unconditioning contexts")

        # Extract context embeddings from the context dicts
        cond_embeds = conditioning_context["context"]
        uncond_embeds = unconditioning_context["context"]

        # --- Helper to get individual score functions for DiffSplat UNet ---
        def get_cond_score(x_t, t):
            # Prepare input with correct multi-view format for DiffSplat
            prepared_input, prepared_embeds = self._prepare_unet_input(x_t, model, cond_embeds)
            # DiffSplat UNet expects (sample, timestep, encoder_hidden_states)
            return model(prepared_input, t, encoder_hidden_states=prepared_embeds).sample
            
        def get_uncond_score(x_t, t):
            # Prepare input with correct multi-view format for DiffSplat
            prepared_input, prepared_embeds = self._prepare_unet_input(x_t, model, uncond_embeds)
            # DiffSplat UNet expects (sample, timestep, encoder_hidden_states)
            return model(prepared_input, t, encoder_hidden_states=prepared_embeds).sample

        results = {}
        steps_to_analyze = {"t50": 0, "t1": -1, "t20": -20} 

        for name, t_index in steps_to_analyze.items():
            if abs(t_index) > len(intermediates['x_inter']):
                continue

            latents = intermediates['x_inter'][t_index].to("cuda")
            timestep_int = intermediates['timesteps'][t_index]
            
            # Handle timestep conversion
            if isinstance(timestep_int, torch.Tensor):
                timestep_int = timestep_int.item()
            elif isinstance(timestep_int, (list, tuple)):
                timestep_int = timestep_int[0]
                
            timestep_tensor = torch.tensor(timestep_int, device=latents.device, dtype=torch.long)

            try:
                # Calculate score difference and perturbation
                s_delta = get_cond_score(latents, timestep_tensor) - get_uncond_score(latents, timestep_tensor)
                
                delta = 1e-3
                s_delta_norm = torch.linalg.norm(s_delta)
                if s_delta_norm < 1e-6:
                    continue
                
                perturbation = delta * s_delta / s_delta_norm
                latents_perturbed = latents + perturbation

                # Approximate Hessian-vector products separately
                h_s_cond = get_cond_score(latents_perturbed, timestep_tensor) - get_cond_score(latents, timestep_tensor)
                h_s_uncond = get_uncond_score(latents_perturbed, timestep_tensor) - get_uncond_score(latents, timestep_tensor)
                
                # Get per-pixel magnitudes for visualization
                # Handle different tensor shapes gracefully
                if len(h_s_cond.shape) > 4:  # (B, C, T, H, W) for video
                    cond_magnitudes = torch.linalg.norm(h_s_cond.squeeze(0), dim=(0, 1)).flatten()
                    uncond_magnitudes = torch.linalg.norm(h_s_uncond.squeeze(0), dim=(0, 1)).flatten()
                elif len(h_s_cond.shape) == 4:  # (B, C, H, W) for image
                    cond_magnitudes = torch.linalg.norm(h_s_cond.squeeze(0), dim=0).flatten()
                    uncond_magnitudes = torch.linalg.norm(h_s_uncond.squeeze(0), dim=0).flatten()
                else:  # Other shapes
                    cond_magnitudes = h_s_cond.flatten().abs()
                    uncond_magnitudes = h_s_uncond.flatten().abs()
                
                results[name] = {
                    "cond_magnitudes": torch.sort(cond_magnitudes).values.cpu().tolist(),
                    "uncond_magnitudes": torch.sort(uncond_magnitudes).values.cpu().tolist()
                }

            except Exception as e:
                print(f"Error in Hessian computation at {name}: {e}")
                print(f"Latents shape: {latents.shape}")
                print(f"Expected UNet input channels: {model.conv_in.weight.shape[1]}")
                print(f"Cond embeds shape: {cond_embeds.shape}")
                print(f"Uncond embeds shape: {uncond_embeds.shape}")
                print(f"Timestep: {timestep_tensor}")
                print(f"Model config - view_concat_condition: {getattr(model.config, 'view_concat_condition', 'Not set')}")
                print(f"Model config - input_concat_plucker: {getattr(model.config, 'input_concat_plucker', 'Not set')}")
                print(f"Model config - input_concat_binary_mask: {getattr(model.config, 'input_concat_binary_mask', 'Not set')}")
                raise e

        # Primary scalar metric at final step
        final_metric_val = 0.0
        if "t1" in results:
             h_s_cond_t1 = torch.tensor(results['t1']['cond_magnitudes'])
             h_s_uncond_t1 = torch.tensor(results['t1']['uncond_magnitudes'])
             final_metric_val = torch.sum((h_s_cond_t1 - h_s_uncond_t1)**2).item()

        return {
            "hessian_sail_norm": final_metric_val,
            "visualizations": results
        }