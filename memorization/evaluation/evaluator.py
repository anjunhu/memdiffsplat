# Memory-optimized evaluator.py with DiffSplat multi-view support and updated attention controller

import os
import json
import logging
import torch
import imageio
import numpy as np
from PIL import Image
from typing import Dict, List, Any
from einops import rearrange

# Ensure all metric classes are imported
from ..metrics import (
    NoiseDiffNormMetric, HessianMetric,
    DiversityMetric, BrightEndingMetric, XAttnEntropyMetric,
    InvMMMetric, PLaplaceMetric
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiffSplatEvaluator:
    """
    Orchestrates the evaluation of a single prompt against multiple memorization metrics
    for the DiffSplat model (3D Gaussian Splatting diffusion).
    
    Updated to work with the new attention controller integration pattern.
    """
    def __init__(self, pipeline: Any, gsvae: Any, gsrecon: Any, metrics: List[Any], device: str = "cuda"):
        self.pipeline = pipeline
        self.gsvae = gsvae
        self.gsrecon = gsrecon
        self.metrics = metrics
        self.device = device

    def _prepare_contexts_for_hessian(self, prompt: str, guidance_scale: float):
        """Prepare conditioning and unconditioning contexts for Hessian metrics."""
        try:
            # Encode the prompt using DiffSplat's pipeline
            prompt_embeds = self.pipeline._encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=""
            )
            
            # Split into unconditional and conditional parts
            uncond_embeds, cond_embeds = prompt_embeds.chunk(2)
            
            # Create context dictionaries
            conditioning_context = {"context": cond_embeds}
            unconditioning_context = {"context": uncond_embeds}
            
            return conditioning_context, unconditioning_context
        except Exception as e:
            logger.error(f"Error preparing contexts: {e}")
            return None, None

    def process_single_prompt_single_seed(self,
                                          prompt: str,
                                          seed: int,
                                          num_inference_steps: int,
                                          guidance_scale: float,
                                          camera_params: Dict[str, Any],
                                          render_params: Dict[str, Any],
                                          unlearning_artifacts: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generates multi-view 3D scene for a single prompt and seed, then computes all metrics.
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            camera_params: Dictionary with camera parameters (elevations, azimuths, radius, fxfycxcy, etc.)
            render_params: Dictionary with rendering parameters (height, width, opacity_threshold, etc.)
            unlearning_artifacts: Dictionary with controller and attention_map_dir
        """
        torch.manual_seed(seed)
        
        # Extract attention controller
        controller = unlearning_artifacts.get("controller") if unlearning_artifacts else None
        attention_map_dir = unlearning_artifacts.get("attention_map_dir") if unlearning_artifacts else None
        
        # Prepare contexts for Hessian metrics if needed
        conditioning_context = None
        unconditioning_context = None
        needs_contexts = any(getattr(metric, 'requires_contexts', False) for metric in self.metrics)
        
        if needs_contexts:
            conditioning_context, unconditioning_context = self._prepare_contexts_for_hessian(prompt, guidance_scale)
        
        # Reset controller before generation if available
        if controller is not None:
            controller.reset()
            logger.debug("Reset attention controller for new generation")
        
        # --- Run the DiffSplat Pipeline ---
        try:
            # Extract camera parameters
            input_C2W = camera_params['input_C2W']
            input_fxfycxcy = camera_params['input_fxfycxcy']
            plucker = camera_params.get('plucker')
            num_views = camera_params.get('num_views', 4)
            
            # Generate latent Gaussians with attention controller
            pipeline_output = self.pipeline(
                None,  # No input image for text-to-3D
                prompt=prompt, 
                negative_prompt=camera_params.get('negative_prompt', ''),
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale,
                triangle_cfg_scaling=camera_params.get('triangle_cfg_scaling', False),
                min_guidance_scale=camera_params.get('min_guidance_scale', 1.0), 
                max_guidance_scale=guidance_scale,
                output_type="latent", 
                eta=camera_params.get('eta', 1.0), 
                generator=torch.Generator(device=self.device).manual_seed(seed),
                plucker=plucker, 
                num_views=num_views,
                init_std=camera_params.get('init_std', 0.0), 
                init_noise_strength=camera_params.get('init_noise_strength', 0.98), 
                init_bg=camera_params.get('init_bg', 0.0),
                guess_mode=camera_params.get('guess_mode', False),
                controlnet_conditioning_scale=camera_params.get('controlnet_scale', 1.0),
                # --- ATTENTION CONTROLLER PASSED DIRECTLY TO PIPELINE ---
                attention_controller=controller
            )
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {"error": f"Pipeline failed: {str(e)}"}
        
        # Extract Gaussian latents and decode/render
        try:
            latents = pipeline_output.images
            latents = latents / self.gsvae.scaling_factor + self.gsvae.shift_factor
            
            # Render multi-view images
            render_outputs = self.gsvae.decode_and_render_gslatents(
                self.gsrecon,
                latents, 
                input_C2W.unsqueeze(0), 
                input_fxfycxcy.unsqueeze(0),
                height=render_params['height'], 
                width=render_params['width'],
                opacity_threshold=render_params.get('opacity_threshold', 0.0),
            )
            
            # Extract rendered images (V_in, 3, H, W)
            rendered_images = render_outputs["image"].squeeze(0)
            
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            return {"error": f"Rendering failed: {str(e)}"}
        
        # Create intermediates dict from pipeline output
        intermediates = {
            'uncond_noise': getattr(pipeline_output, 'uncond_noise', []),
            'text_noise': getattr(pipeline_output, 'text_noise', []),
            'x_inter': getattr(pipeline_output, 'x_inter', []),
            'timesteps': getattr(pipeline_output, 'timesteps', [])
        }
        
        # Capture attention maps from controller if available
        captured_attention_maps = None
        if controller and hasattr(controller, 'get_attention_maps'):
            try:
                captured_attention_maps = controller.get_attention_maps()
                if captured_attention_maps:
                    logger.debug(f"Captured attention maps from {len(captured_attention_maps)} timesteps")
                    for timestep, layers in captured_attention_maps.items():
                        logger.debug(f"  Timestep {timestep}: {list(layers.keys())}")
            except Exception as e:
                logger.warning(f"Failed to extract attention maps from controller: {e}")
        
        # --- Compute Metrics ---
        metric_scores = {}
        for metric in self.metrics:
            # Skip diversity metric, it's handled cross-seed
            if isinstance(metric, DiversityMetric):
                continue
                
            try:
                logger.debug(f"Computing metric: {metric.name}")
                
                # Prepare metric-specific arguments
                kwargs = {
                    'prompt': prompt,
                    'seed': seed
                }
                
                # Add intermediates if required
                if getattr(metric, 'requires_intermediates', False):
                    kwargs['intermediates'] = intermediates
                
                # Add model if required  
                if getattr(metric, 'requires_model', False):
                    kwargs['model'] = self.pipeline.unet
                
                # Add contexts if required
                if getattr(metric, 'requires_contexts', False):
                    if conditioning_context is None or unconditioning_context is None:
                        logger.warning(f"Skipping {metric.name} - contexts not available")
                        metric_scores[metric.name] = {"error": "Contexts not available"}
                        continue
                    kwargs['conditioning_context'] = conditioning_context
                    kwargs['unconditioning_context'] = unconditioning_context
                
                # Add attention information if required
                if getattr(metric, 'requires_attention_maps', False):
                    kwargs['controller'] = controller
                    kwargs['attention_map_dir'] = attention_map_dir
                    # Also provide captured attention maps directly
                    kwargs['captured_attention_maps'] = captured_attention_maps
                
                # Legacy support for metrics expecting image tensor
                if isinstance(metric, BrightEndingMetric):
                    # For multi-view, we can pass the first view or all views
                    # BrightEndingMetric will need to be adapted for multi-view
                    kwargs['image_tensor'] = rendered_images

                # Compute the metric
                score = metric.measure(**kwargs)
                metric_scores[metric.name] = score
                logger.debug(f"{metric.name} score: {score}")

            except Exception as e:
                logger.error(f"Failed to compute metric {metric.name} for prompt '{prompt}'. Error: {e}", exc_info=True)
                metric_scores[metric.name] = {"error": str(e)}
        
        # Prepare result with attention information
        result = {
            "rendered_images": rendered_images,  # (V_in, 3, H, W) tensor
            "metrics": metric_scores,
            "x_inter": intermediates.get('x_inter', []),
            "timesteps": intermediates.get('timesteps', []),
            # Keep individual components for potential further analysis
            "uncond_noise": intermediates.get('uncond_noise', []),
            "text_noise": intermediates.get('text_noise', []),
        }
        
        # Add attention maps info to result if available
        if captured_attention_maps:
            result["attention_maps_available"] = True
            result["attention_timesteps"] = list(captured_attention_maps.keys())
            result["attention_layers"] = {
                timestep: list(layers.keys()) 
                for timestep, layers in captured_attention_maps.items()
            }
        else:
            result["attention_maps_available"] = False
                
        return result


def save_run_outputs(result: Dict[str, Any], output_dir: str, base_filename: str):
    """Saves the multi-view images, montage, and metrics JSON file with memory optimization."""
    if "error" in result:
        # Save error info
        os.makedirs(output_dir, exist_ok=True)
        error_path = os.path.join(output_dir, f"{base_filename}_error.json")
        with open(error_path, 'w') as f:
            json.dump(result, f, indent=4)
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics (with better serialization handling)
    metrics_path = os.path.join(output_dir, f"{base_filename}_metrics.json")
    
    def serialize_metric_value(obj):
        """Convert tensors and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {k: serialize_metric_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_metric_value(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    # Add additional metadata to metrics
    metrics_data = {
        "metrics": serialize_metric_value(result["metrics"]),
        "generation_info": {
            "num_views": result["rendered_images"].shape[0] if "rendered_images" in result else 0,
            "image_resolution": list(result["rendered_images"].shape[2:]) if "rendered_images" in result else [],
            "attention_maps_available": result.get("attention_maps_available", False),
            "attention_timesteps": result.get("attention_timesteps", []),
            "attention_layers": result.get("attention_layers", {})
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    # Save multi-view images with memory optimization
    try:
        rendered_images = result["rendered_images"]
        num_views = rendered_images.shape[0]
        
        # Convert to numpy with memory management
        images_np = rendered_images.cpu().numpy()  # (V, 3, H, W)
        
        # Free GPU memory immediately
        del rendered_images
        torch.cuda.empty_cache()
        
        # Convert from (V, 3, H, W) to (V, H, W, 3) and ensure uint8
        images_np = np.transpose(images_np, (0, 2, 3, 1))  # (V, H, W, 3)
        images_np = (images_np * 255).astype(np.uint8)
        
        # Save individual view images
        view_paths = []
        for view_idx in range(num_views):
            view_path = os.path.join(output_dir, f"{base_filename}_view_{view_idx:02d}.png")
            view_image = Image.fromarray(images_np[view_idx])
            view_image.save(view_path, optimize=True)
            view_paths.append(view_path)
        
        # Create multi-view montage (horizontal concatenation)
        try:
            # Concatenate all views horizontally
            montage_array = np.concatenate(images_np, axis=1)  # (H, V*W, 3)
            montage_path = os.path.join(output_dir, f"{base_filename}_multiview.png")
            montage_image = Image.fromarray(montage_array)
            montage_image.save(montage_path, optimize=True)
            
            # Also create a smaller preview montage
            preview_height = 128
            if montage_array.shape[0] > preview_height:
                ratio = preview_height / montage_array.shape[0]
                preview_width = int(montage_array.shape[1] * ratio)
                preview_montage = Image.fromarray(montage_array).resize((preview_width, preview_height), Image.Resampling.LANCZOS)
                preview_path = os.path.join(output_dir, f"{base_filename}_multiview_preview.png")
                preview_montage.save(preview_path, optimize=True)
                result["preview_montage_path"] = preview_path
            
            result["montage_path"] = montage_path
            
        except Exception as e:
            logger.warning(f"Could not create montage: {e}")
        
        # Clean up
        del images_np
        
        # Add saved paths to the result dict for logging
        result["view_paths"] = view_paths
        result["num_views_saved"] = len(view_paths)
        
        logger.info(f"Saved {len(view_paths)} individual views and montage for {base_filename}")
            
    except Exception as e:
        logger.error(f"Error saving images: {e}")
        # Save a placeholder file to indicate the attempt was made
        error_image_path = os.path.join(output_dir, f"{base_filename}_image_error.txt")
        with open(error_image_path, 'w') as f:
            f.write(f"Image saving failed: {str(e)}")
        result["image_error"] = str(e)


def multiview_tensor_to_images(rendered_images: torch.Tensor) -> List[Image.Image]:
    """
    Convert multi-view rendered images tensor to list of PIL Images.
    
    Args:
        rendered_images: Tensor of shape (V, 3, H, W) where V is number of views
        
    Returns:
        List of PIL Images, one for each view
    """
    try:
        # Convert tensor to numpy and ensure proper format
        images_np = rendered_images.cpu().numpy()  # (V, 3, H, W)
        images_np = np.transpose(images_np, (0, 2, 3, 1))  # (V, H, W, 3)
        images_np = (images_np * 255).astype(np.uint8)
        
        # Convert each view to PIL Image
        pil_images = []
        for view_idx in range(images_np.shape[0]):
            pil_image = Image.fromarray(images_np[view_idx])
            pil_images.append(pil_image)
            
        return pil_images
        
    except Exception as e:
        logger.error(f"Error converting multi-view tensor to images: {e}")
        return []


def cleanup_memory():
    """Aggressive memory cleanup."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()