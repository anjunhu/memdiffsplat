import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Optional, List, Any
import math
from contextlib import contextmanager

from .base import BaseMetric

class InvMMMetric(BaseMetric):
    """
    Inversion-based Memorization Measure (InvMM) for DiffSplat.
    
    Adapted for DiffSplat's multi-view 3D Gaussian Splatting pipeline using gsvae.
    
    The key insight is that DiffSplat uses gsvae.get_gslatents() to encode multi-view
    images into GS latents, which can then be used for the inversion process.
    """
    
    def __init__(self, 
                 num_tokens: int = 75,
                 train_lr: float = 1e-1,
                 train_num_steps: int = 500,  # Reduced for efficiency
                 tau: float = 2.0,
                 init_kl_weight: float = 1.0,
                 similarity_threshold: float = 0.5):
        super().__init__()
        self.num_tokens = num_tokens
        self.train_lr = train_lr
        self.train_num_steps = train_num_steps
        self.tau = tau
        self.init_kl_weight = init_kl_weight
        self.similarity_threshold = similarity_threshold
    
    @property
    def name(self) -> str:
        return "InvMM_Score"
        
    @property
    def metric_type(self) -> str:
        return "per_seed"
        
    @property
    def requires_model(self) -> bool:
        return True

    def measure(self, model=None, images=None, gsvae=None, gsrecon=None, 
                camera_params=None, **kwargs) -> Dict:
        """
        Measures memorization by inverting generated images back to GS latent space.
        
        For DiffSplat, we use gsvae.get_gslatents() to encode rendered multi-view 
        images back to GS latents, then measure the KL divergence of the learned
        latent distribution.
        
        Args:
            model: The DiffSplat pipeline
            images: Rendered images tensor [B, V, C, H, W] or [V, C, H, W]
            gsvae: DiffSplat's Gaussian Splatting VAE
            gsrecon: DiffSplat's GS reconstruction module
            camera_params: Camera parameters dict with input_C2W, input_fxfycxcy
            
        Returns:
            Dict with InvMM score and related metrics
        """
        if model is None:
            return {"error": "InvMMMetric requires model"}
        
        if images is None:
            return {"error": "InvMMMetric requires generated images"}
        
        # Check if we have DiffSplat components
        if gsvae is not None and gsrecon is not None and camera_params is not None:
            return self._measure_diffsplat(model, images, gsvae, gsrecon, camera_params, **kwargs)
        
        # Fallback to standard SD InvMM (for non-DiffSplat models)
        return self._measure_standard(model, images, **kwargs)
    
    def _measure_diffsplat(self, model, images, gsvae, gsrecon, camera_params, **kwargs) -> Dict:
        """
        DiffSplat-specific InvMM using latent space analysis.
        
        Since DiffSplat's gsrecon expects additional channels (normals, coords) that aren't
        available from rendered images, we use a latent-space approach instead:
        1. If latents are provided, analyze their distribution directly
        2. Measure how well the latents can be approximated by a simple Gaussian
        3. Higher KL divergence indicates more complex/memorized patterns
        """
        device = images.device
        
        # Check if latents were passed directly (preferred for DiffSplat)
        latents = kwargs.get('latents')
        
        # Debug: Log what we received
        print(f"[InvMMMetric] kwargs keys: {list(kwargs.keys())}")
        print(f"[InvMMMetric] latents is None: {latents is None}")
        if latents is not None:
            print(f"[InvMMMetric] latents type: {type(latents)}, shape: {latents.shape if hasattr(latents, 'shape') else 'N/A'}")
        
        if latents is not None and hasattr(latents, 'shape') and latents.numel() > 0:
            print(f"[InvMMMetric] Using provided latents, shape: {latents.shape}")
            return self._measure_from_latents(latents, device)
        
        # Fallback: Try to use gsvae encoding (may fail if model expects normals/coords)
        # Get model dtype from gsvae's VAE
        model_dtype = next(gsvae.vae.parameters()).dtype
        print(f"[InvMMMetric] Model dtype: {model_dtype}, images dtype: {images.dtype}")
        
        # Handle image dimensions - ensure [B, V, C, H, W] format for gsvae
        if len(images.shape) == 5:  # [B, V, C, H, W]
            imgs_for_encode = images
        elif len(images.shape) == 4:  # [V, C, H, W]
            imgs_for_encode = images.unsqueeze(0)  # [1, V, C, H, W]
        else:
            return {
                "skipped": True,
                "reason": f"Unexpected image shape: {images.shape}",
                "note": "Expected [B, V, C, H, W] or [V, C, H, W]"
            }
        
        num_views = imgs_for_encode.shape[1]
        num_channels = imgs_for_encode.shape[2]
        
        # Check if model expects additional channels
        expected_channels = 3  # RGB
        if hasattr(gsvae, 'opt'):
            if getattr(gsvae.opt, 'input_normal', False):
                expected_channels += 3
            if getattr(gsvae.opt, 'input_coord', False):
                expected_channels += 3
            if getattr(gsvae.opt, 'input_mr', False):
                expected_channels += 2
        
        if num_channels < expected_channels:
            print(f"[InvMMMetric] Model expects {expected_channels} channels but got {num_channels}")
            print(f"[InvMMMetric] Cannot re-encode rendered images without normals/coords")
            return {
                "skipped": True,
                "reason": f"Model expects {expected_channels} input channels, got {num_channels}",
                "note": "DiffSplat InvMM requires latents to be passed directly, or full input channels"
            }
        
        # Get camera parameters
        if camera_params is None:
            return {
                "skipped": True,
                "reason": "Missing camera parameters for DiffSplat InvMM",
                "note": "Need camera_params dict with input_C2W and input_fxfycxcy"
            }
        
        input_C2W = camera_params.get('input_C2W')
        input_fxfycxcy = camera_params.get('input_fxfycxcy')
        
        if input_C2W is None or input_fxfycxcy is None:
            return {
                "skipped": True,
                "reason": "Missing camera parameters for DiffSplat InvMM",
                "note": "Need input_C2W and input_fxfycxcy in camera_params"
            }
        
        try:
            # Clone and ensure proper dimensions for camera params
            input_C2W = input_C2W.clone()
            input_fxfycxcy = input_fxfycxcy.clone()
            
            # input_C2W comes as [V, 4, 4], need [B, V, 4, 4]
            if len(input_C2W.shape) == 3:
                input_C2W = input_C2W.unsqueeze(0)  # [1, V, 4, 4]
            
            # input_fxfycxcy comes as [V, 4], need [B, V, 4]  
            if len(input_fxfycxcy.shape) == 2:
                input_fxfycxcy = input_fxfycxcy.unsqueeze(0)  # [1, V, 4]
            
            # Convert all inputs to model dtype
            imgs_for_encode = imgs_for_encode.to(dtype=model_dtype)
            input_C2W = input_C2W.to(dtype=model_dtype)
            input_fxfycxcy = input_fxfycxcy.to(dtype=model_dtype)
            
            print(f"[InvMMMetric] Encoding shapes - images: {imgs_for_encode.shape}, C2W: {input_C2W.shape}, fxfycxcy: {input_fxfycxcy.shape}")
            print(f"[InvMMMetric] Dtypes - images: {imgs_for_encode.dtype}, C2W: {input_C2W.dtype}, fxfycxcy: {input_fxfycxcy.dtype}")
            
            # Get target GS latents from rendered images
            with torch.no_grad():
                z_target = gsvae.get_gslatents(
                    gsrecon, 
                    imgs_for_encode, 
                    input_C2W, 
                    input_fxfycxcy
                )
                
                # Apply scaling
                z_target = gsvae.scaling_factor * (z_target - gsvae.shift_factor)
            
            return self._measure_from_latents(z_target, device)
            
        except Exception as e:
            print(f"[InvMMMetric] Error in DiffSplat InvMM: {e}")
            import traceback
            traceback.print_exc()
            return {
                "skipped": True,
                "reason": f"DiffSplat InvMM failed: {str(e)}",
                "note": "Error during GS latent encoding. This typically happens when latents are not passed correctly. "
                        "Ensure the evaluator passes 'latents' in kwargs. "
                        f"Received kwargs keys: {list(kwargs.keys())}"
            }
    
    def _measure_from_latents(self, z_target, device) -> Dict:
        """
        Measure InvMM score from latent representations.
        
        The approach:
        1. Optimize learnable parameters to match the target latents
        2. Measure KL divergence as memorization indicator
        """
        print(f"[InvMMMetric] Measuring from latents, shape: {z_target.shape}, dtype: {z_target.dtype}")
        
        # Convert to float32 for optimization stability
        z_target_f32 = z_target.detach().float()
        
        # Initialize learnable latent distribution parameters (in float32)
        mu = torch.zeros_like(z_target_f32, device=device, requires_grad=True)
        logvar = torch.zeros_like(z_target_f32, device=device, requires_grad=True)
        
        # Setup optimizer
        opt = Adam([mu, logvar], lr=self.train_lr)
        
        # Optimization loop - minimize reconstruction + KL
        kl_weight = self.init_kl_weight
        best_recon_loss = float('inf')
        
        for step in range(self.train_num_steps):
            opt.zero_grad()
            
            # Sample from learned distribution
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_sample = mu + eps * std
            
            # Reconstruction loss (MSE to target latents in float32)
            recon_loss = F.mse_loss(z_sample, z_target_f32)
            
            # KL divergence: KL(q(z|x) || N(0,1))
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Total loss
            total_loss = recon_loss + kl_weight * kl_loss
            
            # Backward (need to enable grad temporarily)
            with torch.enable_grad():
                total_loss.backward()
                opt.step()
            
            if recon_loss.item() < best_recon_loss:
                best_recon_loss = recon_loss.item()
            
            # Adaptive KL weight
            if (step + 1) % 100 == 0:
                if recon_loss.item() < 0.01:
                    kl_weight = max(kl_weight * 0.5, 0.001)
                else:
                    kl_weight = min(kl_weight * 1.1, 10.0)
        
        # Final InvMM score is the KL divergence of the learned distribution
        with torch.no_grad():
            final_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()
            final_recon = F.mse_loss(mu, z_target_f32).item()
        
        print(f"[InvMMMetric] DiffSplat InvMM - KL: {final_kl:.4f}, Recon: {final_recon:.6f}")
        
        return {
            "invmm_score": final_kl,
            "reconstruction_loss": final_recon,
            "success_rate": 1.0 if final_recon < 0.1 else 0.0,
            "method": "diffsplat_latent"
        }
    
    def _measure_standard(self, model, images, **kwargs) -> Dict:
        """
        Standard InvMM for regular SD models (fallback).
        """
        # Check if UNet expects non-standard input channels
        unet = model.unet if hasattr(model, 'unet') else model
        if hasattr(unet, 'config') and hasattr(unet.config, 'in_channels'):
            in_channels = unet.config.in_channels
            if in_channels != 4:
                return {
                    "skipped": True,
                    "reason": f"UNet expects {in_channels} input channels, not standard 4",
                    "note": "Standard InvMM only works with 4-channel latent models"
                }
        
        # Try to find VAE
        vae = None
        if hasattr(model, 'vae'):
            vae = model.vae
        elif hasattr(model, 'first_stage_model'):
            vae = model.first_stage_model
        
        if vae is None:
            return {
                "skipped": True,
                "reason": "Cannot find VAE in model",
                "note": "Standard InvMMMetric requires VAE for image encoding"
            }
        
        # Handle multi-view images
        is_multiview = len(images.shape) == 5
        if is_multiview:
            batch_size, num_views = images.shape[0], images.shape[1]
            images = images.reshape(-1, *images.shape[2:])
        else:
            batch_size = images.shape[0]
            num_views = 1
        
        device = images.device
        total_samples = images.shape[0]
        
        # Get VAE dtype and convert images
        vae_dtype = next(vae.parameters()).dtype
        print(f"[InvMMMetric] VAE dtype: {vae_dtype}, images dtype: {images.dtype}")
        
        invmm_scores = []
        
        for i in range(total_samples):
            img = images[i:i+1].to(dtype=vae_dtype)  # Match VAE dtype
            
            # Encode with VAE
            img_normalized = img * 2 - 1
            encoder_posterior = vae.encode(img_normalized)
            if hasattr(encoder_posterior, 'latent_dist'):
                z_target = encoder_posterior.latent_dist.sample()
            elif hasattr(encoder_posterior, 'sample'):
                z_target = encoder_posterior.sample()
            else:
                z_target = encoder_posterior
            
            # Simple KL-based score (simplified InvMM)
            # Use float32 for optimization stability
            z_target_f32 = z_target.detach().float()
            mu = torch.zeros_like(z_target_f32, requires_grad=True)
            logvar = torch.zeros_like(z_target_f32, requires_grad=True)
            
            opt = Adam([mu, logvar], lr=self.train_lr)
            
            for step in range(self.train_num_steps):
                opt.zero_grad()
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z_sample = mu + eps * std
                
                recon_loss = F.mse_loss(z_sample, z_target_f32)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = recon_loss + self.init_kl_weight * kl_loss
                with torch.enable_grad():
                    loss.backward()
                    opt.step()
            
            with torch.no_grad():
                invmm = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()
            
            invmm_scores.append(invmm)
        
        finite_scores = [s for s in invmm_scores if s != float("inf")]
        
        if not finite_scores:
            return {"invmm_score": float("inf"), "success_rate": 0.0}
        
        avg_invmm = sum(finite_scores) / len(finite_scores)
        success_rate = len(finite_scores) / len(invmm_scores)
        
        return {
            "invmm_score": avg_invmm,
            "success_rate": success_rate,
            "method": "standard_vae"
        }
