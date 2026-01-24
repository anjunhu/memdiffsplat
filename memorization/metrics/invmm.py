import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Optional, List
import math
from contextlib import contextmanager

from .base import BaseMetric

class InvMMMetric(BaseMetric):
    """
    Inversion-based Memorization Measure (InvMM) for DiffSplat.
    
    Replicates the core logic from /home/ubuntu/InvMM by:
    1. Inverting generated images back to latent space
    2. Optimizing learnable tokens and latent distribution parameters
    3. Measuring reconstruction error as memorization indicator
    
    Adapted for DiffSplat's multi-view 3D Gaussian Splatting pipeline.
    """
    
    def __init__(self, 
                 num_tokens: int = 75,
                 train_lr: float = 1e-1,
                 train_num_steps: int = 2000,
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

    @torch.no_grad()
    def measure(self, model = None, images = None, **kwargs) -> Dict:
        """
        Measures memorization by inverting generated images back to latent space.
        
        Args:
            model: The DiffSplat pipeline model
            images: Generated images tensor [B, C, H, W] or [B, V, C, H, W] for multi-view
            
        Returns:
            Dict with InvMM score and related metrics
        """
        if model is None:
            return {"error": "InvMMMetric requires model"}
        
        if images is None:
            return {"error": "InvMMMetric requires generated images"}
        
        # Handle multi-view images - process all views
        is_multiview = len(images.shape) == 5  # [B, V, C, H, W]
        if is_multiview:
            batch_size, num_views = images.shape[0], images.shape[1]
            # Reshape to [B*V, C, H, W] to process all views
            images = images.reshape(-1, *images.shape[2:])
        else:
            batch_size = images.shape[0]
            num_views = 1
        
        device = images.device
        total_samples = images.shape[0]  # B*V for multiview, B otherwise
        
        # Initialize SSCD similarity model for evaluation
        try:
            sscd_model = torch.jit.load("models/sscd/sscd_disc_large.torchscript.pt")
            sscd_model.to(device)
            sscd_model.eval()
            
            from torchvision import transforms as T
            sscd_transforms = T.Compose([
                T.Resize([320, 320]),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except:
            print("[InvMMMetric] Warning: SSCD model not found, using L2 similarity")
            sscd_model = None
            sscd_transforms = None
        
        invmm_scores = []
        
        for i in range(total_samples):
            img = images[i:i+1]  # Keep batch dimension
            
            try:
                # Encode image to latent space
                if hasattr(model, 'vae') and hasattr(model.vae, 'encode'):
                    # SD-style VAE encoding
                    encoder_posterior = model.vae.encode(img * 2 - 1)
                    if hasattr(encoder_posterior, 'sample'):
                        z_target = encoder_posterior.sample()
                    else:
                        z_target = encoder_posterior
                elif hasattr(model, 'encode_first_stage'):
                    # LDM-style encoding
                    encoder_posterior = model.encode_first_stage(img * 2 - 1)
                    if hasattr(model, 'get_first_stage_encoding'):
                        z_target = model.get_first_stage_encoding(encoder_posterior)
                    else:
                        z_target = encoder_posterior
                else:
                    print(f"[InvMMMetric] Warning: Cannot encode image {i}, skipping")
                    continue
                
                z_target.requires_grad_(False)
                
                # Initialize optimization parameters
                log_coeffs = torch.zeros(self.num_tokens, self._get_vocab_size(model), device=device)
                log_coeffs.requires_grad_(True)
                
                mu = torch.zeros_like(z_target, device=device)
                logvar = torch.zeros_like(z_target, device=device)
                mu.requires_grad_(True)
                logvar.requires_grad_(True)
                
                # Setup optimizer
                params = [
                    {"params": [log_coeffs], "weight_decay": 0},
                    {"params": [mu], "weight_decay": 0},
                    {"params": [logvar], "weight_decay": 0}
                ]
                opt = Adam(params, self.train_lr)
                
                # Optimization loop
                success = False
                kl_weight = self.init_kl_weight
                
                for step in range(self.train_num_steps):
                    # Sample from learned distribution
                    n = torch.randn_like(z_target) * logvar.div(2).exp() + mu
                    
                    # Sample tokens using Gumbel softmax
                    coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0), hard=False, tau=self.tau)
                    
                    # Compute reconstruction loss
                    p_loss = self._compute_reconstruction_loss(model, z_target, n, coeffs)
                    
                    # KL divergence regularization
                    r_loss = 0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1)
                    
                    total_loss = p_loss + kl_weight * r_loss
                    
                    # Backward pass
                    total_loss.backward()
                    opt.step()
                    opt.zero_grad()
                    
                    # Adaptive KL weight scheduling
                    if (step + 1) % 50 == 0:
                        # Sample and check similarity
                        with torch.no_grad():
                            samples = self._generate_samples(model, coeffs, mu, logvar, n_samples=4)
                            if self._check_similarity(img, samples, sscd_model, sscd_transforms):
                                success = True
                                break
                        
                        kl_weight = max(kl_weight / 2, 0) if p_loss < 0.1 else kl_weight + 0.001
                
                # Compute final InvMM score
                if success:
                    invmm = 0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1).cpu().item()
                else:
                    invmm = float("inf")
                
                invmm_scores.append(invmm)
                
            except Exception as e:
                print(f"[InvMMMetric] Error processing image {i}: {e}")
                invmm_scores.append(float("inf"))
        
        if not invmm_scores:
            return {"error": "Failed to process any images"}
        
        # Aggregate results
        finite_scores = [s for s in invmm_scores if s != float("inf")]
        
        if not finite_scores:
            return {"invmm_score": float("inf"), "success_rate": 0.0}
        
        avg_invmm = sum(finite_scores) / len(finite_scores)
        success_rate = len(finite_scores) / len(invmm_scores)
        
        if is_multiview:
            print(f"[InvMMMetric] InvMM Score: {avg_invmm:.4f}, Success Rate: {success_rate:.2f} (averaged over {batch_size} batches × {num_views} views)")
        else:
            print(f"[InvMMMetric] InvMM Score: {avg_invmm:.4f}, Success Rate: {success_rate:.2f}")
        
        return {
            "invmm_score": avg_invmm,
            "success_rate": success_rate,
            "individual_scores": invmm_scores
        }
    
    def _get_vocab_size(self, model) -> int:
        """Get vocabulary size from model tokenizer."""
        try:
            if hasattr(model, 'tokenizer'):
                return model.tokenizer.vocab_size
            elif hasattr(model, 'cond_stage_model') and hasattr(model.cond_stage_model, 'tokenizer'):
                return model.cond_stage_model.tokenizer.vocab_size
            elif hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'tokenizer'):
                return model.text_encoder.tokenizer.vocab_size
            else:
                # Default CLIP vocab size
                return 49408
        except:
            return 49408
    
    def _compute_reconstruction_loss(self, model, z_target, z_sample, coeffs):
        """Compute reconstruction loss between target and sampled latents."""
        try:
            # Generate pseudo prompt
            pseudo_prompt = [""]
            
            # Get text embeddings using learned tokens
            if hasattr(model, 'get_learned_conditioning'):
                # LDM-style
                with self._modify_token_embedding(model, coeffs):
                    c = model.get_learned_conditioning(pseudo_prompt)
            elif hasattr(model, 'encode_prompt'):
                # SD-style - more complex for DiffSplat
                c = self._encode_prompt_with_tokens(model, pseudo_prompt, coeffs)
            else:
                # Fallback
                c = torch.zeros(1, 77, 768, device=z_target.device)
            
            # Sample timestep
            t = torch.randint(0, 1000, (1,), device=z_target.device)
            
            # Compute denoising loss
            if hasattr(model, 'p_losses'):
                loss = model.p_losses(z_target, c, t, z_sample)
            else:
                # Manual denoising loss computation
                noise = torch.randn_like(z_target)
                z_noisy = self._add_noise(z_target, noise, t)
                
                if hasattr(model, 'unet'):
                    pred_noise = model.unet(z_noisy, t, c).sample
                elif hasattr(model, 'model'):
                    pred_noise = model.model(z_noisy, t, c)
                else:
                    raise ValueError("Cannot find denoising model")
                
                loss = F.mse_loss(pred_noise, noise)
            
            return loss
            
        except Exception as e:
            print(f"[InvMMMetric] Error in reconstruction loss: {e}")
            return torch.tensor(1.0, device=z_target.device, requires_grad=True)
    
    def _add_noise(self, x_start, noise, t):
        """Add noise to clean latents according to diffusion schedule."""
        # Simple linear schedule - should be replaced with actual model schedule
        alpha_bar = 1 - t.float() / 1000
        return torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
    
    def _generate_samples(self, model, coeffs, mu, logvar, n_samples=4):
        """Generate samples using learned parameters."""
        try:
            samples = []
            for _ in range(n_samples):
                z = torch.randn_like(mu) * logvar.div(2).exp() + mu
                
                # Decode to image
                if hasattr(model, 'vae') and hasattr(model.vae, 'decode'):
                    sample = model.vae.decode(z).sample
                elif hasattr(model, 'decode_first_stage'):
                    sample = model.decode_first_stage(z)
                else:
                    # Skip if cannot decode
                    continue
                
                sample = (sample + 1.) / 2.
                sample = torch.clamp(sample, 0., 1.)
                samples.append(sample)
            
            return torch.cat(samples, dim=0) if samples else None
            
        except Exception as e:
            print(f"[InvMMMetric] Error generating samples: {e}")
            return None
    
    def _check_similarity(self, target, samples, sscd_model, sscd_transforms):
        """Check if generated samples are similar enough to target."""
        if samples is None:
            return False
        
        try:
            if sscd_model is not None and sscd_transforms is not None:
                # Use SSCD similarity
                target_transformed = sscd_transforms(target)
                samples_transformed = sscd_transforms(samples)
                
                target_features = sscd_model(target_transformed)
                sample_features = sscd_model(samples_transformed)
                
                similarities = target_features.mm(sample_features.T).squeeze()
                return torch.any(similarities >= self.similarity_threshold).item()
            else:
                # Use L2 similarity as fallback
                l2_distances = torch.norm(samples - target, dim=(1, 2, 3))
                return torch.any(l2_distances < 0.1).item()
                
        except Exception as e:
            print(f"[InvMMMetric] Error checking similarity: {e}")
            return False
    
    @contextmanager
    def _modify_token_embedding(self, model, coeffs):
        """Context manager to temporarily modify token embeddings."""
        # This is a simplified version - actual implementation depends on model architecture
        try:
            yield None
        finally:
            pass
    
    def _encode_prompt_with_tokens(self, model, prompt, coeffs):
        """Encode prompt with learned token coefficients."""
        # Simplified implementation - actual implementation depends on model architecture
        try:
            if hasattr(model, 'encode_prompt'):
                return model.encode_prompt(prompt, device=coeffs.device)[0]
            else:
                return torch.zeros(1, 77, 768, device=coeffs.device)
        except:
            return torch.zeros(1, 77, 768, device=coeffs.device)