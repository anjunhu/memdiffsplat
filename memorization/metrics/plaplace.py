# DiffSplat/memorization/metrics/plaplace.py

import torch
import torch.nn.functional as F
import math
from typing import Dict, Optional, Callable, Tuple
import numpy as np

from .base import BaseMetric

class PLaplaceMetric(BaseMetric):
    """
    p-Laplace Memorization Metric for DiffSplat.
    
    Replicates the core logic from /home/ubuntu/metrics/pLaplace by:
    1. Computing p-Laplace operator at latent points using boundary flux method
    2. Sampling sphere directions around latent centers
    3. Evaluating score function (∇ log p) at boundary points - this is directly
       the negative noise prediction from the diffusion model, NOT computed via autograd
    4. Computing flux through boundary as memorization indicator
    
    Key insight: The score function for diffusion models is the noise prediction
    (up to sign and scaling). No autograd or finite differences needed for the score -
    we just evaluate the UNet directly.
    
    Adapted for DiffSplat's multi-view 3D Gaussian Splatting pipeline.
    """
    
    def __init__(self, 
                 p: float = 1.0,
                 radius_factor: float = 0.1,
                 n_samples: int = 128,
                 timesteps_to_measure: Optional[list] = None):
        super().__init__()
        self.p = p
        self.radius_factor = radius_factor
        self.n_samples = n_samples
        self.timesteps_to_measure = timesteps_to_measure or [50, 100, 200, 500]
    
    @property
    def name(self) -> str:
        return f"pLaplace_p{self.p}_Metric"
        
    @property
    def metric_type(self) -> str:
        return "per_seed"
        
    @property
    def requires_model(self) -> bool:
        return True
        
    @property
    def requires_contexts(self) -> bool:
        return True

    @torch.no_grad()
    def measure(self, model = None, conditioning_context = None, 
                unconditioning_context = None, latents = None, **kwargs) -> Dict:
        """
        Measures memorization using p-Laplace analysis at latent points.
        
        No gradients needed - the score function is directly the UNet output.
        
        Args:
            model: The DiffSplat UNet model
            conditioning_context: Conditional text embeddings
            unconditioning_context: Unconditional text embeddings  
            latents: Latent tensors to analyze [B, C, H, W]
            
        Returns:
            Dict with p-Laplace scores at different timesteps
        """
        if model is None:
            return {"error": "PLaplaceMetric requires model"}
        
        if conditioning_context is None or unconditioning_context is None:
            return {"error": "PLaplaceMetric requires conditioning contexts"}
        
        if latents is None:
            return {"error": "PLaplaceMetric requires latents"}
        
        device = latents.device
        
        # Handle multi-view latents - process all views
        is_multiview = len(latents.shape) == 5  # [B, V, C, H, W]
        if is_multiview:
            batch_size, num_views = latents.shape[0], latents.shape[1]
            # Reshape to [B*V, C, H, W] to process all views
            latents = latents.reshape(-1, *latents.shape[2:])
        else:
            batch_size = latents.shape[0]
            num_views = 1
        
        # Extract context tensors
        cond_context = conditioning_context.get("context", conditioning_context)
        uncond_context = unconditioning_context.get("context", unconditioning_context)
        
        if isinstance(cond_context, (list, tuple)):
            cond_context = cond_context[0]
        if isinstance(uncond_context, (list, tuple)):
            uncond_context = uncond_context[0]
        
        plaplace_scores = {}
        
        for timestep in self.timesteps_to_measure:
            try:
                timestep_scores = []
                
                total_samples = latents.shape[0]  # B*V for multiview, B otherwise
                for i in range(total_samples):
                    latent_center = latents[i]  # [C, H, W]
                    
                    # Create gradient function for this timestep and context
                    def get_logp_gradients(points):
                        return self._compute_score_gradients(
                            model, points, timestep, cond_context, uncond_context
                        )
                    
                    # Compute p-Laplace using boundary flux method
                    plaplace_score = self._compute_p_laplace_boundary_torch(
                        center=latent_center,
                        radius_factor=self.radius_factor,
                        p=self.p,
                        get_logp_gradients=get_logp_gradients,
                        n_samples=self.n_samples
                    )
                    
                    timestep_scores.append(plaplace_score.item())
                
                # Average across all samples (batch * views)
                avg_score = sum(timestep_scores) / len(timestep_scores)
                plaplace_scores[f"t{timestep}"] = avg_score
                
                if is_multiview:
                    print(f"[PLaplaceMetric] t={timestep}: {avg_score:.6f} (averaged over {batch_size} batches × {num_views} views)")
                else:
                    print(f"[PLaplaceMetric] t={timestep}: {avg_score:.6f}")
                
            except Exception as e:
                print(f"[PLaplaceMetric] Error at timestep {timestep}: {e}")
                plaplace_scores[f"t{timestep}"] = 0.0
        
        # Compute aggregate metrics
        all_scores = [v for v in plaplace_scores.values() if v != 0.0]
        if all_scores:
            plaplace_scores["mean"] = sum(all_scores) / len(all_scores)
            plaplace_scores["max"] = max(all_scores)
            plaplace_scores["min"] = min(all_scores)
        else:
            plaplace_scores["mean"] = 0.0
            plaplace_scores["max"] = 0.0
            plaplace_scores["min"] = 0.0
        
        print(f"[PLaplaceMetric] Final scores: mean={plaplace_scores['mean']:.6f}")
        
        return plaplace_scores
    
    def _compute_score_gradients(self, model, points, timestep, cond_context, uncond_context):
        """
        Compute score function (∇ log p) at given points.
        
        For diffusion models, the score function is directly related to the noise prediction:
            score = -noise_pred / σ(t)
        
        This does NOT use autograd or finite differences - the score is the direct output 
        of the denoising network, matching the approach in /home/ubuntu/metrics/pLaplace.
        
        Args:
            model: UNet model
            points: Tensor of shape [N, C, H, W] - points to evaluate
            timestep: Diffusion timestep
            cond_context: Conditional context
            uncond_context: Unconditional context
            
        Returns:
            Score (gradient of log p) of same shape as points
        """
        return self._compute_score_at_points(model, points, timestep, cond_context, uncond_context)
    
    def _compute_score_at_points(self, model, points, timestep, cond_context, uncond_context):
        """
        Compute the score function (negative noise prediction) at given points.
        
        Args:
            model: UNet model
            points: Tensor of shape [N, C, H, W]
            timestep: Diffusion timestep
            cond_context: Conditional context
            uncond_context: Unconditional context
            
        Returns:
            Score tensor of shape [N, C, H, W]
        """
        n_points = points.shape[0]
        device = points.device
        original_channels = points.shape[1]
        
        # Prepare timestep tensor
        t = torch.full((n_points,), timestep, device=device, dtype=torch.long)
        
        # Prepare contexts for CFG
        if cond_context.shape[0] == 1:
            cond_context_expanded = cond_context.expand(n_points, -1, -1)
        else:
            cond_context_expanded = cond_context
        if uncond_context.shape[0] == 1:
            uncond_context_expanded = uncond_context.expand(n_points, -1, -1)
        else:
            uncond_context_expanded = uncond_context
        
        # Check if model expects additional input channels
        expected_in_channels = None
        if hasattr(model, 'config') and hasattr(model.config, 'in_channels'):
            expected_in_channels = model.config.in_channels
        elif hasattr(model, 'in_channels'):
            expected_in_channels = model.in_channels
        
        # Pad if necessary
        if expected_in_channels and points.shape[1] < expected_in_channels:
            padding_channels = expected_in_channels - points.shape[1]
            padding = torch.zeros(
                points.shape[0], padding_channels, 
                points.shape[2], points.shape[3],
                device=device, dtype=points.dtype
            )
            points_padded = torch.cat([points, padding], dim=1)
        else:
            points_padded = points
        
        # Concatenate for CFG
        context = torch.cat([uncond_context_expanded, cond_context_expanded], dim=0)
        points_cfg = torch.cat([points_padded, points_padded], dim=0)
        t_cfg = torch.cat([t, t], dim=0)
        
        # Forward pass through UNet
        with torch.no_grad():
            if hasattr(model, 'unet'):
                output = model.unet(points_cfg, t_cfg, encoder_hidden_states=context)
                if hasattr(output, 'sample'):
                    noise_pred = output.sample
                else:
                    noise_pred = output
            elif hasattr(model, 'model'):
                noise_pred = model.model(points_cfg, t_cfg, context)
            else:
                output = model(points_cfg, t_cfg, encoder_hidden_states=context)
                if hasattr(output, 'sample'):
                    noise_pred = output.sample
                else:
                    noise_pred = output
            
            # Split CFG predictions
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            
            # Apply CFG
            guidance_scale = 7.5
            noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Only keep original channels
            if noise_pred_guided.shape[1] > original_channels:
                noise_pred_guided = noise_pred_guided[:, :original_channels]
            
            # Score = -noise_pred
            score = -noise_pred_guided
        
        return score
    
    def _sample_sphere_normals_nd_torch(self, tensor_shape, n_samples=128, device="cuda", epsilon=1e-12):
        """
        Returns n_samples random directions on the unit sphere in R^(product of tensor_shape).
        The result has shape (n_samples, *tensor_shape).
        
        Replicates pLaplace core function.
        """
        # 1) Draw Gaussian noise: shape (n_samples, *tensor_shape)
        x = torch.randn((n_samples, *tensor_shape), device=device)
        # 2) Flatten to (n_samples, -1)
        x_flat = x.view(n_samples, -1)
        D = x_flat.shape[1]
        sqrt_d = math.sqrt(D)
        # 3) Compute norms per sample
        norms = x_flat.norm(dim=1, keepdim=True).clamp(min=epsilon)
        # 4) Divide each sample by its norm to get unit vectors
        x_unit_flat = x_flat / norms
        # 5) Reshape back to (n_samples, *tensor_shape)
        x_unit = x_unit_flat.view_as(x)
        return x_unit, sqrt_d
    
    def _compute_p_laplace_boundary_torch(self, center, radius_factor, p, get_logp_gradients, n_samples=128, epsilon=1e-12):
        """
        Monte-Carlo approximation of the p-Laplace at 'center' using the boundary approach.
        
        Replicates pLaplace core function for torch tensors.
        
        Args:
            center: shape (*latent_dims). Typically (C, H, W) for latents.
            radius_factor: float or torch scalar
            p: 1, 2, etc.
            get_logp_gradients: function from (N, *latent_dims) -> (N, *latent_dims), in Torch
            n_samples: number of boundary samples
            epsilon: small constant to avoid division by zero
        """
        device = center.device
        
        # 1) Sample random unit normals in the same dimension as 'center' shape => (n_samples, *center.shape)
        normals, sqrt_d = self._sample_sphere_normals_nd_torch(
            tensor_shape=center.shape,
            n_samples=n_samples,
            device=device,
            epsilon=epsilon,
        )
        
        # 2) Points on boundary: broadcast center => (n_samples, *center.shape)
        #    shape => (n_samples, *latent_dims)
        points_on_sphere = center.unsqueeze(0) + radius_factor * sqrt_d * normals.to(center.dtype)
        
        # 3) Evaluate gradient at each boundary point
        #    shape => (n_samples, *latent_dims)
        grads = get_logp_gradients(points_on_sphere)
        
        n_samples_ = grads.shape[0]  # grads is (n_samples, *latent_dims)
        grads_flat = grads.view(n_samples_, -1).to(torch.float32)  # Use float32 for stability
        normals_flat = sqrt_d * normals.view(n_samples_, -1).to(torch.float32)
        
        # Dot products of each pair
        dot_vals = torch.sum(grads_flat * normals_flat, dim=1)
        
        # Norm of each gradient, norm of each normal
        norm_grads = grads_flat.norm(dim=1)
        norm_normals = normals_flat.norm(dim=1)
        
        denom = (norm_grads * norm_normals).clamp(min=1e-8)
        flux_vals = dot_vals / denom  # shape (n_samples,)
        
        if abs(p - 1.0) > 1e-9:  # p != 1
            scale = norm_grads ** (p - 2)
            flux_vals *= scale
        
        return flux_vals.mean()
    
    def _compute_p_laplace_volume(self, center, radius, p, get_logp_gradients, n_samples=128, delta=1e-3):
        """
        Alternative volumetric approach for p-Laplace computation.
        
        Approximates p-Laplace by volumetric sampling and numerical divergence:
        1) sample points inside the ball
        2) do finite-difference to estimate div( ||grad||^(p-2)*grad ) for each
        3) average
        """
        device = center.device
        center_flat = center.flatten()
        dim = len(center_flat)
        
        # Sample points in ball
        points_in_ball = self._sample_ball_points(dim, radius, n_samples)
        points_in_ball = torch.from_numpy(points_in_ball).float().to(device)
        
        values = []
        for pt in points_in_ball:
            actual_pt = pt + center_flat  # shift by center
            actual_pt_shaped = actual_pt.reshape_as(center).unsqueeze(0)
            
            val = self._numeric_p_lap_at_point(actual_pt_shaped, p, get_logp_gradients, delta=delta)
            values.append(val)
        
        return torch.tensor(values).mean() if values else torch.tensor(0.0)
    
    def _sample_ball_points(self, dimension, radius, n_samples):
        """
        Samples n_samples points uniformly in a 'dimension'-dim ball of radius 'radius'.
        Returns an (n_samples x dimension) NumPy array.
        """
        dirs = np.random.randn(n_samples, dimension)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        
        # radius^(1/d) approach
        u = np.random.rand(n_samples)
        r = radius * (u ** (1.0 / dimension))
        
        return dirs * r[:, None]
    
    def _numeric_p_lap_at_point(self, point, p, get_logp_gradients, delta=1e-3, epsilon=1e-8):
        """
        Numerically approximates p-laplace = div(||grad||^{p-2} * grad) at a single point
        via finite differences.
        """
        point = point.squeeze(0)  # Remove batch dimension
        original_shape = point.shape
        point_flat = point.flatten()
        dim = len(point_flat)
        
        def p_field(x_flat):
            x_shaped = x_flat.reshape(original_shape).unsqueeze(0)
            grad = get_logp_gradients(x_shaped).squeeze(0).flatten()
            norm_grad = grad.norm() + epsilon
            factor = norm_grad ** (p - 2)
            return factor * grad
        
        # finite-differences for divergence
        div_val = 0.0
        for i in range(dim):
            e_i = torch.zeros_like(point_flat)
            e_i[i] = 1.0
            
            f_plus = p_field(point_flat + delta * e_i)
            f_minus = p_field(point_flat - delta * e_i)
            div_val += (f_plus[i] - f_minus[i]) / (2 * delta)
        
        return div_val.item()