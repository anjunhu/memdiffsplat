# DiffSplat/measure_assumptions.py

import torch
import numpy as np
import pandas as pd
import os
import sys
import logging
import argparse
import gc
import time
import json
import re
from tqdm import tqdm
import torch.nn.functional as F
from scipy.stats import kstest, pearsonr, spearmanr, wasserstein_distance, entropy
import warnings
from typing import Dict, List, Tuple, Optional
from omegaconf import OmegaConf

warnings.filterwarnings('ignore')

# --- Path Setup ---
sys.path.append(os.path.join(os.path.dirname(__file__)))

# --- DiffSplat Imports ---
from extensions.diffusers_diffsplat import UNetMV2DConditionModel, StableMVDiffusionPipeline
from src.models import GSAutoencoderKL, GSRecon
from src.options import opt_dict
import src.utils.util as util
import src.utils.geo_util as geo_util

from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from diffusers.models.autoencoders import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

# --- Memorization Framework Imports ---
from memorization.data.dataloaders import DatasetManager
from memorization.metrics.hessian import HessianMetric
from memorization.metrics.noisediffnorm import NoiseDiffNormMetric


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
N_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
NUM_VIEWS = 4  # DiffSplat multi-view
HESSIAN_STEPS_TO_MEASURE = list(range(0, N_INFERENCE_STEPS, 5))
OUTPUT_DIR = "diffsplat_geometric_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def setup_camera_parameters(opt, device, num_views=4, elevation=10.0, distance=1.4):
    """Setup camera parameters for multi-view generation (from run_baseline.py)."""
    
    # Get camera intrinsics
    fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5], device=device).float()
    
    # Setup camera poses for multi-view generation
    elevations = torch.tensor([-elevation] * num_views, device=device).deg2rad().float()
    azimuths = torch.tensor([0., 90., 180., 270.][:num_views], device=device).deg2rad().float()
    radius = torch.tensor([distance] * num_views, device=device).float()
    
    input_C2W = geo_util.orbit_camera(elevations, azimuths, radius, is_degree=False)  # (V_in, 4, 4)
    input_C2W[:, :3, 1:3] *= -1  # OpenGL -> OpenCV
    input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(input_C2W.shape[0], 1)  # (V_in, 4)
    
    # Get plucker embeddings if needed
    plucker = None
    if opt.input_concat_plucker:
        H = W = opt.input_res
        plucker, _ = geo_util.plucker_ray(H, W, input_C2W.unsqueeze(0), input_fxfycxcy.unsqueeze(0))
        plucker = plucker.squeeze(0)  # (V_in, 6, H, W)
        if opt.view_concat_condition:
            plucker = torch.cat([plucker[0:1, ...], plucker], dim=0)  # (V_in+1, 6, H, W)
    
    return {
        'input_C2W': input_C2W,
        'input_fxfycxcy': input_fxfycxcy, 
        'plucker': plucker,
        'num_views': num_views,
        'elevation': elevation,
        'distance': distance,
        'fxfycxcy': fxfycxcy
    }

def load_diffsplat_pipeline(config_file, device, ckpt_iter=13020):
    """Load DiffSplat pipeline (adapted from run_baseline.py)."""
    
    # Parse the config file and options
    configs = util.get_configs(config_file, [])
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)
    
    # Initialize UNet
    in_channels = 4  # hard-coded for SD 1.5/2.1
    if opt.input_concat_plucker:
        in_channels += 6
    if opt.input_concat_binary_mask:
        in_channels += 1
        
    unet_from_pretrained_kwargs = {
        "sample_size": opt.input_res // 8,  # `8` hard-coded for SD 1.5/2.1
        "in_channels": in_channels,
        "zero_init_conv_in": opt.zero_init_conv_in,
        "view_concat_condition": opt.view_concat_condition,
        "input_concat_plucker": opt.input_concat_plucker,
        "input_concat_binary_mask": opt.input_concat_binary_mask,
    }
    
    # Load base models from opt.pretrained_model_name_or_path (from config)
    tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16")
    vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")
    
    # Load custom models
    gsvae = GSAutoencoderKL(opt)
    gsrecon = GSRecon(opt)
    
    # Load scheduler
    noise_scheduler = DDIMScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
        
    if opt.common_tricks:
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config.rescale_betas_zero_snr = True
    if opt.prediction_type is not None:
        noise_scheduler.config.prediction_type = opt.prediction_type
    if opt.beta_schedule is not None:
        noise_scheduler.config.beta_schedule = opt.beta_schedule

    # Set up checkpoint directory structure (from config defaults)
    exp_dir = "out/gsdiff_gobj83k_sd15__render"  # Default from configs
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    # Load UNet checkpoint
    logger.info(f"Loading UNet from checkpoint: {ckpt_dir}")
    path = os.path.join(ckpt_dir, f"{ckpt_iter:06d}")
        
    assert os.path.exists(path), f"Checkpoint path not found: {path}"
    
    # Load UNet
    os.system(f"python3 extensions/merge_safetensors.py {path}/unet_ema")  # merge safetensors for loading
    unet, loading_info = UNetMV2DConditionModel.from_pretrained_new(
        path, subfolder="unet_ema",
        low_cpu_mem_usage=False, 
        ignore_mismatched_sizes=True, 
        output_loading_info=True, 
        **unet_from_pretrained_kwargs
    )
    
    for key in loading_info.keys():
        assert len(loading_info[key]) == 0  # no missing_keys, unexpected_keys, mismatched_keys, error_msgs
    
    # Load pretrained GSVAE and GSRecon (using default paths from config)
    logger.info("Loading GSVAE and GSRecon")
    gsvae = util.load_ckpt(
        os.path.join("out", "gsvae_gobj265k_sd", "checkpoints"),
        -1,  # Latest checkpoint
        None,
        gsvae,
    )
    
    gsrecon = util.load_ckpt(
        os.path.join("out", "gsrecon_gobj265k_cnp_even4", "checkpoints"),
        -1,  # Latest checkpoint
        None,
        gsrecon,
    )
    
    # Move to device
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    gsvae = gsvae.to(device)
    gsrecon = gsrecon.to(device)
    unet = unet.to(device)
    
    # Set to eval mode
    text_encoder.eval()
    vae.eval()
    gsvae.eval()
    gsrecon.eval()
    unet.eval()
    
    # Freeze all parameters
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    gsvae.requires_grad_(False)
    gsrecon.requires_grad_(False)
    unet.requires_grad_(False)

    # Set diffusion pipeline
    pipeline = StableMVDiffusionPipeline(
        text_encoder=text_encoder, tokenizer=tokenizer,
        vae=vae, unet=unet,
        scheduler=noise_scheduler,
    )
    
    pipeline.set_progress_bar_config(disable=True)
    
    logger.info("DiffSplat pipeline loaded successfully")
    return pipeline, gsvae, gsrecon, opt

class DiffSplatGeometricEvaluator:
    """DiffSplat evaluator using the robust geometric measurement functions."""
    
    def __init__(self, pipeline, gsvae, gsrecon, opt):
        self.pipeline = pipeline
        self.gsvae = gsvae
        self.gsrecon = gsrecon
        self.opt = opt
        self.device = pipeline.device

    def run_and_collect(self, prompt, camera_params, seed=42, guidance_scale=7.5, num_inference_steps=50):
        """Run DiffSplat generation and collect intermediates for geometric analysis."""
        torch.manual_seed(seed)
        
        # Extract camera parameters
        input_C2W = camera_params['input_C2W']
        input_fxfycxcy = camera_params['input_fxfycxcy']
        plucker = camera_params.get('plucker')
        num_views = camera_params.get('num_views', 4)
        
        # --- Use the ACTUAL DiffSplat Pipeline Call (like in evaluator.py) ---
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
            init_std=0.1, 
            init_noise_strength=0.95, 
            init_bg=2.5,
            guess_mode=camera_params.get('guess_mode', False),
            controlnet_conditioning_scale=camera_params.get('controlnet_scale', 1.0),
        )
        
        # Extract Gaussian latents and render multi-view images
        latents = pipeline_output.images
        latents = latents / self.gsvae.scaling_factor + self.gsvae.shift_factor
        
        # Render multi-view images
        render_outputs = self.gsvae.decode_and_render_gslatents(
            self.gsrecon,
            latents, 
            input_C2W.unsqueeze(0), 
            input_fxfycxcy.unsqueeze(0),
            height=256, 
            width=256,
            opacity_threshold=0.0,
        )
        
        # Extract rendered images (V_in, 3, H, W)
        rendered_images = render_outputs["image"].squeeze(0)
        
        # Create intermediates dict from pipeline output
        intermediates = {
            'uncond_noise': getattr(pipeline_output, 'uncond_noise', []),
            'text_noise': getattr(pipeline_output, 'text_noise', []),
            'x_inter': getattr(pipeline_output, 'x_inter', []),
            'timesteps': getattr(pipeline_output, 'timesteps', [])
        }
        
        # Prepare contexts for geometric analysis (like in evaluator.py)
        prompt_embeds = self.pipeline._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=""
        )
        
        # Split into unconditional and conditional parts
        uncond_embeds, cond_embeds = prompt_embeds.chunk(2)
        
        return intermediates, cond_embeds, uncond_embeds, rendered_images, plucker

# ============================================================================
# --- GEOMETRIC ASSUMPTION TESTS (Adapted from assumption_measurements_sept17.py) ---
# ============================================================================
def compute_score_matching_fixed(
    intermediates: Dict,
    model,
    cond_context,
    scheduler,
    plucker: Optional[torch.Tensor] = None,
) -> float:
    """
    (A1) Score-matching consistency for DiffSplat with proper multi-view + Plücker conditioning.
    Divergence is taken w.r.t. the latent channels only (first 4).
    """
    import math
    import numpy as np
    import torch
    import torch.nn.functional as F

    x_inter = intermediates.get("x_inter", [])
    timesteps = intermediates.get("timesteps", [])
    assert len(x_inter) >= 3, "Need at least 3 timesteps for score matching test"

    # UNet device/dtype
    m_param = next(model.parameters())
    m_device, m_dtype = m_param.device, m_param.dtype

    # Config flags
    num_views = int(getattr(model.config, "num_views", 4))
    use_view_concat = bool(getattr(model.config, "view_concat_condition", False))
    use_plucker = bool(getattr(model.config, "input_concat_plucker", False))
    use_mask = bool(getattr(model.config, "input_concat_binary_mask", False))

    # Normalize cond_context to tensor and move to device/dtype
    if isinstance(cond_context, dict) and "context" in cond_context:
        cond_ctx = cond_context["context"]
    else:
        cond_ctx = cond_context
    cond_ctx = cond_ctx.to(m_device, dtype=m_dtype)

    def _pack_unet_inputs(x_lat: torch.Tensor, pl: Optional[torch.Tensor]):
        """
        Returns:
          x_in_flat: (B*V_eff, C_total, H, W)
          embeds_flat: (B*V_eff, S, D)
          x_lat_leaf: (B, V_eff, 4, H, W)  # requires_grad=True
        """
        # Ensure 4D (N,4,H,W)
        if x_lat.ndim == 3:
            x_lat = x_lat.unsqueeze(0)
        x_lat = x_lat.to(m_device, dtype=m_dtype)

        N, C, H, W = x_lat.shape
        assert C == 4, f"Expected 4 latent channels, got {C}"

        # Shape to (B,V,4,H,W)
        if N % num_views == 0:
            B = N // num_views
            V = num_views
            x_bv = x_lat.view(B, V, 4, H, W)
        else:
            B = 1
            V_in = N
            x_bv = x_lat.view(1, V_in, 4, H, W)
            if V_in < num_views:
                reps = math.ceil(num_views / V_in)
                x_bv = x_bv.repeat(1, reps, 1, 1, 1)[:, :num_views]
            elif V_in > num_views:
                x_bv = x_bv[:, :num_views]
            V = num_views

        # Optional leading condition view of zeros
        if use_view_concat:
            zero_cond = torch.zeros(B, 1, 4, H, W, device=m_device, dtype=m_dtype)
            x_bv = torch.cat([zero_cond, x_bv], dim=1)
        V_eff = x_bv.shape[1]

        # Plücker planes
        pl_bv = None
        if use_plucker:
            if pl is None:
                pl = torch.zeros(V_eff, 6, H, W, device=m_device, dtype=m_dtype)
            else:
                pl = pl.to(m_device, dtype=m_dtype)
                if pl.shape[-2:] != (H, W):
                    pl = F.interpolate(pl, size=(H, W), mode="bilinear", align_corners=False)
                if pl.shape[0] == 1 and V_eff > 1:
                    pl = pl.repeat(V_eff, 1, 1, 1)
                elif pl.shape[0] < V_eff:
                    reps = math.ceil(V_eff / pl.shape[0])
                    pl = pl.repeat(reps, 1, 1, 1)[:V_eff]
                elif pl.shape[0] > V_eff:
                    pl = pl[:V_eff]
            pl_bv = pl.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B,V_eff,6,H,W)

        # Binary mask channel
        mask_bv = None
        if use_mask:
            mask_bv = torch.ones(B, V_eff, 1, H, W, device=m_device, dtype=m_dtype)
            if use_view_concat:
                mask_bv[:, 0] = 0.0

        # Stack channels per view
        parts = [x_bv]               # +4
        if pl_bv is not None: parts.append(pl_bv)   # +6
        if mask_bv is not None: parts.append(mask_bv)  # +1
        x_bv_full = torch.cat(parts, dim=2)         # (B,V_eff,C_total,H,W)

        # Match model's expected input channels
        expected_c = model.conv_in.weight.shape[1]
        cur_c = x_bv_full.shape[2]
        if cur_c < expected_c:
            pad = torch.zeros(B, V_eff, expected_c - cur_c, H, W, device=m_device, dtype=m_dtype)
            x_bv_full = torch.cat([x_bv_full, pad], dim=2)
        elif cur_c > expected_c:
            x_bv_full = x_bv_full[:, :, :expected_c]

        # Make latent channels a leaf with grad; extras constant
        x_lat_leaf = x_bv_full[:, :, :4, :, :].contiguous().detach().requires_grad_(True)
        x_extras   = x_bv_full[:, :, 4:, :, :].contiguous().detach()
        x_bv_full  = torch.cat([x_lat_leaf, x_extras], dim=2)

        # Flatten to (BV,C,H,W)
        BV = x_bv_full.shape[0] * x_bv_full.shape[1]
        x_in_flat = x_bv_full.view(BV, x_bv_full.shape[2], H, W)

        # Expand text embeds to BV
        embeds = cond_ctx
        if embeds.shape[0] == 1:
            embeds = embeds.repeat(BV, 1, 1)
        elif embeds.shape[0] != BV:
            if BV % embeds.shape[0] == 0:
                embeds = embeds.repeat(BV // embeds.shape[0], 1, 1)
            else:
                embeds = embeds.repeat(BV, 1, 1)
        embeds = embeds.to(m_device, dtype=m_dtype)

        return x_in_flat, embeds, x_lat_leaf  # shapes: (BV,C,H,W), (BV,S,D), (B,V_eff,4,H,W)

    # Probe a few spread-out steps
    n = len(timesteps)
    probe_idxs = sorted(set([max(0, n // 5), max(0, n // 2), max(0, int(0.8 * n) - 1)]))
    scores = []

    for i in probe_idxs:
        x_t = x_inter[i]
        t_int = int(timesteps[i])

        x_in, embeds, x_lat_leaf = _pack_unet_inputs(x_t, plucker)

        # Timestep vector on UNet device (FIX: keep on CUDA to satisfy time embedding)
        t_vec = torch.full((x_in.shape[0],), t_int, device=m_device, dtype=torch.long)

        # Alpha schedule for score conversion
        if hasattr(scheduler, "alphas_cumprod"):
            alphas = scheduler.alphas_cumprod
            if isinstance(alphas, torch.Tensor) and alphas.device != m_device:
                alphas = alphas.to(m_device)
            alpha_bar_t = alphas[t_int] if isinstance(alphas, torch.Tensor) else torch.tensor(float(alphas[t_int]), device=m_device, dtype=m_dtype)
        else:
            alpha_bar_t = torch.tensor(max(0.01, 1.0 - float(t_int) / 1000.0), device=m_device, dtype=m_dtype)

        # Forward UNet
        with torch.enable_grad():
            noise_pred = model(x_in, t_vec, encoder_hidden_states=embeds).sample
            sigma_t = torch.sqrt(1.0 - alpha_bar_t)
            score_all = -noise_pred / sigma_t  # (BV, C_total, H, W)

            # Reshape back, take latent channels
            B, V_eff, H, W = x_lat_leaf.shape[0], x_lat_leaf.shape[1], x_lat_leaf.shape[-2], x_lat_leaf.shape[-1]
            score_bv = score_all.view(B, V_eff, score_all.shape[1], H, W)
            score_lat = score_bv[:, :, :4, :, :]

            # Hutchinson: multiple v's; retain_graph for all but the last
            n_samples = 3
            divs = []
            for k in range(n_samples):
                v = torch.randn_like(x_lat_leaf)
                y = (score_lat * v).sum()  # scalar
                g = torch.autograd.grad(
                    y, x_lat_leaf,
                    retain_graph=(k < n_samples - 1),
                    create_graph=False,
                    only_inputs=True
                )[0]  # J^T v
                divs.append((g * v).sum().item())

        div_est = float(np.mean(divs))
        score_norm_sq = 0.5 * float((score_lat ** 2).sum().item())
        residual = abs(div_est + score_norm_sq)
        scores.append(1.0 / (1.0 + residual / (abs(score_norm_sq) + 1e-6)))

    return float(np.mean(scores)) if scores else 0.0


def compute_hessian_score_proportionality_fixed(intermediates: Dict, hessian_viz: Dict, scheduler) -> float:
    """A2: Fixed Gaussian Local Structure test for DiffSplat.

    Pairs score magnitudes with Hessian traces using the *labels* in hessian_viz
    (e.g., 't20','t19',...,'t1') rather than trying to match raw scheduler timesteps.
    """
    import numpy as np
    import torch

    text_noise = intermediates.get('text_noise', [])
    timesteps = intermediates.get('timesteps', [])
    if len(text_noise) < 3 or not hessian_viz:
        return 0.0

    # Parse available Hessian labels and infer the total count (e.g., 20 from 't20')
    keys = [k for k in hessian_viz.keys() if isinstance(k, str) and k.startswith('t')]
    if not keys:
        return 0.0
    # max number in labels is treated as T (e.g., 't20' -> 20)
    T = max(int(k[1:]) for k in keys)

    pairs = []
    for k in keys:
        try:
            s = int(k[1:])            # e.g., 't19' -> 19
        except Exception:
            continue
        idx = T - s                   # 't20' -> 0, 't19' -> 1, ..., 't1' -> T-1

        if idx < 0 or idx >= len(text_noise):
            continue

        # Map to scheduler index and convert noise pred to score
        t_int = int(timesteps[idx]) if idx < len(timesteps) else int(timesteps[-1])
        if hasattr(scheduler, 'alphas_cumprod') and 0 <= t_int < len(scheduler.alphas_cumprod):
            alpha_bar_t = scheduler.alphas_cumprod[t_int]
        else:
            alpha_bar_t = torch.tensor(max(0.01, 1.0 - float(t_int) / 1000.0), device=text_noise[0].device)

        sigma_t = torch.sqrt(1.0 - alpha_bar_t)
        score = -text_noise[idx] / sigma_t
        score_mag_sq = float(torch.norm(score).pow(2).item())

        # Hessian trace proxy: negative sum of top magnitudes (first 100 for stability)
        mags = hessian_viz[k].get('cond_magnitudes', [])
        if len(mags) == 0:
            continue
        htrace = -float(np.sum(mags[:100]))

        pairs.append((score_mag_sq, htrace))

    if len(pairs) < 3:
        return 0.0

    sm2, ht = zip(*pairs)
    # Correlation (direction ↑): large scores should align with large curvature in magnitude
    corr = np.corrcoef(sm2, ht)[0, 1] if len(pairs) > 1 else 0.0
    return float(corr)


def compute_improved_covariance_metrics(cond_noise: List[torch.Tensor], 
                                      uncond_noise: List[torch.Tensor]) -> Dict:
    """A4: Fixed Covariance Commutativity Test for DiffSplat multi-view"""
    assert len(cond_noise) >= 10, f"Need at least 10 conditional noise samples, got {len(cond_noise)}"

    # Use more samples for better covariance estimation
    n_samples = min(20, len(cond_noise))
    # Flatten multi-view tensors: [B, C, H, W] -> [B, C*H*W] 
    cond_samples = [p.flatten(start_dim=1).flatten() for p in cond_noise[-n_samples:]]
    uncond_samples = [p.flatten(start_dim=1).flatten() for p in uncond_noise[-n_samples:]]
    
    # Stack into matrices: each row is a sample, each column is a feature (pixel)
    cond_matrix = torch.stack(cond_samples, dim=0).cpu().numpy()
    uncond_matrix = torch.stack(uncond_samples, dim=0).cpu().numpy()
    
    # Subsample features if too large (important for multi-view data)
    n_features = min(2000, cond_matrix.shape[1])
    if cond_matrix.shape[1] > n_features:
        indices = np.random.choice(cond_matrix.shape[1], n_features, replace=False)
        cond_matrix = cond_matrix[:, indices]
        uncond_matrix = uncond_matrix[:, indices]
    
    # Compute sample means
    cond_mean = np.mean(cond_matrix, axis=0)
    uncond_mean = np.mean(uncond_matrix, axis=0)
    
    # Mean equality test
    mean_diff_norm = np.linalg.norm(cond_mean - uncond_mean)
    mean_scale = max(np.linalg.norm(cond_mean), np.linalg.norm(uncond_mean), 1e-8)
    mean_equality = max(0.0, 1.0 - mean_diff_norm / mean_scale)
    
    # Compute covariances with regularization for numerical stability
    reg_factor = 1e-6
    cond_cov = np.cov(cond_matrix, rowvar=False) + reg_factor * np.eye(n_features)
    uncond_cov = np.cov(uncond_matrix, rowvar=False) + reg_factor * np.eye(n_features)
    
    # Eigendecomposition for both matrices
    cond_vals, cond_vecs = np.linalg.eigh(cond_cov)
    uncond_vals, uncond_vecs = np.linalg.eigh(uncond_cov)
    
    # Sort by eigenvalue magnitude
    cond_idx = np.argsort(np.abs(cond_vals))[::-1]
    uncond_idx = np.argsort(np.abs(uncond_vals))[::-1]
    
    cond_vals = cond_vals[cond_idx]
    uncond_vals = uncond_vals[uncond_idx]
    cond_vecs = cond_vecs[:, cond_idx]
    uncond_vecs = uncond_vecs[:, uncond_idx]
    
    # Eigenvalue correlation
    n_eigs = min(50, len(cond_vals), len(uncond_vals))
    eigval_corr, _ = pearsonr(cond_vals[:n_eigs], uncond_vals[:n_eigs])
    eigval_corr = abs(eigval_corr) if not np.isnan(eigval_corr) else 0.0
    
    # Eigenvector alignment (using SVD of cross-correlation matrix)
    n_vecs = min(200, cond_vecs.shape[1])
    cross_corr = cond_vecs[:, :n_vecs].T @ uncond_vecs[:, :n_vecs]
    svd_vals = np.linalg.svd(cross_corr, compute_uv=False)
    eigvec_align = np.mean(svd_vals)
    
    # Commutativity test: ||AB - BA||_F / max(||A||_F, ||B||_F)
    commutator = cond_cov @ uncond_cov - uncond_cov @ cond_cov
    commutator_norm = np.linalg.norm(commutator, 'fro')
    ref_norm = max(np.linalg.norm(cond_cov, 'fro'), np.linalg.norm(uncond_cov, 'fro'), 1e-8)
    
    # Non-commutativity ratio
    non_commut_ratio = commutator_norm / ref_norm
    
    # Commutativity score: maps [0,∞] → [0,1], higher = more commutative
    commutativity = 1.0 / (1.0 + non_commut_ratio)
    
    return {
        'a4_eigval_corr': float(eigval_corr),
        'a4_eigvec_align': float(eigvec_align),
        'a4_commutativity': float(commutativity),
        'a4_mean_equality': float(mean_equality),
    }

def compute_prior_structure_deviation(initial_latents: Optional[torch.Tensor]) -> float:
    """A5: Mean-Field Gaussian Prior Test for DiffSplat multi-view latents"""
    assert initial_latents is not None, "A5: No initial_latents provided"
        
    flat = initial_latents.flatten().cpu().numpy()
    assert len(flat) >= 30, f"A5: Too few samples ({len(flat)}) for reliable test"
        
    # Standardize to test against N(0,1)
    flat_mean = np.mean(flat)
    flat_std = np.std(flat)
    
    assert flat_std >= 1e-8, "A5: Zero variance detected"
        
    std_latents = (flat - flat_mean) / flat_std
    
    # KS test against standard normal
    _, p_val = kstest(std_latents, 'norm')
    return float(p_val)

def compute_score_explosion_indicator(uncond_noise: List[torch.Tensor], 
                                    text_noise: List[torch.Tensor]) -> float:
    """A7: Multi-view explosion detection for DiffSplat"""
    all_norms = []
    for noise_list in [uncond_noise, text_noise]:
        for pred in noise_list:
            norm = torch.norm(pred).item()
            if norm > 1e-10:  # Avoid numerical issues
                all_norms.append(norm)
    
    assert len(all_norms) >= 5, f"Need at least 5 noise norms, got {len(all_norms)}"
    
    max_norm = np.max(all_norms)
    mean_norm = np.mean(all_norms)
    
    assert mean_norm >= 1e-8, "Mean norm too small"
    
    explosion_ratio = max_norm / mean_norm
    return float(min(1.0, max(0.0, (explosion_ratio - 1.0) / 10.0)))


# Additional geometric tests (simplified implementations)
def compute_sharpness_rank_persistence(hessian_viz: Dict) -> float:
    """A3: Sharpness persistence test for DiffSplat"""
    assert len(hessian_viz) >= 3, f"Need at least 3 Hessian visualizations, got {len(hessian_viz)}"
        
    timesteps = sorted(hessian_viz.keys(), reverse=True)
    max_timesteps_to_test = min(10, len(timesteps))
    timesteps = timesteps[:max_timesteps_to_test]
    
    rankings = []
    for t in timesteps:
        mags = hessian_viz[t].get('cond_magnitudes', [])
        if len(mags) >= 10:
            ranking = np.argsort(mags)[::-1][:min(15, len(mags))]
            rankings.append(ranking)
    
    assert len(rankings) >= 2, f"Need at least 2 valid rankings, got {len(rankings)}"
    
    correlations = []
    for i in range(len(rankings) - 1):
        r1, r2 = rankings[i], rankings[i+1]
        min_len = min(len(r1), len(r2))
        if min_len >= 5:
            comparison_len = min(10, min_len)
            corr, _ = spearmanr(r1[:comparison_len], r2[:comparison_len])
            if not np.isnan(corr):
                correlations.append(abs(corr))
    
    assert len(correlations) > 0, "No valid correlations computed"
    return np.mean(correlations)



# Add these functions and classes to DiffSplat/measure_assumptions.py
# Insert after the existing geometric test functions but before the main computation function

def jensen_shannon_divergence(p, q):
    """
    Compute Jensen-Shannon divergence between two probability distributions
    Compatible with older scipy versions
    """
    # Ensure inputs are numpy arrays and normalized
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize to ensure they sum to 1
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute Jensen-Shannon divergence
    m = 0.5 * (p + q)
    js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    
    # Convert to base 2 and take square root for Jensen-Shannon distance
    js_distance = np.sqrt(js_div / np.log(2))
    
    return js_distance

class EnhancedA3SharpnessPersistence:
    """
    Improved A3 measurement that directly tests the assumption:
    'Memorization-indicative sharpness patterns persist through reverse process'
    
    Key improvements:
    1. Multi-scale temporal windows (short + long range persistence)
    2. Distribution-based persistence (not just rankings)
    3. Pattern disruption detection (violations break patterns)
    4. Spatial coherence analysis
    """
    
    def __init__(self):
        # Multiple temporal scales to catch different violation types
        self.temporal_windows = [3, 7, 15]  # Short, medium, long-range persistence
        self.min_eigenvalues = 20  # Use more eigenvalues
        self.persistence_threshold = 0.3  # Calibrated threshold
        
    def extract_eigenvalue_patterns(self, hessian_viz: Dict) -> Dict:
        """Extract eigenvalue patterns across timesteps with better handling"""
        timesteps = sorted(hessian_viz.keys(), reverse=True)  # t=T to t=0
        
        patterns = {
            'eigenvalue_distributions': [],
            'top_eigenvalue_ratios': [],
            'spectral_gaps': [],
            'timesteps': timesteps
        }
        
        for t in timesteps:
            mags = hessian_viz[t].get('cond_magnitudes', [])
            if len(mags) < self.min_eigenvalues:
                # If insufficient data, mark as None but don't skip
                patterns['eigenvalue_distributions'].append(None)
                patterns['top_eigenvalue_ratios'].append(None)
                patterns['spectral_gaps'].append(None)
                continue
            
            # Sort eigenvalues in descending order
            eigvals = np.array(sorted(mags, reverse=True))
            
            # 1. Full distribution (normalized for comparison)
            # Normalize to create probability distribution
            eigval_dist = eigvals / np.sum(eigvals) if np.sum(eigvals) > 0 else None
            patterns['eigenvalue_distributions'].append(eigval_dist)
            
            # 2. Top eigenvalue concentration ratios
            if len(eigvals) >= 5:
                top_5_ratio = np.sum(eigvals[:5]) / np.sum(eigvals)
                top_10_ratio = np.sum(eigvals[:10]) / np.sum(eigvals) if len(eigvals) >= 10 else top_5_ratio
                patterns['top_eigenvalue_ratios'].append([top_5_ratio, top_10_ratio])
            else:
                patterns['top_eigenvalue_ratios'].append(None)
            
            # 3. Spectral gaps (measure of eigenvalue clustering)
            if len(eigvals) >= 3:
                gaps = []
                for i in range(min(10, len(eigvals)-1)):  # First 10 gaps
                    gap = eigvals[i] - eigvals[i+1]
                    gaps.append(gap)
                patterns['spectral_gaps'].append(gaps)
            else:
                patterns['spectral_gaps'].append(None)
        
        return patterns
    
    def measure_distribution_persistence(self, distributions: List[np.ndarray], 
                                       window_size: int) -> float:
        """
        Measure persistence using distribution-based metrics
        Key insight: Violations disrupt the eigenvalue distribution shape
        """
        valid_dists = [d for d in distributions if d is not None]
        if len(valid_dists) < window_size:
            return 0.0
        
        persistence_scores = []
        
        for i in range(len(valid_dists) - window_size + 1):
            window_dists = valid_dists[i:i+window_size]
            
            # Compute pairwise distribution similarities within window
            similarities = []
            for j in range(len(window_dists)):
                for k in range(j+1, len(window_dists)):
                    dist1, dist2 = window_dists[j], window_dists[k]
                    
                    # Ensure distributions have same length by truncating to minimum
                    min_len = min(len(dist1), len(dist2))
                    if min_len < 5:  # Need minimum eigenvalues for reliable comparison
                        continue
                    
                    dist1_trunc = dist1[:min_len]
                    dist2_trunc = dist2[:min_len]
                    
                    # Jensen-Shannon divergence (symmetric, bounded)
                    try:
                        # Ensure non-negative and normalized
                        dist1_norm = np.abs(dist1_trunc) / np.sum(np.abs(dist1_trunc))
                        dist2_norm = np.abs(dist2_trunc) / np.sum(np.abs(dist2_trunc))
                        
                        js_div = jensen_shannon_divergence(dist1_norm, dist2_norm)
                        if not np.isnan(js_div) and np.isfinite(js_div):
                            similarity = 1.0 - js_div  # Convert divergence to similarity
                            similarities.append(similarity)
                    except Exception:
                        continue
            
            if similarities:
                persistence_scores.append(np.mean(similarities))
        
        return np.mean(persistence_scores) if persistence_scores else 0.0
    
    def measure_pattern_coherence(self, patterns: Dict, window_size: int) -> Dict:
        """
        Measure multiple aspects of pattern persistence
        """
        results = {}
        
        # 1. Distribution persistence
        dist_persistence = self.measure_distribution_persistence(
            patterns['eigenvalue_distributions'], window_size
        )
        results['distribution_persistence'] = dist_persistence
        
        # 2. Top eigenvalue ratio persistence
        ratio_data = [r for r in patterns['top_eigenvalue_ratios'] if r is not None]
        if len(ratio_data) >= window_size:
            ratio_persistence = []
            for i in range(len(ratio_data) - window_size + 1):
                window_ratios = ratio_data[i:i+window_size]
                
                # Measure stability of top-5 and top-10 ratios
                top5_ratios = [r[0] for r in window_ratios]
                top10_ratios = [r[1] for r in window_ratios]
                
                top5_stability = 1.0 - np.std(top5_ratios) if len(top5_ratios) > 1 else 1.0
                top10_stability = 1.0 - np.std(top10_ratios) if len(top10_ratios) > 1 else 1.0
                
                ratio_persistence.append((top5_stability + top10_stability) / 2)
            
            results['ratio_persistence'] = np.mean(ratio_persistence) if ratio_persistence else 0.0
        else:
            results['ratio_persistence'] = 0.0
        
        # 3. Spectral gap persistence
        gap_data = [g for g in patterns['spectral_gaps'] if g is not None]
        if len(gap_data) >= window_size:
            gap_persistence = []
            for i in range(len(gap_data) - window_size + 1):
                window_gaps = gap_data[i:i+window_size]
                
                # Compare gap patterns using correlation
                gap_correlations = []
                for j in range(len(window_gaps)):
                    for k in range(j+1, len(window_gaps)):
                        gaps1, gaps2 = window_gaps[j], window_gaps[k]
                        min_gaps = min(len(gaps1), len(gaps2))
                        if min_gaps >= 3:
                            try:
                                corr, _ = spearmanr(gaps1[:min_gaps], gaps2[:min_gaps])
                                if not np.isnan(corr):
                                    gap_correlations.append(abs(corr))
                            except Exception:
                                continue
                
                if gap_correlations:
                    gap_persistence.append(np.mean(gap_correlations))
            
            results['gap_persistence'] = np.mean(gap_persistence) if gap_persistence else 0.0
        else:
            results['gap_persistence'] = 0.0
        
        return results
    
    def detect_pattern_disruption(self, patterns: Dict) -> Dict:
        """
        Detect specific types of pattern disruption that indicate A3 violations
        """
        disruption_indicators = {}
        
        # 1. Sudden distribution changes (violation signature)
        distributions = patterns['eigenvalue_distributions']
        valid_dists = [(i, d) for i, d in enumerate(distributions) if d is not None]
        
        if len(valid_dists) >= 3:
            disruption_scores = []
            for i in range(len(valid_dists) - 1):
                idx1, dist1 = valid_dists[i]
                idx2, dist2 = valid_dists[i + 1]
                
                min_len = min(len(dist1), len(dist2))
                if min_len >= 5:
                    dist1_norm = np.abs(dist1[:min_len]) / np.sum(np.abs(dist1[:min_len]))
                    dist2_norm = np.abs(dist2[:min_len]) / np.sum(np.abs(dist2[:min_len]))
                    
                    try:
                        # Large JS divergence indicates disruption
                        js_div = jensen_shannon_divergence(dist1_norm, dist2_norm)
                        if not np.isnan(js_div):
                            disruption_scores.append(js_div)
                    except Exception:
                        continue
            
            disruption_indicators['mean_disruption'] = np.mean(disruption_scores) if disruption_scores else 0.0
            disruption_indicators['max_disruption'] = np.max(disruption_scores) if disruption_scores else 0.0
        else:
            disruption_indicators['mean_disruption'] = 0.0
            disruption_indicators['max_disruption'] = 0.0
        
        # 2. Non-monotonic eigenvalue evolution (should be smooth for valid models)
        ratio_data = [r for r in patterns['top_eigenvalue_ratios'] if r is not None]
        if len(ratio_data) >= 3:
            top5_trajectory = [r[0] for r in ratio_data]
            
            # Measure non-monotonicity as number of direction changes
            direction_changes = 0
            for i in range(1, len(top5_trajectory) - 1):
                prev_slope = top5_trajectory[i] - top5_trajectory[i-1]
                next_slope = top5_trajectory[i+1] - top5_trajectory[i]
                if prev_slope * next_slope < 0:  # Sign change = direction change
                    direction_changes += 1
            
            # Normalize by maximum possible changes
            max_changes = len(top5_trajectory) - 2
            disruption_indicators['trajectory_chaos'] = direction_changes / max_changes if max_changes > 0 else 0.0
        else:
            disruption_indicators['trajectory_chaos'] = 0.0
        
        return disruption_indicators
    
    def compute_comprehensive_a3_score(self, hessian_viz: Dict) -> Dict:
        """
        Main function: Compute comprehensive A3 violation score
        """
        if len(hessian_viz) < 5:  # Need minimum timesteps
            return self._default_results()
        
        # Extract eigenvalue patterns
        patterns = self.extract_eigenvalue_patterns(hessian_viz)
        
        results = {}
        
        # Measure persistence at multiple temporal scales
        for window_size in self.temporal_windows:
            window_key = f'window_{window_size}'
            coherence = self.measure_pattern_coherence(patterns, window_size)
            results[window_key] = coherence
        
        # Detect pattern disruptions
        disruptions = self.detect_pattern_disruption(patterns)
        results['disruptions'] = disruptions
        
        # Compute overall A3 violation score
        overall_score = self._compute_overall_score(results)
        results['a3_violation_score'] = overall_score
        results['a3_persistence_score'] = 1.0 - overall_score  # Invert for persistence
        
        # Diagnostic information
        results['diagnostic'] = {
            'total_timesteps': len(patterns['timesteps']),
            'valid_distributions': sum(1 for d in patterns['eigenvalue_distributions'] if d is not None),
            'sufficient_data': len([d for d in patterns['eigenvalue_distributions'] if d is not None]) >= max(self.temporal_windows)
        }
        
        return results
    
    def _compute_overall_score(self, results: Dict) -> float:
        """
        Combine multiple metrics into overall A3 violation score
        Higher score = more violation = less persistence
        """
        persistence_scores = []
        disruption_scores = []
        
        # Collect persistence scores (lower = more violation)
        for window_key in [f'window_{w}' for w in self.temporal_windows]:
            if window_key in results:
                window_results = results[window_key]
                for metric in ['distribution_persistence', 'ratio_persistence', 'gap_persistence']:
                    if metric in window_results:
                        persistence_scores.append(window_results[metric])
        
        # Collect disruption scores (higher = more violation)
        if 'disruptions' in results:
            disruptions = results['disruptions']
            for metric in ['mean_disruption', 'max_disruption', 'trajectory_chaos']:
                if metric in disruptions:
                    disruption_scores.append(disruptions[metric])
        
        # Combine scores
        if not persistence_scores and not disruption_scores:
            return 0.0
        
        # Average persistence (invert so higher = more violation)
        avg_persistence = np.mean(persistence_scores) if persistence_scores else 1.0
        persistence_violation = 1.0 - avg_persistence
        
        # Average disruption (higher = more violation)
        avg_disruption = np.mean(disruption_scores) if disruption_scores else 0.0
        
        # Weighted combination
        overall_violation = 0.6 * persistence_violation + 0.4 * avg_disruption
        
        return np.clip(overall_violation, 0.0, 1.0)
    
    def _default_results(self) -> Dict:
        """Return default results when insufficient data"""
        return {
            'a3_violation_score': 0.0,
            'a3_persistence_score': 0.0,
            'diagnostic': {'insufficient_data': True}
        }


def compute_enhanced_sharpness_persistence_diffsplat(hessian_viz: Dict) -> float:
    """
    Drop-in replacement for the current A3 function in DiffSplat
    Returns persistence score (higher = better persistence = assumption holds)
    """
    enhanced_a3 = EnhancedA3SharpnessPersistence()
    results = enhanced_a3.compute_comprehensive_a3_score(hessian_viz)
    
    # Return persistence score for compatibility with existing code
    persistence_score = results.get('a3_persistence_score', 0.0)
    
    # Log diagnostic info
    if 'diagnostic' in results:
        diag = results['diagnostic']
        if diag.get('insufficient_data', False):
            logger.info(f"A3 Warning: Insufficient data for reliable A3 measurement")
        elif not diag.get('sufficient_data', True):
            logger.info(f"A3 Warning: Limited data - {diag.get('valid_distributions', 0)}/{diag.get('total_timesteps', 0)} valid timesteps")
    
    return persistence_score


def a3_hotspot_jaccard_persistence(hessian_viz, q=0.05, lags=(1,2,4)):
    """
    Higher = more persistence. Sensitive to relocation of high-curvature 'hotspots'.
    hessian_viz[t]['cond_magnitudes'] is a flat list/1D array per timestep.
    """
    if len(hessian_viz) < 3:
        return 0.0
    ts = sorted(hessian_viz.keys(), reverse=True)  # t=T...0
    vecs = []
    for t in ts:
        mags = np.asarray(hessian_viz[t].get('cond_magnitudes', []), dtype=np.float64)
        if mags.size == 0:
            vecs.append(None); continue
        k = max(1, int(np.ceil(q * mags.size)))
        idx = np.argpartition(mags, -k)[-k:]  # top-q% indices (unordered)
        vecs.append(set(idx.tolist()))
    scores = []
    for lag in lags:
        for i in range(len(vecs)-lag):
            A, B = vecs[i], vecs[i+lag]
            if A is None or B is None or len(A)==0 or len(B)==0: 
                continue
            inter = len(A & B); union = len(A | B)
            scores.append(inter/union if union>0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0

def a3_temporal_autocorr(hessian_viz, max_lag=2):
    # Build T x N matrix (time x position)
    ts = sorted(hessian_viz.keys(), reverse=True)
    rows = []
    for t in ts:
        v = np.asarray(hessian_viz[t].get('cond_magnitudes', []), dtype=np.float64)
        if v.size==0: 
            return 0.0
        rows.append(v)
    M = np.vstack(rows)  # shape: T x N
    if M.shape[0] < max_lag+2: 
        return 0.0
    M = (M - M.mean(axis=0, keepdims=True)) / (M.std(axis=0, keepdims=True) + 1e-9)
    acc = []
    for lag in range(1, max_lag+1):
        # corr at each position between t and t+lag, then median over positions & time
        c_list = []
        for t in range(M.shape[0]-lag):
            a, b = M[t], M[t+lag]
            num = (a*b).mean()
            c_list.append(num)  # already normalized to unit var per position
        if c_list:
            acc.append(np.median(c_list))
    return float(np.mean(acc)) if acc else 0.0

def a3_windowed_distribution_persistence(hessian_viz, window=4, use_emd=True):
    ts = sorted(hessian_viz.keys(), reverse=True)
    dists = []
    for t in ts:
        v = np.asarray(hessian_viz[t].get('cond_magnitudes', []), dtype=np.float64)
        if v.size==0: 
            dists.append(None); continue
        v = np.abs(v)
        v = v / (v.sum()+1e-12)
        dists.append(v)
    vals = []
    for i in range(len(dists)-window+1):
        chunk = [d for d in dists[i:i+window] if d is not None]
        if len(chunk) < 2: 
            continue
        pair_sims = []
        for j in range(len(chunk)-1):
            a = chunk[j]; b = chunk[j+1]
            m = min(a.size, b.size)
            a = a[:m]; b = b[:m]
            if use_emd:
                # EMD on indices as support
                x = np.arange(m)
                dist = wasserstein_distance(x, x, a, b)
                sim = 1.0 / (1.0 + dist)  # map to [0,1]
            else:
                # JS-based
                eps = 1e-12
                a_ = a/(a.sum()+eps); b_ = b/(b.sum()+eps)
                m_ = 0.5*(a_+b_)
                js = 0.5*(np.sum(a_*np.log((a_+eps)/(m_+eps))) + 
                          0.5*np.sum(b_*np.log((b_+eps)/(m_+eps))))
                sim = 1.0 - np.sqrt(js/np.log(2)+0.0)
            pair_sims.append(sim)
        if pair_sims:
            vals.append(np.mean(pair_sims))
    return float(np.mean(vals)) if vals else 0.0

def a3_subspace_overlap(hessian_viz, k=8):
    """
    Note: This requires Hv_stack data which may not be available in standard DiffSplat
    Include for completeness but may return 0.0 if data not present
    """
    from numpy.linalg import svd
    ts = sorted([t for t in hessian_viz if 'Hv_stack' in hessian_viz[t]], reverse=True)
    if len(ts) < 2: 
        return 0.0
    # Orthonormal bases via SVD of Hv_stack
    bases = []
    for t in ts:
        H = hessian_viz[t]['Hv_stack']  # D x R
        if H is None or H.ndim != 2 or min(H.shape) < k:
            return 0.0
        U, _, _ = svd(H, full_matrices=False)
        bases.append(U[:, :k])          # D x k
    sims = []
    for i in range(len(bases)-1):
        Q1, Q2 = bases[i], bases[i+1]
        _, s, _ = svd(Q1.T @ Q2, full_matrices=False)  # singular vals = cos(principal angles)
        sims.append(np.mean(s))  # [0,1]
    return float(np.mean(sims)) if sims else 0.0


def compute_geometric_metrics_diffsplat(run_output, hypotheses_to_test=None):
    """Main function using DiffSplat adapted implementations."""
    intermediates, cond_context, uncond_context, model, scheduler, rendered_images, plucker = run_output
    
    if hypotheses_to_test is None:
        hypotheses_to_test = list(range(1, 8))

    results = {}
    hessian_viz = {}

    # Compute Hessian visualizations for A2, A3
    if any(h in hypotheses_to_test for h in [2, 3]):
        hessian_metric = HessianMetric(timesteps_to_measure=HESSIAN_STEPS_TO_MEASURE)
        hessian_result = hessian_metric.measure(
            intermediates=intermediates, model=model,
            conditioning_context=cond_context, unconditioning_context=uncond_context,
            plucker=plucker  # Pass plucker embeddings
        )
        hessian_viz = hessian_result.get('visualizations', {})

    # Add Noise Diff Norm for reference
    norm_traj = [torch.norm(tn - un).item() for un, tn in zip(intermediates['uncond_noise'], intermediates['text_noise'])]
    results['noise_diff_norm_mean'] = np.mean(norm_traj)

    # A1: Score matching consistency (multiple alternatives)
    if 1 in hypotheses_to_test:
        results['a1_score_matching_fixed'] = compute_score_matching_fixed(
            intermediates, model, cond_context, scheduler, plucker 
        )
    
    # A2: Score-Hessian proportionality
    if 2 in hypotheses_to_test:
        results['a2_score_hessian_proportionality'] = compute_hessian_score_proportionality_fixed(
            intermediates, hessian_viz, scheduler
        )

    # A3: Sharpness persistence 
    if 3 in hypotheses_to_test:
        results['a3_sharpness_rank_persistence'] = compute_enhanced_sharpness_persistence_diffsplat(hessian_viz)
        results['a3_hotspot_jaccard'] = a3_hotspot_jaccard_persistence(hessian_viz, q=0.05, lags=(1,2,4))
        results['a3_temporal_autocorr'] = a3_temporal_autocorr(hessian_viz, max_lag=2)
        results['a3_dist_persistence_emd'] = a3_windowed_distribution_persistence(hessian_viz, window=4, use_emd=True)
        results['a3_subspace_overlap_k8'] = a3_subspace_overlap(hessian_viz, k=8)

    # A4: Comprehensive covariance analysis
    if 4 in hypotheses_to_test:
        covariance_metrics = compute_improved_covariance_metrics(
            intermediates.get('text_noise', []), intermediates.get('uncond_noise', [])
        )
        results.update(covariance_metrics)

    # A5: Gaussian prior test
    if 5 in hypotheses_to_test:
        results['a5_prior_deviation_pval'] = compute_prior_structure_deviation(
            intermediates.get('text_noise', [])[1]
        )

    # A7: Score explosion detection
    if 7 in hypotheses_to_test:
        results['a7_score_explosion_indicator'] = compute_score_explosion_indicator(
            intermediates.get('uncond_noise', []), intermediates.get('text_noise', [])
        )
        
    return results

def print_comprehensive_table_results_diffsplat(all_results, model_name="DiffSplat", verbose=False):
    """Print comprehensive results table for DiffSplat."""
    
    def get_stats(results, key):
        if not results:
            return (0.0, 0.0)
        values = [r.get(key) for r in results if r and r.get(key) is not None and not np.isnan(r.get(key))]
        return (np.mean(values), np.std(values)) if values else (0.0, 0.0)
    
    def get_aggregate_stats(all_results, key):
        """Get statistics across ALL datasets combined"""
        all_values = []
        for dataset_results in all_results.values():
            values = [r.get(key) for r in dataset_results if r and r.get(key) is not None and not np.isnan(r.get(key))]
            all_values.extend(values)
        return (np.mean(all_values), np.std(all_values)) if all_values else (0.0, 0.0)
    
    # Comprehensive metrics for DiffSplat
    comprehensive_metrics = [
        # A1 alternatives
        ("(A1) Score Matching Fixed - BEST", "a1_score_matching_fixed", "↓"),
        
        # A2
        ("(A2) Score-Hessian Proportionality", "a2_score_hessian_proportionality", "↑"),
        
        # A3
        ("(A3) Rank Persistence", "a3_sharpness_rank_persistence", "↑"),
        ("(A3) Hotspot Jaccard", "a3_hotspot_jaccard", "↑"),
        ("(A3) Temporal Autocorr", "a3_temporal_autocorr", "↑"),
        ("(A3) Distributional Persistence (EMD)", "a3_dist_persistence_emd", "↑"),
        ("(A3) Subspace Overlap (k=8)", "a3_subspace_overlap_k8", "↑"),

        # A4 comprehensive
        # ("(A4) Eigenvalue Correlation", "a4_eigval_corr", "↑"),
        ("(A4) Eigenspace Alignment", "a4_eigvec_align", "↑"),
        # ("(A4) Covariance Commutativity", "a4_commutativity", "↑"),
        # ("(A4) Mean Equality", "a4_mean_equality", "↑"),
        
        # A5, A7
        ("(A5) Gaussian Prior p-value", "a5_prior_deviation_pval", "↑"),
        ("(A7) Score Explosion Indicator", "a7_score_explosion_indicator", "↓"),
        
        # Reference metrics
        ("Noise Diff Norm Mean", "noise_diff_norm_mean", "varies"),
    ]
    
    if verbose:
        print(f"\n{'='*140}")
        print(f"COMPREHENSIVE GEOMETRIC ASSUMPTION VALIDATION: {model_name} [VERBOSE MODE]")
        print(f"{'='*140}")
        
        print(f"{'Assumption Test':<50} | {'Dataset':<20} | {'Mean ± Std':<20} | {'Dir':<4}")
        print("-" * 140)
        
        for name, key, direction in comprehensive_metrics:
            for dataset_name, results in all_results.items():
                if results:
                    mean_val, std_val = get_stats(results, key)
                    print(f"{name:<50} | {dataset_name:<20} | {mean_val:.4f} ± {std_val:.4f}    | {direction}")
        
        print(f"{'='*140}")
        
    else:
        print(f"\n{'='*100}")
        print(f"COMPREHENSIVE GEOMETRIC ASSUMPTION VALIDATION: {model_name} [AGGREGATE]")
        print(f"{'='*100}")
        
        print(f"{'Assumption Test':<50} | {'Overall Mean ± Std':<30} | {'Dir':<4}")
        print("-" * 100)
        
        for name, key, direction in comprehensive_metrics:
            agg_mean, agg_std = get_aggregate_stats(all_results, key)
            if agg_mean != 0.0 or agg_std != 0.0:
                print(f"{name:<50} | {agg_mean:.4f} ± {agg_std:.4f}           | {direction}")
        
        print(f"{'='*100}")
        
        # Show per-dataset sample counts
        print(f"\nDATASET BREAKDOWN:")
        for dataset_name, results in all_results.items():
            print(f"  {dataset_name}: {len(results)} samples")
    
    # Summary statistics
    print(f"\nSUMMARY:")
    total_samples = sum(len(results) for results in all_results.values())
    print(f"Total successful samples: {total_samples}")
    print(f"Datasets processed: {len(all_results)}")
    print(f"{'='*100}\n")

def process_dataset(evaluator, dataset_name, dataset, camera_params, hypotheses_to_test):
    """Process a single dataset."""
    logger.info(f"\n=== Processing {dataset_name} ===")
    
    dataset_results = []
    
    for idx, batch in enumerate(tqdm(dataset, desc=f"DiffSplat {dataset_name}")):
        prompt = batch['prompt'][0] if isinstance(batch['prompt'], (list, tuple)) else batch['prompt']
        if not prompt.strip():
            logger.warning(f"Empty prompt at index {idx}")
            continue
        
        logger.info(f"Processing prompt #{idx}: '{prompt[:60]}...'")
        
        run_output = evaluator.run_and_collect(
            prompt, camera_params, seed=idx, 
            guidance_scale=GUIDANCE_SCALE, 
            num_inference_steps=N_INFERENCE_STEPS
        )
        intermediates, cond_context, uncond_context, rendered_images, plucker = run_output
        
        # Save multiview images
        if rendered_images is not None:
            for view_idx in range(rendered_images.shape[0]):
                img_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{idx:03d}_view{view_idx}.png")
                if view_idx < rendered_images.shape[0]:
                    import torchvision
                    torchvision.utils.save_image(rendered_images[view_idx], img_path)
        
        # Prepare full output for geometric analysis
        full_output = (intermediates, cond_context, uncond_context, 
                      evaluator.pipeline.unet, evaluator.pipeline.scheduler, rendered_images, plucker)
        
        metrics = compute_geometric_metrics_diffsplat(full_output, hypotheses_to_test)
        
        # Extract metadata 
        is_memorized = batch.get('label', [''])[0] if 'label' in batch else False
        if isinstance(is_memorized, str):
            is_memorized = "memorized" in is_memorized.lower()
            
        metrics.update({
            "prompt": prompt, 
            "dataset": dataset_name,
            "is_memorized": is_memorized
        })
        
        dataset_results.append(metrics)
        logger.info(f"Successfully computed metrics for sample {idx}")
        cleanup_memory()
    
    logger.info(f"Completed {dataset_name}: {len(dataset_results)} successful samples")
    return dataset_results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run DiffSplat geometric analysis")
    parser.add_argument(
        '--config', type=str, default="configs/gsdiff_sd15.yaml",
        help='DiffSplat config file path'
    )
    parser.add_argument(
        '--ckpt_iter', type=int, default=13020,
        help='Checkpoint iteration to load'
    )
    parser.add_argument(
        '-t', '--test', type=int, nargs='+', choices=range(1, 8),
        default=list(range(1, 8)), help='Select hypotheses to test (A1-A7)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show per-dataset breakdown (detailed view)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Running DiffSplat geometric analysis for hypotheses: {args.test}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load DiffSplat pipeline
    pipeline, gsvae, gsrecon, opt = load_diffsplat_pipeline(args.config, device, args.ckpt_iter)

    evaluator = DiffSplatGeometricEvaluator(pipeline, gsvae, gsrecon, opt)
    
    # Setup camera parameters
    camera_params = setup_camera_parameters(opt, device, num_views=NUM_VIEWS)
    
    # Create dataset configuration like run_baseline.py
    dataset_config = {
        'laion_memorized': {
            'path': 'data/nemo-prompts/memorized_laion_prompts.csv',
            'max_prompts_per_cluster': 10,
            'max_clusters': 15,
            'source_type': 'csv'
        },
    }
    
    # Create dataset manager and load datasets
    manager = DatasetManager(dataset_config)
    datasets = manager.load_datasets()
    dataloaders = manager.create_dataloaders(datasets, batch_size=1)
    
    all_results = {}
    
    # Process datasets using dataloader structure
    for dataset_name, dataloader in dataloaders.items():
        results = process_dataset(evaluator, dataset_name, dataloader, camera_params, args.test)
        all_results[dataset_name] = results

    # Print results
    logger.info(f"\nFinal Results Summary:")
    total_samples = sum(len(results) for results in all_results.values())
    logger.info(f"Total successful samples: {total_samples}")
    
    print_comprehensive_table_results_diffsplat(
        all_results, 
        model_name="DiffSplat", 
        verbose=args.verbose
    )
    
    # Save results
    results_dir = "diffsplat_geometric_results"
    os.makedirs(results_dir, exist_ok=True)
    
    for dataset_name, results in all_results.items():
        if results:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(results_dir, f"{dataset_name}_metrics.csv"), index=False)
            logger.info(f"Saved {len(results)} results to {dataset_name}_metrics.csv")

if __name__ == "__main__":
    main()