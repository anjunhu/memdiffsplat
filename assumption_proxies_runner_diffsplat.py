# DiffSplat/assumption_proxies_runner.py
# Self-contained: all geometric proxy functions + runner in one file.
# No dependency on assumption_proxies.py.
#
# Adapted from DanceGRPO for DiffSplat's multi-view 3D architecture.
# Key differences:
#   - Model: UNetMV2DConditionModel with Plücker ray embeddings + view_concat_condition
#   - Latent shape: [B*V, C_total, H, W] where C_total = 4 + 6(plucker) + 1(mask)
#   - Score matching: autograd on latent channels only (first 4), extras held constant
#
# Usage:
#   python assumption_proxies_runner.py
#   python assumption_proxies_runner.py --config configs/gsdiff_sd15.yaml --ckpt_iter 13020
#   python assumption_proxies_runner.py --n_prompts 20

import math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
import gc
import logging
import warnings
from scipy.stats import kstest, pearsonr, spearmanr
from typing import Dict, Optional, List

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))

from tqdm import tqdm

# --- DiffSplat Imports ---
from extensions.diffusers_diffsplat import UNetMV2DConditionModel, StableMVDiffusionPipeline
from src.models import GSAutoencoderKL, GSRecon
from src.options import opt_dict
import src.utils.util as util
import src.utils.geo_util as geo_util

from diffusers.schedulers import DDIMScheduler
from diffusers.models.autoencoders import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
LAION_PATHS = {
    'memorized':   'data/nemo-prompts/memorized_laion_prompts.csv',
    'unmemorized': 'data/nemo-prompts/unmemorized_laion_prompts.csv',
}
N_PROMPTS = 5
N_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
NUM_VIEWS = 4


# =====================================================================================
# --- CORE METRIC CLASSES ---
# =====================================================================================

class AdaptedNoiseDiffNormMetric:
    def measure(self, intermediates: Dict, **kwargs) -> Dict:
        uncond_noise, text_noise = intermediates['uncond_noise'], intermediates['text_noise']
        norm_traj = [torch.norm(tn - un).item() for un, tn in zip(uncond_noise, text_noise)]
        return {"noise_diff_norm_mean": np.mean(norm_traj), "noise_diff_norm_traj": norm_traj}


class HessianMetric:
    """Single-direction FD Hessian with Plücker-aware packing."""
    def __init__(self, timesteps_to_measure: Optional[List[int]] = None):
        self.timesteps_to_measure = timesteps_to_measure or [0, 10, 20, 30, 40]

    def measure(self, intermediates: Dict, model, **kwargs) -> Dict:
        cond_context = kwargs.get("conditioning_context")
        uncond_context = kwargs.get("unconditioning_context")
        plucker = kwargs.get("plucker")

        results = {}
        timesteps = intermediates['timesteps']
        x_inter = intermediates['x_inter']
        indices = [idx for idx in self.timesteps_to_measure if 0 <= idx < len(x_inter)]

        m_param = next(model.parameters())
        device, dtype = m_param.device, m_param.dtype

        for t_idx in indices:
            try:
                x_t = x_inter[t_idx].to(device=device, dtype=dtype)
                t_val = timesteps[t_idx]

                x_packed, embeds = _pack_unet_input(x_t, model, cond_context, plucker)
                x_packed_u, embeds_u = _pack_unet_input(x_t, model, uncond_context, plucker)

                t_vec = torch.full((x_packed.shape[0],), t_val, device=device, dtype=torch.long)

                s_cond = model(x_packed, t_vec, encoder_hidden_states=embeds).sample
                s_uncond = model(x_packed_u, t_vec, encoder_hidden_states=embeds_u).sample
                s_delta = s_cond - s_uncond
                s_delta_norm = torch.linalg.norm(s_delta)
                if s_delta_norm < 1e-6:
                    continue

                delta = 1e-3
                # Perturb only the first 4 latent channels to avoid dilution
                # across non-latent channels (Plücker rays, mask)
                latent_dir = s_delta / s_delta_norm
                perturbation = torch.zeros_like(x_packed)
                n_latent_ch = min(latent_dir.shape[1], 4)
                perturbation[:, :n_latent_ch] = delta * latent_dir[:, :n_latent_ch]
                x_p = x_packed + perturbation

                h_cond = (model(x_p, t_vec, encoder_hidden_states=embeds).sample - s_cond) / delta
                h_uncond = (model(x_p, t_vec, encoder_hidden_states=embeds_u).sample - s_uncond) / delta

                cond_mags = torch.linalg.norm(h_cond.squeeze(0), dim=0).flatten().detach().cpu().tolist()
                uncond_mags = torch.linalg.norm(h_uncond.squeeze(0), dim=0).flatten().detach().cpu().tolist()

                results[int(t_val)] = {"cond_magnitudes": cond_mags, "uncond_magnitudes": uncond_mags}
            except Exception as e:
                logger.warning(f"Hessian calculation failed at timestep index {t_idx}: {e}")
                continue

        return {"visualizations": results}


# =====================================================================================
# --- PACKING HELPER ---
# =====================================================================================

def _pack_unet_input(x_lat, model, cond_context, plucker=None):
    """Pack latents + Plücker + mask into UNet input format.

    Returns (x_in_flat, embeds_flat) ready for model(x_in_flat, t, encoder_hidden_states=embeds_flat).
    """
    m_param = next(model.parameters())
    device, dtype = m_param.device, m_param.dtype

    num_views = int(getattr(model.config, "num_views", 4))
    use_view_concat = bool(getattr(model.config, "view_concat_condition", False))
    use_plucker = bool(getattr(model.config, "input_concat_plucker", False))
    use_mask = bool(getattr(model.config, "input_concat_binary_mask", False))

    if isinstance(cond_context, dict) and "context" in cond_context:
        cond_ctx = cond_context["context"]
    else:
        cond_ctx = cond_context
    cond_ctx = cond_ctx.to(device, dtype=dtype)

    if x_lat.ndim == 3:
        x_lat = x_lat.unsqueeze(0)
    x_lat = x_lat.to(device, dtype=dtype)

    N, C, H, W = x_lat.shape
    assert C == 4, f"Expected 4 latent channels, got {C}"

    if N % num_views == 0:
        B = N // num_views
        x_bv = x_lat.view(B, num_views, 4, H, W)
    else:
        B = 1
        x_bv = x_lat.view(1, N, 4, H, W)
        if N < num_views:
            reps = math.ceil(num_views / N)
            x_bv = x_bv.repeat(1, reps, 1, 1, 1)[:, :num_views]
        elif N > num_views:
            x_bv = x_bv[:, :num_views]

    if use_view_concat:
        zero_cond = torch.zeros(B, 1, 4, H, W, device=device, dtype=dtype)
        x_bv = torch.cat([zero_cond, x_bv], dim=1)
    V_eff = x_bv.shape[1]

    pl_bv = None
    if use_plucker:
        if plucker is None:
            pl = torch.zeros(V_eff, 6, H, W, device=device, dtype=dtype)
        else:
            pl = plucker.to(device, dtype=dtype)
            if pl.shape[-2:] != (H, W):
                pl = F.interpolate(pl, size=(H, W), mode="bilinear", align_corners=False)
            if pl.shape[0] < V_eff:
                reps = math.ceil(V_eff / pl.shape[0])
                pl = pl.repeat(reps, 1, 1, 1)[:V_eff]
            elif pl.shape[0] > V_eff:
                pl = pl[:V_eff]
        pl_bv = pl.unsqueeze(0).repeat(B, 1, 1, 1, 1)

    mask_bv = None
    if use_mask:
        mask_bv = torch.ones(B, V_eff, 1, H, W, device=device, dtype=dtype)
        if use_view_concat:
            mask_bv[:, 0] = 0.0

    parts = [x_bv]
    if pl_bv is not None:
        parts.append(pl_bv)
    if mask_bv is not None:
        parts.append(mask_bv)
    x_bv_full = torch.cat(parts, dim=2)

    expected_c = model.conv_in.weight.shape[1]
    cur_c = x_bv_full.shape[2]
    if cur_c < expected_c:
        pad = torch.zeros(B, V_eff, expected_c - cur_c, H, W, device=device, dtype=dtype)
        x_bv_full = torch.cat([x_bv_full, pad], dim=2)
    elif cur_c > expected_c:
        x_bv_full = x_bv_full[:, :, :expected_c]

    BV = B * V_eff
    x_in_flat = x_bv_full.view(BV, x_bv_full.shape[2], H, W)

    embeds = cond_ctx
    if embeds.shape[0] != BV:
        if BV % embeds.shape[0] == 0:
            embeds = embeds.repeat(BV // embeds.shape[0], 1, 1)
        else:
            embeds = embeds[:1].repeat(BV, 1, 1)

    return x_in_flat, embeds


# =====================================================================================
# --- GEOMETRIC ASSUMPTION TESTS ---
# =====================================================================================

def compute_score_matching_fixed(intermediates, model, cond_context, scheduler,
                                 plucker=None):
    """A1: Score matching consistency via autograd Hutchinson estimator.

    DiffSplat adaptation: packs Plücker + view_concat, takes grad only on latent channels.
    """
    x_inter = intermediates.get("x_inter", [])
    timesteps = intermediates.get("timesteps", [])
    if len(x_inter) < 3:
        return 0.0

    m_param = next(model.parameters())
    m_device, m_dtype = m_param.device, m_param.dtype

    num_views = int(getattr(model.config, "num_views", 4))
    use_view_concat = bool(getattr(model.config, "view_concat_condition", False))
    use_plucker = bool(getattr(model.config, "input_concat_plucker", False))
    use_mask = bool(getattr(model.config, "input_concat_binary_mask", False))

    if isinstance(cond_context, dict) and "context" in cond_context:
        cond_ctx = cond_context["context"]
    else:
        cond_ctx = cond_context
    cond_ctx = cond_ctx.to(m_device, dtype=m_dtype)

    n = len(timesteps)
    probe_idxs = sorted(set([max(0, n // 5), max(0, n // 2), max(0, int(0.8 * n) - 1)]))
    scores = []

    for i in probe_idxs:
        if i >= len(x_inter):
            continue
        x_t = x_inter[i].to(m_device, dtype=m_dtype)
        t_int = int(timesteps[i])

        if x_t.ndim == 3:
            x_t = x_t.unsqueeze(0)
        N, C, H, W = x_t.shape

        if N % num_views == 0:
            B = N // num_views
            x_bv = x_t.view(B, num_views, 4, H, W)
        else:
            B = 1
            x_bv = x_t.view(1, N, 4, H, W)
            if N < num_views:
                x_bv = x_bv.repeat(1, math.ceil(num_views / N), 1, 1, 1)[:, :num_views]
            elif N > num_views:
                x_bv = x_bv[:, :num_views]

        if use_view_concat:
            zero_cond = torch.zeros(B, 1, 4, H, W, device=m_device, dtype=m_dtype)
            x_bv = torch.cat([zero_cond, x_bv], dim=1)
        V_eff = x_bv.shape[1]

        extras_parts = []
        if use_plucker:
            if plucker is None:
                pl = torch.zeros(V_eff, 6, H, W, device=m_device, dtype=m_dtype)
            else:
                pl = plucker.to(m_device, dtype=m_dtype)
                if pl.shape[-2:] != (H, W):
                    pl = F.interpolate(pl, size=(H, W), mode="bilinear", align_corners=False)
                if pl.shape[0] < V_eff:
                    pl = pl.repeat(math.ceil(V_eff / pl.shape[0]), 1, 1, 1)[:V_eff]
                elif pl.shape[0] > V_eff:
                    pl = pl[:V_eff]
            extras_parts.append(pl.unsqueeze(0).repeat(B, 1, 1, 1, 1))
        if use_mask:
            mask = torch.ones(B, V_eff, 1, H, W, device=m_device, dtype=m_dtype)
            if use_view_concat:
                mask[:, 0] = 0.0
            extras_parts.append(mask)

        x_lat_leaf = x_bv.contiguous().detach().requires_grad_(True)
        parts = [x_lat_leaf] + extras_parts
        x_full = torch.cat(parts, dim=2)

        expected_c = model.conv_in.weight.shape[1]
        cur_c = x_full.shape[2]
        if cur_c < expected_c:
            pad = torch.zeros(B, V_eff, expected_c - cur_c, H, W, device=m_device, dtype=m_dtype)
            x_full = torch.cat([x_full, pad], dim=2)
        elif cur_c > expected_c:
            x_full = x_full[:, :, :expected_c]

        BV = B * V_eff
        x_in_flat = x_full.view(BV, x_full.shape[2], H, W)

        embeds = cond_ctx
        if embeds.shape[0] != BV:
            embeds = embeds[:1].repeat(BV, 1, 1) if BV % embeds.shape[0] != 0 else embeds.repeat(BV // embeds.shape[0], 1, 1)

        t_vec = torch.full((BV,), t_int, device=m_device, dtype=torch.long)

        if hasattr(scheduler, "alphas_cumprod"):
            alphas = scheduler.alphas_cumprod
            if isinstance(alphas, torch.Tensor) and alphas.device != m_device:
                alphas = alphas.to(m_device)
            alpha_bar_t = alphas[t_int] if t_int < len(alphas) else torch.tensor(0.01, device=m_device)
        else:
            alpha_bar_t = torch.tensor(max(0.01, 1.0 - float(t_int) / 1000.0), device=m_device, dtype=m_dtype)

        with torch.enable_grad():
            noise_pred = model(x_in_flat, t_vec, encoder_hidden_states=embeds).sample
            sigma_t = torch.sqrt(1.0 - alpha_bar_t)
            score_all = -noise_pred / sigma_t
            score_bv = score_all.view(B, V_eff, score_all.shape[1], H, W)
            score_lat = score_bv[:, :, :4, :, :]

            n_samples = 10
            divs = []
            for k in range(n_samples):
                v = torch.randn_like(x_lat_leaf)
                y = (score_lat * v).sum()
                g = torch.autograd.grad(
                    y, x_lat_leaf,
                    retain_graph=(k < n_samples - 1),
                    create_graph=False, only_inputs=True
                )[0]
                divs.append((g * v).sum().item())

        n_elements = float(x_lat_leaf.numel())
        div_est = float(np.mean(divs)) / n_elements
        score_norm_sq = 0.5 * float((score_lat ** 2).sum().item()) / n_elements
        residual = abs(div_est + score_norm_sq)
        scores.append(max(0.0, 1.0 - residual / (abs(div_est) + abs(score_norm_sq) + 1e-8)))

    return float(np.mean(scores)) if scores else 0.0


def compute_score_matching_fd(intermediates, model, cond_context, scheduler,
                              plucker=None, delta=0.01):
    """A1-FD: Finite-difference variant for cross-model comparability."""
    x_inter = intermediates.get("x_inter", [])
    timesteps = intermediates.get("timesteps", [])
    if len(x_inter) < 3:
        return 0.0

    m_param = next(model.parameters())
    device, dtype = m_param.device, m_param.dtype

    test_indices = [len(timesteps) // 4, len(timesteps) // 2, 3 * len(timesteps) // 4]
    consistency_scores = []

    def _score_fn(x_lat, t_val):
        x_packed, embeds = _pack_unet_input(x_lat, model, cond_context, plucker)
        t_vec = torch.full((x_packed.shape[0],), t_val, device=device, dtype=torch.long)
        with torch.no_grad():
            pred = model(x_packed, t_vec, encoder_hidden_states=embeds).sample
        return pred

    for i in test_indices:
        if i >= len(x_inter):
            continue
        x_t = x_inter[i].to(device=device, dtype=dtype)
        t_val = int(timesteps[i])

        try:
            if hasattr(scheduler, "alphas_cumprod") and t_val < len(scheduler.alphas_cumprod):
                alpha_bar_t = float(scheduler.alphas_cumprod[t_val])
            else:
                alpha_bar_t = max(0.01, 1.0 - float(t_val) / 1000.0)
            sigma_t = float(np.sqrt(max(1e-8, 1.0 - alpha_bar_t)))

            base_pred = _score_fn(x_t, t_val)
            base_score = -base_pred / sigma_t
            score_norm_sq = 0.5 * torch.sum(base_score ** 2).item()

            n_samples = 10
            divergences = []
            for _ in range(n_samples):
                v = torch.randn_like(x_t)
                s_plus = -_score_fn(x_t + delta * v, t_val) / sigma_t
                s_minus = -_score_fn(x_t - delta * v, t_val) / sigma_t
                directional_deriv = (s_plus - s_minus) / (2 * delta)
                divergences.append(torch.sum(v * directional_deriv[:x_t.shape[0], :x_t.shape[1]]).item())

            n_elements = float(x_t.numel())
            avg_div = np.mean(divergences) / n_elements
            score_norm_sq = score_norm_sq / n_elements
            residual = abs(avg_div + score_norm_sq)
            consistency = max(0.0, 1.0 - residual / (abs(avg_div) + abs(score_norm_sq) + 1e-8))
            consistency_scores.append(consistency)

        except Exception as e:
            logger.warning(f"A1-FD failed at t={t_val}: {e}")
            continue

    return float(np.mean(consistency_scores)) if consistency_scores else 0.0


def compute_hessian_viz_fd(intermediates, model, cond_context, uncond_context,
                           plucker=None, timesteps_to_measure=None, n_dirs=20, delta=0.01):
    """Multi-direction FD Hessian magnitudes — comparable across all victim models."""
    x_inter = intermediates['x_inter']
    timesteps = intermediates['timesteps']
    if not x_inter or not timesteps:
        return {}

    if timesteps_to_measure is None:
        step = max(1, len(timesteps) // 5)
        timesteps_to_measure = list(range(0, len(timesteps), step))

    m_param = next(model.parameters())
    device, dtype = m_param.device, m_param.dtype
    results = {}

    with torch.no_grad():
        for t_idx in timesteps_to_measure:
            if t_idx >= len(x_inter):
                continue
            t_val = int(timesteps[t_idx])
            latents = x_inter[t_idx].to(device=device, dtype=dtype)

            x_packed_c, embeds_c = _pack_unet_input(latents, model, cond_context, plucker)
            x_packed_u, embeds_u = _pack_unet_input(latents, model, uncond_context, plucker)
            t_vec = torch.full((x_packed_c.shape[0],), t_val, device=device, dtype=torch.long)

            try:
                cond_mags, uncond_mags = [], []
                for _ in range(n_dirs):
                    # Restrict random directions to the first 4 latent channels
                    # to avoid diluting perturbation energy across Plücker/mask channels
                    v = torch.zeros_like(x_packed_c)
                    v[:, :4] = torch.randn(
                        x_packed_c.shape[0], 4, *x_packed_c.shape[2:],
                        device=device, dtype=dtype
                    )
                    v = v / torch.linalg.norm(v.reshape(-1)).clamp(min=1e-8)

                    s_plus = model(x_packed_c + delta * v, t_vec, encoder_hidden_states=embeds_c).sample
                    s_minus = model(x_packed_c - delta * v, t_vec, encoder_hidden_states=embeds_c).sample
                    cond_mags.append(torch.linalg.norm((s_plus - s_minus) / (2 * delta)).item())

                    s_plus = model(x_packed_u + delta * v, t_vec, encoder_hidden_states=embeds_u).sample
                    s_minus = model(x_packed_u - delta * v, t_vec, encoder_hidden_states=embeds_u).sample
                    uncond_mags.append(torch.linalg.norm((s_plus - s_minus) / (2 * delta)).item())

                results[t_val] = {'cond_magnitudes': cond_mags, 'uncond_magnitudes': uncond_mags}
            except Exception as e:
                logger.warning(f"Hessian-FD failed at t_idx={t_idx}: {e}")
                continue

    return results


def compute_hessian_score_proportionality_fixed(intermediates, hessian_viz, scheduler):
    """A2: Gaussian Local Structure — correlation of ||score||² with Hessian trace."""
    if len(intermediates['text_noise']) < 3 or len(hessian_viz) < 2:
        return 0.0

    score_mags_sq, hessian_traces = [], []
    for i, t_val in enumerate(intermediates['timesteps']):
        t_int = int(t_val)
        if t_int not in hessian_viz:
            continue
        if hasattr(scheduler, 'alphas_cumprod') and t_int < len(scheduler.alphas_cumprod):
            alpha_bar_t = scheduler.alphas_cumprod[t_int]
        else:
            alpha_bar_t = max(0.01, 1.0 - float(t_int) / 1000.0)

        noise_pred = intermediates['text_noise'][i]
        sigma_t = float(np.sqrt(max(1e-8, 1.0 - float(alpha_bar_t))))
        score_mags_sq.append(torch.norm(-noise_pred / sigma_t).item() ** 2)

        mags = hessian_viz[t_int].get('cond_magnitudes', [])
        if mags:
            hessian_traces.append(-np.sum(mags[:100]))
        else:
            score_mags_sq.pop()

    min_len = min(len(score_mags_sq), len(hessian_traces))
    if min_len < 3:
        return 0.0
    try:
        corr, _ = pearsonr(score_mags_sq[:min_len], hessian_traces[:min_len])
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def compute_sharpness_rank_persistence(hessian_viz):
    """A3: Rank persistence of top Hessian eigenvalues across timesteps."""
    if len(hessian_viz) < 3:
        return 0.0
    timesteps = sorted(hessian_viz.keys(), reverse=True)[:10]
    rankings = []
    for t in timesteps:
        mags = hessian_viz[t].get('cond_magnitudes', [])
        if len(mags) >= 10:
            rankings.append(np.argsort(mags)[::-1][:min(15, len(mags))])
    if len(rankings) < 2:
        return 0.0
    correlations = []
    for i in range(len(rankings) - 1):
        r1, r2 = rankings[i], rankings[i + 1]
        n = min(10, len(r1), len(r2))
        if n >= 5:
            try:
                corr, _ = spearmanr(r1[:n], r2[:n])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            except Exception:
                continue
    return float(np.mean(correlations)) if correlations else 0.0


def a3_hotspot_jaccard_persistence(hessian_viz, q=0.05, lags=(1, 2, 4)):
    """A3 alt: Jaccard similarity of top-q% eigenvalue hotspot locations across timesteps.
    Higher = more persistence = assumption holds."""
    if len(hessian_viz) < 3:
        return 0.0
    ts = sorted(hessian_viz.keys(), reverse=True)
    vecs = []
    for t in ts:
        mags = np.asarray(hessian_viz[t].get('cond_magnitudes', []), dtype=np.float64)
        if mags.size == 0:
            vecs.append(None)
            continue
        k = max(1, int(np.ceil(q * mags.size)))
        idx = np.argpartition(mags, -k)[-k:]
        vecs.append(set(idx.tolist()))
    scores = []
    for lag in lags:
        for i in range(len(vecs) - lag):
            A, B = vecs[i], vecs[i + lag]
            if A is None or B is None or len(A) == 0 or len(B) == 0:
                continue
            inter = len(A & B)
            union = len(A | B)
            scores.append(inter / union if union > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def a3_temporal_autocorr(hessian_viz, max_lag=2):
    """A3 alt: Temporal autocorrelation of eigenvalue magnitude patterns.
    Higher = more persistence = assumption holds."""
    ts = sorted(hessian_viz.keys(), reverse=True)
    rows = []
    for t in ts:
        v = np.asarray(hessian_viz[t].get('cond_magnitudes', []), dtype=np.float64)
        if v.size == 0:
            return 0.0
        rows.append(v)
    M = np.vstack(rows)
    if M.shape[0] < max_lag + 2:
        return 0.0
    M = (M - M.mean(axis=0, keepdims=True)) / (M.std(axis=0, keepdims=True) + 1e-9)
    acc = []
    for lag in range(1, max_lag + 1):
        c_list = []
        for t in range(M.shape[0] - lag):
            a, b = M[t], M[t + lag]
            c_list.append((a * b).mean())
        if c_list:
            acc.append(np.median(c_list))
    return float(np.mean(acc)) if acc else 0.0


def compute_improved_covariance_metrics(cond_noise, uncond_noise):
    """A4: Covariance commutativity between conditional and unconditional noise."""
    if len(cond_noise) < 10:
        return {'a4_eigval_corr': 0.0, 'a4_eigvec_align': 0.0,
                'a4_commutativity': 0.0, 'a4_mean_equality': 0.0}

    n = min(20, len(cond_noise))
    cond_mat = torch.stack([p.flatten() for p in cond_noise[-n:]]).cpu().numpy()
    uncond_mat = torch.stack([p.flatten() for p in uncond_noise[-n:]]).cpu().numpy()

    n_feat = min(2000, cond_mat.shape[1])
    if cond_mat.shape[1] > n_feat:
        idx = np.random.choice(cond_mat.shape[1], n_feat, replace=False)
        cond_mat, uncond_mat = cond_mat[:, idx], uncond_mat[:, idx]

    cond_mean, uncond_mean = np.mean(cond_mat, 0), np.mean(uncond_mat, 0)
    mean_diff = np.linalg.norm(cond_mean - uncond_mean)
    mean_scale = max(np.linalg.norm(cond_mean), np.linalg.norm(uncond_mean), 1e-8)
    mean_equality = max(0.0, 1.0 - mean_diff / mean_scale)

    try:
        reg = 1e-6
        cond_cov = np.cov(cond_mat, rowvar=False) + reg * np.eye(n_feat)
        uncond_cov = np.cov(uncond_mat, rowvar=False) + reg * np.eye(n_feat)

        cond_vals, cond_vecs = np.linalg.eigh(cond_cov)
        uncond_vals, uncond_vecs = np.linalg.eigh(uncond_cov)

        for vals, vecs in [(cond_vals, cond_vecs), (uncond_vals, uncond_vecs)]:
            order = np.argsort(np.abs(vals))[::-1]
            vals[:] = vals[order]
            vecs[:] = vecs[:, order]

        n_eigs = min(50, len(cond_vals), len(uncond_vals))
        eigval_corr, _ = pearsonr(cond_vals[:n_eigs], uncond_vals[:n_eigs])
        eigval_corr = abs(eigval_corr) if not np.isnan(eigval_corr) else 0.0

        n_vecs = min(20, cond_vecs.shape[1])
        svd_vals = np.linalg.svd(cond_vecs[:, :n_vecs].T @ uncond_vecs[:, :n_vecs], compute_uv=False)
        eigvec_align = float(np.mean(svd_vals))

        commutator = cond_cov @ uncond_cov - uncond_cov @ cond_cov
        ref_norm = max(np.linalg.norm(cond_cov, 'fro'), np.linalg.norm(uncond_cov, 'fro'), 1e-8)
        commutativity = 1.0 / (1.0 + np.linalg.norm(commutator, 'fro') / ref_norm)

    except np.linalg.LinAlgError:
        eigval_corr = eigvec_align = commutativity = 0.0

    return {
        'a4_eigval_corr': float(eigval_corr),
        'a4_eigvec_align': float(eigvec_align),
        'a4_commutativity': float(commutativity),
        'a4_mean_equality': float(mean_equality),
    }


def compute_prior_structure_deviation(initial_latents):
    """A5: KS test of initial noise against N(0,1)."""
    if initial_latents is None:
        return 0.0
    flat = initial_latents.flatten().cpu().numpy()
    if len(flat) < 30 or np.std(flat) < 1e-8:
        return 0.0
    std_flat = (flat - np.mean(flat)) / np.std(flat)
    try:
        _, p_val = kstest(std_flat, 'norm')
        return float(p_val)
    except Exception:
        return 0.0


def compute_score_explosion_indicator(uncond_noise, text_noise):
    """A7: Ratio of max to mean noise norm (boundary regularity)."""
    all_norms = [torch.norm(p).item()
                 for lst in [uncond_noise, text_noise]
                 for p in lst
                 if torch.norm(p).item() > 1e-10]
    if len(all_norms) < 5:
        return 0.0
    explosion = np.max(all_norms) / (np.mean(all_norms) + 1e-8)
    return float(min(1.0, max(0.0, (explosion - 1.0) / 10.0)))


def compute_score_sensitivity_to_input_perturbations(intermediates, model, cond_context, uncond_context):
    """Memorization detector: coefficient of variation of noise diff norms."""
    try:
        uncond_noise = intermediates['uncond_noise']
        text_noise = intermediates['text_noise']
        if len(uncond_noise) < 2 or len(text_noise) < 2:
            return 0.0
        noise_diffs = [torch.norm(text_noise[i] - uncond_noise[i]).item()
                       for i in range(min(len(uncond_noise), len(text_noise)))]
        if len(noise_diffs) < 2:
            return 0.0
        return float(np.std(noise_diffs) / (np.mean(noise_diffs) + 1e-8))
    except Exception:
        return 0.0


def compute_magnitude_bias_test(intermediates, **kwargs):
    """Direct test for systematic magnitude bias in predictions."""
    text_noise = intermediates['text_noise']
    uncond_noise = intermediates['uncond_noise']
    if len(text_noise) < 3:
        return 0.0
    text_mean = np.mean([torch.norm(p).item() for p in text_noise])
    uncond_mean = np.mean([torch.norm(p).item() for p in uncond_noise])
    return abs(text_mean - uncond_mean) / (uncond_mean + 1e-8)


def compute_prediction_consistency_test(intermediates, **kwargs):
    """Coefficient of variation of conditional noise norms across timesteps."""
    text_noise = intermediates['text_noise']
    if len(text_noise) < 5:
        return 0.0
    mags = [torch.norm(p).item() for p in text_noise]
    return float(np.std(mags) / (np.mean(mags) + 1e-8))


# =====================================================================================
# --- PIPELINE LOADING ---
# =====================================================================================

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def setup_camera_parameters(opt, device, num_views=4, elevation=10.0, distance=1.4):
    """Setup camera parameters for multi-view generation."""
    fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5], device=device).float()

    elevations = torch.tensor([-elevation] * num_views, device=device).deg2rad().float()
    azimuths = torch.tensor([0., 90., 180., 270.][:num_views], device=device).deg2rad().float()
    radius = torch.tensor([distance] * num_views, device=device).float()

    input_C2W = geo_util.orbit_camera(elevations, azimuths, radius, is_degree=False)
    input_C2W[:, :3, 1:3] *= -1
    input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(input_C2W.shape[0], 1)

    plucker = None
    if opt.input_concat_plucker:
        H = W = opt.input_res
        plucker, _ = geo_util.plucker_ray(H, W, input_C2W.unsqueeze(0), input_fxfycxcy.unsqueeze(0))
        plucker = plucker.squeeze(0)
        if opt.view_concat_condition:
            plucker = torch.cat([plucker[0:1, ...], plucker], dim=0)

    return {
        'input_C2W': input_C2W,
        'input_fxfycxcy': input_fxfycxcy,
        'plucker': plucker,
        'num_views': num_views,
    }


def load_diffsplat_pipeline(config_file, device, ckpt_iter=13020):
    """Load DiffSplat pipeline."""
    configs = util.get_configs(config_file, [])
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)

    in_channels = 4
    if opt.input_concat_plucker:
        in_channels += 6
    if opt.input_concat_binary_mask:
        in_channels += 1

    unet_kwargs = {
        "sample_size": opt.input_res // 8,
        "in_channels": in_channels,
        "zero_init_conv_in": opt.zero_init_conv_in,
        "view_concat_condition": opt.view_concat_condition,
        "input_concat_plucker": opt.input_concat_plucker,
        "input_concat_binary_mask": opt.input_concat_binary_mask,
    }

    tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16")
    vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")
    gsvae = GSAutoencoderKL(opt)
    gsrecon = GSRecon(opt)

    noise_scheduler = DDIMScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
    if opt.common_tricks:
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config.rescale_betas_zero_snr = True
    if opt.prediction_type is not None:
        noise_scheduler.config.prediction_type = opt.prediction_type
    if opt.beta_schedule is not None:
        noise_scheduler.config.beta_schedule = opt.beta_schedule

    exp_dir = "out/gsdiff_gobj83k_sd15__render"
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    path = os.path.join(ckpt_dir, f"{ckpt_iter:06d}")
    assert os.path.exists(path), f"Checkpoint path not found: {path}"

    os.system(f"python3 extensions/merge_safetensors.py {path}/unet_ema")
    unet, loading_info = UNetMV2DConditionModel.from_pretrained_new(
        path, subfolder="unet_ema",
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
        output_loading_info=True, **unet_kwargs
    )

    gsvae = util.load_ckpt(os.path.join("out", "gsvae_gobj265k_sd", "checkpoints"), -1, None, gsvae)
    gsrecon = util.load_ckpt(os.path.join("out", "gsrecon_gobj265k_cnp_even4", "checkpoints"), -1, None, gsrecon)

    for m in [text_encoder, vae, gsvae, gsrecon, unet]:
        m.to(device).eval().requires_grad_(False)

    pipeline = StableMVDiffusionPipeline(
        text_encoder=text_encoder, tokenizer=tokenizer,
        vae=vae, unet=unet, scheduler=noise_scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    logger.info("DiffSplat pipeline loaded successfully")
    return pipeline, gsvae, gsrecon, opt


# =====================================================================================
# --- EVALUATOR ---
# =====================================================================================

class DiffSplatGeometricEvaluator:
    """Runs DiffSplat generation and collects intermediates for geometric analysis."""

    def __init__(self, pipeline, gsvae, gsrecon, opt):
        self.pipeline = pipeline
        self.gsvae = gsvae
        self.gsrecon = gsrecon
        self.opt = opt
        self.device = pipeline.device

    def run_and_collect(self, prompt, camera_params, seed=42, guidance_scale=7.5,
                        num_inference_steps=50):
        torch.manual_seed(seed)

        plucker = camera_params.get('plucker')
        num_views = camera_params.get('num_views', 4)

        pipeline_output = self.pipeline(
            None, prompt=prompt, negative_prompt='',
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            triangle_cfg_scaling=False,
            min_guidance_scale=1.0, max_guidance_scale=guidance_scale,
            output_type="latent", eta=1.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            plucker=plucker, num_views=num_views,
            init_std=0.1, init_noise_strength=0.95, init_bg=2.5,
            guess_mode=False, controlnet_conditioning_scale=1.0,
        )

        intermediates = {
            'uncond_noise': getattr(pipeline_output, 'uncond_noise', []),
            'text_noise': getattr(pipeline_output, 'text_noise', []),
            'x_inter': getattr(pipeline_output, 'x_inter', []),
            'timesteps': getattr(pipeline_output, 'timesteps', []),
            'initial_latents': getattr(pipeline_output, 'initial_latents', None),
        }

        prompt_embeds = self.pipeline._encode_prompt(
            prompt=prompt, device=self.device,
            num_images_per_prompt=1, do_classifier_free_guidance=True,
            negative_prompt=""
        )
        uncond_embeds, cond_embeds = prompt_embeds.chunk(2)

        latents = pipeline_output.images
        latents = latents / self.gsvae.scaling_factor + self.gsvae.shift_factor
        render_outputs = self.gsvae.decode_and_render_gslatents(
            self.gsrecon, latents,
            camera_params['input_C2W'].unsqueeze(0),
            camera_params['input_fxfycxcy'].unsqueeze(0),
            height=256, width=256, opacity_threshold=0.0,
        )
        rendered_images = render_outputs["image"].squeeze(0)

        return intermediates, cond_embeds, uncond_embeds, rendered_images, plucker


# =====================================================================================
# --- METRIC AGGREGATION ---
# =====================================================================================

def compute_all_geometric_metrics(intermediates, cond_context, uncond_context,
                                   model, scheduler, plucker=None):
    """Compute all geometric assumption proxies for a single generation."""
    results = {}

    ndn_metric = AdaptedNoiseDiffNormMetric()
    results.update(ndn_metric.measure(intermediates))

    hessian_metric = HessianMetric(timesteps_to_measure=list(range(0, N_INFERENCE_STEPS, 5)))
    hessian_result = hessian_metric.measure(
        intermediates=intermediates, model=model,
        conditioning_context=cond_context, unconditioning_context=uncond_context,
        plucker=plucker
    )
    hessian_viz = hessian_result.get('visualizations', {})

    hessian_viz_fd = compute_hessian_viz_fd(
        intermediates, model, cond_context, uncond_context, plucker)

    results['a1_score_matching_consistency'] = compute_score_matching_fixed(
        intermediates, model, cond_context, scheduler, plucker)
    results['a1_score_matching_consistency_fd'] = compute_score_matching_fd(
        intermediates, model, cond_context, scheduler, plucker)

    results['a2_score_hessian_proportionality'] = compute_hessian_score_proportionality_fixed(
        intermediates, hessian_viz, scheduler)
    results['a2_score_hessian_proportionality_fd'] = compute_hessian_score_proportionality_fixed(
        intermediates, hessian_viz_fd, scheduler)

    results['a3_sharpness_rank_persistence'] = compute_sharpness_rank_persistence(hessian_viz)
    results['a3_sharpness_rank_persistence_fd'] = compute_sharpness_rank_persistence(hessian_viz_fd)
    results['a3_hotspot_jaccard'] = a3_hotspot_jaccard_persistence(hessian_viz, q=0.05, lags=(1, 2, 4))
    results['a3_temporal_autocorr'] = a3_temporal_autocorr(hessian_viz, max_lag=2)

    results.update(compute_improved_covariance_metrics(
        intermediates['text_noise'], intermediates['uncond_noise']))

    results['a5_prior_deviation_pval'] = compute_prior_structure_deviation(
        intermediates.get('initial_latents'))

    results['a7_score_explosion_indicator'] = compute_score_explosion_indicator(
        intermediates['uncond_noise'], intermediates['text_noise'])

    results['a1_magnitude_bias'] = compute_magnitude_bias_test(intermediates)
    results['a1_prediction_consistency'] = compute_prediction_consistency_test(intermediates)
    results['score_sensitivity_to_input_perturbations'] = compute_score_sensitivity_to_input_perturbations(
        intermediates, model, cond_context, uncond_context)

    results['hessian_by_timestep'] = {
        t: data.get('cond_magnitudes', []) for t, data in hessian_viz.items()}

    return results


# =====================================================================================
# --- PRINTING / REPORTING ---
# =====================================================================================

def print_results_table(mem_results, unmem_results, model_name="DiffSplat"):
    print(f"\n{'='*120}")
    print(f"GEOMETRIC ASSUMPTION VALIDATION: {model_name.upper()}")
    print(f"{'='*120}")

    def stats(results, key):
        vals = [r.get(key, 0) for r in results if r and key in r and not np.isnan(r.get(key, 0))]
        return (np.mean(vals), np.std(vals)) if vals else (0.0, 0.0)

    tests = [
        ('(A1) Score Matching Consistency',    'a1_score_matching_consistency',       '↓'),
        ('(A1) Score Matching Consistency FD', 'a1_score_matching_consistency_fd',    '↓'),
        ('(A1) Magnitude Bias',               'a1_magnitude_bias',                   '↑'),
        ('(A1) Prediction Consistency',        'a1_prediction_consistency',           '↑'),
        ('(A2) Score-Hessian Proportionality', 'a2_score_hessian_proportionality',   '↓'),
        ('(A2) Score-Hessian Prop. FD',        'a2_score_hessian_proportionality_fd','↓'),
        ('(A3) Rank Persistence',              'a3_sharpness_rank_persistence',       '↓'),
        ('(A3) Rank Persistence FD',           'a3_sharpness_rank_persistence_fd',    '↓'),
        ('(A3) Hotspot Jaccard',               'a3_hotspot_jaccard',                  '↑'),
        ('(A3) Temporal Autocorr',             'a3_temporal_autocorr',                '↑'),
        ('(A4) Eigenspace Alignment',          'a4_eigvec_align',                     '↓'),
        ('(A5) Gaussian Prior p-value',        'a5_prior_deviation_pval',             '↓'),
        ('(A7) Score Explosion',               'a7_score_explosion_indicator',        '↑'),
        ('NDN Mean',                           'noise_diff_norm_mean',                '↑'),
        ('Score Sensitivity',                  'score_sensitivity_to_input_perturbations', '↑'),
    ]

    print(f"{'Assumption Test':<45} | {'Memorized':<18} | {'Unmemorized':<18} | Δ")
    print("-" * 120)
    for name, key, direction in tests:
        mm, ms = stats(mem_results, key)
        um, us = stats(unmem_results, key)
        print(f"{name:<45} | {mm:.3f}±{ms:.3f}        | {um:.3f}±{us:.3f}        | {direction}")
    print(f"{'='*120}")


# =====================================================================================
# --- RUNNER ---
# =====================================================================================

def process_category(evaluator, category, csv_path, camera_params, n_prompts):
    logger.info(f"Processing {category.upper()} prompts")
    df = pd.read_csv(csv_path, sep=';').head(n_prompts)
    category_results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"DiffSplat {category}"):
        prompt = str(row['Caption']).strip()
        if not prompt:
            continue
        try:
            intermediates, cond_embeds, uncond_embeds, rendered_images, plucker = \
                evaluator.run_and_collect(
                    prompt, camera_params, seed=idx, guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=N_INFERENCE_STEPS)
            metrics = compute_all_geometric_metrics(
                intermediates, cond_embeds, uncond_embeds,
                evaluator.pipeline.unet, evaluator.pipeline.scheduler, plucker)
            category_results.append(metrics)
            cleanup_memory()
        except Exception as e:
            logger.error(f"Failed on sample {idx}: {e}")
            continue

    return category_results


def main():
    """Measure geometric assumption proxies for DiffSplat.

    Usage:
      python assumption_proxies_runner.py
      python assumption_proxies_runner.py --config configs/gsdiff_sd15.yaml --ckpt_iter 13020
      python assumption_proxies_runner.py --n_prompts 20
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Measure geometric assumptions for DiffSplat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/gsdiff_sd15.yaml",
                        help="DiffSplat config file path")
    parser.add_argument("--ckpt_iter", type=int, default=13020,
                        help="Checkpoint iteration to load")
    parser.add_argument("--n_prompts", type=int, default=N_PROMPTS,
                        help=f"Number of prompts per category (default: {N_PROMPTS})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    pipeline, gsvae, gsrecon, opt = load_diffsplat_pipeline(args.config, device, args.ckpt_iter)
    evaluator = DiffSplatGeometricEvaluator(pipeline, gsvae, gsrecon, opt)
    camera_params = setup_camera_parameters(opt, device, num_views=NUM_VIEWS)

    all_results = {}
    for category, path in LAION_PATHS.items():
        if not os.path.exists(path):
            logger.warning(f"Dataset not found: {path}, skipping.")
            continue
        all_results[category] = process_category(
            evaluator, category, path, camera_params, args.n_prompts)

    mem_results = all_results.get('memorized', [])
    unmem_results = all_results.get('unmemorized', [])

    print_results_table(mem_results, unmem_results, model_name="DiffSplat")


if __name__ == "__main__":
    main()
