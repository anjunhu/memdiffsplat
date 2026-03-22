"""
InvMM (Inversion-based Memorization Measure) for DiffSplat.
Reference: https://github.com/Maryeon/InvMM, https://arxiv.org/abs/2405.05846
Adapted for DiffSplat's StableDiffusionPipeline-based multi-view pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from torchvision import transforms as T

from .base import BaseMetric


class InvMMMetric(BaseMetric):

    def __init__(self, train_num_steps=500, lr=1e-1, num_samples=4,
                 init_kl_weight=1.0, observation_cycle=50,
                 weight_increment=1e-3, progress_threshold=1e-3, verbose=True,
                 prompt_inversion=True, sscd_early_stop=True,
                 num_tokens=75, tau=2.0, sscd_beta=0.5,
                 sample_num_noise=4, sscd_model_path="sscd_disc_mixup.torchscript.pt",
                 similarity_threshold=0.5):
        super().__init__()
        self._steps = train_num_steps
        self._lr = lr
        self._num_samples = num_samples
        self._init_kl_weight = init_kl_weight
        self._C = observation_cycle
        self._delta = weight_increment
        self._ksi = progress_threshold
        self._verbose = verbose
        self._prompt_inversion = prompt_inversion
        self._sscd_early_stop = sscd_early_stop
        self._num_tokens = num_tokens
        self._tau = tau
        self._beta = sscd_beta
        self._sample_num_noise = sample_num_noise
        self._sscd_model_path = sscd_model_path
        self._sscd_model = None
        self._sscd_transforms = T.Compose([
            T.Resize([320, 320]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def name(self):
        return "InvMM_Score"

    @property
    def metric_type(self):
        return "per_seed"

    @property
    def requires_model(self):
        return True

    @property
    def requires_intermediates(self):
        return True

    def _load_sscd(self, device):
        if self._sscd_model is None:
            import os
            if not os.path.exists(self._sscd_model_path):
                print(f"[InvMM] SSCD not found at {self._sscd_model_path}, disabling early stop")
                self._sscd_early_stop = False
                return
            self._sscd_model = torch.jit.load(self._sscd_model_path).to(device).eval()

    @torch.no_grad()
    def _sscd_similarity(self, img_a, img_b):
        a = self._sscd_model(self._sscd_transforms(img_a))
        b = self._sscd_model(self._sscd_transforms(img_b))
        return a.mm(b.T).squeeze()

    def _get_alpha_bar(self, scheduler, t, shape):
        ab = scheduler.alphas_cumprod.to(t.device)[t]
        while ab.dim() < len(shape):
            ab = ab.unsqueeze(-1)
        return ab.sqrt(), (1 - ab).sqrt()

    def _compute_kl(self, mu, logvar):
        return 0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1)

    def _denoising_loss(self, unet, scheduler, x_start, noise, cond):
        """x0-prediction denoising error (Eq. 12)."""
        B = noise.shape[0]
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=noise.device)
        sqrt_ab, sqrt_1mab = self._get_alpha_bar(scheduler, t, noise.shape)
        x_t = sqrt_ab * x_start.expand_as(noise) + sqrt_1mab * noise
        with torch.no_grad():
            model_dtype = next(unet.parameters()).dtype
            x_in = x_t.to(model_dtype)
            # DiffSplat UNet conv_in expects more channels than the raw latent
            # (e.g. 10 = 4 latent + 6 plucker). Pad with zeros for missing channels.
            expected_in = unet.config.in_channels
            if x_in.shape[1] < expected_in:
                pad = torch.zeros(x_in.shape[0], expected_in - x_in.shape[1],
                                  *x_in.shape[2:], dtype=x_in.dtype, device=x_in.device)
                x_in = torch.cat([x_in, pad], dim=1)
            pred_noise = unet(x_in, t, encoder_hidden_states=cond.to(model_dtype)).sample
            # Trim output back to latent channels if UNet outputs extra channels
            pred_noise = pred_noise[:, :x_t.shape[1]]
        pred_x0 = (x_t - sqrt_1mab * pred_noise.float()) / sqrt_ab.clamp(min=1e-8)
        return F.mse_loss(pred_x0, x_start.expand_as(pred_x0), reduction="none").mean(dim=list(range(1, pred_x0.dim()))).mean()

    def _get_prompt_cond(self, pipeline, log_coeffs, device):
        voc_emb = pipeline.text_encoder.get_input_embeddings().weight.detach()
        coeffs = F.gumbel_softmax(log_coeffs.expand(self._num_samples, -1, -1),
                                  hard=False, tau=self._tau)
        tok_emb = torch.bmm(coeffs, voc_emb.expand(self._num_samples, -1, -1))

        embed_layer = pipeline.text_encoder.text_model.embeddings
        def hook_fn(module, args, kwargs):
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            if input_ids is not None:
                inputs_embeds = module.token_embedding(input_ids)
                inputs_embeds[:, 1:1+self._num_tokens] = tok_emb
                kwargs["inputs_embeds"] = inputs_embeds
                kwargs.pop("input_ids", None)
            return args, kwargs

        h = embed_layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
        try:
            tok = pipeline.tokenizer([""] * self._num_samples, padding="max_length",
                                     max_length=pipeline.tokenizer.model_max_length,
                                     return_tensors="pt").input_ids.to(device)
            out = pipeline.text_encoder(tok)
        finally:
            h.remove()
        return out[0]

    @torch.no_grad()
    def _sample_and_check_sscd(self, pipeline, mu, logvar, ref_img, cond, device):
        if not hasattr(self, '_ddim_scheduler') or self._ddim_scheduler is None:
            from diffusers import DDIMScheduler
            self._ddim_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            self._ddim_scheduler.set_timesteps(50)
        scheduler = self._ddim_scheduler
        model_dtype = next(pipeline.unet.parameters()).dtype
        n = torch.randn(self._sample_num_noise, *mu.shape[1:], device=device) * logvar.div(2).exp() + mu
        latents = n
        for t_step in scheduler.timesteps:
            t_batch = t_step.expand(latents.shape[0]).to(device)
            noise_pred = pipeline.unet(latents.to(model_dtype), t_batch, encoder_hidden_states=cond[:latents.shape[0]].to(model_dtype)).sample
            latents = scheduler.step(noise_pred.float(), t_step, latents).prev_sample
        images = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor).sample
        images = ((images + 1.0) / 2.0).clamp(0, 1)
        sims = self._sscd_similarity(ref_img.expand_as(images), images)
        return torch.any(sims >= self._beta).item()

    def _run_inversion(self, pipeline, unet, scheduler, x_start, cond, device, ref_img=None):
        mu = torch.zeros_like(x_start, requires_grad=True)
        logvar = torch.zeros_like(x_start, requires_grad=True)
        params = [{"params": [mu]}, {"params": [logvar]}]

        log_coeffs = None
        if self._prompt_inversion and hasattr(pipeline, "text_encoder"):
            vocab_size = pipeline.text_encoder.get_input_embeddings().weight.shape[0]
            log_coeffs = torch.zeros(self._num_tokens, vocab_size, device=device, requires_grad=True)
            params.append({"params": [log_coeffs]})

        opt = torch.optim.Adam(params, lr=self._lr)
        kl_weight = self._init_kl_weight
        p_loss_prev = float("inf")
        p_loss_history = []

        for step in range(self._steps):
            opt.zero_grad()
            noise = torch.randn(self._num_samples, *x_start.shape[1:], device=device) * logvar.div(2).exp() + mu
            step_cond = self._get_prompt_cond(pipeline, log_coeffs, device) if log_coeffs is not None else cond

            p_loss = self._denoising_loss(unet, scheduler, x_start, noise, step_cond)
            r_loss = self._compute_kl(mu, logvar)
            (p_loss + kl_weight * r_loss).backward()
            opt.step()

            p_loss_history.append(p_loss.item())
            p_loss_now = np.mean(p_loss_history[-100:])

            if step > 0 and (step + 1) % self._C == 0:
                if p_loss_prev - p_loss_now < self._ksi:
                    kl_weight = max(kl_weight / 2, 0)
                else:
                    kl_weight += self._delta
                p_loss_prev = p_loss_now

                if self._sscd_early_stop and ref_img is not None and self._sscd_model is not None:
                    check_cond = step_cond.detach() if log_coeffs is not None else cond
                    if self._sample_and_check_sscd(pipeline, mu.detach(), logvar.detach(), ref_img, check_cond, device):
                        if self._verbose:
                            print(f"[InvMM] SSCD convergence at step {step}")
                        break
                    else:
                        kl_weight += self._delta

            if self._verbose and step % max(1, self._steps // 5) == 0:
                print(f"[InvMM] step {step}/{self._steps}  "
                      f"p_loss={p_loss.item():.6f}  kl={r_loss.item():.6f}  λ={kl_weight:.4f}")

        with torch.no_grad():
            return self._compute_kl(mu, logvar).item()

    def measure(self, model=None, images=None, latents=None, **kwargs) -> Dict:
        if model is None:
            return {"invmm_score": None, "error": "requires model"}

        pipeline = model
        unet = getattr(pipeline, "unet", None)
        scheduler = getattr(pipeline, "scheduler", None)
        if unet is None or scheduler is None:
            return {"invmm_score": None, "error": "pipeline missing unet/scheduler"}

        device = next(unet.parameters()).device
        if self._sscd_early_stop:
            self._load_sscd(device)

        x_start = None
        ref_img = None
        if latents is not None:
            x_start = latents
        elif images is not None and hasattr(pipeline, "vae"):
            with torch.no_grad():
                img = images[:, 0] if images.dim() == 5 else images
                img = img[:1].to(device, dtype=pipeline.vae.dtype)
                ref_img = img.clone() if img.min() >= 0 else ((img + 1) / 2).clamp(0, 1)
                if img.min() >= 0:
                    img = img * 2 - 1
                posterior = pipeline.vae.encode(img)
                x_start = posterior.latent_dist.mode() if hasattr(posterior, "latent_dist") else posterior.sample()

        if x_start is None:
            return {"invmm_score": None, "error": "no latent available"}

        x_start = x_start[:1].to(device).detach()

        cond = None
        if not self._prompt_inversion and hasattr(pipeline, "text_encoder") and hasattr(pipeline, "tokenizer"):
            with torch.no_grad():
                tok = pipeline.tokenizer([""], padding="max_length",
                                         max_length=pipeline.tokenizer.model_max_length,
                                         return_tensors="pt").input_ids.to(device)
                cond = pipeline.text_encoder(tok)[0].expand(self._num_samples, -1, -1)

        invmm = self._run_inversion(pipeline, unet, scheduler, x_start, cond, device, ref_img=ref_img)
        if self._verbose:
            print(f"[InvMM] Final score: {invmm:.6f}")
        return {"invmm_score": invmm, "success_rate": 1.0}
