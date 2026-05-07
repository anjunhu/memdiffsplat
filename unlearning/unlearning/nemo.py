"""
NeMo — Finding Nemo: Memorisation Neurons (portable version).
Operates via DiffusionModelAdapter; no model-specific imports.
Reference: Hintersdorf et al., 2024.
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from .adapter import DiffusionModelAdapter


class NeMoEditor:
    """
    Finds and deactivates cross-attention value neurons that drive memorisation.
    Works with any model that implements DiffusionModelAdapter.
    """

    def __init__(self,
                 adapter: DiffusionModelAdapter,
                 non_mem_prompts: List[str],
                 ssim_threshold: float = 0.5,
                 initial_theta: float = 6.0,
                 min_theta: float = 1.2,
                 theta_step: float = 0.25,
                 initial_k: int = 5,
                 k_step: int = 1,
                 num_ssim_seeds: int = 10):
        self.adapter = adapter
        self.ssim_threshold = ssim_threshold
        self.initial_theta = initial_theta
        self.min_theta = min_theta
        self.theta_step = theta_step
        self.initial_k = initial_k
        self.k_step = k_step
        self.num_ssim_seeds = num_ssim_seeds
        self._hooks: List = []

        self._value_layers = self._get_value_layers()
        print(f"NeMo: pre-computing activation stats on {len(non_mem_prompts)} non-memorised prompts...")
        self._non_mem_stats = self._precompute_stats(non_mem_prompts)
        print("NeMoEditor ready.")

    def _get_value_layers(self) -> Dict[str, nn.Module]:
        """Get cross-attention to_v layers via the adapter."""
        layers = self.adapter.get_attention_layers("cross")
        return {k: v for k, v in layers.items() if k.endswith("to_v")}

    @torch.no_grad()
    def _collect_activations(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        activations: Dict[str, List] = {n: [] for n in self._value_layers}
        hooks = []
        for name, layer in self._value_layers.items():
            def _hook(mod, inp, out, _n=name):
                activations[_n].append(out.detach())
            hooks.append(layer.register_forward_hook(_hook))

        text_emb = self.adapter.encode_text(prompts)
        c = self.adapter.make_conditioning(text_emb)
        t = torch.tensor([999] * len(prompts), device=self.adapter.device)
        x = self.adapter.randn_latent(len(prompts))
        self.adapter.apply_model(x, t, c)

        for h in hooks:
            h.remove()
        return {n: torch.cat(v, dim=0) for n, v in activations.items()}

    def _precompute_stats(self, prompts: List[str]) -> Dict[str, Dict]:
        acts = self._collect_activations(prompts)
        stats = {}
        for name, act in acts.items():
            abs_act = act.abs().mean(dim=1)  # avg over token dim
            stats[name] = {"mean": abs_act.mean(0), "std": abs_act.std(0)}
        return stats

    @torch.no_grad()
    def _noise_diffs(self, prompt: str, neurons: Optional[Dict] = None) -> torch.Tensor:
        if neurons:
            self.register_hooks(neurons)
        text_emb = self.adapter.encode_text([prompt])
        null_emb = self.adapter.null_embedding(1)
        c = self.adapter.make_conditioning(text_emb)
        uc = self.adapter.make_conditioning(null_emb)
        t = torch.tensor([999], device=self.adapter.device)
        diffs = []
        for seed in range(self.num_ssim_seeds):
            gen = torch.Generator(device=self.adapter.device).manual_seed(seed)
            x = self.adapter.randn_latent(1, generator=gen)
            noise = self.adapter.apply_model(x, t, c)
            diffs.append(noise - x)
        if neurons:
            self.remove_hooks()
        return torch.cat(diffs, dim=0)

    def _mem_score(self, prompt: str, neurons: Optional[Dict] = None) -> float:
        diffs = self._noise_diffs(prompt, neurons)
        max_s = 0.0
        for i in range(len(diffs)):
            for j in range(i + 1, len(diffs)):
                a = (diffs[i] - diffs[i].min()) / (diffs[i].max() - diffs[i].min() + 1e-8)
                b = (diffs[j] - diffs[j].min()) / (diffs[j].max() - diffs[j].min() + 1e-8)
                s = ssim(a.unsqueeze(0), b.unsqueeze(0), data_range=1.0).item()
                max_s = max(max_s, s)
        return max_s

    def find_neurons(self, prompt: str) -> Dict[str, List[int]]:
        """Return {layer_name: [neuron_indices]} to block for this prompt."""
        acts = self._collect_activations([prompt])
        theta, k = self.initial_theta, self.initial_k
        candidates: Dict[str, List[int]] = {}

        while True:
            candidates = {}
            for name in self._value_layers:
                mean_abs = acts[name].abs().mean(dim=1).squeeze()
                z = (mean_abs - self._non_mem_stats[name]["mean"]) / (
                    self._non_mem_stats[name]["std"] + 1e-6)
                ood = (z > theta).nonzero().squeeze(-1).tolist()
                topk = torch.topk(mean_abs, min(k, len(mean_abs))).indices.tolist()
                candidates[name] = sorted(set(ood + topk))

            score = self._mem_score(prompt, candidates)
            if score < self.ssim_threshold or theta <= self.min_theta:
                break
            theta -= self.theta_step
            k += self.k_step

        return candidates

    def register_hooks(self, neurons: Dict[str, List[int]]):
        self.remove_hooks()
        for name, layer in self._value_layers.items():
            if name in neurons and neurons[name]:
                idx = neurons[name]
                def _hook(mod, inp, out, _idx=idx):
                    out[:, :, _idx] = 0.0
                    return out
                self._hooks.append(layer.register_forward_hook(_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
