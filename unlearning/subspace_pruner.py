"""
SubspacePruner — portable version.
Identifies FFN weights in the memorisation subspace and zeroes them.
Reference: "Memorized Images in Diffusion Models Share a Subspace".
"""
from copy import deepcopy
from typing import Dict, List
import torch
import torch.nn as nn
from .adapter import DiffusionModelAdapter


class SubspacePruner:
    """
    Prunes FFN weights that are disproportionately activated by memorised prompts.
    Works with any model that implements DiffusionModelAdapter.
    """

    def __init__(self, adapter: DiffusionModelAdapter, sparsity: float = 0.001):
        self.adapter = adapter
        self.sparsity = sparsity
        self._ffn_layers = adapter.get_attention_layers("ffn")
        print(f"SubspacePruner: found {len(self._ffn_layers)} FFN layers.")

    @torch.no_grad()
    def _collect_activations(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Run one denoising step and collect FFN input activations."""
        activations: Dict[str, List] = {n: [] for n in self._ffn_layers}
        hooks = []
        for name, layer in self._ffn_layers.items():
            def _hook(mod, inp, out, _n=name):
                activations[_n].append(inp[0].detach().cpu())
            hooks.append(layer.register_forward_hook(_hook))

        batch = len(prompts)
        text_emb = self.adapter.encode_text(prompts)
        c = self.adapter.make_conditioning(text_emb)
        t = torch.tensor([999] * batch, device=self.adapter.device)
        x = self.adapter.randn_latent(batch)
        self.adapter.apply_model(x, t, c)

        for h in hooks:
            h.remove()

        result = {}
        for name, act_list in activations.items():
            act = act_list[0]  # (batch * seq_len, in_dim)
            in_dim = act.shape[-1]
            act = act.view(batch, -1, in_dim).mean(dim=1)  # (batch, in_dim)
            result[name] = act.T  # (in_dim, batch)
        return result

    def find_memorization_subspace(self, memorised_prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Return pruning masks {layer_name: bool_tensor} for memorised prompts."""
        mem_acts = self._collect_activations(memorised_prompts)
        null_acts = self._collect_activations([""] * len(memorised_prompts))

        masks = {}
        for name, layer in self._ffn_layers.items():
            W = layer.weight.data.float()
            H_mem = mem_acts[name].to(self.adapter.device, dtype=torch.float32)
            H_null = null_acts[name].to(self.adapter.device, dtype=torch.float32)

            S_mem = W.abs() * H_mem.norm(dim=1)
            S_null = W.abs() * H_null.norm(dim=1)

            s_abs = max(1, int(W.shape[1] * self.sparsity))
            threshold = torch.topk(S_mem, s_abs, dim=1).values[:, -1].unsqueeze(1)
            masks[name] = (S_mem >= threshold) & (S_mem > S_null)
        return masks

    def prune_model_weights(self, masks: Dict[str, torch.Tensor]):
        """Zero out the identified weights in-place."""
        for name, mask in masks.items():
            self._ffn_layers[name].weight.data[mask] = 0.0
        print("SubspacePruner: pruning complete.")
