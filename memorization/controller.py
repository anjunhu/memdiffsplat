# DiffSplat/memorization/controller.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any, Dict, Optional, Tuple


class RecordingAttnProcessor(nn.Module):
    """
    Wrapper around the *original* attention processor.
    - Calls the original processor to produce the model's output (no behavioral change).
    - Separately computes attention probabilities for logging (no_grad).
    - Handles multi-view via num_views.
    """
    def __init__(self, controller, module_name: str, base_processor: nn.Module):
        super().__init__()
        self.controller = controller
        self.module_name = module_name
        self.base = base_processor  # << keep the original

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **cross_attention_kwargs,   # e.g., num_views
    ):
        # === 1) FORWARD: use the original processor exactly ===
        out = self.base(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **cross_attention_kwargs,
        )

        # === 2) SIDE PATH: recompute attention probs for logging only ===
        # Only capture CROSS-ATTENTION (attn2) - check early to avoid unnecessary computation
        suffix = self.module_name.split(".")[-1]  # attn1/attn2/Attention
        if suffix != "attn2":
            return out  # Skip self-attention, return output immediately
        
        try:
            with torch.no_grad():
                is_cross = encoder_hidden_states is not None
                ksrc = encoder_hidden_states if is_cross else hidden_states

                # QKV as in Attention.forward
                q = attn.to_q(hidden_states)
                k = attn.to_k(ksrc)

                # Optional pre-normalization on q/k used by some backbones
                if hasattr(attn, "norm_q") and attn.norm_q is not None:
                    q = attn.norm_q(q)
                if hasattr(attn, "norm_k") and attn.norm_k is not None:
                    k = attn.norm_k(k)

                # Heads
                H = attn.heads
                q_bh = attn.head_to_batch_dim(q)  # (B*H, Q, Dh)
                k_bh = attn.head_to_batch_dim(k)  # (B*H, K, Dh)

                # Some processors rescale q/k before softmax
                if hasattr(attn, "scale_qk") and attn.scale_qk is not None:
                    q_bh = q_bh * attn.scale_qk
                    k_bh = k_bh * attn.scale_qk

                # Softmax scale (fallback to 1/sqrt(Dh))
                if hasattr(attn, "scale") and attn.scale is not None:
                    scale = attn.scale
                elif hasattr(attn, "softmax_scale") and attn.softmax_scale is not None:
                    scale = attn.softmax_scale
                else:
                    scale = 1.0 / math.sqrt(q_bh.shape[-1])

                # Base scores
                scores = torch.matmul(q_bh.float(), k_bh.float().transpose(-2, -1)) * scale

                # Add positional/relative bias if the Attention module provides it
                bias = None
                if hasattr(attn, "get_attention_bias"):
                    try:
                        bias = attn.get_attention_bias(q_bh, k_bh, dtype=scores.dtype)
                    except TypeError:
                        try:
                            bias = attn.get_attention_bias(q_bh, k_bh)
                        except Exception:
                            bias = None
                if bias is not None:
                    scores = scores + bias

                # Attention mask
                if attention_mask is not None:
                    if attention_mask.dtype == torch.bool:
                        add_mask = torch.zeros_like(scores)
                        add_mask = add_mask.masked_fill(attention_mask, float("-inf"))
                        scores = scores + add_mask
                    else:
                        scores = scores + attention_mask.float()

                probs = scores.softmax(dim=-1)  # (B*H, Q, K)

                # Reshape and handle num_views
                BH, Q, Klen = probs.shape
                assert BH % H == 0
                B_eff = BH // H
                probs_bhqk = probs.view(B_eff, H, Q, Klen)

                num_views = cross_attention_kwargs.get("num_views", None)
                if num_views is not None and B_eff % num_views == 0:
                    B_true = B_eff // num_views
                    probs_bvhqk = probs_bhqk.view(B_true, num_views, H, Q, Klen)
                else:
                    probs_bvhqk = probs_bhqk.unsqueeze(1)  # (B, 1, H, Q, K)

                # Average heads
                probs_bvqk = probs_bvhqk.mean(dim=2)  # (B, V, Q, K)

                # Fuse or keep views per controller policy
                keep_per_view = getattr(self.controller, "keep_per_view", True)
                fuse_mode = getattr(self.controller, "fuse_views", None)
                if fuse_mode in ("mean", "max"):
                    if fuse_mode == "mean":
                        probs_to_store = probs_bvqk.mean(dim=1)
                    else:
                        probs_to_store, _ = probs_bvqk.max(dim=1)
                else:
                    probs_to_store = probs_bvqk if keep_per_view else probs_bvqk.mean(dim=1)

                # Determine layer name from module name
                module_parts = self.module_name.split('.')
                layer_name = f"seq_len_{Q}"  # Default fallback
                transformer_idx = None
                
                if 'down_blocks' in self.module_name:
                    for i, part in enumerate(module_parts):
                        if part == 'down_blocks' and i + 1 < len(module_parts):
                            block_idx = module_parts[i + 1]
                            layer_name = f"down_{block_idx}"
                            break
                elif 'mid_block' in self.module_name:
                    layer_name = "mid"
                elif 'up_blocks' in self.module_name:
                    for i, part in enumerate(module_parts):
                        if part == 'up_blocks' and i + 1 < len(module_parts):
                            block_idx = module_parts[i + 1]
                            layer_name = f"up_{block_idx}"
                            break
                
                # Extract transformer block index
                for i, part in enumerate(module_parts):
                    if part == 'transformer_blocks' and i + 1 < len(module_parts):
                        transformer_idx = module_parts[i + 1]
                        break
                
                # Build unique layer name
                if transformer_idx is not None:
                    final_layer_name = f"{layer_name}_t{transformer_idx}_{suffix}"
                else:
                    final_layer_name = f"{layer_name}_{suffix}"

                self.controller.store_attention(final_layer_name, probs_to_store.cpu())

        except Exception as e:
            print(f"[RecordingAttnProcessor] store failed for {self.module_name}: {e}")

        return out


class AttentionStore:
    """
    Simple attention controller that captures attention without modifying forward methods.
    """
    
    def __init__(self, input_height=512, input_width=512, store_timesteps=None):
        self.input_height = input_height
        self.input_width = input_width
        self.expected_resolutions = self._calculate_expected_resolutions()
        self.store_timesteps = store_timesteps or list(range(100))
        self.current_timestep = None
        self.timestep_counter = 0
        self.reset()

    def _calculate_expected_resolutions(self):
        """Calculate expected spatial resolutions for all UNet blocks."""
        latent_h = self.input_height // 8
        latent_w = self.input_width // 8
        resolutions = {}
        
        # Down blocks: progressively smaller
        current_h, current_w = latent_h, latent_w
        for i in range(4):
            seq_len = current_h * current_w
            layer_name = f"down_{i}"
            resolutions[seq_len] = {
                'layer_name': layer_name,
                'height': current_h,
                'width': current_w,
                'spatial_dims': (current_h, current_w)
            }
            current_h = max(1, current_h // 2)
            current_w = max(1, current_w // 2)
        
        # Mid block
        mid_seq_len = max(1, latent_h // 8) * max(1, latent_w // 8)
        resolutions[mid_seq_len] = {
            'layer_name': 'mid',
            'height': max(1, latent_h // 8),
            'width': max(1, latent_w // 8),
            'spatial_dims': (max(1, latent_h // 8), max(1, latent_w // 8))
        }
        
        # Up blocks
        up_sizes = []
        current_h, current_w = latent_h, latent_w
        for i in range(4):
            current_h = max(1, current_h // 2)
            current_w = max(1, current_w // 2)
            up_sizes.append((current_h, current_w))
        
        up_sizes.reverse()
        for i, (h, w) in enumerate(up_sizes):
            seq_len = h * w
            layer_name = f"up_{i}"
            if seq_len not in resolutions:
                resolutions[seq_len] = {
                    'layer_name': layer_name,
                    'height': h,
                    'width': w,
                    'spatial_dims': (h, w)
                }
        
        print(f"[AttentionStore] Expected resolutions: {list(resolutions.keys())}")
        print(f"[AttentionStore] Layer names: {[v['layer_name'] for v in resolutions.values()]}")
        
        return resolutions

    def reset(self):
        self.attention_store = {}
        self.timestep_counter = 0
        self.current_timestep = None
        print(f"[AttentionStore] Reset")

    def set_timestep(self, timestep):
        self.current_timestep = timestep
        self.attention_store[timestep] = {}
        
    def increment_timestep(self):
        self.timestep_counter += 1

    def store_attention(self, layer_name: str, attention_probs: torch.Tensor):
        # Decide the storage key
        key = self.current_timestep if self.current_timestep is not None else self.timestep_counter
        
        # Filter by store_timesteps
        if self.store_timesteps is not None and len(self.store_timesteps) > 0:
            if (key not in self.store_timesteps) and (self.timestep_counter not in self.store_timesteps):
                return
                
        if key not in self.attention_store:
            self.attention_store[key] = {}
        
        # Store with full layer name
        self.attention_store[key][layer_name] = attention_probs.detach().cpu()
        
        # Also store aggregated by block type and block index for BE metric compatibility
        if '_' in layer_name:
            parts = layer_name.split('_')
            if len(parts) >= 1:
                block_type = parts[0]  # down, mid, or up
                if block_type in ['down', 'mid', 'up']:
                    # Store aggregated by block type (e.g., "down")
                    if block_type not in self.attention_store[key]:
                        self.attention_store[key][block_type] = attention_probs.detach().cpu()
                    else:
                        prev = self.attention_store[key][block_type]
                        if prev.shape == attention_probs.shape:
                            self.attention_store[key][block_type] = (prev + attention_probs.detach().cpu()) / 2.0
                    
                    # Store by block index (e.g., "down_0", "down_1")
                    if len(parts) >= 2 and parts[1].isdigit():
                        block_with_idx = f"{block_type}_{parts[1]}"
                        if block_with_idx not in self.attention_store[key]:
                            self.attention_store[key][block_with_idx] = attention_probs.detach().cpu()
                        else:
                            prev = self.attention_store[key][block_with_idx]
                            if prev.shape == attention_probs.shape:
                                self.attention_store[key][block_with_idx] = (prev + attention_probs.detach().cpu()) / 2.0

    def get_attention_maps(self, timestep=None, layer=None):
        print(f"[AttentionStore] Available timesteps: {list(self.attention_store.keys())}")
        if timestep is not None:
            return self.attention_store.get(timestep, {}).get(layer) if layer else self.attention_store.get(timestep, {})
        return self.attention_store

    def get_spatial_dims_for_layer(self, layer_name):
        for resolution_info in self.expected_resolutions.values():
            if resolution_info['layer_name'] == layer_name:
                return resolution_info['spatial_dims']
        return None


# Global variable to store current controller
_current_controller = None


def register_attention_control(pipeline: Any, controller: AttentionStore):
    """
    Register attention capture using processor wrappers.
    This approach works reliably because processors receive encoder_hidden_states directly.
    """
    global _current_controller
    _current_controller = controller
    hook_handles = []

    # Update controller dimensions from pipeline
    if hasattr(pipeline, 'unet') and hasattr(pipeline.unet.config, 'sample_size'):
        height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = height
        if controller.input_height == controller.input_width == 512:
            controller.input_height = height
            controller.input_width = width
            controller.expected_resolutions = controller._calculate_expected_resolutions()

    # Find attention modules
    def find_attention_modules(module, prefix=""):
        mods = []
        for name, sub in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            
            # Direct Attention modules
            if sub.__class__.__name__ == "Attention":
                mods.append((full, sub))
            
            # BasicTransformerBlock with attn1/attn2
            elif "BasicTransformerBlock" in sub.__class__.__name__:
                if hasattr(sub, "attn1"):
                    mods.append((f"{full}.attn1", sub.attn1))
                if hasattr(sub, "attn2"):
                    mods.append((f"{full}.attn2", sub.attn2))
            
            # Transformer2DModel blocks
            elif "Transformer2DModel" in sub.__class__.__name__:
                if hasattr(sub, "transformer_blocks"):
                    for i, block in enumerate(sub.transformer_blocks):
                        if hasattr(block, "attn1"):
                            mods.append((f"{full}.transformer_blocks.{i}.attn1", block.attn1))
                        if hasattr(block, "attn2"):
                            mods.append((f"{full}.transformer_blocks.{i}.attn2", block.attn2))
            
            # Recursively search children
            mods.extend(find_attention_modules(sub, full))
        return mods

    attn_modules = find_attention_modules(pipeline.unet, "unet")
    print(f"[AttentionStore] Found {len(attn_modules)} attention modules")
    
    # Print module names for debugging
    module_names_by_block = {}
    for module_name, _ in attn_modules:
        block_type = module_name.split('.')[1] if '.' in module_name else 'unknown'
        if block_type not in module_names_by_block:
            module_names_by_block[block_type] = []
        module_names_by_block[block_type].append(module_name)
    
    for block_type, names in sorted(module_names_by_block.items()):
        print(f"[AttentionStore]   {block_type}: {len(names)} modules")

    # Wrap processors
    for module_name, module in attn_modules:
        try:
            original = module.processor
            wrapper = RecordingAttnProcessor(controller, module_name, original)
            module.set_processor(wrapper)
            hook_handles.append(("processor", module, original))
        except Exception as e:
            print(f"[AttentionStore] Failed to wrap processor on {module_name}: {e}")

    print(f"[AttentionStore] Wrapped processors on {len(hook_handles)} modules")
    return hook_handles


def unregister_attention_control(hook_handles: List[Any]):
    """Remove all registered processor wrappers."""
    global _current_controller
    _current_controller = None

    restored = 0
    for item in hook_handles:
        try:
            if isinstance(item, tuple) and item[0] == "processor":
                module, original = item[1], item[2]
                module.set_processor(original)
                restored += 1
            else:
                item.remove()
        except Exception as e:
            print(f"[AttentionStore] Error removing hook: {e}")

    print(f"[AttentionStore] Restored {restored} processors")
