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
        try:
            with torch.no_grad():
                is_cross = encoder_hidden_states is not None
                ksrc = encoder_hidden_states if is_cross else hidden_states

                # QKV as in Attention.forward
                q = attn.to_q(hidden_states)
                k = attn.to_k(ksrc)
                v = attn.to_v(ksrc)

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
                # (Diffusers Attention often has get_attention_bias or similar API)
                bias = None
                if hasattr(attn, "get_attention_bias"):
                    # Some variants take (Q,K) lengths or spatial shapes; try the common signature.
                    try:
                        bias = attn.get_attention_bias(q_bh, k_bh, dtype=scores.dtype)
                    except TypeError:
                        try:
                            bias = attn.get_attention_bias(q_bh, k_bh)
                        except Exception:
                            bias = None
                if bias is not None:
                    # Broadcast/add if shapes differ (B*H, Q, K) vs (1 or B*H, Q, K)
                    scores = scores + bias

                # Attention mask (additive -inf mask or boolean) as passed by pipeline
                if attention_mask is not None:
                    # If boolean, convert to additive mask with large negative value
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
                        probs_to_store = probs_bvqk.mean(dim=1)          # (B, Q, K)
                    else:
                        probs_to_store, _ = probs_bvqk.max(dim=1)        # (B, Q, K)
                else:
                    probs_to_store = probs_bvqk if keep_per_view else probs_bvqk.mean(dim=1)

                # Map by spatial length -> friendly layer name
                seq_len = Q
                layer_name = self.controller.expected_resolutions.get(
                    seq_len, {"layer_name": f"seq_len_{seq_len}"}
                )["layer_name"]
                suffix = self.module_name.split(".")[-1]  # attn1/attn2/Attention

                self.controller.store_attention(f"{layer_name}_{suffix}", probs_to_store.cpu())

        except Exception as e:
            print(f"[RecordingAttnProcessor] store failed for {self.module_name}: {e}")

        return out


class PassiveRecordingAttnProcessor(nn.Module):
    """
    Records attention weights without recomputing them.
    Only captures what the original processor already computed.
    """
    def __init__(self, controller, module_name: str, base_processor: nn.Module):
        super().__init__()
        self.controller = controller
        self.module_name = module_name
        self.base = base_processor

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # Call original processor exactly as-is - no changes to computation
        result = self.base(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        
        # Only try to capture if the attention module has stored weights
        try:
            if hasattr(attn, 'attn_weights') and attn.attn_weights is not None:
                probs = attn.attn_weights.detach()
                
                # Process for storage (same logic as before)
                seq_len = probs.shape[-2]
                layer_name = self.controller.expected_resolutions.get(
                    seq_len, {"layer_name": f"seq_len_{seq_len}"}
                )["layer_name"]
                suffix = self.module_name.split(".")[-1]
                
                # Handle multi-view and store
                self.controller.store_attention(f"{layer_name}_{suffix}", probs.cpu())
                
        except Exception:
            # Silently continue - don't print errors that might affect generation
            pass
            
        return result


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
        latent_h = self.input_height // 8
        latent_w = self.input_width // 8
        resolutions = {}
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
        return resolutions

    def reset(self):
        self.attention_store = {}
        self.timestep_counter = 0
        self.current_timestep = None
        print(f"[AttentionStore] Reset")

    def set_timestep(self, timestep):
        self.current_timestep = timestep
        self.attention_store[timestep] = {}
        print(f"[AttentionStore] Set timestep: {timestep}")
        
    def increment_timestep(self):
        self.timestep_counter += 1
        print(f"[AttentionStore] Incremented to step: {self.timestep_counter}")

    def store_attention(self, layer_name: str, attention_probs: torch.Tensor):
        # Decide the storage key
        key = self.current_timestep if self.current_timestep is not None else self.timestep_counter
        # Let the user specify either real timesteps or indices
        if self.store_timesteps is not None and len(self.store_timesteps) > 0:
            if (key not in self.store_timesteps) and (self.timestep_counter not in self.store_timesteps):
                return
        if key not in self.attention_store:
            self.attention_store[key] = {}
        self.attention_store[key][layer_name] = attention_probs.detach().cpu()
        base = layer_name.split('_')[0] if layer_name.startswith("down_") else layer_name
        if base not in self.attention_store[key]:
            self.attention_store[key][base] = attention_probs.detach().cpu()
        else:
            prev = self.attention_store[key][base]
            if prev.shape == attention_probs.shape:
                self.attention_store[key][base] = (prev + attention_probs.detach().cpu()) / 2.0
        print(f"[AttentionStore] Stored attention for timestep {key}, layer {layer_name}, shape {attention_probs.shape}")

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


# Global variable to store current controller (needed for processor hook)
_current_controller = None


def create_simple_attention_hook(controller, module_name):
    """Create a simple hook that tries to capture attention from module attributes."""
    def attention_hook(module, inputs, outputs):
        try:
            # Strategy 1: Check if attention processor stored attention weights
            if hasattr(module, 'processor') and hasattr(module.processor, 'attn_weights'):
                attn_weights = module.processor.attn_weights
                if attn_weights is not None and torch.is_tensor(attn_weights):
                    print(f"[Hook] Found processor attention: {attn_weights.shape}")
                    
                    # Determine layer name
                    seq_len = attn_weights.shape[-2]
                    if seq_len in controller.expected_resolutions:
                        layer_name = controller.expected_resolutions[seq_len]['layer_name']
                    else:
                        layer_name = f"seq_len_{seq_len}"
                    
                    # Average over heads if needed
                    if attn_weights.dim() == 4:
                        attn_weights = attn_weights.mean(dim=1)
                    
                    controller.store_attention(f"{layer_name}_{module_name.split('.')[-1]}", attn_weights)
                    return
            
            # Strategy 2: Check common attention weight attributes
            attr_names = [
                'attn_weights', 'attention_probs', 'last_attn_slice', 
                'attention_weights', 'attn_probs', '_attn_weights'
            ]
            
            for attr_name in attr_names:
                if hasattr(module, attr_name):
                    attn_val = getattr(module, attr_name)
                    if attn_val is not None and torch.is_tensor(attn_val) and attn_val.dim() >= 3:
                        print(f"[Hook] Found attention in {attr_name}: {attn_val.shape}")
                        
                        # Determine layer name
                        seq_len = attn_val.shape[-2] if attn_val.dim() >= 3 else attn_val.shape[1]
                        if seq_len in controller.expected_resolutions:
                            layer_name = controller.expected_resolutions[seq_len]['layer_name']
                        else:
                            layer_name = f"seq_len_{seq_len}"
                        
                        # Average over heads if needed
                        if attn_val.dim() == 4:
                            attn_val = attn_val.mean(dim=1)
                        
                        controller.store_attention(f"{layer_name}_{module_name.split('.')[-1]}", attn_val)
                        return
                        
        except Exception as e:
            print(f"[Hook] Error in {module_name}: {e}")
    
    return attention_hook


def register_attention_control(pipeline: Any, controller: AttentionStore):
    global _current_controller
    _current_controller = controller
    hook_handles = []

    # pass real H/W to controller if it was default 512
    if hasattr(pipeline, 'unet') and hasattr(pipeline.unet.config, 'sample_size'):
        height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = height
        if controller.input_height == controller.input_width == 512:
            controller.input_height = height
            controller.input_width = width
            controller.expected_resolutions = controller._calculate_expected_resolutions()

    # find attention modules
    def find_attention_modules(module, prefix=""):
        mods = []
        for name, sub in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if sub.__class__.__name__ == "Attention":
                mods.append((full, sub))
            elif "BasicTransformerBlock" in sub.__class__.__name__:
                if hasattr(sub, "attn1"):
                    mods.append((f"{full}.attn1", sub.attn1))
                if hasattr(sub, "attn2"):
                    mods.append((f"{full}.attn2", sub.attn2))
            mods.extend(find_attention_modules(sub, full))
        return mods

    attn_modules = find_attention_modules(pipeline.unet, "unet")
    print(f"[AttentionStore] Found {len(attn_modules)} attention modules")

    for module_name, module in attn_modules:
        try:
            original = module.processor
            wrapper = RecordingAttnProcessor(controller, module_name, original)
            # wrapper = PassiveRecordingAttnProcessor(controller, module_name, original)
            module.set_processor(wrapper)
            hook_handles.append(("processor", module, original))
        except Exception as e:
            print(f"[AttentionStore] Failed to wrap processor on {module_name}: {e}")

    print(f"[AttentionStore] Wrapped processors on {len(hook_handles)} modules")
    return hook_handles


def unregister_attention_control(hook_handles: List[Any]):
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
                # any future non-processor hooks
                item.remove()
        except Exception as e:
            print(f"[AttentionStore] Error removing hook: {e}")

    print(f"[AttentionStore] Restored {restored} processors")




class AttentionCaptureHook:
    """Captures attention weights using forward hooks without interfering with computation."""
    
    def __init__(self, controller, module_name):
        self.controller = controller
        self.module_name = module_name
    
    def __call__(self, module, input, output):
        try:
            # The key insight: capture during the hook, AFTER forward pass completes
            with torch.no_grad():
                hidden_states, encoder_hidden_states = input[0], input[1] if len(input) > 1 else None
                
                # Get the attention module reference
                attn = module
                
                # Recompute attention ONLY in the hook (not during forward)
                is_cross = encoder_hidden_states is not None
                ksrc = encoder_hidden_states if is_cross else hidden_states

                q = attn.to_q(hidden_states)
                k = attn.to_k(ksrc)
                
                # Handle normalization if present
                if hasattr(attn, "norm_q") and attn.norm_q is not None:
                    q = attn.norm_q(q)
                if hasattr(attn, "norm_k") and attn.norm_k is not None:
                    k = attn.norm_k(k)

                # Reshape for attention computation
                H = attn.heads
                q_bh = attn.head_to_batch_dim(q)
                k_bh = attn.head_to_batch_dim(k)

                # Get scale
                if hasattr(attn, "scale") and attn.scale is not None:
                    scale = attn.scale
                else:
                    scale = 1.0 / math.sqrt(q_bh.shape[-1])

                # Compute attention scores
                scores = torch.matmul(q_bh.float(), k_bh.float().transpose(-2, -1)) * scale
                probs = scores.softmax(dim=-1)

                # Process for storage
                BH, Q, K = probs.shape
                B_eff = BH // H
                probs_bhqk = probs.view(B_eff, H, Q, K)
                probs_bvqk = probs_bhqk.mean(dim=1)  # Average over heads

                # Handle CFG (take conditional part only)
                if B_eff % 2 == 0 and B_eff > 1:
                    cond_batch_size = B_eff // 2
                    probs_to_store = probs_bvqk[cond_batch_size:]
                else:
                    probs_to_store = probs_bvqk

                # Determine layer name and store
                seq_len = Q
                layer_name = self.controller.expected_resolutions.get(
                    seq_len, {"layer_name": f"seq_len_{seq_len}"}
                )["layer_name"]
                suffix = self.module_name.split(".")[-1]
                
                self.controller.store_attention(f"{layer_name}_{suffix}", probs_to_store.cpu())
                
        except Exception as e:
            # Don't print during generation - just log
            pass


def register_attention_control(pipeline: Any, controller: AttentionStore):
    """Register hooks to capture attention without interfering with generation."""
    hook_handles = []
    
    # Update controller dimensions
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
            if sub.__class__.__name__ == "Attention":
                mods.append((full, sub))
            elif "BasicTransformerBlock" in sub.__class__.__name__:
                if hasattr(sub, "attn1"):
                    mods.append((f"{full}.attn1", sub.attn1))
                if hasattr(sub, "attn2"):
                    mods.append((f"{full}.attn2", sub.attn2))
            mods.extend(find_attention_modules(sub, full))
        return mods

    attn_modules = find_attention_modules(pipeline.unet, "unet")
    print(f"Found {len(attn_modules)} attention modules")

    # Register forward hooks instead of processor wrappers
    for module_name, module in attn_modules:
        try:
            hook_fn = AttentionCaptureHook(controller, module_name)
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)
        except Exception as e:
            print(f"Failed to register hook on {module_name}: {e}")

    print(f"Registered {len(hook_handles)} attention capture hooks")
    return hook_handles


def unregister_attention_control(hook_handles: List[Any]):
    """Remove all registered hooks."""
    removed = 0
    for handle in hook_handles:
        try:
            handle.remove()
            removed += 1
        except Exception as e:
            print(f"Error removing hook: {e}")
    
    print(f"Removed {removed} hooks")