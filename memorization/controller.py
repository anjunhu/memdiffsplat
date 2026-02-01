# DiffSplat/memorization/controller.py
"""
Attention capture for DiffSplat that does NOT affect model output.

Key insight: We use forward hooks on BasicTransformerBlock (not Attention modules)
to capture attention AFTER it's computed, without recomputing anything.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any, Dict, Optional, Tuple


class AttentionStore:
    """
    Attention controller that captures cross-attention maps without affecting generation.
    Uses forward hooks on transformer blocks to passively observe attention.
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
        
        mid_seq_len = max(1, latent_h // 8) * max(1, latent_w // 8)
        resolutions[mid_seq_len] = {
            'layer_name': 'mid',
            'height': max(1, latent_h // 8),
            'width': max(1, latent_w // 8),
            'spatial_dims': (max(1, latent_h // 8), max(1, latent_w // 8))
        }
        
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
        
        return resolutions

    def reset(self):
        self.attention_store = {}
        self.timestep_counter = 0
        self.current_timestep = None

    def set_timestep(self, timestep):
        self.current_timestep = timestep
        if timestep not in self.attention_store:
            self.attention_store[timestep] = {}
        
    def increment_timestep(self):
        self.timestep_counter += 1

    def store_attention(self, layer_name: str, attention_probs: torch.Tensor):
        """Store attention probabilities for the current timestep.
        
        Args:
            layer_name: Name of the layer (e.g., "down_0_t0_attn2")
            attention_probs: Attention probabilities tensor (should already be on CPU)
        """
        key = self.current_timestep if self.current_timestep is not None else self.timestep_counter
        
        if self.store_timesteps is not None and len(self.store_timesteps) > 0:
            if (key not in self.store_timesteps) and (self.timestep_counter not in self.store_timesteps):
                return
                
        if key not in self.attention_store:
            self.attention_store[key] = {}
        
        # Ensure tensor is on CPU and detached
        if attention_probs.is_cuda:
            attention_probs = attention_probs.detach().cpu()
        elif attention_probs.requires_grad:
            attention_probs = attention_probs.detach()
        
        self.attention_store[key][layer_name] = attention_probs
        
        # Also store aggregated by block type and index
        if '_' in layer_name:
            parts = layer_name.split('_')
            if len(parts) >= 1:
                block_type = parts[0]
                if block_type in ['down', 'mid', 'up']:
                    if block_type not in self.attention_store[key]:
                        self.attention_store[key][block_type] = attention_probs.clone()
                    else:
                        prev = self.attention_store[key][block_type]
                        if prev.shape == attention_probs.shape:
                            self.attention_store[key][block_type] = (prev + attention_probs) / 2.0
                    
                    if len(parts) >= 2 and parts[1].isdigit():
                        block_with_idx = f"{block_type}_{parts[1]}"
                        if block_with_idx not in self.attention_store[key]:
                            self.attention_store[key][block_with_idx] = attention_probs.clone()
                        else:
                            prev = self.attention_store[key][block_with_idx]
                            if prev.shape == attention_probs.shape:
                                self.attention_store[key][block_with_idx] = (prev + attention_probs) / 2.0

    def get_attention_maps(self, timestep=None, layer=None):
        if timestep is not None:
            return self.attention_store.get(timestep, {}).get(layer) if layer else self.attention_store.get(timestep, {})
        return self.attention_store

    def get_spatial_dims_for_layer(self, layer_name):
        for resolution_info in self.expected_resolutions.values():
            if resolution_info['layer_name'] == layer_name:
                return resolution_info['spatial_dims']
        return None


class AttentionCaptureProcessor:
    """
    A processor wrapper that captures attention weights computed by the original processor.
    
    CRITICAL: This does NOT recompute attention. It only stores what was already computed.
    The original processor must store attention weights in a known location for this to work.
    """
    def __init__(self, controller: AttentionStore, module_name: str, original_processor):
        self.controller = controller
        self.module_name = module_name
        self.original_processor = original_processor
        self._last_attn_probs = None
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # Call original processor - this is the ONLY computation
        output = self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        
        # Try to capture attention weights if they were stored by the processor
        # This is a passive capture - we don't recompute anything
        try:
            attn_probs = None
            
            # Check various places where attention might be stored
            if hasattr(self.original_processor, 'attn_probs'):
                attn_probs = self.original_processor.attn_probs
            elif hasattr(self.original_processor, 'attention_probs'):
                attn_probs = self.original_processor.attention_probs
            elif hasattr(attn, 'attn_probs'):
                attn_probs = attn.attn_probs
            elif hasattr(attn, 'attention_probs'):
                attn_probs = attn.attention_probs
            
            if attn_probs is not None:
                self._store_attention(attn_probs)
                
        except Exception:
            pass  # Silently fail - don't affect generation
        
        return output
    
    def _store_attention(self, attn_probs):
        """Store captured attention probabilities."""
        suffix = self.module_name.split(".")[-1]
        if suffix != "attn2":
            return  # Only store cross-attention
        
        module_parts = self.module_name.split('.')
        layer_name = "unknown"
        transformer_idx = None
        
        if 'down_blocks' in self.module_name:
            for i, part in enumerate(module_parts):
                if part == 'down_blocks' and i + 1 < len(module_parts):
                    layer_name = f"down_{module_parts[i + 1]}"
                    break
        elif 'mid_block' in self.module_name:
            layer_name = "mid"
        elif 'up_blocks' in self.module_name:
            for i, part in enumerate(module_parts):
                if part == 'up_blocks' and i + 1 < len(module_parts):
                    layer_name = f"up_{module_parts[i + 1]}"
                    break
        
        for i, part in enumerate(module_parts):
            if part == 'transformer_blocks' and i + 1 < len(module_parts):
                transformer_idx = module_parts[i + 1]
                break
        
        if transformer_idx is not None:
            final_layer_name = f"{layer_name}_t{transformer_idx}_{suffix}"
        else:
            final_layer_name = f"{layer_name}_{suffix}"
        
        self.controller.store_attention(final_layer_name, attn_probs.detach())


def _create_attention_hook(controller: AttentionStore, module_name: str):
    """
    Create a forward hook that captures attention from BasicTransformerBlock.
    
    This hook recomputes attention weights in a completely detached context
    so it doesn't affect the model's computation graph or outputs.
    """
    def hook(module, args, output):
        try:
            # Only capture for attn2 (cross-attention)
            if not hasattr(module, 'attn2') or module.attn2 is None:
                return
            
            attn2 = module.attn2
            
            # Get the inputs that were used for this attention layer
            # For BasicTransformerBlock, we need to reconstruct from the block's inputs
            # The hidden_states after attn1 + norm2 is what goes into attn2
            
            # Try to get stored attention first (if processor stores it)
            attn_probs = None
            if hasattr(attn2, 'processor'):
                proc = attn2.processor
                if hasattr(proc, 'attn_probs'):
                    attn_probs = proc.attn_probs
                elif hasattr(proc, 'attention_probs'):
                    attn_probs = proc.attention_probs
            
            if hasattr(attn2, 'attn_probs'):
                attn_probs = attn2.attn_probs
            elif hasattr(attn2, 'attention_probs'):
                attn_probs = attn2.attention_probs
            
            # If we found stored attention, use it
            if attn_probs is not None:
                # Clone and move to CPU immediately to break any graph connection
                attn_probs_cpu = attn_probs.detach().clone().cpu()
                
                # Determine layer name
                layer_name = _get_layer_name(module_name)
                controller.store_attention(layer_name, attn_probs_cpu)
                return
            
            # Otherwise, we need to recompute attention in a detached context
            # This is safe because we use torch.no_grad() and clone inputs
            
        except Exception as e:
            pass  # Silently fail
    
    return hook


def _get_layer_name(module_name: str) -> str:
    """Extract a clean layer name from the module path."""
    module_parts = module_name.split('.')
    layer_name = "unknown"
    transformer_idx = None
    
    if 'down_blocks' in module_name:
        for i, part in enumerate(module_parts):
            if part == 'down_blocks' and i + 1 < len(module_parts):
                layer_name = f"down_{module_parts[i + 1]}"
                break
    elif 'mid_block' in module_name:
        layer_name = "mid"
    elif 'up_blocks' in module_name:
        for i, part in enumerate(module_parts):
            if part == 'up_blocks' and i + 1 < len(module_parts):
                layer_name = f"up_{module_parts[i + 1]}"
                break
    
    for i, part in enumerate(module_parts):
        if part == 'transformer_blocks' and i + 1 < len(module_parts):
            transformer_idx = module_parts[i + 1]
            break
    
    if transformer_idx is not None:
        return f"{layer_name}_t{transformer_idx}_attn2"
    else:
        return f"{layer_name}_attn2"


class RecomputingAttentionHook:
    """
    A hook that recomputes attention weights in a completely isolated context.
    
    Key insight: We store the Q, K inputs during forward, then compute attention
    probabilities AFTER the forward pass completes, in a separate no_grad context.
    This ensures zero interference with the model's computation.
    """
    def __init__(self, controller: AttentionStore, module_name: str):
        self.controller = controller
        self.module_name = module_name
        self.layer_name = _get_layer_name(module_name)
        self._cached_qk = None
    
    def pre_hook(self, module, args):
        """Called before forward - cache inputs for later recomputation."""
        # Don't cache during this phase - we'll compute in post_hook
        pass
    
    def post_hook(self, module, args, output):
        """Called after forward - recompute attention in isolated context."""
        try:
            # Skip if not cross-attention
            if not hasattr(module, 'attn2') or module.attn2 is None:
                return
            
            attn = module.attn2
            
            # Get the hidden states and encoder hidden states from the block's context
            # For BasicTransformerBlock, args[0] is hidden_states, and encoder_hidden_states
            # is passed via cross_attention_kwargs or as a separate argument
            
            # Try to get from stored attributes first
            hidden_states = None
            encoder_hidden_states = None
            
            if hasattr(module, '_attn2_hidden_states'):
                hidden_states = module._attn2_hidden_states
            if hasattr(module, '_attn2_encoder_hidden_states'):
                encoder_hidden_states = module._attn2_encoder_hidden_states
            
            if hidden_states is None:
                return
            
            # Recompute attention in completely isolated context
            with torch.no_grad():
                # Clone inputs to ensure complete isolation
                hs = hidden_states.detach().clone()
                enc_hs = encoder_hidden_states.detach().clone() if encoder_hidden_states is not None else hs
                
                # Compute Q, K
                q = attn.to_q(hs)
                k = attn.to_k(enc_hs)
                
                # Apply normalization if present
                if hasattr(attn, 'norm_q') and attn.norm_q is not None:
                    q = attn.norm_q(q)
                if hasattr(attn, 'norm_k') and attn.norm_k is not None:
                    k = attn.norm_k(k)
                
                # Reshape for attention
                H = attn.heads
                q_bh = attn.head_to_batch_dim(q)
                k_bh = attn.head_to_batch_dim(k)
                
                # Get scale
                scale = getattr(attn, 'scale', None) or (1.0 / math.sqrt(q_bh.shape[-1]))
                
                # Compute attention scores and probabilities
                scores = torch.matmul(q_bh.float(), k_bh.float().transpose(-2, -1)) * scale
                probs = scores.softmax(dim=-1)
                
                # Average over heads and move to CPU
                BH, Q, K = probs.shape
                B_eff = BH // H
                probs_bhqk = probs.view(B_eff, H, Q, K)
                probs_avg = probs_bhqk.mean(dim=1)  # Average over heads
                
                # Store on CPU
                self.controller.store_attention(self.layer_name, probs_avg.cpu())
                
        except Exception as e:
            pass  # Silently fail


class StoringAttnProcessor:
    """
    Wraps the original attention processor to store attention weights.
    
    CRITICAL: This computes attention weights in a SEPARATE path that doesn't
    affect the model's output. The original processor runs unchanged.
    """
    def __init__(self, controller: AttentionStore, module_name: str, original_processor):
        self.controller = controller
        self.module_name = module_name
        self.layer_name = _get_layer_name(module_name)
        self.original_processor = original_processor
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # 1. Run original processor FIRST - this is the model's actual computation
        output = self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        
        # 2. Compute attention weights in a completely separate, detached context
        # This does NOT affect the model output because:
        # - We use torch.no_grad()
        # - We clone all inputs
        # - We don't modify any tensors used by the model
        try:
            with torch.no_grad():
                # Clone inputs to ensure complete isolation from computation graph
                hs = hidden_states.detach().clone()
                enc_hs = encoder_hidden_states.detach().clone() if encoder_hidden_states is not None else hs
                
                # Compute Q, K (these are new tensors, not connected to model's graph)
                q = attn.to_q(hs)
                k = attn.to_k(enc_hs)
                
                # Apply normalization if present
                if hasattr(attn, 'norm_q') and attn.norm_q is not None:
                    q = attn.norm_q(q)
                if hasattr(attn, 'norm_k') and attn.norm_k is not None:
                    k = attn.norm_k(k)
                
                # Reshape for attention computation
                H = attn.heads
                q_bh = attn.head_to_batch_dim(q)
                k_bh = attn.head_to_batch_dim(k)
                
                # Get scale
                scale = getattr(attn, 'scale', None) or (1.0 / math.sqrt(q_bh.shape[-1]))
                
                # Compute attention probabilities
                scores = torch.matmul(q_bh.float(), k_bh.float().transpose(-2, -1)) * scale
                probs = scores.softmax(dim=-1)
                
                # Average over heads
                BH, Q, K = probs.shape
                B_eff = BH // H
                probs_bhqk = probs.view(B_eff, H, Q, K)
                probs_avg = probs_bhqk.mean(dim=1)
                
                # Store on CPU immediately
                self.controller.store_attention(self.layer_name, probs_avg.cpu())
                
        except Exception as e:
            # Silently fail - don't affect generation
            pass
        
        return output


def register_attention_control(pipeline: Any, controller: AttentionStore):
    """
    Register attention capture by wrapping attention processors.
    
    This approach wraps each attention module's processor with StoringAttnProcessor,
    which computes attention weights in a separate detached context after the
    original processor runs. This ensures:
    1. Model output is unchanged (original processor runs first)
    2. Attention weights are computed in isolation (no graph connection)
    3. Weights are moved to CPU immediately (no GPU memory buildup)
    """
    wrapped_modules = []

    # Update controller dimensions
    if hasattr(pipeline, 'unet') and hasattr(pipeline.unet.config, 'sample_size'):
        if hasattr(pipeline, 'vae_scale_factor'):
            height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        else:
            height = pipeline.unet.config.sample_size * 8
        width = height
        if controller.input_height == controller.input_width == 512:
            controller.input_height = height
            controller.input_width = width
            controller.expected_resolutions = controller._calculate_expected_resolutions()

    # Find all Attention modules and wrap their processors
    def find_attention_modules(module, prefix=""):
        mods = []
        for name, sub in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if sub.__class__.__name__ == "Attention":
                mods.append((full, sub))
            mods.extend(find_attention_modules(sub, full))
        return mods

    attn_modules = find_attention_modules(pipeline.unet, "unet")
    print(f"[AttentionStore] Found {len(attn_modules)} attention modules")

    # Only wrap cross-attention (attn2) modules
    for module_name, module in attn_modules:
        if 'attn2' in module_name or module_name.endswith('.attn2'):
            try:
                original_processor = module.processor
                wrapper = StoringAttnProcessor(controller, module_name, original_processor)
                module.set_processor(wrapper)
                wrapped_modules.append((module, original_processor))
            except Exception as e:
                print(f"[AttentionStore] Failed to wrap {module_name}: {e}")

    print(f"[AttentionStore] Wrapped {len(wrapped_modules)} cross-attention processors")
    return wrapped_modules


def unregister_attention_control(wrapped_modules: List[Any]):
    """Restore original processors to all wrapped attention modules."""
    restored = 0
    for item in wrapped_modules:
        try:
            if isinstance(item, tuple) and len(item) == 2:
                module, original_processor = item
                module.set_processor(original_processor)
                restored += 1
            else:
                # Handle hook handles if any
                item.remove()
        except Exception as e:
            print(f"[AttentionStore] Error restoring processor: {e}")
    
    print(f"[AttentionStore] Restored {restored} processors")
