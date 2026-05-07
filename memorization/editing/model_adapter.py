"""
DiffusionModelAdapter for DiffSplat (multi-view SD-family pipeline).
The UNet is UNetMV2DConditionModel — same layer naming as SD1.5.
"""
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from unlearning.adapter import DiffusionModelAdapter


class DiffSplatPipelineAdapter(DiffusionModelAdapter):
    """
    Wraps a StableMVDiffusionPipeline (DiffSplat) to expose the
    DiffusionModelAdapter interface.
    """

    def __init__(self, pipeline, image_size: int = 256):
        self._pipe = pipeline
        self._image_size = image_size

    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        tokenizer = self._pipe.tokenizer
        text_encoder = self._pipe.text_encoder
        inputs = tokenizer(
            prompts, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            return text_encoder(**inputs).last_hidden_state

    def null_embedding(self, batch_size: int) -> torch.Tensor:
        return self.encode_text([""] * batch_size)

    def apply_model(self, x_t: torch.Tensor, t: torch.Tensor,
                    conditioning: Dict) -> torch.Tensor:
        encoder_hidden_states = conditioning["context"]
        if t.shape[0] == 1 and x_t.shape[0] > 1:
            t = t.expand(x_t.shape[0])
        # DiffSplat's conv_in may expect extra channels (Plücker rays, binary mask)
        # beyond the 4 latent channels. Pad with zeros — NeMo only needs cross-attention
        # activations, so geometric correctness of the extra channels doesn't matter.
        expected_c = self._pipe.unet.conv_in.weight.shape[1]
        if x_t.shape[1] < expected_c:
            pad = torch.zeros(x_t.shape[0], expected_c - x_t.shape[1],
                              *x_t.shape[2:], device=x_t.device, dtype=x_t.dtype)
            x_t = torch.cat([x_t, pad], dim=1)
        return self._pipe.unet(
            x_t, t, encoder_hidden_states=encoder_hidden_states
        ).sample

    def make_conditioning(self, text_emb: torch.Tensor) -> Dict:
        return {"context": text_emb}

    def get_attention_layers(self, layer_type: str = "cross") -> Dict[str, nn.Module]:
        layers = {}
        for name, module in self._pipe.unet.named_modules():
            if layer_type == "cross":
                if "attn2" in name and name.endswith(("to_k", "to_v")):
                    layers[name] = module
            elif layer_type == "ffn":
                if name.endswith("ff.net.2"):
                    layers[name] = module
        return layers

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        s = self._image_size // 8
        return (4, s, s)

    @property
    def device(self) -> torch.device:
        return self._pipe.unet.device

    @property
    def dtype(self) -> torch.dtype:
        return next(self._pipe.unet.parameters()).dtype
