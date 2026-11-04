"""BiCrossMamba representation backend from the author's supplied source.

The source angular classifier is removed here because StandardDetector owns
its classifier/loss. The controlled DeepFense grid uses the same cross-entropy
head as its existing controls. This is explicit in every generated config.
"""
import torch
from torch import nn
from deepfense.models.base_model import BaseBackend
from deepfense.utils.registry import register_backend


@register_backend('BiCrossMambaST')
class BiCrossMambaST(BaseBackend):
    def __init__(self, config):
        super().__init__(config)
        # Keep Mamba optional for all existing framework backends.
        from deepfense.models.vendor.bicrossmamba.bicrossmamba import BiCrossMamba_ST
        self.model = BiCrossMamba_ST(config)
        self.model.proj_ATTN = nn.Identity()
        self.output_dim = 128
        # The raw sinc encoder already produced the 64-channel map.
        # Drop the SSL projection encoder so it is not trained.
        if config.get("input_format") == "encoded_map":
            self.model.encoder = nn.Identity()

    def forward(self, x, **kwargs):
        # The raw sinc encoder already returns the spectro-temporal map.
        # SSL frontends still return [batch, frames, input_dim].
        if x.ndim == 4:
            if x.shape[1] != 64:
                raise ValueError(f'Expected encoded map [batch, 64, freq, time], got {tuple(x.shape)}')
            return self.model.forward_from_map(x)
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f'Expected [batch, frames, {self.input_dim}], got {tuple(x.shape)}')
        return self.model(x)
