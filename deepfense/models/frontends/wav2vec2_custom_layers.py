import os

from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
from deepfense.models.frontends.layer_custom_bands import (
    CustomLayerBandsFrontendMixin,
    init_custom_layers_frontend,
)


@register_frontend("wav2vec2_custom_layers")
class Wav2Vec2CustomLayersWrapper(CustomLayerBandsFrontendMixin, BaseFrontend):
    """
    XLSR/Wav2Vec2 frontend with learnable multi-band layer pooling.

    For ``n_bands=2`` (default):
      1. Two independent softmax-weighted sums over encoder layers (shallow / deep).
      2. Linear projection of each band to ``proj_dim`` (default 128).
      3. 8-head cross-attention (configurable via ``cross_attn_heads``) where the
         deep band queries the shallow band, with residual.
      4. Orthogonality penalty ``λ · (a · b)`` exposed via ``get_auxiliary_loss()``.

    Set ``n_bands=1`` for a single learned weighted sum (no cross-attention).
    Set ``n_bands=3`` for three bands with pairwise orthogonality; cross-attention
    uses the deepest band as query and the shallowest as key/value.
    """

    frontend_type = "wav2vec2"
    default_source = "fairseq"
    frontend_hop: int = 320

    def __init__(self, config):
        super().__init__(config)
        self.source = config.get("source", self.default_source)
        self.ckpt_path = config.get("ckpt_path")
        if self.ckpt_path is None:
            raise ValueError("ckpt_path must be provided in config")

        if self.source == "fairseq":
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.ckpt_path}")
            from deepfense.models.modules.fairseq_local import load_fairseq_model

            self.model = load_fairseq_model(self.ckpt_path)
        elif self.source == "huggingface":
            from transformers import Wav2Vec2Model

            self.model = Wav2Vec2Model.from_pretrained(self.ckpt_path)
        else:
            raise ValueError(f"Unknown source: {self.source}. Must be 'fairseq' or 'huggingface'")

        init_custom_layers_frontend(self, config)

    def forward(self, input_data, mask=None):
        return self._forward_custom_layers(input_data, mask=mask)
