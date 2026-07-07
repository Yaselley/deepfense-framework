import os

from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
from deepfense.models.frontends.layer_mixing import LayerWeightedSumFrontendMixin
from deepfense.models.frontends.layer_sum_utils import init_layer_sum_frontend


@register_frontend("wav2vec2_layer_sum")
class Wav2Vec2LayerSumWrapper(LayerWeightedSumFrontendMixin, BaseFrontend):
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

        init_layer_sum_frontend(self, config)

    def forward(self, input_data, mask=None):
        return self._forward_layer_sum(input_data, mask=mask)
