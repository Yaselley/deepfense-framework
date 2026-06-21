import os

from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
from deepfense.models.frontends.freeze_policy import apply_ssl_freeze_policy
from deepfense.models.frontends.layer_mixing import LayerWeightedSumFrontendMixin


@register_frontend("hubert_layer_sum")
class HubertLayerSumWrapper(LayerWeightedSumFrontendMixin, BaseFrontend):
    frontend_type = "hubert"
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
            from transformers import HubertModel

            self.model = HubertModel.from_pretrained(self.ckpt_path)
        else:
            raise ValueError(f"Unknown source: {self.source}. Must be 'fairseq' or 'huggingface'")

        self.freeze_summary = apply_ssl_freeze_policy(self.model, self.source, config)
        self._init_layer_sum_config(config)

    def forward(self, input_data, mask=None):
        return self._forward_layer_sum(input_data, mask=mask)
