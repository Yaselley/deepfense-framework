import torch

from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
from deepfense.models.frontends.freeze_policy import apply_ssl_freeze_policy
from deepfense.models.frontends.layer_mixing import LayerWeightedSumFrontendMixin


@register_frontend("wavlm_layer_sum")
class WavLMLayerSumWrapper(LayerWeightedSumFrontendMixin, BaseFrontend):
    frontend_type = "wavlm"
    default_source = "unil"
    frontend_hop: int = 320

    def __init__(self, config):
        super().__init__(config)
        self.source = config.get("source", self.default_source)
        self.ckpt_path = config.get("ckpt_path")
        if self.ckpt_path is None:
            raise ValueError("ckpt_path must be provided in config")

        if self.source == "unil":
            from deepfense.models.modules.wavlm.wavlm import WavLM, WavLMConfig

            checkpoint = torch.load(self.ckpt_path)
            cfg = WavLMConfig(checkpoint["cfg"])
            self.model = WavLM(cfg)
            self.model.load_state_dict(checkpoint["model"], strict=False)
        elif self.source == "huggingface":
            from transformers import WavLMModel

            self.model = WavLMModel.from_pretrained(self.ckpt_path)
        else:
            raise ValueError(f"Unknown source: {self.source}")

        self.freeze_summary = apply_ssl_freeze_policy(self.model, self.source, config)
        self._init_layer_sum_config(config)

    def forward(self, input_data, mask=None):
        return self._forward_layer_sum(input_data, mask=mask)
