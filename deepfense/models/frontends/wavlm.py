import torch
import torch.nn as nn
from deepfense.models.modules.wavlm.wavlm import WavLM, WavLMConfig
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend

@register_frontend("wavlm")
class WavLMWrapper(BaseFrontend):
    def __init__(self, config):
        super().__init__(config)

        self.ckpt_path = config.get("ckpt_path", None)
        checkpoint = torch.load(self.ckpt_path)

        cfg = WavLMConfig(checkpoint["cfg"])
        self.model = WavLM(cfg)
        self.model.load_state_dict(checkpoint["model"], strict=False)

    def forward(self, input_data, mask=None):

        x, layers = self.model.extract_features(
            input_data, mask=False, ret_layer_results=True
        )[0]
        layer_results = torch.stack(layers, dim=1).permute(2, 1, 0, 3).contiguous()

        return x
