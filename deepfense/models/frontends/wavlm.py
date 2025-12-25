import torch
import torch.nn as nn
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
import logging

logger = logging.getLogger(__name__)

@register_frontend("wavlm")
class WavLMWrapper(BaseFrontend):
    def __init__(self, config):
        super().__init__(config)

        self.source = config.get("source", "unil")
        self.ckpt_path = config.get("ckpt_path", None)
        self.freeze = config.get("freeze", False)

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

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_data, mask=None):
        if self.source == "unil":

            x, layers = self.model.extract_features(
                input_data, mask=False, ret_layer_results=True
            )[0]
            return x

        elif self.source == "huggingface":
            attention_mask = None
            if mask is not None:
                attention_mask = mask.long()

            outputs = self.model(input_data, attention_mask=attention_mask)
            return outputs.last_hidden_state
