import torch
import torch.nn as nn
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
import logging

logger = logging.getLogger(__name__)

@register_frontend("wavlm")
class WavLMWrapper(BaseFrontend):
    # WavLM-Base/Large feature extractor: stride 5*2*2*2*2*2*2 = 320 samples.
    frontend_hop: int = 320

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
        """
        ``mask``: ``(B, T_audio)`` 1 = valid, 0 = padding (collate convention).
        Both UniL ``WavLM.extract_features`` and HuggingFace ``WavLMModel`` accept
        a padding mask; we just translate polarity here so partial-deepfake
        batches don't leak padding into features.
        """
        if self.source == "unil":
            padding_mask = None
            if mask is not None:
                padding_mask = mask.eq(0)
            x, layers = self.model.extract_features(
                input_data,
                padding_mask=padding_mask,
                mask=False,
                ret_layer_results=True,
            )[0]
            return x

        elif self.source == "huggingface":
            attention_mask = None
            if mask is not None:
                attention_mask = mask.long()

            outputs = self.model(input_data, attention_mask=attention_mask)
            return outputs.last_hidden_state
