import torch
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
from deepfense.models.frontends.freeze_policy import apply_ssl_freeze_policy
import logging

logger = logging.getLogger(__name__)


@register_frontend("wavlm")
class WavLMWrapper(BaseFrontend):
    def __init__(self, config):
        super().__init__(config)

        self.source = config.get("source", "unil")
        self.ckpt_path = config.get("ckpt_path", None)

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

    def forward(self, input_data, mask=None):
        if self.source == "unil":
            padding_mask = mask.to(input_data.device).eq(0) if mask is not None else None
            out, _ = self.model.extract_features(
                input_data,
                padding_mask=padding_mask,
                mask=False,
                ret_layer_results=False,
            )
            return out

        if self.source == "huggingface":
            attention_mask = mask.to(input_data.device).long() if mask is not None else None
            outputs = self.model(input_data, attention_mask=attention_mask)
            return outputs.last_hidden_state

        raise ValueError(f"Unknown source: {self.source}")
