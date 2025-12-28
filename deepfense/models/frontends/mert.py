import torch
import torch.nn as nn
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
import logging

logger = logging.getLogger(__name__)

@register_frontend("mert")
class MERTWrapper(BaseFrontend):
    """
    Wrapper for MERT (Music Audio Pre-training) models.
    Source: HuggingFace (m-a-p/MERT-v1-95M, etc.)
    """
    def __init__(self, config):
        super().__init__(config)

        self.source = config.get("source", "huggingface") 
        self.ckpt_path = config.get("ckpt_path", "m-a-p/MERT-v1-95M")
        self.freeze = config.get("freeze", False)
        self.trust_remote_code = config.get("trust_remote_code", True)

        if self.source == "huggingface":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.ckpt_path, 
                trust_remote_code=self.trust_remote_code
            )
        else:
            raise ValueError(f"MERT only supports 'huggingface' source. Got {self.source}")

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_data, mask=None):
        # MERT via HF expects input_values
        # mask: 1=valid, 0=pad. HF: 1=valid, 0=pad.
        
        attention_mask = None
        if mask is not None:
            attention_mask = mask.long()

        # MERT forward
        outputs = self.model(input_data, attention_mask=attention_mask)
        
        # Last hidden state: [B, T, D]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        else:
            # Fallback for tuple output
            return outputs[0]
