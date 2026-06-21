import os
import torch
import torch.nn as nn
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
from deepfense.models.frontends.freeze_policy import apply_ssl_freeze_policy
import logging

logger = logging.getLogger(__name__)


@register_frontend("hubert")
class HubertWrapper(BaseFrontend):
    # HuBERT-Base/Large/XL share the same conv stack with stride 320.
    frontend_hop: int = 320

    def __init__(self, config):
        super().__init__(config)

        self.source = config.get("source", "fairseq")
        self.ckpt_path = config.get("ckpt_path", None)
        self.freeze = config.get("freeze", False)

        if self.ckpt_path is None:
            raise ValueError("ckpt_path must be provided in config")

        if self.source == "fairseq":
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint file not found: {self.ckpt_path}. "
                    "Please verify the path is correct."
                )
            from deepfense.models.modules.fairseq_local import load_fairseq_model
            self.model = load_fairseq_model(self.ckpt_path)

        elif self.source == "huggingface":
            from transformers import HubertModel
            self.model = HubertModel.from_pretrained(self.ckpt_path)

        else:
            raise ValueError(f"Unknown source: {self.source}. Must be 'fairseq' or 'huggingface'")

        self.freeze_summary = apply_ssl_freeze_policy(self.model, self.source, config)

    def forward(self, input_data, mask=None):
        """``mask``: 1 = valid sample, 0 = padding. See Wav2VecWrapper."""
        if self.source == "fairseq":
            padding_mask = None
            if mask is not None:
                padding_mask = mask.eq(0)
            emb = self.model(
                input_data,
                padding_mask=padding_mask,
                mask=False,
                features_only=True,
            )
            return emb["x"]

        elif self.source == "huggingface":
            attention_mask = None
            if mask is not None:
                attention_mask = mask.long()
            outputs = self.model(input_data, attention_mask=attention_mask)
            return outputs.last_hidden_state
