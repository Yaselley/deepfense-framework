import os
import torch
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
from deepfense.models.frontends.freeze_policy import apply_ssl_freeze_policy
import logging

logger = logging.getLogger(__name__)


@register_frontend("hubert")
class HubertWrapper(BaseFrontend):
    def __init__(self, config):
        super().__init__(config)

        self.source = config.get("source", "fairseq")
        self.ckpt_path = config.get("ckpt_path", None)

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
        if self.source == "fairseq":
            padding_mask = mask.to(input_data.device).eq(0) if mask is not None else None
            emb = self.model(
                input_data,
                padding_mask=padding_mask,
                mask=False,
                features_only=True,
            )
            return emb["x"]

        if self.source == "huggingface":
            attention_mask = mask.to(input_data.device).long() if mask is not None else None
            outputs = self.model(input_data, attention_mask=attention_mask)
            return outputs.last_hidden_state

        raise ValueError(f"Unknown source: {self.source}")
