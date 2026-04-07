import os
import torch
import torch.nn as nn
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
import logging

logger = logging.getLogger(__name__)


@register_frontend("wav2vec2")
class Wav2VecWrapper(BaseFrontend):
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
            from transformers import Wav2Vec2Model
            self.model = Wav2Vec2Model.from_pretrained(self.ckpt_path)

        else:
            raise ValueError(f"Unknown source: {self.source}. Must be 'fairseq' or 'huggingface'")

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_data, mask=None):
        if self.source == "fairseq":
            emb = self.model(input_data, mask=False, features_only=True)
            return emb["x"]

        elif self.source == "huggingface":
            attention_mask = None
            if mask is not None:
                attention_mask = mask.long()
            outputs = self.model(input_data, attention_mask=attention_mask)
            return outputs.last_hidden_state
